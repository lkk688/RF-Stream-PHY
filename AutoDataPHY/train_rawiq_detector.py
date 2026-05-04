#!/usr/bin/env python3
"""
train_rawiq_detector.py – End-to-End Deep Learning Preamble Detector.

Operates directly on two raw representations extracted from NPZ archives:
  (A) corr_norm_window : 512-point NCC cross-correlation snippet around
                         the detected STF peak  → 1D CNN "xcorr detector"
  (B) preamble_iq      : raw complex IQ samples of the STF+LTF preamble
                         (800 samples → 1600 floats real/imag)  → 1D CNN

Both models are trained and evaluated, ROC-AUC compared against
gate_model_v2 (hand-crafted features, 61-dim MLP).

Outputs: rf_stream/rawiq_model/  (xcorr_model.pt, pream_model.pt, metrics.json)
"""
import argparse, glob, json, os, sys, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
try:
    from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
    SKLEARN = True
except ImportError:
    SKLEARN = False

# 【终极修复 1】禁用 CuDNN，绕过底层 1D 卷积的 Segfault 漏洞
torch.backends.cudnn.enabled = False

# ── Constants ────────────────────────────────────────────────────────────────
XCORR_WIN = 512        # corr_norm window around peak (samples)
XCORR_PAD = 256        # samples before peak
PREAM_LEN = 800        # STF(6×80) + LTF(4×80) samples
PREAM_FLOAT = PREAM_LEN * 2  # real + imag interleaved

# ── Feature extraction from NPZ ──────────────────────────────────────────────
# def extract_xcorr(npz, meta):
#     """512-float window of corr_norm around the best NCC peak."""
#     if "corr_norm" not in npz.files:
#         return None
#     corr = np.asarray(npz["corr_norm"], dtype=np.float32)
#     peak = int(meta.get("ncc_best_idx", meta.get("xc_best_idx", len(corr)//2)))
#     lo = max(0, peak - XCORR_PAD)
#     hi = lo + XCORR_WIN
#     if hi > len(corr):
#         lo = max(0, len(corr) - XCORR_WIN)
#         hi = len(corr)
#     win = corr[lo:hi]
#     # zero-pad if shorter
#     if len(win) < XCORR_WIN:
#         win = np.pad(win, (0, XCORR_WIN - len(win)))
#     return win.astype(np.float32)

# def extract_preamble_iq(npz, meta):
#     """1600-float preamble IQ (real, imag interleaved)."""
#     if "rxw" not in npz.files:
#         return None
#     rxw = np.asarray(npz["rxw"], dtype=np.complex64)
#     stf = int(meta.get("stf_idx", 0))
#     end = stf + PREAM_LEN
#     if end > len(rxw):
#         stf = max(0, len(rxw) - PREAM_LEN)
#         end = len(rxw)
#     seg = rxw[stf:end]
#     if len(seg) < PREAM_LEN:
#         seg = np.pad(seg, (0, PREAM_LEN - len(seg)))
#     out = np.empty(PREAM_FLOAT, dtype=np.float32)
#     out[0::2] = seg.real
#     out[1::2] = seg.imag
#     return out

# ── Feature extraction from NPZ ──────────────────────────────────────────────
def extract_xcorr(npz, meta):
    """512-float window of corr_norm around the best NCC peak."""
    if "corr_norm" not in npz.files:
        return None
    
    # 1. FORCE COPY: Detach from the npz file memory
    corr = np.array(npz["corr_norm"], dtype=np.float32, copy=True)
    
    peak = int(meta.get("ncc_best_idx", meta.get("xc_best_idx", len(corr)//2)))
    lo = max(0, peak - XCORR_PAD)
    hi = lo + XCORR_WIN
    if hi > len(corr):
        lo = max(0, len(corr) - XCORR_WIN)
        hi = len(corr)
    win = corr[lo:hi]
    # zero-pad if shorter
    if len(win) < XCORR_WIN:
        win = np.pad(win, (0, XCORR_WIN - len(win)))
        
    # 2. Return a safe copy, not a memory slice
    return win.copy()

def extract_preamble_iq(npz, meta):
    """1600-float preamble IQ (real, imag interleaved)."""
    if "rxw" not in npz.files:
        return None
        
    # 3. FORCE COPY: Detach from the npz file memory
    rxw = np.array(npz["rxw"], dtype=np.complex64, copy=True)
    
    stf = int(meta.get("stf_idx", 0))
    end = stf + PREAM_LEN
    if end > len(rxw):
        stf = max(0, len(rxw) - PREAM_LEN)
        end = len(rxw)
    seg = rxw[stf:end]
    if len(seg) < PREAM_LEN:
        seg = np.pad(seg, (0, PREAM_LEN - len(seg)))
    out = np.empty(PREAM_FLOAT, dtype=np.float32)
    out[0::2] = seg.real
    out[1::2] = seg.imag
    
    # 4. Return safe copy
    return out.copy()

# ── Dataset ──────────────────────────────────────────────────────────────────
def load_dataset(run_dirs):
    xcorr_pos, xcorr_neg = [], []
    pream_pos, pream_neg = [], []
    skipped = 0
    for rd in run_dirs:
        for f in sorted(glob.glob(os.path.join(rd, "cap_*.npz"))):
            label = 1 if "_ok.npz" in f else 0
            
            # 5. CONTEXT MANAGER: Immediately closes the file handle so Linux doesn't crash
            with np.load(f, allow_pickle=True) as npz:
                meta = {}
                if "meta_json" in npz.files:
                    try:
                        mj = npz["meta_json"].item()
                        meta = json.loads(mj if isinstance(mj, str) else mj.decode())
                    except Exception:
                        pass
                
                xc = extract_xcorr(npz, meta)
                pr = extract_preamble_iq(npz, meta)
                
            # (File is now safely closed, data is sitting safely in RAM)
            if xc is None and pr is None:
                skipped += 1; continue
            if xc is not None:
                (xcorr_pos if label else xcorr_neg).append(xc)
            if pr is not None:
                (pream_pos if label else pream_neg).append(pr)
                
    print(f"  xcorr  pos={len(xcorr_pos)} neg={len(xcorr_neg)}  skipped={skipped}")
    print(f"  pream  pos={len(pream_pos)} neg={len(pream_neg)}")
    return (xcorr_pos, xcorr_neg), (pream_pos, pream_neg)

class SeqDS(Dataset):
    def __init__(self, items):  
        # 强制将离散的 NumPy 切片堆叠成一块全新的、连续的内存
        # 彻底阻断底层的指针游离和 CuDNN 连续性报错
        self.X = torch.tensor(np.stack([item[0] for item in items]), dtype=torch.float32)
        self.Y = torch.tensor([item[1] for item in items], dtype=torch.float32)
        
    def __len__(self): 
        return len(self.Y)
        
    def __getitem__(self, i):
        return self.X[i], self.Y[i]
    
# class SeqDS(Dataset):
#     def __init__(self, items):  # items: list of (array, label)
#         self.items = items
#     def __len__(self): return len(self.items)
#     def __getitem__(self, i):
#         x, y = self.items[i]
#         return torch.from_numpy(x).float(), torch.tensor(float(y))

def balance_and_split(pos, neg, val_frac=0.2, seed=42):
    rng = random.Random(seed)
    # balance
    if len(pos) > len(neg):
        neg = (neg * (len(pos) // len(neg) + 1))[:len(pos)]
    elif len(neg) > len(pos):
        pos = (pos * (len(neg) // len(pos) + 1))[:len(neg)]
    all_items = [(x, 1) for x in pos] + [(x, 0) for x in neg]
    rng.shuffle(all_items)
    n_val = max(1, int(val_frac * len(all_items)))
    return all_items[n_val:], all_items[:n_val]

# ── 1D CNN ───────────────────────────────────────────────────────────────────
# class CNN1D(nn.Module):
#     def __init__(self, in_len, n_in_ch=1):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv1d(n_in_ch, 32, kernel_size=31, padding=15), nn.ReLU(),
#             nn.MaxPool1d(4),
#             nn.Conv1d(32, 64, kernel_size=15, padding=7), nn.ReLU(),
#             nn.MaxPool1d(4),
#             nn.Conv1d(64, 128, kernel_size=7, padding=3), nn.ReLU(),
#             nn.AdaptiveAvgPool1d(8),
#         )
#         self.fc = nn.Sequential(
#             nn.Linear(128 * 8, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1)
#         )
#     def forward(self, x):
#         if x.dim() == 2:
#             x = x.unsqueeze(1)  # (B, 1, L)
#         return self.fc(self.conv(x).view(x.size(0), -1)).squeeze(-1)

class CNN1D(nn.Module):
    def __init__(self, in_len, n_in_ch=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_in_ch, 32, kernel_size=31, padding=15), nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=15, padding=7), nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(64, 128, kernel_size=7, padding=3), nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 8, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1)
        )
    def forward(self, x):
        if x.dim() == 2:
            # 【终极修复 2】不仅 unsqueeze，还要强制内存连续，满足底层卷积要求
            x = x.unsqueeze(1).contiguous()
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        return self.fc(out).squeeze(-1)
    
class PreamCNN(nn.Module):
    """2-channel (I,Q) preamble 1D CNN."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=31, padding=15), nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=15, padding=7), nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(64, 128, kernel_size=7, padding=3), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 8, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1)
        )
        
    def forward(self, x):
        B = x.size(0)
        # Added .contiguous() to prevent CuDNN segfaults!
        x2 = x.view(B, 800, 2).permute(0, 2, 1).contiguous()
        return self.fc(self.conv(x2).view(B, -1)).squeeze(-1)
    
    # def forward(self, x):
    #     # x: (B, 1600)  → reshape to (B, 2, 800)
    #     B = x.size(0)
    #     x2 = x.view(B, 800, 2).permute(0, 2, 1)  # (B, 2, 800)
    #     return self.fc(self.conv(x2).view(B, -1)).squeeze(-1)

# ── Training loop ─────────────────────────────────────────────────────────────
from torch.utils.data import TensorDataset

def prepare_tensor_dataset(items):
    """专门用来处理数据清洗和底层内存打包的辅助函数"""
    # 1. 提取数据并堆叠
    X_np = np.stack([item[0] for item in items])
    Y_np = np.array([item[1] for item in items], dtype=np.float32)
    
    # 2. 致命错误修复：清洗掉因 ADC 饱和引发的 NaN 和 Inf
    X_np = np.nan_to_num(X_np, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 3. 转换为 Tensor 并强制要求内存极度连续 (contiguous)，防止 CuDNN 寻址越界
    X_t = torch.from_numpy(X_np).float().contiguous()
    Y_t = torch.from_numpy(Y_np).float().contiguous()
    
    return TensorDataset(X_t, Y_t)

# ── Training loop ─────────────────────────────────────────────────────────────
# ── Training loop ─────────────────────────────────────────────────────────────
def run_training(name, model, trn_items, val_items, device, epochs=50, lr=1e-3):
    print(f"  [{name}] Preparing and cleaning tensors...")
    trn_ds = prepare_tensor_dataset(trn_items)
    val_ds = prepare_tensor_dataset(val_items)
    
    trn_ld = DataLoader(trn_ds, batch_size=64, shuffle=True, num_workers=0)
    val_ld = DataLoader(val_ds, batch_size=128, shuffle=False)
    
    print(f"  [{name}] Moving model to {device}...")
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.BCEWithLogitsLoss()
    history = []
    
    print(f"  [{name}] Starting training loop...")
    for ep in range(epochs):
        model.train()
        tr_loss = 0.0
        
        for batch_idx, (xb, yb) in enumerate(trn_ld):
            # 探针：只在第一个 Epoch 的第一个 Batch 打印
            if ep == 0 and batch_idx == 0:
                print("    -> [Debug] Batch 0 loaded from RAM")
                
            xb, yb = xb.to(device), yb.to(device)
            if ep == 0 and batch_idx == 0:
                print("    -> [Debug] Batch 0 moved to GPU")
                
            opt.zero_grad()
            
            out = model(xb)
            if ep == 0 and batch_idx == 0:
                print("    -> [Debug] Batch 0 forward pass complete")
                
            loss = crit(out, yb)
            if ep == 0 and batch_idx == 0:
                print("    -> [Debug] Batch 0 loss computed")
                
            loss.backward()
            if ep == 0 and batch_idx == 0:
                print("    -> [Debug] Batch 0 backward pass complete")
                
            opt.step()
            tr_loss += loss.item() * len(xb)
            
        sched.step()
        tr_loss /= len(trn_items)
        
        model.eval(); vl = 0.0
        with torch.no_grad():
            for xb, yb in val_ld:
                xb, yb = xb.to(device), yb.to(device)
                vl += crit(model(xb), yb).item() * len(xb)
        vl /= len(val_items)
        history.append({"epoch": ep+1, "train_loss": tr_loss, "val_loss": vl})
        if (ep+1) % 10 == 0:
            print(f"  [{name}] ep={ep+1:3d}  tr={tr_loss:.4f}  val={vl:.4f}")
            
    # Evaluate (保持之前的代码...)
    model.eval(); preds, trues = [], []
    with torch.no_grad():
        for xb, yb in val_ld:
            xb = xb.to(device)
            preds.extend(torch.sigmoid(model(xb)).cpu().tolist())
            trues.extend(yb.tolist())
            
    preds = np.array(preds); trues = np.array(trues)
    metrics = {"model": name, "n_val": len(trues)}
    if SKLEARN:
        auc = roc_auc_score(trues, preds)
        prec, rec, thr = precision_recall_curve(trues, preds)
        f1 = 2*prec*rec/(prec+rec+1e-12)
        bi = np.argmax(f1[:-1])
        fpr, tpr, _ = roc_curve(trues, preds)
        metrics.update({"roc_auc": float(auc), "best_thr": float(thr[bi]),
                        "best_prec": float(prec[bi]), "best_rec": float(rec[bi]),
                        "best_f1": float(f1[bi]),
                        "roc_fpr": fpr.tolist(), "roc_tpr": tpr.tolist()})
        print(f"  [{name}] AUC={auc:.4f}  thr={thr[bi]:.4f}  "
              f"P={prec[bi]:.3f}  R={rec[bi]:.3f}")
    return metrics, model.state_dict(), history

# def run_training(name, model, trn_items, val_items, device, epochs=50, lr=1e-3):
#     trn_ds = SeqDS(trn_items); val_ds = SeqDS(val_items)
#     trn_ld = DataLoader(trn_ds, batch_size=64, shuffle=True, num_workers=0)
#     val_ld = DataLoader(val_ds, batch_size=128, shuffle=False)
#     model = model.to(device)
#     opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
#     sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
#     crit = nn.BCEWithLogitsLoss()
#     history = []
#     for ep in range(epochs):
#         model.train()
#         tr_loss = 0.0
#         for xb, yb in trn_ld:
#             opt.zero_grad()
#             loss = crit(model(xb.to(device)), yb.to(device))
#             loss.backward(); opt.step()
#             tr_loss += loss.item() * len(xb)
#         sched.step()
#         tr_loss /= len(trn_items)
#         model.eval(); vl = 0.0
#         with torch.no_grad():
#             for xb, yb in val_ld:
#                 vl += crit(model(xb.to(device)), yb.to(device)).item() * len(xb)
#         vl /= len(val_items)
#         history.append({"epoch": ep+1, "train_loss": tr_loss, "val_loss": vl})
#         if (ep+1) % 10 == 0:
#             print(f"  [{name}] ep={ep+1:3d}  tr={tr_loss:.4f}  val={vl:.4f}")
#     # Evaluate
#     model.eval(); preds, trues = [], []
#     with torch.no_grad():
#         for xb, yb in val_ld:
#             preds.extend(torch.sigmoid(model(xb.to(device))).cpu().tolist())
#             trues.extend(yb.tolist())
#     preds = np.array(preds); trues = np.array(trues)
#     metrics = {"model": name, "n_val": len(trues)}
#     if SKLEARN:
#         auc = roc_auc_score(trues, preds)
#         prec, rec, thr = precision_recall_curve(trues, preds)
#         f1 = 2*prec*rec/(prec+rec+1e-12)
#         bi = np.argmax(f1[:-1])
#         fpr, tpr, _ = roc_curve(trues, preds)
#         metrics.update({"roc_auc": float(auc), "best_thr": float(thr[bi]),
#                         "best_prec": float(prec[bi]), "best_rec": float(rec[bi]),
#                         "best_f1": float(f1[bi]),
#                         "roc_fpr": fpr.tolist(), "roc_tpr": tpr.tolist()})
#         print(f"  [{name}] AUC={auc:.4f}  thr={thr[bi]:.4f}  "
#               f"P={prec[bi]:.3f}  R={rec[bi]:.3f}")
#     return metrics, model.state_dict(), history

# ── Main ──────────────────────────────────────────────────────────────────────
def main(args):
    run_dirs = []
    for p in args.run_dirs:
        run_dirs += sorted(glob.glob(p))
    run_dirs = [r for r in run_dirs if os.path.isdir(r)]
    print(f"[rawiq] {len(run_dirs)} run dirs")

    (xc_pos, xc_neg), (pr_pos, pr_neg) = load_dataset(run_dirs)
    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    print(f"[rawiq] device={device}")

    os.makedirs(args.out_dir, exist_ok=True)
    all_metrics = {}

    # --- XCorr model ---
    if xc_pos and xc_neg:
        trn, val = balance_and_split(xc_pos, xc_neg)
        print(f"\n[xcorr CNN] trn={len(trn)} val={len(val)}")
        model_xc = CNN1D(in_len=XCORR_WIN)
        m, sd, hist = run_training("xcorr", model_xc, trn, val, device, args.epochs)
        all_metrics["xcorr"] = m
        torch.save({"state_dict": sd, "arch": "CNN1D", "in_len": XCORR_WIN,
                    "xcorr_pad": XCORR_PAD}, os.path.join(args.out_dir, "xcorr_model.pt"))
        with open(os.path.join(args.out_dir, "xcorr_history.json"), "w") as fp:
            json.dump(hist, fp)

    # --- Preamble IQ model ---
    if pr_pos and pr_neg:
        trn, val = balance_and_split(pr_pos, pr_neg)
        print(f"\n[preamble CNN] trn={len(trn)} val={len(val)}")
        model_pr = PreamCNN()
        m, sd, hist = run_training("pream", model_pr, trn, val, device, args.epochs)
        all_metrics["pream"] = m
        torch.save({"state_dict": sd, "arch": "PreamCNN",
                    "pream_len": PREAM_LEN}, os.path.join(args.out_dir, "pream_model.pt"))
        with open(os.path.join(args.out_dir, "pream_history.json"), "w") as fp:
            json.dump(hist, fp)

    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fp:
        json.dump(all_metrics, fp, indent=2)
    print(f"\n[rawiq] done → {args.out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dirs", nargs="+",
                    default=["rf_stream/ber_sweep_v3/run_*",
                             "rf_stream/ber_sweep/run_20260428_233037",
                             "rf_stream/ber_sweep/run_20260429_001722",
                             "rf_stream/ber_sweep/run_20260429_003513",
                             "rf_stream/ber_sweep/run_20260429_070949"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--out_dir", default="rf_stream/rawiq_model")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    main(args)
