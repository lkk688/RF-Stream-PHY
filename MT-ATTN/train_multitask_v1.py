#!/usr/bin/env python3
"""
train_multitask_v1.py – Multi-Task Preamble CNN: Gate + Modulation + SNR.

Stage 1 of the multi-task PHY intelligence system.
A shared 1D-CNN backbone processes the raw 800-sample preamble IQ window
(STF + LTF, extracted before demodulation) and feeds three task heads:

  Head 1 – Gate    : binary OK/fail classification   (BCEWithLogitsLoss)
  Head 2 – Mod     : QPSK vs QAM-16 classification   (BCEWithLogitsLoss)
  Head 3 – SNR     : receive SNR regression (dB)      (MSELoss, OK only)

All labels come automatically from NPZ metadata — no human annotation:
  gate   ← CRC pass/fail (filename suffix _ok vs _fail*)
  mod    ← meta_json["bps"] (2 → QPSK=0, 4 → QAM16=1)
  snr_db ← meta_json["snr_db"]  (available only in _ok packets)

Ablation control via --tasks:
  --tasks gate          → single-task baseline (reproduces PreamCNN)
  --tasks gate mod      → 2-task: gate + modulation
  --tasks gate mod snr  → full 3-task model (default)

Outputs: rf_stream/multitask_model/
  multitask_v1.pt    checkpoint (state_dict + norm stats + task config)
  metrics.json       per-task: AUC, accuracy, RMSE, ROC curve data
  history.json       per-epoch loss breakdown per task
"""
import argparse, glob, json, os, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from sklearn.metrics import (roc_auc_score, roc_curve,
                                 precision_recall_curve, accuracy_score)
    SKLEARN = True
except ImportError:
    SKLEARN = False

# Prevent CuDNN segfaults on Conv1D (known issue with certain GPU drivers)
torch.backends.cudnn.enabled = False

# ── Constants ────────────────────────────────────────────────────────────────
PREAM_LEN   = 800          # STF(6×~80) + LTF(4×80) complex samples
PREAM_FLOAT = PREAM_LEN * 2  # interleaved real/imag → 1600 floats

# ── Preamble extraction from NPZ ─────────────────────────────────────────────
def extract_preamble_iq(npz, meta):
    """Return 1600-float preamble IQ (real/imag interleaved), or None."""
    if "rxw" not in npz.files:
        return None
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
    return out.copy()

# ── Data loading ──────────────────────────────────────────────────────────────
def load_dataset(run_dirs):
    """
    Returns list of dicts:
      pream  : float32[1600]  preamble IQ
      gate   : int (1=OK, 0=fail)
      mod    : int (0=QPSK, 1=QAM16) or -1 if unknown
      snr_db : float or NaN
      has_snr: bool
    """
    samples = []
    skipped = 0
    for rd in run_dirs:
        for f in sorted(glob.glob(os.path.join(rd, "cap_*.npz"))):
            gate_label = 1 if "_ok.npz" in f else 0
            with np.load(f, allow_pickle=True) as npz:
                meta = {}
                if "meta_json" in npz.files:
                    try:
                        mj = npz["meta_json"].item()
                        meta = json.loads(mj if isinstance(mj, str) else mj.decode())
                    except Exception:
                        pass
                pream = extract_preamble_iq(npz, meta)

            if pream is None:
                skipped += 1
                continue

            bps = meta.get("bps", -1)
            mod_label = 0 if bps == 2 else (1 if bps == 4 else -1)

            # Only use SNR from successfully decoded packets (CRC pass)
            raw_snr = meta.get("snr_db", None) if gate_label == 1 else None
            if raw_snr is None or (isinstance(raw_snr, float) and np.isnan(raw_snr)):
                snr_db = np.nan
                has_snr = False
            else:
                snr_db = float(raw_snr)
                has_snr = True

            samples.append(dict(pream=pream, gate=gate_label,
                                mod=mod_label, snr_db=snr_db, has_snr=has_snr))

    ok  = sum(1 for s in samples if s["gate"] == 1)
    fail = sum(1 for s in samples if s["gate"] == 0)
    qpsk = sum(1 for s in samples if s["mod"] == 0)
    qam  = sum(1 for s in samples if s["mod"] == 1)
    snr_valid = sum(1 for s in samples if s["has_snr"])
    print(f"  loaded {len(samples)} samples  (skipped {skipped} without rxw)")
    print(f"  gate: ok={ok}  fail={fail}")
    print(f"  mod:  qpsk={qpsk}  qam16={qam}  unknown={len(samples)-qpsk-qam}")
    print(f"  snr:  valid={snr_valid}")
    return samples

def balance_gate(samples, seed=42):
    """Oversample the minority gate class so ok/fail are balanced."""
    ok   = [s for s in samples if s["gate"] == 1]
    fail = [s for s in samples if s["gate"] == 0]
    rng = random.Random(seed)
    if len(ok) > len(fail):
        fail = (fail * (len(ok) // len(fail) + 1))[:len(ok)]
    elif len(fail) > len(ok):
        ok = (ok * (len(fail) // len(ok) + 1))[:len(fail)]
    combined = ok + fail
    rng.shuffle(combined)
    return combined

def train_val_split(samples, val_frac=0.2, seed=42):
    rng = random.Random(seed)
    s = list(samples); rng.shuffle(s)
    n_val = max(1, int(val_frac * len(s)))
    return s[n_val:], s[:n_val]

def build_tensors(samples, snr_mu=None, snr_sigma=None):
    """
    Pack samples into tensors. Returns (X, gate_y, mod_y, snr_y, snr_mask).
    snr_y is normalised; snr_mu/snr_sigma computed from samples if not given.
    """
    X    = np.stack([s["pream"] for s in samples]).astype(np.float32)
    gate = np.array([s["gate"]  for s in samples], dtype=np.float32)
    mod  = np.array([s["mod"]   for s in samples], dtype=np.float32)
    snr_raw  = np.array([s["snr_db"] if s["has_snr"] else np.nan
                         for s in samples], dtype=np.float32)
    snr_mask = np.array([s["has_snr"] for s in samples], dtype=bool)

    # Normalise SNR with training-set statistics
    if snr_mu is None:
        valid = snr_raw[snr_mask]
        snr_mu    = float(np.mean(valid))  if len(valid) > 0 else 0.0
        snr_sigma = float(np.std(valid))   if len(valid) > 1 else 1.0
        snr_sigma = max(snr_sigma, 1e-3)
    snr_norm = np.where(snr_mask, (snr_raw - snr_mu) / snr_sigma, 0.0).astype(np.float32)

    # Clean NaN/Inf in preamble (ADC saturation)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return (
        torch.from_numpy(X).float().contiguous(),
        torch.from_numpy(gate).float(),
        torch.from_numpy(mod).float(),
        torch.from_numpy(snr_norm).float(),
        torch.from_numpy(snr_mask),
        snr_mu, snr_sigma,
    )

# ── Model ─────────────────────────────────────────────────────────────────────
class MultiTaskPreamCNN(nn.Module):
    """
    Shared 2-channel Conv1D backbone with three task heads.

    Input : (B, 1600) float — real/imag interleaved preamble IQ
    Heads :
      gate_logit  (B,)  binary gate score
      mod_logit   (B,)  binary modulation score (QPSK=0 / QAM16=1)
      snr_pred    (B,)  normalised SNR estimate
    """
    def __init__(self, embed_dim=256, tasks=("gate", "mod", "snr")):
        super().__init__()
        self.tasks = tasks
        self.embed_dim = embed_dim

        # Shared Conv1D encoder — identical to PreamCNN backbone
        self.encoder = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=31, padding=15),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=15, padding=7),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
        )  # output: (B, 128, 8) → flatten → 1024-dim

        # Shared projection to embed_dim
        self.proj = nn.Sequential(
            nn.Linear(128 * 8, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # Task-specific heads
        if "gate" in tasks:
            self.gate_head = nn.Linear(embed_dim, 1)
        if "mod" in tasks:
            self.mod_head  = nn.Linear(embed_dim, 1)
        if "snr" in tasks:
            self.snr_head  = nn.Sequential(
                nn.Linear(embed_dim, 32), nn.ReLU(),
                nn.Linear(32, 1),
            )

    def encode(self, x):
        B = x.size(0)
        # (B, 1600) → (B, 2, 800)
        x2 = x.view(B, PREAM_LEN, 2).permute(0, 2, 1).contiguous()
        feat = self.encoder(x2).view(B, -1)   # (B, 1024)
        return self.proj(feat)                  # (B, embed_dim)

    def forward(self, x):
        emb = self.encode(x)
        out = {}
        if "gate" in self.tasks:
            out["gate"] = self.gate_head(emb).squeeze(-1)
        if "mod" in self.tasks:
            out["mod"]  = self.mod_head(emb).squeeze(-1)
        if "snr" in self.tasks:
            out["snr"]  = self.snr_head(emb).squeeze(-1)
        return out

# ── Training ──────────────────────────────────────────────────────────────────
def run_epoch(model, loader, opt, device, tasks, w_gate, w_mod, w_snr, train=True):
    if train:
        model.train()
    else:
        model.eval()
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()
    total = gate_t = mod_t = snr_t = n = 0
    with torch.set_grad_enabled(train):
        for xb, gy, my, sy, sm in loader:
            xb = xb.to(device)
            gy, my, sy, sm = gy.to(device), my.to(device), sy.to(device), sm.to(device)
            out = model(xb)
            loss = torch.tensor(0.0, device=device)
            if "gate" in tasks:
                gl = bce(out["gate"], gy)
                loss = loss + w_gate * gl
                gate_t += gl.item() * len(xb)
            if "mod" in tasks:
                # only include samples with known mod label (-1 excluded)
                mod_mask = (my >= 0)
                if mod_mask.any():
                    ml = bce(out["mod"][mod_mask], my[mod_mask])
                    loss = loss + w_mod * ml
                    mod_t += ml.item() * mod_mask.sum().item()
            if "snr" in tasks and sm.any():
                sl = mse(out["snr"][sm], sy[sm])
                loss = loss + w_snr * sl
                snr_t += sl.item() * sm.sum().item()
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(xb)
            n += len(xb)
    return total / max(n, 1), gate_t / max(n, 1), mod_t / max(n, 1), snr_t / max(n, 1)

def collect_preds(model, loader, device, tasks):
    model.eval()
    g_pred, g_true = [], []
    m_pred, m_true = [], []
    s_pred, s_true = [], []
    with torch.no_grad():
        for xb, gy, my, sy, sm in loader:
            xb = xb.to(device)
            out = model(xb)
            if "gate" in tasks:
                g_pred.extend(torch.sigmoid(out["gate"]).cpu().tolist())
                g_true.extend(gy.tolist())
            if "mod" in tasks:
                mask = (my >= 0)
                if mask.any():
                    m_pred.extend(torch.sigmoid(out["mod"][mask]).cpu().tolist())
                    m_true.extend(my[mask].tolist())
            if "snr" in tasks:
                mask = sm.bool()
                if mask.any():
                    s_pred.extend(out["snr"][mask].cpu().tolist())
                    s_true.extend(sy[mask].tolist())
    return (np.array(g_pred), np.array(g_true),
            np.array(m_pred), np.array(m_true),
            np.array(s_pred), np.array(s_true))

def compute_metrics(g_pred, g_true, m_pred, m_true, s_pred, s_true,
                    snr_mu, snr_sigma, tasks):
    metrics = {}
    if "gate" in tasks and SKLEARN and len(g_true) > 0:
        auc = roc_auc_score(g_true, g_pred)
        prec, rec, thr = precision_recall_curve(g_true, g_pred)
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        bi = np.argmax(f1[:-1])
        fpr, tpr, _ = roc_curve(g_true, g_pred)
        metrics["gate"] = {
            "roc_auc": float(auc),
            "best_thr": float(thr[bi]),
            "best_prec": float(prec[bi]),
            "best_rec": float(rec[bi]),
            "best_f1": float(f1[bi]),
            "roc_fpr": fpr.tolist(), "roc_tpr": tpr.tolist(),
        }
        print(f"  [gate]  AUC={auc:.4f}  thr={thr[bi]:.4f}  "
              f"P={prec[bi]:.3f}  R={rec[bi]:.3f}")

    if "mod" in tasks and len(m_true) > 0:
        m_bin = (np.array(m_pred) > 0.5).astype(int)
        acc = float(np.mean(m_bin == np.array(m_true).astype(int)))
        n_classes = len(np.unique(m_true))
        if SKLEARN and n_classes >= 2:
            m_auc = roc_auc_score(m_true, m_pred)
        else:
            m_auc = -1.0  # undefined with single class (e.g., single-modulation run)
        metrics["mod"] = {"accuracy": acc, "roc_auc": float(m_auc), "n_val": len(m_true)}
        print(f"  [mod]   acc={acc:.4f}  AUC={m_auc:.4f}  n={len(m_true)}  classes={n_classes}")

    if "snr" in tasks and len(s_true) > 0:
        s_pred_db = np.array(s_pred) * snr_sigma + snr_mu
        s_true_db = np.array(s_true) * snr_sigma + snr_mu
        rmse = float(np.sqrt(np.mean((s_pred_db - s_true_db) ** 2)))
        mae  = float(np.mean(np.abs(s_pred_db - s_true_db)))
        ss_res = np.sum((s_true_db - s_pred_db) ** 2)
        ss_tot = np.sum((s_true_db - np.mean(s_true_db)) ** 2)
        r2 = float(1 - ss_res / (ss_tot + 1e-12))
        metrics["snr"] = {
            "rmse_db": rmse, "mae_db": mae, "r2": r2,
            "snr_mu": snr_mu, "snr_sigma": snr_sigma,
        }
        print(f"  [snr]   RMSE={rmse:.2f}dB  MAE={mae:.2f}dB  R²={r2:.4f}  n={len(s_true)}")

    return metrics

# ── Main ──────────────────────────────────────────────────────────────────────
def main(args):
    tasks = tuple(args.tasks)
    print(f"[multitask-v1] tasks={tasks}  embed_dim={args.embed_dim}")
    print(f"[multitask-v1] loss weights  gate={args.w_gate}  mod={args.w_mod}  snr={args.w_snr}")

    # Gather run directories
    run_dirs = []
    for p in args.run_dirs:
        run_dirs += sorted(glob.glob(p))
    run_dirs = [r for r in run_dirs if os.path.isdir(r)]
    print(f"[multitask-v1] {len(run_dirs)} run dirs")

    # Load
    all_samples = load_dataset(run_dirs)
    if len(all_samples) < 50:
        print("ERROR: too few samples"); return

    # Balance gate classes, then split train/val
    balanced = balance_gate(all_samples, seed=42)
    trn_s, val_s = train_val_split(balanced, val_frac=0.2, seed=42)
    print(f"[multitask-v1] trn={len(trn_s)}  val={len(val_s)}")

    # Build tensors (compute SNR norm stats from training set)
    Xtr, gtr, mtr, str_, smtr, snr_mu, snr_sigma = build_tensors(trn_s)
    Xva, gva, mva, sva, smva, _, _ = build_tensors(val_s, snr_mu, snr_sigma)
    print(f"[multitask-v1] SNR norm: mu={snr_mu:.1f}dB  sigma={snr_sigma:.1f}dB")

    trn_ds = TensorDataset(Xtr, gtr, mtr, str_, smtr)
    val_ds = TensorDataset(Xva, gva, mva, sva, smva)
    trn_ld = DataLoader(trn_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_ld = DataLoader(val_ds, batch_size=256,             shuffle=False, num_workers=0)

    device = ("cuda" if (args.device == "auto" and torch.cuda.is_available())
              else args.device)
    print(f"[multitask-v1] device={device}")

    model = MultiTaskPreamCNN(embed_dim=args.embed_dim, tasks=tasks).to(device)
    n_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[multitask-v1] params={n_param:,}")

    opt   = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    history = []
    best_auc = 0.0
    best_state = None

    for ep in range(args.epochs):
        tr_tot, tr_g, tr_m, tr_s = run_epoch(
            model, trn_ld, opt, device, tasks,
            args.w_gate, args.w_mod, args.w_snr, train=True)
        va_tot, va_g, va_m, va_s = run_epoch(
            model, val_ld, None, device, tasks,
            args.w_gate, args.w_mod, args.w_snr, train=False)
        sched.step()

        rec = {"epoch": ep+1,
               "train_total": tr_tot, "train_gate": tr_g,
               "train_mod": tr_m,    "train_snr": tr_s,
               "val_total": va_tot,  "val_gate": va_g,
               "val_mod": va_m,      "val_snr": va_s}
        history.append(rec)

        if (ep+1) % 10 == 0:
            print(f"  ep={ep+1:3d}  "
                  f"tr=[{tr_tot:.4f}|g={tr_g:.4f}|m={tr_m:.4f}|s={tr_s:.4f}]  "
                  f"va=[{va_tot:.4f}|g={va_g:.4f}|m={va_m:.4f}|s={va_s:.4f}]")

        # Track best gate AUC checkpoint
        if "gate" in tasks and SKLEARN and (ep+1) % 5 == 0:
            gp, gt, *_ = collect_preds(model, val_ld, device, tasks)
            if len(gt) > 0:
                auc = roc_auc_score(gt, gp)
                if auc > best_auc:
                    best_auc = auc
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        best_state = model.state_dict()

    # Final evaluation with best checkpoint
    model.load_state_dict(best_state)
    print(f"\n[multitask-v1] final evaluation (best gate AUC={best_auc:.4f})")
    g_pred, g_true, m_pred, m_true, s_pred, s_true = collect_preds(
        model, val_ld, device, tasks)
    metrics = compute_metrics(g_pred, g_true, m_pred, m_true, s_pred, s_true,
                              snr_mu, snr_sigma, tasks)

    # Save
    os.makedirs(args.out_dir, exist_ok=True)
    ckpt = {
        "state_dict": best_state,
        "tasks": list(tasks),
        "embed_dim": args.embed_dim,
        "pream_len": PREAM_LEN,
        "snr_mu": snr_mu,
        "snr_sigma": snr_sigma,
        "metrics": metrics,
    }
    torch.save(ckpt, os.path.join(args.out_dir, "multitask_v1.pt"))
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fp:
        json.dump(metrics, fp, indent=2)
    with open(os.path.join(args.out_dir, "history.json"), "w") as fp:
        json.dump(history, fp, indent=2)
    print(f"[multitask-v1] saved → {args.out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dirs", nargs="+",
                    default=[
                        "rf_stream/ber_sweep_v4/run_*",
                        "rf_stream/ber_sweep_v3/run_*",
                        "rf_stream/ber_sweep_v2/run_*",
                        "rf_stream/ber_sweep/run_20260428_233037",
                        "rf_stream/ber_sweep/run_20260429_001722",
                        "rf_stream/ber_sweep/run_20260429_003513",
                        "rf_stream/ber_sweep/run_20260429_070949",
                    ])
    ap.add_argument("--tasks",     nargs="+",  default=["gate", "mod", "snr"],
                    choices=["gate", "mod", "snr"],
                    help="Tasks to train (enables ablation study)")
    ap.add_argument("--embed_dim", type=int,   default=256)
    ap.add_argument("--epochs",    type=int,   default=80)
    ap.add_argument("--batch_size",type=int,   default=64)
    ap.add_argument("--lr",        type=float, default=1e-3)
    ap.add_argument("--w_gate",    type=float, default=1.0,
                    help="Gate task loss weight")
    ap.add_argument("--w_mod",     type=float, default=0.5,
                    help="Modulation task loss weight")
    ap.add_argument("--w_snr",     type=float, default=0.5,
                    help="SNR task loss weight")
    ap.add_argument("--out_dir",   default="rf_stream/multitask_model")
    ap.add_argument("--device",    default="auto")
    args = ap.parse_args()
    main(args)
