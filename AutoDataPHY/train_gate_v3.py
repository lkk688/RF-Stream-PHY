#!/usr/bin/env python3
"""
train_gate_v3.py – Hard-Negative Mining gate for QAM16.

Positives : status=ok  (any modulation, all v3 sweep dirs)
Negatives : save_tag=fail_hard AND modulation=qam16  only
            (the hardest cases: high-z, high-NCC packets that still fail CRC)

Compares ROC AUC vs gate_model_v2 (all-neg baseline).
Outputs: rf_stream/gate_model_v3/gate_model.pt + metrics.json
"""
import argparse, glob, json, os, sys, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
try:
    from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve
    SKLEARN = True
except ImportError:
    SKLEARN = False

# ── feature extraction (same 61-feature set as v2, no post-demod) ──────────
sys.path.insert(0, os.path.dirname(__file__))
from train_neural_gate import (
    load_npz_features, corr_features, topk_ncc_features,
    energy_ds_features, energy_features_from_rxw, MLP, _safe_float
)

EXCLUDE_POST_DEMOD = ["probe_evm", "cfo_hz", "snr_db",
                      "meta_probe_evm", "meta_snr_db", "meta_cfo_hz",
                      "eng_"]  # skip rxw-derived energy (use energy_ds instead)

def load_npz_fast(path):
    """Load NPZ features WITHOUT loading rxw (avoids 2 MB per file)."""
    z = np.load(path, allow_pickle=True)
    corr    = z["corr_norm"]  if "corr_norm"  in z.files else None
    ncc_arr = z["topk_ncc"]   if "topk_ncc"   in z.files else None
    energy_d= z["energy_ds"]  if "energy_ds"  in z.files else None

    feats = {}
    feats |= corr_features(corr, topk=8)
    feats |= topk_ncc_features(ncc_arr)
    feats |= energy_ds_features(energy_d)

    meta = {}
    if "meta_json" in z.files:
        try:
            mj = z["meta_json"]
            if isinstance(mj, np.ndarray): mj = mj.item()
            if isinstance(mj, (bytes, bytearray)): meta = json.loads(mj.decode("utf-8", errors="ignore"))
            elif isinstance(mj, str): meta = json.loads(mj)
        except Exception: pass

    feats["meta_xc_best_peak"] = _safe_float(meta.get("xc_best_peak", np.nan))
    feats["meta_peak"]         = _safe_float(meta.get("peak",         np.nan))
    feats["meta_z"]            = _safe_float(meta.get("z",            np.nan))
    feats["meta_ncc_best"]     = _safe_float(meta.get("ncc_best",     np.nan))
    feats["meta_med"]          = _safe_float(meta.get("med",          np.nan))
    feats["meta_mad"]          = _safe_float(meta.get("mad",          np.nan))
    feats["meta_rx_gain"]      = _safe_float(meta.get("rx_gain",      np.nan))
    feats["meta_bps"]          = _safe_float(meta.get("bps",          np.nan))
    return feats, meta

# ── Dataset ──────────────────────────────────────────────────────────────────
def load_samples(run_dirs, neg_mod="qam16", neg_tag="fail_hard"):
    pos, neg = [], []
    for rd in run_dirs:
        for f in sorted(glob.glob(os.path.join(rd, "cap_*.npz"))):
            label = 1 if "_ok.npz" in f else 0
            feats, meta = load_npz_fast(f)
            tag = meta.get("save_tag", "ok" if label else "fail")
            mod = meta.get("modulation", "")
            if label == 0 and (mod != neg_mod or tag != neg_tag):
                continue  # skip non-target negatives
            feat_vec = np.array([v for k, v in sorted(feats.items())
                                 if not any(s in k for s in EXCLUDE_POST_DEMOD)],
                                dtype=np.float32)
            if label == 1:
                pos.append((feat_vec, 1))
            else:
                neg.append((feat_vec, 0))
    print(f"  loaded: {len(pos)} positives, {len(neg)} negatives "
          f"(neg_mod={neg_mod}, neg_tag={neg_tag})")
    return pos, neg

class SimpleDS(Dataset):
    def __init__(self, items, mu, sigma):
        self.items = items
        self.mu = mu; self.sigma = sigma
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        x, y = self.items[i]
        x = (x - self.mu) / (self.sigma + 1e-8)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.from_numpy(x).float(), torch.tensor(float(y))

# ── Train ─────────────────────────────────────────────────────────────────────
def train(args):
    run_dirs = []
    for pattern in args.run_dirs:
        run_dirs += sorted(glob.glob(pattern))
    run_dirs = [r for r in run_dirs if os.path.isdir(r)]
    print(f"[v3] {len(run_dirs)} run dirs")

    pos, neg = load_samples(run_dirs, args.neg_mod, args.neg_tag)
    if not pos or not neg:
        print("ERROR: empty pos or neg"); return

    # balance: up-sample minority
    if len(pos) > len(neg):
        neg = neg * (len(pos) // len(neg) + 1)
    elif len(neg) > len(pos):
        pos = pos * (len(neg) // len(pos) + 1)
    random.shuffle(neg); random.shuffle(pos)
    all_items = pos + neg
    random.shuffle(all_items)

    # train/val split
    n_val = max(1, int(0.2 * len(all_items)))
    val_items = all_items[:n_val]
    trn_items = all_items[n_val:]

    # normalisation stats from training set
    X_trn = np.stack([x for x, _ in trn_items])
    mu = X_trn.mean(0); sigma = X_trn.std(0)
    d_in = X_trn.shape[1]

    trn_ds = SimpleDS(trn_items, mu, sigma)
    val_ds = SimpleDS(val_items, mu, sigma)
    trn_ld = DataLoader(trn_ds, batch_size=128, shuffle=True)
    val_ld = DataLoader(val_ds, batch_size=256, shuffle=False)

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    print(f"[v3] d_in={d_in}  device={device}  n_trn={len(trn_items)} n_val={len(val_items)}")

    model = MLP(d_in, hidden=128).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    history = []
    for epoch in range(args.epochs):
        model.train()
        tr_loss = 0.0
        for xb, yb in trn_ld:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward(); opt.step()
            tr_loss += loss.item() * len(xb)
        tr_loss /= len(trn_items)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_ld:
                val_loss += criterion(model(xb.to(device)), yb.to(device)).item() * len(xb)
        val_loss /= len(val_items)
        history.append({"epoch": epoch+1, "train_loss": tr_loss, "val_loss": val_loss})
        if (epoch+1) % 10 == 0:
            print(f"  epoch {epoch+1:3d}  tr={tr_loss:.4f}  val={val_loss:.4f}")

    # Evaluation
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for xb, yb in val_ld:
            logits = model(xb.to(device))
            probs = torch.sigmoid(logits).cpu().numpy()
            all_pred.extend(probs.tolist())
            all_true.extend(yb.numpy().tolist())
    all_pred = np.array(all_pred); all_true = np.array(all_true)

    metrics = {}
    if SKLEARN:
        auc = roc_auc_score(all_true, all_pred)
        prec, rec, thr = precision_recall_curve(all_true, all_pred)
        # F1-optimal threshold
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        best_i = np.argmax(f1[:-1])
        best_thr = float(thr[best_i])
        metrics = {
            "roc_auc": float(auc),
            "best_threshold": best_thr,
            "best_precision": float(prec[best_i]),
            "best_recall": float(rec[best_i]),
            "best_f1": float(f1[best_i]),
            "n_val": len(all_true),
            "neg_mod": args.neg_mod, "neg_tag": args.neg_tag,
        }
        # save ROC curve data for comparison plot
        fpr, tpr, roc_thr = roc_curve(all_true, all_pred)
        metrics["roc_fpr"] = fpr.tolist()
        metrics["roc_tpr"] = tpr.tolist()
        print(f"[v3] AUC={auc:.4f}  best_thr={best_thr:.4f}  "
              f"prec={prec[best_i]:.3f}  rec={rec[best_i]:.3f}")

    os.makedirs(args.out_dir, exist_ok=True)
    # Save model
    #feature_names = sorted([k for k, _ in trn_items[0][0].reshape(-1).__class__.__mro__])
    schema = {"d_in": d_in, "hidden": 128, "threshold": metrics.get("best_threshold", 0.5),
              "mu": mu.tolist(), "sigma": sigma.tolist(),
              "exclude_post_demod": True, "neg_mod": args.neg_mod, "neg_tag": args.neg_tag}
    torch.save({"state_dict": model.state_dict(), "schema": schema},
               os.path.join(args.out_dir, "gate_model.pt"))
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fp:
        json.dump(metrics, fp, indent=2)
    with open(os.path.join(args.out_dir, "history.json"), "w") as fp:
        json.dump(history, fp, indent=2)
    print(f"[v3] saved → {args.out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dirs", nargs="+",
                    default=["rf_stream/ber_sweep_v3/run_*",
                             "rf_stream/ber_sweep_v2/run_*",
                             "rf_stream/ber_sweep/run_20260428_233037",
                             "rf_stream/ber_sweep/run_20260429_001722",
                             "rf_stream/ber_sweep/run_20260429_003513",
                             "rf_stream/ber_sweep/run_20260429_070949"])
    ap.add_argument("--neg_mod", default="qam16")
    ap.add_argument("--neg_tag", default="fail_hard")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--out_dir", default="rf_stream/gate_model_v3")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    train(args)
