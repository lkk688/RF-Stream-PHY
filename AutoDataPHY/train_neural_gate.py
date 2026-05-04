#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_neural_gate.py  (multi-mode)

Goal:
  Train a 2-class "Neural Gate" that predicts whether a capture window contains
  a real decodable packet (positive) or not (negative), to reduce false accepts
  before running heavy demod/CRC.

Supports 3 data modes automatically:
  A) Strong: cap_*_ok.npz + cap_*_fail.npz exist
  B) Weak : only captures.csv (+ optional ok npz). Uses CSV rows as labeled samples.
  C) Mixed: both exist, but fail npz might be sparse. It merges both sources.

Expected file layout under --run_dir:
  captures.csv
  cap_000123_ok.npz   (optional but recommended)
  cap_000456_fail.npz (optional)
  rx_config.json (optional; not required)

NPZ format (what this script expects):
  - corr_norm: float32 array (len ~ xcorr_search - len(stf)+1) or smaller
  - rxw: complex64 array (optional; if present we can compute energy features)
  - meta_json: bytes (optional)

If your fail NPZ contains only rxw/corr_norm/meta_json that's enough.

Features:
  - If NPZ corr_norm exists: use a compact summary of corr_norm (quantiles + topK).
  - If NPZ rxw exists: compute energy stats and "gate_ratio" like maxe/p10.
  - If only CSV: use CSV numeric columns (xc_best_peak, maxe, p10, peak, snr_db, etc.)
    (auto-detect columns).

Model:
  - Small MLP on standardized features.

Outputs:
  - gate_model.pt (Torch state_dict + feature schema)
  - thresholds.json (recommend threshold for target recall)
  - metrics.json + optional plots in out_dir

Usage:
  python3 train_neural_gate.py --run_dir rf_stream/ber_sweep/run_20260427_230640 --out_dir gate_out

Notes:
  - This is a "window-level" gate. Later you can extend to "candidate-level" gate by
    building per-candidate features (xc peak, local corr shape).
"""

import argparse
import glob
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# optional torch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# optional plotting
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    PLOT_OK = True
except Exception:
    PLOT_OK = False


# -------------------------
# Utilities
# -------------------------
def _safe_float(x, default=np.nan):
    try:
        if x is None:
            return default
        if isinstance(x, str) and x.strip() == "":
            return default
        return float(x)
    except Exception:
        return default


def _npz_label_from_name(path: str) -> Optional[int]:
    base = os.path.basename(path)
    if base.endswith("_ok.npz"):
        return 1
    if base.endswith("_fail.npz"):
        return 0
    return None


def _cap_id_from_name(path: str) -> Optional[int]:
    m = re.search(r"cap_(\d+)_", os.path.basename(path))
    if not m:
        return None
    return int(m.group(1))


def corr_features(corr: np.ndarray, topk: int = 8) -> Dict[str, float]:
    """Robust scalar features from corr_norm."""
    if corr is None or corr.size == 0:
        return {f"corr_q{q}": 0.0 for q in (10, 50, 90, 99)} | {f"corr_top{i}": 0.0 for i in range(topk)}
    c = np.asarray(corr, dtype=np.float32)
    qs = {
        "corr_q10": float(np.percentile(c, 10)),
        "corr_q50": float(np.percentile(c, 50)),
        "corr_q90": float(np.percentile(c, 90)),
        "corr_q99": float(np.percentile(c, 99)),
        "corr_max": float(np.max(c)),
        "corr_mean": float(np.mean(c)),
        "corr_std": float(np.std(c)),
    }
    # top-k peaks
    k = min(topk, c.size)
    idx = np.argpartition(c, -k)[-k:]
    vals = np.sort(c[idx])[::-1]
    for i in range(topk):
        qs[f"corr_top{i}"] = float(vals[i]) if i < vals.size else 0.0
    # peak-to-median ratio
    med = float(np.median(c) + 1e-12)
    qs["corr_p2m"] = float(np.max(c) / med)
    return qs


def topk_ncc_features(topk_ncc: np.ndarray) -> Dict[str, float]:
    """Features from the top-K NCC vector (step7 NPZ field)."""
    if topk_ncc is None or topk_ncc.size == 0:
        return {"ncc_max": 0.0, "ncc_mean": 0.0, "ncc_std": 0.0,
                "ncc_top0": 0.0, "ncc_top1": 0.0, "ncc_top2": 0.0}
    v = np.asarray(topk_ncc, dtype=np.float32)
    v_sorted = np.sort(v)[::-1]
    return {
        "ncc_max":  float(v_sorted[0]) if v_sorted.size > 0 else 0.0,
        "ncc_top0": float(v_sorted[0]) if v_sorted.size > 0 else 0.0,
        "ncc_top1": float(v_sorted[1]) if v_sorted.size > 1 else 0.0,
        "ncc_top2": float(v_sorted[2]) if v_sorted.size > 2 else 0.0,
        "ncc_mean": float(np.mean(v)),
        "ncc_std":  float(np.std(v)),
        "ncc_p2m":  float(v_sorted[0] / (float(np.median(v)) + 1e-12)),
    }


def energy_ds_features(energy_ds: np.ndarray) -> Dict[str, float]:
    """Features from downsampled energy trace (step7 energy_ds field, stride=16)."""
    if energy_ds is None or energy_ds.size == 0:
        return {"eds_p10": 0.0, "eds_p50": 0.0, "eds_p90": 0.0,
                "eds_max": 0.0, "eds_ratio": 0.0, "eds_z": 0.0}
    e = np.asarray(energy_ds, dtype=np.float32)
    p10 = float(np.percentile(e, 10))
    p50 = float(np.percentile(e, 50))
    p90 = float(np.percentile(e, 90))
    mx  = float(np.max(e))
    mad = float(np.median(np.abs(e - p50))) + 1e-12
    z   = float((mx - p50) / (1.4826 * mad))
    return {
        "eds_p10":   p10,
        "eds_p50":   p50,
        "eds_p90":   p90,
        "eds_max":   mx,
        "eds_ratio": mx / (p10 + 1e-12),
        "eds_z":     z,
    }


def energy_features_from_rxw(rxw: np.ndarray, win: int = 512) -> Dict[str, float]:
    """Compute energy stats similar to your online gate."""
    if rxw is None or rxw.size == 0:
        return {"eng_p10": 0.0, "eng_max": 0.0, "eng_ratio": 0.0, "eng_mean": 0.0, "eng_std": 0.0}
    x = np.asarray(rxw)
    p = (x.real * x.real + x.imag * x.imag).astype(np.float32)
    if p.size < win:
        e = p
    else:
        # simple moving average
        kernel = np.ones(win, dtype=np.float32) / float(win)
        e = np.convolve(p, kernel, mode="valid").astype(np.float32)
    p10 = float(np.percentile(e, 10))
    mx = float(np.max(e))
    return {
        "eng_p10": p10,
        "eng_max": mx,
        "eng_ratio": float(mx / (p10 + 1e-12)),
        "eng_mean": float(np.mean(e)),
        "eng_std": float(np.std(e)),
    }


def load_npz_features(path: str, energy_win: int = 512, corr_topk: int = 8) -> Tuple[Dict[str, float], Dict]:
    z = np.load(path, allow_pickle=True)
    corr     = z["corr_norm"]  if "corr_norm"  in z.files else None
    rxw      = z["rxw"]        if "rxw"        in z.files else None
    ncc_arr  = z["topk_ncc"]   if "topk_ncc"   in z.files else None
    energy_d = z["energy_ds"]  if "energy_ds"  in z.files else None

    feats = {}
    feats |= corr_features(corr, topk=corr_topk)
    feats |= energy_features_from_rxw(rxw, win=energy_win)
    feats |= topk_ncc_features(ncc_arr)
    feats |= energy_ds_features(energy_d)

    # optional meta_json
    meta = {}
    if "meta_json" in z.files:
        try:
            mj = z["meta_json"]
            if isinstance(mj, np.ndarray):
                mj = mj.item()
            if isinstance(mj, (bytes, bytearray)):
                meta = json.loads(mj.decode("utf-8", errors="ignore"))
            elif isinstance(mj, str):
                meta = json.loads(mj)
        except Exception:
            meta = {}

    # Step7 fields are top-level in meta (not nested under "diag")
    feats["meta_xc_best_peak"] = _safe_float(meta.get("xc_best_peak", np.nan))
    feats["meta_peak"]         = _safe_float(meta.get("peak",         np.nan))
    feats["meta_probe_evm"]    = _safe_float(meta.get("probe_evm",    np.nan))
    feats["meta_snr_db"]       = _safe_float(meta.get("snr_db",       np.nan))
    feats["meta_cfo_hz"]       = _safe_float(meta.get("cfo_hz",       np.nan))
    # step7-specific
    feats["meta_z"]            = _safe_float(meta.get("z",            np.nan))
    feats["meta_ncc_best"]     = _safe_float(meta.get("ncc_best",     np.nan))
    feats["meta_med"]          = _safe_float(meta.get("med",          np.nan))
    feats["meta_mad"]          = _safe_float(meta.get("mad",          np.nan))
    feats["meta_rx_gain"]      = _safe_float(meta.get("rx_gain",      np.nan))
    feats["meta_bps"]          = _safe_float(meta.get("bps",          np.nan))
    return feats, meta


def read_captures_csv(csv_path: str) -> List[Dict]:
    import csv
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def csv_row_to_features(row: Dict) -> Dict[str, float]:
    """
    Weak-mode features extracted directly from captures.csv.
    Supports both step6 and step7 column sets.
    """
    keys_numeric = [
        "peak", "p10", "eg_th", "maxe",
        "xc_best_peak", "xc_best_idx",
        "probe_evm", "cfo_hz", "snr_db",
        "rx_gain", "bps",
        # step7 additions
        "med", "mad", "z", "ncc_best", "ncc_best_idx",
    ]
    feats = {}
    for k in keys_numeric:
        if k in row:
            feats[f"csv_{k}"] = _safe_float(row.get(k), np.nan)
    # derived: gate ratio
    mx  = _safe_float(row.get("maxe"), np.nan)
    p10 = _safe_float(row.get("p10"),  np.nan)
    if np.isfinite(mx) and np.isfinite(p10):
        feats["csv_gate_ratio"] = float(mx / (p10 + 1e-12))
    xc = _safe_float(row.get("xc_best_peak"), np.nan)
    if np.isfinite(xc) and np.isfinite(p10):
        feats["csv_xc_over_p10"] = float(xc / (p10 + 1e-12))
    # step7: MAD z-score and NCC
    z_val = _safe_float(row.get("z"), np.nan)
    ncc   = _safe_float(row.get("ncc_best"), np.nan)
    med_v = _safe_float(row.get("med"), np.nan)
    mad_v = _safe_float(row.get("mad"), np.nan)
    if np.isfinite(z_val):
        feats["csv_z_log"] = float(np.log1p(max(z_val, 0.0)))
    if np.isfinite(ncc):
        feats["csv_ncc_best"] = ncc
    if np.isfinite(med_v) and np.isfinite(mad_v) and mad_v > 0:
        feats["csv_snr_mad"] = float(med_v / (mad_v + 1e-12))
    return feats


def csv_row_to_label(row: Dict) -> Optional[int]:
    """
    Weak label:
      ok => 1
      anything else (no_crc/skip) => 0
    """
    s = (row.get("status", "") or "").strip().lower()
    if s == "ok":
        return 1
    if s in ("no_crc", "skip", "no", "fail", "bad"):
        return 0
    if s == "":
        return None
    # default: treat non-ok as negative
    return 0


# -------------------------
# Dataset
# -------------------------
@dataclass
class Sample:
    x: np.ndarray
    y: int


class GateDataset(Dataset):
    def __init__(self, samples: List[Sample]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i: int):
        s = self.samples[i]
        return torch.from_numpy(s.x).float(), torch.tensor(s.y).float()


# -------------------------
# Model
# -------------------------
class MLP(nn.Module):
    def __init__(self, d_in: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# -------------------------
# Training / Metrics
# -------------------------
def split_train_val(samples: List[Sample], val_frac: float = 0.2, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(samples))
    rng.shuffle(idx)
    n_val = int(len(samples) * val_frac)
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]
    tr = [samples[i] for i in tr_idx]
    va = [samples[i] for i in val_idx]
    return tr, va


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def compute_pr_roc(y_true: np.ndarray, y_score: np.ndarray):
    # simple sweep over thresholds
    order = np.argsort(y_score)[::-1]
    y_true = y_true[order]
    y_score = y_score[order]
    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)
    tp = 0
    fp = 0
    tps = []
    fps = []
    ths = []
    last = None
    for i in range(len(y_score)):
        s = y_score[i]
        if last is None or s != last:
            tps.append(tp)
            fps.append(fp)
            ths.append(s)
            last = s
        if y_true[i] == 1:
            tp += 1
        else:
            fp += 1
    tps.append(tp); fps.append(fp); ths.append(y_score[-1])
    tps = np.array(tps, dtype=np.float32)
    fps = np.array(fps, dtype=np.float32)
    ths = np.array(ths, dtype=np.float32)
    # roc
    tpr = tps / (P + 1e-12)
    fpr = fps / (N + 1e-12)
    # precision/recall
    prec = tps / (tps + fps + 1e-12)
    rec = tpr
    return {"ths": ths, "tpr": tpr, "fpr": fpr, "prec": prec, "rec": rec, "P": int(P), "N": int(N)}


def recommend_threshold(y_true: np.ndarray, y_prob: np.ndarray, target_recall: float = 0.98):
    m = compute_pr_roc(y_true, y_prob)
    rec = m["rec"]
    prec = m["prec"]
    ths = m["ths"]

    ok = np.where(rec >= target_recall)[0]
    if ok.size == 0:
        # fallback: best F1
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        j = int(np.argmax(f1))
        return float(ths[j]), {"mode": "best_f1", "prec": float(prec[j]), "rec": float(rec[j])}

    # among those, choose highest precision
    j = ok[np.argmax(prec[ok])]
    return float(ths[j]), {"mode": "target_recall", "prec": float(prec[j]), "rec": float(rec[j])}


def train_loop(model, tr_loader, va_loader, lr=1e-3, epochs=20, device="cpu"):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    best = {"va_loss": 1e9, "state": None}
    history = []

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        n = 0
        for xb, yb in tr_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            tr_loss += float(loss.item()) * xb.size(0)
            n += xb.size(0)
        tr_loss /= max(n, 1)

        model.eval()
        va_loss = 0.0
        n = 0
        all_logits = []
        all_y = []
        with torch.no_grad():
            for xb, yb in va_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = loss_fn(logits, yb)
                va_loss += float(loss.item()) * xb.size(0)
                n += xb.size(0)
                all_logits.append(logits.cpu().numpy())
                all_y.append(yb.cpu().numpy())
        va_loss /= max(n, 1)

        y_true = np.concatenate(all_y).astype(np.int32)
        y_prob = sigmoid(np.concatenate(all_logits))
        thr, info = recommend_threshold(y_true, y_prob, target_recall=0.98)

        history.append({
            "epoch": ep,
            "tr_loss": tr_loss,
            "va_loss": va_loss,
            "thr": thr,
            "thr_info": info,
        })
        print(f"[ep {ep:02d}] tr_loss={tr_loss:.4f} va_loss={va_loss:.4f} thr@rec98={thr:.4f} "
              f"(prec={info['prec']:.3f} rec={info['rec']:.3f})")

        if va_loss < best["va_loss"]:
            best["va_loss"] = va_loss
            best["state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return history


# -------------------------
# Feature alignment + standardization
# -------------------------
def build_matrix(feature_dicts: List[Dict[str, float]], feature_names: List[str]) -> np.ndarray:
    X = np.zeros((len(feature_dicts), len(feature_names)), dtype=np.float32)
    for i, d in enumerate(feature_dicts):
        for j, k in enumerate(feature_names):
            v = d.get(k, np.nan)
            X[i, j] = np.float32(v) if np.isfinite(v) else np.nan
    return X


def impute_and_standardize(X: np.ndarray):
    # impute NaN with column median
    X2 = X.copy()
    med = np.nanmedian(X2, axis=0)
    inds = np.where(~np.isfinite(X2))
    X2[inds] = np.take(med, inds[1])
    # standardize
    mu = np.mean(X2, axis=0)
    sd = np.std(X2, axis=0) + 1e-6
    Xn = (X2 - mu) / sd
    return Xn.astype(np.float32), mu.astype(np.float32), sd.astype(np.float32), med.astype(np.float32)


# -------------------------
# Main loader (multi-mode)
# -------------------------
def gather_samples(
    run_dirs: List[str],
    energy_win: int,
    corr_topk: int,
    use_csv: bool = True,
    max_npz_per_dir: int = 0,
) -> Tuple[List[Dict], List[int]]:
    """
    Return (feature_dicts, labels).
    Accepts a list of run directories; mixes NPZ (strong) + CSV (weak).
    max_npz_per_dir: if > 0, cap NPZ per label per dir to avoid huge-run dominance.
    """
    feats: List[Dict] = []
    labels: List[int] = []

    for run_dir in run_dirs:
        run_dir = os.path.abspath(run_dir)
        ok_paths   = sorted(glob.glob(os.path.join(run_dir, "cap_*_ok.npz")))
        fail_paths = sorted(glob.glob(os.path.join(run_dir, "cap_*_fail.npz")))

        if ok_paths or fail_paths:
            # subsample if requested (keeps large runs from dominating)
            if max_npz_per_dir > 0:
                rng = np.random.default_rng(seed=42)
                if len(ok_paths) > max_npz_per_dir:
                    ok_paths = list(rng.choice(ok_paths, max_npz_per_dir, replace=False))
                if len(fail_paths) > max_npz_per_dir:
                    fail_paths = list(rng.choice(fail_paths, max_npz_per_dir, replace=False))
            npz_paths = ok_paths + fail_paths
            print(f"[load] {os.path.basename(run_dir)}: {len(ok_paths)} ok + {len(fail_paths)} fail NPZ")
            for p in npz_paths:
                y = _npz_label_from_name(p)
                if y is None:
                    continue
                try:
                    f, _ = load_npz_features(p, energy_win=energy_win, corr_topk=corr_topk)
                except Exception as exc:
                    print(f"  [warn] skip {os.path.basename(p)}: {exc}")
                    continue
                f["src_npz"] = 1.0
                f["src_csv"] = 0.0
                feats.append(f)
                labels.append(int(y))

        # CSV weak samples
        csv_path = os.path.join(run_dir, "captures.csv")
        if use_csv and os.path.exists(csv_path):
            rows = read_captures_csv(csv_path)
            n_before = len(feats)
            for r in rows:
                y = csv_row_to_label(r)
                if y is None:
                    continue
                f = csv_row_to_features(r)
                if not f:
                    continue
                f["src_npz"] = 0.0
                f["src_csv"] = 1.0
                feats.append(f)
                labels.append(int(y))
            print(f"[load] {os.path.basename(run_dir)}: {len(feats)-n_before} CSV rows")

    if not feats:
        raise RuntimeError("No samples found across any run_dir.")

    pos = sum(labels)
    print(f"[data] total {len(labels)} samples: pos={pos} neg={len(labels)-pos}")
    return feats, labels


def main():
    ap = argparse.ArgumentParser("Train Neural Gate (multi-mode, multi-dir)")
    ap.add_argument("--run_dirs", nargs="+",
                    help="One or more run directories (captures.csv and/or cap_*_ok/fail.npz)")
    ap.add_argument("--run_dir", help="Single run directory (backward-compat alias for --run_dirs)")
    ap.add_argument("--out_dir", default="gate_out")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--energy_win", type=int, default=512)
    ap.add_argument("--corr_topk", type=int, default=8)
    ap.add_argument("--no_csv",       action="store_true", help="Disable weak CSV samples (NPZ-only)")
    ap.add_argument("--max_npz_per_dir", type=int, default=2000,
                    help="Cap OK and FAIL NPZ per dir to avoid large-run dominance (0=unlimited)")
    ap.add_argument("--target_recall", type=float, default=0.98)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--exclude_post_demod", action="store_true",
                    help="Drop probe_evm/cfo_hz/snr_db features (unavailable during pre-demod online inference)")
    ap.add_argument("--exclude_features", nargs="*", default=[],
                    help="Additional feature name substrings to exclude")
    ap.add_argument("--device", default="auto",
                    help="Torch device: auto | cuda | cpu (default: auto)")
    args = ap.parse_args()

    # Resolve run directories
    run_dirs: List[str] = []
    if args.run_dirs:
        run_dirs.extend(args.run_dirs)
    if args.run_dir:
        run_dirs.append(args.run_dir)
    if not run_dirs:
        ap.error("Provide --run_dirs or --run_dir")

    os.makedirs(args.out_dir, exist_ok=True)

    feat_dicts, y = gather_samples(
        run_dirs=run_dirs,
        energy_win=args.energy_win,
        corr_topk=args.corr_topk,
        use_csv=(not args.no_csv),
        max_npz_per_dir=args.max_npz_per_dir,
    )

    # Build union of feature names
    feature_names = sorted({k for d in feat_dicts for k in d.keys()})

    # Drop features unavailable during online pre-demod inference
    exclude_substrings: List[str] = list(args.exclude_features or [])
    if args.exclude_post_demod:
        exclude_substrings += ["probe_evm", "cfo_hz", "snr_db"]
    if exclude_substrings:
        before = len(feature_names)
        feature_names = [f for f in feature_names
                         if not any(s in f for s in exclude_substrings)]
        print(f"[feat] excluded {before - len(feature_names)} features matching {exclude_substrings} "
              f"→ {len(feature_names)} remain")

    X = build_matrix(feat_dicts, feature_names)
    Xn, mu, sd, med = impute_and_standardize(X)

    samples = [Sample(x=Xn[i], y=int(y[i])) for i in range(len(y))]
    tr, va = split_train_val(samples, val_frac=args.val_frac, seed=args.seed)
    print(f"[data] samples={len(samples)} train={len(tr)} val={len(va)} pos={sum(s.y for s in samples)}")

    tr_loader = DataLoader(GateDataset(tr), batch_size=args.batch, shuffle=True, drop_last=False)
    va_loader = DataLoader(GateDataset(va), batch_size=args.batch, shuffle=False, drop_last=False)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"[device] using {device}  (cuda_available={torch.cuda.is_available()})")
    model = MLP(d_in=Xn.shape[1], hidden=args.hidden, dropout=args.dropout).to(device)

    hist = train_loop(model, tr_loader, va_loader, lr=args.lr, epochs=args.epochs, device=device)

    # Evaluate on val
    model.eval()
    all_logits = []
    all_y = []
    with torch.no_grad():
        for xb, yb in va_loader:
            xb = xb.to(device)
            logits = model(xb).cpu().numpy()
            all_logits.append(logits)
            all_y.append(yb.numpy())
    y_true = np.concatenate(all_y).astype(np.int32)
    y_prob = sigmoid(np.concatenate(all_logits))
    thr, info = recommend_threshold(y_true, y_prob, target_recall=args.target_recall)

    metrics = compute_pr_roc(y_true, y_prob)
    out = {
        "run_dir": args.run_dir,
        "n_val": int(len(y_true)),
        "pos_val": int(np.sum(y_true == 1)),
        "neg_val": int(np.sum(y_true == 0)),
        "target_recall": args.target_recall,
        "recommended_threshold": thr,
        "thr_info": info,
    }

    # Save model + schema
    save_obj = {
        "state_dict": model.state_dict(),
        "feature_names": feature_names,
        "mu": torch.from_numpy(mu),
        "sd": torch.from_numpy(sd),
        "median": torch.from_numpy(med),
        "recommended_threshold": thr,
        "thr_info": info,
    }
    model_path = os.path.join(args.out_dir, "gate_model.pt")
    torch.save(save_obj, model_path)

    with open(os.path.join(args.out_dir, "thresholds.json"), "w") as f:
        json.dump(out, f, indent=2)

    with open(os.path.join(args.out_dir, "history.json"), "w") as f:
        json.dump(hist, f, indent=2)

    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump({
            **out,
            "roc_auc_approx": float(np.trapz(metrics["tpr"], metrics["fpr"])),
        }, f, indent=2)

    print(f"\n[save] model     : {model_path}")
    print(f"[save] thresholds: {os.path.join(args.out_dir,'thresholds.json')}")
    print(f"[save] metrics   : {os.path.join(args.out_dir,'metrics.json')}")
    print(f"       => recommended_threshold={thr:.4f}  prec={info['prec']:.3f}  rec={info['rec']:.3f}")

    # Optional plots
    if args.plot and PLOT_OK:
        # Loss curves
        fig = plt.figure(figsize=(8, 4))
        plt.plot([h["epoch"] for h in hist], [h["tr_loss"] for h in hist], label="train")
        plt.plot([h["epoch"] for h in hist], [h["va_loss"] for h in hist], label="val")
        plt.grid(True); plt.legend(); plt.title("Loss")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "loss.png"), dpi=140)
        plt.close(fig)

        # ROC
        fig = plt.figure(figsize=(5, 5))
        plt.plot(metrics["fpr"], metrics["tpr"])
        plt.plot([0, 1], [0, 1], "--")
        plt.grid(True); plt.title("ROC"); plt.xlabel("FPR"); plt.ylabel("TPR")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "roc.png"), dpi=140)
        plt.close(fig)

        # PR
        fig = plt.figure(figsize=(5, 5))
        plt.plot(metrics["rec"], metrics["prec"])
        plt.grid(True); plt.title("PR"); plt.xlabel("Recall"); plt.ylabel("Precision")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "pr.png"), dpi=140)
        plt.close(fig)

        # Score hist by label
        fig = plt.figure(figsize=(8, 4))
        plt.hist(y_prob[y_true == 1], bins=50, alpha=0.6, label="pos(ok)")
        plt.hist(y_prob[y_true == 0], bins=50, alpha=0.6, label="neg(non-ok)")
        plt.axvline(thr, linestyle="--", label=f"thr={thr:.3f}")
        plt.grid(True); plt.legend(); plt.title("Gate score distribution (val)")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "score_hist.png"), dpi=140)
        plt.close(fig)

        print(f"[plot] saved plots to {args.out_dir}")

if __name__ == "__main__":
    main()

"""
#current data
python3 rf_stream/train_neural_gate.py \
  --run_dir rf_stream/ber_sweep/run_20260427_230640 \
  --out_dir gate_qpsk_csv \
  --plot
"""
