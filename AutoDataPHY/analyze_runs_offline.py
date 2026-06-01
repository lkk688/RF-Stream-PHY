#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_runs_offline.py

Offline analyzer for rf_stream Step6/7 runs.

Expected run_dir structure:
  run_dir/
    captures.csv
    rx_config.json (optional)
    run_summary.json (optional)
    cap_000001_ok.npz  (many)
    cap_000002_fail.npz  (step7 hard-negatives)
    good_packets/...

Step7 additions to captures.csv:
  med, mad, z          -- MAD z-score gate features
  ncc_best, ncc_best_idx -- NCC gate features

This script:
  1) Loads captures.csv
  2) Computes derived features for debugging false detections
  3) Generates a figure pack (PNG) including:
     - status/reason counts and time series (grid overview)
     - step7 gate features: z, ncc_best distributions and scatter
     - BER-per-gain and decode-rate-per-gain curves
     - distributions ok vs fail
     - ROC/PR curves for gate candidates (incl. z, ncc_best)
     - correlation-shape overlay for OK packets
  4) Optionally compares multiple run dirs side-by-side.

Usage examples:
  # Single run
  python3 analyze_runs_offline.py \
    --run_dirs rf_stream/ber_sweep/run_20260428_233037 \
    --out_dir rf_stream/ber_sweep/_offline_plots_qpsk_step7

  # Compare QPSK vs QAM16
  python3 analyze_runs_offline.py \
    --run_dirs rf_stream/ber_sweep/run_20260428_233037 rf_stream/ber_sweep/run_QAM16_RUN \
    --out_dir rf_stream/ber_sweep/_offline_plots_compare

  # Background noise only
  python3 analyze_runs_offline.py \
    --run_dirs rf_stream/bg_noise/run_20260428_232452 \
    --out_dir rf_stream/bg_noise/_plots
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# Helpers
# ----------------------------
def safe_mad(x: np.ndarray) -> float:
    x = np.asarray(x)
    if x.size == 0:
        return 0.0
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(mad)

def robust_z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    med = np.nanmedian(x)
    mad = safe_mad(x)
    if mad < 1e-12:
        return (x - med) * 0.0
    return (x - med) / (1.4826 * mad)

def read_json_if_exists(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def figsave(fig, out_path: str):
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  -> {os.path.basename(out_path)}")

def parse_meta_bytes(meta_json_obj) -> Optional[dict]:
    try:
        if meta_json_obj is None:
            return None
        if isinstance(meta_json_obj, (bytes, bytearray)):
            return json.loads(meta_json_obj.decode("utf-8"))
        if hasattr(meta_json_obj, "tobytes"):
            bb = meta_json_obj.tobytes()
            return json.loads(bb.decode("utf-8"))
    except Exception:
        return None
    return None


# ----------------------------
# Loading run
# ----------------------------
@dataclass
class RunData:
    run_dir: str
    captures: pd.DataFrame
    cfg: Optional[dict]
    summary: Optional[dict]
    ok_npz_paths: List[str]
    fail_npz_paths: List[str]

def load_run(run_dir: str) -> RunData:
    run_dir = os.path.abspath(run_dir)
    cap_csv = os.path.join(run_dir, "captures.csv")
    if not os.path.exists(cap_csv):
        raise FileNotFoundError(f"captures.csv not found: {cap_csv}")

    df = pd.read_csv(cap_csv)

    # Step6 + Step7 columns; fill missing with NaN
    needed_cols = [
        "cap", "status", "reason",
        "peak", "p10", "eg_th", "maxe",
        "med", "mad", "z",                      # step7 energy gate
        "xc_best_peak", "xc_best_idx",
        "ncc_best", "ncc_best_idx",              # step7 NCC gate
        "stf_idx", "ltf_start", "payload_start",
        "probe_evm", "cfo_hz", "snr_db",
        "seq", "payload_len", "modulation", "bps", "rx_gain",
        "ber", "n_bits", "n_bit_errors",
    ]
    for c in needed_cols:
        if c not in df.columns:
            df[c] = np.nan

    if "cap" in df.columns:
        df["cap"] = pd.to_numeric(df["cap"], errors="coerce").fillna(-1).astype(int)

    cfg     = read_json_if_exists(os.path.join(run_dir, "rx_config.json"))
    summary = read_json_if_exists(os.path.join(run_dir, "run_summary.json"))

    ok_npz_paths   = sorted(glob.glob(os.path.join(run_dir, "cap_*_ok.npz")))
    fail_npz_paths = sorted(glob.glob(os.path.join(run_dir, "cap_*_fail.npz")))
    return RunData(
        run_dir=run_dir, captures=df, cfg=cfg, summary=summary,
        ok_npz_paths=ok_npz_paths, fail_npz_paths=fail_npz_paths,
    )


# ----------------------------
# Feature engineering
# ----------------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    d["is_ok"] = (d["status"].astype(str) == "ok")

    eps = 1e-12
    d["gate_ratio"]    = d["maxe"] / (d["eg_th"] + eps)
    d["gate_margin"]   = d["maxe"] - d["eg_th"]
    d["gate_ratio_p10"] = d["maxe"] / (d["p10"] + eps)

    # Numeric casts for step7 columns
    for col in ["z", "ncc_best", "med", "mad"]:
        d[f"{col}_num"] = pd.to_numeric(d[col], errors="coerce")

    # Robust z-scores
    z_score_cols = [
        "xc_best_peak", "gate_ratio", "gate_margin",
        "probe_evm", "snr_db", "cfo_hz", "peak", "maxe",
        "z", "ncc_best",
    ]
    for col in z_score_cols:
        if col in d.columns:
            arr = pd.to_numeric(d[col], errors="coerce").to_numpy(dtype=np.float64)
            d[f"{col}_z"] = robust_z(np.nan_to_num(arr, nan=np.nanmedian(arr[~np.isnan(arr)]) if np.any(~np.isnan(arr)) else 0.0))

    d["abs_cfo"]          = np.abs(pd.to_numeric(d["cfo_hz"],      errors="coerce"))
    d["snr_db_num"]       = pd.to_numeric(d["snr_db"],      errors="coerce")
    d["probe_evm_num"]    = pd.to_numeric(d["probe_evm"],   errors="coerce")
    d["xc_best_peak_num"] = pd.to_numeric(d["xc_best_peak"], errors="coerce")
    d["rx_gain_num"]      = pd.to_numeric(d["rx_gain"],     errors="coerce")
    d["n_bits_num"]       = pd.to_numeric(d["n_bits"],       errors="coerce")
    d["n_bit_errors_num"] = pd.to_numeric(d["n_bit_errors"], errors="coerce")

    return d


# ----------------------------
# ROC/PR utilities
# ----------------------------
def roc_curve(scores: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    order  = np.argsort(-scores)
    scores = scores[order]; labels = labels[order]
    P = np.sum(labels == 1); N = np.sum(labels == 0)
    if P == 0 or N == 0:
        return np.array([0, 1]), np.array([0, 1]), np.array([np.inf, -np.inf])
    tps = np.cumsum(labels == 1); fps = np.cumsum(labels == 0)
    return fps / (N + 1e-12), tps / (P + 1e-12), scores.copy()

def pr_curve(scores: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    order  = np.argsort(-scores)
    scores = scores[order]; labels = labels[order]
    P = np.sum(labels == 1)
    if P == 0:
        return np.array([1.0]), np.array([0.0]), np.array([np.inf])
    tps  = np.cumsum(labels == 1); fps = np.cumsum(labels == 0)
    prec = tps / (tps + fps + 1e-12)
    rec  = tps / (P + 1e-12)
    return prec, rec, scores.copy()


# ----------------------------
# Plot: grid overview
# ----------------------------
def plot_grid_overview(df: pd.DataFrame, out_path: str, title: str):
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    fig.suptitle(title, fontsize=13)

    ok  = df[df["is_ok"] == True]
    bad = df[df["is_ok"] == False]

    # Row 0
    ax = axes[0, 0]
    df["status"].value_counts().head(15).plot(kind="bar", ax=ax)
    ax.set_title("Status counts"); ax.tick_params(axis="x", rotation=30)

    ax = axes[0, 1]
    df["reason"].value_counts().head(15).plot(kind="bar", ax=ax)
    ax.set_title("Reason counts"); ax.tick_params(axis="x", rotation=30)

    ax = axes[0, 2]
    ax.plot(df["cap"], pd.to_numeric(df["peak"], errors="coerce"), linewidth=0.6)
    ax.set_title("peak vs cap"); ax.set_xlabel("cap"); ax.grid(True)

    ax = axes[0, 3]
    ax.plot(df["cap"], pd.to_numeric(df["maxe"], errors="coerce"), label="maxe", linewidth=0.6)
    ax.plot(df["cap"], pd.to_numeric(df["eg_th"], errors="coerce"), label="eg_th", linewidth=0.6)
    ax.plot(df["cap"], pd.to_numeric(df["p10"],  errors="coerce"), label="p10",  linewidth=0.6)
    ax.legend(fontsize=8); ax.set_title("energy terms vs cap"); ax.set_xlabel("cap"); ax.grid(True)

    # Row 1 – step7 gate features
    ax = axes[1, 0]
    z_all = df["z_num"].dropna()
    if len(z_all) > 0:
        ax.plot(df["cap"], df["z_num"], linewidth=0.4, alpha=0.6)
        ax.set_yscale("symlog", linthresh=1.0)
    ax.set_title("z (MAD z-score) vs cap"); ax.set_xlabel("cap"); ax.grid(True)

    ax = axes[1, 1]
    zok = ok["z_num"].dropna(); zbd = bad["z_num"].dropna()
    if len(zok) > 0:
        ax.hist(np.log10(np.clip(zok, 1e-3, None)), bins=60, alpha=0.6, label=f"ok (n={len(zok)})")
    if len(zbd) > 0:
        ax.hist(np.log10(np.clip(zbd, 1e-3, None)), bins=60, alpha=0.6, label=f"not_ok (n={len(zbd)})")
    ax.set_title("log10(z) distribution"); ax.legend(fontsize=8); ax.grid(True)

    ax = axes[1, 2]
    nok = ok["ncc_best_num"].dropna(); nbd = bad["ncc_best_num"].dropna()
    if len(nok) > 0:
        ax.hist(nok, bins=60, alpha=0.6, label=f"ok (n={len(nok)})")
    if len(nbd) > 0:
        ax.hist(nbd, bins=60, alpha=0.6, label=f"not_ok (n={len(nbd)})")
    ax.set_title("ncc_best distribution"); ax.legend(fontsize=8); ax.grid(True)

    ax = axes[1, 3]
    xok = ok["xc_best_peak_num"].dropna(); xbd = bad["xc_best_peak_num"].dropna()
    if len(xok) > 0:
        ax.hist(xok, bins=60, alpha=0.6, label="ok")
    if len(xbd) > 0:
        ax.hist(xbd, bins=60, alpha=0.6, label="not_ok")
    ax.set_title("xc_best_peak distribution"); ax.legend(fontsize=8); ax.grid(True)

    # Row 2
    ax = axes[2, 0]
    xok = ok["snr_db_num"].dropna(); xbd = bad["snr_db_num"].dropna()
    if len(xok) > 0:
        ax.hist(xok, bins=50, alpha=0.6, label="ok")
    if len(xbd) > 0:
        ax.hist(xbd, bins=50, alpha=0.6, label="not_ok")
    ax.set_title("snr_db distribution"); ax.legend(fontsize=8); ax.grid(True)

    ax = axes[2, 1]
    ax.scatter(pd.to_numeric(df["snr_db"], errors="coerce"),
               pd.to_numeric(df["probe_evm"], errors="coerce"),
               s=5, alpha=0.25)
    ax.set_xlabel("snr_db"); ax.set_ylabel("probe_evm")
    ax.set_title("EVM vs SNR"); ax.grid(True)

    ax = axes[2, 2]
    st   = df["status"].astype(str).fillna("na")
    uniq = list(dict.fromkeys(st.tolist()))
    m    = {s: i for i, s in enumerate(uniq)}
    y    = np.array([m[s] for s in st], dtype=int)
    ax.plot(df["cap"], y, linewidth=0.8)
    ax.set_yticks(list(m.values())); ax.set_yticklabels(list(m.keys()))
    ax.set_title("status timeline"); ax.grid(True)

    ax = axes[2, 3]
    ax.plot(df["cap"], pd.to_numeric(df["abs_cfo"], errors="coerce"), linewidth=0.5)
    ax.set_title("|cfo_hz| vs cap"); ax.set_xlabel("cap"); ax.grid(True)

    figsave(fig, out_path)


# ----------------------------
# Plot: BER per gain
# ----------------------------
def plot_ber_per_gain(df: pd.DataFrame, summary: Optional[dict],
                      out_path: str, title: str):
    ok_df = df[df["is_ok"] == True].copy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(title, fontsize=13)

    gains = sorted(df["rx_gain_num"].dropna().unique())

    # ─ BER vs rx_gain ────────────────────────────────────────────────────────
    ax = axes[0]
    bers, g_plot = [], []
    for g in gains:
        sub    = ok_df[ok_df["rx_gain_num"] == g]
        nbits  = sub["n_bits_num"].sum()
        nerr   = sub["n_bit_errors_num"].sum()
        ber    = float(nerr / nbits) if nbits > 0 else np.nan
        bers.append(ber); g_plot.append(g)

    # Also overlay from run_summary if available
    if summary and "ber_per_gain" in summary:
        sg, sb = [], []
        for gk, v in summary["ber_per_gain"].items():
            b = v.get("ber")
            if b is not None and b > 0:
                sg.append(float(gk)); sb.append(b)
        if sg:
            idx = np.argsort(sg)
            ax.semilogy(np.array(sg)[idx], np.array(sb)[idx],
                        "s--", linewidth=1.5, alpha=0.7, label="summary.json")

    valid = [(g, b) for g, b in zip(g_plot, bers) if not np.isnan(b)]
    if valid:
        gv, bv = zip(*valid)
        ax.scatter(gv, [max(b, 1e-6) for b in bv], s=60, zorder=5)
        nonzero = [(g, b) for g, b in zip(gv, bv) if b > 0]
        if nonzero:
            g2, b2 = zip(*nonzero)
            ax.semilogy(g2, b2, "o-", linewidth=2, label="from CSV")
    ax.set_xlabel("rx_gain (dB)"); ax.set_ylabel("BER")
    ax.set_title("BER vs RX Gain"); ax.grid(True, which="both", alpha=0.4)
    ax.legend(fontsize=8)

    # ─ Decode rate vs rx_gain ─────────────────────────────────────────────────
    ax = axes[1]
    by_gain = df.groupby("rx_gain_num")
    dr = by_gain["is_ok"].mean()
    ct = by_gain.size()
    bars = ax.bar(dr.index, dr.values, width=2.5, color="steelblue", alpha=0.7)
    for bar, (g, n) in zip(bars, ct.items()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"n={n}", ha="center", va="bottom", fontsize=7)
    ax.set_xlabel("rx_gain (dB)"); ax.set_ylabel("decode rate")
    ax.set_title("Decode Rate vs RX Gain"); ax.set_ylim(0, 1.1); ax.grid(True, axis="y")

    # ─ Median SNR vs rx_gain ──────────────────────────────────────────────────
    ax = axes[2]
    snr_med = ok_df.groupby("rx_gain_num")["snr_db_num"].agg(["median", "std"])
    evm_med = ok_df.groupby("rx_gain_num")["probe_evm_num"].agg("median")
    if not snr_med.empty:
        ax.errorbar(snr_med.index, snr_med["median"], yerr=snr_med["std"],
                    fmt="o-", linewidth=2, capsize=4, label="SNR (dB)")
        ax.set_xlabel("rx_gain (dB)"); ax.set_ylabel("Median SNR (dB)")
        ax2 = ax.twinx()
        ax2.plot(evm_med.index, evm_med.values, "s--", color="orange",
                 linewidth=1.5, label="EVM")
        ax2.set_ylabel("Median EVM", color="orange")
    ax.set_title("SNR & EVM vs RX Gain"); ax.grid(True)
    lines1, lbls1 = ax.get_legend_handles_labels()
    lines2, lbls2 = (ax2.get_legend_handles_labels() if not snr_med.empty else ([], []))
    ax.legend(lines1 + lines2, lbls1 + lbls2, fontsize=8)

    figsave(fig, out_path)


# ----------------------------
# Plot: step7 gate features
# ----------------------------
def plot_step7_features(df: pd.DataFrame, out_path: str, title: str):
    has_z   = "z_num" in df.columns and df["z_num"].notna().sum() > 10
    has_ncc = "ncc_best_num" in df.columns and df["ncc_best_num"].notna().sum() > 10
    if not has_z and not has_ncc:
        print(f"  (skipping step7 features — no z/ncc data in this run)")
        return

    ok  = df[df["is_ok"] == True]
    bad = df[df["is_ok"] == False]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(title, fontsize=13)

    # z vs cap
    ax = axes[0, 0]
    ax.plot(df["cap"], df["z_num"], linewidth=0.4, alpha=0.5, color="steelblue")
    if not ok.empty:
        ax.scatter(ok["cap"], ok["z_num"], s=6, color="green",  alpha=0.4, label="ok",  zorder=3)
    if not bad.empty:
        ax.scatter(bad["cap"], bad["z_num"], s=6, color="red",    alpha=0.2, label="skip", zorder=3)
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_title("z score time series"); ax.set_xlabel("cap"); ax.grid(True); ax.legend(fontsize=8)

    # ncc_best vs cap
    ax = axes[0, 1]
    ax.plot(df["cap"], df["ncc_best_num"], linewidth=0.4, alpha=0.5, color="darkorange")
    if not ok.empty:
        ax.scatter(ok["cap"], ok["ncc_best_num"], s=6, color="green", alpha=0.4, label="ok", zorder=3)
    ax.set_title("ncc_best time series"); ax.set_xlabel("cap"); ax.grid(True); ax.legend(fontsize=8)

    # z vs ncc scatter
    ax = axes[0, 2]
    if not ok.empty:
        ax.scatter(ok["ncc_best_num"], ok["z_num"], s=5, alpha=0.3, color="green",  label=f"ok (n={len(ok)})")
    if not bad.empty:
        ax.scatter(bad["ncc_best_num"], bad["z_num"], s=5, alpha=0.2, color="red",    label=f"skip (n={len(bad)})")
    ax.set_xlabel("ncc_best"); ax.set_ylabel("z")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.axvline(0.15, color="k", linestyle="--", linewidth=0.8, label="ncc_min=0.15")
    ax.axhline(8.0,  color="k", linestyle=":",  linewidth=0.8, label="z_th=8")
    ax.set_title("z vs ncc_best gate scatter"); ax.legend(fontsize=7); ax.grid(True)

    # z vs SNR (OK only)
    ax = axes[1, 0]
    if not ok.empty and ok["snr_db_num"].notna().any():
        sc = ax.scatter(ok["z_num"], ok["snr_db_num"], c=ok["rx_gain_num"],
                        s=8, alpha=0.4, cmap="plasma")
        plt.colorbar(sc, ax=ax, label="rx_gain (dB)")
    ax.set_xlabel("z score"); ax.set_ylabel("SNR (dB)")
    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_title("SNR vs z (OK packets, colored by rx_gain)"); ax.grid(True)

    # ncc_best vs SNR (OK only)
    ax = axes[1, 1]
    if not ok.empty and ok["snr_db_num"].notna().any():
        sc = ax.scatter(ok["ncc_best_num"], ok["snr_db_num"], c=ok["rx_gain_num"],
                        s=8, alpha=0.4, cmap="plasma")
        plt.colorbar(sc, ax=ax, label="rx_gain (dB)")
    ax.set_xlabel("ncc_best"); ax.set_ylabel("SNR (dB)")
    ax.set_title("SNR vs ncc_best (OK, colored by rx_gain)"); ax.grid(True)

    # z vs EVM (OK only)
    ax = axes[1, 2]
    if not ok.empty and ok["probe_evm_num"].notna().any():
        ax.scatter(ok["z_num"], ok["probe_evm_num"], s=5, alpha=0.3, color="purple")
    ax.set_xlabel("z score"); ax.set_ylabel("EVM")
    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_title("EVM vs z (OK packets)"); ax.grid(True)

    figsave(fig, out_path)


# ----------------------------
# Plot: ROC/PR
# ----------------------------
def plot_roc_pr(df: pd.DataFrame, out_path: str, title: str, score_cols: List[str]):
    labels = df["is_ok"].astype(int).to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=13)

    ax = axes[0]
    for col in score_cols:
        s = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        fpr, tpr, _ = roc_curve(s, labels)
        ax.plot(fpr, tpr, label=col)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("ROC"); ax.grid(True); ax.legend(fontsize=7)

    ax = axes[1]
    for col in score_cols:
        s = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        prec, rec, _ = pr_curve(s, labels)
        ax.plot(rec, prec, label=col)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall"); ax.grid(True); ax.legend(fontsize=7)

    figsave(fig, out_path)


# ----------------------------
# Plot: correlation shape overlay
# ----------------------------
def plot_corr_shape_overlay(run: RunData, out_path: str, max_npz: int = 120):
    paths = run.ok_npz_paths[:max_npz]
    if len(paths) == 0:
        print("  (skipping corr overlay — no OK npz)")
        return

    shapes = []
    for p in paths:
        try:
            z = np.load(p, allow_pickle=True)
            if "corr_norm" not in z:
                continue
            c  = np.asarray(z["corr_norm"], dtype=np.float32)
            if c.size < 64:
                continue
            pk = int(np.argmax(c))
            L  = 256
            s  = max(0, pk - L); e = min(c.size, pk + L)
            seg = c[s:e].copy()
            seg = seg / (np.max(seg) + 1e-12)
            tgt = 2 * L
            seg = np.pad(seg, (0, tgt - seg.size)) if seg.size < tgt else seg[:tgt]
            shapes.append(seg)
        except Exception:
            continue

    if len(shapes) < 5:
        print(f"  (skipping corr overlay — only {len(shapes)} usable shapes)")
        return

    A    = np.stack(shapes, axis=0)
    mean = np.mean(A, axis=0)
    p10  = np.percentile(A, 10, axis=0)
    p90  = np.percentile(A, 90, axis=0)
    x    = np.arange(A.shape[1]) - (A.shape[1] // 2)

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.fill_between(x, p10, p90, alpha=0.25, label="p10..p90")
    ax.plot(x, mean, linewidth=2.0, label="mean shape")
    ax.set_title(f"corr_norm peak-centered shape (OK)  N={A.shape[0]}")
    ax.set_xlabel("samples offset from peak"); ax.set_ylabel("normalized corr")
    ax.grid(True); ax.legend()
    figsave(fig, out_path)


# ----------------------------
# Plot: threshold recommendation
# ----------------------------
def recommend_thresholds(df: pd.DataFrame) -> Dict[str, float]:
    ok  = df[df["is_ok"] == True]
    bad = df[df["is_ok"] == False]
    if len(ok) < 10 or len(bad) < 10:
        return {"note": "not enough samples for threshold recommendation"}

    xc = pd.to_numeric(df["xc_best_peak"], errors="coerce").fillna(0.0).to_numpy()
    gr = pd.to_numeric(df["gate_ratio"],   errors="coerce").fillna(0.0).to_numpy()
    y  = df["is_ok"].astype(int).to_numpy()

    xc_q = np.unique(np.quantile(xc, np.linspace(0.50, 0.99, 25)))
    gr_q = np.unique(np.quantile(gr, np.linspace(0.50, 0.99, 25)))

    best = None
    target_fa = 0.10
    for txc in xc_q:
        for tgr in gr_q:
            pred  = (xc >= txc) & (gr >= tgr)
            ok_m  = (y == 1); bad_m = (y == 0)
            okr   = float(np.mean(pred[ok_m]))  if np.any(ok_m)  else 0.0
            fa    = float(np.mean(pred[bad_m])) if np.any(bad_m) else 0.0
            prec  = float(np.sum((pred) & (y == 1)) / (np.sum(pred) + 1e-12))
            if fa > target_fa:
                continue
            score = okr + 0.25 * prec
            if best is None or score > best["score"]:
                best = {"t_xc": float(txc), "t_gr": float(tgr),
                        "ok_recall": okr, "false_accept": fa,
                        "precision": prec, "score": score}

    if best is None:
        best2 = None
        for txc in xc_q:
            for tgr in gr_q:
                pred  = (xc >= txc) & (gr >= tgr)
                ok_m  = (y == 1); bad_m = (y == 0)
                okr   = float(np.mean(pred[ok_m]))  if np.any(ok_m)  else 0.0
                fa    = float(np.mean(pred[bad_m])) if np.any(bad_m) else 0.0
                prec  = float(np.sum((pred) & (y == 1)) / (np.sum(pred) + 1e-12))
                score = okr - 0.3 * fa
                if best2 is None or score > best2["score"]:
                    best2 = {"t_xc": float(txc), "t_gr": float(tgr),
                             "ok_recall": okr, "false_accept": fa,
                             "precision": prec, "score": score,
                             "note": "constraint relaxed"}
        return best2 if best2 is not None else {"note": "threshold search failed"}
    return best


# ----------------------------
# Plot: compare multiple runs
# ----------------------------
def plot_compare_runs(runs: List[RunData], out_dir: str):
    ensure_dir(out_dir)
    frames = []
    for r in runs:
        df  = add_features(r.captures)
        mod = (str(df["modulation"].dropna().iloc[0])
               if df["modulation"].notna().any() else "unknown")
        df["run_tag"] = f"{mod} | {os.path.basename(r.run_dir)}"
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.suptitle("Run comparison", fontsize=14)

    def hist(ax, col, title):
        for tag, g in all_df.groupby("run_tag"):
            x = pd.to_numeric(g[col], errors="coerce").dropna()
            if len(x) == 0:
                continue
            ax.hist(x, bins=60, alpha=0.35, label=f"{tag} (n={len(g)})")
        ax.set_title(title); ax.grid(True); ax.legend(fontsize=7)

    hist(axes[0, 0], "xc_best_peak", "xc_best_peak")
    hist(axes[0, 1], "gate_ratio",   "gate_ratio=maxe/eg_th")
    hist(axes[0, 2], "snr_db",       "snr_db")
    hist(axes[0, 3], "probe_evm",    "probe_evm")

    hist(axes[1, 0], "abs_cfo",      "|cfo_hz|")
    hist(axes[1, 1], "z_num",        "z (MAD z-score)")
    hist(axes[1, 2], "ncc_best_num", "ncc_best")

    # decode rate per run
    ax = axes[1, 3]
    bars, labels = [], []
    for tag, g in all_df.groupby("run_tag"):
        bars.append(float(np.mean(g["is_ok"]))); labels.append(tag)
    ax.bar(range(len(bars)), bars)
    ax.set_xticks(range(len(bars)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax.set_ylim(0, 1.0); ax.set_title("decode rate"); ax.grid(True, axis="y")

    figsave(fig, os.path.join(out_dir, "compare_distributions.png"))

    # BER per gain for all runs
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 5))
    fig2.suptitle("BER vs RX Gain — all runs", fontsize=13)
    for tag, g in all_df.groupby("run_tag"):
        ok_g = g[g["is_ok"] == True]
        gains = sorted(ok_g["rx_gain_num"].dropna().unique())
        bers  = []
        for gain in gains:
            sub   = ok_g[ok_g["rx_gain_num"] == gain]
            nb    = sub["n_bits_num"].sum(); ne = sub["n_bit_errors_num"].sum()
            bers.append(float(ne / nb) if nb > 0 else np.nan)
        nonzero = [(g2, b) for g2, b in zip(gains, bers) if not np.isnan(b) and b > 0]
        if nonzero:
            gv, bv = zip(*nonzero)
            ax2.semilogy(gv, bv, "o-", linewidth=2, label=tag)
        zero = [(g2, b) for g2, b in zip(gains, bers) if not np.isnan(b) and b == 0]
        if zero:
            gv2, _ = zip(*zero)
            ax2.scatter(gv2, [1e-7] * len(gv2), marker="v", s=60, label=f"{tag} BER=0")
    ax2.set_xlabel("rx_gain (dB)"); ax2.set_ylabel("BER")
    ax2.grid(True, which="both", alpha=0.4); ax2.legend(fontsize=8)
    figsave(fig2, os.path.join(out_dir, "compare_ber_per_gain.png"))


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dirs", nargs="+", required=True,
                    help="One or more run directories containing captures.csv")
    ap.add_argument("--out_dir", required=True,
                    help="Where to write plots and reports")
    ap.add_argument("--max_npz_for_corr_overlay", type=int, default=120)
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    runs = [load_run(d) for d in args.run_dirs]
    for r in runs:
        df       = add_features(r.captures)
        run_name = os.path.basename(r.run_dir)
        mod      = (str(df["modulation"].dropna().iloc[0])
                    if df["modulation"].notna().any() else "unknown")
        title    = (f"{run_name} | mod={mod} | n={len(df)} | "
                    f"ok={int(df['is_ok'].sum())} ({100*df['is_ok'].mean():.1f}%)")

        print(f"\n[{run_name}]  rows={len(df)}  ok={int(df['is_ok'].sum())}  "
              f"ok_rate={df['is_ok'].mean():.3f}  "
              f"ok_npz={len(r.ok_npz_paths)}  fail_npz={len(r.fail_npz_paths)}")

        # Grid overview
        plot_grid_overview(
            df,
            out_path=os.path.join(args.out_dir, f"{run_name}_grid.png"),
            title=f"Streaming RX diagnostics (grid) — {title}",
        )

        # Step7 gate features
        plot_step7_features(
            df,
            out_path=os.path.join(args.out_dir, f"{run_name}_step7_gates.png"),
            title=f"Step7 gate features (z / ncc_best) — {title}",
        )

        # BER per gain
        plot_ber_per_gain(
            df, r.summary,
            out_path=os.path.join(args.out_dir, f"{run_name}_ber_per_gain.png"),
            title=f"BER & Decode Rate vs RX Gain — {title}",
        )

        # ROC/PR for candidate gates
        score_cols = [
            "xc_best_peak_num", "gate_ratio", "gate_ratio_z",
            "xc_best_peak_z",   "snr_db_num",
            "z_num", "ncc_best_num",
        ]
        df["score_comp1"] = (df.get("xc_best_peak_z", 0).fillna(0)
                             + 0.4 * df.get("gate_ratio_z", 0).fillna(0))
        df["score_comp2"] = (df.get("z_num", 0).fillna(0)
                             + 10.0 * df.get("ncc_best_num", 0).fillna(0))
        score_cols += ["score_comp1", "score_comp2"]

        plot_roc_pr(
            df,
            out_path=os.path.join(args.out_dir, f"{run_name}_roc_pr.png"),
            title=f"ROC/PR candidate gates — {title}",
            score_cols=score_cols,
        )

        # Threshold recommendation
        rec = recommend_thresholds(df)
        with open(os.path.join(args.out_dir, f"{run_name}_threshold_recommend.json"), "w") as f:
            json.dump(rec, f, indent=2)

        # Corr shape overlay
        plot_corr_shape_overlay(
            r,
            out_path=os.path.join(args.out_dir, f"{run_name}_corr_shape_ok_overlay.png"),
            max_npz=args.max_npz_for_corr_overlay,
        )

        # Text report
        rep_path = os.path.join(args.out_dir, f"{run_name}_report.txt")
        with open(rep_path, "w") as f:
            f.write(f"Run: {r.run_dir}\n")
            f.write(f"Rows: {len(df)}  OK: {int(df['is_ok'].sum())}  "
                    f"OK rate: {float(df['is_ok'].mean()):.3f}\n")
            f.write(f"OK npz: {len(r.ok_npz_paths)}  FAIL npz: {len(r.fail_npz_paths)}\n")
            f.write(f"Modulation: {mod}\n\n")
            f.write("Status counts:\n" + df["status"].value_counts().to_string() + "\n\n")
            f.write("Reason counts:\n" + df["reason"].value_counts().head(20).to_string() + "\n\n")

            # Per-gain BER summary from CSV
            ok_df = df[df["is_ok"] == True]
            f.write("BER per rx_gain (from CSV):\n")
            for g in sorted(df["rx_gain_num"].dropna().unique()):
                sub  = ok_df[ok_df["rx_gain_num"] == g]
                nb   = sub["n_bits_num"].sum(); ne = sub["n_bit_errors_num"].sum()
                ber  = ne / nb if nb > 0 else None
                dr   = len(sub) / max(len(df[df["rx_gain_num"] == g]), 1)
                bers = f"{ber:.4e}" if ber is not None else "no_ber_data"
                f.write(f"  rx_gain={g:5.1f} dB  BER={bers}  "
                        f"decode_rate={dr:.3f}  n_ok={len(sub)}\n")

            if df["z_num"].notna().any():
                f.write(f"\nz stats: median={df['z_num'].median():.1f}  "
                        f"p95={df['z_num'].quantile(0.95):.1f}  "
                        f"p5={df['z_num'].quantile(0.05):.1f}\n")
            if df["ncc_best_num"].notna().any():
                f.write(f"ncc_best stats: median={df['ncc_best_num'].median():.3f}  "
                        f"p95={df['ncc_best_num'].quantile(0.95):.3f}  "
                        f"p5={df['ncc_best_num'].quantile(0.05):.3f}\n")

            f.write("\nThreshold recommendation (xc + gate_ratio):\n")
            f.write(json.dumps(rec, indent=2) + "\n")
        print(f"  -> {run_name}_report.txt")

    if len(runs) >= 2:
        plot_compare_runs(runs, out_dir=args.out_dir)

    print(f"\n[OK] Wrote {len(runs)} run(s) to: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()


'''
# ── Quick-start commands ──────────────────────────────────────────────────────

# Single step7 QPSK run
python3 rf_stream/analyze_runs_offline.py \
  --run_dirs rf_stream/ber_sweep/run_20260428_233037 \
  --out_dir  rf_stream/ber_sweep/_offline_plots_qpsk_step7

# Background noise
python3 rf_stream/analyze_runs_offline.py \
  --run_dirs rf_stream/bg_noise/run_20260428_232452 \
             rf_stream/bg_noise/run_20260428_232555 \
             rf_stream/bg_noise/run_20260428_232658 \
  --out_dir  rf_stream/bg_noise/_plots

# Compare QPSK vs QAM16 (after QAM16 sweep)
python3 rf_stream/analyze_runs_offline.py \
  --run_dirs rf_stream/ber_sweep/run_QPSK rf_stream/ber_sweep/run_QAM16 \
  --out_dir  rf_stream/ber_sweep/_offline_plots_compare
'''
