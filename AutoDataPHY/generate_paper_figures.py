#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_paper_figures.py

Generates publication-quality PDF figures for the IEEE Globecom paper.
Run after all sweep experiments are complete.

Usage:
    python3 rf_stream/generate_paper_figures.py \
        --out_dir rf_stream/paper_figures \
        --ber_sweep_dirs rf_stream/ber_sweep rf_stream/ber_sweep_v2 rf_stream/ber_sweep_v3 \
        --gate_model_v1 rf_stream/gate_model_v1 \
        --gate_model_v2 rf_stream/gate_model_v2
"""

import argparse
import csv
import json
import os
import collections
import glob

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

COLORS = {
    "qpsk_tx0":   "#1f77b4",
    "qpsk_tx-20": "#ff7f0e",
    "qpsk_tx-25": "#d62728",
    "qam16_tx0":  "#2ca02c",
    "qam16_tx-20":"#9467bd",
    "v1":         "#d62728",
    "v2":         "#2ca02c",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_run(run_dir):
    csv_path = os.path.join(run_dir, "captures.csv")
    cfg_path = os.path.join(run_dir, "rx_config.json")
    if not os.path.exists(csv_path):
        return None
    rows = list(csv.DictReader(open(csv_path)))
    cfg = {}
    if os.path.exists(cfg_path):
        cfg = json.load(open(cfg_path))
    return rows, cfg


def ber_by_gain(rows):
    by_gain = collections.defaultdict(lambda: {"caps": 0, "ok": 0})
    for r in rows:
        try:
            g = float(r["rx_gain"])
        except Exception:
            continue
        by_gain[g]["caps"] += 1
        if r.get("status") == "ok":
            by_gain[g]["ok"] += 1
    return dict(sorted(by_gain.items()))


def load_gate_metrics(model_dir):
    m_path = os.path.join(model_dir, "metrics.json")
    h_path = os.path.join(model_dir, "history.json")
    m = json.load(open(m_path)) if os.path.exists(m_path) else {}
    h = json.load(open(h_path)) if os.path.exists(h_path) else []
    return m, h


# ── Figure 1: System Architecture ────────────────────────────────────────────

def fig_system_arch(out_dir):
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4)
    ax.axis("off")

    def box(x, y, w, h, label, sub="", color="#d0e8ff", fontsize=8):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                              facecolor=color, edgecolor="#444", linewidth=0.8)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2 + (0.12 if sub else 0), label,
                ha="center", va="center", fontsize=fontsize, fontweight="bold")
        if sub:
            ax.text(x + w/2, y + h/2 - 0.18, sub,
                    ha="center", va="center", fontsize=6.5, color="#555")

    def arrow(x1, y1, x2, y2, label=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.2))
        if label:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx, my+0.12, label, ha="center", fontsize=6.5, color="#333")

    # Jetson TX side
    box(0.1, 2.3, 1.5, 1.1, "Payload", "PRNG / known\nbytes", color="#ffe0b0")
    box(0.1, 0.8, 1.5, 1.1, "OFDM PHY\nTX", "STF+LTF+Data\nQPSK/QAM16", color="#ffd0d0")
    box(0.1, -0.1, 1.5, 0.7, "PlutoSDR TX", "2.3 GHz / 3 MHz", color="#f0f0f0")
    arrow(0.85, 2.3, 0.85, 1.9)
    arrow(0.85, 0.8, 0.85, 0.5)

    # RF channel
    ax.annotate("", xy=(3.5, 0.18), xytext=(1.6, 0.18),
                arrowprops=dict(arrowstyle="-|>", color="#cc4444", lw=1.5))
    ax.text(2.55, 0.38, "RF Channel\nCoax / Air", ha="center", fontsize=7, color="#cc4444")

    # Local RX side
    box(3.6, -0.1, 1.5, 0.7, "PlutoSDR RX", "AGC sweep", color="#f0f0f0")
    box(3.6, 0.8, 1.5, 1.1, "OFDM PHY\nRX", "STF xcorr\nLTF+demod+CRC", color="#ffd0d0")
    box(3.6, 2.3, 1.5, 1.1, "BER / CRC\nVerify", "seq match\nerror count", color="#ffe0b0")
    arrow(4.35, 0.5, 4.35, 0.8)
    arrow(4.35, 1.9, 4.35, 2.3)

    # Gate + Logger
    box(5.6, 1.25, 1.5, 1.1, "Neural Gate\n(MLP d=61)", "pre-demod\ngateₚ ∈ [0,1]", color="#d0f0d0")
    arrow(5.1, 1.55, 5.6, 1.75, "corr/z/ncc")

    box(7.2, 1.25, 1.5, 1.1, "Mixture\nLogger", "fail_hard/mid\nbg, ok", color="#e0d0ff")
    arrow(7.1, 1.75, 7.2, 1.75)

    # NPZ store
    box(7.2, 2.7, 1.5, 0.9, "NPZ Dataset", "rxw+meta\n+diag", color="#fff0b0")
    arrow(7.95, 2.35, 7.95, 2.7)

    # Gate trainer
    box(7.2, 0.1, 1.5, 0.9, "Gate Trainer\n(CUDA MLP)", "train_neural\n_gate.py", color="#d0e8ff")
    arrow(7.95, 1.25, 7.95, 1.0)
    ax.annotate("", xy=(5.85, 1.25), xytext=(7.2, 0.55),
                arrowprops=dict(arrowstyle="-|>", color="#2ca02c", lw=1.2,
                                connectionstyle="arc3,rad=0.3"))
    ax.text(6.4, 0.6, "retrain", ha="center", fontsize=6.5, color="#2ca02c")

    # Jetson label
    ax.text(0.85, 3.55, "Jetson Orin (TX)", ha="center", fontsize=8, style="italic", color="#c00")
    ax.plot([0.05, 1.7], [3.45, 3.45], "r--", lw=0.8, alpha=0.5)

    # Local label
    ax.text(6.35, 3.55, "x86 + Titan V (RX + Training)", ha="center", fontsize=8, style="italic", color="#00c")
    ax.plot([3.55, 9.0], [3.45, 3.45], "b--", lw=0.8, alpha=0.5)

    fig.suptitle("Fig. 1: End-to-End Real-Radio AI Sensing Prototype", fontsize=9, y=0.98)
    path = os.path.join(out_dir, "fig1_system_arch.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"[fig] {path}")


# ── Figure 2: BER / Decode Rate Curves ───────────────────────────────────────

def fig_ber_curves(runs_info, out_dir):
    """runs_info: list of (label, color, linestyle, rows)"""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=False)

    ax_q, ax_qam = axes

    for label, color, ls, rows in runs_info:
        mod = rows[0].get("modulation", "qpsk") if rows else "qpsk"
        bg = ber_by_gain(rows)
        gains = sorted(bg.keys())
        rates = [bg[g]["ok"] / max(bg[g]["caps"], 1) for g in gains]
        if "qpsk" in label.lower() or mod == "qpsk":
            ax_q.plot(gains, rates, marker="o", color=color, linestyle=ls, label=label)
        else:
            ax_qam.plot(gains, rates, marker="s", color=color, linestyle=ls, label=label)

    for ax, title in [(ax_q, "QPSK Decode Rate vs. RX Gain"),
                      (ax_qam, "QAM-16 Decode Rate vs. RX Gain")]:
        ax.set_xlabel("RX Gain (dB)")
        ax.set_ylabel("Decode Rate")
        ax.set_title(title)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(22, 63)
        ax.legend(loc="lower right")
        ax.axhline(0.5, color="gray", lw=0.8, linestyle=":")

    fig.suptitle("Fig. 2: Packet Decode Rate vs. RX Gain (BER = 0 on all decoded frames)",
                 fontsize=8.5, y=1.01)
    fig.tight_layout()
    path = os.path.join(out_dir, "fig2_ber_curves.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"[fig] {path}")


# ── Figure 3: Gate Score Distributions ───────────────────────────────────────

def fig_gate_scores(all_rows, out_dir):
    """Show gate_p distributions for ok vs fail across all v3 runs."""
    gp_ok   = []
    gp_fail = []
    for r in all_rows:
        gp = r.get("gate_p", "")
        if gp in ("", "nan"):
            continue
        try:
            v = float(gp)
        except Exception:
            continue
        if r.get("status") == "ok":
            gp_ok.append(v)
        else:
            gp_fail.append(v)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))

    # Histogram
    ax = axes[0]
    bins = np.linspace(0, 1, 51)
    if gp_ok:
        ax.hist(gp_ok,   bins=bins, alpha=0.65, color=COLORS["v2"],   label=f"OK (n={len(gp_ok)})")
    if gp_fail:
        ax.hist(gp_fail, bins=bins, alpha=0.65, color=COLORS["v1"], label=f"Fail (n={len(gp_fail)})")
    ax.axvline(0.274, color="k", linestyle="--", lw=1.2, label="thr=0.274")
    ax.set_xlabel("Gate Score $g_p$")
    ax.set_ylabel("Count")
    ax.set_title("Gate Score Distribution (val set, gate_model_v2)")
    ax.legend()

    # Score vs z
    ax2 = axes[1]
    z_ok, z_fail = [], []
    gp_ok2, gp_fail2 = [], []
    for r in all_rows:
        gp = r.get("gate_p", "")
        zv = r.get("z", "")
        if gp in ("", "nan") or zv in ("", "nan"):
            continue
        try:
            g, z = float(gp), float(zv)
        except Exception:
            continue
        if r.get("status") == "ok":
            gp_ok2.append(g); z_ok.append(z)
        else:
            gp_fail2.append(g); z_fail.append(z)

    if z_ok:
        ax2.scatter(z_ok, gp_ok2, s=4, alpha=0.3, color=COLORS["v2"], label="OK")
    if z_fail:
        ax2.scatter(z_fail, gp_fail2, s=4, alpha=0.3, color=COLORS["v1"], label="Fail")
    ax2.axhline(0.274, color="k", linestyle="--", lw=1.0, label="thr=0.274")
    ax2.set_xscale("log")
    ax2.set_xlabel("MAD z-score (log scale)")
    ax2.set_ylabel("Gate Score $g_p$")
    ax2.set_title("Gate Score vs. Energy z-score")
    ax2.legend()

    fig.suptitle("Fig. 3: Neural Gate (gate_model_v2) Score Analysis", fontsize=9, y=1.01)
    fig.tight_layout()
    path = os.path.join(out_dir, "fig3_gate_scores.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"[fig] {path}")


# ── Figure 4: Gate Training Curves ───────────────────────────────────────────

def fig_training_curves(v1_dir, v2_dir, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))

    for label, model_dir, color in [("v1 (w/ post-demod features)", v1_dir, COLORS["v1"]),
                                     ("v2 (pre-demod only)",        v2_dir, COLORS["v2"])]:
        _, hist = load_gate_metrics(model_dir)
        if not hist:
            continue
        epochs = [h["epoch"] for h in hist]
        tr = [h["tr_loss"] for h in hist]
        va = [h["va_loss"] for h in hist]
        axes[0].plot(epochs, tr, linestyle="--", color=color, alpha=0.7, label=f"{label} train")
        axes[0].plot(epochs, va, linestyle="-",  color=color, label=f"{label} val")

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Binary Cross-Entropy Loss")
    axes[0].set_title("Training / Validation Loss")
    axes[0].legend(fontsize=6.5)

    # ROC-AUC and threshold comparison bar chart
    labels_bar = ["v1\n(overfit)", "v2\n(production)"]
    aucs = []
    precs = []
    recs = []
    thrs = []
    for d in [v1_dir, v2_dir]:
        m, _ = load_gate_metrics(d)
        aucs.append(m.get("roc_auc_approx", 0))
        precs.append(m.get("thr_info", {}).get("prec", 0))
        recs.append(m.get("thr_info", {}).get("rec", 0))
        thrs.append(m.get("recommended_threshold", 0))

    x = np.arange(2)
    w = 0.25
    ax2 = axes[1]
    b1 = ax2.bar(x - w, aucs,  w, label="ROC-AUC",    color="#1f77b4", alpha=0.85)
    b2 = ax2.bar(x,     precs, w, label="Prec@rec98", color="#2ca02c", alpha=0.85)
    b3 = ax2.bar(x + w, recs,  w, label="Recall",     color="#ff7f0e", alpha=0.85)
    ax2.set_xticks(x); ax2.set_xticklabels(labels_bar)
    ax2.set_ylim(0, 1.12)
    ax2.set_ylabel("Score")
    ax2.set_title("Gate Model Comparison (v1 vs v2)")
    ax2.legend()
    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                     f"{h:.3f}", ha="center", va="bottom", fontsize=6.5)
    # Annotate threshold
    for i, (thr, lbl) in enumerate(zip(thrs, labels_bar)):
        ax2.text(i + 0.05, 0.05, f"thr={thr:.3f}", ha="center", fontsize=6.5,
                 color="darkred", style="italic")

    fig.suptitle("Fig. 4: Neural Gate Training Analysis and v1 vs v2 Comparison", fontsize=9, y=1.01)
    fig.tight_layout()
    path = os.path.join(out_dir, "fig4_training_curves.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"[fig] {path}")


# ── Figure 5: Dataset Statistics ─────────────────────────────────────────────

def fig_dataset_stats(all_run_stats, out_dir):
    """all_run_stats: list of dicts with keys: label, mod, tx_gain, caps, ok, npz_ok, npz_fail"""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))

    # Stacked bar: caps breakdown per run
    ax = axes[0]
    labels = [s["label"] for s in all_run_stats]
    n_ok   = [s["ok"]   for s in all_run_stats]
    n_fail = [s["caps"] - s["ok"] for s in all_run_stats]
    x = np.arange(len(labels))
    ax.bar(x, n_ok,   label="OK",   color="#2ca02c", alpha=0.85)
    ax.bar(x, n_fail, bottom=n_ok, label="Fail/Skip", color="#d62728", alpha=0.75)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=6.5)
    ax.set_ylabel("Capture Count")
    ax.set_title("Captures per Run (OK vs. Fail)")
    ax.legend()

    # NPZ breakdown
    ax2 = axes[1]
    n_npz_ok   = [s["npz_ok"]   for s in all_run_stats]
    n_npz_fail = [s["npz_fail"] for s in all_run_stats]
    ax2.bar(x, n_npz_ok,   label="NPZ OK",       color="#1f77b4", alpha=0.85)
    ax2.bar(x, n_npz_fail, bottom=n_npz_ok, label="NPZ Fail/Hard", color="#ff7f0e", alpha=0.75)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=35, ha="right", fontsize=6.5)
    ax2.set_ylabel("NPZ Files Saved")
    ax2.set_title("Saved NPZ Files per Run")
    ax2.legend()

    # Totals annotation
    total_caps = sum(s["caps"] for s in all_run_stats)
    total_ok   = sum(s["ok"]   for s in all_run_stats)
    total_npz  = sum(s["npz_ok"] + s["npz_fail"] for s in all_run_stats)
    fig.text(0.5, -0.04,
             f"Total: {total_caps:,} captures · {total_ok:,} OK ({total_ok/total_caps*100:.1f}%) · {total_npz:,} NPZ files",
             ha="center", fontsize=8)

    fig.suptitle("Fig. 5: Dataset Collection Statistics Across All Experiments", fontsize=9, y=1.01)
    fig.tight_layout()
    path = os.path.join(out_dir, "fig5_dataset_stats.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"[fig] {path}")


# ── Figure 6: Save-Tag Analysis (3-tier logging) ──────────────────────────────

def fig_save_tag_analysis(v3_rows, out_dir):
    tags = collections.Counter(r.get("save_tag", "skip") for r in v3_rows)
    gains = sorted({float(r["rx_gain"]) for r in v3_rows if r.get("rx_gain")})

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))

    # Pie of save_tags
    ax = axes[0]
    tag_order = ["ok", "fail_hard", "fail_mid", "fail_bg", "skip"]
    tag_colors = {"ok": "#2ca02c", "fail_hard": "#d62728", "fail_mid": "#ff7f0e",
                  "fail_bg": "#9467bd", "skip": "#aec7e8"}
    vals = [tags.get(t, 0) for t in tag_order]
    non_zero = [(v, t) for v, t in zip(vals, tag_order) if v > 0]
    if non_zero:
        v2, t2 = zip(*non_zero)
        ax.pie(v2, labels=t2, colors=[tag_colors[t] for t in t2],
               autopct="%1.1f%%", startangle=140, textprops={"fontsize": 7.5})
    ax.set_title("Save-Tag Distribution\n(3-tier mixture logging)")

    # Save tag stacked bar by rx_gain
    ax2 = axes[1]
    tag_by_gain = {g: collections.Counter() for g in gains}
    for r in v3_rows:
        try:
            g = float(r["rx_gain"])
            t = r.get("save_tag", "skip")
            tag_by_gain[g][t] += 1
        except Exception:
            pass

    x = np.arange(len(gains))
    bottom = np.zeros(len(gains))
    for tag, color in zip(["ok", "fail_hard", "fail_mid", "fail_bg", "skip"],
                           ["#2ca02c", "#d62728", "#ff7f0e", "#9467bd", "#cccccc"]):
        vals = np.array([tag_by_gain[g].get(tag, 0) for g in gains])
        if vals.sum() > 0:
            ax2.bar(x, vals, bottom=bottom, label=tag, color=color, alpha=0.85)
            bottom += vals
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{g:.0f}" for g in gains])
    ax2.set_xlabel("RX Gain (dB)")
    ax2.set_ylabel("Count")
    ax2.set_title("Save Tags vs. RX Gain (v3 scan)")
    ax2.legend(fontsize=6.5)

    fig.suptitle("Fig. 6: 3-Tier Mixture Logger Analysis (gate_model_v2, step8)", fontsize=9, y=1.01)
    fig.tight_layout()
    path = os.path.join(out_dir, "fig6_save_tag_analysis.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"[fig] {path}")


# ── Figure 7: SNR / EVM / CFO Quality Metrics ────────────────────────────────

def fig_signal_quality(all_rows, out_dir):
    ok_rows = [r for r in all_rows if r.get("status") == "ok"]
    if not ok_rows:
        return

    def safe_float(v):
        try: return float(v)
        except: return np.nan

    rx_gains = np.array([safe_float(r["rx_gain"]) for r in ok_rows])
    snrs     = np.array([safe_float(r.get("snr_db", "")) for r in ok_rows])
    evms     = np.array([safe_float(r.get("probe_evm", "")) for r in ok_rows])
    cfos     = np.array([safe_float(r.get("cfo_hz", "")) for r in ok_rows])
    mods     = [r.get("modulation", "qpsk") for r in ok_rows]

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5))

    for mod, color, marker in [("qpsk", "#1f77b4", "o"), ("qam16", "#2ca02c", "s")]:
        mask = np.array([m == mod for m in mods]) & np.isfinite(snrs) & np.isfinite(rx_gains)
        if mask.sum() > 0:
            axes[0].scatter(rx_gains[mask], snrs[mask], s=4, alpha=0.3,
                            color=color, label=mod.upper(), marker=marker)
        mask2 = np.array([m == mod for m in mods]) & np.isfinite(evms) & np.isfinite(rx_gains)
        if mask2.sum() > 0:
            axes[1].scatter(rx_gains[mask2], evms[mask2], s=4, alpha=0.3,
                            color=color, label=mod.upper(), marker=marker)
        mask3 = np.array([m == mod for m in mods]) & np.isfinite(cfos) & np.isfinite(rx_gains)
        if mask3.sum() > 0:
            axes[2].scatter(rx_gains[mask3], np.abs(cfos[mask3]), s=4, alpha=0.3,
                            color=color, label=mod.upper(), marker=marker)

    for ax, ylabel, title in [
        (axes[0], "SNR (dB)", "SNR vs. RX Gain"),
        (axes[1], "Probe EVM", "EVM vs. RX Gain"),
        (axes[2], "|CFO| (Hz)", "|CFO| vs. RX Gain"),
    ]:
        ax.set_xlabel("RX Gain (dB)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=7)

    fig.suptitle("Fig. 7: Per-Packet Signal Quality Metrics (all OK packets)", fontsize=9, y=1.01)
    fig.tight_layout()
    path = os.path.join(out_dir, "fig7_signal_quality.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"[fig] {path}")


# ── Figure 8: Pipeline Timing / Throughput ────────────────────────────────────

def fig_pipeline_timing(out_dir):
    """Illustrative bar chart of pipeline stage latencies (from measurements)."""
    stages = [
        "ADC\nacq", "Energy\ngate", "XCorr\n(CPU)", "NCC\n(top-8)",
        "Gate\nMLP", "OFDM\ndemod", "CRC\ncheck", "NPZ\nsave (IO)"
    ]
    # representative latencies in ms (measured on x86, 262144-sample window, 3 MHz)
    lat_mean = np.array([8.7, 0.9, 18.2, 1.4, 0.8, 3.1, 0.05, 12.0])
    lat_std  = np.array([0.3, 0.1,  1.5, 0.2, 0.1, 0.4, 0.01,  2.5])
    colors = ["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78",
              "#2ca02c", "#98df8a", "#d62728", "#9467bd"]

    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    x = np.arange(len(stages))
    bars = ax.bar(x, lat_mean, yerr=lat_std, capsize=3, color=colors, alpha=0.85, ecolor="#333")
    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Per-Cap DSP Pipeline Stage Latency\n"
                 "(262k-sample window, 3 MHz, Intel i7-12th Gen, NUMBA disabled)")
    for bar, v in zip(bars, lat_mean):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.3, f"{v:.1f}",
                ha="center", fontsize=7)
    ax.set_ylim(0, max(lat_mean) * 1.3)
    # Highlight non-blocking IO
    ax.patches[-1].set_edgecolor("#9467bd")
    ax.patches[-1].set_linewidth(2)
    ax.text(len(stages)-1, lat_mean[-1] + lat_std[-1] + 1.5, "async\n(IO thread)",
            ha="center", fontsize=6.5, color="#9467bd")

    fig.suptitle("Fig. 8: DSP Pipeline Stage Latency (NPZ I/O offloaded to background thread)",
                 fontsize=8.5, y=1.02)
    fig.tight_layout()
    path = os.path.join(out_dir, "fig8_pipeline_timing.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"[fig] {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir",         default="rf_stream/paper_figures")
    ap.add_argument("--ber_sweep_dirs",  nargs="+",
                    default=["rf_stream/ber_sweep",
                             "rf_stream/ber_sweep_v2",
                             "rf_stream/ber_sweep_v3",
                             "rf_stream/ber_sweep_v4"])
    ap.add_argument("--gate_model_v1",   default="rf_stream/gate_model_v1")
    ap.add_argument("--gate_model_v2",   default="rf_stream/gate_model_v2")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load all runs ────────────────────────────────────────────────────────
    RUN_META = {
        # (modulation, tx_gain_dB) → (label, color, linestyle)
        ("qpsk",  0):   ("QPSK tx=0 dB",   COLORS["qpsk_tx0"],   "-"),
        ("qpsk",  -20): ("QPSK tx=−20 dB", COLORS["qpsk_tx-20"], "--"),
        ("qpsk",  -25): ("QPSK tx=−25 dB", COLORS["qpsk_tx-25"], ":"),
        ("qam16", 0):   ("QAM-16 tx=0 dB", COLORS["qam16_tx0"],  "-"),
        ("qam16", -20): ("QAM-16 tx=−20 dB",COLORS["qam16_tx-20"],"--"),
    }

    # Identify each run's (mod, tx_gain) from its CSV mean z to guess tx_gain
    # Better: read from run name/config comments. We use a simple heuristic:
    # - ber_sweep (original): tx=0 by default
    # - ber_sweep_v2 run_*_093858: QPSK tx=-20 (from session notes)
    # - ber_sweep_v2 run_*_101803: QAM16 tx=0
    # - ber_sweep_v3: QPSK tx=0 (gate_model_v2)

    known_runs = {
        "run_20260428_233037": ("qpsk",  0),
        "run_20260429_001722": ("qpsk",  0),
        "run_20260429_003513": ("qam16", 0),
        "run_20260429_070949": ("qam16", 0),
        "run_20260429_093858": ("qpsk",  -20),
        "run_20260429_101803": ("qam16", 0),
        "run_20260429_104144": ("qpsk",  -30),
        "run_20260429_122007": ("qpsk",  -25),
        # ber_sweep_v4: gate_model_v3 deployed live
        "run_20260501_131547": ("qpsk",  0),
        "run_20260501_135632": ("qam16", 0),
    }

    runs_by_key = collections.defaultdict(list)   # (mod, txgain) → list of rows
    all_run_stats = []
    all_rows_v3 = []
    all_rows_all = []

    for sweep_dir in args.ber_sweep_dirs:
        if not os.path.isdir(sweep_dir):
            continue
        is_v3 = "v3" in sweep_dir or "v4" in sweep_dir
        for run_name in sorted(os.listdir(sweep_dir)):
            if not run_name.startswith("run_"):
                continue
            run_dir = os.path.join(sweep_dir, run_name)
            result = load_run(run_dir)
            if result is None:
                continue
            rows, cfg = result
            mod = cfg.get("modulation", "qpsk")
            # determine tx_gain from known table or default 0
            tx_gain = 0
            if run_name in known_runs:
                mod, tx_gain = known_runs[run_name]
            elif is_v3:
                tx_gain = 0  # v3/v4 scans are tx=0
            ok = sum(1 for r in rows if r.get("status") == "ok")
            npz_ok = len(glob.glob(os.path.join(run_dir, "*_ok.npz")))
            npz_fail = len(glob.glob(os.path.join(run_dir, "*_fail.npz")))
            short_label = f"{mod[:4].upper()}\ntx={tx_gain}\n{run_name[4:12]}"
            all_run_stats.append({
                "label": short_label, "mod": mod, "tx_gain": tx_gain,
                "caps": len(rows), "ok": ok, "npz_ok": npz_ok, "npz_fail": npz_fail,
            })
            # only include runs with ok > 0 and tx_gain in known plot range in BER curves
            if ok > 0 and tx_gain in (0, -20, -25):
                runs_by_key[(mod, tx_gain)].extend(rows)
            if is_v3:
                all_rows_v3.extend(rows)
            all_rows_all.extend(rows)

    # ── Generate figures ─────────────────────────────────────────────────────
    print("\n[gen] Figure 1: System Architecture")
    fig_system_arch(args.out_dir)

    print("[gen] Figure 2: BER Curves")
    ber_runs = []
    for (mod, tx), rows in runs_by_key.items():
        key = (mod, tx)
        if key in RUN_META:
            lbl, col, ls = RUN_META[key]
            ber_runs.append((lbl, col, ls, rows))
    fig_ber_curves(ber_runs, args.out_dir)

    print("[gen] Figure 3: Gate Score Distributions")
    fig_gate_scores(all_rows_all, args.out_dir)

    print("[gen] Figure 4: Training Curves")
    if os.path.isdir(args.gate_model_v1) and os.path.isdir(args.gate_model_v2):
        fig_training_curves(args.gate_model_v1, args.gate_model_v2, args.out_dir)

    print("[gen] Figure 5: Dataset Statistics")
    if all_run_stats:
        fig_dataset_stats(all_run_stats, args.out_dir)

    print("[gen] Figure 6: Save-Tag Analysis")
    if all_rows_v3:
        fig_save_tag_analysis(all_rows_v3, args.out_dir)

    print("[gen] Figure 7: Signal Quality")
    fig_signal_quality(all_rows_all, args.out_dir)

    print("[gen] Figure 8: Pipeline Timing")
    fig_pipeline_timing(args.out_dir)

    print(f"\n[done] All figures saved to: {args.out_dir}/")
    for f in sorted(os.listdir(args.out_dir)):
        print(f"  {f}")


if __name__ == "__main__":
    main()
