#!/usr/bin/env python3
"""
generate_multitask_figures_v2.py  –  Publication figures for MT-PreamCNN-Attn v2.

Reads:  rf_stream/multitask_model_v2/metrics_*.json
Writes to rf_stream/paper_figures/:
  fig16_v2_domain.pdf       Cross-domain performance grid (gate / mod / SNR, all 4 baseline models)
  fig17_v2_phaseaug.pdf     Phase-aug before→after modulation accuracy (the headline result)
  fig18_v2_roc_ota.pdf      AirLink ROC curves: zero-shot vs mixed (CNN and Attn)
  fig15_v2_summary.pdf      Updated summary bar (replaces fig15_mt_summary_bar.pdf)

Usage:
  python3 rf_stream/generate_multitask_figures_v2.py
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator

# ── paths ─────────────────────────────────────────────────────────────────────
METRICS_DIR = "rf_stream/multitask_model_v2"
OUT_DIR     = "rf_stream/paper_figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ── style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":  "sans-serif",
    "font.size":    8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
    "figure.dpi":   150,
})

# color palette
C_CNN_ZS   = "#4878CF"   # medium blue – CNN zero-shot
C_CNN_MX   = "#1F4E8C"   # dark blue   – CNN mixed
C_ATN_ZS   = "#E07B39"   # orange      – Attn zero-shot
C_ATN_MX   = "#A02020"   # dark red    – Attn mixed
C_COAX     = "#F0F0F0"   # light grey  – CoaxSweep bars (coax maintained)
C_AUG      = "#5CAD5C"   # green       – phase-aug improvement bars
COLORS_4   = [C_CNN_ZS, C_CNN_MX, C_ATN_ZS, C_ATN_MX]

MODEL_LABELS = [
    "CNN\nZero-shot",
    "CNN\nMixed",
    "Attn\nZero-shot",
    "Attn\nMixed",
]

# ── data loading ──────────────────────────────────────────────────────────────
def load_metrics(key: str) -> dict:
    path = os.path.join(METRICS_DIR, f"metrics_{key}.json")
    with open(path) as f:
        return json.load(f)

def get_row(arch: str, mode: str, aug: bool = False):
    """Return (coax, airlink) metric dicts for one model variant."""
    key = f"{arch}_{mode}" + ("_phaseaug" if aug else "")
    m = load_metrics(key)
    coax = m["coax_val"]
    air  = m["airlink_zeroshot" if mode == "zeroshot" else "airlink_held_out"]
    return coax, air

# ── helpers ───────────────────────────────────────────────────────────────────
def annotate_bar(ax, bar, value: str, offset: float = 0, fontsize: float = 6.5, **kwargs):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + offset,
        value,
        ha="center", va="bottom", fontsize=fontsize, **kwargs
    )

def add_delta_bracket(ax, x_left: float, x_right: float, y: float,
                      delta_str: str, color: str = "#228B22"):
    """Draw a bracket + delta label above two bars."""
    ax.annotate(
        "", xy=(x_right, y), xytext=(x_left, y),
        arrowprops=dict(arrowstyle="-", color=color, lw=1.0),
    )
    ax.text((x_left + x_right) / 2, y + 0.005, delta_str,
            ha="center", va="bottom", fontsize=6, color=color, fontweight="bold")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 16 — Cross-domain performance: gate / mod / SNR (all 4 baseline models)
# Two-column figure, 3 panels
# ═════════════════════════════════════════════════════════════════════════════
def fig_cross_domain():
    # Collect data for all 4 baseline model variants
    variants = [
        ("cnn",  "zeroshot"),
        ("cnn",  "mixed"),
        ("attn", "zeroshot"),
        ("attn", "mixed"),
    ]
    coax_gate, air_gate = [], []
    coax_mod,  air_mod  = [], []
    coax_snr,  air_snr  = [], []
    for arch, mode in variants:
        c, a = get_row(arch, mode, aug=False)
        coax_gate.append(c["gate"]["roc_auc"])
        air_gate.append(a["gate"]["roc_auc"])
        coax_mod.append(c["mod"]["accuracy"] * 100)
        air_mod.append(a["mod"]["accuracy"] * 100)
        coax_snr.append(c["snr"]["rmse_db"])
        air_snr.append(a["snr"]["rmse_db"])

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.2))
    x  = np.arange(4)
    w  = 0.32
    xlabels = ["\n".join(lb.split("\n")) for lb in MODEL_LABELS]

    # ── Panel (a): Gate AUC ───────────────────────────────────────────────────
    ax = axes[0]
    b1 = ax.bar(x - w/2, coax_gate, w, label="CoaxSweep",
                color=COLORS_4, edgecolor="white", linewidth=0.5)
    b2 = ax.bar(x + w/2, air_gate,  w, label="AirLink",
                color=COLORS_4, edgecolor="white", linewidth=0.5,
                hatch="///", alpha=0.85)
    for bar, v in zip(b1, coax_gate):
        annotate_bar(ax, bar, f"{v:.4f}", offset=0.0003, fontsize=5.5)
    for bar, v in zip(b2, air_gate):
        annotate_bar(ax, bar, f"{v:.4f}", offset=0.0003, fontsize=5.5)
    ax.set_xticks(x); ax.set_xticklabels(xlabels, fontsize=7.5)
    ax.set_ylabel("Gate ROC-AUC")
    ax.set_ylim([0.82, 1.010])
    ax.set_title("(a) Gate Detection AUC")
    ax.yaxis.set_minor_locator(MultipleLocator(0.02))
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    # annotate domain gap recovery arrows
    ax.annotate("", xy=(2.5 + w/2, 0.986), xytext=(0.5 + w/2, 0.850),
                arrowprops=dict(arrowstyle="->", color="#1A5276", lw=1.2,
                                connectionstyle="arc3,rad=-0.25"))
    ax.text(2.0, 0.875, "+0.14 ↑\nmixed\ntraining", fontsize=5.5,
            color="#1A5276", ha="center")

    # ── Panel (b): Modulation Accuracy ───────────────────────────────────────
    ax = axes[1]
    b1 = ax.bar(x - w/2, coax_mod, w, label="CoaxSweep",
                color=COLORS_4, edgecolor="white", linewidth=0.5)
    b2 = ax.bar(x + w/2, air_mod,  w, label="AirLink",
                color=COLORS_4, edgecolor="white", linewidth=0.5,
                hatch="///", alpha=0.85)
    for bar, v in zip(b1, coax_mod):
        annotate_bar(ax, bar, "100%", offset=0.3, fontsize=5.5)
    for bar, v in zip(b2, air_mod):
        annotate_bar(ax, bar, f"{v:.1f}%", offset=0.3, fontsize=5.5)
    ax.set_xticks(x); ax.set_xticklabels(xlabels, fontsize=7.5)
    ax.set_ylabel("Modulation Accuracy (%)")
    ax.set_ylim([0, 110])
    ax.set_title("(b) Modulation Classification")
    ax.axhline(50, color="gray", ls=":", lw=1.0, alpha=0.7)
    ax.text(3.5, 51.5, "chance", color="gray", fontsize=6, ha="right")
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    # collapse annotation
    ax.annotate("modulation\ncollapse", xy=(0 + w/2, air_mod[0] + 1),
                xytext=(0 + w/2, air_mod[0] + 18),
                fontsize=5.5, ha="center", color="#8B0000",
                arrowprops=dict(arrowstyle="->", color="#8B0000", lw=0.8))

    # ── Panel (c): SNR RMSE ───────────────────────────────────────────────────
    ax = axes[2]
    b1 = ax.bar(x - w/2, coax_snr, w, label="CoaxSweep",
                color=COLORS_4, edgecolor="white", linewidth=0.5)
    b2 = ax.bar(x + w/2, air_snr,  w, label="AirLink",
                color=COLORS_4, edgecolor="white", linewidth=0.5,
                hatch="///", alpha=0.85)
    for bar, v in zip(b1, coax_snr):
        annotate_bar(ax, bar, f"{v:.2f}", offset=0.05, fontsize=5.5)
    for bar, v in zip(b2, air_snr):
        annotate_bar(ax, bar, f"{v:.2f}", offset=0.05, fontsize=5.5)
    ax.set_xticks(x); ax.set_xticklabels(xlabels, fontsize=7.5)
    ax.set_ylabel("SNR RMSE (dB)")
    ax.set_ylim([0, 5.5])
    ax.set_title("(c) SNR Estimation RMSE")
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    # 23% annotation on coax (CNN Mixed vs Attn Mixed)
    y_brace = 1.25
    ax.annotate("", xy=(x[3] - w/2, y_brace), xytext=(x[1] - w/2, y_brace),
                arrowprops=dict(arrowstyle="<->", color="#2E7D32", lw=1.2))
    ax.text((x[1] + x[3]) / 2 - w/2, y_brace + 0.08,
            "−23%", fontsize=6, ha="center", color="#2E7D32", fontweight="bold")

    # shared legend
    coax_patch = mpatches.Patch(facecolor="grey", edgecolor="white",
                                linewidth=0.5, label="CoaxSweep (cable)")
    air_patch  = mpatches.Patch(facecolor="grey", edgecolor="white",
                                linewidth=0.5, hatch="///", alpha=0.85,
                                label="AirLink (OTA)")
    legend_handles = [
        mpatches.Patch(facecolor=C_CNN_ZS, label="CNN Zero-shot"),
        mpatches.Patch(facecolor=C_CNN_MX, label="CNN Mixed"),
        mpatches.Patch(facecolor=C_ATN_ZS, label="Attn Zero-shot"),
        mpatches.Patch(facecolor=C_ATN_MX, label="Attn Mixed"),
        coax_patch, air_patch,
    ]
    fig.legend(handles=legend_handles, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, -0.14), fontsize=6.5,
               framealpha=0.8, edgecolor="grey")

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    out = os.path.join(OUT_DIR, "fig16_v2_domain.pdf")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  saved {out}")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 17 — Phase-aug before/after: modulation accuracy (headline result)
# Two-column figure, 2 panels
# ═════════════════════════════════════════════════════════════════════════════
def fig_phaseaug():
    variants = [
        ("cnn",  "zeroshot"),
        ("cnn",  "mixed"),
        ("attn", "zeroshot"),
        ("attn", "mixed"),
    ]
    air_mod_base, air_mod_aug = [], []
    air_snr_base, air_snr_aug = [], []

    for arch, mode in variants:
        _, a_base = get_row(arch, mode, aug=False)
        _, a_aug  = get_row(arch, mode, aug=True)
        air_mod_base.append(a_base["mod"]["accuracy"] * 100)
        air_mod_aug.append(a_aug["mod"]["accuracy"] * 100)
        air_snr_base.append(a_base["snr"]["rmse_db"])
        air_snr_aug.append(a_aug["snr"]["rmse_db"])

    deltas_mod = [a - b for a, b in zip(air_mod_aug, air_mod_base)]
    deltas_snr = [b - a for a, b in zip(air_snr_aug, air_snr_base)]  # lower is better

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
    x = np.arange(4)
    w = 0.32

    # ── Panel (a): Modulation Accuracy ───────────────────────────────────────
    ax = axes[0]
    bars_base = ax.bar(x - w/2, air_mod_base, w,
                       color=COLORS_4, edgecolor="white", linewidth=0.5,
                       label="Baseline (no aug)", alpha=0.6)
    bars_aug  = ax.bar(x + w/2, air_mod_aug, w,
                       color=COLORS_4, edgecolor="white", linewidth=0.5,
                       label="+Phase Aug", alpha=1.0)

    for bar, v in zip(bars_base, air_mod_base):
        annotate_bar(ax, bar, f"{v:.1f}%", offset=0.4, fontsize=6.0)
    for bar, v, d in zip(bars_aug, air_mod_aug, deltas_mod):
        annotate_bar(ax, bar, f"{v:.1f}%", offset=0.4, fontsize=6.0, fontweight="bold")

    # delta annotations with color-coded arrows
    for i, (bbar, abar, d) in enumerate(zip(bars_base, bars_aug, deltas_mod)):
        y_top = max(bbar.get_height(), abar.get_height()) + 3
        ax.annotate(
            f"+{d:.1f} pp",
            xy=(abar.get_x() + abar.get_width() / 2, abar.get_height() + 0.4),
            xytext=(bbar.get_x() + bbar.get_width() / 2 + w / 2,
                    abar.get_height() + 5 + i % 2 * 3),
            fontsize=6.5, ha="center", color="#1B5E20", fontweight="bold",
            arrowprops=dict(arrowstyle="-|>", color="#1B5E20", lw=0.8),
        )

    # highlight best result
    best_bar = bars_aug[3]   # Attn mixed
    ax.add_patch(mpatches.FancyBboxPatch(
        (best_bar.get_x() - 0.03, best_bar.get_height() - 2),
        best_bar.get_width() + 0.06,
        3.0,
        boxstyle="round,pad=0.02",
        linewidth=1.5, edgecolor="#A02020", facecolor="none",
    ))
    ax.text(best_bar.get_x() + best_bar.get_width() / 2,
            best_bar.get_height() + 1.2,
            "★ Best", ha="center", va="bottom",
            fontsize=6.5, color="#A02020", fontweight="bold")

    ax.axhline(50, color="gray", ls=":", lw=0.9, alpha=0.7)
    ax.text(3.45, 51.5, "chance", color="gray", fontsize=5.5, ha="right")
    ax.set_xticks(x); ax.set_xticklabels(MODEL_LABELS, fontsize=8)
    ax.set_ylabel("AirLink Modulation Accuracy (%)")
    ax.set_ylim([0, 110])
    ax.set_title("(a) Phase-Aug Impact: Modulation Accuracy")
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)

    # baseline hatch pattern for legend
    base_patch = mpatches.Patch(facecolor="grey", edgecolor="white",
                                linewidth=0.5, alpha=0.6, label="Baseline (no aug)")
    aug_patch  = mpatches.Patch(facecolor="grey", edgecolor="white",
                                linewidth=0.5, label="+Phase Rotation Aug")
    ax.legend(handles=[base_patch, aug_patch], loc="upper left",
              fontsize=7, framealpha=0.85)

    # ── Panel (b): SNR RMSE on AirLink ───────────────────────────────────────
    ax = axes[1]
    bars_base = ax.bar(x - w/2, air_snr_base, w,
                       color=COLORS_4, edgecolor="white", linewidth=0.5,
                       alpha=0.6, label="Baseline")
    bars_aug  = ax.bar(x + w/2, air_snr_aug, w,
                       color=COLORS_4, edgecolor="white", linewidth=0.5,
                       alpha=1.0, label="+Phase Aug")

    for bar, v in zip(bars_base, air_snr_base):
        annotate_bar(ax, bar, f"{v:.2f}", offset=0.04, fontsize=6.0)
    for bar, v in zip(bars_aug, air_snr_aug):
        annotate_bar(ax, bar, f"{v:.2f}", offset=0.04, fontsize=6.0, fontweight="bold")

    # delta annotations (RMSE reduction = improvement)
    for bbar, abar, d in zip(bars_base, bars_aug, deltas_snr):
        ax.annotate(
            f"−{d:.2f}",
            xy=(abar.get_x() + abar.get_width() / 2, abar.get_height() + 0.04),
            xytext=(abar.get_x() + abar.get_width() / 2, abar.get_height() + 0.45),
            fontsize=6.5, ha="center", color="#1B5E20", fontweight="bold",
            arrowprops=dict(arrowstyle="-|>", color="#1B5E20", lw=0.8),
        )

    ax.set_xticks(x); ax.set_xticklabels(MODEL_LABELS, fontsize=8)
    ax.set_ylabel("AirLink SNR RMSE (dB)")
    ax.set_ylim([0, 5.8])
    ax.set_title("(b) Phase-Aug Impact: SNR RMSE")
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.legend(handles=[base_patch, aug_patch], loc="upper right",
              fontsize=7, framealpha=0.85)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig17_v2_phaseaug.pdf")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  saved {out}")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 18 — AirLink ROC curves: zero-shot vs mixed (CNN and Attn)
# Single-column figure, 2 panels
# ═════════════════════════════════════════════════════════════════════════════
def fig_roc_ota():
    configs = [
        ("cnn",  "zeroshot", False, C_CNN_ZS, "--", "CNN Zero-shot"),
        ("cnn",  "mixed",    False, C_CNN_MX, "-",  "CNN Mixed"),
        ("attn", "zeroshot", False, C_ATN_ZS, "--", "Attn Zero-shot"),
        ("attn", "mixed",    False, C_ATN_MX, "-",  "Attn Mixed"),
        ("cnn",  "mixed",    True,  C_CNN_MX, "-.", "CNN Mixed +Aug"),
        ("attn", "mixed",    True,  C_ATN_MX, "-.", "Attn Mixed +Aug"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))

    # Panel (a): CoaxSweep ROC (all baseline models)
    ax = axes[0]
    for arch, mode, aug, color, ls, label in configs[:4]:
        key = f"{arch}_{mode}" + ("_phaseaug" if aug else "")
        m = load_metrics(key)
        split = "coax_val"
        fpr = np.array(m[split]["gate"]["roc_fpr"])
        tpr = np.array(m[split]["gate"]["roc_tpr"])
        auc = m[split]["gate"]["roc_auc"]
        ax.plot(fpr, tpr, color=color, ls=ls, lw=1.6,
                label=f"{label} ({auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.6, alpha=0.3)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("(a) CoaxSweep — Gate ROC")
    ax.set_xlim([-0.005, 0.12]); ax.set_ylim([0.94, 1.002])
    ax.legend(fontsize=6.5, loc="lower right")
    ax.grid(alpha=0.3, linewidth=0.5)

    # Panel (b): AirLink ROC (all configs including phase-aug)
    ax = axes[1]
    for arch, mode, aug, color, ls, label in configs:
        key = f"{arch}_{mode}" + ("_phaseaug" if aug else "")
        m = load_metrics(key)
        split = "airlink_zeroshot" if mode == "zeroshot" else "airlink_held_out"
        fpr = np.array(m[split]["gate"]["roc_fpr"])
        tpr = np.array(m[split]["gate"]["roc_tpr"])
        auc = m[split]["gate"]["roc_auc"]
        lw = 2.0 if aug else 1.6
        ax.plot(fpr, tpr, color=color, ls=ls, lw=lw,
                label=f"{label} ({auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.6, alpha=0.3)
    # shade high-recall region
    ax.axhline(0.95, color="gray", ls=":", lw=0.8, alpha=0.5)
    ax.text(0.98, 0.955, "R=0.95", fontsize=5.5, color="gray", ha="right")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("(b) AirLink (OTA) — Gate ROC")
    ax.set_xlim([-0.01, 1.01]); ax.set_ylim([-0.01, 1.02])
    ax.legend(fontsize=6.0, loc="lower right")
    ax.grid(alpha=0.3, linewidth=0.5)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig18_v2_roc_ota.pdf")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  saved {out}")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 15 (updated) — Full performance summary: all 8 model variants
# Replaces fig15_mt_summary_bar.pdf
# ═════════════════════════════════════════════════════════════════════════════
def fig_summary_v2():
    """
    Grouped bar chart showing all 8 model variants (4 baseline + 4 phaseaug)
    for both domains (CoaxSweep, AirLink).  Three panels: gate / mod / SNR.
    """
    arch_modes = [("cnn","zeroshot"),("cnn","mixed"),("attn","zeroshot"),("attn","mixed")]
    rows = []
    for aug in [False, True]:
        for arch, mode in arch_modes:
            c, a = get_row(arch, mode, aug=aug)
            rows.append({
                "label": ("CNN-ZS","CNN-Mx","Attn-ZS","Attn-Mx")[
                    arch_modes.index((arch,mode))] + ("\n+Aug" if aug else ""),
                "aug": aug,
                "arch": arch, "mode": mode,
                "coax_gate": c["gate"]["roc_auc"],
                "air_gate":  a["gate"]["roc_auc"],
                "coax_mod":  c["mod"]["accuracy"] * 100,
                "air_mod":   a["mod"]["accuracy"] * 100,
                "coax_snr":  c["snr"]["rmse_db"],
                "air_snr":   a["snr"]["rmse_db"],
            })

    n = len(rows)
    x = np.arange(n)
    w = 0.36
    # alternate colors: baseline solid, phaseaug lighter + hatch
    def bar_color(i):
        base = COLORS_4[i % 4]
        return base

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.6))
    xlabels = [r["label"] for r in rows]

    def sep_line(ax):
        ax.axvline(3.5, color="grey", ls="--", lw=0.8, alpha=0.6)

    # ── Gate AUC ──────────────────────────────────────────────────────────────
    ax = axes[0]
    for i, r in enumerate(rows):
        c_fill  = bar_color(i)
        alpha   = 1.0 if r["aug"] else 0.55
        hatch   = ".." if r["aug"] else ""
        b1 = ax.bar(i - w/2, r["coax_gate"], w, color=c_fill, alpha=alpha,
                    hatch=hatch, edgecolor="white", linewidth=0.5)
        b2 = ax.bar(i + w/2, r["air_gate"],  w, color=c_fill, alpha=alpha,
                    hatch=hatch + "///", edgecolor="white", linewidth=0.5)
        annotate_bar(ax, b1[0], f"{r['coax_gate']:.4f}", offset=0.0002, fontsize=4.8)
        annotate_bar(ax, b2[0], f"{r['air_gate']:.4f}",  offset=0.0002, fontsize=4.8)
    sep_line(ax)
    ax.set_xticks(x); ax.set_xticklabels(xlabels, fontsize=5.5, rotation=15, ha="right")
    ax.set_ylabel("Gate ROC-AUC"); ax.set_ylim([0.82, 1.008])
    ax.set_title("(a) Gate AUC"); ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.text(1.5, 0.835, "Baseline", ha="center", fontsize=6, color="grey")
    ax.text(5.5, 0.835, "+Phase Aug", ha="center", fontsize=6, color="grey")

    # ── Modulation Accuracy ───────────────────────────────────────────────────
    ax = axes[1]
    for i, r in enumerate(rows):
        c_fill = bar_color(i)
        alpha  = 1.0 if r["aug"] else 0.55
        hatch  = ".." if r["aug"] else ""
        b1 = ax.bar(i - w/2, r["coax_mod"], w, color=c_fill, alpha=alpha,
                    hatch=hatch, edgecolor="white", linewidth=0.5)
        b2 = ax.bar(i + w/2, r["air_mod"],  w, color=c_fill, alpha=alpha,
                    hatch=hatch + "///", edgecolor="white", linewidth=0.5)
        annotate_bar(ax, b2[0], f"{r['air_mod']:.1f}", offset=0.3, fontsize=4.8)
    sep_line(ax)
    ax.axhline(50, color="gray", ls=":", lw=0.8, alpha=0.6)
    ax.set_xticks(x); ax.set_xticklabels(xlabels, fontsize=5.5, rotation=15, ha="right")
    ax.set_ylabel("Modulation Accuracy (%)"); ax.set_ylim([0, 115])
    ax.set_title("(b) Mod Accuracy"); ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    # best result annotation
    ax.annotate("92.3%\n★ Best", xy=(7 + w/2, 92.3), xytext=(6.2, 102),
                fontsize=6, color="#A02020", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#A02020", lw=0.8))

    # ── SNR RMSE ─────────────────────────────────────────────────────────────
    ax = axes[2]
    for i, r in enumerate(rows):
        c_fill = bar_color(i)
        alpha  = 1.0 if r["aug"] else 0.55
        hatch  = ".." if r["aug"] else ""
        b1 = ax.bar(i - w/2, r["coax_snr"], w, color=c_fill, alpha=alpha,
                    hatch=hatch, edgecolor="white", linewidth=0.5)
        b2 = ax.bar(i + w/2, r["air_snr"],  w, color=c_fill, alpha=alpha,
                    hatch=hatch + "///", edgecolor="white", linewidth=0.5)
        annotate_bar(ax, b2[0], f"{r['air_snr']:.2f}", offset=0.04, fontsize=4.8)
    sep_line(ax)
    ax.set_xticks(x); ax.set_xticklabels(xlabels, fontsize=5.5, rotation=15, ha="right")
    ax.set_ylabel("SNR RMSE (dB)"); ax.set_ylim([0, 5.5])
    ax.set_title("(c) SNR RMSE"); ax.grid(axis="y", alpha=0.3, linewidth=0.5)

    # shared legend
    coax_p = mpatches.Patch(facecolor="grey", alpha=0.7, label="CoaxSweep (cable)")
    air_p  = mpatches.Patch(facecolor="grey", alpha=0.7, hatch="///", label="AirLink (OTA)")
    base_p = mpatches.Patch(facecolor="grey", alpha=0.55, label="Baseline (no aug)")
    aug_p  = mpatches.Patch(facecolor="grey", alpha=1.0, hatch="..", label="+Phase Aug")
    color_handles = [
        mpatches.Patch(facecolor=C_CNN_ZS, label="CNN Zero-shot"),
        mpatches.Patch(facecolor=C_CNN_MX, label="CNN Mixed"),
        mpatches.Patch(facecolor=C_ATN_ZS, label="Attn Zero-shot"),
        mpatches.Patch(facecolor=C_ATN_MX, label="Attn Mixed"),
    ]
    fig.legend(handles=color_handles + [coax_p, air_p, base_p, aug_p],
               ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.18),
               fontsize=6.0, framealpha=0.8, edgecolor="grey")

    plt.tight_layout(rect=[0, 0.11, 1, 1])
    out = os.path.join(OUT_DIR, "fig15_v2_summary.pdf")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  saved {out}")


# ── main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[mt-figures-v2] generating...")
    fig_cross_domain()   # fig16
    fig_phaseaug()       # fig17
    fig_roc_ota()        # fig18
    fig_summary_v2()     # fig15 updated
    print("[mt-figures-v2] done")
