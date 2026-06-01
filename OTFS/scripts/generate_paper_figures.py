#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_paper_figures.py — regenerate all paper figures from data files.

Reads:
    OTFS/results/doppler_sweep_snr{10,15,20}.json
    OTFS/results/monostatic_summary.json
Writes:
    OTFS/paper_figures/fig_otfs_chain.pdf
    OTFS/paper_figures/fig_doppler_sweep.pdf
    OTFS/paper_figures/fig_closed_grid.pdf
    OTFS/paper_figures/fig_monostatic_cfo.pdf
    OTFS/paper_figures/fig_monostatic_pkt.pdf
"""

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RES  = os.path.join(ROOT, "results")
FIG  = os.path.join(ROOT, "paper_figures")
os.makedirs(FIG, exist_ok=True)

# IEEE-friendly defaults: small fonts, tight layout, vector PDF
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":          8,
    "axes.labelsize":     8,
    "axes.titlesize":     9,
    "xtick.labelsize":    7,
    "ytick.labelsize":    7,
    "legend.fontsize":    7,
    "lines.linewidth":    1.2,
    "lines.markersize":   4,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "grid.linestyle":     ":",
    "savefig.bbox":       "tight",
    "savefig.dpi":        300,
    "pdf.fonttype":       42,    # TrueType, for IEEE compliance
})


# ──────────────────────────────────────────────────────────────────────────
# Figure 1 — system architecture (TX chain + channel + RX chain)
# ──────────────────────────────────────────────────────────────────────────
def fig_chain():
    fig, ax = plt.subplots(figsize=(7.0, 1.7))
    ax.set_xlim(0, 100); ax.set_ylim(0, 12); ax.axis("off")

    def box(x, y, w, h, text, fc="#e8f0ff", ec="#1a4480"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                                     linewidth=0.9, facecolor=fc, edgecolor=ec))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.5)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                      arrowstyle="-|>", mutation_scale=8, lw=0.7,
                                      color="#444"))

    # ── TX row (y=8) ──
    tx_y = 8
    boxes_tx = [
        ("bytes", 1, 7),
        ("scrambler\n+ LDPC enc", 9, 11),
        ("QPSK map", 21, 8),
        ("DD-grid\n+ pilot", 30, 9),
        ("ISFFT\n+ CP", 40, 8),
        ("STF/LTF\npreamble", 49, 9),
        ("PlutoSDR\nTX",  59, 8),
    ]
    px = 1
    for t, x, w in boxes_tx:
        box(x, tx_y, w, 3, t)
        if px > 1: arrow(px, tx_y + 1.5, x, tx_y + 1.5)
        px = x + w
    ax.text(34, tx_y + 4.0, "TX (Jetson Orin / x86)", ha="center", fontsize=8,
            color="#1a4480", style="italic")

    # ── Channel block ──
    ch_x, ch_y, ch_w, ch_h = 70, 5.5, 16, 4
    ax.add_patch(FancyBboxPatch((ch_x, ch_y), ch_w, ch_h, boxstyle="round,pad=0.2",
                                 linewidth=1.0, facecolor="#fff2cc", edgecolor="#996600"))
    ax.text(ch_x + ch_w/2, ch_y + ch_h/2 + 0.5,
            "Wireless channel\n(2-tap doubly-selective\n+ Doppler)",
            ha="center", va="center", fontsize=7.5)
    arrow(67, tx_y + 1.5, 70, ch_y + ch_h/2 + 0.5)

    # ── RX row (y=1) ──
    rx_y = 1
    boxes_rx = [
        ("PlutoSDR\nRX",     1,  8),
        ("Sync\n(Schmidl-\nCox + xcorr)", 11, 11),
        ("SFFT\n+ DD-LMMSE", 24, 10),
        ("Soft demap\n(QPSK LLR)", 36, 11),
        ("LDPC dec\n(5G NR BG2)", 49, 11),
        ("descram\n+ CRC", 62, 9),
        ("payload", 73, 7),
    ]
    px = 1
    for t, x, w in boxes_rx:
        box(x, rx_y, w, 3, t, fc="#eff5e8", ec="#3a6b2c")
        if px > 1: arrow(px, rx_y + 1.5, x, rx_y + 1.5)
        px = x + w
    ax.text(40, rx_y - 0.7, "RX (x86)", ha="center", fontsize=8,
            color="#3a6b2c", style="italic")

    arrow(ch_x + ch_w / 2, ch_y, ch_x + ch_w / 2, rx_y + 3)
    fig.savefig(os.path.join(FIG, "fig_otfs_chain.pdf"))
    plt.close(fig)
    print("wrote fig_otfs_chain.pdf")


# ──────────────────────────────────────────────────────────────────────────
# Figure 2 — software-loopback Doppler sweep (OTFS vs OFDM, 3 SNRs)
# ──────────────────────────────────────────────────────────────────────────
def fig_doppler():
    snrs = [10, 15, 20]
    data = {}
    for s in snrs:
        p = os.path.join(RES, f"doppler_sweep_snr{s}.json")
        data[s] = json.load(open(p))

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), sharey=True)
    colors = {"otfs": "#1f77b4", "ofdm": "#d62728"}

    for ax, snr in zip(axes, snrs):
        rows = data[snr]["rows"]
        fd   = np.array([r["fd_hz"] for r in rows])
        otfs = 100.0 * np.array([r["otfs_pkt_ok"] for r in rows]) / rows[0]["n_trials"]
        ofdm = 100.0 * np.array([r["ofdm_pkt_ok"] for r in rows]) / rows[0]["n_trials"]
        ax.plot(fd, otfs, "o-", color=colors["otfs"], label="OTFS + LDPC")
        ax.plot(fd, ofdm, "s--", color=colors["ofdm"], label="OFDM + LDPC")
        ax.set_xscale("symlog", linthresh=100)
        ax.set_xlim(0, 6000)
        ax.set_ylim(-3, 105)
        ax.set_xlabel("Doppler $f_d$ (Hz)")
        ax.set_title(f"SNR = {snr} dB")
        ax.set_xticks([0, 100, 500, 1000, 2000, 5000])
        ax.set_xticklabels(["0", "100", "500", "1k", "2k", "5k"])
        # Annotate OFDM cliff
        if snr == 20:
            ax.annotate("OFDM cliff", xy=(300, 3.3), xytext=(900, 30),
                        fontsize=7, color=colors["ofdm"],
                        arrowprops=dict(arrowstyle="->", color=colors["ofdm"],
                                        lw=0.6, shrinkA=2))
            ax.annotate("OTFS Doppler\nceiling (k_max=1)", xy=(5000, 0),
                        xytext=(2200, 70), fontsize=7, color=colors["otfs"],
                        arrowprops=dict(arrowstyle="->", color=colors["otfs"],
                                        lw=0.6, shrinkA=2))
    axes[0].set_ylabel("Packet decode rate (%)")
    axes[0].legend(loc="upper right", framealpha=0.92)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_doppler_sweep.pdf"))
    plt.close(fig)
    print("wrote fig_doppler_sweep.pdf")


# ──────────────────────────────────────────────────────────────────────────
# Figure 3 — closed 2x2 grid (OTA, fc=2.45 GHz, tx_gain=0)
# ──────────────────────────────────────────────────────────────────────────
def fig_closed_grid():
    # Numbers come from OTFS_VS_OFDM_LDPC_CLOSED_GRID.md
    labels  = ["OFDM\nno-FEC", "OFDM\n+LDPC", "OTFS\n+LDPC"]
    val_0   = [0.0,  32.9,  35.8]
    val_n5  = [0.0,   5.3,   0.0]   # current OTA channel back-to-back
    val_n15 = [0.0,   0.0,   0.0]

    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    x = np.arange(len(labels)); w = 0.27
    b1 = ax.bar(x - w, val_0,   w, label="$g_\\mathrm{TX}$ = 0 dB",
                color="#1f77b4", edgecolor="#08306b", linewidth=0.6)
    b2 = ax.bar(x,     val_n5,  w, label="$g_\\mathrm{TX}$ = −5 dB",
                color="#ff7f0e", edgecolor="#8c4a08", linewidth=0.6)
    b3 = ax.bar(x + w, val_n15, w, label="$g_\\mathrm{TX}$ = −15 dB",
                color="#bdbdbd", edgecolor="#525252", linewidth=0.6)
    for bars in [b1, b2, b3]:
        for r in bars:
            h = r.get_height()
            if h > 0:
                ax.text(r.get_x() + r.get_width()/2, h + 1.2,
                        f"{h:.0f}", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Pkt decode rate (\\%)")
    ax.set_ylim(0, 45)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=6.5)
    ax.set_title("OTA closed 2$\\times$2 grid, fc = 2.45 GHz")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_closed_grid.pdf"))
    plt.close(fig)
    print("wrote fig_closed_grid.pdf")


# ──────────────────────────────────────────────────────────────────────────
# Figure 4 — monostatic CFO histogram (shared LO eliminates drift)
# ──────────────────────────────────────────────────────────────────────────
def fig_mono_cfo():
    d = json.load(open(os.path.join(RES, "monostatic_summary.json")))
    tp = d["two_pluto_static"]; ms = d["mono_static"]
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    # Show on a shifted x-axis: bin around mean - mean for two_pluto so both fit
    cfo_tp = np.array(tp["cfo_samples"]) - tp["cfo_mean"]
    cfo_ms = np.array(ms["cfo_samples"]) - ms["cfo_mean"]
    bins = np.linspace(-400, 400, 30)
    ax.hist(cfo_tp, bins=bins, alpha=0.55, color="#d62728",
            label=f"Two-Pluto\n($\\mu$ = {tp['cfo_mean']:+.0f} Hz, $\\sigma$ = {tp['cfo_std']:.0f} Hz)",
            edgecolor="#7f1414", linewidth=0.5)
    ax.hist(cfo_ms, bins=bins, alpha=0.55, color="#1f77b4",
            label=f"Monostatic\n($\\mu$ = {ms['cfo_mean']:+.0f} Hz, $\\sigma$ = {ms['cfo_std']:.0f} Hz)",
            edgecolor="#08306b", linewidth=0.5)
    ax.set_xlabel("CFO deviation from per-setup mean (Hz)")
    ax.set_ylabel("Count")
    ax.set_title(r"Single-Pluto monostatic eliminates 4.5\,kHz LO offset")
    ax.legend(loc="upper left", framealpha=0.92, fontsize=6.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_monostatic_cfo.pdf"))
    plt.close(fig)
    print("wrote fig_monostatic_cfo.pdf")


# ──────────────────────────────────────────────────────────────────────────
# Figure 5 — monostatic packet-rate (motion vs static across 4 conditions)
# ──────────────────────────────────────────────────────────────────────────
def fig_mono_pkt():
    d = json.load(open(os.path.join(RES, "monostatic_summary.json")))
    pr = d["pkt_rates"]
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    labels = ["Two-Pluto\nstatic", "Two-Pluto\nmotion", "Monostatic\nstatic", "Monostatic\nmotion"]
    vals = [pr["two_pluto_static"]*100, pr["two_pluto_motion"]*100,
            pr["mono_static"]*100,      pr["mono_motion"]*100]
    colors = ["#bdbdbd", "#ff7f0e", "#aec7e8", "#1f77b4"]
    bars = ax.bar(labels, vals, color=colors,
                   edgecolor=["#525252", "#8c4a08", "#08306b", "#08306b"],
                   linewidth=0.7)
    for r, v in zip(bars, vals):
        ax.text(r.get_x() + r.get_width()/2, v + 0.8, f"{v:.1f}%",
                ha="center", fontsize=7.5, fontweight="bold")
    ax.set_ylabel("OTFS+LDPC packet decode rate (\\%)")
    ax.set_ylim(0, 55)
    ax.set_title("Monostatic + motion: 0\\,\\% $\\to$ 45.3\\,\\%")
    # Annotation pointing at the winning bar
    ax.annotate("delay-Doppler\ndiversity", xy=(3, 45.3), xytext=(1.7, 47),
                fontsize=7, ha="center",
                arrowprops=dict(arrowstyle="->", lw=0.7, color="black"))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_monostatic_pkt.pdf"))
    plt.close(fig)
    print("wrote fig_monostatic_pkt.pdf")


# ──────────────────────────────────────────────────────────────────────────
# Figure 6 — pilot amplitude std (motion signature)
# ──────────────────────────────────────────────────────────────────────────
def fig_pilot_std():
    d = json.load(open(os.path.join(RES, "monostatic_summary.json")))
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    labels = ["Two-Pluto\nstatic", "Two-Pluto\nmotion", "Monostatic\nstatic", "Monostatic\nmotion"]
    keys = ["two_pluto_static", "two_pluto_motion", "mono_static", "mono_motion"]
    vals = [d[k]["pilot_std"] for k in keys]
    bars = ax.bar(labels, vals, color=["#bdbdbd", "#ff7f0e", "#aec7e8", "#1f77b4"],
                   edgecolor="#444", linewidth=0.6)
    for r, v in zip(bars, vals):
        ax.text(r.get_x() + r.get_width()/2, v + 8, f"{v:.0f}",
                ha="center", fontsize=7.5)
    ax.set_ylabel("Pilot magnitude std (a.u.)")
    ax.set_ylim(0, 550)
    ax.set_title("Motion modulates pilot amplitude $\\sim$5$\\times$")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_pilot_std.pdf"))
    plt.close(fig)
    print("wrote fig_pilot_std.pdf")


if __name__ == "__main__":
    fig_chain()
    fig_doppler()
    fig_closed_grid()
    fig_mono_cfo()
    fig_mono_pkt()
    fig_pilot_std()
    print("\nAll figures written to:", FIG)
