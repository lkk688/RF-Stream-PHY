#!/usr/bin/env python3
"""
generate_benchmark_figure.py — fig19_v2_latency.pdf

Two-panel publication figure comparing inference latency and throughput
for MT-PreamCNN (CNN) and MT-PreamCNN-Attn (Attn) across platforms and backends.

Left:  Latency (ms, log scale) vs batch size — 4 configurations
Right: Throughput (samples/s) vs batch size — same 4 configurations

Configurations shown:
  x86 PyTorch CPU   — baseline, no optimization
  x86 ORT CPU       — graph-optimized ONNX Runtime
  Jetson ORT CPU    — edge ARM CPU baseline
  Jetson TRT FP16   — TensorRT FP16 GPU on Orin

Reads:
  rf_stream/benchmark_results_x86.json    (x86 host)
  rf_stream/benchmark_results_jetson.json (Jetson Orin)

Output:
  rf_stream/paper_figures/fig19_v2_latency.pdf
"""

import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

OUT_DIR  = "rf_stream/paper_figures"
X86_JSON = "rf_stream/benchmark_results_x86.json"
JET_JSON = "rf_stream/benchmark_results_jetson.json"

BATCH_SIZES = [1, 8, 32, 64]

# Consistent color / style palette
CONFIGS = [
    # (label,            json_file,  backend,        arch,   color,    marker, ls)
    ("x86 PyTorch-CPU",  X86_JSON,  "pytorch_cpu",  "cnn",  "#4878CF", "o", "-"),
    ("x86 PyTorch-CPU",  X86_JSON,  "pytorch_cpu",  "attn", "#4878CF", "s", "--"),
    ("x86 ORT-CPU",      X86_JSON,  "ort_cpu",      "cnn",  "#1F4E8C", "o", "-"),
    ("x86 ORT-CPU",      X86_JSON,  "ort_cpu",      "attn", "#1F4E8C", "s", "--"),
    ("Jetson ORT-CPU",   JET_JSON,  "ort_cpu",      "cnn",  "#E07B39", "o", "-"),
    ("Jetson ORT-CPU",   JET_JSON,  "ort_cpu",      "attn", "#E07B39", "s", "--"),
    ("Jetson TRT-FP16",  JET_JSON,  "trtexec",      "cnn",  "#A02020", "o", "-"),
    ("Jetson TRT-FP16",  JET_JSON,  "trtexec",      "attn", "#A02020", "s", "--"),
]


def load_json(path):
    with open(path) as f:
        return json.load(f)


def get_series(data, backend, arch, metric):
    """Extract metric values for each batch size."""
    vals = []
    bdata = data.get("backends", {}).get(backend, {}).get(arch, {})
    for bs in BATCH_SIZES:
        entry = bdata.get(str(bs), {})
        vals.append(entry.get(metric, float("nan")))
    return np.array(vals)


def fig_latency_throughput():
    x86 = load_json(X86_JSON)
    jet = load_json(JET_JSON)
    data_map = {X86_JSON: x86, JET_JSON: jet}

    os.makedirs(OUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))

    ax_lat, ax_tput = axes

    x = np.arange(len(BATCH_SIZES))
    width_bar = 0.09
    bar_offsets = np.linspace(-3.5, 3.5, len(CONFIGS)) * width_bar

    arch_labels = {"cnn": "CNN", "attn": "Attn"}

    # ── Left: latency (log scale, line+marker per config) ─────────────────
    legend_handles = {}
    for i, (label, jpath, backend, arch, color, marker, ls) in enumerate(CONFIGS):
        data = data_map[jpath]
        lat = get_series(data, backend, arch, "mean_ms")
        lkey = label
        akey = arch_labels[arch]
        ls_final = ls
        lw = 1.8 if arch == "cnn" else 1.8
        alpha = 0.95

        h, = ax_lat.plot(x, lat, color=color, marker=marker, ls=ls_final,
                         lw=lw, markersize=5, alpha=alpha,
                         label=f"{label} ({akey})")
        legend_handles[f"{label}_{arch}"] = h

    ax_lat.set_yscale("log")
    ax_lat.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v:.2f}" if v < 1 else f"{v:.1f}"))
    ax_lat.set_xticks(x)
    ax_lat.set_xticklabels([str(b) for b in BATCH_SIZES])
    ax_lat.set_xlabel("Batch Size", fontsize=9)
    ax_lat.set_ylabel("Latency (ms, log scale)", fontsize=9)
    ax_lat.set_title("(a) Inference Latency", fontsize=9, fontweight="bold")
    ax_lat.grid(True, which="both", ls=":", alpha=0.4)
    ax_lat.tick_params(labelsize=8)

    # Annotation: TRT FP16 bs=1
    trt_cnn_lat = get_series(jet, "trtexec", "cnn", "mean_ms")[0]
    trt_attn_lat = get_series(jet, "trtexec", "attn", "mean_ms")[0]
    ax_lat.annotate(f"{trt_cnn_lat:.2f}ms\n(TRT CNN)",
                    xy=(0, trt_cnn_lat), xytext=(0.3, trt_cnn_lat * 2.2),
                    fontsize=6.5, color="#A02020",
                    arrowprops=dict(arrowstyle="->", color="#A02020", lw=0.8))
    ax_lat.annotate(f"{trt_attn_lat:.2f}ms\n(TRT Attn)",
                    xy=(0, trt_attn_lat), xytext=(0.5, trt_attn_lat * 4.5),
                    fontsize=6.5, color="#A02020",
                    arrowprops=dict(arrowstyle="->", color="#A02020", lw=0.8))

    # ── Right: throughput (samples/s) ─────────────────────────────────────
    for i, (label, jpath, backend, arch, color, marker, ls) in enumerate(CONFIGS):
        data = data_map[jpath]
        tput = get_series(data, backend, arch, "throughput_sps")
        akey = arch_labels[arch]
        ax_tput.plot(x, tput / 1e3, color=color, marker=marker, ls=ls,
                     lw=1.8, markersize=5, alpha=0.95,
                     label=f"{label} ({akey})")

    ax_tput.set_xticks(x)
    ax_tput.set_xticklabels([str(b) for b in BATCH_SIZES])
    ax_tput.set_xlabel("Batch Size", fontsize=9)
    ax_tput.set_ylabel("Throughput (k samples/s)", fontsize=9)
    ax_tput.set_title("(b) Inference Throughput", fontsize=9, fontweight="bold")
    ax_tput.grid(True, ls=":", alpha=0.4)
    ax_tput.tick_params(labelsize=8)

    # ── Shared legend (deduplicate by label) ──────────────────────────────
    # Group by backend/platform: CNN solid, Attn dashed — show 4 platform entries
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_entries = []
    seen_labels = set()
    for label, jpath, backend, arch, color, marker, ls in CONFIGS:
        if label not in seen_labels:
            legend_entries.append(
                Line2D([0], [0], color=color, lw=2, label=label))
            seen_labels.add(label)
    # Architecture markers
    legend_entries.append(Line2D([0], [0], color="gray", lw=1.8, marker="o",
                                  ls="-",  label="CNN (solid)"))
    legend_entries.append(Line2D([0], [0], color="gray", lw=1.8, marker="s",
                                  ls="--", label="Attn (dashed)"))

    fig.legend(handles=legend_entries, loc="lower center", ncol=3,
               fontsize=7.5, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.18))

    fig.suptitle("MT-PreamCNN Deployment Benchmark: x86 vs. Jetson Orin",
                 fontsize=10, fontweight="bold", y=1.01)

    plt.tight_layout(rect=[0, 0.12, 1, 1])
    out = os.path.join(OUT_DIR, "fig19_v2_latency.pdf")
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved: {out}")


def fig_latency_bar():
    """Alternative: grouped bar chart for batch=1 and batch=64 side-by-side."""
    x86 = load_json(X86_JSON)
    jet = load_json(JET_JSON)
    data_map = {X86_JSON: x86, JET_JSON: jet}

    configs_bar = [
        ("x86\nPyTorch-CPU", X86_JSON,  "pytorch_cpu", "#4878CF"),
        ("x86\nORT-CPU",     X86_JSON,  "ort_cpu",     "#1F4E8C"),
        ("Jetson\nORT-CPU",  JET_JSON,  "ort_cpu",     "#E07B39"),
        ("Jetson\nTRT-FP16", JET_JSON,  "trtexec",     "#A02020"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=False)

    for ax_idx, (bs, ax) in enumerate(zip([1, 64], axes)):
        n = len(configs_bar)
        x = np.arange(n)
        w = 0.35

        for j, arch in enumerate(["cnn", "attn"]):
            lats = []
            for label, jpath, backend, color in configs_bar:
                data = data_map[jpath]
                bdata = data.get("backends", {}).get(backend, {}).get(arch, {})
                lats.append(bdata.get(str(bs), {}).get("mean_ms", float("nan")))

            colors = [cfg[3] for cfg in configs_bar]
            offset = (j - 0.5) * w
            bars = ax.bar(x + offset, lats, width=w,
                          color=colors, alpha=0.85 if j == 0 else 0.55,
                          edgecolor="white", linewidth=0.5,
                          label=f"{'CNN' if j==0 else 'Attn'}")

            for bar, val in zip(bars, lats):
                if not np.isnan(val):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.01 * ax.get_ylim()[1],
                            f"{val:.2f}", ha="center", va="bottom",
                            fontsize=6.5, rotation=0)

        ax.set_xticks(x)
        ax.set_xticklabels([c[0] for c in configs_bar], fontsize=8)
        ax.set_ylabel("Latency (ms)", fontsize=9)
        ax.set_title(f"({'ab'[ax_idx]}) Batch size = {bs}", fontsize=9, fontweight="bold")
        ax.grid(axis="y", ls=":", alpha=0.4)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=8, framealpha=0.8)

    fig.suptitle("MT-PreamCNN Inference Latency: x86 vs. Jetson Orin (FP16 TRT)",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig19_v2_latency_bar.pdf")
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved: {out}")


def print_summary_table():
    x86 = load_json(X86_JSON)
    jet = load_json(JET_JSON)
    data_map = {X86_JSON: x86, JET_JSON: jet}

    print("\n=== Latency Summary (ms) — batch size = 1 ===")
    print(f"{'Platform/Backend':<28} {'CNN':>8} {'Attn':>8}  {'Speedup (Attn)':>14}")
    print("-" * 62)
    for label, jpath, backend, color in [
        ("x86 PyTorch-CPU",  X86_JSON, "pytorch_cpu", ""),
        ("x86 ORT-CPU",      X86_JSON, "ort_cpu",     ""),
        ("Jetson ORT-CPU",   JET_JSON, "ort_cpu",     ""),
        ("Jetson TRT-FP16",  JET_JSON, "trtexec",     ""),
    ]:
        data = data_map[jpath]
        cnn_lat  = data["backends"].get(backend, {}).get("cnn",  {}).get("1", {}).get("mean_ms", float("nan"))
        attn_lat = data["backends"].get(backend, {}).get("attn", {}).get("1", {}).get("mean_ms", float("nan"))
        if not np.isnan(cnn_lat):
            speedup = cnn_lat / attn_lat if attn_lat > 0 else float("nan")
            print(f"{label:<28} {cnn_lat:>8.2f} {attn_lat:>8.2f}  {speedup:>13.1f}×")

    print("\n=== Throughput Summary (sps) — batch size = 64 ===")
    print(f"{'Platform/Backend':<28} {'CNN':>10} {'Attn':>10}")
    print("-" * 52)
    for label, jpath, backend, color in [
        ("x86 PyTorch-CPU",  X86_JSON, "pytorch_cpu", ""),
        ("x86 ORT-CPU",      X86_JSON, "ort_cpu",     ""),
        ("Jetson ORT-CPU",   JET_JSON, "ort_cpu",     ""),
        ("Jetson TRT-FP16",  JET_JSON, "trtexec",     ""),
    ]:
        data = data_map[jpath]
        cnn_t  = data["backends"].get(backend, {}).get("cnn",  {}).get("64", {}).get("throughput_sps", float("nan"))
        attn_t = data["backends"].get(backend, {}).get("attn", {}).get("64", {}).get("throughput_sps", float("nan"))
        print(f"{label:<28} {cnn_t:>10.0f} {attn_t:>10.0f}")


if __name__ == "__main__":
    print_summary_table()
    fig_latency_throughput()
    fig_latency_bar()
    print("\nDone.")
