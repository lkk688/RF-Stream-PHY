#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stage_data_for_hf.py — Stage raw experiment data from /tmp/ into
                       OTFS/data_staging/ for HuggingFace upload.

Layout produced:
    OTFS/data_staging/
        closed_grid_0db_otfs/           # 35.8% pkt rate, fc=2.45 GHz
        closed_grid_0db_ofdm/           # 32.9% pkt rate, fc=2.45 GHz
        closed_grid_n5db_otfs_v2/       # 0% (channel past cliff), monostatic-grade gain reset
        closed_grid_n5db_ofdm_v2/       # 0/92 (raw BER 0.47)
        ota_doppler/                    # 6 runs: {otfs,ofdm}_fd{0,1000,3000}
        monostatic/                     # static + motion runs (single Pluto)
        tx_iq/                          # TX waveform IQ files (small)
        DATASET.md                      # Top-level dataset card
"""

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime

ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STAGE = os.path.join(ROOT, "data_staging")

# (src, dst_split, description, key_result)
RUNS = [
    # closed-grid 0 dB
    ("/tmp/otfs_ldpc_now/run_otfs_rx_20260531_152717",
     "closed_grid_0db_otfs",
     "OTFS+LDPC OTA, fc=2.45 GHz, tx_gain=0 dB, 500 caps, paper Sec. 5.2",
     "179/500 = 35.8% pkt rate"),

    ("/home/lkk/Developer/AIsensing/rf_stream_rx_runs_step10_ldpc_0dB_v2/run_20260531_152144",
     "closed_grid_0db_ofdm",
     "OFDM+LDPC OTA, fc=2.45 GHz, tx_gain=0 dB, step10 RX with NPZ save; LDPC post-decoded by ofdm_ldpc_postdecode.py, paper Sec. 5.2",
     "161/489 attempts = 32.9% pkt rate (post-LDPC)"),

    # closed-grid -5 dB (v2 = post Pluto-reset, paper Sec. 5.4)
    ("/tmp/otfs_ldpc_n5_v2/run_otfs_rx_20260531_161601",
     "closed_grid_n5db_otfs_v2",
     "OTFS+LDPC OTA, fc=2.45 GHz, tx_gain=-5 dB, 500 caps, after deterministic Pluto gain reset (3x gain rewrite); channel below LDPC cliff so 0/500",
     "0/500 = 0.0% (snr_post median ~-4 dB)"),

    ("/tmp/ofdm_ldpc_n5_v2/run_20260531_162432",
     "closed_grid_n5db_ofdm_v2",
     "OFDM+LDPC OTA, fc=2.45 GHz, tx_gain=-5 dB, step10 RX, back-to-back with OTFS-n5db_v2; channel past LDPC cliff",
     "0/92 attempts = 0%, raw BER median 0.47"),
]

# OTA Doppler: 6 subdirs each with a single run dir
OTA_DOPPLER_RUNS = [
    ("/tmp/ota_doppler/otfs_fd0",     "ota_doppler/otfs_fd0",     "OTFS+LDPC, fc=2.42 GHz, synthetic Doppler fd=0 Hz (multipath only)",       "71/500 = 14.2%"),
    ("/tmp/ota_doppler/otfs_fd1000",  "ota_doppler/otfs_fd1000",  "OTFS+LDPC, fc=2.42 GHz, synthetic Doppler fd=1000 Hz",                     "19/500 = 3.8%"),
    ("/tmp/ota_doppler/otfs_fd3000",  "ota_doppler/otfs_fd3000",  "OTFS+LDPC, fc=2.42 GHz, synthetic Doppler fd=3000 Hz (OTFS k_max ceiling)", "0/500 = 0.0%"),
    ("/tmp/ota_doppler/ofdm_fd0",     "ota_doppler/ofdm_fd0",     "OFDM+LDPC, fc=2.42 GHz, synthetic Doppler fd=0 Hz",                        "0/23 attempts = 0% (raw BER 0.47)"),
    ("/tmp/ota_doppler/ofdm_fd1000",  "ota_doppler/ofdm_fd1000",  "OFDM+LDPC, fc=2.42 GHz, synthetic Doppler fd=1000 Hz",                     "0/46 attempts = 0%"),
    ("/tmp/ota_doppler/ofdm_fd3000",  "ota_doppler/ofdm_fd3000",  "OFDM+LDPC, fc=2.42 GHz, synthetic Doppler fd=3000 Hz",                     "0/38 attempts = 0%"),
]

# Monostatic runs
MONO_RUNS = [
    ("/tmp/mono/static/run_otfs_rx_20260531_201511",
     "monostatic/static",
     "Single-Pluto monostatic loopback, OTFS+LDPC, RX antenna cable-extended and HELD STILL, paper Sec. 5.4",
     "0/300 = 0%, mean CFO -8 Hz (vs -4549 Hz two-Pluto)"),
    ("/tmp/mono/motion/run_otfs_rx_20260531_202038",
     "monostatic/motion",
     "Single-Pluto monostatic loopback, OTFS+LDPC, user hand-waves RX antenna ~30 cm at ~1 swing/sec, paper Sec. 5.4",
     "136/300 = 45.3% pkt rate -- motion induces delay-Doppler diversity"),
]

# TX IQ files (small)
TX_IQ_FILES = [
    "/tmp/otfs_ldpc_seed42.iq",
    "/tmp/otfs_ldpc_seed42.iq.tx_meta.json",
    "/tmp/otfs_ldpc_seed42.iq.tx_truth.jsonl",
    "/tmp/otfs_ldpc_seed42_fd0.iq",
    "/tmp/otfs_ldpc_seed42_fd1000.iq",
    "/tmp/otfs_ldpc_seed42_fd3000.iq",
    "/tmp/otfs_ldpc_seed42_fd5.iq",
    "/tmp/ofdm_ldpc_seed42.iq",
    "/tmp/ofdm_ldpc_seed42.iq.tx_meta.json",
    "/tmp/ofdm_ldpc_seed42.iq.tx_truth.jsonl",
    "/tmp/ofdm_ldpc_seed42_fd0.iq",
    "/tmp/ofdm_ldpc_seed42_fd1000.iq",
    "/tmp/ofdm_ldpc_seed42_fd3000.iq",
]


def copy_run(src: str, dst_parent: str, dst_name: str, description: str, key_result: str) -> dict:
    dst_dir = os.path.join(STAGE, dst_parent, dst_name) if "/" in dst_parent + dst_name else os.path.join(STAGE, dst_name)
    # Allow nested layout: ota_doppler/otfs_fd0
    dst_dir = os.path.join(STAGE, dst_name)
    os.makedirs(dst_dir, exist_ok=True)
    if not os.path.isdir(src):
        print(f"  [SKIP] missing: {src}")
        return {"split": dst_name, "src": src, "n_files": 0, "size_bytes": 0,
                "description": description, "key_result": key_result, "status": "missing"}
    # rsync for speed + accurate size accounting
    subprocess.run(["rsync", "-a", "--info=stats0", src + "/", dst_dir + "/"], check=True)
    n_files = sum(len(files) for _, _, files in os.walk(dst_dir))
    size = subprocess.check_output(["du", "-sb", dst_dir]).split()[0]
    # Per-split README
    write_split_readme(dst_dir, dst_name, src, description, key_result, n_files, int(size))
    return {"split": dst_name, "src": src, "n_files": n_files,
            "size_bytes": int(size), "description": description,
            "key_result": key_result, "status": "ok"}


def write_split_readme(dst_dir: str, name: str, src: str,
                        description: str, key_result: str,
                        n_files: int, size_bytes: int):
    with open(os.path.join(dst_dir, "README.md"), "w") as f:
        f.write(f"# {name}\n\n")
        f.write(f"**Description:** {description}\n\n")
        f.write(f"**Key result:** {key_result}\n\n")
        f.write(f"**Source:** `{src}`\n\n")
        f.write(f"**Stats:** {n_files} files, {size_bytes/1e6:.1f} MB\n\n")
        f.write("**Common columns in each `cap_*.npz`:**\n")
        f.write("- `rxw` (complex64) — raw RX samples (one capture window)\n")
        f.write("- `meta_json` (str) — per-capture metadata: cfo_hz, pilot_rx, snr_post, etc.\n")
        f.write("- (OFDM step10 only) `Yeq_data` (complex64, shape `(num_syms, 48)`) — equalised data subcarrier symbols ready for LDPC post-decode\n\n")
        f.write("**Filename suffix convention:** `_ok` = packet decoded, `_fail`/`_bg` = decode failed / pure background.\n\n")
        f.write("**Reproducibility:** see `../README.md` (top-level dataset card) and the paper, [`globecom2026_otfs_doppler.pdf`](https://github.com/lkk688/RF-Stream-PHY/tree/main/OTFS/paper).\n")


def stage_tx_iq():
    dst = os.path.join(STAGE, "tx_iq"); os.makedirs(dst, exist_ok=True)
    copied = []
    for src in TX_IQ_FILES:
        if not os.path.exists(src):
            print(f"  [SKIP] missing: {src}")
            continue
        shutil.copy2(src, dst)
        copied.append(os.path.basename(src))
    with open(os.path.join(dst, "README.md"), "w") as f:
        f.write("# tx_iq — Transmit IQ waveforms\n\n")
        f.write("Pre-built TX waveforms used in all experiments. Format: raw complex64 little-endian, no header.\n\n")
        f.write("Files:\n")
        for n in copied:
            f.write(f"- `{n}`\n")
        f.write("\nThe `_fdN.iq` variants are the same waveform with the 2-tap synthetic Doppler injection at "
                "`fd = N Hz` applied via `OTFS/inject_doppler_to_iq.py`. See paper Sec. IV.\n\n")
        f.write("**Replay:** `python3 OTFS/rf_stream_tx_playback.py --iq_file <name>.iq --fc 2.42e9 --fs 3e6 --tx_gain 0 --cyclic`\n")
    return copied


def write_top_readme(manifest: list, tx_iq_files: list):
    total_bytes = sum(r.get("size_bytes", 0) for r in manifest)
    today = datetime.now().strftime("%Y-%m-%d")
    md = []
    md.append("---")
    md.append("license: cc-by-4.0")
    md.append("task_categories:")
    md.append("  - audio-classification")
    md.append("language: [en]")
    md.append("size_categories: [1G<n<10G]")
    md.append("tags:")
    md.append("  - rf")
    md.append("  - sdr")
    md.append("  - otfs")
    md.append("  - ofdm")
    md.append("  - ldpc")
    md.append("  - plutosdr")
    md.append("  - doppler")
    md.append("  - mobility")
    md.append("  - isac")
    md.append("---\n")
    md.append("# RF-Stream-PHY OTFS — Raw IQ and Decode Captures\n")
    md.append("Companion dataset for the paper:\n")
    md.append("> *Hardware-Validated OTFS+LDPC on PlutoSDR: Synthetic Doppler Injection and Monostatic-Loopback Mobility Sensing*, "
              "K. Liu, submitted IEEE Globecom 2026.\n")
    md.append("Code: https://github.com/lkk688/RF-Stream-PHY (subproject `OTFS/`).\n")
    md.append(f"Total: **{len(manifest)} experiment runs**, **{total_bytes/1e9:.2f} GB**, captured {today}.\n")
    md.append("## Splits\n")
    md.append("| Split | Description | Key result | # files | Size (MB) |")
    md.append("|---|---|---:|---:|---:|")
    for r in manifest:
        if r.get("status") != "ok": continue
        md.append(f"| `{r['split']}/` | {r['description']} | {r['key_result']} | {r['n_files']} | {r['size_bytes']/1e6:.1f} |")
    md.append("")
    md.append("Plus `tx_iq/` — pre-built TX waveforms (cyclic playback inputs).\n")

    md.append("## File format\n")
    md.append("Each `cap_*.npz` contains:\n")
    md.append("- `rxw` (complex64) — raw RX samples (one capture window, length depends on RX script)\n")
    md.append("- `meta_json` (str) — per-capture metadata as JSON: cfo_hz, pilot_rx, snr_post, stf_idx, etc.\n")
    md.append("- *(OFDM step10 captures only)* `Yeq_data` (complex64, shape `(num_syms, 48)`) — equalised data subcarrier symbols ready for LDPC post-decode without re-running sync/equalisation.\n")
    md.append("- *(OFDM step10 captures only)* `H` (complex64) — LTF channel estimate\n")
    md.append("\nFilename suffixes: `_ok` (packet decoded successfully), `_bg` (background-noise window, no signal), `_fail` (signal present, decode failed). Each run dir also has a `run_summary.json` and a `rx_config.json` (where applicable).\n")

    md.append("## Reproducing the paper\n")
    md.append("Re-decode the OFDM closed-grid 0 dB run through the LDPC post-decoder:\n")
    md.append("```bash")
    md.append("python3 OTFS/ofdm_ldpc_postdecode.py \\")
    md.append("    --rx_npz_glob 'closed_grid_0db_ofdm/cap_*.npz' \\")
    md.append("    --ref_seed 42 --ref_len 4096 --chunk_bytes 64 \\")
    md.append("    --ldpc_k 256 --ldpc_n 512 --num_ofdm_syms 16 --payload_bytes 64")
    md.append("# Expected: 161 / 489 attempts = 32.9% pkt rate")
    md.append("```\n")

    md.append("## License\n")
    md.append("CC-BY-4.0. Please cite the paper if you use this dataset.\n")
    md.append("```bibtex")
    md.append("@inproceedings{liu2026otfs,")
    md.append("  title  = {Hardware-Validated {OTFS}+{LDPC} on {PlutoSDR}:")
    md.append("            Synthetic Doppler Injection and Monostatic-Loopback Mobility Sensing},")
    md.append("  author = {Liu, Kaikai},")
    md.append("  booktitle = {Proc. IEEE Globecom},")
    md.append("  year   = {2026}")
    md.append("}")
    md.append("```")
    with open(os.path.join(STAGE, "DATASET.md"), "w") as f:
        f.write("\n".join(md))


def main():
    os.makedirs(STAGE, exist_ok=True)
    manifest = []
    print(f"Staging into: {STAGE}\n")
    for src, dst, desc, kr in RUNS:
        print(f"  staging {dst:40s} ...")
        manifest.append(copy_run(src, "", dst, desc, kr))
    for src, dst, desc, kr in OTA_DOPPLER_RUNS:
        print(f"  staging {dst:40s} ...")
        manifest.append(copy_run(src, "", dst, desc, kr))
    for src, dst, desc, kr in MONO_RUNS:
        print(f"  staging {dst:40s} ...")
        manifest.append(copy_run(src, "", dst, desc, kr))
    print(f"  staging tx_iq                                   ...")
    tx_files = stage_tx_iq()

    write_top_readme(manifest, tx_files)

    # Manifest as machine-readable JSON
    with open(os.path.join(STAGE, "MANIFEST.json"), "w") as f:
        json.dump({"staged_at": datetime.now().isoformat(),
                    "splits": manifest,
                    "tx_iq_files": tx_files}, f, indent=2)

    total_bytes = sum(r.get("size_bytes", 0) for r in manifest)
    total_files = sum(r.get("n_files", 0) for r in manifest)
    print(f"\n=== STAGED {len(manifest)} runs, {total_files} files, {total_bytes/1e9:.2f} GB ===")
    print(f"Top-level dataset card: {STAGE}/DATASET.md")
    print(f"Manifest:               {STAGE}/MANIFEST.json")


if __name__ == "__main__":
    main()
