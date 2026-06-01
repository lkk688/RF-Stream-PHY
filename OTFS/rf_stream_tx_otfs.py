#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rf_stream_tx_otfs.py — OTFS TX, Phase-1 hardware bring-up.

Mirrors `rf_stream_tx_step6phy.py` exactly in:
  • Pluto streaming model (non-cyclic, ring buffer)
  • Threaded producer / TX worker
  • IQ-swap compensation (sdr.tx(np.conj(buf) * 4096))
  • Per-packet tx_truth.jsonl with same field names + extras for OTFS
  • Packet header (MAGIC || seq || total || len || payload || CRC32)

Replaces:
  • OFDM data symbols → OTFS DD-grid payload (embedded pilot, Raviteja 2019)
  • Modulation lookup is shared with step6 — bit-for-bit compatible.

The same TX preamble (STF + LTF) is sent so the existing AutoDataPHY
synchroniser / CFO estimator works unchanged.  Only the data segment is OTFS.

Example
-------
    python3 rf_stream_tx_otfs.py \
      --uri ip:192.168.3.2 --fc 2.4e9 --fs 3e6 --tx_gain 0 \
      --modulation qpsk --ref_seed 42 --ref_len 256 \
      --otfs_M 64 --otfs_N 8 --otfs_l_max 6 --otfs_k_max 1
"""

import argparse
import json
import os
import queue
import threading
import time
import zlib
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional

import numpy as np

from otfs.otfs_core import OTFSConfig, MOD_BPS
from otfs.otfs_frame import (
    bytes_to_otfs_samples,
    create_schmidl_cox_stf,
    create_ltf,
    build_packet_bytes,
    expected_frame_length,
)


# =============================================================================
# Helpers
# =============================================================================
def fit_to_fixed_len(sig: np.ndarray, fixed_len: int) -> np.ndarray:
    if len(sig) == fixed_len:
        return sig
    if len(sig) > fixed_len:
        return sig[:fixed_len].copy()
    out = np.zeros(fixed_len, dtype=np.complex64)
    out[:len(sig)] = sig
    return out


# =============================================================================
# Config
# =============================================================================
@dataclass
class TxConfig:
    uri: str
    fc: float
    fs: float
    tx_gain: float
    tx_bw: float
    fixed_len: int
    cyclic: bool
    repeat: int
    stf_repeats: int
    ltf_symbols: int
    modulation: str
    bps: int
    tone_duration_ms: float
    tone_freq_hz: float
    gap_short: int
    gap_long: int
    tx_scale: float
    idle_amp: float
    send_interval_s: float
    beacon_period_s: float
    out_root: str
    run_id: str
    # OTFS-specific
    otfs_M: int
    otfs_N: int
    otfs_l_max: int
    otfs_k_max: int
    otfs_pilot_doppler: int
    otfs_pilot_boost_db: float


def make_idle_buffer(cfg: TxConfig) -> np.ndarray:
    if cfg.idle_amp <= 0:
        return np.zeros(cfg.fixed_len, dtype=np.complex64)
    rng = np.random.default_rng(0)
    n = (rng.standard_normal(cfg.fixed_len) + 1j * rng.standard_normal(cfg.fixed_len)).astype(np.complex64)
    return (n / (np.max(np.abs(n)) + 1e-9) * cfg.idle_amp).astype(np.complex64)


# =============================================================================
# TX worker (non-cyclic streaming, IQ-swap compensation — identical to step6)
# =============================================================================
def tx_worker(stop_ev: threading.Event, q: "queue.Queue[np.ndarray]", cfg: TxConfig):
    import adi
    sdr = adi.Pluto(uri=cfg.uri)

    # ── DDS-state auto-reset ──────────────────────────────────────────────
    # Phase-1 hardware lesson: if anything (a CW test, an earlier script, a
    # crashed prior run) left the Pluto's DDS-direct path enabled, the host
    # DMA path is silently attenuated by ~30 dB.  Zeroing all DDS scales and
    # pushing a tiny zero-buffer through DMA reliably puts the channel back
    # under software control regardless of prior state.
    try:
        sdr.dds_scales = (0.0,) * 8
    except Exception:
        pass

    sdr.sample_rate          = int(cfg.fs)
    sdr.tx_lo                = int(cfg.fc)
    sdr.tx_rf_bandwidth      = int(cfg.tx_bw)
    sdr.tx_hardwaregain_chan0 = float(cfg.tx_gain)
    sdr.tx_enabled_channels  = [0]
    try:
        sdr.tx_destroy_buffer()
    except Exception:
        pass
    sdr.tx_cyclic_buffer = False
    try:
        sdr.tx(np.zeros(1024, dtype=np.complex64))    # arm the DMA
        sdr.tx_destroy_buffer()
    except Exception:
        pass
    sdr.tx_cyclic_buffer = bool(cfg.cyclic)

    idle = fit_to_fixed_len(make_idle_buffer(cfg), cfg.fixed_len).astype(np.complex64)
    last_sig: Optional[np.ndarray] = None
    last_beacon_t = 0.0

    print(f"[TX] worker started.  fixed_len={cfg.fixed_len}  modulation={cfg.modulation}  "
          f"tx_gain={cfg.tx_gain} dB  cyclic={cfg.cyclic}")
    print(f"[TX] OTFS grid M={cfg.otfs_M} N={cfg.otfs_N} l_max={cfg.otfs_l_max} k_max={cfg.otfs_k_max}")

    # In cyclic mode we push the FIRST buffer we receive, then sit and let
    # Pluto loop it forever.  This gives ~100 % on-air duty cycle, which is
    # what we want for cable BER characterisation.
    if cfg.cyclic:
        try:
            while not stop_ev.is_set():
                try:
                    buf = q.get(timeout=0.5)
                    break
                except queue.Empty:
                    continue
            buf = buf.astype(np.complex64)
            # Auto-size in cyclic mode: trust the producer's buffer length so
            # multi-payload cyclic mode (where the producer concatenates K
            # frames into one big buffer) works without --fixed_len juggling.
            actual_len = int(buf.size)
            tx_data = (np.conj(buf) * 4096.0).astype(np.complex64)
            sdr.tx(tx_data)
            print(f"[TX] cyclic buffer armed  ({actual_len} samples = "
                  f"{actual_len/cfg.fs*1e3:.2f} ms loop).  Pluto streaming until SIGINT.")
            while not stop_ev.is_set():
                time.sleep(0.1)
        finally:
            try:
                sdr.tx_destroy_buffer()
            except Exception:
                pass
            print("[TX] cyclic worker stopped.")
        return

    try:
        while not stop_ev.is_set():
            now = time.time()
            buf = None
            try:
                buf = q.get_nowait()
            except queue.Empty:
                pass

            if buf is not None:
                buf = fit_to_fixed_len(buf, cfg.fixed_len).astype(np.complex64)
                last_sig = buf.copy()
                last_beacon_t = now
            elif cfg.beacon_period_s > 0 and last_sig is not None and (now - last_beacon_t) >= cfg.beacon_period_s:
                buf = last_sig
                last_beacon_t = now
            else:
                buf = idle

            tx_data = (np.conj(buf) * 4096.0).astype(np.complex64)
            sdr.tx(tx_data)

            if cfg.send_interval_s > 0:
                time.sleep(cfg.send_interval_s)
    finally:
        try:
            sdr.tx_destroy_buffer()
        except Exception:
            pass
        print("[TX] worker stopped.")


# =============================================================================
# Producer: packet mode
# =============================================================================
def producer_packets(
    stop_ev: threading.Event,
    q: "queue.Queue[np.ndarray]",
    cfg: TxConfig,
    infile: str,
    payload_str: str,
    payload_len: int,
    chunk_bytes: int,
    ref_seed: int = 0,
    ref_len: int = 0,
    ldpc=None,
):
    stf = create_schmidl_cox_stf(cfg.stf_repeats)
    ltf = create_ltf(cfg.ltf_symbols)
    otfs_cfg = OTFSConfig(
        M=cfg.otfs_M, N=cfg.otfs_N,
        l_max=cfg.otfs_l_max, k_max=cfg.otfs_k_max,
        pilot_doppler=cfg.otfs_pilot_doppler,
        pilot_boost_db=cfg.otfs_pilot_boost_db,
    )

    if ref_len > 0:
        rng = np.random.default_rng(ref_seed)
        data = rng.integers(0, 256, size=ref_len, dtype=np.uint8).tobytes()
        print(f"[TX] BER-mode reference payload: {len(data)} bytes  seed={ref_seed}")
    elif infile:
        with open(infile, "rb") as fh:
            data = fh.read()
    elif payload_str:
        data = payload_str.encode("utf-8")
    else:
        rng = np.random.default_rng(54321)
        data = rng.integers(0, 256, size=payload_len, dtype=np.uint8).tobytes()

    chunks = [data[i: i + chunk_bytes] for i in range(0, len(data), chunk_bytes)] if chunk_bytes > 0 else [data]
    total = len(chunks)

    out_dir = os.path.join(cfg.out_root, cfg.run_id)
    truth_path = os.path.join(out_dir, "tx_truth.jsonl")

    # ── Build the per-chunk frame signal once (reused below) ──────────────────
    def _build_one_frame(seq: int, ch: bytes):
        fb = build_packet_bytes(seq, total, ch)
        sig, info = bytes_to_otfs_samples(
            fb, cfg.repeat, stf, ltf, otfs_cfg, cfg.modulation,
            cfg.fs, cfg.tone_duration_ms, cfg.tone_freq_hz,
            cfg.gap_short, cfg.gap_long, cfg.tx_scale, ldpc=ldpc,
        )
        return fb, sig, info

    # ── Cyclic multi-payload mode: build all K frames once, concat, queue
    #    a single big buffer, then idle.  Worker pushes it cyclically.  This
    #    is the path used for dataset collection: every cycle of the on-air
    #    waveform contains K *different* payloads so the receiver sees a
    #    varied training distribution instead of memorising one packet.
    if cfg.cyclic and total > 1:
        # Single-frame length determines the slot size.  Pad each slot to the
        # largest single frame so absolute offset of frame k is k * slot_len.
        max_len = 0
        prebuilt = []
        for seq, ch in enumerate(chunks):
            fb, sig, info = _build_one_frame(seq, ch)
            prebuilt.append((seq, ch, fb, sig, info))
            max_len = max(max_len, sig.size)

        slot_len = ((max_len + 63) // 64) * 64        # round up to 64 samples
        big = np.zeros(slot_len * total, dtype=np.complex64)
        with open(truth_path, "a") as truth_fh:
            for seq, ch, fb, sig, info in prebuilt:
                offset = seq * slot_len
                big[offset : offset + sig.size] = sig
                truth = {
                    "seq":           seq,
                    "total":         total,
                    "payload_len":   len(ch),
                    "payload_crc32": zlib.crc32(ch) & 0xFFFFFFFF,
                    "frame_crc32":   info.frame_crc32,
                    "modulation":    cfg.modulation,
                    "tx_gain_db":    cfg.tx_gain,
                    "repeat":        cfg.repeat,
                    "stf_repeats":   cfg.stf_repeats,
                    "ltf_symbols":   cfg.ltf_symbols,
                    "frame_bits_len": int(len(fb) * 8),
                    "waveform":      "OTFS",
                    "tx_mode":       "cyclic_multi_payload",
                    "slot_len_samples": int(slot_len),
                    "cycle_len_samples": int(slot_len * total),
                    "otfs": {
                        "M":              cfg.otfs_M,
                        "N":              cfg.otfs_N,
                        "l_max":          cfg.otfs_l_max,
                        "k_max":          cfg.otfs_k_max,
                        "pilot_delay":    0,
                        "pilot_doppler":  cfg.otfs_pilot_doppler,
                        "pilot_boost_db": cfg.otfs_pilot_boost_db,
                        "pilot_amp":      info.pilot_amp,
                        "num_blocks":     info.num_blocks,
                        "data_cells_per_block": info.data_cells_per_block,
                        "bits_per_block": info.bits_per_block,
                        "bits_total":     info.bits_total,
                        "bits_padded":    info.bits_padded,
                    },
                    "t_send": time.time(),
                }
                truth_fh.write(json.dumps(truth) + "\n")
            truth_fh.flush()

        print(f"[TX] multi-payload cyclic mode: {total} distinct frames, "
              f"slot_len={slot_len} samples, cycle_len={slot_len*total} samples "
              f"({slot_len*total/cfg.fs*1e3:.2f} ms / cycle).")
        try:
            q.put(big, timeout=1.0)
        except queue.Full:
            pass
        print("[TX] multi-payload buffer queued — producer done.")
        while not stop_ev.is_set():
            time.sleep(0.5)
        return

    with open(truth_path, "a") as truth_fh:
        while not stop_ev.is_set():
            for seq, ch in enumerate(chunks):
                if stop_ev.is_set():
                    break
                fb = build_packet_bytes(seq, total, ch)
                sig, info = bytes_to_otfs_samples(
                    fb, cfg.repeat, stf, ltf, otfs_cfg, cfg.modulation,
                    cfg.fs, cfg.tone_duration_ms, cfg.tone_freq_hz,
                    cfg.gap_short, cfg.gap_long, cfg.tx_scale, ldpc=ldpc,
                )

                truth = {
                    "seq":           seq,
                    "total":         total,
                    "payload_len":   len(ch),
                    "payload_crc32": zlib.crc32(ch) & 0xFFFFFFFF,
                    "frame_crc32":   info.frame_crc32,
                    "modulation":    cfg.modulation,
                    "tx_gain_db":    cfg.tx_gain,
                    "repeat":        cfg.repeat,
                    "stf_repeats":   cfg.stf_repeats,
                    "ltf_symbols":   cfg.ltf_symbols,
                    "frame_bits_len": int(len(fb) * 8),
                    "waveform":      "OTFS",
                    "otfs": {
                        "M":              cfg.otfs_M,
                        "N":              cfg.otfs_N,
                        "l_max":          cfg.otfs_l_max,
                        "k_max":          cfg.otfs_k_max,
                        "pilot_delay":    0,
                        "pilot_doppler":  cfg.otfs_pilot_doppler,
                        "pilot_boost_db": cfg.otfs_pilot_boost_db,
                        "pilot_amp":      info.pilot_amp,
                        "num_blocks":     info.num_blocks,
                        "data_cells_per_block": info.data_cells_per_block,
                        "bits_per_block": info.bits_per_block,
                        "bits_total":     info.bits_total,
                        "bits_padded":    info.bits_padded,
                    },
                    "t_send": time.time(),
                }
                truth_fh.write(json.dumps(truth) + "\n")
                truth_fh.flush()

                try:
                    q.put(sig, timeout=0.1)
                except queue.Full:
                    pass

                if seq == 0 or seq == total - 1 or (seq % 20 == 0):
                    print(f"[TX] enq seq={seq}/{total-1}  payload={len(ch)}B  "
                          f"OTFS blocks={info.num_blocks}  pilot_amp={info.pilot_amp:.3f}")

    print("[TX] producer done – idling.")
    while not stop_ev.is_set():
        time.sleep(0.2)


# =============================================================================
# Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser("OTFS PHY TX — Phase 1 hardware bring-up")
    ap.add_argument("--uri",     required=True)
    ap.add_argument("--fc",      type=float, required=True)
    ap.add_argument("--fs",      type=float, required=True)
    ap.add_argument("--tx_gain", type=float, default=-20.0)
    ap.add_argument("--tx_bw",   type=float, default=0.0)

    ap.add_argument("--modulation", default="qpsk",
                    choices=list(MOD_BPS.keys()),
                    help="Modulation scheme (shared with step6 OFDM)")

    ap.add_argument("--fixed_len",     type=int,   default=65536)
    ap.add_argument("--send_interval", type=float, default=0.0)
    ap.add_argument("--queue_depth",   type=int,   default=8)

    ap.add_argument("--repeat",      type=int, default=1, choices=[1, 2, 4])
    ap.add_argument("--stf_repeats", type=int, default=6)
    ap.add_argument("--ltf_symbols", type=int, default=4)

    ap.add_argument("--tone_duration_ms", type=float, default=0.0)
    ap.add_argument("--tone_freq_hz",     type=float, default=100e3)
    ap.add_argument("--gap_short",  type=int, default=1000)
    ap.add_argument("--gap_long",   type=int, default=3000)
    ap.add_argument("--tx_scale",   type=float, default=0.8)
    ap.add_argument("--idle_amp",   type=float, default=0.0)

    ap.add_argument("--payload",     type=str, default="")
    ap.add_argument("--payload_len", type=int, default=64)
    ap.add_argument("--infile",      type=str, default="")
    ap.add_argument("--chunk_bytes", type=int, default=256,
                    help="Bytes per OTFS frame.  Sized to fit ~3 DD blocks at "
                         "M=64,N=8,QPSK with l_max=6,k_max=1 (≈ 119 B/block).")
    ap.add_argument("--ref_seed",    type=int, default=0)
    ap.add_argument("--ref_len",     type=int, default=0,
                    help="BER mode: transmit known PRNG payload of this length in bytes")
    ap.add_argument("--beacon_period", type=float, default=0.2)

    # OTFS-specific
    ap.add_argument("--otfs_M",     type=int, default=64,
                    help="Delay bins (= N_FFT for fair OFDM-vs-OTFS compare)")
    ap.add_argument("--otfs_N",     type=int, default=8,
                    help="Doppler bins per OTFS block")
    ap.add_argument("--otfs_l_max", type=int, default=6,
                    help="Max channel delay spread, in delay cells")
    ap.add_argument("--otfs_k_max", type=int, default=1,
                    help="Max channel Doppler spread, in Doppler cells")
    ap.add_argument("--otfs_pilot_doppler", type=int, default=4,
                    help="Pilot Doppler index (default N/2)")
    ap.add_argument("--otfs_pilot_boost_db", type=float, default=12.0,
                    help="Pilot amplitude over data-RMS, in dB")

    ap.add_argument("--out_root", default="rf_stream_tx_runs")
    ap.add_argument("--cyclic", action="store_true",
                    help="Use Pluto cyclic-buffer mode (loops the first frame forever, "
                         "~100%% on-air duty cycle — good for cable BER characterisation)")
    ap.add_argument("--fec", choices=["none", "ldpc"], default="none",
                    help="Outer FEC.  ldpc uses 5G NR rate-1/2 BG2.")
    ap.add_argument("--ldpc_k", type=int, default=256,
                    help="LDPC info bits per block (k=256, n=512 → rate 1/2)")
    ap.add_argument("--ldpc_n", type=int, default=512,
                    help="LDPC coded bits per block")
    ap.add_argument("--build_only", type=str, default="",
                    help="If set, build the multi-payload cyclic buffer offline (no Pluto), "
                         "save complex64 IQ to this file, write tx_truth.jsonl + tx_meta.json "
                         "alongside.  Use rf_stream_tx_playback.py on the target to play it.")
    args = ap.parse_args()

    bps    = MOD_BPS[args.modulation]
    run_id = datetime.now().strftime("run_otfs_%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out_root, run_id)
    os.makedirs(out_dir, exist_ok=True)

    cfg = TxConfig(
        uri=args.uri, fc=args.fc, fs=args.fs,
        tx_gain=args.tx_gain,
        tx_bw=(args.tx_bw if args.tx_bw > 0 else args.fs * 1.2),
        fixed_len=args.fixed_len,
        cyclic=bool(args.cyclic),
        repeat=args.repeat,
        stf_repeats=args.stf_repeats,
        ltf_symbols=args.ltf_symbols,
        modulation=args.modulation,
        bps=bps,
        tone_duration_ms=args.tone_duration_ms,
        tone_freq_hz=args.tone_freq_hz,
        gap_short=args.gap_short,
        gap_long=args.gap_long,
        tx_scale=args.tx_scale,
        idle_amp=args.idle_amp,
        send_interval_s=args.send_interval,
        beacon_period_s=args.beacon_period,
        out_root=args.out_root,
        run_id=run_id,
        otfs_M=args.otfs_M,
        otfs_N=args.otfs_N,
        otfs_l_max=args.otfs_l_max,
        otfs_k_max=args.otfs_k_max,
        otfs_pilot_doppler=args.otfs_pilot_doppler,
        otfs_pilot_boost_db=args.otfs_pilot_boost_db,
    )

    ldpc = None
    if args.fec == "ldpc":
        from otfs.otfs_fec import OTFSLDPC, FECConfig
        ldpc = OTFSLDPC(FECConfig(enabled=True, k=args.ldpc_k, n=args.ldpc_n))
        print(f"[TX] FEC=LDPC  k={args.ldpc_k} n={args.ldpc_n} rate={args.ldpc_k/args.ldpc_n:.2f}")

    # ── Build-only mode: assemble the multi-payload IQ buffer and dump it
    #    (used when the target machine cannot run LDPC encode locally,
    #    e.g. Jetson without PyTorch — we build on x86, scp the IQ file,
    #    then use rf_stream_tx_playback.py on Jetson).
    if args.build_only:
        otfs_cfg = OTFSConfig(M=args.otfs_M, N=args.otfs_N,
                               l_max=args.otfs_l_max, k_max=args.otfs_k_max,
                               pilot_doppler=args.otfs_pilot_doppler,
                               pilot_boost_db=args.otfs_pilot_boost_db)
        stf = create_schmidl_cox_stf(args.stf_repeats)
        ltf = create_ltf(args.ltf_symbols)
        if args.ref_len > 0:
            rng  = np.random.default_rng(args.ref_seed)
            data = rng.integers(0, 256, size=args.ref_len, dtype=np.uint8).tobytes()
        else:
            data = b"build_only_default_payload"
        chunks = [data[i: i + args.chunk_bytes]
                   for i in range(0, len(data), args.chunk_bytes)]
        total  = len(chunks)
        print(f"[build_only] {total} chunks of up to {args.chunk_bytes} bytes")

        prebuilt = []
        max_len = 0
        for seq, ch in enumerate(chunks):
            fb = build_packet_bytes(seq, total, ch)
            sig, info = bytes_to_otfs_samples(
                fb, args.repeat, stf, ltf, otfs_cfg, args.modulation,
                args.fs, args.tone_duration_ms, args.tone_freq_hz,
                args.gap_short, args.gap_long, args.tx_scale, ldpc=ldpc,
            )
            prebuilt.append((seq, ch, fb, sig, info))
            max_len = max(max_len, sig.size)

        slot_len = ((max_len + 63) // 64) * 64
        big = np.zeros(slot_len * total, dtype=np.complex64)
        truth_path = args.build_only + ".tx_truth.jsonl"
        meta_path  = args.build_only + ".tx_meta.json"
        with open(truth_path, "w") as truth_fh:
            for seq, ch, fb, sig, info in prebuilt:
                offset = seq * slot_len
                big[offset : offset + sig.size] = sig
                truth_fh.write(json.dumps({
                    "seq": seq, "total": total, "payload_len": len(ch),
                    "payload_crc32": zlib.crc32(ch) & 0xFFFFFFFF,
                    "frame_crc32":   info.frame_crc32,
                    "modulation":    args.modulation, "tx_gain_db": args.tx_gain,
                    "repeat":        args.repeat, "stf_repeats": args.stf_repeats,
                    "ltf_symbols":   args.ltf_symbols, "frame_bits_len": int(len(fb)*8),
                    "waveform": "OTFS", "tx_mode": "build_only_multi_payload",
                    "slot_len_samples": int(slot_len),
                    "cycle_len_samples": int(slot_len * total),
                    "fec": args.fec, "ldpc_k": args.ldpc_k, "ldpc_n": args.ldpc_n,
                    "otfs": {"M": args.otfs_M, "N": args.otfs_N,
                              "l_max": args.otfs_l_max, "k_max": args.otfs_k_max,
                              "pilot_delay": 0, "pilot_doppler": args.otfs_pilot_doppler,
                              "pilot_boost_db": args.otfs_pilot_boost_db,
                              "pilot_amp": info.pilot_amp,
                              "num_blocks": info.num_blocks,
                              "data_cells_per_block": info.data_cells_per_block,
                              "bits_per_block": info.bits_per_block,
                              "bits_total": info.bits_total,
                              "bits_padded": info.bits_padded},
                }) + "\n")
        big.tofile(args.build_only)
        with open(meta_path, "w") as f:
            json.dump({
                "iq_file": args.build_only, "samples": int(big.size),
                "fs": args.fs, "fc": args.fc, "modulation": args.modulation,
                "n_chunks": total, "slot_len_samples": int(slot_len),
                "fec": args.fec, "ldpc_k": args.ldpc_k, "ldpc_n": args.ldpc_n,
                "otfs": {"M": args.otfs_M, "N": args.otfs_N,
                          "l_max": args.otfs_l_max, "k_max": args.otfs_k_max,
                          "pilot_doppler": args.otfs_pilot_doppler,
                          "pilot_boost_db": args.otfs_pilot_boost_db,
                          "num_blocks_per_frame": prebuilt[0][4].num_blocks},
                "ref_seed": args.ref_seed, "ref_len": args.ref_len,
                "chunk_bytes": args.chunk_bytes,
            }, f, indent=2)
        print(f"[build_only] wrote {big.size} samples → {args.build_only}")
        print(f"[build_only]   tx_truth → {truth_path}")
        print(f"[build_only]   tx_meta  → {meta_path}")
        print(f"[build_only]   per-frame: num_blocks={prebuilt[0][4].num_blocks}")
        return

    expected_len = expected_frame_length(
        args.chunk_bytes + 14,   # MAGIC+seq+total+len+CRC overhead
        args.repeat,
        OTFSConfig(M=cfg.otfs_M, N=cfg.otfs_N,
                   l_max=cfg.otfs_l_max, k_max=cfg.otfs_k_max),
        cfg.modulation,
        cfg.stf_repeats, cfg.ltf_symbols, cfg.gap_short, cfg.gap_long,
        tone_samples=int(cfg.tone_duration_ms * cfg.fs / 1000.0),
        ldpc=ldpc,
    )
    if expected_len > cfg.fixed_len:
        print(f"[TX] WARNING: expected frame length {expected_len} > fixed_len {cfg.fixed_len}; "
              f"consider --fixed_len {expected_len}")

    meta = {
        **asdict(cfg),
        "ref_seed":     args.ref_seed,
        "ref_len":      args.ref_len,
        "chunk_bytes":  args.chunk_bytes,
        "expected_frame_len": expected_len,
        "start_time":   datetime.now().isoformat(),
        "waveform":     "OTFS",
    }
    with open(os.path.join(out_dir, "tx_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    print("\n" + "=" * 70)
    print("Streaming TX  (OTFS — Phase 1)")
    print("=" * 70)
    print(f"  URI={cfg.uri}  fc={cfg.fc/1e6:.3f}MHz  fs={cfg.fs/1e6:.1f}Msps")
    print(f"  tx_gain={cfg.tx_gain}dB  modulation={cfg.modulation}  bps={bps}")
    print(f"  OTFS M={cfg.otfs_M} N={cfg.otfs_N} l_max={cfg.otfs_l_max} k_max={cfg.otfs_k_max}")
    print(f"  pilot_doppler={cfg.otfs_pilot_doppler}  pilot_boost={cfg.otfs_pilot_boost_db}dB")
    print(f"  expected_frame_len ≈ {expected_len} samples (fixed_len={cfg.fixed_len})")
    print(f"  run_id={run_id}  out_dir={out_dir}")
    print("=" * 70)

    q       = queue.Queue(maxsize=int(args.queue_depth))
    stop_ev = threading.Event()

    t_tx   = threading.Thread(target=tx_worker, args=(stop_ev, q, cfg), daemon=True)
    t_prod = threading.Thread(
        target=producer_packets,
        args=(stop_ev, q, cfg, args.infile, args.payload,
              args.payload_len, args.chunk_bytes, args.ref_seed, args.ref_len,
              ldpc),
        daemon=True,
    )

    t_tx.start()
    t_prod.start()

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        stop_ev.set()
        time.sleep(0.5)
        print("[TX] exit.")


if __name__ == "__main__":
    main()
