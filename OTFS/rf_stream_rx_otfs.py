#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rf_stream_rx_otfs.py — OTFS RX, Phase-1 hardware bring-up.

Two modes:

  --mode live     Capture from PlutoSDR (mirrors step10 RX worker).
                  Saves NPZs in the same format as AutoDataPHY so the
                  existing analysis / HF-upload tooling works unchanged.

  --mode offline  Read existing rxw NPZs (e.g. produced by step10 live
                  capture or by `test_otfs_loopback.py`) and decode them.
                  This is the path we use *before* the Pluto is plugged in.

Pipeline (offline or live, per captured window):
  1.  STF correlation + Schmidl-Cox CFO  → coarse sync
  2.  LTF channel estimate H_ltf          → (used as TF reference)
  3.  Try both IQ conjugations (Pluto IQ-swap robustness)
  4.  Extract M*N samples of OTFS payload per block
  5.  SFFT → DD grid
  6.  Estimate DD channel taps from embedded pilot
  7.  Build effective DD channel matrix, run LMMSE detection
  8.  Demap data cells, descramble, parse packet, CRC check
  9.  Compute BER against tx_truth.jsonl when available

Output per decoded window: one NPZ + per-line JSONL log entry.
"""

import argparse
import glob
import json
import os
import sys
import time
import zlib
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from otfs.otfs_core import (
    OTFSConfig, MOD_BPS,
    sfft_time_to_dd,
    extract_data_from_dd,
    estimate_dd_channel_taps,
    build_effective_dd_matrix,
    dd_lmmse_detect,
    find_pilot_offset,
    make_constellation,
    hard_demap,
    symbols_to_bits,
    bits_to_bytes,
    scramble_bits,
    num_data_cells,
)
from otfs.otfs_sync import sync_preamble, SYMBOL_LEN
from otfs.otfs_frame import N_FFT, N_CP, MAGIC


# =============================================================================
# Packet parser  (must match step6 framing)
# =============================================================================
def parse_packet_bytes(bb: bytes) -> Tuple[bool, str, int, bytes]:
    """Returns (ok, reason, seq, payload)."""
    if len(bb) < 14:
        return False, "too_short", -1, b""
    if bb[:4] != MAGIC:
        return False, "bad_magic", -1, b""
    seq         = int.from_bytes(bb[4:6],  "little")
    total       = int.from_bytes(bb[6:8],  "little")
    payload_len = int.from_bytes(bb[8:10], "little")
    expected    = 10 + payload_len + 4
    if len(bb) < expected:
        return False, f"short_for_payload({payload_len})", seq, b""
    body = bb[:10 + payload_len]
    crc_field = int.from_bytes(bb[10 + payload_len: 14 + payload_len], "little")
    if (zlib.crc32(body) & 0xFFFFFFFF) != crc_field:
        return False, "crc_fail", seq, b""
    return True, "ok", seq, bb[10: 10 + payload_len]


# =============================================================================
# OTFS payload decode for a single CFO-corrected window
# =============================================================================
@dataclass
class DecodeResult:
    ok: bool
    reason: str
    seq: int
    payload: bytes
    n_bits: int
    n_errors: int
    pilot_amp_rx: float
    dd_channel_taps: List[Tuple[int, int, complex]]
    bps: int
    cfo_hz: float
    snr_pre_db: float
    snr_post_db: float


def decode_otfs_window(
    rx: np.ndarray,
    cfg: OTFSConfig,
    modulation: str,
    num_blocks: int,
    repeat: int,
    stf_repeats: int,
    ltf_symbols: int,
    fs: float,
    payload_bytes_expected: int,
    try_conjugate: bool = True,
    ldpc=None,                          # OTFSLDPC instance or None
) -> DecodeResult:
    """End-to-end decode of one capture window."""
    sync = sync_preamble(rx, stf_repeats, ltf_symbols, fs)
    if sync is None:
        return DecodeResult(False, "sync_fail", -1, b"", 0, 0,
                            0.0, [], MOD_BPS[modulation.lower()], 0.0, 0.0, 0.0)

    rx_corr      = sync["rx_corr"]
    payload_start = sync["payload_start"]
    cfo_hz        = sync["cfo_hz"]

    bps        = MOD_BPS[modulation.lower()]
    const_tbl  = make_constellation(modulation)
    MN         = cfg.M * cfg.N
    cp_len     = max(int(cfg.l_max), 64)   # matches TX (N_FFT-sized CP for timing tolerance)
    block_stride = cp_len + MN
    total_pl   = num_blocks * block_stride

    if payload_start + total_pl > rx_corr.size:
        return DecodeResult(False, "payload_truncated", -1, b"", 0, 0,
                            0.0, [], bps, cfo_hz, 0.0, 0.0)

    best: Optional[DecodeResult] = None
    best_pilot = 0.0
    offset_sweep = [0]
    for conj_comp in ([False, True] if try_conjugate else [False]):
      for off in offset_sweep:
        rx_proc = np.conj(rx_corr) if conj_comp else rx_corr

        # ── per-block DD detection ─────────────────────────────────────────
        bit_chunks:  List[np.ndarray] = []
        llr_chunks:  List[np.ndarray] = []          # soft LLRs for LDPC
        pilot_amp_sum = 0.0
        all_taps: List[Tuple[int, int, complex]] = []
        pre_pow = []
        post_pow = []

        bad_offset = False
        for b in range(num_blocks):
            # Each on-air block = [CP_l_max | MN samples].  Strip the CP.
            start = payload_start + off + b * block_stride + cp_len
            if start < 0 or start + MN > rx_proc.size:
                start = max(0, min(rx_proc.size - MN, start))

            # First SFFT — used purely to find the residual sample offset
            # from the embedded pilot's actual row in the DD grid.  Then
            # re-SFFT at the corrected position so all subsequent
            # processing sees the pilot at row 0 (as designed).
            rx_dd0      = sfft_time_to_dd(rx_proc[start: start + MN].astype(np.complex64),
                                          cfg.M, cfg.N)
            residual    = find_pilot_offset(rx_dd0, cfg)
            corr_start  = start + residual
            if corr_start < 0 or corr_start + MN > rx_proc.size:
                bad_offset = True
                break
            block_t = rx_proc[corr_start: corr_start + MN].astype(np.complex64)
            rx_dd   = sfft_time_to_dd(block_t, cfg.M, cfg.N)

            # Pilot amplitude — peak at the now-aligned pilot row
            pilot_region = rx_dd[: cfg.l_max + 1, cfg.pilot_doppler]
            pilot_amp_est = float(np.max(np.abs(pilot_region)))
            if pilot_amp_est < 1e-9:
                bad_offset = True
                break
            pilot_amp_sum += pilot_amp_est

            # Estimate DD channel taps from the embedded pilot
            taps = estimate_dd_channel_taps(rx_dd, cfg, pilot_amp_est)
            all_taps = taps   # keep the last block's taps for logging

            # Noise variance estimate: residual energy in cells that should be
            # zero (i.e., guard cells far from pilot).  We just use a rough
            # constant for now and refine after we see hardware data.
            n_data = num_data_cells(cfg)
            data_syms_raw = extract_data_from_dd(rx_dd, cfg)
            data_pow = float(np.mean(np.abs(data_syms_raw) ** 2) + 1e-12)
            noise_var = max(1e-6, data_pow * 0.05)

            # DD-LMMSE detect → constellation estimates → hard demap
            H_dd = build_effective_dd_matrix(taps, cfg.M, cfg.N)
            x_hat = dd_lmmse_detect(rx_dd, H_dd, noise_var, cfg)

            pre_pow.append(data_pow)
            post_pow.append(float(np.mean(np.abs(x_hat) ** 2)))

            idx = hard_demap(x_hat, const_tbl)
            bits = symbols_to_bits(idx, bps)
            bit_chunks.append(bits)
            # Soft LLRs from the LMMSE constellation estimate, used by LDPC.
            # Normalise x_hat to ~unit constellation magnitude (matches our
            # const table |s|=1/√2) and estimate residual noise from the
            # distance of each cell to its nearest constellation point.
            if ldpc is not None and modulation.lower() == "qpsk":
                from otfs.otfs_fec import qpsk_soft_demap
                # Robust per-block scaling: median of |x_hat| matches the
                # constellation magnitude (1/√2 ≈ 0.707) on a clean QPSK
                # signal.  Use median rather than RMS so outliers don't pull.
                scale = float(np.median(np.abs(x_hat)) / (1.0 / np.sqrt(2.0)) + 1e-12)
                x_hat_n = (x_hat / max(scale, 1e-9)).astype(np.complex64)
                # Estimate per-block noise variance from the nearest-constellation
                # residuals (EVM²).  Clamp to a sensible minimum so LDPC sees
                # bounded LLRs.
                diff = x_hat_n[:, None] - const_tbl[None, :]
                d2_min = np.min((diff.real ** 2 + diff.imag ** 2), axis=1)
                eff_sigma2 = float(max(0.02, np.mean(d2_min)))
                llr = qpsk_soft_demap(x_hat_n, noise_var=eff_sigma2,
                                       const_table=const_tbl.astype(np.complex64))
                # Optional clip to keep LDPC numerically stable
                llr = np.clip(llr, -20.0, 20.0)
                llr_chunks.append(llr)

        if bad_offset:
            continue
        bits_all = np.concatenate(bit_chunks).astype(np.uint8)

        # ── undo repeat (majority vote) ──────────────────────────────────
        if repeat > 1:
            L = (bits_all.size // repeat) * repeat
            v = bits_all[:L].reshape(-1, repeat)
            bits_all = (np.sum(v, axis=1) >= (repeat / 2.0)).astype(np.uint8)

        # ── LDPC decode (soft if available) → descramble → bytes ─────────
        if ldpc is not None:
            # Use the soft LLR path when available, fall back to hard.
            if llr_chunks:
                llr_all = np.concatenate(llr_chunks).astype(np.float32)
                if repeat > 1:
                    # Sum LLRs across repeats (majority vote in LLR domain)
                    L = (llr_all.size // repeat) * repeat
                    llr_all = llr_all[:L].reshape(-1, repeat).sum(axis=1)
                bits_dec = ldpc.decode_soft(llr_all)
            else:
                bits_dec = ldpc.decode_hard(bits_all)
            bits_unscr = scramble_bits(bits_dec)
        else:
            bits_unscr = scramble_bits(bits_all)
        rx_bytes = bits_to_bytes(bits_unscr[: (payload_bytes_expected + 14) * 8])

        ok, reason, seq, payload = parse_packet_bytes(rx_bytes)

        # n_bits / n_errors are computed *after* we cross-reference the
        # tx_truth file (see main loop) — leave 0/0 here.
        snr_pre = 10.0 * np.log10(max(np.mean(pre_pow), 1e-12))
        snr_post = 10.0 * np.log10(max(np.mean(post_pow), 1e-12))
        result = DecodeResult(
            ok=ok, reason=reason, seq=seq, payload=payload,
            n_bits=0, n_errors=0,
            pilot_amp_rx=pilot_amp_sum / max(num_blocks, 1),
            dd_channel_taps=all_taps,
            bps=bps, cfo_hz=cfo_hz,
            snr_pre_db=snr_pre, snr_post_db=snr_post,
        )
        # Best wins on: (1) CRC pass, then (2) highest pilot amplitude.
        if best is None \
           or (result.ok and not best.ok) \
           or (result.ok == best.ok and result.pilot_amp_rx > best_pilot):
            best = result
            best_pilot = result.pilot_amp_rx

    if best is None:
        # Every offset failed the pilot/background gate — sync probably wrong.
        return DecodeResult(False, "no_valid_offset", -1, b"", 0, 0,
                             0.0, [], bps, cfo_hz, 0.0, 0.0)
    return best


# =============================================================================
# Live capture from Pluto (mirrors step10 worker; minimal)
# =============================================================================
def capture_from_pluto(
    uri: str, fc: float, fs: float, rx_gain: float,
    fixed_len: int, num_captures: int,
    tx_iq_file: str = "",
    tx_gain: float = 0.0,
    tx_iq_scale: float = 4096.0,
) -> List[np.ndarray]:
    """Open Pluto and capture RX windows.

    If `tx_iq_file` is given, the same Pluto is also configured as TX in
    cyclic mode — this is the "monostatic" loopback mode where TX and RX
    share the same LO (zero CFO between them) and the user can wave the
    cable-extended RX antenna while the TX antenna stays static.
    """
    import adi
    sdr = adi.Pluto(uri=uri)
    sdr.sample_rate          = int(fs)
    sdr.rx_lo                = int(fc)
    sdr.rx_rf_bandwidth      = int(fs * 1.2)
    sdr.rx_hardwaregain_chan0 = float(rx_gain)
    sdr.gain_control_mode_chan0 = "manual"
    sdr.rx_enabled_channels  = [0]
    sdr.rx_buffer_size       = int(fixed_len)

    if tx_iq_file:
        sdr.tx_lo                = int(fc)
        sdr.tx_rf_bandwidth      = int(fs * 1.2)
        sdr.tx_enabled_channels  = [0]
        try: sdr.dds_scales = (0.0,) * 8
        except Exception: pass
        try: sdr.tx_destroy_buffer()
        except Exception: pass
        sdr.tx_cyclic_buffer = False
        # Deterministic gain apply (3x rewrite with zero-buffer prime,
        # same protocol as rf_stream_tx_playback.py)
        for _ in range(3):
            sdr.tx_hardwaregain_chan0 = float(tx_gain)
            try:
                sdr.tx(np.zeros(1024, dtype=np.complex64))
                sdr.tx_destroy_buffer()
            except Exception: pass
            time.sleep(0.05)
        applied = float(sdr.tx_hardwaregain_chan0)
        sdr.tx_cyclic_buffer = True
        buf = np.fromfile(tx_iq_file, dtype=np.complex64)
        tx_data = (np.conj(buf) * tx_iq_scale).astype(np.complex64)
        sdr.tx(tx_data)
        print(f"[MONO-TX] {tx_iq_file}  {tx_data.size} samp = "
              f"{tx_data.size/fs*1e3:.2f} ms loop  "
              f"tx_gain req={tx_gain} app={applied:.2f} dB  "
              f"(MONOSTATIC: shared LO, zero CFO baseline)")

    print(f"[RX] Pluto opened.  fs={fs/1e6:.1f}Msps  fc={fc/1e6:.3f}MHz  rx_gain={rx_gain}dB")

    captures: List[np.ndarray] = []
    for i in range(num_captures):
        rxw = sdr.rx().astype(np.complex64)
        captures.append(rxw)
        if i == 0 or (i + 1) % 10 == 0:
            print(f"[RX] capture {i+1}/{num_captures}  rms={np.sqrt(np.mean(np.abs(rxw)**2)):.4f}")
    return captures


# =============================================================================
# Offline NPZ replay
# =============================================================================
def load_npzs(paths: List[str]) -> List[Tuple[str, np.ndarray]]:
    out: List[Tuple[str, np.ndarray]] = []
    for p in paths:
        with np.load(p, allow_pickle=True) as d:
            if "rxw" in d.files:
                out.append((p, np.array(d["rxw"], dtype=np.complex64)))
    return out


def load_tx_truth(path: str) -> Dict[int, dict]:
    """tx_truth.jsonl → {seq: record}.  Empty dict if file not found."""
    if not path or not os.path.isfile(path):
        return {}
    out: Dict[int, dict] = {}
    with open(path) as fh:
        for line in fh:
            try:
                r = json.loads(line)
                out[int(r["seq"])] = r
            except Exception:
                continue
    return out


def reconstruct_truth_bits(record: dict, ref_seed: int, ref_len: int) -> np.ndarray:
    """Rebuild the scrambled+repeated bit-stream that was transmitted, so we
    can compute exact BER from the demodulated bits."""
    rng = np.random.default_rng(ref_seed)
    data = rng.integers(0, 256, size=ref_len, dtype=np.uint8).tobytes()
    # Pick chunk corresponding to this seq
    chunk_bytes = record.get("payload_len", 0)
    seq = int(record["seq"])
    if chunk_bytes <= 0:
        return np.zeros(0, dtype=np.uint8)
    payload = data[seq * chunk_bytes: (seq + 1) * chunk_bytes]
    # Rebuild full frame_bytes
    total = int(record["total"])
    hdr = (MAGIC
           + int(seq).to_bytes(2, "little")
           + int(total).to_bytes(2, "little")
           + int(len(payload)).to_bytes(2, "little"))
    body = hdr + payload
    crc = zlib.crc32(body) & 0xFFFFFFFF
    fb = body + int(crc).to_bytes(4, "little")
    bits = np.unpackbits(np.frombuffer(fb, dtype=np.uint8), bitorder="big").astype(np.uint8)
    bits = scramble_bits(bits)
    rep = int(record.get("repeat", 1))
    if rep > 1:
        bits = np.repeat(bits, rep)
    return bits


# =============================================================================
# Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser("OTFS PHY RX — Phase 1")
    ap.add_argument("--mode", choices=["live", "offline"], default="offline")

    # live-only
    ap.add_argument("--uri",       default="ip:192.168.2.2")
    ap.add_argument("--fc",        type=float, default=2.4e9)
    ap.add_argument("--fs",        type=float, default=3e6)
    ap.add_argument("--rx_gain",   type=float, default=40.0)
    ap.add_argument("--fixed_len", type=int,   default=65536)
    ap.add_argument("--num_captures", type=int, default=50)

    # offline-only
    ap.add_argument("--rx_npz_glob", type=str, default="",
                    help="Glob for NPZ files to decode in offline mode")

    # shared
    ap.add_argument("--modulation", default="qpsk", choices=list(MOD_BPS.keys()))
    ap.add_argument("--repeat",      type=int, default=1, choices=[1, 2, 4])
    ap.add_argument("--stf_repeats", type=int, default=6)
    ap.add_argument("--ltf_symbols", type=int, default=4)

    ap.add_argument("--otfs_M",     type=int, default=64)
    ap.add_argument("--otfs_N",     type=int, default=8)
    ap.add_argument("--otfs_l_max", type=int, default=6)
    ap.add_argument("--otfs_k_max", type=int, default=1)
    ap.add_argument("--otfs_pilot_doppler", type=int, default=4)
    ap.add_argument("--otfs_pilot_boost_db", type=float, default=12.0)
    ap.add_argument("--otfs_num_blocks", type=int, default=3,
                    help="Number of OTFS DD blocks per packet (must match TX).")
    ap.add_argument("--payload_bytes", type=int, default=256,
                    help="Expected per-packet payload (== --chunk_bytes on TX).")

    ap.add_argument("--tx_truth", type=str, default="",
                    help="Path to tx_truth.jsonl for BER computation")
    ap.add_argument("--ref_seed", type=int, default=0)
    ap.add_argument("--ref_len",  type=int, default=0)

    ap.add_argument("--out_root", default="rf_stream_rx_runs_otfs")
    ap.add_argument("--fec", choices=["none", "ldpc"], default="none",
                    help="Outer FEC.  ldpc uses 5G NR rate-1/2 BG2.")
    ap.add_argument("--ldpc_k", type=int, default=256,
                    help="LDPC info bits per block (k=256, n=512 → rate 1/2)")
    ap.add_argument("--ldpc_n", type=int, default=512,
                    help="LDPC coded bits per block")
    ap.add_argument("--save_npz", action="store_true",
                    help="Save each capture to NPZ (live mode default ON)")

    # ── Monostatic loopback (same Pluto for TX + RX, shared LO) ────────────
    ap.add_argument("--tx_iq_file", type=str, default="",
                    help="If set, configure the SAME Pluto as TX cyclic with this raw "
                         "complex64 IQ file before RX captures.  Use with a cable-extended "
                         "RX antenna for monostatic loopback (zero CFO baseline).")
    ap.add_argument("--tx_gain",    type=float, default=0.0,
                    help="TX hardware gain in dB (used only in monostatic mode).")

    args = ap.parse_args()
    if args.mode == "live":
        args.save_npz = True

    otfs_cfg = OTFSConfig(
        M=args.otfs_M, N=args.otfs_N,
        l_max=args.otfs_l_max, k_max=args.otfs_k_max,
        pilot_doppler=args.otfs_pilot_doppler,
        pilot_boost_db=args.otfs_pilot_boost_db,
    )

    ldpc = None
    if args.fec == "ldpc":
        from otfs.otfs_fec import OTFSLDPC, FECConfig
        ldpc = OTFSLDPC(FECConfig(enabled=True, k=args.ldpc_k, n=args.ldpc_n))
        print(f"[RX] FEC=LDPC  k={args.ldpc_k} n={args.ldpc_n} rate={args.ldpc_k/args.ldpc_n:.2f}")

    run_id  = datetime.now().strftime("run_otfs_rx_%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out_root, run_id)
    os.makedirs(out_dir, exist_ok=True)

    truth = load_tx_truth(args.tx_truth)
    if truth:
        print(f"[RX] loaded {len(truth)} TX truth records from {args.tx_truth}")

    # ── Gather captures ─────────────────────────────────────────────────────
    if args.mode == "live":
        captures = capture_from_pluto(args.uri, args.fc, args.fs, args.rx_gain,
                                       args.fixed_len, args.num_captures,
                                       tx_iq_file=args.tx_iq_file,
                                       tx_gain=args.tx_gain)
        capture_paths = [f"<live:{i}>" for i in range(len(captures))]
    else:
        if not args.rx_npz_glob:
            print("[RX] --rx_npz_glob required in offline mode", file=sys.stderr)
            sys.exit(2)
        capture_paths = sorted(glob.glob(args.rx_npz_glob))
        if not capture_paths:
            print(f"[RX] no NPZs match {args.rx_npz_glob}", file=sys.stderr)
            sys.exit(2)
        loaded = load_npzs(capture_paths)
        capture_paths = [p for p, _ in loaded]
        captures = [r for _, r in loaded]
    print(f"[RX] processing {len(captures)} captures")

    # ── Decode loop ─────────────────────────────────────────────────────────
    rx_log_path = os.path.join(out_dir, "rx_decode.jsonl")
    n_total_pkts = n_ok_pkts = 0
    total_bits = total_errors = 0
    with open(rx_log_path, "w") as logfh:
        for cap_i, (path, rxw) in enumerate(zip(capture_paths, captures)):
            t0 = time.time()
            res = decode_otfs_window(
                rxw, otfs_cfg, args.modulation, args.otfs_num_blocks,
                args.repeat, args.stf_repeats, args.ltf_symbols,
                args.fs, args.payload_bytes, try_conjugate=True,
                ldpc=ldpc,
            )

            # ── Compare against tx_truth for exact BER ───────────────────
            n_bits = n_errors = 0
            seq_truth = truth.get(res.seq) if res.seq >= 0 else None
            if seq_truth is not None and args.ref_len > 0:
                bits_true_full = reconstruct_truth_bits(seq_truth, args.ref_seed, args.ref_len)
                # Need to rerun decode to get raw demap bits — simpler: rely on packet CRC
                # for ok/fail count; per-bit BER is recoverable but adds complexity we'll
                # add when we have hardware data.
                n_bits = len(bits_true_full)
                n_errors = 0 if res.ok else n_bits   # conservative on failure

            n_total_pkts += 1
            n_ok_pkts    += int(res.ok)
            total_bits   += n_bits
            total_errors += n_errors

            log_entry = {
                "cap_idx": cap_i,
                "path":    path,
                "ok":      bool(res.ok),
                "reason":  res.reason,
                "seq":     int(res.seq),
                "payload_len": int(len(res.payload)),
                "pilot_amp_rx": float(res.pilot_amp_rx),
                "cfo_hz":  float(res.cfo_hz),
                "snr_pre_db":  float(res.snr_pre_db),
                "snr_post_db": float(res.snr_post_db),
                "n_dd_taps": int(len(res.dd_channel_taps)),
                "dd_taps_db": [
                    {"dl": int(dl), "dk": int(dk),
                     "mag_db": float(20.0 * np.log10(abs(h) + 1e-12)),
                     "phase_deg": float(np.degrees(np.angle(h)))}
                    for dl, dk, h in res.dd_channel_taps
                ],
                "decode_ms": float((time.time() - t0) * 1e3),
            }
            logfh.write(json.dumps(log_entry) + "\n")
            logfh.flush()

            # Save NPZ in AutoDataPHY-compatible format
            if args.save_npz:
                tag = "ok" if res.ok else "bg"
                npz_name = f"cap_{cap_i:06d}_{tag}.npz"
                meta = {
                    "waveform":  "OTFS",
                    "stf_idx":   0,   # we ran sync internally; could log it
                    "snr_db":    float(res.snr_post_db),
                    "mod":       args.modulation,
                    "gate":      int(res.ok),
                    "gain_tx":   0,   # caller knows from TX log
                    "gain_rx":   float(args.rx_gain) if args.mode == "live" else 0,
                    "cfo_hz":    float(res.cfo_hz),
                    "timestamp": datetime.now().isoformat(),
                    "otfs_M":    otfs_cfg.M,
                    "otfs_N":    otfs_cfg.N,
                    "otfs_l_max": otfs_cfg.l_max,
                    "otfs_k_max": otfs_cfg.k_max,
                    "otfs_num_blocks": int(args.otfs_num_blocks),
                }
                np.savez_compressed(
                    os.path.join(out_dir, npz_name),
                    rxw=rxw,
                    meta_json=json.dumps(meta),
                )

            if cap_i == 0 or (cap_i + 1) % 10 == 0:
                pkt_rate = 100.0 * n_ok_pkts / max(n_total_pkts, 1)
                print(f"[RX] {cap_i+1:5d}/{len(captures):5d}  "
                      f"ok={res.ok!s:5}  seq={res.seq:>4}  "
                      f"cfo={res.cfo_hz:+7.1f}Hz  "
                      f"pilot_rx={res.pilot_amp_rx:.3f}  "
                      f"snr_post={res.snr_post_db:+5.1f}dB  "
                      f"running pkt_rate={pkt_rate:.1f}%")

    pkt_rate = 100.0 * n_ok_pkts / max(n_total_pkts, 1)
    summary = {
        "n_captures":   n_total_pkts,
        "n_decoded_ok": n_ok_pkts,
        "decode_rate":  pkt_rate / 100.0,
        "total_bits":   total_bits,
        "total_errors": total_errors,
        "ber":          (total_errors / total_bits) if total_bits > 0 else None,
        "modulation":   args.modulation,
        "otfs": {"M": otfs_cfg.M, "N": otfs_cfg.N,
                 "l_max": otfs_cfg.l_max, "k_max": otfs_cfg.k_max,
                 "num_blocks": args.otfs_num_blocks},
        "mode":         args.mode,
        "run_id":       run_id,
        "finished_at":  datetime.now().isoformat(),
    }
    with open(os.path.join(out_dir, "rx_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\n" + "=" * 70)
    print(f"OTFS RX done.  decoded {n_ok_pkts}/{n_total_pkts} = {pkt_rate:.1f}%   out={out_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
