#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
doppler_sweep_loopback.py — Pure-software Doppler sweep comparing
                            OTFS+LDPC vs OFDM+LDPC, no Pluto required.

Each trial:
    1. Build a TX frame (OTFS or OFDM) with optional LDPC, using the same
       primitives the hardware TX uses.
    2. Apply a synthetic 2-tap doubly-selective channel:
            y[n] = x[n] + alpha * x[n-tau] * exp(j*2*pi*fd*n/fs)
       (matches the channel `inject_doppler_to_iq.py` writes into the IQ).
    3. Add complex AWGN at a fixed SNR.
    4. Sync (Schmidl-Cox + xcorr) and decode through the same RX primitives
       the live RX scripts use, then run LDPC soft decode.
    5. Tally raw-BER, post-LDPC packet decode rate.

Sweep `--doppler_hz` for both waveforms back-to-back at fixed SNR.

Usage:
    python3 rf_stream/doppler_sweep_loopback.py \\
        --snr_db 15 --n_trials 30 \\
        --doppler_list 0,100,500,1000,2000,3000,5000 \\
        --tap_db -3 --tap_delay 3
"""

import argparse
import json
import os
import sys
import zlib
import time
from typing import Tuple, List, Optional, Dict

import numpy as np

# ── path setup ───────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# OTFS primitives
from otfs.otfs_core import (
    OTFSConfig, MOD_BPS,
    sfft_time_to_dd,
    estimate_dd_channel_taps,
    build_effective_dd_matrix,
    dd_lmmse_detect,
    make_constellation,
    hard_demap,
    symbols_to_bits as otfs_symbols_to_bits,
    bits_to_bytes as otfs_bits_to_bytes,
    bits_from_bytes as otfs_bits_from_bytes,
    scramble_bits as otfs_scramble_bits,
    num_data_cells,
)
from otfs.otfs_frame import (
    bytes_to_otfs_samples,
    create_schmidl_cox_stf as create_stf_otfs,
    create_ltf as create_ltf_otfs,
    build_packet_bytes as build_packet_bytes_otfs,
)
from otfs.otfs_sync import sync_preamble as sync_preamble_otfs
from otfs.otfs_fec import OTFSLDPC, FECConfig, qpsk_soft_demap

# OFDM primitives — pull directly from the live TX/RX modules
from rf_stream_tx_step6phy import (
    bytes_to_ofdm_samples,
    create_schmidl_cox_stf as create_stf_ofdm,
    create_ltf as create_ltf_ofdm,
    build_packet_bytes as build_packet_bytes_ofdm,
    make_constellation as make_const_ofdm,
    N_FFT, N_CP, SYMBOL_LEN, N_DATA,
    DATA_BINS, PILOT_BINS,
)
from rf_stream_rx_step10phy import (
    extract_ofdm_symbol,
    channel_estimate_from_ltf,
    equalize_mmse,
    pilot_residual_correction,
    create_ltf_ref,
    USED_BINS,
)


FS    = 3.0e6
MAGIC = b"AIS1"


# ─────────────────────────────────────────────────────────────────────────────
# Channel
# ─────────────────────────────────────────────────────────────────────────────
def apply_2tap_doppler(sig: np.ndarray, fs: float, fd_hz: float,
                       tap_delay: int, tap_db: float,
                       rng: np.random.Generator) -> np.ndarray:
    """y[n] = x[n] + alpha * x[n-tau] * exp(j(2*pi*fd*n/fs + phi))

    `phi` is a random per-trial phase so back-to-back trials don't all see
    the same constructive/destructive interference pattern.
    """
    alpha = 10.0 ** (tap_db / 20.0)
    n = np.arange(sig.size, dtype=np.float64)
    phi = rng.uniform(-np.pi, np.pi)
    phasor = np.exp(1j * (2.0 * np.pi * fd_hz * n / fs + phi)).astype(np.complex64)
    x_del = np.zeros_like(sig)
    if tap_delay > 0:
        x_del[tap_delay:] = sig[:-tap_delay]
    else:
        x_del[:] = sig
    return (sig + alpha * phasor * x_del).astype(np.complex64)


def apply_awgn(sig: np.ndarray, snr_db: float,
               rng: np.random.Generator) -> np.ndarray:
    sig_p = float(np.mean(np.abs(sig) ** 2) + 1e-12)
    noise_p = sig_p / (10.0 ** (snr_db / 10.0))
    n = (rng.standard_normal(sig.size) + 1j * rng.standard_normal(sig.size)).astype(np.complex64)
    return (sig + n * np.sqrt(noise_p / 2.0)).astype(np.complex64)


def parse_packet_bytes(bb: bytes) -> Tuple[bool, int, bytes]:
    if len(bb) < 14 or bb[:4] != MAGIC:
        return False, -1, b""
    seq  = int.from_bytes(bb[4:6], "little")
    plen = int.from_bytes(bb[8:10], "little")
    if len(bb) < 10 + plen + 4:
        return False, seq, b""
    body = bb[: 10 + plen]
    crc  = int.from_bytes(bb[10 + plen: 14 + plen], "little")
    if (zlib.crc32(body) & 0xFFFFFFFF) != crc:
        return False, seq, b""
    return True, seq, bb[10: 10 + plen]


# ─────────────────────────────────────────────────────────────────────────────
# OTFS path
# ─────────────────────────────────────────────────────────────────────────────
def otfs_trial(
    payload: bytes,
    cfg: OTFSConfig,
    modulation: str,
    stf_repeats: int,
    ltf_symbols: int,
    fd_hz: float,
    snr_db: float,
    tap_delay: int,
    tap_db: float,
    ldpc: Optional[OTFSLDPC],
    rng: np.random.Generator,
) -> Dict:
    stf = create_stf_otfs(stf_repeats)
    ltf = create_ltf_otfs(ltf_symbols)
    fb  = build_packet_bytes_otfs(seq=0, total=1, payload=payload)
    tx, info = bytes_to_otfs_samples(
        fb, repeat=1, stf=stf, ltf=ltf, cfg=cfg,
        modulation=modulation, fs=FS,
        tone_duration_ms=0.0, tone_freq_hz=0.0,
        gap_short=200, gap_long=400, tx_scale=0.9,
        ldpc=ldpc,
    )

    rx = apply_2tap_doppler(tx, FS, fd_hz, tap_delay, tap_db, rng)
    rx = apply_awgn(rx, snr_db, rng)

    sync = sync_preamble_otfs(rx, stf_repeats, ltf_symbols, FS)
    if sync is None:
        return {"ok": False, "reason": "sync_fail", "raw_ber": 0.5,
                "ldpc_ok": False}

    rx_corr      = sync["rx_corr"]
    payload_start = sync["payload_start"]
    MN = cfg.M * cfg.N
    const_tbl = make_constellation(modulation)
    bps = MOD_BPS[modulation.lower()]
    cp_len = max(int(cfg.l_max), 64)
    block_stride = cp_len + MN

    raw_bits_blocks: List[np.ndarray] = []
    llrs_blocks:     List[np.ndarray] = []
    for b in range(info.num_blocks):
        start = payload_start + b * block_stride + cp_len
        if start + MN > rx_corr.size:
            return {"ok": False, "reason": f"truncated_{b}",
                    "raw_ber": 0.5, "ldpc_ok": False}
        blk = rx_corr[start: start + MN]
        rx_dd = sfft_time_to_dd(blk, cfg.M, cfg.N)
        pilot_amp_est = float(np.max(np.abs(
            rx_dd[cfg.pilot_delay: cfg.pilot_delay + cfg.l_max + 1, :])))
        if pilot_amp_est < 1e-9:
            return {"ok": False, "reason": f"no_pilot_{b}",
                    "raw_ber": 0.5, "ldpc_ok": False}
        taps = estimate_dd_channel_taps(rx_dd, cfg, pilot_amp_est)
        H_dd = build_effective_dd_matrix(taps, cfg.M, cfg.N)
        data_pow = float(np.mean(np.abs(rx_dd) ** 2) + 1e-12)
        noise_var = max(1e-6, data_pow * 0.05)
        x_hat = dd_lmmse_detect(rx_dd, H_dd, noise_var, cfg)

        # raw hard demap
        idx = hard_demap(x_hat, const_tbl)
        raw_bits_blocks.append(otfs_symbols_to_bits(idx, bps))

        if ldpc is not None:
            # match the live RX scaling
            scale = float(np.median(np.abs(x_hat)) / (1.0 / np.sqrt(2.0)) + 1e-12)
            xn = (x_hat / max(scale, 1e-9)).astype(np.complex64)
            diff = xn[:, None] - const_tbl[None, :]
            d2 = np.min(diff.real ** 2 + diff.imag ** 2, axis=1)
            eff_s2 = float(max(0.02, np.mean(d2)))
            llr = qpsk_soft_demap(xn, noise_var=eff_s2,
                                  const_table=const_tbl.astype(np.complex64))
            llr = np.clip(llr, -20.0, 20.0).astype(np.float32)
            llrs_blocks.append(llr)

    raw_bits = np.concatenate(raw_bits_blocks).astype(np.uint8)
    raw_descr = otfs_scramble_bits(raw_bits)
    rx_bytes_raw = otfs_bits_to_bytes(raw_descr)
    raw_ok, _, _ = parse_packet_bytes(rx_bytes_raw)

    # raw BER vs the (scrambled + ldpc-encoded) on-air bits
    if ldpc is not None:
        tx_bits_oa = ldpc.encode(otfs_scramble_bits(otfs_bits_from_bytes(fb)))
    else:
        tx_bits_oa = otfs_scramble_bits(otfs_bits_from_bytes(fb))
    n = min(raw_bits.size, tx_bits_oa.size)
    raw_ber = float(np.mean(raw_bits[:n] != tx_bits_oa[:n]))

    ldpc_ok = False
    if ldpc is not None and llrs_blocks:
        all_llr = np.concatenate(llrs_blocks)
        info_bits = ldpc.decode_soft(all_llr)
        unscr = otfs_scramble_bits(info_bits[: len(fb) * 8])
        rx_bytes_ldpc = otfs_bits_to_bytes(unscr)
        ok, _, payload_rx = parse_packet_bytes(rx_bytes_ldpc)
        ldpc_ok = ok and (payload_rx == payload)

    return {"ok": raw_ok, "reason": "ok",
            "raw_ber": raw_ber, "ldpc_ok": ldpc_ok}


# ─────────────────────────────────────────────────────────────────────────────
# OFDM path
# ─────────────────────────────────────────────────────────────────────────────
def _ofdm_decode_data(rx: np.ndarray, ltf_start: int, data_start: int,
                       num_syms: int, ltf_freq_ref: np.ndarray,
                       ltf_symbols: int, pilot_vals: np.ndarray,
                       const_tbl: np.ndarray, bps: int,
                       noise_var: float) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (Yeq_data, hard_bits) — same structure step10 saves to NPZ."""
    H, _ = channel_estimate_from_ltf(rx, ltf_start, ltf_freq_ref, ltf_symbols)
    if H is None:
        return np.zeros((0, N_DATA), dtype=np.complex64), np.zeros(0, dtype=np.uint8)

    Yeq_data = np.zeros((num_syms, N_DATA), dtype=np.complex64)
    for si in range(num_syms):
        Y = extract_ofdm_symbol(rx, data_start + si * SYMBOL_LEN)
        if Y is None:
            break
        Ye = equalize_mmse(Y, H, noise_var, USED_BINS)
        sign = 1 if si % 2 == 0 else -1
        Hres = pilot_residual_correction(Ye, sign, pilot_vals, bypass_th=0.10)
        # residual is a complex correction; apply phase-only by dividing
        Ye2 = Ye / Hres
        Yeq_data[si] = Ye2[DATA_BINS]

    syms = Yeq_data.flatten()
    # Hard demap
    idx = np.argmin(np.abs(syms[:, None] - const_tbl[None, :]) ** 2, axis=1)
    hard_bits = np.empty(syms.size * bps, dtype=np.uint8)
    if bps == 2:
        hard_bits[0::2] = (idx >> 1) & 1
        hard_bits[1::2] = idx & 1
    else:
        for b in range(bps):
            hard_bits[b::bps] = (idx >> (bps - 1 - b)) & 1
    return Yeq_data, hard_bits


def ofdm_trial(
    payload: bytes,
    modulation: str,
    stf_repeats: int,
    ltf_symbols: int,
    fd_hz: float,
    snr_db: float,
    tap_delay: int,
    tap_db: float,
    ldpc: Optional[OTFSLDPC],
    rng: np.random.Generator,
) -> Dict:
    stf = create_stf_ofdm(stf_repeats)
    ltf = create_ltf_ofdm(ltf_symbols)
    fb  = build_packet_bytes_ofdm(seq=0, total=1, payload=payload)
    bps = MOD_BPS[modulation.lower()]
    const_tbl = make_const_ofdm(modulation)

    tx = bytes_to_ofdm_samples(
        fb, repeat=1, stf=stf, ltf=ltf,
        const_table=const_tbl, bps=bps, fs=FS,
        tone_duration_ms=0.0, tone_freq_hz=0.0,
        gap_short=200, gap_long=400, tx_scale=0.9,
        ldpc=ldpc,
    )

    rx = apply_2tap_doppler(tx, FS, fd_hz, tap_delay, tap_db, rng)
    rx = apply_awgn(rx, snr_db, rng)

    # In loopback we know the frame layout exactly — skip the noisy sync.
    #   tx = [gL(400) | tone(0) | gS(200) | stf | ltf | ofdm | gL(400)]
    gap_long = 400
    gap_short = 200
    stf_off  = gap_long + gap_short
    ltf_off  = stf_off + stf.size
    data_off = ltf_off + ltf.size

    # LTF freq-domain reference matches what step6 TX puts on the wire
    # (alternating +1/-1 across used subcarriers, indexed in bin/IFFT-input space).
    _, ltf_freq_ref = create_ltf_ref(ltf_symbols)

    pilot_vals = np.array([1, 1, 1, -1], dtype=np.complex64)
    if ldpc is not None:
        bits_total = ldpc.cfg.n * ((len(fb) * 8 + ldpc.cfg.k - 1) // ldpc.cfg.k)
    else:
        bits_total = len(fb) * 8
    num_syms = int(np.ceil(bits_total / (N_DATA * bps)))

    # noise-variance estimate: use the trailing gap_long zero block as noise sample
    noise_sample = rx[data_off + num_syms * SYMBOL_LEN:
                       data_off + num_syms * SYMBOL_LEN + gap_long]
    noise_var = float(np.mean(np.abs(noise_sample) ** 2) + 1e-9) if noise_sample.size > 0 else 1e-6

    Yeq_data, hard_bits = _ofdm_decode_data(
        rx, ltf_off, data_off, num_syms, ltf_freq_ref, ltf_symbols,
        pilot_vals, const_tbl, bps, noise_var
    )

    # raw BER vs on-air bits
    if ldpc is not None:
        tx_bits_oa = ldpc.encode(otfs_scramble_bits(otfs_bits_from_bytes(fb)))
    else:
        tx_bits_oa = otfs_scramble_bits(otfs_bits_from_bytes(fb))
    n = min(hard_bits.size, tx_bits_oa.size)
    raw_ber = float(np.mean(hard_bits[:n] != tx_bits_oa[:n])) if n > 0 else 0.5

    # raw packet check
    raw_descr = otfs_scramble_bits(hard_bits[: len(fb) * 8])
    rx_bytes_raw = otfs_bits_to_bytes(raw_descr)
    raw_ok, _, _ = parse_packet_bytes(rx_bytes_raw)

    ldpc_ok = False
    if ldpc is not None and Yeq_data.size > 0:
        syms = Yeq_data.flatten().astype(np.complex64)
        scale = float(np.median(np.abs(syms)) / (1.0 / np.sqrt(2.0)) + 1e-12)
        xn = (syms / max(scale, 1e-9)).astype(np.complex64)
        diff = xn[:, None] - const_tbl[None, :]
        d2 = np.min(diff.real ** 2 + diff.imag ** 2, axis=1)
        eff_s2 = float(max(0.02, np.mean(d2)))
        llr = qpsk_soft_demap(xn, noise_var=eff_s2,
                              const_table=const_tbl.astype(np.complex64))
        llr = np.clip(llr, -20.0, 20.0).astype(np.float32)
        info_bits = ldpc.decode_soft(llr)
        unscr = otfs_scramble_bits(info_bits[: len(fb) * 8])
        rx_bytes_ldpc = otfs_bits_to_bytes(unscr)
        ok, _, payload_rx = parse_packet_bytes(rx_bytes_ldpc)
        ldpc_ok = ok and (payload_rx == payload)

    return {"ok": raw_ok, "reason": "ok",
            "raw_ber": raw_ber, "ldpc_ok": ldpc_ok}


# ─────────────────────────────────────────────────────────────────────────────
# Sweep driver
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser("OTFS vs OFDM software-loopback Doppler sweep")
    ap.add_argument("--modulation",  default="qpsk", choices=list(MOD_BPS.keys()))
    ap.add_argument("--stf_repeats", type=int, default=6)
    ap.add_argument("--ltf_symbols", type=int, default=4)
    ap.add_argument("--otfs_M",      type=int, default=64)
    ap.add_argument("--otfs_N",      type=int, default=8)
    ap.add_argument("--otfs_l_max",  type=int, default=6)
    ap.add_argument("--otfs_k_max",  type=int, default=1)
    ap.add_argument("--otfs_pilot_doppler",   type=int,   default=4)
    ap.add_argument("--otfs_pilot_boost_db",  type=float, default=12.0)
    ap.add_argument("--payload_bytes", type=int, default=64)
    ap.add_argument("--snr_db",        type=float, default=15.0)
    ap.add_argument("--n_trials",      type=int, default=30)
    ap.add_argument("--doppler_list",  type=str,
                    default="0,100,500,1000,2000,3000,5000",
                    help="Comma-separated Doppler frequencies (Hz) to test.")
    ap.add_argument("--tap_delay",     type=int,   default=3)
    ap.add_argument("--tap_db",        type=float, default=-3.0)
    ap.add_argument("--ldpc_k",        type=int,   default=256)
    ap.add_argument("--ldpc_n",        type=int,   default=512)
    ap.add_argument("--no_ldpc",       action="store_true")
    ap.add_argument("--seed",          type=int,   default=42)
    ap.add_argument("--out_json", default="rf_stream/doppler_sweep_results.json")
    args = ap.parse_args()

    cfg = OTFSConfig(M=args.otfs_M, N=args.otfs_N,
                     l_max=args.otfs_l_max, k_max=args.otfs_k_max,
                     pilot_doppler=args.otfs_pilot_doppler,
                     pilot_boost_db=args.otfs_pilot_boost_db)
    ldpc = None
    if not args.no_ldpc:
        ldpc = OTFSLDPC(FECConfig(enabled=True, k=args.ldpc_k, n=args.ldpc_n))

    fd_list = [float(x) for x in args.doppler_list.split(",") if x.strip()]
    rng = np.random.default_rng(args.seed)

    print(f"\n{'='*78}")
    print(f"  Software-loopback Doppler sweep — SNR={args.snr_db} dB, "
          f"channel: direct + {args.tap_db}dB @ delay={args.tap_delay} samples")
    print(f"  n_trials={args.n_trials}, modulation={args.modulation}, "
          f"LDPC={'rate1/2 k=%d n=%d'%(args.ldpc_k,args.ldpc_n) if ldpc else 'OFF'}")
    print(f"{'='*78}\n")

    hdr = f"  {'fd(Hz)':>8} | {'OTFS pkt':>10} {'OTFS raw_ber':>13} | {'OFDM pkt':>10} {'OFDM raw_ber':>13}"
    print(hdr); print("  " + "-" * (len(hdr) - 2))

    results = {"snr_db": args.snr_db, "n_trials": args.n_trials,
               "tap_delay": args.tap_delay, "tap_db": args.tap_db,
               "ldpc": (None if ldpc is None
                        else {"k": args.ldpc_k, "n": args.ldpc_n}),
               "rows": []}

    for fd in fd_list:
        otfs_ok = 0
        otfs_raw_ok = 0
        otfs_raw_bers = []
        ofdm_ok = 0
        ofdm_raw_ok = 0
        ofdm_raw_bers = []
        t0 = time.time()
        for t in range(args.n_trials):
            payload = rng.integers(0, 256, size=args.payload_bytes,
                                    dtype=np.uint8).tobytes()
            o = otfs_trial(payload, cfg, args.modulation,
                            args.stf_repeats, args.ltf_symbols,
                            fd, args.snr_db, args.tap_delay, args.tap_db,
                            ldpc, rng)
            f = ofdm_trial(payload, args.modulation,
                            args.stf_repeats, args.ltf_symbols,
                            fd, args.snr_db, args.tap_delay, args.tap_db,
                            ldpc, rng)
            otfs_raw_bers.append(o["raw_ber"])
            ofdm_raw_bers.append(f["raw_ber"])
            if o["ok"]:     otfs_raw_ok += 1
            if o["ldpc_ok"]: otfs_ok    += 1
            if f["ok"]:     ofdm_raw_ok += 1
            if f["ldpc_ok"]: ofdm_ok    += 1
        dt = time.time() - t0

        otfs_pct = 100.0 * otfs_ok / args.n_trials
        ofdm_pct = 100.0 * ofdm_ok / args.n_trials
        otfs_rb  = float(np.mean(otfs_raw_bers))
        ofdm_rb  = float(np.mean(ofdm_raw_bers))
        print(f"  {fd:>8.0f} | {otfs_ok:>3d}/{args.n_trials} ({otfs_pct:5.1f}%)"
              f"  {otfs_rb:>10.4f} | "
              f"{ofdm_ok:>3d}/{args.n_trials} ({ofdm_pct:5.1f}%)  {ofdm_rb:>10.4f}"
              f"   [{dt:.1f}s]")

        results["rows"].append({
            "fd_hz": fd,
            "otfs_pkt_ok": otfs_ok, "otfs_raw_ok": otfs_raw_ok,
            "otfs_raw_ber_mean": otfs_rb,
            "ofdm_pkt_ok": ofdm_ok, "ofdm_raw_ok": ofdm_raw_ok,
            "ofdm_raw_ber_mean": ofdm_rb,
            "n_trials": args.n_trials,
            "elapsed_s": dt,
        })

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()
