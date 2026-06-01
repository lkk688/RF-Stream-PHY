#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_otfs_loopback.py — Software-only loopback sanity test for the OTFS chain.

Runs entirely without Pluto.  Builds a TX frame using the same code paths as
rf_stream_tx_otfs.py, applies a synthetic doubly-selective channel + AWGN,
then decodes through the same primitives the RX uses.  Produces:

    rf_stream/otfs_loopback_results.json   # numeric summary
    rf_stream/otfs_loopback_ber.png        # BER vs SNR curve

Three channel scenarios are tested:

    awgn         AWGN-only, no multipath or Doppler.  Sanity baseline.
    static_mp    3-tap multipath, no Doppler.  Both OFDM and OTFS handle it.
    doubly_sel   3-tap multipath + Doppler shift on each tap.  This is the
                 regime where OTFS *should* outperform single-tap OFDM
                 equalisation — and where the previous AIradar OTFS code
                 failed because it ran TF-domain one-tap equalisation.

Pass criterion for Phase 1:
    awgn        BER < 1e-3 at SNR >= 15 dB
    static_mp   BER < 1e-2 at SNR >= 15 dB
    doubly_sel  BER < 1e-1 at SNR >= 15 dB  (loose — refined in Phase 2)

If those pass, the TX/RX chain is correct enough to take to hardware.
"""

import argparse
import json
import os
import sys
import zlib
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from otfs.otfs_core import (
    OTFSConfig, MOD_BPS,
    sfft_time_to_dd,
    estimate_dd_channel_taps,
    build_effective_dd_matrix,
    dd_lmmse_detect,
    make_constellation,
    hard_demap,
    symbols_to_bits,
    bits_to_bytes,
    bits_from_bytes,
    scramble_bits,
    num_data_cells,
)
from otfs.otfs_frame import (
    bytes_to_otfs_samples,
    create_schmidl_cox_stf,
    create_ltf,
    build_packet_bytes,
    N_FFT, N_CP,
)
from otfs.otfs_sync import sync_preamble


FS = 3.0e6   # 3 MHz, same as our Pluto chain


# ─────────────────────────────────────────────────────────────────────────────
# Channel models
# ─────────────────────────────────────────────────────────────────────────────

def apply_awgn(sig: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    sig_p = float(np.mean(np.abs(sig) ** 2) + 1e-12)
    noise_p = sig_p / (10.0 ** (snr_db / 10.0))
    n = (rng.standard_normal(sig.size) + 1j * rng.standard_normal(sig.size)).astype(np.complex64)
    return (sig + n * np.sqrt(noise_p / 2.0)).astype(np.complex64)


def apply_static_multipath(
    sig: np.ndarray,
    taps_amp_db: List[float] = (0.0, -6.0, -10.0),
    taps_delay: List[int]   = (0, 2, 5),
    rng: np.random.Generator = None,
) -> np.ndarray:
    """Linear convolution with fixed-delay taps; random phases each call."""
    if rng is None:
        rng = np.random.default_rng()
    L = max(taps_delay) + 1
    h = np.zeros(L, dtype=np.complex64)
    for a_db, d in zip(taps_amp_db, taps_delay):
        amp = 10.0 ** (a_db / 20.0)
        phase = rng.uniform(-np.pi, np.pi)
        h[d] = amp * np.exp(1j * phase)
    return np.convolve(sig, h, mode="full")[: sig.size].astype(np.complex64)


def apply_doubly_selective(
    sig: np.ndarray,
    fs: float,
    fc: float = 2.4e9,
    velocity_mps: float = 3.0,         # walking RX
    taps_amp_db: List[float] = (0.0, -6.0, -10.0),
    taps_delay: List[int]   = (0, 2, 5),
    rng: np.random.Generator = None,
) -> np.ndarray:
    """3-tap channel where each tap has an independent Doppler shift.

    For walking RX (3 m/s) at 2.4 GHz, max Doppler ≈ 2.4e9 * 3 / 3e8 = 24 Hz.
    For walking + reflectors, taps see ±f_d uniformly.
    """
    if rng is None:
        rng = np.random.default_rng()
    c     = 3e8
    fd_max = abs(fc * velocity_mps / c)
    t      = np.arange(sig.size, dtype=np.float64) / fs

    out = np.zeros(sig.size, dtype=np.complex64)
    for a_db, d in zip(taps_amp_db, taps_delay):
        amp = 10.0 ** (a_db / 20.0)
        phase = rng.uniform(-np.pi, np.pi)
        fd = rng.uniform(-fd_max, fd_max)
        delayed = np.roll(sig, d)
        delayed[:d] = 0
        doppler = np.exp(1j * (2 * np.pi * fd * t + phase)).astype(np.complex64)
        out += (amp * delayed * doppler).astype(np.complex64)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# TX/RX harness
# ─────────────────────────────────────────────────────────────────────────────

def build_tx_frame(
    payload: bytes,
    cfg: OTFSConfig,
    modulation: str,
    repeat: int,
    stf_repeats: int,
    ltf_symbols: int,
) -> Tuple[np.ndarray, dict]:
    """Build one OTFS TX frame and return (sig, sidecar)."""
    stf = create_schmidl_cox_stf(stf_repeats)
    ltf = create_ltf(ltf_symbols)
    fb  = build_packet_bytes(seq=0, total=1, payload=payload)
    sig, info = bytes_to_otfs_samples(
        fb, repeat, stf, ltf, cfg, modulation,
        fs=FS, tone_duration_ms=0.0, tone_freq_hz=0.0,
        gap_short=200, gap_long=400, tx_scale=0.9,
    )
    sidecar = {
        "frame_bytes":      fb,
        "num_blocks":       info.num_blocks,
        "pilot_amp":        info.pilot_amp,
        "bits_per_block":   info.bits_per_block,
        "bits_padded":      info.bits_padded,
        "data_cells_per_block": info.data_cells_per_block,
    }
    return sig, sidecar


def decode_rx_frame(
    rx: np.ndarray,
    cfg: OTFSConfig,
    modulation: str,
    repeat: int,
    stf_repeats: int,
    ltf_symbols: int,
    num_blocks: int,
) -> Tuple[bytes, np.ndarray, dict]:
    """Decode one captured frame and return (payload_bytes, raw_demap_bits, info)."""
    sync = sync_preamble(rx, stf_repeats, ltf_symbols, FS)
    if sync is None:
        return b"", np.zeros(0, dtype=np.uint8), {"reason": "sync_fail"}

    rx_corr      = sync["rx_corr"]
    payload_start = sync["payload_start"]
    cfo_hz        = sync["cfo_hz"]
    const_tbl     = make_constellation(modulation)
    bps           = MOD_BPS[modulation.lower()]
    MN            = cfg.M * cfg.N

    raw_bit_chunks: List[np.ndarray] = []
    pilot_amps: List[float] = []
    tap_count = 0
    cp_len = max(int(cfg.l_max), 64)   # matches TX (N_FFT-sized CP)
    block_stride = cp_len + MN
    for b in range(num_blocks):
        # Each block on the wire is [CP_l_max | MN samples].  Skip the CP.
        start = payload_start + b * block_stride + cp_len
        if start + MN > rx_corr.size:
            return b"", np.zeros(0, dtype=np.uint8), {"reason": f"truncated_block{b}"}
        block_t = rx_corr[start: start + MN]
        rx_dd   = sfft_time_to_dd(block_t, cfg.M, cfg.N)
        pilot_amp_est = float(np.max(np.abs(
            rx_dd[cfg.pilot_delay: cfg.pilot_delay + cfg.l_max + 1, :])))
        pilot_amps.append(pilot_amp_est)
        if pilot_amp_est < 1e-9:
            raw_bit_chunks.append(np.zeros(num_data_cells(cfg) * bps, dtype=np.uint8))
            continue
        taps  = estimate_dd_channel_taps(rx_dd, cfg, pilot_amp_est)
        tap_count += len(taps)
        H_dd  = build_effective_dd_matrix(taps, cfg.M, cfg.N)

        # noise variance: residual power in data cells beyond pilot
        data_pow  = float(np.mean(np.abs(rx_dd) ** 2) + 1e-12)
        noise_var = max(1e-6, data_pow * 0.05)

        x_hat  = dd_lmmse_detect(rx_dd, H_dd, noise_var, cfg)
        idx    = hard_demap(x_hat, const_tbl)
        bits   = symbols_to_bits(idx, bps)
        raw_bit_chunks.append(bits)

    raw_bits = np.concatenate(raw_bit_chunks).astype(np.uint8)

    # Undo repeat + descramble
    bits_post = raw_bits.copy()
    if repeat > 1:
        L = (bits_post.size // repeat) * repeat
        v = bits_post[:L].reshape(-1, repeat)
        bits_post = (np.sum(v, axis=1) >= (repeat / 2.0)).astype(np.uint8)
    bits_unscr = scramble_bits(bits_post)
    rx_bytes   = bits_to_bytes(bits_unscr)

    info = {
        "reason":      "ok",
        "cfo_hz":      cfo_hz,
        "pilot_amps":  pilot_amps,
        "n_dd_taps":   tap_count,
    }
    return rx_bytes, raw_bits, info


# ─────────────────────────────────────────────────────────────────────────────
# Per-scenario BER measurement
# ─────────────────────────────────────────────────────────────────────────────

def run_scenario(
    name: str,
    snr_list: List[float],
    n_trials: int,
    payload_bytes: int,
    cfg: OTFSConfig,
    modulation: str,
    repeat: int,
    stf_repeats: int,
    ltf_symbols: int,
    channel_fn,
    seed: int = 0,
) -> Dict[str, list]:
    """Return {'snr_db': [...], 'ber': [...], 'pkt_ok_rate': [...]}."""
    rng = np.random.default_rng(seed)

    res = {"snr_db": [], "ber": [], "pkt_ok_rate": [], "n_dd_taps_mean": []}
    for snr in snr_list:
        n_pkts_ok = 0
        n_bits_total = 0
        n_bit_errors = 0
        n_taps_sum = 0
        n_taps_count = 0
        for trial in range(n_trials):
            payload = rng.integers(0, 256, size=payload_bytes, dtype=np.uint8).tobytes()
            tx_sig, side = build_tx_frame(payload, cfg, modulation, repeat, stf_repeats, ltf_symbols)

            # Channel
            if channel_fn is not None:
                rx_sig = channel_fn(tx_sig, rng)
            else:
                rx_sig = tx_sig.copy()
            rx_sig = apply_awgn(rx_sig, snr, rng)

            # Decode
            rx_bytes, raw_bits, info = decode_rx_frame(
                rx_sig, cfg, modulation, repeat, stf_repeats, ltf_symbols, side["num_blocks"]
            )
            if info["reason"] == "ok":
                n_taps_sum += info["n_dd_taps"]
                n_taps_count += 1

            # Per-bit BER vs the *scrambled+repeated* transmitted bits
            #   (so we count true PHY-level errors, not packet-level)
            tx_bits_full = bits_from_bytes(side["frame_bytes"])
            tx_bits_full = scramble_bits(tx_bits_full)
            if repeat > 1:
                tx_bits_full = np.repeat(tx_bits_full, repeat)
            L = min(raw_bits.size, tx_bits_full.size)
            if L > 0:
                n_bits_total += L
                n_bit_errors += int(np.sum(raw_bits[:L] != tx_bits_full[:L]))

            # Packet success: payload matches
            if len(rx_bytes) >= 14 and rx_bytes[:4] == b"AIS1":
                payload_len = int.from_bytes(rx_bytes[8:10], "little")
                if payload_len == payload_bytes and len(rx_bytes) >= 10 + payload_len + 4:
                    rx_payload = rx_bytes[10: 10 + payload_len]
                    if rx_payload == payload:
                        n_pkts_ok += 1

        ber = n_bit_errors / max(n_bits_total, 1)
        pkt_rate = n_pkts_ok / n_trials
        mean_taps = (n_taps_sum / max(n_taps_count, 1))
        res["snr_db"].append(float(snr))
        res["ber"].append(float(ber))
        res["pkt_ok_rate"].append(float(pkt_rate))
        res["n_dd_taps_mean"].append(float(mean_taps))
        print(f"[{name:>11}] SNR={snr:5.1f}dB  BER={ber:.3e}  "
              f"pkt_ok={pkt_rate*100:5.1f}%  taps={mean_taps:4.1f}  "
              f"(n_trials={n_trials})")
    return res


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(all_res: Dict[str, dict], out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    colors = {"awgn": "tab:blue", "static_mp": "tab:orange", "doubly_sel": "tab:red"}
    markers = {"awgn": "o", "static_mp": "s", "doubly_sel": "^"}
    for name, r in all_res.items():
        ax1.semilogy(r["snr_db"], np.maximum(r["ber"], 1e-6),
                     color=colors.get(name, "gray"),
                     marker=markers.get(name, "x"), label=name, lw=1.8)
        ax2.plot(r["snr_db"], np.array(r["pkt_ok_rate"]) * 100,
                 color=colors.get(name, "gray"),
                 marker=markers.get(name, "x"), label=name, lw=1.8)
    for ax in (ax1, ax2):
        ax.grid(True, ls=":", alpha=0.5)
        ax.set_xlabel("SNR (dB)")
        ax.legend()
    ax1.set_ylabel("BER")
    ax1.set_title("OTFS loopback — bit error rate")
    ax2.set_ylabel("Packet decode rate (%)")
    ax2.set_title("OTFS loopback — packet success")
    ax2.set_ylim(-2, 102)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser("OTFS software loopback sanity test")
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
    ap.add_argument("--payload_bytes", type=int, default=64)
    ap.add_argument("--snr_db", type=str, default="0,5,10,15,20,25,30")
    ap.add_argument("--n_trials", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_json", default="rf_stream/otfs_loopback_results.json")
    ap.add_argument("--out_plot", default="rf_stream/otfs_loopback_ber.png")
    ap.add_argument("--scenarios", default="awgn,static_mp,doubly_sel")
    args = ap.parse_args()

    cfg = OTFSConfig(
        M=args.otfs_M, N=args.otfs_N,
        l_max=args.otfs_l_max, k_max=args.otfs_k_max,
        pilot_doppler=args.otfs_pilot_doppler,
        pilot_boost_db=args.otfs_pilot_boost_db,
    )
    snr_list = [float(x) for x in args.snr_db.split(",")]

    print(f"OTFS loopback config:")
    print(f"  modulation={args.modulation} repeat={args.repeat}")
    print(f"  M={cfg.M} N={cfg.N} l_max={cfg.l_max} k_max={cfg.k_max}")
    print(f"  num_data_cells={num_data_cells(cfg)}  bits/block={num_data_cells(cfg)*MOD_BPS[args.modulation]}")
    print(f"  payload_bytes={args.payload_bytes}  trials per SNR={args.n_trials}")

    scenarios = {}
    if "awgn" in args.scenarios:
        scenarios["awgn"] = None
    if "static_mp" in args.scenarios:
        scenarios["static_mp"] = lambda x, rng: apply_static_multipath(x, rng=rng)
    if "doubly_sel" in args.scenarios:
        scenarios["doubly_sel"] = lambda x, rng: apply_doubly_selective(x, fs=FS, rng=rng)

    all_res = {}
    for name, fn in scenarios.items():
        print(f"\n--- scenario: {name} ---")
        all_res[name] = run_scenario(
            name, snr_list, args.n_trials, args.payload_bytes,
            cfg, args.modulation, args.repeat,
            args.stf_repeats, args.ltf_symbols, fn, seed=args.seed,
        )

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as fh:
        json.dump({
            "config": {
                "modulation": args.modulation, "repeat": args.repeat,
                "otfs": {"M": cfg.M, "N": cfg.N, "l_max": cfg.l_max, "k_max": cfg.k_max,
                          "pilot_boost_db": cfg.pilot_boost_db},
                "payload_bytes": args.payload_bytes, "n_trials": args.n_trials,
            },
            "scenarios": all_res,
        }, fh, indent=2)
    print(f"\n[json] wrote {args.out_json}")
    plot_results(all_res, args.out_plot)


if __name__ == "__main__":
    main()
