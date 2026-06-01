#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inject_doppler_to_iq.py — Apply a synthetic 2-tap doubly-selective channel
                          to a pre-built IQ file, for OTFS vs OFDM Doppler
                          sweep without physical mobility.

Channel model (per-sample):

    y[n] = x[n]  +  alpha * x[n - tau]  *  exp(j * 2*pi * fd * n / fs)

The reflector tap carries a controllable Doppler shift `fd` (Hz).  This is
what we'd see from a single moving scatterer behind a static direct path —
the classic two-path Doppler-multipath model OTFS is designed to handle.

For cyclic playback on PlutoSDR, the exponential phasor must repeat exactly
across the buffer boundary, otherwise every loop introduces a phase
discontinuity that looks like noise.  Phase continuity requires
`fd * L / fs` to be an integer, so this script quantises `fd` to the nearest
multiple of `fs / L`.  With L=569344 and fs=3e6 the step is ~5.27 Hz —
fine for the Hz-to-kHz Doppler range we care about.

Usage:
    python3 rf_stream/inject_doppler_to_iq.py \\
        --in_iq /tmp/otfs_ldpc_seed42.iq \\
        --out_iq /tmp/otfs_ldpc_seed42_fd1000.iq \\
        --fs 3e6 --doppler_hz 1000 --tap_delay 3 --tap_db -3
"""

import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser("Inject 2-tap Doppler channel into a raw IQ file")
    ap.add_argument("--in_iq",       required=True)
    ap.add_argument("--out_iq",      required=True)
    ap.add_argument("--fs",          type=float, required=True,
                    help="Sample rate (Hz) — must match the playback fs.")
    ap.add_argument("--doppler_hz",  type=float, required=True,
                    help="Doppler frequency for the reflector tap (Hz). "
                         "Quantised to nearest multiple of fs/L for cyclic-buffer "
                         "phase continuity.")
    ap.add_argument("--tap_delay",   type=int, default=3,
                    help="Reflector delay in samples (cyclic shift). Default 3 "
                         "(~1 us at fs=3 MHz, well within OFDM CP=16).")
    ap.add_argument("--tap_db",      type=float, default=-3.0,
                    help="Reflector amplitude in dB relative to direct path. "
                         "Default -3 dB.  Use < -60 to bypass channel injection "
                         "(passthrough).")
    ap.add_argument("--rescale",     action="store_true",
                    help="Rescale output so peak |y| matches peak |x| (preserves "
                         "the headroom assumed by the Pluto IQ-swap + 4096x DAC "
                         "scaling).  Default: on (recommended).")
    ap.add_argument("--no_rescale",  dest="rescale", action="store_false")
    ap.set_defaults(rescale=True)
    args = ap.parse_args()

    x = np.fromfile(args.in_iq, dtype=np.complex64)
    L = x.size
    fs = float(args.fs)
    print(f"[doppler] loaded {L} samples ({L/fs*1e3:.2f} ms at fs={fs/1e6:.1f} MHz)")

    alpha = 10.0 ** (args.tap_db / 20.0) if args.tap_db > -60 else 0.0
    if alpha == 0.0:
        print("[doppler] tap_db < -60 dB -> bypass: writing input unchanged")
        x.astype(np.complex64).tofile(args.out_iq)
        return

    # Quantise Doppler so the phasor is phase-continuous across the cyclic loop
    step = fs / L
    n_cycles = int(round(args.doppler_hz / step))
    fd_q = n_cycles * step
    if abs(fd_q - args.doppler_hz) > 0.005:
        print(f"[doppler] fd quantised: requested {args.doppler_hz:.3f} Hz -> "
              f"{fd_q:.4f} Hz  (step fs/L = {step:.4f} Hz, n_cycles = {n_cycles})")
    else:
        print(f"[doppler] fd = {fd_q:.4f} Hz  ({n_cycles} cycles per buffer)")

    n     = np.arange(L, dtype=np.float64)
    phasor = np.exp(1j * 2.0 * np.pi * fd_q * n / fs).astype(np.complex64)
    x_del  = np.roll(x, args.tap_delay)
    y      = (x + (alpha * phasor) * x_del).astype(np.complex64)

    px_in  = float(np.sqrt(np.mean(np.abs(x) ** 2)))
    px_out = float(np.sqrt(np.mean(np.abs(y) ** 2)))
    peak_in  = float(np.max(np.abs(x)))
    peak_out = float(np.max(np.abs(y)))
    print(f"[doppler] before: peak={peak_in:.4f}  rms={px_in:.4f}")
    print(f"[doppler] after : peak={peak_out:.4f}  rms={px_out:.4f}")

    if args.rescale and peak_out > 0:
        scale = peak_in / peak_out
        y = (y * scale).astype(np.complex64)
        print(f"[doppler] rescaled by {scale:.4f} so peak matches input "
              f"(prevents DAC clipping when alpha boosts peaks)")

    y.tofile(args.out_iq)
    print(f"[doppler] wrote {y.size} samples to {args.out_iq}")
    print(f"  channel: y[n] = x[n] + ({alpha:.3f}) * x[n-{args.tap_delay}] "
          f"* exp(j*2*pi*{fd_q:.2f}*n/fs)")


if __name__ == "__main__":
    main()
