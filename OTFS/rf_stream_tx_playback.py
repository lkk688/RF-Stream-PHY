#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rf_stream_tx_playback.py — Replay a pre-built complex64 IQ waveform on PlutoSDR.

Used to run experiments that need code unavailable on Jetson (e.g. PyTorch
LDPC encoder).  Workflow:

    1. On x86:  `python rf_stream/rf_stream_tx_otfs.py --build_only ...`
                builds the full multi-payload cyclic IQ buffer (with LDPC if
                requested), saves to disk.
    2. `scp` the IQ file to Jetson.
    3. On Jetson:  `python rf_stream/rf_stream_tx_playback.py --iq_file ... --cyclic`
                loads the IQ buffer and tx_cyclic streams forever until SIGINT.

The IQ file format is a raw little-endian complex64 array (no header) — just
the time-domain samples already scaled and IQ-conjugated for Pluto.
"""

import argparse
import numpy as np
import time


def main():
    ap = argparse.ArgumentParser("Pluto TX waveform playback")
    ap.add_argument("--uri",     required=True)
    ap.add_argument("--fc",      type=float, required=True)
    ap.add_argument("--fs",      type=float, required=True)
    ap.add_argument("--tx_gain", type=float, default=0.0)
    ap.add_argument("--tx_bw",   type=float, default=0.0)
    ap.add_argument("--iq_file", required=True,
                    help="Path to complex64 raw IQ buffer (no header).")
    ap.add_argument("--cyclic",  action="store_true",
                    help="Use Pluto cyclic buffer (loops forever).")
    ap.add_argument("--no_iq_swap", action="store_true",
                    help="Skip the np.conj() that compensates Pluto's TX IQ inversion. "
                         "Default: apply (matches rf_stream_tx_step6phy and tx_otfs).")
    ap.add_argument("--scale", type=float, default=4096.0,
                    help="Multiplier from normalised float → DAC int range.")
    args = ap.parse_args()

    import adi

    buf = np.fromfile(args.iq_file, dtype=np.complex64)
    print(f"[playback] loaded {buf.size} samples from {args.iq_file} "
          f"({buf.size / args.fs * 1e3:.2f} ms at fs={args.fs/1e6:.1f}MHz)")

    sdr = adi.Pluto(uri=args.uri)

    # ── DDS reset (Phase-1 hardware fix) ─────────────────────────────────
    try:
        sdr.dds_scales = (0.0,) * 8
    except Exception:
        pass

    sdr.sample_rate          = int(args.fs)
    sdr.tx_lo                = int(args.fc)
    sdr.tx_rf_bandwidth      = int(args.tx_bw if args.tx_bw > 0 else args.fs * 1.2)
    sdr.tx_enabled_channels  = [0]
    try:
        sdr.tx_destroy_buffer()
    except Exception:
        pass
    sdr.tx_cyclic_buffer = False

    # Deterministic gain apply: AD9361's tx_hardwaregain register sometimes
    # silently fails to update after a rapid TX-mode restart (DMA still
    # cached the previous setting).  Re-write gain 3x with a zero-buffer
    # DMA prime between each write, so by the third pass the register is
    # guaranteed to match.
    for _ in range(3):
        sdr.tx_hardwaregain_chan0 = float(args.tx_gain)
        try:
            sdr.tx(np.zeros(1024, dtype=np.complex64))
            sdr.tx_destroy_buffer()
        except Exception:
            pass
        time.sleep(0.05)

    applied = float(sdr.tx_hardwaregain_chan0)
    print(f"[playback] tx_gain requested={args.tx_gain} dB  applied={applied:.2f} dB")

    sdr.tx_cyclic_buffer = bool(args.cyclic)

    # Apply Pluto IQ-swap compensation by default
    if args.no_iq_swap:
        tx_data = (buf * args.scale).astype(np.complex64)
    else:
        tx_data = (np.conj(buf) * args.scale).astype(np.complex64)

    print(f"[playback] sending {tx_data.size} samples  cyclic={args.cyclic}  "
          f"gain={args.tx_gain}dB  fc={args.fc/1e6:.3f}MHz")
    sdr.tx(tx_data)

    if args.cyclic:
        print(f"[playback] cyclic buffer armed ({tx_data.size / args.fs * 1e3:.2f} ms loop). "
              "Streaming until SIGINT.")
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            try: sdr.tx_destroy_buffer()
            except Exception: pass
    else:
        # Non-cyclic: just one shot
        print("[playback] one-shot done.")


if __name__ == "__main__":
    main()
