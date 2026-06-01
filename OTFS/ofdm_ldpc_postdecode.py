#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ofdm_ldpc_postdecode.py — Post-decode step10 OFDM RX captures with an LDPC
                          outer code.

The step10 RX (`rf_stream_rx_step10phy.py`) writes one NPZ per capture when
run with `--save_npz`.  Each NPZ contains `Yeq_data` — the equalized,
pilot-corrected OFDM data symbols (shape: max_ofdm_syms × 48 data subcarriers,
complex64) — already the optimal LMMSE input for any downstream decoder.

This script reads those NPZs, computes per-cell soft LLRs via QPSK Gray-code
demap, runs LDPC decode (must match the LDPC config the TX used), descrambles
and parses the packet, and reports both the raw BER (no FEC) and the post-LDPC
packet-decode rate.

Usage:
    python3 rf_stream/ofdm_ldpc_postdecode.py \
        --rx_npz_glob 'rf_stream_rx_runs_*/run_*/cap_*.npz' \
        --tx_truth /tmp/ofdm_ldpc_seed42.iq.tx_truth.jsonl \
        --ref_seed 42 --ref_len 4096 --chunk_bytes 64 \
        --ldpc_k 256 --ldpc_n 512 \
        --num_ofdm_syms 16
"""

import argparse
import glob
import json
import os
import sys
import zlib
from typing import List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from otfs.otfs_core import scramble_bits, bits_to_bytes, bits_from_bytes
from otfs.otfs_fec import OTFSLDPC, FECConfig, qpsk_soft_demap


MAGIC = b"AIS1"

# Step6 OFDM constellation (QPSK Gray)
QPSK = np.array([1+1j, -1+1j, -1-1j, 1-1j], dtype=np.complex64) / np.sqrt(2.0)


def parse_packet_bytes(bb: bytes) -> Tuple[bool, str, int, bytes]:
    if len(bb) < 14:                       return False, "too_short", -1, b""
    if bb[:4] != MAGIC:                    return False, "bad_magic", -1, b""
    seq      = int.from_bytes(bb[4:6],  "little")
    total    = int.from_bytes(bb[6:8],  "little")
    plen     = int.from_bytes(bb[8:10], "little")
    expected = 10 + plen + 4
    if len(bb) < expected:                 return False, "short_payload", seq, b""
    body     = bb[: 10 + plen]
    crc_f    = int.from_bytes(bb[10 + plen: 14 + plen], "little")
    if (zlib.crc32(body) & 0xFFFFFFFF) != crc_f: return False, "crc_fail", seq, b""
    return True, "ok", seq, bb[10: 10 + plen]


def build_packet_bytes(seq: int, total: int, payload: bytes) -> bytes:
    hdr = (MAGIC
           + int(seq).to_bytes(2, "little")
           + int(total).to_bytes(2, "little")
           + int(len(payload)).to_bytes(2, "little"))
    body = hdr + payload
    crc  = zlib.crc32(body) & 0xFFFFFFFF
    return body + int(crc).to_bytes(4, "little")


def reconstruct_candidate_bits(seed: int, ref_len: int, chunk_bytes: int,
                                 ldpc: OTFSLDPC, num_ofdm_syms: int) -> List[Tuple[int, np.ndarray]]:
    """Return [(seq, on-air coded bits), ...] matching what TX put on the wire."""
    rng    = np.random.default_rng(seed)
    data   = rng.integers(0, 256, size=ref_len, dtype=np.uint8).tobytes()
    chunks = [data[i: i + chunk_bytes] for i in range(0, len(data), chunk_bytes)]
    total  = len(chunks)
    n_data = 48          # data subcarriers per OFDM symbol (step6)
    bps    = 2
    bits_per_sym = n_data * bps
    expected_size = num_ofdm_syms * bits_per_sym

    out = []
    for seq, ch in enumerate(chunks):
        fb       = build_packet_bytes(seq, total, ch)
        bits_raw = bits_from_bytes(fb)
        bits     = scramble_bits(bits_raw)
        bits     = ldpc.encode(bits)
        if bits.size < expected_size:
            bits = np.pad(bits, (0, expected_size - bits.size))
        out.append((seq, bits[:expected_size].astype(np.uint8)))
    return out


def main():
    ap = argparse.ArgumentParser("OFDM + LDPC post-decoder")
    ap.add_argument("--rx_npz_glob", required=True)
    ap.add_argument("--tx_truth",    default="",
                    help="Optional — kept for parity with extract scripts; not strictly needed.")
    ap.add_argument("--ref_seed",    type=int, default=42)
    ap.add_argument("--ref_len",     type=int, required=True)
    ap.add_argument("--chunk_bytes", type=int, default=64)
    ap.add_argument("--ldpc_k",      type=int, default=256)
    ap.add_argument("--ldpc_n",      type=int, default=512)
    ap.add_argument("--num_ofdm_syms", type=int, required=True,
                    help="Number of OFDM data symbols per packet (must match TX).")
    ap.add_argument("--llr_clip",    type=float, default=20.0)
    ap.add_argument("--payload_bytes", type=int, default=64)
    args = ap.parse_args()

    ldpc = OTFSLDPC(FECConfig(enabled=True, k=args.ldpc_k, n=args.ldpc_n))
    cands = reconstruct_candidate_bits(args.ref_seed, args.ref_len, args.chunk_bytes,
                                         ldpc, args.num_ofdm_syms)
    print(f"[OFDM-LDPC] {len(cands)} candidate truth payloads built")
    print(f"[OFDM-LDPC] each candidate on-air = {cands[0][1].size} bits = {args.num_ofdm_syms} OFDM symbols")

    files = sorted(glob.glob(args.rx_npz_glob))
    if not files:
        print(f"no NPZs match {args.rx_npz_glob}", file=sys.stderr); sys.exit(2)
    print(f"[OFDM-LDPC] processing {len(files)} captures")

    n_attempted = 0
    n_pkt_ok    = 0
    n_raw_pkt_ok = 0
    ber_post_ldpc = []
    ber_raw       = []
    for fi, fp in enumerate(files):
        try:
            with np.load(fp, allow_pickle=True) as d:
                if "Yeq_data" not in d.files:
                    continue
                Yeq = np.asarray(d["Yeq_data"], dtype=np.complex64)
        except Exception:
            continue

        # Yeq_data shape: (max_ofdm_syms, 48 data subcarriers).  We need
        # the first `num_ofdm_syms` rows.  Captures that didn't probe that
        # many symbols are skipped.
        if Yeq.shape[0] < args.num_ofdm_syms:
            continue
        syms = Yeq[: args.num_ofdm_syms].flatten()    # (num_ofdm_syms * 48,) complex

        # Step6 OFDM cells already constellation-scaled around |s|=1/√2 after
        # the step10 MMSE equaliser.  Just clean per-block normalisation.
        scale = float(np.median(np.abs(syms)) / (1.0 / np.sqrt(2.0)) + 1e-12)
        syms_n = (syms / max(scale, 1e-9)).astype(np.complex64)

        # EVM-based noise variance for soft demap
        diff = syms_n[:, None] - QPSK[None, :]
        d2_min = np.min(diff.real ** 2 + diff.imag ** 2, axis=1)
        eff_s2 = float(max(0.02, np.mean(d2_min)))

        # Hard demap → raw bits (for the no-FEC baseline)
        idx = np.argmin(np.abs(syms_n[:, None] - QPSK[None, :]) ** 2, axis=1)
        hard_bits = np.empty(syms_n.size * 2, dtype=np.uint8)
        hard_bits[0::2] = (idx >> 1) & 1
        hard_bits[1::2] = idx & 1

        # Soft LLRs
        llr = qpsk_soft_demap(syms_n, noise_var=eff_s2, const_table=QPSK)
        llr = np.clip(llr, -args.llr_clip, args.llr_clip).astype(np.float32)
        # Length must be multiple of n=512 for the LDPC decoder; we get
        # 16 OFDM × 48 data × 2 bps = 1536 = exactly 3 LDPC blocks.

        n_attempted += 1
        # Pick the candidate with smallest Hamming distance to the hard demap —
        # we don't trust step10's MAGIC parsing because the LDPC layer changes
        # the bit positions of the header.
        dists = [int(np.sum(hard_bits[:c.size] != c)) for _, c in cands]
        seq_pred = int(np.argmin(dists))
        d_min    = dists[seq_pred]
        ber_raw.append(d_min / hard_bits.size)
        if d_min == 0:
            n_raw_pkt_ok += 1

        # Decode LDPC → 3 LDPC blocks × k=256 = 768 info bits
        bits_dec = ldpc.decode_soft(llr)
        # Descramble first (frame_bytes * 8) = 78*8 = 624 bits worth
        n_info = (args.payload_bytes + 14) * 8
        unscr  = scramble_bits(bits_dec[: n_info])
        rx_bytes = bits_to_bytes(unscr)

        ok, reason, seq, payload = parse_packet_bytes(rx_bytes)
        if ok and len(payload) == args.payload_bytes:
            n_pkt_ok += 1
            # BER after LDPC vs the expected info bits for predicted seq
            # (we approximate: predicted seq from raw hard-demap distance)
            expected_info = scramble_bits(bits_from_bytes(build_packet_bytes(
                seq, len(cands), payload)))
            ber_post_ldpc.append(0.0)
        else:
            # Compute BER vs the predicted seq's info
            fb_pred = build_packet_bytes(seq_pred, len(cands),
                                          # dummy payload — only header bits matter for BER alignment
                                          b"\x00" * args.payload_bytes)
            expected_info = scramble_bits(bits_from_bytes(fb_pred))
            n_cmp = min(unscr.size, expected_info.size)
            ber_post_ldpc.append(float(np.mean(unscr[:n_cmp] != expected_info[:n_cmp])))

        if (fi + 1) % 100 == 0 or (fi + 1) == len(files):
            print(f"  {fi+1:>5}/{len(files):>5}  attempts={n_attempted:>4}  "
                  f"raw_pkt_ok={n_raw_pkt_ok}  ldpc_pkt_ok={n_pkt_ok}")

    ber_raw = np.asarray(ber_raw)
    print(f"\n=== OFDM + LDPC POSTDECODE SUMMARY ===")
    print(f"  files processed       : {len(files)}")
    print(f"  attempted decodes     : {n_attempted}")
    print(f"  raw pkt ok (no FEC)   : {n_raw_pkt_ok} ({100*n_raw_pkt_ok/max(n_attempted,1):.1f}%)")
    print(f"  LDPC pkt ok           : {n_pkt_ok} ({100*n_pkt_ok/max(n_attempted,1):.1f}%)")
    if ber_raw.size:
        print(f"  raw BER  mean={ber_raw.mean():.4f}  median={np.median(ber_raw):.4f}  "
              f"p10={np.percentile(ber_raw,10):.4f}  p90={np.percentile(ber_raw,90):.4f}")


if __name__ == "__main__":
    main()
