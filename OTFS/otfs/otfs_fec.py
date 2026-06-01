"""
otfs_fec.py — LDPC forward-error-correction wrapper for the OTFS chain.

Reuses the 5G NR LDPC encoder/decoder from `sdradi/sdr_ldpc.py` (PyTorch).
The interface here is bit-array-in, bit-array-out (no batching exposed to
callers), so it slots cleanly into our existing
    bits_raw → scramble → [FEC encode] → bits_to_symbols
    symbols → demap → [FEC decode] → descramble → bytes
pipeline.

Supports both **hard-decision** (input = demapped bit array) and **soft**
(input = per-bit LLRs from the constellation demap) decoding modes.  Soft is
~ 1.5–3 dB stronger than hard on AWGN, ~ much stronger on real OTA channels
where the LMMSE output has graded reliability per cell.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch

# Lazy import — only loaded when LDPC is actually requested
_LDPC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "..", "sdradi")
_LDPC_PATH = os.path.normpath(_LDPC_PATH)
if _LDPC_PATH not in sys.path:
    sys.path.insert(0, _LDPC_PATH)


@dataclass
class FECConfig:
    """LDPC configuration.

    The 5G-NR LDPC encoder only supports a discrete set of (k, n) pairs
    determined by the base-graph + lifting factor.  Sensible choices for
    our OTFS frames (894 raw bits / OTFS block, 64-byte payloads):

      k=256, n=512   — rate 1/2, 1 OTFS block holds ~1.7 LDPC blocks
      k=512, n=1024  — rate 1/2, 2 OTFS blocks hold ~1.75 LDPC blocks
      k=384, n=768   — rate 1/2 (when supported by BG2 with Z values)
    """
    enabled:   bool = False
    k:         int = 256          # info bits per LDPC block
    n:         int = 512          # coded bits per LDPC block (n/k = 1/rate)
    soft_demap: bool = True       # True → LLR input, False → hard ±L
    llr_mag:   float = 10.0       # used in hard-decision mode
    device:    str = "cpu"        # "cuda" if a GPU is available

    @property
    def rate(self) -> float:
        return self.k / self.n


class OTFSLDPC:
    """Lightweight wrapper around `sdr_ldpc.LDPC5GEncoder/Decoder`.

    Methods
    -------
    encode(bits) -> coded_bits
        bits: 1-D uint8/int array of 0/1 values, arbitrary length.
              Padded to a multiple of `k`, then encoded block-by-block.
        Returns 1-D uint8 of length ⌈len/k⌉ · n.

    decode_hard(bits) -> info_bits
        bits: 1-D uint8/int array, length must be multiple of `n`.
              Each block converted to ±llr_mag and decoded.

    decode_soft(llr) -> info_bits
        llr: 1-D float array of LLRs (positive → bit=0, negative → bit=1),
             length must be multiple of `n`.
    """

    def __init__(self, cfg: FECConfig):
        if not cfg.enabled:
            raise ValueError("OTFSLDPC instantiated with FECConfig.enabled=False")

        # Defer the heavy import to construction time so unrelated code
        # paths (loopback, sync test, etc.) don't pay for it.
        from sdr_ldpc import LDPC5GEncoder, LDPC5GDecoder

        self.cfg = cfg
        self.encoder = LDPC5GEncoder(cfg.k, cfg.n, num_bits_per_symbol=2,
                                       device=cfg.device)
        self.decoder = LDPC5GDecoder(self.encoder, device=cfg.device)

    # ── Encode ────────────────────────────────────────────────────────────
    def encode(self, bits: np.ndarray) -> np.ndarray:
        bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
        k    = self.cfg.k
        pad  = (-bits.size) % k
        if pad:
            bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
        n_blocks = bits.size // k
        u   = torch.from_numpy(bits.reshape(n_blocks, k).astype(np.float32)).to(self.cfg.device)
        c   = self.encoder.encode(u)             # (B, n) float
        out = c.detach().cpu().numpy().astype(np.uint8).reshape(-1)
        return out

    def expected_coded_len(self, raw_bits: int) -> int:
        """Length of the encoded bit stream for `raw_bits` info bits."""
        k = self.cfg.k; n = self.cfg.n
        n_blocks = (raw_bits + k - 1) // k
        return n_blocks * n

    # ── Decode (hard) ─────────────────────────────────────────────────────
    def decode_hard(self, bits: np.ndarray) -> np.ndarray:
        bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
        return self._decode_llr(self._hard_to_llr(bits))

    def _hard_to_llr(self, bits: np.ndarray) -> np.ndarray:
        # Standard convention: LLR = log P(0)/P(1)
        # bit=0 → +L; bit=1 → -L
        llr = np.where(bits == 0, self.cfg.llr_mag, -self.cfg.llr_mag)
        return llr.astype(np.float32)

    # ── Decode (soft) ─────────────────────────────────────────────────────
    def decode_soft(self, llr: np.ndarray) -> np.ndarray:
        llr = np.asarray(llr, dtype=np.float32).reshape(-1)
        return self._decode_llr(llr)

    def _decode_llr(self, llr: np.ndarray) -> np.ndarray:
        n = self.cfg.n
        # Truncate to a multiple of n — any trailing partial block is dropped
        n_blocks = llr.size // n
        if n_blocks == 0:
            return np.zeros(0, dtype=np.uint8)
        llr = llr[: n_blocks * n].reshape(n_blocks, n)
        L = torch.from_numpy(llr).to(self.cfg.device)
        u = self.decoder.decode(L)                # (B, k) float, hard bits
        out = u.detach().cpu().numpy().astype(np.uint8).reshape(-1)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# QPSK soft demap → per-bit LLRs (for the soft-decision LDPC path)
# ─────────────────────────────────────────────────────────────────────────────

def qpsk_soft_demap(
    x_hat: np.ndarray,
    noise_var: float = 1.0,
    const_table: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute per-bit log-likelihood-ratios from QPSK constellation estimates.

    Step6 QPSK Gray table:
       idx 0: (b0,b1) = (0,0) → (+1+j)/√2
       idx 1: (b0,b1) = (0,1) → (-1+j)/√2
       idx 2: (b0,b1) = (1,0) → (-1-j)/√2
       idx 3: (b0,b1) = (1,1) → (+1-j)/√2

    LLR formula (max-log approximation for QPSK):
        LLR(b_k) = (min_{s: b_k=1} |y-s|² − min_{s: b_k=0} |y-s|²) / σ²

    Returns interleaved bit-LLRs of length 2 * len(x_hat):
        [LLR(b0_cell0), LLR(b1_cell0), LLR(b0_cell1), LLR(b1_cell1), ...]
    """
    if const_table is None:
        const_table = np.array([1+1j, -1+1j, -1-1j, 1-1j], dtype=np.complex64) / np.sqrt(2.0)
    # Distances |y - s|² for all 4 symbols
    diff = x_hat[:, None] - const_table[None, :]      # (N, 4)
    d2   = (diff.real ** 2 + diff.imag ** 2).astype(np.float32)

    # bit-table: which symbols carry b0=0/1 and b1=0/1
    # idx -> (b0, b1) :  0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
    b0_0 = np.array([0, 1])      # symbols where b0=0
    b0_1 = np.array([2, 3])
    b1_0 = np.array([0, 2])
    b1_1 = np.array([1, 3])

    # Max-log LLR (positive → bit=0, negative → bit=1)
    llr_b0 = (d2[:, b0_1].min(axis=1) - d2[:, b0_0].min(axis=1)) / noise_var
    llr_b1 = (d2[:, b1_1].min(axis=1) - d2[:, b1_0].min(axis=1)) / noise_var
    # Interleave: (N,) and (N,) → (2N,)  [b0, b1, b0, b1, ...]
    out = np.empty(x_hat.size * 2, dtype=np.float32)
    out[0::2] = llr_b0
    out[1::2] = llr_b1
    return out
