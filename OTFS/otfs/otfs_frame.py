"""
otfs_frame.py — Frame builder for OTFS over our Pluto pair.

Frame layout (identical preamble to rf_stream_tx_step6phy.py for compatibility
with the existing AutoDataPHY synchronisation pipeline):

    | gap_long | tone | gap_short | STF | LTF | OTFS payload | gap_long |

The OTFS payload is one (or more) DD blocks of length M*N samples produced
by `isfft_dd_to_time`.  We do NOT insert a CP between Doppler symbols (the
OTFS literature relies on the cyclic structure within the DD block).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import zlib

import numpy as np

from .otfs_core import (
    OTFSConfig,
    bits_from_bytes,
    bits_to_symbols,
    build_dd_grid,
    isfft_dd_to_time,
    make_constellation,
    num_data_cells,
    scramble_bits,
    MOD_BPS,
)


# ─────────────────────────────────────────────────────────────────────────────
# Preamble (Schmidl-Cox STF + LTF) — identical to step6 TX so existing RX
# synchronisers work unchanged.
# ─────────────────────────────────────────────────────────────────────────────

N_FFT = 64
N_CP  = 16


def _sc_to_bin(k: int) -> int:
    return (k + N_FFT) % N_FFT


def create_schmidl_cox_stf(num_repeats: int = 6) -> np.ndarray:
    rng = np.random.default_rng(42)
    X = np.zeros(N_FFT, dtype=np.complex64)
    even_subs = np.array([k for k in range(-26, 27, 2) if k != 0], dtype=int)
    bpsk = rng.choice([-1.0, 1.0], size=len(even_subs)).astype(np.float32)
    for i, k in enumerate(even_subs):
        X[_sc_to_bin(int(k))] = bpsk[i]
    x = np.fft.ifft(np.fft.ifftshift(X)) * np.sqrt(N_FFT)
    stf = np.tile(x.astype(np.complex64), num_repeats)
    return np.concatenate([stf[-N_CP:], stf]).astype(np.complex64)


def create_ltf(num_symbols: int = 4) -> np.ndarray:
    X = np.zeros(N_FFT, dtype=np.complex64)
    used = np.array([k for k in range(-26, 27) if k != 0], dtype=int)
    for i, k in enumerate(used):
        X[_sc_to_bin(int(k))] = 1.0 if (i % 2 == 0) else -1.0
    x = np.fft.ifft(np.fft.ifftshift(X)) * np.sqrt(N_FFT)
    ltf_sym = np.concatenate([x[-N_CP:], x]).astype(np.complex64)
    return np.tile(ltf_sym, num_symbols).astype(np.complex64)


# ─────────────────────────────────────────────────────────────────────────────
# Packet framing (identical to step6 — MAGIC || seq || total || len || payload
#                                       || CRC32, all little-endian)
# ─────────────────────────────────────────────────────────────────────────────

MAGIC = b"AIS1"


def build_packet_bytes(seq: int, total: int, payload: bytes) -> bytes:
    hdr = (MAGIC
           + int(seq).to_bytes(2, "little")
           + int(total).to_bytes(2, "little")
           + int(len(payload)).to_bytes(2, "little"))
    body = hdr + payload
    crc = zlib.crc32(body) & 0xFFFFFFFF
    return body + int(crc).to_bytes(4, "little")


# ─────────────────────────────────────────────────────────────────────────────
# OTFS payload builder
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OTFSPayloadInfo:
    """Bookkeeping returned alongside the time-domain signal — needed by the
    RX (and by tx_truth.jsonl for offline BER checks)."""
    num_blocks: int
    pilot_amp: float
    data_cells_per_block: int
    bits_per_block: int
    bits_total: int
    bits_padded: int
    payload_crc32: int
    frame_crc32: int


def bytes_to_otfs_samples(
    frame_bytes: bytes,
    repeat: int,
    stf: np.ndarray,
    ltf: np.ndarray,
    cfg: OTFSConfig,
    modulation: str,
    fs: float,
    tone_duration_ms: float,
    tone_freq_hz: float,
    gap_short: int,
    gap_long: int,
    tx_scale: float,
    ldpc=None,                          # OTFSLDPC instance or None
) -> Tuple[np.ndarray, OTFSPayloadInfo]:
    """Build one full TX waveform carrying `frame_bytes` over OTFS payload.

    The frame may need multiple OTFS DD-blocks; we concatenate them in time.
    All DD blocks share the same embedded-pilot scheme so the RX can
    estimate the channel from each block independently.

    Pipeline:
        bits = from_bytes(frame_bytes)
        bits = scramble(bits)
        if ldpc: bits = ldpc.encode(bits)        ← FEC outer code
        if repeat>1: bits = np.repeat(bits, repeat)
        symbols ← bits_to_symbols(...)
        DD grid ← build_dd_grid(symbols, ...)
        time-domain ← ISFFT + CP
    """
    const_tbl = make_constellation(modulation)
    bps       = MOD_BPS[modulation.lower()]

    # 1. Scramble bits + (optional) LDPC encode + diversity repeat
    bits_raw = bits_from_bytes(frame_bytes)
    bits     = scramble_bits(bits_raw)
    if ldpc is not None:
        bits = ldpc.encode(bits)
    if repeat > 1:
        bits = np.repeat(bits, repeat)

    # 2. Compute how many DD blocks we need
    n_data_cells   = num_data_cells(cfg)
    bits_per_block = n_data_cells * bps
    num_blocks     = int(np.ceil(len(bits) / bits_per_block))
    bits_padded    = num_blocks * bits_per_block
    if bits.size < bits_padded:
        bits = np.pad(bits, (0, bits_padded - bits.size))

    # 3. Build each DD block, ISFFT to time-domain, prepend per-block CP.
    #    The CP serves two purposes here:
    #      (a) make linear channel convolution act as cyclic over the
    #          MN-sample block (CP-OTFS, Raviteja §III-B); l_max samples
    #          would be enough for the channel alone, but
    #      (b) absorb residual timing-sync error from the STF/LTF stage.
    #    Phase-1 hardware showed that sample-level mis-alignment after
    #    coarse sync is the dominant failure mode on cable.  We size the
    #    CP to N_FFT (= 64) samples so any sync error within ±N_FFT is
    #    transparent to the DD decoder.
    payload_chunks: List[np.ndarray] = []
    pilot_amp = 0.0
    cp_len    = max(int(cfg.l_max), N_FFT)
    for b in range(num_blocks):
        block_bits = bits[b * bits_per_block: (b + 1) * bits_per_block]
        block_syms = bits_to_symbols(block_bits, const_tbl, bps)
        dd_grid, p_amp = build_dd_grid(block_syms, cfg)
        pilot_amp = p_amp
        block_t   = isfft_dd_to_time(dd_grid)
        if cp_len > 0:
            block_t = np.concatenate([block_t[-cp_len:], block_t]).astype(np.complex64)
        payload_chunks.append(block_t)
    payload = np.concatenate(payload_chunks).astype(np.complex64)

    # 4. Optional CFO tone (kept zero by default, same as step6)
    tone_samps = int(tone_duration_ms * fs / 1000.0)
    if tone_samps > 0:
        t = np.arange(tone_samps, dtype=np.float32) / float(fs)
        tone = np.exp(2j * np.pi * float(tone_freq_hz) * t).astype(np.complex64) * 0.5
    else:
        tone = np.zeros(0, dtype=np.complex64)

    gS = np.zeros(int(gap_short), dtype=np.complex64)
    gL = np.zeros(int(gap_long),  dtype=np.complex64)

    sig = np.concatenate([gL, tone, gS, stf, ltf, payload, gL]).astype(np.complex64)
    sig = sig / (np.max(np.abs(sig)) + 1e-9) * float(tx_scale)

    info = OTFSPayloadInfo(
        num_blocks=num_blocks,
        pilot_amp=float(pilot_amp),
        data_cells_per_block=int(n_data_cells),
        bits_per_block=int(bits_per_block),
        bits_total=int(bits_raw.size),
        bits_padded=int(bits_padded),
        payload_crc32=0,   # caller fills in if needed
        frame_crc32=zlib.crc32(frame_bytes) & 0xFFFFFFFF,
    )
    return sig, info


# ─────────────────────────────────────────────────────────────────────────────
# Sample-budget helper — useful when sizing fixed_len on the TX
# ─────────────────────────────────────────────────────────────────────────────

def expected_frame_length(
    frame_bytes: int,
    repeat: int,
    cfg: OTFSConfig,
    modulation: str,
    stf_repeats: int,
    ltf_symbols: int,
    gap_short: int,
    gap_long: int,
    tone_samples: int = 0,
    ldpc=None,                                  # OTFSLDPC instance, optional
) -> int:
    """Return the time-domain length (samples) of one assembled OTFS frame."""
    bps            = MOD_BPS[modulation.lower()]
    bits_per_block = num_data_cells(cfg) * bps
    raw_bits       = frame_bytes * 8
    coded_bits     = ldpc.expected_coded_len(raw_bits) if ldpc is not None else raw_bits
    bits           = coded_bits * repeat
    num_blocks     = int(np.ceil(bits / bits_per_block))
    cp_len         = max(int(cfg.l_max), N_FFT)
    payload_len    = num_blocks * (cp_len + cfg.M * cfg.N)
    stf_len        = (stf_repeats + 1) * (N_FFT // 2) * 2  # rough; CP included
    # exact STF length: num_repeats * N_FFT + N_CP — matches create_schmidl_cox_stf
    stf_len = stf_repeats * N_FFT + N_CP
    ltf_len = ltf_symbols * (N_FFT + N_CP)
    return int(gap_long + tone_samples + gap_short + stf_len + ltf_len + payload_len + gap_long)
