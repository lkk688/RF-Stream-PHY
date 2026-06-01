"""
otfs_core.py — pure-numpy OTFS primitives for the Phase-1 hardware bring-up.

OTFS conventions used throughout this module
============================================

    DD grid shape :  X[l, k]   with  l ∈ [0, M-1]  (delay bin)
                                     k ∈ [0, N-1]  (Doppler bin)

    M = N_FFT (delay bins per OFDM symbol — matches our existing OFDM chain)
    N = number of OFDM-equivalent symbols per OTFS block (Doppler bins)

    ISFFT  : DD  -> TF       — IFFT along Doppler, FFT along delay
    Heisenberg: TF -> Time   — IFFT along delay (Tx)
    Wigner  : Time -> TF     — FFT along delay  (Rx)
    SFFT   : TF  -> DD       — FFT along Doppler, IFFT along delay

We use the bi-orthogonal rectangular pulse formulation without CP between
Doppler symbols (matching Raviteja et al. 2018 and most software OTFS
implementations).  This is what the previous AIradar code used in
`AIradar_comm_dataset_g1.py:602–638`, and those primitives are correct.

The contribution of this module over the AIradar version is:
 1.  Embedded-pilot DD frame builder (Raviteja 2019) with guard region.
 2.  Effective DD channel matrix construction from delay/Doppler tap list.
 3.  Real DD-LMMSE detector — the AIradar code did one-tap TF equalization
     which is essentially OFDM and discards OTFS's diversity gain.

All functions operate on plain numpy arrays (complex64) so they can be
called from both the synthetic loopback test and the live RX script.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OTFSConfig:
    """Geometry of the OTFS DD grid and the embedded-pilot scheme."""
    M: int = 64                # delay bins  (= OFDM N_FFT, for fair compare)
    N: int = 8                 # Doppler bins
    l_max: int = 6             # max channel delay spread (cells)
    k_max: int = 1             # max channel Doppler spread (cells)
    pilot_delay: int = 0       # pilot row in DD grid (l_p)
    pilot_doppler: int = 4     # pilot column in DD grid (k_p)  — defaults to N/2
    pilot_boost_db: float = 12.0  # pilot amplitude over data symbol RMS

    @property
    def grid_size(self) -> int:
        return self.M * self.N

    def __post_init__(self) -> None:
        if self.pilot_doppler >= self.N:
            self.pilot_doppler = self.N // 2
        if self.l_max >= self.M:
            raise ValueError("l_max must be < M")
        if self.k_max * 4 + 1 >= self.N:
            raise ValueError("guard region exceeds Doppler dimension")


# ─────────────────────────────────────────────────────────────────────────────
# DD grid layout: data vs guard vs pilot cells
# ─────────────────────────────────────────────────────────────────────────────

def dd_cell_mask(cfg: OTFSConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Return (data_mask, guard_mask) both shape (M, N), bool.

    Embedded-pilot scheme (Raviteja 2019, Fig. 2).  Guard cells protect the
    pilot-readout region from leakage by surrounding data symbols.  Because
    the effective DD channel is *cyclic* over (M, N), the guard must extend
    on BOTH sides of the pilot in delay (and in Doppler).

        delay rows kept clear of data:
            forward  : [l_p,           l_p + l_max]   (pilot spread direction)
            backward : [l_p - l_max,   l_p - 1]       (cyclic wrap of data)

        Doppler cols kept clear of data:
            [k_p - 2 k_max, k_p + 2 k_max]            (Doppler spread + frac. tail)
    """
    M, N = cfg.M, cfg.N
    guard = np.zeros((M, N), dtype=bool)

    # Build the delay-row set (with cyclic wrap) and Doppler-col set
    rows_set = set()
    for dl in range(-cfg.l_max, cfg.l_max + 1):
        rows_set.add((cfg.pilot_delay + dl) % M)
    rows = np.array(sorted(rows_set), dtype=int)

    cols_set = set()
    for dk in range(-2 * cfg.k_max, 2 * cfg.k_max + 1):
        cols_set.add((cfg.pilot_doppler + dk) % N)
    cols = np.array(sorted(cols_set), dtype=int)

    for r in rows:
        for c in cols:
            guard[r, c] = True

    data_mask = ~guard
    return data_mask, guard


def data_indices(cfg: OTFSConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Return (l_idx, k_idx) arrays of equal length giving every data cell.

    The order is reproducible (row-major over `data_mask`) so TX and RX
    agree on which bits map to which (l, k) cell.
    """
    data_mask, _ = dd_cell_mask(cfg)
    l_idx, k_idx = np.nonzero(data_mask)
    return l_idx.astype(np.int32), k_idx.astype(np.int32)


def num_data_cells(cfg: OTFSConfig) -> int:
    return int(data_indices(cfg)[0].size)


# ─────────────────────────────────────────────────────────────────────────────
# DD ↔ Time-domain transforms
# ─────────────────────────────────────────────────────────────────────────────

def isfft_dd_to_time(dd_grid: np.ndarray) -> np.ndarray:
    """OTFS Tx: DD grid (M, N) → time-domain signal of length M*N.

    Implementation matches AIradar_comm_dataset_g1.py:614–617.
        ISFFT : tf = FFT_M(dd, axis=delay) · IFFT_N(·, axis=Doppler)
        Heisenberg: time_grid = IFFT_M(tf, axis=delay)
        Serialize column-by-column (Fortran order) → length-MN signal.
    """
    M, N = dd_grid.shape
    tf = np.fft.fft(dd_grid, axis=0)        # FFT along delay  → TF along time
    tf = np.fft.ifft(tf, axis=1)            # IFFT along Doppler → TF along freq
    time_grid = np.fft.ifft(tf, axis=0)     # IFFT delay → time samples
    return time_grid.flatten(order="F").astype(np.complex64)


def sfft_time_to_dd(time_sig: np.ndarray, M: int, N: int) -> np.ndarray:
    """OTFS Rx: time-domain signal of length M*N → DD grid (M, N).

    Implementation matches AIradar_comm_dataset_g1.py:633–638.
        Wigner: tf = FFT_M(time_grid, axis=delay)
        SFFT  : dd = FFT_N(tf, axis=Doppler) · IFFT_M(·, axis=delay)
    """
    if time_sig.size != M * N:
        raise ValueError(f"time signal length {time_sig.size} ≠ M*N = {M*N}")
    time_grid = time_sig.reshape((M, N), order="F").astype(np.complex64)
    tf = np.fft.fft(time_grid, axis=0)      # Wigner along delay
    dd = np.fft.fft(tf, axis=1)             # SFFT along Doppler
    dd = np.fft.ifft(dd, axis=0)            # complete SFFT
    return dd.astype(np.complex64)


# ─────────────────────────────────────────────────────────────────────────────
# Embedded-pilot frame builder
# ─────────────────────────────────────────────────────────────────────────────

def build_dd_grid(
    data_symbols: np.ndarray,
    cfg: OTFSConfig,
) -> Tuple[np.ndarray, float]:
    """Place `data_symbols` into the data cells of an embedded-pilot DD grid.

    Parameters
    ----------
    data_symbols : 1-D complex array, length >= num_data_cells(cfg).
                   Treated as constellation samples (already mapped).
    cfg          : OTFSConfig.

    Returns
    -------
    dd_grid      : complex64 array, shape (M, N), with pilot+guard+data.
    pilot_amp    : pilot amplitude actually written into the grid (linear).
    """
    M, N = cfg.M, cfg.N
    l_idx, k_idx = data_indices(cfg)
    n_data = l_idx.size
    if data_symbols.size < n_data:
        raise ValueError(f"need {n_data} data symbols, got {data_symbols.size}")

    dd = np.zeros((M, N), dtype=np.complex64)
    dd[l_idx, k_idx] = data_symbols[:n_data].astype(np.complex64)

    rms_data = float(np.sqrt(np.mean(np.abs(data_symbols[:n_data]) ** 2) + 1e-12))
    pilot_amp = rms_data * (10.0 ** (cfg.pilot_boost_db / 20.0))
    dd[cfg.pilot_delay, cfg.pilot_doppler] = np.complex64(pilot_amp)

    return dd, pilot_amp


def extract_data_from_dd(rx_dd: np.ndarray, cfg: OTFSConfig) -> np.ndarray:
    """Return the data-cell values (1-D) from a received DD grid."""
    l_idx, k_idx = data_indices(cfg)
    return rx_dd[l_idx, k_idx].astype(np.complex64)


# ─────────────────────────────────────────────────────────────────────────────
# DD-pilot fine timing — directly extract residual sample-offset from the
# embedded pilot's actual position in the DD grid.
# ─────────────────────────────────────────────────────────────────────────────

def find_pilot_offset(
    rx_dd: np.ndarray,
    cfg: "OTFSConfig",
) -> int:
    """Return the residual sample offset suggested by the embedded pilot.

    If the SFFT window is mis-aligned by `k` samples in time, the OTFS DD
    response shifts in the delay direction by `k` (mod M).  So if the pilot
    was embedded at row `cfg.pilot_delay = 0` but we now find its peak at
    row `r_obs`, the receiver is offset by `-r_obs` samples (the SFFT window
    is `r_obs` samples too LATE — advance by `r_obs` to re-align).

    We search the entire pilot-Doppler column (`cfg.pilot_doppler`) for the
    peak magnitude.  Returns the integer offset in samples (between
    `-M//2` and `+M//2-1` after wrapping).
    """
    col = rx_dd[:, cfg.pilot_doppler]
    r_obs = int(np.argmax(np.abs(col)))
    # Wrap to nearest of (-M/2, +M/2)
    M = cfg.M
    if r_obs > M // 2:
        r_obs -= M
    # We are `r_obs - cfg.pilot_delay` samples too late.  The decoder
    # convention is: shift OFFSET in samples to add to payload_start.
    return int(r_obs - cfg.pilot_delay)


# ─────────────────────────────────────────────────────────────────────────────
# DD-domain channel estimation from the embedded pilot
# ─────────────────────────────────────────────────────────────────────────────

def estimate_dd_channel_taps(
    rx_dd: np.ndarray,
    cfg: OTFSConfig,
    pilot_amp: float,
) -> List[Tuple[int, int, complex]]:
    """Estimate DD channel taps [(l, k, h_complex), …] from the embedded pilot.

    The received DD response to a unit-impulse at (l_p, k_p) is a sparse
    set of taps localised in the guard region.  We read out the magnitude/
    phase of each guard cell, divide by `pilot_amp`, and keep the strongest
    taps above 5% of the peak — the rest are typically noise.
    """
    M, N = cfg.M, cfg.N
    taps: List[Tuple[int, int, complex]] = []
    peak = 0.0
    raw: List[Tuple[int, int, complex]] = []
    for dl in range(0, cfg.l_max + 1):
        for dk in range(-cfg.k_max, cfg.k_max + 1):
            l = (cfg.pilot_delay + dl) % M
            k = (cfg.pilot_doppler + dk) % N
            h = complex(rx_dd[l, k]) / (pilot_amp + 1e-12)
            raw.append((dl, dk, h))
            peak = max(peak, abs(h))
    threshold = 0.05 * peak
    for dl, dk, h in raw:
        if abs(h) >= threshold:
            taps.append((dl, dk, h))
    return taps


# ─────────────────────────────────────────────────────────────────────────────
# Effective DD channel matrix
# ─────────────────────────────────────────────────────────────────────────────

def build_effective_dd_matrix(
    taps: List[Tuple[int, int, complex]],
    M: int,
    N: int,
) -> np.ndarray:
    """Construct the MN×MN effective DD channel matrix H_DD.

    For DD-symbol-domain input x ∈ C^{MN} (vec of DD grid, column-major),
    the receiver observes  y = H_DD · x  (+ noise).

    H_DD[(l', k'), (l, k)] = Σ_i h_i · δ(l' = (l + l_i) mod M)
                                     · ψ((k' - k - k_i) mod N)
    where ψ accounts for the additional inter-Doppler interference from the
    rectangular Tx/Rx window — for the simple (integer Doppler, no fractional)
    case we use ψ(Δk) = (1/N) Σ_n exp(j 2π n (k_i - Δk) / N)
                       = 1   if Δk = k_i,   0  otherwise.
    That is, integer-Doppler delta — sufficient for the Phase-1 hardware
    bring-up; fractional-Doppler refinement can replace this kernel later.
    """
    MN = M * N
    H = np.zeros((MN, MN), dtype=np.complex64)
    for l_tap, k_tap, h in taps:
        for k in range(N):
            for l in range(M):
                l_out = (l + l_tap) % M
                k_out = (k + k_tap) % N
                col = k * M + l            # column-major flatten of (l, k)
                row = k_out * M + l_out
                H[row, col] += np.complex64(h)
    return H


# ─────────────────────────────────────────────────────────────────────────────
# DD-LMMSE detector
# ─────────────────────────────────────────────────────────────────────────────

def dd_lmmse_detect(
    rx_dd: np.ndarray,
    H_dd: np.ndarray,
    noise_var: float,
    cfg: OTFSConfig,
) -> np.ndarray:
    """Closed-form LMMSE on the full DD vector, return data-cell estimates.

    x̂ = (H^H H + σ² I)^{-1} H^H y
    Then read out the data cells from the un-vectorized grid.

    For M=64, N=8 → MN=512 → a 512×512 solve.  This is cheap enough that
    we don't need message passing yet.  Move to MP only when MN > ~4096.
    """
    M, N = cfg.M, cfg.N
    y = rx_dd.flatten(order="F").astype(np.complex64)
    HhH = H_dd.conj().T @ H_dd
    A = HhH + np.complex64(noise_var) * np.eye(M * N, dtype=np.complex64)
    Hhy = H_dd.conj().T @ y
    x_hat = np.linalg.solve(A, Hhy)
    x_grid = x_hat.reshape((M, N), order="F")
    return extract_data_from_dd(x_grid, cfg)


# ─────────────────────────────────────────────────────────────────────────────
# Constellation helpers (compatible with rf_stream_tx_step6phy.py)
# ─────────────────────────────────────────────────────────────────────────────

MOD_BPS = {"bpsk": 1, "qpsk": 2, "qam8": 3, "qam16": 4, "qam32": 5}


def make_constellation(mod: str) -> np.ndarray:
    """Identical to `rf_stream_tx_step6phy.make_constellation`.

    Reproduced here verbatim so the OTFS demapper matches the OFDM chain
    bit-for-bit; if we drift, BER becomes meaningless.
    """
    mod = mod.lower()
    if mod == "bpsk":
        return np.array([-1+0j, 1+0j], dtype=np.complex64)
    if mod == "qpsk":
        return np.array([1+1j, -1+1j, -1-1j, 1-1j], dtype=np.complex64) / np.sqrt(2.0)
    if mod == "qam8":
        phase_idx = [0, 1, 3, 2, 6, 7, 5, 4]
        return np.exp(1j * np.pi / 4.0 * np.array(phase_idx, dtype=float)).astype(np.complex64)
    if mod == "qam16":
        g = {0b00: -3, 0b01: -1, 0b11: 1, 0b10: 3}
        t = np.array([g[(i >> 2) & 3] + 1j * g[i & 3] for i in range(16)], dtype=np.complex64)
        return t / np.sqrt(10.0)
    if mod == "qam32":
        pts = np.array(
            [r + 1j * m for r in (-5, -3, -1, 1, 3, 5) for m in (-5, -3, -1, 1, 3, 5)
             if not (abs(r) == 5 and abs(m) == 5)],
            dtype=np.complex64,
        )
        pts /= np.sqrt(np.mean(np.abs(pts) ** 2))
        order = np.lexsort((pts.real, -pts.imag))
        return pts[order]
    raise ValueError(f"unsupported modulation {mod!r}")


def hard_demap(symbols: np.ndarray, const_table: np.ndarray) -> np.ndarray:
    """Minimum-distance demapping → int labels of shape (N,)."""
    diff = symbols[:, None] - const_table[None, :]
    return np.argmin(np.abs(diff) ** 2, axis=1).astype(np.int32)


def bits_to_symbols(bits: np.ndarray, const_table: np.ndarray, bps: int) -> np.ndarray:
    """MSB-first bit-array → complex symbols, identical to step6 TX."""
    n = len(bits) // bps
    if n == 0:
        return np.zeros(0, dtype=np.complex64)
    b = bits[:n * bps].astype(np.int32).reshape(n, bps)
    powers = 2 ** np.arange(bps - 1, -1, -1, dtype=np.int32)
    indices = (b * powers).sum(axis=1).astype(np.int32)
    return const_table[indices]


def symbols_to_bits(idx: np.ndarray, bps: int) -> np.ndarray:
    """Inverse of `bits_to_symbols` (MSB-first)."""
    out = np.zeros(idx.size * bps, dtype=np.uint8)
    for b in range(bps):
        out[b::bps] = ((idx >> (bps - 1 - b)) & 1).astype(np.uint8)
    return out


def bits_from_bytes(b: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(b, dtype=np.uint8), bitorder='big').astype(np.uint8)


def bits_to_bytes(bits: np.ndarray) -> bytes:
    return np.packbits(bits, bitorder='big').tobytes()


def scramble_bits(bits: np.ndarray, seed: int = 0x7F) -> np.ndarray:
    """Same 7-bit LFSR as step6; self-inverse (XOR)."""
    state = seed
    out = np.zeros_like(bits)
    for i in range(len(bits)):
        b7 = (state >> 6) & 1
        b4 = (state >> 3) & 1
        out[i] = bits[i] ^ b7
        state = ((state << 1) | (b7 ^ b4)) & 0x7F
    return out
