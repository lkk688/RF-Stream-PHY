"""
otfs_sync.py — Preamble synchronisation and CFO/channel estimation.

Wraps the proven Schmidl–Cox + LTF code from rf_stream_rx_step10phy.py so
the OTFS RX can reuse the same primitives without depending on the whole
step10 RX module.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .otfs_frame import N_FFT, N_CP, create_schmidl_cox_stf, create_ltf

SYMBOL_LEN = N_FFT + N_CP


# ─────────────────────────────────────────────────────────────────────────────
# Convolution helpers (pure numpy — Numba not available everywhere)
# ─────────────────────────────────────────────────────────────────────────────

def xcorr_mag_valid(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """|x ⋆ conj(h)| (valid region only). x and h are complex64 1-D arrays."""
    nx, nh = x.shape[0], h.shape[0]
    if nx < nh:
        return np.zeros(0, dtype=np.float32)
    n_out = nx - nh + 1
    h_rev = np.conj(h)[::-1]
    full = np.convolve(x, h_rev, mode="full")
    valid = full[nh - 1: nh - 1 + n_out]
    return np.abs(valid).astype(np.float32)


def moving_energy(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 0 or x.size < win:
        return np.zeros(0, dtype=np.float32)
    p = (np.abs(x) ** 2).astype(np.float64)
    csum = np.concatenate([[0.0], np.cumsum(p)])
    return ((csum[win:] - csum[:-win]) / win).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Schmidl-Cox STF detection → CFO estimate
# ─────────────────────────────────────────────────────────────────────────────

def schmidl_cox_metric(
    rx: np.ndarray,
    half_period: int,
    n_pairs: int,
) -> np.ndarray:
    """Classical Schmidl-Cox timing metric  M(d) = |R(d)|² / P(d)².

        R(d) = Σ_{m=0}^{L-1} rx[d + m] * conj(rx[d + m + P])
        P(d) = Σ_{m=0}^{L-1} |rx[d + m + P]|²
        M(d) = |R(d)|² / (P(d)² + ε)

    where P = `half_period` and L = `n_pairs * P` is the number of paired
    samples used in the sum (typically the STF body length minus one period).

    Properties (Schmidl & Cox 1997):
      • M(d) is invariant to any rotation of `rx`  → CFO doesn't move the peak.
      • M(d) approaches 1.0 at the start of the STF and 0 elsewhere, *regardless
        of how many identical frame copies are present in the buffer*.
        That's the key property that ordinary cross-correlation lacks.
      • For our 6-repeat STF (period P = N_FFT/2 = 32 samples), L = 5*P = 160
        gives a clean plateau over the STF body.

    Returns a real-valued array of length `len(rx) - n_pairs*half_period -
    half_period + 1`, indexed by `d` (the candidate STF-onset sample).
    """
    P = int(half_period)
    L = int(n_pairs * P)
    if rx.size < L + P:
        return np.zeros(0, dtype=np.float32)

    x       = rx.astype(np.complex64)
    prod    = x[:-P] * np.conj(x[P:])              # element-wise pair product
    pow2    = (np.abs(x[P:]) ** 2).astype(np.float32)

    # Rolling sums of length L  →  R(d) and P(d)
    csum_r  = np.concatenate([[0.0 + 0.0j], np.cumsum(prod)])
    csum_p  = np.concatenate([[0.0], np.cumsum(pow2)])
    n_out   = prod.size - L + 1
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)
    R       = csum_r[L : L + n_out] - csum_r[: n_out]
    P_norm  = csum_p[L : L + n_out] - csum_p[: n_out]

    M = (np.abs(R) ** 2) / (P_norm ** 2 + 1e-12)
    return M.astype(np.float32)


def detect_stf(
    rx: np.ndarray,
    stf_ref: np.ndarray,
    stf_repeats: int = 6,
    score_threshold: float = 0.2,
    use_schmidl: bool = True,
) -> Optional[int]:
    """Return the sample index of STF onset (after CP), or None.

    Two methods are available:

      `use_schmidl=True`  : Schmidl-Cox |R|²/P² metric — rotation-invariant and
                            unambiguous under cyclic-buffer TX (multiple identical
                            frames in the RX window).  This is the right detector
                            for Phase 1.5+.
      `use_schmidl=False` : Cross-correlation against `stf_ref`, normalised by
                            STF-window energy.  Was the Phase-1 default; falls back
                            to one-of-many-peaks when several identical frames
                            are present.

    The Schmidl metric peaks ~CP samples *before* the STF body (because the
    half-period autocorrelation holds across the boundary between CP and
    body).  We shift by `N_CP` to return the same index convention as the
    cross-correlation path (i.e. start-of-CP).
    """
    if use_schmidl:
        P = N_FFT // 2   # 32 samples — STF half-period
        # Use most of the body in pair-correlation.  n_pairs * P samples are
        # paired with the next n_pairs * P samples; for 6-period STF body
        # (length 192 samples = 6 × P), we can sum over up to 5 pairs.
        n_pairs = max(1, 2 * stf_repeats - 2)
        M = schmidl_cox_metric(rx, half_period=P, n_pairs=n_pairs)
        if M.size == 0:
            return None
        peak_val = float(np.max(M))
        if peak_val < score_threshold:
            return None
        peak_idx = int(np.argmax(M))

        # Walk left to the rising edge of the plateau (tight threshold so we
        # bracket only the inner "ideal" plateau region).
        edge_th = 0.97 * peak_val
        i = peak_idx
        while i > 0 and M[i - 1] >= edge_th:
            i -= 1
        schmidl_left = max(0, int(i))

        # Refine with cross-correlation against `stf_ref` in a tight ±32
        # window around the Schmidl left-edge.  Schmidl gives a CFO-immune
        # but slightly-sloped left edge; the cross-correlation peak inside
        # this small window gives the exact sample-accurate STF start.
        radius = N_FFT
        lo = max(0, schmidl_left - radius)
        hi = min(rx.size - stf_ref.size, schmidl_left + radius)
        if hi <= lo:
            return schmidl_left
        local_corr = xcorr_mag_valid(
            rx[lo: hi + stf_ref.size].astype(np.complex64),
            stf_ref.astype(np.complex64),
        )
        if local_corr.size == 0:
            return schmidl_left
        return int(lo + int(np.argmax(local_corr)))

    # Legacy cross-correlation path
    corr = xcorr_mag_valid(rx.astype(np.complex64), stf_ref.astype(np.complex64))
    if corr.size == 0:
        return None
    energy = moving_energy(rx, stf_ref.size)
    if energy.size == 0:
        return int(np.argmax(corr))
    n = min(corr.size, energy.size)
    score = corr[:n] / np.sqrt(energy[:n] + 1e-12)
    idx = int(np.argmax(score))
    return idx


def estimate_cfo_schmidl(
    rx: np.ndarray,
    stf_idx: int,
    stf_repeats: int,
    fs: float,
) -> float:
    """Coarse CFO estimate from STF half-period correlation.

    The STF has CP + stf_repeats * N_FFT samples; the FFT-frame half-period
    is P = N_FFT/2 (since the STF was built from EVEN subcarriers only,
    its time-domain signal repeats every N_FFT/2).
    """
    P = N_FFT // 2
    stf_len = stf_repeats * N_FFT + N_CP
    sc_s = stf_idx
    sc_e = stf_idx + stf_len - P
    if sc_e <= sc_s or sc_e + P > rx.size:
        return 0.0
    a = rx[sc_s + P: sc_e + P].astype(np.complex64)
    b = rx[sc_s: sc_e].astype(np.complex64)
    R = np.sum(a * np.conj(b))
    if R == 0:
        return 0.0
    return float(np.angle(R)) * fs / (2.0 * np.pi * P)


def apply_cfo_correction(rx: np.ndarray, cfo_hz: float, fs: float) -> np.ndarray:
    n = np.arange(rx.size, dtype=np.float32)
    return (rx * np.exp(-1j * 2.0 * np.pi * (cfo_hz / fs) * n)).astype(np.complex64)


# ─────────────────────────────────────────────────────────────────────────────
# LTF-based channel estimate (used as initial guess + per-subcarrier SNR)
# ─────────────────────────────────────────────────────────────────────────────

def create_ltf_ref(num_symbols: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    X = np.zeros(N_FFT, dtype=np.complex64)
    used = np.array([k for k in range(-26, 27) if k != 0], dtype=int)
    for i, k in enumerate(used):
        X[(k + N_FFT) % N_FFT] = 1.0 if (i % 2 == 0) else -1.0
    x = np.fft.ifft(np.fft.ifftshift(X)) * np.sqrt(N_FFT)
    ltf_sym = np.concatenate([x[-N_CP:], x]).astype(np.complex64)
    return np.tile(ltf_sym, num_symbols).astype(np.complex64), X


def extract_ofdm_symbol(rx: np.ndarray, start: int) -> Optional[np.ndarray]:
    if start + SYMBOL_LEN > rx.shape[0]:
        return None
    return np.fft.fftshift(np.fft.fft(rx[start + N_CP: start + SYMBOL_LEN]))


def channel_estimate_from_ltf(
    rx: np.ndarray,
    ltf_start: int,
    ltf_freq_ref: np.ndarray,
    ltf_symbols: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Average LTF symbols, return per-subcarrier H and per-subcarrier SNR (dB).

    Identical formula to step10's `channel_estimate_from_ltf` so OFDM/OTFS
    receivers see the same H estimate from the same captured signal.
    """
    Ys = []
    for i in range(ltf_symbols):
        Y = extract_ofdm_symbol(rx, ltf_start + i * SYMBOL_LEN)
        if Y is None:
            return None, None
        Ys.append(Y)
    Yavg = np.mean(Ys, axis=0)
    H = np.zeros(N_FFT, dtype=np.complex64)
    for idx in range(N_FFT):
        if np.abs(ltf_freq_ref[idx]) > 1e-6:
            H[idx] = Yavg[idx] / ltf_freq_ref[idx]
    # Per-subcarrier SNR (dB) from inter-LTF variance
    snr = np.zeros(N_FFT, dtype=np.float32)
    if len(Ys) >= 2:
        var = np.var(np.stack(Ys, axis=0), axis=0)
        sig_pow = np.abs(Yavg) ** 2
        snr_lin = sig_pow / (var + 1e-12)
        snr = (10.0 * np.log10(snr_lin + 1e-12)).astype(np.float32)
    return H, snr


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: full preamble sync pass
# ─────────────────────────────────────────────────────────────────────────────

def sync_preamble(
    rx: np.ndarray,
    stf_repeats: int,
    ltf_symbols: int,
    fs: float,
    ltf_fine_sweep: int = 24,
) -> Optional[dict]:
    """Run full STF/CFO/LTF pipeline.  Returns dict with sync info, or None.

    Phase-1 OTFS hardware bring-up taught us that coarse STF sync alone
    leaves ±N samples of timing uncertainty, which is *fatal* for OTFS
    (the CP only absorbs `l_max` ≈ 6 samples).  After STF + CFO we do a
    fine timing search over ±`ltf_fine_sweep` samples around the expected
    LTF position by maximising the correlation against the LTF time-domain
    reference — this is the same fix step10 RX uses.

    Keys:
        stf_idx        : sample index of STF onset (coarse, CP-included)
        cfo_hz         : coarse CFO from Schmidl-Cox
        ltf_start      : sample index of LTF onset *after* fine sync
        payload_start  : sample index where OTFS payload begins
        H_ltf          : (N_FFT,) complex64 channel estimate
        snr_ltf        : (N_FFT,) per-subcarrier SNR (dB)
        rx_corr        : CFO-corrected rx (same length as input)
        ltf_corr_peak  : peak LTF correlation magnitude (sync quality)
    """
    stf_ref = create_schmidl_cox_stf(stf_repeats)
    stf_len = stf_repeats * N_FFT + N_CP

    stf_idx = detect_stf(rx, stf_ref, stf_repeats=stf_repeats)
    if stf_idx is None:
        return None

    cfo_hz = estimate_cfo_schmidl(rx, stf_idx, stf_repeats, fs)
    rx_corr = apply_cfo_correction(rx, cfo_hz, fs)

    ltf_start_coarse = stf_idx + stf_len
    if ltf_start_coarse + ltf_symbols * SYMBOL_LEN >= rx.size:
        return None

    # ── LTF fine timing search (single LTF symbol = CP + N_FFT samples) ──
    ltf_ref_full, ltf_freq_ref = create_ltf_ref(ltf_symbols)
    ltf_td_ref = ltf_ref_full[:SYMBOL_LEN].astype(np.complex64)
    search_s = max(0, ltf_start_coarse - ltf_fine_sweep)
    search_e = min(rx_corr.shape[0] - SYMBOL_LEN,
                    ltf_start_coarse + ltf_fine_sweep)
    ltf_peak = 0.0
    if search_e > search_s:
        sb        = rx_corr[search_s: search_e + SYMBOL_LEN].astype(np.complex64)
        corr_ltf  = xcorr_mag_valid(sb, ltf_td_ref)
        if corr_ltf.size > 0:
            best     = int(np.argmax(corr_ltf))
            ltf_start = search_s + best
            ltf_peak  = float(corr_ltf[best])
        else:
            ltf_start = ltf_start_coarse
    else:
        ltf_start = ltf_start_coarse

    H, snr_db = channel_estimate_from_ltf(rx_corr, ltf_start, ltf_freq_ref, ltf_symbols)
    if H is None:
        return None

    payload_start = ltf_start + ltf_symbols * SYMBOL_LEN
    return {
        "stf_idx":       int(stf_idx),
        "cfo_hz":        float(cfo_hz),
        "ltf_start":     int(ltf_start),
        "payload_start": int(payload_start),
        "ltf_corr_peak": float(ltf_peak),
        "H_ltf":         H,
        "snr_ltf":       snr_db,
        "rx_corr":       rx_corr,
    }
