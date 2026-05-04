#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rf_stream_rx_step10phy.py  – Step 10 RX: MMSE Equalization + OTA Robustness

Extends step9 with improvements targeted at OTA antenna operation:

Step9 had:
  • LTF ZF channel estimation
  • Per-symbol pilot residual correction (phase-only, adaptive bypass)
  • Fixed energy_z_th=8.0 regardless of modulation

Step10 adds:
  1. MMSE equalization (--equalization mmse | zf):
       H_mmse[k] = H[k]* / (|H[k]|² + noise_var)
       Reduces noise enhancement at subcarriers with poor channel response.
       Noise variance estimated from LTF SNR measurement.

  2. Modulation-aware energy detection threshold (--auto_z_th):
       QAM16 has ~3 dB higher PAPR than QPSK → lower average energy per sample.
       Auto-scales energy_z_th: qpsk × 1.0, qam16 × 0.65, bpsk × 1.2
       Prevents QAM16 OTA packets being missed by the energy detector.

  3. Pilot correction weight (--pilot_weight 0.0–1.0):
       Blends LTF-static H with per-symbol pilot correction:
         H_eff = (1-w) * H_ltf + w * H_pilot_corrected
       Weight 0.0 = pure LTF (step8), 1.0 = full pilot correction (step9).
       Default 0.5 for balanced OTA performance.

All gate / mixture-logging / MT inferencer logic inherited from step8/9.
"""

import argparse
import csv
import json
import os
import queue
import random
import threading
import time
import zlib
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Tuple

import numpy as np

# Optional Torch for neural gate
try:
    import torch
    import torch.nn as _tnn
    TORCH_OK = True
except Exception:
    TORCH_OK = False


# ─────────────────────────────────────────────────────────────────────────────
# Optional Numba JIT
# ─────────────────────────────────────────────────────────────────────────────
try:
    from numba import njit
    NUMBA_OK = True
except Exception:
    NUMBA_OK = False
    def njit(*a, **kw):
        def deco(f): return f
        return deco


# ─────────────────────────────────────────────────────────────────────────────
# PHY parameters  (must match TX)
# ─────────────────────────────────────────────────────────────────────────────
N_FFT      = 64
N_CP       = 16
SYMBOL_LEN = N_FFT + N_CP

PILOT_SUBCARRIERS = np.array([-21, -7, 7, 21], dtype=int)
DATA_SUBCARRIERS  = np.array(
    [k for k in range(-26, 27) if k != 0 and k not in set(PILOT_SUBCARRIERS)],
    dtype=int,
)
N_DATA = len(DATA_SUBCARRIERS)   # 48
MAGIC  = b"AIS1"

def sc_to_bin(k: int) -> int:
    return (k + N_FFT) % N_FFT

DATA_BINS  = np.array([sc_to_bin(int(k)) for k in DATA_SUBCARRIERS], dtype=int)
PILOT_BINS = np.array([sc_to_bin(int(k)) for k in PILOT_SUBCARRIERS], dtype=int)
USED_BINS  = np.array([sc_to_bin(int(k)) for k in range(-26, 27) if k != 0], dtype=int)


# ─────────────────────────────────────────────────────────────────────────────
# Modulation tables  (exact copy of TX definitions)
# ─────────────────────────────────────────────────────────────────────────────
MOD_BPS = {"bpsk": 1, "qpsk": 2, "qam8": 3, "qam16": 4, "qam32": 5}


def make_constellation(mod: str) -> np.ndarray:
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
    raise ValueError(f"Unknown modulation: {mod!r}")


def symbols_to_bits(syms: np.ndarray, table: np.ndarray, bps: int) -> np.ndarray:
    dist    = np.abs(syms[:, np.newaxis] - table[np.newaxis, :]) ** 2
    indices = np.argmin(dist, axis=1).astype(np.int32)
    shifts  = np.arange(bps - 1, -1, -1, dtype=np.int32)
    return ((indices[:, np.newaxis] >> shifts) & 1).astype(np.uint8).ravel()


def bits_to_symbols(bits: np.ndarray, table: np.ndarray, bps: int) -> np.ndarray:
    n = len(bits) // bps
    if n == 0:
        return np.zeros(0, dtype=np.complex64)
    b      = bits[:n * bps].astype(np.int32).reshape(n, bps)
    powers = 2 ** np.arange(bps - 1, -1, -1, dtype=np.int32)
    return table[(b * powers).sum(axis=1).astype(np.int32)]


def mod_evm(syms: np.ndarray, table: np.ndarray) -> float:
    dist = np.abs(syms[:, np.newaxis] - table[np.newaxis, :]) ** 2
    return float(np.sqrt(np.mean(np.min(dist, axis=1))))


def get_rotation_candidates(mod: str) -> list:
    if mod == "bpsk":
        return [1.0 + 0j, -1.0 + 0j]
    if mod in ("qpsk", "qam16", "qam32"):
        return [1.0 + 0j, 1j, -1.0 + 0j, -1j]
    if mod == "qam8":
        return [np.exp(1j * np.pi / 4.0 * k) for k in range(8)]
    return [1.0 + 0j]


# ─────────────────────────────────────────────────────────────────────────────
# Numba JIT kernels
# ─────────────────────────────────────────────────────────────────────────────
@njit(cache=True)
def xcorr_mag_valid(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    nx, nh = x.shape[0], h.shape[0]
    nout = nx - nh + 1
    if nout <= 0:
        return np.zeros(0, dtype=np.float32)
    out = np.zeros(nout, dtype=np.float32)
    for i in range(nout):
        ar, ai = 0.0, 0.0
        for k in range(nh):
            a = x[i + k]; b = h[k]
            ar += a.real * b.real + a.imag * b.imag
            ai += a.imag * b.real - a.real * b.imag
        out[i] = (ar * ar + ai * ai) ** 0.5
    return out

@njit(cache=True)
def moving_energy(x: np.ndarray, win: int) -> np.ndarray:
    n = x.shape[0]
    if win <= 0 or n < win:
        return np.zeros(0, dtype=np.float32)
    out = np.zeros(n - win + 1, dtype=np.float32)
    s   = 0.0
    for i in range(win):
        a = x[i]; s += a.real * a.real + a.imag * a.imag
    out[0] = s / win
    for i in range(1, out.shape[0]):
        a0 = x[i - 1]; a1 = x[i + win - 1]
        s -= a0.real * a0.real + a0.imag * a0.imag
        s += a1.real * a1.real + a1.imag * a1.imag
        out[i] = s / win
    return out


# ─────────────────────────────────────────────────────────────────────────────
# PHY helpers
# ─────────────────────────────────────────────────────────────────────────────
def bits_from_bytes(b: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(b, dtype=np.uint8), bitorder='big').astype(np.uint8)

def bits_to_bytes(bits: np.ndarray) -> bytes:
    L = (len(bits) // 8) * 8
    return np.packbits(bits[:L]).tobytes() if L > 0 else b""

def scramble_bits(bits: np.ndarray, seed: int = 0x7F) -> np.ndarray:
    state, out = seed, np.zeros_like(bits)
    for i in range(len(bits)):
        b7 = (state >> 6) & 1
        b4 = (state >> 3) & 1
        out[i] = bits[i] ^ b7
        state = ((state << 1) | (b7 ^ b4)) & 0x7F
    return out

def majority_vote(bits: np.ndarray, repeat: int) -> np.ndarray:
    if repeat <= 1:
        return bits
    L = (len(bits) // repeat) * repeat
    if L <= 0:
        return np.zeros(0, dtype=np.uint8)
    return (np.sum(bits[:L].reshape(-1, repeat), axis=1) >= (repeat / 2)).astype(np.uint8)

def create_schmidl_cox_stf(num_repeats: int = 6) -> np.ndarray:
    rng = np.random.default_rng(42)
    X   = np.zeros(N_FFT, dtype=np.complex64)
    for i, k in enumerate(np.array([k for k in range(-26, 27, 2) if k != 0], dtype=int)):
        X[sc_to_bin(int(k))] = rng.choice([-1.0, 1.0])
    x = np.fft.ifft(np.fft.ifftshift(X)) * np.sqrt(N_FFT)
    s = np.tile(x.astype(np.complex64), num_repeats)
    return np.concatenate([s[-N_CP:], s]).astype(np.complex64)

def create_ltf_ref(num_symbols: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    X    = np.zeros(N_FFT, dtype=np.complex64)
    used = np.array([k for k in range(-26, 27) if k != 0], dtype=int)
    for i, k in enumerate(used):
        X[sc_to_bin(int(k))] = 1.0 if (i % 2 == 0) else -1.0
    x       = np.fft.ifft(np.fft.ifftshift(X)) * np.sqrt(N_FFT)
    ltf_sym = np.concatenate([x[-N_CP:], x]).astype(np.complex64)
    return np.tile(ltf_sym, num_symbols).astype(np.complex64), X

def extract_ofdm_symbol(rx: np.ndarray, start: int) -> Optional[np.ndarray]:
    if start + SYMBOL_LEN > rx.shape[0]:
        return None
    return np.fft.fftshift(np.fft.fft(rx[start + N_CP: start + SYMBOL_LEN]))

def channel_estimate_from_ltf(
    rx: np.ndarray, ltf_start: int, ltf_freq_ref: np.ndarray, ltf_symbols: int
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    Ys = []
    for i in range(ltf_symbols):
        Y = extract_ofdm_symbol(rx, ltf_start + i * SYMBOL_LEN)
        if Y is None:
            break
        Ys.append(Y)
    if not Ys:
        return None, None
    Yavg = np.mean(np.stack(Ys), axis=0)
    H    = np.ones(N_FFT, dtype=np.complex64)
    used = np.array([k for k in range(-26, 27) if k != 0], dtype=int)
    for k in used:
        idx = sc_to_bin(int(k))
        if np.abs(ltf_freq_ref[idx]) > 1e-6:
            H[idx] = Yavg[idx] / ltf_freq_ref[idx]
    snr_db = None
    if len(Ys) >= 2:
        Y_arr     = np.stack(Ys)[:, USED_BINS]
        noise_var = np.var(Y_arr, axis=0).real + 1e-10
        sig_var   = np.abs(Yavg[USED_BINS]) ** 2
        snr_db    = 10.0 * np.log10(sig_var / noise_var + 1e-10)
    return H, snr_db


# ─────────────────────────────────────────────────────────────────────────────
# Step 10 additions
# ─────────────────────────────────────────────────────────────────────────────

# PAPR scaling for energy_z_th auto-adjustment (relative to QPSK baseline)
MOD_PAPR_SCALE = {
    "bpsk":  1.20,   # BPSK OFDM: lower PAPR → slightly higher threshold OK
    "qpsk":  1.00,   # QPSK baseline
    "qam8":  0.80,
    "qam16": 0.65,   # QAM16: ~3 dB higher PAPR → need lower threshold
    "qam32": 0.55,
}


def equalize_mmse(
    Y: np.ndarray,
    H: np.ndarray,
    noise_var: float,
    active_bins: np.ndarray,
) -> np.ndarray:
    """Per-subcarrier MMSE equalization.

    Y_eq[k] = Y[k] * H[k]* / (|H[k]|² + noise_var)

    Reduces noise enhancement on weak subcarriers vs ZF (Y/H).
    noise_var is the per-subcarrier noise power estimate from LTF.
    """
    Yeq = Y.copy()
    H_active = H[active_bins]
    Y_active  = Y[active_bins]
    H_sq      = np.abs(H_active) ** 2
    Yeq[active_bins] = Y_active * np.conj(H_active) / (H_sq + noise_var + 1e-10)
    return Yeq


def pilot_residual_correction(
    Ye_ltf: np.ndarray,
    pilot_sign: int,
    pilot_vals: np.ndarray,
    bypass_th: float = 0.25,
) -> np.ndarray:
    """Per-symbol residual channel correction interpolated from 4 data pilots.

    Phase-only correction; magnitude correction is left to LTF bulk estimate.
    Bypassed when residual deviation is small (flat/static channel).

    Ye_ltf     : LTF-equalized OFDM symbol (N_FFT, complex64, all bins)
    pilot_sign : +1 or -1 alternating every symbol (matches TX)
    pilot_vals : known pilot reference before sign (shape 4, complex64)
    bypass_th  : if mean |residual - 1| < bypass_th, return all-ones (no-op)

    Returns H_res (N_FFT, complex64) -- correction factor (1.0 on flat channels).
    """
    ONES = np.ones(N_FFT, dtype=np.complex64)

    expected  = (pilot_sign * pilot_vals).astype(np.complex64)
    actual    = Ye_ltf[PILOT_BINS]
    residual  = actual / (expected + 1e-10)   # complex residual at 4 pilots

    # Bypass if residual is very close to 1.0 (static flat channel)
    if float(np.mean(np.abs(residual - 1.0))) < bypass_th:
        return ONES

    # Sort pilot bins ascending for interpolation
    sort_ord    = np.argsort(PILOT_BINS)
    pilot_idx_s = PILOT_BINS[sort_ord].astype(float)
    res_s       = residual[sort_ord]

    # Nearest-neighbor extrapolation at edges
    if pilot_idx_s[0] > 0:
        pilot_idx_s = np.insert(pilot_idx_s, 0, 0.0)
        res_s       = np.insert(res_s, 0, res_s[0])
    if pilot_idx_s[-1] < N_FFT - 1:
        pilot_idx_s = np.append(pilot_idx_s, float(N_FFT - 1))
        res_s       = np.append(res_s, res_s[-1])

    # Phase-only interpolation -- avoids magnitude noise from noisy pilots.
    # Magnitude correction is left entirely to the LTF bulk estimate.
    res_pha   = np.unwrap(np.angle(res_s))
    all_idx   = np.arange(N_FFT, dtype=float)
    res_pha_i = np.interp(all_idx, pilot_idx_s, res_pha)

    return np.exp(1j * res_pha_i).astype(np.complex64)
def parse_packet_bytes(bb: bytes) -> Tuple[bool, str, int, bytes]:
    if len(bb) < 14:
        return False, "too_short", -1, b""
    if bb[:4] != MAGIC:
        return False, "bad_magic", -1, b""
    seq  = int.from_bytes(bb[4:6], "little")
    plen = int.from_bytes(bb[8:10], "little")
    need = 10 + plen + 4
    if len(bb) < need:
        return False, "need_more", seq, b""
    body   = bb[:10 + plen]
    crc_rx = int.from_bytes(bb[10 + plen: 10 + plen + 4], "little")
    if (zlib.crc32(body) & 0xFFFFFFFF) != crc_rx:
        return False, "crc_fail", seq, b""
    return True, "ok", seq, bb[10: 10 + plen]


# ─────────────────────────────────────────────────────────────────────────────
# Ring buffer
# ─────────────────────────────────────────────────────────────────────────────
class RingBuffer:
    def __init__(self, size: int):
        self.size   = int(size)
        self.buf    = np.zeros(self.size, dtype=np.complex64)
        self.w      = 0
        self.filled = False

    def push(self, x: np.ndarray):
        n = x.shape[0]
        if n >= self.size:
            self.buf[:] = x[-self.size:]
            self.w = 0; self.filled = True
            return
        end = self.w + n
        if end <= self.size:
            self.buf[self.w: end] = x
        else:
            n1 = self.size - self.w
            self.buf[self.w:]          = x[:n1]
            self.buf[:end - self.size] = x[n1:]
        self.w = end % self.size
        if not self.filled and self.w == 0:
            self.filled = True

    def get_window(self, length: int) -> np.ndarray:
        length = int(length)
        if length > self.size:
            raise ValueError("window > ring")
        if not self.filled and self.w < length:
            return np.zeros(0, dtype=np.complex64)
        start = (self.w - length) % self.size
        if start < self.w:
            return self.buf[start: self.w].copy()
        return np.concatenate([self.buf[start:], self.buf[:self.w]]).copy()


# ─────────────────────────────────────────────────────────────────────────────
# Neural Gate Inferencer
# ─────────────────────────────────────────────────────────────────────────────
class GateInferencer:
    """
    Loads gate_model.pt produced by train_neural_gate.py and provides
    per-window gate_p predictions.  Never blocks decoding.
    """

    def __init__(self, model_path: str):
        if not TORCH_OK:
            raise RuntimeError("torch not available; cannot load gate model")
        data = torch.load(model_path, map_location="cpu", weights_only=False)
        self.feature_names: list = data["feature_names"]
        self.mu  = data["mu"].numpy().astype(np.float32)
        self.sd  = data["sd"].numpy().astype(np.float32)
        self.recommended_threshold = float(data.get("recommended_threshold", 0.5))

        sd = data["state_dict"]
        d_in   = int(sd["net.0.weight"].shape[1])
        hidden = int(sd["net.0.weight"].shape[0])

        class _MLP(_tnn.Module):
            def __init__(self, d, h):
                super().__init__()
                self.net = _tnn.Sequential(
                    _tnn.Linear(d, h), _tnn.ReLU(), _tnn.Dropout(0.0),
                    _tnn.Linear(h, h), _tnn.ReLU(), _tnn.Dropout(0.0),
                    _tnn.Linear(h, 1),
                )
            def forward(self, x):
                return self.net(x).squeeze(-1)

        self._model = _MLP(d_in, hidden)
        self._model.load_state_dict(sd, strict=True)
        self._model.eval()
        print(f"[Gate] loaded {model_path}  d_in={d_in}  hidden={hidden}"
              f"  thr={self.recommended_threshold:.4f}")

    # ── Feature helpers (mirrors train_neural_gate.py) ───────────────────────

    @staticmethod
    def _corr_features(corr: np.ndarray, topk: int = 8) -> dict:
        if corr is None or corr.size == 0:
            out = {f"corr_q{q}": 0.0 for q in (10, 50, 90, 99)}
            out.update({f"corr_top{i}": 0.0 for i in range(topk)})
            out.update({"corr_max": 0.0, "corr_mean": 0.0, "corr_std": 0.0, "corr_p2m": 0.0})
            return out
        c = np.asarray(corr, dtype=np.float32)
        k = min(topk, c.size)
        idx = np.argpartition(c, -k)[-k:]
        vals = np.sort(c[idx])[::-1]
        med  = float(np.median(c)) + 1e-12
        out  = {
            "corr_q10":  float(np.percentile(c, 10)),
            "corr_q50":  float(np.percentile(c, 50)),
            "corr_q90":  float(np.percentile(c, 90)),
            "corr_q99":  float(np.percentile(c, 99)),
            "corr_max":  float(np.max(c)),
            "corr_mean": float(np.mean(c)),
            "corr_std":  float(np.std(c)),
            "corr_p2m":  float(np.max(c) / med),
        }
        for i in range(topk):
            out[f"corr_top{i}"] = float(vals[i]) if i < vals.size else 0.0
        return out

    @staticmethod
    def _topk_ncc_features(topk_ncc: np.ndarray) -> dict:
        if topk_ncc is None or topk_ncc.size == 0:
            return {"ncc_max": 0.0, "ncc_mean": 0.0, "ncc_std": 0.0,
                    "ncc_top0": 0.0, "ncc_top1": 0.0, "ncc_top2": 0.0, "ncc_p2m": 0.0}
        v  = np.asarray(topk_ncc, dtype=np.float32)
        vs = np.sort(v)[::-1]
        return {
            "ncc_max":  float(vs[0]) if vs.size > 0 else 0.0,
            "ncc_top0": float(vs[0]) if vs.size > 0 else 0.0,
            "ncc_top1": float(vs[1]) if vs.size > 1 else 0.0,
            "ncc_top2": float(vs[2]) if vs.size > 2 else 0.0,
            "ncc_mean": float(np.mean(v)),
            "ncc_std":  float(np.std(v)),
            "ncc_p2m":  float(vs[0] / (float(np.median(v)) + 1e-12)),
        }

    @staticmethod
    def _energy_ds_features(energy_ds: np.ndarray) -> dict:
        if energy_ds is None or energy_ds.size == 0:
            return {"eds_p10": 0.0, "eds_p50": 0.0, "eds_p90": 0.0,
                    "eds_max": 0.0, "eds_ratio": 0.0, "eds_z": 0.0}
        e   = np.asarray(energy_ds, dtype=np.float32)
        p10 = float(np.percentile(e, 10))
        p50 = float(np.percentile(e, 50))
        p90 = float(np.percentile(e, 90))
        mx  = float(np.max(e))
        mad = float(np.median(np.abs(e - p50))) + 1e-12
        return {
            "eds_p10": p10, "eds_p50": p50, "eds_p90": p90,
            "eds_max": mx,
            "eds_ratio": mx / (p10 + 1e-12),
            "eds_z":    float((mx - p50) / (1.4826 * mad)),
        }

    @staticmethod
    def _energy_features_from_rxw(rxw: np.ndarray, win: int = 512) -> dict:
        if rxw is None or rxw.size == 0:
            return {"eng_p10": 0.0, "eng_max": 0.0, "eng_ratio": 0.0,
                    "eng_mean": 0.0, "eng_std": 0.0}
        x   = np.asarray(rxw)
        p   = (x.real ** 2 + x.imag ** 2).astype(np.float32)
        e   = (np.convolve(p, np.ones(win, dtype=np.float32) / float(win), mode="valid")
               if p.size >= win else p)
        p10 = float(np.percentile(e, 10))
        mx  = float(np.max(e))
        return {
            "eng_p10":   p10,
            "eng_max":   mx,
            "eng_ratio": float(mx / (p10 + 1e-12)),
            "eng_mean":  float(np.mean(e)),
            "eng_std":   float(np.std(e)),
        }

    def _build_feature_dict(self, corr_norm, topk_ncc, energy_ds, rxw, meta) -> dict:
        feats: dict = {}
        feats.update(self._corr_features(corr_norm))
        feats.update(self._energy_features_from_rxw(rxw))
        feats.update(self._topk_ncc_features(topk_ncc))
        feats.update(self._energy_ds_features(energy_ds))

        def sf(x):
            try:
                return np.nan if x is None else float(x)
            except Exception:
                return np.nan

        # meta_ prefix (matches training load_npz_features meta extraction)
        feats["meta_xc_best_peak"] = sf(meta.get("xc_best_peak"))
        feats["meta_peak"]         = sf(meta.get("peak"))
        feats["meta_probe_evm"]    = sf(meta.get("probe_evm"))
        feats["meta_snr_db"]       = sf(meta.get("snr_db"))
        feats["meta_cfo_hz"]       = sf(meta.get("cfo_hz"))
        feats["meta_z"]            = sf(meta.get("z"))
        feats["meta_ncc_best"]     = sf(meta.get("ncc_best"))
        feats["meta_med"]          = sf(meta.get("med"))
        feats["meta_mad"]          = sf(meta.get("mad"))
        feats["meta_rx_gain"]      = sf(meta.get("rx_gain"))
        feats["meta_bps"]          = sf(meta.get("bps"))

        # csv_ prefix (matches training csv_row_to_features)
        for k in ["peak", "p10", "eg_th", "maxe", "xc_best_peak", "xc_best_idx",
                  "probe_evm", "cfo_hz", "snr_db", "rx_gain", "bps",
                  "med", "mad", "z", "ncc_best", "ncc_best_idx"]:
            feats[f"csv_{k}"] = sf(meta.get(k))
        mx  = sf(meta.get("maxe"));  p10 = sf(meta.get("p10"))
        xc  = sf(meta.get("xc_best_peak"))
        z_v = sf(meta.get("z"));     ncc_v = sf(meta.get("ncc_best"))
        med_v = sf(meta.get("med")); mad_v = sf(meta.get("mad"))
        if np.isfinite(mx) and np.isfinite(p10):
            feats["csv_gate_ratio"] = float(mx / (p10 + 1e-12))
            if np.isfinite(xc):
                feats["csv_xc_over_p10"] = float(xc / (p10 + 1e-12))
        if np.isfinite(z_v):
            feats["csv_z_log"] = float(np.log1p(max(z_v, 0.0)))
        if np.isfinite(ncc_v):
            feats["csv_ncc_best"] = ncc_v
        if np.isfinite(med_v) and np.isfinite(mad_v) and mad_v > 0:
            feats["csv_snr_mad"] = float(med_v / (mad_v + 1e-12))
        return feats

    def predict(
        self,
        corr_norm: np.ndarray,
        topk_ncc: np.ndarray,
        energy_ds: np.ndarray,
        rxw: np.ndarray,
        meta: dict,
    ) -> float:
        feats = self._build_feature_dict(corr_norm, topk_ncc, energy_ds, rxw, meta)
        vec   = np.array([feats.get(k, np.nan) for k in self.feature_names], dtype=np.float32)
        np.nan_to_num(vec, copy=False, nan=0.0)
        vec   = (vec - self.mu) / (self.sd + 1e-8)
        with torch.no_grad():
            x     = torch.from_numpy(vec[np.newaxis, :])
            logit = self._model(x).item()
        return float(1.0 / (1.0 + np.exp(-logit)))


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Task Preamble CNN Inferencer
# ─────────────────────────────────────────────────────────────────────────────
class MultiTaskInferencer:
    """
    Drop-in replacement for GateInferencer using multitask_v1.pt.

    Input : 800 complex IQ samples (STF+LTF) from rxw at best NCC offset.
    Outputs: gate_p (float, returned); mod_p and snr_pred stored as attributes.

    Auto-detected at load time when checkpoint contains key "tasks".
    """
    PREAM_LEN = 800

    def __init__(self, model_path: str):
        if not TORCH_OK:
            raise RuntimeError("torch not available; cannot load multi-task gate")
        # Disable CuDNN to prevent Conv1D segfaults on this GPU driver
        import torch.backends.cudnn as _cudnn
        _cudnn.enabled = False

        data = torch.load(model_path, map_location="cpu", weights_only=False)
        self.tasks    = data.get("tasks", ["gate"])
        self._embed   = int(data.get("embed_dim", 256))
        self._snr_mu  = float(data.get("snr_mu",    0.0))
        self._snr_sig = float(data.get("snr_sigma",  1.0))
        self.recommended_threshold = 0.5
        _m = data.get("metrics", {})
        if "gate" in _m:
            self.recommended_threshold = float(_m["gate"].get("best_thr", 0.5))

        # Inline MultiTaskPreamCNN architecture
        import torch.nn as _nn
        _e = self._embed

        class _MTModel(_nn.Module):
            def __init__(self, tasks, embed):
                super().__init__()
                self._tasks = tasks
                self.encoder = _nn.Sequential(
                    _nn.Conv1d(2, 32, 31, padding=15), _nn.BatchNorm1d(32), _nn.ReLU(),
                    _nn.MaxPool1d(4),
                    _nn.Conv1d(32, 64, 15, padding=7), _nn.BatchNorm1d(64), _nn.ReLU(),
                    _nn.MaxPool1d(4),
                    _nn.Conv1d(64, 128, 7, padding=3), _nn.BatchNorm1d(128), _nn.ReLU(),
                    _nn.AdaptiveAvgPool1d(8),
                )
                self.proj = _nn.Sequential(
                    _nn.Linear(128 * 8, embed), _nn.ReLU(), _nn.Dropout(0.0))
                if "gate" in tasks:
                    self.gate_head = _nn.Linear(embed, 1)
                if "mod" in tasks:
                    self.mod_head = _nn.Linear(embed, 1)
                if "snr" in tasks:
                    self.snr_head = _nn.Sequential(
                        _nn.Linear(embed, 32), _nn.ReLU(), _nn.Linear(32, 1))

            def forward(self, x):
                B = x.size(0)
                x2 = x.view(B, 800, 2).permute(0, 2, 1).contiguous()
                emb = self.proj(self.encoder(x2).view(B, -1))
                out = {}
                if "gate" in self._tasks:
                    out["gate"] = self.gate_head(emb).squeeze(-1)
                if "mod" in self._tasks:
                    out["mod"]  = self.mod_head(emb).squeeze(-1)
                if "snr" in self._tasks:
                    out["snr"]  = self.snr_head(emb).squeeze(-1)
                return out

        self._model = _MTModel(self.tasks, _e)
        self._model.load_state_dict(data["state_dict"], strict=True)
        self._model.eval()

        # Attributes updated after each predict() call for optional logging
        self.last_mod_p    = float("nan")   # 0=QPSK, 1=QAM16 (sigmoid output)
        self.last_snr_pred = float("nan")   # dB (de-normalised)

        print(f"[Gate/MT] loaded {model_path}")
        print(f"  tasks={self.tasks}  embed={_e}  thr={self.recommended_threshold:.4f}")
        print(f"  SNR norm: mu={self._snr_mu:.1f}dB  sigma={self._snr_sig:.1f}dB")

    def predict(
        self,
        corr_norm:  np.ndarray,
        topk_ncc:   np.ndarray,
        energy_ds:  np.ndarray,
        rxw:        np.ndarray,
        meta:       dict,
    ) -> float:
        # Extract preamble IQ at best NCC candidate offset
        stf = int(meta.get("ncc_best_idx", meta.get("xc_best_idx", 0)))
        end = stf + self.PREAM_LEN
        if end > len(rxw):
            stf = max(0, len(rxw) - self.PREAM_LEN)
            end = len(rxw)
        seg = np.array(rxw[stf:end], dtype=np.complex64, copy=True)
        if len(seg) < self.PREAM_LEN:
            seg = np.pad(seg, (0, self.PREAM_LEN - len(seg)))

        # Build 1600-float interleaved real/imag tensor
        feat = np.empty(self.PREAM_LEN * 2, dtype=np.float32)
        feat[0::2] = seg.real
        feat[1::2] = seg.imag
        np.nan_to_num(feat, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        x = torch.from_numpy(feat[np.newaxis, :]).float().contiguous()

        with torch.no_grad():
            out = self._model(x)

        gate_p = float(torch.sigmoid(out["gate"]).item()) if "gate" in out else 0.5

        if "mod" in out:
            self.last_mod_p = float(torch.sigmoid(out["mod"]).item())
        if "snr" in out:
            snr_norm = float(out["snr"].item())
            self.last_snr_pred = snr_norm * self._snr_sig + self._snr_mu

        return gate_p


# ─────────────────────────────────────────────────────────────────────────────
# Mixture Logger
# ─────────────────────────────────────────────────────────────────────────────
class MixtureLogger:
    """
    Decides save_tag for each processed window using 3 tiers:

      "ok"        – decoded OK, always saved
      "fail_hard" – high-confidence packet-like fail: always saved (hard negative)
      "fail_mid"  – medium-confidence: saved with mid_prob
      "fail_bg"   – low-confidence / pure background: saved with bg_save_prob
      "skip"      – not saved

    Tier thresholds (any one condition fires the tier):
      hard: z >= hard_z  OR  ncc >= hard_ncc  OR  gate_p >= hard_gate_p
      mid:  z >= mid_z   OR  ncc >= mid_ncc   OR  gate_p >= mid_gate_p
      bg:   z >= energy_z_th (already passed gate) but below mid
    """

    def __init__(
        self,
        gate_threshold: float,
        fail_save_ncc_min: float,
        fail_save_z_min: float,
        fail_npz_prob: float,   # kept for CLI compat; used as mid_prob
        bg_save_prob: float,
        has_gate: bool,
        # hard tier
        hard_z: float = 200.0,
        hard_ncc: float = 0.70,
        hard_gate_p: float = 0.80,
        # mid tier
        mid_z: float = 30.0,
        mid_ncc: float = 0.30,
        mid_gate_p: float = 0.30,
        mid_prob: float = 0.5,
    ):
        self.gate_threshold = gate_threshold
        self.fail_ncc_min   = fail_save_ncc_min
        self.fail_z_min     = fail_save_z_min
        self.fail_npz_prob  = fail_npz_prob
        self.bg_save_prob   = bg_save_prob
        self.has_gate       = has_gate
        self.hard_z         = hard_z
        self.hard_ncc       = hard_ncc
        self.hard_gate_p    = hard_gate_p
        self.mid_z          = mid_z
        self.mid_ncc        = mid_ncc
        self.mid_gate_p     = mid_gate_p
        self.mid_prob       = mid_prob

    def decide_ok(self) -> str:
        return "ok"

    def decide_fail(self, z: float, ncc_best: float, gate_p: float) -> str:
        gp = gate_p if np.isfinite(gate_p) else 0.0
        # hard tier: always save
        if z >= self.hard_z or ncc_best >= self.hard_ncc or gp >= self.hard_gate_p:
            return "fail_hard"
        # mid tier: save with mid_prob
        if z >= self.mid_z or ncc_best >= self.mid_ncc or gp >= self.mid_gate_p:
            if self.fail_npz_prob >= 1.0 or random.random() < self.mid_prob:
                return "fail_mid"
            return "skip"
        # low tier: background save
        if z >= self.fail_z_min or ncc_best >= self.fail_ncc_min:
            if self.bg_save_prob > 0.0 and random.random() < self.bg_save_prob:
                return "fail_bg"
        return "skip"

    def decide_bg(self) -> str:
        if self.bg_save_prob > 0.0 and random.random() < self.bg_save_prob:
            return "fail_bg"
        return "skip"


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RxConfig:
    uri: str; fc: float; fs: float
    rx_gain: float; rx_bw: float
    rx_buf: int; kernel_buffers: int

    repeat: int; stf_repeats: int; ltf_symbols: int
    modulation: str; bps: int

    ring_size: int; proc_window: int; proc_hop: int
    energy_win: int; energy_mult: float; energy_z_th: float
    xcorr_search: int; xcorr_topk: int; xcorr_min_peak: float
    ncc_min: float
    ltf_off_sweep: int; max_ofdm_syms_probe: int; max_ofdm_syms_cap: int
    kp: float; ki: float

    ref_seed: int; ref_len: int; chunk_bytes: int
    save_dir: str; save_npz: bool; mode: str
    save_fail_npz: bool
    fail_save_ncc_min: float; fail_save_z_min: float; fail_npz_prob: float

    rx_gain_sweep: list
    gain_step_s: float
    max_caps: int

    # Step 8/9 additions
    gate_model_path: str
    gate_threshold: float
    bg_save_prob: float
    hard_z: float; hard_ncc: float; hard_gate_p: float
    mid_z: float;  mid_ncc: float;  mid_gate_p: float; mid_prob: float

    # Step 10 additions
    use_mmse: bool        # True → MMSE equalization; False → ZF (step8/9)
    pilot_weight: float   # 0.0 = no pilot correction, 1.0 = full (step9)
    auto_z_th: bool       # True → scale energy_z_th by MOD_PAPR_SCALE


# ─────────────────────────────────────────────────────────────────────────────
# RX acquisition worker
# ─────────────────────────────────────────────────────────────────────────────
def rx_acq_worker(
    stop_ev: threading.Event,
    q: "queue.Queue[np.ndarray]",
    cfg: RxConfig,
    current_gain: list,
):
    import adi
    sdr = adi.Pluto(uri=cfg.uri)
    try:
        sdr.rx_destroy_buffer()
    except Exception:
        pass
    sdr.sample_rate             = int(cfg.fs)
    sdr.rx_lo                   = int(cfg.fc)
    sdr.rx_rf_bandwidth         = int(cfg.rx_bw)
    sdr.gain_control_mode_chan0  = "manual"
    sdr.rx_hardwaregain_chan0    = float(cfg.rx_gain)
    sdr.rx_buffer_size          = int(cfg.rx_buf)
    sdr.rx_enabled_channels     = [0]
    try:
        if hasattr(sdr, "_rxadc") and sdr._rxadc is not None:
            sdr._rxadc.set_kernel_buffers_count(int(cfg.kernel_buffers))
    except Exception:
        pass
    for _ in range(4):
        sdr.rx()

    sweep       = cfg.rx_gain_sweep
    gain_idx    = 0
    sweep_start = time.time()
    if sweep:
        sdr.rx_hardwaregain_chan0 = float(sweep[0])
        current_gain[0]          = float(sweep[0])
        print(f"[RX] gain sweep start → {sweep[0]} dB")

    print("[RX] acq worker started. rx_buf =", cfg.rx_buf)
    try:
        while not stop_ev.is_set():
            if sweep and (time.time() - sweep_start) >= cfg.gain_step_s:
                gain_idx    = (gain_idx + 1) % len(sweep)
                new_g       = float(sweep[gain_idx])
                sdr.rx_hardwaregain_chan0 = new_g
                current_gain[0]          = new_g
                sweep_start = time.time()
                print(f"[RX] gain sweep → {new_g} dB")

            x = sdr.rx().astype(np.complex64)
            if np.median(np.abs(x)) > 100:
                x = x / (2 ** 14)
            x = x - np.mean(x)
            q.put(x, block=True)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            sdr.rx_destroy_buffer()
        except Exception:
            pass
        print("[RX] acq worker stopped.")


# ─────────────────────────────────────────────────────────────────────────────
# DSP thread
# ─────────────────────────────────────────────────────────────────────────────
def dsp_thread(
    stop_ev: threading.Event,
    q: "queue.Queue[np.ndarray]",
    cfg: RxConfig,
    current_gain: list,
):
    os.makedirs(cfg.save_dir, exist_ok=True)
    good_dir = os.path.join(cfg.save_dir, "good_packets")
    os.makedirs(good_dir, exist_ok=True)

    # ── Neural gate (optional) ────────────────────────────────────────────────
    inferencer: Optional[GateInferencer] = None
    if cfg.gate_model_path:
        try:
            # Auto-detect model type from checkpoint keys
            _ckpt_keys = torch.load(cfg.gate_model_path, map_location="cpu",
                                     weights_only=False).keys() if TORCH_OK else []
            if "tasks" in _ckpt_keys:
                inferencer = MultiTaskInferencer(cfg.gate_model_path)
            else:
                inferencer = GateInferencer(cfg.gate_model_path)
            thr = (cfg.gate_threshold if cfg.gate_threshold > 0.0
                   else inferencer.recommended_threshold)
        except Exception as exc:
            print(f"[Gate] WARNING: could not load {cfg.gate_model_path}: {exc}")
            thr = cfg.gate_threshold if cfg.gate_threshold > 0.0 else 0.5
    else:
        thr = cfg.gate_threshold if cfg.gate_threshold > 0.0 else 0.5

    mix_logger = MixtureLogger(
        gate_threshold    = thr,
        fail_save_ncc_min = cfg.fail_save_ncc_min,
        fail_save_z_min   = cfg.fail_save_z_min,
        fail_npz_prob     = cfg.fail_npz_prob,
        bg_save_prob      = cfg.bg_save_prob,
        has_gate          = inferencer is not None,
        hard_z            = cfg.hard_z,
        hard_ncc          = cfg.hard_ncc,
        hard_gate_p       = cfg.hard_gate_p,
        mid_z             = cfg.mid_z,
        mid_ncc           = cfg.mid_ncc,
        mid_gate_p        = cfg.mid_gate_p,
        mid_prob          = cfg.mid_prob,
    )

    # Step 10: effective energy z-threshold (modulation-aware scaling)
    eff_z_th = cfg.energy_z_th
    if cfg.auto_z_th:
        scale    = MOD_PAPR_SCALE.get(cfg.modulation, 1.0)
        eff_z_th = cfg.energy_z_th * scale
        print(f"[RX] auto_z_th: {cfg.energy_z_th:.1f} × {scale:.2f} = {eff_z_th:.2f}"
              f"  ({cfg.modulation})")

    # ── Non-blocking IO thread for NPZ saves ─────────────────────────────────
    _npz_q: "queue.Queue" = queue.Queue(maxsize=256)

    def _io_worker():
        while True:
            item = _npz_q.get()
            if item is None:          # sentinel → exit
                break
            try:
                _save_npz(**item)
            except Exception as exc:
                print(f"[IO] NPZ save error: {exc}")
            finally:
                _npz_q.task_done()

    _io_thread = threading.Thread(target=_io_worker, daemon=True, name="npz-io")
    _io_thread.start()

    def enqueue_npz(**kwargs):
        try:
            _npz_q.put_nowait(kwargs)
        except queue.Full:
            print("[IO] NPZ queue full – dropping save")

    stf_ref       = create_schmidl_cox_stf(cfg.stf_repeats).astype(np.complex64)
    stf_ref_e     = float(np.sqrt(np.sum(np.abs(stf_ref) ** 2)) + 1e-12)
    stf_len       = stf_ref.shape[0]
    ltf_ref_full, ltf_freq_ref = create_ltf_ref(cfg.ltf_symbols)
    ltf_td_ref    = ltf_ref_full[:SYMBOL_LEN].astype(np.complex64)

    const_tbl  = make_constellation(cfg.modulation)
    bps        = cfg.bps
    rot_cands  = get_rotation_candidates(cfg.modulation)

    ref_expected_frames: Optional[list] = None
    if cfg.ref_len > 0:
        rng        = np.random.default_rng(cfg.ref_seed)
        ref_data   = rng.integers(0, 256, size=cfg.ref_len, dtype=np.uint8).tobytes()
        cs         = cfg.chunk_bytes if cfg.chunk_bytes > 0 else cfg.ref_len
        ref_chunks = [ref_data[i: i + cs] for i in range(0, len(ref_data), cs)]
        total      = len(ref_chunks)
        ref_expected_frames = []
        for seq, ch in enumerate(ref_chunks):
            fb = _build_packet_bytes(seq, total, ch)
            ref_expected_frames.append((seq, fb))
        print(f"[RX] BER mode: {cfg.ref_len}B ref payload, {total} packets, seed={cfg.ref_seed}")

    ring          = RingBuffer(cfg.ring_size)
    samples_since = 0
    cap = 0; good = 0

    # CSV – gate_p, save_tag, mt_mod_p, mt_snr_pred
    csv_path = os.path.join(cfg.save_dir, "captures.csv")
    fcsv     = open(csv_path, "w", newline="")
    writer   = csv.DictWriter(fcsv, fieldnames=[
        "cap", "status", "reason", "peak",
        "p10", "eg_th", "maxe", "med", "mad", "z",
        "xc_best_peak", "xc_best_idx",
        "ncc_best", "ncc_best_idx",
        "stf_idx", "ltf_start", "payload_start",
        "probe_evm", "cfo_hz", "snr_db",
        "seq", "payload_len",
        "modulation", "bps", "rx_gain",
        "ber", "n_bits", "n_bit_errors",
        "gate_p", "save_tag",
    ])
    writer.writeheader()
    fcsv.flush()

    gain_ber: dict = {}

    def _record_ber(rx_gain: float, n_bits: int, n_errors: int):
        k = round(float(rx_gain), 2)
        if k not in gain_ber:
            gain_ber[k] = {"n_bits": 0, "n_errors": 0, "n_pkts": 0}
        gain_ber[k]["n_bits"]   += n_bits
        gain_ber[k]["n_errors"] += n_errors
        gain_ber[k]["n_pkts"]   += 1

    # ── inner demod ───────────────────────────────────────────────────────────
    def try_demod_at(rxw: np.ndarray, stf_idx: int):
        ltf0 = stf_idx + stf_len

        P    = N_FFT // 2
        sc_s = stf_idx
        sc_e = min(rxw.shape[0] - P, stf_idx + stf_len - P)
        if sc_e > sc_s:
            R       = np.sum(rxw[sc_s + P: sc_e + P].astype(np.complex64) *
                             np.conj(rxw[sc_s: sc_e].astype(np.complex64)))
            cfo_est = float(np.angle(R)) * cfg.fs / (2.0 * np.pi * P)
        else:
            cfo_est = 0.0

        n_arr   = np.arange(rxw.shape[0], dtype=np.float32)
        rxw_cfo = (rxw * np.exp(-1j * 2.0 * np.pi * (cfo_est / cfg.fs) * n_arr)).astype(np.complex64)

        search_s = max(0, ltf0 - cfg.ltf_off_sweep)
        search_e = min(rxw.shape[0] - SYMBOL_LEN, ltf0 + cfg.ltf_off_sweep)
        if search_e > search_s and NUMBA_OK:
            sb        = rxw_cfo[search_s: search_e + SYMBOL_LEN].astype(np.complex64)
            corr_ltf  = xcorr_mag_valid(sb, ltf_td_ref)
            ltf_start = search_s + int(np.argmax(corr_ltf)) if corr_ltf.size > 0 else ltf0
        else:
            ltf_start = ltf0

        payload_start = ltf_start + cfg.ltf_symbols * SYMBOL_LEN
        pilot_vals    = np.array([1, 1, 1, -1], dtype=np.complex64)
        rxw_cfo_conj  = np.conj(rxw_cfo)
        bpos          = N_DATA * bps

        best_ok = False; best_reason = "bad_magic"; best_seq = -1
        best_payload = b""; best_evm = 1e9; best_diag = {}
        best_n_bits = 0; best_n_errors = 0

        for conj_comp in [False, True]:
            rxw_proc = rxw_cfo_conj if conj_comp else rxw_cfo

            H, snr_db_arr = channel_estimate_from_ltf(
                rxw_proc, ltf_start, ltf_freq_ref, cfg.ltf_symbols
            )
            if H is None:
                continue
            snr_mean = float(np.mean(snr_db_arr)) if snr_db_arr is not None else 0.0

            ds_per_sym    = []
            pilot_ph_list = []
            evm_list      = []
            h_per_sym     = []   # per-symbol residual H for NPZ / analysis

            for si in range(cfg.max_ofdm_syms_probe):
                Y = extract_ofdm_symbol(rxw_proc, payload_start + si * SYMBOL_LEN)
                if Y is None:
                    break

                # ── Stage 1: bulk channel equalization ───────────────────────
                if cfg.use_mmse and snr_db_arr is not None:
                    snr_lin  = 10.0 ** (snr_db_arr / 10.0)
                    h_sq     = np.abs(H[USED_BINS]) ** 2
                    noise_var = float(np.mean(h_sq / (snr_lin + 1e-10)))
                    Ye = equalize_mmse(Y, H, noise_var, USED_BINS)
                else:
                    Ye = Y.copy()
                    for k in range(-26, 27):
                        if k == 0: continue
                        idx = sc_to_bin(k)
                        if np.abs(H[idx]) > 1e-6:
                            Ye[idx] = Ye[idx] / (H[idx] + 1e-12)

                pilot_sign = 1 if (si % 2 == 0) else -1

                # ── Stage 2: per-symbol residual correction (pilot_weight) ────
                H_res = pilot_residual_correction(Ye, pilot_sign, pilot_vals)
                h_per_sym.append(H_res.copy())
                if cfg.pilot_weight > 0.0:
                    w       = cfg.pilot_weight
                    H_blend = ((1.0 - w) * np.ones(N_FFT, dtype=np.complex64)
                               + w * H_res)
                    for k in range(-26, 27):
                        if k == 0: continue
                        idx = sc_to_bin(k)
                        if np.abs(H_blend[idx]) > 1e-6:
                            Ye[idx] = Ye[idx] / (H_blend[idx] + 1e-12)

                # ── Stage 3: scalar phase correction (same as step8) ─────────
                rp  = Ye[PILOT_BINS]
                ph  = float(np.angle(np.sum(rp * np.conj(pilot_sign * pilot_vals))))
                Ye *= np.exp(-1j * ph).astype(np.complex64)
                pilot_ph_list.append(ph)

                ds = Ye[DATA_BINS]
                ds_per_sym.append(ds)
                evm_list.append(mod_evm(ds, const_tbl))

            if not ds_per_sym:
                continue
            data_syms_all = np.concatenate(ds_per_sym).astype(np.complex64)
            cur_evm       = float(np.mean(evm_list)) if evm_list else 1e9

            for rot in rot_cands:
                syms_rot  = data_syms_all * rot
                bits_raw  = symbols_to_bits(syms_rot, const_tbl, bps)
                bits_mv   = majority_vote(bits_raw, cfg.repeat)
                bits_ds   = scramble_bits(bits_mv)
                bb        = bits_to_bytes(bits_ds)
                ok, reason, seq, payload = parse_packet_bytes(bb)
                if ok:
                    pkt_bytes_count = 4 + 2 + 2 + 2 + len(payload) + 4
                    pkt_syms_needed = int(np.ceil(pkt_bytes_count * 8 / bpos))
                    n_sym_cap       = min(pkt_syms_needed, len(ds_per_sym))
                    pkt_evm         = float(np.mean(evm_list[:n_sym_cap])) if evm_list else cur_evm

                    n_bits = 0; n_errors = 0
                    if ref_expected_frames is not None:
                        seq_mod = seq % len(ref_expected_frames)
                        _, ref_fb = ref_expected_frames[seq_mod]
                        ref_bits_tx_frame = scramble_bits(bits_from_bytes(ref_fb))
                        n_compare  = min(len(bits_mv), len(ref_bits_tx_frame))
                        n_bits     = int(n_compare)
                        n_errors   = int(np.sum(bits_mv[:n_compare] != ref_bits_tx_frame[:n_compare]))

                    best_ok       = True
                    best_reason   = reason
                    best_seq      = seq
                    best_payload  = payload
                    best_evm      = pkt_evm
                    best_n_bits   = n_bits
                    best_n_errors = n_errors
                    best_diag     = {
                        "ltf_start": ltf_start, "payload_start": payload_start,
                        "probe_evm": pkt_evm,   "cfo_hz": cfo_est,
                        "snr_db":    snr_mean,
                        "H":         H.copy(),
                        "snr_db_sc": (snr_db_arr.astype(np.float32)
                                      if snr_db_arr is not None else None),
                        "evm_per_sym":      np.array(evm_list[:n_sym_cap], dtype=np.float32),
                        "pilot_phase_err":  np.array(pilot_ph_list[:n_sym_cap], dtype=np.float32),
                        "Yeq_data":         np.stack(ds_per_sym[:n_sym_cap]).astype(np.complex64),
                        "H_per_sym":        np.stack(h_per_sym[:n_sym_cap]).astype(np.complex64),
                    }
                    break
                elif best_reason == "bad_magic":
                    best_reason = reason

            if not best_ok and cur_evm < best_evm:
                best_evm = cur_evm
                best_diag = {
                    "ltf_start": ltf_start, "payload_start": payload_start,
                    "probe_evm": cur_evm,   "cfo_hz": cfo_est,
                    "snr_db":    snr_mean,
                    "H":         H.copy(),
                    "snr_db_sc": (snr_db_arr.astype(np.float32)
                                  if snr_db_arr is not None else None),
                    "evm_per_sym":     np.array(evm_list, dtype=np.float32),
                    "pilot_phase_err": np.array(pilot_ph_list, dtype=np.float32),
                    "Yeq_data":        (np.stack(ds_per_sym).astype(np.complex64)
                                        if ds_per_sym else None),
                    "H_per_sym":       (np.stack(h_per_sym).astype(np.complex64)
                                        if h_per_sym else None),
                }

            if best_ok:
                break

        if not best_diag:
            best_diag = {
                "ltf_start": ltf_start, "payload_start": payload_start,
                "probe_evm": 0.0, "cfo_hz": cfo_est, "snr_db": 0.0,
            }
        return best_ok, best_reason, best_seq, best_payload, best_diag, best_n_bits, best_n_errors

    # ── main DSP loop ─────────────────────────────────────────────────────────
    print(f"[RX] DSP thread started.  NUMBA_OK={NUMBA_OK}  mod={cfg.modulation}  bps={bps}")
    print(f"[RX] gate={'active thr=' + f'{thr:.4f}' if inferencer else 'none'}  "
          f"bg_save_prob={cfg.bg_save_prob:.3f}")
    try:
        while not stop_ev.is_set():
            try:
                x = q.get(timeout=0.2)
            except queue.Empty:
                continue
            ring.push(x)
            samples_since += x.shape[0]
            if samples_since < cfg.proc_hop:
                continue
            samples_since = 0

            if cfg.max_caps > 0 and cap >= cfg.max_caps:
                stop_ev.set()
                break

            rxw = ring.get_window(cfg.proc_window)
            if rxw.size == 0:
                if cap == 0:
                    print(f"[RX] DBG ring not ready: w={ring.w} filled={ring.filled}", flush=True)
                continue
            cap += 1
            rx_gain_now = float(current_gain[0])
            peak        = float(np.max(np.abs(rxw)))

            e = (moving_energy(rxw.astype(np.complex64), cfg.energy_win)
                 if NUMBA_OK else np.convolve(np.abs(rxw) ** 2,
                                              np.ones(cfg.energy_win) / cfg.energy_win,
                                              mode="valid"))
            if e.size == 0:
                continue

            p10   = float(np.percentile(e, 10))
            maxe  = float(np.max(e))
            eg_th = float(p10 * cfg.energy_mult)
            med   = float(np.median(e))
            mad   = float(np.median(np.abs(e - med))) + 1e-12
            z     = float((maxe - med) / (1.4826 * mad))

            # ── Energy-low: potential background save ────────────────────────
            if z < eff_z_th:
                bg_tag   = mix_logger.decide_bg()
                energy_ds_bg = e[::16].astype(np.float32)
                writer.writerow({
                    "cap": cap, "status": "skip", "reason": "energy_low", "peak": peak,
                    "p10": p10, "eg_th": eg_th, "maxe": maxe,
                    "med": med, "mad": mad, "z": z,
                    "xc_best_peak": 0.0, "xc_best_idx": -1,
                    "ncc_best": 0.0, "ncc_best_idx": -1,
                    "stf_idx": -1, "ltf_start": -1, "payload_start": -1,
                    "probe_evm": "", "cfo_hz": "", "snr_db": "",
                    "seq": "", "payload_len": "",
                    "modulation": cfg.modulation, "bps": bps,
                    "rx_gain": rx_gain_now,
                    "ber": "", "n_bits": "", "n_bit_errors": "",
                    "gate_p": "", "save_tag": bg_tag,
                })
                if bg_tag == "fail_bg" and cfg.save_npz:
                    enqueue_npz(
                        path=os.path.join(cfg.save_dir, f"cap_{cap:06d}_bg.npz"),
                        rxw=rxw, corr=np.zeros(1, dtype=np.float32),
                        corr_norm=np.zeros(1, dtype=np.float32),
                        top_idx=np.zeros(1, dtype=np.int32),
                        topk_ncc=np.zeros(1, dtype=np.float32),
                        e=energy_ds_bg,
                        meta={
                            "cap": cap, "status": "skip", "reason": "bg_sample",
                            "peak": peak, "p10": p10, "maxe": maxe,
                            "med": med, "mad": mad, "z": z,
                            "gate_ratio": float(maxe / (p10 + 1e-12)),
                            "modulation": cfg.modulation, "bps": bps,
                            "rx_gain": rx_gain_now,
                        },
                        diag={},
                    )
                continue

            # ── XCorr & NCC ──────────────────────────────────────────────────
            search_len = min(cfg.xcorr_search, rxw.size)
            xs         = rxw[:search_len].astype(np.complex64)
            corr       = xcorr_mag_valid(xs, stf_ref) if NUMBA_OK else _numpy_xcorr(xs, stf_ref)
            if corr.size == 0:
                continue
            corr_norm  = corr / stf_ref_e

            k       = min(cfg.xcorr_topk, corr_norm.size)
            top_idx = np.argpartition(corr_norm, -k)[-k:]
            top_idx = top_idx[np.argsort(corr_norm[top_idx])[::-1]]
            best_xc_pk = float(corr_norm[top_idx[0]])

            topk_ncc = np.zeros(k, dtype=np.float32)
            for ti, ci in enumerate(top_idx):
                xseg_e = float(np.sqrt(np.sum(np.abs(xs[int(ci): int(ci) + stf_len]) ** 2))) + 1e-12
                topk_ncc[ti] = corr[int(ci)] / (stf_ref_e * xseg_e)

            ncc_best_pos = int(np.argmax(topk_ncc))
            ncc_best     = float(topk_ncc[ncc_best_pos])
            ncc_best_idx = int(top_idx[ncc_best_pos])

            energy_ds = e[::16].astype(np.float32)

            # ── Neural gate (never blocks decode) ────────────────────────────
            gate_p = np.nan
            if inferencer is not None:
                gate_meta = {
                    "peak": peak, "p10": p10, "eg_th": eg_th, "maxe": maxe,
                    "med": med, "mad": mad, "z": z,
                    "xc_best_peak": best_xc_pk, "xc_best_idx": int(top_idx[0]),
                    "ncc_best": ncc_best, "ncc_best_idx": ncc_best_idx,
                    "rx_gain": rx_gain_now, "bps": bps,
                    # probe_evm / cfo_hz / snr_db not yet available → NaN in model
                }
                try:
                    gate_p = inferencer.predict(corr_norm, topk_ncc, energy_ds, rxw, gate_meta)
                    # Capture extra MT predictions for meta_json logging
                    if isinstance(inferencer, MultiTaskInferencer):
                        gate_meta["mt_mod_p"]    = inferencer.last_mod_p
                        gate_meta["mt_snr_pred"] = inferencer.last_snr_pred
                except Exception as exc:
                    print(f"[Gate] predict error: {exc}")

            # ── Demod ────────────────────────────────────────────────────────
            best_ok = False; best_reason = "no_try"; best_seq = -1
            best_payload = b""; best_diag = {}; best_stf = -1
            best_n_bits = 0; best_n_errors = 0

            for ti, cand in enumerate(top_idx):
                if topk_ncc[ti] < cfg.ncc_min:
                    continue
                ok, reason, seq, payload, diag, nb, ne = try_demod_at(rxw, int(cand))
                if ok:
                    best_ok = True; best_reason = "ok"; best_seq = int(seq)
                    best_payload = payload; best_diag = diag; best_stf = int(cand)
                    best_n_bits = nb; best_n_errors = ne
                    break
                elif best_stf < 0:
                    best_reason = reason; best_diag = diag; best_stf = int(cand)

            status      = "ok" if best_ok else "no_crc"
            payload_len = len(best_payload) if best_ok else 0
            ber_val     = (best_n_errors / best_n_bits) if best_n_bits > 0 else ""

            # ── Mixture logging decision ─────────────────────────────────────
            gate_p_val = float(gate_p) if np.isfinite(gate_p) else np.nan
            if best_ok:
                save_tag = mix_logger.decide_ok()
            else:
                save_tag = mix_logger.decide_fail(z, ncc_best, gate_p_val if np.isfinite(gate_p_val) else 0.0)

            writer.writerow({
                "cap": cap, "status": status, "reason": best_reason, "peak": peak,
                "p10": p10, "eg_th": eg_th, "maxe": maxe,
                "med": med, "mad": mad, "z": z,
                "xc_best_peak": best_xc_pk, "xc_best_idx": int(top_idx[0]),
                "ncc_best": ncc_best, "ncc_best_idx": ncc_best_idx,
                "stf_idx": best_stf,
                "ltf_start":     int(best_diag.get("ltf_start",     -1)),
                "payload_start": int(best_diag.get("payload_start", -1)),
                "probe_evm":  float(best_diag.get("probe_evm", 0.0)),
                "cfo_hz":     float(best_diag.get("cfo_hz",    0.0)),
                "snr_db":     float(best_diag.get("snr_db",    0.0)),
                "seq":        (best_seq    if best_ok else ""),
                "payload_len": (payload_len if best_ok else ""),
                "modulation": cfg.modulation, "bps": bps,
                "rx_gain":    rx_gain_now,
                "ber":        (float(ber_val) if ber_val != "" else ""),
                "n_bits":     (best_n_bits    if best_ok else ""),
                "n_bit_errors": (best_n_errors if best_ok else ""),
                "gate_p":   (f"{gate_p_val:.6f}" if np.isfinite(gate_p_val) else ""),
                "save_tag": save_tag,
            })

            if best_ok:
                good += 1
                if best_n_bits > 0:
                    _record_ber(rx_gain_now, best_n_bits, best_n_errors)
                outp = os.path.join(good_dir, f"seq_{best_seq:08d}.bin")
                with open(outp, "wb") as f:
                    f.write(best_payload)

                ber_str = f" ber={float(ber_val):.4f}" if ber_val != "" else ""
                gp_str  = f" gate={gate_p_val:.3f}" if np.isfinite(gate_p_val) else ""
                print(f"[RX] OK  seq={best_seq}  {payload_len}B  "
                      f"gain={rx_gain_now:.0f}dB  z={z:.1f}  ncc={ncc_best:.3f}  "
                      f"cfo={best_diag.get('cfo_hz', 0):.0f}Hz  "
                      f"snr={best_diag.get('snr_db', 0):.1f}dB  "
                      f"evm={best_diag.get('probe_evm', 0):.3f}{ber_str}{gp_str}")
                try:
                    ps = best_payload.decode("utf-8")
                    print(f"     payload (utf8): {ps!r}")
                except Exception:
                    print(f"     payload (hex):  {best_payload[:32].hex()}")

                if cfg.save_npz and save_tag == "ok":
                    enqueue_npz(
                        path=os.path.join(cfg.save_dir, f"cap_{cap:06d}_ok.npz"),
                        rxw=rxw, corr=corr, corr_norm=corr_norm,
                        top_idx=top_idx, topk_ncc=topk_ncc, e=energy_ds,
                        meta={
                            "cap": cap, "status": "ok", "reason": "ok",
                            "seq": best_seq,
                            "peak": peak, "p10": p10, "maxe": maxe,
                            "med": med, "mad": mad, "z": z,
                            "gate_ratio": float(maxe / (p10 + 1e-12)),
                            "xc_best_peak": best_xc_pk, "xc_best_idx": int(top_idx[0]),
                            "ncc_best": ncc_best, "ncc_best_idx": ncc_best_idx,
                            "stf_idx": best_stf,
                            "ltf_start":     best_diag.get("ltf_start",     -1),
                            "payload_start": best_diag.get("payload_start", -1),
                            "probe_evm": best_diag.get("probe_evm", 0.0),
                            "cfo_hz":    best_diag.get("cfo_hz",    0.0),
                            "snr_db":    best_diag.get("snr_db",    0.0),
                            "modulation": cfg.modulation, "bps": bps,
                            "repeat": cfg.repeat, "rx_gain": rx_gain_now,
                            "n_bits": best_n_bits, "n_bit_errors": best_n_errors,
                            "gate_p": (float(gate_p_val) if np.isfinite(gate_p_val) else None),
                            "save_tag": "ok",
                        },
                        diag=best_diag,
                    )

            elif save_tag in ("fail_hard", "fail_mid", "fail_bg") and cfg.save_npz:
                enqueue_npz(
                    path=os.path.join(cfg.save_dir, f"cap_{cap:06d}_fail.npz"),
                    rxw=rxw, corr=corr, corr_norm=corr_norm,
                    top_idx=top_idx, topk_ncc=topk_ncc, e=energy_ds,
                    meta={
                        "cap": cap, "status": status, "reason": best_reason,
                        "save_tag": save_tag,
                        # key signal quality indicators for offline white-box analysis
                        "peak": peak, "p10": p10, "maxe": maxe,
                        "med": med, "mad": mad, "z": z,
                        "gate_ratio": float(maxe / (p10 + 1e-12)),
                        "gate_p": (float(gate_p_val) if np.isfinite(gate_p_val) else None),
                        # xcorr / NCC
                        "xc_best_peak": best_xc_pk, "xc_best_idx": int(top_idx[0]),
                        "ncc_best": ncc_best, "ncc_best_idx": ncc_best_idx,
                        # frame alignment
                        "stf_idx": best_stf,
                        "ltf_start":     best_diag.get("ltf_start",     -1),
                        "payload_start": best_diag.get("payload_start", -1),
                        # post-demod diagnostics (NaN when demod not attempted)
                        "probe_evm": best_diag.get("probe_evm", None),
                        "cfo_hz":    best_diag.get("cfo_hz",    None),
                        "snr_db":    best_diag.get("snr_db",    None),
                        "modulation": cfg.modulation, "bps": bps,
                        "repeat": cfg.repeat, "rx_gain": rx_gain_now,
                    },
                    diag=best_diag,
                )

            if cap % 20 == 0:
                fcsv.flush()

    except KeyboardInterrupt:
        pass
    finally:
        fcsv.flush(); fcsv.close()
        _save_run_summary(cfg, cap, good, gain_ber)
        # drain IO queue before exit
        _npz_q.join()
        _npz_q.put(None)   # sentinel
        _io_thread.join(timeout=10)
        print(f"[RX] DSP thread stopped.  cap={cap}  good={good}")


# ─────────────────────────────────────────────────────────────────────────────
# NPZ helper  (shared by OK, FAIL, and BG paths)
# ─────────────────────────────────────────────────────────────────────────────
def _save_npz(
    path: str,
    rxw: np.ndarray,
    corr: np.ndarray,
    corr_norm: np.ndarray,
    top_idx: np.ndarray,
    topk_ncc: np.ndarray,
    e: np.ndarray,
    meta: dict,
    diag: dict,
):
    kw = dict(
        rxw=rxw.astype(np.complex64),
        corr=corr.astype(np.float32),
        corr_norm=corr_norm.astype(np.float32),
        topk_idx=top_idx.astype(np.int32),
        topk_corr_norm=corr_norm[top_idx].astype(np.float32) if top_idx.size <= corr_norm.size else np.zeros(top_idx.size, np.float32),
        topk_ncc=topk_ncc.astype(np.float32),
        energy_ds=e.astype(np.float32),
        meta_json=np.bytes_(json.dumps(meta).encode()),
    )
    H = diag.get("H")
    if H is not None:
        kw["H"] = H.astype(np.complex64)
    snr_sc = diag.get("snr_db_sc")
    if snr_sc is not None:
        kw["snr_db_sc"] = snr_sc.astype(np.float32)
    evm_s = diag.get("evm_per_sym")
    if evm_s is not None:
        kw["evm_per_sym"] = evm_s.astype(np.float32)
    ph_s = diag.get("pilot_phase_err")
    if ph_s is not None:
        kw["pilot_phase_err"] = ph_s.astype(np.float32)
    Yeq = diag.get("Yeq_data")
    if Yeq is not None:
        kw["Yeq_data"] = Yeq.astype(np.complex64)
    np.savez_compressed(path, **kw)


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────
def _numpy_xcorr(xs: np.ndarray, stf_ref: np.ndarray) -> np.ndarray:
    L    = stf_ref.size
    nout = xs.size - L + 1
    if nout <= 0:
        return np.zeros(0, dtype=np.float32)
    hc   = np.conj(stf_ref)
    corr = np.zeros(nout, dtype=np.float32)
    for i in range(nout):
        corr[i] = np.abs(np.vdot(hc, xs[i: i + L]))
    return corr


def _build_packet_bytes(seq: int, total: int, payload: bytes) -> bytes:
    hdr  = (MAGIC
            + int(seq).to_bytes(2, "little")
            + int(total).to_bytes(2, "little")
            + int(len(payload)).to_bytes(2, "little"))
    body = hdr + payload
    crc  = zlib.crc32(body) & 0xFFFFFFFF
    return body + int(crc).to_bytes(4, "little")


def _save_run_summary(cfg: RxConfig, cap: int, good: int, gain_ber: dict):
    ber_per_gain = {}
    for g, v in gain_ber.items():
        nb = v["n_bits"]; ne = v["n_errors"]
        ber_per_gain[str(g)] = {
            "rx_gain_dB": g, "modulation": cfg.modulation,
            "n_bits": nb, "n_errors": ne,
            "ber": ne / nb if nb > 0 else None,
            "n_pkts": v["n_pkts"],
        }
    summary = {
        "run_id":      os.path.basename(cfg.save_dir),
        "modulation":  cfg.modulation, "bps": cfg.bps,
        "fc_MHz":      cfg.fc / 1e6,   "fs_MHz": cfg.fs / 1e6,
        "rx_gain_dB":  cfg.rx_gain,    "gain_sweep": cfg.rx_gain_sweep,
        "ref_seed":    cfg.ref_seed,   "ref_len": cfg.ref_len,
        "total_cap":   cap,            "good_pkts": good,
        "decode_rate": good / max(cap, 1),
        "ber_per_gain": ber_per_gain,
        "gate_model":  cfg.gate_model_path,
        "gate_threshold": cfg.gate_threshold,
        "end_time":    datetime.now().isoformat(),
    }
    path = os.path.join(cfg.save_dir, "run_summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[RX] run summary → {path}")
    if ber_per_gain:
        print("[RX] BER per gain level:")
        for g, v in sorted(ber_per_gain.items(), key=lambda x: float(x[0])):
            ber = v["ber"]
            print(f"     rx_gain={v['rx_gain_dB']:6.1f} dB  "
                  f"mod={v['modulation']:6s}  "
                  f"BER={ber:.4e}  n_bits={v['n_bits']}  pkts={v['n_pkts']}")


# ─────────────────────────────────────────────────────────────────────────────
# Sweep DSP thread (spectrum analyser mode)
# ─────────────────────────────────────────────────────────────────────────────
def dsp_sweep_thread(stop_ev, q, cfg, current_gain):
    import scipy.signal
    os.makedirs(cfg.save_dir, exist_ok=True)
    ring          = RingBuffer(cfg.ring_size)
    samples_since = 0; cap = 0
    print("[RX] Sweep thread started.")
    try:
        while not stop_ev.is_set():
            try:
                x = q.get(timeout=0.2)
            except queue.Empty:
                continue
            ring.push(x); samples_since += x.shape[0]
            if samples_since < cfg.proc_window:
                continue
            samples_since = 0; cap += 1
            rxw   = ring.get_window(cfg.proc_window)
            f, Px = scipy.signal.welch(rxw, fs=cfg.fs, return_onesided=False, nperseg=4096)
            f     = np.fft.fftshift(f); Px = np.fft.fftshift(Px)
            Pdb   = 10 * np.log10(Px + 1e-12)
            top3  = np.argsort(Pdb)[-3:][::-1]
            print(f"[Sweep] cap={cap}  "
                  + "  ".join(f"{f[i]/1e6:+.3f}MHz({Pdb[i]:.1f}dB)" for i in top3))
    except KeyboardInterrupt:
        pass
    finally:
        print("[RX] Sweep thread stopped. cap=", cap)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser("Step10 PHY RX – MMSE equalization + OTA robustness")
    ap.add_argument("--uri", required=True)
    ap.add_argument("--fc",  type=float, required=True)
    ap.add_argument("--fs",  type=float, required=True)
    ap.add_argument("--rx_gain",  type=float, default=30.0)
    ap.add_argument("--rx_bw",   type=float, default=0.0)
    ap.add_argument("--rx_buf",  type=int,   default=131072)
    ap.add_argument("--kernel_buffers", type=int, default=4)

    ap.add_argument("--modulation", default="qpsk", choices=list(MOD_BPS.keys()))
    ap.add_argument("--repeat",      type=int, default=1, choices=[1, 2, 4])
    ap.add_argument("--stf_repeats", type=int, default=6)
    ap.add_argument("--ltf_symbols", type=int, default=4)

    ap.add_argument("--ring_size",   type=int, default=524288)
    ap.add_argument("--proc_window", type=int, default=262144)
    ap.add_argument("--proc_hop",    type=int, default=65536)
    ap.add_argument("--energy_win",  type=int,   default=512)
    ap.add_argument("--energy_mult", type=float, default=2.5)
    ap.add_argument("--energy_z_th", type=float, default=8.0,
                    help="MAD z-score threshold for energy gate")
    ap.add_argument("--xcorr_search",   type=int,   default=200000)
    ap.add_argument("--xcorr_topk",     type=int,   default=8)
    ap.add_argument("--xcorr_min_peak", type=float, default=0.2,
                    help="Kept for backward compat; gating uses --ncc_min")
    ap.add_argument("--ncc_min",        type=float, default=0.15,
                    help="Min NCC for a top-K candidate to enter demod")
    ap.add_argument("--ltf_off_sweep",  type=int,   default=16)
    ap.add_argument("--probe_syms",     type=int,   default=80,
                    help="Max OFDM symbols to decode per packet")
    ap.add_argument("--max_syms_cap",   type=int,   default=260)
    ap.add_argument("--kp", type=float, default=0.15)
    ap.add_argument("--ki", type=float, default=0.005)

    ap.add_argument("--ref_seed",    type=int, default=0)
    ap.add_argument("--ref_len",     type=int, default=0,
                    help="BER mode: PRNG payload length in bytes (must match TX)")
    ap.add_argument("--chunk_bytes", type=int, default=512)

    ap.add_argument("--rx_gain_sweep", type=str, default="",
                    help="Comma-separated RX gain levels for BER sweep (dB)")
    ap.add_argument("--gain_step_s", type=float, default=15.0)
    ap.add_argument("--max_caps", type=int, default=0,
                    help="Stop after this many DSP caps (0 = unlimited)")

    ap.add_argument("--out_root",   default="rf_stream_rx_runs")
    ap.add_argument("--save_npz",   action="store_true")
    ap.add_argument("--save_fail_npz",    action="store_true",
                    help="Enable FAIL/BG NPZ saving (also controlled by mixture logger)")
    ap.add_argument("--fail_save_ncc_min", type=float, default=0.15)
    ap.add_argument("--fail_save_z_min",   type=float, default=8.0)
    ap.add_argument("--fail_npz_prob",     type=float, default=1.0,
                    help="Subsampling probability for fail_gate saves (0–1)")
    ap.add_argument("--mode", default="packet", choices=["packet", "sweep"])

    # Step 8/9: neural gate + background saving
    ap.add_argument("--gate_model",     type=str,   default="",
                    help="Path to gate_model.pt from train_neural_gate.py")
    ap.add_argument("--gate_threshold", type=float, default=0.0,
                    help="Gate threshold (0 = use recommended_threshold from model)")
    ap.add_argument("--bg_save_prob",   type=float, default=0.05,
                    help="Probability of saving a background-noise window (fail_bg)")
    # 3-tier fail NPZ thresholds
    ap.add_argument("--hard_z",       type=float, default=200.0,
                    help="z threshold for hard (always-save) fail tier")
    ap.add_argument("--hard_ncc",     type=float, default=0.70,
                    help="NCC threshold for hard fail tier")
    ap.add_argument("--hard_gate_p",  type=float, default=0.80,
                    help="gate_p threshold for hard fail tier")
    ap.add_argument("--mid_z",        type=float, default=30.0,
                    help="z threshold for mid (mid_prob sampled) fail tier")
    ap.add_argument("--mid_ncc",      type=float, default=0.30,
                    help="NCC threshold for mid fail tier")
    ap.add_argument("--mid_gate_p",   type=float, default=0.30,
                    help="gate_p threshold for mid fail tier")
    ap.add_argument("--mid_prob",     type=float, default=0.5,
                    help="Save probability for mid fail tier")

    # Step 10: equalization mode, pilot weight, auto energy threshold
    ap.add_argument("--equalization", default="mmse", choices=["zf", "mmse"],
                    help="Equalization: mmse (default, step10) or zf (step8/9 compat)")
    ap.add_argument("--pilot_weight", type=float, default=0.5,
                    help="Pilot correction weight: 0.0=LTF only, 1.0=full pilot corr")
    ap.add_argument("--auto_z_th", action="store_true",
                    help="Scale energy_z_th by modulation PAPR (qam16→×0.65, bpsk→×1.2)")

    args = ap.parse_args()

    sweep_gains = ([float(x.strip()) for x in args.rx_gain_sweep.split(",") if x.strip()]
                   if args.rx_gain_sweep else [])
    bps    = MOD_BPS[args.modulation]
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out_root, run_id)
    os.makedirs(out_dir, exist_ok=True)

    cfg = RxConfig(
        uri=args.uri, fc=args.fc, fs=args.fs,
        rx_gain=sweep_gains[0] if sweep_gains else args.rx_gain,
        rx_bw=(args.rx_bw if args.rx_bw > 0 else args.fs * 1.2),
        rx_buf=args.rx_buf, kernel_buffers=args.kernel_buffers,
        repeat=args.repeat, stf_repeats=args.stf_repeats, ltf_symbols=args.ltf_symbols,
        modulation=args.modulation, bps=bps,
        ring_size=args.ring_size, proc_window=args.proc_window, proc_hop=args.proc_hop,
        energy_win=args.energy_win, energy_mult=args.energy_mult,
        energy_z_th=args.energy_z_th,
        xcorr_search=args.xcorr_search, xcorr_topk=args.xcorr_topk,
        xcorr_min_peak=args.xcorr_min_peak,
        ncc_min=args.ncc_min,
        ltf_off_sweep=args.ltf_off_sweep,
        max_ofdm_syms_probe=args.probe_syms, max_ofdm_syms_cap=args.max_syms_cap,
        kp=args.kp, ki=args.ki,
        ref_seed=args.ref_seed, ref_len=args.ref_len, chunk_bytes=args.chunk_bytes,
        save_dir=out_dir, save_npz=bool(args.save_npz), mode=args.mode,
        save_fail_npz=bool(args.save_fail_npz),
        fail_save_ncc_min=args.fail_save_ncc_min,
        fail_save_z_min=args.fail_save_z_min,
        fail_npz_prob=args.fail_npz_prob,
        rx_gain_sweep=sweep_gains, gain_step_s=args.gain_step_s,
        max_caps=args.max_caps,
        gate_model_path=args.gate_model,
        gate_threshold=args.gate_threshold,
        bg_save_prob=args.bg_save_prob,
        hard_z=args.hard_z, hard_ncc=args.hard_ncc, hard_gate_p=args.hard_gate_p,
        mid_z=args.mid_z,   mid_ncc=args.mid_ncc,   mid_gate_p=args.mid_gate_p,
        mid_prob=args.mid_prob,
        use_mmse=(args.equalization == "mmse"),
        pilot_weight=args.pilot_weight,
        auto_z_th=args.auto_z_th,
    )

    with open(os.path.join(out_dir, "rx_config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    print("\n" + "=" * 70)
    print("Streaming RX  (Step10 PHY – MMSE Equalization + OTA Robustness)")
    print("=" * 70)
    print(f"  out_dir={out_dir}  NUMBA_OK={NUMBA_OK}  TORCH_OK={TORCH_OK}")
    print(f"  modulation={cfg.modulation}  bps={bps}  rx_gain={cfg.rx_gain} dB")
    if sweep_gains:
        print(f"  gain_sweep={sweep_gains}  step={args.gain_step_s}s")
    if cfg.ref_len > 0:
        print(f"  BER mode: ref_seed={cfg.ref_seed}  ref_len={cfg.ref_len}B")
    print(f"  energy_z_th={cfg.energy_z_th}  auto_z_th={cfg.auto_z_th}  ncc_min={cfg.ncc_min}")
    print(f"  equalization={'mmse' if cfg.use_mmse else 'zf'}  pilot_weight={cfg.pilot_weight:.2f}")
    if cfg.gate_model_path:
        print(f"  gate_model={cfg.gate_model_path}  gate_thr={cfg.gate_threshold or 'auto'}")
    print(f"  bg_save_prob={cfg.bg_save_prob}  fail_npz_prob={cfg.fail_npz_prob}")
    print("=" * 70)

    current_gain = [float(cfg.rx_gain)]
    q       = queue.Queue(maxsize=32)
    stop_ev = threading.Event()

    t_acq = threading.Thread(target=rx_acq_worker, args=(stop_ev, q, cfg, current_gain), daemon=True)
    if cfg.mode == "sweep":
        t_dsp = threading.Thread(target=dsp_sweep_thread, args=(stop_ev, q, cfg, current_gain), daemon=True)
    else:
        t_dsp = threading.Thread(target=dsp_thread, args=(stop_ev, q, cfg, current_gain), daemon=True)

    t_acq.start()
    t_dsp.start()

    try:
        while not stop_ev.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        stop_ev.set()
        time.sleep(0.5)
        print("[RX] exit.")


if __name__ == "__main__":
    main()

"""
# ── Quick-start commands ──────────────────────────────────────────────────────

# QPSK BER sweep with neural gate (step8):
python3 rf_stream_rx_step9phy.py \
  --uri ip:192.168.2.2 --fc 2.3e9 --fs 3e6 \
  --modulation qpsk --ref_seed 42 --ref_len 512 --chunk_bytes 512 \
  --probe_syms 80 \
  --rx_gain_sweep "60,55,50,45,40,35,30,25" --gain_step_s 25 \
  --save_npz --save_fail_npz --fail_npz_prob 0.3 \
  --gate_model rf_stream/gate_model_v1/gate_model.pt \
  --bg_save_prob 0.05 \
  --out_root rf_stream/ber_sweep

# QAM16 BER sweep, lower TX gain for non-zero BER:
#   TX: --tx_gain -30
python3 rf_stream_rx_step9phy.py \
  --uri ip:192.168.2.2 --fc 2.3e9 --fs 3e6 \
  --modulation qam16 --ref_seed 42 --ref_len 512 --chunk_bytes 512 \
  --probe_syms 80 \
  --rx_gain_sweep "60,55,50,45,40,35,30,25" --gain_step_s 25 \
  --save_npz --save_fail_npz \
  --gate_model rf_stream/gate_model_v1/gate_model.pt \
  --bg_save_prob 0.05 \
  --out_root rf_stream/ber_sweep

# Background noise collection (no TX):
python3 rf_stream_rx_step9phy.py \
  --uri ip:192.168.2.2 --fc 2.3e9 --fs 3e6 \
  --modulation qpsk --ref_seed 42 --ref_len 512 \
  --rx_gain 60 --max_caps 1000 \
  --save_npz --bg_save_prob 1.0 \
  --out_root rf_stream/bg_noise

# Tune detection thresholds:
#   --energy_z_th 6.0   (lower → more sensitive, more false alarms)
#   --ncc_min 0.10      (lower → try weaker candidates)
#   --fail_npz_prob 0.1 (subsample FAIL saves to 10%)
#   --bg_save_prob 0.02 (save 2% of background windows)
"""
