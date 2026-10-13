#!/usr/bin/env python3
"""OTFS from scratch — a teaching script (NumPy + Matplotlib only).

Walks through the full OTFS chain and writes the figures used in the README tutorial:
  1. put QPSK symbols + one pilot on the delay-Doppler (DD) grid, modulate (ISFFT + Heisenberg),
  2. pass a 3-path doubly-selective channel (delay + Doppler per path),
  3. demodulate back to the DD grid, read the channel taps around the pilot,
  4. equalize with a DD-domain LMMSE, compare with OFDM under the same channel,
  5. use the same OTFS frame as a radar: range-Doppler map by matched filtering.

The full hardware receiver (sync, LDPC, PlutoSDR) lives in ../otfs/; this file keeps only the ideas.
Run:  python3 otfs_tutorial.py      ->  figs/*.png
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
QPSK = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2)


# ── 1. OTFS modulation / demodulation ────────────────────────────────────────────────────────────────
def isfft(X):
    """DD grid (M delay x N Doppler) -> TF grid (M subcarriers x N symbols)."""
    M, N = X.shape
    return np.fft.fft(np.fft.ifft(X, axis=1), axis=0) * np.sqrt(N / M)


def sfft(Xtf):
    """TF grid -> DD grid (inverse of isfft)."""
    M, N = Xtf.shape
    return np.fft.fft(np.fft.ifft(Xtf, axis=0), axis=1) * np.sqrt(M / N)


def otfs_mod(X):
    """ISFFT, then an OFDM modulator (IFFT per symbol), no CP per symbol; the frame gets one CP."""
    M, N = X.shape
    s_tf = np.fft.ifft(isfft(X), axis=0) * np.sqrt(M)      # Heisenberg transform, rectangular pulse
    return s_tf.reshape(-1, order="F")                     # symbol after symbol


def otfs_demod(r, M, N):
    """Wigner transform (FFT per symbol), then SFFT."""
    R = r.reshape(M, N, order="F")
    return sfft(np.fft.fft(R, axis=0) / np.sqrt(M))


# ── 2. channel: P paths, each with gain h, integer delay l (samples) and Doppler k (DD bins) ──────────
def channel(s, paths):
    """r[n] = sum_p h_p * s[n - l_p] * exp(j 2 pi k_p n / MN); the frame CP (>= max delay) makes the delay
    cyclic. Doppler bin k = k / (N T) Hz, i.e. k / (MN) cycles per sample."""
    MN = s.size
    n = np.arange(MN)
    return sum(h * np.roll(s, l) * np.exp(2j * np.pi * k * n / MN) for h, l, k in paths)


def awgn(x, noise_var, rng):
    return x + np.sqrt(noise_var / 2) * (rng.standard_normal(x.size) + 1j * rng.standard_normal(x.size))


# ── 3. frame with an embedded pilot and a guard region ──────────────────────────────────────────────
def make_frame(M, N, lp, kp, lmax, pilot_db, rng):
    """Guard = delay rows lp-lmax .. lp+lmax in EVERY Doppler column: any Doppler fits, and data never lands
    in the cells where the pilot's echoes are read."""
    mask = np.ones((M, N), bool)                           # True = data cell
    mask[np.arange(lp - lmax, lp + lmax + 1) % M, :] = False
    X = np.zeros((M, N), complex)
    sym = rng.integers(0, 4, mask.sum())
    X[mask] = QPSK[sym]
    pilot = 10 ** (pilot_db / 20)
    X[lp, kp] = pilot
    return X, mask, sym, pilot


def estimate_taps(Y, M, N, lp, kp, lmax, kmax, pilot, noise_var):
    """Read the received grid next to the pilot: a path (l, k) shows up at (lp + l, kp + k). A cell is a tap if
    it is 3 sigma above the noise; its complex gain is the reading divided by the reading a unit path would give
    (a known phase: the 'calibration' is computed by passing the pilot through a unit path)."""
    taps = []
    P = np.zeros((M, N), complex); P[lp, kp] = 1.0
    for l in range(0, lmax + 1):
        for k in range(-kmax, kmax + 1):
            v = Y[(lp + l) % M, (kp + k) % N] / pilot
            if abs(v) ** 2 > 9 * noise_var / pilot ** 2:
                c = otfs_demod(channel(otfs_mod(P), [(1.0, l, k)]), M, N)[(lp + l) % M, (kp + k) % N]
                taps.append((v / c, l, k))
    return taps


def dd_matrix(paths, M, N):
    """Effective MN x MN DD-domain channel: column i = response to a unit symbol on DD cell i."""
    MN = M * N
    H = np.zeros((MN, MN), complex)
    for i in range(MN):
        e = np.zeros(MN, complex); e[i] = 1
        H[:, i] = otfs_demod(channel(otfs_mod(e.reshape(M, N, order="F")), paths), M, N).reshape(-1, order="F")
    return H


def lmmse(Y, H, mask, X_pilot, noise_var):
    y = Y.reshape(-1, order="F") - H @ X_pilot.reshape(-1, order="F")     # remove the known pilot
    idx = np.nonzero(mask.reshape(-1, order="F"))[0]
    Hd = H[:, idx]
    x = np.linalg.solve(Hd.conj().T @ Hd + noise_var * np.eye(idx.size), Hd.conj().T @ y)
    G = np.zeros(mask.size, complex); G[idx] = x
    return G.reshape(mask.shape, order="F")[mask]          # same cell order as X[mask] in make_frame


def hard(x):
    return np.argmin(np.abs(x[:, None] - QPSK[None, :]), 1)


# ── 4. OFDM under the same channel (genie per-symbol one-tap CSI = best case for OFDM) ─────────────────
def ofdm_link(M, N, cp, paths, noise_var, rng):
    sym = rng.integers(0, 4, M * N)
    Xtf = QPSK[sym].reshape(M, N, order="F")
    body = np.fft.ifft(Xtf, axis=0) * np.sqrt(M)
    tx = np.concatenate([np.r_[body[-cp:, n], body[:, n]] for n in range(N)])
    L = tx.size
    t = np.arange(L)
    rx = sum(h * np.r_[np.zeros(l), tx[:L - l]] * np.exp(2j * np.pi * k * t / (M * N)) for h, l, k in paths)
    rx = awgn(rx, noise_var, rng)
    Ytf = np.stack([np.fft.fft(rx[n * (M + cp) + cp: (n + 1) * (M + cp)]) / np.sqrt(M) for n in range(N)], 1)
    m = np.arange(M)[:, None]
    H = np.zeros((M, N), complex)
    for h, l, k in paths:                                   # diagonal of each symbol's channel = mean phase
        tb = np.arange(N)[None, :] * (M + cp) + cp + np.arange(M)[:, None]
        H += h * np.exp(-2j * np.pi * m * l / M) * np.exp(2j * np.pi * k * tb / (M * N)).mean(0)[None, :]
    Xhat = Ytf * np.conj(H) / (np.abs(H) ** 2 + noise_var)
    return Xhat.reshape(-1, order="F"), sym


# ── figures ──────────────────────────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(1)
    M, N = 64, 32                     # delay bins x Doppler bins
    lp, kp, lmax, kmax = M // 2, N // 2, 4, 10   # pilot cell, max delay spread, Doppler search range
    snr_db, pilot_db = 20.0, 25.0
    nv = 10 ** (-snr_db / 10)
    assert np.allclose(otfs_demod(otfs_mod(np.arange(M * N).reshape(M, N) + 0j), M, N),
                       np.arange(M * N).reshape(M, N))

    # Fig 1: the grids
    X, mask, sym, pilot = make_frame(M, N, lp, kp, lmax, pilot_db, rng)
    one = np.zeros((M, N), complex); one[5, 3] = 1
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.9), dpi=110)
    lay = np.where(mask, 1.0, 0.0); lay[lp, kp] = 2.0
    ax[0].imshow(lay.T, origin="lower", cmap=matplotlib.colors.ListedColormap(["#E5E7EB", "#2A8C8E", "#C0392B"]),
                 aspect="auto")
    ax[0].set_title("DD grid: data (teal), guard (grey), pilot (red)"); ax[0].set_xlabel("delay bin l")
    ax[0].set_ylabel("Doppler bin k")
    ax[1].imshow(np.real(isfft(one)).T, origin="lower", cmap="RdBu", aspect="auto")
    ax[1].set_title("ONE DD symbol spreads over the whole\nTF plane (real part of its ISFFT)")
    ax[1].set_xlabel("subcarrier m"); ax[1].set_ylabel("OFDM symbol n")
    s = otfs_mod(X)
    ax[2].plot(np.real(s[:4 * M]), lw=0.8, color="#1E3A5F")
    for n in range(1, 4):
        ax[2].axvline(n * M, color="grey", ls=":", lw=0.8)
    ax[2].set_title("transmitted samples (first 4 of N symbols)"); ax[2].set_xlabel("sample")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig1_grids.png")); plt.close(fig)

    # Fig 2: the channel seen at the pilot
    paths = [(1.0, 0, 0), (0.6 * np.exp(1j * 1.0), 2, 3), (0.4 * np.exp(-1j * 2.0), 4, -2)]
    Y = otfs_demod(awgn(channel(otfs_mod(X), paths), nv, rng), M, N)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), dpi=110)
    ax[0].imshow(20 * np.log10(np.abs(Y.T) + 1e-6), origin="lower", aspect="auto", cmap="viridis", vmin=-10, vmax=25)
    ax[0].set_title("received DD grid |Y| (dB): data + pilot response"); ax[0].set_xlabel("delay bin l")
    ax[0].set_ylabel("Doppler bin k")
    Pz = np.zeros((M, N), complex); Pz[lp, kp] = pilot
    Yp = otfs_demod(channel(otfs_mod(Pz), paths), M, N)
    win = (slice(lp - 2, lp + lmax + 3), slice(kp - kmax - 2, kp + kmax + 3))
    ax[1].imshow(np.abs(Yp[win].T) / pilot, origin="lower", aspect="auto", cmap="magma",
                 extent=[-2.5, lmax + 2.5, -kmax - 2.5, kmax + 2.5])
    for h, l, k in paths:
        ax[1].plot(l, k, "o", mfc="none", mec="cyan", ms=16, mew=2)
        ax[1].text(l + 0.35, k + 0.35, f"|h|={abs(h):.1f}", color="cyan")
    ax[1].set_title("pilot response = the channel: one spot per path"); ax[1].set_xlabel("delay l (re pilot)")
    ax[1].set_ylabel("Doppler k (re pilot)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2_channel.png")); plt.close(fig)

    # Fig 3 + 4: equalization, OFDM vs OTFS, SER vs Doppler
    def otfs_link(paths, rng):
        X, mask, sym, pilot = make_frame(M, N, lp, kp, lmax, pilot_db, rng)
        Y = otfs_demod(awgn(channel(otfs_mod(X), paths), nv, rng), M, N)
        taps = estimate_taps(Y, M, N, lp, kp, lmax, kmax, pilot, nv)
        Xp = np.zeros((M, N), complex); Xp[lp, kp] = pilot
        xh = lmmse(Y, dd_matrix(taps, M, N), mask, Xp, nv)
        return xh, sym, len(taps)
    def rand_paths(k, rng):           # same delays/strengths, random phases, Doppler +-k bins
        ph = np.exp(2j * np.pi * rng.random(3))
        return [(1.0 * ph[0], 0, 0), (0.6 * ph[1], 2, k), (0.4 * ph[2], 4, -k)]
    kk = 6
    pk = rand_paths(kk, rng)
    xo, so = ofdm_link(M, N, lmax, pk, nv, rng)
    xt, st, nt = otfs_link(pk, rng)
    fig, ax = plt.subplots(1, 2, figsize=(9, 4.3), dpi=110)
    for a, x, s_, name in ((ax[0], xo, so, "OFDM, perfect per-symbol CSI"), (ax[1], xt, st, f"OTFS, {nt} pilot-estimated taps")):
        a.plot(x.real, x.imag, ".", ms=2, alpha=0.5, color="#1E3A5F")
        a.plot(QPSK.real, QPSK.imag, "r+", ms=14, mew=2)
        a.set_title(f"{name}\nSER = {np.mean(hard(x) != s_):.4f}"); a.set_aspect("equal")
        a.set_xlim(-2, 2); a.set_ylim(-2, 2); a.grid(alpha=0.3)
    fig.suptitle(f"same 3-path channel, Doppler up to {kk} bins (= {kk / N:.2f} of the subcarrier spacing), SNR {snr_db:.0f} dB")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig3_constellation.png")); plt.close(fig)

    ks, ser_o, ser_t = (0, 1, 2, 4, 6, 8), [], []
    for k in ks:
        eo, et = [], []
        for _ in range(8):
            pk = rand_paths(k, rng)
            x, s_ = ofdm_link(M, N, lmax, pk, nv, rng); eo.append(np.mean(hard(x) != s_))
            x, s_, _ = otfs_link(pk, rng); et.append(np.mean(hard(x) != s_))
        ser_o.append(np.mean(eo)); ser_t.append(np.mean(et))
        print(f"Doppler {k} bins: SER OFDM {ser_o[-1]:.4f}  OTFS {ser_t[-1]:.4f}")
    fig, ax = plt.subplots(figsize=(6, 4), dpi=110)
    eps = np.array(ks) / N
    floor = 0.5 / (M * N * 8)
    ax.semilogy(eps, np.maximum(ser_o, floor), "o-", color="#C77B1C", label="OFDM (perfect per-symbol CSI)")
    ax.semilogy(eps, np.maximum(ser_t, floor), "s-", color="#2A8C8E", label="OTFS (embedded pilot + DD-LMMSE)")
    ax.axhline(floor, color="grey", ls=":", lw=0.8)
    ax.set_xlabel("max Doppler / subcarrier spacing"); ax.set_ylabel("symbol error rate"); ax.grid(alpha=0.3, which="both")
    ax.set_title(f"3-path channel, SNR {snr_db:.0f} dB (dotted: no errors)"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig4_ser_vs_doppler.png")); plt.close(fig)

    # Fig 5: radar with the same waveform (monostatic, the transmitter knows its frame: no pilot needed)
    c0, fc, df = 299792458.0, 28e9, 120e3
    Mr, Nr = 256, 128
    fs = Mr * df
    dR, dV = c0 / (2 * fs), (c0 / fc) / 2 * fs / (Mr * Nr)
    Xr = QPSK[rng.integers(0, 4, Mr * Nr)].reshape(Mr, Nr)
    sr = otfs_mod(Xr)
    targets = [(6, 4, 1.0), (15, -2, 0.5), (25, 7, 0.3)]          # (delay bin, Doppler bin, amplitude)
    rr = awgn(channel(sr, [(a, l, k) for l, k, a in targets]), 10 ** (10 / 10), rng)   # -10 dB per sample
    n = np.arange(Mr * Nr)
    rd = np.array([np.fft.fftshift(np.fft.fft(rr * np.conj(np.roll(sr, l)))) for l in range(40)])
    kax = np.arange(-Mr * Nr // 2, Mr * Nr // 2)
    keep = np.abs(kax) <= 12
    P = 20 * np.log10(np.abs(rd[:, keep]) + 1e-9); P -= np.median(P)
    fig, ax = plt.subplots(figsize=(7, 4.3), dpi=110)
    im = ax.imshow(P.T, origin="lower", aspect="auto", cmap="inferno", vmin=0, vmax=50,
                   extent=[-dR / 2, (40 - 0.5) * dR, (kax[keep][0] - 0.5) * dV, (kax[keep][-1] + 0.5) * dV])
    for l, k, a in targets:
        ax.plot(l * dR, k * dV, "o", mfc="none", mec="cyan", ms=16, mew=2)
    ax.set_xlabel(f"range (m), resolution {dR:.1f} m"); ax.set_ylabel(f"radial velocity (m/s), resolution {dV:.1f} m/s")
    ax.set_title(f"OTFS radar, {fc / 1e9:.0f} GHz, {fs / 1e6:.2f} MHz, {Mr}x{Nr} frame: 3 targets (o = truth)")
    fig.colorbar(im, label="dB above median"); fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig5_radar_rd.png"))
    plt.close(fig)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
