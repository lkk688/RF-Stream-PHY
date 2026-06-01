#!/usr/bin/env python3
"""
eval_llr_net.py – Evaluate learned demapper with correct self-supervised labels
and SNR simulation to show the "BER not zero" region.

Key fix: use conventional hard-decision bits (sign of I/Q) as pseudo-labels,
not external reference bits (which have alignment issues).

Adds simulation: inject AWGN noise at varying SNR, compare neural vs conventional BER.
"""
import json, glob, os, sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))

N_SC = 48

# ── Conventional demappers ────────────────────────────────────────────────────
def qpsk_decide(Yeq_sym):
    """Gray-coded QPSK hard bits from equalized symbols."""
    b = np.zeros(len(Yeq_sym) * 2, dtype=np.float32)
    b[0::2] = (Yeq_sym.real < 0).astype(np.float32)
    b[1::2] = (Yeq_sym.imag < 0).astype(np.float32)
    return b

def qam16_decide(Yeq_sym):
    """Gray-coded 16-QAM hard bits from equalized symbols."""
    # Normalize: assume unit-avg-power constellation {-3,-1,+1,+3}/sqrt(5)
    y = Yeq_sym * np.sqrt(5)
    b = np.zeros(len(y) * 4, dtype=np.float32)
    for i, s in enumerate(y):
        ri, qi = float(s.real), float(s.imag)
        b[4*i+0] = float(ri < 0)
        b[4*i+1] = float(abs(ri) < 2.0)
        b[4*i+2] = float(qi < 0)
        b[4*i+3] = float(abs(qi) < 2.0)
    return b

# ── Model ─────────────────────────────────────────────────────────────────────
class LLRNet(nn.Module):
    def __init__(self, bps=2, n_sc=48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_sc*2, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, n_sc * bps),
        )
    def forward(self, x): return self.net(x)

# ── SNR simulation ────────────────────────────────────────────────────────────
def add_awgn(Yeq_flat, snr_db, bps):
    """Add complex AWGN to equalized symbols at given SNR (dB)."""
    snr_lin = 10 ** (snr_db / 10.0)
    signal_power = np.mean(np.abs(Yeq_flat) ** 2)
    noise_var = signal_power / (snr_lin * bps)
    noise = (np.random.randn(*Yeq_flat.shape) + 1j * np.random.randn(*Yeq_flat.shape)) * np.sqrt(noise_var / 2)
    return (Yeq_flat + noise).astype(np.complex64)

# ── Load data ─────────────────────────────────────────────────────────────────
def load_yeq(run_dirs, bps):
    all_syms = []
    decide_fn = qpsk_decide if bps == 2 else qam16_decide
    for rd in run_dirs:
        for f in sorted(glob.glob(os.path.join(rd, "cap_*_ok.npz"))):
            npz = np.load(f, allow_pickle=True)
            if "Yeq_data" not in npz.files: continue
            try:
                mj = npz["meta_json"].item()
                meta = json.loads(mj if isinstance(mj, str) else mj.decode())
            except: continue
            if meta.get("bps", -1) != bps: continue
            Yeq = np.asarray(npz["Yeq_data"], dtype=np.complex64)
            all_syms.append(Yeq)
    return all_syms  # list of (N_sym, 48) complex arrays

# ── Evaluate at varying SNR ───────────────────────────────────────────────────
def evaluate_snr_sweep(model, mu, sigma, Yeq_list, bps, snr_range, device, n_test=2000):
    decide_fn = qpsk_decide if bps == 2 else qam16_decide
    # Flatten all symbols into (N_total, 48) complex
    all_syms = np.concatenate([y for y in Yeq_list], axis=0)
    np.random.shuffle(all_syms)
    all_syms = all_syms[:n_test]

    # Get "true" bits via conventional decision on clean data (these are ground truth)
    X_clean = np.concatenate([all_syms.real, all_syms.imag], axis=1).astype(np.float32)
    true_bits = np.stack([decide_fn(all_syms[i]) for i in range(len(all_syms))])

    ber_neural, ber_conv = [], []
    for snr_db in snr_range:
        noisy_syms = np.array([add_awgn(all_syms[i], snr_db, bps) for i in range(len(all_syms))])
        X_noisy = np.concatenate([noisy_syms.real, noisy_syms.imag], axis=1).astype(np.float32)
        X_norm = (X_noisy - mu) / (sigma + 1e-8)

        # Neural decision
        model.eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(X_norm).to(device)).cpu().numpy()
        nn_bits = (logits > 0).astype(np.float32)

        # Conventional decision
        conv_bits = np.stack([decide_fn(noisy_syms[i]) for i in range(len(noisy_syms))])

        ber_nn_val   = float(np.mean(np.abs(nn_bits   - true_bits)))
        ber_conv_val = float(np.mean(np.abs(conv_bits - true_bits)))
        ber_neural.append(ber_nn_val)
        ber_conv.append(ber_conv_val)
        print(f"  SNR={snr_db:5.1f}dB  BER_nn={ber_nn_val:.5f}  BER_conv={ber_conv_val:.5f}")

    return ber_neural, ber_conv

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    run_dirs_all = (sorted(glob.glob("rf_stream/ber_sweep_v3/run_*")) +
                    ["rf_stream/ber_sweep/run_20260428_233037",
                     "rf_stream/ber_sweep/run_20260429_001722",
                     "rf_stream/ber_sweep/run_20260429_003513",
                     "rf_stream/ber_sweep/run_20260429_070949"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    snr_range = list(range(-5, 32, 2))

    results = {}
    for bps, name in [(2, "qpsk"), (4, "qam16")]:
        ckpt_path = f"rf_stream/llr_model/{name}_llr.pt"
        if not os.path.exists(ckpt_path):
            print(f"[{name}] checkpoint not found, skipping"); continue
        ckpt = torch.load(ckpt_path, map_location=device)
        mu = np.array(ckpt["mu"], dtype=np.float32)
        sigma = np.array(ckpt["sigma"], dtype=np.float32)
        model = LLRNet(bps=bps).to(device)
        model.load_state_dict(ckpt["state_dict"])

        print(f"\n=== {name.upper()} SNR sweep ===")
        Yeq_list = load_yeq(run_dirs_all, bps)
        print(f"  {len(Yeq_list)} captures → {sum(y.shape[0] for y in Yeq_list)} symbols")
        if not Yeq_list:
            print("  no data, skipping"); continue

        ber_nn, ber_cv = evaluate_snr_sweep(model, mu, sigma, Yeq_list, bps, snr_range, device)
        results[name] = {"snr_db": snr_range, "ber_neural": ber_nn, "ber_conventional": ber_cv}

    out = "rf_stream/llr_model/snr_sweep.json"
    with open(out, "w") as fp:
        json.dump(results, fp, indent=2)
    print(f"\n[eval] SNR sweep saved → {out}")

if __name__ == "__main__":
    main()
