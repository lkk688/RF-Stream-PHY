#!/usr/bin/env python3
"""
train_llr_net.py – Learned Demapper / Soft-LLR Network.

Self-supervised training: labels come from the conventional Gray-coded
hard demapper applied to the CLEAN equalized symbols.  The network
learns to replicate this mapping while implicitly capturing hardware
impairments (IQ offset, phase noise, ADC non-linearity, per-subcarrier
SNR variation).

At evaluation time we inject AWGN at varying SNR levels and compare
BER of the neural demapper vs the conventional formula.  Because the
neural model has seen the hardware-distorted clean constellation, it
adapts better to the real noise floor shape than the ideal-AWGN formula.

Input  : (48,) complex equalized subcarrier symbols → 96 floats (real||imag)
Output : (bps * 48,) logits → soft bit estimates
Labels : conventional Gray-coded hard decisions on the SAME clean symbols
         (self-supervised, no reference payload bits needed)

Outputs: rf_stream/llr_model/  (qpsk_llr.pt, qam16_llr.pt, metrics.json)
"""
import argparse, glob, json, os, sys, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

N_SC = 48   # OFDM data subcarriers

# ── Conventional demapper (used as self-supervised label generator) ───────────
def qpsk_hard_bits(Yeq_sym):
    """Gray-coded QPSK hard decisions on 48 subcarriers → 96 bits."""
    b = np.empty(len(Yeq_sym) * 2, dtype=np.float32)
    b[0::2] = (Yeq_sym.real < 0).astype(np.float32)
    b[1::2] = (Yeq_sym.imag < 0).astype(np.float32)
    return b

def qam16_hard_bits(Yeq_sym):
    """Gray-coded 16-QAM hard decisions on 48 subcarriers → 192 bits.
    Assumes constellation normalised so avg symbol power = 1.
    Points at ±1/√5, ±3/√5 → threshold at ±2/√5.
    """
    y = Yeq_sym * np.sqrt(5)          # scale to ±1, ±3
    b = np.empty(len(y) * 4, dtype=np.float32)
    ri = y.real; qi = y.imag
    b[0::4] = (ri < 0).astype(np.float32)       # MSB I
    b[1::4] = (np.abs(ri) < 2.0).astype(np.float32)  # LSB I
    b[2::4] = (qi < 0).astype(np.float32)       # MSB Q
    b[3::4] = (np.abs(qi) < 2.0).astype(np.float32)  # LSB Q
    return b

# ── Dataset loader ─────────────────────────────────────────────────────────────
def load_demapper_data(run_dirs, bps):
    """Load (Yeq_sym, label) pairs. Labels = conventional hard decisions."""
    X, Y = [], []
    skipped = 0
    decide = qpsk_hard_bits if bps == 2 else qam16_hard_bits

    for rd in run_dirs:
        for f in sorted(glob.glob(os.path.join(rd, "cap_*_ok.npz"))):
            npz = np.load(f, allow_pickle=True)
            if "Yeq_data" not in npz.files: skipped += 1; continue
            try:
                mj = npz["meta_json"].item()
                meta = json.loads(mj if isinstance(mj, str) else mj.decode())
            except Exception: skipped += 1; continue
            if meta.get("bps", -1) != bps: continue
            if meta.get("n_bit_errors", 1) != 0: continue   # only CRC-passed packets
            Yeq = np.asarray(npz["Yeq_data"], dtype=np.complex64)  # (N_sym, 48)
            for sym_idx in range(Yeq.shape[0]):
                Yeq_sym = Yeq[sym_idx]
                # normalise per-symbol power to handle rx_gain variation
                pwr = np.mean(np.abs(Yeq_sym) ** 2)
                if pwr < 1e-10: continue
                Yeq_sym_n = Yeq_sym / np.sqrt(pwr)
                x = np.concatenate([Yeq_sym_n.real, Yeq_sym_n.imag]).astype(np.float32)
                y = decide(Yeq_sym)          # self-supervised label
                X.append(x); Y.append(y)

    print(f"  [bps={bps}] {len(X)} symbol samples  (skipped {skipped} files)")
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

# ── Model ─────────────────────────────────────────────────────────────────────
class LLRNet(nn.Module):
    """Per-symbol MLP: 96 → 128 → 64 → bps*48 logits."""
    def __init__(self, bps=2, n_sc=48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_sc*2, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, n_sc * bps),
        )
    def forward(self, x): return self.net(x)

class SDS(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.Y[i]

# ── Train & eval ──────────────────────────────────────────────────────────────
def add_awgn(X_sym, snr_db, bps):
    """Inject AWGN into 96-float (real||imag) symbol features at given SNR."""
    snr_lin = 10 ** (snr_db / 10.0)
    # reconstruct complex, add noise, flatten again
    n = X_sym.shape[0]
    Yr = X_sym[:, :N_SC]; Yi = X_sym[:, N_SC:]
    pwr = np.mean(Yr**2 + Yi**2, axis=1, keepdims=True) + 1e-12
    noise_var = pwr / (snr_lin * bps)
    nr = np.random.randn(n, N_SC).astype(np.float32) * np.sqrt(noise_var)
    ni = np.random.randn(n, N_SC).astype(np.float32) * np.sqrt(noise_var)
    return np.concatenate([Yr + nr, Yi + ni], axis=1)

def train_one(name, bps, X, Y, device, epochs, out_dir):
    """Self-supervised training: labels = conventional hard-demapper on clean symbols."""
    decide = qpsk_hard_bits if bps == 2 else qam16_hard_bits
    n_val = max(1, int(0.2 * len(X)))
    idx = np.random.permutation(len(X))
    val_idx, trn_idx = idx[:n_val], idx[n_val:]
    X_trn, Y_trn = X[trn_idx], Y[trn_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]

    # Global normalisation from training set (already per-symbol power-normalised in loader)
    mu = X_trn.mean(0); sigma = X_trn.std(0) + 1e-8
    X_trn_n = (X_trn - mu) / sigma
    X_val_n  = (X_val  - mu) / sigma

    trn_ds = SDS(X_trn_n, Y_trn); val_ds = SDS(X_val_n, Y_val)
    trn_ld = DataLoader(trn_ds, batch_size=512, shuffle=True, num_workers=2)
    val_ld = DataLoader(val_ds, batch_size=1024, shuffle=False)

    model = LLRNet(bps=bps).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.BCEWithLogitsLoss()

    history = []
    for ep in range(epochs):
        model.train()
        tl = 0.0
        for xb, yb in trn_ld:
            opt.zero_grad()
            loss = crit(model(xb.to(device)), yb.to(device))
            loss.backward(); opt.step(); tl += loss.item() * len(xb)
        sched.step()
        tl /= len(X_trn)
        model.eval(); vl = 0.0
        with torch.no_grad():
            for xb, yb in val_ld:
                vl += crit(model(xb.to(device)), yb.to(device)).item() * len(xb)
        vl /= len(X_val)
        history.append({"epoch": ep+1, "train_loss": float(tl), "val_loss": float(vl)})
        if (ep+1) % 10 == 0:
            print(f"  [{name}] ep={ep+1:3d}  tr={tl:.4f}  val={vl:.4f}")

    # ── SNR sweep evaluation ─────────────────────────────────────────────────
    model.eval()
    snr_range = list(range(-4, 28, 2))
    ber_nn, ber_conv = [], []
    np.random.seed(99)
    for snr_db in snr_range:
        Xn = add_awgn(X_val, snr_db, bps)
        Xn_norm = (Xn - mu) / sigma
        with torch.no_grad():
            logits = model(torch.from_numpy(Xn_norm).to(device)).cpu().numpy()
        pred = (logits > 0).astype(np.float32)
        # Conventional decision on noisy data
        conv = np.stack([decide(Xn[i, :N_SC] + 1j * Xn[i, N_SC:]) for i in range(len(Xn))])
        ber_nn.append(float(np.mean(pred   != Y_val)))
        ber_conv.append(float(np.mean(conv != Y_val)))
        print(f"    SNR={snr_db:4d}dB  BER_nn={ber_nn[-1]:.5f}  BER_conv={ber_conv[-1]:.5f}")

    metrics = {"name": name, "bps": bps, "n_sym_trn": len(X_trn),
               "n_sym_val": len(X_val), "epochs": epochs,
               "final_val_loss": float(history[-1]["val_loss"]),
               "snr_db": snr_range, "ber_neural": ber_nn, "ber_conventional": ber_conv}

    os.makedirs(out_dir, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "mu": mu.tolist(),
                "sigma": sigma.tolist(), "bps": bps},
               os.path.join(out_dir, f"{name}_llr.pt"))
    with open(os.path.join(out_dir, f"{name}_history.json"), "w") as fp:
        json.dump(history, fp)
    return metrics

# ── Main ──────────────────────────────────────────────────────────────────────
def main(args):
    run_dirs = []
    for p in args.run_dirs:
        run_dirs += sorted(glob.glob(p))
    run_dirs = [r for r in run_dirs if os.path.isdir(r)]
    print(f"[llr] {len(run_dirs)} run dirs")

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    print(f"[llr] device={device}  (self-supervised labels: conventional hard demapper)")
    np.random.seed(0)

    all_metrics = {}
    os.makedirs(args.out_dir, exist_ok=True)

    for bps, name in [(2, "qpsk"), (4, "qam16")]:
        print(f"\n=== Learned Demapper: {name.upper()} (bps={bps}) ===")
        X, Y = load_demapper_data(run_dirs, bps)
        if len(X) < 100:
            print(f"  too few samples ({len(X)}), skipping"); continue
        m = train_one(name, bps, X, Y, device, args.epochs, args.out_dir)
        all_metrics[name] = m

    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fp:
        json.dump(all_metrics, fp, indent=2)
    print(f"\n[llr] done → {args.out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dirs", nargs="+",
                    default=["rf_stream/ber_sweep_v3/run_*",
                             "rf_stream/ber_sweep/run_20260428_233037",
                             "rf_stream/ber_sweep/run_20260429_001722",
                             "rf_stream/ber_sweep/run_20260429_003513",
                             "rf_stream/ber_sweep/run_20260429_070949"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--out_dir", default="rf_stream/llr_model")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    main(args)
