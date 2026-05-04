#!/usr/bin/env python3
"""
train_multitask_v2.py – MT-PreamCNN-Attn: Multi-Task Preamble Intelligence
                         with Temporal Self-Attention and OTA Generalisation.

Builds on train_multitask_v1.py with two contributions:

  1. MT-PreamCNN-Attn architecture:
       Shared CNN encoder (identical to v1) followed by a 4-head temporal
       self-attention block over the 50 time-step feature map, before
       adaptive global pooling.  Attention captures long-range preamble
       structure degraded by OTA multipath without adding a new modality.

  2. Cross-domain training and evaluation:
       --mode coax      : train & eval on CoaxSweep (cable) data only
       --mode airlink   : train & eval on AirLink (OTA) data only
       --mode mixed     : train on CoaxSweep+AirLink, eval on held-out AirLink
       --mode zeroshot  : train on CoaxSweep, eval on full AirLink (no fine-tune)

Dataset naming (academic):
  CoaxSweep-I/II/III/IV   : cable gain-sweep experiments (ber_sweep_v1/v2/v3/v4)
  AirLink-QPSK             : OTA QPSK 2400 MHz captures with --save_npz
  AirLink-QAM16            : OTA QAM16 2400 MHz captures with --save_npz + auto_z_th

Outputs: rf_stream/multitask_model_v2/
  mt_preamcnn_attn.pt    checkpoint (state_dict + norm stats + task config)
  metrics_coax.json      within-domain (CoaxSweep) eval metrics
  metrics_airlink.json   AirLink eval metrics (zero-shot or held-out)
  history.json           per-epoch loss breakdown
"""
import argparse, glob, json, os, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset

try:
    from sklearn.metrics import (roc_auc_score, roc_curve,
                                 precision_recall_curve, accuracy_score)
    SKLEARN = True
except ImportError:
    SKLEARN = False

torch.backends.cudnn.enabled = False  # known Conv1D segfault on some GPU drivers

PREAM_LEN   = 800
PREAM_FLOAT = PREAM_LEN * 2  # real/imag interleaved

# PHY preamble constants (must match TX/RX scripts)
_N_FFT  = 64
_N_CP   = 16
_FS_HZ  = 3e6
_STF_L  = _N_FFT // 2   # 32: STF half-period (even-subcarrier Schmidl-Cox)
_STF_BODY_START = _N_CP  # 16: skip CP at preamble start
_STF_BODY_LEN   = 6 * _N_FFT  # 384: 6 × 64-sample STF repeats


def _correct_cfo(seg: np.ndarray) -> np.ndarray:
    """Schmidl-Cox CFO estimation + correction using the STF half-period."""
    body = seg[_STF_BODY_START : _STF_BODY_START + _STF_BODY_LEN]
    L = _STF_L
    # Accumulate cross-correlation over all available L-spaced pairs
    pairs = len(body) // L - 1          # 11 pairs from 384/32=12 half-periods
    if pairs < 1:
        return seg
    r = np.dot(body[:pairs * L].conj(), body[L : (pairs + 1) * L])
    cfo_hz = float(np.angle(r) / (2.0 * np.pi * L / _FS_HZ))
    t = np.arange(len(seg), dtype=np.float32)
    return (seg * np.exp(-1j * 2.0 * np.pi * cfo_hz * t / _FS_HZ)).astype(np.complex64)


# ── Preamble extraction ────────────────────────────────────────────────────────
def extract_preamble_iq(npz, meta, cfo_correct=True):
    if "rxw" not in npz.files:
        return None
    rxw = np.array(npz["rxw"], dtype=np.complex64, copy=True)
    stf = int(meta.get("stf_idx", 0))
    end = stf + PREAM_LEN
    if end > len(rxw):
        stf = max(0, len(rxw) - PREAM_LEN)
        end = len(rxw)
    seg = rxw[stf:end]
    if len(seg) < PREAM_LEN:
        seg = np.pad(seg, (0, PREAM_LEN - len(seg)))
    if cfo_correct:
        seg = _correct_cfo(seg)
    out = np.empty(PREAM_FLOAT, dtype=np.float32)
    out[0::2] = seg.real
    out[1::2] = seg.imag
    return out.copy()


# ── Data loading with domain label ────────────────────────────────────────────
def load_dataset(run_dirs, domain="coax", cfo_correct=True):
    """
    Load NPZ captures from run directories.
    domain: "coax" or "airlink" — stored as metadata for cross-domain analysis.
    cfo_correct: if True, apply Schmidl-Cox CFO estimation + correction before
                 returning the preamble segment (default True).
    Returns list of sample dicts with keys:
      pream, gate, mod, snr_db, has_snr, domain
    """
    samples = []
    skipped = 0
    for rd in run_dirs:
        for f in sorted(glob.glob(os.path.join(rd, "cap_*.npz"))):
            gate_label = 1 if "_ok.npz" in f else 0
            try:
                with np.load(f, allow_pickle=True) as npz:
                    meta = {}
                    if "meta_json" in npz.files:
                        try:
                            mj = npz["meta_json"].item()
                            meta = json.loads(mj if isinstance(mj, str) else mj.decode())
                        except Exception:
                            pass
                    pream = extract_preamble_iq(npz, meta, cfo_correct=cfo_correct)
            except Exception:
                skipped += 1
                continue
            if pream is None:
                skipped += 1
                continue
            bps = meta.get("bps", -1)
            # Only assign mod label to valid (ok) captures; fail NPZs may be WiFi/noise
            if gate_label == 1:
                mod_label = 0 if bps == 2 else (1 if bps == 4 else -1)
            else:
                mod_label = -1
            raw_snr = meta.get("snr_db", None) if gate_label == 1 else None
            if raw_snr is None or (isinstance(raw_snr, float) and np.isnan(raw_snr)):
                snr_db, has_snr = np.nan, False
            else:
                snr_db, has_snr = float(raw_snr), True
            samples.append(dict(pream=pream, gate=gate_label,
                                mod=mod_label, snr_db=snr_db,
                                has_snr=has_snr, domain=domain))

    ok   = sum(1 for s in samples if s["gate"] == 1)
    fail = sum(1 for s in samples if s["gate"] == 0)
    qpsk = sum(1 for s in samples if s["mod"] == 0)
    qam  = sum(1 for s in samples if s["mod"] == 1)
    print(f"  [{domain}] {len(samples)} samples  "
          f"(ok={ok} fail={fail})  (qpsk={qpsk} qam16={qam})  skipped={skipped}")
    return samples


def balance_gate(samples, seed=42):
    ok   = [s for s in samples if s["gate"] == 1]
    fail = [s for s in samples if s["gate"] == 0]
    rng  = random.Random(seed)
    if len(ok) > len(fail):
        fail = (fail * (len(ok) // max(len(fail), 1) + 1))[:len(ok)]
    elif len(fail) > len(ok):
        ok = (ok * (len(fail) // max(len(ok), 1) + 1))[:len(fail)]
    combined = ok + fail
    rng.shuffle(combined)
    return combined


def train_val_split(samples, val_frac=0.2, seed=42):
    rng = random.Random(seed)
    s = list(samples); rng.shuffle(s)
    n_val = max(1, int(val_frac * len(s)))
    return s[n_val:], s[:n_val]


def build_tensors(samples, snr_mu=None, snr_sigma=None):
    X        = np.stack([s["pream"] for s in samples]).astype(np.float32)
    gate     = np.array([s["gate"]  for s in samples], dtype=np.float32)
    mod      = np.array([s["mod"]   for s in samples], dtype=np.float32)
    snr_raw  = np.array([s["snr_db"] if s["has_snr"] else np.nan
                         for s in samples], dtype=np.float32)
    snr_mask = np.array([s["has_snr"] for s in samples], dtype=bool)
    if snr_mu is None:
        valid     = snr_raw[snr_mask]
        snr_mu    = float(np.mean(valid)) if len(valid) > 0 else 0.0
        snr_sigma = float(np.std(valid))  if len(valid) > 1 else 1.0
        snr_sigma = max(snr_sigma, 1e-3)
    snr_norm = np.where(snr_mask, (snr_raw - snr_mu) / snr_sigma, 0.0).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return (torch.from_numpy(X).float(),
            torch.from_numpy(gate).float(),
            torch.from_numpy(mod).float(),
            torch.from_numpy(snr_norm).float(),
            torch.from_numpy(snr_mask),
            snr_mu, snr_sigma)


# ── Phase-augmentation dataset ────────────────────────────────────────────────
class PhaseAugDataset(Dataset):
    """Applies random global phase rotation each epoch to force amplitude-only features.

    QPSK has constant-envelope (single amplitude ring); QAM-16 has three
    amplitude rings.  Both are invariant to a global phase rotation
    z → z·exp(jθ).  Training with random θ ∈ [0, 2π) prevents the model
    from relying on CFO-dependent phase patterns, which differ in sign and
    magnitude between CoaxSweep (+330 Hz) and AirLink (−3000 Hz).
    """
    def __init__(self, X, gate, mod, snr, snr_mask):
        self.X        = X           # (N, PREAM_FLOAT) interleaved I/Q
        self.gate     = gate
        self.mod      = mod
        self.snr      = snr
        self.snr_mask = snr_mask

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        theta = float(torch.rand(1)) * 2.0 * np.pi
        cos_t, sin_t = float(np.cos(theta)), float(np.sin(theta))
        I = x[0::2].clone()
        Q = x[1::2].clone()
        x[0::2] = I * cos_t - Q * sin_t
        x[1::2] = I * sin_t + Q * cos_t
        return x, self.gate[idx], self.mod[idx], self.snr[idx], self.snr_mask[idx]


# ── Models ────────────────────────────────────────────────────────────────────
class MultiTaskPreamCNN(nn.Module):
    """Baseline: shared Conv1D encoder + 3 task heads (identical to v1)."""
    def __init__(self, embed_dim=256, tasks=("gate", "mod", "snr")):
        super().__init__()
        self.tasks = tasks
        self.encoder = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=31, padding=15),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=15, padding=7),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
        )
        self.proj = nn.Sequential(
            nn.Linear(128 * 8, embed_dim), nn.ReLU(), nn.Dropout(0.3))
        if "gate" in tasks: self.gate_head = nn.Linear(embed_dim, 1)
        if "mod"  in tasks: self.mod_head  = nn.Linear(embed_dim, 1)
        if "snr"  in tasks:
            self.snr_head = nn.Sequential(
                nn.Linear(embed_dim, 32), nn.ReLU(), nn.Linear(32, 1))

    def encode(self, x):
        B = x.size(0)
        x2 = x.view(B, PREAM_LEN, 2).permute(0, 2, 1).contiguous()
        return self.proj(self.encoder(x2).view(B, -1))

    def forward(self, x):
        emb = self.encode(x)
        out = {}
        if "gate" in self.tasks: out["gate"] = self.gate_head(emb).squeeze(-1)
        if "mod"  in self.tasks: out["mod"]  = self.mod_head(emb).squeeze(-1)
        if "snr"  in self.tasks: out["snr"]  = self.snr_head(emb).squeeze(-1)
        return out


class TemporalSelfAttention(nn.Module):
    """
    Multi-head self-attention over the temporal dimension of Conv1D output.

    Input:  (B, C, T) — C channels, T time steps
    Output: (B, C, T) — attended features, same shape
    Uses residual connection + LayerNorm for stable training.
    """
    def __init__(self, d_model=128, nhead=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead,
                                          dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, C, T)
        xt = x.permute(0, 2, 1)          # (B, T, C)
        a, _ = self.attn(xt, xt, xt)     # (B, T, C)
        xt = self.norm(xt + self.dropout(a))  # residual + LayerNorm
        return xt.permute(0, 2, 1)        # (B, C, T)


class MultiTaskPreamCNNAttn(nn.Module):
    """
    MT-PreamCNN-Attn: CNN encoder + temporal self-attention + 3 task heads.

    Architecture:
      (B,1600) → CNN[32,64,128 filters; k=31,15,7] → (B,128,50)
               → TemporalSelfAttention(d=128, heads=4)  → (B,128,50)
               → AdaptiveAvgPool(8)                     → (B,128,8)
               → Linear(1024→256) + ReLU + Dropout(0.3) → (B,256)
               → 3 task heads (gate, mod, snr)

    The CNN uses MaxPool(4) after each of the first two conv layers,
    producing 50 time steps at 128 channels. Attention over these 50 steps
    captures long-range preamble correlations introduced by OTA multipath.
    """
    def __init__(self, embed_dim=256, tasks=("gate", "mod", "snr"),
                 nhead=4, attn_dropout=0.1):
        super().__init__()
        self.tasks = tasks

        # CNN: produces (B, 128, 50) — AdaptiveAvgPool moved after attention
        self.cnn = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=31, padding=15),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(4),     # (B,32,200)
            nn.Conv1d(32, 64, kernel_size=15, padding=7),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(4),     # (B,64,50)
            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128), nn.ReLU(),                     # (B,128,50)
        )

        # Temporal self-attention over 50 time steps
        self.attn_block = TemporalSelfAttention(
            d_model=128, nhead=nhead, dropout=attn_dropout)

        # Global pooling
        self.pool = nn.AdaptiveAvgPool1d(8)                     # (B,128,8)

        # Shared embedding projection
        self.proj = nn.Sequential(
            nn.Linear(128 * 8, embed_dim), nn.ReLU(), nn.Dropout(0.3))

        if "gate" in tasks: self.gate_head = nn.Linear(embed_dim, 1)
        if "mod"  in tasks: self.mod_head  = nn.Linear(embed_dim, 1)
        if "snr"  in tasks:
            self.snr_head = nn.Sequential(
                nn.Linear(embed_dim, 32), nn.ReLU(), nn.Linear(32, 1))

    def encode(self, x):
        B = x.size(0)
        x2   = x.view(B, PREAM_LEN, 2).permute(0, 2, 1).contiguous()
        feat = self.cnn(x2)              # (B, 128, 50)
        feat = self.attn_block(feat)     # (B, 128, 50) — attended
        feat = self.pool(feat)           # (B, 128, 8)
        return self.proj(feat.reshape(B, -1))  # (B, embed_dim)

    def forward(self, x):
        emb = self.encode(x)
        out = {}
        if "gate" in self.tasks: out["gate"] = self.gate_head(emb).squeeze(-1)
        if "mod"  in self.tasks: out["mod"]  = self.mod_head(emb).squeeze(-1)
        if "snr"  in self.tasks: out["snr"]  = self.snr_head(emb).squeeze(-1)
        return out


def build_model(arch, tasks, embed_dim, device):
    if arch == "cnn":
        model = MultiTaskPreamCNN(embed_dim=embed_dim, tasks=tasks)
    elif arch == "attn":
        model = MultiTaskPreamCNNAttn(embed_dim=embed_dim, tasks=tasks)
    else:
        raise ValueError(f"Unknown arch: {arch}")
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [{arch}] params={n:,}")
    return model.to(device)


# ── Training utilities ─────────────────────────────────────────────────────────
def run_epoch(model, loader, opt, device, tasks, w_gate, w_mod, w_snr, train=True):
    model.train() if train else model.eval()
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()
    total = gate_t = mod_t = snr_t = n = 0
    with torch.set_grad_enabled(train):
        for xb, gy, my, sy, sm in loader:
            xb = xb.to(device)
            gy, my, sy, sm = gy.to(device), my.to(device), sy.to(device), sm.to(device)
            out  = model(xb)
            loss = torch.tensor(0.0, device=device)
            if "gate" in tasks:
                gl = bce(out["gate"], gy)
                loss = loss + w_gate * gl; gate_t += gl.item() * len(xb)
            if "mod" in tasks:
                mm = (my >= 0)
                if mm.any():
                    ml = bce(out["mod"][mm], my[mm])
                    loss = loss + w_mod * ml; mod_t += ml.item() * mm.sum().item()
            if "snr" in tasks and sm.any():
                sl = mse(out["snr"][sm], sy[sm])
                loss = loss + w_snr * sl; snr_t += sl.item() * sm.sum().item()
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(xb); n += len(xb)
    return total/max(n,1), gate_t/max(n,1), mod_t/max(n,1), snr_t/max(n,1)


def collect_preds(model, loader, device, tasks):
    model.eval()
    g_pred, g_true, m_pred, m_true, s_pred, s_true = [], [], [], [], [], []
    with torch.no_grad():
        for xb, gy, my, sy, sm in loader:
            xb = xb.to(device)
            out = model(xb)
            if "gate" in tasks:
                g_pred.extend(torch.sigmoid(out["gate"]).cpu().tolist())
                g_true.extend(gy.tolist())
            if "mod" in tasks:
                mm = (my >= 0)
                if mm.any():
                    m_pred.extend(torch.sigmoid(out["mod"][mm]).cpu().tolist())
                    m_true.extend(my[mm].tolist())
            if "snr" in tasks:
                ms = sm.bool()
                if ms.any():
                    s_pred.extend(out["snr"][ms].cpu().tolist())
                    s_true.extend(sy[ms].tolist())
    return (np.array(g_pred), np.array(g_true),
            np.array(m_pred), np.array(m_true),
            np.array(s_pred), np.array(s_true))


def compute_metrics(g_pred, g_true, m_pred, m_true, s_pred, s_true,
                    snr_mu, snr_sigma, tasks, tag=""):
    metrics = {}
    if "gate" in tasks and SKLEARN and len(g_true) > 0:
        auc       = roc_auc_score(g_true, g_pred)
        prec, rec, thr = precision_recall_curve(g_true, g_pred)
        f1        = 2 * prec * rec / (prec + rec + 1e-12)
        bi        = np.argmax(f1[:-1])
        fpr, tpr, _ = roc_curve(g_true, g_pred)
        metrics["gate"] = {
            "roc_auc": float(auc),
            "best_thr": float(thr[bi]),
            "best_prec": float(prec[bi]),
            "best_rec": float(rec[bi]),
            "best_f1": float(f1[bi]),
            "roc_fpr": fpr.tolist(), "roc_tpr": tpr.tolist(),
        }
        print(f"  [{tag} gate]  AUC={auc:.4f}  P={prec[bi]:.3f}  R={rec[bi]:.3f}")
    if "mod" in tasks and len(m_true) > 0:
        m_bin = (np.array(m_pred) > 0.5).astype(int)
        acc   = float(np.mean(m_bin == np.array(m_true).astype(int)))
        m_auc = roc_auc_score(m_true, m_pred) if SKLEARN and len(np.unique(m_true)) >= 2 else -1.0
        metrics["mod"] = {"accuracy": acc, "roc_auc": float(m_auc), "n_val": len(m_true)}
        print(f"  [{tag} mod]   acc={acc:.4f}  AUC={m_auc:.4f}")
    if "snr" in tasks and len(s_true) > 0:
        s_pred_db = np.array(s_pred) * snr_sigma + snr_mu
        s_true_db = np.array(s_true) * snr_sigma + snr_mu
        rmse = float(np.sqrt(np.mean((s_pred_db - s_true_db) ** 2)))
        mae  = float(np.mean(np.abs(s_pred_db - s_true_db)))
        r2   = float(1 - np.sum((s_true_db-s_pred_db)**2) /
                     (np.sum((s_true_db-np.mean(s_true_db))**2) + 1e-12))
        metrics["snr"] = {"rmse_db": rmse, "mae_db": mae, "r2": r2,
                          "snr_mu": snr_mu, "snr_sigma": snr_sigma}
        print(f"  [{tag} snr]   RMSE={rmse:.2f}dB  MAE={mae:.2f}dB  R²={r2:.4f}")
    return metrics


# ── Main ──────────────────────────────────────────────────────────────────────
def main(args):
    tasks = tuple(args.tasks)
    cfo_correct = not args.no_cfo_correct
    print(f"\n=== MT-PreamCNN-Attn v2 ===")
    print(f"  arch={args.arch}  mode={args.mode}  tasks={tasks}  cfo_correct={cfo_correct}")

    # ── Coax (cable) dataset ──────────────────────────────────────────────
    coax_dirs = []
    for p in args.coax_dirs:
        coax_dirs += sorted(glob.glob(p))
    coax_dirs = [r for r in coax_dirs if os.path.isdir(r)]
    print(f"\n[data] CoaxSweep: {len(coax_dirs)} run dirs")
    coax_samples = load_dataset(coax_dirs, domain="coax",
                                cfo_correct=cfo_correct) if coax_dirs else []

    # ── AirLink (OTA) dataset ─────────────────────────────────────────────
    airlink_dirs = []
    for p in args.airlink_dirs:
        airlink_dirs += sorted(glob.glob(p))
    airlink_dirs = [r for r in airlink_dirs if os.path.isdir(r)]
    print(f"[data] AirLink: {len(airlink_dirs)} run dirs")
    airlink_samples = load_dataset(airlink_dirs, domain="airlink",
                                   cfo_correct=cfo_correct) if airlink_dirs else []

    # ── Split per mode ────────────────────────────────────────────────────
    if args.mode == "coax":
        # Train/val entirely within CoaxSweep
        balanced = balance_gate(coax_samples)
        trn_s, val_s = train_val_split(balanced)
        eval_sets = {"coax_val": val_s}

    elif args.mode == "airlink":
        # Train/val entirely within AirLink
        if not airlink_samples:
            raise RuntimeError("No AirLink data found — check --airlink_dirs")
        balanced = balance_gate(airlink_samples)
        trn_s, val_s = train_val_split(balanced)
        eval_sets = {"airlink_val": val_s}

    elif args.mode == "zeroshot":
        # Train on CoaxSweep only, evaluate on full AirLink (zero-shot OTA)
        balanced = balance_gate(coax_samples)
        trn_s, val_s = train_val_split(balanced)
        eval_sets = {"coax_val": val_s, "airlink_zeroshot": airlink_samples}

    elif args.mode == "mixed":
        # Train on CoaxSweep + 80% AirLink; eval on held-out 20% AirLink
        if not airlink_samples:
            raise RuntimeError("No AirLink data — check --airlink_dirs")
        al_trn, al_val = train_val_split(airlink_samples, val_frac=0.2)
        combined = coax_samples + al_trn
        balanced = balance_gate(combined)
        trn_s, _  = train_val_split(balanced, val_frac=0.0)
        trn_s     = balanced   # use all balanced data for training
        val_s     = al_val     # AirLink holdout as primary validation
        # Also compute CoaxSweep val for comparison
        _, coax_val = train_val_split(balance_gate(coax_samples))
        eval_sets = {"airlink_held_out": al_val, "coax_val": coax_val}

    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    print(f"\n[split] trn={len(trn_s)}  primary_val={len(list(eval_sets.values())[0])}")

    # ── Build tensors ─────────────────────────────────────────────────────
    Xtr, gtr, mtr, str_, smtr, snr_mu, snr_sigma = build_tensors(trn_s)
    print(f"[data] SNR norm: mu={snr_mu:.1f}dB  sigma={snr_sigma:.1f}dB")

    eval_loaders = {}
    for name, es in eval_sets.items():
        if not es:
            continue
        Xv, gv, mv, sv, smv, *_ = build_tensors(es, snr_mu, snr_sigma)
        ds = TensorDataset(Xv, gv, mv, sv, smv)
        eval_loaders[name] = DataLoader(ds, batch_size=256, shuffle=False)

    if args.phase_aug:
        trn_ds = PhaseAugDataset(Xtr, gtr, mtr, str_, smtr)
        print("[data] Phase-rotation augmentation: ENABLED (forces amplitude-only mod features)")
    else:
        trn_ds = TensorDataset(Xtr, gtr, mtr, str_, smtr)
    trn_ld = DataLoader(trn_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # ── Model ────────────────────────────────────────────────────────────
    device = ("cuda" if (args.device == "auto" and torch.cuda.is_available())
              else args.device)
    print(f"\n[train] device={device}")

    model = build_model(args.arch, tasks, args.embed_dim, device)
    opt   = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    # Primary val loader (first in eval_sets) for checkpoint selection
    primary_loader = list(eval_loaders.values())[0]
    history   = []
    best_auc  = 0.0
    best_state = None

    for ep in range(args.epochs):
        tr_tot, tr_g, tr_m, tr_s = run_epoch(
            model, trn_ld, opt, device, tasks,
            args.w_gate, args.w_mod, args.w_snr, train=True)
        va_tot, va_g, va_m, va_s = run_epoch(
            model, primary_loader, None, device, tasks,
            args.w_gate, args.w_mod, args.w_snr, train=False)
        sched.step()

        history.append({"epoch": ep+1,
                         "train_total": tr_tot, "train_gate": tr_g,
                         "train_mod": tr_m, "train_snr": tr_s,
                         "val_total": va_tot, "val_gate": va_g,
                         "val_mod": va_m, "val_snr": va_s})

        if (ep+1) % 10 == 0:
            print(f"  ep={ep+1:3d}  tr=[{tr_tot:.4f}|g={tr_g:.4f}|m={tr_m:.4f}]"
                  f"  va=[{va_tot:.4f}|g={va_g:.4f}|m={va_m:.4f}]")

        if "gate" in tasks and SKLEARN and (ep+1) % 5 == 0:
            gp, gt, *_ = collect_preds(model, primary_loader, device, tasks)
            if len(gt) > 0:
                auc = roc_auc_score(gt, gp)
                if auc > best_auc:
                    best_auc = auc
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    print(f"\n[eval] best gate AUC={best_auc:.4f} (primary val)")

    # ── Evaluate on all eval sets ─────────────────────────────────────────
    all_metrics = {}
    for name, ldr in eval_loaders.items():
        print(f"\n--- Eval: {name} ---")
        gp, gt, mp, mt, sp, st = collect_preds(model, ldr, device, tasks)
        m = compute_metrics(gp, gt, mp, mt, sp, st, snr_mu, snr_sigma, tasks, tag=name)
        all_metrics[name] = m

    # ── Save ─────────────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    ckpt = {
        "state_dict": best_state,
        "arch": args.arch,
        "mode": args.mode,
        "tasks": list(tasks),
        "embed_dim": args.embed_dim,
        "pream_len": PREAM_LEN,
        "snr_mu": snr_mu,
        "snr_sigma": snr_sigma,
        "metrics": all_metrics,
    }
    aug_tag = "_phaseaug" if args.phase_aug else ""
    ckpt_name = f"mt_preamcnn_{args.arch}_{args.mode}{aug_tag}.pt"
    torch.save(ckpt, os.path.join(args.out_dir, ckpt_name))
    with open(os.path.join(args.out_dir, f"metrics_{args.arch}_{args.mode}{aug_tag}.json"), "w") as fp:
        json.dump(all_metrics, fp, indent=2)
    with open(os.path.join(args.out_dir, f"history_{args.arch}_{args.mode}{aug_tag}.json"), "w") as fp:
        json.dump(history, fp, indent=2)
    print(f"\n[saved] {os.path.join(args.out_dir, ckpt_name)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="MT-PreamCNN-Attn v2 training")

    # Dataset directories (glob patterns)
    ap.add_argument("--coax_dirs", nargs="+", default=[
        "rf_stream/ber_sweep_v4/run_*",
        "rf_stream/ber_sweep_v3/run_*",
        "rf_stream/ber_sweep_v2/run_*",
        "rf_stream/ber_sweep/run_20260428_233037",
        "rf_stream/ber_sweep/run_20260429_001722",
        "rf_stream/ber_sweep/run_20260429_003513",
        "rf_stream/ber_sweep/run_20260429_070949",
    ], help="CoaxSweep (cable) run directory globs")

    ap.add_argument("--airlink_dirs", nargs="+", default=[
        "/tmp/airlink_qpsk/run_*",
        "/tmp/airlink_qam16/run_*",
    ], help="AirLink (OTA) run directory globs")

    # Training mode
    ap.add_argument("--mode", default="coax",
                    choices=["coax", "airlink", "zeroshot", "mixed"],
                    help="coax=cable only; zeroshot=train coax eval airlink; "
                         "mixed=train coax+airlink eval held-out airlink")

    # Architecture
    ap.add_argument("--arch", default="attn", choices=["cnn", "attn"],
                    help="cnn=baseline MT-PreamCNN; attn=MT-PreamCNN-Attn (new)")

    # Task selection (ablation)
    ap.add_argument("--tasks", nargs="+", default=["gate", "mod", "snr"],
                    choices=["gate", "mod", "snr"])

    # Hyperparameters
    ap.add_argument("--embed_dim",  type=int,   default=256)
    ap.add_argument("--epochs",     type=int,   default=80)
    ap.add_argument("--batch_size", type=int,   default=64)
    ap.add_argument("--lr",         type=float, default=1e-3)
    ap.add_argument("--w_gate",     type=float, default=1.0)
    ap.add_argument("--w_mod",      type=float, default=0.5)
    ap.add_argument("--w_snr",      type=float, default=0.5)

    ap.add_argument("--out_dir", default="rf_stream/multitask_model_v2")
    ap.add_argument("--device",  default="auto")
    ap.add_argument("--no_cfo_correct", action="store_true",
                    help="Disable Schmidl-Cox CFO correction (for ablation)")
    ap.add_argument("--phase_aug", action="store_true",
                    help="Random global phase rotation each step to enforce "
                         "amplitude-only modulation features (fixes CFO domain shift)")

    args = ap.parse_args()
    main(args)
