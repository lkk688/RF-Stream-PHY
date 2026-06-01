# MT-PreamCNN Tutorial: Multi-Task OFDM Preamble Intelligence

A step-by-step guide to loading the dataset, training the models, evaluating results, and running inference — locally or from HuggingFace.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Dataset Structure](#2-dataset-structure)
3. [Loading Data Locally](#3-loading-data-locally)
4. [Loading Data from HuggingFace](#4-loading-data-from-huggingface)
5. [Training the Model](#5-training-the-model)
6. [Evaluating a Trained Model](#6-evaluating-a-trained-model)
7. [Using Pretrained Checkpoints](#7-using-pretrained-checkpoints)
8. [Running Inference on New Captures](#8-running-inference-on-new-captures)
9. [Optimized Deployment (ONNX / TensorRT)](#9-optimized-deployment-onnx--tensorrt)
10. [Reproducing Paper Results](#10-reproducing-paper-results)

---

## 1. Prerequisites

### Hardware
- PlutoSDR (TX and RX), or use the provided offline dataset only
- Optional: NVIDIA Jetson Orin for TensorRT deployment benchmark

### Software

```bash
# Python 3.10+
pip install numpy torch torchvision scikit-learn matplotlib
pip install onnxruntime          # CPU inference
pip install huggingface_hub datasets   # HuggingFace access
```

### Repository

```bash
git clone https://github.com/lkk688/AIsensing.git
cd AIsensing
```

---

## 2. Dataset Structure

The dataset spans two RF domains collected at 2400 MHz with a PlutoSDR testbed:

```
rf_stream/
├── ber_sweep_v2/          # CoaxSweep-I  (cable, 6 runs, 1,403 NPZ, 2.5 GB)
├── ber_sweep_v3/          # CoaxSweep-II (cable, 6 runs, 3,282 NPZ, 5.8 GB)
├── ber_sweep_v4/          # CoaxSweep-III(cable, 3 runs, 3,078 NPZ, 5.6 GB)
├── ber_sweep_v6/          # CoaxSweep-IV (cable, 4 runs,   842 NPZ, 1.6 GB)
└── airlink/
    ├── airlink_qpsk/      # AirLink-QPSK-I   (OTA, 1 run,   768 NPZ, 1.4 GB)
    ├── airlink2_qpsk/     # AirLink-QPSK-II  (OTA, 2 runs,  753 NPZ, 1.4 GB)
    ├── airlink_qam16/     # AirLink-QAM16    (OTA, 1 run,   807 NPZ, 1.5 GB)
    └── airlink3_qpsk/     # AirLink-QPSK-III (OTA, 7 runs, 6,536 NPZ, 12 GB)
```

**Each NPZ file is one packet capture attempt (~2 MB).**

```
run_YYYYMMDD_HHMMSS/
├── cap_000001_ok.npz    ← CRC-pass packet (gate label = 1)
├── cap_000002_bg.npz    ← failed capture (gate label = 0)
└── rx_config.json       ← run-level config (gain, frequency, mode)
```

### Inspecting a single NPZ

```python
import numpy as np, json

with np.load("rf_stream/ber_sweep_v3/run_20260429_150637/cap_000001_ok.npz",
             allow_pickle=True) as d:
    rxw  = d["rxw"]         # (262144,) complex64 — full RX window at 3 MHz
    meta = json.loads(d["meta_json"].item())

print(rxw.shape, rxw.dtype)   # (262144,) complex64
print(meta)
# {'stf_idx': 12800, 'snr_db': 24.1, 'mod': 'QPSK', 'gate': 1,
#  'gain_tx': 40, 'gain_rx': 40, 'cfo_hz': 331.2, 'timestamp': '...'}
```

**Labels embedded in filename and meta_json:**
- `_ok.npz` → `gate = 1` (CRC pass); `_bg.npz` or `_fail.npz` → `gate = 0`
- `meta['mod']` → `"QPSK"` or `"QAM16"` → maps to `mod_label = 0 or 1`
- `meta['snr_db']` → float (estimated from pilot comparison)

---

## 3. Loading Data Locally

The training script's loader handles everything: preamble extraction, CFO correction, label assignment, and normalization.

```python
import glob
from rf_stream.train_multitask_v2 import load_dataset, build_tensors

# ── Load CoaxSweep (cable) ─────────────────────────────────────────────────
coax_dirs = sorted(glob.glob("rf_stream/ber_sweep_v*/run_*"))
coax_samples = load_dataset(coax_dirs, domain="coax", cfo_correct=True)
print(f"CoaxSweep: {len(coax_samples)} samples")
# CoaxSweep: 18455 samples

# ── Load AirLink (OTA) ─────────────────────────────────────────────────────
airlink_dirs = sorted(glob.glob("rf_stream/airlink/airlink*/run_*"))
airlink_samples = load_dataset(airlink_dirs, domain="airlink", cfo_correct=True)
print(f"AirLink: {len(airlink_samples)} samples")
# AirLink: 7465 samples

# ── Inspect a sample ──────────────────────────────────────────────────────
s = coax_samples[0]
print(s.keys())
# dict_keys(['pream', 'gate', 'mod', 'snr_db', 'has_snr', 'domain'])
print(s['pream'].shape)   # (1600,) float32 — 800 IQ pairs, interleaved real/imag
print(s['gate'])          # 1
print(s['mod'])           # 0 (QPSK) or 1 (QAM16)
print(s['snr_db'])        # 24.1
```

### What `load_dataset` does internally

```
NPZ file
  └── rxw[stf_idx : stf_idx+800]     extract 800-sample preamble window
       └── Schmidl-Cox CFO correction  rotate by -j·2π·cfo_hz·t/Fs
            └── interleave real/imag   out[0::2]=I, out[1::2]=Q  → float32 (1600,)
```

### Building PyTorch tensors

```python
from rf_stream.train_multitask_v2 import build_tensors, train_val_split, balance_gate

# Optional: balance gate classes (equal ok/fail)
coax_balanced = balance_gate(coax_samples)

# 80/20 train/val split
train_s, val_s = train_val_split(coax_balanced, val_frac=0.2)

# Build tensors — returns (X, gate, mod, snr_norm, snr_mask, snr_mu, snr_sigma)
X_tr, g_tr, m_tr, s_tr, sm_tr, snr_mu, snr_sigma = build_tensors(train_s)
X_va, g_va, m_va, s_va, sm_va, _,      _         = build_tensors(
    val_s, snr_mu=snr_mu, snr_sigma=snr_sigma)

print(X_tr.shape)   # (N, 1600)
print(g_tr.shape)   # (N,)  float32 {0, 1}
print(m_tr.shape)   # (N,)  float32 {-1=unknown, 0=QPSK, 1=QAM16}
print(s_tr.shape)   # (N,)  float32 (z-score normalized)
```

---

## 4. Loading Data from HuggingFace

The dataset is published at `lkk688/ofdm-preamble-intelligence`.

### Option A — Download full split directly

```python
from huggingface_hub import snapshot_download
import os, glob, numpy as np

# Download CoaxSweep-II (~5.8 GB)
local_dir = snapshot_download(
    repo_id="lkk688/ofdm-preamble-intelligence",
    repo_type="dataset",
    allow_patterns="coax/ber_sweep_v3/**",
    local_dir="./hf_data",
)
npz_files = glob.glob(os.path.join(local_dir, "coax/ber_sweep_v3/**/*.npz"),
                      recursive=True)
print(f"Downloaded {len(npz_files)} NPZ files")
```

### Option B — Download a single sub-split for quick experiments

```python
from huggingface_hub import snapshot_download

# Just AirLink QAM16 (807 files, ~1.5 GB) — good for quick experiments
local_dir = snapshot_download(
    repo_id="lkk688/ofdm-preamble-intelligence",
    repo_type="dataset",
    allow_patterns="airlink/airlink_qam16/**",
    local_dir="./hf_data",
)
```

### Option C — Stream individual files without full download

```python
from huggingface_hub import hf_hub_download
import numpy as np, json

# Download one specific file
path = hf_hub_download(
    repo_id="lkk688/ofdm-preamble-intelligence",
    repo_type="dataset",
    filename="coax/ber_sweep_v3/run_20260429_150637/cap_000001_ok.npz",
)
with np.load(path, allow_pickle=True) as d:
    rxw  = d["rxw"]
    meta = json.loads(d["meta_json"].item())
print(f"SNR: {meta['snr_db']:.1f} dB  mod: {meta['mod']}")
```

---

## 5. Training the Model

### Quick start — replicate the best result

```bash
# Best result: MT-PreamCNN-Attn, mixed training, phase-rotation augmentation
# Achieves 92.3% AirLink modulation accuracy, 100% CoaxSweep, AUC 0.985

python3 rf_stream/train_multitask_v2.py \
    --arch attn \
    --mode mixed \
    --phase_aug \
    --airlink_dirs rf_stream/airlink/airlink_qpsk/run_* \
                   rf_stream/airlink/airlink2_qpsk/run_* \
                   rf_stream/airlink/airlink_qam16/run_* \
                   rf_stream/airlink/airlink3_qpsk/run_* \
    --epochs 80
```

Output is saved to `rf_stream/multitask_model_v2/`:
- `mt_preamcnn_attn_mixed_phaseaug.pt` — checkpoint
- `metrics_attn_mixed_phaseaug.json`   — evaluation metrics
- `history_attn_mixed_phaseaug.json`   — per-epoch loss curves

### Training modes explained

| `--mode` | Training data | Eval data | Use case |
|---|---|---|---|
| `coax` | CoaxSweep only | CoaxSweep held-out | Within-domain baseline |
| `zeroshot` | CoaxSweep only | **AirLink** (no fine-tune) | Zero-shot OTA generalization |
| `airlink` | AirLink only | AirLink held-out | OTA-only training |
| `mixed` | CoaxSweep + AirLink | AirLink held-out | **Best OTA performance** |

### Architecture options

| `--arch` | Model | Params | Notes |
|---|---|---|---|
| `cnn` | MT-PreamCNN | 362K | Faster on ARM CPU |
| `attn` | MT-PreamCNN-Attn | 428K | Better SNR RMSE, best OTA mod accuracy |

### Full argument reference

```bash
python3 rf_stream/train_multitask_v2.py --help
# Key arguments:
#   --arch {cnn,attn}          model architecture
#   --mode {coax,zeroshot,airlink,mixed}  training domain
#   --airlink_dirs DIR [DIR …] run directories for AirLink data
#   --epochs N                 training epochs (default 80)
#   --batch_size N             (default 256)
#   --lr LR                    learning rate (default 1e-3)
#   --embed_dim N              embedding dimension (default 256)
#   --phase_aug                enable phase-rotation augmentation
#   --no_gate / --no_mod / --no_snr   disable individual task heads
```

### Phase-rotation augmentation (the key fix for OTA)

Without augmentation, CNN trained on CoaxSweep (+330 Hz CFO) predicts QAM16 for
all AirLink samples because −3000 Hz CFO inverts the phase-roll feature. The fix:

```python
# Enabled by --phase_aug flag
# Each training sample gets a random global rotation θ ~ U[0, 2π)
theta = random() * 2π
I_new =  I·cos(θ) - Q·sin(θ)
Q_new =  I·sin(θ) + Q·cos(θ)
# Forces the model to use amplitude rings (QPSK: 1 ring, QAM16: 3 rings)
# instead of phase-sensitive features
```

Impact: modulation accuracy jumps from 32.7% → 62.9% (zero-shot) and 71.9% → 89.6% (mixed) for CNN; from 74.6% → **92.3%** (mixed) for Attn.

### Training all 8 variants in sequence

```bash
for arch in cnn attn; do
  for mode in zeroshot mixed; do
    for aug in "" "--phase_aug"; do
      suffix="${aug:+_phaseaug}"
      python3 rf_stream/train_multitask_v2.py \
          --arch $arch --mode $mode $aug \
          --airlink_dirs rf_stream/airlink/airlink*/run_* \
          --epochs 80
    done
  done
done
```

---

## 6. Evaluating a Trained Model

### Reading saved metrics

```python
import json

with open("rf_stream/multitask_model_v2/metrics_attn_mixed_phaseaug.json") as f:
    m = json.load(f)

# Coax (cable) domain
print(f"Coax  AUC:     {m['coax']['gate_auc']:.4f}")
print(f"Coax  Mod acc: {m['coax']['mod_acc']:.1%}")
print(f"Coax  SNR RMSE:{m['coax']['snr_rmse']:.2f} dB")

# AirLink (OTA) domain
print(f"OTA   AUC:     {m['airlink']['gate_auc']:.4f}")
print(f"OTA   Mod acc: {m['airlink']['mod_acc']:.1%}")
print(f"OTA   SNR RMSE:{m['airlink']['snr_rmse']:.2f} dB")
```

### Running evaluation manually

```python
import torch, json, glob, numpy as np
from rf_stream.train_multitask_v2 import (
    load_dataset, build_tensors, MultiTaskPreamCNNAttn)
from sklearn.metrics import roc_auc_score, accuracy_score

# Load checkpoint
ckpt  = torch.load("rf_stream/multitask_model_v2/mt_preamcnn_attn_mixed_phaseaug.pt",
                   map_location="cpu")
model = MultiTaskPreamCNNAttn(embed_dim=ckpt["embed_dim"], tasks=ckpt["tasks"])
model.load_state_dict(ckpt["state_dict"])
model.eval()

snr_mu    = ckpt["snr_mu"]
snr_sigma = ckpt["snr_sigma"]

# Load AirLink test data
airlink_dirs = sorted(glob.glob("rf_stream/airlink/airlink*/run_*"))
samples = load_dataset(airlink_dirs, domain="airlink")
X, gate, mod, snr_n, snr_mask, _, _ = build_tensors(
    samples, snr_mu=snr_mu, snr_sigma=snr_sigma)

# Inference
with torch.no_grad():
    out = model(X)
    gate_prob = torch.sigmoid(out["gate"]).numpy()
    mod_prob  = torch.sigmoid(out["mod"]).numpy()
    snr_pred  = out["snr"].numpy() * snr_sigma + snr_mu   # denormalize

# Metrics
gate_np = gate.numpy()
mod_np  = mod.numpy()

auc = roc_auc_score(gate_np, gate_prob)
print(f"Gate AUC: {auc:.4f}")

# Modulation accuracy (only on gate=1 AND known-mod samples)
mask = (gate_np == 1) & (mod_np >= 0)
pred_mod = (mod_prob[mask] > 0.5).astype(int)
true_mod = mod_np[mask].astype(int)
print(f"Mod acc:  {accuracy_score(true_mod, pred_mod):.1%}")

# SNR RMSE (only on samples with ground-truth SNR and gate=1)
snr_mask_np = snr_mask.numpy() & (gate_np == 1)
if snr_mask_np.sum() > 0:
    snr_true = snr_n[snr_mask_np].numpy() * snr_sigma + snr_mu
    rmse = float(np.sqrt(np.mean((snr_pred[snr_mask_np] - snr_true)**2)))
    print(f"SNR RMSE: {rmse:.2f} dB")
```

### Generating publication figures

```bash
# Regenerate all paper figures from saved metrics JSONs
python3 rf_stream/generate_multitask_figures_v2.py

# Output in rf_stream/paper_figures/:
#   fig15_v2_summary.pdf    — all 8 variants, 3 tasks
#   fig16_v2_domain.pdf     — cross-domain comparison (3-panel)
#   fig17_v2_phaseaug.pdf   — phase-aug before/after
#   fig18_v2_roc_ota.pdf    — AirLink ROC curves
```

---

## 7. Using Pretrained Checkpoints

### Download from HuggingFace

```python
from huggingface_hub import hf_hub_download
import torch

# Best model
path = hf_hub_download("lkk688/mt-preamcnn",
                        "mt_preamcnn_attn_mixed_phaseaug.pt")
ckpt = torch.load(path, map_location="cpu")
print(ckpt["metrics"]["airlink"]["mod_acc"])   # 0.9227
```

### Model inventory

| Filename | Arch | Mode | AirLink Mod | Gate AUC |
|---|---|---|---|---|
| `mt_preamcnn_attn_mixed_phaseaug.pt` | Attn | mixed+aug | **92.3%** | 0.985 |
| `mt_preamcnn_cnn_mixed_phaseaug.pt`  | CNN  | mixed+aug | 89.6% | 0.985 |
| `mt_preamcnn_attn_mixed.pt`          | Attn | mixed     | 74.6% | 0.982 |
| `mt_preamcnn_cnn_mixed.pt`           | CNN  | mixed     | 72.0% | 0.983 |
| `mt_preamcnn_attn_zeroshot_phaseaug.pt` | Attn | zeroshot+aug | 61.0% | 0.873 |
| `mt_preamcnn_cnn_zeroshot_phaseaug.pt`  | CNN  | zeroshot+aug | 62.9% | 0.875 |
| `mt_preamcnn_attn_zeroshot.pt`       | Attn | zeroshot  | 48.9% | 0.885 |
| `mt_preamcnn_cnn_zeroshot.pt`        | CNN  | zeroshot  | 32.7% | 0.840 |

---

## 8. Running Inference on New Captures

### Step 1 — Collect a new capture with the step10 RX

```bash
# On the RX host (with PlutoSDR connected):
python3 rf_stream/rf_stream_rx_step10phy.py \
    --save_npz --out_dir /tmp/my_captures \
    --pilot_weight 0.5 --auto_z_th

# Produces /tmp/my_captures/run_YYYYMMDD_HHMMSS/cap_NNNNNN_{ok,bg}.npz
```

### Step 2 — Extract preamble from a new NPZ

```python
import numpy as np, json

PREAM_LEN = 800
FS_HZ     = 3e6
STF_L     = 32   # Schmidl-Cox half-period

def extract_preamble(npz_path: str) -> np.ndarray:
    """Return (1600,) float32 interleaved I/Q preamble with CFO correction."""
    with np.load(npz_path, allow_pickle=True) as d:
        rxw  = np.array(d["rxw"], dtype=np.complex64)
        meta = json.loads(d["meta_json"].item())

    stf = int(meta.get("stf_idx", 0))
    seg = rxw[stf : stf + PREAM_LEN]
    if len(seg) < PREAM_LEN:
        seg = np.pad(seg, (0, PREAM_LEN - len(seg)))

    # Schmidl-Cox CFO estimation and correction
    pairs = len(seg) // STF_L - 1
    if pairs > 0:
        r      = np.dot(seg[:pairs*STF_L].conj(), seg[STF_L:(pairs+1)*STF_L])
        cfo_hz = float(np.angle(r) / (2.0 * np.pi * STF_L / FS_HZ))
        t      = np.arange(len(seg), dtype=np.float32)
        seg    = (seg * np.exp(-1j * 2.0 * np.pi * cfo_hz * t / FS_HZ)).astype(np.complex64)

    out       = np.empty(PREAM_LEN * 2, dtype=np.float32)
    out[0::2] = seg.real
    out[1::2] = seg.imag
    return out
```

### Step 3 — Run the multi-task model

```python
import torch
from rf_stream.train_multitask_v2 import MultiTaskPreamCNNAttn

# Load checkpoint
ckpt  = torch.load("rf_stream/multitask_model_v2/mt_preamcnn_attn_mixed_phaseaug.pt",
                   map_location="cpu")
model = MultiTaskPreamCNNAttn(embed_dim=ckpt["embed_dim"], tasks=ckpt["tasks"])
model.load_state_dict(ckpt["state_dict"])
model.eval()

snr_mu    = ckpt["snr_mu"]
snr_sigma = ckpt["snr_sigma"]

# Single sample inference
pream = extract_preamble("/tmp/my_captures/run_X/cap_000001_ok.npz")
x     = torch.from_numpy(pream).unsqueeze(0)   # (1, 1600)

with torch.no_grad():
    out = model(x)

gate_prob = float(torch.sigmoid(out["gate"]))
mod_prob  = float(torch.sigmoid(out["mod"]))
snr_est   = float(out["snr"]) * snr_sigma + snr_mu

print(f"Gate:  {'PASS' if gate_prob > 0.5 else 'FAIL'}  ({gate_prob:.3f})")
print(f"Mod:   {'QAM16' if mod_prob > 0.5 else 'QPSK'} ({mod_prob:.3f})")
print(f"SNR:   {snr_est:.1f} dB")
```

### Batch inference for a full run directory

```python
import glob, torch
import numpy as np

npz_files = sorted(glob.glob("/tmp/my_captures/run_*/cap_*.npz"))
preambles = np.stack([extract_preamble(f) for f in npz_files])   # (N, 1600)
X         = torch.from_numpy(preambles)

with torch.no_grad():
    out = model(X)

gate_probs = torch.sigmoid(out["gate"]).numpy()
mod_probs  = torch.sigmoid(out["mod"]).numpy()
snr_ests   = out["snr"].numpy() * snr_sigma + snr_mu

for f, g, m, s in zip(npz_files, gate_probs, mod_probs, snr_ests):
    label = "ok" if "_ok" in f else "bg"
    pred  = "PASS" if g > 0.5 else "FAIL"
    mod   = "QAM16" if m > 0.5 else "QPSK"
    print(f"{f.split('/')[-1]}  true={label}  pred={pred}  mod={mod}  SNR={s:.1f}dB")
```

---

## 9. Optimized Deployment (ONNX / TensorRT)

### Export to ONNX

```bash
python3 rf_stream/benchmark_inference.py --export_onnx \
    --archs cnn attn \
    --ckpt_dir rf_stream/multitask_model_v2 \
    --onnx_dir rf_stream/onnx
# Creates rf_stream/onnx/mt_preamcnn_{cnn,attn}.onnx
```

Or download the pre-exported ONNX files:

```python
from huggingface_hub import hf_hub_download

cnn_onnx  = hf_hub_download("lkk688/mt-preamcnn", "onnx/mt_preamcnn_cnn.onnx")
attn_onnx = hf_hub_download("lkk688/mt-preamcnn", "onnx/mt_preamcnn_attn.onnx")
```

### CPU inference with ONNX Runtime

```python
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession(
    "rf_stream/onnx/mt_preamcnn_attn.onnx",
    providers=["CPUExecutionProvider"])

# Single-sample inference — 0.28 ms on x86, 1.45 ms on Jetson ARM
x = pream[np.newaxis, :]   # (1, 1600) float32
gate_logit, mod_logit, snr_norm = sess.run(None, {"preamble": x})

gate_prob = 1 / (1 + np.exp(-gate_logit[0]))   # sigmoid
mod_prob  = 1 / (1 + np.exp(-mod_logit[0]))
snr_est   = snr_norm[0] * snr_sigma + snr_mu
```

### GPU inference with TensorRT FP16 (Jetson Orin)

```bash
# On Jetson: build and benchmark TRT engine
python3 rf_stream/benchmark_inference.py \
    --backend trtexec \
    --archs cnn attn \
    --onnx_dir ~/rf_stream/onnx

# Latency at batch=1:  CNN 0.15 ms  |  Attn 0.25 ms
# Throughput at bs=64: CNN 2189 sps  |  Attn 1173 sps
```

### Latency comparison

| Platform / Backend | CNN | Attn |
|---|---|---|
| x86 PyTorch CPU | 5.01 ms | 2.43 ms |
| x86 ORT CPU | 0.61 ms | 0.28 ms |
| Jetson ORT CPU | 0.62 ms | 1.45 ms |
| **Jetson TRT FP16** | **0.15 ms** | **0.25 ms** |

TensorRT FP16 on Jetson Orin achieves 33× speedup over unoptimized PyTorch CPU.

---

## 10. Reproducing Paper Results

### Full benchmark: all 8 variants

```bash
# Train all models (~6-8 hours total on a GPU):
bash -c '
for arch in cnn attn; do
  for mode in zeroshot mixed; do
    python3 rf_stream/train_multitask_v2.py \
        --arch $arch --mode $mode \
        --airlink_dirs rf_stream/airlink/airlink*/run_* \
        --epochs 80
    python3 rf_stream/train_multitask_v2.py \
        --arch $arch --mode $mode --phase_aug \
        --airlink_dirs rf_stream/airlink/airlink*/run_* \
        --epochs 80
  done
done'
```

### Generate all paper figures

```bash
python3 rf_stream/generate_multitask_figures_v2.py   # fig15–18
python3 rf_stream/generate_benchmark_figure.py        # fig19 (latency)
ls rf_stream/paper_figures/
# fig15_v2_summary.pdf
# fig16_v2_domain.pdf
# fig17_v2_phaseaug.pdf
# fig18_v2_roc_ota.pdf
# fig19_v2_latency.pdf
# fig19_v2_latency_bar.pdf
```

### Compile the paper

```bash
cd rf_stream/paper
pdflatex globecom2026_multitask_phy.tex
# Output: globecom2026_multitask_phy.pdf (10 pages)
```

### Expected results (final numbers from paper)

| Model | CoaxSweep Mod | CoaxSweep SNR RMSE | AirLink Gate AUC | AirLink Mod | AirLink SNR RMSE |
|---|---|---|---|---|---|
| CNN zero-shot | 100% | 0.91 dB | 0.8397 | 32.7% | 4.48 dB |
| CNN mixed | 100% | 1.09 dB | 0.9826 | 72.0% | 1.70 dB |
| CNN zero-shot+aug | 100% | 0.81 dB | 0.8754 | 62.9% | 3.90 dB |
| CNN mixed+aug | 100% | 1.03 dB | 0.9847 | 89.6% | 1.44 dB |
| Attn zero-shot | 100% | 0.83 dB | 0.8851 | 48.9% | 4.36 dB |
| Attn mixed | 100% | 0.84 dB | 0.9820 | 74.6% | 1.54 dB |
| Attn zero-shot+aug | 100% | 0.81 dB | 0.8725 | 61.0% | 4.22 dB |
| **Attn mixed+aug** | **100%** | **0.82 dB** | **0.9850** | **92.3%** | **1.37 dB** |

---

## Citation

```bibtex
@inproceedings{liu2026multitask,
  title     = {Multi-Task Preamble Intelligence: Joint Packet Detection,
               Modulation Classification, and {SNR} Estimation with
               Phase-Rotation Augmentation for {OTA} Generalisation},
  author    = {Liu, Kaikai},
  booktitle = {Proc. IEEE Global Communications Conference (Globecom)},
  year      = {2026}
}
```

**Dataset:** `lkk688/ofdm-preamble-intelligence` on HuggingFace  
**Models:** `lkk688/mt-preamcnn` on HuggingFace  
**Code:** `github.com/lkk688/AIsensing`
