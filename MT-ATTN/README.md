# MT-ATTN — Multi-Task Preamble Intelligence with Temporal Self-Attention

> **Paper:** *Multi-Task Preamble Intelligence: Joint Packet Detection, Modulation Classification, and SNR Estimation with Phase-Rotation Augmentation for OTA Generalisation* — submitted to **IEEE Globecom 2026**.

This subproject extends the AutoDataPHY pipeline with a unified **multi-task neural receiver** that jointly performs three preamble-level tasks from a single 800-sample IQ window:

1. **Gate** — packet vs. background detection (logit / AUC)
2. **Mod**  — QPSK vs. QAM16 modulation classification
3. **SNR**  — continuous SNR estimation (dB, regression)

Two architectures share the same CNN encoder:

- **MT-PreamCNN** (362 K params) — CNN encoder → adaptive pool → 3 heads
- **MT-PreamCNN-Attn** (428 K params) — CNN encoder → 4-head temporal self-attention over 50 time steps → adaptive pool → 3 heads

The key contribution beyond the architecture is **phase-rotation augmentation**, which solves the "modulation collapse" failure when a model trained on cable data (+330 Hz CFO) is evaluated over-the-air (−3000 Hz CFO).

---

## Headline Results

| Model | CoaxSweep Mod | CoaxSweep SNR RMSE | AirLink Gate AUC | AirLink Mod | AirLink SNR RMSE |
|---|---|---|---|---|---|
| CNN zero-shot | 100% | 0.91 dB | 0.8397 | 32.7% | 4.48 dB |
| CNN mixed     | 100% | 1.09 dB | 0.9826 | 72.0% | 1.70 dB |
| CNN mixed + phase-aug | 100% | 1.03 dB | 0.9847 | 89.6% | 1.44 dB |
| Attn zero-shot | 100% | 0.83 dB | 0.8851 | 48.9% | 4.36 dB |
| Attn mixed     | 100% | 0.84 dB | 0.9820 | 74.6% | 1.54 dB |
| **Attn mixed + phase-aug** | **100%** | **0.82 dB** | **0.9850** | **92.3%** | **1.37 dB** |

Phase-rotation augmentation lifts MT-PreamCNN modulation accuracy from 71.96 % → 89.62 % (+17.7 pp) and MT-PreamCNN-Attn from 74.61 % → **92.27 %** (+17.7 pp) under mixed training — architecture-independent benefit.

### Inference Latency

| Platform / Backend | CNN | Attn |
|---|---|---|
| x86 PyTorch-CPU | 5.01 ms | 2.43 ms |
| x86 ORT-CPU (ONNX Runtime) | 0.61 ms | 0.28 ms |
| Jetson Orin ORT-CPU | 0.62 ms | 1.45 ms |
| **Jetson Orin TRT-FP16** | **0.15 ms** | **0.25 ms** |

TensorRT FP16 on Jetson Orin achieves a 33× speedup over unoptimised PyTorch CPU.

---

## File Inventory

### Step 10 PHY receiver

| File | Role |
|---|---|
| `rf_stream_rx_step10phy.py` | Step-10 RX: MMSE equalisation, pilot_weight blending, auto threshold for QAM16, `--save_npz` capture mode |

### Training scripts

| File | Role |
|---|---|
| `train_multitask_v1.py` | First-generation multi-task CNN (no attention) |
| `train_multitask_v2.py` | **Final model** — CNN + temporal self-attention + phase-aug, supports `--mode {coax, zeroshot, airlink, mixed}` and `--arch {cnn, attn}` |

### Inference & deployment

| File | Role |
|---|---|
| `benchmark_inference.py` | Latency / throughput benchmark (PyTorch CPU/GPU, ONNX Runtime, TensorRT) — handles ONNX export with `AdaptiveAvgPool1d` patch |
| `generate_benchmark_figure.py` | Builds the cross-platform latency figure |
| `benchmark_results_x86.json` | Reference benchmark on x86 host |
| `benchmark_results_jetson.json` | Reference benchmark on Jetson Orin |

### Figures and paper

| Path | Role |
|---|---|
| `generate_multitask_figures_v2.py` | Builds fig15–fig18 from `metrics_*.json` |
| `paper_figures/fig15_v2_summary.pdf` | All 8 variants, 3 tasks |
| `paper_figures/fig16_v2_domain.pdf` | Cross-domain comparison (3 panels) |
| `paper_figures/fig17_v2_phaseaug.pdf` | Phase-aug before/after |
| `paper_figures/fig18_v2_roc_ota.pdf` | AirLink ROC curves |
| `paper_figures/fig19_v2_latency.pdf` | Deployment latency / throughput |
| `paper/globecom2026_multitask_phy.tex` | Submitted paper source |
| `paper/globecom2026_multitask_phy.pdf` | Compiled PDF |
| `TUTORIAL.md` | Step-by-step guide: data loading, training, evaluation, ONNX/TRT |

---

## Quick Start

```bash
# 1. Install dependencies
pip install numpy torch scikit-learn matplotlib
pip install onnxruntime              # CPU inference
pip install huggingface_hub          # HuggingFace dataset/models

# 2. Download the dataset (~31 GB) — see ../DATASET.md for partial downloads
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('lkk688/ofdm-preamble-intelligence', repo_type='dataset',
                   local_dir='./data')
"

# 3. Train the best variant (Attn + mixed + phase-aug)
python3 train_multitask_v2.py --arch attn --mode mixed --phase_aug \
    --airlink_dirs ./data/airlink/*/run_* --epochs 80

# 4. (Optional) Benchmark inference on x86 / Jetson
python3 benchmark_inference.py --export_onnx \
    --backend pytorch_cpu ort_cpu --archs cnn attn
```

For the full walkthrough — data loading, evaluation, deployment, paper-reproduction — see [TUTORIAL.md](./TUTORIAL.md).

---

## HuggingFace Companion Repositories

| Resource | URL |
|---|---|
| **Dataset (31 GB, 17,469 NPZ)** | https://huggingface.co/datasets/lkk688/ofdm-preamble-intelligence |
| **Pretrained models (8 checkpoints + ONNX)** | https://huggingface.co/lkk688/mt-preamcnn |

Download the best model directly:

```python
from huggingface_hub import hf_hub_download
import torch

path = hf_hub_download("lkk688/mt-preamcnn", "mt_preamcnn_attn_mixed_phaseaug.pt")
ckpt = torch.load(path, map_location="cpu")
print(ckpt["metrics"]["airlink"]["mod_acc"])   # 0.9227
```

See [../DATASET.md](../DATASET.md) for the full dataset card.

---

## Phase-Rotation Augmentation — the OTA fix

CoaxSweep training data has a stable **+330 Hz CFO** (Pluto TCXO offset). AirLink captures the same Pluto pair over the air at the same carrier — but the offset flips sign and grows in magnitude to **−3000 Hz**. Zero-shot transfer collapses to 32.7 % modulation accuracy: the CNN learns a CFO-sign-dependent phase-roll feature that no longer points in the right direction.

The fix is one line in the data loader. For each training sample we draw a global rotation `θ ∼ U[0, 2π)` and apply it to the complex IQ:

```
x̃[n] = x[n] · exp(jθ)
```

Both QPSK (single-amplitude ring) and 16-QAM (three rings) are invariant under this rotation, so any phase-sensitive feature is wiped out. The model is forced to use amplitude structure, which is preserved across the CFO domain shift.

Effect on AirLink modulation accuracy:
- MT-PreamCNN, mixed: 71.96 % → **89.62 %**  (+17.7 pp)
- MT-PreamCNN-Attn, mixed: 74.61 % → **92.27 %**  (+17.7 pp)

Both architectures gain *exactly* +17.7 pp — confirming the bias is fully corrected regardless of architecture.

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
