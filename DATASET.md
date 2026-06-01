# OFDM Preamble Intelligence Dataset

**HuggingFace repo:** [`lkk688/ofdm-preamble-intelligence`](https://huggingface.co/datasets/lkk688/ofdm-preamble-intelligence)
**License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
**Total:** 17,469 NPZ captures, ~31 GB raw IQ data
**Carrier:** 2400 MHz, 3 MHz sample rate
**Hardware:** ADALM-PlutoSDR (TX on Jetson Orin, RX on x86 Ubuntu)

This dataset is shared across both subprojects in this repo (`AutoDataPHY` and `MT-ATTN`).

---

## Dataset Overview

Raw IQ preamble captures from a PlutoSDR OFDM testbed, collected across two experimental domains:

| Domain | Description | NPZ files |
|---|---|---|
| **CoaxSweep** | Cable (coaxial) link, gain sweep 25–72 dB | 8,605 |
| **AirLink**   | Over-the-air (OTA) at ~40–50 cm, 2.4 GHz WiFi environment | 8,864 |

---

## Experimental Setup

- **TX:** NVIDIA Jetson Orin running `rf_stream_tx_step6phy.py` (PlutoSDR USB)
- **RX:** x86 host running `rf_stream_rx_step{8,9,10}phy.py` (PlutoSDR USB)
- **Carrier frequency:** 2400 MHz (CoaxSweep / AirLink-OTA)
- **Sample rate:** 3 MHz
- **FFT size:** 64 subcarriers, CP length 16
- **Preamble:** 800 samples (STF + LTF), Schmidl–Cox CFO estimation
- **Modulations:** QPSK, QAM16 (alternating per packet)
- **TX gain:** swept 25–72 dB (CoaxSweep), fixed 72 dB (AirLink)
- **RX equalisation:** MMSE with `pilot_weight=0.5`, auto threshold for QAM16

---

## Sub-datasets

### CoaxSweep (coaxial cable baseline)

Collected under controlled coaxial cable conditions with known SNR ground truth.
CFO: stable **+330 Hz** (Pluto TCXO offset). BER = 0 on all decoded packets.

| Sub-split | HF path | Runs | NPZ | Notes |
|---|---|---|---|---|
| CoaxSweep-I   | `coax/ber_sweep_v2/` | 6 | 1,403 | Step8 PHY, QPSK only |
| CoaxSweep-II  | `coax/ber_sweep_v3/` | 6 | 3,282 | Step9 PHY, QPSK + QAM16 |
| CoaxSweep-III | `coax/ber_sweep_v4/` | 3 | 3,078 | Step10 MMSE, pilot_weight sweep |
| CoaxSweep-IV  | `coax/ber_sweep_v6/` | 4 |   842 | QAM16 mixed, hard-negative gate examples |

### AirLink (over-the-air)

Collected at 2400 MHz in a 2.4 GHz WiFi-congested indoor environment.
CFO: **−3000 Hz** (sign reversal vs. CoaxSweep — the domain shift studied in the MT-ATTN paper).
WiFi-triggered fail captures serve as hard-negative examples for the gate head.

| Sub-split | HF path | Runs | NPZ | Notes |
|---|---|---|---|---|
| AirLink-QPSK-I   | `airlink/airlink_qpsk/`  | 1 |   768 | First OTA collection |
| AirLink-QPSK-II  | `airlink/airlink2_qpsk/` | 2 |   753 | Panel antenna, repeat |
| AirLink-QAM16    | `airlink/airlink_qam16/` | 1 |   807 | OTA QAM16, auto_z_th |
| AirLink-QPSK-III | `airlink/airlink3_qpsk/` | 7 | 6,536 | 50 cm, WiFi env, large collection |

---

## NPZ File Format

Each `.npz` file is one packet capture attempt (~2 MB). Keys:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| `rxw`            | (262 144,) | complex64 | Full receive window (IQ samples at 3 MHz) |
| `corr`           | (N,)       | float32   | STF correlation magnitude |
| `corr_norm`      | (N,)       | float32   | Normalised correlation |
| `topk_idx`       | (K,)       | int32     | Top-K preamble candidate positions |
| `topk_corr_norm` | (K,)       | float32   | Correlation at top-K positions |
| `topk_ncc`       | (K,)       | float32   | NCC scores at top-K positions |
| `energy_ds`      | (N,)       | float32   | Downsampled energy envelope |
| `meta_json`      | scalar     | str       | JSON string with metadata (below) |

### `meta_json` fields

```json
{
  "stf_idx":   12345,            // preamble start sample index
  "snr_db":    22.4,             // estimated SNR (dB), from pilot comparison
  "mod":       "QPSK",           // "QPSK" or "QAM16"
  "gate":      1,                // 1 = CRC pass, 0 = fail
  "gain_tx":   72,               // TX gain (dB)
  "gain_rx":   40,               // RX gain (dB)
  "cfo_hz":    -3012.5,          // estimated CFO (Hz)
  "timestamp": "2026-05-03T12:18:05"
}
```

**Filename convention:** `cap_NNNNNN_ok.npz` (CRC pass) or `cap_NNNNNN_bg.npz` (fail / background).

---

## Loading the Dataset

### Download a single sub-split

```python
from huggingface_hub import snapshot_download

# Just AirLink-QAM16 (807 files, ~1.5 GB) — good for quick experiments
local_dir = snapshot_download(
    repo_id="lkk688/ofdm-preamble-intelligence",
    repo_type="dataset",
    allow_patterns="airlink/airlink_qam16/**",
    local_dir="./hf_data",
)
```

### Download the full dataset

```python
from huggingface_hub import snapshot_download

local_dir = snapshot_download(
    repo_id="lkk688/ofdm-preamble-intelligence",
    repo_type="dataset",
    local_dir="./hf_data",
)
```

### Stream a single file without full download

```python
from huggingface_hub import hf_hub_download
import numpy as np, json

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

## Preamble Extraction

The training pipeline extracts 800 complex samples starting at `stf_idx` and applies
Schmidl–Cox CFO correction before converting to real/imag interleaved float32 (1600 values):

```python
import numpy as np, json

PREAM_LEN = 800
FS_HZ     = 3e6
STF_L     = 32   # Schmidl-Cox half-period

def extract_preamble(npz_path):
    with np.load(npz_path, allow_pickle=True) as d:
        meta = json.loads(d["meta_json"].item())
        rxw  = d["rxw"].astype(np.complex64)
    seg = rxw[meta["stf_idx"] : meta["stf_idx"] + PREAM_LEN]
    if len(seg) < PREAM_LEN:
        seg = np.pad(seg, (0, PREAM_LEN - len(seg)))
    # Schmidl-Cox CFO correction
    pairs = len(seg) // STF_L - 1
    r     = np.dot(seg[:pairs * STF_L].conj(), seg[STF_L : (pairs+1) * STF_L])
    cfo   = float(np.angle(r) / (2 * np.pi * STF_L / FS_HZ))
    t     = np.arange(len(seg), dtype=np.float32)
    seg   = (seg * np.exp(-1j * 2 * np.pi * cfo * t / FS_HZ)).astype(np.complex64)
    out       = np.empty(PREAM_LEN * 2, dtype=np.float32)
    out[0::2] = seg.real
    out[1::2] = seg.imag
    return out  # shape (1600,)
```

The complete reference loader (with batched DataLoader integration) is in `MT-ATTN/train_multitask_v2.py` — function `load_dataset` / class `PhaseAugDataset`.

---

## Pretrained Models

Available at [`lkk688/mt-preamcnn`](https://huggingface.co/lkk688/mt-preamcnn):

| File | Params | CoaxSweep Mod | AirLink Mod | AirLink Gate AUC |
|---|---|---|---|---|
| `mt_preamcnn_attn_mixed_phaseaug.pt` | 428 K | 100 % | **92.3 %** | **0.9850** |
| `mt_preamcnn_cnn_mixed_phaseaug.pt`  | 362 K | 100 % | 89.6 %     | 0.9847 |
| `mt_preamcnn_attn_mixed.pt`          | 428 K | 100 % | 74.6 %     | 0.9820 |
| `mt_preamcnn_cnn_mixed.pt`           | 362 K | 100 % | 72.0 %     | 0.9826 |
| `mt_preamcnn_attn_zeroshot_phaseaug.pt` | 428 K | 100 % | 61.0 %  | 0.8725 |
| `mt_preamcnn_cnn_zeroshot_phaseaug.pt`  | 362 K | 100 % | 62.9 %  | 0.8754 |
| `mt_preamcnn_attn_zeroshot.pt`       | 428 K | 100 % | 48.9 %     | 0.8851 |
| `mt_preamcnn_cnn_zeroshot.pt`        | 362 K | 100 % | 32.7 %     | 0.8397 |

Also published: ONNX exports under `onnx/` and reference benchmark JSONs under `benchmark/`.

### Inference latency (best model)

| Platform / Backend | CNN | Attn |
|---|---|---|
| x86 PyTorch-CPU | 5.01 ms | 2.43 ms |
| x86 ORT-CPU | 0.61 ms | 0.28 ms |
| Jetson Orin ORT-CPU | 0.62 ms | 1.45 ms |
| **Jetson Orin TRT-FP16** | **0.15 ms** | **0.25 ms** |

---

## Citation

```bibtex
@inproceedings{liu2026autodataphy,
  title     = {Real-Radio {OFDM} Prototype with Automated Data Collection
               and Neural Gate for Deep Learning-Aided Packet Detection},
  author    = {Liu, Kaikai},
  booktitle = {Proc. IEEE Global Communications Conference (Globecom)},
  year      = {2026}
}

@inproceedings{liu2026multitask,
  title     = {Multi-Task Preamble Intelligence: Joint Packet Detection,
               Modulation Classification, and {SNR} Estimation with
               Phase-Rotation Augmentation for {OTA} Generalisation},
  author    = {Liu, Kaikai},
  booktitle = {Proc. IEEE Global Communications Conference (Globecom)},
  year      = {2026}
}
```
