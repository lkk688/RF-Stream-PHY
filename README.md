# RF-Stream-PHY

**RF-Stream-PHY** is an open research framework for building modern wireless-communication physical layers from scratch using software-defined radios and Python-based ML tools (NumPy, SciPy, PyTorch). The project is designed for researchers, students, and engineers who want to prototype, analyze, and improve communication PHY systems in a flexible software environment, while also connecting those designs to **real hardware experiments**. Instead of treating the physical layer as a fixed black box, RF-Stream-PHY treats it as a programmable, learnable, and data-driven system.

## Subprojects

The repository is organised around three research subprojects, each corresponding to a paper submitted to **IEEE Globecom 2026**:

| Folder | Paper | Focus |
|---|---|---|
| [`AutoDataPHY/`](./AutoDataPHY/) | *Real-Radio OFDM Prototype with Automated Data Collection and Neural Gate for Deep Learning-Aided Packet Detection* | End-to-end OFDM PHY in Python, self-supervised CRC-based labelling, two-stage neural packet gate |
| [`MT-ATTN/`](./MT-ATTN/) | *Multi-Task Preamble Intelligence: Joint Packet Detection, Modulation Classification, and SNR Estimation with Phase-Rotation Augmentation for OTA Generalisation* | Multi-task neural receiver (gate + mod + SNR) with temporal self-attention and phase-rotation augmentation for OTA generalisation |
| [`OTFS/`](./OTFS/) | *Hardware-Validated OTFS+LDPC on PlutoSDR: Synthetic Doppler Injection and Monostatic-Loopback Mobility Sensing* | OTFS PHY with 5G NR LDPC outer coding on commodity PlutoSDR; phase-continuous synthetic Doppler injection; **single-Pluto monostatic loopback** that eliminates LO drift (554× CFO reduction) and a hand-wave motion test that boosts OTFS decode from 0 % → 45 % on the same channel |

`AutoDataPHY` builds the PHY, collects the data, and trains the packet gate.  `MT-ATTN` consumes the same captures to train a unified multi-task model that runs in 0.25 ms on Jetson Orin TRT.  `OTFS` extends the framework to a delay-Doppler waveform and adds infrastructure for controlled-Doppler mobility validation (synthetic injection + monostatic loopback).

## Open-Source Dataset and Models

The full dataset of **17,469 raw IQ captures** (31 GB) and **8 pretrained model checkpoints** (with ONNX exports) collected across both papers are published on the HuggingFace Hub:

| Resource | URL | Size |
|---|---|---|
| **OFDM dataset** | [`lkk688/ofdm-preamble-intelligence`](https://huggingface.co/datasets/lkk688/ofdm-preamble-intelligence) | 31 GB (17,469 NPZ) |
| **OFDM pretrained models** | [`lkk688/mt-preamcnn`](https://huggingface.co/lkk688/mt-preamcnn) | 20 MB (8 .pt + 2 ONNX) |
| **OTFS dataset** | [`lkk688/rf-stream-otfs`](https://huggingface.co/datasets/lkk688/rf-stream-otfs) | 3.0 GB (4,228 NPZ across 7 splits: closed-grid, OTA Doppler, monostatic) |

The dataset spans two RF domains:

- **CoaxSweep** (cable testbed, 4 splits, 8,605 NPZ, ~15 GB) — controlled SNR / gain sweep used to develop the PHY and train the baseline gate.
- **AirLink** (over-the-air at 2400 MHz, 4 splits, 8,864 NPZ, ~16 GB) — live OTA captures in a 2.4 GHz WiFi-congested environment, used to study and solve the **modulation-collapse domain-shift problem** addressed in the MT-ATTN paper.

See [`DATASET.md`](./DATASET.md) for the full dataset card (file format, labelling convention, sub-split breakdown, and code snippets for loading).

A single utility for uploading new captures back to the HuggingFace Hub is in [`upload_to_hf.py`](./upload_to_hf.py).

## Headline Result (MT-ATTN)

The best model — **MT-PreamCNN-Attn** trained with mixed CoaxSweep + AirLink data and phase-rotation augmentation — reaches:

- **100 %** CoaxSweep modulation accuracy
- **92.3 %** AirLink modulation accuracy (vs. 48.9 % zero-shot baseline)
- **0.82 dB** CoaxSweep SNR RMSE, **1.37 dB** AirLink SNR RMSE
- **0.985** AirLink gate AUC
- **0.25 ms** per-packet latency on Jetson Orin (TensorRT FP16)

at only **428 K parameters**.

## Vision

RF-Stream-PHY aims to make PHY-layer communication research more open, reproducible, and extensible by combining:

- **Classical communication system design** — synchronization, channel estimation, equalization, modulation, coding, packet detection
- **Modern scientific computing in Python** — NumPy, SciPy, matplotlib, Jupyter for fast iteration and visualization
- **Deep learning and differentiable modeling** — PyTorch for neural receivers, learned demappers, signal detectors, hardware-adaptive models
- **Hardware-in-the-loop experimentation** — SDR platforms to bridge simulation and real radio deployment

## Design Philosophy

This project is built around four core ideas:

### 1. PHY should be programmable
Modern communication research benefits from software-first design. Building PHY blocks in Python and PyTorch lets us iterate faster, inspect intermediate signals, and connect classical algorithms with machine learning.

### 2. Real hardware matters
Simulation is valuable, but real-world RF behavior often differs from idealized channel models. RF-Stream-PHY is designed for hardware-in-the-loop workflows so that models and algorithms can be evaluated under realistic conditions.

### 3. Data collection should be part of the PHY workflow
A modern PHY system should not only transmit and receive packets, but also generate useful research datasets. Logging, labelling, metadata collection, and hard-negative mining are treated as first-class capabilities (see AutoDataPHY).

### 4. Learning should augment, not replace, the PHY
The project supports both classical DSP pipelines and neural components. This allows direct comparison, hybrid designs, and gradual integration of learned modules where they add practical value (see MT-ATTN).

## Intended Use Cases

RF-Stream-PHY is suitable for:

- Communication systems research
- SDR-based prototyping
- Deep learning for wireless PHY
- Dataset generation for RF ML
- Classroom and lab instruction
- Rapid experimentation with new receiver ideas
- Reproducible hardware-in-the-loop evaluation

## Hardware Used

- **TX:** NVIDIA Jetson Orin + ADALM-PlutoSDR (USB) — runs `rf_stream_tx_step6phy.py`
- **RX:** x86 Ubuntu host + ADALM-PlutoSDR (USB) — runs `rf_stream_rx_step{8,9,10}phy.py`
- **Carrier:** 2400 MHz, 3 MHz sample rate, 64-FFT OFDM with CP=16
- **Inference deployment:** NVIDIA Jetson Orin (CUDA 12.6, TensorRT 10.3)

## Quick Start

```bash
git clone https://github.com/<your-org>/RF-Stream-PHY.git
cd RF-Stream-PHY

# Install minimal Python dependencies
pip install numpy scipy torch scikit-learn matplotlib onnxruntime huggingface_hub

# Subproject 1: build the PHY and collect data
cd AutoDataPHY && cat README.md

# Subproject 2: train the multi-task neural receiver
cd ../MT-ATTN && cat README.md

# Or download the published dataset (~31 GB) for offline experiments
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('lkk688/ofdm-preamble-intelligence', repo_type='dataset',
                   local_dir='./data')
"
```

For a guided walkthrough — data loading, training, evaluation, ONNX/TensorRT deployment, and full paper reproduction — see [`MT-ATTN/TUTORIAL.md`](./MT-ATTN/TUTORIAL.md).

## Status

RF-Stream-PHY is an active research-oriented project under development. The repository will continue to expand with:

- Reusable PHY modules
- Experiment scripts
- Dataset tools
- Training pipelines
- Hardware integration examples
- Documentation for reproducible research workflows

## Contributing

Contributions, discussions, and collaboration are welcome. This project is intended to grow into an open platform for communication PHY research, real-radio experimentation, and learning-based wireless system design.

## Citation

If you use this project in research, please cite both papers:

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

@inproceedings{liu2026otfs,
  title     = {Hardware-Validated {OTFS}+{LDPC} on {PlutoSDR}:
               Synthetic Doppler Injection and Monostatic-Loopback
               Mobility Sensing},
  author    = {Liu, Kaikai},
  booktitle = {Proc. IEEE Global Communications Conference (Globecom)},
  year      = {2026}
}
```

## License

Code: MIT License (see [`LICENSE`](./LICENSE))
Dataset: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
