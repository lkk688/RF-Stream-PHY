# AutoDataPHY — Automated Real-Radio OFDM Data Collection with Neural Packet Detection

> **Paper:** *Real-Radio OFDM Prototype with Automated Data Collection and Neural Gate for Deep Learning-Aided Packet Detection* — submitted to **IEEE Globecom 2026**.

AutoDataPHY is the **data-centric foundation** of the RF-Stream-PHY framework. It builds a complete OFDM PHY in Python on a single Pluto SDR pair (TX on Jetson Orin, RX on x86), and uses every captured packet — both passes **and** failures — to grow a labelled dataset that later trains neural receivers (used by the MT-ATTN subproject in the same repo).

The four contributions implemented here:

1. **End-to-end OFDM PHY in Python** — STF / LTF preamble, 64-FFT OFDM with 4 pilots, QPSK / QAM16 mapping, CRC32-framed packets — running over a Pluto pair at 2.3–2.4 GHz, 3 MHz BW.
2. **Self-supervised packet labelling** — CRC pass marks the capture `_ok.npz`, CRC fail or noise-only window marks it `_bg.npz`. Every received frame produces one labelled training sample with no human annotation.
3. **Hard-negative mining** — WiFi-co-channel interference at 2.4 GHz triggers thousands of "looks like a packet but isn't" captures, harvested automatically to teach the gate to reject non-AIsensing energy.
4. **Two-stage neural gate** — a Stage-1 coarse XGBoost on STF correlation peaks shortlists ~5 % of windows; a Stage-2 1D-CNN (RawIQ Detector) runs only on shortlisted IQ to make the final keep/drop decision. End-to-end gate AUC reaches 0.998 on cable, 0.94 zero-shot OTA.

---

## File Inventory

### PHY pipeline (TX / RX)

| File | Role |
|---|---|
| `rf_stream_tx_step5phy.py` | Step-5 TX baseline (single modulation, fixed gain) |
| `rf_stream_tx_step6phy.py` | **Step-6 TX (current)** — all modulations, non-cyclic streaming, IQ-swap compensation |
| `rf_stream_rx_step5phy.py` | Step-5 RX (single-pass detection) |
| `rf_stream_rx_step6phy.py` | Step-6 RX (multi-modulation, training-data capture) |
| `rf_stream_rx_step8phy.py` | **Step-8 RX** — ZF/MMSE equalisation, gain-sweep BER capture |
| `rf_stream_rx_step9phy.py` | **Step-9 RX** — Step-8 + per-packet SNR estimation from pilots |

### Training scripts

| File | Role |
|---|---|
| `train_gate_v3.py` | Stage-1 gate (XGBoost on STF correlation features) |
| `train_rawiq_detector.py` | Stage-2 RawIQ 1D-CNN gate detector |
| `train_neural_gate.py` | Joint stage-1 + stage-2 ROC sweep |
| `train_stage2_classifier.py` | Stage-2 alternative MLP / classifier head |
| `train_llr_net.py` | LLR demapper (neural soft-info generator) |
| `eval_llr_net.py` | LLR demapper evaluation against classical soft-demap |

### Offline analysis & figures

| File | Role |
|---|---|
| `analyze_rf_stream_captures.py` | Decode rate, SNR, BER summary from a captures directory |
| `analyze_runs_offline.py` | Offline NPZ replay through different RX pipelines |
| `generate_paper_figures.py` | Builds fig1–fig9 (system + BER + ROC) |
| `generate_dl_figures.py` | Builds fig10–fig15 (LLR + dataset + neural gate) |

### Figures and paper

| Path | Role |
|---|---|
| `paper_figures/fig1_system_arch.pdf` | End-to-end system block diagram |
| `paper_figures/fig2_ber_curves.pdf` | Coded BER vs. TX gain |
| `paper_figures/fig3_gate_scores.pdf` | Stage-1 vs. Stage-2 gate score distributions |
| `paper_figures/fig4_training_curves.pdf` | Neural gate training loss / ROC |
| `paper_figures/fig5_dataset_stats.pdf` | Captures per gain / per modulation |
| `paper_figures/fig6_save_tag_analysis.pdf` | Hard-negative tag mix |
| `paper_figures/fig7_signal_quality.pdf` | SNR / EVM histograms |
| `paper_figures/fig8_pipeline_timing.pdf` | Per-stage RX timing |
| `paper_figures/fig9_roc_comparison.pdf` | Two-stage gate ROC |
| `paper_figures/fig10_llr_demapper.pdf` | LLR demapper accuracy |
| `paper_figures/fig11_hardneg_dataset.pdf` | Hard-negative dataset distribution |
| `paper_figures/fig12_mt_roc_ablation.pdf` | Multi-task gate ROC (preview of MT-ATTN) |
| `paper_figures/fig13_mt_loss_curves.pdf` | Multi-task training curves |
| `paper_figures/fig14_mt_snr_scatter.pdf` | Multi-task SNR regression scatter |
| `paper_figures/fig15_mt_summary_bar.pdf` | Multi-task summary bar chart |
| `paper/globecom2026_rf_ai_gate.tex` | Submitted paper source |
| `paper/globecom2026_rf_ai_gate.pdf` | Compiled PDF |

---

## Quick Start

```bash
# 1. Install dependencies
pip install numpy scipy pyadi-iio torch scikit-learn xgboost matplotlib

# 2. Collect a fresh run (requires a Pluto pair, cable or antenna)
#    On Jetson Orin (TX):
python3 rf_stream_tx_step6phy.py --mode qpsk --gain_db 40 --duration_s 120

#    On x86 RX:
python3 rf_stream_rx_step9phy.py --save_npz --out_dir /tmp/run1 --duration_s 120

# 3. Or download an existing run from HuggingFace
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('lkk688/ofdm-preamble-intelligence', repo_type='dataset',
                  allow_patterns='coax/ber_sweep_v3/**', local_dir='./data')
"

# 4. Train the two-stage gate
python3 train_gate_v3.py            --runs ./data/coax/ber_sweep_v3/run_*
python3 train_rawiq_detector.py     --runs ./data/coax/ber_sweep_v3/run_*

# 5. Generate paper figures from logged metrics
python3 generate_paper_figures.py
python3 generate_dl_figures.py
```

---

## Datasets Used in the Paper

The paper draws on four CoaxSweep splits collected with Step-8 / Step-9 RX, all available on HuggingFace:

| Split | Runs | NPZ | Description |
|---|---|---|---|
| `coax/ber_sweep_v2` | 6 | 1,403 | Step-8 PHY, QPSK only |
| `coax/ber_sweep_v3` | 6 | 3,282 | Step-9 PHY, QPSK + QAM16 |
| `coax/ber_sweep_v4` | 3 | 3,078 | Step-10 MMSE, pilot_weight sweep |
| `coax/ber_sweep_v6` | 4 | 842   | QAM16 mixed, hard-negative gate examples |

For hard-negative WiFi captures used to train the gate's "anti-WiFi" robustness, see the `airlink/*` splits — same HF repo — collected at 2400 MHz indoors.

See [../DATASET.md](../DATASET.md) for the full dataset card.

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
```
