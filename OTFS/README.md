# OTFS — Hardware-Validated OTFS+LDPC on PlutoSDR

> **Paper:** *Open Python/PyTorch OTFS-LDPC on PlutoSDR: Hardware-Validated Doppler Injection and Monostatic ISAC Loopback* — K. Liu, **IEEE MILCOM 2026**. LaTeX source lives in our internal development repo. New to OTFS? See the [tutorial in the top-level README](../README.md#tutorial-otfs-from-scratch--communication-and-radar). This subproject contains the full code, results, and the open dataset that the paper builds on.

This subproject implements an end-to-end OTFS (Orthogonal Time Frequency Space) physical layer with 5G NR LDPC outer coding on commodity PlutoSDR, plus the tooling to compare it against an OFDM baseline under controlled Doppler — including a **single-Pluto monostatic loopback** that eliminates LO drift for fine mobility studies.

## Headline results

| Experiment | Setup | OTFS+LDPC | OFDM+LDPC |
|---|---|---:|---:|
| OTA closed-grid (Sec. 5.2 of paper) | $f_c$=2.45 GHz, $g_\mathrm{TX}$=0 dB, 500 caps | **35.8 %** | 32.9 % |
| Software-loopback Doppler sweep, $f_d$=300 Hz (Sec. 5.1) | SNR=20 dB | **100 %** | 3.3 % |
| Software-loopback Doppler sweep, $f_d$=1000 Hz (Sec. 5.1) | SNR=20 dB | **100 %** | 0 % |
| Monostatic loopback CFO baseline (Sec. 5.4) | shared LO | **−8 Hz mean** | — |
| Monostatic + hand-wave motion (Sec. 5.4) | same channel as static | **45.3 %** | — (OTFS-only) |

The full story (closed 2×2 grid, OTA Doppler sweep, software Doppler sweep at SNR ∈ {10, 15, 20} dB, mechanical sanity check, and the monostatic finding) is in [`reports/`](reports/).

## Code layout

### Core PHY package
| File | Purpose |
|---|---|
| [`otfs/otfs_core.py`](otfs/otfs_core.py) | DD-domain primitives: OTFSConfig dataclass, dd_cell_mask, isfft/sfft, build_dd_grid (embedded pilot), estimate_dd_channel_taps, dd_lmmse_detect, find_pilot_offset, constellation tables |
| [`otfs/otfs_frame.py`](otfs/otfs_frame.py) | Frame builder: bytes_to_otfs_samples with optional LDPC encoder; per-block CP, preamble, gap structure |
| [`otfs/otfs_sync.py`](otfs/otfs_sync.py) | Hybrid Schmidl–Cox + cross-correlation + DD-pilot fine-timing sync |
| [`otfs/otfs_fec.py`](otfs/otfs_fec.py) | 5G NR LDPC (rate-1/2, BG2, Z=32) encoder/decoder wrapper around `sdr_ldpc`; QPSK soft demap |

### TX/RX scripts (PlutoSDR-driven)
| File | Purpose |
|---|---|
| [`rf_stream_tx_otfs.py`](rf_stream_tx_otfs.py) | OTFS TX. `--fec ldpc`, `--build_only` to write IQ file, `--cyclic` for cyclic playback |
| [`rf_stream_rx_otfs.py`](rf_stream_rx_otfs.py) | OTFS RX. Supports monostatic loopback via `--tx_iq_file --tx_gain` (same Pluto opens both TX cyclic and RX captures, shared LO) |
| [`rf_stream_tx_playback.py`](rf_stream_tx_playback.py) | Generic raw-IQ playback (Jetson, no PyTorch). Includes 3× deterministic gain rewrite to work around Pluto's gain-apply bug after rapid restarts |
| [`rf_stream_tx_step6phy.py`](rf_stream_tx_step6phy.py) | OFDM TX (for the OTFS-vs-OFDM comparison) |
| [`rf_stream_rx_step10phy.py`](rf_stream_rx_step10phy.py) | OFDM RX (saves `Yeq_data` NPZs for post-decode) |

### Channel injection and offline analysis
| File | Purpose |
|---|---|
| [`inject_doppler_to_iq.py`](inject_doppler_to_iq.py) | Apply a 2-tap doubly-selective channel `y[n]=x[n]+α·x[n−τ]·exp(j2π fd n/fs+φ)` to any pre-built IQ file. Phase-continuous quantisation for cyclic-buffer playback |
| [`doppler_sweep_loopback.py`](doppler_sweep_loopback.py) | Software loopback driver: OTFS+OFDM through the same injected channel, sweep `fd` at fixed SNR. Uses the same TX/RX primitives the live scripts use |
| [`ofdm_ldpc_postdecode.py`](ofdm_ldpc_postdecode.py) | Reads step10 RX NPZs, computes QPSK soft LLRs, runs LDPC decode, reports raw and post-LDPC packet rates |
| [`test_otfs_loopback.py`](test_otfs_loopback.py) | Software loopback BER/pkt sweep across SNR for three channel scenarios (AWGN, static MP, doubly-selective). Regression test for the chain |

### Figures script
- [`scripts/generate_paper_figures.py`](scripts/generate_paper_figures.py) — regenerates all paper figures from `results/*.json` into `OTFS/figures/`.  Useful to inspect or modify the plots without the paper-side LaTeX source.

### Results and reports
- [`results/`](results/) — machine-readable summaries (JSON) used to generate paper figures: `doppler_sweep_snr{10,15,20}.json`, `monostatic_summary.json`, `otfs_loopback_results.json`
- [`reports/`](reports/) — markdown research reports written during the development:
  - [`OTFS_VS_OFDM_LDPC_CLOSED_GRID.md`](reports/OTFS_VS_OFDM_LDPC_CLOSED_GRID.md) — closed 2×2 grid OTA comparison
  - [`OTFS_VS_OFDM_LDPC_FAIR.md`](reports/OTFS_VS_OFDM_LDPC_FAIR.md) — earlier OTFS+LDPC integration milestone
  - [`OTFS_VS_OFDM_DOPPLER_LOOPBACK.md`](reports/OTFS_VS_OFDM_DOPPLER_LOOPBACK.md) — software-loopback Doppler sweep + OTA validation
  - [`MECHANICAL_DOPPLER_SANITY.md`](reports/MECHANICAL_DOPPLER_SANITY.md) — two-Pluto hand-wave sanity check (motion vs synthetic comparison)
  - [`MONOSTATIC_DOPPLER.md`](reports/MONOSTATIC_DOPPLER.md) — single-Pluto monostatic loopback: 554× CFO reduction + motion-boosts-OTFS finding

## Reproducing the headline experiments

### Software-loopback Doppler sweep (no Pluto required)

```bash
for s in 10 15 20; do
  python3 doppler_sweep_loopback.py --snr_db $s --n_trials 30 \
      --doppler_list 0,100,300,500,1000,2000,3000,5000 \
      --out_json results/doppler_sweep_snr${s}.json
done
python3 scripts/generate_paper_figures.py
```

### Monostatic OTFS loopback (single Pluto, antennas at the cable end)

Physical setup: one PlutoSDR, TX antenna on board, RX antenna at the end of a ~50 cm SMA cable, antennas at ~90° cross-polarised orientation.

```bash
# Pre-build the OTFS+LDPC IQ buffer
python3 rf_stream_tx_otfs.py --build_only /tmp/otfs_ldpc_seed42.iq \
    --fec ldpc --ldpc_k 256 --ldpc_n 512 --num_chunks 64 --chunk_bytes 64 \
    --ref_seed 42 --ref_len 4096 --modulation qpsk \
    --otfs_M 64 --otfs_N 8 --otfs_l_max 6 --otfs_k_max 1

# Static baseline (RX held still)
python3 rf_stream_rx_otfs.py --mode live --uri ip:192.168.2.2 \
    --fc 2.42e9 --fs 3e6 --rx_gain 60 \
    --fixed_len 65536 --num_captures 300 \
    --modulation qpsk --otfs_M 64 --otfs_N 8 --otfs_l_max 6 --otfs_k_max 1 \
    --otfs_num_blocks 2 --payload_bytes 64 \
    --ref_seed 42 --ref_len 4096 \
    --tx_truth /tmp/otfs_ldpc_seed42.iq.tx_truth.jsonl \
    --fec ldpc --ldpc_k 256 --ldpc_n 512 \
    --tx_iq_file /tmp/otfs_ldpc_seed42.iq --tx_gain 0 \
    --out_root runs/mono_static

# Then same with --out_root runs/mono_motion, waving the RX antenna
# ~30 cm at ~1 swing/sec for the ~7 s capture phase.
```

### Two-station OTA Doppler injection

```bash
# Generate Doppler-injected IQ on x86 (PyTorch host), scp to Jetson TX
python3 inject_doppler_to_iq.py --in_iq /tmp/otfs_ldpc_seed42.iq \
    --out_iq /tmp/otfs_ldpc_seed42_fd1000.iq \
    --fs 3e6 --doppler_hz 1000 --tap_delay 3 --tap_db -3
scp /tmp/otfs_ldpc_seed42_fd1000.iq jetson:/tmp/
# Then play on Jetson with rf_stream_tx_playback.py --iq_file /tmp/otfs_ldpc_seed42_fd1000.iq --cyclic
# and capture on x86 with rf_stream_rx_otfs.py as usual.
```

## Hardware

Validated on:
- 2 × Analog Devices PlutoSDR (AD9363, Ethernet bridge), `tx_gain` 0 dB, `rx_gain` 60 dB manual, antennas at ~30–50 cm separation OTA, or ~30 cm cable extension monostatic.
- TX host: Jetson Orin Nano (8 GB), Ubuntu 22.04, no PyTorch — uses pre-built IQ playback.
- RX host: x86 Ubuntu 22.04 with PyTorch 2.x, libiio 0.25, pyadi-iio 0.0.18.

Carrier 2.4 GHz ISM band (works at 2.42 / 2.45 / 2.47 GHz depending on interference state; some sessions one of these is unusable due to WiFi). Sample rate 3 MHz.

## Open dataset (HuggingFace)

Raw IQ captures, tx-truth JSONLs, and per-run summary JSONs are deposited at:

| Resource | URL | Size |
|---|---|---:|
| **Dataset** | [`lkk688/rf-stream-otfs`](https://huggingface.co/datasets/lkk688/rf-stream-otfs) | 3.0 GB, 4,228 NPZ |

7 splits matching the paper's experiments:

| Split | Paper sec. | # files | Size | Key result |
|---|---|---:|---:|---|
| `closed_grid_0db_otfs/`     | §V-B | 503  |  83 MB | 179/500 = **35.8 %** |
| `closed_grid_0db_ofdm/`     | §V-B | 493  | 1.1 GB | 161/489 = **32.9 %** post-LDPC |
| `closed_grid_n5db_otfs_v2/` | §V-B | 503  |  83 MB | 0/500 (channel past LDPC cliff) |
| `closed_grid_n5db_ofdm_v2/` | §V-B | 206  | 446 MB | 0/92, raw BER 0.47 |
| `ota_doppler/` (6 sub-runs) | §V-C | 1903 | 1.1 GB | OTFS 14.2 / 3.8 / 0 % at fd = 0 / 1k / 3k Hz |
| `monostatic/` (static + motion) | §V-D | 606 |  96 MB | static 0 %, motion **45.3 %** |
| `tx_iq/` | (TX waveforms) | 14 |  41 MB | raw complex64 IQ for replay |

To download:

```python
from huggingface_hub import snapshot_download
# Whole dataset (~3 GB):
snapshot_download("lkk688/rf-stream-otfs", repo_type="dataset", local_dir="./otfs_data")
# Or just one split:
snapshot_download("lkk688/rf-stream-otfs", repo_type="dataset", local_dir="./otfs_data",
                  allow_patterns="monostatic/**")
```

Upload script (chunked + retry-on-failure, per-split commits): [`scripts/upload_otfs_to_hf.py`](scripts/upload_otfs_to_hf.py). Staging script (rebuilds `data_staging/` from raw run dirs): [`scripts/stage_data_for_hf.py`](scripts/stage_data_for_hf.py). See [`HF_UPLOAD.md`](HF_UPLOAD.md) for end-to-end procedure.

## License

MIT for code (see [`../LICENSE`](../LICENSE)); CC-BY-4.0 for datasets uploaded to HuggingFace.

## Citation

If you use this code or data, please cite:

```bibtex
@inproceedings{liu2026otfs,
  title     = {Open {Python/PyTorch} {OTFS-LDPC} on {PlutoSDR}: Hardware-Validated
               {Doppler} Injection and Monostatic {ISAC} Loopback},
  author    = {Liu, Kaikai},
  booktitle = {Proc. IEEE Military Communications Conference (MILCOM)},
  year      = {2026}
}
```
