# Uploading the OTFS dataset to HuggingFace

This document captures the plan for releasing the OTFS experiment data on the HuggingFace Hub at `lkk688/rf-stream-otfs`.  The data referenced in [`paper/globecom2026_otfs_doppler.pdf`](paper/globecom2026_otfs_doppler.pdf) totals ~1.7 GB (raw IQ captures + tx-truth + summaries) and is too large to commit to the git repo.

## What is in the dataset

Five top-level splits, one per experiment in the paper:

| Split | Description | Approx size | # NPZ |
|---|---|---:|---:|
| `closed_grid_0db_otfs/` | OTA OTFS+LDPC at fc=2.45 GHz, tx_gain=0 dB, 500 caps | ~80 MB | 500 |
| `closed_grid_0db_ofdm/` | OTA OFDM+LDPC at fc=2.45 GHz, tx_gain=0 dB, 489 caps | ~80 MB | 489 |
| `closed_grid_n5db/`     | OTA at tx_gain=−5 dB (both waveforms, all rerun-after-Pluto-reset captures) | ~150 MB | ~900 |
| `ota_doppler/`          | OTA Doppler injection at fc=2.42 GHz, fd ∈ {0, 1000, 3000} Hz, both waveforms | ~600 MB | ~3000 |
| `monostatic/`           | Single-Pluto monostatic loopback (static + hand-wave motion), 300 caps each | ~80 MB | 600 |

Each split contains:
- `cap_*.npz` — one per capture window (rxw + meta_json).  Step10 OFDM splits also include `Yeq_data`.
- `rx_summary.json` — per-run summary (n_captures, n_decoded_ok, decode_rate, run_id, config).
- `tx_truth.jsonl` — per-packet ground-truth bits used for BER (where applicable).
- `README.md` — per-split notes describing channel state, antenna geometry, gains.

## Dataset card (front-matter)

```yaml
---
license: cc-by-4.0
task_categories:
  - audio-classification    # closest HF tag for "signal classification"
language: [en]
size_categories: [1G<n<10G]
tags:
  - rf
  - sdr
  - otfs
  - ofdm
  - ldpc
  - plutosdr
  - doppler
  - mobility
  - isac
configs:
  - config_name: closed_grid_0db_otfs
  - config_name: closed_grid_0db_ofdm
  - config_name: closed_grid_n5db
  - config_name: ota_doppler
  - config_name: monostatic
---
```

## Upload procedure

The repo-level [`upload_to_hf.py`](../upload_to_hf.py) script was proven for the prior 31 GB `ofdm-preamble-intelligence` upload.  For OTFS we will extend it with an `--what otfs` option that:

1. Walks the 5 split directories under a local staging area (e.g. `~/Developer/RF-Stream-PHY/OTFS/data_staging/`).
2. Commits one HF revision per split (not per-NPZ) to avoid git timeouts on Hub-side.
3. Generates a `DATASET.md` from a Python f-string template embedding the dataset card YAML above and a per-split table.

### Staging the data

The raw captures currently live in `/tmp/` on the x86 host:

```bash
mkdir -p ~/Developer/RF-Stream-PHY/OTFS/data_staging/{closed_grid_0db_otfs,closed_grid_0db_ofdm,closed_grid_n5db,ota_doppler,monostatic}

# Closed-grid 0 dB OTFS
cp -r /tmp/otfs_ldpc_now/run_otfs_rx_20260531_152717/* \
      ~/Developer/RF-Stream-PHY/OTFS/data_staging/closed_grid_0db_otfs/

# Closed-grid 0 dB OFDM (via step10 NPZs)
cp -r /tmp/.../run_20260531_152144/* \
      ~/Developer/RF-Stream-PHY/OTFS/data_staging/closed_grid_0db_ofdm/

# tx_gain=-5 dB, including v2 reruns
cp -r /tmp/{ofdm_ldpc_n5,ofdm_ldpc_n5_v2,otfs_ldpc_n5,otfs_ldpc_n5_v2}/* \
      ~/Developer/RF-Stream-PHY/OTFS/data_staging/closed_grid_n5db/

# OTA Doppler sweep (fc=2.42 GHz)
cp -r /tmp/ota_doppler/* \
      ~/Developer/RF-Stream-PHY/OTFS/data_staging/ota_doppler/

# Monostatic + hand-wave
cp -r /tmp/mono/* \
      ~/Developer/RF-Stream-PHY/OTFS/data_staging/monostatic/
```

(Some of these `/tmp/` directories may have been cleaned by the time you read this — they were intermediate locations during the experiment.  The decode summaries in `OTFS/results/` are persistent.)

### Dry-run

```bash
python3 ../upload_to_hf.py --what otfs --dry_run
# Verifies file count and structure, no auth needed.
```

### Real upload

```bash
export HF_TOKEN=hf_xxx_your_write_token
python3 ../upload_to_hf.py --what otfs
# Pushes per-split commits to lkk688/rf-stream-otfs.
```

## When the upload completes

1. Update [`README.md`](README.md) to remove the *(forthcoming)* marker on the HF link.
2. Update the root [`../README.md`](../README.md) similarly.
3. Update the paper's bibliography entry `\cite{hf_dataset}` URL and re-compile if needed.
4. Tag a release in the repo (`v0.1-otfs`) noting the dataset is now public.

## License and citation

- Dataset: CC-BY-4.0 (same as the prior OFDM dataset).
- Code:    MIT (same as the rest of the repo).
- If anyone uses the dataset, please cite the OTFS paper (`liu2026otfs` BibTeX entry in [`README.md`](README.md)).
