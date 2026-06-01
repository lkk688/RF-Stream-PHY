#!/usr/bin/env python3
"""
upload_to_hf.py — Upload OFDM Preamble Intelligence dataset + models to HuggingFace.

Usage:
  # Login first (one-time):
  python3 -c "from huggingface_hub import login; login()"

  # Upload dataset only:
  python3 upload_to_hf.py --what dataset

  # Upload model checkpoints only:
  python3 upload_to_hf.py --what models

  # Upload everything:
  python3 upload_to_hf.py --what all

  # Dry run (print what would be uploaded):
  python3 upload_to_hf.py --what all --dry_run

Repos created:
  Dataset : hf.co/datasets/{HF_USER}/ofdm-preamble-intelligence
  Models  : hf.co/{HF_USER}/mt-preamcnn

The script uses folder_upload for large directories (chunked + resumable).
Progress is logged to upload_log.txt.
"""

import argparse, glob, json, logging, os, sys, time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("upload_log.txt")],
)
log = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
HF_USER       = "lkk688"
DATASET_REPO  = f"{HF_USER}/ofdm-preamble-intelligence"
MODEL_REPO    = f"{HF_USER}/mt-preamcnn"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # rf_stream/

# Dataset source directories (local path → HF repo path)
COAX_DIRS = [
    # ("ber_sweep_v2", "coax/ber_sweep_v2"),  # DONE 2026-05-03 23:27 — 1403 files confirmed
    ("ber_sweep_v3", "coax/ber_sweep_v3"),
    ("ber_sweep_v4", "coax/ber_sweep_v4"),
    ("ber_sweep_v6", "coax/ber_sweep_v6"),
]

AIRLINK_DIRS = [
    ("airlink/airlink_qpsk",   "airlink/airlink_qpsk"),
    ("airlink/airlink2_qpsk",  "airlink/airlink2_qpsk"),
    ("airlink/airlink_qam16",  "airlink/airlink_qam16"),
    ("airlink/airlink3_qpsk",  "airlink/airlink3_qpsk"),
]

# Model files to upload
MODEL_FILES = [
    "multitask_model_v2/mt_preamcnn_cnn_mixed_phaseaug.pt",
    "multitask_model_v2/mt_preamcnn_attn_mixed_phaseaug.pt",
    "multitask_model_v2/mt_preamcnn_cnn_mixed.pt",
    "multitask_model_v2/mt_preamcnn_attn_mixed.pt",
    "multitask_model_v2/mt_preamcnn_cnn_zeroshot_phaseaug.pt",
    "multitask_model_v2/mt_preamcnn_attn_zeroshot_phaseaug.pt",
    "multitask_model_v2/mt_preamcnn_cnn_zeroshot.pt",
    "multitask_model_v2/mt_preamcnn_attn_zeroshot.pt",
]

METRICS_FILES = glob.glob(os.path.join(BASE_DIR, "multitask_model_v2", "metrics_*.json"))
HISTORY_FILES = glob.glob(os.path.join(BASE_DIR, "multitask_model_v2", "history_*.json"))

ONNX_FILES = [
    "onnx/mt_preamcnn_cnn.onnx",
    "onnx/mt_preamcnn_attn.onnx",
]

BENCHMARK_FILES = [
    "benchmark_results_x86.json",
    "benchmark_results_jetson.json",
]


# ── HF helpers ─────────────────────────────────────────────────────────────────

def ensure_repo(api, repo_id, repo_type, private=False, dry_run=False):
    if dry_run:
        log.info(f"[dry_run] would ensure repo: {repo_id} (type={repo_type})")
        return
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        log.info(f"Repo exists: {repo_id}")
    except Exception:
        log.info(f"Creating repo: {repo_id}")
        api.create_repo(repo_id=repo_id, repo_type=repo_type,
                        private=private, exist_ok=True)


def _already_uploaded(api, repo_id, repo_type, repo_path, min_files=1):
    """Return True if repo_path already has at least min_files files on HF."""
    try:
        files = list(api.list_repo_tree(repo_id, repo_type=repo_type,
                                        path_in_repo=repo_path, recursive=False))
        return len(files) >= min_files
    except Exception:
        return False


def upload_folder(api, local_dir, repo_id, repo_path, repo_type, dry_run=False):
    """Upload a directory as one commit per run sub-directory.

    Large directories (thousands of files) cause git commits to time out when
    uploaded as a single commit.  Splitting by run_* sub-directory keeps each
    commit to ~500 files and ~1 GB — reliably below the timeout threshold.
    If no run_* sub-directories exist, upload the whole directory at once.
    """
    if not os.path.isdir(local_dir):
        log.warning(f"  Skipping missing dir: {local_dir}")
        return

    # Find run sub-directories; fall back to flat upload if none
    run_dirs = sorted(d for d in glob.glob(os.path.join(local_dir, "run_*"))
                      if os.path.isdir(d))

    if not run_dirs:
        # Flat directory (e.g. ber_sweep_v5 which has no NPZ): single upload
        n = sum(1 for f in glob.glob(os.path.join(local_dir, "**"), recursive=True)
                if os.path.isfile(f))
        log.info(f"  Uploading folder: {local_dir} → {repo_id}/{repo_path}  ({n} files)")
        if not dry_run:
            t0 = time.time()
            api.upload_folder(folder_path=local_dir, repo_id=repo_id,
                              path_in_repo=repo_path, repo_type=repo_type,
                              commit_message=f"Upload {repo_path}")
            log.info(f"  Done in {(time.time()-t0)/60:.1f} min: {repo_path}")
        return

    # Upload one run at a time — smaller commits, resumable
    total_size = sum(os.path.getsize(f)
                     for f in glob.glob(os.path.join(local_dir, "**"), recursive=True)
                     if os.path.isfile(f))
    log.info(f"  Uploading folder: {local_dir} → {repo_id}/{repo_path}  "
             f"({len(run_dirs)} runs, {total_size/1e9:.1f} GB)")

    for run_dir in run_dirs:
        run_name   = os.path.basename(run_dir)
        run_repo   = f"{repo_path}/{run_name}"
        n_files    = sum(1 for f in glob.glob(os.path.join(run_dir, "*"))
                         if os.path.isfile(f))
        run_size   = sum(os.path.getsize(f)
                         for f in glob.glob(os.path.join(run_dir, "**"), recursive=True)
                         if os.path.isfile(f))

        # Skip if already committed
        if not dry_run and _already_uploaded(api, repo_id, repo_type, run_repo, min_files=1):
            log.info(f"    SKIP (already on HF): {run_repo}  ({n_files} files)")
            continue

        log.info(f"    run: {run_name}  ({n_files} files, {run_size/1e6:.0f} MB)")
        if dry_run:
            continue

        t0 = time.time()
        api.upload_folder(
            folder_path=run_dir,
            repo_id=repo_id,
            path_in_repo=run_repo,
            repo_type=repo_type,
            commit_message=f"Upload {run_repo}",
        )
        log.info(f"    Done in {(time.time()-t0)/60:.1f} min: {run_repo}")

    log.info(f"  Done: {repo_path}")


def upload_file(api, local_path, repo_id, repo_path, repo_type, dry_run=False):
    if not os.path.isfile(local_path):
        log.warning(f"  Skipping missing file: {local_path}")
        return
    size_mb = os.path.getsize(local_path) / 1e6
    log.info(f"  Uploading file: {local_path} → {repo_id}/{repo_path}  ({size_mb:.1f} MB)")
    if dry_run:
        return
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=repo_path,
        repo_id=repo_id,
        repo_type=repo_type,
    )


# ── Dataset card ───────────────────────────────────────────────────────────────

def push_dataset_card(api, dry_run=False):
    readme_path = os.path.join(BASE_DIR, "airlink", "README.md")
    if not os.path.isfile(readme_path):
        log.warning("Dataset README.md not found, skipping card update")
        return
    log.info("Uploading dataset card (README.md)")
    if not dry_run:
        upload_file(api, readme_path, DATASET_REPO, "README.md", "dataset", dry_run=False)


# ── Model card ────────────────────────────────────────────────────────────────

MODEL_CARD = """\
---
license: mit
language:
- en
tags:
- ofdm
- sdr
- wireless-communications
- phy-layer
- modulation-classification
- snr-estimation
- packet-detection
- pytorch
- onnx
- tensorrt
- globecom2026
---

# MT-PreamCNN: Multi-Task Preamble Intelligence

**Paper:** "Multi-Task Preamble Intelligence: Joint Packet Detection, Modulation
Classification, and SNR Estimation with Phase-Rotation Augmentation for OTA
Generalisation" — *IEEE Globecom 2026 (submitted)*

**Dataset:** [lkk688/ofdm-preamble-intelligence](https://huggingface.co/datasets/lkk688/ofdm-preamble-intelligence)
**Code:** [github.com/lkk688/AIsensing](https://github.com/lkk688/AIsensing)

---

## Models

| Checkpoint | Arch | Training | CoaxSweep Mod | AirLink Mod | AirLink Gate AUC |
|---|---|---|---|---|---|
| `mt_preamcnn_attn_mixed_phaseaug.pt` | Attn | mixed+aug | 100% | **92.3%** | 0.9850 |
| `mt_preamcnn_cnn_mixed_phaseaug.pt`  | CNN  | mixed+aug | 100% | 89.6%  | 0.9847 |
| `mt_preamcnn_attn_mixed.pt`          | Attn | mixed     | 100% | 74.6%  | 0.9820 |
| `mt_preamcnn_cnn_mixed.pt`           | CNN  | mixed     | 100% | 72.0%  | 0.9826 |
| `mt_preamcnn_attn_zeroshot_phaseaug.pt` | Attn | zeroshot+aug | 100% | 61.0% | 0.8725 |
| `mt_preamcnn_cnn_zeroshot_phaseaug.pt`  | CNN  | zeroshot+aug | 100% | 62.9% | 0.8754 |
| `mt_preamcnn_attn_zeroshot.pt`       | Attn | zeroshot  | 100% | 48.9%  | 0.8851 |
| `mt_preamcnn_cnn_zeroshot.pt`        | CNN  | zeroshot  | 100% | 32.7%  | 0.8397 |

ONNX exports (opset 17, dynamic batch) are in `onnx/`.

## Inference Latency

| Platform / Backend | CNN | Attn |
|---|---|---|
| x86 PyTorch-CPU | 5.01 ms | 2.43 ms |
| x86 ORT-CPU (ONNX Runtime) | 0.61 ms | 0.28 ms |
| Jetson Orin ORT-CPU | 0.62 ms | 1.45 ms |
| **Jetson Orin TRT-FP16** | **0.15 ms** | **0.25 ms** |

## Usage

```python
import torch, numpy as np
from huggingface_hub import hf_hub_download

# Download best checkpoint
path = hf_hub_download("lkk688/mt-preamcnn",
                        "mt_preamcnn_attn_mixed_phaseaug.pt")

# Rebuild model (copy class definition from train_multitask_v2.py)
ckpt  = torch.load(path, map_location="cpu")
# model = MultiTaskPreamCNNAttn(embed_dim=256, tasks=("gate","mod","snr"))
# model.load_state_dict(ckpt["state_dict"])
# model.eval()

# Input: 1600-float32 vector (800 complex IQ samples, interleaved real/imag)
# x = torch.from_numpy(preamble_float32).unsqueeze(0)   # (1, 1600)
# with torch.no_grad():
#     out = model(x)   # {"gate": ..., "mod": ..., "snr": ...}
```

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
"""


def push_model_card(api, dry_run=False):
    import tempfile
    log.info("Uploading model card (README.md)")
    if dry_run:
        return
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        f.write(MODEL_CARD)
        tmp = f.name
    try:
        api.upload_file(
            path_or_fileobj=tmp,
            path_in_repo="README.md",
            repo_id=MODEL_REPO,
            repo_type="model",
        )
    finally:
        os.unlink(tmp)


# ── Main ───────────────────────────────────────────────────────────────────────

def upload_dataset(api, dry_run=False):
    log.info("=== Uploading Dataset ===")
    ensure_repo(api, DATASET_REPO, "dataset", private=False, dry_run=dry_run)
    push_dataset_card(api, dry_run=dry_run)

    log.info("--- CoaxSweep sub-splits ---")
    for local_name, repo_path in COAX_DIRS:
        local_dir = os.path.join(BASE_DIR, local_name)
        upload_folder(api, local_dir, DATASET_REPO, repo_path, "dataset", dry_run=dry_run)

    log.info("--- AirLink sub-splits ---")
    for local_name, repo_path in AIRLINK_DIRS:
        local_dir = os.path.join(BASE_DIR, local_name)
        upload_folder(api, local_dir, DATASET_REPO, repo_path, "dataset", dry_run=dry_run)


def upload_models(api, dry_run=False):
    log.info("=== Uploading Models ===")
    ensure_repo(api, MODEL_REPO, "model", private=False, dry_run=dry_run)
    push_model_card(api, dry_run=dry_run)

    log.info("--- Checkpoints ---")
    for rel in MODEL_FILES:
        local = os.path.join(BASE_DIR, rel)
        upload_file(api, local, MODEL_REPO, os.path.basename(rel), "model", dry_run=dry_run)

    log.info("--- Metrics & training history ---")
    for f in sorted(METRICS_FILES + HISTORY_FILES):
        upload_file(api, f, MODEL_REPO, f"training_logs/{os.path.basename(f)}",
                    "model", dry_run=dry_run)

    log.info("--- ONNX exports ---")
    for rel in ONNX_FILES:
        local = os.path.join(BASE_DIR, rel)
        upload_file(api, local, MODEL_REPO, f"onnx/{os.path.basename(rel)}",
                    "model", dry_run=dry_run)

    log.info("--- Benchmark results ---")
    for rel in BENCHMARK_FILES:
        local = os.path.join(BASE_DIR, rel)
        upload_file(api, local, MODEL_REPO, f"benchmark/{os.path.basename(rel)}",
                    "model", dry_run=dry_run)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", choices=["dataset", "models", "all"], default="all")
    ap.add_argument("--dry_run", action="store_true",
                    help="Print plan without uploading anything")
    ap.add_argument("--token", default=None,
                    help="HuggingFace API token (or set HF_TOKEN env var)")
    args = ap.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")

    from huggingface_hub import HfApi, login
    if args.dry_run and not token:
        # Allow dry run without auth to preview the plan
        api = HfApi(token="dry-run-no-token")
        log.info("=== DRY RUN — no files will be uploaded ===")
        if args.what in ("dataset", "all"):
            upload_dataset(api, dry_run=True)
        if args.what in ("models", "all"):
            upload_models(api, dry_run=True)
        log.info("=== Dry run complete ===")
        return

    if token:
        login(token=token)
    else:
        # Falls back to cached token from `huggingface-cli login` or hf_hub.login()
        try:
            api = HfApi()
            api.whoami()  # will raise if not logged in
        except Exception:
            log.error("Not logged in. Run:  python3 -c \"from huggingface_hub import login; login()\"")
            log.error("Or pass --token YOUR_TOKEN or set HF_TOKEN env var.")
            sys.exit(1)

    api = HfApi()
    log.info(f"Logged in as: {api.whoami()['name']}")

    if args.dry_run:
        log.info("=== DRY RUN — no files will be uploaded ===")

    if args.what in ("dataset", "all"):
        upload_dataset(api, dry_run=args.dry_run)
    if args.what in ("models", "all"):
        upload_models(api, dry_run=args.dry_run)

    log.info("=== Upload complete ===")


if __name__ == "__main__":
    main()
