#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
upload_otfs_to_hf.py — Upload OTFS data_staging/ to HuggingFace.

Usage:
  # Dry-run (no auth, no network writes):
  python3 upload_otfs_to_hf.py --dry_run

  # Real upload (requires HF_TOKEN env var or --token):
  python3 upload_otfs_to_hf.py

  # Upload only one split (for testing or resume):
  python3 upload_otfs_to_hf.py --only closed_grid_0db_otfs

Repos created:  hf.co/datasets/lkk688/rf-stream-otfs

The script commits one HF revision per split (and one for tx_iq + dataset card).
Per-split commits avoid Hub-side timeouts on the multi-GB upload.
"""

import argparse
import json
import logging
import os
import sys
import time

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("hf_upload")

HF_USER       = "lkk688"
DATASET_REPO  = f"{HF_USER}/rf-stream-otfs"
BASE_DIR      = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STAGE_DIR     = os.path.join(BASE_DIR, "data_staging")


def main():
    ap = argparse.ArgumentParser("Upload OTFS staging dir to HuggingFace")
    ap.add_argument("--dry_run", action="store_true",
                    help="Print file counts and sizes, do not contact HF Hub.")
    ap.add_argument("--token",   default="",
                    help="HF write token. Falls back to HF_TOKEN env var.")
    ap.add_argument("--only",    default="",
                    help="If set, upload only this split (top-level dir under data_staging/). "
                         "Use 'card' to upload just the top-level DATASET.md.")
    args = ap.parse_args()

    if not os.path.isdir(STAGE_DIR):
        log.error("Staging dir not found: %s. Run stage_data_for_hf.py first.", STAGE_DIR)
        sys.exit(1)

    manifest_path = os.path.join(STAGE_DIR, "MANIFEST.json")
    if not os.path.exists(manifest_path):
        log.error("MANIFEST.json missing. Run stage_data_for_hf.py first."); sys.exit(1)
    with open(manifest_path) as f:
        manifest = json.load(f)

    # Token resolution
    token = args.token or os.environ.get("HF_TOKEN", "")
    if not args.dry_run and not token:
        log.error("No token (no --token, no HF_TOKEN env var). Set HF_TOKEN or pass --token.")
        sys.exit(1)

    # Enumerate splits = top-level dirs under STAGE_DIR (depth 1 except ota_doppler which has subdirs)
    splits = []
    for entry in sorted(os.listdir(STAGE_DIR)):
        p = os.path.join(STAGE_DIR, entry)
        if os.path.isdir(p):
            splits.append(entry)

    if args.only:
        splits = [s for s in splits if s == args.only]
        if not splits and args.only != "card":
            log.error("--only %s did not match any staged split. Available: %s",
                       args.only, sorted(os.listdir(STAGE_DIR)))
            sys.exit(1)

    log.info("Target repo: hf.co/datasets/%s", DATASET_REPO)
    log.info("Mode: %s", "DRY-RUN" if args.dry_run else "REAL UPLOAD")
    log.info("Splits to upload: %s", splits + (["card"] if args.only == "card" else []))

    # Total size sanity
    total = 0; total_files = 0
    for s in splits:
        for root, _, files in os.walk(os.path.join(STAGE_DIR, s)):
            for fn in files:
                fp = os.path.join(root, fn)
                total += os.path.getsize(fp)
                total_files += 1
    log.info("Total to upload: %d files, %.2f GB", total_files, total / 1e9)

    if args.dry_run:
        log.info("DRY-RUN complete. No data was uploaded.")
        return

    # Real upload
    from huggingface_hub import HfApi, create_repo

    api = HfApi(token=token)
    try:
        create_repo(DATASET_REPO, repo_type="dataset", token=token, exist_ok=True)
        log.info("Repo ready: %s", DATASET_REPO)
    except Exception as e:
        log.warning("create_repo: %s", e)

    # Upload top-level dataset card first
    card_path = os.path.join(STAGE_DIR, "DATASET.md")
    if os.path.exists(card_path) and args.only in ("", "card"):
        log.info("Uploading dataset card -> README.md ...")
        try:
            api.upload_file(path_or_fileobj=card_path,
                             path_in_repo="README.md",
                             repo_id=DATASET_REPO,
                             repo_type="dataset",
                             commit_message="Add dataset card")
        except Exception as e:
            log.error("Card upload failed: %s", e)

    # Upload manifest as well
    if args.only in ("",):
        try:
            api.upload_file(path_or_fileobj=manifest_path,
                             path_in_repo="MANIFEST.json",
                             repo_id=DATASET_REPO,
                             repo_type="dataset",
                             commit_message="Add MANIFEST.json")
        except Exception as e:
            log.error("Manifest upload failed: %s", e)

    if args.only == "card":
        log.info("Card-only mode: done."); return

    # Per-split folder uploads.  For splits with many large files we batch
    # the files into ~250 MB chunks per commit so a single network hiccup
    # only loses the in-flight chunk (not the whole split).  Each chunk
    # becomes one commit with a clear message that lets us resume.
    CHUNK_BYTES = 250 * 1024 * 1024   # 250 MB per commit
    for s in splits:
        local = os.path.join(STAGE_DIR, s)
        repo_path = s
        n = sum(len(f) for _, _, f in os.walk(local))
        size_bytes = sum(os.path.getsize(os.path.join(r, fn))
                          for r, _, files in os.walk(local) for fn in files)
        size_mb = size_bytes / 1e6
        log.info("Uploading split: %s (%d files, %.1f MB) ...", s, n, size_mb)
        t0 = time.time()

        # Gather all files relative to `local`, sort for deterministic chunking
        all_files = []
        for root, _, fns in os.walk(local):
            for fn in fns:
                fp = os.path.join(root, fn)
                rel = os.path.relpath(fp, local)
                all_files.append((rel, os.path.getsize(fp)))
        all_files.sort()

        # Chunk into ~250 MB batches
        chunks = []; cur = []; cur_bytes = 0
        for rel, sz in all_files:
            if cur and cur_bytes + sz > CHUNK_BYTES:
                chunks.append(cur); cur = []; cur_bytes = 0
            cur.append(rel); cur_bytes += sz
        if cur: chunks.append(cur)

        log.info("  split into %d commit(s), ~%.0f MB each", len(chunks), CHUNK_BYTES / 1e6)

        ok = True
        for ci, chunk in enumerate(chunks, 1):
            chunk_bytes = sum(os.path.getsize(os.path.join(local, rel)) for rel in chunk)
            t1 = time.time()
            # Build CommitOperationAdd list — robust against single-file retries
            from huggingface_hub import CommitOperationAdd
            ops = [CommitOperationAdd(
                       path_in_repo=os.path.join(repo_path, rel),
                       path_or_fileobj=os.path.join(local, rel))
                   for rel in chunk]
            attempts = 0; max_attempts = 4
            while attempts < max_attempts:
                attempts += 1
                try:
                    api.create_commit(
                        repo_id=DATASET_REPO,
                        repo_type="dataset",
                        operations=ops,
                        commit_message=f"Add {s} chunk {ci}/{len(chunks)} ({len(chunk)} files, {chunk_bytes/1e6:.1f} MB)",
                    )
                    log.info("    chunk %d/%d  %d files  %.1f MB  in %.1fs",
                              ci, len(chunks), len(chunk), chunk_bytes/1e6, time.time() - t1)
                    break
                except Exception as e:
                    log.warning("    chunk %d/%d attempt %d/%d failed: %s",
                                 ci, len(chunks), attempts, max_attempts, e)
                    if attempts < max_attempts:
                        wait = 5 * attempts
                        log.warning("    retrying in %ds ...", wait)
                        time.sleep(wait)
                    else:
                        log.error("    chunk %d/%d EXHAUSTED retries", ci, len(chunks))
                        ok = False
                        break
        log.info("  split %s %s in %.1fs", s, "OK" if ok else "PARTIAL", time.time() - t0)

    log.info("Upload complete. Browse: https://huggingface.co/datasets/%s", DATASET_REPO)


if __name__ == "__main__":
    main()
