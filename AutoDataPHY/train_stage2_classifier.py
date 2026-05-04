#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_stage2_classifier.py

Train a stage-2 classifier using existing captures.csv (no manual labels).
Label: y = 1 if status == "ok" else 0.

Typical usage:
  python3 train_stage2_classifier.py \
    --run_dirs rf_stream/ber_sweep/run_20260427_230640 rf_stream/ber_sweep/run_20260427_223155 \
    --out_dir stage2_out \
    --model xgb \
    --target_recall 0.95

This produces:
  stage2_out/stage2_model.joblib
  stage2_out/stage2_scaler.joblib
  stage2_out/threshold_recommendation.json
  stage2_out/metrics.json
  stage2_out/plots/*.png
"""

import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve, roc_curve,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib


def safe_float(x, default=np.nan):
    try:
        if x is None:
            return default
        if isinstance(x, str) and x.strip() == "":
            return default
        return float(x)
    except Exception:
        return default


def load_runs(run_dirs: List[str]) -> pd.DataFrame:
    rows = []
    for rd in run_dirs:
        csv_path = os.path.join(rd, "captures.csv") if os.path.isdir(rd) else rd
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Missing: {csv_path}")
        df = pd.read_csv(csv_path)
        df["run_dir"] = os.path.dirname(csv_path)
        # some runs store modulation column, some not
        if "modulation" not in df.columns:
            df["modulation"] = "unknown"
        rows.append(df)
    out = pd.concat(rows, axis=0, ignore_index=True)
    return out


def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    # label
    y = (df["status"].astype(str).str.lower() == "ok").astype(np.int32)

    # core engineered features (keep simple & robust)
    # Make sure required cols exist
    for c in ["maxe", "eg_th", "p10", "xc_best_peak", "probe_evm", "snr_db", "cfo_hz", "peak"]:
        if c not in df.columns:
            df[c] = np.nan

    # numeric conversion
    num = pd.DataFrame({
        "peak": df["peak"].map(safe_float),
        "p10": df["p10"].map(safe_float),
        "eg_th": df["eg_th"].map(safe_float),
        "maxe": df["maxe"].map(safe_float),
        "xc_best_peak": df["xc_best_peak"].map(safe_float),
        "probe_evm": df["probe_evm"].map(safe_float),
        "snr_db": df["snr_db"].map(safe_float),
        "abs_cfo_hz": df["cfo_hz"].map(lambda v: abs(safe_float(v, 0.0))),
    })

    # derived
    eps = 1e-12
    num["gate_ratio"] = num["maxe"] / (num["eg_th"] + eps)
    num["gate_margin"] = num["maxe"] - num["eg_th"]
    num["log_xc"] = np.log10(num["xc_best_peak"] + 1.0)
    num["log_maxe"] = np.log10(num["maxe"] + 1.0)
    num["log_gate_ratio"] = np.log10(num["gate_ratio"] + 1.0)

    # optional categorical: modulation + rx_gain
    if "rx_gain" in df.columns:
        num["rx_gain"] = df["rx_gain"].map(safe_float)
    else:
        num["rx_gain"] = np.nan

    # handle modulation
    mod = df["modulation"].astype(str).str.lower()
    for m in ["qpsk", "qam16", "qam32", "qam8", "bpsk"]:
        num[f"mod_{m}"] = (mod == m).astype(np.float32)

    # fill missing with medians
    for c in num.columns:
        med = np.nanmedian(num[c].values)
        if np.isnan(med):
            med = 0.0
        num[c] = num[c].fillna(med)

    return num, y


def choose_threshold_for_recall(y_true: np.ndarray, y_score: np.ndarray, target_recall: float):
    # precision_recall_curve returns precision, recall, thresholds (len(th)=len(prec)-1)
    prec, rec, th = precision_recall_curve(y_true, y_score)
    # We want recall >= target_recall, maximize precision (or minimize false accept)
    # pick best precision under recall constraint
    best = None
    for i in range(len(th)):
        if rec[i] >= target_recall:
            cand = (prec[i], rec[i], th[i])
            if best is None or cand[0] > best[0]:
                best = cand
    if best is None:
        # fallback: highest recall available
        i = int(np.argmax(rec))
        # th not defined at last point; safe fallback
        t = float(th[min(i, len(th)-1)]) if len(th) else 0.5
        return {"threshold": t, "precision": float(prec[i]), "recall": float(rec[i])}
    return {"threshold": float(best[2]), "precision": float(best[0]), "recall": float(best[1])}


def plot_curves(out_dir: str, y_true: np.ndarray, y_score: np.ndarray, title_suffix: str):
    os.makedirs(out_dir, exist_ok=True)

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"ROC {title_suffix}")
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "roc.png"), dpi=140)
    plt.close()

    # PR
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall {title_suffix}")
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "pr.png"), dpi=140)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dirs", nargs="+", required=True, help="Run directories or direct captures.csv paths")
    ap.add_argument("--out_dir", default="stage2_out")
    ap.add_argument("--model", choices=["logreg", "rf", "xgb"], default="logreg")
    ap.add_argument("--target_recall", type=float, default=0.95)
    ap.add_argument("--test_size", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--xgb_estimators", type=int, default=400)
    ap.add_argument("--xgb_depth", type=int, default=6)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    plots_dir = os.path.join(args.out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    df = load_runs(args.run_dirs)
    X, y = build_features(df)

    # stratify by label; optionally also by rx_gain buckets
    strat = y
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values, test_size=args.test_size, random_state=args.seed, stratify=strat
    )

    if args.model == "logreg":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=400, class_weight="balanced"))
        ])
    elif args.model == "rf":
        model = Pipeline([
            ("scaler", StandardScaler()),  # not required for RF but ok
            ("clf", RandomForestClassifier(
                n_estimators=600, max_depth=None,
                class_weight="balanced_subsample",
                random_state=args.seed, n_jobs=-1
            ))
        ])
    else:
        # optional dependency
        try:
            import xgboost as xgb
        except Exception as e:
            raise RuntimeError("xgb selected but xgboost not installed. Use --model logreg or rf, or install xgboost.") from e

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", xgb.XGBClassifier(
                n_estimators=args.xgb_estimators,
                max_depth=args.xgb_depth,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                objective="binary:logistic",
                eval_metric="logloss",
                n_jobs=-1,
                random_state=args.seed,
            ))
        ])

    model.fit(X_train, y_train)

    # scores
    y_score = model.predict_proba(X_test)[:, 1]
    y_pred = (y_score >= 0.5).astype(np.int32)

    auc = roc_auc_score(y_test, y_score)
    ap_score = average_precision_score(y_test, y_score)
    cm = confusion_matrix(y_test, y_pred)

    th_rec = choose_threshold_for_recall(y_test, y_score, args.target_recall)
    th = th_rec["threshold"]
    y_pred_th = (y_score >= th).astype(np.int32)
    cm_th = confusion_matrix(y_test, y_pred_th)

    metrics = {
        "n_total": int(len(y)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "roc_auc": float(auc),
        "avg_precision": float(ap_score),
        "cm@0.5": cm.tolist(),
        "cm@th": cm_th.tolist(),
        "threshold_for_target_recall": th_rec,
        "feature_names": list(X.columns),
        "model": args.model,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # plots
    plot_curves(plots_dir, y_test, y_score, title_suffix=f"({args.model})")

    # save model pipeline
    joblib.dump(model, os.path.join(args.out_dir, "stage2_model.joblib"))

    # save an explicit json with deploy thresholds (you can paste into RX)
    with open(os.path.join(args.out_dir, "threshold_recommendation.json"), "w") as f:
        json.dump({
            "target_recall": args.target_recall,
            "threshold": th,
            "precision": th_rec["precision"],
            "recall": th_rec["recall"],
            "notes": "Use score >= threshold to accept packet candidates before CRC/after coarse demod."
        }, f, indent=2)

    print("Saved:", args.out_dir)
    print(json.dumps(metrics, indent=2))
    print("\nDeploy suggestion:")
    print(f"  if stage2_score >= {th:.4f}: accept candidate (expected recall≈{th_rec['recall']:.3f}, precision≈{th_rec['precision']:.3f})")


if __name__ == "__main__":
    main()

"""
python rf_stream/train_stage2_classifier.py \
  --run_dirs rf_stream/ber_sweep/run_20260427_230640 rf_stream/ber_sweep/run_20260427_223155 \
  --out_dir rf_stream/stage2_out_xgb \
  --model xgb \
  --target_recall 0.95
"""
