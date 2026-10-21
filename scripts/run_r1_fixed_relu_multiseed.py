#!/usr/bin/env python3
"""Repeat the fixed ReLU baseline across seeds on the saved R1 outer folds.

Seeds are training initializations (and sklearn's internal early-stopping
split); they are sensitivity analyses, not independent clinical observations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_r1_nested_experiments as core


ARCHITECTURE = (16, 8)
MODEL_ARMS = (
    ("mlp_relu_fixed_unweighted", False),
    ("mlp_relu_fixed_patient_balanced", True),
)
METRICS = (
    "balanced_accuracy", "accuracy", "mcc", "sensitivity", "specificity",
    "roc_auc", "pr_auc", "brier_score",
)


def atomic_json(data: dict[str, Any], path: Path) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temp.replace(path)


def load_splits(path: Path, groups: pd.DataFrame):
    frame = pd.read_csv(path, dtype={"group_id": str})
    required = {"repeat", "fold", "group_id", "partition"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Split manifest lacks columns: {sorted(required - set(frame.columns))}")
    known = set(groups["group_id"].astype(str))
    if set(frame["group_id"]) - known:
        raise ValueError("Split manifest contains unknown subjects.")
    result = []
    for (repeat, fold), block in frame.groupby(["repeat", "fold"], sort=True):
        train = set(block.loc[block.partition == "train", "group_id"])
        test = set(block.loc[block.partition == "test", "group_id"])
        if block.duplicated(["group_id"]).any() or not train or not test or train & test or train | test != known:
            raise ValueError(f"Invalid subject partition in repeat {repeat}, fold {fold}.")
        tr = np.flatnonzero(groups.group_id.astype(str).isin(train).to_numpy())
        te = np.flatnonzero(groups.group_id.astype(str).isin(test).to_numpy())
        result.append((int(repeat), int(fold), tr, te))
    if not result:
        raise ValueError("No outer splits were found.")
    return result


def write_summaries(predictions: pd.DataFrame, out_dir: Path) -> None:
    fold_rows, repeat_rows = [], []
    keys = ["feature", "model", "seed", "repeat"]
    for key, block in predictions.groupby(keys, sort=True):
        row = core.metric_row(block.label.to_numpy(int), block.probability.to_numpy(float),
                              key[1], int(key[3]), None)
        row.update(dict(zip(keys, key)))
        repeat_rows.append(row)
    repeat_frame = pd.DataFrame(repeat_rows)
    for (feature, model, seed, repeat, fold), block in predictions.groupby(
        ["feature", "model", "seed", "repeat", "fold"], sort=True
    ):
        row = core.metric_row(block.label.to_numpy(int), block.probability.to_numpy(float),
                              model, int(repeat), int(fold))
        row.update(feature=feature, model=model, seed=int(seed))
        fold_rows.append(row)
    seed_rows = []
    for (feature, model, seed), block in repeat_frame.groupby(["feature", "model", "seed"], sort=True):
        row = {"feature": feature, "model": model, "seed": int(seed), "n_repeats": len(block)}
        for metric in METRICS:
            row[metric + "_mean"] = float(block[metric].mean())
            row[metric + "_repeat_sd"] = float(block[metric].std(ddof=1))
        seed_rows.append(row)
    seed_frame = pd.DataFrame(seed_rows)
    aggregate_rows = []
    for (feature, model), block in seed_frame.groupby(["feature", "model"], sort=True):
        row = {"feature": feature, "model": model, "n_seeds": int(block.seed.nunique())}
        for metric in METRICS:
            vals = block[metric + "_mean"].to_numpy(float)
            row[metric + "_mean_across_seeds"] = float(vals.mean())
            row[metric + "_sd_across_seeds"] = float(vals.std(ddof=1))
            row[metric + "_min_seed"] = float(vals.min())
            row[metric + "_max_seed"] = float(vals.max())
        aggregate_rows.append(row)
    core.atomic_csv(pd.DataFrame(fold_rows), out_dir / "outer_fold_metrics.csv")
    core.atomic_csv(repeat_frame, out_dir / "seed_repeat_metrics.csv")
    core.atomic_csv(seed_frame, out_dir / "seed_aggregate_metrics.csv")
    core.atomic_csv(pd.DataFrame(aggregate_rows), out_dir / "aggregate_metrics.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", type=Path, default=Path("svm-mlp-comparison/features_win"))
    parser.add_argument("--split-manifest", type=Path, default=Path("results_r1_nested/outer_split_manifest.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("results_r1_fixed_relu_multiseed"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2026, 3407, 8675309])
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--corr-threshold", type=float, default=0.85)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Seeds must be unique.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = args.out_dir / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    manifest, groups = core.load_manifest(args.feature_dir / "manifest.csv")
    splits = load_splits(args.split_manifest, groups)
    config: dict[str, Any] = {
        "design": "fixed ReLU initialization sensitivity on the shared R1 outer subject folds",
        "architecture": list(ARCHITECTURE), "activation": "relu", "seeds": args.seeds,
        "optimizer": "adam", "learning_rate_init": 0.001, "max_iter": 500,
        "early_stopping": True, "validation_fraction": 0.15, "n_iter_no_change": 20,
        "early_stopping_validation": "scikit-learn random window-level split within outer training partition",
        "arms": {name: {"patient_balanced": patient} for name, patient in MODEL_ARMS},
        "feature_dir": str(args.feature_dir), "split_manifest": str(args.split_manifest),
        "top_k": args.top_k, "corr_threshold": args.corr_threshold,
        "feature_files": core.FEATURE_FILES, "n_rows": len(manifest),
        "n_subjects": len(groups), "n_outer_splits": len(splits),
        "python": sys.version, "platform": platform.platform(), "sklearn": sklearn.__version__,
        "interpretation": "Seed-level variability is descriptive training sensitivity, not independent subject-level evidence.",
    }
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    config["config_hash"] = config_hash
    (args.out_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    predictions: list[dict[str, Any]] = []
    completed: set[tuple[str, int, int, int]] = set()
    if args.resume:
        for path in sorted(checkpoints.glob("fixed_relu_s*_r*_f*.json")):
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved.get("config_hash") != config_hash:
                raise RuntimeError(f"Checkpoint configuration mismatch: {path}")
            key = (saved["feature"], int(saved["seed"]), int(saved["repeat"]), int(saved["fold"]))
            completed.add(key)
            predictions.extend(saved["predictions"])

    started = time.time()
    with threadpool_limits(limits=1):
        for feature, filename in core.FEATURE_FILES.items():
            X, names = core.load_feature_matrix(args.feature_dir / filename, manifest)
            core.log(f"Loaded {feature}: {X.shape[0]} windows x {X.shape[1]} features.")
            for repeat, fold, train_idx, test_idx in splits:
                train_groups = groups.iloc[train_idx].group_id.astype(str).to_numpy()
                test_groups = groups.iloc[test_idx].group_id.astype(str).to_numpy()
                xtr, xte, _, _, selected = core.prepare_fold(
                    X, manifest, train_groups, test_groups, names, args.top_k, args.corr_threshold
                )
                train_rows = core.row_indices(manifest, train_groups)
                test_rows = core.row_indices(manifest, test_groups)
                ytr = manifest.iloc[train_rows].label.to_numpy(int)
                gtr = manifest.iloc[train_rows].group_id.astype(str).to_numpy()
                yte = manifest.iloc[test_rows].label.to_numpy(int)
                gte = manifest.iloc[test_rows].group_id.astype(str).to_numpy()
                for seed in args.seeds:
                    checkpoint = checkpoints / f"fixed_relu_s{seed}_{feature}_r{repeat}_f{fold}.json"
                    key = (feature, seed, repeat, fold)
                    if key in completed:
                        continue
                    fold_predictions = []
                    for model_name, patient_balanced in MODEL_ARMS:
                        row_probs = core.fit_predict(
                            "mlp_relu", {"hidden_layer_sizes": ARCHITECTURE}, xtr, ytr, gtr,
                            xte, seed, patient_balanced, False,
                        )
                        grouped = core.group_probabilities(yte, gte, row_probs)
                        for row in grouped.to_dict(orient="records"):
                            fold_predictions.append({
                                "feature": feature, "model": model_name, "seed": seed,
                                "repeat": repeat, "fold": fold, "group_id": row["group_id"],
                                "label": int(row["label"]), "probability": float(row["probability"]),
                                "threshold": 0.5, "prediction": int(row["probability"] >= 0.5),
                                "architecture": json.dumps(ARCHITECTURE), "activation": "relu",
                                "patient_balanced": patient_balanced,
                                "n_selected_features": len(selected),
                                "selected_features": json.dumps(selected),
                            })
                    atomic_json({"config_hash": config_hash, "feature": feature,
                                 "seed": seed, "repeat": repeat, "fold": fold,
                                 "predictions": fold_predictions}, checkpoint)
                    predictions.extend(fold_predictions)
                    completed.add(key)
                    frame = pd.DataFrame(predictions)
                    core.atomic_csv(frame, args.out_dir / "outer_oof_predictions.csv")
                    write_summaries(frame, args.out_dir)
                    core.log(f"Saved {feature} seed {seed} r{repeat}f{fold} "
                             f"({len(completed)}/{len(core.FEATURE_FILES) * len(args.seeds) * len(splits)} blocks).")

    expected = len(core.FEATURE_FILES) * len(args.seeds) * len(splits)
    if len(completed) != expected:
        raise RuntimeError(f"Incomplete run: {len(completed)} blocks; expected {expected}.")
    frame = pd.DataFrame(predictions)
    core.atomic_csv(frame, args.out_dir / "outer_oof_predictions.csv")
    write_summaries(frame, args.out_dir)
    metadata = {**config, "elapsed_seconds": time.time() - started,
                "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "note": "All preprocessing and feature selection use outer-training subjects only. Predictions are aggregated by subject. The five seeds are repeated fits of the same cohort and do not increase the clinical sample size."}
    atomic_json(metadata, args.out_dir / "run_metadata.json")
    core.log(f"Fixed ReLU multi-seed run complete: {expected} feature/seed/fold blocks in {metadata['elapsed_seconds'] / 60:.1f} min.")


if __name__ == "__main__":
    main()
