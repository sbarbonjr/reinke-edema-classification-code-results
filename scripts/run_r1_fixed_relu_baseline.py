#!/usr/bin/env python3
"""Evaluate a pre-specified ReLU MLP on the shared R1 outer subject folds.

The baseline mirrors the fixed logistic MLP's (16, 8) architecture and seed
(42); only activation differs. No inner-loop architecture or seed selection is
performed. Both unweighted and patient-balanced training are reported.
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
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_r1_nested_experiments as core


ARCHITECTURE = (16, 8)
MODEL_SEED = 42
MODEL_ARMS = (
    ("mlp_relu_fixed_unweighted", False),
    ("mlp_relu_fixed_patient_balanced", True),
)


def atomic_json(data: dict[str, Any], path: Path) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data), encoding="utf-8")
    temp.replace(path)


def load_splits(path: Path, groups: pd.DataFrame) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    frame = pd.read_csv(path, dtype={"group_id": str})
    required = {"repeat", "fold", "group_id", "partition"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Split manifest lacks columns: {sorted(required - set(frame.columns))}")
    known = set(groups["group_id"].astype(str))
    if not set(frame["group_id"]).issubset(known):
        raise ValueError("Split manifest contains unknown subjects.")
    result = []
    for (repeat, fold), block in frame.groupby(["repeat", "fold"], sort=True):
        if block.duplicated(["group_id"]).any():
            raise ValueError(f"Duplicate subject in repeat {repeat}, fold {fold}.")
        train = set(block.loc[block["partition"] == "train", "group_id"])
        test = set(block.loc[block["partition"] == "test", "group_id"])
        if not train or not test or train & test or train | test != known:
            raise ValueError(f"Invalid subject partition in repeat {repeat}, fold {fold}.")
        train_idx = np.flatnonzero(groups["group_id"].astype(str).isin(train).to_numpy())
        test_idx = np.flatnonzero(groups["group_id"].astype(str).isin(test).to_numpy())
        result.append((int(repeat), int(fold), train_idx, test_idx))
    if not result:
        raise ValueError("No outer splits were found.")
    return result


def checkpoint_file(root: Path, feature: str, repeat: int, fold: int) -> Path:
    return root / "checkpoints" / f"fixed_relu_{feature}_r{repeat}_f{fold}.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", type=Path, default=Path("svm-mlp-comparison/features_win"))
    parser.add_argument("--split-manifest", type=Path,
                        default=Path("results_r1_nested/outer_split_manifest.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("results_r1_fixed_relu"))
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--corr-threshold", type=float, default=0.85)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "checkpoints").mkdir(exist_ok=True)
    manifest, groups = core.load_manifest(args.feature_dir / "manifest.csv")
    splits = load_splits(args.split_manifest, groups)
    config: dict[str, Any] = {
        "design": "fixed ReLU counterpart to the fixed logistic MLP; no inner-loop model selection",
        "architecture": list(ARCHITECTURE), "activation": "relu", "seed": MODEL_SEED,
        "optimizer": "adam", "learning_rate_init": 0.001, "max_iter": 500,
        "early_stopping": True, "validation_fraction": 0.15,
        "n_iter_no_change": 20, "arms": {name: {"patient_balanced": patient}
                                            for name, patient in MODEL_ARMS},
        "feature_dir": str(args.feature_dir), "split_manifest": str(args.split_manifest),
        "top_k": args.top_k, "corr_threshold": args.corr_threshold,
        "feature_files": core.FEATURE_FILES, "n_rows": len(manifest),
        "n_subjects": len(groups), "n_outer_splits": len(splits),
        "python": sys.version, "platform": platform.platform(), "sklearn": sklearn.__version__,
    }
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    config["config_hash"] = config_hash
    (args.out_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    completed: set[tuple[str, int, int]] = set()
    predictions: list[dict[str, Any]] = []
    if args.resume:
        for checkpoint in sorted((args.out_dir / "checkpoints").glob("fixed_relu_*_r*_f*.json")):
            stored = json.loads(checkpoint.read_text(encoding="utf-8"))
            if stored.get("config_hash") != config_hash:
                raise RuntimeError(f"Checkpoint configuration mismatch: {checkpoint}")
            completed.add((stored["feature"], int(stored["repeat"]), int(stored["fold"])))
            predictions.extend(stored["predictions"])

    started = time.time()
    with threadpool_limits(limits=1):
        for feature, filename in core.FEATURE_FILES.items():
            X, names = core.load_feature_matrix(args.feature_dir / filename, manifest)
            core.log(f"Loaded {feature}: {X.shape[0]} windows x {X.shape[1]} features.")
            for repeat, fold, train_idx, test_idx in splits:
                key = (feature, repeat, fold)
                if key in completed:
                    continue
                tic = time.time()
                train_groups = groups.iloc[train_idx]["group_id"].astype(str).to_numpy()
                test_groups = groups.iloc[test_idx]["group_id"].astype(str).to_numpy()
                xtr, xte, _, _, selected_features = core.prepare_fold(
                    X, manifest, train_groups, test_groups, names,
                    args.top_k, args.corr_threshold,
                )
                train_rows = core.row_indices(manifest, train_groups)
                test_rows = core.row_indices(manifest, test_groups)
                y_train = manifest.iloc[train_rows]["label"].to_numpy(dtype=int)
                g_train = manifest.iloc[train_rows]["group_id"].astype(str).to_numpy()
                y_test = manifest.iloc[test_rows]["label"].to_numpy(dtype=int)
                g_test = manifest.iloc[test_rows]["group_id"].astype(str).to_numpy()
                fold_predictions: list[dict[str, Any]] = []
                for model_name, patient_balanced in MODEL_ARMS:
                    p_rows = core.fit_predict(
                        "mlp_relu", {"hidden_layer_sizes": ARCHITECTURE},
                        xtr, y_train, g_train, xte, MODEL_SEED,
                        patient_balanced, False,
                    )
                    grouped = core.group_probabilities(y_test, g_test, p_rows)
                    ba = balanced_accuracy_score(
                        grouped["label"], (grouped["probability"] >= 0.5).astype(int)
                    )
                    core.log(f"{feature} r{repeat}f{fold} {model_name}: subject BA={ba:.4f}")
                    for row in grouped.to_dict(orient="records"):
                        fold_predictions.append({
                            "feature": feature, "model": model_name,
                            "repeat": repeat, "fold": fold,
                            "group_id": row["group_id"], "label": int(row["label"]),
                            "probability": float(row["probability"]),
                            "threshold": 0.5,
                            "prediction": int(row["probability"] >= 0.5),
                            "architecture": json.dumps(ARCHITECTURE),
                            "activation": "relu", "seed": MODEL_SEED,
                            "patient_balanced": patient_balanced,
                            "n_selected_features": len(selected_features),
                            "selected_features": json.dumps(selected_features),
                        })
                checkpoint = checkpoint_file(args.out_dir, feature, repeat, fold)
                atomic_json({"config_hash": config_hash, "feature": feature,
                             "repeat": repeat, "fold": fold,
                             "predictions": fold_predictions}, checkpoint)
                predictions.extend(fold_predictions)
                completed.add(key)
                pred_frame = pd.DataFrame(predictions)
                core.atomic_csv(pred_frame, args.out_dir / "outer_oof_predictions.csv")
                core.summarize_predictions(pred_frame, args.out_dir)
                core.log(f"Saved {feature}, repeat {repeat}, fold {fold} "
                         f"in {time.time() - tic:.1f}s ({len(completed)}/"
                         f"{len(core.FEATURE_FILES) * len(splits)} checkpoints).")

    predictions_frame = pd.DataFrame(predictions)
    expected = len(core.FEATURE_FILES) * len(splits) * len(MODEL_ARMS)
    observed = predictions_frame[["feature", "model", "repeat", "fold"]].drop_duplicates().shape[0]
    if observed != expected:
        raise RuntimeError(f"Incomplete run: found {observed} feature/model/fold blocks; expected {expected}.")
    core.atomic_csv(predictions_frame, args.out_dir / "outer_oof_predictions.csv")
    core.summarize_predictions(predictions_frame, args.out_dir)
    metadata = {
        **config, "elapsed_seconds": time.time() - started,
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": "Outer subjects match the saved R1 split manifest exactly. Feature selection and scaling use outer-training data only; probabilities are averaged by subject. No outer-test outcomes were used for model selection or calibration.",
    }
    (args.out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    core.log(f"Fixed ReLU baseline complete in {metadata['elapsed_seconds'] / 60:.1f} minutes.")


if __name__ == "__main__":
    main()
