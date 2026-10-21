#!/usr/bin/env python3
"""Nested subject-grouped experiments for the R1 Reinke-edema revision.

Every feature-selection step and hyperparameter choice is made using only
outer-training subjects. Inner and outer split manifests are shared across
representations. Predictions and calibration diagnostics are at subject level.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits


FEATURE_FILES = {
    "fft_topk_hann": "fft_topk_hann.csv",
    "mfcc_stats": "mfcc_stats.csv",
    "spectral_stats": "spectral_stats.csv",
    "wavlm": "wav2vec2.csv",  # legacy filename; the embedding is WavLM.
    "wpd_haar7_std": "wpd_haar7_std.csv",
}
META_COLUMNS = {
    "file", "filename", "group_id", "segment_id", "start_sample", "start_sec",
    "label", "class_name", "sr", "segment_sec", "target_sec",
}
MLP_SEEDS = (17, 37, 59, 83, 109)
MLP_ARCHITECTURES = ((16,), (32,), (32, 16))


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def load_manifest(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = pd.read_csv(path)
    required = {"group_id", "segment_id", "label"}
    if not required.issubset(manifest.columns):
        raise ValueError(f"Manifest lacks required columns: {sorted(required - set(manifest.columns))}")
    manifest["group_id"] = manifest["group_id"].astype(str)
    manifest["segment_id"] = manifest["segment_id"].astype(int)
    labels_per_group = manifest.groupby("group_id")["label"].nunique()
    if (labels_per_group != 1).any():
        raise ValueError("At least one group has inconsistent labels.")
    groups = (
        manifest.groupby("group_id", sort=True)
        .agg(label=("label", "first"), n_windows=("segment_id", "size"))
        .reset_index()
    )
    return manifest, groups


def load_feature_matrix(path: Path, manifest: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    frame = pd.read_csv(path)
    for key in ("group_id", "segment_id", "label"):
        if key not in frame:
            raise ValueError(f"{path} lacks {key}")
    frame["group_id"] = frame["group_id"].astype(str)
    frame["segment_id"] = frame["segment_id"].astype(int)
    expected = manifest[["group_id", "segment_id", "label"]].sort_values(
        ["group_id", "segment_id"]
    ).reset_index(drop=True)
    observed = frame[["group_id", "segment_id", "label"]].sort_values(
        ["group_id", "segment_id"]
    ).reset_index(drop=True)
    if not expected.equals(observed):
        raise ValueError(f"Feature rows in {path} do not match the shared manifest.")
    feature_names = [
        col for col in frame.columns
        if col not in META_COLUMNS and pd.api.types.is_numeric_dtype(frame[col])
    ]
    if not feature_names:
        raise ValueError(f"No numeric feature columns found in {path}")
    return frame[feature_names].to_numpy(dtype=np.float64), feature_names


def make_group_splits(
    group_table: pd.DataFrame, n_splits: int, seed: int, repeats: int
) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    y = group_table["label"].to_numpy(dtype=int)
    group_ids = group_table["group_id"].to_numpy(dtype=str)
    result = []
    for repeat in range(1, repeats + 1):
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=seed + repeat - 1
        )
        for fold, (tr, te) in enumerate(
            splitter.split(np.zeros((len(y), 1)), y, group_ids), start=1
        ):
            result.append((repeat, fold, tr, te))
    return result


def row_indices(manifest: pd.DataFrame, selected_groups: np.ndarray) -> np.ndarray:
    return np.flatnonzero(manifest["group_id"].isin(set(selected_groups)).to_numpy())


def prepare_fold(
    X: np.ndarray,
    manifest: pd.DataFrame,
    train_groups: np.ndarray,
    test_groups: np.ndarray,
    feature_names: list[str],
    top_k: int,
    corr_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    train_rows = row_indices(manifest, train_groups)
    test_rows = row_indices(manifest, test_groups)
    medians = np.nanmedian(X[train_rows], axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    x_train = np.where(np.isfinite(X[train_rows]), X[train_rows], medians)
    x_test = np.where(np.isfinite(X[test_rows]), X[test_rows], medians)

    group_labels = (
        manifest.iloc[train_rows][["group_id", "label"]]
        .drop_duplicates("group_id").set_index("group_id")["label"]
    )
    train_meta = manifest.iloc[train_rows]
    means = (
        pd.DataFrame(x_train, columns=feature_names)
        .assign(group_id=train_meta["group_id"].to_numpy())
        .groupby("group_id", sort=True)[feature_names].mean()
    )
    y_group = group_labels.loc[means.index].to_numpy(dtype=float)
    corr_y = np.zeros(len(feature_names), dtype=float)
    for j in range(len(feature_names)):
        vals = means.iloc[:, j].to_numpy(dtype=float)
        if np.std(vals) > 1e-12:
            c = np.corrcoef(vals, y_group)[0, 1]
            corr_y[j] = 0.0 if not np.isfinite(c) else float(c)
    corr_x = np.corrcoef(means.to_numpy(dtype=float), rowvar=False)
    corr_x = np.nan_to_num(corr_x, nan=0.0, posinf=0.0, neginf=0.0)
    keep = np.ones(len(feature_names), dtype=bool)
    for i in range(len(feature_names)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(feature_names)):
            if keep[j] and abs(corr_x[i, j]) > corr_threshold:
                if abs(corr_y[i]) >= abs(corr_y[j]):
                    keep[j] = False
                else:
                    keep[i] = False
                    break
    eligible = np.flatnonzero(keep)
    ranked = eligible[np.argsort(np.abs(corr_y[eligible]))[::-1]]
    chosen = ranked[: min(top_k, len(ranked))]
    return (
        x_train[:, chosen], x_test[:, chosen],
        manifest.iloc[train_rows]["label"].to_numpy(dtype=int),
        manifest.iloc[train_rows]["group_id"].to_numpy(dtype=str),
        [feature_names[i] for i in chosen],
    )


def train_weights(y: np.ndarray, groups: np.ndarray, patient_balanced: bool) -> np.ndarray | None:
    if not patient_balanced:
        return None
    data = pd.DataFrame({"group": groups, "label": y})
    group_label = data.groupby("group")["label"].first()
    class_group_count = group_label.value_counts().to_dict()
    windows_per_group = data["group"].value_counts().to_dict()
    weights = np.array([
        1.0 / windows_per_group[g] / class_group_count[int(label)]
        for g, label in zip(groups, y)
    ], dtype=float)
    weights *= len(weights) / weights.sum()
    return weights


def estimator(kind: str, params: dict[str, Any], seed: int, class_balanced: bool):
    if kind == "logreg":
        return LogisticRegression(
            C=float(params["C"]), max_iter=3000, solver="lbfgs",
            class_weight="balanced" if class_balanced else None,
            random_state=seed,
        )
    if kind == "svm_linear":
        return SVC(
            kernel="linear", C=float(params["C"]), probability=True,
            class_weight="balanced" if class_balanced else None,
            random_state=seed,
        )
    if kind == "svm_rbf":
        return SVC(
            kernel="rbf", C=float(params["C"]), gamma=params["gamma"],
            probability=True, class_weight="balanced" if class_balanced else None,
            random_state=seed,
        )
    if kind == "mlp_relu":
        return MLPClassifier(
            hidden_layer_sizes=tuple(params["hidden_layer_sizes"]),
            activation="relu", solver="adam", learning_rate_init=1e-3,
            max_iter=500, early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=20, random_state=seed,
        )
    if kind == "mlp_logistic_legacy":
        return MLPClassifier(
            hidden_layer_sizes=(16, 8), activation="logistic", solver="adam",
            learning_rate_init=1e-3, max_iter=500, early_stopping=True,
            validation_fraction=0.15, n_iter_no_change=20, random_state=seed,
        )
    if kind == "mlp_arch_logistic":
        return MLPClassifier(
            hidden_layer_sizes=tuple(params["hidden_layer_sizes"]),
            activation="logistic", solver="adam", learning_rate_init=1e-3,
            max_iter=500, early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=20, random_state=seed,
        )
    raise ValueError(f"Unknown estimator kind: {kind}")


def fit_predict(
    kind: str,
    params: dict[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    g_train: np.ndarray,
    x_eval: np.ndarray,
    seed: int,
    patient_balanced: bool,
    class_balanced: bool,
) -> np.ndarray:
    imputer = SimpleImputer(strategy="median")
    xtr = imputer.fit_transform(x_train)
    xev = imputer.transform(x_eval)
    scaler = StandardScaler()
    xtr = scaler.fit_transform(xtr)
    xev = scaler.transform(xev)
    clf = estimator(kind, params, seed, class_balanced)
    weights = train_weights(y_train, g_train, patient_balanced)
    if weights is None:
        clf.fit(xtr, y_train)
    else:
        clf.fit(xtr, y_train, sample_weight=weights)
    classes = list(clf.classes_)
    return clf.predict_proba(xev)[:, classes.index(1)]


def group_probabilities(
    y_rows: np.ndarray, group_rows: np.ndarray, prob_rows: np.ndarray
) -> pd.DataFrame:
    joined = pd.DataFrame({"group_id": group_rows, "label": y_rows, "probability": prob_rows})
    result = joined.groupby("group_id", sort=True).agg(
        label=("label", "first"), probability=("probability", "mean")
    ).reset_index()
    return result


def candidate_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for C in (0.01, 0.1, 1.0, 10.0, 100.0):
        specs.append({"name": "logreg_window_balanced", "kind": "logreg",
                      "params": {"C": C}, "patient_balanced": False, "class_balanced": True,
                      "seeds": (42,)})
        specs.append({"name": "logreg_patient_balanced", "kind": "logreg",
                      "params": {"C": C}, "patient_balanced": True, "class_balanced": False,
                      "seeds": (42,)})
        specs.append({"name": "svm_linear_window_balanced", "kind": "svm_linear",
                      "params": {"C": C}, "patient_balanced": False, "class_balanced": True,
                      "seeds": (42,)})
        specs.append({"name": "svm_linear_patient_balanced", "kind": "svm_linear",
                      "params": {"C": C}, "patient_balanced": True, "class_balanced": False,
                      "seeds": (42,)})
    for C in (0.1, 1.0, 10.0, 100.0):
        for gamma in ("scale", 0.01, 0.1):
            specs.append({"name": "svm_rbf_window_balanced", "kind": "svm_rbf",
                          "params": {"C": C, "gamma": gamma},
                          "patient_balanced": False, "class_balanced": True, "seeds": (42,)})
            specs.append({"name": "svm_rbf_patient_balanced", "kind": "svm_rbf",
                          "params": {"C": C, "gamma": gamma},
                          "patient_balanced": True, "class_balanced": False, "seeds": (42,)})
    for architecture in MLP_ARCHITECTURES:
        specs.append({"name": "mlp_relu_window_unweighted", "kind": "mlp_relu",
                      "params": {"hidden_layer_sizes": architecture},
                      "patient_balanced": False, "class_balanced": False, "seeds": MLP_SEEDS})
        specs.append({"name": "mlp_relu_patient_balanced", "kind": "mlp_relu",
                      "params": {"hidden_layer_sizes": architecture},
                      "patient_balanced": True, "class_balanced": False, "seeds": MLP_SEEDS})
    specs.append({"name": "mlp_logistic_legacy", "kind": "mlp_logistic_legacy",
                  "params": {"hidden_layer_sizes": (16, 8)},
                  "patient_balanced": False, "class_balanced": False, "seeds": (42,)})
    return specs


def evaluate_config_inner(
    spec: dict[str, Any],
    inner_data: list[dict[str, Any]],
) -> tuple[float, list[float]]:
    scores = []
    for split in inner_data:
        seed_scores = []
        for model_seed in spec["seeds"]:
            p_rows = fit_predict(
                spec["kind"], spec["params"], split["x_train"], split["y_train"],
                split["g_train"], split["x_valid"], int(model_seed),
                spec["patient_balanced"], spec["class_balanced"],
            )
            groups = group_probabilities(split["y_valid"], split["g_valid"], p_rows)
            seed_scores.append(balanced_accuracy_score(
                groups["label"], (groups["probability"] >= 0.5).astype(int)
            ))
        scores.append(float(np.mean(seed_scores)))
    return float(np.mean(scores)), scores


def metric_row(y: np.ndarray, p: np.ndarray, name: str, repeat: int, fold: int | None) -> dict[str, Any]:
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "model": name, "repeat": repeat, "fold": fold, "n_subjects": int(len(y)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "accuracy": float(accuracy_score(y, pred)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "roc_auc": safe_auc(y, p),
        "pr_auc": float(average_precision_score(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
        "threshold": 0.5,
    }


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False)
    temp.replace(path)


def checkpoint_path(root: Path, feature: str, repeat: int, fold: int) -> Path:
    return root / "checkpoints" / f"{feature}_r{repeat}_f{fold}.json"


def run_outer_fold(
    feature: str,
    X: np.ndarray,
    feature_names: list[str],
    manifest: pd.DataFrame,
    group_table: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    repeat: int,
    fold: int,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_groups = group_table.iloc[train_idx]["group_id"].to_numpy(dtype=str)
    test_groups = group_table.iloc[test_idx]["group_id"].to_numpy(dtype=str)
    outer_rows = []
    tuning_rows = []
    all_specs = candidate_specs()
    names = sorted({spec["name"] for spec in all_specs})
    outer_train_groups = train_groups
    inner_group_table = group_table[
        group_table["group_id"].isin(set(outer_train_groups))
    ].reset_index(drop=True)
    inner_splits = make_group_splits(
        inner_group_table, args.inner_folds,
        args.seed + repeat * 101 + fold * 13, 1,
    )
    inner_data = []
    for _, _, tr_idx, va_idx in inner_splits:
        tr_groups = inner_group_table.iloc[tr_idx]["group_id"].to_numpy(dtype=str)
        va_groups = inner_group_table.iloc[va_idx]["group_id"].to_numpy(dtype=str)
        if not set(tr_groups).issubset(set(outer_train_groups)) or not set(va_groups).issubset(set(outer_train_groups)):
            raise RuntimeError("Inner split escaped the outer-training subjects.")
        xtr, xva, _, _, _ = prepare_fold(
            X, manifest, tr_groups, va_groups, feature_names,
            args.top_k, args.corr_threshold,
        )
        tr_rows = row_indices(manifest, tr_groups)
        va_rows = row_indices(manifest, va_groups)
        inner_data.append({
            "x_train": xtr,
            "x_valid": xva,
            "y_train": manifest.iloc[tr_rows]["label"].to_numpy(dtype=int),
            "g_train": manifest.iloc[tr_rows]["group_id"].to_numpy(dtype=str),
            "y_valid": manifest.iloc[va_rows]["label"].to_numpy(dtype=int),
            "g_valid": manifest.iloc[va_rows]["group_id"].to_numpy(dtype=str),
        })
    for model_name in names:
        specs = [s for s in all_specs if s["name"] == model_name]
        scored = []
        for spec in specs:
            score, per_fold = evaluate_config_inner(
                spec, inner_data,
            ) if spec["kind"] != "mlp_logistic_legacy" else (0.0, [])
            if per_fold:
                scored.append((score, spec))
                tuning_rows.append({
                    "feature": feature, "model": model_name, "repeat": repeat, "outer_fold": fold,
                    "params": json.dumps(spec["params"], sort_keys=True),
                    "inner_mean_ba": score,
                    "inner_fold_ba": json.dumps(per_fold),
                    "selection": "candidate",
                })
        selected = max(scored, key=lambda item: item[0])[1] if scored else specs[0]
        if scored:
            for row in reversed(tuning_rows):
                if (row["feature"], row["model"], row["repeat"], row["outer_fold"]) == (
                    feature, model_name, repeat, fold
                ):
                    row["selection"] = (
                        "selected" if row["params"] == json.dumps(selected["params"], sort_keys=True)
                        else "candidate"
                    )
        xtr, xte, _, _, selected_features = prepare_fold(
            X, manifest, train_groups, test_groups, feature_names,
            args.top_k, args.corr_threshold,
        )
        tr_rows = row_indices(manifest, train_groups)
        te_rows = row_indices(manifest, test_groups)
        ytr = manifest.iloc[tr_rows]["label"].to_numpy(dtype=int)
        gtr = manifest.iloc[tr_rows]["group_id"].to_numpy(dtype=str)
        yte = manifest.iloc[te_rows]["label"].to_numpy(dtype=int)
        gte = manifest.iloc[te_rows]["group_id"].to_numpy(dtype=str)
        seed_probs = []
        for model_seed in selected["seeds"]:
            seed_probs.append(fit_predict(
                selected["kind"], selected["params"], xtr, ytr, gtr, xte,
                int(model_seed), selected["patient_balanced"], selected["class_balanced"],
            ))
        mean_probs = np.mean(seed_probs, axis=0)
        group_pred = group_probabilities(yte, gte, mean_probs)
        for row in group_pred.to_dict(orient="records"):
            outer_rows.append({
                "feature": feature, "model": model_name, "repeat": repeat, "fold": fold,
                "group_id": row["group_id"], "label": int(row["label"]),
                "probability": float(row["probability"]),
                "threshold": 0.5, "prediction": int(row["probability"] >= 0.5),
                "selected_params": json.dumps(selected["params"], sort_keys=True),
                "seed_predictions": json.dumps([
                    float(group_probabilities(yte, gte, p)["probability"].to_numpy()[
                        list(group_pred["group_id"]).index(row["group_id"])
                    ]) for p in seed_probs
                ]),
                "n_selected_features": len(selected_features),
                "selected_features": json.dumps(selected_features),
            })
    return outer_rows, tuning_rows


def summarize_predictions(predictions: pd.DataFrame, out_dir: Path) -> None:
    fold_metrics = []
    repeat_metrics = []
    for (feature, model, repeat, fold), block in predictions.groupby(
        ["feature", "model", "repeat", "fold"], sort=True
    ):
        row = metric_row(block["label"].to_numpy(int), block["probability"].to_numpy(float),
                         model, int(repeat), int(fold))
        row["feature"] = feature
        fold_metrics.append(row)
    for (feature, model, repeat), block in predictions.groupby(
        ["feature", "model", "repeat"], sort=True
    ):
        row = metric_row(block["label"].to_numpy(int), block["probability"].to_numpy(float),
                         model, int(repeat), None)
        row["feature"] = feature
        repeat_metrics.append(row)
    fold_frame = pd.DataFrame(fold_metrics)
    repeat_frame = pd.DataFrame(repeat_metrics)
    metrics = [
        "balanced_accuracy", "accuracy", "mcc", "sensitivity", "specificity",
        "roc_auc", "pr_auc", "brier_score",
    ]
    agg_rows = []
    for (feature, model), block in repeat_frame.groupby(["feature", "model"], sort=True):
        row: dict[str, Any] = {"feature": feature, "model": model, "n_repeats": int(len(block))}
        for metric in metrics:
            row[f"{metric}_mean"] = float(block[metric].mean())
            row[f"{metric}_sd"] = float(block[metric].std(ddof=1)) if len(block) > 1 else float("nan")
        agg_rows.append(row)
    atomic_csv(fold_frame, out_dir / "outer_fold_metrics.csv")
    atomic_csv(repeat_frame, out_dir / "outer_repeat_metrics.csv")
    atomic_csv(pd.DataFrame(agg_rows), out_dir / "aggregate_metrics.csv")

    cal_rows = []
    for (feature, model, repeat), block in predictions.groupby(
        ["feature", "model", "repeat"], sort=True
    ):
        bins = min(5, max(2, block["probability"].nunique()))
        try:
            bin_id = pd.qcut(block["probability"], q=bins, labels=False, duplicates="drop")
        except ValueError:
            bin_id = pd.Series(np.zeros(len(block), dtype=int), index=block.index)
        tmp = block.assign(calibration_bin=np.asarray(bin_id))
        for bin_no, b in tmp.groupby("calibration_bin", sort=True):
            cal_rows.append({
                "feature": feature, "model": model, "repeat": repeat,
                "bin": int(bin_no), "n_subjects": int(len(b)),
                "mean_predicted_probability": float(b["probability"].mean()),
                "observed_positive_fraction": float(b["label"].mean()),
            })
    atomic_csv(pd.DataFrame(cal_rows), out_dir / "calibration_bins.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-dir", type=Path,
                    default=Path("svm-mlp-comparison/features_win"))
    ap.add_argument("--out-dir", type=Path, default=Path("results_r1_nested"))
    ap.add_argument("--outer-folds", type=int, default=5)
    ap.add_argument("--outer-repeats", type=int, default=3)
    ap.add_argument("--inner-folds", type=int, default=4)
    ap.add_argument("--top-k", type=int, default=16)
    ap.add_argument("--corr-threshold", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=20261012)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "checkpoints").mkdir(exist_ok=True)
    manifest, group_table = load_manifest(args.feature_dir / "manifest.csv")
    if len(group_table) < args.outer_folds:
        raise ValueError("Outer folds exceed number of subjects.")
    if int(group_table["label"].value_counts().min()) < args.outer_folds:
        raise ValueError("Each class must have at least outer_folds subjects.")
    outer_splits = make_group_splits(
        group_table, args.outer_folds, args.seed, args.outer_repeats
    )
    split_manifest = []
    for repeat, fold, tr, te in outer_splits:
        split_manifest.extend(
            {"repeat": repeat, "fold": fold, "group_id": group_table.iloc[i]["group_id"],
             "partition": "train"} for i in tr
        )
        split_manifest.extend(
            {"repeat": repeat, "fold": fold, "group_id": group_table.iloc[i]["group_id"],
             "partition": "test"} for i in te
        )
    atomic_csv(pd.DataFrame(split_manifest), args.out_dir / "outer_split_manifest.csv")
    config = {
        "arguments": {
            k: str(v) if isinstance(v, Path) else v
            for k, v in vars(args).items() if k != "resume"
        },
        "feature_files": FEATURE_FILES,
        "mlp_seeds": MLP_SEEDS,
        "mlp_architectures": MLP_ARCHITECTURES,
        "n_rows": len(manifest), "n_subjects": len(group_table),
        "class_subjects": {str(k): int(v) for k, v in group_table["label"].value_counts().to_dict().items()},
        "python": sys.version, "platform": platform.platform(), "sklearn": sklearn.__version__,
    }
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    config["config_hash"] = config_hash
    (args.out_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    all_predictions: list[dict[str, Any]] = []
    all_tuning: list[dict[str, Any]] = []
    completed: set[tuple[str, int, int]] = set()
    if args.resume:
        for ckpt in sorted((args.out_dir / "checkpoints").glob("*_r*_f*.json")):
            stored = json.loads(ckpt.read_text(encoding="utf-8"))
            if stored.get("config_hash") != config_hash:
                raise RuntimeError(f"Checkpoint configuration mismatch: {ckpt}")
            completed.add((
                stored["feature"], int(stored["repeat"]), int(stored["fold"])
            ))
            all_predictions.extend(stored["predictions"])
            all_tuning.extend(stored["tuning"])
        if all_predictions:
            atomic_csv(pd.DataFrame(all_predictions), args.out_dir / "outer_oof_predictions.csv")
            atomic_csv(pd.DataFrame(all_tuning), args.out_dir / "inner_model_selection.csv")
            summarize_predictions(pd.DataFrame(all_predictions), args.out_dir)
    started = time.time()
    with threadpool_limits(limits=1):
        for feature, filename in FEATURE_FILES.items():
            feature_path = args.feature_dir / filename
            X, feature_names = load_feature_matrix(feature_path, manifest)
            log(f"Loaded {feature}: {X.shape[0]} windows x {X.shape[1]} features.")
            for repeat, fold, train_idx, test_idx in outer_splits:
                ckpt = checkpoint_path(args.out_dir, feature, repeat, fold)
                if args.resume and (feature, repeat, fold) in completed:
                    log(f"Resume: already complete {feature}, repeat {repeat}, fold {fold}.")
                    continue
                tic = time.time()
                preds, tuning = run_outer_fold(
                    feature, X, feature_names, manifest, group_table,
                    train_idx, test_idx, repeat, fold, args,
                )
                checkpoint = {
                    "config_hash": config_hash,
                    "feature": feature, "repeat": repeat, "fold": fold,
                    "predictions": preds, "tuning": tuning,
                }
                temp = ckpt.with_suffix(".json.tmp")
                temp.write_text(json.dumps(checkpoint), encoding="utf-8")
                temp.replace(ckpt)
                all_predictions.extend(preds)
                all_tuning.extend(tuning)
                completed.add((feature, repeat, fold))
                atomic_csv(pd.DataFrame(all_predictions), args.out_dir / "outer_oof_predictions.csv")
                atomic_csv(pd.DataFrame(all_tuning), args.out_dir / "inner_model_selection.csv")
                summarize_predictions(pd.DataFrame(all_predictions), args.out_dir)
                log(f"Saved {feature}, repeat {repeat}, fold {fold}: "
                    f"{len(preds)} subject predictions, {len(tuning)} inner candidates "
                    f"in {time.time() - tic:.1f}s.")
    metadata = {
        **config, "elapsed_seconds": time.time() - started,
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": "Brier scores and calibration bins assess uncalibrated out-of-fold probabilities; no outer outcomes are used to fit a calibration transform.",
    }
    (args.out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    log(f"All nested experiments complete in {metadata['elapsed_seconds'] / 3600:.2f} h.")


if __name__ == "__main__":
    main()
