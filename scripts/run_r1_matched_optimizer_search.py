#!/usr/bin/env python3
"""Nested, matched-budget optimizer comparison for MLP architecture search.

Each search method receives the same number of objective calls and uses the
same inner subject-group splits, feature-selection rule, training weights, and
MLP initialization seed within an outer split. The selected architecture is
then evaluated on that outer split's held-out subjects.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

# When this file is launched by path, Python puts ``scripts/`` rather than the
# repository root on sys.path. The optimizer implementations live at the root.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for import_root in (REPOSITORY_ROOT, REPOSITORY_ROOT / "scripts"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from run_experiments import (
    arch_tuple,
    bits_to_arch,
    bpso,
    bde,
    bsade,
    bwoa,
    random_search,
)
from run_r1_nested_experiments import (
    FEATURE_FILES,
    fit_predict,
    group_probabilities,
    load_feature_matrix,
    load_manifest,
    make_group_splits,
    prepare_fold,
    row_indices,
)

METHODS = ("random", "optuna", "bpso", "bwoa", "bde", "bsade")


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False)
    temp.replace(path)


def evaluate_architecture(bits: np.ndarray, inner_data: list[dict[str, Any]], seed: int) -> float:
    n1, n2, n3 = bits_to_arch(bits)
    hidden = arch_tuple(n1, n2, n3)
    kind = "mlp_arch_logistic" if hidden else "logreg"
    params = {"hidden_layer_sizes": hidden} if hidden else {"C": 1.0}
    fold_scores = []
    for split in inner_data:
        probabilities = fit_predict(
            kind, params,
            split["x_train"], split["y_train"], split["g_train"],
            split["x_valid"], seed,
            patient_balanced=True,
            class_balanced=False,
        )
        subject = group_probabilities(
            split["y_valid"], split["g_valid"], probabilities
        )
        prediction = (subject["probability"].to_numpy() >= 0.5).astype(int)
        truth = subject["label"].to_numpy(dtype=int)
        from sklearn.metrics import balanced_accuracy_score
        fold_scores.append(float(balanced_accuracy_score(truth, prediction)))
    return float(np.mean(fold_scores))


def run_optimizer(method: str, objective, budget: int, seed: int):
    rng = np.random.default_rng(seed)
    if method == "random":
        return random_search(objective, dim=15, n_eval=budget, rng=rng)
    if method == "optuna":
        import optuna

        def trial_objective(trial):
            values = [trial.suggest_int(f"n{i}", 0, 31) for i in range(1, 4)]
            bits = np.concatenate([
                np.array([(value >> shift) & 1 for shift in range(4, -1, -1)],
                         dtype=np.int8)
                for value in values
            ])
            return float(objective(bits))

        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
        )
        study.optimize(trial_objective, n_trials=budget, show_progress_bar=False)
        best_values = [study.best_params[f"n{i}"] for i in range(1, 4)]
        best_bits = np.concatenate([
            np.array([(value >> shift) & 1 for shift in range(4, -1, -1)],
                     dtype=np.int8)
            for value in best_values
        ])
        history = [float(trial.value) for trial in study.trials]
        return SimpleNamespace(
            best_bits=best_bits, best_score=float(study.best_value), history=history
        )
    population = 20
    if budget < population or budget % population != 0:
        raise ValueError("Binary optimizer budget must be a positive multiple of 20.")
    iterations = budget // population - 1
    if method == "bpso":
        return bpso(objective, 15, population, iterations, rng, transfer="s")
    if method == "bwoa":
        return bwoa(objective, 15, population, iterations, rng, transfer="s")
    if method == "bde":
        return bde(objective, 15, population, iterations, rng, transfer="s")
    if method == "bsade":
        return bsade(objective, 15, population, iterations, rng, transfer="s")
    raise ValueError(f"Unknown optimizer: {method}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", type=Path,
                        default=Path("svm-mlp-comparison/features_win"))
    parser.add_argument("--out-dir", type=Path,
                        default=Path("results_r1_nested/optimizer_search"))
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--outer-repeats", type=int, default=3)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--budget", type=int, default=200,
                        help="Equal number of optimizer objective calls per method.")
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--corr-threshold", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.budget < 20 or args.budget % 20:
        raise ValueError("--budget must be a multiple of 20 and at least 20.")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.out_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    manifest, group_table = load_manifest(args.feature_dir / "manifest.csv")
    outer_splits = make_group_splits(
        group_table, args.outer_folds, args.seed, args.outer_repeats
    )
    completed: set[tuple[str, int, int]] = set()
    optimizer_records: list[dict[str, Any]] = []
    prediction_records: list[dict[str, Any]] = []
    split_path = args.out_dir / "outer_split_manifest.csv"
    nested_split_path = Path("results_r1_nested/outer_split_manifest.csv")
    if nested_split_path.exists():
        split_reference = pd.read_csv(nested_split_path)
        split_reference["group_id"] = split_reference["group_id"].astype(str)
        expected = []
        for repeat, fold, tr, te in outer_splits:
            expected.extend(
                {"repeat": repeat, "fold": fold,
                 "group_id": str(group_table.iloc[i]["group_id"]), "partition": "train"}
                for i in tr
            )
            expected.extend(
                {"repeat": repeat, "fold": fold,
                 "group_id": str(group_table.iloc[i]["group_id"]), "partition": "test"}
                for i in te
            )
        expected_frame = pd.DataFrame(expected).sort_values(
            ["repeat", "fold", "group_id", "partition"]
        ).reset_index(drop=True)
        reference_frame = split_reference.sort_values(
            ["repeat", "fold", "group_id", "partition"]
        ).reset_index(drop=True)
        if not expected_frame.equals(reference_frame):
            raise RuntimeError("Optimizer run outer splits differ from the nested-model run.")
    else:
        split_records = []
        for repeat, fold, tr, te in outer_splits:
            for i in tr:
                split_records.append({
                    "repeat": repeat, "fold": fold,
                    "group_id": str(group_table.iloc[i]["group_id"]), "partition": "train",
                })
            for i in te:
                split_records.append({
                    "repeat": repeat, "fold": fold,
                    "group_id": str(group_table.iloc[i]["group_id"]), "partition": "test",
                })
        atomic_csv(pd.DataFrame(split_records), split_path)

    settings = {
        "feature_dir": str(args.feature_dir), "out_dir": str(args.out_dir),
        "outer_folds": args.outer_folds, "outer_repeats": args.outer_repeats,
        "inner_folds": args.inner_folds, "objective_calls_per_method": args.budget,
        "top_k": args.top_k, "corr_threshold": args.corr_threshold,
        "seed": args.seed, "methods": METHODS,
        "candidate_space": "15 bits, three 5-bit hidden widths in [0,31]; zero widths omitted",
        "training": "logistic activation; early stopping; subject-and-class-balanced sample weights",
        "objective": "mean inner-fold subject-level balanced accuracy at fixed threshold 0.5",
    }
    config_hash = hashlib.sha256(
        json.dumps(settings, sort_keys=True).encode("utf-8")
    ).hexdigest()
    settings["config_hash"] = config_hash
    (args.out_dir / "run_config.json").write_text(
        json.dumps(settings, indent=2), encoding="utf-8"
    )

    if args.resume:
        for checkpoint in sorted(checkpoint_dir.glob("*_r*_f*.json")):
            item = json.loads(checkpoint.read_text(encoding="utf-8"))
            if item.get("config_hash") != config_hash:
                raise RuntimeError(f"Checkpoint configuration mismatch: {checkpoint}")
            completed.add((item["feature"], int(item["repeat"]), int(item["fold"])))
            optimizer_records.extend(item["optimizers"])
            prediction_records.extend(item["predictions"])
        if optimizer_records:
            atomic_csv(pd.DataFrame(optimizer_records), args.out_dir / "optimizer_search_results.csv")
            atomic_csv(pd.DataFrame(prediction_records), args.out_dir / "optimizer_outer_predictions.csv")

    for feature, filename in FEATURE_FILES.items():
        X, feature_names = load_feature_matrix(args.feature_dir / filename, manifest)
        log(f"Loaded {feature}, shape={X.shape}.")
        for repeat, fold, tr_idx, te_idx in outer_splits:
            key = (feature, repeat, fold)
            if args.resume and key in completed:
                log(f"Resume: already complete {feature}, repeat {repeat}, fold {fold}.")
                continue
            tic = time.time()
            train_groups = group_table.iloc[tr_idx]["group_id"].to_numpy(dtype=str)
            test_groups = group_table.iloc[te_idx]["group_id"].to_numpy(dtype=str)
            inner_table = group_table.iloc[tr_idx].reset_index(drop=True)
            inner_splits = make_group_splits(
                inner_table, args.inner_folds,
                args.seed + repeat * 101 + fold * 13, 1,
            )
            inner_data = []
            for _, _, itr, iva in inner_splits:
                gtr = inner_table.iloc[itr]["group_id"].to_numpy(dtype=str)
                gva = inner_table.iloc[iva]["group_id"].to_numpy(dtype=str)
                xtr, xva, _, _, _ = prepare_fold(
                    X, manifest, gtr, gva, feature_names,
                    args.top_k, args.corr_threshold,
                )
                rtr, rva = row_indices(manifest, gtr), row_indices(manifest, gva)
                inner_data.append({
                    "x_train": xtr, "x_valid": xva,
                    "y_train": manifest.iloc[rtr]["label"].to_numpy(dtype=int),
                    "g_train": manifest.iloc[rtr]["group_id"].to_numpy(dtype=str),
                    "y_valid": manifest.iloc[rva]["label"].to_numpy(dtype=int),
                    "g_valid": manifest.iloc[rva]["group_id"].to_numpy(dtype=str),
                })
            xtr_outer, xte_outer, _, _, selected_features = prepare_fold(
                X, manifest, train_groups, test_groups, feature_names,
                args.top_k, args.corr_threshold,
            )
            rtr = row_indices(manifest, train_groups)
            rte = row_indices(manifest, test_groups)
            ytr = manifest.iloc[rtr]["label"].to_numpy(dtype=int)
            gtr = manifest.iloc[rtr]["group_id"].to_numpy(dtype=str)
            yte = manifest.iloc[rte]["label"].to_numpy(dtype=int)
            gte = manifest.iloc[rte]["group_id"].to_numpy(dtype=str)
            model_seed = args.seed + repeat * 1000 + fold * 31
            current_optimizers = []
            current_predictions = []
            for method in METHODS:
                method_started = time.time()
                objective_scores: list[float] = []
                architectures: set[tuple[int, ...]] = set()

                def objective(bits: np.ndarray) -> float:
                    architecture = arch_tuple(*bits_to_arch(bits))
                    architectures.add(architecture)
                    value = evaluate_architecture(bits, inner_data, model_seed)
                    objective_scores.append(value)
                    return value

                method_seed = args.seed + repeat * 1000 + fold * 31
                result = run_optimizer(method, objective, args.budget, method_seed)
                if result.best_bits is None:
                    raise RuntimeError(f"{method} did not return an architecture.")
                best_arch = arch_tuple(*bits_to_arch(result.best_bits))
                best_kind = "mlp_arch_logistic" if best_arch else "logreg"
                best_params = (
                    {"hidden_layer_sizes": best_arch} if best_arch else {"C": 1.0}
                )
                probability_rows = fit_predict(
                    best_kind, best_params, xtr_outer, ytr, gtr, xte_outer,
                    model_seed, patient_balanced=True, class_balanced=False,
                )
                subject = group_probabilities(yte, gte, probability_rows)
                for item in subject.to_dict(orient="records"):
                    current_predictions.append({
                        "feature": feature, "method": method, "repeat": repeat,
                        "fold": fold, "group_id": item["group_id"],
                        "label": int(item["label"]),
                        "probability": float(item["probability"]),
                        "prediction": int(item["probability"] >= 0.5),
                        "threshold": 0.5,
                        "hidden_layer_sizes": json.dumps(best_arch),
                        "best_inner_ba": float(result.best_score),
                    })
                current_optimizers.append({
                    "feature": feature, "method": method, "repeat": repeat,
                    "fold": fold, "objective_calls": len(objective_scores),
                    "unique_architectures": len(architectures),
                    "duplicate_calls": len(objective_scores) - len(architectures),
                    "best_inner_ba": float(result.best_score),
                    "best_bits": json.dumps(result.best_bits.astype(int).tolist()),
                    "hidden_layer_sizes": json.dumps(best_arch),
                    "history": json.dumps([float(x) for x in result.history]),
                    "objective_values": json.dumps(objective_scores),
                    "search_seconds": time.time() - method_started,
                    "n_selected_features": len(selected_features),
                    "selected_features": json.dumps(selected_features),
                })
            elapsed = time.time() - tic
            checkpoint = {
                "config_hash": config_hash,
                "feature": feature, "repeat": repeat, "fold": fold,
                "budget": args.budget, "optimizers": current_optimizers,
                "predictions": current_predictions,
            }
            target = checkpoint_dir / f"{feature}_r{repeat}_f{fold}.json"
            temp = target.with_suffix(".json.tmp")
            temp.write_text(json.dumps(checkpoint), encoding="utf-8")
            temp.replace(target)
            completed.add(key)
            optimizer_records.extend(current_optimizers)
            prediction_records.extend(current_predictions)
            atomic_csv(pd.DataFrame(optimizer_records), args.out_dir / "optimizer_search_results.csv")
            atomic_csv(pd.DataFrame(prediction_records), args.out_dir / "optimizer_outer_predictions.csv")
            log(f"Saved {feature}, repeat {repeat}, fold {fold}: "
                f"{len(current_optimizers)} methods x {args.budget} calls "
                f"in {elapsed / 60:.1f} min.")

    if prediction_records:
        predictions = pd.DataFrame(prediction_records)
        metric_rows = []
        for (feature, method, repeat, fold), block in predictions.groupby(
            ["feature", "method", "repeat", "fold"], sort=True
        ):
            y = block["label"].to_numpy(dtype=int)
            p = block["probability"].to_numpy(dtype=float)
            from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, roc_auc_score
            metric_rows.append({
                "feature": feature, "method": method, "repeat": repeat, "fold": fold,
                "n_subjects": len(block),
                "balanced_accuracy": float(balanced_accuracy_score(y, p >= 0.5)),
                "mcc": float(matthews_corrcoef(y, p >= 0.5)),
                "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan"),
            })
        atomic_csv(pd.DataFrame(metric_rows), args.out_dir / "optimizer_outer_fold_metrics.csv")
        repeat_rows = []
        for (feature, method, repeat), block in predictions.groupby(
            ["feature", "method", "repeat"], sort=True
        ):
            y = block["label"].to_numpy(dtype=int)
            p = block["probability"].to_numpy(dtype=float)
            from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, roc_auc_score
            repeat_rows.append({
                "feature": feature, "method": method, "repeat": repeat,
                "n_subjects": len(block),
                "balanced_accuracy": float(balanced_accuracy_score(y, p >= 0.5)),
                "mcc": float(matthews_corrcoef(y, p >= 0.5)),
                "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan"),
            })
        repeat_frame = pd.DataFrame(repeat_rows)
        atomic_csv(repeat_frame, args.out_dir / "optimizer_outer_repeat_metrics.csv")
        aggregate = repeat_frame.groupby(["feature", "method"], as_index=False).agg(
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_sd=("balanced_accuracy", "std"),
            mcc_mean=("mcc", "mean"), mcc_sd=("mcc", "std"),
            roc_auc_mean=("roc_auc", "mean"), roc_auc_sd=("roc_auc", "std"),
        )
        atomic_csv(aggregate, args.out_dir / "optimizer_aggregate_metrics.csv")
    log("Matched-budget optimizer experiments complete.")


if __name__ == "__main__":
    main()
