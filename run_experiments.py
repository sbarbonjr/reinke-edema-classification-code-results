#!/usr/bin/env python3
"""
run_experiments.py

End-to-end experiments for windowed (segmented) audio feature CSVs with leakage control
via group-aware CV (group_id = original WAV stem).

What it does
- Loads a feature CSV produced by prepare_dataset.py (rows = segments).
- Runs baselines:
    * LogisticRegression (as "0 hidden layer" baseline)
    * SVM (linear)
    * SVM (RBF) (fixed params unless you also run optuna on svm_rbf)
    * MLP with a fixed default architecture
- Runs optimization:
    * MLP architecture optimization (paper-aligned encoding: 15 bits -> 3x5 bits -> 0..31 neurons, up to 3 layers):
        - random search
        - BPSO / WOA / DE / SADE (binary with S/V transfer)
        - optuna (TPE)
    * Optional: SVM-RBF optimization with optuna (continuous C/gamma)
- Leakage control:
    * StratifiedGroupKFold when available (sklearn >= 1.1), otherwise GroupKFold
- Metrics:
    * segment-level
    * file-level metrics (group_id aggregation) by mean-proba and majority vote

IMPORTANT
- Any feature selection must be performed INSIDE CV to avoid leakage.
  This script supports a paper-like correlation filter (redundancy threshold + top-k).
  It is applied train-only within each fold.

Batch mode
- With --run_opt --opt_all, you can run ALL optimization methods in a single call and
  save per-method JSON plus a summary_best_<csvstem>.json with the best overall.

Example (paper-like):
python3 run_experiments.py \
  --csv features_wpd_paper/wpd_haar7_std.csv \
  --out_dir results \
  --cv_splits 5 --seed 42 \
  --feature_select corr --corr_thresh 0.85 --top_k 16 \
  --run_baselines \
  --run_opt --opt_all \
  --opt_targets "mlp_arch" \
  --opt_methods "random,bpso,bwoa,bde,bsade,optuna" \
  --opt_metric acc --opt_level group_mean \
  --n_eval 200 --n_pop 20 --n_iter 30 --transfer s

Also optimize SVM-RBF with optuna in the same run:
  --opt_targets "mlp_arch,svm_rbf"
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAVE_SGK = True
except Exception:
    HAVE_SGK = False

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)


META_COLS = {
    "file", "filename", "group_id", "segment_id", "start_sample", "start_sec",
    "label", "class_name", "sr", "segment_sec", "target_sec"
}


# ============================
# CV splitter (group-aware)
# ============================

def make_splitter(
    n_splits: int,
    y: np.ndarray,
    groups: np.ndarray,
    seed: int,
    n_repeats: int = 1,
):
    """
    Yield tuples: (repeat_idx, fold_idx, tr_idx, te_idx).
    """
    if n_repeats < 1:
        raise ValueError("n_repeats must be >= 1")

    if n_repeats > 1 and not HAVE_SGK:
        raise ValueError(
            "Repeated group CV requires StratifiedGroupKFold (sklearn >= 1.1). "
            "Upgrade sklearn or set --cv_repeats 1."
        )

    for rep in range(n_repeats):
        if HAVE_SGK:
            splitter = StratifiedGroupKFold(
                n_splits=n_splits, shuffle=True, random_state=seed + rep
            ).split(np.zeros_like(y), y, groups)
        else:
            splitter = GroupKFold(n_splits=n_splits).split(np.zeros_like(y), y, groups)

        for fold, (tr, te) in enumerate(splitter, start=1):
            yield rep + 1, fold, tr, te


# ============================
# Feature selection (train-only)
# ============================

def select_features_train_only_corr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: List[str],
    corr_thresh: float = 0.85,
    top_k: int = 16
) -> Tuple[np.ndarray, List[str]]:
    """
    Paper-inspired:
    1) Remove redundancy: if abs(corr(fi,fj)) > corr_thresh, keep feature with higher abs(corr(fi,y)).
    2) Rank remaining by abs(corr(fi,y)) and keep top_k.
    """
    X = X_train.astype(np.float64, copy=False)
    y = y_train.astype(np.float64, copy=False)

    # corr(feature, y)
    fy = np.zeros(X.shape[1], dtype=np.float64)
    for j in range(X.shape[1]):
        xj = X[:, j]
        if np.std(xj) < 1e-12:
            fy[j] = 0.0
        else:
            c = np.corrcoef(xj, y)[0, 1]
            fy[j] = 0.0 if np.isnan(c) else float(c)

    # feature-feature correlation
    C = np.corrcoef(X, rowvar=False)
    C = np.nan_to_num(C, nan=0.0)

    keep = np.ones(X.shape[1], dtype=bool)
    for i in range(X.shape[1]):
        if not keep[i]:
            continue
        for j in range(i + 1, X.shape[1]):
            if not keep[j]:
                continue
            if abs(C[i, j]) > corr_thresh:
                if abs(fy[i]) >= abs(fy[j]):
                    keep[j] = False
                else:
                    keep[i] = False
                    break

    idx = np.where(keep)[0]
    ranked = idx[np.argsort(np.abs(fy[idx]))[::-1]]
    selected = ranked[: min(top_k, len(ranked))]
    sel_names = [feature_names[k] for k in selected]
    return X_train[:, selected], sel_names


def transform_with_selected(X: np.ndarray, feature_names: List[str], selected_names: List[str]) -> np.ndarray:
    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    idx = [name_to_idx[n] for n in selected_names]
    return X[:, idx]


# ============================
# Models
# ============================

def make_svm_linear(seed: int) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="linear", probability=True, class_weight="balanced", random_state=seed)),
    ])


def make_svm_rbf(C: float, gamma: float, seed: int) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=C, gamma=gamma, probability=True, class_weight="balanced", random_state=seed)),
    ])


def make_logreg(seed: int, C: float = 1.0) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=C,
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=seed
        )),
    ])


def make_mlp(hidden_layers: Tuple[int, ...], seed: int) -> Pipeline:
    """
    Paper-like fixed hyperparams:
      activation=logistic, solver=adam, lr=1e-3, max_iter=500
    """
    if len(hidden_layers) == 0:
        return make_logreg(seed)

    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation="logistic",
            solver="adam",
            learning_rate_init=1e-3,
            max_iter=500,
            early_stopping=True,
            n_iter_no_change=20,
            random_state=seed,
        )),
    ])


# ============================
# Metrics + aggregation
# ============================

def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    if len(np.unique(y_true)) < 2:
        return None
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return None


def aggregate_to_group(
    y_true: np.ndarray,
    y_prob1: np.ndarray,
    groups: np.ndarray,
    mode: str = "mean_proba"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (group_ids, y_true_group, y_prob1_group)
    - mean_proba: average prob per group
    - majority: probability is fraction of segments predicted positive (threshold 0.5 on segment prob)
    """
    df = pd.DataFrame({"g": groups, "y": y_true, "p": y_prob1})
    if mode == "majority":
        df["pred"] = (df["p"] >= 0.5).astype(int)
        agg = df.groupby("g").agg(y=("y", "max"), p=("pred", "mean")).reset_index()
    else:
        agg = df.groupby("g").agg(y=("y", "max"), p=("p", "mean")).reset_index()

    return agg["g"].to_numpy(), agg["y"].to_numpy().astype(int), agg["p"].to_numpy().astype(float)


def compute_report(y_true: np.ndarray, y_prob1: np.ndarray, thr: float = 0.5) -> Dict[str, object]:
    y_pred = (y_prob1 >= thr).astype(int)
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    auc = safe_auc(y_true, y_prob1)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    return {"acc": acc, "f1": f1, "auc": auc, "cm": cm}


def safe_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    if len(np.unique(y_true)) < 2:
        return None
    try:
        return float(average_precision_score(y_true, y_score))
    except Exception:
        return None


def compute_report_extended(y_true: np.ndarray, y_prob1: np.ndarray, thr: float = 0.5) -> Dict[str, object]:
    y_pred = (y_prob1 >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(x) for x in cm.ravel()]
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": safe_auc(y_true, y_prob1),
        "pr_auc": safe_pr_auc(y_true, y_prob1),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def holm_correction(pvals: List[float]) -> List[float]:
    order = np.argsort(np.array(pvals, dtype=float))
    m = len(pvals)
    adj = np.empty(m, dtype=float)
    prev = 0.0
    for rank, idx in enumerate(order):
        p = float(pvals[idx])
        v = min(1.0, (m - rank) * p)
        v = max(prev, v)
        adj[idx] = v
        prev = v
    return adj.tolist()


# ============================
# Objective config
# ============================

@dataclass
class ObjectiveConfig:
    feature_select: str  # none | corr
    corr_thresh: float
    top_k: int
    cv_splits: int
    cv_repeats: int
    seed: int
    opt_metric: str   # acc | f1 | auc
    opt_level: str    # segment | group_mean | group_majority


def objective_from_eval(eval_block: Dict[str, object], cfg: ObjectiveConfig) -> float:
    """
    Extract the scalar objective from evaluation dict.
    AUC can be None if fold has only one class; we treat None as -inf.
    """
    level_map = {
        "segment": "segment",
        "group_mean": "group_mean_proba",
        "group_majority": "group_majority",
    }
    level_key = level_map.get(cfg.opt_level, "segment")
    metric = cfg.opt_metric

    try:
        v = eval_block[level_key][metric]
    except Exception:
        return -1e9

    if v is None:
        return -1e9
    return float(v)


def run_pairwise_wilcoxon_with_holm(
    fold_group_df: pd.DataFrame,
    metric: str = "balanced_accuracy",
    aggregate: str = "group_mean_proba",
) -> pd.DataFrame:
    """
    Paired nonparametric comparison over repeated CV blocks (repeat, fold).
    """
    from scipy.stats import wilcoxon

    if fold_group_df.empty:
        return pd.DataFrame()

    use = fold_group_df.copy()
    use = use[use["aggregate"] == aggregate]
    if use.empty:
        return pd.DataFrame()

    wide = use.pivot_table(
        index=["repeat", "fold"],
        columns="model",
        values=metric,
        aggfunc="mean",
    ).dropna(axis=0, how="any")

    models = list(wide.columns)
    if len(models) < 2 or len(wide) < 2:
        return pd.DataFrame()

    rows = []
    raw_pvals = []

    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            a, b = models[i], models[j]
            diff = wide[a] - wide[b]
            try:
                stat, pval = wilcoxon(diff.to_numpy())
            except ValueError:
                stat, pval = np.nan, 1.0

            rows.append({
                "model_a": a,
                "model_b": b,
                "n_blocks": int(len(diff)),
                "mean_diff_a_minus_b": float(diff.mean()),
                "median_diff_a_minus_b": float(diff.median()),
                "wins_a": int((diff > 0).sum()),
                "wins_b": int((diff < 0).sum()),
                "ties": int((diff == 0).sum()),
                "wilcoxon_stat": float(stat) if pd.notna(stat) else np.nan,
                "p_value": float(pval),
            })
            raw_pvals.append(float(pval))

    out = pd.DataFrame(rows)
    out["p_value_holm"] = holm_correction(raw_pvals)
    out["superior"] = out.apply(
        lambda r: (
            r["model_a"] if (r["p_value_holm"] < 0.05 and r["mean_diff_a_minus_b"] > 0)
            else (r["model_b"] if (r["p_value_holm"] < 0.05 and r["mean_diff_a_minus_b"] < 0)
                  else "no_significant_difference")
        ),
        axis=1,
    )
    return out.sort_values(["p_value_holm", "p_value", "model_a", "model_b"]).reset_index(drop=True)


# ============================
# Architecture encoding (paper)
# 15 bits -> 3x5 bits -> n1,n2,n3 in [0..31]
# ============================

def bits_to_arch(bits: np.ndarray) -> Tuple[int, int, int]:
    bits = bits.astype(int).flatten()
    if bits.size != 15:
        raise ValueError("bits must have length 15")

    def b2i(b):
        val = 0
        for x in b:
            val = (val << 1) | int(x)
        return val

    return b2i(bits[0:5]), b2i(bits[5:10]), b2i(bits[10:15])


def arch_tuple(n1: int, n2: int, n3: int) -> Tuple[int, ...]:
    return tuple([n for n in (n1, n2, n3) if n > 0])


# ============================
# Generic CV evaluator (returns full reports)
# ============================

def evaluate_model_cv(
    model_factory: Callable[[], Pipeline],
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_names: List[str],
    cfg: ObjectiveConfig,
) -> Dict[str, object]:
    eval_block, _, _ = evaluate_model_cv_with_folds(
        model_factory=model_factory,
        X=X,
        y=y,
        groups=groups,
        feature_names=feature_names,
        cfg=cfg,
    )
    return eval_block


def evaluate_model_cv_with_folds(
    model_factory: Callable[[], Pipeline],
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_names: List[str],
    cfg: ObjectiveConfig,
) -> Tuple[Dict[str, object], List[Dict[str, object]], List[Dict[str, object]]]:
    seg_true_all, seg_prob_all, seg_group_all = [], [], []
    fold_rows_segment: List[Dict[str, object]] = []
    fold_rows_group: List[Dict[str, object]] = []

    for rep, fold, tr, te in make_splitter(
        cfg.cv_splits, y, groups, cfg.seed, n_repeats=cfg.cv_repeats
    ):
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]
        gte = groups[te]

        if cfg.feature_select == "corr":
            Xtr_sel, sel_names = select_features_train_only_corr(
                Xtr, ytr, feature_names, corr_thresh=cfg.corr_thresh, top_k=cfg.top_k
            )
            Xte_sel = transform_with_selected(Xte, feature_names, sel_names)
        else:
            Xtr_sel, Xte_sel = Xtr, Xte

        model = model_factory()
        model.fit(Xtr_sel, ytr)

        if hasattr(model, "predict_proba"):
            prob1 = model.predict_proba(Xte_sel)[:, 1]
        else:
            df = model.decision_function(Xte_sel)
            prob1 = 1.0 / (1.0 + np.exp(-df))

        seg_true_all.append(yte)
        seg_prob_all.append(prob1)
        seg_group_all.append(gte)

        seg_fold = compute_report_extended(yte, prob1)
        _, gy_m, gp_m = aggregate_to_group(yte, prob1, gte, mode="mean_proba")
        _, gy_j, gp_j = aggregate_to_group(yte, prob1, gte, mode="majority")
        grp_mean = compute_report_extended(gy_m, gp_m)
        grp_maj = compute_report_extended(gy_j, gp_j)

        fold_rows_segment.append({
            "repeat": int(rep),
            "fold": int(fold),
            "level": "segment",
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "n_groups_test": int(pd.Series(gte).nunique()),
            "pos_train": int(ytr.sum()),
            "pos_test": int(yte.sum()),
            "mcc": seg_fold["mcc"],
            "balanced_accuracy": seg_fold["balanced_accuracy"],
            "tn": seg_fold["tn"],
            "fp": seg_fold["fp"],
            "fn": seg_fold["fn"],
            "tp": seg_fold["tp"],
            "roc_auc": seg_fold["roc_auc"],
            "pr_auc": seg_fold["pr_auc"],
        })
        fold_rows_group.append({
            "repeat": int(rep),
            "fold": int(fold),
            "level": "group",
            "aggregate": "group_mean_proba",
            "n_groups_test": int(pd.Series(gte).nunique()),
            "n_test_segments": int(len(te)),
            "mcc": grp_mean["mcc"],
            "balanced_accuracy": grp_mean["balanced_accuracy"],
            "tn": grp_mean["tn"],
            "fp": grp_mean["fp"],
            "fn": grp_mean["fn"],
            "tp": grp_mean["tp"],
            "roc_auc": grp_mean["roc_auc"],
            "pr_auc": grp_mean["pr_auc"],
        })
        fold_rows_group.append({
            "repeat": int(rep),
            "fold": int(fold),
            "level": "group",
            "aggregate": "group_majority",
            "n_groups_test": int(pd.Series(gte).nunique()),
            "n_test_segments": int(len(te)),
            "mcc": grp_maj["mcc"],
            "balanced_accuracy": grp_maj["balanced_accuracy"],
            "tn": grp_maj["tn"],
            "fp": grp_maj["fp"],
            "fn": grp_maj["fn"],
            "tp": grp_maj["tp"],
            "roc_auc": grp_maj["roc_auc"],
            "pr_auc": grp_maj["pr_auc"],
        })

    seg_true = np.concatenate(seg_true_all)
    seg_prob1 = np.concatenate(seg_prob_all)
    seg_groups = np.concatenate(seg_group_all)

    seg_report = compute_report(seg_true, seg_prob1)

    _, gy, gp_mean = aggregate_to_group(seg_true, seg_prob1, seg_groups, mode="mean_proba")
    _, gy2, gp_maj = aggregate_to_group(seg_true, seg_prob1, seg_groups, mode="majority")

    group_report_mean = compute_report(gy, gp_mean)
    group_report_maj = compute_report(gy2, gp_maj)

    eval_block = {
        "segment": seg_report,
        "group_mean_proba": group_report_mean,
        "group_majority": group_report_maj,
    }
    return eval_block, fold_rows_segment, fold_rows_group


# ============================
# Generic CV objective runner (scalar objective)
# (used inside optimizers)
# ============================

def cv_objective_model(
    model_factory: Callable[[], Pipeline],
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_names: List[str],
    cfg: ObjectiveConfig,
) -> float:
    # We compute predictions across folds and then compute the same evaluation dict,
    # to match the final reported objective (segment or group-level).
    seg_true_all, seg_prob_all, seg_group_all = [], [], []

    for _, _, tr, te in make_splitter(
        cfg.cv_splits, y, groups, cfg.seed, n_repeats=cfg.cv_repeats
    ):
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]
        gte = groups[te]

        if cfg.feature_select == "corr":
            Xtr_sel, sel_names = select_features_train_only_corr(
                Xtr, ytr, feature_names, corr_thresh=cfg.corr_thresh, top_k=cfg.top_k
            )
            Xte_sel = transform_with_selected(Xte, feature_names, sel_names)
        else:
            Xtr_sel, Xte_sel = Xtr, Xte

        model = model_factory()
        model.fit(Xtr_sel, ytr)

        if hasattr(model, "predict_proba"):
            prob1 = model.predict_proba(Xte_sel)[:, 1]
        else:
            df = model.decision_function(Xte_sel)
            prob1 = 1.0 / (1.0 + np.exp(-df))

        seg_true_all.append(yte)
        seg_prob_all.append(prob1)
        seg_group_all.append(gte)

    seg_true = np.concatenate(seg_true_all)
    seg_prob1 = np.concatenate(seg_prob_all)
    seg_groups = np.concatenate(seg_group_all)

    eval_block = {
        "segment": compute_report(seg_true, seg_prob1),
        "group_mean_proba": compute_report(*aggregate_to_group(seg_true, seg_prob1, seg_groups, mode="mean_proba")[1:]),
        "group_majority": compute_report(*aggregate_to_group(seg_true, seg_prob1, seg_groups, mode="majority")[1:]),
    }
    return objective_from_eval(eval_block, cfg)


def cv_objective_mlp_bits(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_names: List[str],
    bits: np.ndarray,
    cfg: ObjectiveConfig,
) -> float:
    n1, n2, n3 = bits_to_arch(bits)
    hl = arch_tuple(n1, n2, n3)
    return cv_objective_model(
        model_factory=lambda hl=hl: make_mlp(hl, seed=cfg.seed),
        X=X, y=y, groups=groups, feature_names=feature_names, cfg=cfg
    )


# ============================
# Binary optimizers (simple & robust)
# ============================

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def s_transfer(v: np.ndarray) -> np.ndarray:
    return sigmoid(v)


def v_transfer(v: np.ndarray) -> np.ndarray:
    return np.abs(np.tanh(v))


def sample_bits_from_prob(p: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return (rng.random(p.shape) < p).astype(np.int8)


@dataclass
class OptResult:
    best_bits: Optional[np.ndarray]
    best_score: float
    history: List[float]
    best_params: Optional[Dict[str, float]] = None  # for continuous opt (SVM)


def random_search(objective: Callable[[np.ndarray], float], dim: int, n_eval: int, rng: np.random.Generator) -> OptResult:
    best_s = -1e9
    best_x = None
    hist = []
    for _ in range(n_eval):
        x = rng.integers(0, 2, size=dim, dtype=np.int8)
        s = float(objective(x))
        hist.append(s)
        if s > best_s:
            best_s = s
            best_x = x.copy()
    return OptResult(best_bits=best_x, best_score=float(best_s), history=hist)


def random_search_params(
    objective: Callable[[Dict[str, float]], float],
    sampler: Callable[[np.random.Generator], Dict[str, float]],
    n_eval: int,
    rng: np.random.Generator,
) -> OptResult:
    best_s = -1e9
    best_p: Optional[Dict[str, float]] = None
    hist: List[float] = []
    for _ in range(n_eval):
        params = sampler(rng)
        s = float(objective(params))
        hist.append(s)
        if s > best_s:
            best_s = s
            best_p = params
    return OptResult(best_bits=None, best_score=float(best_s), history=hist, best_params=best_p)


def bpso(
    objective: Callable[[np.ndarray], float],
    dim: int,
    n_particles: int,
    n_iter: int,
    rng: np.random.Generator,
    w: float = 0.72,
    c1: float = 1.49,
    c2: float = 1.49,
    transfer: str = "s",
) -> OptResult:
    transfer_fn = s_transfer if transfer == "s" else v_transfer

    X = rng.integers(0, 2, size=(n_particles, dim), dtype=np.int8)
    V = rng.normal(0, 1, size=(n_particles, dim)).astype(np.float32)

    pbest = X.copy()
    pbest_s = np.array([objective(x) for x in X], dtype=np.float32)

    g_idx = int(np.argmax(pbest_s))
    gbest = pbest[g_idx].copy()
    gbest_s = float(pbest_s[g_idx])

    hist = [gbest_s]

    for _ in range(n_iter):
        r1 = rng.random((n_particles, dim))
        r2 = rng.random((n_particles, dim))

        V = (w * V
             + c1 * r1 * (pbest - X)
             + c2 * r2 * (gbest - X))

        P = transfer_fn(V)
        X = sample_bits_from_prob(P, rng)

        scores = np.array([objective(x) for x in X], dtype=np.float32)
        improved = scores > pbest_s
        pbest[improved] = X[improved]
        pbest_s[improved] = scores[improved]

        g_idx = int(np.argmax(pbest_s))
        if float(pbest_s[g_idx]) > gbest_s:
            gbest_s = float(pbest_s[g_idx])
            gbest = pbest[g_idx].copy()

        hist.append(gbest_s)

    return OptResult(best_bits=gbest, best_score=gbest_s, history=[float(x) for x in hist])


def bde(
    objective: Callable[[np.ndarray], float],
    dim: int,
    n_pop: int,
    n_iter: int,
    rng: np.random.Generator,
    F: float = 0.5,
    CR: float = 0.9,
    transfer: str = "s",
) -> OptResult:
    transfer_fn = s_transfer if transfer == "s" else v_transfer

    X = rng.integers(0, 2, size=(n_pop, dim), dtype=np.int8)
    scores = np.array([objective(x) for x in X], dtype=np.float32)

    best_idx = int(np.argmax(scores))
    best = X[best_idx].copy()
    best_s = float(scores[best_idx])
    hist = [best_s]

    for _ in range(n_iter):
        for i in range(n_pop):
            idxs = [j for j in range(n_pop) if j != i]
            a, b, c = rng.choice(idxs, size=3, replace=False)

            v = X[a].astype(np.float32) + F * (X[b].astype(np.float32) - X[c].astype(np.float32))
            p = transfer_fn(v)
            mutant = sample_bits_from_prob(p, rng)

            cross_mask = rng.random(dim) < CR
            trial = X[i].copy()
            trial[cross_mask] = mutant[cross_mask]

            s = float(objective(trial))
            if s > float(scores[i]):
                X[i] = trial
                scores[i] = s

        best_idx = int(np.argmax(scores))
        if float(scores[best_idx]) > best_s:
            best_s = float(scores[best_idx])
            best = X[best_idx].copy()

        hist.append(best_s)

    return OptResult(best_bits=best, best_score=best_s, history=[float(x) for x in hist])


def bwoa(
    objective: Callable[[np.ndarray], float],
    dim: int,
    n_pop: int,
    n_iter: int,
    rng: np.random.Generator,
    transfer: str = "s",
) -> OptResult:
    transfer_fn = s_transfer if transfer == "s" else v_transfer

    X = rng.integers(0, 2, size=(n_pop, dim), dtype=np.int8)
    scores = np.array([objective(x) for x in X], dtype=np.float32)

    best_idx = int(np.argmax(scores))
    best = X[best_idx].copy()
    best_s = float(scores[best_idx])
    hist = [best_s]

    for t in range(n_iter):
        a = 2 - 2 * (t / max(1, n_iter - 1))
        for i in range(n_pop):
            r1, r2 = rng.random(), rng.random()
            A = 2 * a * r1 - a
            C = 2 * r2
            p_choice = rng.random()
            l = rng.uniform(-1, 1)

            Xi = X[i].astype(np.float32)
            Xb = best.astype(np.float32)

            if p_choice < 0.5:
                D = np.abs(C * Xb - Xi)
                v = Xb - A * D
            else:
                D = np.abs(Xb - Xi)
                v = D * np.exp(l) * np.cos(2 * np.pi * l) + Xb

            prob = transfer_fn(v)
            Xnew = sample_bits_from_prob(prob, rng)

            s = float(objective(Xnew))
            if s > float(scores[i]):
                X[i] = Xnew
                scores[i] = s

        best_idx = int(np.argmax(scores))
        if float(scores[best_idx]) > best_s:
            best_s = float(scores[best_idx])
            best = X[best_idx].copy()

        hist.append(best_s)

    return OptResult(best_bits=best, best_score=best_s, history=[float(x) for x in hist])


def bsade(
    objective: Callable[[np.ndarray], float],
    dim: int,
    n_pop: int,
    n_iter: int,
    rng: np.random.Generator,
    transfer: str = "s",
) -> OptResult:
    transfer_fn = s_transfer if transfer == "s" else v_transfer

    X = rng.integers(0, 2, size=(n_pop, dim), dtype=np.int8)
    F = rng.uniform(0.3, 0.9, size=n_pop)
    CR = rng.uniform(0.1, 0.9, size=n_pop)
    scores = np.array([objective(x) for x in X], dtype=np.float32)

    best_idx = int(np.argmax(scores))
    best = X[best_idx].copy()
    best_s = float(scores[best_idx])
    hist = [best_s]

    for _ in range(n_iter):
        for i in range(n_pop):
            if rng.random() < 0.1:
                F[i] = rng.uniform(0.3, 0.9)
            if rng.random() < 0.1:
                CR[i] = rng.uniform(0.1, 0.9)

            idxs = [j for j in range(n_pop) if j != i]
            a, b, c = rng.choice(idxs, size=3, replace=False)

            v = X[a].astype(np.float32) + F[i] * (X[b].astype(np.float32) - X[c].astype(np.float32))
            p = transfer_fn(v)
            mutant = sample_bits_from_prob(p, rng)

            cross_mask = rng.random(dim) < CR[i]
            trial = X[i].copy()
            trial[cross_mask] = mutant[cross_mask]

            s = float(objective(trial))
            if s > float(scores[i]):
                X[i] = trial
                scores[i] = s

        best_idx = int(np.argmax(scores))
        if float(scores[best_idx]) > best_s:
            best_s = float(scores[best_idx])
            best = X[best_idx].copy()

        hist.append(best_s)

    return OptResult(best_bits=best, best_score=best_s, history=[float(x) for x in hist])


# ============================
# Optuna searches
# ============================

def optuna_search_mlp(objective_bits: Callable[[np.ndarray], float], n_trials: int, seed: int) -> OptResult:
    import optuna

    def trial_obj(trial: "optuna.Trial") -> float:
        n1 = trial.suggest_int("n1", 0, 31)
        n2 = trial.suggest_int("n2", 0, 31)
        n3 = trial.suggest_int("n3", 0, 31)
        bits = []
        for v in (n1, n2, n3):
            bits.extend([(v >> k) & 1 for k in range(4, -1, -1)])
        return float(objective_bits(np.array(bits, dtype=np.int8)))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(trial_obj, n_trials=n_trials, show_progress_bar=True)

    bp = study.best_params
    bits = []
    for v in (bp["n1"], bp["n2"], bp["n3"]):
        bits.extend([(v >> k) & 1 for k in range(4, -1, -1)])
    hist = [float(t.value) for t in study.trials if t.value is not None]
    return OptResult(best_bits=np.array(bits, dtype=np.int8), best_score=float(study.best_value), history=hist)


def optuna_search_svm_rbf(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_names: List[str],
    cfg: ObjectiveConfig,
    n_trials: int,
    seed: int,
) -> OptResult:
    import optuna

    def trial_obj(trial: "optuna.Trial") -> float:
        # log ranges are typically sensible for SVM
        C = trial.suggest_float("C", 1e-3, 1e3, log=True)
        gamma = trial.suggest_float("gamma", 1e-6, 1e0, log=True)
        return float(cv_objective_model(
            model_factory=lambda C=C, gamma=gamma: make_svm_rbf(C, gamma, seed=cfg.seed),
            X=X, y=y, groups=groups, feature_names=feature_names, cfg=cfg
        ))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(trial_obj, n_trials=n_trials, show_progress_bar=True)

    hist = [float(t.value) for t in study.trials if t.value is not None]
    best_params = {"C": float(study.best_params["C"]), "gamma": float(study.best_params["gamma"])}
    return OptResult(best_bits=None, best_score=float(study.best_value), history=hist, best_params=best_params)


def optuna_search_svm_linear(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_names: List[str],
    cfg: ObjectiveConfig,
    n_trials: int,
    seed: int,
) -> OptResult:
    import optuna

    def trial_obj(trial: "optuna.Trial") -> float:
        C = trial.suggest_float("C", 1e-3, 1e3, log=True)
        return float(cv_objective_model(
            model_factory=lambda C=C: Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="linear", C=C, probability=True, class_weight="balanced", random_state=cfg.seed)),
            ]),
            X=X, y=y, groups=groups, feature_names=feature_names, cfg=cfg
        ))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(trial_obj, n_trials=n_trials, show_progress_bar=True)
    hist = [float(t.value) for t in study.trials if t.value is not None]
    return OptResult(best_bits=None, best_score=float(study.best_value), history=hist,
                     best_params={"C": float(study.best_params["C"])})


def optuna_search_logreg(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_names: List[str],
    cfg: ObjectiveConfig,
    n_trials: int,
    seed: int,
) -> OptResult:
    import optuna

    def trial_obj(trial: "optuna.Trial") -> float:
        C = trial.suggest_float("C", 1e-3, 1e3, log=True)
        return float(cv_objective_model(
            model_factory=lambda C=C: make_logreg(seed=cfg.seed, C=C),
            X=X, y=y, groups=groups, feature_names=feature_names, cfg=cfg
        ))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(trial_obj, n_trials=n_trials, show_progress_bar=True)
    hist = [float(t.value) for t in study.trials if t.value is not None]
    return OptResult(best_bits=None, best_score=float(study.best_value), history=hist,
                     best_params={"C": float(study.best_params["C"])})


# ============================
# Batch runner
# ============================

def run_one_optimizer(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_names: List[str],
    cfg: ObjectiveConfig,
    opt_target: str,
    opt_method: str,
    args: argparse.Namespace,
    collect_folds: bool = False,
) -> object:
    rng = np.random.default_rng(cfg.seed)

    if opt_target == "mlp_arch":
        def obj_bits(bits: np.ndarray) -> float:
            return cv_objective_mlp_bits(
                X=X, y=y, groups=groups, feature_names=feature_names, bits=bits, cfg=cfg
            )

        dim = 15
        if opt_method == "random":
            opt_res = random_search(obj_bits, dim=dim, n_eval=args.n_eval, rng=rng)
        elif opt_method == "bpso":
            opt_res = bpso(obj_bits, dim=dim, n_particles=args.n_pop, n_iter=args.n_iter,
                           rng=rng, w=args.w, c1=args.c1, c2=args.c2, transfer=args.transfer)
        elif opt_method == "bwoa":
            opt_res = bwoa(obj_bits, dim=dim, n_pop=args.n_pop, n_iter=args.n_iter,
                           rng=rng, transfer=args.transfer)
        elif opt_method == "bde":
            opt_res = bde(obj_bits, dim=dim, n_pop=args.n_pop, n_iter=args.n_iter,
                          rng=rng, F=args.F, CR=args.CR, transfer=args.transfer)
        elif opt_method == "bsade":
            opt_res = bsade(obj_bits, dim=dim, n_pop=args.n_pop, n_iter=args.n_iter,
                            rng=rng, transfer=args.transfer)
        elif opt_method == "optuna":
            opt_res = optuna_search_mlp(obj_bits, n_trials=args.n_eval, seed=args.seed)
        else:
            raise ValueError(f"Unknown opt_method for mlp_arch: {opt_method}")

        n1, n2, n3 = bits_to_arch(opt_res.best_bits)
        hl = arch_tuple(n1, n2, n3)

        eval_block, seg_rows, grp_rows = evaluate_model_cv_with_folds(
            model_factory=lambda hl=hl: make_mlp(hl, cfg.seed),
            X=X, y=y, groups=groups, feature_names=feature_names, cfg=cfg
        )
        obj_value = objective_from_eval(eval_block, cfg)

        rec = {
            "opt_target": "mlp_arch",
            "method": opt_method,
            "transfer": args.transfer if opt_method in ["bpso", "bwoa", "bde", "bsade"] else None,
            "objective": {"metric": cfg.opt_metric, "level": cfg.opt_level},
            "best_objective_value": float(obj_value),
            "best_value_seen_by_optimizer": float(opt_res.best_score),
            "best_bits": opt_res.best_bits.astype(int).tolist(),
            "best_n1n2n3": [int(n1), int(n2), int(n3)],
            "hidden_layer_sizes": list(map(int, hl)),
            "history": [float(x) for x in (opt_res.history or [])],
            "evaluation": eval_block,
        }
        if collect_folds:
            return rec, seg_rows, grp_rows
        return rec

    if opt_target in {"svm_rbf", "svm_linear", "logreg"}:
        if opt_target == "svm_rbf":
            def sampler(rrng: np.random.Generator) -> Dict[str, float]:
                return {"C": float(10 ** rrng.uniform(-3, 3)), "gamma": float(10 ** rrng.uniform(-6, 0))}

            def obj_from_params(pp: Dict[str, float]) -> float:
                return float(cv_objective_model(
                    model_factory=lambda pp=pp: make_svm_rbf(float(pp["C"]), float(pp["gamma"]), cfg.seed),
                    X=X, y=y, groups=groups, feature_names=feature_names, cfg=cfg
                ))

            if opt_method == "optuna":
                opt_res = optuna_search_svm_rbf(
                    X=X, y=y, groups=groups, feature_names=feature_names, cfg=cfg,
                    n_trials=args.n_eval, seed=args.seed
                )
            elif opt_method == "random":
                opt_res = random_search_params(obj_from_params, sampler, args.n_eval, rng)
            else:
                raise ValueError("svm_rbf supports only optuna or random.")

            C = float(opt_res.best_params["C"])
            gamma = float(opt_res.best_params["gamma"])
            eval_block, seg_rows, grp_rows = evaluate_model_cv_with_folds(
                model_factory=lambda C=C, gamma=gamma: make_svm_rbf(C, gamma, cfg.seed),
                X=X, y=y, groups=groups, feature_names=feature_names, cfg=cfg
            )

        elif opt_target == "svm_linear":
            def sampler(rrng: np.random.Generator) -> Dict[str, float]:
                return {"C": float(10 ** rrng.uniform(-3, 3))}

            def obj_from_params(pp: Dict[str, float]) -> float:
                C = float(pp["C"])
                return float(cv_objective_model(
                    model_factory=lambda C=C: Pipeline([
                        ("scaler", StandardScaler()),
                        ("clf", SVC(kernel="linear", C=C, probability=True, class_weight="balanced", random_state=cfg.seed)),
                    ]),
                    X=X, y=y, groups=groups, feature_names=feature_names, cfg=cfg
                ))

            if opt_method == "optuna":
                opt_res = optuna_search_svm_linear(
                    X=X, y=y, groups=groups, feature_names=feature_names, cfg=cfg,
                    n_trials=args.n_eval, seed=args.seed
                )
            elif opt_method == "random":
                opt_res = random_search_params(obj_from_params, sampler, args.n_eval, rng)
            else:
                raise ValueError("svm_linear supports only optuna or random.")

            C = float(opt_res.best_params["C"])
            eval_block, seg_rows, grp_rows = evaluate_model_cv_with_folds(
                model_factory=lambda C=C: Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf", SVC(kernel="linear", C=C, probability=True, class_weight="balanced", random_state=cfg.seed)),
                ]),
                X=X, y=y, groups=groups, feature_names=feature_names, cfg=cfg
            )

        else:  # logreg
            def sampler(rrng: np.random.Generator) -> Dict[str, float]:
                return {"C": float(10 ** rrng.uniform(-3, 3))}

            def obj_from_params(pp: Dict[str, float]) -> float:
                C = float(pp["C"])
                return float(cv_objective_model(
                    model_factory=lambda C=C: make_logreg(seed=cfg.seed, C=C),
                    X=X, y=y, groups=groups, feature_names=feature_names, cfg=cfg
                ))

            if opt_method == "optuna":
                opt_res = optuna_search_logreg(
                    X=X, y=y, groups=groups, feature_names=feature_names, cfg=cfg,
                    n_trials=args.n_eval, seed=args.seed
                )
            elif opt_method == "random":
                opt_res = random_search_params(obj_from_params, sampler, args.n_eval, rng)
            else:
                raise ValueError("logreg supports only optuna or random.")

            C = float(opt_res.best_params["C"])
            eval_block, seg_rows, grp_rows = evaluate_model_cv_with_folds(
                model_factory=lambda C=C: make_logreg(seed=cfg.seed, C=C),
                X=X, y=y, groups=groups, feature_names=feature_names, cfg=cfg
            )

        obj_value = objective_from_eval(eval_block, cfg)
        rec = {
            "opt_target": opt_target,
            "method": opt_method,
            "transfer": None,
            "objective": {"metric": cfg.opt_metric, "level": cfg.opt_level},
            "best_objective_value": float(obj_value),
            "best_value_seen_by_optimizer": float(opt_res.best_score),
            "best_params": opt_res.best_params,
            "history": [float(x) for x in (opt_res.history or [])],
            "evaluation": eval_block,
        }
        if collect_folds:
            return rec, seg_rows, grp_rows
        return rec

    raise ValueError(f"Unknown opt_target: {opt_target}")


# ============================
# Main
# ============================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="Feature CSV (rows=segments).")
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cv_splits", type=int, default=5)
    ap.add_argument("--cv_repeats", type=int, default=1,
                    help="Number of repeated Group CV runs (different random seeds per repeat).")

    # Feature selection inside CV
    ap.add_argument("--feature_select", type=str, default="corr", choices=["none", "corr"])
    ap.add_argument("--corr_thresh", type=float, default=0.85)
    ap.add_argument("--top_k", type=int, default=16)

    # Baselines toggles
    ap.add_argument("--run_baselines", action="store_true")
    ap.add_argument("--svm_C", type=float, default=10.0)
    ap.add_argument("--svm_gamma", type=float, default=0.01)

    # Optimization objective choice
    ap.add_argument("--opt_metric", type=str, default="acc", choices=["acc", "f1", "auc"],
                    help="Metric to optimize.")
    ap.add_argument("--opt_level", type=str, default="group_mean",
                    choices=["segment", "group_mean", "group_majority"],
                    help="Level to optimize: segment or group (file) aggregation.")

    # Optimization toggles
    ap.add_argument("--run_opt", action="store_true")
    ap.add_argument("--opt_target", type=str, default="mlp_arch",
                    choices=["mlp_arch", "svm_rbf", "svm_linear", "logreg"],
                    help="Optimization target for single-run mode (when --opt_all is not set).")
    ap.add_argument("--opt_method", type=str, default="optuna",
                    choices=["random", "bpso", "bwoa", "bde", "bsade", "optuna"])
    ap.add_argument("--transfer", type=str, default="s", choices=["s", "v"])
    ap.add_argument("--n_eval", type=int, default=200, help="Random evals OR Optuna trials.")
    ap.add_argument("--n_pop", type=int, default=20)
    ap.add_argument("--n_iter", type=int, default=30)
    ap.add_argument("--w", type=float, default=0.72)
    ap.add_argument("--c1", type=float, default=1.49)
    ap.add_argument("--c2", type=float, default=1.49)
    ap.add_argument("--F", type=float, default=0.5)
    ap.add_argument("--CR", type=float, default=0.9)

    # Batch mode for optimization
    ap.add_argument("--opt_all", action="store_true",
                    help="Run a batch of optimizers in one call and save summary_best_<csvstem>.json.")
    ap.add_argument("--opt_targets", type=str, default="mlp_arch,svm_rbf",
                    help="Comma-separated targets: mlp_arch,svm_rbf,svm_linear,logreg.")
    ap.add_argument("--opt_methods", type=str, default="random,bpso,bwoa,bde,bsade,optuna",
                    help="Comma-separated methods. Allowed by target: "
                         "mlp_arch={random,bpso,bwoa,bde,bsade,optuna}, "
                         "svm_rbf/svm_linear/logreg={random,optuna}.")

    # A fixed MLP baseline
    ap.add_argument("--mlp_default", type=str, default="16,8",
                    help="Comma-separated hidden sizes for a fixed MLP baseline (e.g., '16,8' or '32'). Use '' for 0 layers.")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    if "label" not in df.columns or "group_id" not in df.columns:
        raise ValueError("CSV must contain at least: label, group_id")

    feature_names = [
        c for c in df.columns
        if c not in META_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]
    if len(feature_names) == 0:
        raise ValueError("No numeric feature columns found (excluding metadata).")

    X = df[feature_names].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=int)
    groups = df["group_id"].astype(str).to_numpy()

    cfg = ObjectiveConfig(
        feature_select=args.feature_select,
        corr_thresh=args.corr_thresh,
        top_k=args.top_k,
        cv_splits=args.cv_splits,
        cv_repeats=args.cv_repeats,
        seed=args.seed,
        opt_metric=args.opt_metric,
        opt_level=args.opt_level,
    )

    results: Dict[str, object] = {
        "csv": str(Path(args.csv).resolve()),
        "n_rows_segments": int(len(df)),
        "n_groups_files": int(pd.Series(groups).nunique()),
        "n_features_total": int(X.shape[1]),
        "settings": vars(args),
        "baselines": {},
        "optimization": {},
    }
    csvstem = Path(args.csv).stem
    fold_rows_segment_all: List[Dict[str, object]] = []
    fold_rows_group_all: List[Dict[str, object]] = []

    # ------------------------
    # Baselines
    # ------------------------
    if args.run_baselines:
        print("[RUN] Baselines...")

        ev, seg_rows, grp_rows = evaluate_model_cv_with_folds(
            model_factory=lambda: make_logreg(args.seed),
            X=X, y=y, groups=groups, feature_names=feature_names, cfg=cfg
        )
        results["baselines"]["logreg"] = ev
        for r in seg_rows:
            fold_rows_segment_all.append({"dataset": csvstem, "model": "logreg", **r})
        for r in grp_rows:
            fold_rows_group_all.append({"dataset": csvstem, "model": "logreg", **r})

        ev, seg_rows, grp_rows = evaluate_model_cv_with_folds(
            model_factory=lambda: make_svm_linear(args.seed),
            X=X, y=y, groups=groups, feature_names=feature_names, cfg=cfg
        )
        results["baselines"]["svm_linear"] = ev
        for r in seg_rows:
            fold_rows_segment_all.append({"dataset": csvstem, "model": "svm_linear", **r})
        for r in grp_rows:
            fold_rows_group_all.append({"dataset": csvstem, "model": "svm_linear", **r})

        ev, seg_rows, grp_rows = evaluate_model_cv_with_folds(
            model_factory=lambda: make_svm_rbf(args.svm_C, args.svm_gamma, args.seed),
            X=X, y=y, groups=groups, feature_names=feature_names, cfg=cfg
        )
        results["baselines"]["svm_rbf"] = ev
        for r in seg_rows:
            fold_rows_segment_all.append({"dataset": csvstem, "model": "svm_rbf", **r})
        for r in grp_rows:
            fold_rows_group_all.append({"dataset": csvstem, "model": "svm_rbf", **r})

        if args.mlp_default.strip() == "":
            hl = ()
        else:
            hl = tuple(int(x) for x in args.mlp_default.split(",") if x.strip())
        ev, seg_rows, grp_rows = evaluate_model_cv_with_folds(
            model_factory=lambda hl=hl: make_mlp(hl, args.seed),
            X=X, y=y, groups=groups, feature_names=feature_names, cfg=cfg
        )
        results["baselines"]["mlp_fixed"] = ev
        for r in seg_rows:
            fold_rows_segment_all.append({"dataset": csvstem, "model": "mlp_fixed", **r})
        for r in grp_rows:
            fold_rows_group_all.append({"dataset": csvstem, "model": "mlp_fixed", **r})

    # ------------------------
    # Optimization
    # ------------------------
    if args.run_opt:
        if args.opt_all:
            print(f"[RUN] Batch optimization: targets={args.opt_targets} methods={args.opt_methods} "
                  f"objective={args.opt_metric}@{args.opt_level}")

            targets = [t.strip() for t in args.opt_targets.split(",") if t.strip()]
            methods = [m.strip() for m in args.opt_methods.split(",") if m.strip()]
            supported_methods = {
                "mlp_arch": {"random", "bpso", "bwoa", "bde", "bsade", "optuna"},
                "svm_rbf": {"random", "optuna"},
                "svm_linear": {"random", "optuna"},
                "logreg": {"random", "optuna"},
            }

            batch_results: List[Dict[str, object]] = []
            for t in targets:
                if t not in supported_methods:
                    print(f"[WARN] Skipping unknown opt_target: {t}")
                    continue
                target_methods = [m for m in methods if m in supported_methods[t]]
                if not target_methods:
                    print(f"[WARN] No compatible methods for target={t}; requested={methods}")
                    continue

                for m in target_methods:
                    print(f"[BATCH] {t} / {m}")
                    rec, seg_rows, grp_rows = run_one_optimizer(
                        X=X, y=y, groups=groups, feature_names=feature_names,
                        cfg=cfg, opt_target=t, opt_method=m, args=args,
                        collect_folds=True,
                    )
                    batch_results.append(rec)
                    model_name = f"opt_{t}_{m}"
                    for r in seg_rows:
                        fold_rows_segment_all.append({"dataset": csvstem, "model": model_name, **r})
                    for r in grp_rows:
                        fold_rows_group_all.append({"dataset": csvstem, "model": model_name, **r})
                    per_path = out_dir / f"opt_{csvstem}_{t}_{m}.json"
                    with open(per_path, "w", encoding="utf-8") as f:
                        json.dump(rec, f, indent=2)
                    print(f"[OK] Saved: {per_path}")

            if not batch_results:
                raise ValueError("opt_all enabled, but no valid targets were selected.")

            best = max(batch_results, key=lambda r: r.get("best_objective_value", -1e9))

            summary = {
                "csv": str(Path(args.csv).resolve()),
                "objective": {"metric": args.opt_metric, "level": args.opt_level},
                "n_candidates": int(len(batch_results)),
                "best_overall": best,
                "leaderboard": sorted(
                    [
                        {
                            "opt_target": r["opt_target"],
                            "method": r["method"],
                            "transfer": r.get("transfer"),
                            "best_objective_value": float(r["best_objective_value"]),
                        }
                        for r in batch_results
                    ],
                    key=lambda d: d["best_objective_value"],
                    reverse=True
                )
            }

            summ_path = out_dir / f"summary_best_{csvstem}.json"
            with open(summ_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            print(f"[OK] Saved: {summ_path}")

            results["optimization"]["batch_summary"] = summary

        else:
            # Single optimizer run
            print(f"[RUN] Optimization (single): target={args.opt_target} method={args.opt_method} "
                  f"objective={args.opt_metric}@{args.opt_level}")
            rec, seg_rows, grp_rows = run_one_optimizer(
                X=X, y=y, groups=groups, feature_names=feature_names,
                cfg=cfg, opt_target=args.opt_target, opt_method=args.opt_method, args=args,
                collect_folds=True,
            )
            results["optimization"]["best"] = {
                "opt_target": rec.get("opt_target"),
                "method": rec.get("method"),
                "transfer": rec.get("transfer"),
                "objective": rec.get("objective"),
                "best_objective_value": rec.get("best_objective_value"),
                "best_value_seen_by_optimizer": rec.get("best_value_seen_by_optimizer"),
                "best_bits": rec.get("best_bits"),
                "best_n1n2n3": rec.get("best_n1n2n3"),
                "hidden_layer_sizes": rec.get("hidden_layer_sizes"),
                "best_params": rec.get("best_params"),
                "history": rec.get("history"),
            }
            results["optimization"]["evaluation"] = rec["evaluation"]
            model_name = f"opt_{args.opt_target}_{args.opt_method}"
            for r in seg_rows:
                fold_rows_segment_all.append({"dataset": csvstem, "model": model_name, **r})
            for r in grp_rows:
                fold_rows_group_all.append({"dataset": csvstem, "model": model_name, **r})

    # ------------------------
    # Save fold-level tables + corrected paired tests
    # ------------------------
    if fold_rows_segment_all:
        seg_df = pd.DataFrame(fold_rows_segment_all)
        seg_path = out_dir / "fold_results_segment.csv"
        seg_df.to_csv(seg_path, index=False)
        print(f"[OK] Saved: {seg_path}")

    if fold_rows_group_all:
        grp_df = pd.DataFrame(fold_rows_group_all)
        grp_path = out_dir / "fold_results_group.csv"
        grp_df.to_csv(grp_path, index=False)
        print(f"[OK] Saved: {grp_path}")

        try:
            stats_df = run_pairwise_wilcoxon_with_holm(
                grp_df, metric="balanced_accuracy", aggregate="group_mean_proba"
            )
            if not stats_df.empty:
                stats_path = out_dir / "stats_pairwise_group_mean_balanced_accuracy.csv"
                stats_df.to_csv(stats_path, index=False)
                print(f"[OK] Saved: {stats_path}")
        except Exception as e:
            print(f"[WARN] Failed to compute corrected paired tests: {e}")

    # ------------------------
    # Save main results JSON
    # ------------------------
    out_path = out_dir / f"results_{Path(args.csv).stem}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[OK] Saved: {out_path}")

    # ------------------------
    # Print quick summary
    # ------------------------
    def _acc(block: Dict[str, object], level: str = "segment") -> Optional[float]:
        try:
            return float(block[level]["acc"])
        except Exception:
            return None

    if args.run_baselines:
        print("\n[SUMMARY] Segment accuracy (baselines):")
        for k, v in results["baselines"].items():
            print(f"  {k:12s}  acc={_acc(v, 'segment'):.4f}")

    if args.run_opt:
        if args.opt_all:
            b = results["optimization"]["batch_summary"]["best_overall"]
            print("\n[SUMMARY] Best overall (batch):")
            print(f"  target={b['opt_target']} method={b['method']} transfer={b.get('transfer')}")
            if b["opt_target"] == "mlp_arch":
                print(f"  arch n1,n2,n3={b['best_n1n2n3']} hidden={b['hidden_layer_sizes']}")
            if b.get("best_params") is not None:
                print(f"  params={b['best_params']}")
            ev = b["evaluation"]
            print(f"  segment acc={ev['segment']['acc']:.4f} "
                  f"group_mean acc={ev['group_mean_proba']['acc']:.4f} "
                  f"group_majority acc={ev['group_majority']['acc']:.4f}")
        else:
            if "evaluation" in results["optimization"]:
                ev = results["optimization"]["evaluation"]
                best = results["optimization"]["best"]
                print("\n[SUMMARY] Optimized model (single):")
                print(f"  target={best.get('opt_target')} method={best.get('method')} transfer={best.get('transfer')}")
                if best.get("opt_target") == "mlp_arch":
                    print(f"  best_arch n1,n2,n3={best.get('best_n1n2n3')} hidden={best.get('hidden_layer_sizes')}")
                if best.get("best_params") is not None:
                    print(f"  params={best.get('best_params')}")
                print(f"  segment acc={ev['segment']['acc']:.4f} "
                      f"group_mean acc={ev['group_mean_proba']['acc']:.4f} "
                      f"group_majority acc={ev['group_majority']['acc']:.4f}")


if __name__ == "__main__":
    main()
