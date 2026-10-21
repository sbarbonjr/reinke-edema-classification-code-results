#!/usr/bin/env python3
"""Generate the R1 classifier heatmap from public aggregate results."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results_r1_nested"

FEATURE_ORDER = [
    "fft_topk_hann", "mfcc_stats", "spectral_stats", "wavlm", "wpd_haar7_std",
]
FEATURE_LABELS = ["FFT-topk", "MFCC", "Spectral", "WavLM", "WPD-Haar7"]
MODEL_ORDER = [
    "logreg_window_balanced", "logreg_patient_balanced",
    "svm_linear_window_balanced", "svm_linear_patient_balanced",
    "svm_rbf_window_balanced", "svm_rbf_patient_balanced",
    "mlp_relu_window_unweighted", "mlp_relu_patient_balanced",
    "mlp_logistic_legacy",
]
MODEL_LABELS = [
    "LogReg\n(CW)", "LogReg\n(PW)",
    "Linear SVM\n(CW)", "Linear SVM\n(PW)",
    "RBF-SVM\n(CW)", "RBF-SVM\n(PW)",
    "ReLU MLP\n(UW)", "ReLU MLP\n(PW)",
    "Fixed logistic\nMLP",
]


def encoder_classifier_heatmap() -> None:
    source = RESULTS / "aggregate_metrics.csv"
    summary = pd.read_csv(source)
    frame = summary.pivot(index="feature", columns="model", values="balanced_accuracy_mean")
    frame = frame.loc[FEATURE_ORDER, MODEL_ORDER]
    values = frame.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(10.4, 4.0))
    image = ax.imshow(values, cmap="YlGnBu", vmin=0.50, vmax=0.95, aspect="auto")
    ax.set_xticks(np.arange(frame.shape[1]), labels=MODEL_LABELS)
    ax.set_yticks(np.arange(frame.shape[0]), labels=FEATURE_LABELS)
    ax.set_xlabel("Classifier and training-weight condition (CW: class-weighted; PW: patient-weighted; UW: unweighted)")
    ax.set_ylabel("Feature representation")
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center", fontsize=8)

    threshold = 0.74
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            ax.text(
                col,
                row,
                f"{values[row, col]:.2f}",
                ha="center",
                va="center",
                color="white" if values[row, col] >= threshold else "black",
                fontsize=8.5,
            )

    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label("Mean subject-level balanced accuracy")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIGURES / "encoder_classifier_family_ba_heatmap_R1.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    encoder_classifier_heatmap()
