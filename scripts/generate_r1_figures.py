#!/usr/bin/env python3
"""Generate the R1 classifier heatmap from public aggregate results."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["pdf.fonttype"] = 42


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
MODEL_LABELS = ["CW", "PW", "CW", "PW", "CW", "PW", "UW", "PW", "Fixed"]
# Classifier families drawn as a second tier below the weighting labels.
FAMILY_LABELS = [
    (0.5, "LogReg"), (2.5, "Linear SVM"), (4.5, "RBF-SVM"),
    (6.5, "ReLU MLP"), (8.0, "Logistic\nMLP"),
]


def encoder_classifier_heatmap() -> None:
    source = RESULTS / "aggregate_metrics.csv"
    summary = pd.read_csv(source)
    frame = summary.pivot(index="feature", columns="model", values="balanced_accuracy_mean")
    frame = frame.loc[FEATURE_ORDER, MODEL_ORDER]
    values = frame.to_numpy(dtype=float)
    # Sized close to the printed width (0.95 textwidth) so fonts match the caption.
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    image = ax.imshow(values, cmap="YlGnBu", vmin=0.50, vmax=0.95, aspect="auto")
    ax.set_xticks(np.arange(frame.shape[1]), labels=MODEL_LABELS)
    ax.set_yticks(np.arange(frame.shape[0]), labels=FEATURE_LABELS, fontsize=8)
    for x, label in FAMILY_LABELS:
        ax.text(x, -0.12, label, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=8)
    ax.set_ylabel("Feature representation", fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center", fontsize=7.5)
    ax.tick_params(length=0)
    # Separate classifier families.
    for boundary in (1.5, 3.5, 5.5, 7.5):
        ax.axvline(boundary, color="white", linewidth=2.5)

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
                fontsize=8,
            )

    colorbar = fig.colorbar(image, ax=ax, pad=0.015, fraction=0.04)
    colorbar.set_label("Mean subject-level BA", fontsize=8)
    colorbar.ax.tick_params(labelsize=7.5)
    fig.tight_layout()
    fig.savefig(FIGURES / "encoder_classifier_family_ba_heatmap_R1.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    encoder_classifier_heatmap()
