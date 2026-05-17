"""Evaluation utilities for the World Bank finance classification project.

This module contains reusable plotting and metric-export utilities for model
comparison. Figures are saved at publication quality so the project can be
reviewed directly from GitHub.

Author: Shivang Sharma
"""

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc

FIGURE_DPI = 300
CLASS_LABELS = ["Low Constraint", "High Constraint"]


def save_metrics_table(metrics: Dict[str, Dict[str, float]], output_path: str) -> None:
    """Save model metrics as a CSV table.

    Parameters
    ----------
    metrics : Dict[str, Dict[str, float]]
        Nested dictionary of model metrics.
    output_path : str
        Path where the CSV file should be written.
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    metrics_frame = pd.DataFrame(metrics).T.reset_index().rename(columns={"index": "model"})
    metrics_frame.to_csv(output_file, index=False)


def plot_roc_curves(
    y_true: np.ndarray,
    predicted_probabilities: Dict[str, np.ndarray],
    save_path: str,
) -> None:
    """Plot ROC curves for multiple classification models.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    predicted_probabilities : Dict[str, np.ndarray]
        Mapping of model name to predicted probability for the positive class.
    save_path : str
        Output path for the figure.
    """
    output_file = Path(save_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    for model_name, probabilities in predicted_probabilities.items():
        false_positive_rate, true_positive_rate, _ = roc_curve(y_true, probabilities)
        model_auc = auc(false_positive_rate, true_positive_rate)
        ax.plot(
            false_positive_rate,
            true_positive_rate,
            linewidth=2,
            label=f"{model_name} (AUC = {model_auc:.3f})",
        )

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Random Classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves: Model Comparison")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, save_path: str) -> None:
    """Plot a confusion matrix with counts and row percentages."""
    output_file = Path(save_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    matrix = confusion_matrix(y_true, y_pred)
    row_totals = matrix.sum(axis=1, keepdims=True)
    percentages = np.divide(matrix, row_totals, where=row_totals != 0) * 100
    labels = np.array(
        [
            [f"{count}\n({pct:.1f}%)" for count, pct in zip(row_counts, row_pcts)]
            for row_counts, row_pcts in zip(matrix, percentages)
        ]
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(matrix, interpolation="nearest")
    plt.colorbar(image, ax=ax)
    ax.set_xticks(np.arange(len(CLASS_LABELS)))
    ax.set_yticks(np.arange(len(CLASS_LABELS)))
    ax.set_xticklabels(CLASS_LABELS)
    ax.set_yticklabels(CLASS_LABELS)
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    ax.set_title(f"Confusion Matrix: {model_name}")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, labels[i, j], ha="center", va="center")

    plt.tight_layout()
    plt.savefig(output_file, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
