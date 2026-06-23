"""Shared utilities for the Deep Learning project notebooks."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


# Anchor output locations to the project, regardless of the notebook's current
# working directory.
PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
METRICS_DIR = OUTPUT_DIR / "metrics"
MODELS_DIR = OUTPUT_DIR / "models"

for directory in (FIGURES_DIR, METRICS_DIR, MODELS_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def get_device():
    """Return the best device available to PyTorch."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_history(history, title, output_path):
    """Save a loss-history chart."""
    plt.figure(figsize=(7, 4))
    for name, values in history.items():
        plt.plot(range(1, len(values) + 1), values, label=name.replace("_", " "))
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_confusion_matrix(matrix, class_names, title, output_path):
    """Save an annotated confusion-matrix chart."""
    plt.figure(figsize=(5, 4))
    plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=30, ha="right")
    plt.yticks(ticks, class_names)
    threshold = matrix.max() / 2 if matrix.size else 0
    for row, column in np.ndindex(matrix.shape):
        plt.text(
            column,
            row,
            str(matrix[row, column]),
            ha="center",
            va="center",
            color="white" if matrix[row, column] > threshold else "black",
        )
    plt.ylabel("Classe reelle")
    plt.xlabel("Classe predite")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Type non serialisable: {type(value).__name__}")


def save_json(payload, output_path):
    """Write a JSON result file, including NumPy and Path values."""
    with Path(output_path).open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, default=_json_default)
