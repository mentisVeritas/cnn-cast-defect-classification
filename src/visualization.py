"""Training curves and confusion matrix plots."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.utils import resolve_path


def plot_loss_curves(history: dict[str, list[float]], save_path: str | Path) -> None:
    save_path = resolve_path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_accuracy_curves(history: dict[str, list[float]], save_path: str | Path) -> None:
    save_path = resolve_path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_accuracy"]) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_accuracy"], label="Train Accuracy")
    plt.plot(epochs, history["val_accuracy"], label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    save_path: str | Path,
) -> None:
    save_path = resolve_path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_all_training_plots(history: dict, config: dict[str, Any]) -> None:
    plot_loss_curves(history, config["visualization"]["loss_curve"])
    plot_accuracy_curves(history, config["visualization"]["accuracy_curve"])


def save_confusion_matrix_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    config: dict[str, Any],
) -> None:
    plot_confusion_matrix(
        y_true,
        y_pred,
        class_names,
        config["evaluation"]["confusion_matrix_plot"],
    )


def plot_metrics_comparison(
    rows: list[dict[str, Any]],
    save_path: str | Path,
    title: str = "Model comparison",
) -> None:
    """Bar chart comparing accuracy / F1 across experiments or models."""
    save_path = resolve_path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    labels = [str(r.get("model") or r.get("experiment", "run")) for r in rows]
    accuracy = [float(r["accuracy"]) for r in rows]
    f1 = [float(r["f1_score"]) for r in rows]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 5))
    ax.bar(x - width / 2, accuracy, width, label="Accuracy")
    ax.bar(x + width / 2, f1, width, label="F1-score")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x, labels, rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_tuning_comparison(
    rows: list[dict[str, Any]],
    save_path: str | Path,
) -> None:
    """Compare tuning experiments: accuracy, F1, and training time."""
    save_path = resolve_path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    labels = [str(r.get("experiment", "exp")) for r in rows]
    accuracy = [float(r["accuracy"]) for r in rows]
    f1 = [float(r["f1_score"]) for r in rows]
    times = [float(r.get("training_time_seconds") or 0) for r in rows]
    if all(t == 0 for t in times):
        times = [i + 1 for i in range(len(labels))]  # placeholder if unknown

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    x = np.arange(len(labels))
    axes[0].bar(x - 0.2, accuracy, 0.4, label="Accuracy")
    axes[0].bar(x + 0.2, f1, 0.4, label="F1")
    axes[0].set_xticks(x, labels, rotation=15, ha="right")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Test metrics by experiment")
    axes[0].legend()
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(labels, times, color="steelblue")
    axes[1].set_title("Training time (seconds)")
    axes[1].set_ylabel("Seconds")
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
