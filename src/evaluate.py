"""Evaluation metrics: accuracy, precision, recall, F1, confusion matrix."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from src.utils import resolve_path


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_labels: list[int] = []
    all_preds: list[int] = []
    all_probs: list[list[float]] = []

    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        preds = outputs.argmax(dim=1).cpu().numpy()

        all_labels.extend(labels.numpy().tolist())
        all_preds.extend(preds.tolist())
        all_probs.extend(probs.tolist())

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
    )


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    pos_label = class_names.index("defect") if "defect" in class_names else 1
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(
                y_true, y_pred, average="binary", pos_label=pos_label, zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_true, y_pred, average="binary", pos_label=pos_label, zero_division=0
            )
        ),
        "f1_score": float(
            f1_score(
                y_true, y_pred, average="binary", pos_label=pos_label, zero_division=0
            )
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, target_names=class_names, zero_division=0
        ),
        "class_names": class_names,
    }


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: list[str],
    config: dict[str, Any],
    logger=None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    y_true, y_pred, _ = collect_predictions(model, loader, device)
    metrics = compute_metrics(y_true, y_pred, class_names)

    metrics_path = resolve_path(config["evaluation"]["metrics_file"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(
            {k: v for k, v in metrics.items() if k != "classification_report"},
            f,
            indent=2,
        )

    report_path = resolve_path(config["evaluation"]["classification_report_file"])
    with report_path.open("w", encoding="utf-8") as f:
        f.write(metrics["classification_report"])

    if logger:
        logger.info(
            f"Evaluation | accuracy={metrics['accuracy']:.4f} "
            f"precision={metrics['precision']:.4f} "
            f"recall={metrics['recall']:.4f} "
            f"f1={metrics['f1_score']:.4f}"
        )

    return metrics, y_true, y_pred
