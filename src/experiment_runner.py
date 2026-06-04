"""Train and evaluate isolated experiments (baseline, hyperparameter tuning)."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import torch.nn as nn

from src.dataset import get_dataloaders
from src.evaluate import compute_metrics, collect_predictions
from src.train_engine import Trainer
from src.utils import get_device, load_config, resolve_path, set_seed
from src.visualization import plot_confusion_matrix


def apply_training_overrides(
    config: dict[str, Any],
    *,
    learning_rate: float | None = None,
    batch_size: int | None = None,
    dropout: float | None = None,
    epochs: int | None = None,
) -> dict[str, Any]:
    """Return a deep copy of config with training overrides applied."""
    cfg = copy.deepcopy(config)
    if learning_rate is not None:
        cfg["training"]["learning_rate"] = learning_rate
    if batch_size is not None:
        cfg["training"]["batch_size"] = batch_size
    if dropout is not None:
        cfg["training"]["dropout"] = dropout
    if epochs is not None:
        cfg["training"]["epochs"] = epochs
    return cfg


def run_train_and_evaluate(
    config: dict[str, Any],
    model: nn.Module,
    output_dir: str | Path,
    experiment_name: str,
    device=None,
    resume_training: bool = False,
) -> dict[str, Any]:
    """
    Train model on train/val splits, evaluate on test split, save metrics and confusion matrix.

    Returns dict with metrics, training_time_seconds, history, checkpoint paths.
    """
    output_dir = resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = get_device(config)

    train_loader, val_loader, test_loader, class_names = get_dataloaders(config, device)

    latest_ckpt = output_dir / "checkpoints" / "latest_checkpoint.pth"
    best_ckpt = output_dir / "best_model.pth"

    cfg = copy.deepcopy(config)
    can_resume = resume_training and latest_ckpt.exists()
    cfg["checkpoint"] = {
        **cfg.get("checkpoint", {}),
        "start_from_scratch": not can_resume,
        "resume_training": can_resume,
        "save_best_model": True,
        "latest_checkpoint": str(latest_ckpt),
        "best_model": str(best_ckpt),
    }

    if can_resume:
        print(f"Resuming training from {latest_ckpt}")

    trainer = Trainer(
        cfg,
        train_loader,
        val_loader,
        class_names,
        device,
        model=model,
        latest_checkpoint=latest_ckpt,
        best_checkpoint=best_ckpt,
        resume_training=can_resume,
    )
    history = trainer.train()

    # Load best weights for test evaluation
    from src.train_engine import load_checkpoint

    load_checkpoint(best_ckpt, trainer.model, trainer.optimizer, device)

    y_true, y_pred, _ = collect_predictions(trainer.model, test_loader, device)
    metrics = compute_metrics(y_true, y_pred, class_names)
    metrics["experiment"] = experiment_name
    metrics["training_time_seconds"] = round(trainer.training_time_seconds, 2)
    metrics["hyperparameters"] = {
        "learning_rate": cfg["training"]["learning_rate"],
        "batch_size": cfg["training"]["batch_size"],
        "dropout": cfg["training"].get("dropout"),
        "epochs_configured": cfg["training"]["epochs"],
        "epochs_trained": len(history.get("train_loss", [])),
    }

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(
            {k: v for k, v in metrics.items() if k != "classification_report"},
            f,
            indent=2,
        )

    cm_path = output_dir / "confusion_matrix.png"
    plot_confusion_matrix(y_true, y_pred, class_names, cm_path)

    history_path = output_dir / "training_history.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return {
        "metrics": metrics,
        "history": history,
        "best_checkpoint": best_ckpt,
        "output_dir": output_dir,
    }


def load_cnn_test_metrics(config: dict[str, Any]) -> dict[str, Any] | None:
    """Load existing custom CNN test metrics if evaluation was already run."""
    path = resolve_path(config["evaluation"]["metrics_file"])
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    data["model"] = "Custom CNN"
    data["training_time_seconds"] = None
    return data
