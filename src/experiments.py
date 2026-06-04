"""Baseline (ResNet18) and learning-rate tuning for the custom CNN.

All experiment logic lives here; scripts/ only provide thin CLI entry points.
"""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
from typing import Any

import torch.nn as nn
from torch.optim import Adam

from src.baseline import build_resnet18_baseline
from src.dataset import get_dataloaders
from src.evaluate import collect_predictions, compute_metrics
from src.model import build_model
from src.train_engine import Trainer, load_checkpoint
from src.utils import get_device, load_config, resolve_path, set_seed
from src.visualization import plot_confusion_matrix, plot_metrics_comparison, plot_tuning_comparison

# Two-run LR comparison (batch size and dropout fixed from config).
LR_TUNING_EXPERIMENTS: tuple[dict[str, Any], ...] = (
    {"id": "A", "lr": 0.001, "batch_size": 16, "dropout": 0.4},
    {"id": "B", "lr": 0.0005, "batch_size": 16, "dropout": 0.4},
)

BASELINE_OUTPUT_DIR = "outputs/baseline"
TUNING_OUTPUT_DIR = "outputs/tuning"


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
    """Train on train/val, evaluate on test, save metrics.json and confusion matrix."""
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

    plot_confusion_matrix(y_true, y_pred, class_names, output_dir / "confusion_matrix.png")

    with (output_dir / "training_history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return {
        "metrics": metrics,
        "history": history,
        "best_checkpoint": best_ckpt,
        "output_dir": output_dir,
    }


def load_cnn_training_meta(config: dict[str, Any]) -> dict[str, Any]:
    """Read training_time_seconds / epochs_trained from main CNN training_history.json."""
    history_path = resolve_path(config["evaluation"]["metrics_file"]).parent / "training_history.json"
    if not history_path.exists():
        return {}
    with history_path.open(encoding="utf-8") as f:
        data = json.load(f)
    epochs = data.get("epochs_trained")
    if epochs is None and data.get("train_loss"):
        epochs = len(data["train_loss"])
    return {
        "training_time_seconds": data.get("training_time_seconds"),
        "epochs_trained": epochs,
    }


def load_cnn_test_metrics(config: dict[str, Any]) -> dict[str, Any] | None:
    """Load custom CNN test metrics from outputs/metrics/evaluation_metrics.json."""
    path = resolve_path(config["evaluation"]["metrics_file"])
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    data["model"] = "Custom CNN"
    meta = load_cnn_training_meta(config)
    data["training_time_seconds"] = meta.get("training_time_seconds")
    data["epochs_trained"] = meta.get("epochs_trained")
    return data


def _write_model_comparison_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["Model", "Accuracy", "Precision", "Recall", "F1"]
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "Model": r["model"],
                    "Accuracy": f"{r['accuracy']:.4f}",
                    "Precision": f"{r['precision']:.4f}",
                    "Recall": f"{r['recall']:.4f}",
                    "F1": f"{r['f1_score']:.4f}",
                }
            )


def _tuning_results_row(metrics: dict) -> dict:
    hp = metrics["hyperparameters"]
    return {
        "experiment": metrics.get("experiment"),
        "learning_rate": hp["learning_rate"],
        "batch_size": hp["batch_size"],
        "dropout": hp["dropout"],
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
        "training_time_seconds": metrics.get("training_time_seconds"),
        "epochs_trained": hp.get("epochs_trained"),
    }


def train_resnet18_baseline(
    config: dict[str, Any] | None = None,
    *,
    epochs: int | None = None,
    output_dir: str | Path = BASELINE_OUTPUT_DIR,
) -> dict[str, Path]:
    """
    Train ResNet18 (transfer learning), evaluate on test set, compare with custom CNN.

    Writes:
        resnet18_metrics.json, resnet18_confusion_matrix.png,
        model_comparison.csv, model_comparison.png
    """
    config = config or load_config()
    if epochs is not None:
        config["training"]["epochs"] = epochs
    set_seed(config["random_seed"])
    device = get_device(config)
    out_dir = resolve_path(output_dir)

    print("Training ResNet18 baseline (transfer learning)...")
    model = build_resnet18_baseline(num_classes=2, pretrained=True)
    result = run_train_and_evaluate(
        config,
        model,
        out_dir,
        experiment_name="ResNet18",
        device=device,
        resume_training=False,
    )
    metrics = result["metrics"]

    metrics_path = out_dir / "resnet18_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(
            {k: v for k, v in metrics.items() if k != "classification_report"},
            f,
            indent=2,
        )

    _, _, test_loader, class_names = get_dataloaders(config, device)
    optimizer = Adam(model.parameters(), lr=config["training"]["learning_rate"])
    load_checkpoint(result["best_checkpoint"], model, optimizer, device)
    y_true, y_pred, _ = collect_predictions(model, test_loader, device)
    cm_path = out_dir / "resnet18_confusion_matrix.png"
    plot_confusion_matrix(y_true, y_pred, class_names, cm_path)

    cnn_metrics = load_cnn_test_metrics(config)
    if cnn_metrics is None:
        print("Warning: run scripts/evaluate.py first for full CNN vs ResNet18 table.")
        rows = [{"model": "ResNet18", **{k: metrics[k] for k in ("accuracy", "precision", "recall", "f1_score")}}]
    else:
        rows = [
            {
                "model": "Custom CNN",
                "accuracy": cnn_metrics["accuracy"],
                "precision": cnn_metrics["precision"],
                "recall": cnn_metrics["recall"],
                "f1_score": cnn_metrics["f1_score"],
            },
            {
                "model": "ResNet18",
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
            },
        ]

    csv_path = out_dir / "model_comparison.csv"
    _write_model_comparison_csv(rows, csv_path)
    plot_path = out_dir / "model_comparison.png"
    plot_metrics_comparison(rows, plot_path, title="Custom CNN vs ResNet18")

    print(f"Baseline metrics: {metrics_path}")
    print(f"Comparison table: {csv_path}")
    return {
        "metrics_json": metrics_path,
        "comparison_csv": csv_path,
        "comparison_plot": plot_path,
        "confusion_matrix": cm_path,
    }


def run_learning_rate_tuning(
    config: dict[str, Any] | None = None,
    *,
    skip_trained: bool = False,
    epochs: int | None = None,
    output_dir: str | Path = TUNING_OUTPUT_DIR,
) -> dict[str, Path]:
    """
    Compare lr=0.001 (A) vs lr=0.0005 (B) on the custom CNN.

    Writes: tuning_results.csv, tuning_results.json, tuning_comparison.png
    """
    config = config or load_config()
    if epochs is not None:
        config["training"]["epochs"] = epochs
    set_seed(config["random_seed"])
    device = get_device(config)
    tuning_root = resolve_path(output_dir)
    tuning_root.mkdir(parents=True, exist_ok=True)

    all_metrics: list[dict] = []

    for exp in LR_TUNING_EXPERIMENTS:
        exp_id = exp["id"]
        out_dir = tuning_root / f"experiment_{exp_id}"
        metrics_file = out_dir / "metrics.json"

        if skip_trained and metrics_file.exists():
            print(f"Skipping experiment {exp_id} (metrics exist).")
            with metrics_file.open(encoding="utf-8") as f:
                all_metrics.append(json.load(f))
            continue

        if exp_id == "A" and not metrics_file.exists():
            existing = load_cnn_test_metrics(config)
            if (
                existing
                and config["training"]["learning_rate"] == exp["lr"]
                and config["training"]["batch_size"] == exp["batch_size"]
                and config["training"]["dropout"] == exp["dropout"]
            ):
                print("Experiment A: reusing main CNN test metrics (same as scripts/train.py).")
                meta = load_cnn_training_meta(config)
                row = {
                    "experiment": "A",
                    "accuracy": existing["accuracy"],
                    "precision": existing["precision"],
                    "recall": existing["recall"],
                    "f1_score": existing["f1_score"],
                    "training_time_seconds": meta.get("training_time_seconds"),
                    "class_names": existing["class_names"],
                    "confusion_matrix": existing["confusion_matrix"],
                    "hyperparameters": {
                        "learning_rate": exp["lr"],
                        "batch_size": exp["batch_size"],
                        "dropout": exp["dropout"],
                        "epochs_configured": config["training"]["epochs"],
                        "epochs_trained": meta.get("epochs_trained"),
                    },
                }
                out_dir.mkdir(parents=True, exist_ok=True)
                with metrics_file.open("w", encoding="utf-8") as f:
                    json.dump(row, f, indent=2)
                all_metrics.append(row)
                continue

        cfg = apply_training_overrides(
            config,
            learning_rate=exp["lr"],
            batch_size=exp["batch_size"],
            dropout=exp["dropout"],
        )
        latest_ckpt = out_dir / "checkpoints" / "latest_checkpoint.pth"
        resume = latest_ckpt.exists() and not metrics_file.exists()

        print(f"\n=== Experiment {exp_id}: lr={exp['lr']} ===")
        model = build_model(cfg, num_classes=2)
        result = run_train_and_evaluate(
            cfg,
            model,
            out_dir,
            experiment_name=exp_id,
            device=device,
            resume_training=resume,
        )
        all_metrics.append(result["metrics"])

    ranked = sorted(all_metrics, key=lambda m: m["f1_score"], reverse=True)
    best = ranked[0]
    best_summary = {
        "best_experiment": best.get("experiment"),
        "best_f1_score": best["f1_score"],
        "best_accuracy": best["accuracy"],
        "recommended_hyperparameters": best.get("hyperparameters"),
    }

    rows = [_tuning_results_row(m) for m in all_metrics]
    json_path = tuning_root / "tuning_results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"experiments": rows, "best": best_summary}, f, indent=2)

    csv_path = tuning_root / "tuning_results.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    plot_path = tuning_root / "tuning_comparison.png"
    plot_tuning_comparison(all_metrics, plot_path)

    print(f"\nBest experiment: {best_summary['best_experiment']} (F1={best_summary['best_f1_score']:.4f})")
    print(f"Results: {csv_path}")
    return {"results_csv": csv_path, "results_json": json_path, "comparison_plot": plot_path}
