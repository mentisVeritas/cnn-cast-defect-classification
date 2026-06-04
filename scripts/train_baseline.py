"""Train ResNet18 baseline and build comparison table with custom CNN.

Usage: python scripts/train_baseline.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import torch
from torch.optim import Adam

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baseline import build_resnet18_baseline
from src.dataset import get_dataloaders
from src.evaluate import collect_predictions
from src.experiment_runner import load_cnn_test_metrics, run_train_and_evaluate
from src.train_engine import load_checkpoint
from src.utils import get_device, load_config, resolve_path, set_seed
from src.visualization import plot_confusion_matrix, plot_metrics_comparison


def _write_comparison_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["Model", "Accuracy", "Precision", "Recall", "F1"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
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


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    config = load_config()
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    set_seed(config["random_seed"])
    device = get_device(config)

    out_dir = resolve_path("outputs/baseline")
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
    plot_confusion_matrix(y_true, y_pred, class_names, out_dir / "resnet18_confusion_matrix.png")

    cnn_metrics = load_cnn_test_metrics(config)
    if cnn_metrics is None:
        print("Warning: custom CNN metrics not found. Run scripts/evaluate.py first.")
        rows = [
            {
                "model": "ResNet18",
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
            }
        ]
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

    _write_comparison_csv(rows, out_dir / "model_comparison.csv")
    plot_metrics_comparison(rows, out_dir / "model_comparison.png", title="Custom CNN vs ResNet18")

    print(f"Baseline done. Metrics: {metrics_path}")
    print(f"Comparison: {out_dir / 'model_comparison.csv'}")


if __name__ == "__main__":
    main()
