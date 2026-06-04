"""Run hyperparameter tuning experiments A–D on the custom CNN.

Usage: python scripts/run_hyperparameter_tuning.py
       python scripts/run_hyperparameter_tuning.py --skip-trained  # skip if metrics exist
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment_runner import apply_training_overrides, load_cnn_test_metrics, run_train_and_evaluate
from src.model import build_model
from src.utils import get_device, load_config, resolve_path, set_seed
from src.visualization import plot_tuning_comparison

EXPERIMENTS = [
    {"id": "A", "lr": 0.001, "batch_size": 16, "dropout": 0.4},
    {"id": "B", "lr": 0.0005, "batch_size": 16, "dropout": 0.4},
    {"id": "C", "lr": 0.001, "batch_size": 32, "dropout": 0.4},
    {"id": "D", "lr": 0.001, "batch_size": 16, "dropout": 0.5},
]


def _metrics_row(metrics: dict) -> dict:
    return {
        "experiment": metrics.get("experiment"),
        "learning_rate": metrics["hyperparameters"]["learning_rate"],
        "batch_size": metrics["hyperparameters"]["batch_size"],
        "dropout": metrics["hyperparameters"]["dropout"],
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
        "training_time_seconds": metrics.get("training_time_seconds"),
        "epochs_trained": metrics["hyperparameters"].get("epochs_trained"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-trained",
        action="store_true",
        help="Skip experiment folder if metrics.json already exists",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override training epochs for all tuning runs",
    )
    args = parser.parse_args()

    config = load_config()
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    set_seed(config["random_seed"])
    device = get_device(config)
    tuning_root = resolve_path("outputs/tuning")
    tuning_root.mkdir(parents=True, exist_ok=True)

    all_metrics: list[dict] = []

    for exp in EXPERIMENTS:
        exp_id = exp["id"]
        out_dir = tuning_root / f"experiment_{exp_id}"
        metrics_file = out_dir / "metrics.json"

        if args.skip_trained and metrics_file.exists():
            print(f"Skipping experiment {exp_id} (already trained).")
            with metrics_file.open(encoding="utf-8") as f:
                all_metrics.append(json.load(f))
            continue

        # Experiment A matches default config — reuse existing CNN eval if available
        if exp_id == "A" and not metrics_file.exists():
            existing = load_cnn_test_metrics(config)
            if existing and config["training"]["learning_rate"] == exp["lr"]:
                if config["training"]["batch_size"] == exp["batch_size"]:
                    if config["training"]["dropout"] == exp["dropout"]:
                        print("Experiment A: using existing custom CNN test metrics.")
                        row = {
                            "experiment": "A",
                            "accuracy": existing["accuracy"],
                            "precision": existing["precision"],
                            "recall": existing["recall"],
                            "f1_score": existing["f1_score"],
                            "training_time_seconds": None,
                            "class_names": existing["class_names"],
                            "confusion_matrix": existing["confusion_matrix"],
                            "hyperparameters": {
                                "learning_rate": exp["lr"],
                                "batch_size": exp["batch_size"],
                                "dropout": exp["dropout"],
                                "epochs_configured": config["training"]["epochs"],
                                "epochs_trained": None,
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

        print(f"\n=== Experiment {exp_id}: lr={exp['lr']} batch={exp['batch_size']} dropout={exp['dropout']} ===")
        if resume:
            print(f"Checkpoint found — resuming from {latest_ckpt}")
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

    # Best configuration by F1 on test set
    ranked = sorted(all_metrics, key=lambda m: m["f1_score"], reverse=True)
    best = ranked[0]
    best_summary = {
        "best_experiment": best.get("experiment"),
        "best_f1_score": best["f1_score"],
        "best_accuracy": best["accuracy"],
        "recommended_hyperparameters": best.get("hyperparameters"),
    }

    rows = [_metrics_row(m) for m in all_metrics]
    json_path = tuning_root / "tuning_results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"experiments": rows, "best": best_summary}, f, indent=2)

    csv_path = tuning_root / "tuning_results.csv"
    fieldnames = [
        "experiment",
        "learning_rate",
        "batch_size",
        "dropout",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "training_time_seconds",
        "epochs_trained",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    plot_tuning_comparison(all_metrics, tuning_root / "tuning_comparison.png")

    print(f"\nTuning complete. Best experiment: {best_summary['best_experiment']}")
    print(f"Results: {csv_path}")


if __name__ == "__main__":
    main()
