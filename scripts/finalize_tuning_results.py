"""Aggregate outputs/tuning/experiment_*/metrics.json into summary files."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.visualization import plot_tuning_comparison


def main() -> None:
    tuning_root = ROOT / "outputs/tuning"
    all_metrics = []
    for exp_id in ("A", "B", "C", "D"):
        p = tuning_root / f"experiment_{exp_id}" / "metrics.json"
        if p.exists():
            with p.open(encoding="utf-8") as f:
                all_metrics.append(json.load(f))
        else:
            print(f"Missing {p}")

    if not all_metrics:
        raise SystemExit("No experiment metrics found.")

    rows = []
    for m in all_metrics:
        hp = m.get("hyperparameters", {})
        rows.append(
            {
                "experiment": m.get("experiment"),
                "learning_rate": hp.get("learning_rate"),
                "batch_size": hp.get("batch_size"),
                "dropout": hp.get("dropout"),
                "accuracy": m["accuracy"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1_score": m["f1_score"],
                "training_time_seconds": m.get("training_time_seconds"),
                "epochs_trained": hp.get("epochs_trained"),
            }
        )

    ranked = sorted(all_metrics, key=lambda x: x["f1_score"], reverse=True)
    best = ranked[0]
    summary = {
        "experiments": rows,
        "best": {
            "best_experiment": best.get("experiment"),
            "best_f1_score": best["f1_score"],
            "best_accuracy": best["accuracy"],
            "recommended_hyperparameters": best.get("hyperparameters"),
        },
    }

    with (tuning_root / "tuning_results.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    fields = list(rows[0].keys())
    with (tuning_root / "tuning_results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    plot_tuning_comparison(all_metrics, tuning_root / "tuning_comparison.png")
    print(f"Best: experiment {summary['best']['best_experiment']}")
    print(f"Wrote {tuning_root / 'tuning_results.csv'}")


if __name__ == "__main__":
    main()
