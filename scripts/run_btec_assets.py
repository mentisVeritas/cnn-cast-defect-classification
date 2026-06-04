#!/usr/bin/env python3
"""Run all BTEC analysis asset generators (no full retraining).

Usage:
    python scripts/run_btec_assets.py
    python scripts/run_btec_assets.py --with-training  # includes tuning + baseline
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list[str]) -> None:
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--with-training",
        action="store_true",
        help="Also run hyperparameter tuning and ResNet18 baseline (long)",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Epochs for training scripts")
    args = parser.parse_args()

    py = sys.executable
    run([py, "scripts/generate_analysis_assets.py"])
    run([py, "scripts/generate_deployment_previews.py"])
    run([py, "scripts/append_btec_notebook.py"])

    if args.with_training:
        tune = [py, "scripts/run_hyperparameter_tuning.py"]
        base = [py, "scripts/train_baseline.py"]
        if args.epochs:
            tune += ["--epochs", str(args.epochs)]
            base += ["--epochs", str(args.epochs)]
        run(tune)
        run(base)

    print("\nBTEC assets ready. See outputs/ and docs/")


if __name__ == "__main__":
    main()
