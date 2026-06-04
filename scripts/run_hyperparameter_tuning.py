"""Compare learning rates 0.001 vs 0.0005 on the custom CNN.

Usage: python scripts/run_hyperparameter_tuning.py [--skip-trained] [--epochs N]
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments import run_learning_rate_tuning
from src.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-run LR tuning (A=0.001, B=0.0005)")
    parser.add_argument("--skip-trained", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    run_learning_rate_tuning(
        load_config(),
        skip_trained=args.skip_trained,
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
