"""Train ResNet18 baseline and build CNN comparison table.

Usage: python scripts/train_baseline.py [--epochs N]
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments import train_resnet18_baseline
from src.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="ResNet18 baseline vs custom CNN")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    train_resnet18_baseline(load_config(), epochs=args.epochs)


if __name__ == "__main__":
    main()
