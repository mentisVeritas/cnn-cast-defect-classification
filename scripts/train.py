"""Train model: python scripts/train.py"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import get_dataloaders
from src.train_engine import Trainer
from src.utils import ensure_output_dirs, get_device, load_config, set_seed, setup_logging
from src.visualization import save_all_training_plots


def main():
    config = load_config()
    set_seed(config["random_seed"])
    ensure_output_dirs(config)
    setup_logging(config)

    device = get_device(config)
    print(f"Device: {device}")

    train_loader, val_loader, _, class_names = get_dataloaders(config, device)
    trainer = Trainer(config, train_loader, val_loader, class_names, device)
    history = trainer.train()

    if history.get("train_loss"):
        save_all_training_plots(history, config)


if __name__ == "__main__":
    main()
