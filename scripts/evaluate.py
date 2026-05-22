"""Evaluate on test set: python scripts/evaluate.py"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import get_dataloaders
from src.evaluate import evaluate_model
from src.inference import load_model_for_inference
from src.utils import get_device, load_config, setup_logging
from src.visualization import save_confusion_matrix_plot


def main():
    config = load_config()
    setup_logging(config)
    device = get_device(config)

    _, _, test_loader, class_names = get_dataloaders(config, device)
    model, class_names, device = load_model_for_inference(config, device=device)

    metrics, y_true, y_pred = evaluate_model(model, test_loader, device, class_names, config)
    save_confusion_matrix_plot(y_true, y_pred, class_names, config)

    print("\nTest set results:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1_score']:.4f}")
    print(metrics["classification_report"])


if __name__ == "__main__":
    main()
