"""Inference: python scripts/inference.py --image path/to/image.jpg"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference import load_model_for_inference, predict_image
from src.utils import get_device, load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    config = load_config()
    device = get_device(config)
    model, class_names, device = load_model_for_inference(config, device=device)

    result = predict_image(args.image, model, class_names, config, device)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
