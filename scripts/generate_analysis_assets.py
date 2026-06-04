"""Generate feature maps, filters, and Grad-CAM figures for BTEC analysis.

Usage: python scripts/generate_analysis_assets.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image

from src.explain import (
    CONV_BLOCKS,
    extract_feature_maps,
    grad_cam,
    image_to_tensor,
    save_feature_map_heatmap,
    save_gradcam_figure,
    visualize_first_conv_filters,
)
from src.inference import load_model_for_inference
from src.utils import get_device, load_config, resolve_path, set_seed


def _find_sample_image(raw_dir: Path, label: str, labels_file: Path) -> Path | None:
    """Return path to first image with given label from label.csv."""
    with labels_file.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row.get("choice") or "").strip().lower() == label:
                name = (row.get("image") or "").strip()
                if name:
                    path = raw_dir / name
                    if path.is_file():
                        return path
    return None


def main() -> None:
    config = load_config()
    set_seed(config["random_seed"])
    device = get_device(config)

    raw_dir = resolve_path(config["data"]["raw_images_dir"])
    labels_file = resolve_path(config["data"]["labels_file"])
    out_maps = resolve_path("outputs/feature_maps")
    out_filters = resolve_path("outputs/filters")
    out_gradcam = resolve_path("outputs/gradcam")

    normal_path = _find_sample_image(raw_dir, "normal", labels_file)
    defect_path = _find_sample_image(raw_dir, "defect", labels_file)
    if not normal_path or not defect_path:
        raise FileNotFoundError(
            "Could not find sample normal/defect images. Run unpack + ensure data/raw_images exists."
        )

    model, class_names, device = load_model_for_inference(config, device=device)
    defect_idx = class_names.index("defect")
    normal_idx = class_names.index("normal")

    samples = {
        "normal": Image.open(normal_path).convert("RGB"),
        "defect": Image.open(defect_path).convert("RGB"),
    }

    print("Saving feature maps...")
    for label, image in samples.items():
        tensor = image_to_tensor(image, config, device)
        for block, (layer_key, _) in CONV_BLOCKS.items():
            fmap = extract_feature_maps(model, tensor, layer_key)
            save_feature_map_heatmap(
                fmap,
                out_maps / f"{label}_{block}.png",
                title=f"{label} — {block} mean activation",
            )

    print("Saving conv1 filters...")
    visualize_first_conv_filters(model, out_filters / "conv1_filters.png")

    print("Saving Grad-CAM...")
    for label, image in samples.items():
        tensor = image_to_tensor(image, config, device)
        target = defect_idx if label == "defect" else normal_idx
        heatmap = grad_cam(model, tensor, target, "conv3 (96 ch)")
        save_gradcam_figure(
            image,
            heatmap,
            out_gradcam / f"{label}_gradcam.png",
            title=f"Grad-CAM → {label}",
        )

    print(f"Done.\n  {out_maps}\n  {out_filters}\n  {out_gradcam}")


if __name__ == "__main__":
    main()
