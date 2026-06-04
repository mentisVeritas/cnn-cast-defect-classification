"""Generate static deployment preview images (BTEC evidence).

These are reproducible matplotlib previews of the Streamlit UI layout.
For live screenshots, see docs/screenshots/README.md.

Usage: python scripts/generate_deployment_previews.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from src.inference import load_model_for_inference, predict_image
from src.utils import get_device, load_config, resolve_path


def _draw_app_frame(fig, title: str, body_fn) -> None:
    fig.patch.set_facecolor("#0e1117")
    gs = GridSpec(1, 2, width_ratios=[1, 2.2], wspace=0.08)
    ax_side = fig.add_subplot(gs[0])
    ax_main = fig.add_subplot(gs[1])
    for ax in (ax_side, ax_main):
        ax.set_facecolor("#262730")
        ax.axis("off")

    ax_side.text(0.05, 0.95, "Sidebar", color="white", fontsize=12, weight="bold", va="top")
    ax_main.text(0.05, 0.95, title, color="white", fontsize=14, weight="bold", va="top")
    body_fn(ax_side, ax_main)


def _save_home(out: Path) -> None:
    fig = plt.figure(figsize=(12, 6))

    def body(ax_side, ax_main):
        ax_side.text(0.05, 0.8, "Upload images", color="#fafafa", fontsize=10)
        ax_side.text(0.05, 0.65, "[ Choose files ]", color="#888", fontsize=9)
        ax_main.text(
            0.05,
            0.75,
            "Cast Defect Classifier\nCNN demo: prediction · feature maps · Grad-CAM",
            color="#fafafa",
            fontsize=11,
            va="top",
        )
        ax_main.text(
            0.05,
            0.45,
            "Upload one or more images in the sidebar.",
            color="#aaaaaa",
            fontsize=10,
            va="top",
        )

    _draw_app_frame(fig, "Cast Defect Classifier", body)
    fig.savefig(out / "01_home.png", dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def _save_prediction(out: Path, image_path: Path, label_hint: str, filename: str) -> None:
    from PIL import Image

    config = load_config()
    device = get_device(config)
    model, class_names, device = load_model_for_inference(config, device=device)
    result = predict_image(image_path, model, class_names, config, device)

    fig = plt.figure(figsize=(12, 6))

    def body(ax_side, ax_main):
        ax_side.text(0.05, 0.8, f"File: {image_path.name}", color="#fafafa", fontsize=9)
        ax_main.text(
            0.55,
            0.55,
            f"Prediction: {result['predicted_class'].upper()}\n"
            f"Confidence: {result['confidence']:.1%}",
            color="#00d4aa",
            fontsize=16,
            ha="center",
            va="center",
            weight="bold",
        )
        ax_ins = ax_main.inset_axes([0.05, 0.15, 0.35, 0.55])
        ax_ins.imshow(Image.open(image_path).convert("RGB"))
        ax_ins.axis("off")
        ax_ins.set_title(label_hint, color="white", fontsize=9)

    _draw_app_frame(fig, "Prediction", body)
    fig.savefig(out / filename, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    out = resolve_path("docs/screenshots")
    out.mkdir(parents=True, exist_ok=True)

    raw = resolve_path("data/raw_images")
    labels = resolve_path("data/label.csv")

    # Pick samples (same logic as analysis script)
    import csv

    normal_img = defect_img = None
    with labels.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = (row.get("image") or "").strip()
            lab = (row.get("choice") or "").strip().lower()
            p = raw / name
            if not p.is_file():
                continue
            if lab == "normal" and normal_img is None:
                normal_img = p
            if lab == "defect" and defect_img is None:
                defect_img = p
            if normal_img and defect_img:
                break

    if not normal_img or not defect_img:
        raise FileNotFoundError("Need sample images in data/raw_images")

    _save_home(out)
    _save_prediction(out, defect_img, "Defect sample", "02_defect_upload.png")
    _save_prediction(out, defect_img, "Defect — result", "03_defect_prediction.png")
    _save_prediction(out, normal_img, "Normal sample", "04_normal_upload.png")
    _save_prediction(out, normal_img, "Normal — result", "05_normal_prediction.png")

    print(f"Saved deployment previews to {out}")


if __name__ == "__main__":
    main()
