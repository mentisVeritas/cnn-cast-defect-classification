"""Single-image inference with probability output."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision import transforms

from src.model import CastDefectCNN

CLASS_NAMES_FALLBACK = ["defect", "normal"]


def _build_inference_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def load_model_for_inference(
    config: dict[str, Any],
    checkpoint_path: str | Path | None = None,
    device: torch.device | None = None,
) -> tuple[CastDefectCNN, list[str], torch.device]:
    if device is None:
        from src.utils import get_device

        device = get_device(config)

    from src.utils import resolve_path

    path = resolve_path(checkpoint_path or config["checkpoint"]["best_model"])
    if not path.exists():
        path = resolve_path(config["checkpoint"]["latest_checkpoint"])
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint found at {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    class_names = checkpoint.get("class_names", CLASS_NAMES_FALLBACK)

    model = CastDefectCNN(
        num_classes=len(class_names),
        dropout=config["training"].get("dropout", 0.3),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, class_names, device


def predict_image(
    image_path: str | Path,
    model: CastDefectCNN,
    class_names: list[str],
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    transform = _build_inference_transform(config["training"]["image_size"])
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

    pred_idx = int(probs.argmax())
    return {
        "predicted_class": class_names[pred_idx],
        "confidence": float(probs[pred_idx]),
        "probabilities": {class_names[i]: float(probs[i]) for i in range(len(class_names))},
        "image_path": str(image_path),
    }


def predict_pil_image(
    image: Image.Image,
    model: CastDefectCNN,
    class_names: list[str],
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Run inference on an in-memory PIL image (e.g. Streamlit upload)."""
    transform = _build_inference_transform(config["training"]["image_size"])
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

    pred_idx = int(probs.argmax())
    return {
        "predicted_class": class_names[pred_idx],
        "confidence": float(probs[pred_idx]),
        "probabilities": {class_names[i]: float(probs[i]) for i in range(len(class_names))},
    }
