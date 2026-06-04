"""Feature map extraction for the custom CNN (conv blocks 1–3)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image

from src.inference import _build_inference_transform
from src.model import CastDefectCNN

# Index of Conv2d inside model.features (before pool).
CONV_BLOCKS: dict[str, tuple[int, str]] = {
    "conv1": (0, "Block 1 · 32 filters"),
    "conv2": (4, "Block 2 · 64 filters"),
    "conv3": (8, "Block 3 · 96 filters"),
}


def image_to_tensor(
    image: Image.Image,
    config: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    """Preprocess PIL image like inference (224, ImageNet norm)."""
    transform = _build_inference_transform(config["training"]["image_size"])
    return transform(image.convert("RGB")).unsqueeze(0).to(device)


def extract_feature_map(
    model: CastDefectCNN,
    tensor: torch.Tensor,
    block: str,
) -> np.ndarray:
    """
    Return mean activation heatmap (H, W) after the given conv block.

    block: one of conv1, conv2, conv3
    """
    if block not in CONV_BLOCKS:
        raise ValueError(f"Unknown block {block!r}. Use: {list(CONV_BLOCKS)}")

    layer_idx, _ = CONV_BLOCKS[block]
    captured: list[torch.Tensor] = []

    def hook(_module, _inputs, output: torch.Tensor) -> None:
        captured.append(output.detach())

    handle = model.features[layer_idx].register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(tensor)
    finally:
        handle.remove()

    activations = captured[0][0].cpu().numpy()  # (C, H, W)
    return activations.mean(axis=0)


def extract_all_feature_maps(
    model: CastDefectCNN,
    tensor: torch.Tensor,
) -> dict[str, np.ndarray]:
    """Mean activation maps for conv1, conv2, conv3."""
    return {name: extract_feature_map(model, tensor, name) for name in CONV_BLOCKS}
