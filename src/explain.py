"""CNN explainability: feature maps and Grad-CAM."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from src.model import CastDefectCNN

# Conv2d layers inside CastDefectCNN.features (indices in Sequential)
CONV_LAYERS: dict[str, int] = {
    "conv1 (32 ch)": 0,
    "conv2 (64 ch)": 4,
    "conv3 (96 ch)": 8,
}

# Short names for saved figure filenames (assignment outputs)
CONV_BLOCKS: dict[str, tuple[str, int]] = {
    "conv1": ("conv1 (32 ch)", 0),
    "conv2": ("conv2 (64 ch)", 4),
    "conv3": ("conv3 (96 ch)", 8),
}


def build_inference_transform(image_size: int) -> transforms.Compose:
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


def image_to_tensor(
    image: Image.Image,
    config: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    transform = build_inference_transform(config["training"]["image_size"])
    return transform(image.convert("RGB")).unsqueeze(0).to(device)


def predict_tensor(
    model: CastDefectCNN,
    tensor: torch.Tensor,
    class_names: list[str],
) -> dict[str, Any]:
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx = int(probs.argmax())
    return {
        "predicted_class": class_names[pred_idx],
        "predicted_index": pred_idx,
        "confidence": float(probs[pred_idx]),
        "probabilities": {class_names[i]: float(probs[i]) for i in range(len(class_names))},
    }


def extract_feature_maps(
    model: CastDefectCNN,
    tensor: torch.Tensor,
    layer_key: str = "conv3 (96 ch)",
) -> np.ndarray:
    """Mean activation per spatial location, shape (H, W)."""
    layer_idx = CONV_LAYERS[layer_key]
    activations: list[torch.Tensor] = []

    def hook(_module, _inp, out):
        activations.append(out.detach())

    handle = model.features[layer_idx].register_forward_hook(hook)
    try:
        model.eval()
        with torch.no_grad():
            model(tensor)
    finally:
        handle.remove()

    feat = activations[0][0]  # (C, H, W)
    return feat.mean(dim=0).cpu().numpy()


def extract_all_feature_maps(
    model: CastDefectCNN,
    tensor: torch.Tensor,
) -> dict[str, np.ndarray]:
    return {name: extract_feature_maps(model, tensor, name) for name in CONV_LAYERS}


def grad_cam(
    model: CastDefectCNN,
    tensor: torch.Tensor,
    target_index: int,
    layer_key: str = "conv3 (96 ch)",
) -> np.ndarray:
    """Grad-CAM heatmap normalized to [0, 1], shape (H, W)."""
    layer_idx = CONV_LAYERS[layer_key]
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def fwd_hook(_module, _inp, out):
        activations.append(out)

    def bwd_hook(_module, _grad_in, grad_out):
        gradients.append(grad_out[0])

    fwd = model.features[layer_idx].register_forward_hook(fwd_hook)
    bwd = model.features[layer_idx].register_full_backward_hook(bwd_hook)

    model.eval()
    tensor = tensor.clone().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    logits = model(tensor)
    score = logits[0, target_index]
    score.backward()

    fwd.remove()
    bwd.remove()

    acts = activations[0][0]  # (C, H, W)
    grads = gradients[0][0]
    weights = grads.mean(dim=(1, 2))
    cam = (weights[:, None, None] * acts).sum(dim=0)
    cam = torch.relu(cam).detach().cpu().numpy()
    cam -= cam.min()
    if cam.max() > 0:
        cam /= cam.max()
    return cam


def overlay_heatmap(
    image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.45,
) -> Image.Image:
    """Resize heatmap to image size and blend with RGB image."""
    import matplotlib.cm as cm

    image = image.convert("RGB")
    w, h = image.size
    heat_resized = np.array(
        Image.fromarray((heatmap * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    ) / 255.0
    colored = cm.jet(heat_resized)[:, :, :3]
    base = np.array(image, dtype=np.float32) / 255.0
    blended = (1 - alpha) * base + alpha * colored
    blended = (np.clip(blended, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(blended)


def save_feature_map_heatmap(
    fmap: np.ndarray,
    save_path: str | Path,
    title: str,
) -> None:
    """Save a single-channel activation heatmap to PNG."""
    import matplotlib.pyplot as plt

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(fmap, cmap="viridis")
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def save_gradcam_figure(
    image: Image.Image,
    heatmap: np.ndarray,
    save_path: str | Path,
    title: str,
) -> None:
    """Save original + Grad-CAM overlay side by side."""
    import matplotlib.pyplot as plt

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    overlay = overlay_heatmap(image, heatmap, alpha=0.5)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(image)
    axes[0].set_title("Input")
    axes[0].axis("off")
    axes[1].imshow(overlay)
    axes[1].set_title(title)
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def visualize_first_conv_filters(
    model: CastDefectCNN,
    save_path: str | Path,
    max_filters: int = 32,
) -> None:
    """Plot weights of the first Conv2d layer (learned filters)."""
    import matplotlib.pyplot as plt

    conv1 = model.features[0]
    if not isinstance(conv1, torch.nn.Conv2d):
        raise TypeError("Expected first features layer to be Conv2d")

    weights = conv1.weight.detach().cpu()  # (out_ch, in_ch, k, k)
    n = min(max_filters, weights.shape[0])
    cols = 8
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    axes_flat = np.atleast_1d(axes).flat

    for i in range(rows * cols):
        ax = axes_flat[i]
        ax.axis("off")
        if i >= n:
            continue
        w = weights[i]
        # RGB from 3 input channels
        filt = w.permute(1, 2, 0).numpy()
        filt = (filt - filt.min()) / (filt.max() - filt.min() + 1e-8)
        ax.imshow(filt)

    fig.suptitle("Conv Block 1 — learned filters (first 32 of 32)", fontsize=11)
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
