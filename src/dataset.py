"""Dataset loaders and transforms."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.utils import get_dataloader_kwargs, get_device, resolve_path


def _build_transforms(config: dict[str, Any], split: str) -> transforms.Compose:
    image_size = config["training"]["image_size"]
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if split == "train" and config.get("augmentation", {}).get("train"):
        aug = config["augmentation"]["train"]
        steps: list = [
            transforms.Resize((image_size, image_size)),
        ]
        if aug.get("random_horizontal_flip", False):
            steps.append(transforms.RandomHorizontalFlip())
        if aug.get("random_rotation", 0):
            steps.append(transforms.RandomRotation(aug["random_rotation"]))
        steps.extend([transforms.ToTensor(), normalize])
        return transforms.Compose(steps)

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )


def get_dataloaders(
    config: dict[str, Any],
    device=None,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    if device is None:
        device = get_device(config)

    base = resolve_path(config["data"]["processed_dir"])
    data_dirs = {"train": base / "train", "val": base / "val", "test": base / "test"}

    for name, path in data_dirs.items():
        if not path.exists():
            raise FileNotFoundError(
                f"{name} not found: {path}. Run: python scripts/split_dataset.py"
            )

    train_ds = datasets.ImageFolder(
        data_dirs["train"], transform=_build_transforms(config, "train")
    )
    val_ds = datasets.ImageFolder(
        data_dirs["val"], transform=_build_transforms(config, "val")
    )
    test_ds = datasets.ImageFolder(
        data_dirs["test"], transform=_build_transforms(config, "test")
    )

    kwargs = get_dataloader_kwargs(config, device)
    train_loader = DataLoader(train_ds, shuffle=True, **kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **kwargs)

    return train_loader, val_loader, test_loader, train_ds.classes
