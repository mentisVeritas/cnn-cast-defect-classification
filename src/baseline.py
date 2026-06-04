"""ResNet18 transfer-learning baseline for comparison with the custom CNN."""

from __future__ import annotations

import torch.nn as nn
from torchvision import models


def build_resnet18_baseline(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """ResNet18 with ImageNet weights; final FC layer replaced for binary classes."""
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
