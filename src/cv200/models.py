from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torchvision import models


@dataclass(frozen=True)
class ModelConfig:
    arch: str  # "resnet18" | "resnet50" | ...
    num_classes: int
    pretrained: bool


class SmallCNN(nn.Module):
    """
    Simple convolutional baseline for Task 1 sanity checks.

    Uses a few conv/pool blocks followed by global average pooling, so it does not
    depend on a fixed input resolution (beyond being "reasonably large").
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /4
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /8
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


def build_model(cfg: ModelConfig) -> nn.Module:
    """
    Minimal model factory for portfolio purposes.
    Extensible: add EfficientNet / ViT / ConvNeXt etc.
    """
    arch = cfg.arch.lower()

    if arch == "small_cnn":
        return SmallCNN(num_classes=cfg.num_classes)

    if arch == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if cfg.pretrained else None
        m = models.resnet18(weights=weights)
        m.fc = nn.Linear(m.fc.in_features, cfg.num_classes)
        return m
    if arch == "resnet34":
        weights = models.ResNet34_Weights.DEFAULT if cfg.pretrained else None
        m = models.resnet34(weights=weights)
        m.fc = nn.Linear(m.fc.in_features, cfg.num_classes)
        return m
    if arch == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if cfg.pretrained else None
        m = models.resnet50(weights=weights)
        m.fc = nn.Linear(m.fc.in_features, cfg.num_classes)
        return m

    if arch == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if cfg.pretrained else None
        m = models.efficientnet_b0(weights=weights)
        in_features = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_features, cfg.num_classes)
        return m

    if arch == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if cfg.pretrained else None
        m = models.convnext_tiny(weights=weights)
        in_features = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_features, cfg.num_classes)
        return m

    if arch == "swin_b":
        weights = models.Swin_B_Weights.DEFAULT if cfg.pretrained else None
        m = models.swin_b(weights=weights)
        in_features = m.head.in_features
        m.head = nn.Linear(in_features, cfg.num_classes)
        return m

    if arch == "vit_b_16":
        weights = models.ViT_B_16_Weights.DEFAULT if cfg.pretrained else None
        m = models.vit_b_16(weights=weights)
        in_features = m.heads.head.in_features
        m.heads.head = nn.Linear(in_features, cfg.num_classes)
        return m

    raise ValueError(f"Unknown arch: {cfg.arch}")


@torch.inference_mode()
def topk_from_logits(logits: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    probs = torch.softmax(logits, dim=-1)
    return torch.topk(probs, k=k, dim=-1)
