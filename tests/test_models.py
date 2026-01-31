from __future__ import annotations

import pytest
import torch

from cv200.models import ModelConfig, build_model


@pytest.mark.parametrize(
    "arch,image_size", [("small_cnn", 64), ("resnet18", 224), ("efficientnet_b0", 224)]
)
def test_model_forward_output_shape(arch: str, image_size: int) -> None:
    num_classes = 7
    model = build_model(ModelConfig(arch=arch, num_classes=num_classes, pretrained=False))
    model.eval()

    x = torch.randn(2, 3, image_size, image_size)
    with torch.inference_mode():
        y = model(x)
    assert tuple(y.shape) == (2, num_classes)
