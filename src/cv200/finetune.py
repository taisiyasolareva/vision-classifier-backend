from __future__ import annotations

"""
Fine-tuning utilities (single source of truth).

This module exists to keep `cv200.train` readable and to make fine-tuning behavior
explicit, testable, and consistent across architectures (head-only → partial unfreeze → full fine-tune).
"""

from torch import nn


def _head_module(model: nn.Module) -> nn.Module | None:
    """
    Best-effort classifier/head discovery across supported torchvision families.

    Supported (minimum):
    - ResNet*:          `model.fc`
    - EfficientNet-B0:  `model.classifier`
    - ConvNeXt-Tiny:    `model.classifier`
    - Swin-B:           `model.head`
    - ViT-B/16:         `model.heads`
    - SmallCNN:         `model.classifier`
    """
    if hasattr(model, "fc"):
        return getattr(model, "fc")
    if hasattr(model, "classifier"):
        return getattr(model, "classifier")
    if hasattr(model, "head"):
        return getattr(model, "head")
    if hasattr(model, "heads"):
        return getattr(model, "heads")
    return None


def split_backbone_and_head(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """
    Split model parameters into (backbone_params, head_params).

    Returns parameter lists (not modules) to support:
    - optimizer parameter groups (discriminative LRs)
    - consistent filtering by `requires_grad`
    """
    head = _head_module(model)
    if head is None:
        # Unknown model; treat everything as backbone.
        return list(model.parameters()), []

    head_params = list(head.parameters())
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    return backbone_params, head_params


def freeze_backbone(model: nn.Module) -> None:
    """
    Freeze all backbone parameters; keep head trainable.

    "Frozen" strategy (Task 2 baseline): train only the classifier/head.
    """
    for p in model.parameters():
        p.requires_grad = False

    head = _head_module(model)
    if head is not None:
        for p in head.parameters():
            p.requires_grad = True


def unfreeze_last_n(model: nn.Module, n: int) -> None:
    """
    Partial fine-tuning: unfreeze the last N "blocks/stages" of the backbone + head.

    What "block" means (best-effort, model-family aware):
    - ResNet*: unfreeze `layer4` (N=1), `layer3` (N=2), `layer2` (N=3), `layer1` (N=4)
    - EfficientNet/ConvNeXt/Swin: unfreeze the last N modules of `model.features` (Sequential)
    - ViT: unfreeze the last N encoder layers in `model.encoder.layers` if present

    Notes:
    - This is intentionally conservative and explicit; for research-grade control,
      you can add architecture-specific handling here.
    """
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}")

    # Start from a clean frozen state (head trainable).
    freeze_backbone(model)

    # ResNet family: layer1..layer4 are the main stages.
    if hasattr(model, "layer4"):
        stages: list[nn.Module] = []
        for name in ("layer4", "layer3", "layer2", "layer1"):
            if hasattr(model, name):
                stages.append(getattr(model, name))
        for stage in stages[:n]:
            for p in stage.parameters():
                p.requires_grad = True
        return

    # EfficientNet/ConvNeXt/Swin: many torchvision models expose a Sequential `.features`.
    features = getattr(model, "features", None)
    if isinstance(features, nn.Sequential) and len(features) > 0:
        # Unfreeze last N modules of the sequential.
        for block in list(features)[-n:]:
            for p in block.parameters():
                p.requires_grad = True
        return

    # ViT: encoder.layers is often an iterable container.
    encoder = getattr(model, "encoder", None)
    layers = getattr(encoder, "layers", None) if encoder is not None else None
    if layers is not None and hasattr(layers, "__len__"):
        # Try to slice like a list/ModuleList.
        try:
            last_layers = list(layers)[-n:]
        except TypeError:
            last_layers = []
        for layer in last_layers:
            for p in layer.parameters():
                p.requires_grad = True
        return

    # Fallback: we can't identify backbone blocks; keep head-only training.
    return


def count_trainable_params(model: nn.Module) -> int:
    """Count trainable parameters (absolute count)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_params(model: nn.Module) -> int:
    """Count total parameters (absolute count)."""
    return sum(p.numel() for p in model.parameters())
