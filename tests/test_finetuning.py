from __future__ import annotations

from cv200.finetune import (
    count_trainable_params,
    freeze_backbone,
    split_backbone_and_head,
    unfreeze_last_n,
)
from cv200.models import ModelConfig, build_model


def test_freeze_backbone_makes_only_head_trainable() -> None:
    model = build_model(ModelConfig(arch="resnet50", num_classes=3, pretrained=False))
    freeze_backbone(model)
    backbone_params, head_params = split_backbone_and_head(model)

    assert head_params, "Expected to detect head parameters"
    assert backbone_params, "Expected to detect backbone parameters"

    assert all(p.requires_grad for p in head_params), "Head params should be trainable"
    assert all(not p.requires_grad for p in backbone_params), "Backbone params should be frozen"


def test_unfreeze_last_n_monotonic_increases_trainable_params_for_resnet() -> None:
    model = build_model(ModelConfig(arch="resnet50", num_classes=3, pretrained=False))

    freeze_backbone(model)
    n_frozen = count_trainable_params(model)

    unfreeze_last_n(model, 1)
    n1 = count_trainable_params(model)

    unfreeze_last_n(model, 2)
    n2 = count_trainable_params(model)

    # Full fine-tune: mark everything trainable
    for p in model.parameters():
        p.requires_grad = True
    n_full = count_trainable_params(model)

    assert n_frozen > 0, "Even frozen mode should keep the head trainable"
    assert n_frozen < n1 < n2 <= n_full, "Expected monotonic increase with deeper unfreezing"


def test_split_backbone_and_head_is_disjoint_and_covers_all_params() -> None:
    model = build_model(ModelConfig(arch="resnet50", num_classes=3, pretrained=False))
    backbone_params, head_params = split_backbone_and_head(model)

    ids_backbone = {id(p) for p in backbone_params}
    ids_head = {id(p) for p in head_params}
    ids_all = {id(p) for p in model.parameters()}

    assert ids_backbone.isdisjoint(ids_head), "Backbone and head params must not overlap"
    assert ids_backbone.union(ids_head) == ids_all, "Split must cover all parameters exactly once"
