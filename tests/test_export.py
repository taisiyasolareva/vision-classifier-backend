from __future__ import annotations

import sys
from pathlib import Path

import torch

from cv200.checkpointing import CheckpointMeta, save_checkpoint
from cv200.models import ModelConfig, build_model
from cv200.utils import save_json


def _make_dummy_checkpoint_dir(tmp_path: Path) -> tuple[Path, Path]:
    """
    Create a tiny checkpoint + metadata layout compatible with cv200.export.
    Returns: (ckpt_path, ckpt_dir)
    """
    ckpt_dir = tmp_path / "run"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    arch = "small_cnn"
    num_classes = 3
    image_size = 64

    model = build_model(ModelConfig(arch=arch, num_classes=num_classes, pretrained=False))
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1e-3)

    meta = CheckpointMeta(
        arch=arch,
        num_classes=num_classes,
        task="task1",
        pretrained=False,
        fine_tune_strategy="none",
        unfreeze_last_n=None,
        trainable_params_m=1.0,
        image_size=image_size,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        dataset_fingerprint="dummy",
        epoch=1,
    )
    ckpt_path = ckpt_dir / "checkpoint_best.pt"
    save_checkpoint(ckpt_path, model=model, optimizer=opt, epoch=1, meta=meta)

    # Required by API loader: labels.json + preprocess.json
    save_json(ckpt_dir / "labels.json", {"0": "0", "1": "1", "2": "2"})
    save_json(
        ckpt_dir / "preprocess.json",
        {
            "task": "task1",
            "image_size": image_size,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
    )
    save_json(ckpt_dir / "run_meta.json", {"meta": meta.to_json(), "preprocess": {"task": "task1"}})

    return ckpt_path, ckpt_dir


def test_export_writes_artifact_and_api_can_load(tmp_path: Path, monkeypatch) -> None:
    ckpt_path, _ = _make_dummy_checkpoint_dir(tmp_path)

    out_dir = tmp_path / "artifact"

    from cv200 import export as export_mod

    argv = ["cv200.export", "--ckpt", str(ckpt_path), "--output", str(out_dir), "--device", "cpu"]
    monkeypatch.setattr(sys, "argv", argv)
    export_mod.main()

    assert (out_dir / "model.ts").exists()
    assert (out_dir / "labels.json").exists()
    assert (out_dir / "preprocess.json").exists()

    # API artifact loader should be able to load TorchScript + preprocess.
    from cv200.api import _load_artifact_cached

    artifact = _load_artifact_cached(str(out_dir))
    assert "model" in artifact and "preprocess" in artifact and "labels" in artifact
