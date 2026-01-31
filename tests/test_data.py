from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image
from torchvision.transforms import v2 as T

from cv200.data import DataConfig, assert_task1_no_resize, build_dataloaders, build_transforms


def test_task1_transforms_have_no_resize() -> None:
    # Should not raise
    train_tf, val_tf, _ = build_transforms(task="task1", image_size=64)
    assert_task1_no_resize(train_tf)
    assert_task1_no_resize(val_tf)


def test_task2_transforms_can_include_resize() -> None:
    # Task 2 uses Resize; our guardrail should reject it if mistakenly applied to Task 1.
    resize_tf = T.Compose([T.ToImage(), T.Resize((224, 224))])
    with pytest.raises(AssertionError):
        assert_task1_no_resize(resize_tf)


def _write_img(path: Path, *, size: int = 64) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (size, size), color=(10, 20, 30)).save(path)


def test_build_dataloaders_on_tiny_imagefolder(tmp_path: Path) -> None:
    # Create tiny ImageFolder layout: 2 classes, a few images each
    root = tmp_path / "data"
    for split in ("train", "val"):
        for cls in ("0", "1"):
            for i in range(3):
                _write_img(root / split / cls / f"{split}_{cls}_{i}.jpg", size=64)

    cfg = DataConfig(
        data_root=root,
        batch_size=2,
        num_workers=0,
        task="task1",
        image_size=64,
    )
    train_dl, val_dl, idx_to_class, pp = build_dataloaders(cfg)

    assert pp.task == "task1"
    assert pp.image_size == 64
    assert len(idx_to_class) == 2

    xb, yb = next(iter(train_dl))
    assert xb.shape[0] == 2
    assert xb.shape[1] == 3
    assert xb.shape[2] == 64
    assert xb.shape[3] == 64
    assert yb.shape[0] == 2

    xb2, yb2 = next(iter(val_dl))
    assert xb2.shape[1] == 3
    assert yb2.ndim == 1
