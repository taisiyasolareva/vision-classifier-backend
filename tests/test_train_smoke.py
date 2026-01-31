from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PIL import Image


def _write_image(path: Path, *, size: int = 64) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (size, size), color=(123, 50, 200))
    img.save(path)


def _make_tiny_imagefolder(
    root: Path, *, num_classes: int = 3, n_train: int = 6, n_val: int = 3
) -> Path:
    for split, n in (("train", n_train), ("val", n_val)):
        for cls in range(num_classes):
            for i in range(n):
                _write_image(root / split / str(cls) / f"img_{i}.jpg")
    return root


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_train_smoke_task1_and_task2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    End-to-end smoke test that:
    - builds dataloaders from an ImageFolder layout
    - runs 1 epoch
    - writes artifacts (run_meta.json + checkpoints)

    CI-safe: uses --no-pretrained to avoid downloads.
    """
    data_root = _make_tiny_imagefolder(tmp_path / "data")

    from cv200 import train as train_mod

    # Task 1 smoke: use SmallCNN for speed
    out1 = tmp_path / "runs_task1"
    argv1 = [
        "cv200.train",
        "--data-root",
        str(data_root),
        "--task",
        "task1",
        "--arch",
        "small_cnn",
        "--epochs",
        "1",
        "--batch-size",
        "4",
        "--num-workers",
        "0",
        "--limit-train-samples",
        "8",
        "--limit-val-samples",
        "6",
        "--device",
        "cpu",
        "--output-dir",
        str(out1),
        "--notes",
        "smoke-test",
    ]
    monkeypatch.setattr(sys, "argv", argv1)
    train_mod.main()

    assert (out1 / "run_meta.json").exists()
    assert (out1 / "labels.json").exists()
    assert (out1 / "checkpoint_last.pt").exists()
    assert (out1 / "checkpoint_best.pt").exists()

    # Task 2 smoke: validate fine-tuning mechanics without pretrained downloads
    out2 = tmp_path / "runs_task2"
    argv2 = [
        "cv200.train",
        "--data-root",
        str(data_root),
        "--task",
        "task2",
        "--arch",
        "resnet18",
        "--no-pretrained",
        "--fine-tune-strategy",
        "frozen",
        "--epochs",
        "1",
        "--batch-size",
        "4",
        "--num-workers",
        "0",
        "--limit-train-samples",
        "8",
        "--limit-val-samples",
        "6",
        "--device",
        "cpu",
        "--output-dir",
        str(out2),
        "--notes",
        "smoke-test",
    ]
    monkeypatch.setattr(sys, "argv", argv2)
    train_mod.main()

    assert (out2 / "run_meta.json").exists()
    assert (out2 / "labels.json").exists()
    assert (out2 / "checkpoint_last.pt").exists()
    assert (out2 / "checkpoint_best.pt").exists()
