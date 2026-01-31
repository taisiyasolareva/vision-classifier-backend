from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2 as T

from cv200.utils import PreprocessConfig

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _iter_transforms(tf: object) -> list[object]:
    """
    Flatten a torchvision transform graph (best-effort).

    Supports torchvision v2 containers like Compose / RandomApply which expose a
    `.transforms` attribute.
    """
    out: list[object] = [tf]
    children = getattr(tf, "transforms", None)
    if isinstance(children, list):
        for ch in children:
            out.extend(_iter_transforms(ch))
    return out


def assert_task1_no_resize(tf: object) -> None:
    """
    Task 1 guardrail: resizing/rescaling is forbidden.

    This mirrors the original homework's "must-pass asserts": if someone
    accidentally introduces a resize-like op into Task 1, we fail fast.
    """
    forbidden = {"Resize", "RandomResizedCrop"}
    hits: list[str] = []
    for t in _iter_transforms(tf):
        name = t.__class__.__name__
        if name in forbidden:
            hits.append(name)
    if hits:
        raise AssertionError(
            "Task 1 forbids resize-like transforms. Found: "
            + ", ".join(sorted(set(hits)))
            + ". Use Pad/Crop only (no rescaling)."
        )


@dataclass(frozen=True)
class DataConfig:
    data_root: Path
    batch_size: int
    num_workers: int
    task: str  # "task1" | "task2"
    image_size: int = 224


def build_transforms(
    task: str, image_size: int
) -> tuple[torch.nn.Module, torch.nn.Module, PreprocessConfig]:
    """
    Task 1 constraint: no resizing/rescaling. We use pad+crop (no scale change).
    Task 2: resize allowed (and typically required for pretrained backbones).
    """
    mean, std = IMAGENET_MEAN, IMAGENET_STD

    if task == "task1":
        train_tf = T.Compose(
            [
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                # No Resize / no rescaling:
                # - Pad ensures we can crop even if images are smaller than target.
                # - Crop does not rescale; it only selects a window.
                T.Pad(padding=image_size // 2, fill=0),
                T.RandomCrop(size=(image_size, image_size)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=15),
                T.Normalize(mean=mean, std=std),
            ]
        )
        val_tf = T.Compose(
            [
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                T.Pad(padding=image_size // 2, fill=0),
                T.CenterCrop(size=(image_size, image_size)),
                T.Normalize(mean=mean, std=std),
            ]
        )
        # Guardrail: Task 1 must never include resize-like operations.
        assert_task1_no_resize(train_tf)
        assert_task1_no_resize(val_tf)
    elif task == "task2":
        train_tf = T.Compose(
            [
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                T.Resize(size=(image_size, image_size), antialias=True),
                T.RandomHorizontalFlip(p=0.5),
                T.RandAugment(num_ops=2, magnitude=9),
                T.Normalize(mean=mean, std=std),
            ]
        )
        val_tf = T.Compose(
            [
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                T.Resize(size=(image_size, image_size), antialias=True),
                T.Normalize(mean=mean, std=std),
            ]
        )
    else:
        raise ValueError(f"Unknown task: {task}. Expected 'task1' or 'task2'.")

    pp = PreprocessConfig(task=task, image_size=image_size, mean=mean, std=std)
    return train_tf, val_tf, pp


def build_dataloaders(
    cfg: DataConfig,
) -> tuple[DataLoader, DataLoader, dict[int, str], PreprocessConfig]:
    train_dir = cfg.data_root / "train"
    val_dir = cfg.data_root / "val"
    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")
    if not val_dir.exists():
        raise FileNotFoundError(f"Val directory not found: {val_dir}")

    train_tf, val_tf, pp = build_transforms(task=cfg.task, image_size=cfg.image_size)
    train_ds = ImageFolder(str(train_dir), transform=train_tf)
    val_ds = ImageFolder(str(val_dir), transform=val_tf)

    # idx -> class_name mapping (stable for export/serving)
    idx_to_class = {idx: cls_name for cls_name, idx in train_ds.class_to_idx.items()}

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.num_workers > 0,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.num_workers > 0,
    )
    return train_dl, val_dl, idx_to_class, pp
