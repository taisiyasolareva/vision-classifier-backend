from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

import torch

from cv200.utils import PreprocessConfig, save_json


@dataclass(frozen=True)
class CheckpointMeta:
    arch: str
    num_classes: int
    task: str
    pretrained: bool
    fine_tune_strategy: str  # "none" | "frozen" | "partial" | "full"
    image_size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    unfreeze_last_n: int | None = None
    trainable_params_m: float | None = None
    dataset_fingerprint: str | None = None
    train_top1: float | None = None
    val_top1: float | None = None
    val_top5: float | None = None
    epoch: int | None = None

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["mean"] = list(self.mean)
        d["std"] = list(self.std)
        return d


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    meta: CheckpointMeta,
    scheduler: Any | None = None,
    best_val_top1: float | None = None,
    include_optimizer: bool = True,
    include_scheduler: bool = True,
) -> None:
    """
    Save checkpoint with atomic write (write to .tmp then rename).

    Args:
        path: Checkpoint file path
        model: Model to save
        optimizer: Optimizer to save (if include_optimizer=True)
        epoch: Current epoch number
        meta: Checkpoint metadata
        scheduler: Scheduler to save (if include_scheduler=True)
        best_val_top1: Best validation top-1 accuracy (for resume)
        include_optimizer: If False, skip optimizer state (saves ~50% space)
        include_scheduler: If False, skip scheduler state
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # Atomic write: write to .tmp then rename
    tmp_path = p.with_suffix(p.suffix + ".tmp")

    state = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "meta": meta.to_json(),
    }

    if include_optimizer:
        state["optimizer_state_dict"] = optimizer.state_dict()

    if include_scheduler and scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()

    if best_val_top1 is not None:
        state["best_val_top1"] = best_val_top1

    torch.save(state, tmp_path)
    # Atomic rename (safer on Colab/network filesystems)
    os.replace(tmp_path, p)


def load_checkpoint(
    path: str | Path, *, map_location: str | torch.device = "cpu"
) -> dict[str, Any]:
    return torch.load(Path(path), map_location=map_location)


def write_run_metadata(
    output_dir: str | Path,
    *,
    meta: CheckpointMeta,
    preprocess: PreprocessConfig,
    cli_args: Dict[str, Any] | None = None,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "meta": meta.to_json(),
        "preprocess": preprocess.to_json(),
    }
    if cli_args is not None:
        payload["cli_args"] = cli_args
    save_json(out / "run_meta.json", payload)
    save_json(out / "preprocess.json", preprocess.to_json())
