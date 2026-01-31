from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> int:
    """Best-effort seeding across common RNG sources."""
    logging.getLogger(__name__).info("Setting random seed to %s", seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    # Trade-off: disabling CuDNN benchmark improves reproducibility but can reduce throughput.
    torch.backends.cudnn.benchmark = False
    return seed


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: str | Path, payload: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def dataset_fingerprint(root: str | Path, max_files: int | None = None) -> str:
    """
    Compute a simple fingerprint of the dataset directory.

    Uses file relative paths and sizes; this is not cryptographically strong but
    is sufficient to detect most content/layout changes for reproducibility.
    """
    root_path = Path(root).resolve()
    h = sha256()
    count = 0
    for dirpath, _, filenames in os.walk(root_path):
        dirpath_p = Path(dirpath)
        for name in sorted(filenames):
            file_path = dirpath_p / name
            rel = file_path.relative_to(root_path)
            try:
                size = file_path.stat().st_size
            except OSError:
                size = -1
            h.update(str(rel).encode("utf-8"))
            h.update(str(size).encode("utf-8"))
            count += 1
            if max_files is not None and count >= max_files:
                break
        if max_files is not None and count >= max_files:
            break
    return h.hexdigest()


@dataclass(frozen=True)
class PreprocessConfig:
    task: str  # "task1" | "task2"
    image_size: int  # used for crop/resize targets
    mean: tuple[float, float, float]
    std: tuple[float, float, float]

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["mean"] = list(self.mean)
        d["std"] = list(self.std)
        return d
