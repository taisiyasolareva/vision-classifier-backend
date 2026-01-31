from __future__ import annotations

"""
Basic EDA script for the CV200 dataset.

Usage:
  python scripts/eda.py --data-root /path/to/data_root --output-dir reports/eda

It will generate:
  - class_counts.png      : bar plot of per-class image counts
  - resolution_stats.json : basic width/height statistics
  - samples_grid.png      : grid of sample images from a subset of classes
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid

# Allow running this script without `pip install -e .`
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)


def compute_class_distribution(ds: ImageFolder) -> Counter:
    counts: Counter[int] = Counter()
    for _, label in ds.samples:
        counts[int(label)] += 1
    return counts


def compute_resolution_stats(ds: ImageFolder, max_samples: int | None = 10_000) -> dict[str, Any]:
    widths: list[int] = []
    heights: list[int] = []

    n = 0
    for path, _ in ds.samples:
        img = Image.open(path).convert("RGB")
        w, h = img.size
        widths.append(w)
        heights.append(h)
        n += 1
        if max_samples is not None and n >= max_samples:
            break

    widths_arr = np.array(widths)
    heights_arr = np.array(heights)

    def _summ(a: np.ndarray) -> dict[str, float]:
        return {
            "min": float(a.min()),
            "max": float(a.max()),
            "mean": float(a.mean()),
            "median": float(np.median(a)),
        }

    return {
        "num_samples": int(len(widths)),
        "width": _summ(widths_arr),
        "height": _summ(heights_arr),
    }


def plot_class_histogram(class_counts: Counter, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    indices = sorted(class_counts.keys())
    counts = [class_counts[i] for i in indices]

    plt.figure(figsize=(16, 5))
    plt.bar(indices, counts)
    plt.xlabel("Class index")
    plt.ylabel("Image count")
    plt.title("Per-class image counts")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_samples_grid(ds: ImageFolder, out_path: Path, num_classes: int = 10) -> None:
    """
    Create a 2x5 grid montage: randomly sample 1 image from each of 10 classes.

    Args:
        ds: ImageFolder dataset
        out_path: Output path for the grid image
        num_classes: Number of classes to sample (default: 10 for 2x5 grid)
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import random

    # Build mapping: class_idx -> list of image paths
    by_class: dict[int, list[str]] = {}
    for path, label in ds.samples:
        by_class.setdefault(int(label), []).append(path)

    # Randomly sample 1 image from each of num_classes classes
    selected_imgs: list[np.ndarray] = []
    class_indices = sorted(by_class.keys())[:num_classes]

    for cls_idx in class_indices:
        paths = by_class[cls_idx]
        if paths:
            # Randomly select 1 image from this class
            selected_path = random.choice(paths)
            img = Image.open(selected_path).convert("RGB")
            selected_imgs.append(np.array(img))

    if not selected_imgs:
        print(f"Warning: No images found to create grid")
        return

    # Convert to a torch tensor grid via PIL → numpy → tensor inside make_grid
    import torch

    tensors = [torch.from_numpy(img).permute(2, 0, 1).float() / 255.0 for img in selected_imgs]
    # Create 2x5 grid (nrow=5 means 5 columns, so 2 rows)
    grid = make_grid(tensors, nrow=5)
    grid_np = (grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    Image.fromarray(grid_np).save(out_path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, required=True, help="Root with train/ and val/ folders")
    p.add_argument(
        "--output-dir", type=str, default="reports/eda", help="Output directory for EDA artifacts"
    )
    args = p.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.output_dir)
    train_dir = data_root / "train"

    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")

    ds = ImageFolder(str(train_dir))

    # Class distribution
    class_counts = compute_class_distribution(ds)
    plot_class_histogram(class_counts, out_dir / "class_counts.png")
    _save_json(out_dir / "class_counts.json", {str(k): int(v) for k, v in class_counts.items()})

    # Resolution statistics
    res_stats = compute_resolution_stats(ds)
    _save_json(out_dir / "resolution_stats.json", res_stats)

    # Sample grid
    save_samples_grid(ds, out_dir / "samples_grid.png")

    print(f"EDA artifacts written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
