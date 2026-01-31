from __future__ import annotations

"""
Dataset profiling script for CV200.

Usage:
  python scripts/profile_data.py --data-root /path/to/data --output reports/data_profile.json

It computes:
  - Per-channel mean and std (RGB) over a random sample of training images (default: 1000 images).
  - Per-class image counts.
  - Imbalance metrics: max/min ratio and standard deviation of class counts.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from torchvision.datasets import ImageFolder

# Allow running this script without `pip install -e .`
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)


def compute_class_counts(ds: ImageFolder) -> Counter:
    counts: Counter[int] = Counter()
    for _, label in ds.samples:
        counts[int(label)] += 1
    return counts


def compute_mean_std(ds: ImageFolder, num_samples: int = 1000) -> tuple[list[float], list[float]]:
    """
    Compute per-channel mean and std over a random subset of training images.

    Args:
        ds: ImageFolder dataset
        num_samples: Number of random images to sample

    Returns:
        Tuple of (mean, std) as 3-element lists [R, G, B]
    """
    import random

    # Randomly sample images
    all_samples = list(ds.samples)
    if len(all_samples) > num_samples:
        sampled = random.sample(all_samples, num_samples)
    else:
        sampled = all_samples

    # Collect all pixel values
    all_pixels = []
    for path, _ in sampled:
        img = Image.open(path).convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0  # [H, W, 3]
        pixels = arr.reshape(-1, 3)  # [N, 3]
        all_pixels.append(pixels)

    if not all_pixels:
        raise RuntimeError("No images found to compute statistics.")

    # Stack all pixels
    all_pixels = np.vstack(all_pixels)  # [total_pixels, 3]

    # Compute mean and std per channel
    mean = np.mean(all_pixels, axis=0)  # [3]
    std = np.std(all_pixels, axis=0)  # [3]

    return mean.tolist(), std.tolist()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data-root", type=str, required=True, help="Root directory with train/ and val/ folders"
    )
    p.add_argument(
        "--output",
        type=str,
        default="reports/data_profile.json",
        help="Output path for JSON profile",
    )
    p.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of random images to sample for mean/std computation",
    )
    args = p.parse_args()

    data_root = Path(args.data_root)
    train_dir = data_root / "train"
    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")

    ds = ImageFolder(str(train_dir))

    # Class counts
    class_counts = compute_class_counts(ds)
    counts_dict = {str(k): int(v) for k, v in sorted(class_counts.items())}

    # Imbalance metrics
    counts_list = [int(v) for v in counts_dict.values()]
    if counts_list:
        max_count = max(counts_list)
        min_count = min(counts_list)
        max_min_ratio = float(max_count) / float(min_count) if min_count > 0 else None
        counts_std = float(np.std(counts_list))
    else:
        max_min_ratio = None
        counts_std = 0.0

    # Mean/std over sampled images
    mean, std = compute_mean_std(ds, num_samples=args.num_samples)

    # Output format matching specification
    profile = {
        "mean": mean,  # [R, G, B]
        "std": std,  # [R, G, B]
        "class_counts": counts_dict,
        "imbalance": {
            "max_min_ratio": max_min_ratio,
            "std": counts_std,
        },
    }

    output_path = Path(args.output)
    _save_json(output_path, profile)
    print(f"Wrote data profile to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
