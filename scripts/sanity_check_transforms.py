from __future__ import annotations

"""
Sanity-check script for cv200 transforms.

It samples N random images from the train set, applies the configured train/val
transforms for a given task, and saves montages showing raw, train-transformed, and
val-transformed images.

Usage:
  python scripts/sanity_check_transforms.py \
    --task task1 \
    --data-root /path/to/data \
    --output-dir runs/sanity_check \
    --num-samples 16
"""

import argparse
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid

# Allow running this script without `pip install -e .`
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cv200.data import build_transforms  # noqa: E402


def _sample_indices(n_total: int, n_samples: int) -> list[int]:
    n_samples = min(n_samples, n_total)
    return random.sample(range(n_total), n_samples)


def _save_grid(
    images: torch.Tensor,
    out_path: Path,
    title: str,
    nrow: int = 4,
) -> None:
    """Save a grid montage of images."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Clamp values to [0, 1] for visualization
    # Normalized images may be outside [0, 1], so we use normalize=True
    # which will scale each image independently to [0, 1] range
    grid = make_grid(images, nrow=nrow, normalize=True, scale_each=True)

    # Convert to numpy and ensure values are in [0, 1]
    grid_np = grid.permute(1, 2, 0).cpu().numpy()
    grid_np = grid_np.clip(0, 1)

    plt.figure(figsize=(12, 12))
    plt.axis("off")
    plt.title(title, fontsize=14)
    plt.imshow(grid_np)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--task", type=str, choices=["task1", "task2"], required=True, help="Task (task1 or task2)"
    )
    p.add_argument(
        "--data-root", type=str, required=True, help="Root directory with train/ and val/ folders"
    )
    p.add_argument(
        "--output-dir", type=str, required=True, help="Output directory for montage images"
    )
    p.add_argument("--num-samples", type=int, default=16, help="Number of random images to sample")
    p.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Target image size for transforms. If omitted: task1 defaults to 64, task2 defaults to 224.",
    )
    args = p.parse_args()

    if args.image_size is None:
        args.image_size = 64 if args.task == "task1" else 224

    data_root = Path(args.data_root)
    train_dir = data_root / "train"
    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")

    # Load dataset and build transforms
    ds = ImageFolder(str(train_dir))
    train_tf, val_tf, _ = build_transforms(args.task, args.image_size)

    # Sample random indices
    indices = _sample_indices(len(ds), args.num_samples)

    # Calculate grid layout (nrow = sqrt of num_samples, rounded)
    nrow = int(args.num_samples**0.5)
    if nrow * nrow < args.num_samples:
        nrow += 1

    raw_tensors = []
    train_tensors = []
    val_tensors = []

    from torchvision.transforms import ToTensor

    to_tensor = ToTensor()

    for idx in indices:
        path, _ = ds.samples[idx]
        img = ds.loader(path)  # PIL Image

        # Raw image (convert to tensor, scale to [0, 1])
        raw_tensors.append(to_tensor(img))

        # Apply transforms
        train_tensors.append(train_tf(img))
        val_tensors.append(val_tf(img))

    # Stack into batches
    raw_batch = torch.stack(raw_tensors)
    train_batch = torch.stack(train_tensors)
    val_batch = torch.stack(val_tensors)

    # Save montages
    out_dir = Path(args.output_dir)
    _save_grid(
        raw_batch, out_dir / "samples_raw.png", title="Raw Samples (Before Transforms)", nrow=nrow
    )
    _save_grid(
        train_batch,
        out_dir / "samples_train_transformed.png",
        title=f"Train Transforms ({args.task})",
        nrow=nrow,
    )
    _save_grid(
        val_batch,
        out_dir / "samples_val_transformed.png",
        title=f"Validation Transforms ({args.task})",
        nrow=nrow,
    )

    print(f"Wrote transform sanity-check montages to: {out_dir.resolve()}")
    print(f"  - samples_raw.png")
    print(f"  - samples_train_transformed.png")
    print(f"  - samples_val_transformed.png")


if __name__ == "__main__":
    main()
