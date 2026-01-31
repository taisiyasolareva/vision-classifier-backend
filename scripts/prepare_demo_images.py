#!/usr/bin/env python3
"""
Prepare sample images for web demo.

This script copies a few random images from your validation set to a public directory
that can be hosted alongside your frontend.

Usage:
    python scripts/prepare_demo_images.py \
        --data-root /path/to/data_root \
        --output static/sample_images \
        --num-samples 20
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from torchvision.datasets import ImageFolder


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare sample images for web demo")
    p.add_argument(
        "--data-root", type=str, required=True, help="Root directory with train/ and val/ folders"
    )
    p.add_argument(
        "--output",
        type=str,
        default="static/sample_images",
        help="Output directory for sample images",
    )
    p.add_argument("--num-samples", type=int, default=20, help="Number of random images to copy")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = p.parse_args()

    data_root = Path(args.data_root)
    val_dir = data_root / "val"
    if not val_dir.exists():
        raise FileNotFoundError(f"Validation directory not found: {val_dir}")

    # Load validation dataset
    val_dataset = ImageFolder(str(val_dir))
    print(f"Found {len(val_dataset)} validation images across {len(val_dataset.classes)} classes")

    # Sample random images
    random.seed(args.seed)
    indices = random.sample(range(len(val_dataset)), min(args.num_samples, len(val_dataset)))

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy images
    copied = []
    for idx in indices:
        img_path, class_idx = val_dataset.samples[idx]
        class_name = val_dataset.classes[class_idx]

        # Create filename: class_idx_class_name_original_name.jpg
        original_name = Path(img_path).name
        new_name = f"{class_idx:03d}_{class_name}_{original_name}"
        dest_path = output_dir / new_name

        shutil.copy2(img_path, dest_path)
        copied.append(
            {
                "path": str(dest_path.relative_to(output_dir)),
                "class_idx": class_idx,
                "class_name": class_name,
            }
        )
        print(f"Copied: {new_name}")

    # Save metadata JSON
    import json

    metadata = {
        "num_samples": len(copied),
        "images": copied,
    }
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Copied {len(copied)} images to {output_dir}")
    print(f"✓ Saved metadata to {metadata_path}")
    print(
        f"\nTo use in your frontend, host these images and update SAMPLE_IMAGES array in index.html"
    )


if __name__ == "__main__":
    main()
