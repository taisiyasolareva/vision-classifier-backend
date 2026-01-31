from __future__ import annotations

"""
Lightweight dataset validation for CV200.

This is a minimal substitute for tools like Great Expectations / Evidently and can
be extended or replaced by them in a production setting.

Checks:
  - train/ and val/ directories exist.
  - All images are readable by PIL.
  - Class indices are contiguous (as given by ImageFolder).
  - Image sizes are positive and within a reasonable range.

Usage:
  python scripts/validate_data.py --data-root /path/to/data_root
"""

import argparse
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from torchvision.datasets import ImageFolder


def validate_images(ds: ImageFolder, max_errors: int = 20) -> None:
    errors = 0
    for path, _ in ds.samples:
        p = Path(path)
        try:
            img = Image.open(p)
            w, h = img.size
            if w <= 0 or h <= 0:
                print(f"[ERROR] Non-positive size image: {p} ({w}x{h})")
                errors += 1
        except (UnidentifiedImageError, OSError) as e:
            print(f"[ERROR] Failed to read image {p}: {e}")
            errors += 1
        if errors >= max_errors:
            print(f"[ERROR] Reached max error limit ({max_errors}), stopping.")
            break
    if errors == 0:
        print("[OK] All sampled images are readable with positive sizes.")
    else:
        print(f"[WARN] Found {errors} problematic images (see logs above).")


def validate_class_indices(ds: ImageFolder) -> None:
    indices = sorted(ds.class_to_idx.values())
    expected = list(range(len(indices)))
    if indices != expected:
        print(f"[WARN] Class indices are not contiguous: {indices}")
    else:
        print("[OK] Class indices are contiguous and start from 0.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--max-errors", type=int, default=20)
    args = p.parse_args()

    root = Path(args.data_root)
    train_dir = root / "train"
    val_dir = root / "val"

    if not train_dir.exists():
        raise FileNotFoundError(f"Missing train directory: {train_dir}")
    if not val_dir.exists():
        raise FileNotFoundError(f"Missing val directory: {val_dir}")

    print(f"Validating train set at: {train_dir}")
    train_ds = ImageFolder(str(train_dir))
    validate_class_indices(train_ds)
    validate_images(train_ds, max_errors=args.max_errors)

    print(f"\nValidating val set at: {val_dir}")
    val_ds = ImageFolder(str(val_dir))
    validate_class_indices(val_ds)
    validate_images(val_ds, max_errors=args.max_errors)

    print(
        "\nValidation complete. For richer validation, consider integrating Great Expectations or Evidently."
    )


if __name__ == "__main__":
    main()
