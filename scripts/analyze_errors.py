from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running this script without `pip install -e .`
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cv200.analysis import analyze_errors  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (.pt)")
    p.add_argument(
        "--data-root", type=str, required=True, help="Dataset root containing train/ and val/"
    )
    p.add_argument(
        "--task",
        type=str,
        choices=["task1", "task2"],
        required=True,
        help="Task (used for transforms)",
    )
    p.add_argument(
        "--output-dir", type=str, required=True, help="Output directory for analysis artifacts"
    )
    p.add_argument(
        "--top-k-errors", type=int, default=20, help="Number of confident wrong predictions to save"
    )
    args = p.parse_args()

    analyze_errors(
        checkpoint_path=args.checkpoint,
        data_root=Path(args.data_root),
        task=args.task,
        output_dir=Path(args.output_dir),
        top_k_errors=args.top_k_errors,
    )
    print(f"Wrote error analysis artifacts to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
