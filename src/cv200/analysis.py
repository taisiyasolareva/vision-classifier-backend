from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from cv200.checkpointing import load_checkpoint
from cv200.data import build_transforms
from cv200.models import ModelConfig, build_model
from cv200.utils import save_json


class ImageFolderWithPaths(ImageFolder):
    """torchvision ImageFolder that also returns the underlying file path."""

    def __getitem__(self, index: int):  # type: ignore[override]
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, int(target), str(path)


def _sanitize_image_path(path: str, data_root: Path) -> str:
    """Return a relative path suitable for public artifacts (e.g. val/class_000/00001.jpg)."""
    path_str = str(path)
    # Strip to val/... if present to avoid leaking absolute or Colab paths.
    if "/val/" in path_str:
        return "val/" + path_str.split("/val/", 1)[-1]
    try:
        return str(Path(path_str).relative_to(data_root))
    except ValueError:
        return path_str


@dataclass(frozen=True)
class _PredRecord:
    path: str
    true: int
    pred: int
    pred_conf: float
    true_conf: float


def analyze_errors(
    checkpoint_path: str,
    data_root: Path,
    task: str,
    output_dir: Path,
    top_k_errors: int = 20,
) -> None:
    """
    Error analysis on the validation set.

    Produces:
    - top_errors.json: top-K most confident wrong predictions
    - hardest_examples.json: per-class list of hardest correct predictions (lowest confidence on correct class)
    - error_summary.txt: short text summary of common failure patterns
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt = load_checkpoint(checkpoint_path, map_location="cpu")
    meta = ckpt.get("meta", {})
    arch = str(meta.get("arch", "resnet18"))

    # Use checkpoint meta if available; otherwise use provided `task`.
    task_from_ckpt = meta.get("task")
    if task_from_ckpt is not None:
        task = str(task_from_ckpt)

    image_size = int(meta.get("image_size", 224))

    train_dir = data_root / "train"
    val_dir = data_root / "val"
    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")
    if not val_dir.exists():
        raise FileNotFoundError(f"Val directory not found: {val_dir}")

    # Label mapping comes from train/ (stable ImageFolder class_to_idx).
    train_ds = ImageFolder(str(train_dir))
    idx_to_class = {idx: cls_name for cls_name, idx in train_ds.class_to_idx.items()}
    num_classes = len(idx_to_class)

    # Deterministic validation transforms.
    _, val_tf, _ = build_transforms(task=task, image_size=image_size)
    val_ds = ImageFolderWithPaths(str(val_dir), transform=val_tf)
    val_dl = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

    # Load model weights (pretrained flag irrelevant for state_dict compatibility).
    model = build_model(ModelConfig(arch=arch, num_classes=num_classes, pretrained=False))
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    # Prefer GPU acceleration when available:
    # - CUDA on Linux/Windows GPUs
    # - MPS on Apple Silicon
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = model.to(device)
    model.eval()

    softmax = nn.Softmax(dim=-1)

    wrong: list[_PredRecord] = []
    hardest_correct: dict[int, list[_PredRecord]] = defaultdict(list)
    confusion_pairs: Counter[tuple[int, int]] = Counter()
    per_class_total: Counter[int] = Counter()
    per_class_wrong: Counter[int] = Counter()

    for images, targets, paths in val_dl:
        images = images.to(device)
        targets_t = torch.tensor(targets, device=device, dtype=torch.long)
        logits = model(images)
        probs = softmax(logits)
        pred = torch.argmax(probs, dim=-1)

        pred_conf = probs.gather(1, pred.view(-1, 1)).squeeze(1)
        true_conf = probs.gather(1, targets_t.view(-1, 1)).squeeze(1)

        pred_cpu = pred.detach().to("cpu").tolist()
        t_cpu = targets_t.detach().to("cpu").tolist()
        pred_conf_cpu = pred_conf.detach().to("cpu").tolist()
        true_conf_cpu = true_conf.detach().to("cpu").tolist()

        # NOTE: keep Python 3.9 compatibility (zip(strict=...) is 3.10+).
        for pth, t, pr, pc, tc in zip(paths, t_cpu, pred_cpu, pred_conf_cpu, true_conf_cpu):
            per_class_total[int(t)] += 1
            rec = _PredRecord(
                path=_sanitize_image_path(str(pth), data_root),
                true=int(t),
                pred=int(pr),
                pred_conf=float(pc),
                true_conf=float(tc),
            )
            if rec.pred != rec.true:
                wrong.append(rec)
                per_class_wrong[int(t)] += 1
                confusion_pairs[(int(t), int(pr))] += 1
            else:
                # Keep the lowest-confidence correct predictions per class.
                hardest_correct[int(t)].append(rec)

    # Top-K most confident wrong predictions
    wrong_sorted = sorted(wrong, key=lambda r: r.pred_conf, reverse=True)[: max(0, top_k_errors)]
    top_errors_payload: list[dict[str, Any]] = []
    for r in wrong_sorted:
        top_errors_payload.append(
            {
                "image_path": r.path,
                "predicted_class": r.pred,
                "true_class": r.true,
                "confidence": r.pred_conf,
                "predicted_class_name": idx_to_class.get(r.pred, str(r.pred)),
                "true_class_name": idx_to_class.get(r.true, str(r.true)),
            }
        )
    save_json(output_dir / "top_errors.json", top_errors_payload)

    # Per-class hardest correct examples: keep up to K=3 with lowest true_conf.
    hardest_k = 3
    hardest_payload: dict[str, list[dict[str, Any]]] = {}
    for cls_idx, recs in hardest_correct.items():
        recs_sorted = sorted(recs, key=lambda r: r.true_conf)[:hardest_k]
        hardest_payload[str(cls_idx)] = [
            {
                "image_path": r.path,
                "predicted_class": r.pred,
                "true_class": r.true,
                "confidence": r.true_conf,
                "predicted_class_name": idx_to_class.get(r.pred, str(r.pred)),
                "true_class_name": idx_to_class.get(r.true, str(r.true)),
            }
            for r in recs_sorted
        ]
    save_json(output_dir / "hardest_examples.json", hardest_payload)

    # Text summary of common failure patterns
    total_n = sum(per_class_total.values())
    wrong_n = len(wrong)
    acc = 1.0 - (float(wrong_n) / float(max(1, total_n)))

    worst_classes = []
    for cls_idx, n_tot in per_class_total.items():
        n_wrong = per_class_wrong.get(cls_idx, 0)
        err_rate = float(n_wrong) / float(max(1, n_tot))
        worst_classes.append((cls_idx, err_rate, n_tot))
    worst_classes.sort(key=lambda x: x[1], reverse=True)

    top_confusions = confusion_pairs.most_common(10)

    lines: list[str] = []
    lines.append(f"Checkpoint: {checkpoint_path}")
    lines.append(f"Task: {task} | Arch: {arch} | Image size: {image_size}")
    lines.append(f"Val samples: {total_n} | Wrong: {wrong_n} | Top-1 acc (approx): {acc:.4f}")
    lines.append("")
    lines.append("Top confusions (true -> pred) by count:")
    if not top_confusions:
        lines.append("  (none)")
    else:
        for (t, pr), cnt in top_confusions:
            lines.append(
                f"  {t} ({idx_to_class.get(t, t)}) -> {pr} ({idx_to_class.get(pr, pr)}): {cnt}"
            )
    lines.append("")
    lines.append("Most error-prone classes by error rate (top 10):")
    for cls_idx, err_rate, n_tot in worst_classes[:10]:
        lines.append(
            f"  class={cls_idx} name={idx_to_class.get(cls_idx, cls_idx)} err_rate={err_rate:.3f} n={n_tot}"
        )
    lines.append("")
    lines.append("Notes:")
    lines.append("- Review top_errors.json to spot systematic label confusion or ambiguous images.")
    lines.append(
        "- Review hardest_examples.json for borderline samples that the model barely gets right."
    )

    (output_dir / "error_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
