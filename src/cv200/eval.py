from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch import nn

from cv200.checkpointing import load_checkpoint
from cv200.data import DataConfig, build_dataloaders
from cv200.models import ModelConfig, build_model
from cv200.train import _device_from_arg
from cv200.utils import save_json


def _save_confusion_matrix_csv(cm: torch.Tensor, *, out_path: Path) -> None:
    """
    Save confusion matrix as CSV with rows=predicted and cols=actual.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = int(cm.shape[0])
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pred\\actual", *list(range(n))])
        cm_cpu = cm.to("cpu").tolist()
        for pred_i in range(n):
            w.writerow([pred_i, *cm_cpu[pred_i]])


def _save_confusion_matrix_png(
    cm: torch.Tensor,
    *,
    out_path: Path,
    class_order: list[int],
    title: str,
) -> None:
    """
    Save a confusion matrix heatmap PNG using matplotlib.

    For readability, callers typically pass a subset order (e.g., top-N classes by support).
    """
    # Local import so eval can still run (minus PNG) if matplotlib isn't installed.
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub = cm[torch.tensor(class_order), :][:, torch.tensor(class_order)].to("cpu").numpy()

    plt.figure(figsize=(10, 10))
    plt.imshow(sub, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.colorbar()

    n = len(class_order)
    if n <= 30:
        ticks = list(range(n))
        labels = [str(i) for i in class_order]
        plt.xticks(ticks, labels, rotation=90, fontsize=6)
        plt.yticks(ticks, labels, fontsize=6)
    else:
        # Too dense; keep ticks sparse.
        step = max(1, n // 10)
        ticks = list(range(0, n, step))
        labels = [str(class_order[i]) for i in ticks]
        plt.xticks(ticks, labels, rotation=90, fontsize=6)
        plt.yticks(ticks, labels, fontsize=6)

    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


@torch.inference_mode()
def eval_detailed(
    model: nn.Module,
    dl,
    *,
    device: torch.device,
    loss_fn: nn.Module,
    num_classes: int,
) -> tuple[
    float,
    float,
    float,
    dict[int, float],
    float,
    float,
    list[tuple[int, float, int]],
    torch.Tensor,
    torch.Tensor,
]:
    """
    Evaluate and compute:
    - val loss
    - micro accuracy (top-1)
    - top-5 accuracy
    - per-class accuracy
    - macro accuracy (mean of per-class accuracies)
    - list of worst classes: (class_idx, acc, count)
    - confusion matrix (rows=predicted, cols=actual)
    - per-class support counts (val totals)
    """
    model.eval()
    running_loss = 0.0
    total = 0
    correct_top1 = 0
    correct_top5 = 0

    per_class_total = torch.zeros(num_classes, dtype=torch.long)
    per_class_correct = torch.zeros(num_classes, dtype=torch.long)
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long)

    for images, targets in dl:
        images = images.to(device)
        targets = targets.to(device)

        logits = model(images)
        loss = loss_fn(logits, targets)

        bsz = targets.numel()
        running_loss += float(loss.item()) * bsz
        total += int(bsz)

        pred1 = torch.argmax(logits, dim=-1)
        correct_mask = pred1.eq(targets)
        correct_top1 += int(correct_mask.sum().item())

        # Top-5 accuracy (batch)
        correct_top5 += int(
            (torch.topk(logits, k=min(5, logits.shape[-1]), dim=-1).indices == targets[:, None])
            .any(dim=1)
            .sum()
            .item()
        )

        # Per-class totals/correct + confusion matrix (on CPU for stable bincount)
        pred_cpu = pred1.detach().to("cpu")
        t_cpu = targets.detach().to("cpu")
        c_cpu = correct_mask.detach().to("cpu")
        per_class_total += torch.bincount(t_cpu, minlength=num_classes)
        if c_cpu.any():
            per_class_correct += torch.bincount(t_cpu[c_cpu], minlength=num_classes)
        # Confusion matrix: rows=predicted, cols=actual
        flat = pred_cpu * num_classes + t_cpu
        cm += torch.bincount(flat, minlength=num_classes * num_classes).reshape(
            num_classes, num_classes
        )

    val_loss = running_loss / max(1, total)
    micro = float(correct_top1) / float(max(1, total))
    top5 = float(correct_top5) / float(max(1, total))

    # Per-class accuracies
    per_class_acc: dict[int, float] = {}
    totals = per_class_total.numpy()
    corrects = per_class_correct.numpy()
    accs: list[float] = []
    for i in range(num_classes):
        if totals[i] > 0:
            acc = float(corrects[i]) / float(totals[i])
            accs.append(acc)
        else:
            acc = 0.0
        per_class_acc[i] = acc

    macro = float(sum(accs) / max(1, len(accs)))

    # Worst classes (by accuracy, among classes that exist in val)
    worst: list[tuple[int, float, int]] = [
        (i, per_class_acc[i], int(totals[i])) for i in range(num_classes) if totals[i] > 0
    ]
    worst.sort(key=lambda x: x[1])
    worst5 = worst[:5]

    return val_loss, micro, top5, per_class_acc, macro, micro, worst5, cm, per_class_total


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--save-confusion-matrix",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="If enabled, save confusion_matrix.csv and a readable confusion_matrix.png.",
    )
    args = p.parse_args()

    device = _device_from_arg(args.device)
    ckpt = load_checkpoint(args.ckpt, map_location="cpu")
    meta = ckpt.get("meta", {})

    task = meta.get("task", "task2")
    image_size = int(meta.get("image_size", 224))
    arch = meta.get("arch", "resnet18")

    data_cfg = DataConfig(
        data_root=Path(args.data_root),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        task=task,
        image_size=image_size,
    )
    _, val_dl, idx_to_class, _ = build_dataloaders(data_cfg)
    num_classes = len(idx_to_class)

    # Whether it was pretrained doesn’t affect state_dict shape; we rebuild a matching head.
    model = build_model(ModelConfig(arch=arch, num_classes=num_classes, pretrained=False))
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    val_loss, val_top1, val_top5, per_class_acc, macro_acc, micro_acc, worst5, cm, support = (
        eval_detailed(model, val_dl, device=device, loss_fn=loss_fn, num_classes=num_classes)
    )

    # Save per-class metrics next to the checkpoint
    out_dir = Path(args.ckpt).resolve().parent
    save_json(
        out_dir / "per_class_accuracy.json", {str(k): float(v) for k, v in per_class_acc.items()}
    )
    if args.save_confusion_matrix:
        _save_confusion_matrix_csv(cm, out_path=out_dir / "confusion_matrix.csv")

        # Plot a readable subset: top-N classes by validation support (default N=50).
        support_np = support.numpy()
        order = list(reversed(sorted(range(num_classes), key=lambda i: int(support_np[i]))))
        top_n = min(50, num_classes)
        order = order[:top_n]
        _save_confusion_matrix_png(
            cm,
            out_path=out_dir / "confusion_matrix.png",
            class_order=order,
            title=f"Confusion Matrix (Top {top_n} classes by support)",
        )

    print(f"val_loss={val_loss:.4f} val_top1={val_top1:.4f} val_top5={val_top5:.4f}")
    print(f"Macro accuracy: {macro_acc:.4f}, Micro accuracy: {micro_acc:.4f}")

    # Worst classes
    print("Worst classes (bottom 5):")
    for cls_idx, acc, cnt in worst5:
        cls_name = idx_to_class.get(int(cls_idx), str(cls_idx))
        print(f"  class={cls_idx} name={cls_name} acc={acc:.4f} n={cnt}")


if __name__ == "__main__":
    main()
