from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from cv200.checkpointing import CheckpointMeta, save_checkpoint, write_run_metadata
from cv200.data import DataConfig, build_dataloaders
from cv200.finetune import (
    count_total_params,
    count_trainable_params,
    freeze_backbone,
    split_backbone_and_head,
    unfreeze_last_n,
)
from cv200.metrics import accuracy_topk
from cv200.models import ModelConfig, build_model
from cv200.utils import dataset_fingerprint, ensure_dir, seed_everything


def _append_jsonl(path: Path, row: dict[str, object]) -> None:
    """
    Append one record to a JSONL file (one JSON object per line).
    This keeps per-epoch metrics available even when W&B is disabled/offline.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def _device_from_arg(device: str | None) -> torch.device:
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Apple Silicon / macOS Metal backend
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _limit_dataloader_samples(
    dl: DataLoader,
    *,
    limit_samples: int | None,
    seed: int,
    shuffle: bool,
) -> DataLoader:
    """
    Return a new DataLoader backed by a deterministic Subset of `dl.dataset`.

    This is "smoke mode": run on a tiny subset to validate the pipeline end-to-end
    before launching long runs.
    """
    if limit_samples is None:
        return dl
    if limit_samples <= 0:
        raise ValueError(f"--limit-*-samples must be > 0, got {limit_samples}")

    ds = dl.dataset
    n = len(ds)
    if limit_samples >= n:
        return dl

    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n, generator=g)[:limit_samples].tolist()
    subset = Subset(ds, idx)

    # Preserve dataloader behavior; keep shuffling deterministic across runs.
    dl_gen = torch.Generator().manual_seed(seed)
    return DataLoader(
        subset,
        batch_size=dl.batch_size,
        shuffle=shuffle,
        num_workers=dl.num_workers,
        pin_memory=dl.pin_memory,
        drop_last=getattr(dl, "drop_last", False),
        persistent_workers=False,
        generator=dl_gen,
    )


class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing cross-entropy loss."""

    def __init__(self, smoothing: float = 0.1, num_classes: int = 200) -> None:
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))


class WarmupScheduler:
    """Wrapper for LR schedulers with warmup."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_scheduler: torch.optim.lr_scheduler._LRScheduler | ReduceLROnPlateau | None,
        warmup_epochs: int,
        base_lrs: list[float],
    ) -> None:
        self.optimizer = optimizer
        self.base_scheduler = base_scheduler
        self.warmup_epochs = warmup_epochs
        self.base_lrs = base_lrs
        self.current_epoch = 0

    def step(self, metrics: float | None = None) -> None:
        if self.current_epoch < self.warmup_epochs:
            # Linear warmup
            factor = (self.current_epoch + 1) / self.warmup_epochs
            for base_lr, param_group in zip(
                self.base_lrs, self.optimizer.param_groups, strict=False
            ):
                param_group["lr"] = base_lr * factor
        else:
            # Use base scheduler
            if self.base_scheduler is not None:
                if isinstance(self.base_scheduler, ReduceLROnPlateau):
                    if metrics is not None:
                        self.base_scheduler.step(metrics)
                else:
                    self.base_scheduler.step()
        self.current_epoch += 1

    def get_last_lr(self) -> list[float]:
        return [pg["lr"] for pg in self.optimizer.param_groups]

    def state_dict(self) -> dict:
        """Return scheduler state for checkpointing."""
        state = {
            "current_epoch": self.current_epoch,
            "warmup_epochs": self.warmup_epochs,
            "base_lrs": self.base_lrs,
        }
        if self.base_scheduler is not None:
            # Base scheduler state (if it has state_dict)
            if hasattr(self.base_scheduler, "state_dict"):
                state["base_scheduler_state"] = self.base_scheduler.state_dict()
        return state

    def load_state_dict(self, state: dict) -> None:
        """Load scheduler state from checkpoint."""
        self.current_epoch = state.get("current_epoch", 0)
        self.warmup_epochs = state.get("warmup_epochs", 0)
        self.base_lrs = state.get("base_lrs", [])
        if self.base_scheduler is not None and "base_scheduler_state" in state:
            if hasattr(self.base_scheduler, "load_state_dict"):
                self.base_scheduler.load_state_dict(state["base_scheduler_state"])


def train_one_epoch(
    model: nn.Module,
    dl,
    *,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_fn: nn.Module,
    scheduler: WarmupScheduler | None = None,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_top1 = 0.0
    n = 0

    for images, targets in tqdm(dl, desc="train", leave=False):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()

        bsz = images.shape[0]
        running_loss += loss.item() * bsz
        running_top1 += accuracy_topk(logits.detach(), targets, k=1).item() * bsz
        n += bsz

    return running_loss / max(1, n), running_top1 / max(1, n)


@torch.inference_mode()
def eval_one_epoch(
    model: nn.Module,
    dl,
    *,
    device: torch.device,
    loss_fn: nn.Module,
) -> tuple[float, float, float]:
    model.eval()
    running_loss = 0.0
    running_top1 = 0.0
    running_top5 = 0.0
    n = 0

    for images, targets in tqdm(dl, desc="val", leave=False):
        images = images.to(device)
        targets = targets.to(device)
        logits = model(images)
        loss = loss_fn(logits, targets)

        bsz = images.shape[0]
        running_loss += loss.item() * bsz
        running_top1 += accuracy_topk(logits, targets, k=1).item() * bsz
        running_top5 += accuracy_topk(logits, targets, k=5).item() * bsz
        n += bsz

    return running_loss / max(1, n), running_top1 / max(1, n), running_top5 / max(1, n)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--task", type=str, choices=["task1", "task2"], required=True)
    p.add_argument("--arch", type=str, default="resnet18")
    p.add_argument(
        "--pretrained",
        default=None,
        action=argparse.BooleanOptionalAction,
        help="Use pretrained weights (allowed for task2, forbidden for task1). "
        "If omitted: defaults to True for task2 and False for task1.",
    )
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Input image size. If omitted: task1 defaults to 64 (dataset-native), task2 defaults to 224.",
    )
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--lr-backbone",
        type=float,
        default=None,
        help="Backbone LR for fine-tuning (defaults to --lr if omitted).",
    )
    p.add_argument(
        "--lr-head",
        type=float,
        default=None,
        help="Head/classifier LR for fine-tuning (defaults to --lr if omitted).",
    )
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--output-dir", type=str, required=True)
    # Task 2 specific: fine-tuning strategy
    p.add_argument(
        "--fine-tune-strategy",
        type=str,
        choices=["frozen", "partial", "full"],
        default="full",
        help="For Task 2: frozen (backbone frozen), partial (last N layers), full (all trainable)",
    )
    p.add_argument(
        "--unfreeze-last-n",
        type=int,
        default=1,
        help="For partial strategy: number of backbone layers to unfreeze",
    )
    # Scheduler options
    p.add_argument(
        "--scheduler",
        type=str,
        choices=["none", "steplr", "cosine", "plateau"],
        default="none",
        help="LR scheduler type",
    )
    p.add_argument("--scheduler-step-size", type=int, default=30, help="For StepLR: step size")
    p.add_argument("--scheduler-gamma", type=float, default=0.1, help="For StepLR: gamma")
    p.add_argument(
        "--scheduler-t-max",
        type=int,
        default=None,
        help="For CosineAnnealing: T_max (defaults to epochs)",
    )
    p.add_argument("--warmup-epochs", type=int, default=0, help="Number of warmup epochs")
    # Optimizer choice
    p.add_argument(
        "--optimizer",
        type=str,
        choices=["adamw", "sgd"],
        default="adamw",
        help="Optimizer type",
    )
    p.add_argument("--momentum", type=float, default=0.9, help="For SGD: momentum")
    # Regularization
    p.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="Label smoothing factor (0.0 = no smoothing)",
    )
    # Smoke-mode / fast debug (tiny subset run)
    p.add_argument(
        "--limit-train-samples",
        type=int,
        default=None,
        help="If set, train on a deterministic subset of N samples (smoke mode).",
    )
    p.add_argument(
        "--limit-val-samples",
        type=int,
        default=None,
        help="If set, validate on a deterministic subset of N samples (smoke mode).",
    )
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML config file to override CLI defaults",
    )
    p.add_argument(
        "--notes",
        type=str,
        default="",
        help="Free-form notes: what changed in this run and why (persisted into run_meta.json via cli_args).",
    )
    # Optional experiment tracking (non-interactive)
    p.add_argument(
        "--wandb-project", type=str, default=None, help="If set, enable Weights & Biases logging."
    )
    p.add_argument("--wandb-run-name", type=str, default=None, help="Optional W&B run name.")
    p.add_argument(
        "--wandb-log-model",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="If enabled, log best checkpoint + metadata as a W&B artifact at the end.",
    )
    p.add_argument(
        "--slim-checkpoint",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="If enabled, save checkpoints without optimizer/scheduler state (saves ~50% space, but can't resume optimizer state).",
    )
    p.add_argument(
        "--early-stop-patience",
        type=int,
        default=None,
        help="Early stopping patience: stop if val_top1 doesn't improve for N epochs. Requires --early-stop-min-epochs.",
    )
    p.add_argument(
        "--early-stop-min-epochs",
        type=int,
        default=None,
        help="Minimum epochs before early stopping can trigger. Use with --early-stop-patience.",
    )
    args = p.parse_args()

    # Task 1 default preset (timm-inspired, Task1-adapted).
    # Applied only when: task1 + no YAML config + user didn't override relevant flags.
    if args.task == "task1" and args.config is None:
        # Snapshot argparse defaults for a small set of fields we may override.
        _defaults = {
            "optimizer": "adamw",
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "scheduler": "none",
            "warmup_epochs": 0,
            "label_smoothing": 0.0,
        }

        # timm-ish starting point (adapted; Task 1 constraints are enforced elsewhere)
        _timm_task1 = {
            "optimizer": "sgd",
            # timm's ImageNet script uses lr_base=0.1 at lr_base_size=256 with linear scaling.
            # With batch_size=128 (single-process), that corresponds to lr ≈ 0.05.
            "lr": 0.05,
            "weight_decay": 2e-5,
            "scheduler": "cosine",
            "warmup_epochs": 5,
            "label_smoothing": 0.1,
        }

        for k, v in _timm_task1.items():
            if hasattr(args, k) and getattr(args, k) == _defaults.get(k):
                setattr(args, k, v)

    # Optionally override defaults from YAML config
    if args.config is not None:
        import sys

        import yaml

        with open(args.config, "r", encoding="utf-8") as f:
            cfg_yaml = yaml.safe_load(f) or {}

        # Track which CLI arguments were explicitly provided
        # Map argparse dest names to their CLI flag names
        flag_to_dest: dict[str, str] = {}
        for action in getattr(p, "_actions", []):
            dest = getattr(action, "dest", None)
            if isinstance(dest, str) and dest != "help" and hasattr(action, "option_strings"):
                for option in action.option_strings:
                    if option.startswith("--"):
                        # Remove -- prefix and convert hyphens to underscores
                        flag_name = option[2:].replace("-", "_")
                        flag_to_dest[flag_name] = dest

        # Parse sys.argv to find explicitly provided flags
        provided_dests: set[str] = set()
        i = 1  # Skip script name
        while i < len(sys.argv):
            arg = sys.argv[i]
            if arg.startswith("--"):
                # Handle --flag=value format
                if "=" in arg:
                    flag_name = arg.split("=")[0][2:].replace("-", "_")
                else:
                    flag_name = arg[2:].replace("-", "_")

                # Map flag name to dest name
                if flag_name in flag_to_dest:
                    provided_dests.add(flag_to_dest[flag_name])
            i += 1

        # Apply YAML as defaults, but allow explicit CLI values to win.
        # This enables "hardware overrides" like `--device mps --batch-size 32` while using a base config.
        for key, value in cfg_yaml.items():
            if not hasattr(args, key):
                continue
            # Only override if user did not explicitly provide this flag
            if key not in provided_dests:
                setattr(args, key, value)

    # Task-specific defaults (after YAML overrides).
    if args.image_size is None:
        args.image_size = 64 if args.task == "task1" else 224
    if args.lr_head is None:
        args.lr_head = args.lr
    if args.lr_backbone is None:
        args.lr_backbone = args.lr

    # Make pretrained explicit (not only inferred from task).
    if args.pretrained is None:
        args.pretrained = args.task == "task2"
    if args.task == "task1" and args.pretrained:
        raise ValueError("Task 1 forbids pretrained weights. Use --no-pretrained.")

    seed_everything(args.seed)
    device = _device_from_arg(args.device)
    out = ensure_dir(args.output_dir)

    wandb_run = None
    if args.wandb_project:
        # Non-interactive: do NOT call wandb.login(). Use WANDB_API_KEY env var instead.
        if not os.environ.get("WANDB_API_KEY"):
            print(
                "[WARN] --wandb-project was provided but WANDB_API_KEY is not set. "
                "W&B init may fail unless you're already logged in."
            )
        try:
            import wandb  # type: ignore

            wandb_run = wandb.init(
                project=args.wandb_project, name=args.wandb_run_name, config=vars(args)
            )
        except Exception as e:
            print(f"[WARN] Failed to initialize W&B (continuing without it): {e}")
            wandb_run = None

    pretrained = bool(args.pretrained)
    smoke_mode = (args.limit_train_samples is not None) or (args.limit_val_samples is not None)
    if smoke_mode:
        print("=" * 72)
        print("SMOKE MODE ENABLED (fast debug run)")
        print(
            f"- limit_train_samples={args.limit_train_samples} | limit_val_samples={args.limit_val_samples} | "
            f"epochs={args.epochs} | batch_size={args.batch_size} | device={device}"
        )
        print("=" * 72)

    if args.task == "task2" and pretrained:
        print("Task2 + pretrained=True => fine-tuning mode enabled")

    data_cfg = DataConfig(
        data_root=Path(args.data_root),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        task=args.task,
        image_size=args.image_size,
    )
    train_dl, val_dl, idx_to_class, preprocess = build_dataloaders(data_cfg)
    num_classes = len(idx_to_class)

    if smoke_mode:
        train_dl = _limit_dataloader_samples(
            train_dl, limit_samples=args.limit_train_samples, seed=args.seed, shuffle=True
        )
        val_dl = _limit_dataloader_samples(
            val_dl, limit_samples=args.limit_val_samples, seed=args.seed, shuffle=False
        )

    # Basic dataset fingerprint for reproducibility
    ds_fingerprint = dataset_fingerprint(args.data_root)

    model = build_model(
        ModelConfig(arch=args.arch, num_classes=num_classes, pretrained=pretrained)
    ).to(device)

    # Apply fine-tuning strategy for Task 2 (even if pretrained=False; CI-safe + explicit mechanics)
    if args.task == "task2":
        if args.fine_tune_strategy == "frozen":
            freeze_backbone(model)
        elif args.fine_tune_strategy == "partial":
            unfreeze_last_n(model, args.unfreeze_last_n)
        # "full" means all layers trainable (default)

    total_params = count_total_params(model)
    trainable_params = count_trainable_params(model)
    trainable_params_m = trainable_params / 1e6
    trainable_pct = (100.0 * trainable_params / total_params) if total_params > 0 else 0.0

    fine_tune_strategy_for_meta = args.fine_tune_strategy if args.task == "task2" else "none"
    unfreeze_last_n_for_meta = (
        int(args.unfreeze_last_n)
        if (args.task == "task2" and args.fine_tune_strategy == "partial")
        else None
    )

    print(
        "Fine-tuning summary: "
        f"arch={args.arch} pretrained={pretrained} strategy={fine_tune_strategy_for_meta} "
        f"unfreeze_last_n={unfreeze_last_n_for_meta} "
        f"trainable_params={trainable_params_m:.2f}M ({trainable_pct:.1f}% trainable)"
    )

    # Setup optimizer
    backbone_params, head_params = split_backbone_and_head(model)
    head_trainable = [p for p in head_params if p.requires_grad]
    backbone_trainable = [p for p in backbone_params if p.requires_grad]

    param_groups: list[dict[str, object]] = []
    if backbone_trainable:
        param_groups.append({"params": backbone_trainable, "lr": float(args.lr_backbone)})
    if head_trainable:
        param_groups.append({"params": head_trainable, "lr": float(args.lr_head)})
    if not param_groups:
        raise RuntimeError(
            "No trainable parameters found. Check fine-tuning strategy / model freezing."
        )

    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    else:  # sgd
        optimizer = torch.optim.SGD(
            param_groups, weight_decay=args.weight_decay, momentum=args.momentum
        )

    # Setup loss function
    if args.label_smoothing > 0.0:
        loss_fn = LabelSmoothingCrossEntropy(
            smoothing=args.label_smoothing, num_classes=num_classes
        )
        print(f"Using label smoothing: {args.label_smoothing}")
    else:
        loss_fn = nn.CrossEntropyLoss()

    # Setup scheduler
    base_scheduler = None
    if args.scheduler == "steplr":
        base_scheduler = StepLR(
            optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma
        )
    elif args.scheduler == "cosine":
        t_max = args.scheduler_t_max if args.scheduler_t_max is not None else args.epochs
        base_scheduler = CosineAnnealingLR(optimizer, T_max=t_max)
    elif args.scheduler == "plateau":
        base_scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    scheduler = None
    if args.warmup_epochs > 0 or base_scheduler is not None:
        scheduler = WarmupScheduler(
            optimizer,
            base_scheduler,
            args.warmup_epochs,
            base_lrs=[pg["lr"] for pg in optimizer.param_groups],
        )
        print(f"Scheduler: {args.scheduler} with {args.warmup_epochs} warmup epochs")
    elif base_scheduler is not None:
        scheduler = WarmupScheduler(
            optimizer, base_scheduler, 0, base_lrs=[pg["lr"] for pg in optimizer.param_groups]
        )

    best_val_top1 = -1.0
    best_path = out / "checkpoint_best.pt"

    # Persist label mapping and preprocess settings for serving
    from cv200.utils import save_json

    save_json(out / "labels.json", {str(k): v for k, v in idx_to_class.items()})
    write_run_metadata(
        out,
        meta=CheckpointMeta(
            arch=args.arch,
            num_classes=num_classes,
            task=args.task,
            pretrained=pretrained,
            fine_tune_strategy=fine_tune_strategy_for_meta,
            unfreeze_last_n=unfreeze_last_n_for_meta,
            trainable_params_m=float(trainable_params_m),
            image_size=args.image_size,
            mean=preprocess.mean,
            std=preprocess.std,
            dataset_fingerprint=ds_fingerprint,
        ),
        preprocess=preprocess,
        cli_args=vars(args),
    )

    print(f"Device: {device}")
    print(f"Classes: {num_classes}")
    print(f"Output: {out}")

    # Clean old per-epoch checkpoints if they exist (prevents Drive bloat)
    if out.exists():
        for fname in os.listdir(out):
            if fname.startswith("checkpoint_epoch"):
                old_path = out / fname
                print(f"[INFO] Removing old checkpoint: {old_path}")
                old_path.unlink()

    # Resume logic: check for checkpoint_last.pt
    start_epoch = 1
    best_val_top1 = -1.0
    last_ckpt_path = out / "checkpoint_last.pt"

    if last_ckpt_path.exists():
        print(f"Resuming from checkpoint: {last_ckpt_path}")
        from cv200.checkpointing import load_checkpoint

        ckpt = load_checkpoint(last_ckpt_path, map_location=device)

        # Load model state
        model.load_state_dict(ckpt["model_state_dict"])

        # Load optimizer state (if present and not slim mode)
        if "optimizer_state_dict" in ckpt and not args.slim_checkpoint:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        # Load scheduler state (if present and not slim mode)
        if "scheduler_state_dict" in ckpt and scheduler is not None and not args.slim_checkpoint:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        # Resume from next epoch
        start_epoch = ckpt["epoch"] + 1
        best_val_top1 = ckpt.get("best_val_top1", -1.0)

        # Also check if best checkpoint exists and load its metric
        best_ckpt_path = out / "checkpoint_best.pt"
        if best_ckpt_path.exists():
            best_ckpt = load_checkpoint(best_ckpt_path, map_location=device)
            best_val_top1 = max(best_val_top1, best_ckpt.get("best_val_top1", -1.0))

        print(f"→ Resumed at epoch {start_epoch}, best val top1 = {best_val_top1:.4f}")
        if args.slim_checkpoint:
            print(
                "[INFO] Slim checkpoint mode: optimizer/scheduler state not restored (will restart)"
            )

    # Early stopping setup
    early_stop_enabled = (
        args.early_stop_patience is not None and args.early_stop_min_epochs is not None
    )
    if early_stop_enabled:
        if args.early_stop_patience <= 0:
            raise ValueError("--early-stop-patience must be > 0")
        if args.early_stop_min_epochs <= 0:
            raise ValueError("--early-stop-min-epochs must be > 0")
        print(
            f"[INFO] Early stopping enabled: patience={args.early_stop_patience}, min_epochs={args.early_stop_min_epochs}"
        )
        epochs_without_improvement = 0
        best_val_top1_seen = best_val_top1  # Track best seen during this run (for early stopping)

    metrics_path = out / "metrics.jsonl"
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss, train_top1 = train_one_epoch(
            model,
            train_dl,
            optimizer=optimizer,
            device=device,
            loss_fn=loss_fn,
            scheduler=scheduler,
        )
        val_loss, val_top1, val_top5 = eval_one_epoch(model, val_dl, device=device, loss_fn=loss_fn)
        dt = time.time() - t0

        # Update scheduler after epoch (for plateau, use val_loss)
        if scheduler is not None:
            if args.scheduler == "plateau":
                scheduler.step(metrics=val_loss)
            else:
                scheduler.step()

        current_lr = scheduler.get_last_lr()[0] if scheduler is not None else args.lr
        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"train loss {train_loss:.4f} top1 {train_top1:.4f} | "
            f"val loss {val_loss:.4f} top1 {val_top1:.4f} top5 {val_top5:.4f} | "
            f"lr {current_lr:.2e} | {dt:.1f}s"
        )

        lrs = scheduler.get_last_lr() if scheduler is not None else [float(args.lr)]
        metrics_row: dict[str, object] = {
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "train_top1": float(train_top1),
            "val_loss": float(val_loss),
            "val_top1": float(val_top1),
            "val_top5": float(val_top5),
            "lr_min": float(min(lrs)),
            "lr_max": float(max(lrs)),
            "step_time_s": float(dt),
        }
        for gi, lr_val in enumerate(lrs):
            metrics_row[f"lr_group_{gi}"] = float(lr_val)

        try:
            _append_jsonl(metrics_path, metrics_row)
        except Exception as e:
            print(f"[WARN] Failed to append metrics to {metrics_path}: {e}")

        if wandb_run is not None:
            try:
                import wandb  # type: ignore

                wandb.log(metrics_row, step=epoch)
            except Exception:
                # If logging fails mid-run, keep training.
                pass

        meta = CheckpointMeta(
            arch=args.arch,
            num_classes=num_classes,
            task=args.task,
            pretrained=pretrained,
            fine_tune_strategy=fine_tune_strategy_for_meta,
            unfreeze_last_n=unfreeze_last_n_for_meta,
            trainable_params_m=float(trainable_params_m),
            image_size=args.image_size,
            mean=preprocess.mean,
            std=preprocess.std,
            dataset_fingerprint=ds_fingerprint,
            train_top1=train_top1,
            val_top1=val_top1,
            val_top5=val_top5,
            epoch=epoch,
        )

        # Save checkpoint_last.pt (overwrite every epoch, atomic write)
        save_checkpoint(
            out / "checkpoint_last.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            meta=meta,
            scheduler=scheduler,
            best_val_top1=best_val_top1,
            include_optimizer=not args.slim_checkpoint,
            include_scheduler=not args.slim_checkpoint,
        )

        # Save checkpoint_best.pt (only on improvement, atomic write)
        if val_top1 > best_val_top1:
            best_val_top1 = val_top1
            save_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                meta=meta,
                scheduler=scheduler,
                best_val_top1=best_val_top1,
                include_optimizer=not args.slim_checkpoint,
                include_scheduler=not args.slim_checkpoint,
            )
            print(f"✓ New best model saved (val_top1={best_val_top1:.4f})")

        # Early stopping logic
        if early_stop_enabled:
            # Update best seen (for early stopping tracking)
            if val_top1 > best_val_top1_seen:
                best_val_top1_seen = val_top1
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # Check if we should stop
            if epoch >= args.early_stop_min_epochs:
                if epochs_without_improvement >= args.early_stop_patience:
                    print(
                        f"\n[EARLY STOP] No improvement for {epochs_without_improvement} epochs (since epoch {epoch - epochs_without_improvement})"
                    )
                    print(
                        f"  Best val_top1: {best_val_top1_seen:.4f} (at epoch {epoch - epochs_without_improvement})"
                    )
                    print(f"  Current val_top1: {val_top1:.4f}")
                    print(f"  Stopping at epoch {epoch} (saved ~{args.epochs - epoch} epochs)")
                    break

    if wandb_run is not None and args.wandb_log_model:
        try:
            import wandb  # type: ignore

            artifact = wandb.Artifact(name=f"{args.arch}-{args.task}", type="model")
            if best_path.exists():
                artifact.add_file(str(best_path))
            # Attach run metadata if present
            for fname in ("run_meta.json", "labels.json", "preprocess.json"):
                pth = out / fname
                if pth.exists():
                    artifact.add_file(str(pth))
            wandb.log_artifact(artifact)
        except Exception as e:
            print(f"[WARN] Failed to log model artifact to W&B: {e}")
        try:
            wandb.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
