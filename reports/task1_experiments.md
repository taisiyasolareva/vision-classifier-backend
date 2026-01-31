## Task 1 Experiments – No Pretrain, No Resize

**Source of record:** Summary metrics are in `reports/results_summary.md`. The tables below are filled from completed runs; empty rows are templates for future ablations.

This document tracks experiments for **Task 1** under the strict constraints:

- **No pretraining** – all models trained from scratch on the train set.
- **No resize / rescaling** – only padding + cropping allowed (no `Resize` that changes pixel scale).
- **No external data** and **no training on `val/`**.

The goal is to start from a **simple baseline** and iteratively improve it in a disciplined way.

---

## 0. Run bookkeeping (required)

To make results credible and comparable, use a single convention:

- **Run ID**: `T1-<short>` (e.g., `T1-B0`)
- **Run folder**: `runs/<RUN_ID>/`
- **W&B run name**: exactly `<RUN_ID>`
- **Notes**: every run must include `--notes "<exactly one change + why>"`

**Task 1 PASS gate** (assignment parity): **val_top1 ≥ 0.44** (no pretrain, no resize, no external data, no val leakage).

Template full run command (W&B required for full runs):

```bash
python -m cv200.train \
  --data-root "/path/to/data_root" \
  --task task1 \
  --config "configs/<config>.yaml" \
  --wandb-project "cv200" \
  --wandb-run-name "T1-B0" \
  --notes "Baseline Task1: resnet18 scratch, default augs, AdamW" \
  --output-dir "runs/T1-B0"
```

After training, evaluate the best checkpoint (to generate confusion matrix and per-class metrics artifacts):

```bash
python -m cv200.eval --data-root "/path/to/data_root" --ckpt "runs/T1-B0/checkpoint_best.pt"
```

---

## Smoke run checklist (fast validation before long runs)

Run this **before** any full experiment. If this fails, do not start long trainings.

### Smoke (Task 1, CPU OK)

```bash
python -m cv200.train --data-root /path/to/data_root --task task1 --arch resnet18 \
  --epochs 1 --batch-size 16 --num-workers 0 --device cpu \
  --limit-train-samples 32 --limit-val-samples 32 \
  --notes "smoke: task1 pipeline" \
  --output-dir runs/SMOKE-T1
```

### Smoke pass criteria (must hold)

- `runs/SMOKE-T1/checkpoint_last.pt` exists
- `runs/SMOKE-T1/run_meta.json` exists
- Metrics printed are finite (no NaNs)

---

## 1. Baseline Architectures

### 1.1 Initial Baseline (Small CNN or ResNet18 from Scratch)

- **Model**: `resnet18` (from scratch)  
- **Augmentations**: minimal, Task 1–valid pad/crop (see `configs/task1_resnet18.yaml`)  
- **Optimizer / Scheduler**: AdamW + cosine (timm-style config)  
- **Metrics** (from `reports/results_task1.json`):
  - **T1-B0**: `val_top1 = 16.89%`, `val_top5 = 41.08%` (6/15 epochs, incomplete)
  - **T1-B0-FAST**: `val_top1 = 14.53%`, `val_top5 = 35.17%` (5/1 epochs, complete short-run baseline)

### 1.2 Alternative Architectures

Try at least one alternative (e.g. lighter ConvNet, ResNet-34 from scratch).

| ID  | Model        | Params (M) | Augmentations                  | Optimizer / Scheduler | val_top1 | val_top5 | Notes                             |
|-----|-------------|-----------:|--------------------------------|-----------------------|:--------:|:--------:|-----------------------------------|
| T1-B0  | resnet18     | 11.28 | basic crop+flip (Task 1)       | AdamW + cosine        | 16.89%   | 41.08%   | Baseline (6/15 epochs, incomplete) |
| T1-B0-FAST | resnet18 | 11.28 | basic crop+flip                | AdamW + cosine        | 14.53%   | 35.17%   | Short-run complete baseline       |
| T2  | custom_cnn   |           | basic crop+flip                | AdamW + StepLR        |          |          | Template: smaller, compare capacity |
| T3  | resnet34     |           | basic crop+flip                | SGD + cosine          |          |          | Template: deeper model, scratch   |

> You can estimate parameter counts via tools like `torchinfo.summary` during local exploration.

---

## 2. Augmentation Ablations

Study the effect of **augmentation strength** under the no-resize constraint.

### 2.1 Augmentation Variants

Suggested progression:

- **A1** – Minimal: `Pad + RandomCrop + HorizontalFlip`
- **A2** – +Rotation: add small `RandomRotation`
- **A3** – +Perspective / stronger spatial transforms (still no resize)
- **A4** – +Mild color jitter (if visually reasonable for this dataset)

Fill in results for a fixed architecture (e.g. resnet18 from scratch):

| ID  | Model    | Augmentations                                                | val_top1 | val_top5 | Δ vs A1 | Notes                               |
|-----|---------|--------------------------------------------------------------|:--------:|:--------:|:-------:|-------------------------------------|
| A1  | resnet18 | Pad+RandomCrop+HFlip                                       | 16.89    | 41.08    |   –     | Baseline (T1-B0, 6/15 epochs)      |
| A2  | resnet18 | A1 + small rotation                                         |          |          |         | Check robustness to orientation     |
| A3  | resnet18 | A2 + mild perspective                                      |          |          |         | Stronger spatial aug                |
| A4  | resnet18 | A3 + mild color jitter                                     |          |          |         | Domain uncertainty, lighting shift  |

---

## 3. Optimizer & LR Schedule Ablations

Keep the **model and augmentations fixed** (e.g. best config from Section 2), and vary optimizer and LR schedule.

| ID  | Model    | Optimizer         | LR Schedule      | Epochs | val_top1 | val_top5 | Notes                                  |
|-----|---------|-------------------|------------------|:------:|:--------:|:--------:|----------------------------------------|
| O1  | resnet18 | SGD + momentum    | StepLR           |        |          |          | Classical baseline                     |
| O2  | resnet18 | SGD + momentum    | CosineAnnealing  |        |          |          | Smoother decay                         |
| O3  | resnet18 | AdamW             | StepLR           |        |          |          | Often faster convergence               |
| O4  | resnet18 | AdamW             | CosineAnnealing  |        |          |          | Combine AdamW + cosine                 |

Comment on:

- Convergence speed (epochs/time to reach certain accuracy).
- Stability (spikes, overfitting).

---

## 4. Crop Strategy Ablations (Respecting “No Resize”)

Compare different strategies for handling variable image sizes without resizing:

- **C1** – `Pad + RandomCrop(image_size, image_size)`
- **C2** – `Pad + CenterCrop(image_size, image_size)` (less variation)
- **C3** – Random resized crop with scale constrained to **avoid** effective resampling (if you attempt something advanced, document it carefully).

| ID  | Model    | Crop Strategy                 | val_top1 | val_top5 | Notes                                      |
|-----|---------|-------------------------------|:--------:|:--------:|--------------------------------------------|
| C1  | resnet18 | Pad+RandomCrop                |          |          | More variation, better robustness          |
| C2  | resnet18 | Pad+CenterCrop                |          |          | More stable, possibly less robust          |
| C3  | resnet18 | (Custom, documented clearly) |          |          | Must still satisfy “no-resize” constraint  |

---

## 5. Summary & Narrative

Once the above tables are filled, write a short summary (1–3 paragraphs):

- **Baseline**: what was your simplest working model and accuracy?
- **Key gains**:
  - Which augmentations provided the biggest lift?
  - Which optimizer/schedule combination worked best and why?
  - Which crop strategy balanced robustness and stability?
- **Final Task 1 configuration**:
  - Model architecture.
  - Augmentation policy.
  - Optimizer/scheduler.
  - Achieved `val_top1` / `val_top5`.

This narrative should read like an **experiment log from a senior ML engineer**: clear baselines, controlled changes, and data-driven conclusions.


