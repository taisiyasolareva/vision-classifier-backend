## Task 2 Experiments – Pretrain + Resize

**Source of record:** The mandatory run matrix in **§0.1** is the source of record; summary metrics are in `reports/results_summary.md`. Sections 1–3 below are optional ablation templates (filled where we have runs).

Task 2 lifts the strict constraints of Task 1:

- **Pretraining allowed** – you can use ImageNet-pretrained backbones.
- **Resize allowed** – you can rescale inputs to a fixed resolution.
- **Same integrity rules**: no external data, no training on `val/`.

The goal is to build **strong, modern baselines** via transfer learning and then refine them with careful scheduling and regularization.

---

## Experiment discipline (“one change per run”)

To keep this report credible and interpretable, enforce the following rule:

> **Each run changes exactly one dimension**: fine-tune depth **OR** augmentation **OR** optimizer **OR** schedule.  
> If more than one thing changed, the run is not comparable and should be marked as invalid.

Always record:
- **What changed** and **why**
- Whether the run is **comparable** to the baseline it’s being compared against

---

## Run bookkeeping (required)

Use the same convention everywhere (folders, reports, and W&B):

- **Run ID**: `T2-<short>` (e.g., `T2-P1`)
- **Run folder**: `runs/<RUN_ID>/`
- **W&B run name**: exactly `<RUN_ID>`
- **Notes**: every run must include `--notes "<exactly one change + why>"`

W&B setup (non-interactive):

```bash
export WANDB_API_KEY="..."
export WANDB_MODE=online  # or "offline"
```

Template full run (W&B required):

```bash
python -m cv200.train \
  --data-root "/path/to/data_root" \
  --task task2 \
  --config "configs/<config>.yaml" \
  --wandb-project "cv200" \
  --wandb-run-name "T2-B0" \
  --notes "Baseline Task2 (one change per run thereafter)" \
  --output-dir "runs/T2-B0"
```

After training, evaluate the best checkpoint:

```bash
python -m cv200.eval --data-root "/path/to/data_root" --ckpt "runs/T2-B0/checkpoint_best.pt"
```

---

## Smoke run checklist (fast validation before long runs)

Run this **before** any full experiment. If this fails, do not start long trainings.

### Smoke (Task 2, offline-safe)

The smoke YAMLs are meant to validate **fine-tuning mechanics** quickly on CPU. For offline safety,
ensure `pretrained: false` in the smoke config (no downloads).

```bash
python -m cv200.train --data-root /path/to/data_root --task task2 \
  --config configs/smoke_task2_resnet50_frozen.yaml \
  --notes "smoke: task2 fine-tuning mechanics (offline-safe)" \
  --output-dir runs/SMOKE-T2-FROZEN
```

### Smoke pass criteria (must hold)

- `runs/SMOKE-T2-FROZEN/checkpoint_last.pt` exists
- `runs/SMOKE-T2-FROZEN/run_meta.json` exists
- Metrics printed are finite (no NaNs)

---

## 0. Required run matrix (must-run) + PASS/FAIL gate

**Task 2 PASS** (original assignment parity): **val_top1 ≥ 0.84** (no external data, no val leakage).

Run each config with the **same dataset** and **same seed** (default: `1337` in configs). Record the metrics in the table below.

### Best-so-far (tracked with completion status)

From `reports/results_summary.md` (auto-generated):

- **Best-so-far (incomplete)**: `T2-LR` — **val_top1 = 86.36%**, **val_top5 = 96.75%** (8/15 epochs) → **PASS**, but **incomplete**.
- **Best complete (currently recorded)**: `T2-ARCH-RUNNERUP-P1` — **val_top1 = 83.58%**, **val_top5 = 95.37%** (2/2 epochs) → **FAIL** (just below 84%).
- **Full-budget baselines**: `T2-P1` (**78.96%**) and `T2-P2` (**78.31%**) at 15/15 epochs.

Template command:

```bash
python -m cv200.train \
  --data-root "/path/to/data_root" \
  --task task2 \
  --config "configs/<config>.yaml" \
  --output-dir "./runs/<run_id>"
```

### 0.1 Mandatory Task 2 run matrix

| ID    | Config                           | Model         | Pretrained | Strategy | Unfreeze | LR(backbone) | LR(head) | Warmup | Scheduler | Trainable Params (M) | val_top1 | val_top5 | Overfit notes |
|------:|----------------------------------|---------------|-----------:|----------|---------:|-------------:|---------:|:------:|-----------|---------------------:|:--------:|:--------:|--------------|
| T2-B0-F | `configs/task2_resnet50_frozen.yaml`   | resnet50      | Yes        | frozen   | -        | 3e-5         | 3e-4     | 3      | cosine    |                      | 82.14%   | 94.47%   | Complete (15/15) |
| T2-P1 | `configs/task2_resnet50_partial1.yaml` | resnet50      | Yes        | partial  | 1        | 2e-5         | 2e-4     | 3      | cosine    |                      | 78.96%   | 92.56%   | Complete (15/15) |
| T2-P2 | `configs/task2_resnet50_partial2.yaml` | resnet50      | Yes        | partial  | 2        | 1.5e-5       | 1.5e-4   | 3      | cosine    |                      | 78.31%   | 92.77%   | Complete (15/15) |
| T2-FT | `configs/task2_resnet50_full.yaml`     | resnet50      | Yes        | full     | -        | 5e-5         | 1e-4     | 3      | cosine    |                      | 63.36%   | 86.06%   | Incomplete (1/15) |
| T2-SCOUT-CN-F | `configs/task2_convnext_tiny_full.yaml`| convnext_tiny | Yes        | frozen   | -        | (scout)      | (scout)  | (scout) | (scout)   |                      | 80.86%   | 94.04%   | Scout run (2/2 epochs): quick backbone check |

**PASS/FAIL summary:**
- Best-so-far run: **`T2-LR`** (incomplete, 8/15 epochs)
- Best-so-far `val_top1`: **86.36%** (**PASS**, but incomplete)
- Best complete `val_top1`: **83.58%** (`T2-ARCH-RUNNERUP-P1`, 2/2 epochs → FAIL)

---

## 1. Pretrained Baselines – Frozen vs Fine-Tuned

Start with classic transfer-learning setups for `resnet18` / `resnet50`:

- **Frozen backbone + linear head**.
- **Partially fine-tuned** (e.g. last N layers unfrozen).
- **Fully fine-tuned** (all layers trainable).

### 1.1 ResNet18 / ResNet50 Transfer Learning Matrix

Use a consistent augmentation / optimizer setup initially, then refine later.

| ID  | Model     | Pretrained | Strategy                  | Trainable Params (M) | val_top1 | val_top5 | Notes                            |
|-----|----------|-----------:|---------------------------|---------------------:|:--------:|:--------:|----------------------------------|
| R18-F | resnet18  | Yes       | Frozen backbone + head    |                      |          |          | Fast baseline                    |
| R18-P | resnet18  | Yes       | Last few layers unfrozen  |                      |          |          | Check marginal gain vs R18-F     |
| R18-A | resnet18  | Yes       | Fully fine-tuned          |                      |          |          | More capacity, risk of overfit   |
| R50-F | resnet50  | Yes       | Frozen backbone + head    |                      |          |          | Stronger backbone, cheap to try  |
| R50-P | resnet50  | Yes       | Last few layers unfrozen  |                      |          |          | Often a good trade-off           |
| R50-A | resnet50  | Yes       | Fully fine-tuned          |                      |          |          | Max capacity, needs good schedule|

Comment on:

- How much you gain by unfreezing more layers.
- Overfitting behavior vs dataset size.

---

## 2. Modern CNN/Hybrid Backbones (EfficientNet, ConvNeXt, etc.)

Optionally, add at least one more modern backbone:

- `efficientnet_b0` / `b3`
- `convnext_tiny` / `small`

### 2.1 Modern Backbone Comparison

| ID  | Model          | Pretrained | Strategy                | Params (M) | val_top1 | val_top5 | Notes                             |
|-----|---------------|-----------:|-------------------------|-----------:|:--------:|:--------:|-----------------------------------|
| E0  | efficientnet_b0 | Yes       | Fully fine-tuned        |            |          |          | Lightweight, strong baseline      |
| C0  | convnext_tiny | Yes        | Fully fine-tuned        |            |          |          | Stronger, more modern backbone    |

Focus on:

- Accuracy vs parameter count and latency (if you measure it).
- Any training instabilities or sensitivity to LR.

---

## 3. LR Warmup, Cosine Schedules & Batch Size

Once you have a good baseline (e.g. R50-P or ConvNeXt-Tiny), experiment with:

- **Learning rate warmup** (few epochs/steps).
- **Cosine decay** vs StepLR vs ReduceLROnPlateau.
- **Batch size** vs throughput and memory usage on your hardware.

### 3.1 Scheduler Ablations

Fix model and batch size, vary only scheduler and warmup:

| ID  | Model       | Warmup | Scheduler            | Base LR  | Epochs | val_top1 | val_top5 | Notes                        |
|-----|------------|:------:|----------------------|---------:|:------:|:--------:|:--------:|------------------------------|
| S1  | resnet50-P |  No    | StepLR               |          |        |          |          | Baseline                     |
| S2  | resnet50-P | Yes    | StepLR               |          |        |          |          | Warmup helps convergence?    |
| S3  | resnet50-P | Yes    | CosineAnnealing      |          |        |          |          | Common fine-tuning choice    |
| S4  | resnet50-P | Yes    | ReduceLROnPlateau    |          |        |          |          | Adaptive to val loss         |

### 3.2 Batch Size / Throughput Trade-offs

At least qualitatively record:

- Small vs large batch sizes (e.g. 64 vs 256) for your best model and schedule.
- Approximate GPU memory usage and steps/sec or images/sec.

| ID  | Model       | Batch Size | LR    | Epochs | val_top1 | val_top5 | GPU Mem (GB) | Notes                       |
|-----|------------|-----------:|------:|:------:|:--------:|:--------:|-------------:|-----------------------------|
| B1  | resnet50-P | 64         |       |        |          |          |              | Baseline                     |
| B2  | resnet50-P | 128        |       |        |          |          |              | Better throughput?           |
| B3  | resnet50-P | 256        |       |        |          |          |              | Might need LR scaling        |

---

## 4. Overfitting & Regularization

With powerful pretrained models, overfitting can be a concern, especially if the dataset is not huge.

Explore:

- **Weight decay** variations.
- **Dropout** in the classification head.
- Possibly **label smoothing**.

Track overfitting by:

- Monitoring train vs val loss / accuracy curves.
- Checking if stronger regularization helps validation metrics without hurting convergence too much.

You can integrate these results into existing tables (e.g., adding columns for weight decay / dropout) or create a separate small table.

---

## 5. Summary & Narrative

In 2–4 paragraphs, synthesize:

- **Best non-finetuned baseline**:
  - Which frozen-backbone configuration gives decent performance with minimal compute?
- **Best fine-tuned model**:
  - Which model + strategy + schedule achieved the highest `val_top1` / `val_top5`?
  - How much better is it than the frozen baseline?
- **Regularization & stability**:
  - What did you learn about overfitting and how to mitigate it (weight decay, dropout, augmentations)?
- **Cost vs performance**:
  - Comment on the trade-off between model size / training cost / inference latency versus accuracy.

This section will feed directly into your main report and README, showing **disciplined use of transfer learning** rather than just “I tried some pretrained models”.


