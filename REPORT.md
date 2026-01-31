# End-to-End Vision Classification System — Report (Portfolio Narrative)

**Project:** Production Computer Vision: 200‑Class Image Classification.  
**Purpose:** One project demonstrating end-to-end production ML ownership — data → training → evaluation → error analysis → export → serving — for senior ML engineer roles. Blog-post style: concise, honest, evidence-driven.

## Executive Summary 

- **Problem**: 200-class image classification under unknown domain and strict constraints (Task 1/Task 2).
- **What I built**: reproducible training + evaluation + error analysis + export + FastAPI serving.
- **Best results**:
  - Task 1 (no pretrain, no resize): **best-so-far val_top1 = 16.89%** (`T1-B0`, **incomplete** at 6/15 epochs) → **FAIL** vs threshold 44%.
  - Task 2 (pretrain + resize): **best-so-far val_top1 = 86.36%** (`T2-LR`, **incomplete** at 8/15 epochs) → **PASS** vs threshold 84%. Best complete (as currently recorded): **83.58%** (`T2-ARCH-RUNNERUP-P1`, 2/2 epochs) → **near-pass**.
- **Serving**: Live HF Space serves **T2-LR** (86.36% val_top1). **Live demo (UI):** [vision-classifier-ui.vercel.app](https://vision-classifier-ui.vercel.app/). Local CPU benchmarks: `POST /predict` p50 **37.16 ms**, p95 **41.48 ms**, throughput **26.54 img/s** (`reports/serving_benchmark.json`).
- **Key learnings**:
  - Optimization details (LR schedule / regularization) can dominate gains early: `T2-LR` and `T2-REG` outperformed the 15-epoch partial-unfreeze baselines before completion.
  - “Best-so-far” can be misleading if runs stop early; I track **complete vs incomplete** explicitly and re-run promising configs to completion before finalizing.

**What this demonstrates:** End-to-end ownership (no handoff gaps); rigor via run metadata and honest best-so-far vs best-complete; production mindset (TorchScript export, FastAPI, latency/throughput benchmarks); error analysis that drives next steps; reproducibility (CI, determinism) so a team could audit or extend; and explicit “what I’d do next” with trade-offs (early stopping, quantization, calibration).

## 1) Problem & Constraints

### Task 1 (constraint-based)
- No pretrained weights
- No resize (pad/crop OK)
- No external data
- No training on val

### Task 2 (performance-focused)
- Pretrained weights allowed
- Resize allowed
- Still no external data, no val leakage

## 2) Data & Validation

- Dataset layout and integrity checks: `scripts/validate_data.py`
- Dataset stats (EDA): `scripts/eda.py`, `scripts/profile_data.py`
- Key dataset facts (from EDA): **64×64**, **200 classes**, **balanced** (500/train/class)

## 3) Modeling Roadmap (what I tried, in order)

### 3.1 Task 1: from-scratch baselines
- Baseline(s): **ResNet18 from scratch** at **64×64** with Task 1–valid pad/crop transforms (`configs/task1_resnet18.yaml`, run family `T1-*`).
- Best-so-far: `T1-B0` reached **val_top1 16.89%** at **epoch 6/15** (incomplete).
- What didn’t work (yet): under the no-pretrain/no-resize constraint, the from-scratch baseline is still far from the **44%** PASS bar; next steps are completing the full schedule for the best configs and adding targeted regularization/augmentation ablations documented in `reports/task1_experiments.md`.

### 3.2 Task 2: transfer learning and fine-tuning

Run matrix (must-run): see `reports/task2_experiments.md` and configs under `configs/task2_*.yaml`.

- Frozen → partial unfreeze → full fine-tune
- Discriminative LRs (head vs backbone)
- Warmup + cosine schedule

## 4) Results

### Task 1 Results
- Best-so-far (incomplete): `T1-B0` — **val_top1 16.89%**, **val_top5 41.08%** (epoch 6/15) → FAIL vs 44% PASS bar.
- Best complete (short-run baseline): `T1-B0-FAST` — **val_top1 14.53%**, **val_top5 35.17%** (recorded as complete in `reports/results_summary.md`).
- Notes:
  - Task 1 is tracked as a strict “constraint mode” baseline and is currently incomplete vs the target threshold; the experiment log is kept in `reports/task1_experiments.md`.
  - Before claiming final Task 1 results, the plan is to (1) complete the best candidate run to full budget, then (2) run error analysis and targeted ablations.

### Task 2 Results
- Best-so-far (incomplete): `T2-LR` — **val_top1 86.36%**, **val_top5 96.75%** (epoch 8/15) → PASS vs 84% bar, but incomplete.
- Best complete (as currently recorded): `T2-ARCH-RUNNERUP-P1` — **val_top1 83.58%**, **val_top5 95.37%** (2/2 epochs) → just below PASS bar.
- Best complete full-budget baselines: `T2-P1` (**78.96%**) and `T2-P2` (**78.31%**) at 15/15 epochs.
- Notes / concrete findings:
  - **Partial unfreeze depth (P1 vs P2)**: `T2-P1` (78.96%) slightly outperformed `T2-P2` (78.31%) under the same 15-epoch budget → deeper unfreezing was not automatically better.
  - **LR / regularization tweaks show strong early signal**: `T2-LR` (86.36% at 8/15) and `T2-REG` (85.34% at 7/15) beat the full-budget partial-unfreeze baselines before completion, suggesting optimization knobs dominate late-stage performance.
  - **Architecture comparison is promising but needs full-budget confirmation**: runner-up runs at 2 epochs already reach ~83.5% top-1, indicating architecture choice matters; final selection requires completing the full training schedule on the strongest candidate.

**Honest note (best-so-far policy):** the current top-performing Task 2 run is **incomplete**. The next step is to **re-run the `T2-LR` configuration to completion (15/15 epochs)** (or apply early stopping with logged criteria) to validate that the gain holds before calling it the final best model.

## 5) Error Analysis (what the model gets wrong)

Artifacts (committed):
- `reports/error_analysis/T2-LR/per_class_accuracy.json`
- `reports/error_analysis/T2-LR/confusion_matrix.csv` + `reports/error_analysis/T2-LR/confusion_matrix.png`
- `reports/error_analysis/T2-LR/top_errors.json` (most confident wrong predictions)
- `reports/error_analysis/T2-LR/hardest_examples.json` (borderline correct predictions)
- `reports/error_analysis/T2-LR/error_summary.txt` (human-readable summary)

Summary:
- **Eval snapshot** (ConvNeXt-Tiny checkpoint analyzed in `reports/error_analysis/T2-LR/`): **val_top1 ≈ 87.10%** over **10,000** validation images (`reports/error_analysis/T2-LR/error_summary.txt`).
- **Most common confusions (true → pred)** by count:
  - `class_049 → class_168` (17)
  - `class_025 → class_015` (11)
  - `class_033 → class_072` (11)
  - `class_077 → class_099` (11)
  - `class_148 → class_041` (11)
- **Most error-prone classes** by error rate (each class has 50 val images): `class_039` (42%), `class_049` (42%), `class_033` (38%), `class_115` (36%), `class_148` (36%).
- **Interpretation / next steps**:
  - High-confidence wrong predictions in `top_errors.json` are the first place to check for systematic label ambiguity, visually similar classes, or preprocessing mismatches.
  - Use the worst-class list to drive targeted follow-ups (class-specific augmentations, calibration checks for overconfidence, or higher-resolution/stronger backbone validation if allowed).

## 6) Serving & Performance

Benchmarks:
- `scripts/benchmark_api.py` → `reports/serving_benchmark.json`
- Write-up: `reports/serving.md`

Key numbers:
- Single image p50/p95: **37.16 ms / 41.48 ms** (local CPU, `POST /predict`, 100 requests)
- Single image throughput: **26.54 img/s** (`reports/serving_benchmark.json`)
- Batch throughput (HF Spaces, includes network + shared infra): benchmarked at batch sizes **4/8/16** — see `reports/serving.md` and `reports/serving_benchmark_hf_b*.json`.

## 7) Reproducibility & Engineering

- Determinism guidance: `docs/REPRODUCIBILITY.md`
- CI: `.github/workflows/ci.yml`
- Tooling: `pyproject.toml`, `Makefile`
- Exportable artifact format for serving: `cv200.export`

## 8) Lessons Learned

- What worked:
  - A strict run naming + metadata scheme (`run_meta.json`, `metrics.jsonl`) made it easy to aggregate results across local + Colab/Drive runs and avoid “hand-copied” metrics.
  - Exporting a TorchScript artifact + serving behind FastAPI enabled realistic deployment and benchmarking early (before final model selection).
- What failed / constraints:
  - Long-running GPU experiments can be interrupted; incomplete runs can look “best” mid-training. The fix is explicitly tracking completion status and re-running promising configs to completion.
  - Task 1 is materially harder under strict constraints; it needs additional careful experimentation (not just “turn the knobs from Task 2”).
- What I’d do next:
  - Confirm the best-so-far config with a full-budget run and add early stopping + checkpoint selection criteria.
  - Expand serving benchmarks (batch sizes, concurrency) and evaluate trade-offs (quantization / distillation) if latency or memory become constraints.
  - Run full error analysis on the final best checkpoint and use it to drive a small set of targeted improvements (augmentation policy, calibration, or architectural swap).








