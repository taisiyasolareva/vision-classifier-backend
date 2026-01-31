# End-to-End Vision Classification System — API (vision-classifier-api)

**Production Computer Vision: 200‑Class Image Classification.**  
Reproducible training + disciplined experiments + TorchScript export + FastAPI serving + React demo. *Portfolio piece demonstrating end-to-end production ML ownership for senior ML engineer roles.*

This repository is the **API + training pipeline** (GitHub repo: **vision-classifier-backend**). The React UI lives in the separate repo **vision-classifier-frontend**.

**Proof points:** Top‑1 86.36% / Top‑5 96.75% *(best‑so‑far; run completion tracked)* · CPU p50 ~37 ms per image *(FastAPI, local benchmark)* · Dockerized + CI + clear run bookkeeping.

**CTAs:** [Read the Report](REPORT.md) · [API Docs](https://solarevat-cv200.hf.space/docs) · Live Demo: link when frontend is deployed.

## Live API

- **HF Spaces API base URL**: `https://solarevat-cv200.hf.space`
  - **API docs**: `https://solarevat-cv200.hf.space/docs`
  - **Health**: `https://solarevat-cv200.hf.space/healthz`
  - **Deployed model**: **T2-LR** (best-so-far, 86.36% val_top1, 8/15 epochs).

## Results (explicit best-so-far vs best-complete policy)

Source of truth: `reports/results_summary.md` (auto-generated).

**Policy**:
- **Best-so-far** may be **incomplete** (stopped early / interrupted). It is shown because it indicates promising configurations.
- **Best complete** is the best run that meets the completion threshold recorded by the summarizer.

### Task 2 (pretrain + resize allowed)

- **Best-so-far (INCOMPLETE)**: `T2-LR` — **top-1 86.36% / top-5 96.75%** (8/15 epochs)
- **Best complete (recorded)**: `T2-ARCH-RUNNERUP-P1` — **top-1 83.58% / top-5 95.37%** (2/2 epochs)

### Task 1 (no pretrain, no resize)

- **Best-so-far (INCOMPLETE)**: `T1-B0` — **top-1 16.89% / top-5 41.08%** (6/15 epochs)

## Serving performance (local CPU)

From `REPORT.md` / `reports/serving.md`:
- `POST /predict` p50 **37.16 ms**, p95 **41.48 ms**, throughput **26.54 img/s** (batch_size=1)

## API (backend routes)

The backend serves these **root** endpoints:
- `GET /healthz`
- `POST /predict`
- `POST /predict_batch`
- `GET /docs`

Note: the frontend dev server may proxy requests under `/api/*`, but that is **not** a backend route prefix.

## Install

Prereqs: Python 3.9–3.11 (GPU optional).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

If editable install fails, run with:

```bash
PYTHONPATH=src python -m cv200.train --help
```

## Quick start (local API)

### 1) Train (requires dataset)

```bash
python -m cv200.train --data-root /path/to/data_root --task task2 --output-dir runs/T2-LOCAL
```

### 2) Export (checkpoint → artifact)

```bash
python -m cv200.export --ckpt runs/T2-LOCAL/checkpoint_best.pt --output runs/T2-LOCAL/artifact
```

### 3) Serve

```bash
MODEL_ARTIFACT_DIR=./runs/T2-LOCAL/artifact \
  uvicorn cv200.api:app --host 0.0.0.0 --port 8000
```

### 4) Predict

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@image.jpg" \
  -F "top_k=5"
```

## Dataset notes

The original dataset is not redistributed in this repo. See `DATASET.md` for the required layout, constraints, and licensing notes.

## Reproducibility / architecture / report (recommended reading)

- Portfolio narrative: `REPORT.md`
- Architecture: `docs/ARCHITECTURE.md`
- Reproducibility: `docs/REPRODUCIBILITY.md`
- Best results (auto-generated): `reports/results_summary.md`
- Serving benchmarks: `reports/serving.md`
- Error analysis artifacts: `reports/error_analysis/T2-LR/`