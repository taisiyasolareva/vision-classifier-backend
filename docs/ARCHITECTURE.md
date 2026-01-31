# Architecture

This document gives a high-level view of the **End-to-End Vision Classification System** (Production Computer Vision: 200‑Class Image Classification): training, evaluation, export, and serving.

## System Architecture (end-to-end)

```
Client (curl / web / batch job)
        |
        | HTTP (multipart/form-data)
        v
FastAPI (src/cv200/api.py)
  - validates input (type/size/dims)
  - loads artifact (TorchScript + labels + preprocess)
  - runs inference
        |
        v
Response (top-k class predictions)
```

## Data Pipeline (training/eval)

```
ImageFolder layout
data_root/
  train/<class_id>/*.jpg
  val/<class_id>/*.jpg
        |
        v
Task-specific transforms (src/cv200/data.py)
  - Task 1: pad + crop (NO resize)
  - Task 2: resize + stronger aug
        |
        v
DataLoader
        |
        v
Model forward pass (src/cv200/models.py)
```

## Training Pipeline

```
Config (CLI + optional YAML)  ->  cv200.train
        |
        v
Build dataloaders + transforms
        |
        v
Build model
  - Task 1: from scratch
  - Task 2: optional pretrained + fine-tuning strategies (frozen/partial/full)
        |
        v
Train loop (epochs)
  - metrics: train_loss/train_top1, val_loss/val_top1/val_top5
  - checkpoints: checkpoint_last.pt, checkpoint_best.pt
  - metadata: run_meta.json (args + preprocess + fingerprint)
  - metrics logging: metrics.jsonl (per-epoch JSON Lines format)
```

**Metrics Logging**: Each epoch's metrics are appended to `metrics.jsonl` in JSON Lines format. This enables easy extraction of final results and learning curve analysis without requiring W&B.

## Serving Pipeline

```
Export (cv200.export)
  -> artifact/
     - model.ts          (TorchScript traced model)
     - labels.json       (class_id -> class_name mapping)
     - preprocess.json   (transform config: task, image_size, mean, std)
     - run_meta.json     (optional: full run metadata for traceability)
        |
        v
FastAPI loads artifact on demand (cached via @lru_cache)
  - Reads MODEL_ARTIFACT_DIR from environment (server-side only)
  - Validates artifact files exist (model.ts, labels.json, preprocess.json)
  - Builds preprocessing pipeline from preprocess.json
        |
        v
Request -> Image validation -> Preprocess -> TorchScript inference -> Top-k -> JSON response
```

**Security Note**: The API uses `MODEL_ARTIFACT_DIR` environment variable (server-side only). Clients cannot provide filesystem paths, ensuring safe public deployment.








