# Serving Performance Report

This report documents inference performance for the **End-to-End Vision Classification System** FastAPI service (`src/cv200/api.py`) using the benchmark script `scripts/benchmark_api.py`.

## Hardware / Environment

This repo includes two benchmark contexts:

### Local (developer machine)

- **Machine**: MacBook Air (Apple Silicon)
- **OS**: macOS
- **CPU**: Apple Silicon (M-series)
- **GPU**: Not used (CPU inference only)
- **Note**: measured locally against `http://127.0.0.1:8000` (no network); not directly comparable to HF timings.
- **Runtime (dependency versions)**: FastAPI 0.115.6, Uvicorn 0.34.0 (`requirements.txt`)
- **Core ML deps (constraints)**: torch >= 2.9.0 (< 3.0.0), torchvision >= 0.21.0 (< 1.0.0) (`requirements.txt`)
- **Python support**: 3.9–3.11 (`README.md`, `pyproject.toml`); CI uses Python 3.10 (`.github/workflows/ci.yml`)
- **Service runtime**: Uvicorn + FastAPI (single process)

### Deployed (Hugging Face Spaces)

- **API base**: `https://solarevat-cv200.hf.space`
- **Live UI**: [https://vision-classifier-ui.vercel.app/](https://vision-classifier-ui.vercel.app/)
- **Artifact identity (server-reported)**: `artifact_dir=/app/artifact`, `artifact_ok=true` (from `/healthz`, captured in `reports/serving_benchmark_hf_*.json`)
- **Note**: includes network overhead + shared deployment resources, so it is not directly comparable to local CPU timings.

## Methodology

We benchmark using:

### Deployed HF Spaces (recommended for portfolio)

```bash
python scripts/benchmark_api.py \
  --url https://solarevat-cv200.hf.space \
  --num-requests 100 \
  --batch-size 1 \
  --out reports/serving_benchmark_hf_b1.json
```

### Local (optional)

```bash
MODEL_ARTIFACT_DIR=/path/to/artifact uvicorn cv200.api:app --host 0.0.0.0 --port 8000
python scripts/benchmark_api.py --url http://127.0.0.1:8000 --num-requests 100 --batch-size 1 --out reports/serving_benchmark.json
```

- **Endpoints**:
  - `POST /predict` for `batch-size=1`
  - `POST /predict_batch` for `batch-size>1`
- **Metrics**:
  - **Latency**: p50 / p95 / p99 (ms)
  - **Throughput**: requests/sec and images/sec
- **Artifacts**:
  - Raw JSON output is saved to: `reports/serving_benchmark.json`

## Single-Image Performance (batch_size=1)

### Deployed HF Spaces

**Results** (from `reports/serving_benchmark_hf_b1.json`):

| Endpoint | batch_size | p50 (ms) | p95 (ms) | p99 (ms) | req/s | img/s | Notes |
|---------:|-----------:|---------:|---------:|---------:|------:|------:|------|
| /predict | 1          | 888.22   | 2337.94  | 4992.48  | 0.91  | 0.91  | Mean latency: 1100.59 ms |

### Local (developer machine)

**Results** (from `reports/serving_benchmark.json`):

| Endpoint | batch_size | p50 (ms) | p95 (ms) | p99 (ms) | req/s | img/s | Notes |
|---------:|-----------:|---------:|---------:|---------:|------:|------:|------|
| /predict | 1          | 37.16    | 41.48    | 42.53    | 26.54 | 26.54 | Mean latency: 37.62 ms |

### What this means (practical interpretation)

- **HF vs local**: deployed numbers include network + shared infra, so they appear much slower than local loopback benchmarks.
- **Batching**: use `/predict_batch` to increase **throughput** (img/s) at the cost of per-request latency.

## Batch Performance (latency vs batch size)

Benchmark a few batch sizes:

```bash
python scripts/benchmark_api.py --url http://localhost:8000 --artifact-dir runs/task2/artifact --num-requests 100 --batch-size 1
python scripts/benchmark_api.py --url http://localhost:8000 --artifact-dir runs/task2/artifact --num-requests 100 --batch-size 4
python scripts/benchmark_api.py --url http://localhost:8000 --artifact-dir runs/task2/artifact --num-requests 100 --batch-size 8
python scripts/benchmark_api.py --url http://localhost:8000 --artifact-dir runs/task2/artifact --num-requests 100 --batch-size 16
```

**Results**:

| Endpoint       | batch_size | p50 (ms) | p95 (ms) | img/s | Notes |
|---------------:|-----------:|---------:|---------:|------:|------|
| /predict       | 1          | 888.22   | 2337.94  | 0.91  | HF Spaces (`reports/serving_benchmark_hf_b1.json`) |
| /predict_batch | 4          | 1691.15  | 3468.23  | 1.73  | HF Spaces (`reports/serving_benchmark_hf_b4.json`) |
| /predict_batch | 8          | 1950.43  | 2323.39  | 4.04  | HF Spaces (`reports/serving_benchmark_hf_b8.json`) |
| /predict_batch | 16         | 2632.88  | 4133.95  | 5.84  | HF Spaces (`reports/serving_benchmark_hf_b16.json`) |

## Comparison Table (optional, strong signal)

If you benchmark multiple exported artifacts (different models / sizes), record them here:

| Model / Artifact | Endpoint | batch_size | p50 (ms) | p95 (ms) | img/s | Notes |
|------------------|---------:|-----------:|---------:|---------:|------:|------|
| ResNet50 (`SMOKE-T2-FULL`) | /predict | 1 | 37.16 | 41.48 | 26.54 | CPU inference, single process |
| HF deployed (`artifact_dir=/app/artifact`) | /predict_batch | 16 | 2632.88 | 4133.95 | 5.84 | HF Spaces (`reports/serving_benchmark_hf_b16.json`) |

## Takeaways

- **Primary bottleneck**: CPU inference (no GPU acceleration). Single-threaded processing limits throughput.
- **Performance**: ~37ms per image (p50) with 26.5 images/sec throughput on CPU. P95 latency stays under 42ms, indicating consistent performance.
- **Best trade-off** (accuracy vs latency): Local CPU numbers above use artifact from SMOKE-T2-FULL; best-model (T2-LR) latency on HF Spaces is documented in the batch tables. Latency depends on model size and preprocessing.
- **Next optimizations**:
  - [x] TorchScript export (already using optimized model.ts)
  - [ ] Batch sizing & concurrency tuning (test batch_size > 1 for better throughput)
  - [ ] GPU acceleration (if available, would significantly improve throughput)
  - [ ] Quantization / distillation (if latency-constrained)








