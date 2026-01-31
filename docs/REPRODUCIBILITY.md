# Reproducibility & Determinism

This project aims for **best-effort reproducibility** (same code + same data + same config → very similar results).
Exact bit-for-bit determinism can be difficult in deep learning due to GPU kernels, parallel data loading, and nondeterministic ops.

## What we do in code

- **Seeding**: `cv200.utils.seed_everything(seed)` seeds:
  - Python's `random` module
  - NumPy's RNG (`np.random.seed`)
  - PyTorch's CPU RNG (`torch.manual_seed`)
  - PyTorch's CUDA RNG (`torch.cuda.manual_seed_all`)
  - Python hash seed (`PYTHONHASHSEED` environment variable)
- **CuDNN settings**:
  - `torch.backends.cudnn.deterministic = True` (use deterministic algorithms)
  - `torch.backends.cudnn.benchmark = False` (more reproducible, potentially slower)

`cv200.train` calls `seed_everything(--seed)` at startup. The seed is persisted into `run_meta.json` under `cli_args.seed` (all CLI arguments are saved via `vars(args)`).

## Common sources of nondeterminism

- **DataLoader parallelism**: `num_workers > 0` can change sample ordering and timing; augmentation randomness can diverge between workers.
- **Random augmentations**: transforms like `RandomCrop`, `RandomRotation`, `RandAugment`, etc. are stochastic by design.
- **GPU kernels / floating point**: some CUDA operations can be nondeterministic or have slight numeric differences between runs/hardware.
- **Library versions**: changing `torch/torchvision` versions (or even CPU BLAS) can change results.

## Paths in reports and JSON artifacts

- **Run paths** in `reports/results_task1.json`, `reports/results_task2.json`, and `reports/experiment_summary.json` may be local (`runs/<RUN_ID>`) or from Colab/Drive; for public sharing they are redacted to generic `runs/<RUN_ID>` so the repo does not leak machine-specific or personal paths.
- **Error analysis** (`reports/error_analysis/*/top_errors.json`, `hardest_examples.json`) stores image paths as relative paths (e.g. `val/class_000/00001.jpg`) so artifacts are shareable without local paths.

## Best-effort reproduction steps

1. **Fix your environment**
   - Use a consistent Python version and keep `requirements.txt` the same.
   - Record GPU + driver + CUDA versions if you train on GPU.

2. **Use the same data**
   - Ensure the dataset path and contents are unchanged.
   - This repo computes a dataset fingerprint (SHA256 hash of file paths and sizes) and saves it in `run_meta.json` under `meta.dataset_fingerprint`.
   - You can verify data consistency by comparing fingerprints across runs.

3. **Run with a fixed seed**

```bash
python -m cv200.train \
  --data-root "/path/to/data_root" \
  --task task1 \
  --arch resnet18 \
  --seed 1337 \
  --output-dir "./runs/repro_task1"
```

4. **For maximum determinism**
   - Set `--num-workers 0`
   - Prefer CPU runs when validating exact determinism

```bash
python -m cv200.train \
  --data-root "/path/to/data_root" \
  --task task1 \
  --arch resnet18 \
  --seed 1337 \
  --num-workers 0 \
  --device cpu \
  --output-dir "./runs/repro_task1_cpu"
```

## What “reproducible” means here (practically)

- **Same run setup** should produce **very similar** learning curves and final metrics.
- If you need **exact** determinism, run on CPU with `--num-workers 0`, and keep all versions fixed.








