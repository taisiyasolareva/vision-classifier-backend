# Dataset Documentation

## Dataset Structure

The dataset is expected in a standard `ImageFolder` layout compatible with `torchvision.datasets.ImageFolder`:

```text
data_root/
  train/
    0/  img1.jpg, img2.jpg, ...
    1/  ...
    ...
    199/  ...
  val/
    0/  ...
    1/  ...
    ...
    199/  ...
```

- **Train/val split**: 
  - `train/`: Used exclusively for training models
  - `val/`: Used exclusively for model selection and reporting validation metrics
- **Number of classes**: 200 (class indices `0` through `199`)
- **Images per class (this project’s dataset instance)**: 500 training images per class (see EDA findings below)

> **Note (public repo)**: The dataset itself is **not** committed to this repository. You must download or prepare it separately and point `--data-root` to its location when running training/evaluation scripts.
>
> If you keep data locally under this repo (e.g., `./data/...`), it must remain **gitignored** (see `.gitignore`). Do not publish course/private datasets.

## Licensing

The original assignment dataset was provided as part of a course and is **not redistributed** here for licensing reasons.

To reproduce this project for portfolio purposes, you can:

- Use a **similarly structured public dataset** such as:
  - **ImageNet subset**: A curated subset of ImageNet with 200 classes
  - **CIFAR-100 extended**: Extended version with additional classes
  - **Tiny-ImageNet**: 200 classes, 500 training images per class
  - Other multi-class natural image datasets converted into the `ImageFolder` layout above
- Clearly state in your own fork/deployment which public dataset you used and under which license

When using any dataset:

- Respect the original license and terms of use
- Do **not** include private or proprietary images in any public repository
- Ensure compliance with dataset-specific usage restrictions

## Constraints

The project follows strict experimental constraints to ensure fair comparison and realistic production scenarios:

### Task 1 (Constraint Track)

- **No pretraining**: Models must be trained from scratch on the training set only
- **No resize operation**: You may use padding and cropping, but **no rescaling / `Resize` transform** that changes the pixel scale of the image
- **No external data**: Do not augment training with external datasets
- **No training on validation data**: The `val/` directory is for evaluation only

### Task 2 (Performance Track)

- **Pretraining allowed**: You may use ImageNet-pretrained models or similar pretrained backbones
- **Resize allowed**: You may rescale inputs to a fixed resolution for modern backbones
- **Same data integrity rules**:
  - No external training data
  - No training on validation set

### General Rules (Both Tasks)

- **No external data for training**: Only use the provided train/val splits
- **No training on validation set**: The validation set is strictly for evaluation and model selection

### Documentation Requirements

In all experiments, document:

- Which dataset you used (name, source, version/date)
- Any filtering/preprocessing you applied (e.g., removing corrupt images, filtering classes)
- The exact train/val split and any random seeds used to create it (if applicable)

## Exploratory Data Analysis (EDA)

To better understand the dataset characteristics, run the EDA script:

```bash
python scripts/eda.py \
  --data-root /path/to/data_root \
  --output-dir reports/eda
```

**Note**: If your dataset is nested (e.g., extracted from a zip), adjust the path accordingly (example):

```bash
python scripts/eda.py \
  --data-root /path/to/extracted_root \
  --output-dir reports/eda
```

### Generated Artifacts

The EDA script generates the following outputs in `reports/eda/`:

- **`class_counts.json`** – Raw per-class image counts (JSON format)
- **`class_counts.png`** – Bar plot visualization of class distribution (committed; contains counts only, no raw images)
- **`resolution_stats.json`** – Image resolution statistics (min/median/max width and height)
- **`samples_grid.png`** – (optional) 2×5 grid montage showing sample images from 10 randomly selected classes
  - If your dataset is not redistributable (e.g., a course dataset), **do not commit** this file in a public repo.

### Visualizations (optional)

If you generate plots locally (e.g., `class_counts.png`), do not commit them to a public repo if the underlying images are restricted. The JSON outputs (`class_counts.json`, `resolution_stats.json`) are sufficient to verify dataset structure and counts.

### Key Findings

Based on the EDA results:

- **Class Balance**: The dataset is perfectly balanced with 500 training images per class across all 200 classes. This eliminates the need for class weighting or sampling strategies to handle imbalance.

- **Resolution Characteristics**: All images are uniformly sized at 64×64 pixels. This fixed resolution simplifies preprocessing:
  - For **Task 1** (no resize): Images can be used directly with padding/cropping strategies
  - For **Task 2** (resize allowed): Images can be upscaled to standard input sizes (e.g., 224×224) for pretrained models

- **Dataset Scale**: With 200 classes × 500 images = 100,000 training images, this is a mid-scale classification task that requires careful regularization to prevent overfitting.

These findings inform augmentation choices and model design decisions documented in the experiment reports.



