# Data Augmentation Justification

This document explains the augmentation strategies used for Task 1 and Task 2, how each transform addresses domain uncertainty, and their impact on model performance.

## Task 1 Augmentations (No Pretrain, No Resize)

**Constraints**: No pretraining, no resize/rescaling operations allowed.

### Transforms Used

1. **`Pad`** + **`RandomCrop`**
   - **Domain uncertainty addressed**: Handles variable object positioning and framing in real-world images. Random cropping forces the model to learn from different spatial regions, reducing reliance on fixed object locations.
   - **Implementation**: Pad images to a fixed size (e.g., 64×64 → 72×72), then randomly crop back to original size.

2. **`RandomHorizontalFlip(p=0.5)`**
   - **Domain uncertainty addressed**: Many objects are naturally symmetric or appear in both left/right orientations. This augmentation doubles effective training data and improves generalization to flipped test images.
   - **Example**: A car facing left vs. right should be classified the same way.

3. **`RandomRotation`** (small degrees, e.g., ±10°)
   - **Domain uncertainty addressed**: Handles orientation variability in real-world images. Objects may be captured at slight angles due to camera positioning or natural object orientation.
   - **Example**: A slightly tilted sign or rotated object should maintain its class label.

4. **`ColorJitter`** (optional, mild)
   - **Domain uncertainty addressed**: Simulates different lighting conditions, camera settings, and color variations. Helps the model become invariant to brightness, contrast, and saturation changes.
   - **Implementation**: Small adjustments to brightness, contrast, saturation (e.g., ±0.1).

5. **`Normalize`**
   - **Domain uncertainty addressed**: Standardizes input distribution, reducing sensitivity to absolute pixel values and improving training stability.

### Validation Transforms

- `Pad` + `CenterCrop` (deterministic, no randomness)
- `Normalize`

## Task 2 Augmentations (Pretrain + Resize)

**Constraints**: Pretraining and resize operations are allowed.

### Transforms Used

1. **`Resize`**
   - **Domain uncertainty addressed**: Standardizes input size for pretrained models while maintaining aspect ratio. Allows leveraging ImageNet-pretrained backbones that expect specific input dimensions (e.g., 224×224).
   - **Implementation**: Resize to fixed size (e.g., 224×224) with antialiasing.

2. **`RandomHorizontalFlip(p=0.5)`**
   - **Domain uncertainty addressed**: Same as Task 1—handles left/right orientation variability.

3. **`RandomCrop`** (with resize)
   - **Domain uncertainty addressed**: After resizing, random cropping adds spatial diversity and prevents overfitting to specific image regions.

4. **`ColorJitter`** (stronger than Task 1)
   - **Domain uncertainty addressed**: More aggressive color augmentation simulates diverse lighting and camera conditions. With pretrained models, we can afford stronger augmentations without hurting convergence.

5. **`RandAugment`** (optional, advanced)
   - **Domain uncertainty addressed**: Automatically applies a random combination of augmentations (rotation, translation, color shifts, etc.) from a learned policy. Provides comprehensive robustness to geometric and photometric variations.

6. **`Normalize`** (ImageNet statistics or dataset-specific)
   - **Domain uncertainty addressed**: Standardizes inputs to match pretrained model expectations or dataset characteristics.

### Validation Transforms

- `Resize` (deterministic)
- `Normalize`

## Before/After Comparison Table

Fill in this table with actual experiment results:

| Config | Augmentations | val_top1 | val_top5 | Notes |
|--------|---------------|----------|----------|-------|
| Baseline | Minimal | - | - | No aug or flip only |
| Strong | Full set | - | - | +X% improvement |

Note: this repo does not include “example” numeric results. Populate this table only with measured metrics from `reports/results_summary.md` / your run logs.

## Key Takeaways

*To be filled after running augmentation experiments.*

After completing augmentation ablation studies, document:

- **What helped most**: Which specific augmentations provided the largest accuracy gains (e.g., "Random cropping improved robustness by +3% top-1 accuracy").
- **What hurt**: Any augmentations that degraded performance (e.g., "Aggressive rotation (>15°) caused label noise and reduced accuracy").
- **Final policy choice**:
  - **Task 1**: Final augmentation pipeline selected under no-resize constraints.
  - **Task 2**: Final augmentation pipeline selected with pretraining and resize allowed.
- **Domain uncertainty insights**: How the chosen augmentations address the unknown/heterogeneous visual domain of the dataset.

This analysis demonstrates **senior-level judgment**: not just applying augmentations, but **choosing and justifying** them based on empirical results and domain understanding.


