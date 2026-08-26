# AURASeg reproducibility notes

This document records the implementation and evaluation conventions used by the current AURASeg paper/preprint.

## Environment

- Python 3.8+
- PyTorch 2.x
- CUDA-capable NVIDIA GPU for training
- Training GPU used for the reported AURASeg experiments: NVIDIA GeForce RTX 5060
- Input resolution: 384 x 640
- Number of classes: 2 (drivable / non-drivable)
- Random seed: 42
- Mixed precision: enabled

## AURASeg training

Canonical entry point:

```bash
python benchmark_models/train_auraseg_r18_wacv.py \
  --fusion-type mul \
  --attention-mode full \
  --use-sobel \
  --use-gate \
  --seed 42
```

The paper configuration uses:

- ImageNet-pretrained ResNet-18 encoder
- APUD width: 128 channels
- AdamW optimizer
- encoder learning rate: 1e-4
- decoder learning rate: 1e-3
- weight decay: 0.01
- cosine annealing with minimum learning rate 1e-6
- maximum 50 epochs
- micro-batch size 4
- gradient accumulation: 2 steps
- effective batch size: 8
- validation batch size: 4
- early stopping patience: 10
- minimum validation improvement: 1e-4
- checkpoint selection: validation mIoU

The training script writes a `config.json` for each run with the resolved configuration and git commit.

## Loss

The main segmentation objective is

```text
0.5 * focal + 0.5 * Dice
```

with focal alpha 0.25, gamma 2, and Dice smoothing 1.0.

RBRM boundary supervision uses a target produced from the semantic mask with a 3x3 morphological gradient and an unweighted BCE-with-logits objective with weight 0.2.

Each of the four APUD stages has a training-only auxiliary segmentation head. The summed auxiliary focal + Dice objectives have weight 0.1.

Implementation: `benchmark_models/wacv_losses.py`.

## Data augmentation and normalization

Training augmentation:

- horizontal flip, p=0.5
- shift +/-0.1, scale +/-0.1, rotation +/-15 degrees, p=0.5
- brightness/contrast +/-0.2, p=0.3
- Gaussian noise variance range 10-50, p=0.2

ImageNet normalization:

```text
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

Discrete masks are resized with nearest-neighbor interpolation.

## Dataset splits

### Gazebo

- train: 2,483
- validation: 294
- test: 420

### GMRPD

- train: 3,000
- validation: 360
- test: 536

### MIX

MIX is the fixed union of the Gazebo and GMRPD held-out partitions used by the paper. It is a benchmark composition, not an additional dataset.

### CARL-D

- train: 8,372
- validation: 1,046
- test: 1,046

CARL-D masks are decoded before resizing. The current loader maps:

```text
road RGB        (17, 163, 74) -> 1
black fallback  (0, 0, 0)     -> 1
background RGB  (15, 16, 65)  -> 0
```

Unexpected RGB values raise an error. The implementation is in `benchmark_models/unified_dataset.py`.

## Region metrics

Region metrics are accumulated over foreground pixels. The paper reports foreground IoU and F1 in the main comparison table. Precision, recall, accuracy and mIoU are also computed by the toolkit.

Implementation: `benchmark_models/wacv_metrics.py`.

## Boundary metrics

For each prediction/target pair:

1. convert to binary foreground masks;
2. extract each contour using a 3x3 morphological gradient;
3. dilate both predicted and target boundaries by `k` iterations;
4. compute boundary IoU, precision, recall and F1;
5. macro-average boundary metrics over images.

The main paper uses `k=2` for every compared model. The diagnostic sensitivity table also reports `k=1` and `k=3` for the same AURASeg predictions.

Implementation: `benchmark_models/wacv_metrics.py`.

## Baseline comparison protocol

The reported comparison is controlled at the dataset split, input resolution and evaluation levels. Common optimization settings are used where compatible. Architecture-specific objectives are retained when they are integral to a baseline's defining formulation.

The purpose of the table is therefore a controlled same-data/same-resolution/same-evaluation comparison. It should not be interpreted as a claim that every baseline is reproduced at its independently optimized published training schedule.

General-baseline and boundary-focused training/adaptation scripts are kept in `benchmark_models/`.

## Evaluation toolkit release

The model definition, training entry points, split/loading logic, CARL-D RGB decoding, loss implementation, region metrics and boundary metrics used by the current paper are available in this repository. This file documents the paper-facing protocol so that the numerical evaluation can be reproduced without relying on legacy experiment folders.
