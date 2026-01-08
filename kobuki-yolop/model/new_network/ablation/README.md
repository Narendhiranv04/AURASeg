# Ablation Study for AURASeg

This directory contains the complete ablation study setup for the AURASeg drivable area segmentation model.

## Overview

The ablation study progressively adds components to evaluate their contribution:

| Version | Model | Components |
|---------|-------|------------|
| **V1** | Base | CSPDarknet + SPP + Simple Decoder |
| **V2** | +ASSPLite | V1 + Lightweight ASPP (coming soon) |
| **V3** | +APUD | V2 + Attention Pyramid Upsampling Decoder (coming soon) |
| **V4** | +RBRM | V3 + Residual Boundary Refinement Module (coming soon) |

## Quick Start

### 1. Install Requirements

```bash
pip install -r requirements.txt
```

### 2. Run Base Ablation (V1)

```bash
python run_base_ablation.py \
    --image-dir /path/to/images \
    --mask-dir /path/to/masks \
    --epochs 100 \
    --batch-size 4
```

Or use the full training script with config:

```bash
python train_ablation.py \
    --model ablation_v1_base \
    --image-dir /path/to/images \
    --mask-dir /path/to/masks
```

## Files

```
ablation/
├── __init__.py              # Package initialization
├── config.py                # Configuration management
├── losses.py                # Focal + Dice losses with deep supervision
├── ablation_v1_base.py      # Base model (CSPDarknet + SPP + Simple decoder)
├── train_ablation.py        # Full training script with metrics/plots
├── run_base_ablation.py     # Quick-start script for base model
├── utils.py                 # Metrics, visualization utilities
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## Configuration

All hyperparameters are managed through dataclasses in `config.py`:

```python
from config import get_base_config

config = get_base_config()

# Modify as needed
config.training.epochs = 100
config.optimizer.learning_rate = 1e-4
config.loss.lambda_focal = 1.0
config.loss.lambda_dice = 1.0

# Save config
config.save("my_config.json")
```

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | 100 | Number of training epochs |
| `batch_size` | 4 | Batch size |
| `learning_rate` | 1e-4 | Initial learning rate |
| `lambda_focal` | 1.0 | Focal loss weight |
| `lambda_dice` | 1.0 | Dice loss weight |
| `focal_alpha` | 0.25 | Focal loss alpha (class balance) |
| `focal_gamma` | 2.0 | Focal loss gamma (focusing) |
| `supervision_weights` | [0.1, 0.2, 0.3, 0.4] | Deep supervision weights (coarse to fine) |

## Loss Function

The combined loss uses:

$$L_{total} = \lambda_f \cdot L_{focal} + \lambda_d \cdot L_{dice}$$

For models with deep supervision (V3, V4):

$$L_{total} = \sum_{i=1}^{4} w_i \cdot (\lambda_f \cdot L_{focal}^i + \lambda_d \cdot L_{dice}^i)$$

Where $w_i$ follows a decaying schedule: `[0.1, 0.2, 0.3, 0.4]` (higher weight for finer scales).

## Outputs

Training produces:

```
runs/ablation/v1_base/
├── checkpoints/
│   ├── best.pth              # Best model (highest val mIoU)
│   ├── latest.pth            # Latest checkpoint
│   └── epoch_N.pth           # Periodic checkpoints
├── predictions/
│   └── epoch_N_sample_K.png  # Visualization samples
├── tensorboard/              # TensorBoard logs
├── config.json               # Saved configuration
├── training_history.json     # Training metrics history
└── training_history.png      # Training curves plot
```

## Metrics

The following metrics are computed:

- **mIoU**: Mean Intersection over Union
- **mDice**: Mean Dice Score
- **Per-class IoU/Dice**: For each class
- **Precision/Recall/F1**: For drivable area class
- **Accuracy**: Overall pixel accuracy

## Model Architecture (V1 Base)

```
Input (3, 640, 384)
    │
    ▼
┌─────────────────┐
│   CSPDarknet    │
│   Encoder       │
│                 │
│ Focus ─────────────► c1 (64, 320, 192)
│   │             │
│ Conv+CSP ──────────► c2 (128, 160, 96)
│   │             │
│ Conv+CSP ──────────► c3 (256, 80, 48)
│   │             │
│ Conv+CSP ──────────► c4 (512, 40, 24)
│   │             │
│ Conv+SPP+CSP ──────► c5 (1024, 20, 12)
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  Simple Decoder │
│  (Bilinear Up   │
│   + Skip Conn)  │
└─────────────────┘
    │
    ▼
Output (2, 640, 384)
```

## TensorBoard

Monitor training with TensorBoard:

```bash
tensorboard --logdir runs/ablation/
```

## Citation

If you use this code, please cite:

```bibtex
@article{auraseg2024,
  title={AURASeg: Attention-aware Upsampling for Real-time Autonomous Segmentation},
  author={Your Name},
  year={2024}
}
```
