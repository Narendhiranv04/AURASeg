# ABLATION_11_DEC - AURASeg Ablation Study

**Paper:** AURASeg - Attention Guided Upsampling with Residual Boundary-Assistive Refinement for Drivable-Area Segmentation

**Date:** December 11, 2025

---

## Architecture Overview

This ablation study implements the **correct** architecture from the paper:

### Encoder: CSPDarknet-53
```
Input (3, H, W)
    ↓
Focus Layer → (32, H/2, W/2)
    ↓
Conv1 (CSP) → (64, H/4, W/4)      → c1 (skip)
    ↓
Conv2 (CSP) → (128, H/8, W/8)     → c2 (skip)
    ↓
Conv3 (CSP) → (256, H/16, W/16)   → c3 (skip)
    ↓
Conv4 (CSP) → (512, H/32, W/32)   → c4 (skip)
    ↓
Conv5 (CSP) → (1024, H/32, W/32)  → c5 (to SPP/ASPP-Lite)
```

### Context Modules

**V1 - SPP (Spatial Pyramid Pooling):**
- MaxPool kernels: 5, 9, 13
- Input: 1024 channels, Output: 256 channels

**V2 - ASPP-Lite (4 branches, as per paper):**
- Branch 1: Conv 1×1, dilation=1, 128 filters
- Branch 2: Conv 3×3, dilation=1, 128 filters
- Branch 3: Conv 3×3, dilation=6, 128 filters
- Branch 4: Conv 3×3, dilation=12, 128 filters
- Fusion: Concatenate (512 ch) → Conv 1×1 → 256 channels

### Decoder
```
ASPP/SPP output (256, H/32)
    ↓ + c4
Block 4 → (256, H/16)
    ↓ + c3
Block 3 → (128, H/8)
    ↓ + c2
Block 2 → (64, H/4)
    ↓ + c1
Block 1 → (32, H/2)
    ↓
Output → (num_classes, H, W)
```

---

## Directory Structure

```
ABLATION_11_DEC/
├── models/
│   ├── __init__.py          # Module exports
│   ├── backbone.py          # CSPDarknet-53, Focus, CSPBlock, SPP
│   ├── aspp_lite.py         # ASPP-Lite (4 branches)
│   ├── decoder.py           # Decoder with skip connections
│   ├── v1_base_spp.py       # V1: CSPDarknet + SPP + Decoder
│   └── v2_base_assplite.py  # V2: CSPDarknet + ASPP-Lite + Decoder
├── losses.py                 # Focal, Dice, Combined, Boundary losses
├── config.py                 # Training configuration
├── utils.py                  # Dataset, metrics, helpers
├── train.py                  # Training script
├── runs/                     # Training outputs
│   ├── v1_base_spp/
│   │   ├── checkpoints/
│   │   ├── logs/
│   │   └── visualizations/
│   └── v2_base_assplite/
│       ├── checkpoints/
│       ├── logs/
│       └── visualizations/
└── README.md                 # This file
```

---

## Usage

### Train V1 (CSPDarknet + SPP baseline)
```bash
cd c:\Users\naren\Documents\AURASeg
python kobuki-yolop/model/new_network/ABLATION_11_DEC/train.py --model v1_base_spp --epochs 50 --batch-size 8
```

### Train V2 (CSPDarknet + ASPP-Lite)
```bash
cd c:\Users\naren\Documents\AURASeg
python kobuki-yolop/model/new_network/ABLATION_11_DEC/train.py --model v2_base_assplite --epochs 50 --batch-size 8
```

### Resume Training
```bash
python kobuki-yolop/model/new_network/ABLATION_11_DEC/train.py --model v1_base_spp --resume path/to/checkpoint.pth
```

### Custom Data Paths
```bash
python kobuki-yolop/model/new_network/ABLATION_11_DEC/train.py \
    --model v1_base_spp \
    --image-dir "kobuki-yolop/final_dataset/images" \
    --mask-dir "kobuki-yolop/final_dataset/bdd_seg_gt"
```

---

## Loss Function

**V1 & V2:** Combined Loss = Focal Loss + Dice Loss

- **Focal Loss:** Addresses class imbalance (α=0.25, γ=2.0)
- **Dice Loss:** Directly optimizes segmentation overlap

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Image Size | (384, 640) |
| Batch Size | 8 |
| Epochs | 50 |
| Optimizer | AdamW |
| Learning Rate | 1e-3 |
| Scheduler | Cosine Annealing |
| Weight Decay | 1e-4 |
| Early Stopping | 15 epochs |
| Mixed Precision | Enabled |

---

## Ablation Study Plan

| Version | Encoder | Context Module | Decoder | Loss |
|---------|---------|----------------|---------|------|
| V1 | CSPDarknet-53 | SPP | 4-block | Focal + Dice |
| V2 | CSPDarknet-53 | ASPP-Lite | 4-block | Focal + Dice |
| V3 (future) | CSPDarknet-53 | ASPP-Lite | 4-block + Attention | Focal + Dice |
| V4 (future) | CSPDarknet-53 | ASPP-Lite | 4-block + Attention + RBRM | Focal + Dice + Boundary |

---

## Expected Results

After training, results will be saved in:
- `runs/v1_base_spp/checkpoints/best.pth` - Best V1 model
- `runs/v2_base_assplite/checkpoints/best.pth` - Best V2 model
- `runs/*/logs/` - TensorBoard logs
- `runs/*/visualizations/` - Prediction visualizations

---

## Notes

- This implementation uses the **correct CSPDarknet-53** encoder as described in the paper
- Previous implementations incorrectly used ResNet-18
- ASPP-Lite now has **4 branches** (1×1 d=1, 3×3 d=1, 3×3 d=6, 3×3 d=12) as per paper Figure 3
