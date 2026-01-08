"""
Benchmark Models for AURASeg Comparison
========================================

This module provides state-of-the-art segmentation models for benchmarking
against AURASeg (V1-V4) in the RAL paper.

Models:
    1. DeepLabV3+ (ResNet-50) - ECCV 2018 - CNN + ASPP
    2. SegFormer (MiT-B2) - NeurIPS 2021 - Transformer
    3. UPerNet (Swin-T) - ECCV 2018 + Swin 2021 - Hierarchical Transformer
    4. DPT (ViT-Base) - ICCV 2021 - Dense Prediction Transformer
    5. Mask2Former (Swin-S) - CVPR 2022 - Universal Segmentation

All models use ImageNet pretrained backbones for fair comparison.
"""

from .model_factory import (
    get_benchmark_model,
    get_model_info,
    BENCHMARK_MODELS,
)

__all__ = [
    'get_benchmark_model',
    'get_model_info',
    'BENCHMARK_MODELS',
]
