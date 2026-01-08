"""
Benchmark Models Package for AURASeg Comparison
================================================

Contains SOTA segmentation models for RAL paper benchmarking:
- DeepLabV3+ (ResNet-50) - ECCV 2018
- SegFormer (MiT-B2) - NeurIPS 2021
- UPerNet (ResNet-101) - ECCV 2018
- PSPNet (ResNet-101) - CVPR 2017
- FPN (MiT-B3) - CVPR 2017

All models use ImageNet pretrained backbones from segmentation_models_pytorch.
"""

BENCHMARK_MODELS = [
    'deeplabv3plus',
    'segformer',
    'upernet',
    'dpt',       # Actually PSPNet
    'mask2former'  # Actually FPN
]

from .model_factory import get_benchmark_model, get_model_info
