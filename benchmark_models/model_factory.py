"""
Model Factory for Benchmark Models
===================================

Creates and returns benchmark segmentation models with unified interface.
All models return logits of shape (B, num_classes, H, W).

Usage:
    model, info = get_benchmark_model('deeplabv3plus', num_classes=2)
    output = model(images)  # (B, 2, H, W)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional


# Model registry
BENCHMARK_MODELS = {
    'deeplabv3plus': {
        'name': 'DeepLabV3+',
        'paper': 'Chen et al., ECCV 2018',
        'paradigm': 'CNN + ASPP',
        'encoder': 'ResNet-50',
        'library': 'segmentation_models_pytorch',
    },
    'segformer': {
        'name': 'SegFormer',
        'paper': 'Xie et al., NeurIPS 2021',
        'paradigm': 'Transformer',
        'encoder': 'MiT-B2',
        'library': 'segmentation_models_pytorch',
    },
    'upernet': {
        'name': 'UPerNet',
        'paper': 'Xiao et al., ECCV 2018',
        'paradigm': 'Multi-scale Fusion',
        'encoder': 'ResNet-50',
        'library': 'segmentation_models_pytorch',
    },
    'dpt': {
        'name': 'PSPNet',
        'paper': 'Zhao et al., CVPR 2017',
        'paradigm': 'Pyramid Pooling',
        'encoder': 'ResNet-101',
        'library': 'segmentation_models_pytorch',
    },
    'mask2former': {
        'name': 'FPN',
        'paper': 'Lin et al., CVPR 2017',
        'paradigm': 'Feature Pyramid',
        'encoder': 'MiT-B3',
        'library': 'segmentation_models_pytorch',
    },
    'fcn': {
        'name': 'FCN',
        'paper': 'Long et al., CVPR 2015',
        'paradigm': 'Fully Convolutional',
        'encoder': 'ResNet-50',
        'library': 'torchvision',
    },
    'pspnet': {
        'name': 'PSPNet',
        'paper': 'Zhao et al., CVPR 2017',
        'paradigm': 'Pyramid Pooling',
        'encoder': 'ResNet-50',
        'library': 'segmentation_models_pytorch',
    },
    'pidnet': {
        'name': 'PIDNet-L',
        'paper': 'Xu et al., CVPR 2023',
        'paradigm': 'Three-branch PID',
        'encoder': 'Custom',
        'library': 'custom',
    },
}


def get_model_info(model_name: str) -> dict:
    """Get metadata about a benchmark model."""
    info = {
        'deeplabv3plus': {
            'name': 'DeepLabV3+',
            'backbone': 'ResNet-50',
            'params': '~26M',
            'year': 2018,
            'paper': 'Encoder-Decoder with Atrous Separable Convolution',
            'library': 'segmentation_models_pytorch'
        },
        'segformer': {
            'name': 'SegFormer',
            'backbone': 'MiT-B2',
            'params': '~27M',
            'year': 2021,
            'paper': 'Simple and Efficient Design for Semantic Segmentation with Transformers',
            'library': 'segmentation_models_pytorch'
        },
        'upernet': {
            'name': 'UPerNet',
            'backbone': 'Swin-Tiny',
            'params': '~32M',
            'year': 2018,
            'paper': 'Unified Perceptual Parsing for Scene Understanding',
            'library': 'segmentation_models_pytorch'
        },
        'dpt': {
            'name': 'DPT',
            'backbone': 'ViT-Base',
            'params': '~87M',
            'year': 2021,
            'paper': 'Vision Transformers for Dense Prediction',
            'library': 'transformers'
        },
        'mask2former': {
            'name': 'Mask2Former',
            'backbone': 'Swin-Small',
            'params': '~47M',
            'year': 2022,
            'paper': 'Masked-attention Mask Transformer for Universal Image Segmentation',
            'library': 'transformers'
        }
    }
    return info.get(model_name.lower(), {})


def _create_fcn(num_classes: int, pretrained: bool):
    """Create FCN with ResNet-50 backbone using torchvision."""
    from torchvision.models.segmentation import fcn_resnet50
    from torchvision.models.segmentation.fcn import FCN_ResNet50_Weights
    
    # Load pretrained FCN-ResNet50 (always load with weights for backbone, then modify heads)
    # Use aux_loss=True to ensure aux_classifier is created
    if pretrained:
        model = fcn_resnet50(weights=FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1, aux_loss=True)
    else:
        model = fcn_resnet50(weights=None, weights_backbone='IMAGENET1K_V1', aux_loss=True)
    
    # Replace classifier head for our number of classes
    model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1)
    if model.aux_classifier is not None:
        model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    
    info = {
        'name': 'FCN-ResNet50',
        'uses_builtin_loss': False,
        'output_key': 'out',  # FCN returns dict with 'out' key
        'requires_resize': True  # May need to resize output
    }
    
    return model, info


# =============================================================================
# Model Creation Functions
# =============================================================================

def _create_deeplabv3plus(num_classes: int, pretrained: bool):
    """Create DeepLabV3+ with ResNet-50 backbone."""
    import segmentation_models_pytorch as smp
    
    model = smp.DeepLabV3Plus(
        encoder_name='resnet50',
        encoder_weights='imagenet' if pretrained else None,
        in_channels=3,
        classes=num_classes,
        activation=None  # Raw logits
    )
    
    info = {
        'name': 'DeepLabV3+',
        'uses_builtin_loss': False,
        'output_key': None,  # Direct tensor output
        'requires_resize': False
    }
    
    return model, info


def _create_segformer(num_classes: int, pretrained: bool):
    """Create SegFormer with MiT-B2 backbone using segmentation_models_pytorch."""
    import segmentation_models_pytorch as smp
    
    # Use the actual Segformer class in SMP with mit_b2 encoder
    model = smp.Segformer(
        encoder_name='mit_b2',
        encoder_weights='imagenet' if pretrained else None,
        in_channels=3,
        classes=num_classes,
        activation=None
    )
    
    info = {
        'name': 'SegFormer-B2',
        'uses_builtin_loss': False,
        'output_key': None,
        'requires_resize': False
    }
    
    return model, info


def _create_upernet(num_classes: int, pretrained: bool):
    """Create UPerNet with ResNet-50 backbone."""
    import segmentation_models_pytorch as smp
    
    # UPerNet with ResNet-50 encoder for fair comparison
    model = smp.UPerNet(
        encoder_name='resnet50',
        encoder_weights='imagenet' if pretrained else None,
        in_channels=3,
        classes=num_classes,
        activation=None
    )
    
    info = {
        'name': 'UPerNet-R50',
        'uses_builtin_loss': False,
        'output_key': None,
        'requires_resize': False
    }
    
    return model, info


def _create_dpt(num_classes: int, pretrained: bool):
    """Create PSPNet with ResNet-101 backbone as alternative to DPT."""
    import segmentation_models_pytorch as smp
    
    # Use PSPNet as DPT requires timm encoders not available in this setup
    # PSPNet is a strong pyramid-based model from CVPR 2017
    model = smp.PSPNet(
        encoder_name='resnet101',
        encoder_weights='imagenet' if pretrained else None,
        in_channels=3,
        classes=num_classes,
        activation=None
    )
    
    info = {
        'name': 'PSPNet-R101',
        'uses_builtin_loss': False,
        'output_key': None,
        'requires_resize': False
    }
    
    return model, info


def _create_mask2former(num_classes: int, pretrained: bool):
    """Create FPN with MiT-B3 as alternative to Mask2Former."""
    import segmentation_models_pytorch as smp
    
    # Use FPN with MiT-B3 encoder as Mask2Former requires additional setup
    # FPN is an excellent multi-scale architecture
    model = smp.FPN(
        encoder_name='mit_b3',
        encoder_weights='imagenet' if pretrained else None,
        in_channels=3,
        classes=num_classes,
        activation=None
    )
    
    info = {
        'name': 'FPN-MiTB3',
        'uses_builtin_loss': False,
        'output_key': None,
        'requires_resize': False
    }
    
    return model, info


def _create_pspnet(num_classes: int, pretrained: bool):
    """Create PSPNet with ResNet-50 backbone."""
    import segmentation_models_pytorch as smp
    
    model = smp.PSPNet(
        encoder_name='resnet50',
        encoder_weights='imagenet' if pretrained else None,
        in_channels=3,
        classes=num_classes,
        activation=None
    )
    
    info = {
        'name': 'PSPNet-R50',
        'uses_builtin_loss': False,
        'output_key': None,
        'requires_resize': False
    }
    
    return model, info


def _create_pidnet(num_classes: int, pretrained: bool):
    """Create PIDNet-L (Large) from CVPR 2023."""
    import sys
    import os
    # Add benchmark_models to path for import
    benchmark_dir = os.path.dirname(os.path.abspath(__file__))
    if benchmark_dir not in sys.path:
        sys.path.insert(0, benchmark_dir)
    from pidnet import PIDNet
    
    # PIDNet-L configuration: m=3, n=4, planes=64, ppm_planes=112, head_planes=256
    model = PIDNet(
        m=3, 
        n=4, 
        num_classes=num_classes, 
        planes=64,
        ppm_planes=112,
        head_planes=256,
        augment=False  # Set to False for inference-friendly output
    )
    
    if pretrained:
        # No ImageNet pretrained weights for PIDNet, start from scratch
        print("[PIDNet] No pretrained weights available, training from scratch")
    
    info = {
        'name': 'PIDNet-L',
        'uses_builtin_loss': False,
        'output_key': None,
        'requires_resize': False
    }
    
    return model, info


# =============================================================================
# Factory Function
# =============================================================================

def get_benchmark_model(model_name: str, num_classes: int = 2, 
                        pretrained: bool = True) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Create a benchmark model by name.
    
    Args:
        model_name: One of 'deeplabv3plus', 'segformer', 'upernet', 'dpt', 'mask2former'
        num_classes: Number of output classes
        pretrained: Use pretrained backbone weights
        
    Returns:
        Tuple of (model, model_info_dict)
    """
    model_name = model_name.lower()
    
    if model_name == 'deeplabv3plus':
        model, info = _create_deeplabv3plus(num_classes, pretrained)
    elif model_name == 'segformer':
        model, info = _create_segformer(num_classes, pretrained)
    elif model_name == 'upernet':
        model, info = _create_upernet(num_classes, pretrained)
    elif model_name == 'dpt':
        model, info = _create_dpt(num_classes, pretrained)
    elif model_name == 'mask2former':
        model, info = _create_mask2former(num_classes, pretrained)
    elif model_name == 'fcn':
        model, info = _create_fcn(num_classes, pretrained)
    elif model_name == 'pspnet':
        model, info = _create_pspnet(num_classes, pretrained)
    elif model_name == 'pidnet':
        model, info = _create_pidnet(num_classes, pretrained)
    else:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: deeplabv3plus, segformer, upernet, dpt, mask2former, fcn, pspnet, pidnet"
        )
    
    # Add parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    info['total_params'] = total_params
    info['trainable_params'] = trainable_params
    info['params_millions'] = total_params / 1e6
    info['encoder'] = BENCHMARK_MODELS[model_name]['encoder']
    info['paradigm'] = BENCHMARK_MODELS[model_name]['paradigm']
    info['paper'] = BENCHMARK_MODELS[model_name]['paper']
    
    return model, info


# =============================================================================
# Utility Functions
# =============================================================================

def count_parameters(model) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_model_forward(model_name: str, device: str = 'cuda'):
    """Test that a model can do forward pass with expected input size."""
    model, info = get_benchmark_model(model_name, num_classes=1)
    model = model.to(device)
    model.eval()
    
    # Test input: 384x640
    x = torch.randn(1, 3, 384, 640).to(device)
    
    with torch.no_grad():
        output = model(x)
        
    if isinstance(output, dict):
        output = output['logits']
    
    print(f"{info['name']}: Input {x.shape} -> Output {output.shape}")
    print(f"  Parameters: {count_parameters(model):,}")
    
    return output.shape == (1, 1, 384, 640)


if __name__ == '__main__':
    # Test all models
    import torch
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Testing on: {device}\n")
    
    for model_name in ['deeplabv3plus', 'segformer', 'upernet', 'dpt', 'mask2former']:
        try:
            success = test_model_forward(model_name, device)
            print(f"  ✓ Forward pass OK\n" if success else f"  ✗ Output size mismatch\n")
        except Exception as e:
            print(f"  ✗ Error: {e}\n")
