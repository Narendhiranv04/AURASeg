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
    'auraseg_r18': {
        'name': 'AURASeg-R18',
        'paper': 'Ours (WACV 2027)',
        'paradigm': 'ASPP-Lite + APUD + RBRM',
        'encoder': 'ResNet-18',
        'library': 'custom',
    },
    'auraseg_r18_apud128': {
        'name': 'AURASeg-R18 (APUD-128)',
        'paper': 'Ours (WACV 2027)',
        'paradigm': 'ASPP-Lite + APUD-128 + Full RBRM',
        'encoder': 'ResNet-18',
        'library': 'custom',
    },
    'auraseg_r18_apud64': {
        'name': 'AURASeg-R18 (APUD-64)',
        'paper': 'Ours (WACV 2027)',
        'paradigm': 'ASPP-Lite + APUD-64 + Full RBRM',
        'encoder': 'ResNet-18',
        'library': 'custom',
    },
    'auraseg_r18_fast': {
        'name': 'AURASeg-R18-Fast',
        'paper': 'Ours (WACV 2027 Real-Time Variant)',
        'paradigm': 'ASPP-Lite + APUD-128 + FastRBRM',
        'encoder': 'ResNet-18',
        'library': 'custom',
    },
    'auraseg_r50': {
        'name': 'AURASeg-R50',
        'paper': 'Ours (WACV 2027)',
        'paradigm': 'ASPP-Lite + APUD + RBRM',
        'encoder': 'ResNet-50',
        'library': 'custom',
    },
    'auraseg_r50_fast': {
        'name': 'AURASeg-R50-Fast',
        'paper': 'Ours (WACV 2027 Real-Time Variant)',
        'paradigm': 'ASPP-Lite + APUD-128 + FastRBRM',
        'encoder': 'ResNet-50',
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


def _create_auraseg_r18(num_classes: int, pretrained: bool):
    """Create AURASeg with ResNet-18 backbone (Base APUD-128 + Full RBRM)."""
    import sys
    import os
    benchmark_dir = os.path.dirname(os.path.abspath(__file__))
    if benchmark_dir not in sys.path:
        sys.path.insert(0, benchmark_dir)
    from auraseg_exportable import AURASeg_V4_ResNet
    
    model = AURASeg_V4_ResNet(
        backbone='resnet18',
        num_classes=num_classes,
        decoder_channels=128,
        fast_rbrm=False,
        encoder_weights='imagenet' if pretrained else None
    )
    
    info = {
        'name': 'AURASeg-R18',
        'uses_builtin_loss': False,
        'output_key': None,
        'requires_resize': False
    }
    return model, info


def _create_auraseg_r18_apud128(num_classes: int, pretrained: bool):
    """Create AURASeg with ResNet-18 and APUD-128 with full RBRM."""
    from auraseg_exportable import AURASeg_V4_ResNet
    model = AURASeg_V4_ResNet(
        backbone='resnet18',
        num_classes=num_classes,
        decoder_channels=128,
        fast_rbrm=False,
        encoder_weights='imagenet' if pretrained else None
    )
    info = {
        'name': 'AURASeg-R18-APUD128',
        'uses_builtin_loss': False,
        'output_key': None,
        'requires_resize': False
    }
    return model, info


def _create_auraseg_r18_apud64(num_classes: int, pretrained: bool):
    """Create AURASeg with ResNet-18 and APUD-64 with full RBRM."""
    from auraseg_exportable import AURASeg_V4_ResNet
    model = AURASeg_V4_ResNet(
        backbone='resnet18',
        num_classes=num_classes,
        decoder_channels=64,
        fast_rbrm=False,
        encoder_weights='imagenet' if pretrained else None
    )
    info = {
        'name': 'AURASeg-R18-APUD64',
        'uses_builtin_loss': False,
        'output_key': None,
        'requires_resize': False
    }
    return model, info


def _create_auraseg_r18_fast(num_classes: int, pretrained: bool):
    """Create AURASeg-Fast with ResNet-18 backbone (Real-Time Deployment Variant)."""
    import sys
    import os
    benchmark_dir = os.path.dirname(os.path.abspath(__file__))
    if benchmark_dir not in sys.path:
        sys.path.insert(0, benchmark_dir)
    from auraseg_exportable import auraseg_resnet18_fast
    
    model = auraseg_resnet18_fast(
        num_classes=num_classes,
        pretrained=pretrained
    )
    
    info = {
        'name': 'AURASeg-R18-Fast',
        'uses_builtin_loss': False,
        'output_key': None,
        'requires_resize': False
    }
    return model, info


def _create_auraseg_r50(num_classes: int, pretrained: bool):
    """Create AURASeg with ResNet-50 backbone (Base APUD-128 + Full RBRM)."""
    import sys
    import os
    benchmark_dir = os.path.dirname(os.path.abspath(__file__))
    if benchmark_dir not in sys.path:
        sys.path.insert(0, benchmark_dir)
    from auraseg_exportable import AURASeg_V4_ResNet
    
    model = AURASeg_V4_ResNet(
        backbone='resnet50',
        num_classes=num_classes,
        decoder_channels=128,
        fast_rbrm=False,
        encoder_weights='imagenet' if pretrained else None
    )
    
    info = {
        'name': 'AURASeg-R50',
        'uses_builtin_loss': False,
        'output_key': None,
        'requires_resize': False
    }
    return model, info


def _create_auraseg_r50_fast(num_classes: int, pretrained: bool):
    """Create AURASeg-Fast with ResNet-50 backbone (Real-Time Deployment Variant)."""
    import sys
    import os
    benchmark_dir = os.path.dirname(os.path.abspath(__file__))
    if benchmark_dir not in sys.path:
        sys.path.insert(0, benchmark_dir)
    from auraseg_exportable import auraseg_resnet50_fast
    
    model = auraseg_resnet50_fast(
        num_classes=num_classes,
        pretrained=pretrained
    )
    
    info = {
        'name': 'AURASeg-R50-Fast',
        'uses_builtin_loss': False,
        'output_key': None,
        'requires_resize': False
    }
    return model, info


# =============================================================================
# Factory Function
# =============================================================================

# Map common model aliases to canonical keys
MODEL_ALIASES = {
    'fcn-r50': 'fcn',
    'fcn_r50': 'fcn',
    'fcn50': 'fcn',
    'pspnet-r50': 'pspnet',
    'pspnet_r50': 'pspnet',
    'pspnet50': 'pspnet',
    'upernet-r50': 'upernet',
    'upernet_r50': 'upernet',
    'upernet50': 'upernet',
    'segformer-b2': 'segformer',
    'segformer_b2': 'segformer',
    'segformerb2': 'segformer',
    'pidnet-l': 'pidnet',
    'pidnet_l': 'pidnet',
    'pidnetl': 'pidnet',
    'auraseg': 'auraseg_r18',
    'auraseg-r18': 'auraseg_r18',
    'auraseg_r18': 'auraseg_r18',
    'auraseg-r18-apud128': 'auraseg_r18_apud128',
    'auraseg_r18_apud128': 'auraseg_r18_apud128',
    'auraseg-r18-128': 'auraseg_r18_apud128',
    'auraseg-apud128': 'auraseg_r18_apud128',
    'auraseg-r18-apud64': 'auraseg_r18_apud64',
    'auraseg_r18_apud64': 'auraseg_r18_apud64',
    'auraseg-r18-64': 'auraseg_r18_apud64',
    'auraseg-apud64': 'auraseg_r18_apud64',
    'auraseg-r18-fast': 'auraseg_r18_fast',
    'auraseg_r18_fast': 'auraseg_r18_fast',
    'auraseg-r18-opt': 'auraseg_r18_fast',
    'auraseg-fast': 'auraseg_r18_fast',
    'auraseg_fast': 'auraseg_r18_fast',
    'auraseg-opt': 'auraseg_r18_fast',
    'auraseg-r50': 'auraseg_r50',
    'auraseg_r50': 'auraseg_r50',
    'auraseg-r50-fast': 'auraseg_r50_fast',
    'auraseg_r50_fast': 'auraseg_r50_fast',
    'auraseg-r50-opt': 'auraseg_r50_fast',
}


def get_benchmark_model(model_name: str, num_classes: int = 2, 
                        pretrained: bool = True) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Create a benchmark model by name.
    
    Args:
        model_name: One of 'deeplabv3plus', 'segformer', 'upernet', 'dpt', 'mask2former',
                    'fcn', 'pspnet', 'pidnet', 'auraseg_r18', 'auraseg_r50'
        num_classes: Number of output classes
        pretrained: Use pretrained backbone weights
        
    Returns:
        Tuple of (model, model_info_dict)
    """
    model_name = model_name.lower().strip()
    canonical_name = MODEL_ALIASES.get(model_name, model_name)
    
    if canonical_name == 'deeplabv3plus':
        model, info = _create_deeplabv3plus(num_classes, pretrained)
    elif canonical_name == 'segformer':
        model, info = _create_segformer(num_classes, pretrained)
    elif canonical_name == 'upernet':
        model, info = _create_upernet(num_classes, pretrained)
    elif canonical_name == 'dpt':
        model, info = _create_dpt(num_classes, pretrained)
    elif canonical_name == 'mask2former':
        model, info = _create_mask2former(num_classes, pretrained)
    elif canonical_name == 'fcn':
        model, info = _create_fcn(num_classes, pretrained)
    elif canonical_name == 'pspnet':
        model, info = _create_pspnet(num_classes, pretrained)
    elif canonical_name == 'pidnet':
        model, info = _create_pidnet(num_classes, pretrained)
    elif canonical_name == 'auraseg_r18':
        model, info = _create_auraseg_r18(num_classes, pretrained)
    elif canonical_name == 'auraseg_r18_apud128':
        model, info = _create_auraseg_r18_apud128(num_classes, pretrained)
    elif canonical_name == 'auraseg_r18_apud64':
        model, info = _create_auraseg_r18_apud64(num_classes, pretrained)
    elif canonical_name == 'auraseg_r18_fast':
        model, info = _create_auraseg_r18_fast(num_classes, pretrained)
    elif canonical_name == 'auraseg_r50':
        model, info = _create_auraseg_r50(num_classes, pretrained)
    elif canonical_name == 'auraseg_r50_fast':
        model, info = _create_auraseg_r50_fast(num_classes, pretrained)
    else:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: deeplabv3plus, segformer, upernet, dpt, mask2former, fcn, pspnet, pidnet, auraseg_r18, auraseg_r18_apud128, auraseg_r18_apud64, auraseg_r18_fast, auraseg_r50, auraseg_r50_fast"
        )
    
    # Add parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    info['total_params'] = total_params
    info['trainable_params'] = trainable_params
    info['params_millions'] = total_params / 1e6
    info['encoder'] = BENCHMARK_MODELS.get(canonical_name, {}).get('encoder', 'N/A')
    info['paradigm'] = BENCHMARK_MODELS.get(canonical_name, {}).get('paradigm', 'N/A')
    info['paper'] = BENCHMARK_MODELS.get(canonical_name, {}).get('paper', 'N/A')
    
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
