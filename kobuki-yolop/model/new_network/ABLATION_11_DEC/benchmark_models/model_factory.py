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
        'paper': 'Xiao et al., ECCV 2018 + Swin',
        'paradigm': 'Hierarchical Transformer',
        'encoder': 'Swin-Tiny',
        'library': 'segmentation_models_pytorch',
    },
    'dpt': {
        'name': 'DPT',
        'paper': 'Ranftl et al., ICCV 2021',
        'paradigm': 'Vision Transformer',
        'encoder': 'ViT-Base',
        'library': 'segmentation_models_pytorch',
    },
    'mask2former': {
        'name': 'Mask2Former',
        'paper': 'Cheng et al., CVPR 2022',
        'paradigm': 'Universal Segmentation',
        'encoder': 'Swin-Small',
        'library': 'transformers',
    },
}


def get_deeplabv3plus(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """
    Create DeepLabV3+ with ResNet-50 backbone.
    
    Args:
        num_classes: Number of output classes
        pretrained: Use ImageNet pretrained weights
        
    Returns:
        DeepLabV3+ model
    """
    import segmentation_models_pytorch as smp
    
    model = smp.DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights="imagenet" if pretrained else None,
        in_channels=3,
        classes=num_classes,
        encoder_output_stride=16,
        decoder_channels=256,
        decoder_atrous_rates=(12, 24, 36),
        upsampling=4,
    )
    return model


def get_segformer(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """
    Create SegFormer with MiT-B2 backbone.
    
    Args:
        num_classes: Number of output classes
        pretrained: Use ImageNet pretrained weights
        
    Returns:
        SegFormer model
    """
    import segmentation_models_pytorch as smp
    
    model = smp.Segformer(
        encoder_name="mit_b2",
        encoder_weights="imagenet" if pretrained else None,
        in_channels=3,
        classes=num_classes,
        decoder_segmentation_channels=256,
        upsampling=4,
    )
    return model


def get_upernet(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """
    Create UPerNet with Swin-Tiny backbone.
    
    Args:
        num_classes: Number of output classes
        pretrained: Use ImageNet pretrained weights
        
    Returns:
        UPerNet model
    """
    import segmentation_models_pytorch as smp
    
    model = smp.UPerNet(
        encoder_name="tu-swin_tiny_patch4_window7_224",
        encoder_weights="imagenet" if pretrained else None,
        in_channels=3,
        classes=num_classes,
        decoder_channels=256,
        upsampling=4,
    )
    return model


def get_dpt(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """
    Create DPT (Dense Prediction Transformer) with ViT-Base backbone.
    
    Args:
        num_classes: Number of output classes
        pretrained: Use ImageNet pretrained weights
        
    Returns:
        DPT model
    """
    import segmentation_models_pytorch as smp
    
    model = smp.DPT(
        encoder_name="tu-vit_base_patch16_384",
        encoder_weights="imagenet" if pretrained else None,
        in_channels=3,
        classes=num_classes,
        encoder_depth=4,
        decoder_intermediate_channels=(256, 512, 1024, 1024),
        decoder_fusion_channels=256,
        dynamic_img_size=True,  # Required for non-square inputs
    )
    return model


class Mask2FormerWrapper(nn.Module):
    """
    Wrapper for Mask2Former from HuggingFace Transformers.
    
    Adapts the Mask2Former API to return standard segmentation output.
    Uses built-in loss when labels are provided.
    """
    
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        
        from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerConfig
        
        self.num_classes = num_classes
        
        if pretrained:
            # Load pretrained model and modify for our classes
            self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
                "facebook/mask2former-swin-small-ade-semantic",
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
            )
        else:
            config = Mask2FormerConfig(
                num_labels=num_classes,
                backbone_config={
                    "model_type": "swin",
                    "image_size": 384,
                    "patch_size": 4,
                    "num_channels": 3,
                    "embed_dim": 96,
                    "depths": [2, 2, 18, 2],
                    "num_heads": [3, 6, 12, 24],
                    "window_size": 7,
                },
            )
            self.model = Mask2FormerForUniversalSegmentation(config)
        
        # Image processor for preprocessing
        from transformers import Mask2FormerImageProcessor
        self.processor = Mask2FormerImageProcessor.from_pretrained(
            "facebook/mask2former-swin-small-ade-semantic",
            do_resize=False,  # We handle resizing ourselves
            do_normalize=False,  # We handle normalization ourselves
        )
    
    def forward(self, pixel_values: torch.Tensor, 
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            pixel_values: Input images (B, 3, H, W), already normalized
            labels: Optional ground truth masks (B, H, W) for training
            
        Returns:
            Dictionary with 'logits' and optionally 'loss'
        """
        B, C, H, W = pixel_values.shape
        
        # Mask2Former expects pixel_values in a specific format
        outputs = self.model(pixel_values=pixel_values)
        
        # Get semantic segmentation from mask predictions
        # masks_queries_logits: (B, num_queries, H/4, W/4)
        # class_queries_logits: (B, num_queries, num_classes+1)
        
        masks = outputs.masks_queries_logits  # (B, Q, H', W')
        classes = outputs.class_queries_logits  # (B, Q, num_classes+1)
        
        # Compute semantic segmentation logits
        # For each pixel, sum weighted mask contributions
        masks = torch.nn.functional.interpolate(
            masks, size=(H, W), mode='bilinear', align_corners=False
        )
        
        # Softmax over classes (excluding null class at last index)
        class_probs = torch.softmax(classes, dim=-1)[..., :-1]  # (B, Q, num_classes)
        
        # Weighted sum: (B, Q, H, W) x (B, Q, num_classes) -> (B, num_classes, H, W)
        # Reshape for einsum
        masks = masks.sigmoid()  # (B, Q, H, W)
        logits = torch.einsum('bqhw,bqc->bchw', masks, class_probs)
        
        result = {'logits': logits}
        
        # Compute loss if labels provided (for training)
        if labels is not None:
            # Use built-in loss computation
            mask_labels = []
            class_labels = []
            
            for b in range(B):
                unique_classes = labels[b].unique()
                unique_classes = unique_classes[unique_classes != 255]  # Ignore label
                
                masks_b = []
                classes_b = []
                for cls in unique_classes:
                    mask = (labels[b] == cls).float()
                    masks_b.append(mask)
                    classes_b.append(cls)
                
                if len(masks_b) > 0:
                    mask_labels.append(torch.stack(masks_b))
                    class_labels.append(torch.tensor(classes_b, device=labels.device))
                else:
                    # Empty mask case
                    mask_labels.append(torch.zeros(1, H, W, device=labels.device))
                    class_labels.append(torch.tensor([0], device=labels.device))
            
            # Re-run with labels for loss
            outputs_with_loss = self.model(
                pixel_values=pixel_values,
                mask_labels=mask_labels,
                class_labels=class_labels,
            )
            result['loss'] = outputs_with_loss.loss
        
        return result
    
    def get_logits(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Get segmentation logits only (for inference)."""
        return self.forward(pixel_values)['logits']


def get_mask2former(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """
    Create Mask2Former with Swin-Small backbone.
    
    Args:
        num_classes: Number of output classes
        pretrained: Use pretrained weights
        
    Returns:
        Mask2Former wrapper model
    """
    return Mask2FormerWrapper(num_classes=num_classes, pretrained=pretrained)


# Factory function mapping
_MODEL_FACTORY = {
    'deeplabv3plus': get_deeplabv3plus,
    'segformer': get_segformer,
    'upernet': get_upernet,
    'dpt': get_dpt,
    'mask2former': get_mask2former,
}


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
    
    if model_name not in _MODEL_FACTORY:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {list(_MODEL_FACTORY.keys())}"
        )
    
    model = _MODEL_FACTORY[model_name](num_classes=num_classes, pretrained=pretrained)
    info = BENCHMARK_MODELS[model_name].copy()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    info['total_params'] = total_params
    info['trainable_params'] = trainable_params
    info['params_millions'] = total_params / 1e6
    
    return model, info


def get_model_info(model_name: str) -> Dict[str, Any]:
    """Get model metadata without creating the model."""
    model_name = model_name.lower()
    if model_name not in BENCHMARK_MODELS:
        raise ValueError(f"Unknown model: {model_name}")
    return BENCHMARK_MODELS[model_name].copy()


if __name__ == "__main__":
    """Test all benchmark models."""
    import torch
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing benchmark models on {device}")
    print("=" * 70)
    
    # Test input: (B, C, H, W) = (2, 3, 384, 640)
    test_input = torch.randn(2, 3, 384, 640).to(device)
    
    for model_name in BENCHMARK_MODELS.keys():
        print(f"\nTesting {model_name}...")
        try:
            model, info = get_benchmark_model(model_name, num_classes=2, pretrained=True)
            model = model.to(device)
            model.eval()
            
            with torch.no_grad():
                if model_name == 'mask2former':
                    output = model.get_logits(test_input)
                else:
                    output = model(test_input)
            
            print(f"  ✓ {info['name']} ({info['encoder']})")
            print(f"    Params: {info['params_millions']:.1f}M")
            print(f"    Input:  {tuple(test_input.shape)}")
            print(f"    Output: {tuple(output.shape)}")
            
            # Verify output shape
            assert output.shape == (2, 2, 384, 640), f"Unexpected output shape: {output.shape}"
            print(f"    ✓ Shape verified: (B, 2, 384, 640)")
            
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
    
    print("\n" + "=" * 70)
    print("Benchmark model tests complete!")
