"""
V1 Base Model: CSPDarknet-53 + SPP + Decoder
Paper: AURASeg - Attention Guided Upsampling with Residual Boundary-Assistive Refinement

This is the BASELINE model for ablation study.
Architecture:
    - Encoder: CSPDarknet-53 (Focus + 5 Conv stages)
    - Context Module: SPP (Spatial Pyramid Pooling with MaxPool 5, 9, 13)
    - Decoder: 4 blocks with skip connections to c1, c2, c3, c4
    - Loss: Focal + Dice

This model establishes the baseline performance using SPP as the context module.
V2 will replace SPP with ASPP-Lite to measure the impact.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import CSPDarknet53, SPP
from .decoder import Decoder


class V1BaseSPP(nn.Module):
    """
    V1 Ablation Model: CSPDarknet-53 + SPP + Decoder
    
    This is the baseline model for the ablation study.
    Uses SPP (Spatial Pyramid Pooling) as the context aggregation module.
    
    Architecture:
        Input (3, H, W)
            ↓
        CSPDarknet-53 Encoder
            ├─ c1 (64, H/4, W/4)   ─────────────────────────────┐
            ├─ c2 (128, H/8, W/8)  ───────────────────────┐     │
            ├─ c3 (256, H/16, W/16) ─────────────────┐    │     │
            ├─ c4 (512, H/32, W/32) ────────────┐    │    │     │
            └─ c5 (1024, H/32, W/32)             │    │    │     │
                ↓                                │    │    │     │
        SPP (MaxPool 5,9,13)                     │    │    │     │
                ↓                                │    │    │     │
            (256, H/32, W/32)                    │    │    │     │
                ↓                                ↓    ↓    ↓     ↓
        Decoder ←─ c4 ←─────────────────────────┘    │    │     │
            ↓                                         │    │     │
        Decoder ←─ c3 ←──────────────────────────────┘    │     │
            ↓                                              │     │
        Decoder ←─ c2 ←───────────────────────────────────┘     │
            ↓                                                    │
        Decoder ←─ c1 ←─────────────────────────────────────────┘
            ↓
        Output (num_classes, H, W)
    """
    
    def __init__(self, in_channels=3, num_classes=2):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Encoder: CSPDarknet-53
        self.encoder = CSPDarknet53(in_channels=in_channels)
        
        # Context Module: SPP
        # Input: c5 (1024 channels), Output: 256 channels
        self.spp = SPP(in_channels=1024, out_channels=256, kernels=(5, 9, 13))
        
        # Decoder with skip connections
        # encoder_channels = [c1, c2, c3, c4] = [64, 128, 256, 512]
        self.decoder = Decoder(
            encoder_channels=self.encoder.out_channels[:4],  # [64, 128, 256, 512]
            aspp_channels=256,  # SPP output channels
            num_classes=num_classes
        )
    
    def forward(self, x):
        """
        Args:
            x: Input image (B, 3, H, W)
            
        Returns:
            Segmentation logits (B, num_classes, H, W)
        """
        input_shape = x.shape[2:]  # (H, W)
        
        # Encoder: Extract multi-scale features
        c1, c2, c3, c4, c5 = self.encoder(x)
        
        # Context Module: SPP on deepest features
        context = self.spp(c5)  # (B, 256, H/32, W/32)
        
        # Decoder: Progressive upsampling with skip connections
        out = self.decoder(context, c1, c2, c3, c4, input_shape)
        
        return out
    
    def get_encoder_params(self):
        """Get encoder parameters for separate learning rate"""
        return self.encoder.parameters()
    
    def get_decoder_params(self):
        """Get decoder + SPP parameters for separate learning rate"""
        return list(self.spp.parameters()) + list(self.decoder.parameters())


def v1_base_spp(num_classes=2, pretrained=False):
    """
    Factory function to create V1 Base SPP model
    
    Args:
        num_classes: Number of segmentation classes (default: 2 for drivable area)
        pretrained: Not used, for API compatibility
        
    Returns:
        V1BaseSPP model
    """
    return V1BaseSPP(in_channels=3, num_classes=num_classes)


if __name__ == "__main__":
    # Test V1 model
    print("=" * 60)
    print("V1 Base Model: CSPDarknet-53 + SPP + Decoder")
    print("=" * 60)
    
    model = V1BaseSPP(in_channels=3, num_classes=2)
    
    # Test with standard input size
    x = torch.randn(2, 3, 384, 640)
    
    print(f"\nInput shape: {x.shape}")
    
    # Forward pass
    with torch.no_grad():
        out = model(x)
    
    print(f"Output shape: {out.shape}")
    
    # Parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    spp_params = sum(p.numel() for p in model.spp.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    
    print(f"\nParameter Counts:")
    print(f"  Encoder (CSPDarknet-53): {encoder_params:,}")
    print(f"  SPP:                     {spp_params:,}")
    print(f"  Decoder:                 {decoder_params:,}")
    print(f"  Total:                   {total_params:,}")
    
    # Memory estimation
    print(f"\nModel size: ~{total_params * 4 / 1024 / 1024:.2f} MB (float32)")
