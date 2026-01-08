"""
V2 Base Model: CSPDarknet-53 + ASPP-Lite + Decoder
Paper: AURASeg - Attention Guided Upsampling with Residual Boundary-Assistive Refinement

This model replaces SPP with ASPP-Lite to evaluate the impact of 
dilated convolutions for multi-scale context aggregation.

Architecture:
    - Encoder: CSPDarknet-53 (Focus + 5 Conv stages)
    - Context Module: ASPP-Lite (4 branches: 1×1 d=1, 3×3 d=1, 3×3 d=6, 3×3 d=12)
    - Decoder: 4 blocks with skip connections to c1, c2, c3, c4
    - Loss: Focal + Dice

Comparison with V1:
    V1 uses SPP (MaxPool-based spatial pyramid)
    V2 uses ASPP-Lite (Dilated convolution-based atrous pyramid)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import CSPDarknet53
from .aspp_lite import ASPPLite
from .decoder import Decoder


class V2BaseASPPLite(nn.Module):
    """
    V2 Ablation Model: CSPDarknet-53 + ASPP-Lite + Decoder
    
    Replaces SPP with ASPP-Lite for better multi-scale context with
    dilated convolutions.
    
    ASPP-Lite configuration (from paper):
        Branch 1: Conv 1×1, dilation=1, 128 filters
        Branch 2: Conv 3×3, dilation=1, 128 filters
        Branch 3: Conv 3×3, dilation=6, 128 filters
        Branch 4: Conv 3×3, dilation=12, 128 filters
        
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
        ASPP-Lite (4 parallel branches)          │    │    │     │
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
        
        # Context Module: ASPP-Lite with 4 branches
        # Input: c5 (1024 channels), Output: 256 channels
        # Branch channels: 128 each (as per paper)
        self.aspp_lite = ASPPLite(
            in_channels=1024, 
            out_channels=256, 
            branch_channels=128
        )
        
        # Decoder with skip connections
        # encoder_channels = [c1, c2, c3, c4] = [64, 128, 256, 512]
        self.decoder = Decoder(
            encoder_channels=self.encoder.out_channels[:4],  # [64, 128, 256, 512]
            aspp_channels=256,  # ASPP-Lite output channels
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
        
        # Context Module: ASPP-Lite on deepest features
        context = self.aspp_lite(c5)  # (B, 256, H/32, W/32)
        
        # Decoder: Progressive upsampling with skip connections
        out = self.decoder(context, c1, c2, c3, c4, input_shape)
        
        return out
    
    def get_encoder_params(self):
        """Get encoder parameters for separate learning rate"""
        return self.encoder.parameters()
    
    def get_decoder_params(self):
        """Get decoder + ASPP-Lite parameters for separate learning rate"""
        return list(self.aspp_lite.parameters()) + list(self.decoder.parameters())


def v2_base_assplite(num_classes=2, pretrained=False):
    """
    Factory function to create V2 Base ASPP-Lite model
    
    Args:
        num_classes: Number of segmentation classes (default: 2 for drivable area)
        pretrained: Not used, for API compatibility
        
    Returns:
        V2BaseASPPLite model
    """
    return V2BaseASPPLite(in_channels=3, num_classes=num_classes)


if __name__ == "__main__":
    # Test V2 model
    print("=" * 60)
    print("V2 Base Model: CSPDarknet-53 + ASPP-Lite + Decoder")
    print("=" * 60)
    
    model = V2BaseASPPLite(in_channels=3, num_classes=2)
    
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
    aspp_params = sum(p.numel() for p in model.aspp_lite.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    
    print(f"\nParameter Counts:")
    print(f"  Encoder (CSPDarknet-53): {encoder_params:,}")
    print(f"  ASPP-Lite:               {aspp_params:,}")
    print(f"  Decoder:                 {decoder_params:,}")
    print(f"  Total:                   {total_params:,}")
    
    # Memory estimation
    print(f"\nModel size: ~{total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    
    # Comparison note
    print("\n" + "=" * 60)
    print("V2 vs V1 Comparison:")
    print("  V1 uses SPP (MaxPool-based)")
    print("  V2 uses ASPP-Lite (Dilated convolution-based)")
    print("  Both have same decoder architecture")
    print("=" * 60)
