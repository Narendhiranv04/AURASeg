"""
V3 APUD Model: CSPDarknet-53 + ASPP-Lite + APUD Decoder
Paper: AURASeg - Attention Guided Upsampling with Residual Boundary-Assistive Refinement

This model replaces the standard decoder with APUD (Attention Progressive
Upsampling Decoder) blocks that use SE and Spatial attention for better
feature fusion.

Architecture:
    - Encoder: CSPDarknet-53 (Focus + 5 Conv stages)
    - Context Module: ASPP-Lite (4 branches: 1×1 d=1, 3×3 d=1, 3×3 d=6, 3×3 d=12)
    - Decoder: 4 APUD blocks with attention-guided skip connections
    - Deep Supervision: Auxiliary losses at each APUD output
    - Loss: Focal + Dice + Deep Supervision

APUD Block Architecture (Fig. 3):
    x_low (deeper, smaller) → 1×1 → SE Attention → Upsample
    x_high (shallower, larger) → 1×1
    Fusion: Upsample(SE(x_low)) ⊗ x_high
    Spatial: SpatialAttention(x_high)
    Output: Refinement(Fusion + Spatial)

Comparison with V2:
    V2 uses standard decoder blocks (concat + conv)
    V3 uses APUD blocks (SE + Spatial attention fusion)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import CSPDarknet53
from .aspp_lite import ASPPLite
from .apud import APUDDecoder


class V3APUD(nn.Module):
    """
    V3 Ablation Model: CSPDarknet-53 + ASPP-Lite + APUD Decoder
    
    Replaces standard decoder with APUD blocks for attention-guided
    feature fusion and adds deep supervision.
    
    APUD configuration:
        - SE reduction ratio: 16
        - Spatial attention kernel: 7×7
        - Decoder channels: 256 (constant through all blocks)
        - Deep supervision weights: [0.1, 0.2, 0.3, 0.4] (coarse to fine)
        
    Architecture:
        Input (3, H, W)
            ↓
        CSPDarknet-53 Encoder
            ├─ c2 (128, H/4, W/4)
            ├─ c3 (256, H/8, W/8)
            ├─ c4 (512, H/16, W/16)
            └─ c5 (1024, H/32, W/32)
                ↓
        ASPP-Lite (256, H/32, W/32)
                ↓
        APUD-1: (ASPP, c5) → 256 @ H/32  [Supervision 1]
                ↓
        APUD-2: (APUD-1, c4) → 256 @ H/16  [Supervision 2]
                ↓
        APUD-3: (APUD-2, c3) → 256 @ H/8   [Supervision 3]
                ↓
        APUD-4: (APUD-3, c2) → 256 @ H/4   [Supervision 4]
                ↓
        Seg Head → (num_classes, H, W)
    """
    
    def __init__(self, in_channels: int = 3, num_classes: int = 2,
                 decoder_channels: int = 256, se_reduction: int = 16):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Encoder: CSPDarknet-53
        self.encoder = CSPDarknet53(in_channels=in_channels)
        
        # Context Module: ASPP-Lite with 4 branches
        # Input: c5 (1024 channels), Output: 256 channels
        self.aspp_lite = ASPPLite(
            in_channels=1024, 
            out_channels=256, 
            branch_channels=128
        )
        
        # APUD Decoder with deep supervision
        # encoder_channels = [c1, c2, c3, c4] = [64, 128, 256, 512]
        self.decoder = APUDDecoder(
            encoder_channels=[64, 128, 256, 512],
            neck_channels=256,
            decoder_channels=decoder_channels,
            num_classes=num_classes,
            se_reduction=se_reduction
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor, return_aux: bool = True) -> dict:
        """
        Args:
            x: Input image (B, 3, H, W)
            return_aux: Whether to return auxiliary supervision outputs
            
        Returns:
            Dictionary with:
                - 'main': Main segmentation output (B, num_classes, H, W)
                - 'aux': List of auxiliary outputs at different scales (if return_aux=True)
        """
        # Encoder: Extract multi-scale features
        c1, c2, c3, c4, c5 = self.encoder(x)
        # c1: (B, 64, H/4, W/4)
        # c2: (B, 128, H/8, W/8)
        # c3: (B, 256, H/16, W/16)
        # c4: (B, 512, H/32, W/32)
        # c5: (B, 1024, H/32, W/32) - used by ASPP-Lite
        
        # Context Module: ASPP-Lite on deepest features
        context = self.aspp_lite(c5)  # (B, 256, H/32, W/32)
        
        # APUD Decoder with optional auxiliary outputs
        # Pass [c1, c2, c3, c4] as skip connections
        encoder_features = [c1, c2, c3, c4]
        outputs = self.decoder(context, encoder_features, return_aux=return_aux)
        
        return outputs
    
    def get_encoder_params(self):
        """Get encoder parameters for separate learning rate"""
        return self.encoder.parameters()
    
    def get_decoder_params(self):
        """Get decoder + ASPP-Lite parameters for separate learning rate"""
        return list(self.aspp_lite.parameters()) + list(self.decoder.parameters())


def v3_apud(num_classes: int = 2, pretrained: bool = False):
    """
    Factory function to create V3 APUD model
    
    Args:
        num_classes: Number of segmentation classes (default: 2 for drivable area)
        pretrained: Not used, for API compatibility
        
    Returns:
        V3APUD model
    """
    return V3APUD(in_channels=3, num_classes=num_classes)


if __name__ == "__main__":
    # Test V3 model
    print("=" * 60)
    print("V3 APUD Model: CSPDarknet-53 + ASPP-Lite + APUD Decoder")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = V3APUD(in_channels=3, num_classes=2).to(device)
    
    # Test with standard input size
    x = torch.randn(2, 3, 384, 640).to(device)
    
    print(f"\nInput shape: {x.shape}")
    
    # Forward pass with auxiliary outputs
    with torch.no_grad():
        outputs = model(x, return_aux=True)
    
    print(f"\nMain output shape: {outputs['main'].shape}")
    print(f"Auxiliary outputs:")
    for i, aux in enumerate(outputs['aux']):
        print(f"  Aux-{i+1}: {aux.shape}")
    
    # Forward pass without auxiliary outputs (inference mode)
    with torch.no_grad():
        outputs_no_aux = model(x, return_aux=False)
    
    print(f"\nInference mode (return_aux=False):")
    print(f"  Main output: {outputs_no_aux['main'].shape}")
    print(f"  Aux outputs: {'aux' in outputs_no_aux and outputs_no_aux.get('aux') is not None}")
    
    # Parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    aspp_params = sum(p.numel() for p in model.aspp_lite.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    
    print(f"\nParameter Counts:")
    print(f"  Encoder (CSPDarknet-53): {encoder_params:,}")
    print(f"  ASPP-Lite:               {aspp_params:,}")
    print(f"  APUD Decoder:            {decoder_params:,}")
    print(f"  Total:                   {total_params:,}")
    
    # Memory estimation
    print(f"\nModel size: ~{total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    
    # Comparison note
    print("\n" + "=" * 60)
    print("V3 vs V2 Comparison:")
    print("  V2 uses standard decoder (concat + conv)")
    print("  V3 uses APUD decoder (SE + Spatial attention)")
    print("  V3 adds deep supervision at 4 scales")
    print("=" * 60)
