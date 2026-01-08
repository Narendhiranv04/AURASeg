"""
APUD Module: Attention Progressive Upsampling Decoder
Paper: AURASeg - Attention Guided Upsampling with Residual Boundary-Assistive Refinement

This module implements the APUD block which replaces standard decoder blocks
with attention-guided feature fusion.

APUD Block Architecture (from Fig. 3):
    x_low (deeper features, smaller spatial) → 1×1 Transform → SE Attention → Upsample
    x_high (shallower features, larger spatial) → 1×1 Transform
    
    Fusion: Upsample(SE(x_low_transformed)) ⊗ x_high_transformed
    Spatial: SpatialAttention(x_high_transformed)
    Output: Refinement(Fusion + Spatial)

Components:
    1. SE (Squeeze-and-Excitation) Attention - Channel attention
    2. Spatial Attention - Spatial focus using max+avg pooling
    3. APUD Block - Full attention-guided upsampling block
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEAttention(nn.Module):
    """
    Squeeze-and-Excitation Attention Module
    
    Applies channel-wise attention by:
    1. Global Average Pooling (squeeze)
    2. FC → ReLU → FC → Sigmoid (excitation)
    3. Channel-wise multiplication (scale)
    
    Args:
        channels: Number of input channels
        reduction: Reduction ratio for FC layers (default: 16)
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Excitation: FC → ReLU → FC → Sigmoid
        reduced_channels = max(channels // reduction, 8)  # Ensure at least 8 channels
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Channel-reweighted tensor (B, C, H, W)
        """
        b, c, _, _ = x.size()
        
        # Squeeze: Global Average Pooling
        y = self.avg_pool(x).view(b, c)  # (B, C)
        
        # Excitation: FC layers
        y = self.fc(y).view(b, c, 1, 1)  # (B, C, 1, 1)
        
        # Scale: Channel-wise multiplication
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module
    
    Applies spatial attention by:
    1. Max-pool and Avg-pool along channel dimension
    2. Concatenate pooled features
    3. 7×7 Conv → Sigmoid
    4. Spatial-wise multiplication
    
    Args:
        kernel_size: Kernel size for spatial conv (default: 7)
    """
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        padding = kernel_size // 2
        
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Spatially-reweighted tensor (B, C, H, W)
        """
        # Pool along channel dimension
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, H, W)
        
        # Concatenate
        y = torch.cat([avg_out, max_out], dim=1)  # (B, 2, H, W)
        
        # Spatial attention map
        y = self.conv(y)  # (B, 1, H, W)
        
        # Spatial-wise multiplication
        return x * y


class ConvBNAct(nn.Module):
    """Standard Convolution + BatchNorm + Activation"""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, stride: int = 1, padding: int = None):
        super().__init__()
        
        if padding is None:
            padding = kernel_size // 2
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                              stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class APUDBlock(nn.Module):
    """
    Attention Progressive Upsampling Decoder Block
    
    Architecture (from Fig. 3):
        x_low (deeper, smaller spatial):
            → 1×1 Transform → SE Attention → Upsample (bilinear)
        
        x_high (shallower, larger spatial):
            → 1×1 Transform
            
        Fusion:
            upsample(SE(x_low_transformed)) ⊗ x_high_transformed
            
        Spatial:
            SpatialAttention(x_high_transformed)
            
        Output:
            Refinement(Fusion + Spatial)
            
    Args:
        low_channels: Channels from deeper (low-res) features
        high_channels: Channels from shallower (high-res) features
        out_channels: Output channels
        se_reduction: SE attention reduction ratio
        spatial_kernel: Spatial attention kernel size
    """
    
    def __init__(self, low_channels: int, high_channels: int, out_channels: int,
                 se_reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        
        # 1×1 Transform for x_low (deeper features)
        self.low_transform = ConvBNAct(low_channels, out_channels, kernel_size=1)
        
        # 1×1 Transform for x_high (shallower features)
        self.high_transform = ConvBNAct(high_channels, out_channels, kernel_size=1)
        
        # SE Attention for low-res features (channel attention)
        self.se_attention = SEAttention(out_channels, reduction=se_reduction)
        
        # Spatial Attention for high-res features
        self.spatial_attention = SpatialAttention(kernel_size=spatial_kernel)
        
        # Refinement Module: 3×3 Conv + BN + ReLU
        self.refinement = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)  # Paper mentions ReLU, but SiLU is modern variant
        )
    
    def forward(self, x_low: torch.Tensor, x_high: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_low: Deeper features, smaller spatial resolution (B, low_ch, H_low, W_low)
            x_high: Shallower features, larger spatial resolution (B, high_ch, H_high, W_high)
            
        Returns:
            Fused and upsampled features (B, out_ch, H_high, W_high)
        """
        # Transform x_low: 1×1 conv
        low_transformed = self.low_transform(x_low)  # (B, out_ch, H_low, W_low)
        
        # Transform x_high: 1×1 conv
        high_transformed = self.high_transform(x_high)  # (B, out_ch, H_high, W_high)
        
        # SE Attention on low features
        low_se = self.se_attention(low_transformed)  # (B, out_ch, H_low, W_low)
        
        # Upsample low features to match high-res spatial size
        low_upsampled = F.interpolate(
            low_se, 
            size=high_transformed.shape[2:], 
            mode='bilinear', 
            align_corners=True
        )  # (B, out_ch, H_high, W_high)
        
        # Multiplicative Fusion: upsample(SE(x_low)) ⊗ x_high
        fusion = low_upsampled * high_transformed  # (B, out_ch, H_high, W_high)
        
        # Spatial Attention on high features
        high_spatial = self.spatial_attention(high_transformed)  # (B, out_ch, H_high, W_high)
        
        # Combine: Fusion + Spatial
        combined = fusion + high_spatial  # (B, out_ch, H_high, W_high)
        
        # Refinement
        output = self.refinement(combined)  # (B, out_ch, H_high, W_high)
        
        return output


class APUDDecoder(nn.Module):
    """
    Full APUD Decoder with 4 blocks and deep supervision
    
    Based on Fig. 2 of AURASeg paper:
        ASPP-Lite output (256, H/32, W/32)
            ↓
        APUD-1: (ASPP, c4=512@H/32) → 256 @ H/16  [Supervision 1]
            ↓
        APUD-2: (APUD-1, c3=256@H/16) → 256 @ H/8  [Supervision 2]
            ↓
        APUD-3: (APUD-2, c2=128@H/8) → 256 @ H/4   [Supervision 3]
            ↓
        APUD-4: (APUD-3, c1=64@H/4) → 256 @ H/2    [Supervision 4]
            ↓
        Seg Head → num_classes @ H/1
        
    Note: x_low is the deeper/smaller features that get upsampled
          x_high is the shallower/larger features (skip connections)
        
    Args:
        encoder_channels: List of encoder output channels [c1, c2, c3, c4]
        neck_channels: Channels from ASPP-Lite output
        decoder_channels: Output channels for each APUD block
        num_classes: Number of segmentation classes
        se_reduction: SE attention reduction ratio
    """
    
    def __init__(self, 
                 encoder_channels: list = [64, 128, 256, 512],
                 neck_channels: int = 256,
                 decoder_channels: int = 256,
                 num_classes: int = 2,
                 se_reduction: int = 16):
        super().__init__()
        
        c1, c2, c3, c4 = encoder_channels
        
        # APUD blocks (4 levels)
        # APUD-1: ASPP output (256@H/32) + c4 (512@H/32) → 256@H/16
        # x_low=ASPP, x_high=c4 (but c4 is at H/32, we need to upsample output)
        # Actually, looking at Fig 3: x_low gets SE attention and upsampled to x_high size
        # So if we want output at H/16, we need x_high to be at H/16
        # But c4 is at H/32... Let me re-read the architecture
        
        # Re-interpreting based on actual encoder outputs:
        # c1: H/4, c2: H/8, c3: H/16, c4: H/32, c5: H/32
        # 
        # APUD-1: (ASPP@H/32, c3@H/16) → out@H/16  (ASPP upsampled to c3 size)
        # APUD-2: (APUD-1@H/16, c2@H/8) → out@H/8  (APUD-1 upsampled to c2 size)
        # APUD-3: (APUD-2@H/8, c1@H/4) → out@H/4   (APUD-2 upsampled to c1 size)
        # APUD-4: Not needed, or we add final upsampling
        
        # Actually, let's match the paper exactly:
        # The paper shows 4 APUD blocks connecting to conv2, conv3, conv4 (and conv1 goes to final)
        # So: APUD-1 connects to conv4, APUD-2 to conv3, APUD-3 to conv2, APUD-4 to conv1
        
        # For our backbone (c1=H/4, c2=H/8, c3=H/16, c4=H/32):
        # APUD-1: (ASPP@H/32, c4@H/32) → out@H/32 (no upsampling, just fusion)
        # APUD-2: (APUD-1@H/32, c3@H/16) → out@H/16 (upsample APUD-1 to match c3)
        # APUD-3: (APUD-2@H/16, c2@H/8) → out@H/8 (upsample APUD-2 to match c2)
        # APUD-4: (APUD-3@H/8, c1@H/4) → out@H/4 (upsample APUD-3 to match c1)
        
        # APUD-1: ASPP (256@H/32) + c4 (512@H/32) → 256@H/32
        self.apud1 = APUDBlock(neck_channels, c4, decoder_channels, se_reduction)
        
        # APUD-2: APUD-1 (256@H/32) + c3 (256@H/16) → 256@H/16
        self.apud2 = APUDBlock(decoder_channels, c3, decoder_channels, se_reduction)
        
        # APUD-3: APUD-2 (256@H/16) + c2 (128@H/8) → 256@H/8
        self.apud3 = APUDBlock(decoder_channels, c2, decoder_channels, se_reduction)
        
        # APUD-4: APUD-3 (256@H/8) + c1 (64@H/4) → 256@H/4
        self.apud4 = APUDBlock(decoder_channels, c1, decoder_channels, se_reduction)
        
        # Supervision heads (one for each APUD block output)
        # Each outputs at its native resolution
        self.aux_head1 = self._make_aux_head(decoder_channels, num_classes)  # @ H/32
        self.aux_head2 = self._make_aux_head(decoder_channels, num_classes)  # @ H/16
        self.aux_head3 = self._make_aux_head(decoder_channels, num_classes)  # @ H/8
        self.aux_head4 = self._make_aux_head(decoder_channels, num_classes)  # @ H/4
        
        # Main segmentation head (from APUD-4 output)
        self.seg_head = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels // 2),
            nn.SiLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(decoder_channels // 2, num_classes, 1)
        )
    
    def _make_aux_head(self, in_channels: int, num_classes: int) -> nn.Sequential:
        """Create auxiliary supervision head"""
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels // 4, num_classes, 1)
        )
    
    def forward(self, neck_out: torch.Tensor, 
                encoder_features: list,
                return_aux: bool = True) -> dict:
        """
        Args:
            neck_out: ASPP-Lite output (B, 256, H/32, W/32)
            encoder_features: [c1, c2, c3, c4] from encoder
                c1: (B, 64, H/4, W/4)
                c2: (B, 128, H/8, W/8)
                c3: (B, 256, H/16, W/16)
                c4: (B, 512, H/32, W/32)
            return_aux: Whether to return auxiliary supervision outputs
            
        Returns:
            Dictionary with 'main' and optionally 'aux' outputs
        """
        c1, c2, c3, c4 = encoder_features
        
        # APUD-1: neck(256@H/32) + c4(512@H/32) → out1(256@H/32)
        out1 = self.apud1(neck_out, c4)
        
        # APUD-2: out1(256@H/32) + c3(256@H/16) → out2(256@H/16)
        out2 = self.apud2(out1, c3)
        
        # APUD-3: out2(256@H/16) + c2(128@H/8) → out3(256@H/8)
        out3 = self.apud3(out2, c2)
        
        # APUD-4: out3(256@H/8) + c1(64@H/4) → out4(256@H/4)
        out4 = self.apud4(out3, c1)
        
        # Main segmentation output (upsample to full resolution)
        main_out = self.seg_head(out4)  # (B, num_classes, H/4, W/4)
        main_out = F.interpolate(main_out, scale_factor=4, mode='bilinear', align_corners=True)
        
        result = {'main': main_out}
        
        if return_aux:
            # Auxiliary outputs at native resolutions (for supervision)
            aux1 = self.aux_head1(out1)  # @ H/32
            aux2 = self.aux_head2(out2)  # @ H/16
            aux3 = self.aux_head3(out3)  # @ H/8
            aux4 = self.aux_head4(out4)  # @ H/4
            
            result['aux'] = [aux1, aux2, aux3, aux4]
        
        return result


if __name__ == "__main__":
    print("=" * 60)
    print("Testing APUD Module")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test SE Attention
    print("\n1. Testing SE Attention:")
    se = SEAttention(256, reduction=16).to(device)
    x = torch.randn(2, 256, 12, 20).to(device)
    out = se(x)
    print(f"   Input: {x.shape} → Output: {out.shape}")
    assert out.shape == x.shape, "SE Attention shape mismatch!"
    print("   ✓ SE Attention passed")
    
    # Test Spatial Attention
    print("\n2. Testing Spatial Attention:")
    sa = SpatialAttention(kernel_size=7).to(device)
    x = torch.randn(2, 256, 12, 20).to(device)
    out = sa(x)
    print(f"   Input: {x.shape} → Output: {out.shape}")
    assert out.shape == x.shape, "Spatial Attention shape mismatch!"
    print("   ✓ Spatial Attention passed")
    
    # Test APUD Block
    print("\n3. Testing APUD Block:")
    apud = APUDBlock(256, 512, 256).to(device)
    x_low = torch.randn(2, 256, 12, 20).to(device)
    x_high = torch.randn(2, 512, 24, 40).to(device)
    out = apud(x_low, x_high)
    print(f"   x_low: {x_low.shape}")
    print(f"   x_high: {x_high.shape}")
    print(f"   Output: {out.shape}")
    assert out.shape == (2, 256, 24, 40), "APUD Block shape mismatch!"
    print("   ✓ APUD Block passed")
    
    # Test Full APUD Decoder
    print("\n4. Testing APUD Decoder:")
    decoder = APUDDecoder(
        encoder_channels=[64, 128, 256, 512],  # [c1, c2, c3, c4]
        neck_channels=256,
        decoder_channels=256,
        num_classes=2
    ).to(device)
    
    # Simulate inputs matching CSPDarknet-53 outputs
    neck = torch.randn(2, 256, 12, 20).to(device)  # ASPP output @ H/32, W/32
    c1 = torch.randn(2, 64, 96, 160).to(device)    # H/4, W/4
    c2 = torch.randn(2, 128, 48, 80).to(device)    # H/8, W/8
    c3 = torch.randn(2, 256, 24, 40).to(device)    # H/16, W/16
    c4 = torch.randn(2, 512, 12, 20).to(device)    # H/32, W/32
    
    encoder_features = [c1, c2, c3, c4]
    
    result = decoder(neck, encoder_features, return_aux=True)
    
    print(f"   Neck input: {neck.shape}")
    print(f"   Main output: {result['main'].shape}")
    print(f"   Aux outputs:")
    for i, aux in enumerate(result['aux']):
        print(f"     Aux-{i+1}: {aux.shape}")
    
    # Expected main output: (2, 2, 384, 640)
    assert result['main'].shape == (2, 2, 384, 640), f"Main output shape mismatch! Got {result['main'].shape}"
    print("   ✓ APUD Decoder passed")
    
    # Count parameters
    params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"\n   Decoder Parameters: {params:,}")
    
    print("\n" + "=" * 60)
    print("All APUD tests passed!")
    print("=" * 60)
