"""
Ablation V3: APUD (Attention Progressive Upsampling Decoder)

Building on V2 (CSPDarknet + ASPPLite), this version adds the APUD decoder
with SE Attention and Spatial Attention modules as specified in the paper.

APUD Architecture (per paper):
1. Squeeze-and-Excitation (SE) Attention: Channel-level reweighting via 
   Global Average Pooling → FC → ReLU → FC → Sigmoid
2. Spatial Attention: Aggregates max and average-pooled cues across channels,
   then applies 7x7 convolution to produce a spatial mask
3. Progressive upsampling with skip connections from encoder

APUD Block Flow:
- x_high (smaller/deeper) → 1x1 transform → SE Attention → multiply with x_low
- x_low (larger/shallower) → 1x1 transform → used in both SE multiply and Spatial Attention
- Multiply result → Upsample → Add with Spatial Attention output
- Sum → 3x3 Conv + BatchNorm + ReLU refinement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


# =============================================================================
# Encoder: CSPDarknet (same as V2)
# =============================================================================

class ConvBNAct(nn.Module):
    """Standard Conv + BatchNorm + Activation block"""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, 
                 stride: int = 1, padding: int = None, groups: int = 1,
                 act: bool = True):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, 
                              groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """CSP Bottleneck with residual connection"""
    def __init__(self, in_ch: int, out_ch: int, shortcut: bool = True, 
                 expansion: float = 0.5):
        super().__init__()
        hidden_ch = int(out_ch * expansion)
        self.cv1 = ConvBNAct(in_ch, hidden_ch, 1)
        self.cv2 = ConvBNAct(hidden_ch, out_ch, 3)
        self.add = shortcut and in_ch == out_ch
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.cv2(self.cv1(x))
        return x + out if self.add else out


class CSPBlock(nn.Module):
    """Cross Stage Partial block"""
    def __init__(self, in_ch: int, out_ch: int, n: int = 1, 
                 shortcut: bool = True, expansion: float = 0.5):
        super().__init__()
        hidden_ch = int(out_ch * expansion)
        self.cv1 = ConvBNAct(in_ch, hidden_ch, 1)
        self.cv2 = ConvBNAct(in_ch, hidden_ch, 1)
        self.cv3 = ConvBNAct(2 * hidden_ch, out_ch, 1)
        self.m = nn.Sequential(*[Bottleneck(hidden_ch, hidden_ch, shortcut, 1.0) 
                                  for _ in range(n)])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class Focus(nn.Module):
    """Focus module - space to depth"""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 1):
        super().__init__()
        self.conv = ConvBNAct(in_ch * 4, out_ch, kernel_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(torch.cat([
            x[..., ::2, ::2],
            x[..., 1::2, ::2],
            x[..., ::2, 1::2],
            x[..., 1::2, 1::2]
        ], dim=1))


class CSPDarknet(nn.Module):
    """CSPDarknet backbone encoder"""
    def __init__(self, base_channels: int = 64, base_depth: int = 3):
        super().__init__()
        # Stem
        self.stem = Focus(3, base_channels, 3)
        
        # Stage 1: 1/4 resolution
        self.stage1 = nn.Sequential(
            ConvBNAct(base_channels, base_channels * 2, 3, 2),
            CSPBlock(base_channels * 2, base_channels * 2, base_depth, True)
        )
        
        # Stage 2: 1/8 resolution
        self.stage2 = nn.Sequential(
            ConvBNAct(base_channels * 2, base_channels * 4, 3, 2),
            CSPBlock(base_channels * 4, base_channels * 4, base_depth * 3, True)
        )
        
        # Stage 3: 1/16 resolution
        self.stage3 = nn.Sequential(
            ConvBNAct(base_channels * 4, base_channels * 8, 3, 2),
            CSPBlock(base_channels * 8, base_channels * 8, base_depth * 3, True)
        )
        
        # Stage 4: 1/32 resolution
        self.stage4 = nn.Sequential(
            ConvBNAct(base_channels * 8, base_channels * 16, 3, 2),
            CSPBlock(base_channels * 16, base_channels * 16, base_depth, True)
        )
        
        self.out_channels = [
            base_channels * 2,   # 1/4: 128
            base_channels * 4,   # 1/8: 256
            base_channels * 8,   # 1/16: 512
            base_channels * 16   # 1/32: 1024
        ]
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)      # 1/2
        c2 = self.stage1(x)   # 1/4
        c3 = self.stage2(c2)  # 1/8
        c4 = self.stage3(c3)  # 1/16
        c5 = self.stage4(c4)  # 1/32
        return [c2, c3, c4, c5]


# =============================================================================
# Neck: ASPPLite (same as V2)
# =============================================================================

class ASPPLite(nn.Module):
    """
    Lightweight ASPP with depthwise separable convolutions
    """
    def __init__(self, in_channels: int, out_channels: int, 
                 dilations: List[int] = [1, 6, 12, 18]):
        super().__init__()
        
        # Global average pooling branch
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBNAct(in_channels, out_channels, 1)
        )
        
        # 1x1 convolution branch
        self.conv1x1 = ConvBNAct(in_channels, out_channels, 1)
        
        # Dilated depthwise separable convolution branches
        self.dilated_branches = nn.ModuleList()
        for d in dilations[1:]:  # Skip dilation=1, handled by conv1x1
            self.dilated_branches.append(nn.Sequential(
                # Depthwise
                nn.Conv2d(in_channels, in_channels, 3, padding=d, dilation=d, 
                          groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.SiLU(inplace=True),
                # Pointwise
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True)
            ))
        
        # Fusion: 1 (gap) + 1 (1x1) + 3 (dilated) = 5 branches
        self.fusion = ConvBNAct(out_channels * 5, out_channels, 1)
        self.dropout = nn.Dropout2d(0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2:]
        
        # Global pooling branch
        gap = F.interpolate(self.gap(x), size=(h, w), mode='bilinear', align_corners=False)
        
        # 1x1 branch
        conv1x1 = self.conv1x1(x)
        
        # Dilated branches
        dilated_outs = [branch(x) for branch in self.dilated_branches]
        
        # Concatenate all branches
        out = torch.cat([gap, conv1x1] + dilated_outs, dim=1)
        out = self.fusion(out)
        out = self.dropout(out)
        
        return out


# =============================================================================
# APUD Components: SE Attention and Spatial Attention (FROM PAPER)
# =============================================================================

class SEAttention(nn.Module):
    """
    Squeeze-and-Excitation Attention Module
    
    As specified in the paper:
    "The Squeeze-and-Excitation Attention Module reweighs feature mappings 
    at the channel level through global context modeling."
    
    Process: Global Average Pool → FC → ReLU → FC → Sigmoid → Channel reweighting
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        # Squeeze: global average pooling
        y = self.avg_pool(x).view(b, c)
        # Excitation: FC layers with sigmoid
        y = self.fc(y).view(b, c, 1, 1)
        # Scale: channel-wise multiplication
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module
    
    As specified in the paper:
    "The Spatial Attention Module refines feature maps by aggregating max and 
    average-pooled cues across channels, then applying a lightweight convolution 
    to produce a spatial mask."
    
    Process: [MaxPool, AvgPool] across channels → Concat → 7x7 Conv → Sigmoid
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel-wise max and average pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, H, W]
        # Concatenate pooled features
        y = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W]
        # 7x7 conv to produce spatial attention mask
        y = self.conv(y)
        # Sigmoid activation
        return x * self.sigmoid(y)


class APUDBlock(nn.Module):
    """
    Attention Progressive Upsampling Decoder Block
    
    Architecture from paper diagram:
    1. x_high (deeper features) → 1x1 transform → SE Attention
    2. x_low (shallower features) → 1x1 transform
    3. SE output × x_low_transformed → Upsample
    4. x_low_transformed → Spatial Attention
    5. Upsampled features + Spatial Attention output → Sum
    6. Sum → 3x3 Conv + BN + ReLU refinement
    
    Args:
        high_ch: Channels in high-level (deeper, smaller) features
        low_ch: Channels in low-level (shallower, larger) features  
        out_ch: Output channels
        se_reduction: Reduction ratio for SE attention
    """
    def __init__(self, high_ch: int, low_ch: int, out_ch: int, se_reduction: int = 16):
        super().__init__()
        
        # 1x1 transforms for dimensional consistency
        self.transform_high = ConvBNAct(high_ch, out_ch, 1)
        self.transform_low = ConvBNAct(low_ch, out_ch, 1)
        
        # SE Attention on transformed high features
        self.se_attention = SEAttention(out_ch, reduction=se_reduction)
        
        # Spatial Attention on transformed low features
        self.spatial_attention = SpatialAttention(kernel_size=7)
        
        # Refinement module: 3x3 Conv + BN + ReLU
        self.refinement = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True)
        )
    
    def forward(self, x_high: torch.Tensor, x_low: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_high: High-level features (deeper, smaller spatial size)
            x_low: Low-level features (shallower, larger spatial size)
        """
        # Transform to same channel dimension
        high_transformed = self.transform_high(x_high)  # [B, out_ch, H_high, W_high]
        low_transformed = self.transform_low(x_low)      # [B, out_ch, H_low, W_low]
        
        # Apply SE attention to high-level features
        high_se = self.se_attention(high_transformed)    # [B, out_ch, H_high, W_high]
        
        # Upsample SE-attended high features to match low-level size
        high_upsampled = F.interpolate(
            high_se, 
            size=low_transformed.shape[2:], 
            mode='bilinear', 
            align_corners=False
        )  # [B, out_ch, H_low, W_low]
        
        # Multiply upsampled high features with transformed low features
        fused = high_upsampled * low_transformed  # [B, out_ch, H_low, W_low]
        
        # Apply Spatial Attention to transformed low features
        low_spatial = self.spatial_attention(low_transformed)  # [B, out_ch, H_low, W_low]
        
        # Add spatial attention output to fused features
        out = fused + low_spatial  # [B, out_ch, H_low, W_low]
        
        # Refinement
        out = self.refinement(out)  # [B, out_ch, H_low, W_low]
        
        return out


class APUDDecoder(nn.Module):
    """
    Attention Progressive Upsampling Decoder
    
    Progressive decoder that uses SE + Spatial Attention to fuse multi-scale 
    features from the encoder while progressively upsampling.
    
    Takes encoder features at 4 scales (1/4, 1/8, 1/16, 1/32) and the neck output,
    progressively fuses them bottom-up with attention mechanisms.
    """
    def __init__(self, encoder_channels: List[int], neck_channels: int, 
                 decoder_channels: int = 256, se_reduction: int = 16):
        super().__init__()
        
        # encoder_channels: [128, 256, 512, 1024] for CSPDarknet with base=64
        # neck_channels: output of ASPPLite (256)
        
        # APUD Block 1: Neck (1/32) + Encoder Stage 4 (1/32) → 1/32
        # Since both are at 1/32, we just fuse them
        self.apud1 = APUDBlock(
            high_ch=neck_channels,      # 256 from ASPP
            low_ch=encoder_channels[3], # 1024 from encoder stage 4
            out_ch=decoder_channels,
            se_reduction=se_reduction
        )
        
        # APUD Block 2: APUD1 output (1/32) + Encoder Stage 3 (1/16) → 1/16
        self.apud2 = APUDBlock(
            high_ch=decoder_channels,    # 256 from APUD1
            low_ch=encoder_channels[2],  # 512 from encoder stage 3
            out_ch=decoder_channels,
            se_reduction=se_reduction
        )
        
        # APUD Block 3: APUD2 output (1/16) + Encoder Stage 2 (1/8) → 1/8
        self.apud3 = APUDBlock(
            high_ch=decoder_channels,    # 256 from APUD2
            low_ch=encoder_channels[1],  # 256 from encoder stage 2
            out_ch=decoder_channels,
            se_reduction=se_reduction
        )
        
        # APUD Block 4: APUD3 output (1/8) + Encoder Stage 1 (1/4) → 1/4
        self.apud4 = APUDBlock(
            high_ch=decoder_channels,    # 256 from APUD3
            low_ch=encoder_channels[0],  # 128 from encoder stage 1
            out_ch=decoder_channels,
            se_reduction=se_reduction
        )
        
        self.out_channels = decoder_channels
    
    def forward(self, encoder_features: List[torch.Tensor], neck_feature: torch.Tensor) -> torch.Tensor:
        """
        Args:
            encoder_features: [c2, c3, c4, c5] from encoder at 1/4, 1/8, 1/16, 1/32
            neck_feature: Output from ASPPLite at 1/32
        
        Returns:
            Decoded features at 1/4 resolution
        """
        c2, c3, c4, c5 = encoder_features
        
        # Progressive upsampling with attention
        # Stage 1: Fuse neck with c5 (both at 1/32)
        x = self.apud1(neck_feature, c5)  # 1/32
        
        # Stage 2: Fuse with c4, upsample to 1/16
        x = self.apud2(x, c4)  # 1/16
        
        # Stage 3: Fuse with c3, upsample to 1/8
        x = self.apud3(x, c3)  # 1/8
        
        # Stage 4: Fuse with c2, upsample to 1/4
        x = self.apud4(x, c2)  # 1/4
        
        return x


# =============================================================================
# Segmentation Head
# =============================================================================

class SegmentationHead(nn.Module):
    """Lightweight segmentation head"""
    def __init__(self, in_channels: int, num_classes: int, scale_factor: int = 4):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBNAct(in_channels, in_channels // 2, 3),
            nn.Dropout2d(0.1),
            nn.Conv2d(in_channels // 2, num_classes, 1)
        )
        self.scale_factor = scale_factor
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.scale_factor != 1:
            x = F.interpolate(x, scale_factor=self.scale_factor, 
                              mode='bilinear', align_corners=False)
        return x


# =============================================================================
# Complete V3 Model: CSPDarknet + ASPPLite + APUD
# =============================================================================

class AblationV3APUD(nn.Module):
    """
    Ablation Study V3: APUD (Attention Progressive Upsampling Decoder)
    
    Architecture:
    - Encoder: CSPDarknet (same as V1, V2)
    - Neck: ASPPLite (same as V2)
    - Decoder: APUD with SE Attention + Spatial Attention (NEW in V3)
    
    The APUD decoder uses:
    1. Squeeze-and-Excitation (SE) Attention for channel-level reweighting
    2. Spatial Attention for spatial-level refinement
    3. Progressive upsampling with skip connections from encoder
    """
    def __init__(self, num_classes: int = 3, in_channels: int = 3,
                 base_channels: int = 64, base_depth: int = 3, 
                 decoder_channels: int = 256, aspp_out_channels: int = 256, 
                 se_reduction: int = 16, deep_supervision: bool = False):
        super().__init__()
        
        # Store for compatibility (deep_supervision not used in V3 base)
        self.deep_supervision = deep_supervision
        
        # Encoder
        self.encoder = CSPDarknet(base_channels, base_depth)
        
        # Neck: ASPPLite on the deepest encoder features
        self.neck = ASPPLite(
            in_channels=self.encoder.out_channels[-1],  # 1024
            out_channels=aspp_out_channels               # 256
        )
        
        # Decoder: APUD
        self.decoder = APUDDecoder(
            encoder_channels=self.encoder.out_channels,  # [128, 256, 512, 1024]
            neck_channels=aspp_out_channels,              # 256
            decoder_channels=decoder_channels,            # 256
            se_reduction=se_reduction
        )
        
        # Segmentation head - decoder outputs at 1/4, we upsample 4x
        self.seg_head = SegmentationHead(
            in_channels=decoder_channels,
            num_classes=num_classes,
            scale_factor=4
        )
        
        self._init_weights()
    
    def _init_weights(self):
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder - multi-scale features
        encoder_features = self.encoder(x)  # [c2, c3, c4, c5]
        
        # Neck - process deepest features with ASPP
        neck_out = self.neck(encoder_features[-1])  # 1/32
        
        # Decoder - progressive upsampling with attention
        decoder_out = self.decoder(encoder_features, neck_out)  # 1/4
        
        # Segmentation head - final prediction
        out = self.seg_head(decoder_out)  # 1/1
        
        return out


def create_model(num_classes: int = 3, **kwargs) -> AblationV3APUD:
    """Factory function to create V3 APUD model"""
    return AblationV3APUD(num_classes=num_classes, **kwargs)


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    # Test the model
    model = AblationV3APUD(num_classes=3)
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"V3 APUD Model Architecture")
    print(f"=" * 50)
    print(f"Encoder: CSPDarknet")
    print(f"Neck: ASPPLite")
    print(f"Decoder: APUD (SE Attention + Spatial Attention)")
    print(f"=" * 50)
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
    
    # Test forward pass
    x = torch.randn(1, 3, 384, 640)
    with torch.no_grad():
        y = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    # Verify attention modules
    print(f"\nAttention Modules in APUD:")
    print(f"  - SE Attention: Channel reweighting via GAP → FC → ReLU → FC → Sigmoid")
    print(f"  - Spatial Attention: Max/Avg pool → 7x7 Conv → Sigmoid spatial mask")
