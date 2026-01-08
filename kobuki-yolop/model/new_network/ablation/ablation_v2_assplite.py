"""
Ablation Model V2: CSPDarknet + ASPPLite
========================================
Second ablation model for the study.

Architecture:
- Encoder: CSPDarknet backbone (Focus + Conv + BottleneckCSP)
- Neck: ASPPLite (replaces SPP from V1)
- Decoder: Simple bilinear upsampling with conv layers (same as V1)

ASPPLite (as per paper):
- 3 parallel dilated convolutions with dilation rates: 1, 6, 12
- Each branch: 3x3 Conv + BatchNorm + ReLU
- 128 filters per branch
- NO Global Average Pooling (unlike standard ASPP)
- Outputs are concatenated for multi-scale feature fusion

This is the model WITH ASPPLite, but WITHOUT:
- APUD decoder (added in V3)
- RBRM boundary refinement (added in V4)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def autopad(k, p=None):
    """Auto-calculate padding for 'same' convolution."""
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Hardswish(nn.Module):
    """Export-friendly version of nn.Hardswish()."""
    @staticmethod
    def forward(x):
        return x * F.hardtanh(x + 3, 0., 6.) / 6.


class Conv(nn.Module):
    """Standard convolution with BatchNorm and activation."""
    
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = Hardswish() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Standard bottleneck block."""
    
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck - Cross Stage Partial Networks."""
    
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class Focus(nn.Module):
    """Focus layer - reduces spatial size while increasing channels."""
    
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)

    def forward(self, x):
        return self.conv(torch.cat([
            x[..., ::2, ::2], 
            x[..., 1::2, ::2], 
            x[..., ::2, 1::2], 
            x[..., 1::2, 1::2]
        ], 1))


class ASPPLite(nn.Module):
    """
    ASPP-Lite Module (as per paper).
    
    Captures multi-scale contextual information using dilated convolutions.
    
    Features:
    - 3 parallel branches with dilation rates: 1, 6, 12
    - Each branch: 3x3 Conv (128 filters) + BatchNorm + ReLU
    - NO Global Average Pooling (preserves spatial integrity)
    - Outputs concatenated for feature fusion
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels (after 1x1 fusion conv)
        mid_channels: Number of filters per branch (default: 128 as per paper)
    """
    
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = 128):
        super(ASPPLite, self).__init__()
        
        # Branch 1: dilation = 1 (local features)
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, 
                      stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        # Branch 2: dilation = 6 (mid-range dependencies)
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, 
                      stride=1, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        # Branch 3: dilation = 12 (broader receptive field)
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, 
                      stride=1, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        # Fusion: 1x1 convolution to combine all branches
        # Concatenated channels = 3 * mid_channels
        self.fusion = nn.Sequential(
            nn.Conv2d(mid_channels * 3, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Multi-scale fused features (B, out_channels, H, W)
        """
        # Apply parallel branches
        b1 = self.branch1(x)  # dilation=1
        b2 = self.branch2(x)  # dilation=6
        b3 = self.branch3(x)  # dilation=12
        
        # Concatenate along channel dimension
        concat = torch.cat([b1, b2, b3], dim=1)
        
        # Fuse with 1x1 convolution
        output = self.fusion(concat)
        
        return output


class CSPDarknetEncoderASPPLite(nn.Module):
    """
    CSPDarknet Encoder with ASPPLite instead of SPP.
    
    The key difference from V1:
    - Replaces SPP (max pooling based) with ASPPLite (dilated conv based)
    - ASPPLite captures multi-scale context through dilation rates 1, 6, 12
    """
    
    def __init__(self, in_channels: int = 3):
        super(CSPDarknetEncoderASPPLite, self).__init__()
        
        # Stage 0: Focus - 1/2 resolution
        self.focus = Focus(in_channels, 64, k=3, s=1)
        
        # Stage 1: Conv + CSP - 1/4 resolution
        self.conv1 = Conv(64, 128, k=3, s=2)
        self.csp1 = BottleneckCSP(128, 128, n=1)
        
        # Stage 2: Conv + CSP - 1/8 resolution
        self.conv2 = Conv(128, 256, k=3, s=2)
        self.csp2 = BottleneckCSP(256, 256, n=3)
        
        # Stage 3: Conv + CSP - 1/16 resolution
        self.conv3 = Conv(256, 512, k=3, s=2)
        self.csp3 = BottleneckCSP(512, 512, n=3)
        
        # Stage 4: Conv + ASPPLite + CSP - 1/32 resolution
        self.conv4 = Conv(512, 1024, k=3, s=2)
        # ASPPLite replaces SPP here
        self.aspp_lite = ASPPLite(in_channels=1024, out_channels=1024, mid_channels=128)
        self.csp4 = BottleneckCSP(1024, 1024, n=1, shortcut=False)
        
    def forward(self, x) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass returning multi-scale features.
        
        Args:
            x: Input tensor (B, 3, H, W)
            
        Returns:
            Tuple of feature maps at different scales:
            - c1: (B, 64, H/2, W/2)
            - c2: (B, 128, H/4, W/4)  
            - c3: (B, 256, H/8, W/8)
            - c4: (B, 512, H/16, W/16)
            - c5: (B, 1024, H/32, W/32) - after ASPPLite
        """
        c1 = self.focus(x)           # 1/2
        
        c2 = self.conv1(c1)          # 1/4
        c2 = self.csp1(c2)
        
        c3 = self.conv2(c2)          # 1/8
        c3 = self.csp2(c3)
        
        c4 = self.conv3(c3)          # 1/16
        c4 = self.csp3(c4)
        
        c5 = self.conv4(c4)          # 1/32
        c5 = self.aspp_lite(c5)      # ASPPLite instead of SPP
        c5 = self.csp4(c5)
        
        return c1, c2, c3, c4, c5


class SimpleSegmentationHead(nn.Module):
    """
    Simple segmentation decoder head (YOLOP-style).
    Same as V1 - no changes for V2.
    """
    
    def __init__(self, num_classes: int = 2):
        super(SimpleSegmentationHead, self).__init__()
        
        # Stage 1: 1/32 -> 1/16
        self.up1 = nn.Sequential(
            Conv(1024, 512, k=1, s=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.conv1 = Conv(512 + 512, 512, k=3, s=1)
        
        # Stage 2: 1/16 -> 1/8
        self.up2 = nn.Sequential(
            Conv(512, 256, k=1, s=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.conv2 = Conv(256 + 256, 256, k=3, s=1)
        
        # Stage 3: 1/8 -> 1/4
        self.up3 = nn.Sequential(
            Conv(256, 128, k=1, s=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.conv3 = Conv(128 + 128, 128, k=3, s=1)
        
        # Stage 4: 1/4 -> 1/2
        self.up4 = nn.Sequential(
            Conv(128, 64, k=1, s=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.conv4 = Conv(64 + 64, 64, k=3, s=1)
        
        # Final upsampling: 1/2 -> 1/1
        self.up5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Output head
        self.output = nn.Sequential(
            Conv(64, 64, k=3, s=1),
            nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0)
        )
        
    def forward(self, c1, c2, c3, c4, c5) -> torch.Tensor:
        x = self.up1(c5)
        x = self.conv1(torch.cat([x, c4], dim=1))
        
        x = self.up2(x)
        x = self.conv2(torch.cat([x, c3], dim=1))
        
        x = self.up3(x)
        x = self.conv3(torch.cat([x, c2], dim=1))
        
        x = self.up4(x)
        x = self.conv4(torch.cat([x, c1], dim=1))
        
        x = self.up5(x)
        x = self.output(x)
        
        return x


class AblationV2ASPPLite(nn.Module):
    """
    Ablation Model V2: CSPDarknet + ASPPLite
    
    Components:
    - CSPDarknet Encoder with ASPPLite (replaces SPP)
    - Simple bilinear upsampling decoder with skip connections
    
    Key difference from V1:
    - Uses ASPPLite (dilated convs at rates 1, 6, 12) instead of SPP (max pooling)
    - ASPPLite preserves spatial details better through dilated convolutions
    - No Global Average Pooling, maintaining border feature integrity
    """
    
    def __init__(self, num_classes: int = 2, in_channels: int = 3):
        super(AblationV2ASPPLite, self).__init__()
        
        self.encoder = CSPDarknetEncoderASPPLite(in_channels=in_channels)
        self.decoder = SimpleSegmentationHead(num_classes=num_classes)
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Segmentation output (B, num_classes, H, W)
        """
        c1, c2, c3, c4, c5 = self.encoder(x)
        output = self.decoder(c1, c2, c3, c4, c5)
        return output
    
    def get_params_count(self) -> dict:
        """Get parameter count breakdown."""
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        total_params = encoder_params + decoder_params
        
        return {
            'encoder': encoder_params,
            'decoder': decoder_params,
            'total': total_params,
            'total_M': total_params / 1e6
        }


def build_ablation_v2(num_classes: int = 2, pretrained: bool = False) -> AblationV2ASPPLite:
    """Build ablation V2 model with ASPPLite."""
    model = AblationV2ASPPLite(num_classes=num_classes)
    return model


if __name__ == '__main__':
    # Test the model
    print("Testing Ablation V2 Model (ASPPLite)...")
    
    model = AblationV2ASPPLite(num_classes=2)
    x = torch.randn(1, 3, 640, 384)
    
    print(f"Input shape: {x.shape}")
    output = model(x)
    print(f"Output shape: {output.shape}")
    
    params = model.get_params_count()
    print(f"\nParameter count:")
    print(f"  Encoder: {params['encoder']:,}")
    print(f"  Decoder: {params['decoder']:,}")
    print(f"  Total: {params['total']:,} ({params['total_M']:.2f}M)")
    
    # Compare with V1 (SPP) parameter count
    print("\n--- ASPPLite Module Stats ---")
    aspp = ASPPLite(in_channels=1024, out_channels=1024, mid_channels=128)
    aspp_params = sum(p.numel() for p in aspp.parameters())
    print(f"  ASPPLite parameters: {aspp_params:,}")
