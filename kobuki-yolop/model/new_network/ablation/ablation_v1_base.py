"""
Base Ablation Model (V1)
========================
Minimal segmentation model for ablation study baseline.

Architecture:
- Encoder: CSPDarknet backbone (Focus + Conv + BottleneckCSP)
- Neck: SPP (Spatial Pyramid Pooling)
- Decoder: Simple bilinear upsampling with conv layers (YOLOP-style)

This is the BASE model without:
- ASSPLite (added in V2)
- APUD decoder (added in V3)
- RBRM boundary refinement (added in V4)

Reference: YOLOP drivable area segmentation head architecture
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
        """
        Args:
            c1: Input channels
            c2: Output channels
            k: Kernel size
            s: Stride
            p: Padding
            g: Groups
            act: Use activation
        """
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = Hardswish() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Standard bottleneck block."""
    
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        """
        Args:
            c1: Input channels
            c2: Output channels
            shortcut: Use residual connection
            g: Groups
            e: Expansion ratio
        """
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
        """
        Args:
            c1: Input channels
            c2: Output channels
            n: Number of bottleneck blocks
            shortcut: Use residual connection
            g: Groups
            e: Expansion ratio
        """
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


class SPP(nn.Module):
    """Spatial Pyramid Pooling layer (YOLOv3-SPP)."""
    
    def __init__(self, c1, c2, k=(5, 9, 13)):
        """
        Args:
            c1: Input channels
            c2: Output channels
            k: Tuple of pooling kernel sizes
        """
        super(SPP, self).__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class Focus(nn.Module):
    """Focus layer - reduces spatial size while increasing channels."""
    
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """
        Args:
            c1: Input channels
            c2: Output channels
            k: Kernel size
            s: Stride
            p: Padding
            g: Groups
            act: Use activation
        """
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)

    def forward(self, x):
        # x(b,c,h,w) -> y(b,4c,h/2,w/2)
        return self.conv(torch.cat([
            x[..., ::2, ::2], 
            x[..., 1::2, ::2], 
            x[..., ::2, 1::2], 
            x[..., 1::2, 1::2]
        ], 1))


class CSPDarknetEncoder(nn.Module):
    """
    CSPDarknet Encoder backbone.
    
    Produces multi-scale features at 1/2, 1/4, 1/8, 1/16, 1/32 of input resolution.
    """
    
    def __init__(self, in_channels: int = 3):
        super(CSPDarknetEncoder, self).__init__()
        
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
        
        # Stage 4: Conv + SPP + CSP - 1/32 resolution
        self.conv4 = Conv(512, 1024, k=3, s=2)
        self.spp = SPP(1024, 1024, k=(5, 9, 13))
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
            - c5: (B, 1024, H/32, W/32) - after SPP
        """
        c1 = self.focus(x)           # 1/2
        
        c2 = self.conv1(c1)          # 1/4
        c2 = self.csp1(c2)
        
        c3 = self.conv2(c2)          # 1/8
        c3 = self.csp2(c3)
        
        c4 = self.conv3(c3)          # 1/16
        c4 = self.csp3(c4)
        
        c5 = self.conv4(c4)          # 1/32
        c5 = self.spp(c5)
        c5 = self.csp4(c5)
        
        return c1, c2, c3, c4, c5


class SimpleSegmentationHead(nn.Module):
    """
    Simple segmentation decoder head (YOLOP-style).
    
    Uses bilinear upsampling with skip connections from encoder.
    This is the baseline decoder for ablation study.
    """
    
    def __init__(self, num_classes: int = 2):
        super(SimpleSegmentationHead, self).__init__()
        
        # Decoder stages - progressive upsampling
        # From c5 (1024ch, 1/32) to full resolution
        
        # Stage 1: 1/32 -> 1/16
        self.up1 = nn.Sequential(
            Conv(1024, 512, k=1, s=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.conv1 = Conv(512 + 512, 512, k=3, s=1)  # Concat with c4
        
        # Stage 2: 1/16 -> 1/8
        self.up2 = nn.Sequential(
            Conv(512, 256, k=1, s=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.conv2 = Conv(256 + 256, 256, k=3, s=1)  # Concat with c3
        
        # Stage 3: 1/8 -> 1/4
        self.up3 = nn.Sequential(
            Conv(256, 128, k=1, s=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.conv3 = Conv(128 + 128, 128, k=3, s=1)  # Concat with c2
        
        # Stage 4: 1/4 -> 1/2
        self.up4 = nn.Sequential(
            Conv(128, 64, k=1, s=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.conv4 = Conv(64 + 64, 64, k=3, s=1)  # Concat with c1
        
        # Final upsampling: 1/2 -> 1/1
        self.up5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Output head
        self.output = nn.Sequential(
            Conv(64, 64, k=3, s=1),
            nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0)
        )
        
    def forward(self, c1, c2, c3, c4, c5) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            c1-c5: Multi-scale features from encoder
            
        Returns:
            Segmentation output at full resolution (B, num_classes, H, W)
        """
        # Upsample and concatenate with skip connections
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


class AblationBaseModel(nn.Module):
    """
    Base Ablation Model (V1)
    
    Simple segmentation model for ablation study baseline.
    
    Components:
    - CSPDarknet Encoder with SPP
    - Simple bilinear upsampling decoder with skip connections
    
    This model does NOT include:
    - ASSPLite (V2)
    - APUD decoder (V3)
    - RBRM boundary refinement (V4)
    """
    
    def __init__(self, num_classes: int = 2, in_channels: int = 3):
        """
        Args:
            num_classes: Number of segmentation classes
            in_channels: Number of input channels (default: 3 for RGB)
        """
        super(AblationBaseModel, self).__init__()
        
        self.encoder = CSPDarknetEncoder(in_channels=in_channels)
        self.decoder = SimpleSegmentationHead(num_classes=num_classes)
        
        # Initialize weights
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
        # Encoder
        c1, c2, c3, c4, c5 = self.encoder(x)
        
        # Decoder
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


def build_ablation_base(num_classes: int = 2, pretrained: bool = False) -> AblationBaseModel:
    """
    Build ablation base model.
    
    Args:
        num_classes: Number of segmentation classes
        pretrained: Load pretrained weights (not implemented)
        
    Returns:
        AblationBaseModel instance
    """
    model = AblationBaseModel(num_classes=num_classes)
    
    if pretrained:
        # TODO: Load pretrained CSPDarknet weights
        pass
    
    return model


if __name__ == '__main__':
    # Test the model
    print("Testing Ablation Base Model...")
    
    model = AblationBaseModel(num_classes=2)
    x = torch.randn(1, 3, 640, 384)
    
    print(f"Input shape: {x.shape}")
    output = model(x)
    print(f"Output shape: {output.shape}")
    
    params = model.get_params_count()
    print(f"\nParameter count:")
    print(f"  Encoder: {params['encoder']:,}")
    print(f"  Decoder: {params['decoder']:,}")
    print(f"  Total: {params['total']:,} ({params['total_M']:.2f}M)")
    
    # Test with different input sizes
    print("\nTesting with different input sizes...")
    for h, w in [(480, 640), (384, 640), (320, 480)]:
        x = torch.randn(1, 3, h, w)
        output = model(x)
        print(f"  Input: {x.shape} -> Output: {output.shape}")
