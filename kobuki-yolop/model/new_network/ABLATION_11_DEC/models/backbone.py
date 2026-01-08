"""
CSPDarknet-53 Backbone for AURASeg
Paper: AURASeg - Attention Guided Upsampling with Residual Boundary-Assistive Refinement

Architecture (from paper):
    Input (3, H, W)
        ↓
    Focus Layer → (32, H/2, W/2)
        ↓
    Conv1 (CSP) → (64, H/4, W/4)      → c1 (skip connection)
        ↓
    Conv2 (CSP) → (128, H/8, W/8)     → c2 (skip connection)
        ↓
    Conv3 (CSP) → (256, H/16, W/16)   → c3 (skip connection)
        ↓
    Conv4 (CSP) → (512, H/32, W/32)   → c4 (skip connection)
        ↓
    Conv5 (CSP) → (1024, H/32, W/32)  → c5 (to SPP/ASPP-Lite)
"""

import torch
import torch.nn as nn


class ConvBNAct(nn.Module):
    """Standard Convolution + BatchNorm + Activation block"""
    
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, 
                 padding=None, groups=1, act=True):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, 
                              padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Focus(nn.Module):
    """
    Focus layer - Slices input into 4 parts and concatenates
    Reduces spatial dimensions by 2x while increasing channels by 4x
    
    Input: (B, C, H, W)
    Output: (B, out_channels, H/2, W/2)
    """
    
    def __init__(self, in_channels=3, out_channels=32, kernel_size=3):
        super().__init__()
        # After slicing: 4 * in_channels
        self.conv = ConvBNAct(in_channels * 4, out_channels, kernel_size)
    
    def forward(self, x):
        # Slice input into 4 parts (top-left, top-right, bottom-left, bottom-right)
        # x shape: (B, C, H, W)
        return self.conv(
            torch.cat([
                x[..., ::2, ::2],    # Top-left pixels
                x[..., 1::2, ::2],   # Bottom-left pixels  
                x[..., ::2, 1::2],   # Top-right pixels
                x[..., 1::2, 1::2],  # Bottom-right pixels
            ], dim=1)
        )


class Bottleneck(nn.Module):
    """
    Standard bottleneck block with residual connection
    
    Structure:
        x → Conv1x1 → Conv3x3 → + → out
        └─────────────────────────┘
    """
    
    def __init__(self, in_channels, out_channels, shortcut=True, expansion=0.5):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        
        self.conv1 = ConvBNAct(in_channels, hidden_channels, 1)
        self.conv2 = ConvBNAct(hidden_channels, out_channels, 3)
        self.add = shortcut and in_channels == out_channels
    
    def forward(self, x):
        if self.add:
            return x + self.conv2(self.conv1(x))
        return self.conv2(self.conv1(x))


class CSPBlock(nn.Module):
    """
    Cross Stage Partial Block
    Splits input, processes one half through bottlenecks, concatenates with other half
    
    Structure:
        x → split ─┬─→ Conv1x1 → n × Bottleneck → Conv1x1 ─┬─→ Concat → Conv1x1 → out
                   │                                        │
                   └────────────→ Conv1x1 ─────────────────┘
    """
    
    def __init__(self, in_channels, out_channels, n_bottlenecks=1, shortcut=True):
        super().__init__()
        hidden_channels = out_channels // 2
        
        # Main branch
        self.conv1 = ConvBNAct(in_channels, hidden_channels, 1)
        self.bottlenecks = nn.Sequential(*[
            Bottleneck(hidden_channels, hidden_channels, shortcut) 
            for _ in range(n_bottlenecks)
        ])
        self.conv2 = ConvBNAct(hidden_channels, hidden_channels, 1)
        
        # Skip branch
        self.conv3 = ConvBNAct(in_channels, hidden_channels, 1)
        
        # Fusion
        self.conv4 = ConvBNAct(hidden_channels * 2, out_channels, 1)
    
    def forward(self, x):
        # Main branch with bottlenecks
        y1 = self.conv2(self.bottlenecks(self.conv1(x)))
        # Skip branch
        y2 = self.conv3(x)
        # Concatenate and fuse
        return self.conv4(torch.cat([y1, y2], dim=1))


class SPP(nn.Module):
    """
    Spatial Pyramid Pooling
    Uses MaxPool with kernels 5, 9, 13 to capture multi-scale context
    
    Input: (B, C, H, W)
    Output: (B, out_channels, H, W)
    """
    
    def __init__(self, in_channels, out_channels, kernels=(5, 9, 13)):
        super().__init__()
        hidden_channels = in_channels // 2
        
        self.conv1 = ConvBNAct(in_channels, hidden_channels, 1)
        
        self.pools = nn.ModuleList([
            nn.MaxPool2d(k, stride=1, padding=k // 2) for k in kernels
        ])
        
        # After pooling: hidden_channels * (1 + len(kernels))
        self.conv2 = ConvBNAct(hidden_channels * (1 + len(kernels)), out_channels, 1)
    
    def forward(self, x):
        x = self.conv1(x)
        
        # Original + 3 pooled versions
        pooled = [x] + [pool(x) for pool in self.pools]
        
        return self.conv2(torch.cat(pooled, dim=1))


class CSPDarknet53(nn.Module):
    """
    CSPDarknet-53 Encoder as per AURASeg paper
    
    Architecture:
        Focus (32) → Conv1 (64) → Conv2 (128) → Conv3 (256) → Conv4 (512) → Conv5 (1024)
        
    Returns multi-scale features: c1, c2, c3, c4, c5
    
    Feature scales:
        c1: H/4, W/4, 64 channels
        c2: H/8, W/8, 128 channels
        c3: H/16, W/16, 256 channels
        c4: H/32, W/32, 512 channels
        c5: H/32, W/32, 1024 channels
    """
    
    def __init__(self, in_channels=3):
        super().__init__()
        
        # Focus: (3, H, W) → (32, H/2, W/2)
        self.focus = Focus(in_channels, 32, kernel_size=3)
        
        # Conv1: (32, H/2, W/2) → (64, H/4, W/4)
        self.conv1 = nn.Sequential(
            ConvBNAct(32, 64, 3, stride=2),
            CSPBlock(64, 64, n_bottlenecks=1)
        )
        
        # Conv2: (64, H/4, W/4) → (128, H/8, W/8)
        self.conv2 = nn.Sequential(
            ConvBNAct(64, 128, 3, stride=2),
            CSPBlock(128, 128, n_bottlenecks=3)
        )
        
        # Conv3: (128, H/8, W/8) → (256, H/16, W/16)
        self.conv3 = nn.Sequential(
            ConvBNAct(128, 256, 3, stride=2),
            CSPBlock(256, 256, n_bottlenecks=3)
        )
        
        # Conv4: (256, H/16, W/16) → (512, H/32, W/32)
        self.conv4 = nn.Sequential(
            ConvBNAct(256, 512, 3, stride=2),
            CSPBlock(512, 512, n_bottlenecks=1)
        )
        
        # Conv5: (512, H/32, W/32) → (1024, H/32, W/32)
        self.conv5 = nn.Sequential(
            ConvBNAct(512, 1024, 3, stride=1),  # No spatial reduction
            CSPBlock(1024, 1024, n_bottlenecks=1)
        )
        
        # Output channel dimensions for each stage
        self.out_channels = [64, 128, 256, 512, 1024]
    
    def forward(self, x):
        """
        Args:
            x: Input tensor (B, 3, H, W)
            
        Returns:
            c1, c2, c3, c4, c5: Multi-scale feature maps
        """
        x = self.focus(x)      # (B, 32, H/2, W/2)
        
        c1 = self.conv1(x)     # (B, 64, H/4, W/4)
        c2 = self.conv2(c1)    # (B, 128, H/8, W/8)
        c3 = self.conv3(c2)    # (B, 256, H/16, W/16)
        c4 = self.conv4(c3)    # (B, 512, H/32, W/32)
        c5 = self.conv5(c4)    # (B, 1024, H/32, W/32)
        
        return c1, c2, c3, c4, c5


if __name__ == "__main__":
    # Test the backbone
    model = CSPDarknet53(in_channels=3)
    x = torch.randn(2, 3, 384, 640)
    
    c1, c2, c3, c4, c5 = model(x)
    
    print("CSPDarknet-53 Output Shapes:")
    print(f"  Input:  {x.shape}")
    print(f"  c1:     {c1.shape}  (H/4, W/4)")
    print(f"  c2:     {c2.shape}  (H/8, W/8)")
    print(f"  c3:     {c3.shape}  (H/16, W/16)")
    print(f"  c4:     {c4.shape}  (H/32, W/32)")
    print(f"  c5:     {c5.shape}  (H/32, W/32)")
    
    # Parameter count
    params = sum(p.numel() for p in model.parameters())
    print(f"\n  Total Parameters: {params:,}")
