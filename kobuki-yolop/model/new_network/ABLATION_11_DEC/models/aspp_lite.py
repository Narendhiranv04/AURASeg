"""
ASPP-Lite Module for AURASeg
Paper: AURASeg - Attention Guided Upsampling with Residual Boundary-Assistive Refinement

ASPP-Lite Configuration (from paper Fig. 3):
    4 Parallel Branches, each with 128 filters:
    
    Branch 1: Conv 1×1, dilation=1
    Branch 2: Conv 3×3, dilation=1  
    Branch 3: Conv 3×3, dilation=6
    Branch 4: Conv 3×3, dilation=12
    
    All branches are concatenated (128×4 = 512 channels)
    Then fused with Conv 1×1 to output channels
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPPLiteConv(nn.Module):
    """Single ASPP-Lite convolution branch"""
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        
        padding = 0 if kernel_size == 1 else dilation
        
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ASPPLite(nn.Module):
    """
    ASPP-Lite Module as per AURASeg paper
    
    4 Branches with 128 filters each:
        - Branch 1: 1×1 conv, dilation=1
        - Branch 2: 3×3 conv, dilation=1
        - Branch 3: 3×3 conv, dilation=6
        - Branch 4: 3×3 conv, dilation=12
    
    Input: (B, in_channels, H, W)
    Output: (B, out_channels, H, W)
    """
    
    def __init__(self, in_channels=1024, out_channels=256, branch_channels=128):
        super().__init__()
        
        # 4 parallel branches as per paper
        self.branch1 = ASPPLiteConv(in_channels, branch_channels, kernel_size=1, dilation=1)
        self.branch2 = ASPPLiteConv(in_channels, branch_channels, kernel_size=3, dilation=1)
        self.branch3 = ASPPLiteConv(in_channels, branch_channels, kernel_size=3, dilation=6)
        self.branch4 = ASPPLiteConv(in_channels, branch_channels, kernel_size=3, dilation=12)
        
        # Fusion: 4 branches × 128 channels = 512 channels
        self.fusion = nn.Sequential(
            nn.Conv2d(branch_channels * 4, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout2d(0.1)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor (B, in_channels, H, W) - typically c5 from backbone
            
        Returns:
            Fused multi-scale features (B, out_channels, H, W)
        """
        # Apply all 4 branches in parallel
        b1 = self.branch1(x)  # 1×1, d=1
        b2 = self.branch2(x)  # 3×3, d=1
        b3 = self.branch3(x)  # 3×3, d=6
        b4 = self.branch4(x)  # 3×3, d=12
        
        # Concatenate all branches
        concat = torch.cat([b1, b2, b3, b4], dim=1)
        
        # Fuse to output channels
        out = self.fusion(concat)
        out = self.dropout(out)
        
        return out


if __name__ == "__main__":
    # Test ASPP-Lite
    model = ASPPLite(in_channels=1024, out_channels=256, branch_channels=128)
    x = torch.randn(2, 1024, 12, 20)  # Simulating c5 from 384×640 input
    
    out = model(x)
    
    print("ASPP-Lite Output:")
    print(f"  Input:  {x.shape}")
    print(f"  Output: {out.shape}")
    
    # Parameter count
    params = sum(p.numel() for p in model.parameters())
    print(f"\n  Total Parameters: {params:,}")
