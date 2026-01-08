"""
Decoder Module for AURASeg
Paper: AURASeg - Attention Guided Upsampling with Residual Boundary-Assistive Refinement

Decoder Structure:
    Takes multi-scale features from encoder (c1, c2, c3, c4) as skip connections
    Progressively upsamples and fuses with skip connections
    
    Flow:
        SPP/ASPP-Lite output (256, H/32, W/32)
            ↓ Upsample 2x + concat c4
        Block 4 → (256, H/16, W/16)
            ↓ Upsample 2x + concat c3
        Block 3 → (128, H/8, W/8)
            ↓ Upsample 2x + concat c2
        Block 2 → (64, H/4, W/4)
            ↓ Upsample 2x + concat c1
        Block 1 → (32, H/2, W/2)
            ↓ Upsample 2x
        Output → (num_classes, H, W)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    """Standard Convolution + BatchNorm + Activation"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
            
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DecoderBlock(nn.Module):
    """
    Single Decoder Block with skip connection fusion
    
    Structure:
        Upsample (bilinear 2x) → Concat with skip → Conv3x3 → Conv3x3
    """
    
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        
        # After concat: in_channels + skip_channels
        self.conv1 = ConvBNAct(in_channels + skip_channels, out_channels, 3)
        self.conv2 = ConvBNAct(out_channels, out_channels, 3)
    
    def forward(self, x, skip=None):
        """
        Args:
            x: Input features to upsample
            skip: Skip connection features (optional)
            
        Returns:
            Upsampled and fused features
        """
        # Upsample 2x
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        
        # Concatenate with skip connection if provided
        if skip is not None:
            # Ensure spatial dimensions match
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, skip], dim=1)
        
        # Apply convolutions
        x = self.conv1(x)
        x = self.conv2(x)
        
        return x


class Decoder(nn.Module):
    """
    Full Decoder with 4 upsampling blocks
    
    Takes SPP/ASPP-Lite output + encoder skip connections (c1, c2, c3, c4)
    Progressively upsamples to full resolution
    
    Channel progression:
        Input: 256 (from SPP/ASPP-Lite at H/32)
        + c4 (512) → Block4 → 256 (H/16)
        + c3 (256) → Block3 → 128 (H/8)
        + c2 (128) → Block2 → 64 (H/4)
        + c1 (64)  → Block1 → 32 (H/2)
        → Final upsample → (H, W)
    """
    
    def __init__(self, encoder_channels=[64, 128, 256, 512], 
                 aspp_channels=256, num_classes=2):
        super().__init__()
        
        # encoder_channels = [c1, c2, c3, c4] = [64, 128, 256, 512]
        c1_ch, c2_ch, c3_ch, c4_ch = encoder_channels
        
        # Decoder blocks (from bottom to top)
        # Block 4: (256 + 512) → 256, H/32 → H/16
        self.block4 = DecoderBlock(aspp_channels, c4_ch, 256)
        
        # Block 3: (256 + 256) → 128, H/16 → H/8
        self.block3 = DecoderBlock(256, c3_ch, 128)
        
        # Block 2: (128 + 128) → 64, H/8 → H/4
        self.block2 = DecoderBlock(128, c2_ch, 64)
        
        # Block 1: (64 + 64) → 32, H/4 → H/2
        self.block1 = DecoderBlock(64, c1_ch, 32)
        
        # Final convolution before output
        self.final_conv = ConvBNAct(32, 32, 3)
        
        # Segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(32, num_classes, 1),
        )
        
        self.num_classes = num_classes
    
    def forward(self, aspp_out, c1, c2, c3, c4, input_shape=None):
        """
        Args:
            aspp_out: Output from SPP/ASPP-Lite (B, 256, H/32, W/32)
            c1: Encoder feature at H/4 (B, 64, H/4, W/4)
            c2: Encoder feature at H/8 (B, 128, H/8, W/8)
            c3: Encoder feature at H/16 (B, 256, H/16, W/16)
            c4: Encoder feature at H/32 (B, 512, H/32, W/32)
            input_shape: Original input shape (H, W) for final upsampling
            
        Returns:
            Segmentation logits (B, num_classes, H, W)
        """
        # Progressive decoding with skip connections
        x = self.block4(aspp_out, c4)  # (B, 256, H/16, W/16)
        x = self.block3(x, c3)          # (B, 128, H/8, W/8)
        x = self.block2(x, c2)          # (B, 64, H/4, W/4)
        x = self.block1(x, c1)          # (B, 32, H/2, W/2)
        
        # Final processing
        x = self.final_conv(x)          # (B, 32, H/2, W/2)
        
        # Upsample to original resolution
        if input_shape is not None:
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)
        else:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        
        # Segmentation prediction
        out = self.seg_head(x)          # (B, num_classes, H, W)
        
        return out


if __name__ == "__main__":
    # Test decoder
    decoder = Decoder(
        encoder_channels=[64, 128, 256, 512],
        aspp_channels=256,
        num_classes=2
    )
    
    # Simulate inputs for 384×640 image
    aspp_out = torch.randn(2, 256, 12, 20)   # H/32, W/32
    c1 = torch.randn(2, 64, 96, 160)         # H/4, W/4
    c2 = torch.randn(2, 128, 48, 80)         # H/8, W/8
    c3 = torch.randn(2, 256, 24, 40)         # H/16, W/16
    c4 = torch.randn(2, 512, 12, 20)         # H/32, W/32
    
    out = decoder(aspp_out, c1, c2, c3, c4, input_shape=(384, 640))
    
    print("Decoder Output:")
    print(f"  ASPP Input: {aspp_out.shape}")
    print(f"  c1:         {c1.shape}")
    print(f"  c2:         {c2.shape}")
    print(f"  c3:         {c3.shape}")
    print(f"  c4:         {c4.shape}")
    print(f"  Output:     {out.shape}")
    
    # Parameter count
    params = sum(p.numel() for p in decoder.parameters())
    print(f"\n  Total Parameters: {params:,}")
