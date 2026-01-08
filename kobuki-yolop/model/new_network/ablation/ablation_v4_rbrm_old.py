"""
Ablation V4: RBRM (Residual Boundary Refinement Module)

Building on V3 (CSPDarknet + ASPPLite + APUD), this version adds the RBRM
for boundary-aware refinement as specified in the paper.

RBRM Architecture (from paper):
The Residual Boundary Refinement Module (RBRM) enhances segmentation by:
1. Taking low-level features (rich in spatial/boundary details)
2. Taking high-level features (rich in semantic context)
3. Fusing them through concatenation
4. Applying Spatial Attention for boundary awareness
5. Using Laplacian filtering to extract edge/boundary information
6. Combining boundary cues with feature maps for refined output

This is the FULL MODEL with all components:
- Encoder: CSPDarknet
- Neck: ASPPLite  
- Decoder: APUD (SE + Spatial Attention)
- Refinement: RBRM (Boundary-aware refinement)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import numpy as np


# =============================================================================
# Encoder: CSPDarknet (same as V1, V2, V3)
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
# Neck: ASPPLite (same as V2, V3)
# =============================================================================

class ASPPLite(nn.Module):
    """Lightweight ASPP with depthwise separable convolutions"""
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
        for d in dilations[1:]:
            self.dilated_branches.append(nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=d, dilation=d, 
                          groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True)
            ))
        
        # Fusion
        self.fusion = ConvBNAct(out_channels * 5, out_channels, 1)
        self.dropout = nn.Dropout2d(0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2:]
        gap = F.interpolate(self.gap(x), size=(h, w), mode='bilinear', align_corners=False)
        conv1x1 = self.conv1x1(x)
        dilated_outs = [branch(x) for branch in self.dilated_branches]
        out = torch.cat([gap, conv1x1] + dilated_outs, dim=1)
        out = self.fusion(out)
        out = self.dropout(out)
        return out


# =============================================================================
# APUD Components: SE Attention and Spatial Attention (same as V3)
# =============================================================================

class SEAttention(nn.Module):
    """Squeeze-and-Excitation Attention Module"""
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
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    """Spatial Attention Module - Max/Avg pool + 7x7 Conv"""
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        return x * self.sigmoid(y)


class APUDBlock(nn.Module):
    """Attention Progressive Upsampling Decoder Block"""
    def __init__(self, high_ch: int, low_ch: int, out_ch: int, se_reduction: int = 16):
        super().__init__()
        self.transform_high = ConvBNAct(high_ch, out_ch, 1)
        self.transform_low = ConvBNAct(low_ch, out_ch, 1)
        self.se_attention = SEAttention(out_ch, reduction=se_reduction)
        self.spatial_attention = SpatialAttention(kernel_size=7)
        self.refinement = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True)
        )
    
    def forward(self, x_high: torch.Tensor, x_low: torch.Tensor) -> torch.Tensor:
        high_transformed = self.transform_high(x_high)
        low_transformed = self.transform_low(x_low)
        high_se = self.se_attention(high_transformed)
        high_upsampled = F.interpolate(
            high_se, size=low_transformed.shape[2:], 
            mode='bilinear', align_corners=False
        )
        fused = high_upsampled * low_transformed
        low_spatial = self.spatial_attention(low_transformed)
        out = fused + low_spatial
        out = self.refinement(out)
        return out


class APUDDecoder(nn.Module):
    """Attention Progressive Upsampling Decoder"""
    def __init__(self, encoder_channels: List[int], neck_channels: int, 
                 decoder_channels: int = 256, se_reduction: int = 16):
        super().__init__()
        
        self.apud1 = APUDBlock(neck_channels, encoder_channels[3], decoder_channels, se_reduction)
        self.apud2 = APUDBlock(decoder_channels, encoder_channels[2], decoder_channels, se_reduction)
        self.apud3 = APUDBlock(decoder_channels, encoder_channels[1], decoder_channels, se_reduction)
        self.apud4 = APUDBlock(decoder_channels, encoder_channels[0], decoder_channels, se_reduction)
        
        self.out_channels = decoder_channels
    
    def forward(self, encoder_features: List[torch.Tensor], neck_feature: torch.Tensor) -> torch.Tensor:
        c2, c3, c4, c5 = encoder_features
        x = self.apud1(neck_feature, c5)  # 1/32
        x = self.apud2(x, c4)              # 1/16
        x = self.apud3(x, c3)              # 1/8
        x = self.apud4(x, c2)              # 1/4
        return x


# =============================================================================
# RBRM: Residual Boundary Refinement Module (NEW in V4)
# =============================================================================

class RBRM(nn.Module):
    """
    Residual Boundary Refinement Module
    
    As described in the paper:
    - Takes low-level features (rich in boundary/spatial details) and
      high-level features (rich in semantic context)
    - Fuses them through concatenation and convolution
    - Applies Spatial Attention for boundary awareness
    - Uses Laplacian kernel to extract edge/boundary information
    - Combines boundary cues with feature maps via residual connection
    
    The Laplacian filter enhances edge information, helping the network
    focus on boundary regions for improved segmentation accuracy.
    """
    def __init__(self, in_channels: int, out_channels: int, num_classes: int = 2):
        super().__init__()
        
        # Feature fusion: concatenate low + high, then reduce channels
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Spatial attention for boundary awareness
        self.spatial_attention = SpatialAttention(kernel_size=7)
        
        # Pre-prediction for Laplacian extraction
        self.pre_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 2, num_classes, 1)
        )
        
        # Laplacian kernel (fixed, not learnable)
        # This is an edge detection kernel
        laplacian_kernel = torch.tensor([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('laplacian_kernel', laplacian_kernel)
        
        # Output refinement
        self.refine_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.out_channels = out_channels
    
    def forward(self, low_features: torch.Tensor, high_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            low_features: Low-level features (larger spatial size, more boundary detail)
            high_features: High-level features (smaller spatial size, more semantic info)
        
        Returns:
            refined_features: Boundary-refined features
            boundary_logits: Intermediate prediction (for deep supervision or visualization)
        """
        # Upsample high features to match low features
        h, w = low_features.shape[2:]
        high_up = F.interpolate(high_features, size=(h, w), mode='bilinear', align_corners=False)
        
        # Concatenate and fuse
        fused = torch.cat([low_features, high_up], dim=1)
        fused = self.fusion_conv(fused)
        
        # Apply spatial attention for boundary awareness
        fused_sa = self.spatial_attention(fused)
        
        # Get pre-prediction for Laplacian extraction
        pre_logits = self.pre_conv(fused)
        
        # Extract boundary information using Laplacian
        # Take max across classes to get boundary response
        pre_max = pre_logits.max(dim=1, keepdim=True)[0]
        
        # Apply Laplacian filter - need to handle properly for variable devices
        boundary_response = F.conv2d(
            pre_max, 
            self.laplacian_kernel.to(pre_max.dtype), 
            padding=1
        )
        
        # Normalize boundary response to [0, 1] range
        boundary_response = torch.sigmoid(boundary_response)
        
        # Combine: original features + spatial attention + boundary cues
        # Boundary response is broadcast across channels
        refined = fused + fused_sa + fused * boundary_response
        
        # Final refinement
        refined = self.refine_conv(refined)
        
        return refined, pre_logits


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
# Complete V4 Model: CSPDarknet + ASPPLite + APUD + RBRM
# =============================================================================

class AblationV4RBRM(nn.Module):
    """
    Ablation Study V4: RBRM (Residual Boundary Refinement Module)
    
    This is the FULL MODEL with all components:
    - Encoder: CSPDarknet
    - Neck: ASPPLite
    - Decoder: APUD (SE + Spatial Attention)
    - Refinement: RBRM (Boundary-aware refinement) [NEW]
    
    The RBRM takes the decoder output and refines it using:
    1. Fusion with encoder features (low-level boundary details)
    2. Spatial attention for boundary awareness
    3. Laplacian filtering for edge detection
    4. Residual combination for refined output
    """
    def __init__(self, num_classes: int = 2, in_channels: int = 3,
                 base_channels: int = 64, base_depth: int = 3, 
                 decoder_channels: int = 256, aspp_out_channels: int = 256, 
                 se_reduction: int = 16, deep_supervision: bool = False):
        super().__init__()
        
        self.deep_supervision = deep_supervision
        self.num_classes = num_classes
        
        # Encoder
        self.encoder = CSPDarknet(base_channels, base_depth)
        
        # Neck: ASPPLite
        self.neck = ASPPLite(
            in_channels=self.encoder.out_channels[-1],
            out_channels=aspp_out_channels
        )
        
        # Decoder: APUD
        self.decoder = APUDDecoder(
            encoder_channels=self.encoder.out_channels,
            neck_channels=aspp_out_channels,
            decoder_channels=decoder_channels,
            se_reduction=se_reduction
        )
        
        # RBRM: Takes decoder output + encoder low-level features
        # Low features (c2): 128 channels, High features (decoder): 256 channels
        self.rbrm = RBRM(
            in_channels=self.encoder.out_channels[0] + decoder_channels,  # 128 + 256 = 384
            out_channels=decoder_channels,
            num_classes=num_classes
        )
        
        # Final segmentation head
        self.seg_head = SegmentationHead(
            in_channels=decoder_channels,
            num_classes=num_classes,
            scale_factor=4  # 1/4 -> 1/1
        )
        
        # Deep supervision heads (optional)
        if deep_supervision:
            self.aux_head1 = nn.Conv2d(decoder_channels, num_classes, 1)  # From APUD output
            self.aux_head2 = nn.Conv2d(decoder_channels, num_classes, 1)  # From RBRM intermediate
        
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
    
    def forward(self, x: torch.Tensor):
        input_size = x.shape[2:]
        
        # Encoder
        encoder_features = self.encoder(x)  # [c2, c3, c4, c5]
        c2 = encoder_features[0]  # 1/4 resolution, rich in boundary details
        
        # Neck
        neck_out = self.neck(encoder_features[-1])  # 1/32
        
        # Decoder (APUD)
        decoder_out = self.decoder(encoder_features, neck_out)  # 1/4
        
        # RBRM: Refine with boundary information
        refined, boundary_logits = self.rbrm(c2, decoder_out)
        
        # Final prediction
        out = self.seg_head(refined)  # 1/1
        
        if self.training and self.deep_supervision:
            # Auxiliary outputs for deep supervision
            aux1 = self.aux_head1(decoder_out)
            aux1 = F.interpolate(aux1, size=input_size, mode='bilinear', align_corners=False)
            
            aux2 = F.interpolate(boundary_logits, size=input_size, mode='bilinear', align_corners=False)
            
            return out, aux1, aux2
        
        return out


def create_model(num_classes: int = 2, **kwargs) -> AblationV4RBRM:
    """Factory function to create V4 RBRM model"""
    return AblationV4RBRM(num_classes=num_classes, **kwargs)


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    # Test the model
    model = AblationV4RBRM(num_classes=2)
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"V4 RBRM Model Architecture (FULL MODEL)")
    print(f"=" * 60)
    print(f"Encoder: CSPDarknet")
    print(f"Neck: ASPPLite")
    print(f"Decoder: APUD (SE Attention + Spatial Attention)")
    print(f"Refinement: RBRM (Residual Boundary Refinement)")
    print(f"=" * 60)
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
    
    # Test forward pass
    x = torch.randn(1, 3, 384, 640)
    with torch.no_grad():
        y = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    # Test with deep supervision
    print(f"\nWith Deep Supervision:")
    model_ds = AblationV4RBRM(num_classes=2, deep_supervision=True)
    model_ds.train()
    with torch.no_grad():
        outputs = model_ds(x)
    print(f"  Main output: {outputs[0].shape}")
    print(f"  Aux output 1: {outputs[1].shape}")
    print(f"  Aux output 2: {outputs[2].shape}")
    
    print(f"\nRBRM Components:")
    print(f"  - Feature fusion (concat low + high)")
    print(f"  - Spatial Attention for boundary awareness")
    print(f"  - Laplacian kernel for edge extraction")
    print(f"  - Residual combination for refinement")
