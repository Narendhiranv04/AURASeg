"""
AURASeg V4 with ResNet-50 Encoder
==================================

Final model for RAL paper benchmark comparison.

Architecture:
    - Encoder: ResNet-50 (ImageNet Pretrained) from segmentation_models_pytorch
    - Context Module: ASPP-Lite (4 branches, adapted for 2048 input channels)
    - Decoder: APUD (Attention Progressive Upsampling Decoder) with deep supervision
    - Boundary Refinement: RBRM (Residual Boundary Refinement Module)

Key Changes from Original V4 (CSPDarknet):
    - Encoder: CSPDarknet-53 (scratch) → ResNet-50 (ImageNet pretrained)
    - ASPP-Lite input: 1024 → 2048 channels
    - Skip connections: [64, 128, 256, 512] → [64, 256, 512, 1024]

Paper: AURASeg - Attention Guided Upsampling with Residual Boundary-Assistive Refinement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


# =============================================================================
# Utility Modules
# =============================================================================

class ConvBNAct(nn.Module):
    """Standard Convolution + BatchNorm + Activation"""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, stride: int = 1, padding: int = None,
                 activation: str = 'silu'):
        super().__init__()
        
        if padding is None:
            padding = kernel_size // 2
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                              stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        
        if activation == 'silu':
            self.act = nn.SiLU(inplace=True)
        elif activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


# =============================================================================
# ASPP-Lite Module (Adapted for ResNet-50)
# =============================================================================

class ASPPLiteConv(nn.Module):
    """Single ASPP-Lite convolution branch"""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int, dilation: int):
        super().__init__()
        
        padding = 0 if kernel_size == 1 else dilation
        
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class ASPPLite(nn.Module):
    """
    ASPP-Lite Module adapted for ResNet-50 encoder.
    
    4 Branches with branch_channels filters each:
        - Branch 1: 1×1 conv, dilation=1
        - Branch 2: 3×3 conv, dilation=1
        - Branch 3: 3×3 conv, dilation=6
        - Branch 4: 3×3 conv, dilation=12
    
    Input: (B, 2048, H/32, W/32) from ResNet-50 layer4
    Output: (B, 256, H/32, W/32)
    """
    
    def __init__(self, in_channels: int = 2048, out_channels: int = 256, 
                 branch_channels: int = 128):
        super().__init__()
        
        # 4 parallel branches as per paper
        self.branch1 = ASPPLiteConv(in_channels, branch_channels, kernel_size=1, dilation=1)
        self.branch2 = ASPPLiteConv(in_channels, branch_channels, kernel_size=3, dilation=1)
        self.branch3 = ASPPLiteConv(in_channels, branch_channels, kernel_size=3, dilation=6)
        self.branch4 = ASPPLiteConv(in_channels, branch_channels, kernel_size=3, dilation=12)
        
        # Fusion: 4 branches × branch_channels = 4 × 128 = 512 channels
        self.fusion = nn.Sequential(
            nn.Conv2d(branch_channels * 4, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )
        
        self.dropout = nn.Dropout2d(0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        
        concat = torch.cat([b1, b2, b3, b4], dim=1)
        out = self.fusion(concat)
        out = self.dropout(out)
        
        return out


# =============================================================================
# Attention Modules
# =============================================================================

class SEAttention(nn.Module):
    """Squeeze-and-Excitation Channel Attention"""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        reduced_channels = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    """Spatial Attention Module"""
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        return x * y


# =============================================================================
# APUD Block
# =============================================================================

class APUDBlock(nn.Module):
    """
    Attention Progressive Upsampling Decoder Block
    
    x_low (deeper, smaller) → 1×1 → SE → Upsample
    x_high (shallower, larger) → 1×1
    
    Fusion: upsample(SE(x_low_transformed)) ⊗ x_high_transformed + Spatial(x_high)
    Output: Refinement(Fusion)
    """
    
    def __init__(self, low_channels: int, high_channels: int, out_channels: int,
                 se_reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        
        self.low_transform = ConvBNAct(low_channels, out_channels, kernel_size=1)
        self.high_transform = ConvBNAct(high_channels, out_channels, kernel_size=1)
        
        self.se_attention = SEAttention(out_channels, reduction=se_reduction)
        self.spatial_attention = SpatialAttention(kernel_size=spatial_kernel)
        
        self.refinement = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )
    
    def forward(self, x_low: torch.Tensor, x_high: torch.Tensor) -> torch.Tensor:
        # Transform channels
        low = self.low_transform(x_low)
        high = self.high_transform(x_high)
        
        # Apply SE attention to low-res features
        low = self.se_attention(low)
        
        # Upsample low to match high resolution
        low_up = F.interpolate(low, size=high.shape[2:], mode='bilinear', align_corners=True)
        
        # Fusion: element-wise multiply
        fusion = low_up * high
        
        # Apply spatial attention to high-res features
        spatial = self.spatial_attention(high)
        
        # Combine
        combined = fusion + spatial
        
        # Refinement
        out = self.refinement(combined)
        
        return out


# =============================================================================
# RBRM Module (Residual Boundary Refinement)
# =============================================================================

class BoundaryDetectionHead(nn.Module):
    """Boundary Detection using Sobel Operators"""
    
    def __init__(self, in_channels: int = 256, out_channels: int = 64):
        super().__init__()
        
        self.proj = ConvBNAct(in_channels, out_channels, kernel_size=1)
        
        # Sobel kernels (fixed)
        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
        sobel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
        self.register_buffer('sobel_x', sobel_x.repeat(out_channels, 1, 1, 1))
        self.register_buffer('sobel_y', sobel_y.repeat(out_channels, 1, 1, 1))
        
        self.fusion = ConvBNAct(out_channels * 2, out_channels, kernel_size=3)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        
        edge_x = F.conv2d(x, self.sobel_x, padding=1, groups=x.shape[1])
        edge_y = F.conv2d(x, self.sobel_y, padding=1, groups=x.shape[1])
        
        edges = torch.cat([edge_x, edge_y], dim=1)
        edge_features = self.fusion(edges)
        
        return edge_features


class BoundaryEncoder(nn.Module):
    """Lightweight Boundary Encoder (3 stages)"""
    
    def __init__(self, in_channels: int = 64):
        super().__init__()
        
        self.stage1 = nn.Sequential(
            ConvBNAct(in_channels, 128, kernel_size=3, stride=2),
            ConvBNAct(128, 128, kernel_size=3, stride=1)
        )
        
        self.stage2 = nn.Sequential(
            ConvBNAct(128, 256, kernel_size=3, stride=2),
            ConvBNAct(256, 256, kernel_size=3, stride=1)
        )
        
        self.stage3 = nn.Sequential(
            ConvBNAct(256, 512, kernel_size=3, stride=2),
            ConvBNAct(512, 512, kernel_size=3, stride=1)
        )
    
    def forward(self, x: torch.Tensor) -> tuple:
        s1 = self.stage1(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        return s1, s2, s3


class BoundaryDecoder(nn.Module):
    """Boundary Decoder with Skip Connections"""
    
    def __init__(self, out_channels: int = 64):
        super().__init__()
        
        self.up1 = nn.Sequential(
            ConvBNAct(512 + 256, 256, kernel_size=3),
            ConvBNAct(256, 256, kernel_size=3)
        )
        
        self.up2 = nn.Sequential(
            ConvBNAct(256 + 128, 128, kernel_size=3),
            ConvBNAct(128, 128, kernel_size=3)
        )
        
        self.up3 = nn.Sequential(
            ConvBNAct(128 + out_channels, out_channels, kernel_size=3),
            ConvBNAct(out_channels, out_channels, kernel_size=3)
        )
    
    def forward(self, s3: torch.Tensor, s2: torch.Tensor, 
                s1: torch.Tensor, edge_features: torch.Tensor) -> torch.Tensor:
        # Up-1: s3 (512) + s2 (256 skip) → 256
        x = F.interpolate(s3, size=s2.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, s2], dim=1)
        x = self.up1(x)
        
        # Up-2: (256) + s1 (128 skip) → 128
        x = F.interpolate(x, size=s1.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, s1], dim=1)
        x = self.up2(x)
        
        # Up-3: (128) + edge_features (64 skip) → 64
        x = F.interpolate(x, size=edge_features.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, edge_features], dim=1)
        x = self.up3(x)
        
        return x


class RBRMModule(nn.Module):
    """
    Residual Boundary Refinement Module
    
    F_out = F_main + sigmoid(gate) × F_boundary
    """
    
    def __init__(self, in_channels: int = 256, edge_channels: int = 64):
        super().__init__()
        
        self.boundary_head = BoundaryDetectionHead(in_channels, edge_channels)
        self.boundary_encoder = BoundaryEncoder(edge_channels)
        self.boundary_decoder = BoundaryDecoder(edge_channels)
        
        # Project boundary features to main feature channels
        self.boundary_proj = nn.Sequential(
            ConvBNAct(edge_channels, in_channels, kernel_size=1),
            nn.Conv2d(in_channels, in_channels, 1, bias=False)
        )
        
        # Learned fusion gate
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )
        
        # Boundary prediction head
        self.boundary_pred = nn.Conv2d(edge_channels, 1, 1)
    
    def forward(self, x: torch.Tensor, return_boundary: bool = True) -> dict:
        # Edge detection
        edge_features = self.boundary_head(x)
        
        # Boundary encoder
        s1, s2, s3 = self.boundary_encoder(edge_features)
        
        # Boundary decoder
        boundary_features = self.boundary_decoder(s3, s2, s1, edge_features)
        
        # Project to main feature channels
        boundary_proj = self.boundary_proj(boundary_features)
        
        # Learned gate
        concat = torch.cat([x, boundary_proj], dim=1)
        gate = self.fusion_gate(concat)
        
        # Residual fusion
        refined = x + gate * boundary_proj
        
        result = {'features': refined}
        
        if return_boundary:
            boundary_pred = self.boundary_pred(boundary_features)
            # Upsample to full resolution will be done in main model
            result['boundary'] = boundary_pred
        
        return result


# =============================================================================
# APUD Decoder
# =============================================================================

class APUDDecoder(nn.Module):
    """
    APUD Decoder for ResNet-50 encoder.
    
    ResNet-50 feature channels: [64, 256, 512, 1024, 2048]
    Skip connections use: [64, 256, 512, 1024] from layers 0-3
    ASPP-Lite input: 2048 from layer4
    """
    
    def __init__(self, 
                 encoder_channels: list = [64, 256, 512, 1024],
                 neck_channels: int = 256,
                 decoder_channels: int = 256,
                 num_classes: int = 2,
                 se_reduction: int = 16):
        super().__init__()
        
        c1, c2, c3, c4 = encoder_channels
        
        # APUD blocks (4 levels)
        # APUD-1: ASPP(256) + c4(1024) → 256 @ H/16
        self.apud1 = APUDBlock(neck_channels, c4, decoder_channels, se_reduction)
        # APUD-2: APUD1(256) + c3(512) → 256 @ H/8
        self.apud2 = APUDBlock(decoder_channels, c3, decoder_channels, se_reduction)
        # APUD-3: APUD2(256) + c2(256) → 256 @ H/4
        self.apud3 = APUDBlock(decoder_channels, c2, decoder_channels, se_reduction)
        # APUD-4: APUD3(256) + c1(64) → 256 @ H/4
        self.apud4 = APUDBlock(decoder_channels, c1, decoder_channels, se_reduction)
        
        # Auxiliary supervision heads
        self.aux_head1 = self._make_aux_head(decoder_channels, num_classes)
        self.aux_head2 = self._make_aux_head(decoder_channels, num_classes)
        self.aux_head3 = self._make_aux_head(decoder_channels, num_classes)
        self.aux_head4 = self._make_aux_head(decoder_channels, num_classes)
    
    def _make_aux_head(self, in_channels: int, num_classes: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels // 4, num_classes, 1)
        )
    
    def forward(self, neck_out: torch.Tensor, 
                encoder_features: list,
                return_aux: bool = True) -> dict:
        c1, c2, c3, c4 = encoder_features
        
        # APUD-1: neck(256@H/32) + c4(1024@H/16) → out1(256@H/16)
        out1 = self.apud1(neck_out, c4)
        
        # APUD-2: out1(256@H/16) + c3(512@H/8) → out2(256@H/8)
        out2 = self.apud2(out1, c3)
        
        # APUD-3: out2(256@H/8) + c2(256@H/4) → out3(256@H/4)
        out3 = self.apud3(out2, c2)
        
        # APUD-4: out3(256@H/4) + c1(64@H/4) → out4(256@H/4)
        out4 = self.apud4(out3, c1)
        
        result = {'decoder_features': out4}
        
        if return_aux:
            aux1 = self.aux_head1(out1)
            aux2 = self.aux_head2(out2)
            aux3 = self.aux_head3(out3)
            aux4 = self.aux_head4(out4)
            result['aux'] = [aux1, aux2, aux3, aux4]
        
        return result


# =============================================================================
# Main Model: AURASeg V4 with ResNet-50
# =============================================================================

class AURASeg_V4_ResNet50(nn.Module):
    """
    AURASeg V4 with ResNet-50 Encoder
    
    Complete Architecture:
        Input (3, H, W)
            ↓
        ResNet-50 Encoder (ImageNet Pretrained)
            ├─ layer0 (64, H/4)   → c1 (skip)
            ├─ layer1 (256, H/4)  → (not used, same res as layer0)
            ├─ layer2 (512, H/8)  → c2 (skip)
            ├─ layer3 (1024, H/16) → c3 (skip)
            └─ layer4 (2048, H/32) → context input
                ↓
        ASPP-Lite (2048 → 256)
                ↓
        APUD Decoder (4 blocks)
            → decoder_features (256, H/4)
                ↓
        RBRM (Residual Boundary Refinement)
            → refined_features (256, H/4)
            → boundary_pred (1, H)
                ↓
        Seg Head → Main Output (num_classes, H)
    
    Args:
        num_classes: Number of segmentation classes (default: 2)
        decoder_channels: Decoder feature channels (default: 256)
        encoder_weights: Pretrained weights for encoder (default: 'imagenet')
    """
    
    def __init__(self, num_classes: int = 2, decoder_channels: int = 256,
                 encoder_weights: str = 'imagenet'):
        super().__init__()
        
        self.num_classes = num_classes
        
        # ResNet-50 Encoder from segmentation_models_pytorch
        # This provides features at multiple scales
        self.encoder = smp.encoders.get_encoder(
            name="resnet50",
            in_channels=3,
            depth=5,
            weights=encoder_weights
        )
        
        # ResNet-50 output channels: [3, 64, 256, 512, 1024, 2048]
        # Index 0 is input, indices 1-5 are feature levels
        # We use: c1=64 (H/4), c2=256 (H/4), c3=512 (H/8), c4=1024 (H/16), c5=2048 (H/32)
        
        # ASPP-Lite (input from layer4 = 2048 channels)
        self.aspp_lite = ASPPLite(
            in_channels=2048,
            out_channels=256,
            branch_channels=128
        )
        
        # APUD Decoder
        # ResNet-50 skip channels: layer0=64, layer1=256, layer2=512, layer3=1024
        self.decoder = APUDDecoder(
            encoder_channels=[64, 256, 512, 1024],
            neck_channels=256,
            decoder_channels=decoder_channels,
            num_classes=num_classes,
            se_reduction=16
        )
        
        # RBRM Module
        self.rbrm = RBRMModule(
            in_channels=decoder_channels,
            edge_channels=64
        )
        
        # Final Segmentation Head
        self.seg_head = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels // 2),
            nn.SiLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(decoder_channels // 2, num_classes, 1)
        )
        
        # Initialize decoder weights (encoder is pretrained)
        self._init_decoder_weights()
    
    def _init_decoder_weights(self):
        """Initialize decoder weights (encoder uses pretrained)"""
        for module in [self.aspp_lite, self.decoder, self.rbrm, self.seg_head]:
            for m in module.modules():
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
    
    def forward(self, x: torch.Tensor, return_aux: bool = True,
                return_boundary: bool = True) -> dict:
        """
        Args:
            x: Input image (B, 3, H, W)
            return_aux: Whether to return auxiliary outputs (training)
            return_boundary: Whether to return boundary prediction (training)
            
        Returns:
            Dictionary with:
                - 'main': Main segmentation (B, num_classes, H, W)
                - 'aux': List of 4 auxiliary outputs (if return_aux)
                - 'boundary': Boundary prediction (B, 1, H, W) (if return_boundary)
        """
        input_size = x.shape[2:]  # (H, W)
        
        # Encoder: Extract multi-scale features
        # Returns list: [input, layer0, layer1, layer2, layer3, layer4]
        features = self.encoder(x)
        
        # features[0] = input (3, H, W) - not used
        # features[1] = layer0 (64, H/4)  → c1
        # features[2] = layer1 (256, H/4) → c2 (same res as c1, but more channels)
        # features[3] = layer2 (512, H/8) → c3
        # features[4] = layer3 (1024, H/16) → c4
        # features[5] = layer4 (2048, H/32) → context
        
        c1 = features[1]  # (64, H/4)
        c2 = features[2]  # (256, H/4)
        c3 = features[3]  # (512, H/8)
        c4 = features[4]  # (1024, H/16)
        c5 = features[5]  # (2048, H/32)
        
        # ASPP-Lite on deepest features
        context = self.aspp_lite(c5)  # (256, H/32)
        
        # APUD Decoder
        encoder_features = [c1, c2, c3, c4]
        decoder_out = self.decoder(context, encoder_features, return_aux=return_aux)
        
        decoder_features = decoder_out['decoder_features']  # (256, H/4)
        
        # RBRM: Boundary Refinement
        rbrm_out = self.rbrm(decoder_features, return_boundary=return_boundary)
        refined_features = rbrm_out['features']  # (256, H/4)
        
        # Final Segmentation Head
        main_out = self.seg_head(refined_features)  # (num_classes, H/4)
        main_out = F.interpolate(main_out, size=input_size, mode='bilinear', align_corners=True)
        
        result = {'main': main_out}
        
        if return_aux and 'aux' in decoder_out:
            result['aux'] = decoder_out['aux']
        
        if return_boundary and 'boundary' in rbrm_out:
            # Upsample boundary to full resolution
            boundary = F.interpolate(rbrm_out['boundary'], size=input_size, 
                                      mode='bilinear', align_corners=True)
            result['boundary'] = boundary
        
        return result
    
    def get_param_groups(self, lr_encoder: float = 1e-4, lr_decoder: float = 1e-3):
        """
        Get parameter groups with differential learning rates.
        
        Args:
            lr_encoder: LR for pretrained ResNet-50 encoder
            lr_decoder: LR for decoder modules (ASPP, APUD, RBRM, seg_head)
        """
        encoder_params = list(self.encoder.parameters())
        decoder_params = (
            list(self.aspp_lite.parameters()) +
            list(self.decoder.parameters()) +
            list(self.rbrm.parameters()) +
            list(self.seg_head.parameters())
        )
        
        return [
            {'params': encoder_params, 'lr': lr_encoder, 'name': 'encoder'},
            {'params': decoder_params, 'lr': lr_decoder, 'name': 'decoder'}
        ]


# =============================================================================
# Factory Function
# =============================================================================

def auraseg_v4_resnet50(num_classes: int = 2, pretrained_encoder: bool = True):
    """
    Factory function to create AURASeg V4 with ResNet-50 encoder.
    
    Args:
        num_classes: Number of segmentation classes
        pretrained_encoder: Whether to use ImageNet pretrained encoder
    """
    encoder_weights = 'imagenet' if pretrained_encoder else None
    return AURASeg_V4_ResNet50(
        num_classes=num_classes,
        decoder_channels=256,
        encoder_weights=encoder_weights
    )


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("AURASeg V4 with ResNet-50 Encoder")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create model
    model = auraseg_v4_resnet50(num_classes=2, pretrained_encoder=True).to(device)
    
    # Test input
    x = torch.randn(2, 3, 384, 640).to(device)
    print(f"\nInput shape: {x.shape}")
    
    # Forward pass (training mode)
    print("\n1. Training mode (with aux and boundary):")
    model.train()
    with torch.no_grad():
        outputs = model(x, return_aux=True, return_boundary=True)
    
    print(f"   Main output: {outputs['main'].shape}")
    print(f"   Boundary output: {outputs['boundary'].shape}")
    print("   Auxiliary outputs:")
    for i, aux in enumerate(outputs['aux']):
        print(f"     Aux-{i+1}: {aux.shape}")
    
    # Forward pass (inference mode)
    print("\n2. Inference mode:")
    model.eval()
    with torch.no_grad():
        outputs = model(x, return_aux=False, return_boundary=False)
    
    print(f"   Main output: {outputs['main'].shape}")
    
    # Parameter counts
    print("\n3. Parameter Counts:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    aspp_params = sum(p.numel() for p in model.aspp_lite.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    rbrm_params = sum(p.numel() for p in model.rbrm.parameters())
    seg_head_params = sum(p.numel() for p in model.seg_head.parameters())
    
    print(f"   Encoder (ResNet-50): {encoder_params:,}")
    print(f"   ASPP-Lite: {aspp_params:,}")
    print(f"   APUD Decoder: {decoder_params:,}")
    print(f"   RBRM: {rbrm_params:,}")
    print(f"   Seg Head: {seg_head_params:,}")
    print(f"   ─────────────────────────────")
    print(f"   Total: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"   Trainable: {trainable_params:,}")
    
    # Test differential learning rates
    print("\n4. Differential Learning Rates:")
    param_groups = model.get_param_groups(lr_encoder=1e-4, lr_decoder=1e-3)
    for group in param_groups:
        n_params = sum(p.numel() for p in group['params'])
        print(f"   {group['name']}: {n_params:,} params @ lr={group['lr']}")
    
    print("\n" + "=" * 70)
    print("Model ready for training!")
    print("=" * 70)
