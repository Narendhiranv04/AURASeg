"""
AURASeg V4 with Configurable ResNet Backbone
=============================================

Supports both ResNet-50 and ResNet-18 encoders for deployment comparison.

Architecture:
    - Encoder: ResNet-50 or ResNet-18 (ImageNet Pretrained)
    - Context Module: ASPP-Lite (4 branches)
    - Decoder: APUD (Attention Progressive Upsampling Decoder)
    - Boundary Refinement: RBRM (Residual Boundary Refinement Module)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


# =============================================================================
# Utility Modules (same as auraseg_v4_resnet.py)
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
# ASPP-Lite Module
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
    ASPP-Lite Module - adaptive for different encoder output channels.
    """
    
    def __init__(self, in_channels: int = 512, out_channels: int = 256, 
                 branch_channels: int = 64):
        super().__init__()
        
        self.branch1 = ASPPLiteConv(in_channels, branch_channels, kernel_size=1, dilation=1)
        self.branch2 = ASPPLiteConv(in_channels, branch_channels, kernel_size=3, dilation=1)
        self.branch3 = ASPPLiteConv(in_channels, branch_channels, kernel_size=3, dilation=6)
        self.branch4 = ASPPLiteConv(in_channels, branch_channels, kernel_size=3, dilation=12)
        
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
    """Attention Progressive Upsampling Decoder Block"""
    
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
        low = self.low_transform(x_low)
        high = self.high_transform(x_high)
        
        low = self.se_attention(low)
        low_up = F.interpolate(low, size=high.shape[2:], mode='bilinear', align_corners=True)
        
        fusion = low_up * high
        spatial = self.spatial_attention(high)
        combined = fusion + spatial
        out = self.refinement(combined)
        
        return out


# =============================================================================
# RBRM Module (Simplified for ONNX export)
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
    
    def forward(self, x: torch.Tensor):
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
        x = F.interpolate(s3, size=s2.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, s2], dim=1)
        x = self.up1(x)
        
        x = F.interpolate(x, size=s1.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, s1], dim=1)
        x = self.up2(x)
        
        x = F.interpolate(x, size=edge_features.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, edge_features], dim=1)
        x = self.up3(x)
        
        return x


class RBRMModule(nn.Module):
    """Residual Boundary Refinement Module"""
    
    def __init__(self, in_channels: int = 256, edge_channels: int = 64):
        super().__init__()
        
        self.boundary_head = BoundaryDetectionHead(in_channels, edge_channels)
        self.boundary_encoder = BoundaryEncoder(edge_channels)
        self.boundary_decoder = BoundaryDecoder(edge_channels)
        
        self.boundary_proj = nn.Sequential(
            ConvBNAct(edge_channels, in_channels, kernel_size=1),
            nn.Conv2d(in_channels, in_channels, 1, bias=False)
        )
        
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        edge_features = self.boundary_head(x)
        s1, s2, s3 = self.boundary_encoder(edge_features)
        boundary_features = self.boundary_decoder(s3, s2, s1, edge_features)
        
        boundary_proj = self.boundary_proj(boundary_features)
        concat = torch.cat([x, boundary_proj], dim=1)
        gate = self.fusion_gate(concat)
        
        refined = x + gate * boundary_proj
        return refined


# =============================================================================
# APUD Decoder (Configurable for different backbones)
# =============================================================================

class APUDDecoder(nn.Module):
    """APUD Decoder - configurable for ResNet-18/50"""
    
    def __init__(self, 
                 encoder_channels: list,
                 neck_channels: int = 256,
                 decoder_channels: int = 256,
                 num_classes: int = 2,
                 se_reduction: int = 16):
        super().__init__()
        
        c1, c2, c3, c4 = encoder_channels
        
        self.apud1 = APUDBlock(neck_channels, c4, decoder_channels, se_reduction)
        self.apud2 = APUDBlock(decoder_channels, c3, decoder_channels, se_reduction)
        self.apud3 = APUDBlock(decoder_channels, c2, decoder_channels, se_reduction)
        self.apud4 = APUDBlock(decoder_channels, c1, decoder_channels, se_reduction)
    
    def forward(self, neck_out: torch.Tensor, encoder_features: list) -> torch.Tensor:
        c1, c2, c3, c4 = encoder_features
        
        out1 = self.apud1(neck_out, c4)
        out2 = self.apud2(out1, c3)
        out3 = self.apud3(out2, c2)
        out4 = self.apud4(out3, c1)
        
        return out4


# =============================================================================
# Main Model: AURASeg V4 with Configurable ResNet Backbone
# =============================================================================

class AURASeg_V4_ResNet(nn.Module):
    """
    AURASeg V4 with configurable ResNet backbone (18 or 50).
    
    Optimized for ONNX/TensorRT export - returns single tensor output.
    """
    
    # Channel configurations for different ResNet variants
    ENCODER_CONFIGS = {
        'resnet18': {
            'encoder_channels': [64, 64, 128, 256],  # layer0, layer1, layer2, layer3
            'context_channels': 512,  # layer4 output
            'aspp_branch_channels': 64,
        },
        'resnet34': {
            'encoder_channels': [64, 64, 128, 256],
            'context_channels': 512,
            'aspp_branch_channels': 64,
        },
        'resnet50': {
            'encoder_channels': [64, 256, 512, 1024],
            'context_channels': 2048,
            'aspp_branch_channels': 128,
        },
    }
    
    def __init__(self, 
                 backbone: str = 'resnet50',
                 num_classes: int = 2, 
                 decoder_channels: int = 256,
                 encoder_weights: str = 'imagenet'):
        super().__init__()
        
        self.backbone = backbone
        self.num_classes = num_classes
        
        if backbone not in self.ENCODER_CONFIGS:
            raise ValueError(f"Unsupported backbone: {backbone}. Choose from {list(self.ENCODER_CONFIGS.keys())}")
        
        config = self.ENCODER_CONFIGS[backbone]
        
        # Encoder
        self.encoder = smp.encoders.get_encoder(
            name=backbone,
            in_channels=3,
            depth=5,
            weights=encoder_weights
        )
        
        # ASPP-Lite
        self.aspp_lite = ASPPLite(
            in_channels=config['context_channels'],
            out_channels=256,
            branch_channels=config['aspp_branch_channels']
        )
        
        # APUD Decoder
        self.decoder = APUDDecoder(
            encoder_channels=config['encoder_channels'],
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
        
        self._init_decoder_weights()
    
    def _init_decoder_weights(self):
        """Initialize decoder weights"""
        for module in [self.aspp_lite, self.decoder, self.rbrm, self.seg_head]:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass optimized for inference/export.
        Returns single tensor: (B, num_classes, H, W)
        """
        input_size = x.shape[2:]
        
        # Encoder
        features = self.encoder(x)
        
        # For ResNet-18: features[1-5] = [64, 64, 128, 256, 512]
        # For ResNet-50: features[1-5] = [64, 256, 512, 1024, 2048]
        
        c1 = features[1]
        c2 = features[2]
        c3 = features[3]
        c4 = features[4]
        c5 = features[5]
        
        # ASPP-Lite
        context = self.aspp_lite(c5)
        
        # APUD Decoder
        encoder_features = [c1, c2, c3, c4]
        decoder_features = self.decoder(context, encoder_features)
        
        # RBRM
        refined_features = self.rbrm(decoder_features)
        
        # Segmentation Head
        main_out = self.seg_head(refined_features)
        main_out = F.interpolate(main_out, size=input_size, mode='bilinear', align_corners=True)
        
        return main_out


# =============================================================================
# Factory Functions
# =============================================================================

def auraseg_resnet18(num_classes: int = 2, pretrained: bool = True):
    """Create AURASeg with ResNet-18 backbone."""
    return AURASeg_V4_ResNet(
        backbone='resnet18',
        num_classes=num_classes,
        decoder_channels=256,
        encoder_weights='imagenet' if pretrained else None
    )


def auraseg_resnet50(num_classes: int = 2, pretrained: bool = True):
    """Create AURASeg with ResNet-50 backbone."""
    return AURASeg_V4_ResNet(
        backbone='resnet50',
        num_classes=num_classes,
        decoder_channels=256,
        encoder_weights='imagenet' if pretrained else None
    )


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("AURASeg V4 - Multi-Backbone Test")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    for backbone in ['resnet18', 'resnet50']:
        print(f"\n--- Testing {backbone} ---")
        
        model = AURASeg_V4_ResNet(backbone=backbone, num_classes=2).to(device)
        model.eval()
        
        x = torch.randn(1, 3, 384, 640).to(device)
        
        with torch.no_grad():
            out = model(x)
        
        total_params = sum(p.numel() for p in model.parameters()) / 1e6
        
        print(f"  Input: {x.shape}")
        print(f"  Output: {out.shape}")
        print(f"  Params: {total_params:.2f}M")
        
        del model
        torch.cuda.empty_cache()
    
    print("\n" + "=" * 70)
    print("All tests passed!")
