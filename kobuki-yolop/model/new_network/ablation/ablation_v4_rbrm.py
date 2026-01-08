"""
Ablation V4: CSPDarknet + ASPPLite + APUD + Deep Guided Filter RBRM
===================================================================

This is the fourth ablation model that adds the Residual Boundary 
Refinement Module (RBRM) on top of V3.

Architecture:
- Encoder: CSPDarknet (same as V1, V2, V3)
- Neck: ASPPLite (same as V2, V3)
- Decoder: APUD with SE + Spatial Attention (same as V3)
- RBRM: Deep Guided Filter (DGF) for edge-preserving upsampling (NEW in V4)

RBRM Architecture (Deep Guided Filter):
    Instead of standard upsampling or conv-based refinement, we use
    Deep Guided Filtering. The high-res RGB image guides the upsampling
    of the low-res segmentation features.
    
    Output = A * Guidance + b
    
    Where A and b are coefficients computed from the low-res features
    and downsampled guidance, then upsampled bilinearly.
    This mathematically enforces that the output edges align with the
    RGB guidance image edges.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


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
# RBRM: Deep Guided Filter (DGF)
# =============================================================================

class BoxFilter(nn.Module):
    """Box filter for mean calculation"""
    def __init__(self, r):
        super(BoxFilter, self).__init__()
        self.r = r
        self.kernel_size = 2 * r + 1
        self.box_filter = nn.AvgPool2d(kernel_size=self.kernel_size, stride=1, padding=r)

    def forward(self, x):
        return self.box_filter(x)


class EdgeBranch(nn.Module):
    """
    Explicit Edge Detection Branch.
    Predicts a boundary map from the RGB image.
    Increased capacity to 64 channels and 3 blocks for better edge detection.
    """
    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.conv1 = ConvBNAct(in_channels, 64, 3)
        self.res1 = Bottleneck(64, 64, shortcut=True, expansion=1.0)
        self.res2 = Bottleneck(64, 64, shortcut=True, expansion=1.0)
        self.res3 = Bottleneck(64, 64, shortcut=True, expansion=1.0)
        self.conv_out = nn.Conv2d(64, 1, 1) # Output 1-channel edge map
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        edge_logits = self.conv_out(x)
        return edge_logits

class RBRM(nn.Module):
    """
    Enhanced RBRM with Explicit Edge Supervision.
    
    Architecture:
    1. Edge Branch: Predicts edge map from RGB (Supervised by Edge Loss).
    2. Guided Filter: Uses the predicted edge map as the 'Guidance' to refine segmentation.
    
    This ensures the refinement is driven by actual structural boundaries, not just texture.
    """
    def __init__(self, in_channels: int, num_classes: int = 2, r: int = 2):
        super().__init__()
        self.r = r
        self.eps = 1e-8
        
        # 1. Edge Detection Branch (Learns Guidance)
        self.edge_branch = EdgeBranch(in_channels=3)
        
        # 2. Feature Transformation (Optional, can just use coarse logits directly)
        # We keep it identity for now as we trust V3 logits, but we could add a small adapter
        self.adapter = nn.Identity()
        
        self.box_filter = BoxFilter(r)
        
    def forward(self, coarse_logits: torch.Tensor, rgb_image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            coarse_logits: [B, num_classes, H/4, W/4] (Output from V3 SegHead)
            rgb_image: [B, 3, H, W]
        Returns:
            refined_output: [B, num_classes, H, W]
            edge_logits: [B, 1, H, W] (For auxiliary loss)
        """
        # 1. Predict Edges (High-Res Guidance)
        edge_logits = self.edge_branch(rgb_image) # [B, 1, H, W]
        guidance_map = torch.sigmoid(edge_logits) # [0, 1] range
        
        # 2. Prepare Low-Res Guidance I_l
        # Downsample guidance to match coarse logits resolution
        I_l = F.interpolate(guidance_map, size=coarse_logits.shape[2:], mode='bilinear', align_corners=False)
        
        # 3. Prepare Low-Res Target P_l
        P_l = coarse_logits
        
        # 4. Guided Filter Operations
        # Mean filtering
        mean_I = self.box_filter(I_l)
        mean_P = self.box_filter(P_l)
        
        corr_IP = self.box_filter(I_l * P_l)
        var_I = self.box_filter(I_l * I_l) - mean_I * mean_I
        
        cov_IP = corr_IP - mean_I * mean_P
        
        # A = cov_IP / (var_I + eps)
        A = cov_IP / (var_I + self.eps)
        
        # b = mean_P - A * mean_I
        b = mean_P - A * mean_I
        
        # 5. Upsample Coefficients to High-Res
        A_h = F.interpolate(A, size=guidance_map.shape[2:], mode='bilinear', align_corners=False)
        b_h = F.interpolate(b, size=guidance_map.shape[2:], mode='bilinear', align_corners=False)
        
        # 6. Apply Linear Model: O = A * I + b
        # Use the learned guidance map I_h (guidance_map)
        O = A_h * guidance_map + b_h
        
        return O, edge_logits


# =============================================================================
# Segmentation Head (for Deep Supervision)
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
# Complete V4 Model: CSPDarknet + ASPPLite + APUD + RBRM (DGF)
# =============================================================================

class AblationV4RBRM(nn.Module):
    """
    Ablation Study V4: RBRM (Residual Boundary Refinement Module)
    
    Full model with Deep Guided Filter boundary refinement:
    - Encoder: CSPDarknet
    - Neck: ASPPLite
    - Decoder: APUD (SE + Spatial Attention)
    - Refinement: RBRM (Deep Guided Filter) [NEW]
    
    Training Modes:
        1. End-to-end: Train everything together (default)
        2. Two-stage: Load V3 weights, freeze backbone, train only RBRM
    """
    def __init__(self, num_classes: int = 2, in_channels: int = 3,
                 base_channels: int = 64, base_depth: int = 3, 
                 decoder_channels: int = 256, aspp_out_channels: int = 256, 
                 rbrm_base_channels: int = 32, se_reduction: int = 16,
                 deep_supervision: bool = False, freeze_backbone: bool = False):
        super().__init__()
        
        self.deep_supervision = deep_supervision
        self.num_classes = num_classes
        self._freeze_backbone_flag = freeze_backbone
        
        # Encoder: CSPDarknet
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
        
        # Segmentation Head (from V3)
        # We set scale_factor=1 because we want the low-res logits for RBRM
        self.seg_head = SegmentationHead(
            in_channels=decoder_channels,
            num_classes=num_classes,
            scale_factor=1 
        )
        
        # RBRM: Deep Guided Filter
        # Takes coarse logits (from seg_head) and RGB image
        self.rbrm = RBRM(
            in_channels=decoder_channels, # Not used anymore as we pass logits
            num_classes=num_classes,
            r=2 # Radius 2 (5x5 window)
        )
        
        # Deep supervision head (optional) - from APUD before RBRM
        if deep_supervision:
            self.aux_head = SegmentationHead(
                in_channels=decoder_channels,
                num_classes=num_classes,
                scale_factor=4
            )
        
        self._init_weights()
        
        # Apply freezing if requested
        if freeze_backbone:
            self.freeze_backbone()
    
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
    
    def freeze_backbone(self):
        """Freeze encoder, neck (ASPP), decoder (APUD), and seg_head - only train RBRM"""
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.neck.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False
        for param in self.seg_head.parameters():
            param.requires_grad = False
        self._freeze_backbone_flag = True
        print("[V4] Backbone + SegHead frozen. Only RBRM (Deep Guided Filter) is trainable.")
    
    def unfreeze_backbone(self):
        """Unfreeze all parameters"""
        for param in self.parameters():
            param.requires_grad = True
        self._freeze_backbone_flag = False
        print("[V4] All parameters unfrozen.")
    
    def load_v3_weights(self, v3_checkpoint_path: str):
        """
        Load pretrained V3 weights into encoder, neck (ASPP), decoder (APUD), and seg_head.
        RBRM weights are randomly initialized.
        
        Args:
            v3_checkpoint_path: Path to V3 checkpoint (.pth file)
        """
        import os
        if not os.path.exists(v3_checkpoint_path):
            raise FileNotFoundError(f"V3 checkpoint not found: {v3_checkpoint_path}")
        
        checkpoint = torch.load(v3_checkpoint_path, map_location='cpu', weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Filter to only V3-compatible keys (encoder, neck, decoder, seg_head)
        v3_keys = {}
        for k, v in state_dict.items():
            if k.startswith('encoder.') or k.startswith('neck.') or k.startswith('decoder.') or k.startswith('seg_head.'):
                v3_keys[k] = v
        
        # Load with strict=False to ignore missing RBRM keys
        missing, unexpected = self.load_state_dict(v3_keys, strict=False)
        
        loaded_count = len(v3_keys)
        print(f"[V4] Loaded {loaded_count} weight tensors from V3 checkpoint")
        print(f"[V4] Missing keys (RBRM - expected): {len(missing)}")
        if unexpected:
            print(f"[V4] Unexpected keys: {len(unexpected)}")
        
        return self
    
    def get_trainable_params(self):
        """Get only trainable parameters (for optimizer)"""
        return [p for p in self.parameters() if p.requires_grad]
    
    def get_params_count(self):
        """Get parameter counts"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Count by component
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        neck_params = sum(p.numel() for p in self.neck.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        seg_head_params = sum(p.numel() for p in self.seg_head.parameters())
        rbrm_params = sum(p.numel() for p in self.rbrm.parameters())
        
        return {
            'total': total,
            'trainable': trainable,
            'total_M': total / 1e6,
            'trainable_M': trainable / 1e6,
            'encoder_M': encoder_params / 1e6,
            'neck_M': neck_params / 1e6,
            'decoder_M': decoder_params / 1e6,
            'seg_head_M': seg_head_params / 1e6,
            'rbrm_M': rbrm_params / 1e6
        }
    
    def forward(self, x: torch.Tensor):
        input_size = x.shape[2:]
        
        # Store original RGB for RBRM (edge guidance)
        rgb_input = x  # [B, 3, H, W]
        
        # Encoder
        encoder_features = self.encoder(x)  # [c2, c3, c4, c5]
        
        # Neck
        neck_out = self.neck(encoder_features[-1])  # 1/32
        
        # Decoder (APUD) - outputs at 1/4 resolution
        decoder_out = self.decoder(encoder_features, neck_out)  # 1/4
        
        # V3 Segmentation Head (Coarse Prediction)
        # Output: [B, num_classes, H/4, W/4] (since scale_factor=1)
        coarse_logits = self.seg_head(decoder_out)
        
        # RBRM: Deep Guided Filter
        # Takes coarse logits and RGB, outputs refined full-res logits AND edge logits
        out, edge_logits = self.rbrm(coarse_logits, rgb_input)  # [B, num_classes, H, W], [B, 1, H, W]
        
        if self.training:
            # Return edge_logits for auxiliary loss
            if self.deep_supervision:
                aux = F.interpolate(coarse_logits, scale_factor=4, mode='bilinear', align_corners=False)
                return out, aux, edge_logits
            return out, edge_logits
        
        return out


def create_model(num_classes: int = 2, **kwargs) -> AblationV4RBRM:
    """Factory function to create V4 RBRM model"""
    return AblationV4RBRM(num_classes=num_classes, **kwargs)


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    # Test the model
    model = AblationV4RBRM(num_classes=2, deep_supervision=False)
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"V4 RBRM Model Architecture (Deep Guided Filter)")
    print(f"=" * 60)
    print(f"Encoder: CSPDarknet")
    print(f"Neck: ASPPLite")
    print(f"Decoder: APUD (SE Attention + Spatial Attention)")
    print(f"Refinement: RBRM (Deep Guided Filter)")
    print(f"=" * 60)
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
    
    # Test forward pass
    x = torch.randn(1, 3, 480, 640)
    with torch.no_grad():
        y = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    # Test with deep supervision
    print(f"\nWith Deep Supervision (batch=2 for BatchNorm):")
    model_ds = AblationV4RBRM(num_classes=2, deep_supervision=True)
    model_ds.train()
    x2 = torch.randn(2, 3, 480, 640)
    outputs = model_ds(x2)
    print(f"  Main output: {outputs[0].shape}")
    print(f"  Aux output: {outputs[1].shape}")
