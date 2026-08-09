"""
AURASeg WACV ResNet-18
======================

Clean, standalone training-capable ResNet-18 architecture.
Supports new WACV ablations for APUD (fusion/attention) and RBRM (sobel/gate).
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
# ASPP-Lite Module
# =============================================================================

class ASPPLiteConv(nn.Module):
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
    def __init__(self, low_channels: int, high_channels: int, out_channels: int,
                 se_reduction: int = 16, spatial_kernel: int = 7,
                 fusion_type: str = 'mul', attention_mode: str = 'full'):
        super().__init__()
        self.fusion_type = fusion_type
        self.attention_mode = attention_mode
        
        if self.fusion_type not in ['mul', 'add', 'concat']:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")
        if self.attention_mode not in ['full', 'none', 'se', 'spatial']:
            raise ValueError(f"Unknown attention mode: {self.attention_mode}")
            
        self.low_transform = ConvBNAct(low_channels, out_channels, kernel_size=1)
        self.high_transform = ConvBNAct(high_channels, out_channels, kernel_size=1)
        
        if self.attention_mode in ['full', 'se']:
            self.se_attention = SEAttention(out_channels, reduction=se_reduction)
        else:
            self.se_attention = nn.Identity()
            
        if self.attention_mode in ['full', 'spatial']:
            self.spatial_attention = SpatialAttention(kernel_size=spatial_kernel)
        else:
            self.spatial_attention = nn.Identity()
            
        if self.fusion_type == 'concat':
            self.concat_proj = ConvBNAct(out_channels * 2, out_channels, kernel_size=1)
            
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
        
        if self.fusion_type == 'mul':
            fusion = low_up * high
        elif self.fusion_type == 'add':
            fusion = low_up + high
        elif self.fusion_type == 'concat':
            fusion = torch.cat([low_up, high], dim=1)
            fusion = self.concat_proj(fusion)
            
        spatial = self.spatial_attention(high)
        combined = fusion + spatial
        out = self.refinement(combined)
        
        return out

class APUDDecoder(nn.Module):
    def __init__(self, 
                 encoder_channels: list,
                 neck_channels: int = 256,
                 decoder_channels: int = 256,
                 num_classes: int = 2,
                 se_reduction: int = 16,
                 fusion_type: str = 'mul',
                 attention_mode: str = 'full'):
        super().__init__()
        
        c1, c2, c3, c4 = encoder_channels
        
        self.apud1 = APUDBlock(neck_channels, c4, decoder_channels, se_reduction, fusion_type=fusion_type, attention_mode=attention_mode)
        self.apud2 = APUDBlock(decoder_channels, c3, decoder_channels, se_reduction, fusion_type=fusion_type, attention_mode=attention_mode)
        self.apud3 = APUDBlock(decoder_channels, c2, decoder_channels, se_reduction, fusion_type=fusion_type, attention_mode=attention_mode)
        self.apud4 = APUDBlock(decoder_channels, c1, decoder_channels, se_reduction, fusion_type=fusion_type, attention_mode=attention_mode)
        
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
    
    def forward(self, neck_out: torch.Tensor, encoder_features: list, return_aux: bool = True) -> dict:
        c1, c2, c3, c4 = encoder_features
        
        out1 = self.apud1(neck_out, c4)
        out2 = self.apud2(out1, c3)
        out3 = self.apud3(out2, c2)
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
# RBRM Module
# =============================================================================

class BoundaryDetectionHead(nn.Module):
    def __init__(self, in_channels: int = 256, out_channels: int = 64, use_sobel: bool = True):
        super().__init__()
        self.use_sobel = use_sobel
        self.proj = ConvBNAct(in_channels, out_channels, kernel_size=1)
        
        if self.use_sobel:
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
            self.register_buffer('sobel_x', sobel_x.repeat(out_channels, 1, 1, 1))
            self.register_buffer('sobel_y', sobel_y.repeat(out_channels, 1, 1, 1))
            self.fusion = ConvBNAct(out_channels * 2, out_channels, kernel_size=3)
        else:
            self.learned_edge = ConvBNAct(out_channels, out_channels, kernel_size=3)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        if self.use_sobel:
            edge_x = F.conv2d(x, self.sobel_x, padding=1, groups=x.shape[1])
            edge_y = F.conv2d(x, self.sobel_y, padding=1, groups=x.shape[1])
            edges = torch.cat([edge_x, edge_y], dim=1)
            edge_features = self.fusion(edges)
        else:
            edge_features = self.learned_edge(x)
        return edge_features

class BoundaryEncoder(nn.Module):
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
    def __init__(self, in_channels: int = 256, edge_channels: int = 64, 
                 use_sobel: bool = True, use_gate: bool = True):
        super().__init__()
        self.use_gate = use_gate
        
        self.boundary_head = BoundaryDetectionHead(in_channels, edge_channels, use_sobel=use_sobel)
        self.boundary_encoder = BoundaryEncoder(edge_channels)
        self.boundary_decoder = BoundaryDecoder(edge_channels)
        
        self.boundary_proj = nn.Sequential(
            ConvBNAct(edge_channels, in_channels, kernel_size=1),
            nn.Conv2d(in_channels, in_channels, 1, bias=False)
        )
        
        if self.use_gate:
            self.fusion_gate = nn.Sequential(
                nn.Conv2d(in_channels * 2, in_channels, 1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.Sigmoid()
            )
            
        self.boundary_pred = nn.Conv2d(edge_channels, 1, 1)
    
    def forward(self, x: torch.Tensor, return_boundary: bool = False) -> dict:
        edge_features = self.boundary_head(x)
        s1, s2, s3 = self.boundary_encoder(edge_features)
        boundary_features = self.boundary_decoder(s3, s2, s1, edge_features)
        
        boundary_proj = self.boundary_proj(boundary_features)
        
        if self.use_gate:
            concat = torch.cat([x, boundary_proj], dim=1)
            gate = self.fusion_gate(concat)
            refined = x + gate * boundary_proj
        else:
            refined = x + boundary_proj
            
        result = {'features': refined}
        if return_boundary:
            result['boundary'] = self.boundary_pred(boundary_features)
            
        return result

# =============================================================================
# Main Model: AURASeg WACV ResNet-18
# =============================================================================

class AURASeg_R18_WACV(nn.Module):
    """
    AURASeg WACV Model explicitly hardcoded to ResNet-18.
    Supports all ablations required for WACV experiments.
    """
    def __init__(self, 
                 num_classes: int = 2, 
                 decoder_channels: int = 256,
                 encoder_weights: str = 'imagenet',
                 fusion_type: str = 'mul',
                 attention_mode: str = 'full',
                 use_sobel: bool = True,
                 use_gate: bool = True):
        super().__init__()
        
        self.num_classes = num_classes
        
        # ResNet-18 Hardcoded Channels
        encoder_channels = [64, 64, 128, 256] # layer0 to layer3
        context_channels = 512
        aspp_branch_channels = 64
        
        self.encoder = smp.encoders.get_encoder(
            name="resnet18",
            in_channels=3,
            depth=5,
            weights=encoder_weights
        )
        
        self.aspp_lite = ASPPLite(
            in_channels=context_channels,
            out_channels=decoder_channels,
            branch_channels=aspp_branch_channels
        )
        
        self.decoder = APUDDecoder(
            encoder_channels=encoder_channels,
            neck_channels=decoder_channels,
            decoder_channels=decoder_channels,
            num_classes=num_classes,
            fusion_type=fusion_type,
            attention_mode=attention_mode
        )
        
        self.rbrm = RBRMModule(
            in_channels=decoder_channels,
            edge_channels=64,
            use_sobel=use_sobel,
            use_gate=use_gate
        )
        
        self.seg_head = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels // 2),
            nn.SiLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(decoder_channels // 2, num_classes, 1)
        )
        
        self._init_decoder_weights()
    
    def _init_decoder_weights(self):
        for module in [self.aspp_lite, self.decoder, self.rbrm, self.seg_head]:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor, return_aux: bool = False, return_boundary: bool = False) -> dict:
        input_size = x.shape[2:]
        
        features = self.encoder(x)
        # ResNet-18: [1] -> 64, [2] -> 64, [3] -> 128, [4] -> 256, [5] -> 512
        c1, c2, c3, c4, c5 = features[1], features[2], features[3], features[4], features[5]
        
        context = self.aspp_lite(c5)
        
        decoder_out = self.decoder(context, [c1, c2, c3, c4], return_aux=return_aux)
        decoder_features = decoder_out['decoder_features']
        
        rbrm_out = self.rbrm(decoder_features, return_boundary=return_boundary)
        refined_features = rbrm_out['features']
        
        main_out = self.seg_head(refined_features)
        main_out = F.interpolate(main_out, size=input_size, mode='bilinear', align_corners=True)
        
        result = {'main': main_out}
        
        if return_aux and 'aux' in decoder_out:
            result['aux'] = [
                F.interpolate(aux, size=input_size, mode='bilinear', align_corners=True)
                for aux in decoder_out['aux']
            ]
            
        if return_boundary and 'boundary' in rbrm_out:
            boundary = F.interpolate(rbrm_out['boundary'], size=input_size, mode='bilinear', align_corners=True)
            result['boundary'] = boundary
            
        return result

    def get_param_groups(self, lr_encoder: float = 1e-4, lr_decoder: float = 1e-3):
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
