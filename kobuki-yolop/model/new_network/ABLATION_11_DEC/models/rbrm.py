"""
RBRM Module: Residual Boundary Refinement Module
Paper: AURASeg - Attention Guided Upsampling with Residual Boundary-Assistive Refinement

This module implements the Residual Boundary Refinement Module (RBRM) which is 
added after the APUD decoder in V4 to enhance boundary segmentation.

RBRM Architecture:
    1. Boundary Detection Head: Sobel-X + Sobel-Y for edge feature extraction
    2. Boundary Encoder: Lightweight 3-stage encoder (64→128→256→512)
    3. Boundary Decoder: Symmetric decoder with skip connections (512→256→128→64)
    4. Boundary-Guided Attention: Edge features modulate main features
    5. Learned Residual Fusion: F_out = F_main + gate × F_boundary

Key Design Principles:
    - Secondary encoder-decoder for multi-scale edge learning
    - Sobel operators for explicit edge detection (fixed, not learned)
    - Residual connection ensures main features are never lost
    - Learned gate decides where boundary features contribute
    - Boundary head provides explicit supervision signal
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    """Standard Convolution + BatchNorm + ReLU"""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, stride: int = 1, padding: int = None):
        super().__init__()
        
        if padding is None:
            padding = kernel_size // 2
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                              stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class BoundaryDetectionHead(nn.Module):
    """
    Boundary Detection Head using Sobel Operators
    
    Extracts edge features using fixed Sobel-X and Sobel-Y kernels.
    
    Architecture:
        Input (256ch, H/4) 
            → Conv 1×1 (256→64) + BN + ReLU
            → Sobel-X (fixed) → Edge-X (64ch)
            → Sobel-Y (fixed) → Edge-Y (64ch)
            → Concat → Conv 3×3 (128→64) + BN + ReLU
            → Edge Features (64ch, H/4)
    """
    
    def __init__(self, in_channels: int = 256, out_channels: int = 64):
        super().__init__()
        
        # Initial projection to reduce channels
        self.proj = ConvBNReLU(in_channels, out_channels, kernel_size=1)
        
        # Sobel kernels (fixed, not learned)
        # Sobel-X: Horizontal edges
        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
        # Sobel-Y: Vertical edges
        sobel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
        # Expand sobel kernels to handle all channels
        # Shape: (out_channels, 1, 3, 3) for depthwise convolution
        self.register_buffer('sobel_x', sobel_x.repeat(out_channels, 1, 1, 1))
        self.register_buffer('sobel_y', sobel_y.repeat(out_channels, 1, 1, 1))
        
        # Fusion conv after Sobel
        self.fusion = ConvBNReLU(out_channels * 2, out_channels, kernel_size=3)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features (B, 256, H, W)
            
        Returns:
            Edge features (B, 64, H, W)
        """
        # Project to lower channels
        x = self.proj(x)  # (B, 64, H, W)
        
        # Apply Sobel operators (depthwise convolution)
        edge_x = F.conv2d(x, self.sobel_x, padding=1, groups=x.shape[1])
        edge_y = F.conv2d(x, self.sobel_y, padding=1, groups=x.shape[1])
        
        # Concatenate edge responses
        edges = torch.cat([edge_x, edge_y], dim=1)  # (B, 128, H, W)
        
        # Fuse edge features
        edge_features = self.fusion(edges)  # (B, 64, H, W)
        
        return edge_features


class BoundaryEncoder(nn.Module):
    """
    Lightweight Boundary Encoder (3 stages)
    
    Progressively downsamples and increases channels:
        Stage 1: 64 → 128 @ H/2
        Stage 2: 128 → 256 @ H/4
        Stage 3: 256 → 512 @ H/8
        
    Returns features at each stage for skip connections.
    """
    
    def __init__(self, in_channels: int = 64):
        super().__init__()
        
        # Stage 1: 64 → 128, stride=2
        self.stage1 = nn.Sequential(
            ConvBNReLU(in_channels, 128, kernel_size=3, stride=2),
            ConvBNReLU(128, 128, kernel_size=3, stride=1)
        )
        
        # Stage 2: 128 → 256, stride=2
        self.stage2 = nn.Sequential(
            ConvBNReLU(128, 256, kernel_size=3, stride=2),
            ConvBNReLU(256, 256, kernel_size=3, stride=1)
        )
        
        # Stage 3: 256 → 512, stride=2
        self.stage3 = nn.Sequential(
            ConvBNReLU(256, 512, kernel_size=3, stride=2),
            ConvBNReLU(512, 512, kernel_size=3, stride=1)
        )
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: Edge features (B, 64, H, W)
            
        Returns:
            Tuple of (s1, s2, s3, bottleneck):
                s1: (B, 128, H/2, W/2)
                s2: (B, 256, H/4, W/4)
                s3: (B, 512, H/8, W/8) - bottleneck
        """
        s1 = self.stage1(x)   # (B, 128, H/2, W/2)
        s2 = self.stage2(s1)  # (B, 256, H/4, W/4)
        s3 = self.stage3(s2)  # (B, 512, H/8, W/8)
        
        return s1, s2, s3


class BoundaryDecoder(nn.Module):
    """
    Boundary Decoder with Skip Connections (3 stages)
    
    Progressively upsamples and decreases channels:
        Up-1: 512 + 256 (skip) → 256 @ H/4
        Up-2: 256 + 128 (skip) → 128 @ H/2
        Up-3: 128 + 64 (skip) → 64 @ H/1
        
    Uses skip connections from encoder.
    """
    
    def __init__(self, out_channels: int = 64):
        super().__init__()
        
        # Up-1: 512 + 256 → 256
        self.up1 = nn.Sequential(
            ConvBNReLU(512 + 256, 256, kernel_size=3),
            ConvBNReLU(256, 256, kernel_size=3)
        )
        
        # Up-2: 256 + 128 → 128
        self.up2 = nn.Sequential(
            ConvBNReLU(256 + 128, 128, kernel_size=3),
            ConvBNReLU(128, 128, kernel_size=3)
        )
        
        # Up-3: 128 + 64 → 64
        self.up3 = nn.Sequential(
            ConvBNReLU(128 + out_channels, out_channels, kernel_size=3),
            ConvBNReLU(out_channels, out_channels, kernel_size=3)
        )
    
    def forward(self, s3: torch.Tensor, s2: torch.Tensor, 
                s1: torch.Tensor, edge_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s3: Bottleneck from encoder (B, 512, H/8, W/8)
            s2: Skip from encoder stage 2 (B, 256, H/4, W/4)
            s1: Skip from encoder stage 1 (B, 128, H/2, W/2)
            edge_features: Original edge features (B, 64, H, W)
            
        Returns:
            Decoded boundary features (B, 64, H, W)
        """
        # Up-1: Upsample s3 and concat with s2
        x = F.interpolate(s3, size=s2.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, s2], dim=1)  # (B, 512+256, H/4, W/4)
        x = self.up1(x)  # (B, 256, H/4, W/4)
        
        # Up-2: Upsample and concat with s1
        x = F.interpolate(x, size=s1.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, s1], dim=1)  # (B, 256+128, H/2, W/2)
        x = self.up2(x)  # (B, 128, H/2, W/2)
        
        # Up-3: Upsample and concat with edge_features
        x = F.interpolate(x, size=edge_features.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, edge_features], dim=1)  # (B, 128+64, H, W)
        x = self.up3(x)  # (B, 64, H, W)
        
        return x


class LearnedResidualFusion(nn.Module):
    """
    Learned Residual Fusion Module
    
    Fuses main features with boundary features using a learned gate:
        gate = sigmoid(conv1x1(relu(conv3x3(concat(F_main, F_boundary)))))
        F_output = F_main + gate × F_boundary
        
    The gate learns WHERE boundary features should contribute.
    """
    
    def __init__(self, main_channels: int = 256, boundary_channels: int = 256):
        super().__init__()
        
        concat_channels = main_channels + boundary_channels
        
        # Gate network
        self.gate_conv1 = nn.Conv2d(concat_channels, main_channels, kernel_size=3, 
                                     padding=1, bias=False)
        self.gate_bn = nn.BatchNorm2d(main_channels)
        self.gate_act = nn.SiLU(inplace=True)
        
        self.gate_conv2 = nn.Conv2d(main_channels, main_channels, kernel_size=1)
        self.gate_sigmoid = nn.Sigmoid()
        
        # Initialize gate to output ~0.5 (balanced contribution initially)
        nn.init.zeros_(self.gate_conv2.bias)
    
    def forward(self, f_main: torch.Tensor, f_boundary: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f_main: Main features from APUD decoder (B, 256, H, W)
            f_boundary: Boundary features from RBRM (B, 256, H, W)
            
        Returns:
            Fused features (B, 256, H, W)
        """
        # Concatenate features
        concat = torch.cat([f_main, f_boundary], dim=1)  # (B, 512, H, W)
        
        # Compute gate
        gate = self.gate_conv1(concat)
        gate = self.gate_bn(gate)
        gate = self.gate_act(gate)
        gate = self.gate_conv2(gate)
        gate = self.gate_sigmoid(gate)  # (B, 256, H, W), values in [0, 1]
        
        # Residual fusion: F_out = F_main + gate * F_boundary
        f_out = f_main + gate * f_boundary
        
        return f_out


class BoundaryHead(nn.Module):
    """
    Boundary Prediction Head for explicit boundary supervision
    
    Takes boundary features and outputs a binary boundary map (logits).
    Used to compute boundary loss during training.
    
    Note: Outputs logits (no sigmoid) for AMP compatibility with BCE loss.
    """
    
    def __init__(self, in_channels: int = 64):
        super().__init__()
        
        self.conv1 = ConvBNReLU(in_channels, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 1, kernel_size=1)
        # Note: No sigmoid here - output logits for AMP-safe BCE with logits
    
    def forward(self, x: torch.Tensor, target_size: tuple = None) -> torch.Tensor:
        """
        Args:
            x: Boundary features (B, 64, H, W)
            target_size: Optional size to upsample to (H_out, W_out)
            
        Returns:
            Boundary logits (B, 1, H_out, W_out) - NOT sigmoid activated
        """
        x = self.conv1(x)
        x = self.conv2(x)  # Output logits
        
        if target_size is not None:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=True)
        
        return x


class RBRMModule(nn.Module):
    """
    Residual Boundary Refinement Module (RBRM)
    
    Complete RBRM architecture that takes APUD decoder features and refines
    them with boundary-aware processing.
    
    Architecture:
        APUD-4 output (256, H/4)
            │
            ├─────────────────────────────────────┐
            ↓                                     ↓
        BoundaryDetectionHead                  (Main Path)
        (Sobel-X, Sobel-Y → 64ch)                 │
            ↓                                     │
        BoundaryEncoder                           │
        (64→128→256→512, 3 stages)                │
            ↓                                     │
        BoundaryDecoder                           │
        (512→256→128→64, skip connections)        │
            ↓                                     │
        FeatureProjection (64→256)                │
            ↓                                     │
        LearnedResidualFusion ←───────────────────┘
        (F_out = F_main + gate × F_boundary)
            ↓
        Refined Features (256, H/4)
            │
            └─→ BoundaryHead (for supervision)
                → Boundary Prediction (1, H)
    
    Args:
        in_channels: Input channels from APUD decoder (default: 256)
        edge_channels: Channels for edge features (default: 64)
    """
    
    def __init__(self, in_channels: int = 256, edge_channels: int = 64):
        super().__init__()
        
        self.in_channels = in_channels
        self.edge_channels = edge_channels
        
        # Boundary Detection Head (Sobel operators)
        self.boundary_detection = BoundaryDetectionHead(in_channels, edge_channels)
        
        # Boundary Encoder (lightweight 3-stage)
        self.boundary_encoder = BoundaryEncoder(edge_channels)
        
        # Boundary Decoder (with skip connections)
        self.boundary_decoder = BoundaryDecoder(edge_channels)
        
        # Feature Projection: 64 → 256 channels
        self.feature_proj = nn.Sequential(
            ConvBNReLU(edge_channels, in_channels, kernel_size=1),
            ConvBNReLU(in_channels, in_channels, kernel_size=3)
        )
        
        # Learned Residual Fusion
        self.fusion = LearnedResidualFusion(in_channels, in_channels)
        
        # Boundary Head (for explicit boundary supervision)
        self.boundary_head = BoundaryHead(edge_channels)
    
    def forward(self, x: torch.Tensor, return_boundary: bool = True) -> dict:
        """
        Args:
            x: Input features from APUD-4 (B, 256, H/4, W/4)
            return_boundary: Whether to return boundary prediction
            
        Returns:
            Dictionary with:
                - 'features': Refined features (B, 256, H/4, W/4)
                - 'boundary': Boundary prediction (B, 1, H, W) if return_boundary
        """
        # Store original input for residual fusion
        f_main = x
        
        # Step 1: Boundary Detection using Sobel
        edge_features = self.boundary_detection(x)  # (B, 64, H/4, W/4)
        
        # Step 2: Boundary Encoder (multi-scale edge processing)
        s1, s2, s3 = self.boundary_encoder(edge_features)
        # s1: (B, 128, H/8, W/8)
        # s2: (B, 256, H/16, W/16)
        # s3: (B, 512, H/32, W/32)
        
        # Step 3: Boundary Decoder (with skip connections)
        boundary_features = self.boundary_decoder(s3, s2, s1, edge_features)
        # boundary_features: (B, 64, H/4, W/4)
        
        # Step 4: Project to match main feature channels
        f_boundary = self.feature_proj(boundary_features)  # (B, 256, H/4, W/4)
        
        # Step 5: Learned Residual Fusion
        f_refined = self.fusion(f_main, f_boundary)  # (B, 256, H/4, W/4)
        
        result = {'features': f_refined}
        
        if return_boundary:
            # Boundary prediction (upsample to full resolution)
            # Input spatial: H/4, output: H (4× upsample)
            input_h, input_w = x.shape[2], x.shape[3]
            target_size = (input_h * 4, input_w * 4)  # Full resolution
            boundary_pred = self.boundary_head(boundary_features, target_size)
            result['boundary'] = boundary_pred
        
        return result


if __name__ == "__main__":
    print("=" * 60)
    print("Testing RBRM Module")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Test Boundary Detection Head
    print("\n1. Testing BoundaryDetectionHead:")
    bd_head = BoundaryDetectionHead(256, 64).to(device)
    x = torch.randn(2, 256, 96, 160).to(device)
    edge_out = bd_head(x)
    print(f"   Input: {x.shape} → Output: {edge_out.shape}")
    assert edge_out.shape == (2, 64, 96, 160), "BoundaryDetectionHead shape mismatch!"
    print("   ✓ BoundaryDetectionHead passed")
    
    # Test Boundary Encoder
    print("\n2. Testing BoundaryEncoder:")
    encoder = BoundaryEncoder(64).to(device)
    x = torch.randn(2, 64, 96, 160).to(device)
    s1, s2, s3 = encoder(x)
    print(f"   Input: {x.shape}")
    print(f"   s1: {s1.shape}")
    print(f"   s2: {s2.shape}")
    print(f"   s3 (bottleneck): {s3.shape}")
    assert s1.shape == (2, 128, 48, 80), "Encoder s1 shape mismatch!"
    assert s2.shape == (2, 256, 24, 40), "Encoder s2 shape mismatch!"
    assert s3.shape == (2, 512, 12, 20), "Encoder s3 shape mismatch!"
    print("   ✓ BoundaryEncoder passed")
    
    # Test Boundary Decoder
    print("\n3. Testing BoundaryDecoder:")
    decoder = BoundaryDecoder(64).to(device)
    edge_feat = torch.randn(2, 64, 96, 160).to(device)
    decoded = decoder(s3, s2, s1, edge_feat)
    print(f"   s3: {s3.shape}, s2: {s2.shape}, s1: {s1.shape}")
    print(f"   edge_features: {edge_feat.shape}")
    print(f"   Output: {decoded.shape}")
    assert decoded.shape == (2, 64, 96, 160), "BoundaryDecoder shape mismatch!"
    print("   ✓ BoundaryDecoder passed")
    
    # Test Learned Residual Fusion
    print("\n4. Testing LearnedResidualFusion:")
    fusion = LearnedResidualFusion(256, 256).to(device)
    f_main = torch.randn(2, 256, 96, 160).to(device)
    f_boundary = torch.randn(2, 256, 96, 160).to(device)
    f_out = fusion(f_main, f_boundary)
    print(f"   f_main: {f_main.shape}, f_boundary: {f_boundary.shape}")
    print(f"   Output: {f_out.shape}")
    assert f_out.shape == (2, 256, 96, 160), "LearnedResidualFusion shape mismatch!"
    print("   ✓ LearnedResidualFusion passed")
    
    # Test Full RBRM Module
    print("\n5. Testing Full RBRMModule:")
    rbrm = RBRMModule(in_channels=256, edge_channels=64).to(device)
    x = torch.randn(2, 256, 96, 160).to(device)  # APUD-4 output @ H/4
    
    result = rbrm(x, return_boundary=True)
    
    print(f"   Input: {x.shape}")
    print(f"   Refined features: {result['features'].shape}")
    print(f"   Boundary prediction: {result['boundary'].shape}")
    
    assert result['features'].shape == (2, 256, 96, 160), "RBRM features shape mismatch!"
    assert result['boundary'].shape == (2, 1, 384, 640), "RBRM boundary shape mismatch!"
    print("   ✓ RBRMModule passed")
    
    # Count parameters
    params = sum(p.numel() for p in rbrm.parameters() if p.requires_grad)
    print(f"\n   RBRM Parameters: {params:,}")
    
    # Test backward pass
    print("\n6. Testing Backward Pass:")
    loss = result['features'].sum() + result['boundary'].sum()
    loss.backward()
    print("   ✓ Backward pass successful")
    
    print("\n" + "=" * 60)
    print("All RBRM tests passed!")
    print("=" * 60)
