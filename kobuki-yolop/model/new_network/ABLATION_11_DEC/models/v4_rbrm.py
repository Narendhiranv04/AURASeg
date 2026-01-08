"""
V4 RBRM Model: V3 APUD + Residual Boundary Refinement Module
Paper: AURASeg - Attention Guided Upsampling with Residual Boundary-Assistive Refinement

This model extends V3 (APUD) with the Residual Boundary Refinement Module (RBRM)
to enhance boundary segmentation through explicit edge learning.

Architecture:
    - Encoder: CSPDarknet-53 (same as V3)
    - Context Module: ASPP-Lite (same as V3)
    - Decoder: 4 APUD blocks with deep supervision (same as V3)
    - NEW: RBRM after APUD decoder for boundary refinement
    - Loss: Focal + Dice + Deep Supervision + Boundary Loss

RBRM Architecture:
    APUD-4 output (256, H/4)
        ↓
    Boundary Detection (Sobel operators)
        ↓
    Boundary Encoder-Decoder (lightweight U-Net)
        ↓
    Learned Residual Fusion: F_out = F_main + gate × F_boundary
        ↓
    Refined Features → Seg Head → Main Output
                    → Boundary Head → Boundary Supervision

Training Strategy:
    - Initialize from V3 pretrained weights
    - Differential learning rates: 1e-4 (pretrained), 1e-3 (RBRM)
    - End-to-end training (no freezing)
    - Boundary loss at full resolution only
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import CSPDarknet53
from .aspp_lite import ASPPLite
from .apud import APUDBlock, SEAttention, SpatialAttention, ConvBNAct
from .rbrm import RBRMModule


class APUDDecoderForV4(nn.Module):
    """
    APUD Decoder modified for V4 to expose intermediate features for RBRM.
    
    Same as original APUDDecoder but returns APUD-4 features before seg head.
    """
    
    def __init__(self, 
                 encoder_channels: list = [64, 128, 256, 512],
                 neck_channels: int = 256,
                 decoder_channels: int = 256,
                 num_classes: int = 2,
                 se_reduction: int = 16):
        super().__init__()
        
        c1, c2, c3, c4 = encoder_channels
        
        # APUD blocks (4 levels)
        self.apud1 = APUDBlock(neck_channels, c4, decoder_channels, se_reduction)
        self.apud2 = APUDBlock(decoder_channels, c3, decoder_channels, se_reduction)
        self.apud3 = APUDBlock(decoder_channels, c2, decoder_channels, se_reduction)
        self.apud4 = APUDBlock(decoder_channels, c1, decoder_channels, se_reduction)
        
        # Auxiliary supervision heads
        self.aux_head1 = self._make_aux_head(decoder_channels, num_classes)
        self.aux_head2 = self._make_aux_head(decoder_channels, num_classes)
        self.aux_head3 = self._make_aux_head(decoder_channels, num_classes)
        self.aux_head4 = self._make_aux_head(decoder_channels, num_classes)
    
    def _make_aux_head(self, in_channels: int, num_classes: int) -> nn.Sequential:
        """Create auxiliary supervision head"""
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels // 4, num_classes, 1)
        )
    
    def forward(self, neck_out: torch.Tensor, 
                encoder_features: list,
                return_aux: bool = True) -> dict:
        """
        Returns APUD-4 features (before seg head) for RBRM processing.
        """
        c1, c2, c3, c4 = encoder_features
        
        # APUD-1: neck(256@H/32) + c4(512@H/32) → out1(256@H/32)
        out1 = self.apud1(neck_out, c4)
        
        # APUD-2: out1(256@H/32) + c3(256@H/16) → out2(256@H/16)
        out2 = self.apud2(out1, c3)
        
        # APUD-3: out2(256@H/16) + c2(128@H/8) → out3(256@H/8)
        out3 = self.apud3(out2, c2)
        
        # APUD-4: out3(256@H/8) + c1(64@H/4) → out4(256@H/4)
        out4 = self.apud4(out3, c1)
        
        result = {'decoder_features': out4}  # Return features, not final output
        
        if return_aux:
            aux1 = self.aux_head1(out1)
            aux2 = self.aux_head2(out2)
            aux3 = self.aux_head3(out3)
            aux4 = self.aux_head4(out4)
            result['aux'] = [aux1, aux2, aux3, aux4]
        
        return result


class V4RBRM(nn.Module):
    """
    V4 Ablation Model: V3 APUD + Residual Boundary Refinement Module
    
    Extends V3 with RBRM for enhanced boundary segmentation.
    
    Architecture:
        Input (3, H, W)
            ↓
        CSPDarknet-53 Encoder
            ├─ c1 (64, H/4)
            ├─ c2 (128, H/8)
            ├─ c3 (256, H/16)
            ├─ c4 (512, H/32)
            └─ c5 (1024, H/32)
                ↓
        ASPP-Lite (256, H/32)
                ↓
        APUD Decoder (4 blocks with deep supervision)
            → out4 (256, H/4)
                ↓
        RBRM (Residual Boundary Refinement)
            → refined_features (256, H/4)
            → boundary_pred (1, H)
                ↓
        Seg Head → Main Output (num_classes, H)
    
    Args:
        in_channels: Input image channels (default: 3)
        num_classes: Number of segmentation classes (default: 2)
        decoder_channels: Channels for decoder features (default: 256)
        se_reduction: SE attention reduction ratio (default: 16)
        edge_channels: Channels for RBRM edge features (default: 64)
    """
    
    def __init__(self, in_channels: int = 3, num_classes: int = 2,
                 decoder_channels: int = 256, se_reduction: int = 16,
                 edge_channels: int = 64):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Encoder: CSPDarknet-53
        self.encoder = CSPDarknet53(in_channels=in_channels)
        
        # Context Module: ASPP-Lite
        self.aspp_lite = ASPPLite(
            in_channels=1024, 
            out_channels=256, 
            branch_channels=128
        )
        
        # APUD Decoder (modified to return features before seg head)
        self.decoder = APUDDecoderForV4(
            encoder_channels=[64, 128, 256, 512],
            neck_channels=256,
            decoder_channels=decoder_channels,
            num_classes=num_classes,
            se_reduction=se_reduction
        )
        
        # RBRM: Residual Boundary Refinement Module
        self.rbrm = RBRMModule(
            in_channels=decoder_channels,
            edge_channels=edge_channels
        )
        
        # Final Segmentation Head (after RBRM)
        self.seg_head = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels // 2),
            nn.SiLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(decoder_channels // 2, num_classes, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
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
    
    def forward(self, x: torch.Tensor, return_aux: bool = True, 
                return_boundary: bool = True) -> dict:
        """
        Args:
            x: Input image (B, 3, H, W)
            return_aux: Whether to return auxiliary supervision outputs
            return_boundary: Whether to return boundary prediction
            
        Returns:
            Dictionary with:
                - 'main': Main segmentation output (B, num_classes, H, W)
                - 'aux': List of auxiliary outputs [4 scales] (if return_aux)
                - 'boundary': Boundary prediction (B, 1, H, W) (if return_boundary)
        """
        input_size = x.shape[2:]  # (H, W)
        
        # Encoder: Extract multi-scale features
        c1, c2, c3, c4, c5 = self.encoder(x)
        
        # Context Module: ASPP-Lite on deepest features
        context = self.aspp_lite(c5)
        
        # APUD Decoder
        encoder_features = [c1, c2, c3, c4]
        decoder_out = self.decoder(context, encoder_features, return_aux=return_aux)
        
        decoder_features = decoder_out['decoder_features']  # (B, 256, H/4, W/4)
        
        # RBRM: Boundary Refinement
        rbrm_out = self.rbrm(decoder_features, return_boundary=return_boundary)
        refined_features = rbrm_out['features']  # (B, 256, H/4, W/4)
        
        # Final Segmentation Head
        main_out = self.seg_head(refined_features)  # (B, num_classes, H/4, W/4)
        main_out = F.interpolate(main_out, size=input_size, mode='bilinear', align_corners=True)
        
        result = {'main': main_out}
        
        if return_aux and 'aux' in decoder_out:
            result['aux'] = decoder_out['aux']
        
        if return_boundary and 'boundary' in rbrm_out:
            result['boundary'] = rbrm_out['boundary']
        
        return result
    
    def load_v3_weights(self, v3_checkpoint_path: str, strict: bool = False):
        """
        Load pretrained V3 weights (encoder, ASPP, APUD decoder).
        RBRM remains randomly initialized.
        
        Args:
            v3_checkpoint_path: Path to V3 checkpoint
            strict: Whether to strictly enforce matching keys
        """
        checkpoint = torch.load(v3_checkpoint_path, map_location='cpu', weights_only=False)
        v3_state = checkpoint['model_state_dict']
        
        # Filter out RBRM and seg_head keys (these are new in V4)
        v4_state = self.state_dict()
        
        loaded_keys = []
        skipped_keys = []
        
        for k, v in v3_state.items():
            if k in v4_state:
                if v.shape == v4_state[k].shape:
                    v4_state[k] = v
                    loaded_keys.append(k)
                else:
                    skipped_keys.append(f"{k} (shape mismatch: {v.shape} vs {v4_state[k].shape})")
            else:
                # Try mapping V3 decoder keys to V4 decoder
                # V3 uses APUDDecoder, V4 uses APUDDecoderForV4 (same structure)
                skipped_keys.append(k)
        
        self.load_state_dict(v4_state, strict=False)
        
        print(f"Loaded {len(loaded_keys)} keys from V3 checkpoint")
        print(f"Skipped {len(skipped_keys)} keys (new in V4 or shape mismatch)")
        
        return loaded_keys, skipped_keys
    
    def get_param_groups(self, lr_pretrained: float = 1e-4, lr_new: float = 1e-3):
        """
        Get parameter groups with differential learning rates.
        
        Args:
            lr_pretrained: LR for encoder, ASPP, decoder (pretrained from V3)
            lr_new: LR for RBRM and seg_head (new, needs faster learning)
            
        Returns:
            List of param groups for optimizer
        """
        pretrained_params = []
        new_params = []
        
        # Encoder, ASPP, Decoder are pretrained
        pretrained_params.extend(self.encoder.parameters())
        pretrained_params.extend(self.aspp_lite.parameters())
        pretrained_params.extend(self.decoder.parameters())
        
        # RBRM and seg_head are new
        new_params.extend(self.rbrm.parameters())
        new_params.extend(self.seg_head.parameters())
        
        param_groups = [
            {'params': pretrained_params, 'lr': lr_pretrained, 'name': 'pretrained'},
            {'params': new_params, 'lr': lr_new, 'name': 'new'}
        ]
        
        return param_groups


def v4_rbrm(num_classes: int = 2, pretrained_v3: str = None):
    """
    Factory function to create V4 RBRM model
    
    Args:
        num_classes: Number of segmentation classes (default: 2)
        pretrained_v3: Path to V3 checkpoint for initialization
        
    Returns:
        V4RBRM model
    """
    model = V4RBRM(in_channels=3, num_classes=num_classes)
    
    if pretrained_v3 is not None:
        model.load_v3_weights(pretrained_v3)
    
    return model


if __name__ == "__main__":
    print("=" * 60)
    print("V4 RBRM Model: V3 APUD + Residual Boundary Refinement")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create model
    model = V4RBRM(in_channels=3, num_classes=2).to(device)
    
    # Test with standard input size
    x = torch.randn(2, 3, 384, 640).to(device)
    print(f"\nInput shape: {x.shape}")
    
    # Forward pass with all outputs
    print("\n1. Testing forward pass (full):")
    with torch.no_grad():
        outputs = model(x, return_aux=True, return_boundary=True)
    
    print(f"   Main output: {outputs['main'].shape}")
    print(f"   Boundary output: {outputs['boundary'].shape}")
    print(f"   Auxiliary outputs:")
    for i, aux in enumerate(outputs['aux']):
        print(f"     Aux-{i+1}: {aux.shape}")
    
    # Forward pass inference mode (no aux, no boundary)
    print("\n2. Testing forward pass (inference mode):")
    with torch.no_grad():
        outputs_infer = model(x, return_aux=False, return_boundary=False)
    
    print(f"   Main output: {outputs_infer['main'].shape}")
    print(f"   Has aux: {'aux' in outputs_infer}")
    print(f"   Has boundary: {'boundary' in outputs_infer}")
    
    # Parameter counts
    print("\n3. Parameter Counts:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    aspp_params = sum(p.numel() for p in model.aspp_lite.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    rbrm_params = sum(p.numel() for p in model.rbrm.parameters())
    seg_head_params = sum(p.numel() for p in model.seg_head.parameters())
    
    print(f"   Encoder: {encoder_params:,}")
    print(f"   ASPP-Lite: {aspp_params:,}")
    print(f"   APUD Decoder: {decoder_params:,}")
    print(f"   RBRM: {rbrm_params:,}")
    print(f"   Seg Head: {seg_head_params:,}")
    print(f"   ─────────────────────")
    print(f"   Total: {total_params:,}")
    print(f"   Trainable: {trainable_params:,}")
    
    # Test differential learning rates
    print("\n4. Testing Differential Learning Rates:")
    param_groups = model.get_param_groups(lr_pretrained=1e-4, lr_new=1e-3)
    for group in param_groups:
        n_params = sum(p.numel() for p in group['params'])
        print(f"   {group['name']}: {n_params:,} params @ lr={group['lr']}")
    
    # Test backward pass
    print("\n5. Testing Backward Pass:")
    outputs = model(x, return_aux=True, return_boundary=True)
    
    # Simulate loss computation
    main_loss = outputs['main'].sum()
    boundary_loss = outputs['boundary'].sum()
    aux_loss = sum(aux.sum() for aux in outputs['aux'])
    total_loss = main_loss + 0.5 * boundary_loss + aux_loss
    
    total_loss.backward()
    
    # Check gradients
    has_grad = sum(1 for p in model.parameters() if p.grad is not None)
    total_p = sum(1 for p in model.parameters())
    print(f"   Parameters with gradients: {has_grad}/{total_p}")
    print("   ✓ Backward pass successful")
    
    print("\n" + "=" * 60)
    print("All V4 RBRM tests passed!")
    print("=" * 60)
