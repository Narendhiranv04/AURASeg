"""
Source: YangParky/BASeg
Upstream Commit: e88e958fa5f44a26995ec1dd9949291c89449d8d
Paper: "BASeg: Boundary Aware Semantic Segmentation for Autonomous Driving", Neural Networks, 2023.
Adaptation: Binary-class generalization (num_classes=2) and device/AMP portability. Architecture preserved.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn

# Add external_baselines/BASeg to path for internal imports
repo_root = Path(__file__).parent.parent
baseg_root = repo_root / "external_baselines" / "BASeg"
if str(baseg_root) not in sys.path:
    sys.path.insert(0, str(baseg_root))

from model.baseg import BASeg


class BASeg_WACV(nn.Module):
    """
    Clean wrapper for BASeg specifically instantiated for WACV baseline experiments.
    Supports binary segmentation adaptation and cleanly exposes parameter groups for AdamW.
    """
    def __init__(self, num_classes: int = 2, layers: int = 101, pretrained: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.layers = layers
        self.model = BASeg(
            num_classes=num_classes,
            layers=layers,
            multi_grid=(1, 1, 1),
            in_channels=[256, 512, 1024, 2048],
            embed_dim=512,
            criterion=None,
            pretrained=pretrained
        )

    def forward(self, x, canny=None):
        """
        Forward pass:
        - In train mode: returns (main_logits, aux_logits, edge_logits)
          - main_logits: [B, num_classes, H, W]
          - aux_logits:  [B, num_classes, H, W]
          - edge_logits: [B, 1, H, W]
        - In eval mode: returns (main_logits, edge_logits)
          - main_logits: [B, num_classes, H, W]
        """
        return self.model(x)

    def get_param_groups(self, lr_encoder: float = 1e-4, lr_decoder: float = 1e-3, weight_decay: float = 0.01):
        """
        Separates backbone encoder parameters (lr_encoder) from newly initialized
        decoder/boundary/ASPP/CAM/head modules (lr_decoder).
        """
        encoder_params = []
        decoder_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'layer0' in name or 'layer1' in name or 'layer2' in name or 'layer3' in name or 'layer4' in name:
                encoder_params.append(param)
            else:
                decoder_params.append(param)

        return [
            {'params': encoder_params, 'lr': lr_encoder, 'weight_decay': weight_decay},
            {'params': decoder_params, 'lr': lr_decoder, 'weight_decay': weight_decay}
        ]
