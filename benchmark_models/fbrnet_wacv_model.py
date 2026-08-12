"""
Source: little5570/FBRNet
upstream commit: 04f2bf7209d78035019edc8c25bab0d02bd0439f
changes: binary-class generalization + device portability only (Laplacian kernel buffer)
"""

import sys
from pathlib import Path
import torch.nn as nn

# Add FBRNet_official to path to resolve its internal imports cleanly
repo_root = Path(__file__).parent.parent
fbrnet_root = repo_root / "external_baselines" / "FBRNet_official"
if str(fbrnet_root) not in sys.path:
    sys.path.insert(0, str(fbrnet_root))

from lib.models.model import res18PaNew7Brm

class FBRNet_WACV(nn.Module):
    """
    Clean wrapper for FBRNet specifically instantiated for WACV baseline experiments.
    Supports binary segmentation adaptation and cleanly exposes parameters.
    """
    def __init__(self, num_classes: int = 2, aux_mode: str = 'train'):
        super().__init__()
        self.num_classes = num_classes
        self.aux_mode = aux_mode
        self.model = res18PaNew7Brm(n_classes=num_classes, aux_mode=aux_mode)
        
    def forward(self, x):
        """
        Outputs: 
        Train mode: feat_ffm, feat_out32, feat_out16, aux_0, aux_1
        Eval mode: feat_ffm
        """
        return self.model(x)
        
    def get_params(self):
        """
        Passes through FBRNet's native parameter getter which separates 
        weight decay and learning rate multipliers natively.
        """
        return self.model.get_params()
