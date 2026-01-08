"""
Ablation Study Package
======================
Contains models, losses, and training utilities for ablation experiments.

Models:
- V1 Base: CSPDarknet + SPP + Simple decoder
- V2 ASSPLite: V1 + Lightweight ASPP (to be added)
- V3 APUD: V2 + Attention Pyramid Upsampling Decoder (to be added)
- V4 RBRM: V3 + Residual Boundary Refinement Module (to be added)
"""

from .ablation_v1_base import AblationBaseModel, build_ablation_base
from .losses import FocalLoss, DiceLoss, CombinedLoss, SingleScaleLoss
from .config import (
    AblationConfig,
    DataConfig,
    LossConfig,
    OptimizerConfig,
    TrainingConfig,
    ModelConfig,
    LoggingConfig,
    get_config,
    get_base_config,
)

__all__ = [
    # Models
    'AblationBaseModel',
    'build_ablation_base',
    
    # Losses
    'FocalLoss',
    'DiceLoss',
    'CombinedLoss',
    'SingleScaleLoss',
    
    # Config
    'AblationConfig',
    'DataConfig',
    'LossConfig',
    'OptimizerConfig',
    'TrainingConfig',
    'ModelConfig',
    'LoggingConfig',
    'get_config',
    'get_base_config',
]

__version__ = '1.0.0'
