"""
Configuration for AURASeg Ablation Study
Paper: AURASeg - Attention Guided Upsampling with Residual Boundary-Assistive Refinement

All paths and hyperparameters for training V1 and V2 models.
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class DataConfig:
    """Dataset configuration"""
    # Paths (relative to workspace root: c:\Users\naren\Documents\AURASeg)
    image_dir: str = "CommonDataset/images"
    mask_dir: str = "CommonDataset/labels"
    
    # Image settings
    img_size: Tuple[int, int] = (384, 640)  # (H, W)
    num_classes: int = 2  # drivable / non-drivable
    
    # Data split
    train_ratio: float = 0.8
    val_ratio: float = 0.2
    
    # Data loading
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    # Input
    in_channels: int = 3
    num_classes: int = 2
    
    # Encoder (CSPDarknet-53)
    encoder_channels: List[int] = field(default_factory=lambda: [64, 128, 256, 512, 1024])
    
    # ASPP-Lite (for V2)
    aspp_branch_channels: int = 128
    aspp_out_channels: int = 256
    aspp_dilations: List[int] = field(default_factory=lambda: [1, 1, 6, 12])
    
    # SPP (for V1)
    spp_kernels: List[int] = field(default_factory=lambda: [5, 9, 13])
    spp_out_channels: int = 256
    
    # Decoder
    decoder_channels: List[int] = field(default_factory=lambda: [256, 128, 64, 32])


@dataclass
class TrainConfig:
    """Training configuration"""
    # Training params
    epochs: int = 50
    batch_size: int = 8
    
    # Optimizer
    optimizer: str = "adamw"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    
    # Learning rate scheduler
    scheduler: str = "cosine"  # 'cosine', 'step', 'poly'
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    
    # Loss function
    loss_type: str = "combined"  # 'focal', 'dice', 'combined'
    focal_weight: float = 1.0
    dice_weight: float = 1.0
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    
    # Regularization
    dropout: float = 0.1
    
    # Checkpointing
    save_every: int = 5
    val_every: int = 1
    
    # Early stopping
    patience: int = 15
    min_delta: float = 0.001
    
    # Mixed precision
    use_amp: bool = True
    
    # Gradient clipping
    grad_clip: float = 1.0


@dataclass
class AugmentConfig:
    """Data augmentation configuration"""
    # Geometric
    random_flip: bool = True
    flip_prob: float = 0.5
    
    random_rotate: bool = True
    rotate_limit: int = 15
    
    random_scale: bool = True
    scale_range: Tuple[float, float] = (0.8, 1.2)
    
    # Photometric
    random_brightness: bool = True
    brightness_limit: float = 0.2
    
    random_contrast: bool = True
    contrast_limit: float = 0.2
    
    # Normalization
    normalize: bool = True
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass
class AblationConfig:
    """Complete ablation study configuration"""
    # Experiment info
    experiment_name: str = "ablation_11_dec"
    model_version: str = "v1_base_spp"  # or "v2_base_assplite"
    
    # Output directory
    output_dir: str = "kobuki-yolop/model/new_network/ABLATION_11_DEC/runs"
    
    # Sub-configs
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    
    # Device
    device: str = "cuda"
    
    # Reproducibility
    seed: int = 42
    
    def __post_init__(self):
        """Create output directories"""
        self.run_dir = os.path.join(self.output_dir, self.model_version)
        self.checkpoint_dir = os.path.join(self.run_dir, "checkpoints")
        self.log_dir = os.path.join(self.run_dir, "logs")
        self.vis_dir = os.path.join(self.run_dir, "visualizations")


def get_v1_config():
    """Get configuration for V1 (CSPDarknet + SPP)"""
    config = AblationConfig(
        experiment_name="ablation_11_dec",
        model_version="v1_base_spp"
    )
    return config


def get_v2_config():
    """Get configuration for V2 (CSPDarknet + ASPP-Lite)"""
    config = AblationConfig(
        experiment_name="ablation_11_dec",
        model_version="v2_base_assplite"
    )
    return config


if __name__ == "__main__":
    # Print configurations
    print("=" * 60)
    print("Ablation Study Configurations")
    print("=" * 60)
    
    print("\n[V1 Configuration - CSPDarknet + SPP]")
    v1_config = get_v1_config()
    print(f"  Model: {v1_config.model_version}")
    print(f"  Output: {v1_config.run_dir}")
    print(f"  Epochs: {v1_config.train.epochs}")
    print(f"  Batch Size: {v1_config.train.batch_size}")
    print(f"  Learning Rate: {v1_config.train.learning_rate}")
    print(f"  Loss: {v1_config.train.loss_type}")
    
    print("\n[V2 Configuration - CSPDarknet + ASPP-Lite]")
    v2_config = get_v2_config()
    print(f"  Model: {v2_config.model_version}")
    print(f"  Output: {v2_config.run_dir}")
    print(f"  Epochs: {v2_config.train.epochs}")
    print(f"  Batch Size: {v2_config.train.batch_size}")
    print(f"  Learning Rate: {v2_config.train.learning_rate}")
    print(f"  Loss: {v2_config.train.loss_type}")
    
    print("\n" + "=" * 60)
