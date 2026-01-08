"""
Configuration for Ablation Study
================================
Centralized configuration management for all ablation experiments.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import json


@dataclass
class DataConfig:
    """Dataset configuration."""
    # Dataset root directory (CommonDataset)
    dataset_dir: str = "C:/Users/naren/Documents/AURASeg/CommonDataset"
    
    # Optional overrides (if not using standard structure)
    train_image_dir: Optional[str] = None
    train_mask_dir: Optional[str] = None
    val_image_dir: Optional[str] = None
    val_mask_dir: Optional[str] = None
    
    # Image settings
    input_height: int = 640
    input_width: int = 384
    num_classes: int = 2
    
    # Data augmentation
    use_augmentation: bool = True
    horizontal_flip_prob: float = 0.5
    vertical_flip_prob: float = 0.0
    random_crop: bool = False
    color_jitter: bool = True
    
    # Data loading
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class LossConfig:
    """Loss function configuration."""
    # Loss weights
    lambda_focal: float = 1.0
    lambda_dice: float = 1.0
    
    # Focal loss parameters
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    
    # Deep supervision weights (coarse to fine)
    # Decaying schedule: higher weights for finer scales
    supervision_weights: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4])
    
    # For base model (single scale), this is ignored
    use_deep_supervision: bool = False
    
    # Ignore index for loss computation
    ignore_index: int = 255


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""
    # Optimizer
    optimizer: str = "adamw"  # 'adam', 'adamw', 'sgd'
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    momentum: float = 0.9  # For SGD
    betas: Tuple[float, float] = (0.9, 0.999)  # For Adam/AdamW
    
    # Learning rate scheduler
    scheduler: str = "cosine"  # 'cosine', 'step', 'poly', 'none'
    warmup_epochs: int = 5
    warmup_lr: float = 1e-6
    min_lr: float = 1e-6
    
    # For step scheduler
    step_size: int = 30
    gamma: float = 0.1
    
    # For poly scheduler
    poly_power: float = 0.9


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Training settings
    epochs: int = 100
    batch_size: int = 4
    accumulation_steps: int = 1  # Gradient accumulation
    
    # Mixed precision
    use_amp: bool = True
    
    # Checkpointing
    save_every: int = 5
    save_best: bool = True
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 20
    min_delta: float = 1e-4
    
    # Validation
    val_every: int = 1
    val_split: float = 0.1
    
    # Random seed
    seed: int = 42
    
    # Resume training
    resume: Optional[str] = None


@dataclass
class ModelConfig:
    """Model configuration."""
    # Model variant
    model_name: str = "ablation_v1_base"  # 'ablation_v1_base', 'ablation_v2_assplite', etc.
    
    # Architecture settings
    num_classes: int = 2
    in_channels: int = 3
    
    # Pretrained weights
    pretrained: bool = False
    pretrained_path: Optional[str] = None
    
    # Freeze settings
    freeze_encoder: bool = False
    freeze_decoder: bool = False


@dataclass 
class LoggingConfig:
    """Logging and output configuration."""
    # Output directories
    output_dir: str = "./runs/ablation"
    experiment_name: str = "v1_base"
    
    # Logging
    log_every: int = 10
    use_tensorboard: bool = True
    use_wandb: bool = False
    wandb_project: str = "auraseg-ablation"
    
    # Visualization
    save_predictions: bool = True
    save_predictions_every: int = 10
    num_visualization_samples: int = 4


@dataclass
class AblationConfig:
    """Complete configuration for ablation study."""
    data: DataConfig = field(default_factory=DataConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    def save(self, path: str):
        """Save configuration to JSON file."""
        config_dict = {
            'data': self.data.__dict__,
            'loss': self.loss.__dict__,
            'optimizer': self.optimizer.__dict__,
            'training': self.training.__dict__,
            'model': self.model.__dict__,
            'logging': self.logging.__dict__
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'AblationConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        config = cls()
        
        for key, value in config_dict.get('data', {}).items():
            if hasattr(config.data, key):
                setattr(config.data, key, value)
        
        for key, value in config_dict.get('loss', {}).items():
            if hasattr(config.loss, key):
                setattr(config.loss, key, value)
                
        for key, value in config_dict.get('optimizer', {}).items():
            if hasattr(config.optimizer, key):
                setattr(config.optimizer, key, value)
                
        for key, value in config_dict.get('training', {}).items():
            if hasattr(config.training, key):
                setattr(config.training, key, value)
                
        for key, value in config_dict.get('model', {}).items():
            if hasattr(config.model, key):
                setattr(config.model, key, value)
                
        for key, value in config_dict.get('logging', {}).items():
            if hasattr(config.logging, key):
                setattr(config.logging, key, value)
        
        return config
    
    def get_output_path(self) -> str:
        """Get full output path for this experiment."""
        return os.path.join(self.logging.output_dir, self.logging.experiment_name)


def get_base_config() -> AblationConfig:
    """Get configuration for V1 Base ablation model."""
    config = AblationConfig()
    
    # Model settings
    config.model.model_name = "ablation_v1_base"
    config.model.num_classes = 2
    
    # Loss settings (no deep supervision for base)
    config.loss.use_deep_supervision = False
    config.loss.lambda_focal = 1.0
    config.loss.lambda_dice = 1.0
    
    # Training settings
    config.training.epochs = 100
    config.training.batch_size = 4
    config.training.use_amp = True
    
    # Optimizer settings
    config.optimizer.optimizer = "adamw"
    config.optimizer.learning_rate = 1e-4
    config.optimizer.scheduler = "cosine"
    config.optimizer.warmup_epochs = 5
    
    # Logging
    config.logging.experiment_name = "v1_base"
    
    return config


def get_assplite_config() -> AblationConfig:
    """Get configuration for V2 ASSPLite ablation model."""
    config = get_base_config()
    
    config.model.model_name = "ablation_v2_assplite"
    config.logging.experiment_name = "v2_assplite"
    
    return config


def get_apud_config() -> AblationConfig:
    """Get configuration for V3 APUD ablation model."""
    config = get_base_config()
    
    config.model.model_name = "ablation_v3_apud"
    config.logging.experiment_name = "v3_apud"
    
    # Enable deep supervision for APUD
    config.loss.use_deep_supervision = True
    config.loss.supervision_weights = [0.1, 0.2, 0.3, 0.4]
    
    return config


def get_rbrm_config() -> AblationConfig:
    """Get configuration for V4 RBRM (full) ablation model."""
    config = get_apud_config()
    
    config.model.model_name = "ablation_v4_rbrm"
    config.logging.experiment_name = "v4_rbrm"
    
    return config


# Mapping of model names to configs
CONFIG_REGISTRY = {
    "ablation_v1_base": get_base_config,
    "ablation_v2_assplite": get_assplite_config,
    "ablation_v3_apud": get_apud_config,
    "ablation_v4_rbrm": get_rbrm_config,
}


def get_config(model_name: str) -> AblationConfig:
    """Get configuration by model name."""
    if model_name not in CONFIG_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(CONFIG_REGISTRY.keys())}")
    return CONFIG_REGISTRY[model_name]()


if __name__ == '__main__':
    # Test configuration
    print("Testing configuration system...")
    
    config = get_base_config()
    print(f"\nBase config:")
    print(f"  Model: {config.model.model_name}")
    print(f"  Epochs: {config.training.epochs}")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Learning rate: {config.optimizer.learning_rate}")
    print(f"  Deep supervision: {config.loss.use_deep_supervision}")
    print(f"  Output path: {config.get_output_path()}")
    
    # Save and reload
    config.save("./test_config.json")
    loaded_config = AblationConfig.load("./test_config.json")
    print(f"\nLoaded config model: {loaded_config.model.model_name}")
    
    # Cleanup
    import os
    os.remove("./test_config.json")
    
    print("\nAll configs:")
    for name in CONFIG_REGISTRY:
        cfg = get_config(name)
        print(f"  {name}: deep_supervision={cfg.loss.use_deep_supervision}")
