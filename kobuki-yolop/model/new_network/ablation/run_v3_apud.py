#!/usr/bin/env python
r"""
Run Ablation V3 Experiment (APUD - Attention Progressive Upsampling Decoder)
=============================================================================
Training script for the third ablation model with APUD decoder.

Architecture:
- Encoder: CSPDarknet (same as V1, V2)
- Neck: ASPPLite (same as V2)
- Decoder: APUD with SE Attention + Spatial Attention (NEW in V3)

Usage:
    python run_v3_apud.py --dataset-dir C:\path\to\CommonDataset
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_base_config
from train_ablation import AblationTrainer


def main():
    parser = argparse.ArgumentParser(description='Run Ablation V3 Experiment (APUD)')
    
    parser.add_argument('--dataset-dir', type=str, default="C:/Users/naren/Documents/AURASeg/CommonDataset",
                       help='Path to dataset root directory')
    parser.add_argument('--output-dir', type=str, default='./runs/ablation_v3',
                       help='Output directory for results')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--no-amp', action='store_true',
                       help='Disable mixed precision training')
    parser.add_argument('--no-augmentation', action='store_true',
                       help='Disable data augmentation')
    
    args = parser.parse_args()
    
    # Get base config and modify for V3
    config = get_base_config()
    
    # Model: V3 with APUD decoder
    config.model.model_name = "ablation_v3_apud"
    config.logging.experiment_name = "v3_apud"
    
    # Dataset
    config.data.dataset_dir = args.dataset_dir
    
    # Training
    config.training.epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.optimizer.learning_rate = args.lr
    config.data.num_workers = args.num_workers
    config.training.use_amp = not args.no_amp
    config.data.use_augmentation = not args.no_augmentation
    
    # Output directory
    config.output_dir = args.output_dir
    
    print("\n" + "="*60)
    print("Ablation V3 Experiment Configuration (APUD)")
    print("="*60)
    print(f"Model: {config.model.model_name}")
    print(f"  - Encoder: CSPDarknet")
    print(f"  - Neck: ASPPLite")
    print(f"  - Decoder: APUD (SE Attention + Spatial Attention)")
    print(f"Dataset dir: {config.data.dataset_dir}")
    print(f"Epochs: {config.training.epochs}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Learning rate: {config.optimizer.learning_rate}")
    print(f"Mixed precision: {config.training.use_amp}")
    print(f"Augmentation: {config.data.use_augmentation}")
    print(f"Output dir: {config.get_output_path()}")
    print("="*60 + "\n")
    
    # Create trainer and train
    trainer = AblationTrainer(config)
    trainer.train()
    
    print("\nTraining completed!")
    print(f"Results saved to: {config.get_output_path()}")


if __name__ == "__main__":
    main()
