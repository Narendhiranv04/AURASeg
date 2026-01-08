#!/usr/bin/env python
r"""
Run Base Ablation Experiment (V1)
=================================
Quick-start script to run the base ablation model training.

Usage:
    python run_base_ablation.py --image-dir /path/to/images --mask-dir /path/to/masks
    
For Windows:
    python run_base_ablation.py --image-dir C:\path\to\images --mask-dir C:\path\to\masks
"""

import os
import sys
import argparse

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_base_config
from train_ablation import AblationTrainer


def main():
    parser = argparse.ArgumentParser(description='Run Base Ablation Experiment')
    
    # Dataset path
    parser.add_argument('--dataset-dir', type=str, default="C:/Users/naren/Documents/AURASeg/CommonDataset",
                       help='Path to dataset root directory')
    
    # Optional overrides
    parser.add_argument('--output-dir', type=str, default='./runs/ablation',
                       help='Output directory for results')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4,
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
    
    # Get base config
    config = get_base_config()
    
    # Update with command line args
    config.data.dataset_dir = args.dataset_dir
    config.data.num_workers = args.num_workers
    config.data.use_augmentation = not args.no_augmentation
    
    config.training.epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.use_amp = not args.no_amp
    
    config.optimizer.learning_rate = args.lr
    
    config.logging.output_dir = args.output_dir
    config.logging.experiment_name = 'v1_base'
    
    # Print configuration summary
    print("\n" + "="*60)
    print("Base Ablation Experiment Configuration")
    print("="*60)
    print(f"Model: {config.model.model_name}")
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
    
    print("\n" + "="*60)
    print("Training Complete!")
    print(f"Results saved to: {config.get_output_path()}")
    print("="*60)


if __name__ == '__main__':
    main()
