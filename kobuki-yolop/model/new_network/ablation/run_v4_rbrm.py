#!/usr/bin/env python
r"""
Run Ablation V4 Experiment (RBRM - Residual Boundary Refinement Module)
=======================================================================

Two training strategies:
    1. End-to-end: Train everything from scratch (default)
    2. Two-stage: Load V3 weights, freeze backbone, train only RBRM

Architecture:
- Encoder: CSPDarknet (same as V1, V2, V3)
- Neck: ASPPLite (same as V2, V3)
- Decoder: APUD with SE Attention + Spatial Attention (same as V3)
- RBRM: Secondary Encoder-Decoder for boundary refinement (NEW in V4)

Usage:
    # End-to-end training
    python run_v4_rbrm.py --dataset-dir C:\path\to\CommonDataset

    # Two-stage training (RECOMMENDED)
    python run_v4_rbrm.py --dataset-dir C:\path\to\CommonDataset \
        --v3-checkpoint ./runs/ablation_v3/v3_apud/checkpoints/best.pth \
        --freeze-backbone
"""

import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_base_config
from train_ablation import AblationTrainer


def main():
    parser = argparse.ArgumentParser(description='Run Ablation V4 Experiment (RBRM)')
    
    # Basic arguments
    parser.add_argument('--dataset-dir', type=str, default="C:/Users/naren/Documents/AURASeg/CommonDataset",
                       help='Path to dataset root directory')
    parser.add_argument('--output-dir', type=str, default='./runs/ablation_v4',
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
    
    # Two-stage training arguments
    parser.add_argument('--v3-checkpoint', type=str, default=None,
                       help='Path to V3 checkpoint for two-stage training')
    parser.add_argument('--pretrained-checkpoint', type=str, default=None,
                       help='Path to full V4 checkpoint for fine-tuning')
    parser.add_argument('--freeze-backbone', action='store_true',
                       help='Freeze encoder/ASPP/APUD, train only RBRM')
    
    # Deep supervision (disabled by default for this ablation)
    parser.add_argument('--deep-supervision', action='store_true',
                       help='Enable deep supervision (auxiliary outputs)')
    
    # Boundary-weighted loss for RBRM (Idea 1)
    parser.add_argument('--boundary-loss', action='store_true',
                       help='Use boundary-weighted loss (emphasizes edges)')
    parser.add_argument('--boundary-width', type=int, default=5,
                       help='Width of boundary region in pixels')
    parser.add_argument('--boundary-weight', type=float, default=5.0,
                       help='Weight multiplier for boundary pixels')
    parser.add_argument('--interior-weight', type=float, default=0.5,
                       help='Weight for interior pixels')
    
    args = parser.parse_args()
    
    # Determine training mode
    if args.v3_checkpoint and args.freeze_backbone:
        training_mode = "Two-Stage (Frozen Backbone)"
        experiment_name = "v4_rbrm_frozen"
    elif args.pretrained_checkpoint:
        training_mode = "Fine-tune V4"
        experiment_name = "v4_rbrm_finetune"
    elif args.v3_checkpoint:
        training_mode = "Fine-tune from V3"
        experiment_name = "v4_rbrm_finetune"
    else:
        training_mode = "End-to-End"
        experiment_name = "v4_rbrm"
    
    # Get base config and modify for V4
    config = get_base_config()
    
    # Model: V4 with RBRM
    config.model.model_name = "ablation_v4_rbrm"
    config.logging.experiment_name = experiment_name
    
    # Store freeze_backbone in config for model creation
    config.model.freeze_backbone = args.freeze_backbone
    
    # Deep supervision (disabled by default)
    config.model.deep_supervision = args.deep_supervision
    config.loss.use_deep_supervision = args.deep_supervision
    
    # Boundary-weighted loss configuration
    config.loss.use_boundary_loss = args.boundary_loss
    config.loss.boundary_width = args.boundary_width
    config.loss.boundary_weight = args.boundary_weight
    config.loss.interior_weight = args.interior_weight
    
    # Dataset
    config.data.dataset_dir = args.dataset_dir
    
    # Training
    config.training.epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.lr
    config.training.num_workers = args.num_workers
    config.training.use_amp = not args.no_amp
    config.data.use_augmentation = not args.no_augmentation
    
    # Output
    config.logging.output_dir = args.output_dir
    
    # Print config
    print("\n" + "="*60)
    print("Ablation V4: RBRM (Residual Boundary Refinement Module)")
    print("="*60)
    print(f"Training Mode: {training_mode}")
    print(f"Model: {config.model.model_name}")
    print(f"Epochs: {config.training.epochs}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Learning rate: {config.training.learning_rate}")
    print(f"Dataset: {config.data.dataset_dir}")
    print(f"Output: {config.logging.output_dir}")
    print(f"AMP: {config.training.use_amp}")
    print(f"Augmentation: {config.data.use_augmentation}")
    if args.v3_checkpoint:
        print(f"V3 Checkpoint: {args.v3_checkpoint}")
    if args.pretrained_checkpoint:
        print(f"Pretrained Checkpoint: {args.pretrained_checkpoint}")
    print(f"Freeze Backbone: {args.freeze_backbone}")
    print(f"Deep Supervision: {config.model.deep_supervision}")
    print(f"Boundary-Weighted Loss: {config.loss.use_boundary_loss}")
    if config.loss.use_boundary_loss:
        print(f"  Boundary Width: {config.loss.boundary_width}px")
        print(f"  Boundary Weight: {config.loss.boundary_weight}")
        print(f"  Interior Weight: {config.loss.interior_weight}")
    print("="*60 + "\n")
    
    # Initialize trainer
    trainer = AblationTrainer(config)
    
    # Load full V4 checkpoint if provided (fine-tuning)
    if args.pretrained_checkpoint:
        if not os.path.exists(args.pretrained_checkpoint):
            print(f"[ERROR] Pretrained checkpoint not found: {args.pretrained_checkpoint}")
            return
        
        print(f"\n[V4] Loading full V4 weights from: {args.pretrained_checkpoint}")
        checkpoint = torch.load(args.pretrained_checkpoint, map_location='cpu', weights_only=False)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        trainer.model.load_state_dict(state_dict, strict=True)
        print(f"[V4] Successfully loaded full model state.")

    # Load V3 weights if provided (two-stage training)
    elif args.v3_checkpoint:
        if not os.path.exists(args.v3_checkpoint):
            print(f"[ERROR] V3 checkpoint not found: {args.v3_checkpoint}")
            return
        
        print(f"\n[V4] Loading V3 weights from: {args.v3_checkpoint}")
        trainer.model.load_v3_weights(args.v3_checkpoint)
        
        if args.freeze_backbone:
            # Freeze backbone (encoder, neck, decoder)
            trainer.model.freeze_backbone()
            
            # Recreate optimizer with only trainable parameters (RBRM only)
            trainable_params = trainer.model.get_trainable_params()
            trainer.optimizer = trainer._build_optimizer_with_params(trainable_params)
            print(f"[V4] Optimizer recreated with {len(trainable_params)} trainable parameter tensors")
    
    # Print parameter counts
    params = trainer.model.get_params_count()
    print(f"\n[V4] Model Parameters:")
    print(f"  Total: {params['total_M']:.2f}M")
    print(f"  Trainable: {params['trainable_M']:.2f}M")
    print(f"  Encoder: {params['encoder_M']:.2f}M")
    print(f"  Neck (ASPP): {params['neck_M']:.2f}M")
    print(f"  Decoder (APUD): {params['decoder_M']:.2f}M")
    print(f"  RBRM: {params['rbrm_M']:.2f}M")
    
    # Train
    print("\nStarting training...")
    trainer.train()
    
    print("\n✅ V4 Training complete!")
    print(f"Results saved to: {config.logging.output_dir}")


if __name__ == "__main__":
    main()
