"""
Benchmark Model Training Script
================================

Train state-of-the-art segmentation models for comparison with AURASeg.

Models:
    - deeplabv3plus: DeepLabV3+ with ResNet-50 (ECCV 2018)
    - segformer: SegFormer with MiT-B2 (NeurIPS 2021)
    - upernet: UPerNet with Swin-Tiny (ECCV 2018 + Swin)
    - dpt: DPT with ViT-Base (ICCV 2021)
    - mask2former: Mask2Former with Swin-Small (CVPR 2022)

Training Setup (matches AURASeg V4):
    - Optimizer: AdamW (lr=1e-3, weight_decay=1e-4)
    - Scheduler: Cosine annealing (min_lr=1e-6)
    - Loss: Focal + Dice (or model's built-in loss for Mask2Former)
    - Epochs: 50
    - Batch size: 8
    - Image size: 384 x 640
    - Mixed precision: Enabled

Usage:
    python train_benchmark.py --model deeplabv3plus --epochs 50 --batch-size 8
    python train_benchmark.py --model segformer
    python train_benchmark.py --model upernet
    python train_benchmark.py --model dpt
    python train_benchmark.py --model mask2former

All checkpoints saved to: ABLATION_11_DEC/runs/benchmark_{model_name}/
"""

import os
import sys
import argparse
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DataConfig, TrainConfig, AugmentConfig
from utils import (
    DrivableAreaDataset, compute_metrics, set_seed,
    AverageMeter, EarlyStopping, save_checkpoint
)
from losses import CombinedLoss
from benchmark_models import get_benchmark_model, BENCHMARK_MODELS


class BenchmarkTrainer:
    """
    Trainer for benchmark SOTA models.
    
    Uses identical training setup as AURASeg V4 for fair comparison:
        - Same dataset, augmentation, optimizer, scheduler
        - Same loss function (Focal + Dice) except for Mask2Former
        - Same epochs, batch size, image size
    """
    
    def __init__(self, model_name: str, args):
        self.model_name = model_name
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        
        # Set random seed
        set_seed(42)
        
        # Config
        self.data_config = DataConfig()
        self.train_config = TrainConfig()
        self.augment_config = AugmentConfig()
        
        # Override with args
        if args.epochs:
            self.train_config.epochs = args.epochs
        if args.batch_size:
            self.train_config.batch_size = args.batch_size
        if args.lr:
            self.train_config.learning_rate = args.lr
        
        # Setup output directories
        self._setup_directories()
        
        # Initialize components
        self.model, self.model_info = self._build_model()
        self.criterion = self._build_criterion()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.scaler = GradScaler('cuda') if self.train_config.use_amp else None
        
        # Data loaders
        self._setup_data()
        
        # Logging
        self.writer = SummaryWriter(self.log_dir)
        self.best_miou = 0.0
        self.start_epoch = 1
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=self.train_config.patience,
            min_delta=self.train_config.min_delta
        )
        
        # Resume if specified
        if args.resume:
            self._load_checkpoint(args.resume)
    
    def _setup_directories(self):
        """Create output directories."""
        base_dir = Path(__file__).parent.parent / "runs"
        self.run_dir = base_dir / f"benchmark_{self.model_name}"
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.log_dir = self.run_dir / "logs"
        self.vis_dir = self.run_dir / "visualizations"
        
        for d in [self.checkpoint_dir, self.log_dir, self.vis_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        print(f"[INFO] Output directory: {self.run_dir}")
    
    def _build_model(self):
        """Create benchmark model."""
        model, info = get_benchmark_model(
            self.model_name,
            num_classes=self.data_config.num_classes,
            pretrained=True
        )
        model = model.to(self.device)
        
        print(f"[INFO] Model: {info['name']} ({info['encoder']})")
        print(f"[INFO] Parameters: {info['params_millions']:.2f}M")
        
        return model, info
    
    def _build_criterion(self):
        """Build loss function."""
        if self.model_name == 'mask2former':
            # Mask2Former uses its own built-in loss
            return None
        else:
            # Standard Focal + Dice loss
            return CombinedLoss(
                focal_weight=self.train_config.focal_weight,
                dice_weight=self.train_config.dice_weight,
                focal_alpha=self.train_config.focal_alpha,
                focal_gamma=self.train_config.focal_gamma
            ).to(self.device)
    
    def _build_optimizer(self):
        """Build optimizer."""
        return optim.AdamW(
            self.model.parameters(),
            lr=self.train_config.learning_rate,
            weight_decay=self.train_config.weight_decay
        )
    
    def _build_scheduler(self):
        """Build learning rate scheduler."""
        return optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.train_config.epochs,
            eta_min=self.train_config.min_lr
        )
    
    def _setup_data(self):
        """Setup data loaders."""
        # Training dataset
        train_dataset = DrivableAreaDataset(
            image_dir=self.data_config.image_dir,
            mask_dir=self.data_config.mask_dir,
            img_size=self.data_config.img_size,
            split='train',
            transform=True,
            mean=self.augment_config.mean,
            std=self.augment_config.std
        )
        
        # Validation dataset
        val_dataset = DrivableAreaDataset(
            image_dir=self.data_config.image_dir,
            mask_dir=self.data_config.mask_dir,
            img_size=self.data_config.img_size,
            split='val',
            transform=False,
            mean=self.augment_config.mean,
            std=self.augment_config.std
        )
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.train_config.batch_size,
            shuffle=True,
            num_workers=self.data_config.num_workers,
            pin_memory=self.data_config.pin_memory,
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.train_config.batch_size,
            shuffle=False,
            num_workers=self.data_config.num_workers,
            pin_memory=self.data_config.pin_memory
        )
        
        print(f"[INFO] Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    def _load_checkpoint(self, path: str):
        """Load checkpoint for resuming training."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_miou = checkpoint.get('best_miou', 0.0)
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'scaler_state_dict' in checkpoint and self.scaler:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"[INFO] Resumed from epoch {self.start_epoch - 1}, best mIoU: {self.best_miou:.4f}")
    
    def train(self):
        """Main training loop."""
        print("\n" + "=" * 70)
        print(f"BENCHMARK TRAINING: {self.model_info['name']}")
        print("=" * 70)
        print(f"Model: {self.model_info['name']} ({self.model_info['encoder']})")
        print(f"Paradigm: {self.model_info['paradigm']}")
        print(f"Paper: {self.model_info['paper']}")
        print(f"Device: {self.device}")
        print(f"Parameters: {self.model_info['params_millions']:.2f}M")
        print(f"Epochs: {self.train_config.epochs}")
        print(f"Batch size: {self.train_config.batch_size}")
        print(f"Learning rate: {self.train_config.learning_rate}")
        print(f"Image size: {self.data_config.img_size}")
        print(f"Loss: {'Built-in (Mask2Former)' if self.model_name == 'mask2former' else 'Focal + Dice'}")
        print("=" * 70)
        
        print(f"\nStarting training from epoch {self.start_epoch}\n")
        
        for epoch in range(self.start_epoch, self.train_config.epochs + 1):
            print(f"\n{'=' * 60}")
            print(f"Epoch {epoch}/{self.train_config.epochs}")
            print(f"{'=' * 60}")
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate(epoch)
            
            # Update scheduler
            self.scheduler.step()
            
            # Log to tensorboard
            self.writer.add_scalar('train/loss', train_loss, epoch)
            self.writer.add_scalar('val/miou', val_metrics['miou'], epoch)
            self.writer.add_scalar('val/iou_drivable', val_metrics['iou_drivable'], epoch)
            self.writer.add_scalar('val/dice', val_metrics['mdice'], epoch)
            self.writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Print epoch summary
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"\n[TRAIN] Loss: {train_loss:.4f}, LR: {current_lr:.6f}")
            print(f"[VAL] mIoU: {val_metrics['miou']:.4f}, "
                  f"IoU (Drivable): {val_metrics['iou_drivable']:.4f}, "
                  f"Dice: {val_metrics['mdice']:.4f}")
            
            # Save checkpoint
            is_best = val_metrics['miou'] > self.best_miou
            if is_best:
                self.best_miou = val_metrics['miou']
                print(f"  *** New best mIoU: {self.best_miou:.4f} ***")
            
            self._save_checkpoint(epoch, is_best)
            
            # Early stopping check
            if self.early_stopping(val_metrics['miou']):
                print(f"\n[INFO] Early stopping triggered at epoch {epoch}")
                break
        
        # Save final checkpoint
        self._save_checkpoint(epoch, filename='final.pth')
        
        print("\n" + "=" * 60)
        print(f"TRAINING COMPLETE - {self.model_info['name']}")
        print(f"Best mIoU: {self.best_miou:.4f}")
        print(f"Checkpoints: {self.checkpoint_dir}")
        print("=" * 60)
        
        self.writer.close()
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        loss_meter = AverageMeter()
        
        pbar = tqdm(self.train_loader, desc=f"Train Epoch {epoch}", leave=False)
        
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.train_config.use_amp:
                with autocast('cuda'):
                    if self.model_name == 'mask2former':
                        # Mask2Former with built-in loss
                        outputs = self.model(images, labels=masks)
                        loss = outputs['loss']
                    else:
                        # Standard forward + loss
                        outputs = self.model(images)
                        loss = self.criterion(outputs, masks)
                
                self.scaler.scale(loss).backward()
                
                if self.train_config.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.train_config.grad_clip)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                if self.model_name == 'mask2former':
                    outputs = self.model(images, labels=masks)
                    loss = outputs['loss']
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                
                loss.backward()
                
                if self.train_config.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.train_config.grad_clip)
                
                self.optimizer.step()
            
            loss_meter.update(loss.item(), images.size(0))
            pbar.set_postfix({'loss': loss_meter.avg})
        
        return loss_meter.avg
    
    @torch.no_grad()
    def validate(self, epoch: int) -> dict:
        """Validate model."""
        self.model.eval()
        all_metrics = []
        
        for images, masks in tqdm(self.val_loader, desc="Validating", leave=False):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Get predictions
            if self.model_name == 'mask2former':
                outputs = self.model.get_logits(images)
            else:
                outputs = self.model(images)
            
            preds = torch.argmax(outputs, dim=1)
            
            # Compute metrics
            metrics = compute_metrics(preds.cpu(), masks.cpu(), num_classes=2)
            all_metrics.append(metrics)
        
        # Average metrics
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)
        
        return avg_metrics
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False, filename: str = None):
        """Save checkpoint."""
        state = {
            'epoch': epoch,
            'model_name': self.model_name,
            'model_info': self.model_info,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_miou': self.best_miou,
        }
        
        if self.scheduler:
            state['scheduler_state_dict'] = self.scheduler.state_dict()
        if self.scaler:
            state['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save latest
        if filename:
            save_path = self.checkpoint_dir / filename
        else:
            save_path = self.checkpoint_dir / 'latest.pth'
        save_checkpoint(state, str(save_path))
        
        # Save best
        if is_best:
            best_path = self.checkpoint_dir / 'best.pth'
            save_checkpoint(state, str(best_path))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Benchmark Models')
    
    # Model selection
    parser.add_argument('--model', type=str, required=True,
                        choices=list(BENCHMARK_MODELS.keys()),
                        help='Benchmark model to train')
    
    # Training params
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    
    # Resume
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Validate model choice
    if args.model not in BENCHMARK_MODELS:
        print(f"Error: Unknown model '{args.model}'")
        print(f"Available models: {list(BENCHMARK_MODELS.keys())}")
        sys.exit(1)
    
    # Create trainer and start training
    trainer = BenchmarkTrainer(args.model, args)
    trainer.train()


if __name__ == "__main__":
    main()
