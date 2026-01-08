"""
Training Script for V3 APUD Model with Deep Supervision
Paper: AURASeg - Attention Guided Upsampling with Residual Boundary-Assistive Refinement

This script handles V3 training with:
    - APUD decoder with SE and Spatial attention
    - Deep supervision at 4 scales
    - Combined loss: Main (Focal + Dice) + Weighted Auxiliary losses

Usage:
    # Train V3 (APUD + Deep Supervision)
    python train_v3.py --epochs 50 --batch-size 8
    
    # Resume training
    python train_v3.py --resume path/to/checkpoint.pth

All outputs saved to: ABLATION_11_DEC/runs/v3_apud/
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

import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from models.v3_apud import V3APUD
from losses import DeepSupervisionLoss
from utils import (
    DrivableAreaDataset, compute_metrics, set_seed,
    AverageMeter, EarlyStopping, save_checkpoint, load_checkpoint,
    count_parameters
)
from config import AblationConfig, DataConfig, TrainConfig, AugmentConfig


def get_v3_config():
    """Get configuration for V3 (APUD + Deep Supervision)"""
    config = AblationConfig(
        experiment_name="ablation_11_dec",
        model_version="v3_apud"
    )
    return config


class V3Trainer:
    """
    Trainer class for V3 APUD model with deep supervision
    
    Handles:
        - Model initialization with APUD decoder
        - Deep supervision loss computation
        - Training with auxiliary outputs
        - Logging of per-scale losses
    """
    
    def __init__(self, config: AblationConfig, args):
        self.config = config
        self.args = args
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        
        # Set random seed
        set_seed(config.seed)
        
        # Create output directories
        self._setup_directories()
        
        # Initialize components
        self.model = self._build_model()
        self.criterion = self._build_criterion()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.scaler = GradScaler('cuda') if config.train.use_amp else None
        
        # Load data
        self.train_loader, self.val_loader = self._build_dataloaders()
        
        # Training state
        self.start_epoch = 1
        self.best_miou = 0.0
        self.early_stopping = EarlyStopping(
            patience=config.train.patience,
            min_delta=config.train.min_delta,
            mode='max'
        )
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=config.log_dir)
        
        # Resume if checkpoint provided
        if args.resume:
            self._resume_from_checkpoint(args.resume)
        
        self._print_info()
    
    def _setup_directories(self):
        """Create output directories"""
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        os.makedirs(self.config.log_dir, exist_ok=True)
        os.makedirs(self.config.vis_dir, exist_ok=True)
        print(f"[INFO] Output directory: {self.config.run_dir}")
    
    def _build_model(self) -> nn.Module:
        """Build V3 APUD model"""
        model = V3APUD(
            in_channels=self.config.model.in_channels,
            num_classes=self.config.model.num_classes,
            decoder_channels=256,
            se_reduction=16
        )
        model = model.to(self.device)
        return model
    
    def _build_criterion(self):
        """Build deep supervision loss"""
        return DeepSupervisionLoss(
            aux_weights=[0.1, 0.2, 0.3, 0.4],  # Coarse to fine
            main_weight=1.0,
            focal_weight=self.config.train.focal_weight,
            dice_weight=self.config.train.dice_weight,
            focal_alpha=self.config.train.focal_alpha,
            focal_gamma=self.config.train.focal_gamma
        ).to(self.device)
    
    def _build_optimizer(self):
        """Build optimizer"""
        if self.config.train.optimizer.lower() == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.train.learning_rate,
                weight_decay=self.config.train.weight_decay
            )
        elif self.config.train.optimizer.lower() == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.train.learning_rate,
                momentum=0.9,
                weight_decay=self.config.train.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.train.optimizer}")
    
    def _build_scheduler(self):
        """Build learning rate scheduler"""
        if self.config.train.scheduler == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.train.epochs,
                eta_min=self.config.train.min_lr
            )
        elif self.config.train.scheduler == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.5
            )
        else:
            return None
    
    def _build_dataloaders(self):
        """Build train and validation dataloaders"""
        base_image_dir = os.path.join(os.getcwd(), self.config.data.image_dir)
        base_mask_dir = os.path.join(os.getcwd(), self.config.data.mask_dir)
        
        train_dataset = DrivableAreaDataset(
            image_dir=base_image_dir,
            mask_dir=base_mask_dir,
            img_size=self.config.data.img_size,
            split='train',
            transform=True,
            mean=self.config.augment.mean,
            std=self.config.augment.std
        )
        
        val_dataset = DrivableAreaDataset(
            image_dir=base_image_dir,
            mask_dir=base_mask_dir,
            img_size=self.config.data.img_size,
            split='val',
            transform=False,
            mean=self.config.augment.mean,
            std=self.config.augment.std
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.train.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.train.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory
        )
        
        print(f"[INFO] Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
        
        return train_loader, val_loader
    
    def _resume_from_checkpoint(self, checkpoint_path: str):
        """Resume training from checkpoint"""
        checkpoint = load_checkpoint(
            checkpoint_path, 
            self.model, 
            self.optimizer, 
            self.scheduler
        )
        
        self.start_epoch = checkpoint.get('epoch', 0) + 1
        self.best_miou = checkpoint.get('best_miou', 0.0)
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"[INFO] Resuming from epoch {self.start_epoch}")
    
    def _print_info(self):
        """Print training configuration"""
        print("\n" + "=" * 60)
        print("ABLATION STUDY - V3 APUD + DEEP SUPERVISION")
        print("=" * 60)
        print(f"Model: V3 APUD")
        print(f"Device: {self.device}")
        print(f"Parameters: {count_parameters(self.model):,}")
        print(f"Epochs: {self.config.train.epochs}")
        print(f"Batch size: {self.config.train.batch_size}")
        print(f"Learning rate: {self.config.train.learning_rate}")
        print(f"Loss: Deep Supervision (Focal + Dice at 4 scales)")
        print(f"Aux weights: [0.1, 0.2, 0.3, 0.4]")
        print(f"Image size: {self.config.data.img_size}")
        print("=" * 60 + "\n")
    
    def train_epoch(self, epoch: int) -> dict:
        """Train for one epoch with deep supervision"""
        self.model.train()
        
        loss_meters = {
            'total': AverageMeter(),
            'main': AverageMeter(),
            'aux_weighted': AverageMeter()
        }
        
        start_time = time.time()
        
        for batch_idx, (images, masks) in enumerate(self.train_loader):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Mixed precision forward
            if self.config.train.use_amp:
                with autocast('cuda'):
                    # Forward with auxiliary outputs
                    outputs = self.model(images, return_aux=True)
                    
                    # Compute deep supervision loss
                    losses = self.criterion(outputs, masks)
                    total_loss = losses['total']
                
                self.scaler.scale(total_loss).backward()
                
                if self.config.train.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.train.grad_clip
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images, return_aux=True)
                losses = self.criterion(outputs, masks)
                total_loss = losses['total']
                
                total_loss.backward()
                
                if self.config.train.grad_clip > 0:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.train.grad_clip
                    )
                
                self.optimizer.step()
            
            # Update meters
            loss_meters['total'].update(losses['total'].item(), images.size(0))
            loss_meters['main'].update(losses['main'].item(), images.size(0))
            loss_meters['aux_weighted'].update(losses['aux_weighted'].item(), images.size(0))
            
            # Print progress
            if (batch_idx + 1) % 10 == 0:
                elapsed = time.time() - start_time
                print(f"  Batch [{batch_idx + 1}/{len(self.train_loader)}] "
                      f"Total: {loss_meters['total'].avg:.4f} "
                      f"Main: {loss_meters['main'].avg:.4f} "
                      f"Aux: {loss_meters['aux_weighted'].avg:.4f} "
                      f"Time: {elapsed:.1f}s")
        
        return {k: v.avg for k, v in loss_meters.items()}
    
    @torch.no_grad()
    def validate(self, epoch: int) -> dict:
        """Validate model"""
        self.model.eval()
        
        loss_meters = {
            'total': AverageMeter(),
            'main': AverageMeter(),
            'aux_weighted': AverageMeter()
        }
        all_metrics = []
        
        for images, masks in self.val_loader:
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward with auxiliary outputs
            outputs = self.model(images, return_aux=True)
            
            # Compute losses
            losses = self.criterion(outputs, masks)
            
            loss_meters['total'].update(losses['total'].item(), images.size(0))
            loss_meters['main'].update(losses['main'].item(), images.size(0))
            loss_meters['aux_weighted'].update(losses['aux_weighted'].item(), images.size(0))
            
            # Compute metrics on main output
            preds = outputs['main'].argmax(dim=1)
            metrics = compute_metrics(preds.cpu(), masks.cpu(), num_classes=2)
            all_metrics.append(metrics)
        
        # Average metrics
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        
        # Add loss metrics
        avg_metrics['loss_total'] = loss_meters['total'].avg
        avg_metrics['loss_main'] = loss_meters['main'].avg
        avg_metrics['loss_aux'] = loss_meters['aux_weighted'].avg
        
        return avg_metrics
    
    def save_visualizations(self, epoch: int):
        """Save visualization of predictions including auxiliary outputs"""
        self.model.eval()
        
        # Get one batch
        images, masks = next(iter(self.val_loader))
        images = images.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(images, return_aux=True)
        
        main_preds = outputs['main'].argmax(dim=1)
        aux_preds = [aux.argmax(dim=1) for aux in outputs['aux']]
        
        # Denormalize images
        mean = torch.tensor(self.config.augment.mean).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor(self.config.augment.std).view(1, 3, 1, 1).to(self.device)
        images = images * std + mean
        
        # Plot: Input | GT | Main | Aux1 | Aux2 | Aux3 | Aux4
        fig, axes = plt.subplots(4, 7, figsize=(28, 16))
        
        for i in range(min(4, len(images))):
            # Image
            img = images[i].cpu().permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            axes[i, 0].imshow(img)
            axes[i, 0].set_title('Input')
            axes[i, 0].axis('off')
            
            # Ground truth
            axes[i, 1].imshow(masks[i].cpu().numpy(), cmap='gray')
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')
            
            # Main prediction
            axes[i, 2].imshow(main_preds[i].cpu().numpy(), cmap='gray')
            axes[i, 2].set_title('Main (H/1)')
            axes[i, 2].axis('off')
            
            # Auxiliary predictions
            for j, (aux_pred, scale) in enumerate(zip(aux_preds, ['H/32', 'H/16', 'H/8', 'H/4'])):
                axes[i, 3 + j].imshow(aux_pred[i].cpu().numpy(), cmap='gray')
                axes[i, 3 + j].set_title(f'Aux-{j+1} ({scale})')
                axes[i, 3 + j].axis('off')
        
        plt.suptitle(f'V3 APUD - Epoch {epoch} (Deep Supervision Outputs)')
        plt.tight_layout()
        
        save_path = os.path.join(self.config.vis_dir, f'epoch_{epoch:03d}.png')
        plt.savefig(save_path, dpi=150)
        plt.close()
    
    def train(self):
        """Full training loop"""
        print(f"Starting training from epoch {self.start_epoch}")
        
        for epoch in range(self.start_epoch, self.config.train.epochs + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{self.config.train.epochs}")
            print(f"{'='*60}")
            
            # Train
            train_losses = self.train_epoch(epoch)
            
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"\n[TRAIN] Total: {train_losses['total']:.4f}, "
                  f"Main: {train_losses['main']:.4f}, "
                  f"Aux: {train_losses['aux_weighted']:.4f}, "
                  f"LR: {current_lr:.6f}")
            
            # Validate
            if epoch % self.config.train.val_every == 0:
                val_metrics = self.validate(epoch)
                
                print(f"[VAL] Total Loss: {val_metrics['loss_total']:.4f}")
                print(f"      Main Loss: {val_metrics['loss_main']:.4f}")
                print(f"      Aux Loss: {val_metrics['loss_aux']:.4f}")
                print(f"      mIoU: {val_metrics['miou']:.4f}")
                print(f"      IoU (Drivable): {val_metrics['iou_drivable']:.4f}")
                print(f"      Dice: {val_metrics['mdice']:.4f}")
                
                # Log to TensorBoard
                self.writer.add_scalar('Loss/train_total', train_losses['total'], epoch)
                self.writer.add_scalar('Loss/train_main', train_losses['main'], epoch)
                self.writer.add_scalar('Loss/train_aux', train_losses['aux_weighted'], epoch)
                self.writer.add_scalar('Loss/val_total', val_metrics['loss_total'], epoch)
                self.writer.add_scalar('Loss/val_main', val_metrics['loss_main'], epoch)
                self.writer.add_scalar('Loss/val_aux', val_metrics['loss_aux'], epoch)
                self.writer.add_scalar('mIoU/val', val_metrics['miou'], epoch)
                self.writer.add_scalar('LR', current_lr, epoch)
                
                # Save best model
                if val_metrics['miou'] > self.best_miou:
                    self.best_miou = val_metrics['miou']
                    self._save_checkpoint(epoch, is_best=True)
                    print(f"  *** New best mIoU: {self.best_miou:.4f} ***")
                
                # Early stopping check
                if self.early_stopping(val_metrics['miou']):
                    print(f"\n[INFO] Early stopping triggered at epoch {epoch}")
                    break
            
            # Save checkpoint
            if epoch % self.config.train.save_every == 0:
                self._save_checkpoint(epoch, is_best=False)
            
            # Save visualizations
            if epoch % 10 == 0:
                self.save_visualizations(epoch)
        
        # Final save
        self._save_checkpoint(epoch, is_best=False, filename='final.pth')
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print(f"Best mIoU: {self.best_miou:.4f}")
        print(f"Checkpoints saved to: {self.config.checkpoint_dir}")
        print("=" * 60)
        
        self.writer.close()
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False, filename: str = None):
        """Save checkpoint"""
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_miou': self.best_miou,
            'config': self.config,
        }
        
        if self.scheduler is not None:
            state['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.scaler is not None:
            state['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save latest
        if filename:
            save_path = os.path.join(self.config.checkpoint_dir, filename)
        else:
            save_path = os.path.join(self.config.checkpoint_dir, 'latest.pth')
        save_checkpoint(state, save_path)
        
        # Save best
        if is_best:
            best_path = os.path.join(self.config.checkpoint_dir, 'best.pth')
            save_checkpoint(state, best_path)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train V3 APUD Model with Deep Supervision')
    
    # Training params
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate')
    
    # Resume
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()
    
    # Get V3 config
    config = get_v3_config()
    
    # Override config with command line args
    if args.epochs:
        config.train.epochs = args.epochs
    if args.batch_size:
        config.train.batch_size = args.batch_size
    if args.lr:
        config.train.learning_rate = args.lr
    if args.device:
        config.device = args.device
    
    # Reinitialize paths after config updates
    config.__post_init__()
    
    # Create trainer and start training
    trainer = V3Trainer(config, args)
    trainer.train()


if __name__ == "__main__":
    main()
