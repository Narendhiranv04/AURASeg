"""
Training Script for V4 RBRM Model
Paper: AURASeg - Attention Guided Upsampling with Residual Boundary-Assistive Refinement

V4 extends V3 (APUD) with the Residual Boundary Refinement Module (RBRM).

Training Strategy:
    - Initialize from V3 pretrained weights
    - Differential learning rates:
        - Pretrained (encoder, ASPP, decoder): 1e-4
        - New (RBRM, seg_head): 1e-3
    - End-to-end training (no freezing)
    - Loss: Main (Focal + Dice) + Deep Supervision + Boundary Loss

Loss Function:
    L_total = L_main + L_supervision + 0.5 × L_boundary
    
    Where:
        L_main = Focal + Dice at full resolution
        L_supervision = 0.1×L1 + 0.2×L2 + 0.3×L3 + 0.4×L4
        L_boundary = WeightedBCE on boundary prediction

Usage:
    # Train V4 from V3 checkpoint
    python train_v4.py --epochs 50 --batch-size 8
    
    # Resume training
    python train_v4.py --resume path/to/checkpoint.pth

All outputs saved to: ABLATION_11_DEC/runs/v4_rbrm/
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

from models.v4_rbrm import V4RBRM
from losses import V4CombinedLoss
from utils import (
    DrivableAreaDataset, compute_metrics, set_seed,
    AverageMeter, EarlyStopping, save_checkpoint, load_checkpoint,
    count_parameters
)
from config import AblationConfig, DataConfig, TrainConfig, AugmentConfig


def get_v4_config():
    """Get configuration for V4 (RBRM)"""
    config = AblationConfig(
        experiment_name="ablation_11_dec",
        model_version="v4_rbrm"
    )
    return config


class V4Trainer:
    """
    Trainer class for V4 RBRM model
    
    Features:
        - Initialize from V3 pretrained weights
        - Differential learning rates for pretrained vs new modules
        - Deep supervision + boundary loss training
        - Visualization of boundary predictions
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
        """Build V4 RBRM model, optionally loading V3 weights"""
        model = V4RBRM(
            in_channels=self.config.model.in_channels,
            num_classes=self.config.model.num_classes,
            decoder_channels=256,
            se_reduction=16,
            edge_channels=64
        )
        
        # Load V3 pretrained weights if available
        v3_checkpoint = self.args.pretrained_v3
        if v3_checkpoint is None:
            # Try default V3 checkpoint path
            default_v3_path = os.path.join(
                os.path.dirname(self.config.run_dir),
                'v3_apud', 'checkpoints', 'best.pth'
            )
            if os.path.exists(default_v3_path):
                v3_checkpoint = default_v3_path
        
        if v3_checkpoint and os.path.exists(v3_checkpoint):
            print(f"[INFO] Loading V3 pretrained weights from: {v3_checkpoint}")
            model.load_v3_weights(v3_checkpoint)
        else:
            print("[INFO] Training V4 from scratch (no V3 weights found)")
        
        model = model.to(self.device)
        return model
    
    def _build_criterion(self):
        """Build V4 combined loss (main + aux + boundary)"""
        return V4CombinedLoss(
            aux_weights=[0.1, 0.2, 0.3, 0.4],
            main_weight=1.0,
            boundary_weight=0.5,  # λ_bnd = 0.5
            focal_weight=self.config.train.focal_weight,
            dice_weight=self.config.train.dice_weight,
            focal_alpha=self.config.train.focal_alpha,
            focal_gamma=self.config.train.focal_gamma,
            boundary_kernel=5
        ).to(self.device)
    
    def _build_optimizer(self):
        """Build optimizer with differential learning rates"""
        # Get parameter groups
        lr_pretrained = self.config.train.learning_rate  # 1e-4 for pretrained
        lr_new = self.config.train.learning_rate * 10    # 1e-3 for new modules
        
        param_groups = self.model.get_param_groups(
            lr_pretrained=lr_pretrained,
            lr_new=lr_new
        )
        
        if self.config.train.optimizer.lower() == 'adamw':
            return optim.AdamW(
                param_groups,
                weight_decay=self.config.train.weight_decay
            )
        elif self.config.train.optimizer.lower() == 'sgd':
            return optim.SGD(
                param_groups,
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
        total_params = count_parameters(self.model)
        
        # Count RBRM params
        rbrm_params = sum(p.numel() for p in self.model.rbrm.parameters())
        
        print("\n" + "=" * 60)
        print("ABLATION STUDY - V4 RBRM (Residual Boundary Refinement)")
        print("=" * 60)
        print(f"Model: V4 RBRM = V3 APUD + Boundary Refinement")
        print(f"Device: {self.device}")
        print(f"Total Parameters: {total_params:,}")
        print(f"  - RBRM Parameters: {rbrm_params:,}")
        print(f"Epochs: {self.config.train.epochs}")
        print(f"Batch size: {self.config.train.batch_size}")
        print(f"Learning rates:")
        for group in self.optimizer.param_groups:
            print(f"  - {group.get('name', 'unnamed')}: {group['lr']}")
        print(f"Loss: Main + Deep Supervision + 0.5 × Boundary")
        print(f"Aux weights: [0.1, 0.2, 0.3, 0.4]")
        print(f"Image size: {self.config.data.img_size}")
        print("=" * 60 + "\n")
    
    def train_epoch(self, epoch: int) -> dict:
        """Train for one epoch"""
        self.model.train()
        
        loss_meters = {
            'total': AverageMeter(),
            'main': AverageMeter(),
            'aux_weighted': AverageMeter(),
            'boundary': AverageMeter()
        }
        
        start_time = time.time()
        
        for batch_idx, (images, masks) in enumerate(self.train_loader):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Mixed precision forward
            if self.config.train.use_amp:
                with autocast('cuda'):
                    outputs = self.model(images, return_aux=True, return_boundary=True)
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
                outputs = self.model(images, return_aux=True, return_boundary=True)
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
            loss_meters['boundary'].update(losses['boundary'].item(), images.size(0))
            
            # Print progress
            if (batch_idx + 1) % 10 == 0:
                elapsed = time.time() - start_time
                print(f"  Batch [{batch_idx + 1}/{len(self.train_loader)}] "
                      f"Total: {loss_meters['total'].avg:.4f} "
                      f"Main: {loss_meters['main'].avg:.4f} "
                      f"Aux: {loss_meters['aux_weighted'].avg:.4f} "
                      f"Bnd: {loss_meters['boundary'].avg:.4f} "
                      f"Time: {elapsed:.1f}s")
        
        return {k: v.avg for k, v in loss_meters.items()}
    
    @torch.no_grad()
    def validate(self, epoch: int) -> dict:
        """Validate model"""
        self.model.eval()
        
        loss_meters = {
            'total': AverageMeter(),
            'main': AverageMeter(),
            'aux_weighted': AverageMeter(),
            'boundary': AverageMeter()
        }
        all_metrics = []
        
        for images, masks in self.val_loader:
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            outputs = self.model(images, return_aux=True, return_boundary=True)
            losses = self.criterion(outputs, masks)
            
            loss_meters['total'].update(losses['total'].item(), images.size(0))
            loss_meters['main'].update(losses['main'].item(), images.size(0))
            loss_meters['aux_weighted'].update(losses['aux_weighted'].item(), images.size(0))
            loss_meters['boundary'].update(losses['boundary'].item(), images.size(0))
            
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
        avg_metrics['loss_boundary'] = loss_meters['boundary'].avg
        
        return avg_metrics
    
    def save_visualizations(self, epoch: int):
        """Save visualization of predictions including boundary outputs"""
        self.model.eval()
        
        # Get one batch
        images, masks = next(iter(self.val_loader))
        images = images.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(images, return_aux=True, return_boundary=True)
        
        main_preds = outputs['main'].argmax(dim=1)
        boundary_preds = outputs['boundary']
        
        # Denormalize images
        mean = torch.tensor(self.config.augment.mean).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor(self.config.augment.std).view(1, 3, 1, 1).to(self.device)
        images = images * std + mean
        
        # Generate boundary GT for visualization
        boundary_gt = self.criterion.boundary_loss.generate_boundary_gt(masks.to(self.device))
        
        # Plot: Input | GT | Main Pred | Boundary GT | Boundary Pred
        fig, axes = plt.subplots(4, 5, figsize=(25, 20))
        
        for i in range(min(4, len(images))):
            # Image
            img = images[i].cpu().permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            axes[i, 0].imshow(img)
            axes[i, 0].set_title('Input')
            axes[i, 0].axis('off')
            
            # Ground truth
            axes[i, 1].imshow(masks[i].cpu().numpy(), cmap='gray')
            axes[i, 1].set_title('GT Segmentation')
            axes[i, 1].axis('off')
            
            # Main prediction
            axes[i, 2].imshow(main_preds[i].cpu().numpy(), cmap='gray')
            axes[i, 2].set_title('Main Prediction')
            axes[i, 2].axis('off')
            
            # Boundary GT
            axes[i, 3].imshow(boundary_gt[i, 0].cpu().numpy(), cmap='hot')
            axes[i, 3].set_title('Boundary GT')
            axes[i, 3].axis('off')
            
            # Boundary prediction
            axes[i, 4].imshow(boundary_preds[i, 0].cpu().numpy(), cmap='hot')
            axes[i, 4].set_title('Boundary Pred')
            axes[i, 4].axis('off')
        
        plt.suptitle(f'V4 RBRM - Epoch {epoch}')
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
            
            # Get current LR (from first param group)
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"\n[TRAIN] Total: {train_losses['total']:.4f}, "
                  f"Main: {train_losses['main']:.4f}, "
                  f"Aux: {train_losses['aux_weighted']:.4f}, "
                  f"Bnd: {train_losses['boundary']:.4f}, "
                  f"LR: {current_lr:.6f}")
            
            # Validate
            if epoch % self.config.train.val_every == 0:
                val_metrics = self.validate(epoch)
                
                print(f"[VAL] Total Loss: {val_metrics['loss_total']:.4f}")
                print(f"      Main Loss: {val_metrics['loss_main']:.4f}")
                print(f"      Aux Loss: {val_metrics['loss_aux']:.4f}")
                print(f"      Boundary Loss: {val_metrics['loss_boundary']:.4f}")
                print(f"      mIoU: {val_metrics['miou']:.4f}")
                print(f"      IoU (Drivable): {val_metrics['iou_drivable']:.4f}")
                print(f"      Dice: {val_metrics['mdice']:.4f}")
                
                # Log to TensorBoard
                self.writer.add_scalar('Loss/train_total', train_losses['total'], epoch)
                self.writer.add_scalar('Loss/train_main', train_losses['main'], epoch)
                self.writer.add_scalar('Loss/train_aux', train_losses['aux_weighted'], epoch)
                self.writer.add_scalar('Loss/train_boundary', train_losses['boundary'], epoch)
                self.writer.add_scalar('Loss/val_total', val_metrics['loss_total'], epoch)
                self.writer.add_scalar('Loss/val_main', val_metrics['loss_main'], epoch)
                self.writer.add_scalar('Loss/val_aux', val_metrics['loss_aux'], epoch)
                self.writer.add_scalar('Loss/val_boundary', val_metrics['loss_boundary'], epoch)
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
    parser = argparse.ArgumentParser(description='Train V4 RBRM Model')
    
    # Training params
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=None,
                        help='Base learning rate (pretrained modules)')
    
    # Model initialization
    parser.add_argument('--pretrained-v3', type=str, default=None,
                        help='Path to V3 checkpoint for initialization')
    
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
    
    # Get V4 config
    config = get_v4_config()
    
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
    trainer = V4Trainer(config, args)
    trainer.train()


if __name__ == "__main__":
    main()
