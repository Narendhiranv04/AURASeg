"""
Benchmark Model Training Script
================================

Train state-of-the-art segmentation models for comparison with AURASeg.

Models (all from segmentation_models_pytorch):
    - deeplabv3plus: DeepLabV3+ with ResNet-50 (ECCV 2018)
    - segformer: SegFormer with MiT-B2 (NeurIPS 2021)
    - upernet: UPerNet with ResNet-101 (ECCV 2018)
    - dpt: PSPNet with ResNet-101 (CVPR 2017)
    - mask2former: FPN with MiT-B3 (CVPR 2017)

Training Setup (matches AURASeg V4):
    - Optimizer: AdamW (lr=1e-3, weight_decay=1e-4)
    - Scheduler: Cosine annealing (min_lr=1e-6)
    - Loss: Focal + Dice
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

All checkpoints saved to: runs/benchmark_{model_name}/
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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "kobuki-yolop" / "model" / "new_network" / "ABLATION_11_DEC"))

from model_factory import get_benchmark_model, BENCHMARK_MODELS
from unified_dataset import Normalization, UnifiedDrivableAreaDataset


# =============================================================================
# Configuration (matches AURASeg training setup)
# =============================================================================

class Config:
    """Unified configuration for benchmark training."""
    
    # Data paths (relative to AURASeg root)
    DATA_ROOT = Path(__file__).parent.parent
    IMAGE_DIR = DATA_ROOT / "CommonDataset" / "images"
    MASK_DIR = DATA_ROOT / "CommonDataset" / "labels"
    
    # Image size
    IMG_SIZE = (384, 640)  # (H, W)
    NUM_CLASSES = 2  # drivable / non-drivable
    
    # Training hyperparams (identical to AURASeg V4)
    EPOCHS = 50
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    MIN_LR = 1e-6
    
    # Loss weights
    FOCAL_WEIGHT = 1.0
    DICE_WEIGHT = 1.0
    FOCAL_ALPHA = 0.25
    FOCAL_GAMMA = 2.0
    
    # Regularization
    GRAD_CLIP = 1.0
    
    # Early stopping
    PATIENCE = 15
    MIN_DELTA = 0.001
    
    # Mixed precision
    USE_AMP = True
    
    # Data loading
    NUM_WORKERS = 4
    PIN_MEMORY = True
    
    # Normalization
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]


# =============================================================================
# Dataset
# =============================================================================

class DrivableAreaDataset(Dataset):
    """Dataset for drivable area segmentation."""
    
    def __init__(self, image_dir, mask_dir, img_size, split='train', 
                 transform=True, mean=None, std=None):
        self.image_dir = Path(image_dir) / split
        self.mask_dir = Path(mask_dir) / split
        self.img_size = img_size
        self.transform_enabled = transform
        self.mean = mean or Config.MEAN
        self.std = std or Config.STD
        
        # Get all images
        self.images = sorted(list(self.image_dir.glob('*.jpg')) + 
                            list(self.image_dir.glob('*.png')))
        
        print(f"[{split.upper()}] Found {len(self.images)} images")
        
        # Build transforms
        self.transform = self._build_transforms(split, transform)
    
    def _build_transforms(self, split, use_augment):
        """Build albumentations transform pipeline."""
        if split == 'train' and use_augment:
            return A.Compose([
                A.Resize(height=self.img_size[0], width=self.img_size[1]),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.5
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.3
                ),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
                A.Normalize(mean=self.mean, std=self.std),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Resize(height=self.img_size[0], width=self.img_size[1]),
                A.Normalize(mean=self.mean, std=self.std),
                ToTensorV2()
            ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.images[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask_name = img_path.stem + '.png'
        mask_path = self.mask_dir / mask_name
        if not mask_path.exists():
            mask_path = self.mask_dir / (img_path.stem + '.jpg')
        
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # Binarize mask (drivable=1, non-drivable=0)
        mask = (mask > 127).astype(np.uint8)
        
        # Apply transforms
        transformed = self.transform(image=image, mask=mask)
        image = transformed['image'].float()
        mask = transformed['mask'].long()
        
        return image, mask


# =============================================================================
# Loss Functions
# =============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for class imbalance."""
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        # inputs: (B, C, H, W), targets: (B, H, W)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class DiceLoss(nn.Module):
    """Dice Loss for segmentation."""
    
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        # inputs: (B, C, H, W), targets: (B, H, W)
        probs = F.softmax(inputs, dim=1)[:, 1]  # Take foreground class
        targets_float = targets.float()
        
        intersection = (probs * targets_float).sum()
        union = probs.sum() + targets_float.sum()
        
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice


class CombinedLoss(nn.Module):
    """Combined Focal + Dice Loss."""
    
    def __init__(self, focal_weight=1.0, dice_weight=1.0, 
                 focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice_loss = DiceLoss()
    
    def forward(self, inputs, targets):
        focal = self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        return self.focal_weight * focal + self.dice_weight * dice


# =============================================================================
# Metrics
# =============================================================================

def compute_metrics(preds, targets, num_classes=2):
    """Compute segmentation metrics."""
    preds = preds.numpy() if torch.is_tensor(preds) else preds
    targets = targets.numpy() if torch.is_tensor(targets) else targets
    
    metrics = {}
    
    # IoU per class
    ious = []
    for cls in range(num_classes):
        pred_cls = (preds == cls)
        target_cls = (targets == cls)
        
        intersection = (pred_cls & target_cls).sum()
        union = (pred_cls | target_cls).sum()
        
        if union > 0:
            iou = intersection / union
        else:
            iou = 1.0 if intersection == 0 else 0.0
        
        ious.append(iou)
    
    metrics['iou_background'] = ious[0]
    metrics['iou_drivable'] = ious[1] if len(ious) > 1 else 0.0
    metrics['miou'] = np.mean(ious)
    
    # Dice score
    pred_fg = (preds == 1)
    target_fg = (targets == 1)
    intersection = (pred_fg & target_fg).sum()
    dice = (2 * intersection) / (pred_fg.sum() + target_fg.sum() + 1e-6)
    metrics['mdice'] = dice
    
    # Precision, Recall
    tp = (pred_fg & target_fg).sum()
    fp = (pred_fg & ~target_fg).sum()
    fn = (~pred_fg & target_fg).sum()
    
    metrics['precision'] = tp / (tp + fp + 1e-6)
    metrics['recall'] = tp / (tp + fn + 1e-6)
    metrics['f1'] = 2 * metrics['precision'] * metrics['recall'] / (
        metrics['precision'] + metrics['recall'] + 1e-6
    )
    
    return metrics


# =============================================================================
# Utilities
# =============================================================================

class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """Early stopping handler."""
    
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
    
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_checkpoint(state, filepath):
    """Save model checkpoint."""
    torch.save(state, filepath)


# =============================================================================
# Trainer
# =============================================================================

class BenchmarkTrainer:
    """Trainer for benchmark SOTA models."""
    
    def __init__(self, model_name: str, args):
        self.model_name = model_name
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        self.repo_root = Path(__file__).parent.parent
        
        # Set random seed
        set_seed(42)
        
        # Override config with args
        self.config = Config()
        if args.epochs:
            self.config.EPOCHS = args.epochs
        if args.batch_size:
            self.config.BATCH_SIZE = args.batch_size
        if args.lr:
            self.config.LEARNING_RATE = args.lr
        if getattr(args, "num_workers", None) is not None:
            self.config.NUM_WORKERS = args.num_workers

        # Dataset setup
        dataset_root = Path(args.dataset_root)
        if not dataset_root.is_absolute():
            dataset_root = self.repo_root / dataset_root
        self.dataset_root = dataset_root
        self.train_split = args.train_split
        self.val_split = args.val_split
        
        # Setup output directories
        self._setup_directories()
        
        # Initialize components
        self.model, self.model_info = self._build_model()
        self.criterion = self._build_criterion()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.scaler = GradScaler('cuda') if self.config.USE_AMP else None
        
        # Data loaders
        self._setup_data()
        
        # Logging
        self.writer = SummaryWriter(self.log_dir)
        self.best_miou = 0.0
        self.start_epoch = 1
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=self.config.PATIENCE,
            min_delta=self.config.MIN_DELTA
        )
        
        # Resume if specified
        if args.resume:
            self._load_checkpoint(args.resume)
    
    def _setup_directories(self):
        """Create output directories."""
        base_dir = Path(self.args.runs_dir)
        if not base_dir.is_absolute():
            base_dir = self.repo_root / base_dir
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
            num_classes=self.config.NUM_CLASSES,
            pretrained=True
        )
        model = model.to(self.device)
        
        # Check if model returns dict (e.g., FCN)
        self.output_key = info.get('output_key', None)
        
        print(f"[INFO] Model: {info['name']} ({info['encoder']})")
        print(f"[INFO] Parameters: {info['params_millions']:.2f}M")
        if self.output_key:
            print(f"[INFO] Output key: '{self.output_key}' (model returns dict)")
        
        return model, info
    
    def _build_criterion(self):
        """Build loss function - Focal + Dice for all models."""
        return CombinedLoss(
            focal_weight=self.config.FOCAL_WEIGHT,
            dice_weight=self.config.DICE_WEIGHT,
            focal_alpha=self.config.FOCAL_ALPHA,
            focal_gamma=self.config.FOCAL_GAMMA
        ).to(self.device)
    
    def _build_optimizer(self):
        """Build optimizer."""
        return optim.AdamW(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY
        )
    
    def _build_scheduler(self):
        """Build learning rate scheduler."""
        return optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.EPOCHS,
            eta_min=self.config.MIN_LR
        )
    
    def _setup_data(self):
        """Setup data loaders."""
        normalization = Normalization(mean=tuple(self.config.MEAN), std=tuple(self.config.STD))

        # Training dataset
        train_dataset = UnifiedDrivableAreaDataset(
            dataset_root=self.dataset_root,
            split=self.train_split,
            img_size=self.config.IMG_SIZE,
            transform=True,
            normalization=normalization,
            return_names=False,
        )
        
        # Validation dataset
        val_dataset = UnifiedDrivableAreaDataset(
            dataset_root=self.dataset_root,
            split=self.val_split,
            img_size=self.config.IMG_SIZE,
            transform=False,
            normalization=normalization,
            return_names=False,
        )
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=self.config.PIN_MEMORY,
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=self.config.PIN_MEMORY
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
        print(f"Epochs: {self.config.EPOCHS}")
        print(f"Batch size: {self.config.BATCH_SIZE}")
        print(f"Learning rate: {self.config.LEARNING_RATE}")
        print(f"Image size: {self.config.IMG_SIZE}")
        print(f"Loss: Focal + Dice")
        print("=" * 70)
        
        print(f"\nStarting training from epoch {self.start_epoch}\n")
        
        for epoch in range(self.start_epoch, self.config.EPOCHS + 1):
            print(f"\n{'=' * 60}")
            print(f"Epoch {epoch}/{self.config.EPOCHS}")
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
            if self.config.USE_AMP:
                with autocast('cuda'):
                    outputs = self.model(images)
                    # Handle dict output (e.g., FCN)
                    if self.output_key and isinstance(outputs, dict):
                        outputs = outputs[self.output_key]
                    loss = self.criterion(outputs, masks)
                
                self.scaler.scale(loss).backward()
                
                if self.config.GRAD_CLIP > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRAD_CLIP)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                # Handle dict output (e.g., FCN)
                if self.output_key and isinstance(outputs, dict):
                    outputs = outputs[self.output_key]
                loss = self.criterion(outputs, masks)
                
                loss.backward()
                
                if self.config.GRAD_CLIP > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRAD_CLIP)
                
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
            outputs = self.model(images)
            # Handle dict output (e.g., FCN)
            if self.output_key and isinstance(outputs, dict):
                outputs = outputs[self.output_key]
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
                        choices=['deeplabv3plus', 'segformer', 'upernet', 'dpt', 'mask2former', 'fcn', 'pspnet', 'pidnet'],
                        help='Benchmark model to train')
    
    # Training params
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')

    # Dataset / outputs
    parser.add_argument('--dataset-root', type=str, default='CommonDataset',
                        help='Dataset root (e.g., CommonDataset or carl-dataset)')
    parser.add_argument('--train-split', type=str, default='train',
                        help='Training split name')
    parser.add_argument('--val-split', type=str, default='val',
                        help='Validation split name')
    parser.add_argument('--runs-dir', type=str, default='runs',
                        help='Base output directory for runs')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='DataLoader workers')
    
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
    available_models = ['deeplabv3plus', 'segformer', 'upernet', 'dpt', 'mask2former', 'fcn', 'pspnet', 'pidnet']
    if args.model not in available_models:
        print(f"Error: Unknown model '{args.model}'")
        print(f"Available models: {available_models}")
        sys.exit(1)
    
    # Create trainer and start training
    trainer = BenchmarkTrainer(args.model, args)
    trainer.train()


if __name__ == "__main__":
    main()
