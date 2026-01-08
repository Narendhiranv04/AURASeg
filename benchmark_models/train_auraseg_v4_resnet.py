"""
Training Script for AURASeg V4 with ResNet-50 Encoder
======================================================

Final model training for RAL paper benchmark comparison.

Training Configuration:
    - Optimizer: AdamW (weight_decay=0.01)
    - Learning Rate: Differential (encoder=1e-4, decoder=1e-3)
    - Scheduler: CosineAnnealingLR (T_max=50, eta_min=1e-6)
    - Loss: Focal + Dice + Boundary + Auxiliary (deep supervision)
    - Epochs: 50 (early stopping patience=10)
    - Batch Size: 8
    - Mixed Precision: Enabled (AMP)

Usage:
    python train_auraseg_v4_resnet.py --epochs 50
    python train_auraseg_v4_resnet.py --epochs 50 --lr-encoder 1e-4 --lr-decoder 1e-3
    python train_auraseg_v4_resnet.py --resume runs/auraseg_v4_resnet50/checkpoints/latest.pth
"""

import os
import sys
import argparse
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

# Albumentations for augmentation
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False
    print("Warning: albumentations not found, using basic augmentation")

# Import model
sys.path.insert(0, str(Path(__file__).parent))
from auraseg_v4_resnet import auraseg_v4_resnet50
from unified_dataset import Normalization, UnifiedDrivableAreaDataset


# =============================================================================
# Configuration
# =============================================================================

class Config:
    """Training configuration"""
    
    # Dataset paths
    DATA_ROOT = Path(__file__).parent.parent / "CommonDataset"
    IMAGE_DIR = DATA_ROOT / "images"
    MASK_DIR = DATA_ROOT / "labels"
    
    # Output directory
    OUTPUT_DIR = Path(__file__).parent.parent / "runs" / "auraseg_v4_resnet50"
    
    # Model
    NUM_CLASSES = 2
    IMG_SIZE = (384, 640)  # (H, W)
    
    # Training
    EPOCHS = 50
    BATCH_SIZE = 8
    LR_ENCODER = 1e-4  # Lower LR for pretrained encoder
    LR_DECODER = 1e-3  # Higher LR for decoder modules
    WEIGHT_DECAY = 0.01
    
    # Loss weights
    FOCAL_WEIGHT = 0.5
    DICE_WEIGHT = 0.5
    BOUNDARY_WEIGHT = 0.2
    AUX_WEIGHT = 0.1  # Per auxiliary head
    
    # Early stopping
    PATIENCE = 10
    MIN_DELTA = 0.0001
    
    # DataLoader
    NUM_WORKERS = 4
    PIN_MEMORY = True
    
    # Mixed precision
    USE_AMP = True
    
    # ImageNet normalization
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]


# =============================================================================
# Dataset
# =============================================================================

class DrivableAreaDataset(Dataset):
    """Drivable area segmentation dataset"""
    
    def __init__(self, image_dir: Path, mask_dir: Path, img_size: tuple,
                 split: str = 'train', transform: bool = True):
        self.image_dir = image_dir / split
        self.mask_dir = mask_dir / split
        self.img_size = img_size
        self.transform = transform and (split == 'train')
        
        # Get image paths
        self.image_paths = sorted(list(self.image_dir.glob("*.png")) + 
                                  list(self.image_dir.glob("*.jpg")))
        
        print(f"[{split.upper()}] Found {len(self.image_paths)} images")
        
        # Setup augmentations
        if HAS_ALBUMENTATIONS:
            self.train_transform = A.Compose([
                A.Resize(img_size[0], img_size[1]),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, 
                                   rotate_limit=15, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, 
                                           contrast_limit=0.2, p=0.3),
                A.GaussNoise(p=0.2),
                A.Normalize(mean=Config.MEAN, std=Config.STD),
                ToTensorV2()
            ])
            
            self.val_transform = A.Compose([
                A.Resize(img_size[0], img_size[1]),
                A.Normalize(mean=Config.MEAN, std=Config.STD),
                ToTensorV2()
            ])
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> tuple:
        # Load image
        img_path = self.image_paths[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask_name = img_path.stem + ".png"
        mask_path = self.mask_dir / mask_name
        
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        else:
            # Try jpg
            mask_path = self.mask_dir / (img_path.stem + ".jpg")
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            else:
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # Binarize mask (0 = background, 1 = drivable)
        mask = (mask > 127).astype(np.uint8)
        
        # Apply transforms
        if HAS_ALBUMENTATIONS:
            if self.transform:
                augmented = self.train_transform(image=image, mask=mask)
            else:
                augmented = self.val_transform(image=image, mask=mask)
            
            image = augmented['image']
            mask = augmented['mask'].long()
        else:
            # Basic preprocessing without albumentations
            image = cv2.resize(image, (self.img_size[1], self.img_size[0]))
            mask = cv2.resize(mask, (self.img_size[1], self.img_size[0]), 
                             interpolation=cv2.INTER_NEAREST)
            
            image = image.astype(np.float32) / 255.0
            image = (image - np.array(Config.MEAN)) / np.array(Config.STD)
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()
            mask = torch.from_numpy(mask).long()
        
        return image, mask


# =============================================================================
# Loss Functions
# =============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""
    
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_soft = F.softmax(pred, dim=1)
        
        # One-hot encode target
        target_onehot = F.one_hot(target, num_classes=pred.shape[1])
        target_onehot = target_onehot.permute(0, 3, 1, 2).float()
        
        # Compute Dice per class
        dims = (2, 3)
        intersection = (pred_soft * target_onehot).sum(dims)
        union = pred_soft.sum(dims) + target_onehot.sum(dims)
        
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        
        return 1 - dice.mean()


class BoundaryLoss(nn.Module):
    """Binary Cross Entropy loss for boundary prediction"""
    
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Boundary prediction (B, 1, H, W)
            target: Segmentation mask (B, H, W)
        """
        # Generate boundary from segmentation mask using Sobel
        target_float = target.float().unsqueeze(1)  # (B, 1, H, W)
        
        # Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=torch.float32, device=target.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=torch.float32, device=target.device).view(1, 1, 3, 3)
        
        # Compute gradients
        grad_x = F.conv2d(target_float, sobel_x, padding=1)
        grad_y = F.conv2d(target_float, sobel_y, padding=1)
        
        # Gradient magnitude
        boundary_gt = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        boundary_gt = (boundary_gt > 0.1).float()  # Threshold to binary
        
        return self.bce(pred, boundary_gt)


class CombinedLoss(nn.Module):
    """Combined loss for AURASeg training"""
    
    def __init__(self, focal_weight: float = 0.5, dice_weight: float = 0.5,
                 boundary_weight: float = 0.2, aux_weight: float = 0.1):
        super().__init__()
        
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.aux_weight = aux_weight
        
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        self.dice_loss = DiceLoss(smooth=1.0)
        self.boundary_loss = BoundaryLoss()
    
    def forward(self, outputs: dict, target: torch.Tensor) -> dict:
        """
        Args:
            outputs: Model outputs dict with 'main', 'aux', 'boundary'
            target: Ground truth segmentation mask (B, H, W)
        """
        losses = {}
        
        # Main segmentation loss
        main_pred = outputs['main']
        
        # Resize target to match prediction if needed
        if main_pred.shape[2:] != target.shape[1:]:
            target_resized = F.interpolate(
                target.unsqueeze(1).float(), 
                size=main_pred.shape[2:], 
                mode='nearest'
            ).squeeze(1).long()
        else:
            target_resized = target
        
        focal = self.focal_loss(main_pred, target_resized)
        dice = self.dice_loss(main_pred, target_resized)
        
        losses['focal'] = focal
        losses['dice'] = dice
        losses['seg'] = self.focal_weight * focal + self.dice_weight * dice
        
        # Auxiliary losses (deep supervision)
        if 'aux' in outputs:
            aux_loss = 0.0
            for i, aux_pred in enumerate(outputs['aux']):
                # Resize target to aux prediction size
                aux_target = F.interpolate(
                    target.unsqueeze(1).float(),
                    size=aux_pred.shape[2:],
                    mode='nearest'
                ).squeeze(1).long()
                
                aux_focal = self.focal_loss(aux_pred, aux_target)
                aux_dice = self.dice_loss(aux_pred, aux_target)
                aux_loss += self.focal_weight * aux_focal + self.dice_weight * aux_dice
            
            losses['aux'] = aux_loss * self.aux_weight
        else:
            losses['aux'] = torch.tensor(0.0, device=main_pred.device)
        
        # Boundary loss
        if 'boundary' in outputs:
            boundary_pred = outputs['boundary']
            losses['boundary'] = self.boundary_loss(boundary_pred, target_resized) * self.boundary_weight
        else:
            losses['boundary'] = torch.tensor(0.0, device=main_pred.device)
        
        # Total loss
        losses['total'] = losses['seg'] + losses['aux'] + losses['boundary']
        
        return losses


# =============================================================================
# Metrics
# =============================================================================

def compute_metrics(preds: np.ndarray, targets: np.ndarray, num_classes: int = 2) -> dict:
    """Compute segmentation metrics"""
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
    metrics['dice'] = dice
    
    # Precision, Recall, F1
    tp = (pred_fg & target_fg).sum()
    fp = (pred_fg & ~target_fg).sum()
    fn = (~pred_fg & target_fg).sum()
    tn = (~pred_fg & ~target_fg).sum()
    
    metrics['precision'] = tp / (tp + fp + 1e-6)
    metrics['recall'] = tp / (tp + fn + 1e-6)
    metrics['f1'] = 2 * metrics['precision'] * metrics['recall'] / (
        metrics['precision'] + metrics['recall'] + 1e-6
    )
    metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn + 1e-6)
    
    return metrics


def compute_boundary_metrics(preds: np.ndarray, targets: np.ndarray) -> dict:
    """Compute boundary metrics"""
    kernel = np.ones((3, 3), np.uint8)
    
    boundary_ious = []
    boundary_precisions = []
    boundary_recalls = []
    boundary_f1s = []
    
    for i in range(len(preds)):
        pred_binary = (preds[i] == 1).astype(np.uint8)
        target_binary = (targets[i] == 1).astype(np.uint8)
        
        pred_boundary = cv2.morphologyEx(pred_binary, cv2.MORPH_GRADIENT, kernel)
        target_boundary = cv2.morphologyEx(target_binary, cv2.MORPH_GRADIENT, kernel)
        
        pred_boundary = cv2.dilate(pred_boundary, kernel, iterations=2)
        target_boundary = cv2.dilate(target_boundary, kernel, iterations=2)
        
        tp = np.sum((pred_boundary > 0) & (target_boundary > 0))
        fp = np.sum((pred_boundary > 0) & (target_boundary == 0))
        fn = np.sum((pred_boundary == 0) & (target_boundary > 0))
        
        boundary_iou = tp / (tp + fp + fn + 1e-6)
        boundary_precision = tp / (tp + fp + 1e-6)
        boundary_recall = tp / (tp + fn + 1e-6)
        boundary_f1 = 2 * boundary_precision * boundary_recall / (
            boundary_precision + boundary_recall + 1e-6
        )
        
        boundary_ious.append(boundary_iou)
        boundary_precisions.append(boundary_precision)
        boundary_recalls.append(boundary_recall)
        boundary_f1s.append(boundary_f1)
    
    return {
        'boundary_iou': np.mean(boundary_ious),
        'boundary_precision': np.mean(boundary_precisions),
        'boundary_recall': np.mean(boundary_recalls),
        'boundary_f1': np.mean(boundary_f1s)
    }


# =============================================================================
# Trainer
# =============================================================================

class AURASegTrainer:
    """Trainer for AURASeg V4 with ResNet-50"""
    
    def __init__(self, config: Config, device: torch.device):
        self.config = config
        self.device = device
        
        # Setup output directory
        self.output_dir = config.OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        
        # Create model
        self.model = auraseg_v4_resnet50(
            num_classes=config.NUM_CLASSES,
            pretrained_encoder=True
        ).to(device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[INFO] Model parameters: {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"[INFO] Trainable parameters: {trainable_params:,}")
        
        # Loss function
        self.criterion = CombinedLoss(
            focal_weight=config.FOCAL_WEIGHT,
            dice_weight=config.DICE_WEIGHT,
            boundary_weight=config.BOUNDARY_WEIGHT,
            aux_weight=config.AUX_WEIGHT
        )
        
        # Optimizer with differential learning rates
        param_groups = self.model.get_param_groups(
            lr_encoder=config.LR_ENCODER,
            lr_decoder=config.LR_DECODER
        )
        self.optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.EPOCHS,
            eta_min=1e-6
        )
        
        # Mixed precision
        self.scaler = GradScaler(enabled=config.USE_AMP)
        self.use_amp = config.USE_AMP
        
        # Training state
        self.best_miou = 0.0
        self.epochs_without_improvement = 0
        self.start_epoch = 1

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """Load a checkpoint to resume training."""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        try:
            checkpoint = torch.load(
                checkpoint_path, map_location=self.device, weights_only=False
            )
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            if "optimizer_state_dict" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint:
                try:
                    self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                except Exception as exc:
                    print(f"[WARN] Could not load scheduler state: {exc}")
            if checkpoint.get("scaler_state_dict") is not None:
                try:
                    self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
                except Exception as exc:
                    print(f"[WARN] Could not load GradScaler state: {exc}")

            self.best_miou = float(checkpoint.get("best_miou", 0.0))
            self.epochs_without_improvement = int(
                checkpoint.get("epochs_without_improvement", 0)
            )
            self.start_epoch = int(checkpoint.get("epoch", 0)) + 1
            loaded_epoch = int(checkpoint.get("epoch", 0))
        else:
            # Support loading raw model state_dict checkpoints
            self.model.load_state_dict(checkpoint)
            self.best_miou = 0.0
            self.epochs_without_improvement = 0
            self.start_epoch = 1
            loaded_epoch = 0

        print(f"[INFO] Resumed from: {checkpoint_path}")
        print(f"[INFO] Checkpoint epoch: {loaded_epoch} -> start_epoch={self.start_epoch}")
        print(f"[INFO] Best mIoU so far: {self.best_miou:.4f}")
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> dict:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_seg_loss = 0.0
        total_boundary_loss = 0.0
        total_aux_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}")
        
        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            self.optimizer.zero_grad()
            
            with autocast(enabled=self.use_amp):
                outputs = self.model(images, return_aux=True, return_boundary=True)
                losses = self.criterion(outputs, masks)
            
            self.scaler.scale(losses['total']).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += losses['total'].item()
            total_seg_loss += losses['seg'].item()
            total_boundary_loss += losses['boundary'].item()
            total_aux_loss += losses['aux'].item()
            
            pbar.set_postfix({
                'loss': f"{losses['total'].item():.4f}",
                'seg': f"{losses['seg'].item():.4f}",
                'bnd': f"{losses['boundary'].item():.4f}"
            })
        
        n = len(train_loader)
        return {
            'loss': total_loss / n,
            'seg_loss': total_seg_loss / n,
            'boundary_loss': total_boundary_loss / n,
            'aux_loss': total_aux_loss / n
        }
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> dict:
        """Validate model"""
        self.model.eval()
        
        all_preds = []
        all_targets = []
        total_loss = 0.0
        
        for images, masks in tqdm(val_loader, desc="Validating"):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            with autocast(enabled=self.use_amp):
                outputs = self.model(images, return_aux=True, return_boundary=True)
                losses = self.criterion(outputs, masks)
            
            total_loss += losses['total'].item()
            
            preds = torch.argmax(outputs['main'], dim=1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(masks.cpu().numpy())
        
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # Compute metrics
        seg_metrics = compute_metrics(all_preds, all_targets)
        boundary_metrics = compute_boundary_metrics(all_preds, all_targets)
        
        return {
            'loss': total_loss / len(val_loader),
            **seg_metrics,
            **boundary_metrics
        }
    
    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.use_amp else None,
            'best_miou': self.best_miou,
            'epochs_without_improvement': self.epochs_without_improvement,
            'metrics': metrics
        }
        
        # Save latest
        latest_path = self.output_dir / "checkpoints" / "latest.pth"
        torch.save(checkpoint, latest_path)
        
        # Save best
        if is_best:
            best_path = self.output_dir / "checkpoints" / "best.pth"
            torch.save(checkpoint, best_path)
            print(f"  *** New best mIoU: {metrics['miou']:.4f} ***")
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int):
        """Full training loop"""
        
        print("\n" + "=" * 70)
        print("TRAINING: AURASeg V4 with ResNet-50")
        print("=" * 70)
        print(f"Epochs: {epochs}")
        print(f"Encoder LR: {self.config.LR_ENCODER}")
        print(f"Decoder LR: {self.config.LR_DECODER}")
        print(f"Batch size: {self.config.BATCH_SIZE}")
        print(f"Loss: Focal({self.config.FOCAL_WEIGHT}) + Dice({self.config.DICE_WEIGHT}) + Boundary({self.config.BOUNDARY_WEIGHT}) + Aux({self.config.AUX_WEIGHT})")
        print(f"Output: {self.output_dir}")
        print("=" * 70 + "\n")
        
        for epoch in range(self.start_epoch, epochs + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{epochs}")
            print(f"{'='*60}")
            
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            print(f"[TRAIN] Loss: {train_metrics['loss']:.4f} (Seg: {train_metrics['seg_loss']:.4f}, Bnd: {train_metrics['boundary_loss']:.4f})")
            
            # Validate
            val_metrics = self.validate(val_loader)
            print(f"[VAL] Loss: {val_metrics['loss']:.4f}, mIoU: {val_metrics['miou']:.4f}, B-IoU: {val_metrics['boundary_iou']:.4f}")
            print(f"      IoU(Drivable): {val_metrics['iou_drivable']:.4f}, Dice: {val_metrics['dice']:.4f}, B-F1: {val_metrics['boundary_f1']:.4f}")
            
            # Step scheduler
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            print(f"      LR: {current_lr:.6f}")
            
            # Check for improvement
            is_best = val_metrics['miou'] > self.best_miou + self.config.MIN_DELTA
            if is_best:
                self.best_miou = val_metrics['miou']
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_metrics, is_best)
            
            # Early stopping
            if self.epochs_without_improvement >= self.config.PATIENCE:
                print(f"\n[INFO] Early stopping triggered at epoch {epoch}")
                break
        
        print("\n" + "=" * 70)
        print(f"TRAINING COMPLETE - AURASeg V4 with ResNet-50")
        print(f"Best mIoU: {self.best_miou:.4f}")
        print(f"Checkpoints: {self.output_dir / 'checkpoints'}")
        print("=" * 70)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train AURASeg V4 with ResNet-50')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr-encoder', type=float, default=1e-4)
    parser.add_argument('--lr-decoder', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dataset-root', type=str, default='CommonDataset',
                        help='Dataset root (e.g., CommonDataset or carl-dataset)')
    parser.add_argument('--train-split', type=str, default='train',
                        help='Training split name')
    parser.add_argument('--val-split', type=str, default='val',
                        help='Validation split name')
    parser.add_argument('--runs-dir', type=str, default='runs',
                        help='Base output directory for runs')
    parser.add_argument('--run-name', type=str, default='auraseg_v4_resnet50',
                        help='Run directory name under runs-dir')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='DataLoader workers')
    parser.add_argument('--resume', type=str, default=None,
                        help="Path to checkpoint to resume from (or 'latest' to use OUTPUT_DIR/checkpoints/latest.pth)")
    
    args = parser.parse_args()
    
    # Setup config
    config = Config()
    config.EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.LR_ENCODER = args.lr_encoder
    config.LR_DECODER = args.lr_decoder
    config.NUM_WORKERS = args.num_workers

    repo_root = Path(__file__).parent.parent
    dataset_root = Path(args.dataset_root)
    if not dataset_root.is_absolute():
        dataset_root = repo_root / dataset_root

    runs_dir = Path(args.runs_dir)
    if not runs_dir.is_absolute():
        runs_dir = repo_root / runs_dir
    config.OUTPUT_DIR = runs_dir / args.run_name
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Output directory: {config.OUTPUT_DIR}")
    print(f"[INFO] Dataset root: {dataset_root} (train='{args.train_split}', val='{args.val_split}')")

    resume_path = None
    if args.resume:
        if args.resume.lower() in {"latest", "auto"}:
            resume_path = config.OUTPUT_DIR / "checkpoints" / "latest.pth"
        else:
            resume_path = Path(args.resume)
            if not resume_path.is_absolute():
                resume_path = repo_root / resume_path
        print(f"[INFO] Resume checkpoint: {resume_path}")
    
    # Create datasets
    normalization = Normalization(mean=tuple(config.MEAN), std=tuple(config.STD))

    train_dataset = UnifiedDrivableAreaDataset(
        dataset_root=dataset_root,
        split=args.train_split,
        img_size=config.IMG_SIZE,
        transform=True,
        normalization=normalization,
        return_names=False,
    )
    
    val_dataset = UnifiedDrivableAreaDataset(
        dataset_root=dataset_root,
        split=args.val_split,
        img_size=config.IMG_SIZE,
        transform=False,
        normalization=normalization,
        return_names=False,
    )
    
    print(f"[INFO] Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    # Create trainer and start training
    trainer = AURASegTrainer(config, device)
    if resume_path is not None:
        trainer.load_checkpoint(resume_path)
    trainer.train(train_loader, val_loader, config.EPOCHS)


if __name__ == "__main__":
    main()
