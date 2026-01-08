"""
Utility Functions for AURASeg Ablation Study
Paper: AURASeg - Attention Guided Upsampling with Residual Boundary-Assistive Refinement

Contains:
    - Dataset class for BDD100K drivable area
    - Metrics (IoU, Dice, Precision, Recall)
    - Visualization utilities
    - Training helpers
"""

import os
import random
import numpy as np
from PIL import Image
from typing import Tuple, Optional, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF


class DrivableAreaDataset(Dataset):
    """
    Dataset for BDD100K Drivable Area Segmentation
    
    Loads image-mask pairs from specified directories.
    Supports data augmentation for training.
    """
    
    def __init__(self, image_dir: str, mask_dir: str, 
                 img_size: Tuple[int, int] = (384, 640),
                 split: str = 'train',
                 transform: bool = True,
                 mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
                 std: Tuple[float, ...] = (0.229, 0.224, 0.225)):
        """
        Args:
            image_dir: Path to images base directory (contains train/val subdirs)
            mask_dir: Path to masks base directory (contains train/val subdirs)
            img_size: Target size (H, W)
            split: 'train' or 'val'
            transform: Whether to apply augmentation (train only)
            mean: Normalization mean
            std: Normalization std
        """
        # Handle directories with train/val subdirs
        self.image_dir = os.path.join(image_dir, split) if os.path.isdir(os.path.join(image_dir, split)) else image_dir
        self.mask_dir = os.path.join(mask_dir, split) if os.path.isdir(os.path.join(mask_dir, split)) else mask_dir
        self.img_size = img_size
        self.split = split
        self.transform = transform and (split == 'train')
        self.mean = mean
        self.std = std
        
        # Get list of image files
        self.images = self._get_image_list()
        
        print(f"[{split.upper()}] Loaded {len(self.images)} samples from {self.image_dir}")
    
    def _get_image_list(self) -> List[str]:
        """Get list of image filenames"""
        image_files = []
        
        if os.path.exists(self.image_dir):
            for f in os.listdir(self.image_dir):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    # Check if corresponding mask exists (try both .png and .jpg)
                    base_name = os.path.splitext(f)[0]
                    mask_found = False
                    for ext in ['.png', '.jpg', '.jpeg']:
                        mask_path = os.path.join(self.mask_dir, base_name + ext)
                        if os.path.exists(mask_path):
                            image_files.append(f)
                            mask_found = True
                            break
                    if not mask_found:
                        # Try original name replacement
                        mask_name = f.replace('.jpg', '.png').replace('.jpeg', '.png')
                        mask_path = os.path.join(self.mask_dir, mask_name)
                        if os.path.exists(mask_path):
                            image_files.append(f)
        
        return sorted(image_files)
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load image
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        # Load mask - try multiple extensions
        base_name = os.path.splitext(img_name)[0]
        mask_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            candidate = os.path.join(self.mask_dir, base_name + ext)
            if os.path.exists(candidate):
                mask_path = candidate
                break
        if mask_path is None:
            # Fallback
            mask_path = os.path.join(self.mask_dir, img_name.replace('.jpg', '.png').replace('.jpeg', '.png'))
        mask = Image.open(mask_path).convert('L')
        
        # Resize to target size
        image = image.resize((self.img_size[1], self.img_size[0]), Image.BILINEAR)
        mask = mask.resize((self.img_size[1], self.img_size[0]), Image.NEAREST)
        
        # Apply augmentations
        if self.transform:
            image, mask = self._augment(image, mask)
        
        # Convert to tensors
        image = TF.to_tensor(image)
        mask = torch.from_numpy(np.array(mask)).long()
        
        # Normalize image
        image = TF.normalize(image, self.mean, self.std)
        
        # Ensure mask is binary (0 or 1)
        mask = (mask > 0).long()
        
        return image, mask
    
    def _augment(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        """Apply data augmentation"""
        # Random horizontal flip
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        
        # Random rotation
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            image = TF.rotate(image, angle, fill=0)
            mask = TF.rotate(mask, angle, fill=0)
        
        # Random brightness/contrast
        if random.random() > 0.5:
            brightness = random.uniform(0.8, 1.2)
            image = TF.adjust_brightness(image, brightness)
        
        if random.random() > 0.5:
            contrast = random.uniform(0.8, 1.2)
            image = TF.adjust_contrast(image, contrast)
        
        return image, mask


def compute_iou(pred: torch.Tensor, target: torch.Tensor, 
                num_classes: int = 2, eps: float = 1e-6) -> torch.Tensor:
    """
    Compute IoU (Intersection over Union) for each class
    
    Args:
        pred: Predictions (B, H, W) - class indices
        target: Ground truth (B, H, W) - class indices
        num_classes: Number of classes
        eps: Small value to avoid division by zero
        
    Returns:
        IoU for each class (num_classes,)
    """
    ious = []
    
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        
        iou = (intersection + eps) / (union + eps)
        ious.append(iou)
    
    return torch.stack(ious)


def compute_dice(pred: torch.Tensor, target: torch.Tensor,
                 num_classes: int = 2, eps: float = 1e-6) -> torch.Tensor:
    """
    Compute Dice coefficient for each class
    
    Args:
        pred: Predictions (B, H, W) - class indices
        target: Ground truth (B, H, W) - class indices
        num_classes: Number of classes
        eps: Small value to avoid division by zero
        
    Returns:
        Dice for each class (num_classes,)
    """
    dices = []
    
    for cls in range(num_classes):
        pred_cls = (pred == cls).float()
        target_cls = (target == cls).float()
        
        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum()
        
        dice = (2 * intersection + eps) / (union + eps)
        dices.append(dice)
    
    return torch.stack(dices)


def compute_precision_recall_f1(pred: torch.Tensor, target: torch.Tensor,
                                num_classes: int = 2, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute Precision, Recall, and F1-score for each class
    
    Args:
        pred: Predictions (B, H, W) - class indices
        target: Ground truth (B, H, W) - class indices
        num_classes: Number of classes
        eps: Small value to avoid division by zero
        
    Returns:
        Tuple of (Precision, Recall, F1) tensors, each of shape (num_classes,)
    """
    precisions = []
    recalls = []
    f1s = []
    
    for cls in range(num_classes):
        pred_cls = (pred == cls).float()
        target_cls = (target == cls).float()
        
        tp = (pred_cls * target_cls).sum()
        fp = (pred_cls * (1 - target_cls)).sum()
        fn = ((1 - pred_cls) * target_cls).sum()
        
        precision = (tp + eps) / (tp + fp + eps)
        recall = (tp + eps) / (tp + fn + eps)
        f1 = (2 * precision * recall + eps) / (precision + recall + eps)
        
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    
    return torch.stack(precisions), torch.stack(recalls), torch.stack(f1s)


def compute_metrics(pred: torch.Tensor, target: torch.Tensor,
                    num_classes: int = 2) -> Dict[str, float]:
    """
    Compute all evaluation metrics
    
    Args:
        pred: Predictions (B, H, W) - class indices
        target: Ground truth (B, H, W) - class indices
        num_classes: Number of classes
        
    Returns:
        Dictionary of metrics
    """
    iou = compute_iou(pred, target, num_classes)
    dice = compute_dice(pred, target, num_classes)
    precision, recall, f1 = compute_precision_recall_f1(pred, target, num_classes)
    
    # Accuracy
    accuracy = (pred == target).float().mean()
    
    return {
        'iou_bg': iou[0].item(),
        'iou_drivable': iou[1].item(),
        'miou': iou.mean().item(),
        'dice_bg': dice[0].item(),
        'dice_drivable': dice[1].item(),
        'mdice': dice.mean().item(),
        'accuracy': accuracy.item(),
        'precision_drivable': precision[1].item(),
        'recall_drivable': recall[1].item(),
        'f1_drivable': f1[1].item()
    }


def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience: int = 15, min_delta: float = 0.001, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            improvement = score - self.best_score
        else:
            improvement = self.best_score - score
        
        if improvement > self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


def save_checkpoint(state: dict, filepath: str):
    """Save training checkpoint"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)
    print(f"Checkpoint saved: {filepath}")


def load_checkpoint(filepath: str, model: nn.Module, 
                    optimizer: Optional[torch.optim.Optimizer] = None,
                    scheduler: Optional[object] = None) -> dict:
    """Load training checkpoint"""
    checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"Checkpoint loaded: {filepath}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Best mIoU: {checkpoint.get('best_miou', 'N/A'):.4f}")
    
    return checkpoint


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Utility Functions")
    print("=" * 60)
    
    # Test metrics
    print("\n1. Testing Metrics:")
    pred = torch.randint(0, 2, (4, 384, 640))
    target = torch.randint(0, 2, (4, 384, 640))
    
    metrics = compute_metrics(pred, target, num_classes=2)
    for name, value in metrics.items():
        print(f"   {name}: {value:.4f}")
    
    # Test average meter
    print("\n2. Testing AverageMeter:")
    meter = AverageMeter()
    for i in range(10):
        meter.update(i)
    print(f"   Average: {meter.avg:.2f}")
    
    # Test early stopping
    print("\n3. Testing EarlyStopping:")
    early_stop = EarlyStopping(patience=3)
    scores = [0.8, 0.85, 0.84, 0.83, 0.82, 0.81]
    for i, score in enumerate(scores):
        stop = early_stop(score)
        print(f"   Epoch {i}: score={score:.2f}, stop={stop}")
    
    print("\n" + "=" * 60)
    print("All utilities working correctly!")
    print("=" * 60)
