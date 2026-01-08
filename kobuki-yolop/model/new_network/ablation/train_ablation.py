"""
Ablation Study Training Script
==============================
Comprehensive training script for ablation experiments with:
- Focal + Dice loss
- Deep supervision support
- Metrics tracking (IoU, Dice, Precision, Recall)
- Learning rate scheduling with warmup
- Mixed precision training
- TensorBoard logging
- Visualization of predictions
- Early stopping
"""

import os
import sys
import time
import random
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import AblationConfig, get_config
from losses import SingleScaleLoss, CombinedLoss, BoundaryWeightedLoss
from ablation_v1_base import AblationBaseModel
from ablation_v2_assplite import AblationV2ASPPLite
from ablation_v3_apud import AblationV3APUD
from ablation_v4_rbrm import AblationV4RBRM


# ============================================================================
# Dataset
# ============================================================================

class SegmentationDataset(Dataset):
    """Segmentation dataset with augmentation support."""
    
    def __init__(
        self, 
        image_dir: str, 
        mask_dir: str, 
        transform=None,
        target_transform=None,
        use_augmentation: bool = False,
        horizontal_flip_prob: float = 0.5,
        color_jitter: bool = True
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.target_transform = target_transform
        self.use_augmentation = use_augmentation
        self.horizontal_flip_prob = horizontal_flip_prob
        self.color_jitter = color_jitter
        
        # Get image list
        self.images = sorted([f for f in os.listdir(image_dir) 
                            if f.endswith(('.jpg', '.png', '.jpeg'))])
        
        # Color jitter transform
        if color_jitter:
            self.color_transform = transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            )
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        image_path = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(image_path).convert("RGB")
        
        # Load mask
        mask_name = self.images[idx].replace('.jpg', '.png').replace('.jpeg', '.png')
        mask_path = os.path.join(self.mask_dir, mask_name)
        mask = Image.open(mask_path).convert("L")
        
        # Apply augmentation
        if self.use_augmentation:
            # Horizontal flip
            if random.random() < self.horizontal_flip_prob:
                image = transforms.functional.hflip(image)
                mask = transforms.functional.hflip(mask)
            
            # Color jitter (only on image)
            if self.color_jitter:
                image = self.color_transform(image)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
        
        # Convert mask to long tensor and squeeze
        mask = mask.squeeze(0).long()
        
        return image, mask


# ============================================================================
# Metrics
# ============================================================================

class SegmentationMetrics:
    """Compute segmentation metrics."""
    
    def __init__(self, num_classes: int = 2, ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        """Reset all counters."""
        self.intersection = torch.zeros(self.num_classes)
        self.union = torch.zeros(self.num_classes)
        self.pred_sum = torch.zeros(self.num_classes)
        self.target_sum = torch.zeros(self.num_classes)
        self.total_correct = 0
        self.total_pixels = 0
    
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Update metrics with new predictions.
        
        Args:
            pred: Predictions (B, C, H, W) or (B, H, W)
            target: Ground truth (B, H, W)
        """
        # Convert to class predictions if needed
        if pred.dim() == 4:
            pred = pred.argmax(dim=1)
        
        # Move to CPU
        pred = pred.cpu()
        target = target.cpu()
        
        # Create valid mask
        valid_mask = target != self.ignore_index
        
        # Overall accuracy
        self.total_correct += ((pred == target) & valid_mask).sum().item()
        self.total_pixels += valid_mask.sum().item()
        
        # Per-class metrics
        for c in range(self.num_classes):
            pred_c = (pred == c) & valid_mask
            target_c = (target == c) & valid_mask
            
            self.intersection[c] += (pred_c & target_c).sum().item()
            self.union[c] += (pred_c | target_c).sum().item()
            self.pred_sum[c] += pred_c.sum().item()
            self.target_sum[c] += target_c.sum().item()
    
    def compute(self) -> Dict[str, float]:
        """Compute all metrics."""
        metrics = {}
        
        # Overall accuracy
        metrics['accuracy'] = self.total_correct / max(self.total_pixels, 1)
        
        # Per-class IoU
        iou = self.intersection / (self.union + 1e-10)
        for c in range(self.num_classes):
            metrics[f'iou_class{c}'] = iou[c].item()
        metrics['miou'] = iou.mean().item()
        
        # Per-class Dice
        dice = 2 * self.intersection / (self.pred_sum + self.target_sum + 1e-10)
        for c in range(self.num_classes):
            metrics[f'dice_class{c}'] = dice[c].item()
        metrics['mdice'] = dice.mean().item()
        
        # Precision and Recall for positive class (assuming class 1 is drivable area)
        if self.num_classes == 2:
            precision = self.intersection[1] / (self.pred_sum[1] + 1e-10)
            recall = self.intersection[1] / (self.target_sum[1] + 1e-10)
            f1 = 2 * precision * recall / (precision + recall + 1e-10)
            
            metrics['precision'] = precision.item()
            metrics['recall'] = recall.item()
            metrics['f1'] = f1.item()
        
        return metrics


# ============================================================================
# Learning Rate Scheduler with Warmup
# ============================================================================

class WarmupCosineScheduler:
    """Learning rate scheduler with warmup and cosine decay."""
    
    def __init__(
        self, 
        optimizer: optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        warmup_lr: float,
        base_lr: float,
        min_lr: float
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.warmup_lr = warmup_lr
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_epoch = 0
        
    def step(self, epoch: int):
        """Update learning rate for given epoch."""
        self.current_epoch = epoch
        
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.warmup_lr + (self.base_lr - self.warmup_lr) * epoch / self.warmup_epochs
        else:
            # Cosine decay
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']


# ============================================================================
# Trainer
# ============================================================================

class AblationTrainer:
    """Trainer for ablation study experiments."""
    
    def __init__(self, config: AblationConfig):
        self.config = config
        
        # Strict CUDA check
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available! Training cannot proceed on GPU as requested.")
        
        self.device = torch.device('cuda')
        print(f"✅ Using CUDA Device: {torch.cuda.get_device_name(0)}")
        
        # Setup output directory
        self.output_dir = config.get_output_path()
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'predictions'), exist_ok=True)
        
        # Setup text logger
        self.log_file = os.path.join(self.output_dir, 'train_log.txt')
        with open(self.log_file, 'w') as f:
            f.write(f"Training Log - {datetime.now()}\n")
            f.write(f"Model: {config.model.model_name}\n")
            f.write("="*60 + "\n")
        
        # Save config
        config.save(os.path.join(self.output_dir, 'config.json'))
        
        # Set random seed
        self._set_seed(config.training.seed)
        
        # Setup model
        self.model = self._build_model()
        
        # Setup data
        self.train_loader, self.val_loader = self._build_dataloaders()
        
        # Setup loss
        self.criterion = self._build_criterion()
        
        # Setup optimizer and scheduler
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        
        # Mixed precision
        self.scaler = GradScaler() if config.training.use_amp else None
        
        # Metrics
        self.train_metrics = SegmentationMetrics(
            num_classes=config.model.num_classes,
            ignore_index=config.loss.ignore_index
        )
        self.val_metrics = SegmentationMetrics(
            num_classes=config.model.num_classes,
            ignore_index=config.loss.ignore_index
        )
        
        # Training state
        self.current_epoch = 0
        self.best_miou = 0.0
        self.best_epoch = 0
        self.early_stop_counter = 0
        
        # Resume from checkpoint if provided
        if hasattr(config, 'resume') and config.resume:
            if os.path.isfile(config.resume):
                print(f"Loading checkpoint '{config.resume}'")
                checkpoint = torch.load(config.resume, map_location=self.device, weights_only=False)
                self.start_epoch = checkpoint['epoch'] + 1
                self.best_miou = checkpoint.get('best_miou', 0.0)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if self.scheduler and 'scheduler_state_dict' in checkpoint:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print(f"Resumed from epoch {self.start_epoch}")
            else:
                print(f"No checkpoint found at '{config.resume}'")
                self.start_epoch = 0
        else:
            self.start_epoch = 0

        # History for plotting
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_miou': [],
            'val_miou': [],
            'train_dice': [],
            'val_dice': [],
            'lr': []
        }
        
        # TensorBoard
        self.writer = None
        if config.logging.use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(os.path.join(self.output_dir, 'tensorboard'))
        
        print(f"\n{'='*60}")
        print(f"Ablation Study Trainer Initialized")
        print(f"{'='*60}")
        print(f"Model: {config.model.model_name}")
        print(f"Device: {self.device}")
        print(f"Output: {self.output_dir}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"{'='*60}\n")
    
    def log(self, message: str):
        """Log message to console and file."""
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(message + '\n')

    def _set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def _build_model(self) -> nn.Module:
        """Build model based on config."""
        model_name = self.config.model.model_name
        
        if model_name == "ablation_v1_base":
            model = AblationBaseModel(
                num_classes=self.config.model.num_classes,
                in_channels=self.config.model.in_channels
            )
        elif model_name == "ablation_v2_assplite":
            model = AblationV2ASPPLite(
                num_classes=self.config.model.num_classes,
                in_channels=self.config.model.in_channels
            )
        elif model_name == "ablation_v3_apud":
            # V3: CSPDarknet + ASPPLite + APUD decoder
            model = AblationV3APUD(
                num_classes=self.config.model.num_classes,
                in_channels=self.config.model.in_channels,
                deep_supervision=getattr(self.config.model, 'deep_supervision', False)
            )
        elif model_name == "ablation_v4_rbrm":
            # V4: CSPDarknet + ASPPLite + APUD decoder + RBRM
            # Check both config locations for freeze_backbone
            freeze = getattr(self.config.training, 'freeze_backbone', False) or \
                     getattr(self.config.model, 'freeze_backbone', False)
            
            model = AblationV4RBRM(
                num_classes=self.config.model.num_classes,
                in_channels=self.config.model.in_channels,
                deep_supervision=getattr(self.config.model, 'deep_supervision', False),
                freeze_backbone=freeze
            )
            
            # Load V3 weights if provided
            v3_ckpt = getattr(self.config.model, 'v3_checkpoint', None)
            if v3_ckpt:
                print(f"Loading V3 weights from {v3_ckpt}...")
                model.load_v3_weights(v3_ckpt)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Load pretrained weights if specified
        if self.config.model.pretrained_path:
            state_dict = torch.load(self.config.model.pretrained_path, map_location='cpu', weights_only=False)
            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded pretrained weights from {self.config.model.pretrained_path}")
        
        # Freeze layers if specified
        if self.config.model.freeze_encoder:
            for param in model.encoder.parameters():
                param.requires_grad = False
            print("Encoder frozen")
        
        if self.config.model.freeze_decoder:
            for param in model.decoder.parameters():
                param.requires_grad = False
            print("Decoder frozen")
        
        return model.to(self.device)
    
    def _build_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """Build train and validation dataloaders."""
        cfg = self.config.data
        
        # Transforms
        image_transform = transforms.Compose([
            transforms.Resize((cfg.input_height, cfg.input_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        mask_transform = transforms.Compose([
            transforms.Resize((cfg.input_height, cfg.input_width), interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])
        
        # Determine paths
        if cfg.train_image_dir and cfg.train_mask_dir:
            train_img_dir = cfg.train_image_dir
            train_mask_dir = cfg.train_mask_dir
            val_img_dir = cfg.val_image_dir
            val_mask_dir = cfg.val_mask_dir
        else:
            # Use standard structure
            train_img_dir = os.path.join(cfg.dataset_dir, 'images', 'train')
            train_mask_dir = os.path.join(cfg.dataset_dir, 'labels', 'train')
            val_img_dir = os.path.join(cfg.dataset_dir, 'images', 'val')
            val_mask_dir = os.path.join(cfg.dataset_dir, 'labels', 'val')
            
        # Train dataset
        train_dataset = SegmentationDataset(
            image_dir=train_img_dir,
            mask_dir=train_mask_dir,
            transform=image_transform,
            target_transform=mask_transform,
            use_augmentation=cfg.use_augmentation,
            horizontal_flip_prob=cfg.horizontal_flip_prob,
            color_jitter=cfg.color_jitter
        )
        
        # Validation dataset
        val_dataset = SegmentationDataset(
            image_dir=val_img_dir,
            mask_dir=val_mask_dir,
            transform=image_transform,
            target_transform=mask_transform,
            use_augmentation=False, # No augmentation for val
            horizontal_flip_prob=0.0,
            color_jitter=False
        )
        
        print(f"Dataset: {len(train_dataset)} train, {len(val_dataset)} val")
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory
        )
        
        return train_loader, val_loader
    
    def _build_criterion(self):
        """Build loss function."""
        cfg = self.config.loss
        
        # Check boundary loss first (for V4)
        if getattr(cfg, 'use_boundary_loss', False):
            # Boundary-weighted loss for V4 RBRM training
            print("[Loss] Using Boundary-Weighted Loss (emphasizes edges)")
            return BoundaryWeightedLoss(
                lambda_focal=cfg.lambda_focal,
                lambda_dice=cfg.lambda_dice,
                focal_alpha=cfg.focal_alpha,
                focal_gamma=cfg.focal_gamma,
                boundary_width=getattr(cfg, 'boundary_width', 5),
                boundary_weight=getattr(cfg, 'boundary_weight', 5.0),
                interior_weight=getattr(cfg, 'interior_weight', 0.5),
                ignore_index=cfg.ignore_index
            )
            
        if cfg.use_deep_supervision:
            return CombinedLoss(
                num_supervisors=4,
                lambda_focal=cfg.lambda_focal,
                lambda_dice=cfg.lambda_dice,
                focal_alpha=cfg.focal_alpha,
                focal_gamma=cfg.focal_gamma,
                supervision_weights=cfg.supervision_weights,
                ignore_index=cfg.ignore_index
            )
        else:
            return SingleScaleLoss(
                lambda_focal=cfg.lambda_focal,
                lambda_dice=cfg.lambda_dice,
                focal_alpha=cfg.focal_alpha,
                focal_gamma=cfg.focal_gamma,
                ignore_index=cfg.ignore_index
            )
    
    def _build_optimizer(self) -> optim.Optimizer:
        """Build optimizer."""
        cfg = self.config.optimizer
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        
        if cfg.optimizer.lower() == 'adam':
            return optim.Adam(params, lr=cfg.learning_rate, betas=cfg.betas, 
                            weight_decay=cfg.weight_decay)
        elif cfg.optimizer.lower() == 'adamw':
            return optim.AdamW(params, lr=cfg.learning_rate, betas=cfg.betas,
                             weight_decay=cfg.weight_decay)
        elif cfg.optimizer.lower() == 'sgd':
            return optim.SGD(params, lr=cfg.learning_rate, momentum=cfg.momentum,
                           weight_decay=cfg.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {cfg.optimizer}")
    
    def _build_optimizer_with_params(self, params) -> optim.Optimizer:
        """Build optimizer with specific parameters (for frozen backbone training)."""
        cfg = self.config.optimizer
        
        if cfg.optimizer.lower() == 'adam':
            return optim.Adam(params, lr=cfg.learning_rate, betas=cfg.betas, 
                            weight_decay=cfg.weight_decay)
        elif cfg.optimizer.lower() == 'adamw':
            return optim.AdamW(params, lr=cfg.learning_rate, betas=cfg.betas,
                             weight_decay=cfg.weight_decay)
        elif cfg.optimizer.lower() == 'sgd':
            return optim.SGD(params, lr=cfg.learning_rate, momentum=cfg.momentum,
                           weight_decay=cfg.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {cfg.optimizer}")
    
    def _build_scheduler(self):
        """Build learning rate scheduler."""
        cfg = self.config.optimizer
        
        if cfg.scheduler.lower() == 'cosine':
            return WarmupCosineScheduler(
                optimizer=self.optimizer,
                warmup_epochs=cfg.warmup_epochs,
                total_epochs=self.config.training.epochs,
                warmup_lr=cfg.warmup_lr,
                base_lr=cfg.learning_rate,
                min_lr=cfg.min_lr
            )
        elif cfg.scheduler.lower() == 'step':
            # No warmup for step scheduler
            return optim.lr_scheduler.StepLR(
                self.optimizer, step_size=cfg.step_size, gamma=cfg.gamma
            )
        elif cfg.scheduler.lower() == 'none':
            return None
        else:
            raise ValueError(f"Unknown scheduler: {cfg.scheduler}")
    
    def train_epoch(self) -> Tuple[float, Dict]:
        """Train for one epoch."""
        self.model.train()
        self.train_metrics.reset()
        
        total_loss = 0.0
        loss_components = {'focal': 0.0, 'dice': 0.0}
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1} [Train]")
        
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)
            # Forward pass with mixed precision
            with autocast('cuda', enabled=self.config.training.use_amp):
                outputs = self.model(images)
                # Handle V4 RBRM output (segmentation + optional edge branch)
                edge_logits = None
                if isinstance(outputs, tuple):
                    if outputs[-1].shape[1] == 1:
                        edge_logits = outputs[-1]
                        outputs = outputs[:-1]
                    if isinstance(outputs, tuple) and len(outputs) == 1:
                        outputs = outputs[0]

                if self.config.loss.use_deep_supervision:
                    loss, loss_dict = self.criterion(outputs, masks)
                else:
                    loss, loss_dict = self.criterion(outputs, masks)

            # Compute Edge Loss if available
            if edge_logits is not None:
                # Generate edge target on the fly
                with torch.no_grad():
                    mask_float = masks.float().unsqueeze(1)
                    dilated = torch.nn.functional.max_pool2d(mask_float, 3, stride=1, padding=1)
                    eroded = -torch.nn.functional.max_pool2d(-mask_float, 3, stride=1, padding=1)
                    edge_target = (dilated - eroded).abs()

                # Resize edge_logits to match target if needed
                if edge_logits.shape[2:] != edge_target.shape[2:]:
                    edge_logits = torch.nn.functional.interpolate(edge_logits, size=edge_target.shape[2:], mode='bilinear', align_corners=False)

                # Weighted BCE Loss for edges (edges are sparse)
                pos_weight = torch.tensor([5.0]).to(self.device)  # Weight positive edges more
                edge_loss_val = torch.nn.functional.binary_cross_entropy_with_logits(edge_logits, edge_target, pos_weight=pos_weight)

                loss += 0.5 * edge_loss_val  # Add weighted edge loss
                loss_dict['edge'] = edge_loss_val.item()

            # Backward pass
            self.optimizer.zero_grad()

            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            # Update metrics
            with torch.no_grad():
                if self.config.loss.use_deep_supervision:
                    # Use finest output for metrics
                    pred = outputs[-1] if isinstance(outputs, (list, tuple)) else outputs
                else:
                    pred = outputs

                # DEBUG: Check shapes before interpolate
                if isinstance(pred, tuple):
                    # If pred is still a tuple, take the first element (main output)
                    pred = pred[0]

                # Resize prediction to match mask size
                if pred.dim() != 4:
                    print(f"WARNING: pred shape mismatch: {pred.shape}. Expected 4 dims.")
                    if pred.dim() == 3:
                        pred = pred.unsqueeze(1)
                    if pred.shape[0] != masks.shape[0]:
                        # If batch size mismatches, try to fix by repeating or slicing
                        if pred.shape[0] == 1 and masks.shape[0] > 1:
                            pred = pred.repeat(masks.shape[0], 1, 1, 1)
                        else:
                            pred = pred[:masks.shape[0]]
                pred = torch.nn.functional.interpolate(
                    pred, size=masks.shape[1:], mode='bilinear', align_corners=False
                )
                self.train_metrics.update(pred, masks)

            total_loss += loss.item()
            loss_components['focal'] += loss_dict.get('focal', 0)
            loss_components['dice'] += loss_dict.get('dice', 0)

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.scheduler.get_lr():.2e}" if self.scheduler else f"{self.config.optimizer.learning_rate:.2e}"
            })

        # Compute epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        for k in loss_components:
            loss_components[k] /= len(self.train_loader)
        
        metrics = self.train_metrics.compute()
        metrics['loss'] = avg_loss
        metrics['loss_focal'] = loss_components['focal']
        metrics['loss_dice'] = loss_components['dice']
        
        return avg_loss, metrics
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, Dict]:
        """Validate the model."""
        self.model.eval()
        self.val_metrics.reset()
        
        total_loss = 0.0
        loss_components = {'focal': 0.0, 'dice': 0.0}
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch+1} [Val]")
        
        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            with autocast('cuda', enabled=self.config.training.use_amp):
                outputs = self.model(images)
                
                # Handle V4 RBRM output (out, aux, edge_logits) or (out, edge_logits)
                edge_logits = None
                if isinstance(outputs, tuple):
                    # Check if the last element is edge_logits (1 channel)
                    if outputs[-1].shape[1] == 1:
                        edge_logits = outputs[-1]
                        outputs = outputs[:-1] # Remove edge_logits from outputs
                    
                    # If still a tuple and length is 1, unpack it
                    if isinstance(outputs, tuple) and len(outputs) == 1:
                        outputs = outputs[0]
                
                if self.config.loss.use_deep_supervision:
                    loss, loss_dict = self.criterion(outputs, masks)
                else:
                    loss, loss_dict = self.criterion(outputs, masks)
            
            # Update metrics
            if self.config.loss.use_deep_supervision:
                pred = outputs[-1] if isinstance(outputs, (list, tuple)) else outputs
            else:
                pred = outputs
            
            # Debug print for shape mismatch
            if pred.dim() != 4:
                print(f"DEBUG: pred shape mismatch! Expected 4 dims, got {pred.shape}")
                if isinstance(pred, tuple):
                    print(f"DEBUG: pred is a tuple of length {len(pred)}")
                    for idx, p in enumerate(pred):
                        if isinstance(p, torch.Tensor):
                            print(f"DEBUG: pred[{idx}] shape: {p.shape}")
                        else:
                            print(f"DEBUG: pred[{idx}] type: {type(p)}")
            
            pred = torch.nn.functional.interpolate(
                pred, size=masks.shape[1:], mode='bilinear', align_corners=False
            )
            self.val_metrics.update(pred, masks)
            
            total_loss += loss.item()
            loss_components['focal'] += loss_dict.get('focal', 0)
            loss_components['dice'] += loss_dict.get('dice', 0)
        
        # Compute metrics
        avg_loss = total_loss / len(self.val_loader)
        for k in loss_components:
            loss_components[k] /= len(self.val_loader)
        
        metrics = self.val_metrics.compute()
        metrics['loss'] = avg_loss
        metrics['loss_focal'] = loss_components['focal']
        metrics['loss_dice'] = loss_components['dice']
        
        return avg_loss, metrics
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_miou': self.best_miou,
            'config': self.config.__dict__
        }
        
        # Save latest
        latest_path = os.path.join(self.output_dir, 'checkpoints', 'latest.pth')
        torch.save(checkpoint, latest_path)
        
        # Save periodic
        if (self.current_epoch + 1) % self.config.training.save_every == 0:
            epoch_path = os.path.join(
                self.output_dir, 'checkpoints', f'epoch_{self.current_epoch+1}.pth'
            )
            torch.save(checkpoint, epoch_path)
        
        # Save best
        if is_best:
            best_path = os.path.join(self.output_dir, 'checkpoints', 'best.pth')
            torch.save(checkpoint, best_path)
            self.log(f"  -> Saved best model (mIoU: {self.best_miou:.4f})")
    
    def log_tensorboard(self, train_metrics: Dict, val_metrics: Dict, lr: float):
        """Log metrics to TensorBoard."""
        if self.writer is None:
            return
        
        epoch = self.current_epoch
        
        # Loss
        self.writer.add_scalars('Loss', {
            'train': train_metrics['loss'],
            'val': val_metrics['loss']
        }, epoch)
        
        # mIoU
        self.writer.add_scalars('mIoU', {
            'train': train_metrics['miou'],
            'val': val_metrics['miou']
        }, epoch)
        
        # Dice
        self.writer.add_scalars('mDice', {
            'train': train_metrics['mdice'],
            'val': val_metrics['mdice']
        }, epoch)
        
        # Learning rate
        self.writer.add_scalar('Learning_Rate', lr, epoch)
        
        # Per-class metrics
        for c in range(self.config.model.num_classes):
            self.writer.add_scalars(f'IoU_Class{c}', {
                'train': train_metrics.get(f'iou_class{c}', 0),
                'val': val_metrics.get(f'iou_class{c}', 0)
            }, epoch)
    
    def save_predictions(self, num_samples: int = 4):
        """Save prediction visualizations."""
        self.model.eval()
        
        # Get a batch from validation
        try:
            images, masks = next(iter(self.val_loader))
        except StopIteration:
            return
            
        images = images[:num_samples].to(self.device)
        masks = masks[:num_samples]
        
        with torch.no_grad():
            outputs = self.model(images)
            
            # Handle V4 RBRM output
            if isinstance(outputs, tuple):
                # Check if the last element is edge_logits (1 channel)
                if outputs[-1].shape[1] == 1:
                    outputs = outputs[:-1] # Remove edge_logits
                
                # If still a tuple and length is 1, unpack it
                if isinstance(outputs, tuple) and len(outputs) == 1:
                    outputs = outputs[0]
            
            if self.config.loss.use_deep_supervision:
                outputs = outputs[-1] if isinstance(outputs, (list, tuple)) else outputs
            
            outputs = torch.nn.functional.interpolate(
                outputs, size=masks.shape[1:], mode='bilinear', align_corners=False
            )
            preds = outputs.argmax(dim=1).cpu()
        
        # Save each sample
        save_dir = os.path.join(self.output_dir, 'predictions', f'epoch_{self.current_epoch+1}')
        os.makedirs(save_dir, exist_ok=True)
        
        for i in range(len(images)):
            # Denormalize image
            import matplotlib.pyplot as plt
            
            img = images[i].cpu()
            img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img = img + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            
            # Prediction and ground truth
            pred = preds[i].numpy()
            gt = masks[i].cpu().numpy()
            
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.imshow(img)
            plt.title('Image')
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(gt, cmap='gray')
            plt.title('Ground Truth')
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(pred, cmap='gray')
            plt.title('Prediction')
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'sample_{i}.png'))
            plt.close()
            gt = masks[i].numpy()
            
            # Create visualization
            fig_path = os.path.join(
                self.output_dir, 'predictions', 
                f'epoch_{self.current_epoch+1}_sample_{i}.png'
            )
            self._save_visualization(img, gt, pred, fig_path)
    
    def _save_visualization(self, image: np.ndarray, gt: np.ndarray, pred: np.ndarray, path: str):
        """Save a single visualization."""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(image)
        axes[0].set_title('Input Image')
        axes[0].axis('off')
        
        axes[1].imshow(gt, cmap='gray')
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        axes[2].imshow(pred, cmap='gray')
        axes[2].set_title('Prediction')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(path, dpi=100, bbox_inches='tight')
        plt.close()
    
    def save_training_plots(self):
        """Save training history plots."""
        import matplotlib.pyplot as plt
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Loss
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='Train')
        axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', label='Val')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # mIoU
        axes[0, 1].plot(epochs, self.history['train_miou'], 'b-', label='Train')
        axes[0, 1].plot(epochs, self.history['val_miou'], 'r-', label='Val')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('mIoU')
        axes[0, 1].set_title('Mean IoU')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Dice
        axes[1, 0].plot(epochs, self.history['train_dice'], 'b-', label='Train')
        axes[1, 0].plot(epochs, self.history['val_dice'], 'r-', label='Val')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('mDice')
        axes[1, 0].set_title('Mean Dice Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate
        axes[1, 1].plot(epochs, self.history['lr'], 'g-')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_history.png'), dpi=150)
        plt.close()
        
        # Save history as JSON
        import json
        with open(os.path.join(self.output_dir, 'training_history.json'), 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def train(self):
        """Full training loop."""
        self.log(f"\nStarting training for {self.config.training.epochs} epochs...")
        self.log(f"{'='*60}\n")
        
        start_time = time.time()
        
        for epoch in range(self.config.training.epochs):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            # Update learning rate
            if isinstance(self.scheduler, WarmupCosineScheduler):
                lr = self.scheduler.step(epoch)
            elif self.scheduler is not None:
                self.scheduler.step()
                lr = self.scheduler.get_last_lr()[0]
            else:
                lr = self.config.optimizer.learning_rate
            
            # Train
            train_loss, train_metrics = self.train_epoch()
            
            # Validate
            val_loss, val_metrics = self.validate()
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_miou'].append(train_metrics['miou'])
            self.history['val_miou'].append(val_metrics['miou'])
            self.history['train_dice'].append(train_metrics['mdice'])
            self.history['val_dice'].append(val_metrics['mdice'])
            self.history['lr'].append(lr)
            
            # Check for best model
            is_best = val_metrics['miou'] > self.best_miou
            if is_best:
                self.best_miou = val_metrics['miou']
                self.best_epoch = epoch + 1
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1
            
            # Save checkpoint
            self.save_checkpoint(is_best=is_best)
            
            # Log to TensorBoard
            self.log_tensorboard(train_metrics, val_metrics, lr)
            
            # Save predictions periodically
            if (epoch + 1) % self.config.logging.save_predictions_every == 0:
                self.save_predictions()
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start
            self.log(f"\nEpoch {epoch+1}/{self.config.training.epochs} ({epoch_time:.1f}s)")
            self.log(f"  Train - Loss: {train_loss:.4f}, mIoU: {train_metrics['miou']:.4f}, "
                  f"mDice: {train_metrics['mdice']:.4f}")
            self.log(f"  Val   - Loss: {val_loss:.4f}, mIoU: {val_metrics['miou']:.4f}, "
                  f"mDice: {val_metrics['mdice']:.4f}")
            self.log(f"  LR: {lr:.2e}, Best mIoU: {self.best_miou:.4f} (epoch {self.best_epoch})")
            
            # Early stopping
            if self.config.training.early_stopping:
                if self.early_stop_counter >= self.config.training.patience:
                    self.log(f"\nEarly stopping triggered after {epoch+1} epochs")
                    break
        
        # Final saves
        self.save_training_plots()
        
        total_time = time.time() - start_time
        self.log(f"\n{'='*60}")
        self.log(f"Training completed in {total_time/3600:.2f} hours")
        self.log(f"Best mIoU: {self.best_miou:.4f} at epoch {self.best_epoch}")
        self.log(f"{'='*60}")
        
        if self.writer:
            self.writer.close()


# ============================================================================
# Main
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Ablation Study Training')
    
    parser.add_argument('--model', type=str, default='ablation_v1_base',
                       choices=['ablation_v1_base', 'ablation_v2_assplite', 
                               'ablation_v3_apud', 'ablation_v4_rbrm'],
                       help='Model variant to train')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config JSON file (overrides --model)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (overrides config)')
    parser.add_argument('--image-dir', type=str, default=None,
                       help='Image directory (overrides config)')
    parser.add_argument('--mask-dir', type=str, default=None,
                       help='Mask directory (overrides config)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (overrides config)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--no-amp', action='store_true',
                       help='Disable mixed precision training')
    
    # V4 Specific Arguments
    parser.add_argument('--freeze-backbone', action='store_true',
                       help='Freeze backbone for V4 training')
    parser.add_argument('--use-boundary-loss', action='store_true',
                       help='Use boundary-weighted loss for V4')
    parser.add_argument('--v3-checkpoint', type=str, default=None,
                       help='Path to V3 checkpoint for V4 initialization')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config
    if args.config:
        config = AblationConfig.load(args.config)
    else:
        config = get_config(args.model)
    
    # Override with command line args
    if args.epochs:
        config.training.epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.lr:
        config.optimizer.learning_rate = args.lr
    if args.image_dir:
        config.data.image_dir = args.image_dir
    if args.mask_dir:
        config.data.mask_dir = args.mask_dir
    if args.output_dir:
        config.logging.output_dir = args.output_dir
    if args.resume:
        config.resume = args.resume
    if args.no_amp:
        config.training.use_amp = False
        
    # V4 Specific Overrides
    if args.v3_checkpoint:
        config.model.v3_checkpoint = args.v3_checkpoint
    
    # Create trainer and train
    trainer = AblationTrainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
