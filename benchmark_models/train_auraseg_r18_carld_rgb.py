"""
AURASeg WACV ResNet-18 Training Entry Point
===========================================

Training script for new controlled WACV ablations.
Supports configurable fusion, attention, and RBRM components.
Includes deterministic seeding and automatic config saving.

Usage:
    python benchmark_models/train_auraseg_r18_wacv.py --fusion-type mul --attention-mode full --use-sobel --use-gate --seed 42
"""

import os
import sys
import argparse
import time
import json
import random
import shutil
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import cv2
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from auraseg_r18_wacv import AURASeg_R18_WACV
from unified_dataset import Normalization, UnifiedDrivableAreaDataset

from wacv_losses import WACVCombinedLoss
from wacv_metrics import compute_metrics, compute_boundary_metrics

def set_seed(seed: int):
    """Set deterministic seeds for all frameworks"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    """Seed data loader workers"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class Config:
    def __init__(self, args):
        self.DATA_ROOT = Path(__file__).parent.parent / "carl-dataset"
        self.NUM_CLASSES = 2
        self.IMG_SIZE = (384, 640)
        self.EPOCHS = 50
        if getattr(args, 'smoke_test', False):
            self.EPOCHS = 2
        self.MICRO_BATCH_SIZE = 4
        self.GRAD_ACCUM_STEPS = 2
        self.VAL_BATCH_SIZE = 4
        self.EFFECTIVE_BATCH_SIZE = self.MICRO_BATCH_SIZE * self.GRAD_ACCUM_STEPS
        self.LR_ENCODER = 1e-4
        self.LR_DECODER = 1e-3
        self.WEIGHT_DECAY = 0.01
        self.FOCAL_WEIGHT = 0.5
        self.DICE_WEIGHT = 0.5
        self.BOUNDARY_WEIGHT = 0.2
        self.AUX_WEIGHT = 0.1
        self.PATIENCE = 10
        self.MIN_DELTA = 0.0001
        self.NUM_WORKERS = 4
        self.PIN_MEMORY = True
        self.USE_AMP = True
        self.MEAN = [0.485, 0.456, 0.406]
        self.STD = [0.229, 0.224, 0.225]
        
        self.fusion_type = args.fusion_type
        self.attention_mode = args.attention_mode
        self.use_sobel = args.use_sobel
        self.use_gate = args.use_gate
        self.simple_boundary_head = getattr(args, 'simple_boundary_head', False)
        self.seed = args.seed
        self.smoke_test = getattr(args, 'smoke_test', False)
        self.eval_best_only = getattr(args, 'eval_best_only', False)
        self.resume_from = getattr(args, 'resume_from', None)

        
        # WACV Augmentation Parameters
        self.aug_params = {
            'shift_limit': 0.1,
            'scale_limit': 0.1,
            'rotate_limit': 15,
            'brightness_limit': 0.2,
            'contrast_limit': 0.2,
            'gauss_var_limit': (10.0, 50.0),
            'flip_p': 0.5,
            'color_p': 0.3,
            'noise_p': 0.2,
            'geom_p': 0.5
        }
        
        # Name formulation
        sobel_str = "sobel" if self.use_sobel else "nosobel"
        gate_str = "gate" if self.use_gate else "nogate"
        sbh_str = "simplebnd" if self.simple_boundary_head else ""
        self.run_name = f"r18_{self.fusion_type}_{self.attention_mode}_{sobel_str}_{gate_str}"
        if sbh_str:
            self.run_name += f"_{sbh_str}"
        self.run_name += f"_seed{self.seed}"
        
        repo_root = Path(__file__).parent.parent
        if hasattr(args, 'output_root') and args.output_root:
            self.output_root_path = Path(args.output_root)
        else:
            self.output_root_path = repo_root / "runs_carld_rgb"
            
        self.OUTPUT_DIR = self.output_root_path / self.run_name
        self.resume_from = getattr(args, 'resume_from', None)
        
    def to_dict(self):
        import subprocess
        try:
            git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.STDOUT).decode('utf-8').strip()
        except Exception:
            git_commit = "unknown"
            
        return {
            'backbone': 'resnet18',
            'fusion_type': self.fusion_type,
            'attention_mode': self.attention_mode,
            'use_sobel': self.use_sobel,
            'use_gate': self.use_gate,
            'simple_boundary_head': self.simple_boundary_head,
            'seed': self.seed,
            'dataset': str(self.DATA_ROOT),
            'input_resolution': self.IMG_SIZE,
            'optimizer': 'AdamW',
            'learning_rates': {'encoder': self.LR_ENCODER, 'decoder': self.LR_DECODER},
            'weight_decay': self.WEIGHT_DECAY,
            'AMP': self.USE_AMP,
            'loss_configuration': {
                'focal_alpha': 0.25,
                'focal_gamma': 2.0,
                'focal_weight': self.FOCAL_WEIGHT,
                'dice_weight': self.DICE_WEIGHT,
                'boundary_weight': self.BOUNDARY_WEIGHT,
                'boundary_criterion': 'BCEWithLogitsLoss',
                'boundary_target_method': 'morphological (dilate - erode)',
                'boundary_target_kernel': '3x3',
                'aux_weight_per_stage': self.AUX_WEIGHT,
                'auxiliary_formulation': 'L_focal + L_dice',
                'number_of_aux_stages': 4
            },
            'augmentation': self.aug_params,
            'normalization': {
                'mean': self.MEAN,
                'std': self.STD
            },
            'scheduler': 'CosineAnnealingLR',
            'scheduler_eta_min': 1e-6,
            'epochs': self.EPOCHS,
            'micro_batch_size': self.MICRO_BATCH_SIZE,
            'gradient_accumulation_steps': self.GRAD_ACCUM_STEPS,
            'effective_batch_size': self.EFFECTIVE_BATCH_SIZE,
            'val_batch_size': self.VAL_BATCH_SIZE,
            'num_workers': self.NUM_WORKERS,
            'pin_memory': self.PIN_MEMORY,
            'dataloader_drop_last': {'train': True, 'val': False},
            'early_stopping_metric': 'mIoU',
            'patience': self.PATIENCE,
            'min_delta': self.MIN_DELTA,
            'timestamp': datetime.now().isoformat(),
            'git_commit': git_commit
        }


class AURASegTrainer_R18:
    def __init__(self, config: Config, device: torch.device):
        self.config = config
        self.device = device
        
        self.model = AURASeg_R18_WACV(
            num_classes=config.NUM_CLASSES,
            encoder_weights='imagenet',
            fusion_type=config.fusion_type,
            attention_mode=config.attention_mode,
            use_sobel=config.use_sobel,
            use_gate=config.use_gate,
            simple_boundary_head=config.simple_boundary_head
        ).to(device)
        
        self.criterion = WACVCombinedLoss(
            focal_alpha=0.25,
            focal_gamma=2.0,
            dice_smooth=1.0
        )
        
        param_groups = self.model.get_param_groups(
            lr_encoder=config.LR_ENCODER,
            lr_decoder=config.LR_DECODER
        )
        self.optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=config.WEIGHT_DECAY
        )
        
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.EPOCHS,
            eta_min=1e-6
        )
        
        self.scaler = GradScaler(enabled=config.USE_AMP)
        self.best_miou = 0.0

        self.epochs_without_improvement = 0
        self.start_epoch = 1
        
        if config.resume_from:
            print(f"Resuming from {config.resume_from}...")
            checkpoint = torch.load(config.resume_from, map_location=device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if self.config.USE_AMP and checkpoint.get('scaler_state_dict'):
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_miou = checkpoint.get('best_miou', 0.0)
            self.epochs_without_improvement = checkpoint.get('epochs_without_improvement', 0)
            print(f"Resumed at epoch {self.start_epoch} (Best mIoU: {self.best_miou:.4f})")

    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        # 1. Check free space (require 500MB safety margin)
        usage = shutil.disk_usage(self.config.OUTPUT_DIR)
        min_required_bytes = 500 * 1024 * 1024
        if usage.free < min_required_bytes:
            raise RuntimeError(f"Insufficient disk space to safely write checkpoint. Required: 500MB, Free: {usage.free / (1024**2):.2f}MB")
            
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.config.USE_AMP else None,
            'best_miou': self.best_miou,
            'epochs_without_improvement': self.epochs_without_improvement,
            'metrics': metrics
        }
        
        latest_path = self.config.OUTPUT_DIR / "checkpoints" / "latest.pth"
        latest_tmp = latest_path.with_suffix(".pth.tmp")
        
        # 2. Atomic save for latest.pth
        try:
            torch.save(checkpoint, latest_tmp)
            os.replace(latest_tmp, latest_path)
        except Exception as e:
            if latest_tmp.exists():
                latest_tmp.unlink()
        # 2. Save best and last checkpoints
        if is_best:
            torch.save(checkpoint, self.config.OUTPUT_DIR / "checkpoints" / "best.pth")
            
        torch.save(checkpoint, self.config.OUTPUT_DIR / "checkpoints" / "last.pth")

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> dict:
        self.model.train()
        total_loss = total_seg = total_bnd = total_aux = 0.0
        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}")
        
        self.optimizer.zero_grad()
        for i, (images, masks) in enumerate(pbar):
            if self.config.smoke_test and i >= 4:
                break
            images, masks = images.to(self.device), masks.to(self.device)
            
            with autocast(enabled=self.config.USE_AMP):
                outputs = self.model(images, return_aux=True, return_boundary=True)
                losses = self.criterion(outputs, masks)
            
            scaled_loss = losses['total'] / self.config.GRAD_ACCUM_STEPS
            self.scaler.scale(scaled_loss).backward()
            
            if (i + 1) % self.config.GRAD_ACCUM_STEPS == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            
            # Log ORIGINAL (un-divided) loss values
            total_loss += losses['total'].item()
            total_seg += losses['seg'].item()
            total_bnd += losses['boundary'].item()
            total_aux += losses['aux'].item()
            
            pbar.set_postfix({'loss': f"{losses['total'].item():.4f}"})
            
        # For smoke tests or full loops, the scaler.step is handled inside the loop for accum steps.
            
        n = len(train_loader)
        return {'loss': total_loss / n, 'seg': total_seg / n, 'bnd': total_bnd / n, 'aux': total_aux / n}

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> dict:
        self.model.eval()
        all_preds, all_targets = [], []
        total_loss = 0.0
        
        for batch_idx, (images, masks) in enumerate(tqdm(val_loader, desc="Validating")):
            if self.config.smoke_test and batch_idx >= 3:
                break
            images, masks = images.to(self.device), masks.to(self.device)
            with autocast(enabled=self.config.USE_AMP):
                outputs = self.model(images, return_aux=True, return_boundary=True)
                losses = self.criterion(outputs, masks)
                
            total_loss += losses['total'].item()
            preds = torch.argmax(outputs['main'], dim=1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(masks.cpu().numpy())
            
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        seg_metrics = compute_metrics(all_preds, all_targets)
        boundary_metrics = compute_boundary_metrics(all_preds, all_targets)
        return {'loss': total_loss / len(val_loader), **seg_metrics, **boundary_metrics}

    def evaluate_test_set(self, test_loader: DataLoader, output_dir: Path):
        print(f"\n{'='*70}\nEVALUATING ON TEST SET\n{'='*70}")
        best_path = output_dir / "checkpoints" / "best.pth"
        if not best_path.exists():
            raise RuntimeError(f"Cannot find best.pth at {best_path}")
            
        checkpoint = torch.load(best_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best.pth from Epoch {checkpoint['epoch']}")
        
        test_metrics = self.validate(test_loader)
        
        print("\n[TEST RESULTS]")
        print(f"mIoU: {test_metrics['miou']:.4f}")
        print(f"IoU (Drivable): {test_metrics['iou_drivable']:.4f}")
        print(f"F1 (Drivable):  {test_metrics['f1']:.4f}")
        print(f"Boundary F1:    {test_metrics['boundary_f1']:.4f}")
        
        # Save metrics
        results = {
            'best_epoch': checkpoint['epoch'],
            'best_val_miou': checkpoint.get('best_miou', 0.0),
        }
        results.update(test_metrics)
        
        with open(output_dir / "test_results.json", "w") as f:
            json.dump(results, f, indent=4)
            
        import csv
        with open(output_dir / "test_results.csv", "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(results.keys())
            writer.writerow(results.values())
            
        print(f"Test results saved to {output_dir / 'test_results.json'} and test_results.csv")

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        print(f"\n{'='*70}\nTRAINING: {self.config.run_name}\n{'='*70}")
        
        for epoch in range(self.start_epoch, self.config.EPOCHS + 1):
            train_metrics = self.train_epoch(train_loader, epoch)
            val_metrics = self.validate(val_loader)
            
            print(f"[Epoch {epoch}] Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f} | mIoU: {val_metrics['miou']:.4f}")
            
            self.scheduler.step()
            
            is_best = val_metrics['miou'] > self.best_miou + self.config.MIN_DELTA
            if is_best:
                self.best_miou = val_metrics['miou']
                self.epochs_without_improvement = 0
                print(f"  *** New best mIoU: {self.best_miou:.4f} ***")
            else:
                self.epochs_without_improvement += 1
                
            self.save_checkpoint(epoch, val_metrics, is_best)
            
            if self.epochs_without_improvement >= self.config.PATIENCE:
                print(f"Early stopping triggered at epoch {epoch}")
                break

def main():
    parser = argparse.ArgumentParser(description='Train AURASeg WACV ResNet-18 Ablations')
    parser.add_argument('--fusion-type', type=str, choices=['mul', 'add', 'concat'], default='mul')
    parser.add_argument('--attention-mode', type=str, choices=['full', 'none', 'se', 'spatial'], default='full')
    
    parser.add_argument('--use-sobel', action='store_true', default=True, help='Enable fixed Sobel prior (Default: True)')
    parser.add_argument('--no-sobel', action='store_false', dest='use_sobel', help='Disable Sobel (learn edges from scratch)')
    
    parser.add_argument('--use-gate', action='store_true', default=True, help='Enable localized gate (Default: True)')
    parser.add_argument('--no-gate', action='store_false', dest='use_gate', help='Disable gate (direct residual)')
    
    parser.add_argument('--simple-boundary-head', action='store_true', default=False, help='Use simple boundary head instead of RBRM encoder-decoder')
    
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--output-root', type=str, default='runs_carld_rgb', help='Root directory for output (Default: runs_carld_rgb)')
        
    parser.add_argument('--smoke-test', action='store_true', help='Smoke test mode (few batches)')
    parser.add_argument('--eval-best-only', action='store_true', help='Evaluate best checkpoint on test set')
    parser.add_argument('--resume-from', type=str, default=None, help='Path to checkpoint to resume from')

    args = parser.parse_args()
    
    # 1. Set deterministic seed
    set_seed(args.seed)
    
    # 2. Config & Naming
    config = Config(args)
    
    
    if config.smoke_test:
        config.OUTPUT_DIR = Path(args.output_root if args.output_root else 'runs_carld_rgb') / f"{config.run_name}_smoke"
        
    config_path = config.OUTPUT_DIR / "config.json"
    if config.eval_best_only:
        if not config.OUTPUT_DIR.exists():
            raise RuntimeError(f"--eval-best-only requires an existing output directory: {config.OUTPUT_DIR}")

    elif config_path.exists() and not getattr(args, 'resume_from', None) and not config.smoke_test:
        raise RuntimeError(f"Experiment directory already exists. Refusing to overwrite:\\n{config.OUTPUT_DIR}")

        
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (config.OUTPUT_DIR / "checkpoints").mkdir(exist_ok=True)
    
    if not config.eval_best_only:
        with open(config_path, 'w') as f:
            json.dump(config.to_dict(), f, indent=4)
        
    print(f"[INFO] Config saved to {config_path}")
    print(f"[INFO] Run Name: {config.run_name}")
    print(f"[INFO] Seed: {args.seed}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Dataset Preparation
    normalization = Normalization(mean=tuple(config.MEAN), std=tuple(config.STD))
    
    g = torch.Generator()
    g.manual_seed(args.seed)
    
    train_dataset = UnifiedDrivableAreaDataset(
        dataset_root=config.DATA_ROOT, split='train',
        img_size=config.IMG_SIZE, transform=True,
        normalization=normalization, return_names=False,
        aug_params=config.aug_params
    )
    
    val_dataset = UnifiedDrivableAreaDataset(
        dataset_root=config.DATA_ROOT, split='val',
        img_size=config.IMG_SIZE, transform=False,
        normalization=normalization, return_names=False,
        aug_params=config.aug_params
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=config.MICRO_BATCH_SIZE,
        shuffle=True, num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY, drop_last=True,
        worker_init_fn=seed_worker, generator=g
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=config.VAL_BATCH_SIZE,
        shuffle=False, num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY, drop_last=False,
        worker_init_fn=seed_worker
    )
    
    
    test_dataset = UnifiedDrivableAreaDataset(
        dataset_root=config.DATA_ROOT, split='test',
        img_size=config.IMG_SIZE, transform=False,
        normalization=normalization, return_names=False,
        aug_params=config.aug_params
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.VAL_BATCH_SIZE,
        shuffle=False, num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY, drop_last=False,
        worker_init_fn=seed_worker
    )
    
    assert str(config.DATA_ROOT).endswith("carl-dataset"), "Dataset root must be carl-dataset"
    print(f"train={len(train_dataset)}")
    print(f"val={len(val_dataset)}")
    print(f"test={len(test_dataset)}")
    print(f"dataset root {config.DATA_ROOT}")
    
    trainer = AURASegTrainer_R18(config, device)
    
    if args.eval_best_only:
        trainer.evaluate_test_set(test_loader, config.OUTPUT_DIR)
        return
        

    trainer.train(train_loader, val_loader)
    trainer.evaluate_test_set(test_loader, config.OUTPUT_DIR)

if __name__ == "__main__":
    main()
