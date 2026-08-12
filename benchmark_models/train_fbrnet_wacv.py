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
from tqdm import tqdm
import subprocess

sys.path.insert(0, str(Path(__file__).parent))
from fbrnet_wacv_model import FBRNet_WACV
from unified_dataset import Normalization, UnifiedDrivableAreaDataset
from wacv_metrics import compute_metrics, compute_boundary_metrics

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class Config:
    def __init__(self, args):
        repo_root = Path(__file__).parent.parent
        
        self.dataset_type = args.dataset
        if self.dataset_type == 'mix':
            self.DATA_ROOT = repo_root / "CommonDataset"
        elif self.dataset_type == 'carl-d':
            self.DATA_ROOT = repo_root / "carl-dataset"
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_type}")
            
        self.NUM_CLASSES = 2
        self.IMG_SIZE = (384, 640)
        self.EPOCHS = 50
        self.MICRO_BATCH_SIZE = 4
        self.GRAD_ACCUM_STEPS = 2
        self.VAL_BATCH_SIZE = 4
        self.EFFECTIVE_BATCH_SIZE = self.MICRO_BATCH_SIZE * self.GRAD_ACCUM_STEPS
        self.LR_ENCODER = 1e-4
        self.LR_DECODER = 1e-3
        self.WEIGHT_DECAY = 0.01
        self.PATIENCE = 10
        self.MIN_DELTA = 0.0001
        self.NUM_WORKERS = 4
        self.PIN_MEMORY = True
        self.USE_AMP = True
        self.MEAN = [0.485, 0.456, 0.406]
        self.STD = [0.229, 0.224, 0.225]
        
        self.seed = args.seed
        self.smoke_test = args.smoke_test
        
        if self.smoke_test:
            self.run_name = "_smoke"
        else:
            dataset_str = "mix" if self.dataset_type == 'mix' else "carld"
            self.run_name = f"fbrnet_{dataset_str}_seed{self.seed}"
        
        if args.output_root:
            self.output_root_path = Path(args.output_root)
        else:
            self.output_root_path = repo_root / "runs_fbrnet_wacv"
            
        self.OUTPUT_DIR = self.output_root_path / self.run_name
        self.resume_from = args.resume_from
        
    def to_dict(self):
        try:
            git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.STDOUT).decode('utf-8').strip()
        except Exception:
            git_commit = "unknown"
            
        return {
            'model': 'FBRNet',
            'official_repo': 'https://github.com/little5570/FBRNet',
            'official_commit': '04f2bf7209d78035019edc8c25bab0d02bd0439f',
            'adaptation': 'Binary-class generalization (num_classes=2) and device portability (Laplacian buffer). Architecture preserved.',
            'dataset': str(self.DATA_ROOT),
            'dataset_type': self.dataset_type,
            'input_resolution': self.IMG_SIZE,
            'optimizer': 'AdamW',
            'learning_rates': {'encoder': self.LR_ENCODER, 'decoder': self.LR_DECODER},
            'weight_decay': self.WEIGHT_DECAY,
            'AMP': self.USE_AMP,
            'loss': 'Native FBRNet CrossEntropy (weighted: main=1, aux16=2, aux0=5, aux1=3, aux32=0)',
            'normalization': {'mean': self.MEAN, 'std': self.STD},
            'scheduler': 'CosineAnnealingLR',
            'scheduler_eta_min': 1e-6,
            'epochs': self.EPOCHS,
            'micro_batch_size': self.MICRO_BATCH_SIZE,
            'gradient_accumulation_steps': self.GRAD_ACCUM_STEPS,
            'effective_batch_size': self.EFFECTIVE_BATCH_SIZE,
            'val_batch_size': self.VAL_BATCH_SIZE,
            'early_stopping_metric': 'mIoU',
            'patience': self.PATIENCE,
            'seed': self.seed,
            'timestamp': datetime.now().isoformat(),
            'git_commit': git_commit,
            'environment': {
                'python': sys.version.split()[0],
                'torch': torch.__version__,
                'cuda': torch.version.cuda,
                'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'
            }
        }


class FBRNetTrainer:
    def __init__(self, config: Config, device: torch.device):
        self.config = config
        self.device = device
        
        self.model = FBRNet_WACV(num_classes=config.NUM_CLASSES, aux_mode='train').to(device)
        
        encoder_params = []
        decoder_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'model.cp.resnet' in name:
                encoder_params.append(param)
            else:
                decoder_params.append(param)
                
        self.optimizer = torch.optim.AdamW(
            [
                {'params': encoder_params, 'lr': config.LR_ENCODER},
                {'params': decoder_params, 'lr': config.LR_DECODER}
            ],
            weight_decay=config.WEIGHT_DECAY
        )
        
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.EPOCHS, eta_min=1e-6)
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
        usage = shutil.disk_usage(self.config.OUTPUT_DIR)
        min_required_bytes = 500 * 1024 * 1024
        if usage.free < min_required_bytes:
            raise RuntimeError(f"Insufficient disk space to safely write checkpoint. Free: {usage.free / (1024**2):.2f}MB")
            
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
        
        latest_path = self.config.OUTPUT_DIR / "checkpoints" / "last.pth"
        latest_tmp = latest_path.with_suffix(".pth.tmp")
        
        try:
            torch.save(checkpoint, latest_tmp)
            os.replace(latest_tmp, latest_path)
        except Exception as e:
            if latest_tmp.exists(): latest_tmp.unlink()
            raise RuntimeError(f"Failed to save last checkpoint: {e}")
            
        if is_best:
            best_path = self.config.OUTPUT_DIR / "checkpoints" / "best.pth"
            best_tmp = best_path.with_suffix(".pth.tmp")
            try:
                shutil.copy2(latest_path, best_tmp)
                os.replace(best_tmp, best_path)
            except Exception as e:
                if best_tmp.exists(): best_tmp.unlink()
                raise RuntimeError(f"Failed to copy best checkpoint: {e}")

    def compute_loss(self, outputs, targets):
        # outputs: feat_ffm, feat_out32, feat_out16, aux_0, aux_1
        feat_ffm, feat_out32, feat_out16, aux_0, aux_1 = outputs
        
        targets = targets.long()
        loss_main = F.cross_entropy(feat_ffm, targets, ignore_index=255)
        # loss_aux[0] corresponds to feat_out32 -> weighted 0
        loss_aux16 = F.cross_entropy(feat_out16, targets, ignore_index=255) # loss_aux[1] -> weight 2
        loss_aux0 = F.cross_entropy(aux_0, targets, ignore_index=255)       # loss_aux[2] -> weight 5 (2 + 3)
        loss_aux1 = F.cross_entropy(aux_1, targets, ignore_index=255)       # loss_aux[3] -> weight 3
        
        total_loss = loss_main + 2 * loss_aux16 + 5 * loss_aux0 + 3 * loss_aux1
        
        return {'total': total_loss, 'main': loss_main, 'aux16': loss_aux16, 'aux0': loss_aux0, 'aux1': loss_aux1}

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> dict:
        self.model.train()
        self.model.aux_mode = 'train'
        self.model.model.aux_mode = 'train'
        
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}")
        
        self.optimizer.zero_grad()
        for i, (images, masks) in enumerate(pbar):
            images, masks = images.to(self.device), masks.to(self.device)
            
            with autocast(enabled=self.config.USE_AMP):
                outputs = self.model(images)
                losses = self.compute_loss(outputs, masks)
            
            scaled_loss = losses['total'] / self.config.GRAD_ACCUM_STEPS
            self.scaler.scale(scaled_loss).backward()
            
            if (i + 1) % self.config.GRAD_ACCUM_STEPS == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            
            total_loss += losses['total'].item()
            pbar.set_postfix({'loss': f"{losses['total'].item():.4f}"})
            
            if self.config.smoke_test and i >= 3:
                break
            
        if len(train_loader) % self.config.GRAD_ACCUM_STEPS != 0 and not self.config.smoke_test:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            
        n = min(len(train_loader), 4) if self.config.smoke_test else len(train_loader)
        return {'loss': total_loss / n}

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> dict:
        self.model.eval()
        self.model.aux_mode = 'eval'
        self.model.model.aux_mode = 'eval'
        
        all_preds, all_targets = [], []
        
        for i, (images, masks) in enumerate(tqdm(val_loader, desc="Validating")):
            images, masks = images.to(self.device), masks.to(self.device)
            with autocast(enabled=self.config.USE_AMP):
                feat_ffm = self.model(images)[0]
                
            preds = torch.argmax(feat_ffm, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(masks.cpu().numpy())
            
            if self.config.smoke_test and i >= 1:
                break
            
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        seg_metrics = compute_metrics(all_preds, all_targets)
        boundary_metrics = compute_boundary_metrics(all_preds, all_targets, k=2)
        return {**seg_metrics, **boundary_metrics}

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        print(f"\n{'='*70}\nTRAINING: {self.config.run_name}\n{'='*70}")
        
        for epoch in range(self.start_epoch, self.config.EPOCHS + 1):
            train_metrics = self.train_epoch(train_loader, epoch)
            val_metrics = self.validate(val_loader)
            
            print(f"[Epoch {epoch}] Train Loss: {train_metrics['loss']:.4f} | mIoU: {val_metrics['miou']:.4f}")
            
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
                
            if self.config.smoke_test:
                break

def main():
    parser = argparse.ArgumentParser(description='Train FBRNet WACV Baseline')
    parser.add_argument('--dataset', type=str, choices=['mix', 'carl-d'], required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-root', type=str, default='')
    parser.add_argument('--resume-from', type=str, default=None)
    parser.add_argument('--smoke-test', action='store_true', default=False)
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    config = Config(args)
    
    if not args.resume_from and config.OUTPUT_DIR.exists():
        if not args.smoke_test:
            raise RuntimeError(f"Experiment directory already exists. Refusing to overwrite: {config.OUTPUT_DIR}")
        else:
            shutil.rmtree(config.OUTPUT_DIR)
            
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (config.OUTPUT_DIR / "checkpoints").mkdir(exist_ok=True)
    
    with open(config.OUTPUT_DIR / "config.json", 'w') as f:
        json.dump(config.to_dict(), f, indent=4)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    normalization = Normalization(mean=tuple(config.MEAN), std=tuple(config.STD))
    
    # WACV Augmentation Parameters (used identically in AURASeg trainer)
    aug_params = {
        'shift_limit': 0.1, 'scale_limit': 0.1, 'rotate_limit': 15,
        'brightness_limit': 0.2, 'contrast_limit': 0.2, 'gauss_var_limit': (10.0, 50.0),
        'flip_p': 0.5, 'color_p': 0.3, 'noise_p': 0.2, 'geom_p': 0.5
    }
    
    g = torch.Generator()
    g.manual_seed(args.seed)
    
    train_dataset = UnifiedDrivableAreaDataset(
        dataset_root=config.DATA_ROOT, split='train',
        img_size=config.IMG_SIZE, transform=True,
        normalization=normalization, return_names=False,
        aug_params=aug_params
    )
    
    val_dataset = UnifiedDrivableAreaDataset(
        dataset_root=config.DATA_ROOT, split='val',
        img_size=config.IMG_SIZE, transform=False,
        normalization=normalization, return_names=False,
        aug_params=aug_params
    )
    
    print(f"MIX Train Samples: {len(train_dataset)}")
    print(f"MIX Val Samples: {len(val_dataset)}")
    
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
    
    trainer = FBRNetTrainer(config, device)
    trainer.train(train_loader, val_loader)

if __name__ == "__main__":
    main()
