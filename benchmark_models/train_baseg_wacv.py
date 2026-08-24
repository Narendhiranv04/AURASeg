import os
import sys
import argparse
import time
import json
import csv
import random
import shutil
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm
import subprocess

sys.path.insert(0, str(Path(__file__).parent))
from baseg_wacv_model import BASeg_WACV
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


def mask_to_onehot(mask: np.ndarray, num_classes: int = 2) -> np.ndarray:
    """Official BASeg mask to one-hot conversion."""
    _mask = [mask == i for i in range(num_classes)]
    return np.array(_mask).astype(np.uint8)


def onehot_to_binary_edges(mask: np.ndarray, radius: int = 2, num_classes: int = 2) -> np.ndarray:
    """
    Official BASeg boundary ground-truth generation utility.
    Computes Euclidean distance transform from boundaries with radius tolerance.
    """
    if radius < 0:
        return mask

    mask_pad = np.pad(mask, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)
    edgemap = np.zeros(mask.shape[1:], dtype=np.float32)

    for i in range(num_classes):
        dist = distance_transform_edt(mask_pad[i, :]) + distance_transform_edt(1.0 - mask_pad[i, :])
        dist = dist[1:-1, 1:-1]
        dist[dist > radius] = 0
        edgemap += dist

    edgemap = np.expand_dims(edgemap, axis=0)
    edgemap = (edgemap > 0).astype(np.uint8)
    return edgemap


class BASegDatasetWrapper(Dataset):
    """
    Wraps UnifiedDrivableAreaDataset to generate official BASeg boundary targets
    on-the-fly deterministically after semantic mask decoding.
    """
    def __init__(self, base_dataset: UnifiedDrivableAreaDataset, radius: int = 2, num_classes: int = 2):
        self.base_dataset = base_dataset
        self.radius = radius
        self.num_classes = num_classes

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        image, mask = item[0], item[1]

        # mask is torch.Tensor of shape [H, W] with values {0, 1}
        mask_np = mask.numpy().astype(np.uint8)
        onehot = mask_to_onehot(mask_np, self.num_classes)
        edge_np = onehot_to_binary_edges(onehot, self.radius, self.num_classes) # [1, H, W]
        edge = torch.from_numpy(edge_np).float()

        return image, mask, edge


class ImageBasedCrossEntropyLoss2d(nn.Module):
    """
    Official BASeg frequency-weighted CrossEntropy loss per image.
    """
    def __init__(self, classes: int = 2, ignore_index: int = 255, upper_bound: float = 1.0):
        super().__init__()
        self.num_classes = classes
        self.ignore_index = ignore_index
        self.upper_bound = upper_bound
        self.nll_loss = nn.NLLLoss(reduction="mean", ignore_index=ignore_index)

    def calculate_weights(self, target_cpu: np.ndarray) -> np.ndarray:
        hist = np.histogram(target_cpu.flatten(), range(self.num_classes + 1), density=True)[0]
        weights = ((hist != 0) * self.upper_bound * (1 - hist)) + 1
        return weights.astype(np.float32)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        target_cpu = targets.data.cpu().numpy()
        loss = 0.0
        for i in range(inputs.shape[0]):
            weights = self.calculate_weights(target_cpu[i])
            weight_tensor = torch.from_numpy(weights).to(inputs.device)
            loss += F.nll_loss(
                F.log_softmax(inputs[i].unsqueeze(0), dim=1),
                targets[i].unsqueeze(0).long(),
                weight=weight_tensor,
                ignore_index=self.ignore_index
            )
        return loss / max(inputs.shape[0], 1)


def bce2d(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Official BASeg balanced binary cross entropy loss with inverse frequency class balancing.
    """
    n, c, h, w = input.size()
    log_p = input.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
    target_t = target.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)

    pos_index = (target_t == 1)
    neg_index = (target_t == 0)
    ignore_index = (target_t > 1)

    pos_np = pos_index.data.cpu().numpy().astype(bool)
    neg_np = neg_index.data.cpu().numpy().astype(bool)
    ignore_np = ignore_index.data.cpu().numpy().astype(bool)

    weight = np.zeros(log_p.size(), dtype=np.float32)
    pos_num = pos_np.sum()
    neg_num = neg_np.sum()
    sum_num = pos_num + neg_num

    if sum_num > 0:
        weight[pos_np] = neg_num * 1.0 / sum_num
        weight[neg_np] = pos_num * 1.0 / sum_num
    weight[ignore_np] = 0

    weight_tensor = torch.from_numpy(weight).to(input.device)
    loss = F.binary_cross_entropy_with_logits(log_p, target_t, weight=weight_tensor, reduction='mean')
    return loss


class JointEdgeSegLoss(nn.Module):
    """
    Official BASeg multi-task loss module:
    L_total = seg_weight * L_seg + aux_weight * L_aux + edge_weight * 20 * L_edge
    """
    def __init__(self, classes: int = 2, ignore_index: int = 255, aux_weight: float = 0.4,
                 edge_weight: float = 8.0, seg_weight: float = 1.0):
        super().__init__()
        self.num_classes = classes
        self.ignore_index = ignore_index
        self.seg_loss = ImageBasedCrossEntropyLoss2d(classes=classes, ignore_index=ignore_index, upper_bound=1.0)
        self.aux_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.seg_weight = seg_weight
        self.aux_weight = aux_weight
        self.edge_weight = edge_weight

    def forward(self, inputs, targets):
        # inputs: (seg_in, aux_seg_in, edge_in)
        # targets: (seg_mask, edge_mask)
        seg_in, aux_seg_in, edge_in = inputs
        seg_mask, edge_mask = targets

        l_seg = self.seg_loss(seg_in, seg_mask)
        l_aux = self.aux_loss(aux_seg_in, seg_mask.long())
        l_edge = bce2d(edge_in, edge_mask)

        total_loss = self.seg_weight * l_seg + self.aux_weight * l_aux + self.edge_weight * 20.0 * l_edge

        return {
            'total': total_loss,
            'seg': l_seg,
            'aux': l_aux,
            'edge': l_edge
        }


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

        self.MODEL_NAME = "BASeg-R101"
        self.NUM_CLASSES = 2
        self.IMG_SIZE = (384, 640)
        self.EPOCHS = 50
        self.MICRO_BATCH_SIZE = 2
        self.GRAD_ACCUM_STEPS = 4
        self.VAL_BATCH_SIZE = 2
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

        # Official loss weights
        self.SEG_WEIGHT = 1.0
        self.AUX_WEIGHT = 0.4
        self.EDGE_WEIGHT = 8.0
        self.EDGE_SCALING = 20.0
        self.TOTAL_EDGE_COEFF = self.EDGE_WEIGHT * self.EDGE_SCALING # 160.0

        self.seed = args.seed
        self.smoke_test = args.smoke_test
        self.eval_best_only = getattr(args, 'eval_best_only', False)

        if self.smoke_test:
            dataset_str = "mix" if self.dataset_type == 'mix' else "carld"
            self.run_name = f"baseg_{dataset_str}_seed{self.seed}_smoke"
        else:
            dataset_str = "mix" if self.dataset_type == 'mix' else "carld"
            self.run_name = f"baseg_{dataset_str}_seed{self.seed}"

        if args.output_root:
            self.output_root_path = Path(args.output_root)
        else:
            self.output_root_path = repo_root / "runs_baseg_wacv"

        self.OUTPUT_DIR = self.output_root_path / self.run_name
        self.resume_from = args.resume_from
        self.param_count = 0
        self.trainable_param_count = 0

    def to_dict(self):
        try:
            git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.STDOUT).decode('utf-8').strip()
        except Exception:
            git_commit = "unknown"

        return {
            'model': self.MODEL_NAME,
            'official_repo': 'https://github.com/YangParky/BASeg',
            'official_commit': 'e88e958fa5f44a26995ec1dd9949291c89449d8d',
            'official_paper': 'BASeg: Boundary Aware Semantic Segmentation for Autonomous Driving, Neural Networks, 2023',
            'official_backbone': 'Dilated ResNet-101 (OS=8, multi-grid [1, 1, 1])',
            'total_parameters': self.param_count,
            'trainable_parameters': self.trainable_param_count,
            'adaptation': 'Binary drivable segmentation (num_classes=2). Preserved native ASPP, BRM, CAM, AGB, and Canny stream.',
            'dataset': str(self.DATA_ROOT),
            'dataset_type': self.dataset_type,
            'input_resolution': self.IMG_SIZE,
            'optimizer': 'AdamW',
            'learning_rates': {'encoder': self.LR_ENCODER, 'decoder': self.LR_DECODER},
            'weight_decay': self.WEIGHT_DECAY,
            'AMP': self.USE_AMP,
            'loss_formulation': {
                'semantic_loss': 'ImageBasedCrossEntropyLoss2d (weight=1.0)',
                'auxiliary_loss': 'nn.CrossEntropyLoss on stage 3 (weight=0.4)',
                'boundary_loss': 'bce2d with inverse pos/neg balancing (weight=8.0 * 20 = 160.0)',
                'attention_loss': 'disabled (att_weight=0.0)'
            },
            'boundary_target_generation': 'Exact official onehot_to_binary_edges with Euclidean distance transform (radius=2)',
            'canny_prior_handling': 'OpenCV Canny (thresholds 10, 100) on restored RGB input',
            'normalization': {'mean': self.MEAN, 'std': self.STD},
            'scheduler': 'CosineAnnealingLR',
            'scheduler_eta_min': 1e-6,
            'epochs': self.EPOCHS,
            'min_delta': self.MIN_DELTA,
            'num_workers': self.NUM_WORKERS,
            'train_drop_last': True,
            'val_drop_last': False,
            'boundary_tolerance_k': 2,
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
                'torchvision': __import__('torchvision').__version__,
                'cuda': torch.version.cuda if torch.cuda.is_available() else 'None',
                'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'
            }
        }


class BASegTrainer:
    def __init__(self, config: Config, device: torch.device):
        self.config = config
        self.device = device

        self.model = BASeg_WACV(num_classes=config.NUM_CLASSES, layers=101, pretrained=True).to(device)

        total_p = sum(p.numel() for p in self.model.parameters())
        trainable_p = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.config.param_count = total_p
        self.config.trainable_param_count = trainable_p

        print(f"Model: {config.MODEL_NAME}")
        print(f"Total Parameters: {total_p:,} ({total_p/1e6:.2f}M)")
        print(f"Trainable Parameters: {trainable_p:,} ({trainable_p/1e6:.2f}M)")

        param_groups = self.model.get_param_groups(
            lr_encoder=config.LR_ENCODER,
            lr_decoder=config.LR_DECODER,
            weight_decay=config.WEIGHT_DECAY
        )

        self.optimizer = torch.optim.AdamW(param_groups)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.EPOCHS, eta_min=1e-6)
        self.scaler = GradScaler(enabled=config.USE_AMP)

        self.criterion = JointEdgeSegLoss(
            classes=config.NUM_CLASSES,
            ignore_index=255,
            aux_weight=config.AUX_WEIGHT,
            edge_weight=config.EDGE_WEIGHT,
            seg_weight=config.SEG_WEIGHT
        ).to(device)

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

        self.csv_path = config.OUTPUT_DIR / "training_log.csv"
        if not self.csv_path.exists():
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'epoch', 'train_loss', 'val_miou', 'val_iou_drivable',
                    'val_f1', 'val_boundary_iou', 'val_boundary_f1',
                    'encoder_lr', 'decoder_lr', 'is_best'
                ])

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
            if latest_tmp.exists():
                latest_tmp.unlink()
            raise RuntimeError(f"Failed to save last checkpoint: {e}")

        if is_best:
            best_path = self.config.OUTPUT_DIR / "checkpoints" / "best.pth"
            best_tmp = best_path.with_suffix(".pth.tmp")
            try:
                shutil.copy2(latest_path, best_tmp)
                os.replace(best_tmp, best_path)
            except Exception as e:
                if best_tmp.exists():
                    best_tmp.unlink()
                raise RuntimeError(f"Failed to copy best checkpoint: {e}")

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> dict:
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}")

        self.optimizer.zero_grad()
        for i, (images, masks, edges) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)
            edges = edges.to(self.device)

            with autocast(enabled=self.config.USE_AMP):
                outputs = self.model(images)
                loss_dict = self.criterion(outputs, (masks, edges))
                loss = loss_dict['total']

            scaled_loss = loss / self.config.GRAD_ACCUM_STEPS
            self.scaler.scale(scaled_loss).backward()

            if (i + 1) % self.config.GRAD_ACCUM_STEPS == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'seg': f"{loss_dict['seg'].item():.4f}", 'edge': f"{loss_dict['edge'].item():.4f}"})

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
        all_preds, all_targets = [], []

        for i, batch in enumerate(tqdm(val_loader, desc="Validating")):
            images, masks = batch[0].to(self.device), batch[1].to(self.device)

            with autocast(enabled=self.config.USE_AMP):
                out = self.model(images)
                # In eval mode, out is (main_logits, edge_logits) or main_logits
                main_logits = out[0] if isinstance(out, (tuple, list)) else out

            preds = torch.argmax(main_logits, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(masks.cpu().numpy())

            if self.config.smoke_test and i >= 1:
                break

        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        seg_metrics = compute_metrics(all_preds, all_targets)
        boundary_metrics = compute_boundary_metrics(all_preds, all_targets, k=2)
        return {**seg_metrics, **boundary_metrics}

    def evaluate_test_set(self, test_loader: DataLoader):
        print(f"\n{'='*70}\nEVALUATING ON TEST SET\n{'='*70}")
        best_path = self.config.OUTPUT_DIR / "checkpoints" / "best.pth"
        if not best_path.exists():
            raise RuntimeError(f"Cannot find best.pth at {best_path}")

        checkpoint = torch.load(best_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best.pth from Epoch {checkpoint['epoch']} (Val mIoU: {checkpoint.get('best_miou', 0.0):.4f})")

        test_metrics = self.validate(test_loader)

        results_json = {
            'model': self.config.MODEL_NAME,
            'dataset': str(self.config.DATA_ROOT),
            'dataset_type': self.config.dataset_type,
            'checkpoint_epoch': checkpoint['epoch'],
            'val_best_miou': checkpoint.get('best_miou', 0.0),
            'metrics': {k: float(v) for k, v in test_metrics.items()}
        }

        json_path = self.config.OUTPUT_DIR / "test_results.json"
        with open(json_path, 'w') as f:
            json.dump(results_json, f, indent=4)

        csv_path = self.config.OUTPUT_DIR / "test_results.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['metric', 'value'])
            for k, v in test_metrics.items():
                writer.writerow([k, v])

        print("\n[TEST SET METRICS]")
        for k, v in test_metrics.items():
            print(f"{k:24s}: {v:.6f}")
        print(f"Results saved to {json_path} and {csv_path}")
        return test_metrics

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        print(f"\n{'='*70}\nTRAINING: {self.config.run_name}\n{'='*70}")

        for epoch in range(self.start_epoch, self.config.EPOCHS + 1):
            train_metrics = self.train_epoch(train_loader, epoch)
            val_metrics = self.validate(val_loader)

            print(f"[Epoch {epoch}] Train Loss: {train_metrics['loss']:.4f} | Val mIoU: {val_metrics['miou']:.4f}")

            self.scheduler.step()

            is_best = val_metrics['miou'] > self.best_miou + self.config.MIN_DELTA
            if is_best:
                self.best_miou = val_metrics['miou']
                self.epochs_without_improvement = 0
                print(f"  *** New best mIoU: {self.best_miou:.4f} ***")
            else:
                self.epochs_without_improvement += 1

            self.save_checkpoint(epoch, val_metrics, is_best)

            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch,
                    f"{train_metrics['loss']:.6f}",
                    f"{val_metrics['miou']:.6f}",
                    f"{val_metrics['iou_drivable']:.6f}",
                    f"{val_metrics['f1']:.6f}",
                    f"{val_metrics.get('boundary_iou', 0.0):.6f}",
                    f"{val_metrics.get('boundary_f1', 0.0):.6f}",
                    f"{self.optimizer.param_groups[0]['lr']:.2e}",
                    f"{self.optimizer.param_groups[1]['lr']:.2e}",
                    is_best
                ])

            if self.epochs_without_improvement >= self.config.PATIENCE:
                print(f"Early stopping triggered at epoch {epoch}")
                break

            if self.config.smoke_test:
                break

        # Save final checkpoint
        final_path = self.config.OUTPUT_DIR / "checkpoints" / "final.pth"
        if (self.config.OUTPUT_DIR / "checkpoints" / "last.pth").exists():
            shutil.copy2(self.config.OUTPUT_DIR / "checkpoints" / "last.pth", final_path)


def main():
    parser = argparse.ArgumentParser(description='Train BASeg WACV Baseline')
    parser.add_argument('--dataset', type=str, choices=['mix', 'carl-d'], required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-root', type=str, default='')
    parser.add_argument('--resume-from', type=str, default=None)
    parser.add_argument('--smoke-test', action='store_true', default=False)
    parser.add_argument('--eval-best-only', action='store_true', default=False)
    args = parser.parse_args()

    set_seed(args.seed)

    config = Config(args)

    if args.eval_best_only:
        if not config.OUTPUT_DIR.exists():
            raise RuntimeError(f"Experiment directory does not exist for eval-best-only: {config.OUTPUT_DIR}")
        best_path = config.OUTPUT_DIR / "checkpoints" / "best.pth"
        if not best_path.exists():
            raise RuntimeError(f"Cannot find best.pth for eval-best-only: {best_path}")
    else:
        if not args.resume_from and config.OUTPUT_DIR.exists():
            if not args.smoke_test:
                raise RuntimeError(f"Experiment directory already exists. Refusing to overwrite: {config.OUTPUT_DIR}")
            else:
                shutil.rmtree(config.OUTPUT_DIR)

        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        (config.OUTPUT_DIR / "checkpoints").mkdir(exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    normalization = Normalization(mean=tuple(config.MEAN), std=tuple(config.STD))

    aug_params = {
        'shift_limit': 0.1, 'scale_limit': 0.1, 'rotate_limit': 15,
        'brightness_limit': 0.2, 'contrast_limit': 0.2, 'gauss_var_limit': (10.0, 50.0),
        'flip_p': 0.5, 'color_p': 0.3, 'noise_p': 0.2, 'geom_p': 0.5
    }

    g = torch.Generator()
    g.manual_seed(args.seed)

    if not args.eval_best_only:
        train_base_dataset = UnifiedDrivableAreaDataset(
            dataset_root=config.DATA_ROOT, split='train',
            img_size=config.IMG_SIZE, transform=True,
            normalization=normalization, return_names=False,
            aug_params=aug_params
        )
        train_dataset = BASegDatasetWrapper(train_base_dataset, radius=2, num_classes=config.NUM_CLASSES)
        print(f"Train Samples: {len(train_dataset)}")
        train_loader = DataLoader(
            train_dataset, batch_size=config.MICRO_BATCH_SIZE,
            shuffle=True, num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY, drop_last=True,
            worker_init_fn=seed_worker, generator=g
        )
    else:
        train_dataset = None
        train_loader = None

    val_base_dataset = UnifiedDrivableAreaDataset(
        dataset_root=config.DATA_ROOT, split='val',
        img_size=config.IMG_SIZE, transform=False,
        normalization=normalization, return_names=False,
        aug_params=aug_params
    )
    val_dataset = BASegDatasetWrapper(val_base_dataset, radius=2, num_classes=config.NUM_CLASSES)
    print(f"Val Samples: {len(val_dataset)}")

    val_loader = DataLoader(
        val_dataset, batch_size=config.VAL_BATCH_SIZE,
        shuffle=False, num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY, drop_last=False,
        worker_init_fn=seed_worker
    )

    test_loader = None
    if config.dataset_type == 'carl-d':
        test_base_dataset = UnifiedDrivableAreaDataset(
            dataset_root=config.DATA_ROOT, split='test',
            img_size=config.IMG_SIZE, transform=False,
            normalization=normalization, return_names=False,
            aug_params=aug_params
        )
        test_dataset = BASegDatasetWrapper(test_base_dataset, radius=2, num_classes=config.NUM_CLASSES)
        print(f"Test Samples: {len(test_dataset)}")

        test_loader = DataLoader(
            test_dataset, batch_size=config.VAL_BATCH_SIZE,
            shuffle=False, num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY, drop_last=False,
            worker_init_fn=seed_worker
        )
    else:
        # On MIX (CommonDataset), val split serves as the benchmark evaluation split
        test_loader = val_loader

    # Assert exact split counts for CARL-D
    if config.dataset_type == 'carl-d':
        if train_dataset is not None:
            assert len(train_dataset) == 8372, f"CARL-D train split count mismatch: expected 8372, got {len(train_dataset)}"
        assert len(val_dataset) == 1046, f"CARL-D val split count mismatch: expected 1046, got {len(val_dataset)}"
        assert len(test_dataset) == 1046, f"CARL-D test split count mismatch: expected 1046, got {len(test_dataset)}"
        print("[CARL-D ASSERTION PASS] train=8372 (if loaded), val=1046, test=1046 verified.")

    trainer = BASegTrainer(config, device)

    # Write config.json with populated parameter counts
    if not args.eval_best_only:
        with open(config.OUTPUT_DIR / "config.json", 'w') as f:
            json.dump(config.to_dict(), f, indent=4)

    # Print summary configuration
    print(f"\nConfiguration:")
    print(f"  Dataset: {config.DATA_ROOT} ({config.dataset_type})")
    print(f"  Micro batch size: {config.MICRO_BATCH_SIZE}")
    print(f"  Gradient accumulation steps: {config.GRAD_ACCUM_STEPS}")
    print(f"  Effective batch size: {config.EFFECTIVE_BATCH_SIZE}")
    print(f"  Encoder LR: {config.LR_ENCODER:.2e}")
    print(f"  Decoder LR: {config.LR_DECODER:.2e}")
    print(f"  Loss: 1.0 * ImageBasedCE + 0.4 * AuxCE + 160.0 * BalancedBCE(edge)")

    if args.eval_best_only:
        trainer.evaluate_test_set(test_loader)
        return

    trainer.train(train_loader, val_loader)
    trainer.evaluate_test_set(test_loader)


if __name__ == "__main__":
    main()
