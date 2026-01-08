"""Comprehensive Evaluation Script for All Benchmark Models
=========================================================

Computes all metrics for Table IV format:
- IoU (drivable area only)
- F1-Score
- Boundary IoU (BIoU)
- Boundary F1-Score (BF1)
- Precision
- Recall

Evaluates on both MIX (Gazebo+GMRPD) and CARL-D datasets.
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch
import cv2
from tqdm import tqdm
from torch.utils.data import DataLoader
import argparse
import csv

sys.path.insert(0, str(Path(__file__).parent))
from model_factory import get_benchmark_model
from train_benchmark import DrivableAreaDataset, Config
from auraseg_v4_resnet import AURASeg_V4_ResNet50


def compute_boundary_metrics(pred, target, dilation=2, kernel_size=3):
    """Compute boundary metrics: IoU, Precision, Recall, F1
    
    Args:
        pred: Binary prediction mask (H, W) with values {0, 1}
        target: Binary target mask (H, W) with values {0, 1}
        dilation: Number of dilation iterations for boundary tolerance
        kernel_size: Size of morphological kernel
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    pred_boundary = cv2.morphologyEx(pred, cv2.MORPH_GRADIENT, kernel)
    target_boundary = cv2.morphologyEx(target, cv2.MORPH_GRADIENT, kernel)
    
    # Dilate boundaries to allow spatial tolerance
    pred_boundary = cv2.dilate(pred_boundary, kernel, iterations=dilation)
    target_boundary = cv2.dilate(target_boundary, kernel, iterations=dilation)
    
    tp = np.sum((pred_boundary > 0) & (target_boundary > 0))
    fp = np.sum((pred_boundary > 0) & (target_boundary == 0))
    fn = np.sum((pred_boundary == 0) & (target_boundary > 0))
    
    boundary_iou = tp / (tp + fp + fn + 1e-6)
    boundary_precision = tp / (tp + fp + 1e-6)
    boundary_recall = tp / (tp + fn + 1e-6)
    boundary_f1 = 2 * boundary_precision * boundary_recall / (boundary_precision + boundary_recall + 1e-6)
    
    return {
        'boundary_iou': boundary_iou,
        'boundary_precision': boundary_precision,
        'boundary_recall': boundary_recall,
        'boundary_f1': boundary_f1
    }


def compute_all_metrics(preds, targets, num_classes=2, boundary_dilation=2):
    """Compute all segmentation metrics matching V1-V4 format.
    
    Args:
        preds: Predictions array (N, H, W)
        targets: Targets array (N, H, W)
        num_classes: Number of classes
        boundary_dilation: Dilation iterations for boundary tolerance (higher = more tolerant)
    """
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
    
    # Boundary Metrics
    boundary_ious = []
    boundary_precisions = []
    boundary_recalls = []
    boundary_f1s = []
    
    for i in range(len(preds)):
        pred_binary = (preds[i] == 1).astype(np.uint8)
        target_binary = (targets[i] == 1).astype(np.uint8)
        
        b_metrics = compute_boundary_metrics(pred_binary, target_binary, dilation=boundary_dilation)
        boundary_ious.append(b_metrics['boundary_iou'])
        boundary_precisions.append(b_metrics['boundary_precision'])
        boundary_recalls.append(b_metrics['boundary_recall'])
        boundary_f1s.append(b_metrics['boundary_f1'])
    
    metrics['boundary_iou'] = np.mean(boundary_ious)
    metrics['boundary_precision'] = np.mean(boundary_precisions)
    metrics['boundary_recall'] = np.mean(boundary_recalls)
    metrics['boundary_f1'] = np.mean(boundary_f1s)
    
    return metrics


def evaluate_model(model_name, checkpoint_path, val_loader, device, is_fcn=False, is_auraseg=False, boundary_dilation=2):
    """Evaluate a single model."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name.upper()}")
    print(f"{'='*60}")
    
    # Load model
    if is_auraseg:
        model = AURASeg_V4_ResNet50(num_classes=2)
        info = {'name': 'AURASeg V4', 'encoder': 'ResNet50', 'paradigm': 'Multi-scale', 'params_millions': sum(p.numel() for p in model.parameters()) / 1e6}
    else:
        model, info = get_benchmark_model(model_name, num_classes=2, pretrained=False)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model: {info['name']}")
    print(f"Encoder: {info['encoder']}")
    print(f"Parameters: {info['params_millions']:.2f}M")
    print(f"Best mIoU from training: {checkpoint.get('best_miou', 'N/A'):.4f}")
    
    # Run inference
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Evaluating {model_name}"):
            # Handle different dataset return formats
            if len(batch) == 2:
                images, masks = batch
            else:
                images, masks = batch[0], batch[1]  # Ignore filename if present
            
            images = images.to(device)
            outputs = model(images)
            
            # Handle different output formats
            if is_auraseg:
                outputs = outputs['main']
            elif is_fcn:
                if isinstance(outputs, dict):
                    outputs = outputs['out']
            
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.append(preds.cpu().numpy())
            all_targets.append(masks.numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Compute metrics
    metrics = compute_all_metrics(all_preds, all_targets, boundary_dilation=boundary_dilation)
    
    return metrics, info


def print_table_iv(results, dataset_name):
    """Print results in Table IV format."""
    print(f"\n{'='*100}")
    print(f"TABLE IV: {dataset_name} DATASET RESULTS")
    print(f"{'='*100}")
    
    # Header
    header = f"{'Model':<20} | {'IoU':>8} | {'F1':>8} | {'BIoU':>8} | {'BF1':>8} | {'Precision':>10} | {'Recall':>8}"
    print(header)
    print("-" * 100)
    
    # Find best values for each metric
    metrics_names = ['iou_drivable', 'f1', 'boundary_iou', 'boundary_f1', 'precision', 'recall']
    display_names = ['IoU', 'F1', 'BIoU', 'BF1', 'Precision', 'Recall']
    best_values = {}
    for metric in metrics_names:
        values = [results[m]['metrics'][metric] for m in results]
        best_values[metric] = max(values)
    
    # Print rows
    for model_name, data in results.items():
        metrics = data['metrics']
        row = f"{model_name:<20}"
        for metric in metrics_names:
            val = metrics[metric]
            marker = "*" if val == best_values[metric] else ""
            row += f" | {val:>7.4f}{marker}"
        print(row)
    
    print("-" * 100)
    print("* indicates best performance")


def save_table_iv_csv(results, output_path, dataset_name):
    """Save Table IV results to CSV."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'IoU', 'F1', 'BIoU', 'BF1', 'Precision', 'Recall'])
        for model_name, data in results.items():
            metrics = data['metrics']
            writer.writerow([
                model_name,
                f"{metrics['iou_drivable']:.4f}",
                f"{metrics['f1']:.4f}",
                f"{metrics['boundary_iou']:.4f}",
                f"{metrics['boundary_f1']:.4f}",
                f"{metrics['precision']:.4f}",
                f"{metrics['recall']:.4f}"
            ])
    print(f"Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Unified Benchmark Evaluation')
    parser.add_argument('--dataset', choices=['mix', 'carl', 'both'], default='both',
                        help='Dataset to evaluate on')
    parser.add_argument('--split', choices=['val', 'test'], default='test',
                        help='Split to evaluate on')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    config = Config()
    base_dir = Path(__file__).parent.parent
    runs_dir = base_dir / "runs"
    runs_carl_dir = base_dir / "runs_carl"
    
    # All models to evaluate (display_name, model_key, is_fcn, is_auraseg)
    all_models = [
        ('FCN', 'fcn', True, False),
        ('PSPNet', 'pspnet', False, False),
        ('DeepLabV3+', 'deeplabv3plus', False, False),
        ('UPerNet', 'upernet', False, False),
        ('SegFormer', 'segformer', False, False),
        ('Mask2Former', 'mask2former', False, False),
        ('PIDNet', 'pidnet', False, False),
        ('AURASeg (Ours)', 'auraseg', False, True),
    ]
    
    # ========== MIX Dataset Evaluation ==========
    if args.dataset in ['mix', 'both']:
        print(f"\n{'='*80}")
        print("EVALUATING ON MIX DATASET (Gazebo + GMRPD)")
        print(f"{'='*80}")
        
        # Load MIX test dataset
        mix_dataset = DrivableAreaDataset(
            image_dir=config.IMAGE_DIR,
            mask_dir=config.MASK_DIR,
            img_size=config.IMG_SIZE,
            split=args.split,
            transform=False
        )
        
        mix_loader = DataLoader(
            mix_dataset,
            batch_size=8,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        print(f"MIX {args.split} samples: {len(mix_dataset)}")
        
        # Evaluate all models on MIX (dilation=2 for complex boundaries)
        mix_results = {}
        for display_name, model_key, is_fcn, is_auraseg in all_models:
            if is_auraseg:
                ckpt_path = runs_dir / 'auraseg_v4_resnet50' / 'checkpoints' / 'best.pth'
            else:
                ckpt_path = runs_dir / f'benchmark_{model_key}' / 'checkpoints' / 'best.pth'
            
            if not ckpt_path.exists():
                print(f"[SKIP] {display_name}: checkpoint not found at {ckpt_path}")
                continue
            
            metrics, info = evaluate_model(model_key, ckpt_path, mix_loader, device, is_fcn, is_auraseg, boundary_dilation=2)
            mix_results[display_name] = {'metrics': metrics, 'info': info}
        
        # Print and save MIX results
        print_table_iv(mix_results, 'MIX (Gazebo + GMRPD)')
        save_table_iv_csv(mix_results, runs_dir / 'table_iv_mix.csv', 'MIX')
    
    # ========== CARL-D Dataset Evaluation ==========
    if args.dataset in ['carl', 'both']:
        print(f"\n{'='*80}")
        print("EVALUATING ON CARL-D DATASET")
        print(f"{'='*80}")
        
        # Import CARL dataset class from eval_carl_benchmarks
        from eval_carl_benchmarks import CARLTestDataset
        
        carl_dir = base_dir / "carl-dataset"
        carl_dataset = CARLTestDataset(
            image_dir=carl_dir / "test" / "test",
            label_dir=carl_dir / "test" / "labels",
            img_size=config.IMG_SIZE  # (H, W) = (384, 640)
        )
        
        carl_loader = DataLoader(
            carl_dataset,
            batch_size=8,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        print(f"CARL-D test samples: {len(carl_dataset)}")
        
        # Evaluate all models on CARL-D (dilation=8 for simpler polygon-style masks)
        carl_results = {}
        for display_name, model_key, is_fcn, is_auraseg in all_models:
            if is_auraseg:
                ckpt_path = runs_carl_dir / 'auraseg_v4_resnet50' / 'checkpoints' / 'best.pth'
            else:
                ckpt_path = runs_carl_dir / f'benchmark_{model_key}' / 'checkpoints' / 'best.pth'
            
            if not ckpt_path.exists():
                print(f"[SKIP] {display_name}: checkpoint not found at {ckpt_path}")
                continue
            
            metrics, info = evaluate_model(model_key, ckpt_path, carl_loader, device, is_fcn, is_auraseg, boundary_dilation=2)
            carl_results[display_name] = {'metrics': metrics, 'info': info}
        
        # Print and save CARL-D results
        print_table_iv(carl_results, 'CARL-D')
        save_table_iv_csv(carl_results, runs_dir / 'table_iv_carl.csv', 'CARL-D')
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
