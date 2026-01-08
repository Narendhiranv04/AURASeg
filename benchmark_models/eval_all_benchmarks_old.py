"""
Comprehensive Evaluation Script for All Benchmark Models
=========================================================

Computes all metrics matching the V1-V4 evaluation format.
Evaluates: DeepLabV3+, SegFormer, and any other trained models.
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch
import cv2
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from model_factory import get_benchmark_model
from train_benchmark import DrivableAreaDataset, Config


def compute_boundary_metrics(pred, target, dilation=2):
    """Compute boundary metrics: IoU, Precision, Recall, F1"""
    kernel = np.ones((3, 3), np.uint8)
    
    pred_boundary = cv2.morphologyEx(pred, cv2.MORPH_GRADIENT, kernel)
    target_boundary = cv2.morphologyEx(target, cv2.MORPH_GRADIENT, kernel)
    
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


def compute_all_metrics(preds, targets, num_classes=2):
    """Compute all segmentation metrics matching V1-V4 format."""
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
        
        b_metrics = compute_boundary_metrics(pred_binary, target_binary)
        boundary_ious.append(b_metrics['boundary_iou'])
        boundary_precisions.append(b_metrics['boundary_precision'])
        boundary_recalls.append(b_metrics['boundary_recall'])
        boundary_f1s.append(b_metrics['boundary_f1'])
    
    metrics['boundary_iou'] = np.mean(boundary_ious)
    metrics['boundary_precision'] = np.mean(boundary_precisions)
    metrics['boundary_recall'] = np.mean(boundary_recalls)
    metrics['boundary_f1'] = np.mean(boundary_f1s)
    
    return metrics


def evaluate_model(model_name, checkpoint_path, val_loader, device):
    """Evaluate a single model."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name.upper()}")
    print(f"{'='*60}")
    
    # Load model
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
        for images, masks in tqdm(val_loader, desc=f"Evaluating {model_name}"):
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.append(preds.cpu().numpy())
            all_targets.append(masks.numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Compute metrics
    metrics = compute_all_metrics(all_preds, all_targets)
    
    return metrics, info


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = Config()
    runs_dir = Path(__file__).parent.parent / "runs"
    
    # Models to evaluate
    models_to_eval = [
        ('deeplabv3plus', runs_dir / 'benchmark_deeplabv3plus' / 'checkpoints' / 'best.pth'),
        ('segformer', runs_dir / 'benchmark_segformer' / 'checkpoints' / 'best.pth'),
        ('mask2former', runs_dir / 'benchmark_mask2former' / 'checkpoints' / 'best.pth'),
    ]
    
    # Check which models are available
    available_models = []
    for name, ckpt_path in models_to_eval:
        if ckpt_path.exists():
            available_models.append((name, ckpt_path))
            print(f"[OK] {name}: {ckpt_path}")
        else:
            print(f"[MISSING] {name}: {ckpt_path}")
    
    if not available_models:
        print("No trained models found!")
        return
    
    # Load validation dataset
    val_dataset = DrivableAreaDataset(
        image_dir=config.IMAGE_DIR,
        mask_dir=config.MASK_DIR,
        img_size=config.IMG_SIZE,
        split='val',
        transform=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"\nValidation samples: {len(val_dataset)}")
    
    # Evaluate all models
    all_results = {}
    for model_name, ckpt_path in available_models:
        metrics, info = evaluate_model(model_name, ckpt_path, val_loader, device)
        all_results[model_name] = {
            'metrics': metrics,
            'info': info
        }
    
    # Print comparison table
    print("\n")
    print("=" * 100)
    print("BENCHMARK COMPARISON RESULTS")
    print("=" * 100)
    
    # Header
    model_names = list(all_results.keys())
    header = f"{'Metric':<25}"
    for name in model_names:
        header += f" | {name:<15}"
    print(header)
    print("-" * 100)
    
    # Segmentation metrics
    print("\nSEGMENTATION METRICS:")
    seg_metrics = ['miou', 'iou_drivable', 'iou_background', 'dice', 'precision', 'recall', 'f1', 'accuracy']
    for metric in seg_metrics:
        row = f"    {metric:<21}"
        values = [all_results[name]['metrics'][metric] for name in model_names]
        best_val = max(values)
        for i, name in enumerate(model_names):
            val = all_results[name]['metrics'][metric]
            marker = "*" if val == best_val else ""
            row += f" | {val:.4f}{marker:<10}"
        print(row)
    
    # Boundary metrics
    print("\nBOUNDARY METRICS:")
    boundary_metrics = ['boundary_iou', 'boundary_precision', 'boundary_recall', 'boundary_f1']
    for metric in boundary_metrics:
        row = f"    {metric:<21}"
        values = [all_results[name]['metrics'][metric] for name in model_names]
        best_val = max(values)
        for i, name in enumerate(model_names):
            val = all_results[name]['metrics'][metric]
            marker = "*" if val == best_val else ""
            row += f" | {val:.4f}{marker:<10}"
        print(row)
    
    print("=" * 100)
    print("\n* indicates best performance for each metric")
    
    # Save results
    output_path = runs_dir / "benchmark_comparison_results.txt"
    with open(output_path, 'w') as f:
        f.write("BENCHMARK COMPARISON RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        for model_name in model_names:
            info = all_results[model_name]['info']
            metrics = all_results[model_name]['metrics']
            
            f.write(f"Model: {info['name']}\n")
            f.write(f"Encoder: {info['encoder']}\n")
            f.write(f"Paradigm: {info['paradigm']}\n")
            f.write(f"Parameters: {info['params_millions']:.2f}M\n")
            f.write("-" * 40 + "\n")
            f.write("SEGMENTATION METRICS:\n")
            for metric in seg_metrics:
                f.write(f"    {metric}: {metrics[metric]:.4f}\n")
            f.write("BOUNDARY METRICS:\n")
            for metric in boundary_metrics:
                f.write(f"    {metric}: {metrics[metric]:.4f}\n")
            f.write("\n")
    
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
