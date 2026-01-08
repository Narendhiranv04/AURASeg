"""
Comprehensive Evaluation Script for DeepLabV3+
================================================

Computes all metrics matching the V1-V4 evaluation format.
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
    """
    Compute boundary metrics: IoU, Precision, Recall, F1
    
    Args:
        pred: Binary prediction (H, W)
        target: Binary ground truth (H, W)
        dilation: Dilation for boundary extraction
        
    Returns:
        dict with boundary metrics
    """
    kernel = np.ones((3, 3), np.uint8)
    
    # Extract boundaries using morphological gradient
    pred_boundary = cv2.morphologyEx(pred, cv2.MORPH_GRADIENT, kernel)
    target_boundary = cv2.morphologyEx(target, cv2.MORPH_GRADIENT, kernel)
    
    # Dilate boundaries
    pred_boundary = cv2.dilate(pred_boundary, kernel, iterations=dilation)
    target_boundary = cv2.dilate(target_boundary, kernel, iterations=dilation)
    
    # Compute metrics on boundaries
    tp = np.sum((pred_boundary > 0) & (target_boundary > 0))
    fp = np.sum((pred_boundary > 0) & (target_boundary == 0))
    fn = np.sum((pred_boundary == 0) & (target_boundary > 0))
    
    intersection = tp
    union = tp + fp + fn
    
    boundary_iou = tp / (union + 1e-6)
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
    """
    Compute all segmentation metrics matching V1-V4 format.
    
    Returns dict with:
        - miou, iou_drivable, iou_background
        - dice
        - precision, recall, f1, accuracy
        - boundary_iou, boundary_precision, boundary_recall, boundary_f1
    """
    metrics = {}
    
    # ========== Segmentation Metrics ==========
    
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
    
    # Accuracy
    metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn + 1e-6)
    
    # ========== Boundary Metrics ==========
    
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


def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = Config()
    
    print("=" * 80)
    print("DeepLabV3+ Evaluation on Validation Set")
    print("=" * 80)
    
    # Load model
    checkpoint_path = Path(__file__).parent.parent / "runs" / "benchmark_deeplabv3plus" / "checkpoints" / "best.pth"
    
    print(f"Checkpoint: {checkpoint_path}")
    
    model, info = get_benchmark_model('deeplabv3plus', num_classes=2, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model: {info['name']}")
    print(f"Encoder: {info['encoder']}")
    print(f"Parameters: {info['params_millions']:.2f}M")
    print(f"Best mIoU from training: {checkpoint.get('best_miou', 'N/A'):.4f}")
    print()
    
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
    
    print(f"Validation samples: {len(val_dataset)}")
    print()
    
    # Run inference
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.append(preds.cpu().numpy())
            all_targets.append(masks.numpy())
    
    # Concatenate all
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Compute metrics
    metrics = compute_all_metrics(all_preds, all_targets)
    
    # Print results in V1-V4 format
    print()
    print("=" * 80)
    print()
    print("Metric                  | DeepLabV3+")
    print("-" * 80)
    print()
    print("SEGMENTATION METRICS:")
    print(f"    miou                | {metrics['miou']:.4f}")
    print(f"    iou_drivable        | {metrics['iou_drivable']:.4f}")
    print(f"    iou_background      | {metrics['iou_background']:.4f}")
    print(f"    dice                | {metrics['dice']:.4f}")
    print(f"    precision           | {metrics['precision']:.4f}")
    print(f"    recall              | {metrics['recall']:.4f}")
    print(f"    f1                  | {metrics['f1']:.4f}")
    print(f"    accuracy            | {metrics['accuracy']:.4f}")
    print()
    print("BOUNDARY METRICS:")
    print(f"    boundary_iou        | {metrics['boundary_iou']:.4f}")
    print(f"    boundary_precision  | {metrics['boundary_precision']:.4f}")
    print(f"    boundary_recall     | {metrics['boundary_recall']:.4f}")
    print(f"    boundary_f1         | {metrics['boundary_f1']:.4f}")
    print()
    print("=" * 80)
    
    # Save to file
    output_path = Path(__file__).parent.parent / "runs" / "benchmark_deeplabv3plus" / "evaluation_results.txt"
    with open(output_path, 'w') as f:
        f.write("DeepLabV3+ Evaluation Results\n")
        f.write("=" * 60 + "\n\n")
        f.write("SEGMENTATION METRICS:\n")
        f.write(f"    miou:               {metrics['miou']:.4f}\n")
        f.write(f"    iou_drivable:       {metrics['iou_drivable']:.4f}\n")
        f.write(f"    iou_background:     {metrics['iou_background']:.4f}\n")
        f.write(f"    dice:               {metrics['dice']:.4f}\n")
        f.write(f"    precision:          {metrics['precision']:.4f}\n")
        f.write(f"    recall:             {metrics['recall']:.4f}\n")
        f.write(f"    f1:                 {metrics['f1']:.4f}\n")
        f.write(f"    accuracy:           {metrics['accuracy']:.4f}\n\n")
        f.write("BOUNDARY METRICS:\n")
        f.write(f"    boundary_iou:       {metrics['boundary_iou']:.4f}\n")
        f.write(f"    boundary_precision: {metrics['boundary_precision']:.4f}\n")
        f.write(f"    boundary_recall:    {metrics['boundary_recall']:.4f}\n")
        f.write(f"    boundary_f1:        {metrics['boundary_f1']:.4f}\n")
    
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
