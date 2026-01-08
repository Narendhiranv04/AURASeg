"""
CARL Dataset - AURASeg V4 ResNet50 Evaluation
==============================================

Evaluate AURASeg V4 (ResNet50 backbone) trained on CARL dataset.

Usage:
    python eval_auraseg_carl.py
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from auraseg_v4_resnet import AURASeg_V4_ResNet50


# =============================================================================
# Configuration
# =============================================================================

class CARLConfig:
    """Configuration for CARL dataset evaluation."""
    
    # Paths
    CARL_ROOT = Path(__file__).parent.parent / "carl-dataset"
    TEST_IMAGE_DIR = CARL_ROOT / "test" / "test"
    TEST_LABEL_DIR = CARL_ROOT / "test" / "labels"
    
    RUNS_DIR = Path(__file__).parent.parent / "runs_carl"
    CHECKPOINT_PATH = RUNS_DIR / "auraseg_v4_resnet50" / "checkpoints" / "best.pth"
    OUTPUT_DIR = RUNS_DIR / "evaluation_results"
    
    # Model settings
    IMG_SIZE = (384, 640)
    NUM_CLASSES = 2
    
    # DataLoader settings
    BATCH_SIZE = 4
    NUM_WORKERS = 4
    PIN_MEMORY = True
    
    # Normalization (ImageNet)
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]


# =============================================================================
# Dataset
# =============================================================================

class CARLTestDataset(Dataset):
    """CARL Dataset test set loader."""
    
    def __init__(self, image_dir, label_dir, img_size=(384, 640), mean=None, std=None):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.img_size = img_size
        self.mean = mean or [0.485, 0.456, 0.406]
        self.std = std or [0.229, 0.224, 0.225]
        
        # Get all test images
        self.images = []
        self.labels = []
        
        for img_path in sorted(self.image_dir.glob("*.jpg")):
            # CARL labels have ___fuse.png suffix
            label_name = img_path.name + "___fuse.png"
            label_path = self.label_dir / label_name
            
            if label_path.exists():
                self.images.append(img_path)
                self.labels.append(label_path)
        
        print(f"Found {len(self.images)} test samples in CARL dataset")
        
        # Transforms
        self.normalize = T.Normalize(mean=self.mean, std=self.std)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.images[idx]).convert('RGB')
        image = image.resize((self.img_size[1], self.img_size[0]), Image.BILINEAR)
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        image = self.normalize(image)
        
        # Load label
        label = Image.open(self.labels[idx]).convert('L')
        label = label.resize((self.img_size[1], self.img_size[0]), Image.NEAREST)
        label = np.array(label)
        
        # Binarize mask (must match training binarization!)
        label = self._binarize_mask(label)
        label = torch.from_numpy(label)
        
        return image, label, str(self.images[idx].name)
    
    def _binarize_mask(self, mask):
        """
        Convert grayscale mask to binary {0,1}.
        Must match unified_dataset.py binarization used during training!
        
        For CARL with 3 values (0, 21, 109):
        - Training used (mask > 0) which maps both 21 and 109 to drivable
        """
        uniq = np.unique(mask)
        if uniq.size == 0:
            return np.zeros_like(mask, dtype=np.int64)
        if uniq.size == 1:
            return (mask > 0).astype(np.int64)
        if uniq.size == 2:
            return (mask == uniq.max()).astype(np.int64)
        # Three+ values: training used (mask > 0)
        return (mask > 0).astype(np.int64)


# =============================================================================
# Metrics
# =============================================================================

def compute_segmentation_metrics(preds, targets, num_classes=2):
    """Compute comprehensive segmentation metrics."""
    preds = preds.flatten()
    targets = targets.flatten()
    
    # Per-class IoU
    ious = []
    for cls in range(num_classes):
        pred_cls = (preds == cls)
        target_cls = (targets == cls)
        
        intersection = np.sum(pred_cls & target_cls)
        union = np.sum(pred_cls | target_cls)
        
        if union > 0:
            ious.append(intersection / union)
        else:
            ious.append(1.0 if intersection == 0 else 0.0)
    
    iou_background = ious[0]
    iou_drivable = ious[1]
    miou = np.mean(ious)
    
    # Classification metrics for drivable class
    tp = np.sum((preds == 1) & (targets == 1))
    fp = np.sum((preds == 1) & (targets == 0))
    fn = np.sum((preds == 0) & (targets == 1))
    tn = np.sum((preds == 0) & (targets == 0))
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    
    return {
        'miou': miou,
        'iou_drivable': iou_drivable,
        'iou_background': iou_background,
        'dice': dice,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy
    }


def compute_boundary_metrics(pred, target, kernel_size=5):
    """Compute boundary-specific metrics."""
    kernel = np.ones((3, 3), np.uint8)
    
    pred_binary = (pred == 1).astype(np.uint8)
    target_binary = (target == 1).astype(np.uint8)
    
    pred_boundary = cv2.morphologyEx(pred_binary, cv2.MORPH_GRADIENT, kernel)
    target_boundary = cv2.morphologyEx(target_binary, cv2.MORPH_GRADIENT, kernel)
    
    pred_boundary = cv2.dilate(pred_boundary, kernel, iterations=2)
    target_boundary = cv2.dilate(target_boundary, kernel, iterations=2)
    
    pred_bnd = pred_boundary.flatten() > 0
    target_bnd = target_boundary.flatten() > 0
    
    intersection = np.sum(pred_bnd & target_bnd)
    union = np.sum(pred_bnd | target_bnd)
    boundary_iou = intersection / (union + 1e-8)
    
    tp = np.sum(pred_bnd & target_bnd)
    fp = np.sum(pred_bnd & ~target_bnd)
    fn = np.sum(~pred_bnd & target_bnd)
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return {
        'boundary_iou': boundary_iou,
        'boundary_precision': precision,
        'boundary_recall': recall,
        'boundary_f1': f1
    }


class AverageMeter:
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


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_model(model, data_loader, device):
    """Evaluate AURASeg V4 on the test set."""
    seg_keys = ['miou', 'iou_drivable', 'iou_background', 'dice', 'precision', 'recall', 'f1', 'accuracy']
    bnd_keys = ['boundary_iou', 'boundary_precision', 'boundary_recall', 'boundary_f1']
    
    metrics = {k: AverageMeter() for k in seg_keys + bnd_keys}
    
    model.eval()
    with torch.no_grad():
        for images, masks, _ in tqdm(data_loader, desc="Evaluating AURASeg V4"):
            images = images.to(device)
            masks = masks.numpy()
            
            # Get predictions
            outputs = model(images, return_aux=False, return_boundary=False)
            
            # AURASeg V4 returns dict with 'main' key for main output
            if isinstance(outputs, dict):
                outputs = outputs['main']
            
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            # Compute metrics per sample
            for i in range(len(preds)):
                seg_m = compute_segmentation_metrics(preds[i], masks[i])
                for k, v in seg_m.items():
                    metrics[k].update(v)
                
                bnd_m = compute_boundary_metrics(preds[i], masks[i])
                for k, v in bnd_m.items():
                    metrics[k].update(v)
    
    return {k: m.avg for k, m in metrics.items()}


def measure_inference_time(model, device, input_size=(384, 640), num_warmup=10, num_runs=50):
    """Measure model inference time."""
    model.eval()
    x = torch.randn(1, 3, input_size[0], input_size[1]).to(device)
    
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(x)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            _ = model(x)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            times.append(time.perf_counter() - start)
    
    times = np.array(times) * 1000
    
    return {
        'latency_mean_ms': float(np.mean(times)),
        'latency_std_ms': float(np.std(times)),
        'fps': float(1000.0 / np.mean(times))
    }


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 80)
    print("CARL DATASET - AURASeg V4 ResNet50 EVALUATION")
    print("=" * 80)
    
    config = CARLConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    print(f"Checkpoint: {config.CHECKPOINT_PATH}")
    
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load test dataset
    test_dataset = CARLTestDataset(
        image_dir=config.TEST_IMAGE_DIR,
        label_dir=config.TEST_LABEL_DIR,
        img_size=config.IMG_SIZE,
        mean=config.MEAN,
        std=config.STD
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    # Load model
    print("\n--- Loading AURASeg V4 ResNet50 ---")
    model = AURASeg_V4_ResNet50(
        num_classes=config.NUM_CLASSES,
        decoder_channels=256,
        encoder_weights=None  # Will load from checkpoint
    ).to(device)
    
    if config.CHECKPOINT_PATH.exists():
        checkpoint = torch.load(config.CHECKPOINT_PATH, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded checkpoint: {config.CHECKPOINT_PATH}")
        print(f"  Best mIoU from training: {checkpoint.get('best_miou', 'N/A')}")
    else:
        print(f"  ERROR: Checkpoint not found at {config.CHECKPOINT_PATH}")
        return
    
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params / 1e6:.2f}M")
    
    # Evaluate
    print("\n--- Evaluating on CARL Test Set ---")
    metrics = evaluate_model(model, test_loader, device)
    
    # Measure inference time
    print("\n--- Measuring Inference Time ---")
    timing = measure_inference_time(model, device, config.IMG_SIZE)
    
    # Print results
    print("\n" + "=" * 80)
    print("AURASeg V4 ResNet50 - CARL TEST SET RESULTS")
    print("=" * 80)
    
    print("\nSEGMENTATION METRICS:")
    print(f"  mIoU:             {metrics['miou']:.4f}")
    print(f"  IoU (Drivable):   {metrics['iou_drivable']:.4f}")
    print(f"  IoU (Background): {metrics['iou_background']:.4f}")
    print(f"  Dice:             {metrics['dice']:.4f}")
    print(f"  Precision:        {metrics['precision']:.4f}")
    print(f"  Recall:           {metrics['recall']:.4f}")
    print(f"  F1:               {metrics['f1']:.4f}")
    print(f"  Accuracy:         {metrics['accuracy']:.4f}")
    
    print("\nBOUNDARY METRICS:")
    print(f"  Boundary IoU:       {metrics['boundary_iou']:.4f}")
    print(f"  Boundary Precision: {metrics['boundary_precision']:.4f}")
    print(f"  Boundary Recall:    {metrics['boundary_recall']:.4f}")
    print(f"  Boundary F1:        {metrics['boundary_f1']:.4f}")
    
    print("\nINFERENCE PERFORMANCE:")
    print(f"  Parameters:   {total_params / 1e6:.2f}M")
    print(f"  Latency:      {timing['latency_mean_ms']:.2f} ± {timing['latency_std_ms']:.2f} ms")
    print(f"  FPS:          {timing['fps']:.1f}")
    
    print("=" * 80)
    
    # Save results
    result = {
        'Model': 'AURASeg V4-R50',
        'Encoder': 'ResNet-50',
        'Params (M)': total_params / 1e6,
        'mIoU': metrics['miou'],
        'IoU (Drivable)': metrics['iou_drivable'],
        'IoU (Background)': metrics['iou_background'],
        'Dice': metrics['dice'],
        'Precision': metrics['precision'],
        'Recall': metrics['recall'],
        'F1': metrics['f1'],
        'Accuracy': metrics['accuracy'],
        'Boundary IoU': metrics['boundary_iou'],
        'Boundary F1': metrics['boundary_f1'],
        'Boundary Precision': metrics['boundary_precision'],
        'Boundary Recall': metrics['boundary_recall'],
        'Latency (ms)': timing['latency_mean_ms'],
        'FPS': timing['fps']
    }
    
    # Save to CSV
    csv_path = config.OUTPUT_DIR / "carl_auraseg_results.csv"
    pd.DataFrame([result]).to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    # Append to combined results
    combined_csv = config.OUTPUT_DIR / "carl_benchmark_results.csv"
    if combined_csv.exists():
        combined_df = pd.read_csv(combined_csv)
        # Remove existing AURASeg entry if present
        combined_df = combined_df[~combined_df['Model'].str.contains('AURASeg', case=False, na=False)]
        combined_df = pd.concat([combined_df, pd.DataFrame([result])], ignore_index=True)
        combined_df = combined_df.sort_values('mIoU', ascending=False)
        combined_df.to_csv(combined_csv, index=False)
        print(f"Updated combined results: {combined_csv}")
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
