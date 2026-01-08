"""
CARL Dataset Benchmark Evaluation Script
==========================================

Evaluate all 7 benchmark models trained on CARL dataset.

Models:
    - FCN-ResNet50
    - PSPNet-ResNet50
    - DeepLabV3+-ResNet50
    - UPerNet-ResNet50
    - SegFormer-B2
    - Mask2Former
    - PIDNet-L

Metrics:
    - mIoU (Mean Intersection over Union)
    - IoU (Drivable class)
    - IoU (Background class)
    - Dice Score
    - Precision / Recall / F1
    - Accuracy
    - Boundary IoU
    - Boundary F1
    - Boundary Precision / Recall

Usage:
    python eval_carl_benchmarks.py
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
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from model_factory import get_benchmark_model, BENCHMARK_MODELS


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
    
    # Models to evaluate
    BENCHMARK_MODELS = [
        'fcn',
        'pspnet',
        'deeplabv3plus',
        'upernet',
        'segformer',
        'mask2former',
        'pidnet'
    ]


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
        
        # Binarize mask robustly (handles CARL's 21/109 encoding)
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
            # Two values: max is drivable
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
    
    # For binary: class 0 = background, class 1 = drivable
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
    
    # Dice score
    dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
    
    # Accuracy
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
    # Generate boundary masks using morphological operations
    kernel = np.ones((3, 3), np.uint8)
    
    pred_binary = (pred == 1).astype(np.uint8)
    target_binary = (target == 1).astype(np.uint8)
    
    # Extract boundaries using morphological gradient
    pred_boundary = cv2.morphologyEx(pred_binary, cv2.MORPH_GRADIENT, kernel)
    target_boundary = cv2.morphologyEx(target_binary, cv2.MORPH_GRADIENT, kernel)
    
    # Dilate boundaries for tolerance
    pred_boundary = cv2.dilate(pred_boundary, kernel, iterations=2)
    target_boundary = cv2.dilate(target_boundary, kernel, iterations=2)
    
    pred_bnd = pred_boundary.flatten() > 0
    target_bnd = target_boundary.flatten() > 0
    
    # Boundary IoU
    intersection = np.sum(pred_bnd & target_bnd)
    union = np.sum(pred_bnd | target_bnd)
    boundary_iou = intersection / (union + 1e-8)
    
    # Boundary Precision, Recall, F1
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
    """Computes and stores the average and current value."""
    
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
# Model Loading
# =============================================================================

def find_checkpoint(model_name, runs_dir):
    """Find best checkpoint for a model."""
    run_dir = runs_dir / f"benchmark_{model_name}"
    best_path = run_dir / "checkpoints" / "best.pth"
    
    if best_path.exists():
        return best_path
    
    # Try latest
    latest_path = run_dir / "checkpoints" / "latest.pth"
    if latest_path.exists():
        return latest_path
    
    return None


def load_model(model_name, checkpoint_path, device, num_classes=2):
    """Load a trained model from checkpoint."""
    model, info = get_benchmark_model(model_name, num_classes=num_classes, pretrained=False)
    
    if checkpoint_path and Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded checkpoint: {checkpoint_path}")
        print(f"  Best mIoU from training: {checkpoint.get('best_miou', 'N/A'):.4f}" if checkpoint.get('best_miou') else "  Best mIoU: N/A")
    else:
        print(f"  Warning: No checkpoint found at {checkpoint_path}")
        return None, None
    
    model = model.to(device)
    model.eval()
    return model, info


def measure_inference_time(model, device, input_size=(384, 640), num_warmup=10, num_runs=50):
    """Measure model inference time."""
    model.eval()
    x = torch.randn(1, 3, input_size[0], input_size[1]).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(x)
    
    # Synchronize CUDA
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Measure
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
    
    times = np.array(times) * 1000  # Convert to ms
    
    return {
        'latency_mean_ms': float(np.mean(times)),
        'latency_std_ms': float(np.std(times)),
        'fps': float(1000.0 / np.mean(times))
    }


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_model(model, model_name, data_loader, device):
    """Evaluate a single model on the test set."""
    # Metrics storage
    seg_keys = ['miou', 'iou_drivable', 'iou_background', 'dice', 'precision', 'recall', 'f1', 'accuracy']
    bnd_keys = ['boundary_iou', 'boundary_precision', 'boundary_recall', 'boundary_f1']
    
    metrics = {k: AverageMeter() for k in seg_keys + bnd_keys}
    
    model.eval()
    with torch.no_grad():
        for images, masks, _ in tqdm(data_loader, desc=f"Evaluating {model_name}", leave=False):
            images = images.to(device)
            masks = masks.numpy()
            
            # Get predictions - handle different model output formats
            outputs = model(images)
            
            # Handle dict output (FCN returns OrderedDict with 'out' key)
            if isinstance(outputs, dict):
                outputs = outputs['out']
            
            # Resize if needed (FCN may output different size)
            if outputs.shape[-2:] != images.shape[-2:]:
                outputs = F.interpolate(outputs, size=images.shape[-2:], mode='bilinear', align_corners=False)
            
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


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 80)
    print("CARL DATASET - BENCHMARK MODEL EVALUATION")
    print("=" * 80)
    
    # Setup
    config = CARLConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    print(f"Test Image Dir: {config.TEST_IMAGE_DIR}")
    print(f"Test Label Dir: {config.TEST_LABEL_DIR}")
    print(f"Runs Dir: {config.RUNS_DIR}")
    
    # Create output directory
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
    
    # Results storage
    results = []
    
    # Evaluate each model
    for model_name in config.BENCHMARK_MODELS:
        print(f"\n{'=' * 60}")
        print(f"Evaluating: {model_name.upper()}")
        print(f"{'=' * 60}")
        
        # Find checkpoint
        checkpoint_path = find_checkpoint(model_name, config.RUNS_DIR)
        
        if checkpoint_path is None:
            print(f"  [SKIP] No checkpoint found for {model_name}")
            continue
        
        # Load model
        model, info = load_model(model_name, checkpoint_path, device, config.NUM_CLASSES)
        
        if model is None:
            continue
        
        # Evaluate
        metrics = evaluate_model(model, model_name, test_loader, device)
        
        # Measure inference time
        timing = measure_inference_time(model, device, config.IMG_SIZE)
        
        # Compile results
        result = {
            'Model': info['name'],
            'Encoder': info['encoder'],
            'Paradigm': info['paradigm'],
            'Params (M)': info['params_millions'],
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
        results.append(result)
        
        # Print summary
        print(f"\n  Results for {model_name}:")
        print(f"    mIoU:           {metrics['miou']:.4f}")
        print(f"    IoU (Drivable): {metrics['iou_drivable']:.4f}")
        print(f"    Dice:           {metrics['dice']:.4f}")
        print(f"    F1:             {metrics['f1']:.4f}")
        print(f"    Boundary IoU:   {metrics['boundary_iou']:.4f}")
        print(f"    Boundary F1:    {metrics['boundary_f1']:.4f}")
        print(f"    FPS:            {timing['fps']:.1f}")
        
        # Cleanup
        del model
        torch.cuda.empty_cache()
    
    # Create results DataFrame
    if not results:
        print("\nNo models were evaluated!")
        return
    
    results_df = pd.DataFrame(results)
    
    # Sort by mIoU
    results_df = results_df.sort_values('mIoU', ascending=False)
    
    # Save to CSV
    csv_path = config.OUTPUT_DIR / "carl_benchmark_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    # Print full table
    print("\n" + "=" * 120)
    print("CARL DATASET - BENCHMARK COMPARISON RESULTS (TEST SET)")
    print("=" * 120)
    
    # Format for display
    display_cols = ['Model', 'Params (M)', 'mIoU', 'IoU (Drivable)', 'Dice', 
                    'F1', 'Boundary IoU', 'Boundary F1', 'FPS']
    display_df = results_df[display_cols].copy()
    
    # Format numeric columns
    for col in ['mIoU', 'IoU (Drivable)', 'Dice', 'F1', 'Boundary IoU', 'Boundary F1']:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
    display_df['FPS'] = display_df['FPS'].apply(lambda x: f"{x:.1f}")
    display_df['Params (M)'] = display_df['Params (M)'].apply(lambda x: f"{x:.2f}")
    
    print(display_df.to_string(index=False))
    print("=" * 120)
    
    # Save detailed results to text file
    txt_path = config.OUTPUT_DIR / "carl_benchmark_results.txt"
    with open(txt_path, 'w') as f:
        f.write("CARL DATASET - BENCHMARK MODEL EVALUATION RESULTS\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Test samples: {len(test_dataset)}\n")
        f.write("=" * 120 + "\n\n")
        
        f.write("SEGMENTATION METRICS:\n")
        f.write("-" * 120 + "\n")
        header = f"{'Model':<20} | {'mIoU':>8} | {'IoU-Drv':>8} | {'IoU-Bg':>8} | {'Dice':>8} | {'Prec':>8} | {'Recall':>8} | {'F1':>8} | {'Acc':>8}\n"
        f.write(header)
        f.write("-" * 120 + "\n")
        
        for _, row in results_df.iterrows():
            line = f"{row['Model']:<20} | {row['mIoU']:>8.4f} | {row['IoU (Drivable)']:>8.4f} | {row['IoU (Background)']:>8.4f} | {row['Dice']:>8.4f} | {row['Precision']:>8.4f} | {row['Recall']:>8.4f} | {row['F1']:>8.4f} | {row['Accuracy']:>8.4f}\n"
            f.write(line)
        
        f.write("\n\nBOUNDARY METRICS:\n")
        f.write("-" * 100 + "\n")
        header = f"{'Model':<20} | {'Bnd IoU':>10} | {'Bnd Prec':>10} | {'Bnd Recall':>10} | {'Bnd F1':>10}\n"
        f.write(header)
        f.write("-" * 100 + "\n")
        
        for _, row in results_df.iterrows():
            line = f"{row['Model']:<20} | {row['Boundary IoU']:>10.4f} | {row['Boundary Precision']:>10.4f} | {row['Boundary Recall']:>10.4f} | {row['Boundary F1']:>10.4f}\n"
            f.write(line)
        
        f.write("\n\nINFERENCE PERFORMANCE:\n")
        f.write("-" * 80 + "\n")
        header = f"{'Model':<20} | {'Params (M)':>12} | {'Latency (ms)':>14} | {'FPS':>10}\n"
        f.write(header)
        f.write("-" * 80 + "\n")
        
        for _, row in results_df.iterrows():
            line = f"{row['Model']:<20} | {row['Params (M)']:>12.2f} | {row['Latency (ms)']:>14.2f} | {row['FPS']:>10.1f}\n"
            f.write(line)
        
        f.write("\n" + "=" * 120 + "\n")
    
    print(f"Detailed results saved to: {txt_path}")
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
