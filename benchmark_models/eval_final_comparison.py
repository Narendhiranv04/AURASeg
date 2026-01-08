"""
Final Comparison: AURASeg V4-R50 vs Benchmark Models
=====================================================

Evaluates all 4 models on validation dataset with complete metrics.
Saves visualizations for each model.
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
from auraseg_v4_resnet import AURASeg_V4_ResNet50


class DrivableAreaDatasetWithPaths(DrivableAreaDataset):
    """Dataset that also returns image filenames."""
    
    def __getitem__(self, idx):
        image, mask = super().__getitem__(idx)
        img_path = self.images[idx]
        return image, mask, img_path.name


def denormalize(img_tensor, mean, std):
    """Convert normalized CHW tensor to uint8 HWC RGB."""
    img = img_tensor.cpu().numpy()
    if img.ndim == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    img = (img * std) + mean
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img


def mask_to_uint8(mask):
    """Convert binary mask to 0/255 uint8."""
    mask = np.asarray(mask)
    return ((mask > 0).astype(np.uint8) * 255)


def safe_stem(filename):
    """Create filesystem-safe stem from filename."""
    stem = Path(filename).stem
    return stem.replace(' ', '_').replace('/', '_').replace('\\', '_')


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
    """Compute all segmentation metrics."""
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


def evaluate_benchmark_model(model_name, checkpoint_path, val_loader, device, output_dir, config):
    """Evaluate a benchmark model and save visualizations."""
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
    
    # Create output directories
    pred_dir = output_dir / model_name / "pred_masks"
    compare_dir = output_dir / model_name / "compare"
    pred_dir.mkdir(parents=True, exist_ok=True)
    compare_dir.mkdir(parents=True, exist_ok=True)
    
    # Run inference
    all_preds = []
    all_targets = []
    saved = 0
    
    with torch.no_grad():
        for images, masks, names in tqdm(val_loader, desc=f"Evaluating {model_name}"):
            images = images.to(device)
            outputs = model(images)
            
            # Handle dict outputs (FCN returns OrderedDict with 'out' key)
            if isinstance(outputs, dict):
                outputs = outputs['out']
            
            preds = torch.argmax(outputs, dim=1)
            
            # Save visualizations
            for i in range(preds.shape[0]):
                fname = str(names[i])
                stem = safe_stem(fname)
                
                pred_np = preds[i].cpu().numpy()
                mask_np = masks[i].numpy()
                img_np = denormalize(images[i].cpu(), config.MEAN, config.STD)
                
                # Save prediction mask
                pred_png = mask_to_uint8(pred_np)
                cv2.imwrite(str(pred_dir / f"{stem}.png"), pred_png)
                
                # Save side-by-side comparison
                gt_png = mask_to_uint8(mask_np)
                gt_rgb = cv2.cvtColor(gt_png, cv2.COLOR_GRAY2RGB)
                pred_rgb = cv2.cvtColor(pred_png, cv2.COLOR_GRAY2RGB)
                
                compare = np.concatenate([img_np, gt_rgb, pred_rgb], axis=1)
                cv2.imwrite(str(compare_dir / f"{stem}.png"), cv2.cvtColor(compare, cv2.COLOR_RGB2BGR))
                saved += 1
            
            all_preds.append(preds.cpu().numpy())
            all_targets.append(masks.numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    metrics = compute_all_metrics(all_preds, all_targets)
    print(f"  Saved {saved} visualizations to: {output_dir / model_name}")
    
    return metrics, info


def evaluate_auraseg(checkpoint_path, val_loader, device, output_dir, config):
    """Evaluate AURASeg V4-R50 model and save visualizations."""
    print(f"\n{'='*60}")
    print(f"Evaluating: AURASEG V4-R50")
    print(f"{'='*60}")
    
    # Load model
    model = AURASeg_V4_ResNet50(num_classes=2)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    
    info = {
        'name': 'AURASeg V4-R50',
        'encoder': 'ResNet-50',
        'paradigm': 'ASPP-Lite + APUD + RBRM',
        'params_millions': total_params / 1e6
    }
    
    print(f"Model: {info['name']}")
    print(f"Encoder: {info['encoder']}")
    print(f"Parameters: {info['params_millions']:.2f}M")
    print(f"Best mIoU from training: {checkpoint.get('best_miou', 'N/A'):.4f}")
    
    # Create output directories
    model_name = "auraseg_v4_r50"
    pred_dir = output_dir / model_name / "pred_masks"
    compare_dir = output_dir / model_name / "compare"
    pred_dir.mkdir(parents=True, exist_ok=True)
    compare_dir.mkdir(parents=True, exist_ok=True)
    
    # Run inference
    all_preds = []
    all_targets = []
    saved = 0
    
    with torch.no_grad():
        for images, masks, names in tqdm(val_loader, desc="Evaluating AURASeg V4-R50"):
            images = images.to(device)
            outputs = model(images, return_aux=False, return_boundary=False)
            # Model returns dict with 'main' key
            main_out = outputs['main']
            preds = torch.argmax(main_out, dim=1)
            
            # Save visualizations
            for i in range(preds.shape[0]):
                fname = str(names[i])
                stem = safe_stem(fname)
                
                pred_np = preds[i].cpu().numpy()
                mask_np = masks[i].numpy()
                img_np = denormalize(images[i].cpu(), config.MEAN, config.STD)
                
                # Save prediction mask
                pred_png = mask_to_uint8(pred_np)
                cv2.imwrite(str(pred_dir / f"{stem}.png"), pred_png)
                
                # Save side-by-side comparison
                gt_png = mask_to_uint8(mask_np)
                gt_rgb = cv2.cvtColor(gt_png, cv2.COLOR_GRAY2RGB)
                pred_rgb = cv2.cvtColor(pred_png, cv2.COLOR_GRAY2RGB)
                
                compare = np.concatenate([img_np, gt_rgb, pred_rgb], axis=1)
                cv2.imwrite(str(compare_dir / f"{stem}.png"), cv2.cvtColor(compare, cv2.COLOR_RGB2BGR))
                saved += 1
            
            all_preds.append(preds.cpu().numpy())
            all_targets.append(masks.numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    metrics = compute_all_metrics(all_preds, all_targets)
    print(f"  Saved {saved} visualizations to: {output_dir / model_name}")
    
    return metrics, info


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = Config()
    runs_dir = Path(__file__).parent.parent / "runs"
    
    # Create visualization output directory
    viz_output_dir = runs_dir / "visualization-val"
    viz_output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("FINAL COMPARISON: AURASeg V4-R50 vs Benchmark Models")
    print("="*80)
    print(f"Visualizations will be saved to: {viz_output_dir}")
    
    # Models to evaluate
    benchmark_models = [
        ('deeplabv3plus', runs_dir / 'benchmark_deeplabv3plus' / 'checkpoints' / 'best.pth'),
        ('segformer', runs_dir / 'benchmark_segformer' / 'checkpoints' / 'best.pth'),
        ('mask2former', runs_dir / 'benchmark_mask2former' / 'checkpoints' / 'best.pth'),
        ('fcn', runs_dir / 'benchmark_fcn' / 'checkpoints' / 'best.pth'),
        ('upernet', runs_dir / 'benchmark_upernet' / 'checkpoints' / 'best.pth'),
        ('pspnet', runs_dir / 'benchmark_pspnet' / 'checkpoints' / 'best.pth'),
        ('pidnet', runs_dir / 'benchmark_pidnet' / 'checkpoints' / 'best.pth'),
    ]
    
    auraseg_checkpoint = runs_dir / 'auraseg_v4_resnet50' / 'checkpoints' / 'best.pth'
    
    # Check availability
    print("\nChecking model checkpoints:")
    for name, ckpt_path in benchmark_models:
        status = "[OK]" if ckpt_path.exists() else "[MISSING]"
        print(f"  {status} {name}: {ckpt_path}")
    
    auraseg_status = "[OK]" if auraseg_checkpoint.exists() else "[MISSING]"
    print(f"  {auraseg_status} auraseg_v4_r50: {auraseg_checkpoint}")
    
    # Load validation dataset with paths
    val_dataset = DrivableAreaDatasetWithPaths(
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
    
    # Benchmark models
    for model_name, ckpt_path in benchmark_models:
        if ckpt_path.exists():
            metrics, info = evaluate_benchmark_model(model_name, ckpt_path, val_loader, device, viz_output_dir, config)
            all_results[model_name] = {'metrics': metrics, 'info': info}
    
    # AURASeg V4-R50
    if auraseg_checkpoint.exists():
        metrics, info = evaluate_auraseg(auraseg_checkpoint, val_loader, device, viz_output_dir, config)
        all_results['auraseg_v4_r50'] = {'metrics': metrics, 'info': info}
    
    # Print comparison table
    print("\n")
    print("=" * 120)
    print("COMPLETE BENCHMARK COMPARISON RESULTS")
    print("=" * 120)
    
    model_names = list(all_results.keys())
    
    # Header
    header = f"{'Metric':<25}"
    for name in model_names:
        header += f" | {name:<15}"
    print(header)
    print("-" * 120)
    
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
    
    # Model info
    print("\nMODEL INFO:")
    print(f"    {'params (M)':<21}", end="")
    for name in model_names:
        params = all_results[name]['info']['params_millions']
        print(f" | {params:.2f}M{'':<9}", end="")
    print()
    
    print(f"    {'encoder':<21}", end="")
    for name in model_names:
        encoder = all_results[name]['info']['encoder'][:12]
        print(f" | {encoder:<15}", end="")
    print()
    
    print("=" * 120)
    print("\n* indicates best performance for each metric")
    
    # Save results
    output_path = runs_dir / "final_comparison_results.txt"
    with open(output_path, 'w') as f:
        f.write("FINAL COMPARISON: AURASeg V4-R50 vs Benchmark Models\n")
        f.write("=" * 80 + "\n\n")
        
        for model_name in model_names:
            info = all_results[model_name]['info']
            metrics = all_results[model_name]['metrics']
            
            f.write(f"Model: {info['name']}\n")
            f.write(f"Encoder: {info['encoder']}\n")
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
