"""
Benchmark Model Evaluation Script
==================================

Evaluate all trained benchmark models against AURASeg for comprehensive comparison.

Metrics computed:
    - mIoU (Mean Intersection over Union)
    - IoU (Drivable class)
    - Dice Score
    - Precision / Recall / F1
    - Boundary IoU (edge quality)
    - Inference time (FPS)
    - Model parameters

Usage:
    python evaluate_benchmark.py --output results.csv
    python evaluate_benchmark.py --models deeplabv3plus segformer upernet
    python evaluate_benchmark.py --include-auraseg
"""

import os
import sys
import argparse
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
from torch.utils.data import DataLoader

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "kobuki-yolop" / "model" / "new_network" / "ABLATION_11_DEC"))

from model_factory import get_benchmark_model, BENCHMARK_MODELS
from train_benchmark import DrivableAreaDataset, Config, compute_metrics


# =============================================================================
# Extended Metrics
# =============================================================================

def compute_boundary_iou(pred, target, dilation=2):
    """
    Compute Boundary IoU - measures edge quality.
    
    Args:
        pred: Binary prediction (H, W)
        target: Binary ground truth (H, W)
        dilation: Dilation for boundary extraction
        
    Returns:
        Boundary IoU score
    """
    kernel = np.ones((3, 3), np.uint8)
    
    # Extract boundaries using morphological gradient
    pred_boundary = cv2.morphologyEx(pred, cv2.MORPH_GRADIENT, kernel)
    target_boundary = cv2.morphologyEx(target, cv2.MORPH_GRADIENT, kernel)
    
    # Dilate boundaries
    pred_boundary = cv2.dilate(pred_boundary, kernel, iterations=dilation)
    target_boundary = cv2.dilate(target_boundary, kernel, iterations=dilation)
    
    # Compute IoU on boundaries
    intersection = (pred_boundary & target_boundary).sum()
    union = (pred_boundary | target_boundary).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return intersection / union


def measure_inference_time(model, device, input_size=(384, 640), 
                          num_warmup=10, num_runs=100):
    """
    Measure model inference time.
    
    Args:
        model: PyTorch model
        device: Device to run on
        input_size: (H, W) input dimensions
        num_warmup: Warmup iterations
        num_runs: Benchmark iterations
        
    Returns:
        dict with mean/std latency and FPS
    """
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
# Evaluation Functions
# =============================================================================

def load_model(model_name, checkpoint_path, device, num_classes=2):
    """Load a trained model from checkpoint."""
    model, info = get_benchmark_model(model_name, num_classes=num_classes, pretrained=False)
    
    if checkpoint_path and Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded checkpoint: {checkpoint_path}")
        print(f"  Best mIoU from training: {checkpoint.get('best_miou', 'N/A'):.4f}")
    else:
        print(f"  Warning: No checkpoint found at {checkpoint_path}")
        print(f"  Using untrained model (pretrained backbone only)")
        model, info = get_benchmark_model(model_name, num_classes=num_classes, pretrained=True)
    
    model = model.to(device)
    model.eval()
    return model, info


def evaluate_model(model, model_name, data_loader, device):
    """
    Evaluate a single model on the validation set.
    
    Returns:
        dict with all metrics
    """
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, masks in tqdm(data_loader, desc=f"Evaluating {model_name}", leave=False):
            images = images.to(device)
            masks = masks.to(device)
            
            # Get predictions (all SMP models have consistent interface)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.append(preds.cpu().numpy())
            all_targets.append(masks.cpu().numpy())
    
    # Concatenate all
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Compute standard metrics
    metrics = compute_metrics(all_preds, all_targets, num_classes=2)
    
    # Compute boundary IoU
    boundary_ious = []
    for i in range(len(all_preds)):
        pred_binary = (all_preds[i] == 1).astype(np.uint8)
        target_binary = (all_targets[i] == 1).astype(np.uint8)
        b_iou = compute_boundary_iou(pred_binary, target_binary)
        boundary_ious.append(b_iou)
    
    metrics['boundary_iou'] = np.mean(boundary_ious)
    
    return metrics


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


# =============================================================================
# Visualization
# =============================================================================

def visualize_comparison(images, masks, predictions_dict, save_path):
    """
    Create side-by-side visualization of all model predictions.
    
    Args:
        images: List of input images (denormalized)
        masks: List of ground truth masks
        predictions_dict: Dict mapping model_name -> predictions
        save_path: Path to save visualization
    """
    n_samples = min(4, len(images))
    n_models = len(predictions_dict) + 2  # +2 for input and GT
    
    fig, axes = plt.subplots(n_samples, n_models, figsize=(4*n_models, 4*n_samples))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    model_names = ['Input', 'Ground Truth'] + list(predictions_dict.keys())
    
    for i in range(n_samples):
        # Input image
        axes[i, 0].imshow(images[i])
        axes[i, 0].set_title('Input' if i == 0 else '')
        axes[i, 0].axis('off')
        
        # Ground truth
        axes[i, 1].imshow(masks[i], cmap='gray')
        axes[i, 1].set_title('Ground Truth' if i == 0 else '')
        axes[i, 1].axis('off')
        
        # Model predictions
        for j, (name, preds) in enumerate(predictions_dict.items()):
            axes[i, j+2].imshow(preds[i], cmap='gray')
            axes[i, j+2].set_title(name if i == 0 else '')
            axes[i, j+2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_comparison_table(results_df, save_path):
    """Create a formatted comparison table."""
    # Sort by mIoU
    results_df = results_df.sort_values('mIoU', ascending=False)
    
    # Format for display
    display_df = results_df[['Model', 'Encoder', 'Params (M)', 
                             'mIoU', 'IoU (Drivable)', 'Dice', 
                             'Boundary IoU', 'FPS']].copy()
    
    # Format percentages
    for col in ['mIoU', 'IoU (Drivable)', 'Dice', 'Boundary IoU']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
    
    # Format FPS
    display_df['FPS'] = display_df['FPS'].apply(lambda x: f"{x:.1f}")
    
    # Save to CSV
    results_df.to_csv(save_path, index=False)
    print(f"\nResults saved to: {save_path}")
    
    # Print table
    print("\n" + "=" * 100)
    print("BENCHMARK COMPARISON RESULTS")
    print("=" * 100)
    print(display_df.to_string(index=False))
    print("=" * 100)
    
    return display_df


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Evaluate Benchmark Models')
    
    parser.add_argument('--models', type=str, nargs='+', 
                        default=['deeplabv3plus', 'segformer', 'upernet', 'dpt', 'mask2former'],
                        help='Models to evaluate')
    parser.add_argument('--include-auraseg', action='store_true',
                        help='Include AURASeg V4 in comparison')
    parser.add_argument('--output', type=str, default='benchmark_results.csv',
                        help='Output CSV file path')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for evaluation')
    parser.add_argument('--runs-dir', type=str, default=None,
                        help='Directory containing trained model runs')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate comparison visualizations')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    config = Config()
    
    if args.runs_dir:
        runs_dir = Path(args.runs_dir)
    else:
        runs_dir = Path(__file__).parent.parent / "runs"
    
    print(f"Device: {device}")
    print(f"Runs directory: {runs_dir}")
    print(f"Models to evaluate: {args.models}")
    
    # Setup validation dataset
    val_dataset = DrivableAreaDataset(
        image_dir=config.IMAGE_DIR,
        mask_dir=config.MASK_DIR,
        img_size=config.IMG_SIZE,
        split='val',
        transform=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    # Results storage
    results = []
    
    # Evaluate each model
    for model_name in args.models:
        print(f"\n{'=' * 60}")
        print(f"Evaluating: {model_name.upper()}")
        print(f"{'=' * 60}")
        
        # Find checkpoint
        checkpoint_path = find_checkpoint(model_name, runs_dir)
        
        # Load model
        model, info = load_model(model_name, checkpoint_path, device)
        
        # Evaluate
        metrics = evaluate_model(model, model_name, val_loader, device)
        
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
            'Dice': metrics['mdice'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1': metrics['f1'],
            'Boundary IoU': metrics['boundary_iou'],
            'Latency (ms)': timing['latency_mean_ms'],
            'FPS': timing['fps']
        }
        results.append(result)
        
        # Print summary
        print(f"  mIoU: {metrics['miou']:.4f}")
        print(f"  IoU (Drivable): {metrics['iou_drivable']:.4f}")
        print(f"  Dice: {metrics['mdice']:.4f}")
        print(f"  Boundary IoU: {metrics['boundary_iou']:.4f}")
        print(f"  FPS: {timing['fps']:.1f}")
        
        # Cleanup
        del model
        torch.cuda.empty_cache()
    
    # Include AURASeg if requested
    if args.include_auraseg:
        print(f"\n{'=' * 60}")
        print("Including AURASeg V4...")
        print(f"{'=' * 60}")
        
        # Load AURASeg V4
        try:
            from models.v4_rbrm import V4RBRM
            
            auraseg_checkpoint = runs_dir / "ablation_v4" / "v4_rbrm" / "checkpoints" / "best.pth"
            
            if not auraseg_checkpoint.exists():
                # Try alternative locations
                alt_locations = [
                    runs_dir.parent / "kobuki-yolop" / "model" / "new_network" / "ABLATION_11_DEC" / "runs" / "ablation_v4" / "v4_rbrm" / "checkpoints" / "best.pth",
                    Path(__file__).parent.parent / "kobuki-yolop" / "model" / "new_network" / "ABLATION_11_DEC" / "runs" / "ablation_v4" / "v4_rbrm" / "checkpoints" / "best.pth",
                ]
                for alt in alt_locations:
                    if alt.exists():
                        auraseg_checkpoint = alt
                        break
            
            if auraseg_checkpoint.exists():
                model = V4RBRM(in_channels=3, num_classes=2).to(device)
                checkpoint = torch.load(auraseg_checkpoint, map_location=device, weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                
                print(f"  Loaded: {auraseg_checkpoint}")
                
                # Evaluate
                metrics = evaluate_model(model, 'auraseg_v4', val_loader, device)
                timing = measure_inference_time(model, device, config.IMG_SIZE)
                
                # Count params
                total_params = sum(p.numel() for p in model.parameters())
                
                result = {
                    'Model': 'AURASeg V4',
                    'Encoder': 'CSPDarknet-53',
                    'Paradigm': 'ASPP + APUD + RBRM',
                    'Params (M)': total_params / 1e6,
                    'mIoU': metrics['miou'],
                    'IoU (Drivable)': metrics['iou_drivable'],
                    'Dice': metrics['mdice'],
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'F1': metrics['f1'],
                    'Boundary IoU': metrics['boundary_iou'],
                    'Latency (ms)': timing['latency_mean_ms'],
                    'FPS': timing['fps']
                }
                results.append(result)
                
                print(f"  mIoU: {metrics['miou']:.4f}")
                print(f"  Boundary IoU: {metrics['boundary_iou']:.4f}")
                print(f"  FPS: {timing['fps']:.1f}")
                
            else:
                print("  AURASeg V4 checkpoint not found!")
                
        except Exception as e:
            print(f"  Error loading AURASeg: {e}")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save and display results
    output_path = Path(__file__).parent.parent / args.output
    create_comparison_table(results_df, output_path)
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
