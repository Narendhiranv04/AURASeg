"""
Utilities for Ablation Study
============================
Helper functions for metrics, visualization, and evaluation.
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ============================================================================
# Metrics Computation
# ============================================================================

def compute_iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 2, 
               ignore_index: int = 255) -> Dict[str, float]:
    """
    Compute IoU for each class and mean IoU.
    
    Args:
        pred: Predictions (B, H, W) or (B, C, H, W)
        target: Ground truth (B, H, W)
        num_classes: Number of classes
        ignore_index: Index to ignore
    
    Returns:
        Dictionary with IoU values
    """
    if pred.dim() == 4:
        pred = pred.argmax(dim=1)
    
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    
    ious = {}
    for c in range(num_classes):
        pred_c = pred == c
        target_c = target == c
        valid = target != ignore_index
        
        intersection = ((pred_c & target_c) & valid).sum()
        union = ((pred_c | target_c) & valid).sum()
        
        iou = intersection / (union + 1e-10)
        ious[f'iou_class{c}'] = float(iou)
    
    ious['miou'] = np.mean([ious[f'iou_class{c}'] for c in range(num_classes)])
    
    return ious


def compute_dice(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 2,
                ignore_index: int = 255) -> Dict[str, float]:
    """
    Compute Dice score for each class and mean Dice.
    
    Args:
        pred: Predictions (B, H, W) or (B, C, H, W)
        target: Ground truth (B, H, W)
        num_classes: Number of classes
        ignore_index: Index to ignore
    
    Returns:
        Dictionary with Dice values
    """
    if pred.dim() == 4:
        pred = pred.argmax(dim=1)
    
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    
    dices = {}
    for c in range(num_classes):
        pred_c = pred == c
        target_c = target == c
        valid = target != ignore_index
        
        intersection = ((pred_c & target_c) & valid).sum()
        total = (pred_c & valid).sum() + (target_c & valid).sum()
        
        dice = 2 * intersection / (total + 1e-10)
        dices[f'dice_class{c}'] = float(dice)
    
    dices['mdice'] = np.mean([dices[f'dice_class{c}'] for c in range(num_classes)])
    
    return dices


def compute_boundary_metrics(pred: torch.Tensor, target: torch.Tensor,
                           threshold: int = 3) -> Dict[str, float]:
    """
    Compute boundary-aware metrics.
    
    Args:
        pred: Predictions (B, H, W)
        target: Ground truth (B, H, W)
        threshold: Distance threshold for boundary matching
    
    Returns:
        Dictionary with boundary metrics
    """
    from scipy.ndimage import distance_transform_edt, sobel
    
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    
    boundary_ious = []
    boundary_f1s = []
    
    for i in range(pred.shape[0]):
        # Extract boundaries using Sobel
        pred_boundary = np.abs(sobel(pred[i].astype(float))) > 0
        target_boundary = np.abs(sobel(target[i].astype(float))) > 0
        
        # Compute distance transform
        pred_dist = distance_transform_edt(~pred_boundary)
        target_dist = distance_transform_edt(~target_boundary)
        
        # Boundary recall: target boundaries within threshold of pred
        recall = (target_dist[pred_boundary] <= threshold).mean() if pred_boundary.any() else 0
        
        # Boundary precision: pred boundaries within threshold of target
        precision = (pred_dist[target_boundary] <= threshold).mean() if target_boundary.any() else 0
        
        # F1
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        
        # Boundary IoU
        intersection = (pred_boundary & target_boundary).sum()
        union = (pred_boundary | target_boundary).sum()
        biou = intersection / (union + 1e-10)
        
        boundary_ious.append(biou)
        boundary_f1s.append(f1)
    
    return {
        'boundary_iou': np.mean(boundary_ious),
        'boundary_f1': np.mean(boundary_f1s)
    }


# ============================================================================
# Visualization
# ============================================================================

def visualize_prediction(
    image: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    save_path: Optional[str] = None,
    class_names: Optional[List[str]] = None,
    alpha: float = 0.5
) -> Optional[np.ndarray]:
    """
    Visualize prediction overlay on image.
    
    Args:
        image: RGB image (H, W, 3) in range [0, 255]
        gt: Ground truth mask (H, W)
        pred: Prediction mask (H, W)
        save_path: Path to save visualization
        class_names: Names for each class
        alpha: Overlay transparency
    
    Returns:
        Visualization as numpy array if save_path is None
    """
    if class_names is None:
        class_names = ['Background', 'Drivable']
    
    # Color map for classes
    colors = np.array([
        [0, 0, 0],       # Background - black
        [0, 255, 0],     # Drivable - green
        [255, 0, 0],     # Class 2 - red
        [0, 0, 255],     # Class 3 - blue
    ])
    
    # Create colored overlays
    gt_colored = colors[gt.astype(int) % len(colors)]
    pred_colored = colors[pred.astype(int) % len(colors)]
    
    # Overlay on image
    image_float = image.astype(float)
    gt_overlay = (1 - alpha) * image_float + alpha * gt_colored
    pred_overlay = (1 - alpha) * image_float + alpha * pred_colored
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Input Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(gt, cmap='gray')
    axes[0, 1].set_title('Ground Truth')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(pred, cmap='gray')
    axes[1, 0].set_title('Prediction')
    axes[1, 0].axis('off')
    
    # Difference map
    diff = (gt != pred).astype(float)
    axes[1, 1].imshow(diff, cmap='Reds')
    axes[1, 1].set_title('Error Map')
    axes[1, 1].axis('off')
    
    # Add legend
    patches = [mpatches.Patch(color=colors[i]/255, label=name) 
               for i, name in enumerate(class_names)]
    fig.legend(handles=patches, loc='lower center', ncol=len(class_names))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return None
    else:
        fig.canvas.draw()
        result = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        result = result.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()
        return result


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: str,
    title: str = "Training History"
):
    """
    Plot comprehensive training curves.
    
    Args:
        history: Dictionary with training history
        save_path: Path to save plot
        title: Plot title
    """
    epochs = range(1, len(history.get('train_loss', [])) + 1)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(title, fontsize=14)
    
    # Loss
    if 'train_loss' in history:
        axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
        axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # mIoU
    if 'train_miou' in history:
        axes[0, 1].plot(epochs, history['train_miou'], 'b-', label='Train', linewidth=2)
        axes[0, 1].plot(epochs, history['val_miou'], 'r-', label='Val', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('mIoU')
        axes[0, 1].set_title('Mean IoU')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add best marker
        best_idx = np.argmax(history['val_miou'])
        best_val = history['val_miou'][best_idx]
        axes[0, 1].scatter([best_idx + 1], [best_val], color='gold', s=100, 
                          zorder=5, marker='*', label=f'Best: {best_val:.4f}')
    
    # Dice
    if 'train_dice' in history:
        axes[0, 2].plot(epochs, history['train_dice'], 'b-', label='Train', linewidth=2)
        axes[0, 2].plot(epochs, history['val_dice'], 'r-', label='Val', linewidth=2)
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('mDice')
        axes[0, 2].set_title('Mean Dice Score')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
    
    # Learning rate
    if 'lr' in history:
        axes[1, 0].plot(epochs, history['lr'], 'g-', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Per-class IoU
    if 'train_iou_class1' in history:
        axes[1, 1].plot(epochs, history.get('train_iou_class1', []), 'b-', 
                       label='Train (Drivable)', linewidth=2)
        axes[1, 1].plot(epochs, history.get('val_iou_class1', []), 'r-', 
                       label='Val (Drivable)', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('IoU')
        axes[1, 1].set_title('Drivable Area IoU')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    # Gradient of loss (convergence indicator)
    if 'train_loss' in history and len(history['train_loss']) > 1:
        loss_grad = np.gradient(history['train_loss'])
        axes[1, 2].plot(epochs, loss_grad, 'purple', linewidth=2, alpha=0.7)
        axes[1, 2].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('dLoss/dEpoch')
        axes[1, 2].set_title('Loss Gradient (Convergence)')
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_ablation_comparison(
    results: Dict[str, Dict[str, float]],
    save_path: str,
    metric: str = 'miou'
):
    """
    Plot comparison across ablation variants.
    
    Args:
        results: Dictionary mapping model name to metrics
        save_path: Path to save plot
        metric: Metric to compare
    """
    models = list(results.keys())
    values = [results[m].get(metric, 0) for m in models]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(models, values, color=['#3498db', '#e74c3c', '#2ecc71', '#9b59b6'])
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{val:.4f}', ha='center', va='bottom', fontsize=11)
    
    ax.set_xlabel('Model Variant')
    ax.set_ylabel(metric.upper())
    ax.set_title(f'Ablation Study Comparison - {metric.upper()}')
    ax.set_ylim(0, max(values) * 1.15)
    
    # Add improvement annotations
    for i in range(1, len(values)):
        improvement = (values[i] - values[0]) / values[0] * 100
        ax.annotate(f'+{improvement:.1f}%', 
                   xy=(i, values[i]), 
                   xytext=(i, values[i] + 0.05),
                   ha='center', fontsize=9, color='green' if improvement > 0 else 'red')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# Model Analysis
# ============================================================================

def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary with parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Per-module breakdown
    breakdown = {}
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        breakdown[name] = params
    
    return {
        'total': total,
        'trainable': trainable,
        'non_trainable': total - trainable,
        'total_M': total / 1e6,
        'breakdown': breakdown
    }


def compute_flops(model: torch.nn.Module, input_size: Tuple[int, int, int, int]) -> Dict[str, float]:
    """
    Estimate FLOPs for the model.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (B, C, H, W)
    
    Returns:
        Dictionary with FLOPs estimation
    """
    try:
        from thop import profile
        
        device = next(model.parameters()).device
        input_tensor = torch.randn(input_size).to(device)
        
        flops, params = profile(model, inputs=(input_tensor,), verbose=False)
        
        return {
            'flops': flops,
            'flops_G': flops / 1e9,
            'params': params,
            'params_M': params / 1e6
        }
    except ImportError:
        print("Warning: thop not installed. Install with: pip install thop")
        return {'flops': 0, 'flops_G': 0}


def measure_inference_time(
    model: torch.nn.Module,
    input_size: Tuple[int, int, int, int],
    num_runs: int = 100,
    warmup: int = 10
) -> Dict[str, float]:
    """
    Measure inference time.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size
        num_runs: Number of inference runs
        warmup: Number of warmup runs
    
    Returns:
        Dictionary with timing statistics
    """
    device = next(model.parameters()).device
    model.eval()
    
    input_tensor = torch.randn(input_size).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)
    
    # Synchronize if CUDA
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Measure
    import time
    times = []
    
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(input_tensor)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            times.append(time.perf_counter() - start)
    
    times = np.array(times) * 1000  # Convert to ms
    
    return {
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'min_ms': float(np.min(times)),
        'max_ms': float(np.max(times)),
        'fps': 1000.0 / np.mean(times)
    }


# ============================================================================
# Results Management
# ============================================================================

def save_results(results: Dict, path: str):
    """Save results to JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)


def load_results(path: str) -> Dict:
    """Load results from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def aggregate_ablation_results(
    output_dirs: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate results from multiple ablation experiments.
    
    Args:
        output_dirs: List of output directories for each variant
    
    Returns:
        Dictionary mapping model name to metrics
    """
    results = {}
    
    for output_dir in output_dirs:
        history_path = os.path.join(output_dir, 'training_history.json')
        config_path = os.path.join(output_dir, 'config.json')
        
        if os.path.exists(history_path) and os.path.exists(config_path):
            history = load_results(history_path)
            config = load_results(config_path)
            
            model_name = config.get('model', {}).get('model_name', os.path.basename(output_dir))
            
            # Get best metrics
            best_idx = np.argmax(history.get('val_miou', [0]))
            
            results[model_name] = {
                'miou': history['val_miou'][best_idx] if history.get('val_miou') else 0,
                'mdice': history['val_dice'][best_idx] if history.get('val_dice') else 0,
                'best_epoch': best_idx + 1,
                'final_loss': history['val_loss'][-1] if history.get('val_loss') else 0
            }
    
    return results


if __name__ == '__main__':
    # Test utilities
    print("Testing utilities...")
    
    # Test metrics
    pred = torch.randint(0, 2, (4, 256, 256))
    target = torch.randint(0, 2, (4, 256, 256))
    
    iou = compute_iou(pred, target)
    print(f"IoU: {iou}")
    
    dice = compute_dice(pred, target)
    print(f"Dice: {dice}")
    
    print("\nUtilities test complete!")
