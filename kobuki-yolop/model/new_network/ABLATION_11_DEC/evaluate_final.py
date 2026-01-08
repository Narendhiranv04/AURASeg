"""
Final Evaluation Script for V1, V2, V3, V4 Ablation Study
Compares all four model variants with comprehensive metrics.

Models:
    V1: CSPDarknet-53 + SPP + Basic Decoder
    V2: CSPDarknet-53 + ASPP-Lite + Basic Decoder
    V3: CSPDarknet-53 + ASPP-Lite + APUD (Attention) + Deep Supervision
    V4: V3 + RBRM (Residual Boundary Refinement)

Metrics:
    - Segmentation: mIoU, IoU (Drivable), IoU (Background), Dice, Accuracy
    - Classification: Precision, Recall, F1 Score
    - Boundary: Boundary IoU, Boundary F1 (computed on edge pixels only)

Outputs:
    - Full metrics table (txt)
    - Side-by-side visualizations (png)
    - V4 boundary prediction visualizations (png)
"""

import os
import sys
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from pathlib import Path
import torch.nn.functional as F

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import DataConfig, AugmentConfig
from utils import DrivableAreaDataset, compute_metrics, AverageMeter
from models.v1_base_spp import V1BaseSPP
from models.v2_base_assplite import V2BaseASPPLite
from models.v3_apud import V3APUD
from models.v4_rbrm import V4RBRM


def load_model(model_class, checkpoint_path, device, **kwargs):
    """Load model from checkpoint"""
    model = model_class(**kwargs).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"  Loaded: {checkpoint_path}")
    print(f"  Best mIoU: {checkpoint.get('best_miou', 'N/A')}")
    return model


def denormalize(tensor, mean, std):
    """Denormalize image tensor"""
    mean = torch.tensor(mean).view(3, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(3, 1, 1).to(tensor.device)
    return tensor * std + mean


def generate_boundary_gt(segmentation_gt, kernel_size=5):
    """Generate boundary GT using morphological operations"""
    mask = segmentation_gt.float().unsqueeze(1)
    dilated = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    eroded = -F.max_pool2d(-mask, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    boundary = (dilated - eroded).clamp(0, 1)
    return boundary.squeeze(1)


def compute_boundary_metrics(pred, target, kernel_size=5):
    """Compute boundary-specific metrics"""
    # Generate boundary masks
    pred_boundary = generate_boundary_gt(pred.float(), kernel_size)
    target_boundary = generate_boundary_gt(target.float(), kernel_size)
    
    pred_bnd = pred_boundary.cpu().numpy().flatten() > 0.5
    target_bnd = target_boundary.cpu().numpy().flatten() > 0.5
    
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


def compute_extended_metrics(pred, target):
    """Compute extended metrics including F1, Precision, Recall"""
    pred = pred.cpu().numpy().flatten()
    target = target.cpu().numpy().flatten()
    
    # For drivable class (class 1)
    tp = np.sum((pred == 1) & (target == 1))
    fp = np.sum((pred == 1) & (target == 0))
    fn = np.sum((pred == 0) & (target == 1))
    tn = np.sum((pred == 0) & (target == 0))
    
    # Metrics
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    iou_drivable = tp / (tp + fp + fn + 1e-8)
    iou_background = tn / (tn + fp + fn + 1e-8)
    miou = (iou_drivable + iou_background) / 2
    
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


def create_overlay(image, mask, color=[0, 1, 0], alpha=0.4):
    """Create image with mask overlay"""
    overlay = image.copy()
    mask_indices = mask == 1
    for c in range(3):
        overlay[:, :, c][mask_indices] = (
            overlay[:, :, c][mask_indices] * (1 - alpha) + 
            color[c] * alpha
        )
    return overlay


def visualize_all_models(image, gt, preds, save_path):
    """Visualize predictions from all 4 models"""
    img_np = image.permute(1, 2, 0).cpu().numpy()
    img_np = np.clip(img_np, 0, 1)
    
    gt_np = gt.cpu().numpy()
    
    # Create overlays
    overlays = [create_overlay(img_np, gt_np)]
    titles = ['Ground Truth']
    
    model_names = ['V1 (SPP)', 'V2 (ASPP-Lite)', 'V3 (APUD)', 'V4 (RBRM)']
    for pred, name in zip(preds, model_names):
        pred_np = pred.cpu().numpy()
        overlays.append(create_overlay(img_np, pred_np))
        titles.append(name)
    
    # Plot
    fig, axes = plt.subplots(1, 6, figsize=(30, 5))
    
    axes[0].imshow(img_np)
    axes[0].set_title('Input', fontsize=12)
    axes[0].axis('off')
    
    for i, (overlay, title) in enumerate(zip(overlays, titles)):
        axes[i + 1].imshow(overlay)
        axes[i + 1].set_title(title, fontsize=12)
        axes[i + 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def visualize_v4_boundary(image, gt, seg_pred, boundary_pred, boundary_gt, save_path):
    """Visualize V4 boundary predictions"""
    img_np = image.permute(1, 2, 0).cpu().numpy()
    img_np = np.clip(img_np, 0, 1)
    
    gt_np = gt.cpu().numpy()
    seg_np = seg_pred.cpu().numpy()
    bnd_pred_np = boundary_pred.cpu().numpy()
    bnd_gt_np = boundary_gt.cpu().numpy()
    
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    
    axes[0].imshow(img_np)
    axes[0].set_title('Input')
    axes[0].axis('off')
    
    axes[1].imshow(gt_np, cmap='gray')
    axes[1].set_title('GT Segmentation')
    axes[1].axis('off')
    
    axes[2].imshow(seg_np, cmap='gray')
    axes[2].set_title('V4 Prediction')
    axes[2].axis('off')
    
    axes[3].imshow(bnd_gt_np, cmap='hot')
    axes[3].set_title('Boundary GT')
    axes[3].axis('off')
    
    axes[4].imshow(bnd_pred_np, cmap='hot')
    axes[4].set_title('V4 Boundary Pred')
    axes[4].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print("ABLATION STUDY - FINAL EVALUATION: V1 vs V2 vs V3 vs V4")
    print(f"{'='*70}")
    print(f"Device: {device}")
    
    # Paths
    base_dir = 'kobuki-yolop/model/new_network/ABLATION_11_DEC/runs'
    v1_ckpt = os.path.join(base_dir, 'v1_base_spp/checkpoints/best.pth')
    v2_ckpt = os.path.join(base_dir, 'v2_base_assplite/checkpoints/best.pth')
    v3_ckpt = os.path.join(base_dir, 'v3_apud/checkpoints/best.pth')
    v4_ckpt = os.path.join(base_dir, 'v4_rbrm/checkpoints/best.pth')
    
    output_dir = os.path.join(base_dir, 'final_comparison')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'v4_boundary'), exist_ok=True)
    
    # Check which checkpoints exist
    checkpoints = {
        'V1': (v1_ckpt, V1BaseSPP, {}),
        'V2': (v2_ckpt, V2BaseASPPLite, {}),
        'V3': (v3_ckpt, V3APUD, {'in_channels': 3, 'num_classes': 2, 'decoder_channels': 256, 'se_reduction': 16}),
        'V4': (v4_ckpt, V4RBRM, {'in_channels': 3, 'num_classes': 2, 'decoder_channels': 256, 'se_reduction': 16, 'edge_channels': 64})
    }
    
    available_models = {}
    for name, (ckpt, cls, kwargs) in checkpoints.items():
        if os.path.exists(ckpt):
            available_models[name] = (ckpt, cls, kwargs)
        else:
            print(f"  [WARNING] {name} checkpoint not found: {ckpt}")
    
    if not available_models:
        print("No model checkpoints found! Exiting.")
        return
    
    # Load Dataset
    data_config = DataConfig()
    augment_config = AugmentConfig()
    
    val_dataset = DrivableAreaDataset(
        image_dir=data_config.image_dir,
        mask_dir=data_config.mask_dir,
        img_size=data_config.img_size,
        split='val',
        transform=False,
        mean=augment_config.mean,
        std=augment_config.std
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
    print(f"\nValidation samples: {len(val_dataset)}")
    
    # Load models
    print("\n--- Loading Models ---")
    models = {}
    for name, (ckpt, cls, kwargs) in available_models.items():
        print(f"\n{name}:")
        models[name] = load_model(cls, ckpt, device, **kwargs)
    
    # Metrics storage
    seg_keys = ['miou', 'iou_drivable', 'iou_background', 'dice', 'precision', 'recall', 'f1', 'accuracy']
    bnd_keys = ['boundary_iou', 'boundary_precision', 'boundary_recall', 'boundary_f1']
    
    all_metrics = {name: {k: AverageMeter() for k in seg_keys + bnd_keys} for name in models}
    
    print("\n--- Starting Evaluation ---")
    with torch.no_grad():
        for i, (images, masks) in enumerate(tqdm(val_loader, desc="Evaluating")):
            images = images.to(device)
            masks = masks.to(device)
            
            preds = {}
            boundary_pred = None
            
            for name, model in models.items():
                if name in ['V1', 'V2']:
                    out = model(images)
                    pred = torch.argmax(out, dim=1)
                elif name == 'V3':
                    out = model(images, return_aux=False)
                    pred = torch.argmax(out['main'], dim=1)
                elif name == 'V4':
                    out = model(images, return_aux=False, return_boundary=True)
                    pred = torch.argmax(out['main'], dim=1)
                    boundary_pred = out['boundary']
                
                preds[name] = pred
                
                # Compute segmentation metrics
                seg_m = compute_extended_metrics(pred, masks)
                for k, v in seg_m.items():
                    all_metrics[name][k].update(v)
                
                # Compute boundary metrics
                bnd_m = compute_boundary_metrics(pred, masks)
                for k, v in bnd_m.items():
                    all_metrics[name][k].update(v)
            
            # Visualize first 50 samples
            if i < 50:
                img_denorm = denormalize(images[0], val_dataset.mean, val_dataset.std)
                
                # All models comparison
                pred_list = [preds[name][0] for name in ['V1', 'V2', 'V3', 'V4'] if name in preds]
                save_path = os.path.join(output_dir, f"comparison_{i:03d}.png")
                visualize_all_models(img_denorm, masks[0], pred_list, save_path)
                
                # V4 boundary visualization
                if 'V4' in preds and boundary_pred is not None:
                    bnd_gt = generate_boundary_gt(masks)
                    bnd_save = os.path.join(output_dir, 'v4_boundary', f"boundary_{i:03d}.png")
                    visualize_v4_boundary(
                        img_denorm, masks[0], preds['V4'][0],
                        boundary_pred[0, 0], bnd_gt[0], bnd_save
                    )
    
    # Print Results
    print("\n" + "=" * 100)
    print("EVALUATION RESULTS")
    print("=" * 100)
    
    # Header
    model_names = list(models.keys())
    header = f"{'Metric':<22}"
    for name in model_names:
        header += f" | {name:<14}"
    print(header)
    print("-" * 100)
    
    # Segmentation metrics
    print("SEGMENTATION METRICS:")
    for k in seg_keys:
        row = f"  {k:<20}"
        values = [all_metrics[name][k].avg for name in model_names]
        best = max(values)
        for name in model_names:
            val = all_metrics[name][k].avg
            marker = "*" if val == best else " "
            row += f" | {val:.4f}{marker}        "
        print(row)
    
    # Boundary metrics
    print("\nBOUNDARY METRICS:")
    for k in bnd_keys:
        row = f"  {k:<20}"
        values = [all_metrics[name][k].avg for name in model_names]
        best = max(values)
        for name in model_names:
            val = all_metrics[name][k].avg
            marker = "*" if val == best else " "
            row += f" | {val:.4f}{marker}        "
        print(row)
    
    print("=" * 100)
    print("* indicates best performance for each metric\n")
    
    # Compute improvements over V1
    if 'V1' in models:
        print("--- Improvements over V1 (SPP) ---")
        v1_miou = all_metrics['V1']['miou'].avg
        v1_bnd_iou = all_metrics['V1']['boundary_iou'].avg
        
        for name in model_names:
            if name != 'V1':
                miou_delta = (all_metrics[name]['miou'].avg - v1_miou) * 100
                bnd_delta = (all_metrics[name]['boundary_iou'].avg - v1_bnd_iou) * 100
                print(f"  {name}: mIoU {miou_delta:+.2f}%, Boundary IoU {bnd_delta:+.2f}%")
    
    # Save results to file
    results_path = os.path.join(output_dir, 'final_evaluation_results.txt')
    with open(results_path, 'w') as f:
        f.write("ABLATION STUDY - FINAL EVALUATION RESULTS\n")
        f.write("=" * 100 + "\n\n")
        
        f.write("SEGMENTATION METRICS:\n")
        f.write(f"{'Metric':<22}")
        for name in model_names:
            f.write(f" | {name:<14}")
        f.write("\n")
        f.write("-" * 100 + "\n")
        
        for k in seg_keys:
            f.write(f"  {k:<20}")
            for name in model_names:
                f.write(f" | {all_metrics[name][k].avg:.4f}         ")
            f.write("\n")
        
        f.write("\nBOUNDARY METRICS:\n")
        for k in bnd_keys:
            f.write(f"  {k:<20}")
            for name in model_names:
                f.write(f" | {all_metrics[name][k].avg:.4f}         ")
            f.write("\n")
        
        f.write("\n" + "=" * 100 + "\n")
    
    print(f"\nResults saved to: {results_path}")
    print(f"Visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
