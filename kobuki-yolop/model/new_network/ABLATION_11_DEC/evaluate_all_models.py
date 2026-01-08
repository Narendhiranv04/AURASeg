"""
Comprehensive Evaluation Script for V1, V2, V3 Ablation Study
Compares: V1 (SPP) vs V2 (ASPP-Lite) vs V3 (APUD + Deep Supervision)

Outputs:
    - Full metrics table (mIoU, Dice, Precision, Recall, F1)
    - Side-by-side visualizations
    - Per-model auxiliary output visualizations (V3 only)
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

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import DataConfig, AugmentConfig
from utils import DrivableAreaDataset, compute_metrics, AverageMeter
from models.v1_base_spp import V1BaseSPP
from models.v2_base_assplite import V2BaseASPPLite
from models.v3_apud import V3APUD


def load_model(model_class, checkpoint_path, device, **kwargs):
    """Load model from checkpoint"""
    model = model_class(**kwargs).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"  Loaded checkpoint: {checkpoint_path}")
    print(f"  Best mIoU from training: {checkpoint.get('best_miou', 'N/A')}")
    return model


def denormalize(tensor, mean, std):
    """Denormalize image tensor"""
    mean = torch.tensor(mean).view(3, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(3, 1, 1).to(tensor.device)
    return tensor * std + mean


def colorize_mask(mask, color=[0, 1, 0]):
    """Create colored mask overlay"""
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3))
    color_mask[mask == 1] = color
    return color_mask


def create_overlay(image, mask, color=[0, 1, 0], alpha=0.4):
    """Create image with mask overlay"""
    overlay = image.copy()
    color_mask = colorize_mask(mask, color)
    mask_indices = mask == 1
    overlay[mask_indices] = overlay[mask_indices] * (1 - alpha) + color_mask[mask_indices] * alpha
    return overlay


def visualize_comparison_v1v2v3(image, gt, pred_v1, pred_v2, pred_v3, save_path):
    """Visualize V1, V2, V3 predictions side-by-side"""
    # Convert to numpy
    img_np = image.permute(1, 2, 0).cpu().numpy()
    img_np = np.clip(img_np, 0, 1)
    
    gt_np = gt.cpu().numpy()
    v1_np = pred_v1.cpu().numpy()
    v2_np = pred_v2.cpu().numpy()
    v3_np = pred_v3.cpu().numpy()
    
    # Create overlays
    gt_overlay = create_overlay(img_np, gt_np, [0, 1, 0])  # Green
    v1_overlay = create_overlay(img_np, v1_np, [0, 0.8, 0.2])  # Green
    v2_overlay = create_overlay(img_np, v2_np, [0, 0.8, 0.2])  # Green
    v3_overlay = create_overlay(img_np, v3_np, [0, 0.8, 0.2])  # Green
    
    # Plot
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    
    axes[0].imshow(img_np)
    axes[0].set_title("Input Image", fontsize=12)
    axes[0].axis('off')
    
    axes[1].imshow(gt_overlay)
    axes[1].set_title("Ground Truth", fontsize=12)
    axes[1].axis('off')
    
    axes[2].imshow(v1_overlay)
    axes[2].set_title("V1 (SPP)", fontsize=12)
    axes[2].axis('off')
    
    axes[3].imshow(v2_overlay)
    axes[3].set_title("V2 (ASPP-Lite)", fontsize=12)
    axes[3].axis('off')
    
    axes[4].imshow(v3_overlay)
    axes[4].set_title("V3 (APUD)", fontsize=12)
    axes[4].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def visualize_v3_aux_outputs(image, gt, main_pred, aux_preds, save_path):
    """Visualize V3 main and auxiliary outputs"""
    img_np = image.permute(1, 2, 0).cpu().numpy()
    img_np = np.clip(img_np, 0, 1)
    
    gt_np = gt.cpu().numpy()
    main_np = main_pred.cpu().numpy()
    
    # Plot: Input | GT | Main | Aux1 | Aux2 | Aux3 | Aux4
    fig, axes = plt.subplots(1, 7, figsize=(28, 4))
    
    axes[0].imshow(img_np)
    axes[0].set_title('Input')
    axes[0].axis('off')
    
    axes[1].imshow(gt_np, cmap='gray')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    axes[2].imshow(main_np, cmap='gray')
    axes[2].set_title('Main (Full Res)')
    axes[2].axis('off')
    
    scales = ['H/32', 'H/16', 'H/8', 'H/4']
    for i, (aux_pred, scale) in enumerate(zip(aux_preds, scales)):
        aux_np = aux_pred.cpu().numpy()
        axes[3 + i].imshow(aux_np, cmap='gray')
        axes[3 + i].set_title(f'Aux-{i+1} ({scale})')
        axes[3 + i].axis('off')
    
    plt.suptitle('V3 APUD - Deep Supervision Outputs', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


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
    
    # IoU
    iou_drivable = tp / (tp + fp + fn + 1e-8)
    iou_background = tn / (tn + fp + fn + 1e-8)
    miou = (iou_drivable + iou_background) / 2
    
    # Dice
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


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print("ABLATION STUDY - V1 vs V2 vs V3 EVALUATION")
    print(f"{'='*60}")
    print(f"Device: {device}")
    
    # Paths
    base_dir = 'kobuki-yolop/model/new_network/ABLATION_11_DEC/runs'
    v1_ckpt = os.path.join(base_dir, 'v1_base_spp/checkpoints/best.pth')
    v2_ckpt = os.path.join(base_dir, 'v2_base_assplite/checkpoints/best.pth')
    v3_ckpt = os.path.join(base_dir, 'v3_apud/checkpoints/best.pth')
    
    output_dir = os.path.join(base_dir, 'comparison_v1v2v3')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'v3_aux'), exist_ok=True)
    
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
    
    # Load Models
    print("\n--- Loading Models ---")
    print("V1 (SPP):")
    model_v1 = load_model(V1BaseSPP, v1_ckpt, device)
    
    print("\nV2 (ASPP-Lite):")
    model_v2 = load_model(V2BaseASPPLite, v2_ckpt, device)
    
    print("\nV3 (APUD + Deep Supervision):")
    model_v3 = load_model(V3APUD, v3_ckpt, device, 
                          in_channels=3, num_classes=2, 
                          decoder_channels=256, se_reduction=16)
    
    # Metrics storage
    metric_keys = ['miou', 'iou_drivable', 'iou_background', 'dice', 'precision', 'recall', 'f1', 'accuracy']
    metrics_v1 = {k: AverageMeter() for k in metric_keys}
    metrics_v2 = {k: AverageMeter() for k in metric_keys}
    metrics_v3 = {k: AverageMeter() for k in metric_keys}
    
    print("\n--- Starting Evaluation ---")
    with torch.no_grad():
        for i, (images, masks) in enumerate(tqdm(val_loader, desc="Evaluating")):
            images = images.to(device)
            masks = masks.to(device)
            
            # V1 and V2 inference
            out_v1 = model_v1(images)
            out_v2 = model_v2(images)
            pred_v1 = torch.argmax(out_v1, dim=1)
            pred_v2 = torch.argmax(out_v2, dim=1)
            
            # V3 inference with auxiliary outputs
            outputs_v3 = model_v3(images, return_aux=True)
            pred_v3 = torch.argmax(outputs_v3['main'], dim=1)
            aux_preds_v3 = [torch.argmax(aux, dim=1) for aux in outputs_v3['aux']]
            
            # Compute metrics
            m_v1 = compute_extended_metrics(pred_v1, masks)
            m_v2 = compute_extended_metrics(pred_v2, masks)
            m_v3 = compute_extended_metrics(pred_v3, masks)
            
            for k in metric_keys:
                metrics_v1[k].update(m_v1[k])
                metrics_v2[k].update(m_v2[k])
                metrics_v3[k].update(m_v3[k])
            
            # Visualize first 50 samples
            if i < 50:
                img_denorm = denormalize(images[0], val_dataset.mean, val_dataset.std)
                
                # Main comparison
                save_path = os.path.join(output_dir, f"comparison_{i:03d}.png")
                visualize_comparison_v1v2v3(img_denorm, masks[0], pred_v1[0], pred_v2[0], pred_v3[0], save_path)
                
                # V3 auxiliary outputs
                aux_save_path = os.path.join(output_dir, 'v3_aux', f"v3_aux_{i:03d}.png")
                visualize_v3_aux_outputs(img_denorm, masks[0], pred_v3[0], 
                                         [aux[0] for aux in aux_preds_v3], aux_save_path)
    
    # Print Results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"{'Metric':<18} | {'V1 (SPP)':<14} | {'V2 (ASPP-Lite)':<14} | {'V3 (APUD)':<14}")
    print("-" * 80)
    
    for k in metric_keys:
        v1_val = metrics_v1[k].avg
        v2_val = metrics_v2[k].avg
        v3_val = metrics_v3[k].avg
        
        # Highlight best
        best = max(v1_val, v2_val, v3_val)
        v1_str = f"{v1_val:.4f}" + ("*" if v1_val == best else " ")
        v2_str = f"{v2_val:.4f}" + ("*" if v2_val == best else " ")
        v3_str = f"{v3_val:.4f}" + ("*" if v3_val == best else " ")
        
        print(f"{k:<18} | {v1_str:<14} | {v2_str:<14} | {v3_str:<14}")
    
    print("="*80)
    print("* indicates best performance for each metric")
    
    # Compute improvements
    print("\n--- Improvements over V1 (SPP) ---")
    for k in ['miou', 'dice', 'f1']:
        v1_val = metrics_v1[k].avg
        v2_val = metrics_v2[k].avg
        v3_val = metrics_v3[k].avg
        v2_delta = (v2_val - v1_val) * 100
        v3_delta = (v3_val - v1_val) * 100
        print(f"{k}: V2 {v2_delta:+.2f}%, V3 {v3_delta:+.2f}%")
    
    # Save results to file
    results_path = os.path.join(output_dir, 'evaluation_results.txt')
    with open(results_path, 'w') as f:
        f.write("ABLATION STUDY - V1 vs V2 vs V3 EVALUATION\n")
        f.write("="*80 + "\n\n")
        f.write(f"{'Metric':<18} | {'V1 (SPP)':<14} | {'V2 (ASPP-Lite)':<14} | {'V3 (APUD)':<14}\n")
        f.write("-" * 80 + "\n")
        
        for k in metric_keys:
            f.write(f"{k:<18} | {metrics_v1[k].avg:.4f}         | {metrics_v2[k].avg:.4f}         | {metrics_v3[k].avg:.4f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("\nImprovements over V1 (SPP):\n")
        for k in ['miou', 'dice', 'f1']:
            v1_val = metrics_v1[k].avg
            v2_val = metrics_v2[k].avg
            v3_val = metrics_v3[k].avg
            v2_delta = (v2_val - v1_val) * 100
            v3_delta = (v3_val - v1_val) * 100
            f.write(f"{k}: V2 {v2_delta:+.2f}%, V3 {v3_delta:+.2f}%\n")
    
    print(f"\nResults saved to: {results_path}")
    print(f"Visualizations saved to: {output_dir}")
    print(f"V3 auxiliary outputs saved to: {os.path.join(output_dir, 'v3_aux')}")


if __name__ == "__main__":
    main()
