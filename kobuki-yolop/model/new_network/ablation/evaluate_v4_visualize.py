import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ablation_v3_apud import AblationV3APUD
from ablation_v4_rbrm import AblationV4RBRM
from train_ablation import SegmentationDataset

def evaluate_and_visualize(
    dataset_dir,
    v4_checkpoint_path,
    v3_checkpoint_path=None,
    output_dir='./evaluation_results',
    num_samples=20,
    device='cuda'
):
    os.makedirs(output_dir, exist_ok=True)
    
    # Device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data Transforms
    transform = transforms.Compose([
        transforms.Resize((480, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    target_transform = transforms.Compose([
        transforms.Resize((480, 640), interpolation=Image.NEAREST),
        transforms.ToTensor()
    ])
    
    # Dataset
    image_dir = os.path.join(dataset_dir, 'images', 'val')
    mask_dir = os.path.join(dataset_dir, 'labels', 'val')
    
    # Load Validation Dataset directly
    val_dataset = SegmentationDataset(
        image_dir, mask_dir, 
        transform=transform, 
        target_transform=target_transform,
        use_augmentation=False
    )
    
    print(f"Found {len(val_dataset)} validation samples.")
    
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=2)
    
    # Load V4 Model
    print(f"Loading V4 Model from {v4_checkpoint_path}...")
    model_v4 = AblationV4RBRM(num_classes=2).to(device)
    checkpoint_v4 = torch.load(v4_checkpoint_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint_v4:
        model_v4.load_state_dict(checkpoint_v4['model_state_dict'])
    else:
        model_v4.load_state_dict(checkpoint_v4)
    model_v4.eval()
    
    # Load V3 Model (if provided)
    model_v3 = None
    if v3_checkpoint_path and os.path.exists(v3_checkpoint_path):
        print(f"Loading V3 Model from {v3_checkpoint_path}...")
        model_v3 = AblationV3APUD(num_classes=2).to(device)
        checkpoint_v3 = torch.load(v3_checkpoint_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint_v3:
            model_v3.load_state_dict(checkpoint_v3['model_state_dict'])
        else:
            model_v3.load_state_dict(checkpoint_v3)
        model_v3.eval()
    
    # Evaluation Loop
    print("Starting evaluation...")
    
    metrics_v4 = {'iou': [], 'dice': [], 'precision': [], 'recall': [], 'accuracy': [], 'boundary_iou': []}
    metrics_v3 = {'iou': [], 'dice': [], 'precision': [], 'recall': [], 'accuracy': [], 'boundary_iou': []}
    
    with torch.no_grad():
        for i, (image, target) in enumerate(tqdm(val_loader)):
            image = image.to(device)
            target = target.to(device)
            
            # V4 Inference
            output_v4 = model_v4(image)
            if isinstance(output_v4, tuple):
                output_v4 = output_v4[0] # Handle (out, edge_logits)
            pred_v4 = torch.argmax(output_v4, dim=1)
            
            # V3 Inference
            pred_v3 = None
            if model_v3:
                output_v3 = model_v3(image)
                pred_v3 = torch.argmax(output_v3, dim=1)
            
            # Calculate Metrics for V4
            m_v4 = calculate_metrics(pred_v4, target)
            for k, v in m_v4.items():
                metrics_v4[k].append(v)
            
            # Calculate Metrics for V3
            if model_v3:
                m_v3 = calculate_metrics(pred_v3, target)
                for k, v in m_v3.items():
                    metrics_v3[k].append(v)
            
            # Save visualizations for the first N samples
            if i < num_samples:
                save_comparison(
                    image, target, pred_v4, pred_v3, 
                    os.path.join(output_dir, f'sample_{i}.png'),
                    m_v4['iou'], m_v3['iou'] if model_v3 else None
                )
    
    # Print Results
    print(f"\n{'='*60}")
    print(f"{'Metric':<15} | {'V3 Baseline':<15} | {'V4 DGF':<15} | {'Improvement':<15}")
    print(f"{'-'*60}")
    
    keys = ['iou', 'dice', 'precision', 'recall', 'accuracy', 'boundary_iou']
    for k in keys:
        v4_val = np.mean(metrics_v4[k])
        if model_v3:
            v3_val = np.mean(metrics_v3[k])
            imp = v4_val - v3_val
            print(f"{k.upper():<15} | {v3_val:.4f}          | {v4_val:.4f}          | {imp:+.4f}")
        else:
            print(f"{k.upper():<15} | {'N/A':<15} | {v4_val:.4f}          | {'N/A':<15}")
    print(f"{'='*60}")

def calculate_boundary_iou(pred, target, dilation_ratio=0.02):
    """
    Calculate Boundary IoU.
    1. Generate boundary mask from GT (dilation - erosion)
    2. Calculate IoU only within the boundary region
    """
    # Convert to float for pooling
    gt_mask = target.float().unsqueeze(0).unsqueeze(0) # [1, 1, H, W]
    pred_mask = pred.float().unsqueeze(0).unsqueeze(0) # [1, 1, H, W]
    
    # Determine kernel size based on image size
    h, w = target.shape
    diag = np.sqrt(h**2 + w**2)
    kernel_size = int(diag * dilation_ratio)
    if kernel_size % 2 == 0: kernel_size += 1
    
    # Generate boundary region
    dilated = F.max_pool2d(gt_mask, kernel_size, stride=1, padding=kernel_size//2)
    eroded = -F.max_pool2d(-gt_mask, kernel_size, stride=1, padding=kernel_size//2)
    boundary_region = (dilated - eroded).abs() > 0.5
    
    # Calculate IoU in boundary region
    inter = (pred_mask * gt_mask * boundary_region).sum().item()
    union = ((pred_mask + gt_mask) * boundary_region).clamp(0, 1).sum().item()
    
    if union == 0:
        return 1.0
    return inter / union

def calculate_metrics(pred, target):
    """Calculate IoU, Dice, Precision, Recall, Accuracy for binary segmentation"""
    # Calculate Boundary IoU first (needs 2D shape)
    boundary_iou = calculate_boundary_iou(pred, target)
    
    pred = pred.view(-1)
    target = target.view(-1)
    
    # Confusion Matrix Elements
    tp = (pred * target).sum().item()
    fp = (pred * (1 - target)).sum().item()
    fn = ((1 - pred) * target).sum().item()
    tn = ((1 - pred) * (1 - target)).sum().item()
    
    # Metrics
    accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    
    return {
        'iou': iou,
        'dice': dice,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'boundary_iou': boundary_iou
    }

def save_comparison(image, target, pred_v4, pred_v3, save_path, iou_v4, iou_v3):
    # Denormalize image
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(image.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(image.device)
    img_vis = image[0] * std + mean
    img_vis = img_vis.cpu().permute(1, 2, 0).numpy()
    img_vis = np.clip(img_vis, 0, 1)
    
    gt_vis = target[0].cpu().numpy()
    pred_v4_vis = pred_v4[0].cpu().numpy()
    
    cols = 4 if pred_v3 is not None else 3
    fig, axes = plt.subplots(1, cols, figsize=(cols*5, 5))
    
    # Original Image
    axes[0].imshow(img_vis)
    axes[0].set_title("Input Image")
    axes[0].axis('off')
    
    # Ground Truth
    axes[1].imshow(gt_vis, cmap='gray')
    axes[1].set_title("Ground Truth")
    axes[1].axis('off')
    
    # V3 Prediction
    idx = 2
    if pred_v3 is not None:
        pred_v3_vis = pred_v3[0].cpu().numpy()
        axes[idx].imshow(pred_v3_vis, cmap='gray')
        axes[idx].set_title(f"V3 Baseline\nIoU: {iou_v3:.4f}")
        axes[idx].axis('off')
        idx += 1
    
    # V4 Prediction
    axes[idx].imshow(pred_v4_vis, cmap='gray')
    axes[idx].set_title(f"V4 Deep Guided Filter\nIoU: {iou_v4:.4f}")
    axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', type=str, required=True)
    parser.add_argument('--v4-checkpoint', type=str, required=True)
    parser.add_argument('--v3-checkpoint', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default='./evaluation_results')
    args = parser.parse_args()
    
    evaluate_and_visualize(
        args.dataset_dir,
        args.v4_checkpoint,
        args.v3_checkpoint,
        args.output_dir
    )
