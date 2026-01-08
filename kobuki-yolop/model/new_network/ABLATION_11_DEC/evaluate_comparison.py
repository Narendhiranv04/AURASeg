import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF

from config import DataConfig
from utils import DrivableAreaDataset, compute_metrics, AverageMeter
from models.v1_base_spp import V1BaseSPP
from models.v2_base_assplite import V2BaseASPPLite

def load_model(model_class, checkpoint_path, device):
    model = model_class().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def denormalize(tensor, mean, std):
    mean = torch.tensor(mean).view(3, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(3, 1, 1).to(tensor.device)
    return tensor * std + mean

def visualize_comparison(image, gt, pred_v1, pred_v2, save_path):
    # Convert tensors to numpy
    # Image: (3, H, W) -> (H, W, 3)
    img_np = image.permute(1, 2, 0).cpu().numpy()
    img_np = np.clip(img_np, 0, 1)
    
    # Masks: (H, W)
    gt_np = gt.cpu().numpy()
    v1_np = pred_v1.cpu().numpy()
    v2_np = pred_v2.cpu().numpy()
    
    # Create colored masks
    # Green for Drivable Area
    def colorize(mask):
        color_mask = np.zeros_like(img_np)
        color_mask[mask == 1] = [0, 1, 0] # RGB
        return color_mask

    gt_color = colorize(gt_np)
    v1_color = colorize(v1_np)
    v2_color = colorize(v2_np)
    
    # Overlay
    alpha = 0.4
    gt_overlay = img_np.copy()
    mask_indices = gt_np == 1
    gt_overlay[mask_indices] = gt_overlay[mask_indices] * (1 - alpha) + gt_color[mask_indices] * alpha
    
    v1_overlay = img_np.copy()
    mask_indices = v1_np == 1
    v1_overlay[mask_indices] = v1_overlay[mask_indices] * (1 - alpha) + v1_color[mask_indices] * alpha
    
    v2_overlay = img_np.copy()
    mask_indices = v2_np == 1
    v2_overlay[mask_indices] = v2_overlay[mask_indices] * (1 - alpha) + v2_color[mask_indices] * alpha
    
    # Plot
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(img_np)
    axes[0].set_title("Input Image")
    axes[0].axis('off')
    
    axes[1].imshow(gt_overlay)
    axes[1].set_title("Ground Truth")
    axes[1].axis('off')
    
    axes[2].imshow(v1_overlay)
    axes[2].set_title("V1 (SPP)")
    axes[2].axis('off')
    
    axes[3].imshow(v2_overlay)
    axes[3].set_title("V2 (ASPP-Lite)")
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Paths
    v1_ckpt = 'kobuki-yolop/model/new_network/ABLATION_11_DEC/runs/v1_base_spp/checkpoints/best.pth'
    v2_ckpt = 'kobuki-yolop/model/new_network/ABLATION_11_DEC/runs/v2_base_assplite/checkpoints/best.pth'
    output_dir = 'kobuki-yolop/model/new_network/ABLATION_11_DEC/runs/comparison_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load Dataset
    data_config = DataConfig()
    val_dataset = DrivableAreaDataset(
        image_dir=data_config.image_dir,
        mask_dir=data_config.mask_dir,
        img_size=data_config.img_size,
        split='val',
        transform=False
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
    
    # Load Models
    print("Loading V1 Model...")
    model_v1 = load_model(V1BaseSPP, v1_ckpt, device)
    
    print("Loading V2 Model...")
    model_v2 = load_model(V2BaseASPPLite, v2_ckpt, device)
    
    # Metrics
    metric_keys = ['miou', 'dice_drivable', 'accuracy', 'precision_drivable', 'recall_drivable', 'f1_drivable']
    metrics_v1 = {k: AverageMeter() for k in metric_keys}
    metrics_v2 = {k: AverageMeter() for k in metric_keys}
    
    print("Starting Evaluation...")
    with torch.no_grad():
        for i, (images, masks) in enumerate(tqdm(val_loader)):
            images = images.to(device)
            masks = masks.to(device)
            
            # Inference
            out_v1 = model_v1(images)
            out_v2 = model_v2(images)
            
            pred_v1 = torch.argmax(out_v1, dim=1)
            pred_v2 = torch.argmax(out_v2, dim=1)
            
            # Calculate Metrics
            m_v1 = compute_metrics(pred_v1, masks)
            m_v2 = compute_metrics(pred_v2, masks)
            
            for k in metrics_v1.keys():
                metrics_v1[k].update(m_v1[k])
                metrics_v2[k].update(m_v2[k])
            
            # Visualize (save first 50 images)
            if i < 50:
                img_denorm = denormalize(images[0], val_dataset.mean, val_dataset.std)
                save_path = os.path.join(output_dir, f"comparison_{i:03d}.png")
                visualize_comparison(img_denorm, masks[0], pred_v1[0], pred_v2[0], save_path)
    
    # Print Results
    print("\n" + "="*40)
    print("COMPARISON RESULTS")
    print("="*40)
    print(f"{'Metric':<20} | {'V1 (SPP)':<15} | {'V2 (ASPP-Lite)':<15}")
    print("-" * 56)
    
    for k in metrics_v1.keys():
        print(f"{k:<20} | {metrics_v1[k].avg:.4f}          | {metrics_v2[k].avg:.4f}")
        
    # Save to file
    with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
        f.write("COMPARISON RESULTS\n")
        f.write(f"{'Metric':<20} | {'V1 (SPP)':<15} | {'V2 (ASPP-Lite)':<15}\n")
        f.write("-" * 56 + "\n")
        for k in metrics_v1.keys():
            f.write(f"{k:<20} | {metrics_v1[k].avg:.4f}          | {metrics_v2[k].avg:.4f}\n")

if __name__ == "__main__":
    main()
