import os
import sys
import torch
import numpy as np
# import cv2  # Removed dependency
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from ablation_v1_base import AblationBaseModel
from ablation_v2_assplite import AblationV2ASPPLite
from ablation_v3_apud import AblationV3APUD
from ablation_v4_rbrm import AblationV4RBRM
from train_ablation import SegmentationDataset

def load_model(model_class, checkpoint_path, device, **kwargs):
    model = model_class(num_classes=2, in_channels=3, **kwargs)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded {model_class.__name__} from {checkpoint_path}")
    else:
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return None
    model.to(device)
    model.eval()
    return model

def visualize_all():
    # Configuration
    dataset_dir = "C:/Users/naren/Documents/AURASeg/CommonDataset"
    output_dir = "./runs/comparison_results"
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Checkpoint Paths (Updated based on file search)
    checkpoints = {
        'v1': r"C:\Users\naren\Documents\AURASeg\runs\ablation_base_v1\v1_base\checkpoints\best.pth",
        'v2': r"C:\Users\naren\Documents\AURASeg\runs\ablation\v2_assplite\checkpoints\best.pth",
        'v3': r"C:\Users\naren\Documents\AURASeg\kobuki-yolop\model\new_network\ablation\runs\ablation\v3_apud\checkpoints\best.pth",
        'v4': r"C:\Users\naren\Documents\AURASeg\kobuki-yolop\model\new_network\ablation\runs\ablation_v4\v4_rbrm_frozen\checkpoints\best.pth"
    }
    
    # Initialize Models
    models = {}
    models['v1'] = load_model(AblationBaseModel, checkpoints['v1'], device)
    models['v2'] = load_model(AblationV2ASPPLite, checkpoints['v2'], device)
    models['v3'] = load_model(AblationV3APUD, checkpoints['v3'], device, deep_supervision=False)
    models['v4'] = load_model(AblationV4RBRM, checkpoints['v4'], device, deep_supervision=False)
    
    # Check if all models loaded
    for k, v in models.items():
        if v is None:
            print(f"Skipping visualization because model {k} failed to load.")
            return

    # Dataset
    val_img_dir = os.path.join(dataset_dir, 'images', 'val')
    val_mask_dir = os.path.join(dataset_dir, 'labels', 'val')
    
    # Transform for model input (with normalization)
    transform = transforms.Compose([
        transforms.Resize((640, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Transform for mask
    target_transform = transforms.Compose([
        transforms.Resize((640, 384), interpolation=Image.NEAREST),
        transforms.ToTensor()
    ])
    
    val_dataset = SegmentationDataset(
        image_dir=val_img_dir, 
        mask_dir=val_mask_dir, 
        transform=transform,
        target_transform=target_transform,
        use_augmentation=False,
        color_jitter=False
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)
    
    print(f"Starting visualization on {len(val_dataset)} images...")
    
    # Color map for visualization (0: Background, 1: Lane)
    # Background: Black, Lane: Green
    colors = np.array([
        [0, 0, 0],
        [0, 255, 0]
    ], dtype=np.uint8)

    with torch.no_grad():
        for i, (images, masks) in enumerate(tqdm(val_loader)):
            images = images.to(device)
            
            # Get predictions
            preds = {}
            for name, model in models.items():
                output = model(images)
                # Handle deep supervision or tuple outputs if any (V3/V4 might return tuple if DS is on, but we set it to False)
                # Just in case check type
                if isinstance(output, (tuple, list)):
                    output = output[0]
                
                pred = torch.argmax(output, dim=1).cpu().numpy()[0]
                preds[name] = pred

            # Prepare images for grid
            # Original Image - Need to denormalize or reload
            # Simple denormalization for visualization
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
            orig_img = images[0] * std + mean
            orig_img = torch.clamp(orig_img, 0, 1)
            
            img_np = orig_img.permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            
            # Ground Truth
            gt_mask = masks[0].numpy()
            gt_color = colors[gt_mask]
            pil_gt = Image.fromarray(gt_color)
            
            # Predictions
            pil_preds = {}
            for name, pred in preds.items():
                p_color = colors[pred]
                pil_preds[name] = Image.fromarray(p_color)
            
            # Create Grid
            # Row 1: Input, GT
            # Row 2: V1, V2
            # Row 3: V3, V4
            
            w, h = pil_img.size
            grid_img = Image.new('RGB', (w * 2, h * 3))
            
            # Helper to add text
            def add_label(img, text):
                draw = ImageDraw.Draw(img)
                # Default font
                # draw.text((10, 10), text, fill=(255, 255, 255))
                # To make it bigger/clearer without custom font file, we can just draw it. 
                # Or just paste it and know what it is by position.
                # Let's try to draw text.
                draw.text((10, 10), text, fill=(255, 255, 255))
                return img

            # Paste images
            grid_img.paste(add_label(pil_img, "Input"), (0, 0))
            grid_img.paste(add_label(pil_gt, "Ground Truth"), (w, 0))
            
            grid_img.paste(add_label(pil_preds['v1'], "V1 (Base)"), (0, h))
            grid_img.paste(add_label(pil_preds['v2'], "V2 (ASPPLite)"), (w, h))
            
            grid_img.paste(add_label(pil_preds['v3'], "V3 (APUD)"), (0, h * 2))
            grid_img.paste(add_label(pil_preds['v4'], "V4 (RBRM)"), (w, h * 2))
            
            # Save
            save_path = os.path.join(output_dir, f"comparison_{i:04d}.png")
            grid_img.save(save_path)

    print(f"Saved comparison images to {output_dir}")

if __name__ == "__main__":
    visualize_all()
