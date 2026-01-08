"""
Save Predictions for AURASeg V4 Models
======================================

Saves predictions for:
1. AURASeg V4 (CSPDarknet) - ablation study version
2. AURASeg V4-R50 (ResNet-50 backbone) - benchmark version
"""

import os
import sys
from pathlib import Path

import torch
import numpy as np
import cv2
from tqdm import tqdm
from torch.utils.data import DataLoader

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "kobuki-yolop" / "model" / "new_network" / "ABLATION_11_DEC"))

from train_benchmark import DrivableAreaDataset, Config


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


def save_predictions(model, model_name, val_loader, device, output_dir, config, is_auraseg_v4_r50=False):
    """Run inference and save predictions."""
    
    # Create output dirs
    pred_dir = output_dir / "pred_masks"
    compare_dir = output_dir / "compare"
    pred_dir.mkdir(parents=True, exist_ok=True)
    compare_dir.mkdir(parents=True, exist_ok=True)
    
    saved = 0
    
    with torch.no_grad():
        for images, masks, names in tqdm(val_loader, desc=f"Saving {model_name}"):
            images = images.to(device)
            
            # Inference - handle AURASeg V4-R50 specifically
            if is_auraseg_v4_r50:
                outputs = model(images, return_aux=False, return_boundary=False)
                # Model returns dict with 'main' key
                main_out = outputs['main']
                preds = torch.argmax(main_out, dim=1)
            else:
                outputs = model(images)
                
                # Handle different output formats
                if isinstance(outputs, dict):
                    outputs = outputs.get('main', outputs.get('out', list(outputs.values())[0]))
                elif isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                preds = torch.argmax(outputs, dim=1)
            
            # Save each sample
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
    
    print(f"  Saved {saved} predictions to: {output_dir}")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = Config()
    runs_dir = Path(__file__).parent.parent / "runs"
    
    print(f"Device: {device}")
    
    # Load validation dataset
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
    
    print(f"Validation samples: {len(val_dataset)}")
    
    # =========================================================================
    # AURASeg V4 (CSPDarknet) - Ablation version
    # =========================================================================
    print(f"\n{'='*60}")
    print("Processing: AURASeg V4 (CSPDarknet)")
    print(f"{'='*60}")
    
    try:
        # Add ABLATION_11_DEC to path for proper module resolution
        ablation_root = Path(__file__).parent.parent / "kobuki-yolop" / "model" / "new_network" / "ABLATION_11_DEC"
        sys.path.insert(0, str(ablation_root))
        
        from models.v4_rbrm import V4RBRM
        
        ckpt_path = runs_dir / "ablation_v4" / "v4_rbrm" / "checkpoints" / "best.pth"
        
        if ckpt_path.exists():
            model = V4RBRM(in_channels=3, num_classes=2).to(device)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            model.eval()
            
            print(f"  Loaded: {ckpt_path}")
            print(f"  Best mIoU: {ckpt.get('best_miou', 'N/A'):.4f}")
            
            output_dir = runs_dir / "ablation_v4" / "v4_rbrm" / "visualizations" / "val"
            save_predictions(model, "auraseg_v4", val_loader, device, output_dir, config)
            
            del model
            torch.cuda.empty_cache()
        else:
            print(f"  [SKIP] Checkpoint not found: {ckpt_path}")
            
    except Exception as e:
        print(f"  [ERROR] {e}")
    
    # =========================================================================
    # AURASeg V4-R50 (ResNet-50 backbone) - Benchmark version
    # =========================================================================
    print(f"\n{'='*60}")
    print("Processing: AURASeg V4-R50 (ResNet-50)")
    print(f"{'='*60}")
    
    try:
        from auraseg_v4_resnet import AURASeg_V4_ResNet50
        
        ckpt_path = runs_dir / "auraseg_v4_resnet50" / "checkpoints" / "best.pth"
        
        if ckpt_path.exists():
            model = AURASeg_V4_ResNet50(num_classes=2).to(device)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            model.eval()
            
            print(f"  Loaded: {ckpt_path}")
            print(f"  Best mIoU: {ckpt.get('best_miou', 'N/A'):.4f}")
            
            output_dir = runs_dir / "auraseg_v4_resnet50" / "visualizations" / "val"
            save_predictions(model, "auraseg_v4_r50", val_loader, device, output_dir, config, is_auraseg_v4_r50=True)
            
            del model
            torch.cuda.empty_cache()
        else:
            print(f"  [SKIP] Checkpoint not found: {ckpt_path}")
            
    except Exception as e:
        print(f"  [ERROR] {e}")
    
    print("\n" + "="*60)
    print("AURASeg PREDICTIONS SAVED!")
    print("="*60)


if __name__ == "__main__":
    main()
