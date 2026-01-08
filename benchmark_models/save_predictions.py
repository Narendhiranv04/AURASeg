"""
Save Predictions for All Benchmark Models
==========================================

Runs inference on the validation set and saves:
1. Predicted masks (binary PNG)
2. Side-by-side comparison images (Input | GT | Prediction)

Output structure:
    runs/benchmark_<model>/visualizations/val/
        pred_masks/     - Binary prediction masks
        compare/        - Side-by-side comparisons
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

from model_factory import get_benchmark_model
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


def save_model_predictions(model_name, checkpoint_path, val_loader, device, output_dir, config):
    """Run inference and save predictions for one model."""
    
    print(f"\n{'='*60}")
    print(f"Processing: {model_name.upper()}")
    print(f"{'='*60}")
    
    # Load model
    model, info = get_benchmark_model(model_name, num_classes=2, pretrained=False)
    
    if not checkpoint_path.exists():
        print(f"  [SKIP] Checkpoint not found: {checkpoint_path}")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"  Model: {info['name']}")
    print(f"  Params: {info['params_millions']:.2f}M")
    print(f"  Best mIoU: {checkpoint.get('best_miou', 'N/A'):.4f}")
    
    # Create output dirs
    pred_dir = output_dir / "pred_masks"
    compare_dir = output_dir / "compare"
    pred_dir.mkdir(parents=True, exist_ok=True)
    compare_dir.mkdir(parents=True, exist_ok=True)
    
    saved = 0
    
    with torch.no_grad():
        for images, masks, names in tqdm(val_loader, desc=f"Saving {model_name}"):
            images = images.to(device)
            
            # Inference
            outputs = model(images)
            
            # Handle different output formats (OrderedDict for FCN, tensor for others)
            if isinstance(outputs, dict):
                outputs = outputs.get('out', list(outputs.values())[0])
            
            preds = torch.argmax(outputs, dim=1)
            
            # Save each sample
            for i in range(preds.shape[0]):
                fname = str(names[i])
                stem = safe_stem(fname)
                
                # Get numpy arrays
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
                
                # Concatenate: Input | GT | Pred
                compare = np.concatenate([img_np, gt_rgb, pred_rgb], axis=1)
                cv2.imwrite(str(compare_dir / f"{stem}.png"), cv2.cvtColor(compare, cv2.COLOR_RGB2BGR))
                
                saved += 1
    
    print(f"  Saved {saved} predictions to: {output_dir}")
    
    # Cleanup
    del model
    torch.cuda.empty_cache()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = Config()
    runs_dir = Path(__file__).parent.parent / "runs"
    
    print(f"Device: {device}")
    print(f"Dataset: {config.IMAGE_DIR}")
    
    # All models to process
    models = ['fcn', 'pspnet', 'deeplabv3plus', 'upernet', 'segformer', 'mask2former', 'pidnet']
    
    # Load validation dataset with filenames
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
    
    # Process each model
    for model_name in models:
        checkpoint_path = runs_dir / f"benchmark_{model_name}" / "checkpoints" / "best.pth"
        output_dir = runs_dir / f"benchmark_{model_name}" / "visualizations" / "val"
        
        save_model_predictions(
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            val_loader=val_loader,
            device=device,
            output_dir=output_dir,
            config=config
        )
    
    print("\n" + "="*60)
    print("ALL PREDICTIONS SAVED!")
    print("="*60)
    print("\nOutput locations:")
    for model_name in models:
        print(f"  {model_name}: runs/benchmark_{model_name}/visualizations/val/")


if __name__ == "__main__":
    main()
