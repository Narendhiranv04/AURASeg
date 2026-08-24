"""
Ablation Study Inference Script
===============================
Run inference on all 4 ablation models and save predictions as white/black masks.
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import cv2

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / 'benchmark_models'))

# Import ablation models V1-V3
from ablation_v1_base import AblationBaseModel
from ablation_v2_assplite import AblationV2ASPPLite
from ablation_v3_apud import AblationV3APUD

# Import V4 from benchmark_models (the correct architecture)
from auraseg_v4_resnet import AURASeg_V4_ResNet50


def load_image(image_path, target_size=(384, 640)):
    """Load and preprocess image."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((target_size[1], target_size[0]), Image.BILINEAR)
    
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = (img_array - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    
    tensor = torch.from_numpy(img_array.transpose(2, 0, 1)).float()
    return tensor.unsqueeze(0)


def save_prediction(pred, save_path):
    """Save prediction as white/black mask (white=255 for free space, black=0 for background)."""
    mask = np.zeros((pred.shape[0], pred.shape[1]), dtype=np.uint8)
    mask[pred == 1] = 255
    cv2.imwrite(str(save_path), mask)


def copy_image(src_path, dst_path, size=(384, 640)):
    """Copy and resize image."""
    img = Image.open(src_path).convert('RGB')
    img = img.resize((size[1], size[0]), Image.BILINEAR)
    img.save(dst_path)


def copy_mask(src_path, dst_path, size=(384, 640)):
    """Copy and convert ground truth to white/black."""
    mask = np.array(Image.open(src_path).convert('L'))
    mask = cv2.resize(mask, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
    
    # Normalize to binary
    if mask.max() > 1:
        mask = (mask > 127).astype(np.uint8) * 255
    else:
        mask = mask.astype(np.uint8) * 255
    
    cv2.imwrite(str(dst_path), mask)


def get_samples(dataset_dir, split='val', max_samples=50):
    """Get image-mask pairs using os.listdir to handle spaces in filenames."""
    dataset_dir = Path(dataset_dir)
    
    # Try different structures
    if (dataset_dir / 'images' / split).exists():
        img_dir = dataset_dir / 'images' / split
        mask_dir = dataset_dir / 'labels' / split
    elif (dataset_dir / split / 'images').exists():
        img_dir = dataset_dir / split / 'images'
        mask_dir = dataset_dir / split / 'labels'
    else:
        print(f"  Could not find {split} split in {dataset_dir}")
        return []
    
    samples = []
    # Use os.listdir instead of glob to handle spaces in filenames
    if img_dir.exists():
        for filename in os.listdir(img_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')) and filename != 'desktop.ini':
                img_path = img_dir / filename
                stem = Path(filename).stem
                
                # Try different mask naming conventions
                mask_path = None
                # Standard: same stem + .png
                candidate = mask_dir / (stem + '.png')
                if candidate.exists():
                    mask_path = candidate
                # Standard: same stem + .jpg
                if mask_path is None:
                    candidate = mask_dir / (stem + '.jpg')
                    if candidate.exists():
                        mask_path = candidate
                # CARL-D style: filename + ___fuse.png
                if mask_path is None:
                    candidate = mask_dir / (filename + '___fuse.png')
                    if candidate.exists():
                        mask_path = candidate
                
                if mask_path is not None:
                    samples.append((img_path, mask_path))
    
    print(f"  Found {len(samples)} total samples in {img_dir}")
    
    if max_samples and len(samples) > max_samples:
        indices = np.linspace(0, len(samples)-1, max_samples, dtype=int)
        samples = [samples[i] for i in indices]
        print(f"  Subsampled to {len(samples)} samples")
    
    return samples


def run_inference(model, img_tensor, device):
    """Run model inference."""
    img_tensor = img_tensor.to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
        
        if isinstance(output, dict):
            if 'main' in output:
                logits = output['main']
            elif 'out' in output:
                logits = output['out']
            else:
                logits = list(output.values())[0]
        elif isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output
        
        # Ensure full resolution
        if logits.shape[2:] != (384, 640):
            logits = F.interpolate(logits, size=(384, 640), mode='bilinear', align_corners=False)
        
        pred = torch.argmax(logits, dim=1).squeeze().cpu().numpy()
    
    return pred


def main():
    base_dir = Path(__file__).parent.parent.parent.parent.parent  # AURASeg root
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("=" * 70)
    print("ABLATION STUDY INFERENCE (White/Black Masks)")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Base dir: {base_dir}")
    
    # Checkpoints - V4 uses auraseg_v4_resnet50
    checkpoints = {
        'v1_base': base_dir / 'runs' / 'ablation_base_v1' / 'v1_base' / 'checkpoints' / 'best.pth',
        'v2_aspp_lite': base_dir / 'runs' / 'ablation' / 'v2_assplite' / 'checkpoints' / 'best.pth',
        'v3_apud': base_dir / 'runs' / 'ablation' / 'v3_apud' / 'checkpoints' / 'best.pth',
        'v4_rbrm': base_dir / 'runs' / 'auraseg_v4_resnet50' / 'checkpoints' / 'best.pth',
    }
    
    # Verify checkpoints exist
    print("\nCheckpoints:")
    for name, path in checkpoints.items():
        status = "✓" if path.exists() else "✗"
        print(f"  {status} {name}: {path}")
        if not path.exists():
            print(f"    ERROR: Not found!")
            return
    
    # Model classes - V4 uses the full AURASeg_V4_ResNet50
    model_classes = {
        'v1_base': AblationBaseModel,
        'v2_aspp_lite': AblationV2ASPPLite,
        'v3_apud': AblationV3APUD,
        'v4_rbrm': AURASeg_V4_ResNet50,
    }
    
    # Output directory
    output_dir = base_dir / 'runs' / 'ablation_visualization'
    
    # Datasets
    datasets = {
        'CommonDataset': base_dir / 'CommonDataset',
        'CARL-D': base_dir / 'carl-dataset',
    }
    
    # Process each dataset
    for dataset_name, dataset_path in datasets.items():
        if not dataset_path.exists():
            print(f"\nSkipping {dataset_name} (not found)")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing: {dataset_name}")
        print(f"{'='*60}")
        
        samples = get_samples(dataset_path, split='val', max_samples=50)
        
        if len(samples) == 0:
            print(f"  No samples found, skipping...")
            continue
        
        # Create output dirs
        ds_out = output_dir / dataset_name
        dirs = {
            'images': ds_out / 'images',
            'ground_truth': ds_out / 'ground_truth',
            'v1_base': ds_out / 'v1_base',
            'v2_aspp_lite': ds_out / 'v2_aspp_lite', 
            'v3_apud': ds_out / 'v3_apud',
            'v4_rbrm': ds_out / 'v4_rbrm',
        }
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        
        # Process each model
        for model_name, ModelClass in model_classes.items():
            print(f"\n  Loading {model_name}...")
            
            try:
                # Create model
                model = ModelClass(num_classes=2)
                
                # Load checkpoint
                ckpt = torch.load(checkpoints[model_name], map_location=device, weights_only=False)
                if 'model_state_dict' in ckpt:
                    model.load_state_dict(ckpt['model_state_dict'])
                else:
                    model.load_state_dict(ckpt)
                
                model = model.to(device).eval()
                print(f"    ✓ Loaded successfully")
                
                # Run inference
                for img_path, mask_path in tqdm(samples, desc=f"    {model_name}", leave=False):
                    # Create safe filename (replace spaces with underscores)
                    filename = img_path.stem.replace(' ', '_')
                    
                    # Save image and GT (only once, with first model)
                    if model_name == 'v1_base':
                        copy_image(img_path, dirs['images'] / f"{filename}.png")
                        copy_mask(mask_path, dirs['ground_truth'] / f"{filename}.png")
                    
                    # Inference
                    img_tensor = load_image(img_path)
                    pred = run_inference(model, img_tensor, device)
                    save_prediction(pred, dirs[model_name] / f"{filename}.png")
                
                print(f"    ✓ Processed {len(samples)} samples")
                del model
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"    ✗ Error: {e}")
                import traceback
                traceback.print_exc()
    
    print(f"\n{'='*70}")
    print("COMPLETE!")
    print(f"{'='*70}")
    print(f"Output directory: {output_dir}")
    print(f"\nSaved masks as white (255) / black (0)")


if __name__ == "__main__":
    main()
