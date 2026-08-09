"""
Ablation Study Qualitative Visualization
=========================================

Generates prediction outputs for all ablation variants:
- V1 Base: CSPDarknet + SPP + Simple Decoder
- V2 ASPPLite: Base + ASPPLite (replaces SPP)
- V3 APUD: V2 + Attention Progressive Upsampling Decoder
- V4 RBRM: V3 + Residual Boundary Refinement Module (Full AURASeg)

Outputs organized by dataset:
    runs/ablation_visualization/
    ├── CommonDataset/
    │   ├── images/
    │   ├── ground_truth/
    │   ├── v1_base/
    │   ├── v2_aspp_lite/
    │   ├── v3_apud/
    │   └── v4_rbrm/
    └── CARL-D/
        ├── images/
        ├── ground_truth/
        ├── v1_base/
        ├── v2_aspp_lite/
        ├── v3_apud/
        └── v4_rbrm/
"""

import sys
import os
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2
import shutil

# Add paths
ablation_models_dir = str(Path(__file__).parent.parent / "kobuki-yolop" / "model" / "new_network" / "ablation")
sys.path.insert(0, ablation_models_dir)
sys.path.insert(0, str(Path(__file__).parent))


# =============================================================================
# Model Loading Functions
# =============================================================================

def load_v1_base(checkpoint_path, device='cuda'):
    """Load V1 Base model."""
    from ablation_v1_base import AblationBaseModel
    
    model = AblationBaseModel(num_classes=2)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    return model.to(device).eval()


def load_v2_assplite(checkpoint_path, device='cuda'):
    """Load V2 ASPPLite model."""
    from ablation_v2_assplite import AblationV2ASPPLite
    
    model = AblationV2ASPPLite(num_classes=2)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    return model.to(device).eval()


def load_v3_apud(checkpoint_path, device='cuda'):
    """Load V3 APUD model."""
    from ablation_v3_apud import AblationV3APUD
    
    model = AblationV3APUD(num_classes=2)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    return model.to(device).eval()


def load_v4_rbrm(checkpoint_path, device='cuda'):
    """Load V4 RBRM (Full AURASeg) model."""
    from ablation_v4_rbrm_old import AblationV4RBRM
    
    model = AblationV4RBRM(num_classes=2)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    return model.to(device).eval()


# =============================================================================
# Data Loading and Preprocessing
# =============================================================================

def load_image(image_path, target_size=(384, 640)):
    """Load and preprocess image."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((target_size[1], target_size[0]), Image.BILINEAR)
    
    # Normalize
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = (img_array - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    
    # To tensor (C, H, W)
    tensor = torch.from_numpy(img_array.transpose(2, 0, 1)).float()
    return tensor.unsqueeze(0)


def load_mask(mask_path, target_size=(384, 640)):
    """Load ground truth mask."""
    mask = Image.open(mask_path).convert('L')
    mask = mask.resize((target_size[1], target_size[0]), Image.NEAREST)
    return np.array(mask)


def get_dataset_samples(dataset_dir, split='val', max_samples=None):
    """Get list of image-mask pairs from dataset."""
    dataset_dir = Path(dataset_dir)
    
    # Check directory structure
    if (dataset_dir / 'images' / split).exists():
        # CommonDataset structure
        image_dir = dataset_dir / 'images' / split
        mask_dir = dataset_dir / 'labels' / split
    elif (dataset_dir / split / 'images').exists():
        # CARL-D structure
        image_dir = dataset_dir / split / 'images'
        mask_dir = dataset_dir / split / 'labels'
    elif (dataset_dir / split / split).exists():
        # CARL-D test structure
        image_dir = dataset_dir / split / split
        mask_dir = dataset_dir / split / 'labels'
    else:
        raise ValueError(f"Unknown dataset structure in {dataset_dir}")
    
    # Get all image files
    samples = []
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    for ext in image_extensions:
        for img_path in image_dir.glob(f'*{ext}'):
            # Find corresponding mask
            mask_name = img_path.stem + '.png'
            mask_path = mask_dir / mask_name
            
            if not mask_path.exists():
                mask_name = img_path.stem + '.jpg'
                mask_path = mask_dir / mask_name
            
            if mask_path.exists():
                samples.append((img_path, mask_path))
    
    if max_samples and len(samples) > max_samples:
        # Sample evenly
        indices = np.linspace(0, len(samples)-1, max_samples, dtype=int)
        samples = [samples[i] for i in indices]
    
    return samples


# =============================================================================
# Prediction and Visualization
# =============================================================================

def predict(model, image_tensor, device='cuda'):
    """Run model prediction."""
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        
        # Handle different output formats
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
        
        # Get prediction
        pred = torch.argmax(logits, dim=1).squeeze().cpu().numpy()
    
    return pred


def save_prediction_mask(pred, save_path, colormap='binary'):
    """Save prediction as colored mask."""
    if colormap == 'binary':
        # Free space = green, background = black
        mask_colored = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        mask_colored[pred == 1] = [0, 255, 0]  # Green for free space
    else:
        # Use colormap
        mask_colored = cv2.applyColorMap((pred * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    cv2.imwrite(str(save_path), cv2.cvtColor(mask_colored, cv2.COLOR_RGB2BGR))


def save_overlay(image_path, pred, save_path, alpha=0.5):
    """Save prediction overlaid on original image."""
    # Load original image
    img = cv2.imread(str(image_path))
    img = cv2.resize(img, (640, 384))
    
    # Create overlay
    overlay = img.copy()
    overlay[pred == 1] = [0, 255, 0]  # Green for free space
    
    # Blend
    blended = cv2.addWeighted(img, 1-alpha, overlay, alpha, 0)
    cv2.imwrite(str(save_path), blended)


def copy_original_image(image_path, save_path, target_size=(384, 640)):
    """Copy and resize original image."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((target_size[1], target_size[0]), Image.BILINEAR)
    img.save(save_path)


def copy_ground_truth(mask_path, save_path, target_size=(384, 640)):
    """Copy and colorize ground truth mask."""
    mask = np.array(Image.open(mask_path).convert('L'))
    mask = cv2.resize(mask, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)
    
    # Normalize to 0/1 if needed
    if mask.max() > 1:
        mask = (mask > 127).astype(np.uint8)
    
    # Colorize
    mask_colored = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    mask_colored[mask == 1] = [0, 255, 0]
    
    cv2.imwrite(str(save_path), cv2.cvtColor(mask_colored, cv2.COLOR_RGB2BGR))


# =============================================================================
# Main Processing
# =============================================================================

def process_dataset(dataset_name, dataset_dir, models, output_dir, device='cuda', 
                    max_samples=50, split='val'):
    """Process a dataset with all ablation models."""
    
    print(f"\n{'='*60}")
    print(f"Processing: {dataset_name}")
    print(f"{'='*60}")
    
    # Get samples
    samples = get_dataset_samples(dataset_dir, split=split, max_samples=max_samples)
    print(f"Found {len(samples)} samples")
    
    if len(samples) == 0:
        print(f"Warning: No samples found in {dataset_dir}")
        return
    
    # Create output directories
    dataset_output = output_dir / dataset_name
    
    dirs = {
        'images': dataset_output / 'images',
        'ground_truth': dataset_output / 'ground_truth',
        'v1_base': dataset_output / 'v1_base',
        'v2_aspp_lite': dataset_output / 'v2_aspp_lite',
        'v3_apud': dataset_output / 'v3_apud',
        'v4_rbrm': dataset_output / 'v4_rbrm',
    }
    
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    
    # Process each sample
    for img_path, mask_path in tqdm(samples, desc=f"Processing {dataset_name}"):
        filename = img_path.stem
        
        # Save original image
        copy_original_image(img_path, dirs['images'] / f"{filename}.png")
        
        # Save ground truth
        copy_ground_truth(mask_path, dirs['ground_truth'] / f"{filename}.png")
        
        # Load image tensor
        img_tensor = load_image(img_path)
        
        # Get predictions from each model
        for model_name, model in models.items():
            pred = predict(model, img_tensor, device)
            
            # Map model name to directory
            dir_key = model_name.replace('-', '_').lower()
            if 'base' in dir_key:
                dir_key = 'v1_base'
            elif 'aspp' in dir_key:
                dir_key = 'v2_aspp_lite'
            elif 'apud' in dir_key:
                dir_key = 'v3_apud'
            elif 'rbrm' in dir_key:
                dir_key = 'v4_rbrm'
            
            save_path = dirs[dir_key] / f"{filename}.png"
            save_prediction_mask(pred, save_path)
    
    print(f"✓ Saved outputs to: {dataset_output}")


def main():
    # Configuration
    base_dir = Path(__file__).parent.parent
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("=" * 70)
    print("ABLATION STUDY QUALITATIVE VISUALIZATION")
    print("=" * 70)
    print(f"Device: {device}")
    
    # Checkpoint paths
    checkpoints = {
        'v1_base': base_dir / 'runs' / 'ablation_base_v1' / 'v1_base' / 'checkpoints' / 'best.pth',
        'v2_aspp_lite': base_dir / 'runs' / 'ablation' / 'v2_assplite' / 'checkpoints' / 'best.pth',
        'v3_apud': base_dir / 'runs' / 'ablation' / 'v3_apud' / 'checkpoints' / 'best.pth',
        'v4_rbrm': base_dir / 'runs' / 'ablation_v4' / 'v4_rbrm' / 'checkpoints' / 'best.pth',
    }
    
    # Verify checkpoints exist
    print("\nLoading models...")
    for name, path in checkpoints.items():
        if not path.exists():
            print(f"  ERROR: Checkpoint not found: {path}")
            return
        print(f"  ✓ Found: {name}")
    
    # Load models
    print("\nInitializing models...")
    models = {}
    
    print("  Loading V1 Base...")
    models['v1_base'] = load_v1_base(checkpoints['v1_base'], device)
    
    print("  Loading V2 ASPPLite...")
    models['v2_aspp_lite'] = load_v2_assplite(checkpoints['v2_aspp_lite'], device)
    
    print("  Loading V3 APUD...")
    models['v3_apud'] = load_v3_apud(checkpoints['v3_apud'], device)
    
    print("  Loading V4 RBRM (Full AURASeg)...")
    models['v4_rbrm'] = load_v4_rbrm(checkpoints['v4_rbrm'], device)
    
    print("  ✓ All models loaded")
    
    # Output directory
    output_dir = base_dir / 'runs' / 'ablation_visualization'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Dataset paths
    datasets = {
        'CommonDataset': {
            'path': base_dir / 'CommonDataset',
            'split': 'val',
            'max_samples': 50,
        },
        'CARL-D': {
            'path': base_dir / 'carl-dataset',
            'split': 'val',
            'max_samples': 50,
        },
    }
    
    # Process each dataset
    for dataset_name, config in datasets.items():
        if config['path'].exists():
            process_dataset(
                dataset_name=dataset_name,
                dataset_dir=config['path'],
                models=models,
                output_dir=output_dir,
                device=device,
                max_samples=config['max_samples'],
                split=config['split']
            )
        else:
            print(f"\nWarning: Dataset not found: {config['path']}")
    
    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print("\nDirectory structure:")
    print("  ablation_visualization/")
    print("  ├── CommonDataset/")
    print("  │   ├── images/          (original images)")
    print("  │   ├── ground_truth/    (GT masks)")
    print("  │   ├── v1_base/         (Base predictions)")
    print("  │   ├── v2_aspp_lite/    (+ ASPPLite)")
    print("  │   ├── v3_apud/         (+ APUD)")
    print("  │   └── v4_rbrm/         (+ RBRM = Full)")
    print("  └── CARL-D/")
    print("      └── (same structure)")


if __name__ == "__main__":
    main()
