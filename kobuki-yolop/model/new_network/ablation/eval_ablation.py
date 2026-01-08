"""
Evaluation Script for Ablation Study
====================================
Calculates:
- mIoU
- Max F1 Score (at optimal threshold)
- Precision & Recall
- Parameters
- FLOPs
- FPS (Inference Speed)
"""

import os
import sys
import time
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from ablation_v1_base import AblationBaseModel
from ablation_v2_assplite import AblationV2ASPPLite
from ablation_v3_apud import AblationV3APUD
from ablation_v4_rbrm import AblationV4RBRM
from train_ablation import SegmentationDataset, SegmentationMetrics

def calculate_flops(model, input_size, device):
    try:
        from thop import profile
        input = torch.randn(1, 3, input_size[0], input_size[1]).to(device)
        macs, params = profile(model, inputs=(input, ), verbose=False)
        flops = macs * 2  # MACs to FLOPs
        return flops, params
    except ImportError:
        print("Warning: 'thop' library not found. FLOPs calculation skipped.")
        print("Install it via: pip install thop")
        return 0, 0

import argparse

def evaluate_model():
    parser = argparse.ArgumentParser(description='Evaluate Ablation Model')
    parser.add_argument('--model-path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--model-type', type=str, required=True, choices=['v1', 'v2', 'v3', 'v4'], help='Model version (v1, v2, v3, v4)')
    parser.add_argument('--dataset-dir', type=str, default="C:/Users/naren/Documents/AURASeg/CommonDataset", help='Path to dataset')
    
    args = parser.parse_args()

    # Configuration
    model_path = args.model_path
    dataset_dir = args.dataset_dir
    model_version = args.model_type
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Evaluating model: {model_path}")
    print(f"Model Version: {model_version}")
    print(f"Device: {device}")
    
    # Load Config (from saved file if possible, else default)
    config_path = os.path.join(os.path.dirname(model_path), '..', 'config.json')
    if os.path.exists(config_path):
        # Load config logic here if needed, for now we use default + overrides
        pass
    
    # Initialize Model based on version
    if model_version == "v1":
        model = AblationBaseModel(num_classes=2, in_channels=3)
    elif model_version == "v2":
        model = AblationV2ASPPLite(num_classes=2, in_channels=3)
    elif model_version == "v3":
        model = AblationV3APUD(num_classes=2, in_channels=3, deep_supervision=False)
    elif model_version == "v4":
        model = AblationV4RBRM(num_classes=2, in_channels=3, deep_supervision=False)
    else:
        raise ValueError(f"Unknown model version: {model_version}")
    
    # Load Weights
    if os.path.exists(model_path):
        # Set weights_only=False to handle numpy types in checkpoint
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']+1}")
    else:
        print(f"Error: Model not found at {model_path}")
        return

    model.to(device)
    model.eval()
    
    # 1. Calculate Parameters & FLOPs
    input_height, input_width = 640, 384
    flops, params = calculate_flops(model, (input_height, input_width), device)
    
    print("\nModel Complexity:")
    print(f"  Parameters: {params / 1e6:.2f} M")
    print(f"  FLOPs: {flops / 1e9:.2f} G")
    
    # 2. Calculate FPS
    print("\nMeasuring FPS...")
    dummy_input = torch.randn(1, 3, input_height, input_width).to(device)
    
    # Warmup
    for _ in range(50):
        with torch.no_grad():
            _ = model(dummy_input)
            
    # Timing
    num_frames = 200
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_frames):
            _ = model(dummy_input)
    end_time = time.time()
    
    fps = num_frames / (end_time - start_time)
    print(f"  FPS: {fps:.2f} (on {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    
    # 3. Evaluate Metrics on Validation Set
    print("\nEvaluating Metrics on Validation Set...")
    
    val_img_dir = os.path.join(dataset_dir, 'images', 'val')
    val_mask_dir = os.path.join(dataset_dir, 'labels', 'val')
    
    val_dataset = SegmentationDataset(
        image_dir=val_img_dir,
        mask_dir=val_mask_dir,
        transform=transforms.Compose([
            transforms.Resize((input_height, input_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        target_transform=transforms.Compose([
            transforms.Resize((input_height, input_width), interpolation=Image.NEAREST),
            transforms.ToTensor() # Will be converted to long later
        ]),
        use_augmentation=False,
        color_jitter=False
    )
    
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)
    
    metrics = SegmentationMetrics(num_classes=2, ignore_index=255)
    metrics.reset()
    
    # For Max F1, we need to store probabilities and targets
    # This can be memory intensive, so we'll do a simplified version or just standard F1 first
    # Standard F1 (argmax)
    
    with torch.no_grad():
        for images, masks in tqdm(val_loader):
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward
            outputs = model(images)
            
            # Resize output to match mask (if needed, though we resized input)
            # The model output should be same size as input
            
            # Update metrics (Argmax based)
            metrics.update(outputs, masks)
            
    results = metrics.compute()
    
    print("\nEvaluation Results:")
    print(f"  mIoU:      {results['miou']:.4f}")
    print(f"  mDice:     {results['mdice']:.4f}")
    print(f"  Precision: {results.get('precision', 0):.4f}")
    print(f"  Recall:    {results.get('recall', 0):.4f}")
    print(f"  F1 Score:  {results.get('f1', 0):.4f}")
    
    # Note on Max F1
    print("\nNote: Reported F1 is based on standard argmax (threshold 0.5 equivalent).")

if __name__ == "__main__":
    evaluate_model()
