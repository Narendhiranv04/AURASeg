import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
# import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ablation_v1_base import AblationBaseModel
from ablation_v2_assplite import AblationV2ASPPLite
from ablation_v3_apud import AblationV3APUD
from ablation_v4_rbrm import AblationV4RBRM
from train_ablation import SegmentationDataset

try:
    from thop import profile
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("thop not installed. GFLOPs calculation will be skipped.")

def calculate_metrics(pred, target):
    """Calculate IoU, Dice, Precision, Recall, Accuracy for binary segmentation"""
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
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    return {
        'iou': iou,
        'dice': dice,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy
    }

def measure_fps_gflops(model, device, input_size=(1, 3, 480, 640)):
    model.eval()
    x = torch.randn(input_size).to(device)
    
    # GFLOPs
    gflops = 0.0
    if THOP_AVAILABLE:
        try:
            macs, params = profile(model, inputs=(x,), verbose=False)
            gflops = macs / 1e9
        except Exception as e:
            print(f"Error calculating GFLOPs: {e}")
    
    # FPS
    # Warmup
    for _ in range(10):
        _ = model(x)
    
    # Measure
    num_runs = 100
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(x)
    end_time = time.time()
    
    fps = num_runs / (end_time - start_time)
    
    return fps, gflops

def evaluate_model(model, val_loader, device):
    model.eval()
    metrics_list = {'iou': [], 'dice': [], 'precision': [], 'recall': [], 'f1': [], 'accuracy': []}
    
    with torch.no_grad():
        for image, target in tqdm(val_loader, desc="Evaluating", leave=False):
            image = image.to(device)
            target = target.to(device)
            
            output = model(image)
            pred = torch.argmax(output, dim=1)
            
            m = calculate_metrics(pred, target)
            for k, v in m.items():
                metrics_list[k].append(v)
    
    avg_metrics = {k: np.mean(v) for k, v in metrics_list.items()}
    return avg_metrics

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', type=str, required=True)
    parser.add_argument('--v1-ckpt', type=str, required=True)
    parser.add_argument('--v2-ckpt', type=str, required=True)
    parser.add_argument('--v3-ckpt', type=str, required=True)
    parser.add_argument('--v4-ckpt', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='./evaluation_results_all')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Dataset
    # NOTE: Must match training resolution (Height=640, Width=384)
    # Even though original images are landscape (1280x720), the models were trained
    # on this specific resolution/aspect ratio defined in config.py.
    transform = transforms.Compose([
        transforms.Resize((640, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    target_transform = transforms.Compose([
        transforms.Resize((640, 384), interpolation=Image.NEAREST),
        transforms.ToTensor()
    ])
    
    image_dir = os.path.join(args.dataset_dir, 'images', 'val')
    mask_dir = os.path.join(args.dataset_dir, 'labels', 'val')
    
    val_dataset = SegmentationDataset(
        image_dir, mask_dir, 
        transform=transform, 
        target_transform=target_transform,
        use_augmentation=False
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)
    print(f"Validation samples: {len(val_dataset)}")
    
    # Models configuration
    models_config = [
        {
            'name': 'Base Model (V1)',
            'class': AblationBaseModel,
            'ckpt': args.v1_ckpt
        },
        {
            'name': 'Base + ASPP-Lite (V2)',
            'class': AblationV2ASPPLite,
            'ckpt': args.v2_ckpt
        },
        {
            'name': 'Base + ASPP-Lite + APUD (V3)',
            'class': AblationV3APUD,
            'ckpt': args.v3_ckpt
        },
        {
            'name': 'Deep Guided Filter (V4)',
            'class': AblationV4RBRM,
            'ckpt': args.v4_ckpt
        }
    ]
    
    results = []
    
    for config in models_config:
        print(f"\nEvaluating {config['name']}...")
        
        # Initialize model
        model = config['class'](num_classes=2).to(device)
        
        # Load checkpoint
        print(f"Loading weights from {config['ckpt']}")
        checkpoint = torch.load(config['ckpt'], map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        # Calculate Parameters
        params = sum(p.numel() for p in model.parameters()) / 1e6
        
        # Calculate FPS and GFLOPs
        fps, gflops = measure_fps_gflops(model, device)
        
        # Evaluate Metrics
        metrics = evaluate_model(model, val_loader, device)
        
        res = {
            'Model Version': config['name'],
            'Parameters (M)': f"{params:.2f} M",
            'FPS': f"{fps:.2f}",
            'GFLOPs': f"{gflops:.2f}" if gflops > 0 else "N/A",
            'mIoU': f"{metrics['iou']:.4f}",
            'Precision': f"{metrics['precision']:.4f}",
            'Recall': f"{metrics['recall']:.4f}",
            'F1 Score': f"{metrics['f1']:.4f}"
        }
        results.append(res)
        
        # Clean up
        del model
        torch.cuda.empty_cache()
        
    # Print Results
    print("\n" + "="*120)
    print(f"{'Model Version':<30} | {'Params':<12} | {'FPS':<8} | {'GFLOPs':<8} | {'mIoU':<8} | {'Precision':<10} | {'Recall':<8} | {'F1 Score':<8}")
    print("-" * 120)
    
    for res in results:
        print(f"{res['Model Version']:<30} | {res['Parameters (M)']:<12} | {res['FPS']:<8} | {res['GFLOPs']:<8} | {res['mIoU']:<8} | {res['Precision']:<10} | {res['Recall']:<8} | {res['F1 Score']:<8}")
    print("="*120)
    
    # Save to CSV manually
    csv_path = os.path.join(args.output_dir, 'ablation_comparison.csv')
    with open(csv_path, 'w') as f:
        f.write("Model Version,Parameters (M),FPS,GFLOPs,mIoU,Precision,Recall,F1 Score\n")
        for res in results:
            f.write(f"{res['Model Version']},{res['Parameters (M)']},{res['FPS']},{res['GFLOPs']},{res['mIoU']},{res['Precision']},{res['Recall']},{res['F1 Score']}\n")
            
    print(f"\nResults saved to {csv_path}")

if __name__ == "__main__":
    main()
