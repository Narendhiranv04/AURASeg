import os
import sys
import json
import csv
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import subprocess

sys.path.insert(0, str(Path(__file__).parent))
from auraseg_r18_wacv import AURASeg_R18_WACV
from unified_dataset import Normalization, UnifiedDrivableAreaDataset
from wacv_metrics import compute_metrics, compute_boundary_metrics
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

def get_git_commit():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.STDOUT).decode('utf-8').strip()
    except Exception:
        return "unknown"

def run_evaluation():
    print("=" * 60)
    print("WACV BOUNDARY TOLERANCE SENSITIVITY EXPERIMENT")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Configuration
    repo_root = Path(__file__).parent.parent
    checkpoint_path = repo_root / "runs_wacv_new/r18_mul_full_sobel_gate_seed42/checkpoints/best.pth"
    data_root = repo_root / "CommonDataset"
    
    out_dir = repo_root / "runs_wacv_new/boundary_tolerance_reference"
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_dir = out_dir / "predictions"
    preds_dir.mkdir(exist_ok=True)
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Instantiate Model
    model = AURASeg_R18_WACV(
        num_classes=2,
        encoder_weights=None,
        fusion_type='mul',
        attention_mode='full',
        use_sobel=True,
        use_gate=True,
        simple_boundary_head=False
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Dataset Preparation
    normalization = Normalization(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    
    val_dataset = UnifiedDrivableAreaDataset(
        dataset_root=data_root, split='val',
        img_size=(384, 640), transform=False,
        normalization=normalization, return_names=False
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=4,
        shuffle=False, num_workers=4,
        pin_memory=True, drop_last=False
    )
    
    print(f"Dataset Size: {len(val_dataset)} images.")
    
    # INFERENCE
    all_preds = []
    all_targets = []
    
    print("\nRunning Inference...")
    with torch.no_grad():
        for images, masks in tqdm(val_loader):
            images, masks = images.to(device), masks.to(device)
            with autocast():
                outputs = model(images, return_aux=False, return_boundary=False)
            
            preds = torch.argmax(outputs['main'], dim=1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(masks.cpu().numpy())
            
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # METRICS EVALUATION
    print("\nEvaluating Standard Segmentation Metrics...")
    seg_metrics = compute_metrics(all_preds, all_targets)
    
    print("Evaluating Boundary Metrics...")
    k_values = [1, 2, 3]
    boundary_results = {}
    
    for k in k_values:
        print(f"  Computing k={k}...")
        b_metrics = compute_boundary_metrics(all_preds, all_targets, k=k)
        boundary_results[k] = b_metrics
        
    # PARITY CHECK
    print("\n" + "=" * 60)
    print("PARITY CHECK (k=2)")
    print("=" * 60)
    
    ref_iou = 0.8126085301
    ref_f1 = 0.8902360057
    
    actual_iou = boundary_results[2]['boundary_iou']
    actual_f1 = boundary_results[2]['boundary_f1']
    
    diff_iou = abs(actual_iou - ref_iou)
    diff_f1 = abs(actual_f1 - ref_f1)
    
    print(f"Expected BIoU: {ref_iou:.10f} | Actual: {actual_iou:.10f} | Diff: {diff_iou:.10e}")
    print(f"Expected BF1:  {ref_f1:.10f} | Actual: {actual_f1:.10f} | Diff: {diff_f1:.10e}")
    
    if diff_iou > 1e-4 or diff_f1 > 1e-4:
        print("\n[FAIL] k=2 Parity Check Failed. Halting.")
        sys.exit(1)
        
    print("\n[PASS] k=2 Parity Check Passed.")
    
    # SAVE CONFIG & RESULTS
    config = {
        'checkpoint_path': str(checkpoint_path),
        'model_configuration': {
            'fusion_type': 'mul',
            'attention_mode': 'full',
            'use_sobel': True,
            'use_gate': True,
            'simple_boundary_head': False
        },
        'dataset_path': str(data_root),
        'split': 'val',
        'resolution': (384, 640),
        'boundary_gradient_kernel': '3x3',
        'k_values': k_values,
        'timestamp': datetime.now().isoformat(),
        'git_commit': get_git_commit()
    }
    
    with open(out_dir / "evaluation_config.json", 'w') as f:
        json.dump(config, f, indent=4)
        
    results_json = {
        'segmentation': {
            'iou_drivable': float(seg_metrics['iou_drivable']),
            'f1': float(seg_metrics['f1']),
            'precision': float(seg_metrics['precision']),
            'recall': float(seg_metrics['recall'])
        },
        'boundary_tolerance': {
            k: {
                'boundary_iou': float(boundary_results[k]['boundary_iou']),
                'boundary_precision': float(boundary_results[k]['boundary_precision']),
                'boundary_recall': float(boundary_results[k]['boundary_recall']),
                'boundary_f1': float(boundary_results[k]['boundary_f1'])
            } for k in k_values
        }
    }
    
    with open(out_dir / "boundary_tolerance_results.json", 'w') as f:
        json.dump(results_json, f, indent=4)
        
    csv_path = out_dir / "boundary_tolerance_results.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['k', 'boundary_iou', 'boundary_precision', 'boundary_recall', 'boundary_f1'])
        for k in k_values:
            writer.writerow([
                k,
                boundary_results[k]['boundary_iou'],
                boundary_results[k]['boundary_precision'],
                boundary_results[k]['boundary_recall'],
                boundary_results[k]['boundary_f1']
            ])
            
    print("\n" + "=" * 60)
    print("FINAL TABLE")
    print("-" * 60)
    print(f"{'k':<8} {'BIoU':<12} {'BPrec':<12} {'BRec':<12} {'BF1':<12}")
    print("-" * 60)
    for k in k_values:
        res = boundary_results[k]
        marker = " (primary)" if k == 2 else ""
        print(f"{str(k)+marker:<8} {res['boundary_iou']:<12.6f} {res['boundary_precision']:<12.6f} {res['boundary_recall']:<12.6f} {res['boundary_f1']:<12.6f}")
    print("-" * 60)

if __name__ == "__main__":
    run_evaluation()
