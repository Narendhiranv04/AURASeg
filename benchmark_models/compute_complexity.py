"""
Model Complexity Analysis: Parameters, GFLOPs, and FPS
=======================================================

Computes computational complexity metrics for all models.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import time
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from model_factory import get_benchmark_model
from auraseg_v4_resnet import AURASeg_V4_ResNet50

try:
    from thop import profile, clever_format
    HAS_THOP = True
except ImportError:
    HAS_THOP = False
    print("Warning: thop not installed. Install with: pip install thop")
    print("Will use manual FLOP estimation.\n")


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def estimate_gflops_manual(model, input_size=(1, 3, 384, 640), device='cuda'):
    """
    Estimate GFLOPs by measuring forward pass time and using empirical relationships.
    This is a fallback when thop is not available.
    """
    model = model.to(device)
    model.eval()
    
    x = torch.randn(*input_size, device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(x)
    
    torch.cuda.synchronize()
    
    # Time forward pass
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    with torch.no_grad():
        start_event.record()
        for _ in range(100):
            _ = model(x)
        end_event.record()
    
    torch.cuda.synchronize()
    avg_time_ms = start_event.elapsed_time(end_event) / 100
    
    return None, avg_time_ms  # Return None for GFLOPs when can't compute


def compute_gflops(model, input_size=(1, 3, 384, 640), device='cuda'):
    """Compute GFLOPs using thop library."""
    if not HAS_THOP:
        return None
    
    model = model.to(device)
    model.eval()
    
    x = torch.randn(*input_size, device=device)
    
    try:
        # Disable gradient for profiling
        with torch.no_grad():
            macs, params = profile(model, inputs=(x,), verbose=False)
        gflops = macs / 1e9
        return gflops
    except Exception as e:
        print(f"  Error computing GFLOPs: {e}")
        return None


def compute_fps(model, input_size=(1, 3, 384, 640), device='cuda', 
                warmup_iters=50, test_iters=200, is_auraseg=False):
    """
    Compute FPS (Frames Per Second) for a model.
    """
    model = model.to(device)
    model.eval()
    
    x = torch.randn(*input_size, device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_iters):
            if is_auraseg:
                _ = model(x, return_aux=False, return_boundary=False)
            else:
                _ = model(x)
    
    torch.cuda.synchronize()
    
    # Measure time
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    with torch.no_grad():
        start_event.record()
        for _ in range(test_iters):
            if is_auraseg:
                _ = model(x, return_aux=False, return_boundary=False)
            else:
                _ = model(x)
        end_event.record()
    
    torch.cuda.synchronize()
    
    total_time_ms = start_event.elapsed_time(end_event)
    avg_time_ms = total_time_ms / test_iters
    fps = 1000.0 / avg_time_ms
    
    return fps, avg_time_ms


class AURASeg_Wrapper(nn.Module):
    """Wrapper to make AURASeg compatible with thop profiling."""
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        out = self.model(x, return_aux=False, return_boundary=False)
        return out['main']


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cpu':
        print("WARNING: Running on CPU. FPS measurements will be much slower than GPU.")
    
    input_size = (1, 3, 384, 640)  # Batch=1, 3 channels, 384x640
    print(f"Input size: {input_size}")
    print()
    
    # Model configurations
    models_config = {
        'FCN-R50': {
            'type': 'benchmark',
            'name': 'fcn',
            'encoder': 'ResNet-50'
        },
        'PSPNet-R50': {
            'type': 'benchmark',
            'name': 'pspnet',
            'encoder': 'ResNet-50'
        },
        'DeepLabV3+': {
            'type': 'benchmark',
            'name': 'deeplabv3plus',
            'encoder': 'ResNet-50'
        },
        'UPerNet-R50': {
            'type': 'benchmark',
            'name': 'upernet',
            'encoder': 'ResNet-50'
        },
        'SegFormer-B2': {
            'type': 'benchmark', 
            'name': 'segformer',
            'encoder': 'MiT-B2'
        },
        'FPN-MiTB3': {
            'type': 'benchmark',
            'name': 'mask2former',
            'encoder': 'MiT-B3'
        },
        'PIDNet-L': {
            'type': 'benchmark',
            'name': 'pidnet',
            'encoder': 'Custom'
        },
        'AURASeg V4-R50': {
            'type': 'auraseg',
            'encoder': 'ResNet-50'
        }
    }
    
    results = {}
    
    print("="*80)
    print("MODEL COMPLEXITY ANALYSIS")
    print("="*80)
    
    for model_name, config in models_config.items():
        print(f"\nAnalyzing: {model_name}")
        print("-" * 40)
        
        # Create model
        if config['type'] == 'benchmark':
            model, _ = get_benchmark_model(config['name'], num_classes=2)
        else:
            model = AURASeg_V4_ResNet50(num_classes=2)
        
        model = model.to(device)
        model.eval()
        
        # Count parameters
        total_params, trainable_params = count_parameters(model)
        print(f"  Parameters: {total_params / 1e6:.2f}M")
        
        # Compute GFLOPs
        if config['type'] == 'auraseg':
            # Wrap for profiling
            wrapped_model = AURASeg_Wrapper(model)
            gflops = compute_gflops(wrapped_model, input_size, device)
        else:
            gflops = compute_gflops(model, input_size, device)
        
        if gflops is not None:
            print(f"  GFLOPs: {gflops:.2f}")
        else:
            print(f"  GFLOPs: N/A (install thop)")
        
        # Compute FPS
        is_auraseg = config['type'] == 'auraseg'
        fps, latency_ms = compute_fps(model, input_size, device, is_auraseg=is_auraseg)
        print(f"  FPS: {fps:.1f}")
        print(f"  Latency: {latency_ms:.2f} ms")
        
        results[model_name] = {
            'encoder': config['encoder'],
            'params_M': total_params / 1e6,
            'gflops': gflops,
            'fps': fps,
            'latency_ms': latency_ms
        }
        
        # Clear memory
        del model
        if config['type'] == 'auraseg':
            del wrapped_model
        torch.cuda.empty_cache()
    
    # Print summary table
    print("\n")
    print("="*100)
    print("MODEL COMPLEXITY COMPARISON TABLE")
    print("="*100)
    print(f"{'Model':<20} | {'Encoder':<12} | {'Params (M)':<12} | {'GFLOPs':<12} | {'FPS':<10} | {'Latency (ms)':<12}")
    print("-"*100)
    
    for model_name, data in results.items():
        gflops_str = f"{data['gflops']:.2f}" if data['gflops'] is not None else "N/A"
        print(f"{model_name:<20} | {data['encoder']:<12} | {data['params_M']:<12.2f} | {gflops_str:<12} | {data['fps']:<10.1f} | {data['latency_ms']:<12.2f}")
    
    print("="*100)
    
    # Find best values
    min_params = min(r['params_M'] for r in results.values())
    min_gflops = min(r['gflops'] for r in results.values() if r['gflops'] is not None) if any(r['gflops'] for r in results.values()) else None
    max_fps = max(r['fps'] for r in results.values())
    
    print("\nNotes:")
    print(f"  - Input resolution: 384 × 640")
    print(f"  - Batch size: 1")
    print(f"  - Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"  - Fastest model (FPS): {max([k for k, v in results.items() if v['fps'] == max_fps])}")
    print(f"  - Smallest model: {max([k for k, v in results.items() if v['params_M'] == min_params])}")
    
    # Save results
    output_path = Path(__file__).parent.parent / "runs" / "complexity_comparison.txt"
    with open(output_path, 'w') as f:
        f.write("MODEL COMPLEXITY COMPARISON\n")
        f.write("="*100 + "\n")
        f.write(f"Input resolution: 384 × 640\n")
        f.write(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n")
        f.write("\n")
        f.write(f"{'Model':<20} | {'Encoder':<12} | {'Params (M)':<12} | {'GFLOPs':<12} | {'FPS':<10} | {'Latency (ms)':<12}\n")
        f.write("-"*100 + "\n")
        for model_name, data in results.items():
            gflops_str = f"{data['gflops']:.2f}" if data['gflops'] is not None else "N/A"
            f.write(f"{model_name:<20} | {data['encoder']:<12} | {data['params_M']:<12.2f} | {gflops_str:<12} | {data['fps']:<10.1f} | {data['latency_ms']:<12.2f}\n")
        f.write("="*100 + "\n")
    
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
