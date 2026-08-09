"""
On-Device Deployment Benchmark
==============================

Calculates deployment metrics for Table 4:
- Params (M): Number of parameters in millions
- GFLOPs: Giga Floating Point Operations
- Peak Memory (MB): Peak GPU memory usage during inference
- Latency (ms): Time for single inference
- FPS: Frames per second

Usage:
    python deployment_benchmark.py [--include-auraseg]
"""

import sys
import os
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import time
import gc
import numpy as np
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))
from model_factory import get_benchmark_model

try:
    from thop import profile, clever_format
    HAS_THOP = True
except ImportError:
    HAS_THOP = False
    print("Warning: thop not installed. Install with: pip install thop")
    print("GFLOPs will be estimated using ptflops if available.\n")

try:
    from ptflops import get_model_complexity_info
    HAS_PTFLOPS = True
except ImportError:
    HAS_PTFLOPS = False


def count_parameters(model):
    """Count total parameters in millions."""
    total = sum(p.numel() for p in model.parameters())
    return total / 1e6


def compute_gflops_thop(model, input_size=(1, 3, 384, 640), device='cuda'):
    """Compute GFLOPs using thop library."""
    if not HAS_THOP:
        return None
    
    model = model.to(device)
    model.eval()
    x = torch.randn(*input_size, device=device)
    
    try:
        with torch.no_grad():
            macs, params = profile(model, inputs=(x,), verbose=False)
        gflops = macs / 1e9
        return gflops
    except Exception as e:
        print(f"  thop error: {e}")
        return None


def compute_gflops_ptflops(model, input_size=(3, 384, 640), device='cuda'):
    """Compute GFLOPs using ptflops library (backup)."""
    if not HAS_PTFLOPS:
        return None
    
    model = model.to(device)
    model.eval()
    
    try:
        macs, params = get_model_complexity_info(
            model, input_size, 
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False
        )
        gflops = macs / 1e9
        return gflops
    except Exception as e:
        print(f"  ptflops error: {e}")
        return None


def compute_gflops(model, input_size=(1, 3, 384, 640), device='cuda'):
    """Compute GFLOPs using available library."""
    # Try thop first
    gflops = compute_gflops_thop(model, input_size, device)
    if gflops is not None:
        return gflops
    
    # Fallback to ptflops
    gflops = compute_gflops_ptflops(model, input_size[1:], device)
    return gflops


def measure_peak_memory(model, input_size=(1, 3, 384, 640), device='cuda', 
                        warmup_iters=10, test_iters=50):
    """
    Measure peak GPU memory usage during inference.
    Returns peak memory in MB.
    """
    if device == 'cpu' or not torch.cuda.is_available():
        return None
    
    model = model.to(device)
    model.eval()
    
    x = torch.randn(*input_size, device=device)
    
    # Clear cache and reset peak stats
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    
    # Record baseline memory (model loaded)
    baseline_memory = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(x)
    
    torch.cuda.synchronize()
    
    # Reset peak stats after warmup
    torch.cuda.reset_peak_memory_stats()
    
    # Measure peak memory during inference
    with torch.no_grad():
        for _ in range(test_iters):
            _ = model(x)
            torch.cuda.synchronize()
    
    # Get peak memory
    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
    
    return peak_memory


def measure_latency_and_fps(model, input_size=(1, 3, 384, 640), device='cuda',
                            warmup_iters=50, test_iters=200):
    """
    Measure latency (ms) and FPS.
    Returns (fps, latency_ms).
    """
    model = model.to(device)
    model.eval()
    
    x = torch.randn(*input_size, device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(x)
    
    if device == 'cuda' or (isinstance(device, torch.device) and device.type == 'cuda'):
        torch.cuda.synchronize()
        
        # Use CUDA events for precise timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        with torch.no_grad():
            start_event.record()
            for _ in range(test_iters):
                _ = model(x)
            end_event.record()
        
        torch.cuda.synchronize()
        total_time_ms = start_event.elapsed_time(end_event)
    else:
        # CPU timing
        with torch.no_grad():
            start_time = time.perf_counter()
            for _ in range(test_iters):
                _ = model(x)
            end_time = time.perf_counter()
        total_time_ms = (end_time - start_time) * 1000
    
    avg_latency_ms = total_time_ms / test_iters
    fps = 1000.0 / avg_latency_ms
    
    return fps, avg_latency_ms


class FCNWrapper(nn.Module):
    """Wrapper for FCN to get single tensor output."""
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        out = self.model(x)
        if isinstance(out, dict):
            return out['out']
        return out


class AURASeg_Wrapper(nn.Module):
    """Wrapper for AURASeg to get single tensor output."""
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        out = self.model(x, return_aux=False, return_boundary=False)
        if isinstance(out, dict):
            return out['main']
        return out


def run_benchmark(include_auraseg=False):
    """Run deployment benchmark for all models."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 80)
    print("ON-DEVICE DEPLOYMENT BENCHMARK")
    print("=" * 80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Input Resolution: 384 × 640")
    print(f"Batch Size: 1")
    print("=" * 80)
    print()
    
    # Models to benchmark (matching Table 4 in the paper)
    # Order: FCN, DeepLabV3+, UPerNet-R50, SegFormer-B2, PIDNet-S, AURASeg
    models_config = [
        {
            'display_name': 'FCN',
            'factory_name': 'fcn',
            'is_auraseg': False,
            'needs_wrapper': True,  # FCN returns dict
        },
        {
            'display_name': 'DeepLabV3+',
            'factory_name': 'deeplabv3plus',
            'is_auraseg': False,
            'needs_wrapper': False,
        },
        {
            'display_name': 'UPerNet-R50',
            'factory_name': 'upernet',
            'is_auraseg': False,
            'needs_wrapper': False,
        },
        {
            'display_name': 'SegFormer-B2',
            'factory_name': 'segformer',
            'is_auraseg': False,
            'needs_wrapper': False,
        },
        {
            'display_name': 'PIDNet-S',
            'factory_name': 'pidnet',
            'is_auraseg': False,
            'needs_wrapper': False,
        },
    ]
    
    # Optionally add AURASeg
    if include_auraseg:
        models_config.append({
            'display_name': 'AURASeg',
            'factory_name': None,  # Special handling
            'is_auraseg': True,
            'needs_wrapper': True,
        })
    
    results = []
    input_size = (1, 3, 384, 640)
    
    for config in models_config:
        model_name = config['display_name']
        print(f"Benchmarking: {model_name}")
        print("-" * 50)
        
        try:
            # Create model
            if config['is_auraseg']:
                from auraseg_v4_resnet import AURASeg_V4_ResNet50
                model = AURASeg_V4_ResNet50(num_classes=2)
            else:
                model, info = get_benchmark_model(config['factory_name'], num_classes=2)
            
            model = model.to(device)
            model.eval()
            
            # Create wrapped model for GFLOPs computation if needed
            if config['needs_wrapper']:
                if config['is_auraseg']:
                    wrapped_model = AURASeg_Wrapper(model)
                else:
                    wrapped_model = FCNWrapper(model)
            else:
                wrapped_model = model
            
            # 1. Count Parameters
            params_m = count_parameters(model)
            print(f"  Params: {params_m:.2f}M")
            
            # 2. Compute GFLOPs
            gflops = compute_gflops(wrapped_model, input_size, device)
            if gflops is not None:
                print(f"  GFLOPs: {gflops:.2f}")
            else:
                print(f"  GFLOPs: N/A")
            
            # 3. Measure Peak Memory
            if config['is_auraseg']:
                # Special handling for AURASeg forward
                peak_memory = measure_peak_memory_auraseg(model, input_size, device)
            else:
                peak_memory = measure_peak_memory(wrapped_model, input_size, device)
            
            if peak_memory is not None:
                print(f"  Peak Memory: {peak_memory:.1f} MB")
            else:
                print(f"  Peak Memory: N/A (CPU mode)")
            
            # 4. Measure Latency and FPS
            if config['is_auraseg']:
                fps, latency = measure_latency_fps_auraseg(model, input_size, device)
            else:
                fps, latency = measure_latency_and_fps(wrapped_model, input_size, device)
            
            print(f"  Latency: {latency:.2f} ms")
            print(f"  FPS: {fps:.1f}")
            
            results.append({
                'model': model_name,
                'params_m': params_m,
                'gflops': gflops,
                'peak_memory_mb': peak_memory,
                'latency_ms': latency,
                'fps': fps
            })
            
            # Cleanup
            del model
            if config['needs_wrapper']:
                del wrapped_model
            torch.cuda.empty_cache()
            gc.collect()
            
            print()
            
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'model': model_name,
                'params_m': None,
                'gflops': None,
                'peak_memory_mb': None,
                'latency_ms': None,
                'fps': None
            })
            print()
    
    return results


def measure_peak_memory_auraseg(model, input_size, device, warmup_iters=10, test_iters=50):
    """Special peak memory measurement for AURASeg."""
    if device == 'cpu' or not torch.cuda.is_available():
        return None
    
    model = model.to(device)
    model.eval()
    x = torch.randn(*input_size, device=device)
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(x, return_aux=False, return_boundary=False)
    
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
    # Measure
    with torch.no_grad():
        for _ in range(test_iters):
            _ = model(x, return_aux=False, return_boundary=False)
            torch.cuda.synchronize()
    
    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
    return peak_memory


def measure_latency_fps_auraseg(model, input_size, device, warmup_iters=50, test_iters=200):
    """Special latency/FPS measurement for AURASeg."""
    model = model.to(device)
    model.eval()
    x = torch.randn(*input_size, device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(x, return_aux=False, return_boundary=False)
    
    torch.cuda.synchronize()
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    with torch.no_grad():
        start_event.record()
        for _ in range(test_iters):
            _ = model(x, return_aux=False, return_boundary=False)
        end_event.record()
    
    torch.cuda.synchronize()
    total_time_ms = start_event.elapsed_time(end_event)
    
    avg_latency_ms = total_time_ms / test_iters
    fps = 1000.0 / avg_latency_ms
    
    return fps, avg_latency_ms


def print_table(results):
    """Print results in a nicely formatted table."""
    print("\n")
    print("=" * 90)
    print("Table 4. On-device deployment comparison")
    print("=" * 90)
    
    # Header
    header = f"{'Model':<15} | {'Params':<8} | {'GFLOPs':<8} | {'Peak Memory':<12} | {'Latency':<10} | {'FPS':<8}"
    print(header)
    print("-" * 90)
    
    for r in results:
        params_str = f"{r['params_m']:.2f}M" if r['params_m'] is not None else "xx.xx"
        gflops_str = f"{r['gflops']:.2f}" if r['gflops'] is not None else "xx.xx"
        memory_str = f"{r['peak_memory_mb']:.1f} MB" if r['peak_memory_mb'] is not None else "xx.xx"
        latency_str = f"{r['latency_ms']:.2f} ms" if r['latency_ms'] is not None else "xx.xx"
        fps_str = f"{r['fps']:.1f}" if r['fps'] is not None else "xx.xx"
        
        row = f"{r['model']:<15} | {params_str:<8} | {gflops_str:<8} | {memory_str:<12} | {latency_str:<10} | {fps_str:<8}"
        print(row)
    
    print("=" * 90)


def print_latex_table(results):
    """Print results as LaTeX table."""
    print("\n")
    print("% LaTeX Table")
    print("\\begin{table}[t]")
    print("\\centering")
    print("\\caption{On-device deployment comparison on NVIDIA Jetson Nano 4GB.}")
    print("\\label{tab:deployment}")
    print("\\begin{tabular}{lccccc}")
    print("\\toprule")
    print("Model & Params & GFLOPs & Peak Memory & Latency & FPS \\\\")
    print("\\midrule")
    
    for r in results:
        params_str = f"{r['params_m']:.2f}M" if r['params_m'] is not None else "xx.xx"
        gflops_str = f"{r['gflops']:.2f}" if r['gflops'] is not None else "xx.xx"
        memory_str = f"{r['peak_memory_mb']:.0f} MB" if r['peak_memory_mb'] is not None else "xx.xx"
        latency_str = f"{r['latency_ms']:.1f} ms" if r['latency_ms'] is not None else "xx.xx"
        fps_str = f"{r['fps']:.1f}" if r['fps'] is not None else "xx.xx"
        
        model_name = r['model'].replace('_', '\\_')
        print(f"{model_name} & {params_str} & {gflops_str} & {memory_str} & {latency_str} & {fps_str} \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


def save_results(results, output_path):
    """Save results to a text file."""
    with open(output_path, 'w') as f:
        f.write("ON-DEVICE DEPLOYMENT BENCHMARK RESULTS\n")
        f.write("=" * 90 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n")
        f.write(f"Input Resolution: 384 × 640\n")
        f.write(f"Batch Size: 1\n")
        f.write("=" * 90 + "\n\n")
        
        # Header
        header = f"{'Model':<15} | {'Params':<8} | {'GFLOPs':<8} | {'Peak Memory':<12} | {'Latency':<10} | {'FPS':<8}\n"
        f.write(header)
        f.write("-" * 90 + "\n")
        
        for r in results:
            params_str = f"{r['params_m']:.2f}M" if r['params_m'] is not None else "N/A"
            gflops_str = f"{r['gflops']:.2f}" if r['gflops'] is not None else "N/A"
            memory_str = f"{r['peak_memory_mb']:.1f} MB" if r['peak_memory_mb'] is not None else "N/A"
            latency_str = f"{r['latency_ms']:.2f} ms" if r['latency_ms'] is not None else "N/A"
            fps_str = f"{r['fps']:.1f}" if r['fps'] is not None else "N/A"
            
            row = f"{r['model']:<15} | {params_str:<8} | {gflops_str:<8} | {memory_str:<12} | {latency_str:<10} | {fps_str:<8}\n"
            f.write(row)
        
        f.write("=" * 90 + "\n")
    
    print(f"\nResults saved to: {output_path}")


def save_csv(results, output_path):
    """Save results as CSV for easy import."""
    with open(output_path, 'w') as f:
        f.write("Model,Params (M),GFLOPs,Peak Memory (MB),Latency (ms),FPS\n")
        for r in results:
            params = f"{r['params_m']:.2f}" if r['params_m'] is not None else ""
            gflops = f"{r['gflops']:.2f}" if r['gflops'] is not None else ""
            memory = f"{r['peak_memory_mb']:.1f}" if r['peak_memory_mb'] is not None else ""
            latency = f"{r['latency_ms']:.2f}" if r['latency_ms'] is not None else ""
            fps = f"{r['fps']:.1f}" if r['fps'] is not None else ""
            f.write(f"{r['model']},{params},{gflops},{memory},{latency},{fps}\n")
    
    print(f"CSV saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='On-device Deployment Benchmark')
    parser.add_argument('--include-auraseg', action='store_true', 
                        help='Include AURASeg model in benchmark')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for results')
    args = parser.parse_args()
    
    # Run benchmark
    results = run_benchmark(include_auraseg=args.include_auraseg)
    
    # Print formatted table
    print_table(results)
    
    # Print LaTeX table
    print_latex_table(results)
    
    # Save results
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent.parent / "runs"
    output_dir.mkdir(exist_ok=True)
    
    save_results(results, output_dir / "deployment_benchmark_results.txt")
    save_csv(results, output_dir / "deployment_benchmark_results.csv")
    
    print("\n✓ Benchmark complete!")


if __name__ == "__main__":
    main()
