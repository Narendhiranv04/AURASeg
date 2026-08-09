"""
Complete Deployment Benchmark for Table 4
==========================================

Combines all benchmark models + AURASeg variants for the final deployment table.

Metrics:
- Params (M)
- GFLOPs
- Peak Memory (MB)
- Latency (ms)
- FPS
"""

import sys
import os
import gc
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from model_factory import get_benchmark_model
from auraseg_exportable import AURASeg_V4_ResNet

try:
    from thop import profile
    HAS_THOP = True
except ImportError:
    HAS_THOP = False

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False


def count_parameters(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def compute_gflops(model, input_size=(1, 3, 384, 640), device='cuda'):
    if not HAS_THOP:
        return None
    
    model = model.to(device).eval()
    x = torch.randn(*input_size, device=device)
    
    try:
        with torch.no_grad():
            macs, _ = profile(model, inputs=(x,), verbose=False)
        return macs / 1e9
    except:
        return None


def measure_peak_memory(model, input_size=(1, 3, 384, 640), device='cuda',
                        warmup_iters=10, test_iters=50):
    if not torch.cuda.is_available():
        return None
    
    model = model.to(device).eval()
    x = torch.randn(*input_size, device=device)
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(x)
    
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
    with torch.no_grad():
        for _ in range(test_iters):
            _ = model(x)
            torch.cuda.synchronize()
    
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def measure_latency_fps(model, input_size=(1, 3, 384, 640), device='cuda',
                        warmup_iters=50, test_iters=200):
    model = model.to(device).eval()
    x = torch.randn(*input_size, device=device)
    
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(x)
    
    torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    with torch.no_grad():
        start.record()
        for _ in range(test_iters):
            _ = model(x)
        end.record()
    
    torch.cuda.synchronize()
    
    total_ms = start.elapsed_time(end)
    latency = total_ms / test_iters
    fps = 1000.0 / latency
    
    return latency, fps


class FCNWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        out = self.model(x)
        return out['out'] if isinstance(out, dict) else out


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_size = (1, 3, 384, 640)
    
    print("=" * 90)
    print("TABLE 4: ON-DEVICE DEPLOYMENT COMPARISON")
    print("=" * 90)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Input Resolution: 384 × 640")
    print("=" * 90)
    print()
    
    # Models configuration
    models = [
        # Benchmark models
        {'name': 'FCN', 'type': 'benchmark', 'factory': 'fcn', 'wrap': True},
        {'name': 'DeepLabV3+', 'type': 'benchmark', 'factory': 'deeplabv3plus', 'wrap': False},
        {'name': 'UPerNet-R50', 'type': 'benchmark', 'factory': 'upernet', 'wrap': False},
        {'name': 'SegFormer-B2', 'type': 'benchmark', 'factory': 'segformer', 'wrap': False},
        {'name': 'PIDNet-S', 'type': 'benchmark', 'factory': 'pidnet', 'wrap': False},
        # AURASeg variants
        {'name': 'AURASeg-R18', 'type': 'auraseg', 'backbone': 'resnet18'},
        {'name': 'AURASeg-R50', 'type': 'auraseg', 'backbone': 'resnet50'},
    ]
    
    results = []
    
    for config in models:
        name = config['name']
        print(f"Benchmarking: {name}")
        print("-" * 50)
        
        try:
            # Create model
            if config['type'] == 'benchmark':
                model, _ = get_benchmark_model(config['factory'], num_classes=2)
                if config.get('wrap', False):
                    model = FCNWrapper(model)
            else:
                model = AURASeg_V4_ResNet(
                    backbone=config['backbone'],
                    num_classes=2
                )
            
            model = model.to(device).eval()
            
            # Metrics
            params = count_parameters(model)
            gflops = compute_gflops(model, input_size, device)
            memory = measure_peak_memory(model, input_size, device)
            latency, fps = measure_latency_fps(model, input_size, device)
            
            print(f"  Params: {params:.2f}M")
            print(f"  GFLOPs: {gflops:.2f}" if gflops else "  GFLOPs: N/A")
            print(f"  Peak Memory: {memory:.1f} MB" if memory else "  Peak Memory: N/A")
            print(f"  Latency: {latency:.2f} ms")
            print(f"  FPS: {fps:.1f}")
            
            results.append({
                'name': name,
                'params': params,
                'gflops': gflops,
                'memory': memory,
                'latency': latency,
                'fps': fps
            })
            
            del model
            torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
        
        print()
    
    # Print final table
    print("\n")
    print("=" * 95)
    print("Table 4. On-device deployment comparison")
    print("=" * 95)
    print(f"{'Model':<15} | {'Params':<10} | {'GFLOPs':<10} | {'Peak Memory':<12} | {'Latency':<12} | {'FPS':<8}")
    print("-" * 95)
    
    for r in results:
        params = f"{r['params']:.2f}M"
        gflops = f"{r['gflops']:.2f}" if r['gflops'] else "N/A"
        memory = f"{r['memory']:.1f} MB" if r['memory'] else "N/A"
        latency = f"{r['latency']:.2f} ms"
        fps = f"{r['fps']:.1f}"
        
        print(f"{r['name']:<15} | {params:<10} | {gflops:<10} | {memory:<12} | {latency:<12} | {fps:<8}")
    
    print("=" * 95)
    
    # LaTeX table
    print("\n% LaTeX Table for Paper")
    print("\\begin{table}[t]")
    print("\\centering")
    print("\\caption{On-device deployment comparison on NVIDIA Jetson Nano 4GB.}")
    print("\\label{tab:deployment}")
    print("\\begin{tabular}{lccccc}")
    print("\\toprule")
    print("Model & Params & GFLOPs & Peak Memory & Latency & FPS \\\\")
    print("\\midrule")
    
    for r in results:
        name = r['name'].replace('-', ' ').replace('_', ' ')
        params = f"{r['params']:.2f}M"
        gflops = f"{r['gflops']:.2f}" if r['gflops'] else "--"
        memory = f"{r['memory']:.0f} MB" if r['memory'] else "--"
        latency = f"{r['latency']:.1f} ms"
        fps = f"{r['fps']:.1f}"
        
        print(f"{name} & {params} & {gflops} & {memory} & {latency} & {fps} \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    
    # Save results
    output_dir = Path(__file__).parent.parent / "runs"
    output_dir.mkdir(exist_ok=True)
    
    # CSV
    csv_path = output_dir / "table4_deployment_comparison.csv"
    with open(csv_path, 'w') as f:
        f.write("Model,Params (M),GFLOPs,Peak Memory (MB),Latency (ms),FPS\n")
        for r in results:
            gflops = f"{r['gflops']:.2f}" if r['gflops'] else ""
            memory = f"{r['memory']:.1f}" if r['memory'] else ""
            f.write(f"{r['name']},{r['params']:.2f},{gflops},{memory},{r['latency']:.2f},{r['fps']:.1f}\n")
    
    print(f"\n✓ CSV saved to: {csv_path}")
    
    # Text report
    txt_path = output_dir / "table4_deployment_comparison.txt"
    with open(txt_path, 'w') as f:
        f.write("Table 4. On-device deployment comparison\n")
        f.write("=" * 95 + "\n")
        f.write(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n")
        f.write(f"Input Resolution: 384 × 640\n")
        f.write("=" * 95 + "\n\n")
        
        f.write(f"{'Model':<15} | {'Params':<10} | {'GFLOPs':<10} | {'Peak Memory':<12} | {'Latency':<12} | {'FPS':<8}\n")
        f.write("-" * 95 + "\n")
        
        for r in results:
            params = f"{r['params']:.2f}M"
            gflops = f"{r['gflops']:.2f}" if r['gflops'] else "N/A"
            memory = f"{r['memory']:.1f} MB" if r['memory'] else "N/A"
            latency = f"{r['latency']:.2f} ms"
            fps = f"{r['fps']:.1f}"
            f.write(f"{r['name']:<15} | {params:<10} | {gflops:<10} | {memory:<12} | {latency:<12} | {fps:<8}\n")
        
        f.write("=" * 95 + "\n")
    
    print(f"✓ TXT saved to: {txt_path}")


if __name__ == "__main__":
    main()
