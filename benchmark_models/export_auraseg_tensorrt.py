"""
AURASeg ONNX Export and TensorRT Benchmark
============================================

Exports AURASeg models to ONNX format and benchmarks with TensorRT.

Metrics computed:
- Params (M): Number of parameters
- GFLOPs: Computational complexity
- Peak Memory (MB): GPU memory during inference
- Latency (ms): Inference time
- FPS: Frames per second

Usage:
    python export_auraseg_tensorrt.py
    python export_auraseg_tensorrt.py --skip-tensorrt  # ONNX only
"""

import sys
import os
import argparse
import gc
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np
import subprocess
import shutil

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))
from auraseg_exportable import AURASeg_V4_ResNet, auraseg_resnet18, auraseg_resnet50

try:
    from thop import profile
    HAS_THOP = True
except ImportError:
    HAS_THOP = False
    print("Warning: thop not installed. Install with: pip install thop")

try:
    import onnx
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
    print("Warning: onnx/onnxruntime not installed.")
    print("Install with: pip install onnx onnxruntime-gpu")

try:
    import tensorrt as trt
    HAS_TRT = True
    TRT_VERSION = trt.__version__
except ImportError:
    HAS_TRT = False
    TRT_VERSION = None
    print("Warning: TensorRT not installed.")

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    HAS_PYCUDA = True
except ImportError:
    HAS_PYCUDA = False


# =============================================================================
# Model Export Functions
# =============================================================================

def count_parameters(model):
    """Count parameters in millions."""
    return sum(p.numel() for p in model.parameters()) / 1e6


def compute_gflops(model, input_size=(1, 3, 384, 640), device='cuda'):
    """Compute GFLOPs using thop."""
    if not HAS_THOP:
        return None
    
    model = model.to(device).eval()
    x = torch.randn(*input_size, device=device)
    
    try:
        with torch.no_grad():
            macs, _ = profile(model, inputs=(x,), verbose=False)
        return macs / 1e9
    except Exception as e:
        print(f"  GFLOPs computation failed: {e}")
        return None


def export_to_onnx(model, output_path, input_size=(1, 3, 384, 640), 
                   opset_version=17, device='cuda'):
    """
    Export PyTorch model to ONNX format.
    """
    print(f"  Exporting to ONNX: {output_path}")
    
    model = model.to(device).eval()
    dummy_input = torch.randn(*input_size, device=device)
    
    # Dynamic axes for batch size
    dynamic_axes = {
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes
    )
    
    # Verify ONNX model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"  ✓ ONNX model verified")
    
    return output_path


def simplify_onnx(input_path, output_path=None):
    """Simplify ONNX model using onnx-simplifier."""
    try:
        import onnxsim
        
        if output_path is None:
            output_path = input_path
        
        print(f"  Simplifying ONNX model...")
        model = onnx.load(input_path)
        model_simp, check = onnxsim.simplify(model)
        
        if check:
            onnx.save(model_simp, output_path)
            print(f"  ✓ ONNX model simplified")
            return True
        else:
            print(f"  Warning: Simplification check failed")
            return False
    except ImportError:
        print(f"  Note: onnxsim not installed, skipping simplification")
        return False
    except Exception as e:
        print(f"  Warning: Simplification failed: {e}")
        return False


# =============================================================================
# ONNX Runtime Benchmark
# =============================================================================

def benchmark_onnx_runtime(onnx_path, input_size=(1, 3, 384, 640),
                           warmup_iters=50, test_iters=200):
    """
    Benchmark ONNX model with ONNX Runtime.
    """
    print(f"  Benchmarking with ONNX Runtime...")
    
    # Create session with GPU provider
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    session = ort.InferenceSession(onnx_path, sess_options, providers=providers)
    
    # Get input name
    input_name = session.get_inputs()[0].name
    
    # Create input
    dummy_input = np.random.randn(*input_size).astype(np.float32)
    
    # Warmup
    for _ in range(warmup_iters):
        _ = session.run(None, {input_name: dummy_input})
    
    # Benchmark
    start_time = time.perf_counter()
    for _ in range(test_iters):
        _ = session.run(None, {input_name: dummy_input})
    end_time = time.perf_counter()
    
    total_time_ms = (end_time - start_time) * 1000
    avg_latency_ms = total_time_ms / test_iters
    fps = 1000.0 / avg_latency_ms
    
    return {
        'latency_ms': avg_latency_ms,
        'fps': fps,
        'provider': session.get_providers()[0]
    }


def convert_onnx_to_ncnn(onnx_path, param_path, bin_path):
    """Convert ONNX model to NCNN param/bin using Python ncnn or CLI onnx2ncnn."""
    # Try Python ncnn.convert_onnx first (pnnx-based, more robust)
    try:
        import ncnn
        print(f"  Converting ONNX to NCNN via Python ncnn...")
        # ncnn Python package provides convert API
        if hasattr(ncnn, 'convert_onnx'):
            ncnn.convert_onnx(onnx_path, str(param_path), str(bin_path))
            print(f"  ✓ NCNN conversion produced: {param_path}, {bin_path}")
            return True
    except Exception as e:
        print(f"  Python ncnn convert failed: {e}")

    # Fallback to CLI onnx2ncnn
    onnx2ncnn = shutil.which('onnx2ncnn')
    if onnx2ncnn is None:
        # Try looking in ncnn package directory
        try:
            import ncnn
            ncnn_dir = Path(ncnn.__file__).parent
            possible = ncnn_dir / 'onnx2ncnn.exe'
            if possible.exists():
                onnx2ncnn = str(possible)
        except Exception:
            pass

    if onnx2ncnn is None:
        print("  onnx2ncnn not found; skipping NCNN conversion")
        return False

    print(f"  Converting ONNX to NCNN using: {onnx2ncnn}")
    try:
        subprocess.check_call([onnx2ncnn, onnx_path, str(param_path), str(bin_path)])
        print(f"  ✓ NCNN conversion produced: {param_path}, {bin_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  NCNN conversion failed: {e}")
        return False


def benchmark_ncnn(param_path, bin_path, input_size=(384, 640), test_iters=200):
    """Best-effort NCNN benchmark.

    Attempts to run a benchmark via available ncnn tools. If no tooling is found,
    returns None to indicate benchmarking was skipped.
    """
    # Try Python ncnn bindings first
    try:
        import ncnn
        import numpy as np

        print("  Running NCNN benchmark via Python bindings...")
        net = ncnn.Net()
        # Enable Vulkan GPU if available
        if hasattr(net, 'opt'):
            net.opt.use_vulkan_compute = False  # CPU mode more reliable
        ret_param = net.load_param(str(param_path))
        ret_model = net.load_model(str(bin_path))
        if ret_param != 0 or ret_model != 0:
            print(f"  Failed to load NCNN model (param={ret_param}, model={ret_model})")
            raise RuntimeError("NCNN load failed")

        # Prepare random input using ncnn.Mat from numpy
        h, w = input_size
        # Create numpy array and convert to ncnn.Mat
        input_data = np.random.rand(3, h, w).astype(np.float32)
        mat = ncnn.Mat(input_data)

        # Get input/output names from param file
        input_name = "input"   # default
        output_name = "output" # default
        
        # Warmup
        for _ in range(10):
            ex = net.create_extractor()
            ex.input(input_name, mat)
            ret, out = ex.extract(output_name)

        # Timed runs
        import time
        start = time.perf_counter()
        for _ in range(test_iters):
            ex = net.create_extractor()
            ex.input(input_name, mat)
            ret, out = ex.extract(output_name)
        end = time.perf_counter()

        total_ms = (end - start) * 1000.0
        avg_ms = total_ms / test_iters
        fps = 1000.0 / avg_ms
        return {'latency_ms': avg_ms, 'fps': fps}
    except Exception as e:
        # If python bindings not available or failed, try falling back to CLI
        print(f"  NCNN Python benchmark failed: {e}")
        import traceback
        traceback.print_exc()

    # Try CLI benchmark binaries (best-effort): look for common names
    possible_bins = ['ncnn', 'ncnn_benchmark', 'benchmark', 'ncnnbenchmark']
    bin_exec = None
    for name in possible_bins:
        path = shutil.which(name)
        if path:
            bin_exec = path
            break

    if bin_exec is None:
        print("  No NCNN benchmark tool found; skipping NCNN benchmark")
        return None

    print(f"  Running NCNN benchmark via: {bin_exec}")
    try:
        # Example CLI invocation; many ncnn tools accept param/bin and runs.
        # We run best-effort call and parse timing if printed. If format unknown,
        # we will not attempt to parse and will return None.
        proc = subprocess.run([bin_exec, str(param_path), str(bin_path)],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=300)
        out = proc.stdout + proc.stderr
        # Try to find lines like "avg=? ms" or "time=... ms"
        import re
        m = re.search(r"avg[:=]\s*([0-9]+\.?[0-9]*)\s*ms", out)
        if not m:
            m = re.search(r"time[:=]\s*([0-9]+\.?[0-9]*)\s*ms", out)
        if m:
            avg_ms = float(m.group(1))
            fps = 1000.0 / avg_ms
            return {'latency_ms': avg_ms, 'fps': fps}
        else:
            print("  Could not parse NCNN benchmark output; returning raw output as note")
            print(out)
            return None
    except Exception as e:
        print(f"  NCNN benchmark failed: {e}")
        return None


# =============================================================================
# TensorRT Engine Building
# =============================================================================

TRT_LOGGER = trt.Logger(trt.Logger.WARNING) if HAS_TRT else None


def build_tensorrt_engine(onnx_path, engine_path, 
                          input_size=(1, 3, 384, 640),
                          fp16=True, int8=False):
    """
    Build TensorRT engine from ONNX model.
    """
    if not HAS_TRT:
        print("  TensorRT not available!")
        return None
    
    print(f"  Building TensorRT engine (FP16={fp16})...")
    
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # Parse ONNX model
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(f"  ONNX Parse Error: {parser.get_error(error)}")
            return None
    
    # Configure builder
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
    
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print(f"  Using FP16 precision")
    
    if int8 and builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        print(f"  Using INT8 precision")
    
    # Set input shape
    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name
    
    # Min, Opt, Max shapes (batch size 1)
    profile.set_shape(input_name, 
                      min=(1, 3, 384, 640),
                      opt=(1, 3, 384, 640),
                      max=(1, 3, 384, 640))
    config.add_optimization_profile(profile)
    
    # Build engine
    print(f"  Building engine (this may take a few minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        print("  Failed to build TensorRT engine!")
        return None
    
    # Save engine
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)
    
    print(f"  ✓ TensorRT engine saved: {engine_path}")
    
    return engine_path


def load_tensorrt_engine(engine_path):
    """Load TensorRT engine from file."""
    runtime = trt.Runtime(TRT_LOGGER)
    
    with open(engine_path, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    
    return engine


# =============================================================================
# TensorRT Benchmark
# =============================================================================

def benchmark_tensorrt(engine_path, input_size=(1, 3, 384, 640),
                       warmup_iters=50, test_iters=200):
    """
    Benchmark TensorRT engine.
    """
    if not HAS_TRT or not HAS_PYCUDA:
        print("  TensorRT/PyCUDA not available!")
        return None
    
    print(f"  Benchmarking with TensorRT...")
    
    # Load engine
    engine = load_tensorrt_engine(engine_path)
    context = engine.create_execution_context()
    
    # Allocate buffers
    batch_size = input_size[0]
    
    # Get binding shapes
    input_binding_idx = 0
    output_binding_idx = 1
    
    input_shape = input_size
    # Output shape: same H, W with num_classes channels
    output_shape = (batch_size, 2, input_size[2], input_size[3])
    
    # Allocate device memory
    d_input = cuda.mem_alloc(int(np.prod(input_shape) * np.float32().nbytes))
    d_output = cuda.mem_alloc(int(np.prod(output_shape) * np.float32().nbytes))
    
    # Create host buffers
    h_input = np.random.randn(*input_shape).astype(np.float32)
    h_output = np.empty(output_shape, dtype=np.float32)
    
    # Create CUDA stream
    stream = cuda.Stream()
    
    # Set binding shapes for dynamic inputs
    context.set_input_shape("input", input_shape)
    
    # Warmup
    for _ in range(warmup_iters):
        cuda.memcpy_htod_async(d_input, h_input, stream)
        context.execute_async_v3(stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()
    
    # Benchmark
    start_event = cuda.Event()
    end_event = cuda.Event()
    
    start_event.record(stream)
    for _ in range(test_iters):
        cuda.memcpy_htod_async(d_input, h_input, stream)
        context.execute_async_v3(stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
    end_event.record(stream)
    end_event.synchronize()
    
    total_time_ms = start_event.time_till(end_event)
    avg_latency_ms = total_time_ms / test_iters
    fps = 1000.0 / avg_latency_ms
    
    # Cleanup
    del context
    del engine
    
    return {
        'latency_ms': avg_latency_ms,
        'fps': fps
    }


def benchmark_tensorrt_torch(engine_path, input_size=(1, 3, 384, 640),
                              warmup_iters=50, test_iters=200):
    """
    Benchmark TensorRT using torch-tensorrt or manual bindings.
    Fallback method using direct CUDA timing.
    """
    if not HAS_TRT:
        return None
    
    print(f"  Benchmarking TensorRT engine...")
    
    try:
        # Try using torch_tensorrt if available
        import torch_tensorrt
        
        # Load as TorchScript
        trt_model = torch_tensorrt.ts.load(engine_path)
        
        device = torch.device('cuda')
        x = torch.randn(*input_size, device=device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_iters):
                _ = trt_model(x)
        
        torch.cuda.synchronize()
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        with torch.no_grad():
            start_event.record()
            for _ in range(test_iters):
                _ = trt_model(x)
            end_event.record()
        
        torch.cuda.synchronize()
        
        total_time_ms = start_event.elapsed_time(end_event)
        avg_latency_ms = total_time_ms / test_iters
        fps = 1000.0 / avg_latency_ms
        
        return {
            'latency_ms': avg_latency_ms,
            'fps': fps
        }
    except Exception as e:
        print(f"  torch_tensorrt not available, using fallback: {e}")
        return benchmark_tensorrt(engine_path, input_size, warmup_iters, test_iters)


# =============================================================================
# PyTorch Baseline Benchmark
# =============================================================================

def benchmark_pytorch(model, input_size=(1, 3, 384, 640), device='cuda',
                      warmup_iters=50, test_iters=200):
    """Benchmark PyTorch model."""
    model = model.to(device).eval()
    x = torch.randn(*input_size, device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(x)
    
    torch.cuda.synchronize()
    
    # Measure
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    with torch.no_grad():
        start_event.record()
        for _ in range(test_iters):
            _ = model(x)
        end_event.record()
    
    torch.cuda.synchronize()
    
    total_time_ms = start_event.elapsed_time(end_event)
    avg_latency_ms = total_time_ms / test_iters
    fps = 1000.0 / avg_latency_ms
    
    return {
        'latency_ms': avg_latency_ms,
        'fps': fps
    }


def measure_peak_memory_pytorch(model, input_size=(1, 3, 384, 640), device='cuda',
                                warmup_iters=10, test_iters=50):
    """Measure peak GPU memory for PyTorch model."""
    model = model.to(device).eval()
    x = torch.randn(*input_size, device=device)
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(x)
    
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
    # Measure
    with torch.no_grad():
        for _ in range(test_iters):
            _ = model(x)
            torch.cuda.synchronize()
    
    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
    return peak_memory


# =============================================================================
# Main Benchmark Pipeline
# =============================================================================

def benchmark_auraseg_model(backbone: str, output_dir: Path, 
                            skip_tensorrt: bool = False,
                            device='cuda'):
    """
    Full benchmark pipeline for one AURASeg variant.
    """
    print(f"\n{'='*70}")
    print(f"Benchmarking AURASeg with {backbone.upper()} backbone")
    print(f"{'='*70}")
    
    input_size = (1, 3, 384, 640)
    results = {
        'backbone': backbone,
        'params_m': None,
        'gflops': None,
    }
    
    # Create model
    print("\n1. Creating PyTorch model...")
    model = AURASeg_V4_ResNet(backbone=backbone, num_classes=2)
    model = model.to(device).eval()
    
    # Count parameters
    results['params_m'] = count_parameters(model)
    print(f"   Params: {results['params_m']:.2f}M")
    
    # Compute GFLOPs
    results['gflops'] = compute_gflops(model, input_size, device)
    if results['gflops']:
        print(f"   GFLOPs: {results['gflops']:.2f}")
    
    # PyTorch baseline
    print("\n2. PyTorch baseline benchmark...")
    pytorch_results = benchmark_pytorch(model, input_size, device)
    pytorch_memory = measure_peak_memory_pytorch(model, input_size, device)
    
    results['pytorch'] = {
        'latency_ms': pytorch_results['latency_ms'],
        'fps': pytorch_results['fps'],
        'peak_memory_mb': pytorch_memory
    }
    print(f"   PyTorch Latency: {pytorch_results['latency_ms']:.2f} ms")
    print(f"   PyTorch FPS: {pytorch_results['fps']:.1f}")
    print(f"   PyTorch Peak Memory: {pytorch_memory:.1f} MB")
    
    # Export to ONNX
    print("\n3. Exporting to ONNX...")
    onnx_path = output_dir / f"auraseg_{backbone}.onnx"
    
    try:
        export_to_onnx(model, str(onnx_path), input_size, device=device)
        simplify_onnx(str(onnx_path))
        
        # ONNX Runtime benchmark
        print("\n4. ONNX Runtime benchmark...")
        onnx_results = benchmark_onnx_runtime(str(onnx_path), input_size)
        results['onnx'] = {
            'latency_ms': onnx_results['latency_ms'],
            'fps': onnx_results['fps'],
            'provider': onnx_results['provider']
        }
        print(f"   ONNX Runtime Latency: {onnx_results['latency_ms']:.2f} ms")
        print(f"   ONNX Runtime FPS: {onnx_results['fps']:.1f}")
        print(f"   Provider: {onnx_results['provider']}")
        
    except Exception as e:
        print(f"   ONNX export failed: {e}")
        import traceback
        traceback.print_exc()
        results['onnx'] = None
    
    # TensorRT benchmark
    if not skip_tensorrt and HAS_TRT:
        print("\n5. TensorRT optimization...")
        engine_path = output_dir / f"auraseg_{backbone}_fp16.engine"
        
        try:
            build_tensorrt_engine(str(onnx_path), str(engine_path), 
                                  input_size, fp16=True)
            
            print("\n6. TensorRT benchmark...")
            if HAS_PYCUDA:
                trt_results = benchmark_tensorrt(str(engine_path), input_size)
            else:
                trt_results = benchmark_tensorrt_torch(str(engine_path), input_size)
            
            if trt_results:
                results['tensorrt'] = {
                    'latency_ms': trt_results['latency_ms'],
                    'fps': trt_results['fps']
                }
                print(f"   TensorRT Latency: {trt_results['latency_ms']:.2f} ms")
                print(f"   TensorRT FPS: {trt_results['fps']:.1f}")
        except Exception as e:
            print(f"   TensorRT failed: {e}")
            import traceback
            traceback.print_exc()
            results['tensorrt'] = None
    else:
        if skip_tensorrt:
            print("\n5. TensorRT skipped (--skip-tensorrt)")
        else:
            print("\n5. TensorRT not available")
        results['tensorrt'] = None

    # NCNN conversion and benchmark
    results['ncnn'] = None
    try:
        ncnn_param_path = output_dir / f"auraseg_{backbone}.param"
        ncnn_bin_path = output_dir / f"auraseg_{backbone}.bin"
        if convert_onnx_to_ncnn(str(onnx_path), ncnn_param_path, ncnn_bin_path):
            print("\n6. NCNN benchmark...")
            ncnn_results = benchmark_ncnn(ncnn_param_path, ncnn_bin_path,
                                          input_size=(input_size[2], input_size[3]))
            if ncnn_results:
                results['ncnn'] = {
                    'latency_ms': ncnn_results['latency_ms'],
                    'fps': ncnn_results['fps']
                }
                print(f"   NCNN Latency: {ncnn_results['latency_ms']:.2f} ms")
                print(f"   NCNN FPS: {ncnn_results['fps']:.1f}")
    except Exception as e:
        print(f"   NCNN step failed: {e}")
        import traceback
        traceback.print_exc()

    # Cleanup
    del model
    torch.cuda.empty_cache()
    gc.collect()

    return results


def print_summary_table(all_results):
    """Print summary table of all results."""
    print("\n")
    print("=" * 100)
    print("AURASeg DEPLOYMENT BENCHMARK SUMMARY")
    print("=" * 100)
    
    # Header
    print(f"{'Model':<20} | {'Params':<8} | {'GFLOPs':<8} | {'Memory':<10} | {'Runtime':<12} | {'Latency':<10} | {'FPS':<8}")
    print("-" * 100)
    
    for r in all_results:
        backbone = r['backbone']
        params = f"{r['params_m']:.2f}M" if r['params_m'] else "N/A"
        gflops = f"{r['gflops']:.2f}" if r['gflops'] else "N/A"
        
        # PyTorch row
        if r.get('pytorch'):
            memory = f"{r['pytorch']['peak_memory_mb']:.1f} MB"
            latency = f"{r['pytorch']['latency_ms']:.2f} ms"
            fps = f"{r['pytorch']['fps']:.1f}"
            print(f"AURASeg-{backbone:<10} | {params:<8} | {gflops:<8} | {memory:<10} | {'PyTorch':<12} | {latency:<10} | {fps:<8}")
        
        # ONNX row
        if r.get('onnx'):
            latency = f"{r['onnx']['latency_ms']:.2f} ms"
            fps = f"{r['onnx']['fps']:.1f}"
            print(f"{'':20} | {'':8} | {'':8} | {'':10} | {'ONNX RT':<12} | {latency:<10} | {fps:<8}")
        
        # TensorRT row
        if r.get('tensorrt'):
            latency = f"{r['tensorrt']['latency_ms']:.2f} ms"
            fps = f"{r['tensorrt']['fps']:.1f}"
            print(f"{'':20} | {'':8} | {'':8} | {'':10} | {'TensorRT FP16':<12} | {latency:<10} | {fps:<8}")

        # NCNN row
        if r.get('ncnn'):
            latency = f"{r['ncnn']['latency_ms']:.2f} ms"
            fps = f"{r['ncnn']['fps']:.1f}"
            print(f"{'':20} | {'':8} | {'':8} | {'':10} | {'NCNN':<12} | {latency:<10} | {fps:<8}")

    print("=" * 100)


def save_results(all_results, output_dir):
    """Save results to files."""
    
    # Text report
    report_path = output_dir / "auraseg_deployment_results.txt"
    with open(report_path, 'w') as f:
        f.write("AURASeg DEPLOYMENT BENCHMARK RESULTS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n")
        if HAS_TRT:
            f.write(f"TensorRT Version: {TRT_VERSION}\n")
        f.write(f"Input Resolution: 384 × 640\n")
        f.write("=" * 80 + "\n\n")
        
        for r in all_results:
            f.write(f"Backbone: {r['backbone']}\n")
            f.write(f"  Params: {r['params_m']:.2f}M\n")
            f.write(f"  GFLOPs: {r['gflops']:.2f}\n" if r['gflops'] else "  GFLOPs: N/A\n")
            
            if r.get('pytorch'):
                f.write(f"  PyTorch:\n")
                f.write(f"    Latency: {r['pytorch']['latency_ms']:.2f} ms\n")
                f.write(f"    FPS: {r['pytorch']['fps']:.1f}\n")
                f.write(f"    Peak Memory: {r['pytorch']['peak_memory_mb']:.1f} MB\n")
            
            if r.get('onnx'):
                f.write(f"  ONNX Runtime:\n")
                f.write(f"    Latency: {r['onnx']['latency_ms']:.2f} ms\n")
                f.write(f"    FPS: {r['onnx']['fps']:.1f}\n")
            
            if r.get('tensorrt'):
                f.write(f"  TensorRT FP16:\n")
                f.write(f"    Latency: {r['tensorrt']['latency_ms']:.2f} ms\n")
                f.write(f"    FPS: {r['tensorrt']['fps']:.1f}\n")

            if r.get('ncnn'):
                f.write(f"  NCNN:\n")
                f.write(f"    Latency: {r['ncnn']['latency_ms']:.2f} ms\n")
                f.write(f"    FPS: {r['ncnn']['fps']:.1f}\n")

            f.write("\n")
    
    print(f"\nResults saved to: {report_path}")
    
    # CSV for Table 4
    csv_path = output_dir / "auraseg_deployment_results.csv"
    with open(csv_path, 'w') as f:
        f.write("Model,Backbone,Params (M),GFLOPs,Peak Memory (MB),Latency (ms),FPS,Runtime\n")
        
        for r in all_results:
            backbone = r['backbone']
            params = f"{r['params_m']:.2f}" if r['params_m'] else ""
            gflops = f"{r['gflops']:.2f}" if r['gflops'] else ""
            
            # PyTorch
            if r.get('pytorch'):
                memory = f"{r['pytorch']['peak_memory_mb']:.1f}"
                latency = f"{r['pytorch']['latency_ms']:.2f}"
                fps = f"{r['pytorch']['fps']:.1f}"
                f.write(f"AURASeg,{backbone},{params},{gflops},{memory},{latency},{fps},PyTorch\n")
            
            # ONNX
            if r.get('onnx'):
                latency = f"{r['onnx']['latency_ms']:.2f}"
                fps = f"{r['onnx']['fps']:.1f}"
                f.write(f"AURASeg,{backbone},{params},{gflops},,{latency},{fps},ONNX RT\n")
            
            # TensorRT
            if r.get('tensorrt'):
                latency = f"{r['tensorrt']['latency_ms']:.2f}"
                fps = f"{r['tensorrt']['fps']:.1f}"
                f.write(f"AURASeg,{backbone},{params},{gflops},,{latency},{fps},TensorRT FP16\n")

            # NCNN
            if r.get('ncnn'):
                latency = f"{r['ncnn']['latency_ms']:.2f}"
                fps = f"{r['ncnn']['fps']:.1f}"
                f.write(f"AURASeg,{backbone},{params},{gflops},,{latency},{fps},NCNN\n")

    print(f"CSV saved to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='AURASeg ONNX/TensorRT Benchmark')
    parser.add_argument('--skip-tensorrt', action='store_true',
                        help='Skip TensorRT benchmarking')
    parser.add_argument('--backbone', type=str, default='both',
                        choices=['resnet18', 'resnet50', 'both'],
                        help='Which backbone to benchmark')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory')
    args = parser.parse_args()
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent.parent / "runs" / "deployment"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("AURASeg ONNX Export & TensorRT Benchmark")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"ONNX available: {HAS_ONNX}")
    print(f"TensorRT available: {HAS_TRT} (version: {TRT_VERSION})")
    print(f"Output directory: {output_dir}")
    print("=" * 70)
    
    # Benchmark models
    all_results = []
    
    backbones = ['resnet18', 'resnet50'] if args.backbone == 'both' else [args.backbone]
    
    for backbone in backbones:
        results = benchmark_auraseg_model(
            backbone=backbone,
            output_dir=output_dir,
            skip_tensorrt=args.skip_tensorrt,
            device=device
        )
        all_results.append(results)
    
    # Print summary
    print_summary_table(all_results)
    
    # Save results
    save_results(all_results, output_dir)
    
    print("\n✓ Benchmark complete!")


if __name__ == "__main__":
    main()
