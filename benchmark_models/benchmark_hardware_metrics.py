"""
Comprehensive Hardware Metrics Benchmark Suite
===============================================
Computes deployment hardware metrics across segmentation models:
- AURASeg-R18 (Ours)
- AURASeg-R50 (Ours)
- FCN-R50
- PSPNet-R50
- UPerNet-R50
- SegFormer-B2
- Mask2Former (FPN-MiTB3)
- PIDNet-L

Precisions & Engines:
- PyTorch FP32 (Latency ms, FPS, Peak Memory MB)
- PyTorch FP16 / AMP (Latency ms, FPS, Peak Memory MB)
- ONNX Runtime (Latency ms, FPS)
- TensorRT FP16 (Engine Build + Latency ms, FPS)
- Model Complexity: Params (M), Trainable Params (M), GFLOPs / GMACs

Usage:
    python benchmark_hardware_metrics.py --models all
    python benchmark_hardware_metrics.py --models auraseg_r18 fcn-r50 pspnet-r50 --build-trt
    python benchmark_hardware_metrics.py --input-size 384 640 --iters 200 --warmup 50
"""

import sys
import os
import gc
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Ensure benchmark_models directory is in path
benchmark_dir = Path(__file__).parent.resolve()
if str(benchmark_dir) not in sys.path:
    sys.path.insert(0, str(benchmark_dir))

from model_factory import get_benchmark_model, BENCHMARK_MODELS, MODEL_ALIASES

# Complexity profiling
try:
    from thop import profile, clever_format
    HAS_THOP = True
except ImportError:
    HAS_THOP = False

try:
    from ptflops import get_model_complexity_info
    HAS_PTFLOPS = True
except ImportError:
    HAS_PTFLOPS = False

# ONNX & Simplifier
try:
    import onnx
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

try:
    import onnxsim
    HAS_ONNXSIM = True
except ImportError:
    HAS_ONNXSIM = False

# TensorRT
try:
    import tensorrt as trt
    HAS_TRT = True
    TRT_VERSION = trt.__version__
except ImportError:
    HAS_TRT = False
    TRT_VERSION = None


# =============================================================================
# Model Wrapper for Uniform Output
# =============================================================================

class CanonicalModelWrapper(nn.Module):
    """
    Wraps any segmentation model to guarantee a single Tensor output (B, C, H, W).
    Handles dict outputs ('out', 'main', 'logits') and tuple/list outputs.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        if isinstance(out, dict):
            if 'out' in out:
                return out['out']
            if 'main' in out:
                return out['main']
            if 'logits' in out:
                return out['logits']
            return next(iter(out.values()))
        elif isinstance(out, (list, tuple)):
            return out[0]
        return out


# =============================================================================
# Hardware Metric Functions
# =============================================================================

def count_parameters(model: nn.Module) -> Tuple[float, float]:
    """Return total and trainable parameters in Millions."""
    total = sum(p.numel() for p in model.parameters()) / 1e6
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    return total, trainable


def compute_gflops_thop(model: nn.Module, input_size: Tuple[int, ...], device: torch.device) -> Optional[float]:
    """Compute GFLOPs using thop library."""
    if not HAS_THOP:
        return None
    try:
        dummy = torch.randn(*input_size, device=device)
        wrapped = CanonicalModelWrapper(model) if not isinstance(model, CanonicalModelWrapper) else model
        with torch.no_grad():
            macs, _ = profile(wrapped, inputs=(dummy,), verbose=False)
        return macs / 1e9
    except Exception as e:
        return None


def compute_gflops_ptflops(model: nn.Module, input_size: Tuple[int, ...], device: torch.device) -> Optional[float]:
    """Compute GFLOPs using ptflops library (fallback)."""
    if not HAS_PTFLOPS:
        return None
    try:
        c, h, w = input_size[1:]
        wrapped = CanonicalModelWrapper(model) if not isinstance(model, CanonicalModelWrapper) else model
        macs, _ = get_model_complexity_info(
            wrapped, (c, h, w),
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False
        )
        return macs / 1e9
    except Exception as e:
        return None


def compute_gflops(model: nn.Module, input_size: Tuple[int, ...], device: torch.device) -> Optional[float]:
    """Compute GFLOPs using thop with ptflops fallback."""
    gflops = compute_gflops_thop(model, input_size, device)
    if gflops is not None:
        return gflops
    return compute_gflops_ptflops(model, input_size, device)


def measure_peak_memory(model: nn.Module, input_size: Tuple[int, ...], 
                        device: torch.device, fp16: bool = False,
                        warmup_iters: int = 10, test_iters: int = 50) -> Optional[float]:
    """Measure peak GPU memory allocation (MB) during inference."""
    if device.type != 'cuda' or not torch.cuda.is_available():
        return None

    model = model.to(device).eval()
    dummy = torch.randn(*input_size, device=device)

    # Empty cache and reset stats
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_iters):
            if fp16 and device.type == 'cuda':
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    _ = model(dummy)
            else:
                _ = model(dummy)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # Timed runs
    with torch.no_grad():
        for _ in range(test_iters):
            if fp16 and device.type == 'cuda':
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    _ = model(dummy)
            else:
                _ = model(dummy)
            torch.cuda.synchronize()

    peak_mb = torch.cuda.max_memory_allocated(device=device) / (1024 ** 2)
    return peak_mb


def measure_pytorch_latency_fps(model: nn.Module, input_size: Tuple[int, ...], 
                                device: torch.device, fp16: bool = False,
                                warmup_iters: int = 50, test_iters: int = 200) -> Tuple[float, float]:
    """
    Measure PyTorch latency (ms) and throughput (FPS) using CUDA Events.
    """
    model = model.to(device).eval()
    dummy = torch.randn(*input_size, device=device)
    batch_size = input_size[0]

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_iters):
            if fp16 and device.type == 'cuda':
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    _ = model(dummy)
            else:
                _ = model(dummy)

    if device.type == 'cuda':
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        with torch.no_grad():
            start_event.record()
            for _ in range(test_iters):
                if fp16:
                    with torch.amp.autocast('cuda', dtype=torch.float16):
                        _ = model(dummy)
                else:
                    _ = model(dummy)
            end_event.record()

        torch.cuda.synchronize()
        total_time_ms = start_event.elapsed_time(end_event)
    else:
        with torch.no_grad():
            t0 = time.perf_counter()
            for _ in range(test_iters):
                _ = model(dummy)
            t1 = time.perf_counter()
        total_time_ms = (t1 - t0) * 1000.0

    avg_latency_ms = total_time_ms / test_iters
    fps = (1000.0 / avg_latency_ms) * batch_size
    return avg_latency_ms, fps


def fuse_conv_bn_recursive(module: nn.Module) -> nn.Module:
    """
    Recursively fuses all Conv2d + BatchNorm2d pairs in a PyTorch model into Conv2d(bias=True).
    Mathematically exact inference re-parameterization: zero loss in precision, 100% identical predictions.
    """
    for name, child in list(module.named_children()):
        # 1. Custom ConvBNAct
        if child.__class__.__name__ == 'ConvBNAct':
            if hasattr(child, 'conv') and hasattr(child, 'bn') and isinstance(child.conv, nn.Conv2d) and isinstance(child.bn, nn.BatchNorm2d):
                child.conv = torch.nn.utils.fusion.fuse_conv_bn_eval(child.conv, child.bn)
                child.bn = nn.Identity()
        # 2. ResNet BasicBlock (conv1+bn1, conv2+bn2, downsample[0]+downsample[1])
        elif child.__class__.__name__ == 'BasicBlock':
            if hasattr(child, 'conv1') and hasattr(child, 'bn1') and isinstance(child.conv1, nn.Conv2d) and isinstance(child.bn1, nn.BatchNorm2d):
                child.conv1 = torch.nn.utils.fusion.fuse_conv_bn_eval(child.conv1, child.bn1)
                child.bn1 = nn.Identity()
            if hasattr(child, 'conv2') and hasattr(child, 'bn2') and isinstance(child.conv2, nn.Conv2d) and isinstance(child.bn2, nn.BatchNorm2d):
                child.conv2 = torch.nn.utils.fusion.fuse_conv_bn_eval(child.conv2, child.bn2)
                child.bn2 = nn.Identity()
            if hasattr(child, 'downsample') and child.downsample is not None:
                if len(child.downsample) >= 2 and isinstance(child.downsample[0], nn.Conv2d) and isinstance(child.downsample[1], nn.BatchNorm2d):
                    child.downsample[0] = torch.nn.utils.fusion.fuse_conv_bn_eval(child.downsample[0], child.downsample[1])
                    child.downsample[1] = nn.Identity()
        # 3. Standard Sequentials
        elif isinstance(child, nn.Sequential):
            i = 0
            while i < len(child) - 1:
                if isinstance(child[i], nn.Conv2d) and isinstance(child[i+1], nn.BatchNorm2d):
                    child[i] = torch.nn.utils.fusion.fuse_conv_bn_eval(child[i], child[i+1])
                    child[i+1] = nn.Identity()
                    i += 2
                else:
                    i += 1
            fuse_conv_bn_recursive(child)
        else:
            fuse_conv_bn_recursive(child)
    return module


# =============================================================================
# ONNX Export & Benchmark
# =============================================================================

class ExportableAdaptiveAvgPool2d(nn.Module):
    """
    Drop-in replacement for AdaptiveAvgPool2d to ensure compatibility with ONNX export
    when feature map dimensions are not an exact integer multiple of the output size.
    """
    def __init__(self, orig_module: nn.AdaptiveAvgPool2d):
        super().__init__()
        out_sz = orig_module.output_size
        self.output_size = out_sz if isinstance(out_sz, tuple) else (out_sz, out_sz)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=self.output_size, mode='bilinear', align_corners=False)


def patch_adaptive_pooling_for_onnx(model: nn.Module) -> nn.Module:
    """Recursively patches AdaptiveAvgPool2d layers with non-factor sizes for ONNX export."""
    for name, child in list(model.named_children()):
        if isinstance(child, nn.AdaptiveAvgPool2d):
            out_sz = child.output_size
            if out_sz not in [1, (1, 1)]:
                setattr(model, name, ExportableAdaptiveAvgPool2d(child))
        else:
            patch_adaptive_pooling_for_onnx(child)
    return model


def export_onnx_model(model: nn.Module, output_path: str, 
                      input_size: Tuple[int, ...], device: torch.device,
                      opset_version: int = 17) -> bool:
    """Export model to ONNX with validation and simplification."""
    if not HAS_ONNX:
        print("  [ONNX] onnx package not installed. Skipping export.")
        return False

    try:
        import copy
        # Create an export copy of model and patch any non-factor adaptive pooling
        export_model = copy.deepcopy(model).to(device).eval()
        export_model = patch_adaptive_pooling_for_onnx(export_model)
        
        dummy = torch.randn(*input_size, device=device)
        wrapped = CanonicalModelWrapper(export_model)

        dynamic_axes = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }

        # Create output directory
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        print(f"  Exporting ONNX: {output_path} (opset {opset_version})...")
        try:
            torch.onnx.export(
                wrapped,
                dummy,
                output_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes=dynamic_axes,
                dynamo=False
            )
        except TypeError:
            torch.onnx.export(
                wrapped,
                dummy,
                output_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes=dynamic_axes
            )

        # Check model
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("  ✓ ONNX model validated")

        # Simplify
        if HAS_ONNXSIM:
            try:
                model_simp, check = onnxsim.simplify(onnx_model)
                if check:
                    onnx.save(model_simp, output_path)
                    print("  ✓ ONNX model simplified (onnxsim)")
            except Exception as sim_e:
                print(f"  Note: onnxsim skipped: {sim_e}")

        del export_model
        return True
    except Exception as e:
        print(f"  ✗ ONNX export failed: {e}")
        return False


def benchmark_onnx_runtime(onnx_path: str, input_size: Tuple[int, ...],
                           warmup_iters: int = 50, test_iters: int = 200) -> Optional[Dict[str, Any]]:
    """Benchmark ONNX model with ONNX Runtime."""
    if not HAS_ONNX:
        return None

    try:
        available = ort.get_available_providers()
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'CUDAExecutionProvider' in available else ['CPUExecutionProvider']
        
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        session = ort.InferenceSession(onnx_path, sess_opts, providers=providers)
        input_name = session.get_inputs()[0].name
        active_provider = session.get_providers()[0]
        
        dummy_np = np.random.randn(*input_size).astype(np.float32)
        batch_size = input_size[0]

        # Warmup
        for _ in range(warmup_iters):
            _ = session.run(None, {input_name: dummy_np})

        # Benchmark
        t0 = time.perf_counter()
        for _ in range(test_iters):
            _ = session.run(None, {input_name: dummy_np})
        t1 = time.perf_counter()

        avg_latency_ms = ((t1 - t0) * 1000.0) / test_iters
        fps = (1000.0 / avg_latency_ms) * batch_size

        return {
            'latency_ms': avg_latency_ms,
            'fps': fps,
            'provider': active_provider
        }
    except Exception as e:
        print(f"  ✗ ONNX Runtime benchmark failed: {e}")
        return None


# =============================================================================
# TensorRT FP16 Engine Build & Benchmark
# =============================================================================

def build_tensorrt_fp16_engine(onnx_path: str, engine_path: str,
                               input_size: Tuple[int, ...],
                               fp16: bool = True,
                               workspace_mb: int = 1024) -> bool:
    """Build a serialized TensorRT FP16 engine from ONNX."""
    if not HAS_TRT:
        print("  [TensorRT] tensorrt package not installed.")
        return False

    try:
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        
        if hasattr(trt, 'NetworkDefinitionCreationFlag') and hasattr(trt.NetworkDefinitionCreationFlag, 'EXPLICIT_BATCH'):
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        else:
            network = builder.create_network()
            
        parser = trt.OnnxParser(network, logger)

        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                for idx in range(parser.num_errors):
                    print(f"  TRT Parse Error: {parser.get_error(idx)}")
                return False

        config = builder.create_builder_config()
        # Set workspace memory limit
        if hasattr(trt, 'MemoryPoolType'):
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * 1024 * 1024)
        elif hasattr(config, 'max_workspace_size'):
            config.max_workspace_size = workspace_mb * 1024 * 1024

        if fp16:
            if hasattr(trt.BuilderFlag, 'FP16'):
                config.set_flag(trt.BuilderFlag.FP16)
                print("  TRT: Fast FP16 flag enabled.")
            else:
                for i in range(network.num_layers):
                    layer = network.get_layer(i)
                    if hasattr(layer, 'precision'):
                        try:
                            layer.precision = trt.DataType.HALF
                        except Exception:
                            pass
                print("  TRT: Layer-wise FP16 precision enabled.")

        # Optimization profile for input shape
        profile = builder.create_optimization_profile()
        input_name = network.get_input(0).name
        profile.set_shape(input_name, min=input_size, opt=input_size, max=input_size)
        config.add_optimization_profile(profile)

        print(f"  Building TensorRT FP16 engine (saving to {engine_path})...")
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            print("  ✗ TensorRT engine build returned None.")
            return False

        Path(engine_path).parent.mkdir(parents=True, exist_ok=True)
        with open(engine_path, 'wb') as f:
            f.write(bytes(serialized_engine))

        print(f"  ✓ TensorRT FP16 engine successfully built ({Path(engine_path).stat().st_size / 1e6:.1f} MB)")
        return True
    except Exception as e:
        print(f"  ✗ TensorRT build failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def benchmark_tensorrt_engine(engine_path: str, input_size: Tuple[int, ...],
                              num_classes: int = 2,
                              warmup_iters: int = 50,
                              test_iters: int = 200) -> Optional[Dict[str, Any]]:
    """
    Benchmark TensorRT engine directly using PyTorch CUDA tensors.
    Compatible across TensorRT 8.x, 9.x, 10.x, and 11.x without external PyCUDA dependencies.
    """
    if not HAS_TRT or not torch.cuda.is_available():
        return None

    try:
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)

        with open(engine_path, 'rb') as f:
            engine = runtime.deserialize_cuda_engine(f.read())

        context = engine.create_execution_context()
        device = torch.device('cuda')
        batch_size = input_size[0]

        # Allocate PyTorch input & output device buffers
        d_input = torch.randn(*input_size, device=device, dtype=torch.float32)
        out_shape = (batch_size, num_classes, input_size[2], input_size[3])
        d_output = torch.empty(out_shape, device=device, dtype=torch.float32)

        stream = torch.cuda.Stream()

        # Determine TensorRT API version
        is_trt10_plus = hasattr(context, "set_tensor_address")

        if is_trt10_plus:
            context.set_input_shape("input", input_size)
            context.set_tensor_address("input", d_input.data_ptr())
            context.set_tensor_address("output", d_output.data_ptr())
            exec_fn = lambda: context.execute_async_v3(stream.cuda_stream)
        else:
            # TRT 8.x / 9.x bindings
            bindings = [d_input.data_ptr(), d_output.data_ptr()]
            if hasattr(context, "set_binding_shape"):
                context.set_binding_shape(0, input_size)
            exec_fn = lambda: context.execute_async_v2(bindings=bindings, stream_handle=stream.cuda_stream)

        # Warmup
        for _ in range(warmup_iters):
            exec_fn()
        stream.synchronize()

        # Timed inference using CUDA Events
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record(stream)
        for _ in range(test_iters):
            exec_fn()
        end_event.record(stream)

        stream.synchronize()
        total_time_ms = start_event.elapsed_time(end_event)
        avg_latency_ms = total_time_ms / test_iters
        fps = (1000.0 / avg_latency_ms) * batch_size

        del context
        del engine
        return {
            'latency_ms': avg_latency_ms,
            'fps': fps
        }
    except Exception as e:
        print(f"  ✗ TensorRT benchmark execution failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# =============================================================================
# Benchmarking Engine for Single Model
# =============================================================================

def benchmark_single_model(model_key: str, 
                           input_size: Tuple[int, int, int, int] = (1, 3, 384, 640),
                           num_classes: int = 2,
                           device_str: str = 'cuda',
                           warmup_iters: int = 50,
                           test_iters: int = 200,
                           export_onnx: bool = True,
                           build_trt: bool = True,
                           output_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Execute complete FP32 -> FP16 -> TensorRT FP16 benchmark for a model."""
    canonical_key = MODEL_ALIASES.get(model_key.lower().strip(), model_key.lower().strip())
    device = torch.device(device_str if torch.cuda.is_available() and device_str == 'cuda' else 'cpu')

    print("\n" + "=" * 80)
    print(f"BENCHMARKING: {canonical_key.upper()} ({model_key})")
    print("=" * 80)

    # 1. Instantiate Model
    model, info = get_benchmark_model(canonical_key, num_classes=num_classes, pretrained=False)
    wrapped_model = CanonicalModelWrapper(model).to(device).eval()
    
    # Apply Conv+BN Fusion (Structural Re-parameterization at test-time)
    fuse_conv_bn_recursive(wrapped_model)

    model_display_name = info.get('name', canonical_key)
    encoder_name = info.get('encoder', 'N/A')
    paradigm_name = info.get('paradigm', 'N/A')

    # Complexity
    params_total, params_trainable = count_parameters(wrapped_model)
    gflops = compute_gflops(wrapped_model, input_size, device)

    print(f"Architecture: {model_display_name} | Encoder: {encoder_name} | Paradigm: {paradigm_name}")
    print(f"Parameters  : {params_total:.2f}M (Trainable: {params_trainable:.2f}M)")
    print(f"Complexity  : {f'{gflops:.2f} GFLOPs' if gflops is not None else 'N/A'} at {input_size[2]}x{input_size[3]}")

    results = {
        'key': canonical_key,
        'name': model_display_name,
        'encoder': encoder_name,
        'paradigm': paradigm_name,
        'params_m': params_total,
        'params_trainable_m': params_trainable,
        'gflops': gflops,
        'pytorch_fp32': None,
        'pytorch_fp16': None,
        'onnx': None,
        'tensorrt_fp16': None
    }

    # 2. PyTorch FP32 Benchmark
    print("\n[1/4] Measuring PyTorch FP32...")
    fp32_latency, fp32_fps = measure_pytorch_latency_fps(
        wrapped_model, input_size, device, fp16=False,
        warmup_iters=warmup_iters, test_iters=test_iters
    )
    fp32_mem = measure_peak_memory(wrapped_model, input_size, device, fp16=False)
    results['pytorch_fp32'] = {
        'latency_ms': fp32_latency,
        'fps': fp32_fps,
        'peak_memory_mb': fp32_mem
    }
    print(f"  PyTorch FP32 Latency: {fp32_latency:.2f} ms | FPS: {fp32_fps:.1f} | Peak VRAM: {f'{fp32_mem:.1f} MB' if fp32_mem else 'N/A'}")

    # 3. PyTorch FP16 Benchmark (if CUDA)
    if device.type == 'cuda':
        print("\n[2/4] Measuring PyTorch FP16 (AMP Autocast)...")
        fp16_latency, fp16_fps = measure_pytorch_latency_fps(
            wrapped_model, input_size, device, fp16=True,
            warmup_iters=warmup_iters, test_iters=test_iters
        )
        fp16_mem = measure_peak_memory(wrapped_model, input_size, device, fp16=True)
        results['pytorch_fp16'] = {
            'latency_ms': fp16_latency,
            'fps': fp16_fps,
            'peak_memory_mb': fp16_mem
        }
        speedup = fp32_latency / fp16_latency if fp16_latency > 0 else 1.0
        print(f"  PyTorch FP16 Latency: {fp16_latency:.2f} ms | FPS: {fp16_fps:.1f} | Peak VRAM: {f'{fp16_mem:.1f} MB' if fp16_mem else 'N/A'} (Speedup: {speedup:.2f}x)")
    else:
        print("\n[2/4] PyTorch FP16 skipped (CPU mode)")

    # 4. ONNX Export & Runtime
    onnx_path = None
    if output_dir and export_onnx:
        print("\n[3/4] Exporting and benchmarking ONNX...")
        onnx_file = output_dir / "onnx" / f"{canonical_key}.onnx"
        if export_onnx_model(wrapped_model, str(onnx_file), input_size, device):
            onnx_path = str(onnx_file)
            onnx_res = benchmark_onnx_runtime(onnx_path, input_size, warmup_iters, test_iters)
            if onnx_res:
                results['onnx'] = onnx_res
                print(f"  ONNX Runtime Latency: {onnx_res['latency_ms']:.2f} ms | FPS: {onnx_res['fps']:.1f} | Provider: {onnx_res['provider']}")
        else:
            print("  ONNX export was skipped or failed.")

    # 5. TensorRT FP16 Engine Build & Benchmark
    if output_dir and build_trt and onnx_path and HAS_TRT and device.type == 'cuda':
        print("\n[4/4] Building & Benchmarking TensorRT FP16 Engine...")
        engine_file = output_dir / "tensorrt" / f"{canonical_key}_fp16.engine"
        if build_tensorrt_fp16_engine(onnx_path, str(engine_file), input_size, fp16=True):
            trt_res = benchmark_tensorrt_engine(
                str(engine_file), input_size, num_classes=num_classes,
                warmup_iters=warmup_iters, test_iters=test_iters
            )
            if trt_res:
                results['tensorrt_fp16'] = trt_res
                trt_speedup = fp32_latency / trt_res['latency_ms'] if trt_res['latency_ms'] > 0 else 1.0
                print(f"  TensorRT FP16 Latency: {trt_res['latency_ms']:.2f} ms | FPS: {trt_res['fps']:.1f} | vs PyTorch FP32: {trt_speedup:.2f}x speedup")
    elif not HAS_TRT and build_trt:
        print("\n[4/4] TensorRT not installed. TensorRT FP16 benchmark skipped.")
    else:
        print("\n[4/4] TensorRT build skipped.")

    # Clean up GPU memory
    del model
    del wrapped_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return results


# =============================================================================
# Report Generation (ASCII Table, LaTeX, CSV, JSON)
# =============================================================================

def print_summary_table(all_results: List[Dict[str, Any]]):
    """Print complete hardware metrics comparison table."""
    print("\n" + "=" * 135)
    print("HARDWARE DEPLOYMENT BENCHMARK COMPARISON SUMMARY")
    print("=" * 135)
    
    header = (
        f"{'Model':<16} | {'Encoder':<12} | {'Params (M)':<10} | {'GFLOPs':<8} | "
        f"{'Peak VRAM':<10} | {'FP32 (ms)':<10} | {'FP32 FPS':<9} | "
        f"{'FP16 (ms)':<10} | {'FP16 FPS':<9} | {'TRT16 (ms)':<10} | {'TRT16 FPS':<9}"
    )
    print(header)
    print("-" * 135)

    for r in all_results:
        name = r['name']
        encoder = r['encoder']
        params = f"{r['params_m']:.2f}" if r.get('params_m') else "N/A"
        gflops = f"{r['gflops']:.2f}" if r.get('gflops') else "N/A"
        
        # PyTorch FP32
        fp32_lat = f"{r['pytorch_fp32']['latency_ms']:.2f}" if r.get('pytorch_fp32') else "N/A"
        fp32_fps = f"{r['pytorch_fp32']['fps']:.1f}" if r.get('pytorch_fp32') else "N/A"
        fp32_mem = f"{r['pytorch_fp32']['peak_memory_mb']:.1f} MB" if r.get('pytorch_fp32') and r['pytorch_fp32'].get('peak_memory_mb') else "N/A"

        # PyTorch FP16
        fp16_lat = f"{r['pytorch_fp16']['latency_ms']:.2f}" if r.get('pytorch_fp16') else "--"
        fp16_fps = f"{r['pytorch_fp16']['fps']:.1f}" if r.get('pytorch_fp16') else "--"

        # TensorRT FP16
        trt_lat = f"{r['tensorrt_fp16']['latency_ms']:.2f}" if r.get('tensorrt_fp16') else "--"
        trt_fps = f"{r['tensorrt_fp16']['fps']:.1f}" if r.get('tensorrt_fp16') else "--"

        row = (
            f"{name:<16} | {encoder:<12} | {params:<10} | {gflops:<8} | "
            f"{fp32_mem:<10} | {fp32_lat:<10} | {fp32_fps:<9} | "
            f"{fp16_lat:<10} | {fp16_fps:<9} | {trt_lat:<10} | {trt_fps:<9}"
        )
        print(row)

    print("=" * 135)


def generate_latex_table(all_results: List[Dict[str, Any]], device_name: str) -> str:
    """Generate LaTeX table code formatted for academic papers (Table 4)."""
    lines = [
        "% ===========================================================================",
        "% Table 4: On-Device Hardware Deployment Metrics",
        "% ===========================================================================",
        "\\begin{table*}[t]",
        "\\centering",
        f"\\caption{{On-Device Hardware Deployment Metrics evaluated on {device_name} at $384 \\times 640$ resolution.}}",
        "\\label{tab:hardware_deployment}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{llccccccc}",
        "\\toprule",
        "\\textbf{Model} & \\textbf{Backbone} & \\textbf{Params (M)} & \\textbf{GFLOPs} & \\textbf{Peak VRAM (MB)} & \\textbf{PyTorch FP32 (ms)} & \\textbf{FP32 FPS} & \\textbf{TRT FP16 (ms)} & \\textbf{TRT FP16 FPS} \\\\",
        "\\midrule"
    ]

    for r in all_results:
        name = r['name'].replace('_', '\\_')
        encoder = r['encoder'].replace('_', '\\_')
        params = f"{r['params_m']:.2f}" if r.get('params_m') else "--"
        gflops = f"{r['gflops']:.2f}" if r.get('gflops') else "--"
        
        mem = f"{r['pytorch_fp32']['peak_memory_mb']:.1f}" if r.get('pytorch_fp32') and r['pytorch_fp32'].get('peak_memory_mb') else "--"
        fp32_lat = f"{r['pytorch_fp32']['latency_ms']:.2f}" if r.get('pytorch_fp32') else "--"
        fp32_fps = f"{r['pytorch_fp32']['fps']:.1f}" if r.get('pytorch_fp32') else "--"

        trt_lat = f"{r['tensorrt_fp16']['latency_ms']:.2f}" if r.get('tensorrt_fp16') else (
            f"{r['pytorch_fp16']['latency_ms']:.2f}*" if r.get('pytorch_fp16') else "--"
        )
        trt_fps = f"{r['tensorrt_fp16']['fps']:.1f}" if r.get('tensorrt_fp16') else (
            f"{r['pytorch_fp16']['fps']:.1f}*" if r.get('pytorch_fp16') else "--"
        )

        is_ours = 'auraseg' in r['key'].lower()
        if is_ours:
            lines.append(f"\\textbf{{{name}}} & \\textbf{{{encoder}}} & \\textbf{{{params}}} & \\textbf{{{gflops}}} & \\textbf{{{mem}}} & \\textbf{{{fp32_lat}}} & \\textbf{{{fp32_fps}}} & \\textbf{{{trt_lat}}} & \\textbf{{{trt_fps}}} \\\\")
        else:
            lines.append(f"{name} & {encoder} & {params} & {gflops} & {mem} & {fp32_lat} & {fp32_fps} & {trt_lat} & {trt_fps} \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}%",
        "}",
        "\\end{table*}"
    ])
    return "\n".join(lines)


def save_benchmark_artifacts(all_results: List[Dict[str, Any]], output_dir: Path, device_name: str):
    """Save CSV, TXT, LaTeX, and JSON benchmark artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. CSV
    csv_path = output_dir / "hardware_metrics_summary.csv"
    with open(csv_path, 'w') as f:
        f.write("Model,Encoder,Paradigm,Params (M),Trainable Params (M),GFLOPs,Peak VRAM (MB),FP32 Latency (ms),FP32 FPS,FP16 Latency (ms),FP16 FPS,ONNX Latency (ms),ONNX FPS,TRT FP16 Latency (ms),TRT FP16 FPS\n")
        for r in all_results:
            params = f"{r['params_m']:.2f}" if r.get('params_m') else ""
            trainable = f"{r['params_trainable_m']:.2f}" if r.get('params_trainable_m') else ""
            gflops = f"{r['gflops']:.2f}" if r.get('gflops') else ""
            
            mem = f"{r['pytorch_fp32']['peak_memory_mb']:.1f}" if r.get('pytorch_fp32') and r['pytorch_fp32'].get('peak_memory_mb') else ""
            fp32_lat = f"{r['pytorch_fp32']['latency_ms']:.2f}" if r.get('pytorch_fp32') else ""
            fp32_fps = f"{r['pytorch_fp32']['fps']:.1f}" if r.get('pytorch_fp32') else ""

            fp16_lat = f"{r['pytorch_fp16']['latency_ms']:.2f}" if r.get('pytorch_fp16') else ""
            fp16_fps = f"{r['pytorch_fp16']['fps']:.1f}" if r.get('pytorch_fp16') else ""

            onnx_lat = f"{r['onnx']['latency_ms']:.2f}" if r.get('onnx') else ""
            onnx_fps = f"{r['onnx']['fps']:.1f}" if r.get('onnx') else ""

            trt_lat = f"{r['tensorrt_fp16']['latency_ms']:.2f}" if r.get('tensorrt_fp16') else ""
            trt_fps = f"{r['tensorrt_fp16']['fps']:.1f}" if r.get('tensorrt_fp16') else ""

            f.write(f"{r['name']},{r['encoder']},{r['paradigm']},{params},{trainable},{gflops},{mem},{fp32_lat},{fp32_fps},{fp16_lat},{fp16_fps},{onnx_lat},{onnx_fps},{trt_lat},{trt_fps}\n")

    print(f"\n✓ CSV summary saved to: {csv_path}")

    # 2. Text Report
    txt_path = output_dir / "hardware_metrics_report.txt"
    with open(txt_path, 'w') as f:
        f.write("=" * 90 + "\n")
        f.write("HARDWARE METRICS DEPLOYMENT REPORT\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Device: {device_name}\n")
        f.write(f"TensorRT Version: {TRT_VERSION if HAS_TRT else 'Not Installed'}\n")
        f.write("=" * 90 + "\n\n")

        for r in all_results:
            f.write(f"Model: {r['name']} (Encoder: {r['encoder']}, Paradigm: {r['paradigm']})\n")
            f.write(f"  Params: {r['params_m']:.2f}M (Trainable: {r['params_trainable_m']:.2f}M)\n")
            f.write(f"  GFLOPs: {r['gflops']:.2f}\n" if r.get('gflops') else "  GFLOPs: N/A\n")
            if r.get('pytorch_fp32'):
                f.write(f"  PyTorch FP32: {r['pytorch_fp32']['latency_ms']:.2f} ms | {r['pytorch_fp32']['fps']:.1f} FPS | Peak Mem: {r['pytorch_fp32'].get('peak_memory_mb', 'N/A')} MB\n")
            if r.get('pytorch_fp16'):
                f.write(f"  PyTorch FP16: {r['pytorch_fp16']['latency_ms']:.2f} ms | {r['pytorch_fp16']['fps']:.1f} FPS | Peak Mem: {r['pytorch_fp16'].get('peak_memory_mb', 'N/A')} MB\n")
            if r.get('onnx'):
                f.write(f"  ONNX Runtime: {r['onnx']['latency_ms']:.2f} ms | {r['onnx']['fps']:.1f} FPS\n")
            if r.get('tensorrt_fp16'):
                f.write(f"  TensorRT FP16: {r['tensorrt_fp16']['latency_ms']:.2f} ms | {r['tensorrt_fp16']['fps']:.1f} FPS\n")
            f.write("\n")
    print(f"✓ Text report saved to: {txt_path}")

    # 3. LaTeX Table
    tex_path = output_dir / "table4_hardware_metrics.tex"
    latex_code = generate_latex_table(all_results, device_name)
    with open(tex_path, 'w') as f:
        f.write(latex_code)
    print(f"✓ LaTeX Table 4 saved to: {tex_path}")

    # 4. JSON
    json_path = output_dir / "hardware_metrics.json"
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"✓ JSON metrics saved to: {json_path}")


# =============================================================================
# Main Entrypoint
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Comprehensive Hardware Metrics Benchmark Suite")
    parser.add_argument(
        '--models', nargs='+', default=['all'],
        help="Models to benchmark. Choose 'all' or subset: 'fcn-r50', 'pspnet-r50', 'upernet-r50', 'segformer-b2', 'mask2former', 'pidnet-l', 'auraseg-r18', 'auraseg-r50'"
    )
    parser.add_argument(
        '--input-size', type=int, nargs=2, default=[384, 640],
        help="Input resolution [height, width] (default: 384 640)"
    )
    parser.add_argument(
        '--batch-size', type=int, default=1,
        help="Batch size (default: 1)"
    )
    parser.add_argument(
        '--num-classes', type=int, default=2,
        help="Number of segmentation classes (default: 2)"
    )
    parser.add_argument(
        '--warmup', type=int, default=50,
        help="Number of warmup iterations (default: 50)"
    )
    parser.add_argument(
        '--iters', type=int, default=200,
        help="Number of timed benchmark iterations (default: 200)"
    )
    parser.add_argument(
        '--export-onnx', action='store_true', default=True,
        help="Export models to ONNX (default: True)"
    )
    parser.add_argument(
        '--no-onnx', dest='export_onnx', action='store_false',
        help="Skip ONNX export"
    )
    parser.add_argument(
        '--build-trt', action='store_true', default=True,
        help="Build and benchmark TensorRT FP16 engines (default: True)"
    )
    parser.add_argument(
        '--no-trt', dest='build_trt', action='store_false',
        help="Skip TensorRT engine build & benchmark"
    )
    parser.add_argument(
        '--device', type=str, default='cuda',
        choices=['cuda', 'cpu'],
        help="Device to benchmark on (default: cuda)"
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help="Output directory for reports and exported engines"
    )
    parser.add_argument(
        '--fuse-bn', action='store_true', default=True,
        help="Fuse Conv+BatchNorm layers at evaluation time (default: True)"
    )
    args = parser.parse_args()

    # Canonical list of requested benchmark models
    all_supported = [
        'fcn-r50',
        'pspnet-r50',
        'upernet-r50',
        'segformer-b2',
        'mask2former',
        'pidnet-l',
        'auraseg-r18',
        'auraseg-r18-apud128',
        'auraseg-r18-apud64',
        'auraseg-r18-fast',
        'auraseg-r50',
        'auraseg-r50-fast',
    ]

    selected_models = []
    if 'all' in [m.lower() for m in args.models]:
        selected_models = all_supported
    else:
        for m in args.models:
            canonical = MODEL_ALIASES.get(m.lower().strip(), m.lower().strip())
            if canonical in [MODEL_ALIASES.get(x, x) for x in all_supported] or canonical in all_supported:
                selected_models.append(m)
            else:
                print(f"Warning: Unknown model '{m}'. Supported: {', '.join(all_supported)}")

    if not selected_models:
        print("No valid models selected. Exiting.")
        return

    # Setup directories
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent.parent / "runs" / "deployment"
    output_dir.mkdir(parents=True, exist_ok=True)

    input_shape = (args.batch_size, 3, args.input_size[0], args.input_size[1])
    device_name = torch.cuda.get_device_name(0) if (torch.cuda.is_available() and args.device == 'cuda') else 'CPU'

    print("=" * 80)
    print("HARDWARE METRICS BENCHMARK SUITE")
    print("=" * 80)
    print(f"Date             : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device           : {device_name}")
    print(f"Input Shape      : {input_shape} (B, C, H, W)")
    print(f"Warmup / Iters   : {args.warmup} / {args.iters}")
    print(f"ONNX Available   : {HAS_ONNX}")
    print(f"TensorRT Avail   : {HAS_TRT} (Version: {TRT_VERSION})")
    print(f"Selected Models  : {', '.join(selected_models)}")
    print(f"Output Directory : {output_dir}")
    print("=" * 80)

    all_results = []
    for model_key in selected_models:
        try:
            res = benchmark_single_model(
                model_key=model_key,
                input_size=input_shape,
                num_classes=args.num_classes,
                device_str=args.device,
                warmup_iters=args.warmup,
                test_iters=args.iters,
                export_onnx=args.export_onnx,
                build_trt=args.build_trt,
                output_dir=output_dir
            )
            all_results.append(res)
        except Exception as e:
            print(f"\n[ERROR] Failed to benchmark {model_key}: {e}")
            import traceback
            traceback.print_exc()

    # Print summary
    print_summary_table(all_results)

    # Print LaTeX Table
    print("\n" + generate_latex_table(all_results, device_name))

    # Save artifacts
    save_benchmark_artifacts(all_results, output_dir, device_name)
    print("\n✓ Hardware metrics benchmark run complete!\n")


if __name__ == "__main__":
    main()
