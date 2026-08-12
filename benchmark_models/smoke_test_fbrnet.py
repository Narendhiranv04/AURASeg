import os
import sys
import torch
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fbrnet_wacv_model import FBRNet_WACV

def test_structural():
    print("="*60)
    print("FBRNET STRUCTURAL TEST")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.reset_peak_memory_stats()
        
    model = FBRNet_WACV(num_classes=2, aux_mode='train').to(device)
    model.train()
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    
    x = torch.randn(2, 3, 384, 640).to(device)
    y = torch.randint(0, 2, (2, 384, 640)).to(device)
    
    print("\nRunning Forward Pass...")
    outputs = model(x)
    # feat_ffm, feat_out32, feat_out16, aux_0, aux_1 = outputs
    
    assert len(outputs) == 5, f"Expected 5 outputs, got {len(outputs)}"
    
    names = ['feat_ffm (Main)', 'feat_out32 (Aux0)', 'feat_out16 (Aux1)', 'aux_0 (Aux2)', 'aux_1 (Aux3)']
    for name, out in zip(names, outputs):
        print(f"{name} Shape: {list(out.shape)}")
        assert out.shape == (2, 2, 384, 640), f"{name} shape mismatch: {out.shape}"
        assert not torch.isnan(out).any(), f"NaN in {name}"
        assert not torch.isinf(out).any(), f"Inf in {name}"
        
    print("\nComputing Native Loss...")
    feat_ffm, feat_out32, feat_out16, aux_0, aux_1 = outputs
    
    loss_main = F.cross_entropy(feat_ffm, y, ignore_index=255)
    loss_aux16 = F.cross_entropy(feat_out16, y, ignore_index=255) 
    loss_aux0 = F.cross_entropy(aux_0, y, ignore_index=255)       
    loss_aux1 = F.cross_entropy(aux_1, y, ignore_index=255)       
    
    total_loss = loss_main + 2 * loss_aux16 + 5 * loss_aux0 + 3 * loss_aux1
    
    print(f"Loss Main: {loss_main.item():.4f}")
    print(f"Loss Aux16: {loss_aux16.item():.4f}")
    print(f"Loss Aux0: {loss_aux0.item():.4f}")
    print(f"Loss Aux1: {loss_aux1.item():.4f}")
    print(f"Total Loss: {total_loss.item():.4f}")
    
    assert torch.isfinite(total_loss), "Total loss is not finite!"
    
    print("\nRunning Backward Pass...")
    total_loss.backward()
    
    def check_grad(name, module):
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters() if p.requires_grad)
        print(f"[{'PASS' if has_grad else 'FAIL'}] {name} received gradients")
        
    check_grad('Encoder (ResNet)', model.model.cp.resnet)
    check_grad('arASPP', model.model.cp.araspp)
    check_grad('CSFM', model.model.cp.ffm32_16)
    check_grad('LABRM', model.model.cp.brm)
    check_grad('Main Classifier', model.model.conv_out)
    
    if torch.cuda.is_available():
        allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
        print(f"\nPeak CUDA Allocated Memory: {allocated:.2f} MB")
        print(f"Peak CUDA Reserved Memory: {reserved:.2f} MB")
        
    print("\nStructural test SUCCESS.")
    
if __name__ == '__main__':
    test_structural()
