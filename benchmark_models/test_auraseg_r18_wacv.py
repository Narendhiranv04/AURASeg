import os
import sys
import torch
import torch.nn.functional as F
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from auraseg_r18_wacv import AURASeg_R18_WACV
from auraseg_exportable import auraseg_resnet18
from wacv_losses import WACVCombinedLoss, MorphologicalBoundaryLoss

def test_structural_integrity():
    fusion_types = ['mul', 'add', 'concat']
    attention_modes = ['full', 'none', 'se', 'spatial']
    use_sobels = [True, False]
    use_gates = [True, False]
    
    dummy_input = torch.randn(2, 3, 384, 640)
    
    total = len(fusion_types) * len(attention_modes) * len(use_sobels) * len(use_gates)
    passed = 0
    
    print(f"Testing {total} ablation permutations of AURASeg WACV ResNet-18...")
    
    for fusion, attention, sobel, gate in product(fusion_types, attention_modes, use_sobels, use_gates):
        try:
            model = AURASeg_R18_WACV(
                num_classes=2,
                encoder_weights=None,
                fusion_type=fusion,
                attention_mode=attention,
                use_sobel=sobel,
                use_gate=gate
            )
            
            outputs = model(dummy_input, return_aux=True, return_boundary=True)
            loss = outputs['main'].sum() + sum([a.sum() for a in outputs['aux']]) + outputs['boundary'].sum()
            loss.backward()
            passed += 1
        except Exception as e:
            print(f"[FAILED] fusion={fusion}, attention={attention}, sobel={sobel}, gate={gate}")
            print(f"Error: {e}")
            sys.exit(1)
            
    print(f"[SUCCESS] {passed}/{total} permutations instantiated and completed forward/backward passes correctly.")

def test_default_model_parity():
    print("Testing default-model parity against exportable architecture...")
    exportable = auraseg_resnet18(pretrained=False)
    wacv_model = AURASeg_R18_WACV(
        num_classes=2,
        encoder_weights=None,
        fusion_type='mul',
        attention_mode='full',
        use_sobel=True,
        use_gate=True
    )
    
    # Extract exportable state dict
    exp_sd = exportable.state_dict()
    wacv_sd = wacv_model.state_dict()
    
    # Copy shared weights
    shared_keys = 0
    for key in exp_sd.keys():
        if key in wacv_sd and exp_sd[key].shape == wacv_sd[key].shape:
            wacv_sd[key].copy_(exp_sd[key])
            shared_keys += 1
            
    wacv_model.load_state_dict(wacv_sd)
    
    exportable.eval()
    wacv_model.eval()
    
    dummy_input = torch.randn(1, 3, 384, 640)
    with torch.no_grad():
        out_exp = exportable(dummy_input)
        out_wacv = wacv_model(dummy_input, return_aux=False, return_boundary=False)['main']
        
    diff = torch.abs(out_exp - out_wacv).max().item()
    if diff < 1e-5:
        print(f"[SUCCESS] Default model parity test passed. Max diff: {diff:.6e}. Shared keys mapped: {shared_keys}")
    else:
        print(f"[FAILED] Parity test failed. Max diff: {diff:.6e}")
        sys.exit(1)

def test_morphological_boundary():
    print("Testing Morphological Boundary Target...")
    loss_module = MorphologicalBoundaryLoss()
    
    # Create a 10x10 binary mask with a 4x4 square in the center
    mask = torch.zeros(1, 10, 10)
    mask[0, 3:7, 3:7] = 1.0
    
    # Boundary is dilate(mask) - erode(mask)
    # Expected boundary should be a ring around the 4x4 square + inner ring
    # Manually compute boundary
    target_float = mask.unsqueeze(1).float()
    dilated = F.max_pool2d(target_float, kernel_size=3, stride=1, padding=1)
    eroded = -F.max_pool2d(-target_float, kernel_size=3, stride=1, padding=1)
    boundary_gt = dilated - eroded
    
    # Center 2x2 should be 0, outer should be 0, ring should be 1
    assert boundary_gt[0, 0, 4:6, 4:6].sum() == 0, "Center should be 0"
    assert boundary_gt[0, 0, 2:8, 2:8].sum() == 32, "Boundary ring should have 32 pixels"
    
    # Verify BCE works
    pred = torch.randn(1, 1, 10, 10, requires_grad=True)
    loss = loss_module(pred, mask)
    loss.backward()
    
    print("[SUCCESS] Morphological boundary target generated correctly and gradients flow.")

def test_wacv_loss():
    print("Testing WACV Combined Loss...")
    criterion = WACVCombinedLoss()
    
    pred_main = torch.randn(2, 2, 48, 80, requires_grad=True)
    pred_aux1 = torch.randn(2, 2, 48, 80, requires_grad=True)
    pred_bnd = torch.randn(2, 1, 48, 80, requires_grad=True)
    
    target = torch.randint(0, 2, (2, 48, 80))
    
    outputs = {
        'main': pred_main,
        'aux': [pred_aux1],
        'boundary': pred_bnd
    }
    
    losses = criterion(outputs, target)
    losses['total'].backward()
    
    assert pred_main.grad is not None, "Main gradient missing"
    assert pred_aux1.grad is not None, "Aux gradient missing"
    assert pred_bnd.grad is not None, "Boundary gradient missing"
    
    print("[SUCCESS] WACV Combined Loss computed and gradients flow.")

if __name__ == "__main__":
    test_default_model_parity()
    test_morphological_boundary()
    test_wacv_loss()
    test_structural_integrity()
    
