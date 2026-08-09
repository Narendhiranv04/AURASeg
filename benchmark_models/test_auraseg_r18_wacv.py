import os
import sys
import torch
import torch.nn.functional as F
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from auraseg_r18_wacv import AURASeg_R18_WACV

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
        config_name = f"fusion={fusion}, attention={attention}, sobel={sobel}, gate={gate}"
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
            
            # Check shapes
            assert 'main' in outputs, "Missing main output"
            assert outputs['main'].shape == (2, 2, 384, 640), f"Main shape mismatch: {outputs['main'].shape}"
            
            assert 'aux' in outputs, "Missing aux outputs"
            assert len(outputs['aux']) == 4, f"Expected 4 aux outputs, got {len(outputs['aux'])}"
            for i, aux in enumerate(outputs['aux']):
                assert aux.shape == (2, 2, 384, 640), f"Aux {i} shape mismatch: {aux.shape}"
                
            assert 'boundary' in outputs, "Missing boundary output"
            assert outputs['boundary'].shape == (2, 1, 384, 640), f"Boundary shape mismatch: {outputs['boundary'].shape}"
            
            # Check backward pass
            loss = outputs['main'].sum() + sum([a.sum() for a in outputs['aux']]) + outputs['boundary'].sum()
            loss.backward()
            
            passed += 1
            
        except Exception as e:
            print(f"[FAILED] {config_name}")
            print(f"Error: {e}")
            sys.exit(1)
            
    print(f"\\n[SUCCESS] {passed}/{total} permutations instantiated and completed forward/backward passes correctly.")

if __name__ == "__main__":
    test_structural_integrity()
