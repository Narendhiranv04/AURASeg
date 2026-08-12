import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from auraseg_r18_wacv import AURASeg_R18_WACV
from wacv_losses import WACVCombinedLoss

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def run_structural_test():
    print("="*60)
    print("SIMPLE BOUNDARY HEAD STRUCTURAL TEST")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 1. Instantiate default full model
    model_full = AURASeg_R18_WACV(
        num_classes=2, encoder_weights=None, fusion_type='mul', 
        attention_mode='full', use_sobel=True, use_gate=True,
        simple_boundary_head=False
    ).to(device)
    
    # 2. Instantiate simple-boundary-head model
    model_simple = AURASeg_R18_WACV(
        num_classes=2, encoder_weights=None, fusion_type='mul', 
        attention_mode='full', use_sobel=True, use_gate=True,
        simple_boundary_head=True
    ).to(device)
    
    params_full = count_parameters(model_full)
    params_simple = count_parameters(model_simple)
    
    print(f"\nParameter Count (Full RBRM):     {params_full:,}")
    print(f"Parameter Count (Simple Head):   {params_simple:,}")
    
    dummy_input = torch.randn(2, 3, 384, 640).to(device)
    dummy_target = torch.randint(0, 2, (2, 384, 640)).to(device)
    criterion = WACVCombinedLoss().to(device)
    
    print("\n--- Testing Simple Boundary Head Model ---")
    try:
        # Verify forward
        outputs = model_simple(dummy_input, return_aux=True, return_boundary=True)
        print("[SUCCESS] Forward pass completed.")
        
        # Verify shapes
        assert 'main' in outputs, "Missing main output"
        assert outputs['main'].shape == (2, 2, 384, 640), f"Main shape mismatch: {outputs['main'].shape}"
        print("[SUCCESS] Main shape verified.")
        
        assert 'aux' in outputs, "Missing aux output"
        assert len(outputs['aux']) == 4, f"Expected 4 aux, got {len(outputs['aux'])}"
        for i, aux in enumerate(outputs['aux']):
            assert aux.shape == (2, 2, 384, 640), f"Aux shape mismatch: {aux.shape}"
        print("[SUCCESS] Aux shapes verified.")
        
        assert 'boundary' in outputs, "Missing boundary output"
        assert outputs['boundary'].shape == (2, 1, 384, 640), f"Boundary shape mismatch: {outputs['boundary'].shape}"
        print("[SUCCESS] Boundary shape verified.")
        
        # Compute Loss
        losses = criterion(outputs, dummy_target)
        print(f"[SUCCESS] WACVCombinedLoss computed (Total: {losses['total'].item():.4f}).")
        
        # Backward
        losses['total'].backward()
        print("[SUCCESS] Backward pass completed successfully.")
        
    except Exception as e:
        print(f"\n[FAILED] Error during simple model structural test: {e}")
        sys.exit(1)
        
    print("\nAll structural tests passed successfully! Stopping.")

if __name__ == "__main__":
    run_structural_test()
