"""
Test script to verify V4 RBRM integration
"""
import sys
sys.path.insert(0, 'c:/Users/naren/Documents/AURASeg/kobuki-yolop/model/new_network/ABLATION_11_DEC')

import torch

print('Testing V4 RBRM Integration...')
print('='*60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# Test RBRM module
from models.rbrm import RBRMModule
print('\n1. RBRM Module:')
rbrm = RBRMModule(256, 64).to(device)
x = torch.randn(2, 256, 96, 160).to(device)
out = rbrm(x, return_boundary=True)
print(f'   Input: {x.shape}')
print(f'   Features: {out["features"].shape}')
print(f'   Boundary: {out["boundary"].shape}')
rbrm_params = sum(p.numel() for p in rbrm.parameters())
print(f'   Params: {rbrm_params:,}')
print('   ✓ RBRM passed')

# Test V4 model
from models.v4_rbrm import V4RBRM
print('\n2. V4 RBRM Model:')
model = V4RBRM(in_channels=3, num_classes=2).to(device)
x = torch.randn(2, 3, 384, 640).to(device)
with torch.no_grad():
    outputs = model(x, return_aux=True, return_boundary=True)
print(f'   Input: {x.shape}')
print(f'   Main: {outputs["main"].shape}')
print(f'   Boundary: {outputs["boundary"].shape}')
print(f'   Aux outputs: {len(outputs["aux"])}')
for i, aux in enumerate(outputs["aux"]):
    print(f'     Aux-{i+1}: {aux.shape}')
total_params = sum(p.numel() for p in model.parameters())
print(f'   Total Params: {total_params:,}')
print('   ✓ V4 Model passed')

# Test V4 Loss
from losses import V4CombinedLoss
print('\n3. V4 Combined Loss:')
outputs = model(x, return_aux=True, return_boundary=True)  # Need grad-enabled forward
criterion = V4CombinedLoss().to(device)
targets = torch.randint(0, 2, (2, 384, 640)).to(device)
losses = criterion(outputs, targets)
print(f'   Total: {losses["total"].item():.4f}')
print(f'   Main: {losses["main"].item():.4f}')
print(f'   Aux weighted: {losses["aux_weighted"].item():.4f}')
print(f'   Boundary: {losses["boundary"].item():.4f}')
print('   ✓ V4 Loss passed')

# Test backward pass
print('\n4. Backward Pass:')
losses['total'].backward()
grads = sum(1 for p in model.parameters() if p.grad is not None)
total_p = sum(1 for p in model.parameters())
print(f'   Params with gradients: {grads}/{total_p}')
print('   ✓ Backward passed')

# Param group test
print('\n5. Differential Learning Rates:')
param_groups = model.get_param_groups(lr_pretrained=1e-4, lr_new=1e-3)
for group in param_groups:
    n_params = sum(p.numel() for p in group['params'])
    print(f'   {group["name"]}: {n_params:,} params @ lr={group["lr"]}')

print('\n' + '='*60)
print('V4 RBRM INTEGRATION TEST PASSED!')
print('='*60)
