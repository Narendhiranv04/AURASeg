"""
Test V3 Model Integration with Deep Supervision Loss
"""

import torch
from models.v3_apud import V3APUD
from losses import DeepSupervisionLoss

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    
    # Create model and loss
    model = V3APUD().to(device)
    criterion = DeepSupervisionLoss()
    
    # Dummy input
    x = torch.randn(2, 3, 384, 640).to(device)
    targets = torch.randint(0, 2, (2, 384, 640)).to(device)
    
    # Forward pass
    outputs = model(x, return_aux=True)
    print(f'Main output: {outputs["main"].shape}')
    print(f'Aux outputs: {[a.shape for a in outputs["aux"]]}')
    
    # Compute loss
    losses = criterion(outputs, targets)
    print(f'Total loss: {losses["total"].item():.4f}')
    print(f'Main loss: {losses["main"].item():.4f}')
    print(f'Aux weighted: {losses["aux_weighted"].item():.4f}')
    for i, aux_loss in enumerate(losses['aux']):
        print(f'  Aux-{i+1}: {aux_loss.item():.4f}')
    
    # Backward pass
    losses['total'].backward()
    print('Backward pass successful!')
    
    # Check gradients
    grad_count = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_count += 1
    print(f'Parameters with gradients: {grad_count}')

if __name__ == "__main__":
    main()
