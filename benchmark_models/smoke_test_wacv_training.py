import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
import time

sys.path.insert(0, str(Path(__file__).parent))
from auraseg_r18_wacv import AURASeg_R18_WACV
from unified_dataset import Normalization, UnifiedDrivableAreaDataset
from wacv_losses import WACVCombinedLoss

def run_smoke_test():
    print("=" * 60)
    print("WACV TRAINING PIPELINE SMOKE TEST")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory Total: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    
    # Dataset Preparation
    data_root = Path(__file__).parent.parent / "CommonDataset"
    
    aug_params = {
        'shift_limit': 0.1,
        'scale_limit': 0.1,
        'rotate_limit': 15,
        'brightness_limit': 0.2,
        'contrast_limit': 0.2,
        'gauss_var_limit': (10.0, 50.0),
        'flip_p': 0.5,
        'color_p': 0.3,
        'noise_p': 0.2,
        'geom_p': 0.5
    }
    
    normalization = Normalization(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    
    print("\nInstantiating Training Dataset...")
    train_dataset = UnifiedDrivableAreaDataset(
        dataset_root=data_root, split='train',
        img_size=(384, 640), transform=True,
        normalization=normalization, return_names=False,
        aug_params=aug_params
    )
    
    print("\nConstructing Training DataLoader (Micro-Batch Size 4)...")
    train_loader = DataLoader(
        train_dataset, batch_size=4,
        shuffle=True, num_workers=4,
        pin_memory=True, drop_last=True
    )
    
    # Model Preparation
    print("\nInstantiating ResNet-18 WACV Architecture...")
    model = AURASeg_R18_WACV(
        num_classes=2,
        encoder_weights=None,
        fusion_type='mul',
        attention_mode='full',
        use_sobel=True,
        use_gate=True
    ).to(device)
    
    criterion = WACVCombinedLoss().to(device)
    
    print("\nFetching ONE Batch...")
    for images, masks in train_loader:
        break
        
    print(f"Image Shape: {images.shape}")
    print(f"Mask Shape: {masks.shape}")
    print(f"Image dtype: {images.dtype}")
    print(f"Mask dtype: {masks.dtype}")
    print(f"Mask unique values: {torch.unique(masks)}")
    
    print("\nSending Batch to GPU...")
    images, masks = images.to(device), masks.to(device)
    
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(0)
    
    print("\nPerforming Forward Pass with Deep Supervision & Boundary...")
    t0 = time.time()
    with autocast():
        outputs = model(images, return_aux=True, return_boundary=True)
        print(f"Main Output Shape: {outputs['main'].shape}")
        print(f"Aux Output Shapes: {[aux.shape for aux in outputs['aux']]}")
        print(f"Boundary Output Shape: {outputs['boundary'].shape}")
        print("Computing WACVCombinedLoss...")
        losses = criterion(outputs, masks)
        
    print("\nDividing Total Loss by 2 for Accumulation Simulation...")
    scaled_loss = losses['total'] / 2.0
        
    t1 = time.time()
    print(f"Forward Pass Time: {t1 - t0:.3f}s")
    
    print("\nPerforming Backward Pass...")
    t0 = time.time()
    scaled_loss.backward()
    t1 = time.time()
    print(f"Backward Pass Time: {t1 - t0:.3f}s")
    
    print("\n" + "=" * 60)
    print("SMOKE TEST RESULTS")
    print("=" * 60)
    print(f"Total Loss: {losses['total'].item():.4f}")
    print(f"Main Seg Loss: {losses['seg'].item():.4f}")
    print(f"Aux Loss (Total): {losses['aux'].item():.4f}")
    print(f"Boundary Loss: {losses['boundary'].item():.4f}")
    
    if device.type == 'cuda':
        peak_mem_alloc = torch.cuda.max_memory_allocated(0) / (1024**3)
        peak_mem_res = torch.cuda.max_memory_reserved(0) / (1024**3)
        print(f"\nPeak GPU Memory Allocated: {peak_mem_alloc:.2f} GB")
        print(f"Peak GPU Memory Reserved: {peak_mem_res:.2f} GB")
        
    print("\nSuccess! Aborting immediately before optimization.")
    sys.exit(0)

if __name__ == "__main__":
    run_smoke_test()
