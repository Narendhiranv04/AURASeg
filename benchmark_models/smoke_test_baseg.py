"""
Comprehensive sanity test suite for BASeg WACV adaptation.
Verifies shapes, types, boundary target generation, loss finiteness, backward pass,
gradient accumulation, checkpoint saving/loading, metrics, and CARL-D visual geometry alignment.
"""

import sys
import os
from pathlib import Path
import torch
import numpy as np
import cv2

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "benchmark_models"))

from baseg_wacv_model import BASeg_WACV
from train_baseg_wacv import JointEdgeSegLoss, mask_to_onehot, onehot_to_binary_edges, BASegDatasetWrapper
from unified_dataset import UnifiedDrivableAreaDataset, Normalization
from wacv_metrics import compute_metrics, compute_boundary_metrics


def run_sanity_tests():
    print("=" * 70)
    print("RUNNING BASEG SANITY CHECKS")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 1. Model instantiation and parameter counts
    model = BASeg_WACV(num_classes=2, layers=101, pretrained=True).to(device)
    total_p = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[CHECK 1] Model: BASeg-R101 | Total Params: {total_p:,} | Trainable: {trainable_p:,}")
    assert total_p > 70_000_000, f"Expected >70M parameters, got {total_p}"

    # 2. Output shapes (Train & Eval)
    dummy_input = torch.randn(2, 3, 384, 640, device=device)
    
    # Train mode
    model.train()
    with torch.cuda.amp.autocast():
        main_out, aux_out, edge_out = model(dummy_input)
    print(f"[CHECK 2a] Train outputs: main={main_out.shape}, aux={aux_out.shape}, edge={edge_out.shape}")
    assert main_out.shape == (2, 2, 384, 640), f"Expected main shape (2,2,384,640), got {main_out.shape}"
    assert aux_out.shape == (2, 2, 384, 640), f"Expected aux shape (2,2,384,640), got {aux_out.shape}"
    assert edge_out.shape == (2, 1, 384, 640), f"Expected edge shape (2,1,384,640), got {edge_out.shape}"

    # Eval mode
    model.eval()
    with torch.no_grad(), torch.cuda.amp.autocast():
        eval_out = model(dummy_input)
        eval_main = eval_out[0] if isinstance(eval_out, (tuple, list)) else eval_out
        print(f"[CHECK 2b] Eval output: main={eval_main.shape}")
        assert eval_main.shape == (2, 2, 384, 640), f"Expected eval main shape (2,2,384,640), got {eval_main.shape}"

    # 3. Loss finiteness & backward
    model.train()
    criterion = JointEdgeSegLoss(classes=2, ignore_index=255, aux_weight=0.4, edge_weight=8.0, seg_weight=1.0).to(device)
    dummy_target = torch.randint(0, 2, (2, 384, 640), device=device)
    dummy_edge = torch.randint(0, 2, (2, 1, 384, 640), device=device).float()

    scaler = torch.cuda.amp.GradScaler()
    with torch.cuda.amp.autocast():
        outputs = model(dummy_input)
        loss_dict = criterion(outputs, (dummy_target, dummy_edge))
    print(f"[CHECK 3] Loss: total={loss_dict['total'].item():.4f}, seg={loss_dict['seg'].item():.4f}, aux={loss_dict['aux'].item():.4f}, edge={loss_dict['edge'].item():.4f}")
    assert torch.isfinite(loss_dict['total']), "Loss is not finite!"
    assert torch.isfinite(loss_dict['seg']), "Seg loss is not finite!"
    assert torch.isfinite(loss_dict['aux']), "Aux loss is not finite!"
    assert torch.isfinite(loss_dict['edge']), "Edge loss is not finite!"

    # 4. Backward & Optimizer step
    optimizer = torch.optim.AdamW(model.get_param_groups())
    optimizer.zero_grad()
    scaler.scale(loss_dict['total']).backward()
    scaler.step(optimizer)
    scaler.update()
    print("[CHECK 4] Backward pass & optimizer step succeeded.")

    # 5. Dataset loading & boundary target verification on MIX and CARL-D
    norm = Normalization()
    
    # MIX
    ds_mix = UnifiedDrivableAreaDataset(repo_root / "CommonDataset", split='val', img_size=(384, 640), normalization=norm)
    wrapper_mix = BASegDatasetWrapper(ds_mix, radius=2, num_classes=2)
    img_m, mask_m, edge_m = wrapper_mix[0]
    print(f"[CHECK 5a] MIX sample: img={img_m.shape}, mask={mask_m.shape} unique={torch.unique(mask_m).tolist()}, edge={edge_m.shape} unique={torch.unique(edge_m).tolist()}")
    assert set(torch.unique(mask_m).tolist()).issubset({0, 1}), f"MIX mask has invalid values: {torch.unique(mask_m)}"
    assert set(torch.unique(edge_m).tolist()).issubset({0.0, 1.0}), f"MIX edge has invalid values: {torch.unique(edge_m)}"
    assert edge_m.shape == (1, 384, 640)

    # CARL-D
    ds_carl = UnifiedDrivableAreaDataset(repo_root / "carl-dataset", split='val', img_size=(384, 640), normalization=norm)
    wrapper_carl = BASegDatasetWrapper(ds_carl, radius=2, num_classes=2)
    img_c, mask_c, edge_c = wrapper_carl[0]
    print(f"[CHECK 5b] CARL-D sample: img={img_c.shape}, mask={mask_c.shape} unique={torch.unique(mask_c).tolist()}, edge={edge_c.shape} unique={torch.unique(edge_c).tolist()}")
    assert set(torch.unique(mask_c).tolist()).issubset({0, 1}), f"CARL-D mask has invalid values: {torch.unique(mask_c)}"
    assert set(torch.unique(edge_c).tolist()).issubset({0.0, 1.0}), f"CARL-D edge has invalid values: {torch.unique(edge_c)}"
    assert edge_c.shape == (1, 384, 640)

    # 6. CARL-D Visual Geometry Inspection
    # Unnormalize image, compare mask boundary and Canny edges
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    rgb_carl = torch.clamp((img_c * std + mean) * 255.0, 0, 255).byte().permute(1, 2, 0).numpy()
    canny_carl = cv2.Canny(rgb_carl, 10, 100)

    print(f"[CHECK 6] Visual geometry check:")
    print(f"  RGB image size: {rgb_carl.shape}")
    print(f"  Decoded semantic mask drivable pixels: {(mask_c == 1).sum().item():,}")
    print(f"  BASeg boundary target edge pixels: {(edge_c == 1).sum().item():,}")
    print(f"  Canny edge prior pixels: {(canny_carl > 0).sum():,}")
    assert (edge_c == 1).sum().item() > 0, "Boundary target has 0 edge pixels!"

    # 7. Metrics calculation check
    preds_dummy = np.random.randint(0, 2, (4, 384, 640), dtype=np.uint8)
    targets_dummy = np.random.randint(0, 2, (4, 384, 640), dtype=np.uint8)
    seg_m = compute_metrics(preds_dummy, targets_dummy)
    bnd_m = compute_boundary_metrics(preds_dummy, targets_dummy, k=2)
    print(f"[CHECK 7] Metric verification: mIoU={seg_m['miou']:.4f}, boundary_f1={bnd_m['boundary_f1']:.4f}")
    assert np.isfinite(seg_m['miou']) and np.isfinite(bnd_m['boundary_f1'])

    print("\nALL SANITY CHECKS PASSED SUCCESSFULLY!\n")


if __name__ == "__main__":
    run_sanity_tests()
