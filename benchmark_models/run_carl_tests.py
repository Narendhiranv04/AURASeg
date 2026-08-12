import torch
import numpy as np
import cv2
from pathlib import Path
import random
import sys
from PIL import Image

sys.path.insert(0, '/media/naren/Windows/Users/naren/Documents/AURASeg/benchmark_models')
from unified_dataset import UnifiedDrivableAreaDataset, Normalization, decode_carl_rgb_mask
from wacv_metrics import compute_boundary_metrics

DATA_ROOT = Path('/media/naren/Windows/Users/naren/Documents/AURASeg/carl-dataset')
DIAG_DIR = Path('/media/naren/Windows/Users/naren/Documents/AURASeg/runs_carl/diagnostics_rgb')
DIAG_DIR.mkdir(parents=True, exist_ok=True)

print("=== 3. VISUAL SANITY ===")
splits = ['train', 'val', 'test']
for split in splits:
    u_ds = UnifiedDrivableAreaDataset(DATA_ROOT, split, (384, 640), False, Normalization(), return_names=True)
    indices = random.sample(range(len(u_ds)), min(7, len(u_ds)))
    
    for idx in indices:
        img_t, mask_t, name = u_ds[idx]
        
        # Load original RGB annotation
        label_path = u_ds.labels_dir / (Path(name).name + "___fuse.png")
        if not label_path.exists():
            label_path = u_ds.labels_dir / (Path(name).stem + "___fuse.png")
            
        orig_rgb = cv2.imread(str(label_path), cv2.IMREAD_COLOR)
        orig_rgb = cv2.cvtColor(orig_rgb, cv2.COLOR_BGR2RGB)
        
        # decoded binary mask
        decoded = mask_t.numpy().astype(np.uint8) * 255
        
        # image
        # unnormalize
        mean = np.array([0.485, 0.456, 0.406]).reshape(3,1,1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3,1,1)
        img_rgb = (img_t.numpy() * std + mean) * 255.0
        img_rgb = np.clip(img_rgb, 0, 255).astype(np.uint8).transpose(1, 2, 0)
        
        # resize orig_rgb to match
        orig_rgb = cv2.resize(orig_rgb, (640, 384), interpolation=cv2.INTER_NEAREST)
        
        # Save
        decoded_3c = cv2.cvtColor(decoded, cv2.COLOR_GRAY2RGB)
        panel = np.concatenate([img_rgb, orig_rgb, decoded_3c], axis=1)
        panel = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(DIAG_DIR / f"{split}_{idx}_{name}.png"), panel)
print(f"Saved 21 visual diagnostics to {DIAG_DIR}")

print("\n=== 4. GT BOUNDARY SANITY ===")
test_ds = UnifiedDrivableAreaDataset(DATA_ROOT, 'test', (384, 640), False, Normalization(), return_names=True)
bnd_pixels = []
zero_count = 0
nonzero_count = 0
kernel = np.ones((3, 3), np.uint8)

for idx in range(len(test_ds)):
    _, mask_t, _ = test_ds[idx]
    target_binary = mask_t.numpy().astype(np.uint8)
    target_boundary = cv2.morphologyEx(target_binary, cv2.MORPH_GRADIENT, kernel)
    
    cnt = np.sum(target_boundary)
    bnd_pixels.append(cnt)
    if cnt == 0:
        zero_count += 1
    else:
        nonzero_count += 1

print("GT boundary pixels/image (test split):")
print(f"  Mean: {np.mean(bnd_pixels):.2f}")
print(f"  Median: {np.median(bnd_pixels)}")
print(f"  Min: {np.min(bnd_pixels)}")
print(f"  Max: {np.max(bnd_pixels)}")
print(f"  25th percentile: {np.percentile(bnd_pixels, 25)}")
print(f"  75th percentile: {np.percentile(bnd_pixels, 75)}")
print(f"Masks with ZERO GT boundary pixels: {zero_count}")
print(f"Masks with NONZERO GT boundary pixels: {nonzero_count}")

print("\n=== 5. SYNTHETIC METRIC TESTS ===")
# Create a normal polygon (e.g. bottom half is road)
h, w = 384, 640
poly = np.zeros((h, w), dtype=np.uint8)
poly[h//2:, :] = 1

# A. identical
res_A = compute_boundary_metrics(poly[None, ...], poly[None, ...], k=2)
print("A. Identical normal polygon:")
print(f"   BIoU: {res_A['boundary_iou']:.4f}")
print(f"   BF1:  {res_A['boundary_f1']:.4f}")

# B. shifted polygon (shift down by 2 pixels)
poly_shift = np.zeros((h, w), dtype=np.uint8)
poly_shift[h//2+2:, :] = 1
res_B = compute_boundary_metrics(poly_shift[None, ...], poly[None, ...], k=2)
print("B. Shifted road polygon (down by 2 pixels):")
print(f"   BIoU: {res_B['boundary_iou']:.4f}")
print(f"   BF1:  {res_B['boundary_f1']:.4f}")

# C. non-overlapping polygon
poly_non = np.zeros((h, w), dtype=np.uint8)
poly_non[:h//2-10, :] = 1
res_C = compute_boundary_metrics(poly_non[None, ...], poly[None, ...], k=2)
print("C. Non-overlapping polygon:")
print(f"   BIoU: {res_C['boundary_iou']:.4f}")
print(f"   BF1:  {res_C['boundary_f1']:.4f}")

print("\n=== 6. MIX PARITY ===")
MIX_ROOT = Path('/media/naren/Windows/Users/naren/Documents/AURASeg/CommonDataset')
mix_ds = UnifiedDrivableAreaDataset(MIX_ROOT, 'test', (384, 640), False, Normalization(), return_names=True)
img, mask, name = mix_ds[0]
print(f"MIX dataset loaded 0 successfully: {name}, mask unique: {np.unique(mask.numpy())}")
print("MIX parity PASS")
