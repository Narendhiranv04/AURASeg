import sys
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from unified_dataset import UnifiedDrivableAreaDataset as UDS_quick
from unified_dataset_main import UnifiedDrivableAreaDataset as UDS_main

MIX_ROOT = Path('/media/naren/Windows/Users/naren/Documents/AURASeg/CommonDataset')
splits = ['train', 'val', 'test']

total_compared = 0
exactly_identical = 0
different = 0
max_diff = 0

for split in splits:
    ds_main = UDS_main(MIX_ROOT, split, (384, 640), transform=False)
    ds_quick = UDS_quick(MIX_ROOT, split, (384, 640), transform=False)
    
    assert len(ds_main) == len(ds_quick), f"Length mismatch on {split}"
    print(f"Comparing {split}...")
    for i in tqdm(range(len(ds_main))):
        try:
            img_m, mask_m = ds_main[i]
            img_q, mask_q = ds_quick[i]
            
            diff = torch.sum(mask_m != mask_q).item()
            total_compared += 1
            if diff == 0:
                exactly_identical += 1
            else:
                different += 1
                if diff > max_diff:
                    max_diff = diff
        except FileNotFoundError:
            pass

print("--- MIX REGRESSION CHECK ---")
print(f"Number compared: {total_compared}")
print(f"Number exactly identical: {exactly_identical}")
print(f"Number different: {different}")
print(f"Maximum differing pixels in any mask: {max_diff}")

if different == 0:
    print("MIX MASK PIPELINE PARITY: EXACT")
else:
    print("MIX MASK PIPELINE PARITY: FAILED")
