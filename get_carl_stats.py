import cv2
import numpy as np
from pathlib import Path
import sys
from multiprocessing import Pool, cpu_count
import time

DATA_ROOT = Path('/media/naren/Windows/Users/naren/Documents/AURASeg/carl-dataset')
splits = ['train', 'val', 'test']

def process_image(mask_path):
    img = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    
    pixels = img.reshape(-1, 3)
    total = pixels.shape[0]
    
    # Fast unique via 1D array
    pixels_1d = pixels[:, 2].astype(np.uint32) * 65536 + pixels[:, 1].astype(np.uint32) * 256 + pixels[:, 0].astype(np.uint32)
    unique_1d, counts = np.unique(pixels_1d, return_counts=True)
    
    road = 0
    bg = 0
    black = 0
    
    # ROAD: 17, 163, 74 -> BGR: 74, 163, 17 -> 17*65536 + 163*256 + 74 = 1114112 + 41728 + 74 = 1155914
    # BG: 15, 16, 65 -> BGR: 65, 16, 15 -> 15*65536 + 16*256 + 65 = 983040 + 4096 + 65 = 987201
    # BLACK: 0, 0, 0
    ROAD_ID = 17*65536 + 163*256 + 74
    BG_ID = 15*65536 + 16*256 + 65
    BLACK_ID = 0
    
    for u1d, count in zip(unique_1d, counts):
        if u1d == ROAD_ID:
            road += count
        elif u1d == BG_ID:
            bg += count
        elif u1d == BLACK_ID:
            black += count
            
    # Black is mapped to ROAD
    road_total = road + black
    bg_total = bg
            
    return {
        'total': total,
        'road': road_total,
        'bg': bg_total,
        'black': black
    }

if __name__ == '__main__':
    for split in splits:
        split_dir = DATA_ROOT / split / 'labels'
        masks = list(split_dir.glob('*___fuse.png'))
        if not masks:
            continue
            
        print(f"\nProcessing {split} split ({len(masks)} samples)...")
        total_px = 0
        road_px = 0
        bg_px = 0
        black_px = 0
        
        with Pool(processes=cpu_count()) as pool:
            results = pool.imap_unordered(process_image, masks, chunksize=10)
            for res in results:
                if res is not None:
                    total_px += res['total']
                    road_px += res['road']
                    bg_px += res['bg']
                    black_px += res['black']
                    
        print(f"Sample count: {len(masks)}")
        print(f"Road pixel count: {road_px}")
        print(f"Background pixel count: {bg_px}")
        print(f"Road percentage: {road_px / total_px * 100:.4f}%")
        print(f"Background percentage: {bg_px / total_px * 100:.4f}%")
        print(f"Original (0,0,0) pixels: {black_px} ({black_px / total_px * 100:.6f}%)")
