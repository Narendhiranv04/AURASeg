import cv2
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count
from collections import defaultdict
import time

ROAD_RGB = (17, 163, 74) # R, G, B
BACKGROUND_RGB = (15, 16, 65)

# Convert to BGR since cv2 reads in BGR
ROAD_BGR = (74, 163, 17)
BACKGROUND_BGR = (65, 16, 15)

DATA_ROOT = Path('/media/naren/Windows/Users/naren/Documents/AURASeg/carl-dataset')
splits = [
    DATA_ROOT / 'train' / 'labels',
    DATA_ROOT / 'val' / 'labels',
    DATA_ROOT / 'test' / 'labels'
]

def process_image(mask_path):
    img = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    
    pixels = img.reshape(-1, 3)
    total_pixels = pixels.shape[0]
    
    # Fast unique via 1D array
    # B, G, R -> R*65536 + G*256 + B
    pixels_1d = pixels[:, 2].astype(np.uint32) * 65536 + pixels[:, 1].astype(np.uint32) * 256 + pixels[:, 0].astype(np.uint32)
    unique_1d, counts = np.unique(pixels_1d, return_counts=True)
    
    result_counts = {}
    has_unexpected = False
    unexpected_pixels = 0
    
    for u1d, count in zip(unique_1d, counts):
        r = (u1d >> 16) & 0xFF
        g = (u1d >> 8) & 0xFF
        b = u1d & 0xFF
        c_tuple = (b, g, r) # Keep as BGR for result
        result_counts[c_tuple] = count
        if c_tuple != ROAD_BGR and c_tuple != BACKGROUND_BGR:
            unexpected_pixels += count
            has_unexpected = True
            
    return {
        'counts': result_counts,
        'has_unexpected': has_unexpected,
        'unexpected_pixels': unexpected_pixels,
        'total_pixels': total_pixels
    }

if __name__ == '__main__':
    all_masks = []
    for split_dir in splits:
        if split_dir.exists():
            all_masks.extend(list(split_dir.glob('*___fuse.png')))
    
    print(f"Total masks to process: {len(all_masks)}")
    
    color_counts = defaultdict(int)
    unexpected_images = 0
    unexpected_pixels = 0
    total_pixels = 0
    
    start_time = time.time()
    
    with Pool(processes=cpu_count()) as pool:
        # use a smaller chunksize for more frequent updates
        results = pool.imap_unordered(process_image, all_masks, chunksize=5)
        
        for i, res in enumerate(results):
            if res is None:
                continue
            
            for c_tuple, count in res['counts'].items():
                color_counts[c_tuple] += count
            
            if res['has_unexpected']:
                unexpected_images += 1
                
            unexpected_pixels += res['unexpected_pixels']
            total_pixels += res['total_pixels']
            
            if (i + 1) % 500 == 0:
                print(f"Processed {i + 1}/{len(all_masks)} masks... ({(time.time() - start_time):.2f}s)")
                
    print("\n--- RESULTS ---")
    print("Unique Colors (RGB):")
    for bgr, count in color_counts.items():
        rgb = (bgr[2], bgr[1], bgr[0])
        if rgb == ROAD_RGB:
            print(f"  ROAD {rgb}: {count}")
        elif rgb == BACKGROUND_RGB:
            print(f"  BACKGROUND {rgb}: {count}")
        else:
            print(f"  UNEXPECTED {rgb}: {count}")

    print(f"\nImages with only expected colors: {len(all_masks) - unexpected_images}")
    print(f"Images with unexpected colors: {unexpected_images}")
    print(f"Total pixels: {total_pixels}")
    if total_pixels > 0:
        print(f"Unexpected pixels: {unexpected_pixels} ({unexpected_pixels / total_pixels * 100:.6f}%)")
