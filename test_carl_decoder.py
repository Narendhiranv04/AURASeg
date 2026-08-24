import cv2
import numpy as np

def decode_carl_rgb_mask(mask_path, target_size=None):
    mask_img = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
    if mask_img is None:
        raise FileNotFoundError(f"Could not load mask: {mask_path}")
    mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB)

    decoded = np.zeros(mask_img.shape[:2], dtype=np.uint8)
    road_mask = (mask_img[:, :, 0] == 17) & (mask_img[:, :, 1] == 163) & (mask_img[:, :, 2] == 74)
    black_mask = (mask_img[:, :, 0] == 0) & (mask_img[:, :, 1] == 0) & (mask_img[:, :, 2] == 0)
    bg_mask = (mask_img[:, :, 0] == 15) & (mask_img[:, :, 1] == 16) & (mask_img[:, :, 2] == 65)

    valid_pixels = road_mask | black_mask | bg_mask
    if not np.all(valid_pixels):
        invalid_count = np.sum(~valid_pixels)
        raise ValueError(f"Found {invalid_count} pixels with unexpected colors in {mask_path}")

    decoded[road_mask] = 1
    decoded[black_mask] = 1
    
    if target_size is not None:
        decoded = cv2.resize(decoded, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)
        
    return decoded

m = decode_carl_rgb_mask('/media/naren/Windows/Users/naren/Documents/AURASeg/carl-dataset/test/labels/2 1723.jpg___fuse.png')
print(f"Decoded unique: {np.unique(m)}, sum: {np.sum(m)}, expected ~ 677140 (109) + 20 (0) = 677160 vs old logic which mapped 21 to 1 too.")
print(f"Total pixels: {m.size}")
