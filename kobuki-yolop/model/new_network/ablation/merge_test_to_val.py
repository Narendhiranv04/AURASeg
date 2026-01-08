"""
Merge Test to Val Script
========================
Moves labeled test images to the validation set to increase validation size.
Only moves images that have a corresponding label.
"""

import os
import shutil
from tqdm import tqdm

def merge_test_to_val():
    dataset_dir = r"C:\Users\naren\Documents\AURASeg\CommonDataset"
    
    test_img_dir = os.path.join(dataset_dir, 'images', 'test')
    test_lbl_dir = os.path.join(dataset_dir, 'labels', 'test')
    
    val_img_dir = os.path.join(dataset_dir, 'images', 'val')
    val_lbl_dir = os.path.join(dataset_dir, 'labels', 'val')
    
    if not os.path.exists(test_img_dir):
        print("Test directory does not exist.")
        return

    # Get list of test images
    test_images = os.listdir(test_img_dir)
    
    moved_count = 0
    skipped_count = 0
    
    print(f"Scanning {len(test_images)} test images...")
    
    for img_name in tqdm(test_images):
        # Determine label name (assuming same basename, png extension)
        basename = os.path.splitext(img_name)[0]
        label_name = basename + ".png"
        
        src_img = os.path.join(test_img_dir, img_name)
        src_lbl = os.path.join(test_lbl_dir, label_name)
        
        if os.path.exists(src_lbl):
            # Move both to val
            dst_img = os.path.join(val_img_dir, img_name)
            dst_lbl = os.path.join(val_lbl_dir, label_name)
            
            shutil.move(src_img, dst_img)
            shutil.move(src_lbl, dst_lbl)
            moved_count += 1
        else:
            skipped_count += 1
            
    print("\nMerge Complete!")
    print(f"Moved {moved_count} pairs to Validation set.")
    print(f"Skipped {skipped_count} images (no label found).")
    
    # Count final val size
    val_count = len(os.listdir(val_img_dir))
    print(f"New Validation Set Size: {val_count}")

if __name__ == "__main__":
    merge_test_to_val()
