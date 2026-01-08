"""
Dataset Verification and Cleanup Script
=======================================
1. Verifies that every image in Train/Val has a corresponding label.
2. Verifies that every label in Train/Val has a corresponding image.
3. Removes the 'test' directory if it contains only unlabeled data, to comply with "everything is either train or val".
"""

import os
import shutil

def verify_and_clean():
    dataset_dir = r"C:\Users\naren\Documents\AURASeg\CommonDataset"
    
    splits = ['train', 'val']
    
    print("Verifying Dataset Integrity...")
    print("=" * 30)
    
    for split in splits:
        img_dir = os.path.join(dataset_dir, 'images', split)
        lbl_dir = os.path.join(dataset_dir, 'labels', split)
        
        images = set(os.listdir(img_dir))
        labels = set(os.listdir(lbl_dir))
        
        # Check for images without labels
        img_names = {os.path.splitext(f)[0] for f in images}
        lbl_names = {os.path.splitext(f)[0] for f in labels}
        
        missing_labels = img_names - lbl_names
        missing_images = lbl_names - img_names
        
        print(f"\nChecking {split.upper()} split:")
        print(f"  Images: {len(images)}")
        print(f"  Labels: {len(labels)}")
        
        if not missing_labels and not missing_images:
            print(f"  ✅ PERFECT MATCH: All {len(images)} images have labels.")
        else:
            if missing_labels:
                print(f"  ❌ ERROR: {len(missing_labels)} images are missing labels!")
                # print(list(missing_labels)[:5])
            if missing_images:
                print(f"  ❌ ERROR: {len(missing_images)} labels are missing images!")
                # print(list(missing_images)[:5])

    # Cleanup Test Folder
    test_img_dir = os.path.join(dataset_dir, 'images', 'test')
    test_lbl_dir = os.path.join(dataset_dir, 'labels', 'test')
    
    if os.path.exists(test_img_dir):
        test_images = os.listdir(test_img_dir)
        test_labels = os.listdir(test_lbl_dir) if os.path.exists(test_lbl_dir) else []
        
        print(f"\nChecking TEST folder (to be removed):")
        print(f"  Images: {len(test_images)}")
        print(f"  Labels: {len(test_labels)}")
        
        if len(test_labels) == 0:
            print("  Test labels are empty. Removing 'test' directory to clean up...")
            shutil.rmtree(os.path.join(dataset_dir, 'images', 'test'))
            if os.path.exists(test_lbl_dir):
                shutil.rmtree(os.path.join(dataset_dir, 'labels', 'test'))
            print("  ✅ 'test' folder removed. Dataset is now purely Train/Val.")
        else:
            print("  ⚠️ Warning: Test folder still has labels. Not deleting automatically.")

if __name__ == "__main__":
    verify_and_clean()
