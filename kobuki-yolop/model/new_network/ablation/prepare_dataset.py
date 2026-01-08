"""
Dataset Preparation Script
==========================
Combines GMRPD and Gazebo datasets into a common structure for ablation study.

Structure:
    CommonDataset/
        images/
            train/
            val/
            test/
        labels/
            train/
            val/
            test/
"""

import os
import shutil
import random
import glob
from pathlib import Path
from tqdm import tqdm

def prepare_dataset():
    # Configuration
    root_dir = r"C:\Users\naren\Documents\AURASeg"
    
    # Source paths
    gmrpd_images = os.path.join(root_dir, r"GMRPD\GMRPD_modified\images")
    gmrpd_labels = os.path.join(root_dir, r"GMRPD\GMRPD_modified\labels")
    
    gazebo_images = os.path.join(root_dir, r"GAZEBO\dataset\images")
    gazebo_labels = os.path.join(root_dir, r"GAZEBO\dataset\da_seg_annotations")
    
    # Destination path
    dest_dir = os.path.join(root_dir, "CommonDataset")
    
    # Split ratios for Gazebo (GMRPD is already split)
    gazebo_split = {'train': 0.8, 'val': 0.1, 'test': 0.1}
    
    print(f"Preparing dataset at: {dest_dir}")
    
    # Create directories
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(dest_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(dest_dir, 'labels', split), exist_ok=True)
        
    # =========================================================================
    # Process GMRPD (Already split)
    # =========================================================================
    print("\nProcessing GMRPD dataset...")
    
    # Train
    copy_folder(
        os.path.join(gmrpd_images, 'train'), 
        os.path.join(dest_dir, 'images', 'train'),
        "GMRPD Train Images"
    )
    copy_folder(
        os.path.join(gmrpd_labels, 'train'), 
        os.path.join(dest_dir, 'labels', 'train'),
        "GMRPD Train Labels"
    )
    
    # Val
    copy_folder(
        os.path.join(gmrpd_images, 'val'), 
        os.path.join(dest_dir, 'images', 'val'),
        "GMRPD Val Images"
    )
    copy_folder(
        os.path.join(gmrpd_labels, 'val'), 
        os.path.join(dest_dir, 'labels', 'val'),
        "GMRPD Val Labels"
    )
    
    # Test (Images only, labels might be missing)
    copy_folder(
        os.path.join(gmrpd_images, 'test'), 
        os.path.join(dest_dir, 'images', 'test'),
        "GMRPD Test Images"
    )
    # Check if test labels exist
    if os.path.exists(os.path.join(gmrpd_labels, 'test')):
        copy_folder(
            os.path.join(gmrpd_labels, 'test'), 
            os.path.join(dest_dir, 'labels', 'test'),
            "GMRPD Test Labels"
        )
    
    # =========================================================================
    # Process Gazebo (Need to split)
    # =========================================================================
    print("\nProcessing Gazebo dataset...")
    
    # Get all images
    gazebo_img_files = glob.glob(os.path.join(gazebo_images, "*.jpg")) + \
                       glob.glob(os.path.join(gazebo_images, "*.png"))
    
    random.seed(42)
    random.shuffle(gazebo_img_files)
    
    total_gazebo = len(gazebo_img_files)
    train_count = int(total_gazebo * gazebo_split['train'])
    val_count = int(total_gazebo * gazebo_split['val'])
    # Rest for test
    
    splits = {
        'train': gazebo_img_files[:train_count],
        'val': gazebo_img_files[train_count:train_count+val_count],
        'test': gazebo_img_files[train_count+val_count:]
    }
    
    print(f"Gazebo Split: {len(splits['train'])} train, {len(splits['val'])} val, {len(splits['test'])} test")
    
    for split, files in splits.items():
        print(f"Copying Gazebo {split} set...")
        for img_path in tqdm(files):
            # Copy image
            shutil.copy2(img_path, os.path.join(dest_dir, 'images', split))
            
            # Find and copy label
            basename = os.path.splitext(os.path.basename(img_path))[0]
            # Assuming label is .png
            label_path = os.path.join(gazebo_labels, basename + ".png")
            
            if os.path.exists(label_path):
                shutil.copy2(label_path, os.path.join(dest_dir, 'labels', split))
            else:
                print(f"Warning: Label not found for {basename}")

    print("\nDataset preparation complete!")
    
    # Print final stats
    for split in ['train', 'val', 'test']:
        n_img = len(os.listdir(os.path.join(dest_dir, 'images', split)))
        n_lbl = len(os.listdir(os.path.join(dest_dir, 'labels', split)))
        print(f"{split.upper()}: {n_img} images, {n_lbl} labels")

def copy_folder(src, dst, desc):
    if not os.path.exists(src):
        print(f"Source not found: {src}")
        return
        
    files = os.listdir(src)
    print(f"Copying {desc} ({len(files)} files)...")
    for f in tqdm(files):
        src_file = os.path.join(src, f)
        dst_file = os.path.join(dst, f)
        if os.path.isfile(src_file):
            shutil.copy2(src_file, dst_file)

if __name__ == "__main__":
    prepare_dataset()
