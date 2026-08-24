import sys
import cv2
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from train_fbrnet_wacv import Config, seed_worker
from unified_dataset import UnifiedDrivableAreaDataset, Normalization
from torch.utils.data import DataLoader
from fbrnet_wacv_model import FBRNet_WACV
from wacv_metrics import compute_metrics, compute_boundary_metrics
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class CARLTestDataset(Dataset):
    def __init__(self, image_dir, label_dir, img_size=(384, 640), mean=None, std=None):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.img_size = img_size
        self.mean = mean or [0.485, 0.456, 0.406]
        self.std = std or [0.229, 0.224, 0.225]
        
        self.images = []
        self.labels = []
        
        for img_path in sorted(self.image_dir.glob("*.jpg")):
            label_name = img_path.name + "___fuse.png"
            label_path = self.label_dir / label_name
            if label_path.exists():
                self.images.append(img_path)
                self.labels.append(label_path)
        
        self.normalize = T.Normalize(mean=self.mean, std=self.std)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        image = image.resize((self.img_size[1], self.img_size[0]), Image.BILINEAR)
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        image = self.normalize(image)
        
        label = Image.open(self.labels[idx]).convert('L')
        label = label.resize((self.img_size[1], self.img_size[0]), Image.NEAREST)
        label = np.array(label)
        
        uniq = np.unique(label)
        if uniq.size == 0:
            label = np.zeros_like(label, dtype=np.int64)
        elif uniq.size == 1:
            label = (label > 0).astype(np.int64)
        elif uniq.size == 2:
            label = (label == uniq.max()).astype(np.int64)
        else:
            label = (label > 0).astype(np.int64)
            
        label = torch.from_numpy(label)
        return image, label, str(self.images[idx].name)

def old_compute_boundary_metrics(pred, target, kernel_size=5):
    kernel = np.ones((3, 3), np.uint8)
    pred_binary = (pred == 1).astype(np.uint8)
    target_binary = (target == 1).astype(np.uint8)
    
    pred_boundary = cv2.morphologyEx(pred_binary, cv2.MORPH_GRADIENT, kernel)
    target_boundary = cv2.morphologyEx(target_binary, cv2.MORPH_GRADIENT, kernel)
    
    pred_boundary = cv2.dilate(pred_boundary, kernel, iterations=2)
    target_boundary = cv2.dilate(target_boundary, kernel, iterations=2)
    
    pred_bnd = pred_boundary.flatten() > 0
    target_bnd = target_boundary.flatten() > 0
    
    intersection = np.sum(pred_bnd & target_bnd)
    union = np.sum(pred_bnd | target_bnd)
    boundary_iou = intersection / (union + 1e-8)
    
    tp = np.sum(pred_bnd & target_bnd)
    fp = np.sum(pred_bnd & ~target_bnd)
    fn = np.sum(~pred_bnd & target_bnd)
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return {
        'boundary_iou': boundary_iou,
        'boundary_precision': precision,
        'boundary_recall': recall,
        'boundary_f1': f1
    }

class DummyArgs:
    dataset = 'carl-d'
    seed = 42
    smoke_test = False
    output_root = '/media/naren/Windows/Users/naren/Documents/AURASeg/runs_fbrnet_wacv'
    resume_from = None

def get_bnd_stats(pred, target, k=2):
    kernel = np.ones((3, 3), np.uint8)
    pred_binary = (pred == 1).astype(np.uint8)
    target_binary = (target == 1).astype(np.uint8)
    
    pred_boundary = cv2.morphologyEx(pred_binary, cv2.MORPH_GRADIENT, kernel)
    target_boundary = cv2.morphologyEx(target_binary, cv2.MORPH_GRADIENT, kernel)
    
    pre_dil_pred = np.sum(pred_boundary > 0)
    pre_dil_target = np.sum(target_boundary > 0)
    
    pred_boundary_dil = cv2.dilate(pred_boundary, kernel, iterations=k)
    target_boundary_dil = cv2.dilate(target_boundary, kernel, iterations=k)
    
    pred_bnd = pred_boundary_dil > 0
    target_bnd = target_boundary_dil > 0
    
    post_dil_pred = np.sum(pred_bnd)
    post_dil_target = np.sum(target_bnd)
    intersection = np.sum(pred_bnd & target_bnd)
    union = np.sum(pred_bnd | target_bnd)
    
    return {
        'pre_pred': pre_dil_pred,
        'pre_target': pre_dil_target,
        'post_pred': post_dil_pred,
        'post_target': post_dil_target,
        'intersection': intersection,
        'union': union
    }

def main():
    args = DummyArgs()
    config = Config(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    normalization = Normalization(mean=tuple(config.MEAN), std=tuple(config.STD))
    
    # 7. Check Mask Pipeline Identity
    print("\n--- 7. CHECK MASK PIPELINE IDENTITY ---")
    unified_ds = UnifiedDrivableAreaDataset(
        dataset_root=config.DATA_ROOT, split='test', img_size=config.IMG_SIZE,
        transform=False, normalization=normalization, return_names=True
    )
    old_ds = CARLTestDataset(
        image_dir=config.DATA_ROOT / 'test' / 'test',
        label_dir=config.DATA_ROOT / 'test' / 'labels',
        img_size=config.IMG_SIZE, mean=config.MEAN, std=config.STD
    )
    
    num_identical = 0
    num_different = 0
    max_diff_pixels = 0
    
    # We must match them by name since old_ds sorts by .jpg and unified_ds by various
    old_ds_dict = {name: (img, mask) for img, mask, name in old_ds}
    
    for i in tqdm(range(len(unified_ds)), desc="Comparing Target Masks"):
        img_u, mask_u, name_u = unified_ds[i]
        _, mask_o = old_ds_dict[name_u]
        
        m_u = mask_u.numpy()
        m_o = mask_o.numpy()
        
        diff = np.sum(m_u != m_o)
        if diff == 0:
            num_identical += 1
        else:
            num_different += 1
            max_diff_pixels = max(max_diff_pixels, diff)
            
    print(f"Number identical: {num_identical}")
    print(f"Number different: {num_different}")
    print(f"Max differing pixels: {max_diff_pixels}")
    if num_identical == len(unified_ds):
        print("CARL TARGET PIPELINE IDENTITY: PASS")
    else:
        print("CARL TARGET PIPELINE IDENTITY: FAIL")

    
    # Load Model and Predictions
    print("\n--- LOADING MODEL & PREDICTIONS ---")
    model = FBRNet_WACV(num_classes=2, aux_mode='eval').to(device)
    best_path = Path('/media/naren/Windows/Users/naren/Documents/AURASeg/runs_fbrnet_wacv/fbrnet_carld_seed42/checkpoints/best.pth')
    if not best_path.exists():
        print("best.pth NOT FOUND!")
        return
        
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    test_loader = DataLoader(
        unified_ds, batch_size=config.VAL_BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY
    )
    
    all_preds, all_targets, all_names = [], [], []
    with torch.no_grad():
        for images, masks, names in tqdm(test_loader, desc="Running Inference"):
            images = images.to(device)
            with torch.amp.autocast('cuda', enabled=config.USE_AMP):
                feat_ffm = model(images)[0]
            preds = torch.argmax(feat_ffm, dim=1).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(masks.numpy())
            all_names.extend(names)
            
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # 1. Reproduce Saved Test Result
    print("\n--- 1. REPRODUCE SAVED TEST RESULT ---")
    seg_m = compute_metrics(all_preds, all_targets)
    bnd_m = compute_boundary_metrics(all_preds, all_targets, k=2)
    print("Reproduction metrics:")
    print(f"IoU drivable: {seg_m['iou_drivable']:.6f}")
    print(f"F1: {seg_m['f1']:.6f}")
    print(f"BIoU (wacv_metrics): {bnd_m['boundary_iou']:.6f}")
    print(f"BF1 (wacv_metrics): {bnd_m['boundary_f1']:.6f}")
    
    # 2. Cross-check with Old CARL Implementation
    print("\n--- 2. CROSS-CHECK OLD CARL BOUNDARY IMPLEMENTATION (k=2) ---")
    old_bnd_iou_sum, old_bnd_f1_sum = 0.0, 0.0
    old_bnd_prec_sum, old_bnd_recall_sum = 0.0, 0.0
    for i in range(len(all_preds)):
        res = old_compute_boundary_metrics(all_preds[i], all_targets[i], kernel_size=5) # it uses iterations=2 internally
        old_bnd_iou_sum += res['boundary_iou']
        old_bnd_f1_sum += res['boundary_f1']
        old_bnd_prec_sum += res['boundary_precision']
        old_bnd_recall_sum += res['boundary_recall']
        
    print(f"Old BIoU: {old_bnd_iou_sum / len(all_preds):.6f}")
    print(f"Old BPrecision: {old_bnd_prec_sum / len(all_preds):.6f}")
    print(f"Old BRecall: {old_bnd_recall_sum / len(all_preds):.6f}")
    print(f"Old BF1: {old_bnd_f1_sum / len(all_preds):.6f}")

    # 3. Target/Prediction Pixel Statistics
    print("\n--- 3. TARGET/PREDICTION PIXEL STATISTICS ---")
    total_pixels = all_targets.size
    gt_c0 = np.sum(all_targets == 0)
    gt_c1 = np.sum(all_targets == 1)
    pred_c0 = np.sum(all_preds == 0)
    pred_c1 = np.sum(all_preds == 1)
    
    tp = np.sum((all_preds == 1) & (all_targets == 1))
    fp = np.sum((all_preds == 1) & (all_targets == 0))
    fn = np.sum((all_preds == 0) & (all_targets == 1))
    tn = np.sum((all_preds == 0) & (all_targets == 0))
    
    print(f"Total pixels: {total_pixels}")
    print(f"GT class-0: {gt_c0} ({gt_c0/total_pixels*100:.2f}%)")
    print(f"GT class-1: {gt_c1} ({gt_c1/total_pixels*100:.2f}%)")
    print(f"Pred class-0: {pred_c0} ({pred_c0/total_pixels*100:.2f}%)")
    print(f"Pred class-1: {pred_c1} ({pred_c1/total_pixels*100:.2f}%)")
    print(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    iou_d = tp / (tp + fp + fn)
    acc = (tp + tn) / total_pixels
    print(f"Regenerated Precision: {precision:.6f}")
    print(f"Regenerated Recall: {recall:.6f}")
    print(f"Regenerated IoU_d: {iou_d:.6f}")
    print(f"Regenerated Acc: {acc:.6f}")
    
    # 4. Boundary Pixel Statistics
    print("\n--- 4. BOUNDARY PIXEL STATISTICS (k=2) ---")
    pre_pred_sum, pre_target_sum = 0, 0
    post_pred_sum, post_target_sum = 0, 0
    int_sum, uni_sum = 0, 0
    for i in range(len(all_preds)):
        st = get_bnd_stats(all_preds[i], all_targets[i], k=2)
        pre_pred_sum += st['pre_pred']
        pre_target_sum += st['pre_target']
        post_pred_sum += st['post_pred']
        post_target_sum += st['post_target']
        int_sum += st['intersection']
        uni_sum += st['union']
        
    print(f"Total GT bnd pixels (pre-dil): {pre_target_sum}")
    print(f"Total Pred bnd pixels (pre-dil): {pre_pred_sum}")
    print(f"Mean GT bnd pixels/img (pre-dil): {pre_target_sum/len(all_preds):.2f}")
    print(f"Mean Pred bnd pixels/img (pre-dil): {pre_pred_sum/len(all_preds):.2f}")
    print(f"GT bnd-band pixels (k=2): {post_target_sum}")
    print(f"Pred bnd-band pixels (k=2): {post_pred_sum}")
    print(f"Intersection pixels: {int_sum}")
    print(f"Union pixels: {uni_sum}")
    
    regen_biou = int_sum / uni_sum
    # This regen uses global pixels, wacv_metrics uses mean-of-IoUs or global? 
    # Usually metric is mean-of-images. But let's report global just in case, though bnd_m is per-image average.
    
    # 5. Per-Image Distribution
    print("\n--- 5. PER-IMAGE DISTRIBUTION ---")
    img_iou_d = []
    img_bf1 = []
    for i in range(len(all_preds)):
        p = all_preds[i]
        t = all_targets[i]
        _tp = np.sum((p == 1) & (t == 1))
        _fp = np.sum((p == 1) & (t == 0))
        _fn = np.sum((p == 0) & (t == 1))
        iou = _tp / (_tp + _fp + _fn + 1e-8)
        img_iou_d.append(iou)
        
        b_res = compute_boundary_metrics(p[None, ...], t[None, ...], k=2)
        img_bf1.append(b_res['boundary_f1'])
        
    img_iou_d = np.array(img_iou_d)
    img_bf1 = np.array(img_bf1)
    
    for name, arr in zip(["IoU drivable", "BF1"], [img_iou_d, img_bf1]):
        print(f"--- {name} ---")
        print(f"Mean: {np.mean(arr):.6f}")
        print(f"Median: {np.median(arr):.6f}")
        print(f"Min: {np.min(arr):.6f}")
        print(f"25th: {np.percentile(arr, 25):.6f}")
        print(f"75th: {np.percentile(arr, 75):.6f}")
        print(f"Max: {np.max(arr):.6f}")
        
    print(f"BF1 == 0 count: {np.sum(img_bf1 == 0)}")
    print(f"BF1 < 0.01 count: {np.sum(img_bf1 < 0.01)}")
    print(f"BF1 < 0.05 count: {np.sum(img_bf1 < 0.05)}")
    print(f"BF1 >= 0.05 count: {np.sum(img_bf1 >= 0.05)}")
    
    # 6. Visual Forensics
    print("\n--- 6. VISUAL FORENSICS ---")
    diag_dir = Path('/media/naren/Windows/Users/naren/Documents/AURASeg/runs_fbrnet_wacv/fbrnet_carld_seed42/diagnostics')
    diag_dir.mkdir(parents=True, exist_ok=True)
    
    sorted_idx = np.argsort(img_bf1)
    worst = sorted_idx[:4]
    best = sorted_idx[-4:]
    median_idx = len(img_bf1) // 2
    median = sorted_idx[median_idx-2:median_idx+2]
    
    sel_idx = list(worst) + list(median) + list(best)
    sel_labels = ['worst']*4 + ['median']*4 + ['best']*4
    
    kernel = np.ones((3,3), np.uint8)
    for idx, lbl in zip(sel_idx, sel_labels):
        img_path = unified_ds.images[idx]
        image_rgb = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        image_rgb = cv2.resize(image_rgb, (640, 384))
        
        p = all_preds[idx].astype(np.uint8)
        t = all_targets[idx].astype(np.uint8)
        
        p_bnd = cv2.morphologyEx(p, cv2.MORPH_GRADIENT, kernel)
        t_bnd = cv2.morphologyEx(t, cv2.MORPH_GRADIENT, kernel)
        
        # Overlay contours
        gt_overlay = image_rgb.copy()
        gt_overlay[t_bnd > 0] = [0, 255, 0] # Green for GT
        
        pred_overlay = image_rgb.copy()
        pred_overlay[p_bnd > 0] = [255, 0, 0] # Red for Pred
        
        both_overlay = image_rgb.copy()
        both_overlay[t_bnd > 0] = [0, 255, 0]
        both_overlay[p_bnd > 0] = [255, 0, 0]
        both_overlay[(t_bnd > 0) & (p_bnd > 0)] = [255, 255, 0] # Yellow for intersect
        
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        axs[0,0].imshow(image_rgb); axs[0,0].set_title('RGB')
        axs[0,1].imshow(t, cmap='gray'); axs[0,1].set_title('GT Mask')
        axs[0,2].imshow(p, cmap='gray'); axs[0,2].set_title('Pred Mask')
        
        axs[1,0].imshow(gt_overlay); axs[1,0].set_title('GT Contour (Green)')
        axs[1,1].imshow(pred_overlay); axs[1,1].set_title('Pred Contour (Red)')
        axs[1,2].imshow(both_overlay); axs[1,2].set_title(f'Both (BF1: {img_bf1[idx]:.4f})')
        
        for ax in axs.flatten():
            ax.axis('off')
            
        plt.tight_layout()
        save_path = diag_dir / f"{lbl}_{all_names[idx]}.png"
        plt.savefig(save_path, dpi=150)
        plt.close()
        
    print(f"Saved 12 diagnostics to {diag_dir}")
    
    # 8. Optional Diagnostic Tolerance
    print("\n--- 8. OPTIONAL DIAGNOSTIC TOLERANCE ---")
    for k_val in [1, 2, 3, 5, 10]:
        res = compute_boundary_metrics(all_preds, all_targets, k=k_val)
        print(f"k={k_val:2d} -> BIoU: {res['boundary_iou']:.6f} | BF1: {res['boundary_f1']:.6f}")

if __name__ == '__main__':
    main()
