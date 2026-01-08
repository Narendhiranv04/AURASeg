#!/usr/bin/env python3
"""
Create a publication-ready combined figure for a small list of scenes.

This script loads RGB, GT, UPerNet and AURASeg predictions, computes
boundary errors and boundary P/R/F1, and renders a 3-row x 4-column
figure suitable for paper with nice metric boxes.

Usage (example):
  C:/.../.venv/Scripts/python.exe benchmark_models/final_qualitative_figure.py \
    --files scene21_0129.png scene11_0142.png scene23_0021.png \
    --rgb_dir CommonDataset/images/val \
    --gt_dir CommonDataset/labels/val \
    --upernet_dir visualization-val/upernet/pred_masks \
    --auraseg_dir visualization-val/auraseg_v4_r50/pred_masks \
    --out runs/plots/final_qualitative_3.png
"""
import argparse
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple


def load_and_resize(path: Path, size: Tuple[int,int]=(640,360), is_mask: bool=False):
    if not path.exists():
        return None
    if is_mask:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.resize(img, (size[0], size[1]), interpolation=cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA)
    if is_mask:
        _, img = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
        return img.astype(np.uint8)
    return img


def boundary_map(mask: np.ndarray) -> np.ndarray:
    if mask is None:
        return None
    b = (mask > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    B = cv2.morphologyEx(b, cv2.MORPH_GRADIENT, kernel)
    return (B>0).astype(np.uint8)


def compute_boundary_error(gt: np.ndarray, pred: np.ndarray, radius: int=3) -> np.ndarray:
    gt_b = (gt>0).astype(np.uint8)
    pr_b = (pred>0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    B_gt = cv2.morphologyEx(gt_b, cv2.MORPH_GRADIENT, kernel)
    B_pr = cv2.morphologyEx(pr_b, cv2.MORPH_GRADIENT, kernel)
    E = cv2.bitwise_xor(B_gt, B_pr)
    band = cv2.dilate(B_gt, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1)))
    E_final = cv2.bitwise_and(E, band)
    return (E_final>0).astype(np.uint8)


def boundary_f1(gt: np.ndarray, pred: np.ndarray, tol: int=3) -> Tuple[float,float,float]:
    B_gt = boundary_map(gt)
    B_pr = boundary_map(pred)
    if B_gt.sum()==0 and B_pr.sum()==0:
        return 1.0,1.0,1.0
    if B_pr.sum()==0:
        return 0.0,0.0,0.0
    inv_gt = (B_gt==0).astype(np.uint8)
    dist_gt = cv2.distanceTransform(inv_gt, cv2.DIST_L2, 3)
    pred_coords = (B_pr>0)
    matched_pred = (dist_gt[pred_coords] <= tol).sum() if pred_coords.any() else 0
    precision = matched_pred / float(B_pr.sum()) if B_pr.sum()>0 else 0.0
    inv_pr = (B_pr==0).astype(np.uint8)
    dist_pr = cv2.distanceTransform(inv_pr, cv2.DIST_L2, 3)
    gt_coords = (B_gt>0)
    matched_gt = (dist_pr[gt_coords] <= tol).sum() if gt_coords.any() else 0
    recall = matched_gt / float(B_gt.sum()) if B_gt.sum()>0 else 0.0
    if precision+recall==0:
        f1=0.0
    else:
        f1 = 2*precision*recall/(precision+recall)
    return precision, recall, f1


def render_mask_panel(mask: np.ndarray, error_mask: np.ndarray=None) -> np.ndarray:
    h,w = mask.shape
    panel = np.zeros((h,w,3), dtype=np.uint8)
    panel[mask>0] = [255,255,255]
    if error_mask is not None:
        panel[error_mask>0] = [255,0,0]
    return panel


def make_final_figure(files: List[str], out_path: Path,
                      rgb_dir: Path, gt_dir: Path, uper_dir: Path, aura_dir: Path,
                      size=(640,360), tol:int=3):
    n = len(files)
    cols = 4
    plt.rcParams.update({'font.size':14,'axes.titlesize':16})
    fig, axes = plt.subplots(n, cols, figsize=(14, 3.5*n))
    if n==1:
        axes = axes[np.newaxis,:]

    col_titles = ['RGB','Mask','UPerNet-R50','AURASeg V4-R50']
    for j in range(cols):
        axes[0,j].set_title(col_titles[j], pad=12)

    metrics = []
    for i, fname in enumerate(files):
        rgb = load_and_resize(rgb_dir/fname, size=size, is_mask=False)
        gt = load_and_resize(gt_dir/fname, size=size, is_mask=True)
        uper = load_and_resize(uper_dir/fname, size=size, is_mask=True)
        aura = load_and_resize(aura_dir/fname, size=size, is_mask=True)
        if rgb is None:
            rgb = np.full((size[1],size[0],3),200,dtype=np.uint8)
        else:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        if gt is None:
            gt = np.zeros((size[1],size[0]),dtype=np.uint8)
        if uper is None:
            uper = np.zeros_like(gt)
        if aura is None:
            aura = np.zeros_like(gt)

        err_u = compute_boundary_error(gt, uper, radius=3)
        err_a = compute_boundary_error(gt, aura, radius=3)

        p_u, r_u, f1_u = boundary_f1(gt, uper, tol=tol)
        p_a, r_a, f1_a = boundary_f1(gt, aura, tol=tol)
        metrics.append({'file':fname,'u_p':p_u,'u_r':r_u,'u_f1':f1_u,'a_p':p_a,'a_r':r_a,'a_f1':f1_a})

        panels = [rgb, render_mask_panel(gt,None), render_mask_panel(uper,err_u), render_mask_panel(aura,err_a)]
        for j in range(cols):
            ax = axes[i,j]
            ax.imshow(panels[j])
            ax.axis('off')
            for spine in ax.spines.values():
                spine.set_edgecolor('#888')
                spine.set_linewidth(0.6)

        # draw metric boxes on columns 2 and 3 (index 2,3)
        def draw_metric_box(ax, P,R,F, title):
            txt = f"P: {P:.3f}\nR: {R:.3f}\nF1: {F:.3f}"
            ax.text(0.02,0.02,txt,transform=ax.transAxes, fontsize=11, color='white',
                    bbox=dict(facecolor='black', alpha=0.6, pad=6))
            # also small title
            ax.text(0.02,0.30,title,transform=ax.transAxes, fontsize=10, color='white',
                    bbox=dict(facecolor='black', alpha=0.6, pad=3))

        # for UPerNet
        draw_metric_box(axes[i,2], p_u, r_u, f1_u, 'UPerNet')
        # for AURASeg, make box greenish if better
        box_color = (0.0,0.6,0.0,0.7) if f1_a>f1_u else (0.4,0.4,0.4,0.6)
        axes[i,3].text(0.02,0.02,f"P: {p_a:.3f}\nR: {r_a:.3f}\nF1: {f1_a:.3f}",transform=axes[i,3].transAxes, fontsize=11, color='white',
                       bbox=dict(facecolor=box_color, alpha=0.85, pad=6))
        axes[i,3].text(0.02,0.30,'AURASeg',transform=axes[i,3].transAxes, fontsize=10, color='white',
                       bbox=dict(facecolor=box_color, alpha=0.85, pad=3))

        # delta F1 annotation top-right of row
        delta = f1_a - f1_u
        axes[i,3].text(0.98,0.02, f"ΔF1={delta:+.3f}", transform=axes[i,3].transAxes,
                       fontsize=12, color='white', ha='right',
                       bbox=dict(facecolor='navy' if delta>0 else 'darkred', alpha=0.8, pad=4))

    # overall layout and legend
    from matplotlib.patches import Patch
    legend_elems = [Patch(facecolor='white', edgecolor='k', label='Drivable (GT / Pred)'),
                    Patch(facecolor='black', edgecolor='k', label='Background'),
                    Patch(facecolor='red', edgecolor='red', label='Boundary error (Pred vs GT)')]
    fig.legend(handles=legend_elems, loc='lower center', ncol=3, fontsize=12, frameon=False, bbox_to_anchor=(0.5,-0.02))
    plt.tight_layout(rect=[0,0.03,1,0.98])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved final figure: {out_path}")


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', nargs='+', required=True)
    parser.add_argument('--rgb_dir', type=str, default='CommonDataset/images/val')
    parser.add_argument('--gt_dir', type=str, default='CommonDataset/labels/val')
    parser.add_argument('--upernet_dir', type=str, default='visualization-val/upernet/pred_masks')
    parser.add_argument('--auraseg_dir', type=str, default='visualization-val/auraseg_v4_r50/pred_masks')
    parser.add_argument('--out', type=str, default='runs/plots/final_qualitative_3.png')
    parser.add_argument('--size', nargs=2, type=int, default=[640,360])
    parser.add_argument('--tol', type=int, default=3)
    args = parser.parse_args()

    files = args.files
    rgb_dir = Path(args.rgb_dir)
    gt_dir = Path(args.gt_dir)
    uper_dir = Path(args.upernet_dir)
    aura_dir = Path(args.auraseg_dir)
    out = Path(args.out)
    size = tuple(args.size)

    make_final_figure(files, out, rgb_dir, gt_dir, uper_dir, aura_dir, size=size, tol=args.tol)
