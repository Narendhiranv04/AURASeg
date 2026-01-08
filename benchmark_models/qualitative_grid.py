#!/usr/bin/env python3
"""
Create a paper-style qualitative comparison grid and batch exporter.

Supports two modes:
 - Per-image single-row grid for an input list of filenames (--files)
 - Process the entire validation set (--all) and export one image per sample

For each prediction (UPerNet and AURASeg) the script computes a boundary F1
score against GT using a tolerance radius (pixels) and writes per-image metrics
to a CSV summary.

Output: per-image PNGs and a CSV with boundary precision/recall/f1 for each model.

Usage (example):
    C:/.../.venv/Scripts/python.exe benchmark_models/qualitative_grid.py \
        --all \
        --rgb_dir CommonDataset/images/val \
        --gt_dir CommonDataset/labels/val \
        --upernet_dir visualization-val/upernet/pred_masks \
        --auraseg_dir visualization-val/auraseg_v4_r50/pred_masks \
        --out_dir runs/plots/qual_grid_per_image \
        --csv runs/plots/qual_grid_metrics.csv

The script uses OpenCV + NumPy + Matplotlib and writes CSV with standard library.
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional


def load_and_resize(path: Path, size=(640, 360), is_mask: bool=False):
    """Load image or mask and resize to target size.
    For masks, returns binary uint8 0/1.
    """
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
        # Binarize masks robustly
        _, img = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
        img = img.astype(np.uint8)
    return img


def compute_boundary_error(gt: np.ndarray, pred: np.ndarray, radius: int = 3) -> np.ndarray:
    """Compute thin boundary error mask E_final as described.
    Inputs: gt and pred are binary uint8 (0/1) masks of same shape.
    Returns: binary uint8 mask with error pixels (0/1).
    """
    if gt is None or pred is None:
        return np.zeros_like(gt if gt is not None else pred, dtype=np.uint8)

    # Ensure binary
    gt_b = (gt > 0).astype(np.uint8)
    pr_b = (pred > 0).astype(np.uint8)

    # Morphological gradient to get boundaries (1-pixel wide in many cases)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    B_gt = cv2.morphologyEx(gt_b, cv2.MORPH_GRADIENT, kernel)
    B_pr = cv2.morphologyEx(pr_b, cv2.MORPH_GRADIENT, kernel)

    # XOR of boundaries
    E = cv2.bitwise_xor(B_gt, B_pr)

    # Dilate GT boundary to create narrow band
    band_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    band = cv2.dilate(B_gt, band_kernel)

    # Restrict error to band
    E_final = cv2.bitwise_and(E, band)

    # Optionally thin to 1-2px using distance transform threshold
    # but keep as-is -- E_final usually narrow.
    return (E_final > 0).astype(np.uint8)


def boundary_map(mask: np.ndarray) -> np.ndarray:
    """Return a binary boundary map (1-px-ish) from binary mask."""
    if mask is None:
        return None
    b = (mask > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    B = cv2.morphologyEx(b, cv2.MORPH_GRADIENT, kernel)
    return (B > 0).astype(np.uint8)


def boundary_f1(gt: np.ndarray, pred: np.ndarray, tol: int = 3):
    """Compute boundary precision, recall, f1 using distance tolerance in pixels.

    Uses morphological gradient for boundaries and distance transforms.
    Returns (precision, recall, f1).
    """
    if gt is None or pred is None:
        return 0.0, 0.0, 0.0

    B_gt = boundary_map(gt)
    B_pr = boundary_map(pred)

    if B_gt.sum() == 0 and B_pr.sum() == 0:
        return 1.0, 1.0, 1.0
    if B_pr.sum() == 0:
        return 0.0, 0.0, 0.0

    # distance from every pixel to nearest GT boundary
    # compute distance transform on inverted B_gt
    inv_gt = (B_gt == 0).astype(np.uint8)
    dist_gt = cv2.distanceTransform(inv_gt, cv2.DIST_L2, 3)

    # predicted boundary pixels that lie within tol of GT boundary are true positives (for precision)
    pred_coords = (B_pr > 0)
    matched_pred = (dist_gt[pred_coords] <= tol).sum() if pred_coords.any() else 0
    precision = matched_pred / float(B_pr.sum()) if B_pr.sum() > 0 else 0.0

    # similarly for recall: distance transform from predicted boundary
    inv_pr = (B_pr == 0).astype(np.uint8)
    dist_pr = cv2.distanceTransform(inv_pr, cv2.DIST_L2, 3)
    gt_coords = (B_gt > 0)
    matched_gt = (dist_pr[gt_coords] <= tol).sum() if gt_coords.any() else 0
    recall = matched_gt / float(B_gt.sum()) if B_gt.sum() > 0 else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def render_panel_mask(mask: np.ndarray, error_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Render binary mask to RGB panel: drivable=white, background=black.
    If error_mask provided, overlay red pixels where error_mask==1.
    Returns uint8 RGB image.
    """
    h, w = mask.shape
    panel = np.zeros((h, w, 3), dtype=np.uint8)
    # drivable -> white
    panel[mask > 0] = [255, 255, 255]
    # background remains black
    if error_mask is not None:
        # Put red [255,0,0] where error true
        panel[error_mask > 0] = [255, 0, 0]
    return panel


def make_grid(file_list: List[str], out_path: Path,
              rgb_dir: Path, gt_dir: Path,
              upernet_dir: Path, auraseg_dir: Path,
              size=(640, 360), row_labels: Optional[List[str]] = None,
              columns_titles: Optional[List[str]] = None,
              max_columns: int = 4):
    """Create N x 4 grid and save to out_path.
    Columns: RGB | Mask | UPerNet-R50 | AURASeg V4-R50
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    n_rows = len(file_list)
    n_cols = 4
    panels = []

    # This function is now deprecated in favor of make_grid_per_image
    for fname in file_list:
        # candidate filenames: try exact fname or with .png/.jpg
        f = fname
        # Load rgb
        rgb_path = rgb_dir / f
        gt_path = gt_dir / f
        pred_u_path = upernet_dir / f
        pred_a_path = auraseg_dir / f

        # Load images and masks
        rgb = load_and_resize(rgb_path, size=size, is_mask=False)
        gt = load_and_resize(gt_path, size=size, is_mask=True)
        uper = load_and_resize(pred_u_path, size=size, is_mask=True)
        aura = load_and_resize(pred_a_path, size=size, is_mask=True)

        # If some files are missing, warn and create placeholders
        if rgb is None:
            print(f"Warning: RGB not found for {f} at {rgb_path}; inserting gray panel")
            rgb = np.full((size[1], size[0], 3), 200, dtype=np.uint8)
        else:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        if gt is None:
            print(f"Warning: GT mask not found for {f} at {gt_path}; using empty mask")
            gt = np.zeros((size[1], size[0]), dtype=np.uint8)
        if uper is None:
            print(f"Warning: UPerNet pred not found for {f} at {pred_u_path}; using empty mask")
            uper = np.zeros_like(gt)
        if aura is None:
            print(f"Warning: AURASeg pred not found for {f} at {pred_a_path}; using empty mask")
            aura = np.zeros_like(gt)

        # compute boundary errors for predictions (overlay only on preds)
        err_u = compute_boundary_error(gt, uper, radius=3)
        err_a = compute_boundary_error(gt, aura, radius=3)

        # Render panels
        mask_panel = render_panel_mask(gt, error_mask=None)
        u_panel = render_panel_mask(uper, error_mask=err_u)
        a_panel = render_panel_mask(aura, error_mask=err_a)

        panels.append([rgb, mask_panel, u_panel, a_panel])

    # Plot
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
    })

    # Create the figure
    fig_w = 12
    fig_h = max(3.0, n_rows * (size[1] / size[0]) * 2.0)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    col_titles = columns_titles or ["RGB", "Mask", "UPerNet-R50", "AURASeg V4-R50"]
    for col in range(n_cols):
        axes[0, col].set_title(col_titles[col], pad=14)

    for i in range(n_rows):
        for j in range(n_cols):
            ax = axes[i, j]
            img = panels[i][j]
            ax.imshow(img)
            ax.axis('off')
            # left-side row label
            if j == 0 and row_labels:
                ax.set_ylabel(row_labels[i], fontsize=15, rotation=0, labelpad=50, va='center')

            # thin gray frame
            for spine in ax.spines.values():
                spine.set_edgecolor('#888')
                spine.set_linewidth(0.6)

    # Legend (bottom)
    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor='white', edgecolor='k', label='Drivable (GT / Pred)'),
        Patch(facecolor='black', edgecolor='k', label='Background'),
        Patch(facecolor='red', edgecolor='red', label='Boundary error (Pred vs GT)')
    ]
    fig.legend(handles=legend_elems, loc='lower center', ncol=3, fontsize=13, frameon=False, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.subplots_adjust(hspace=0.09, wspace=0.03, bottom=0.08)
    fig.savefig(str(out_path), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")

def make_grid_per_image(fname: str, out_path: Path,
                        rgb_dir: Path, gt_dir: Path,
                        upernet_dir: Path, auraseg_dir: Path,
                        size=(640, 360), tol: int = 3):
    """Create a single-row 4-panel figure for one filename and compute boundary F1s.

    Returns a dict with metrics for UPerNet and AURASeg: precision, recall, f1.
    """
    f = fname
    rgb_path = rgb_dir / f
    gt_path = gt_dir / f
    pred_u_path = upernet_dir / f
    pred_a_path = auraseg_dir / f

    rgb = load_and_resize(rgb_path, size=size, is_mask=False)
    gt = load_and_resize(gt_path, size=size, is_mask=True)
    uper = load_and_resize(pred_u_path, size=size, is_mask=True)
    aura = load_and_resize(pred_a_path, size=size, is_mask=True)

    if rgb is None:
        print(f"Warning: RGB not found for {f} at {rgb_path}; inserting gray panel")
        rgb = np.full((size[1], size[0], 3), 200, dtype=np.uint8)
    else:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    if gt is None:
        print(f"Warning: GT mask not found for {f} at {gt_path}; using empty mask")
        gt = np.zeros((size[1], size[0]), dtype=np.uint8)
    if uper is None:
        print(f"Warning: UPerNet pred not found for {f} at {pred_u_path}; using empty mask")
        uper = np.zeros_like(gt)
    if aura is None:
        print(f"Warning: AURASeg pred not found for {f} at {pred_a_path}; using empty mask")
        aura = np.zeros_like(gt)

    # compute boundary error overlays
    err_u = compute_boundary_error(gt, uper, radius=3)
    err_a = compute_boundary_error(gt, aura, radius=3)

    # compute boundary f1 metrics
    p_u, r_u, f1_u = boundary_f1(gt, uper, tol=tol)
    p_a, r_a, f1_a = boundary_f1(gt, aura, tol=tol)

    # render panels
    mask_panel = render_panel_mask(gt, error_mask=None)
    u_panel = render_panel_mask(uper, error_mask=err_u)
    a_panel = render_panel_mask(aura, error_mask=err_a)

    # assemble figure (single-row)
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
    })
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    titles = ["RGB", "Mask", "UPerNet-R50", "AURASeg V4-R50"]
    imgs = [rgb, mask_panel, u_panel, a_panel]

    for i, ax in enumerate(axes):
        ax.imshow(imgs[i])
        ax.set_title(titles[i], pad=8)
        ax.axis('off')
        for spine in ax.spines.values():
            spine.set_edgecolor('#888')
            spine.set_linewidth(0.6)

    # overlay F1 text on the prediction panels
    txt_kwargs = dict(color='white', fontsize=10, weight='bold', va='top')
    axes[2].text(0.02, 0.02, f"F1={f1_u:.3f}\nP={p_u:.3f} R={r_u:.3f}", transform=axes[2].transAxes,
                 bbox=dict(facecolor='black', alpha=0.6, pad=4), **txt_kwargs)
    axes[3].text(0.02, 0.02, f"F1={f1_a:.3f}\nP={p_a:.3f} R={r_a:.3f}", transform=axes[3].transAxes,
                 bbox=dict(facecolor='black', alpha=0.6, pad=4), **txt_kwargs)

    # legend
    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor='white', edgecolor='k', label='Drivable (GT / Pred)'),
        Patch(facecolor='black', edgecolor='k', label='Background'),
        Patch(facecolor='red', edgecolor='red', label='Boundary error (Pred vs GT)')
    ]
    fig.legend(handles=legend_elems, loc='lower center', ncol=3, fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=300, bbox_inches='tight')
    plt.close(fig)

    metrics = {
        'filename': f,
        'upernet_precision': float(p_u),
        'upernet_recall': float(r_u),
        'upernet_f1': float(f1_u),
        'auraseg_precision': float(p_a),
        'auraseg_recall': float(r_a),
        'auraseg_f1': float(f1_a)
    }
    return metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', nargs='+', help='List of filenames (same stem across folders)')
    parser.add_argument('--all', action='store_true', help='Process all files in GT directory')
    parser.add_argument('--rgb_dir', type=str, default='CommonDataset/images/val')
    parser.add_argument('--gt_dir', type=str, default='CommonDataset/labels/val')
    parser.add_argument('--upernet_dir', type=str, default='visualization-val/upernet')
    parser.add_argument('--auraseg_dir', type=str, default='runs/auraseg_v4_resnet50')
    parser.add_argument('--out_dir', type=str, default='runs/plots/qual_grid_per_image')
    parser.add_argument('--csv', type=str, default='runs/plots/qual_grid_metrics.csv')
    parser.add_argument('--size', nargs=2, type=int, default=[640, 360], help='Panel width height')
    parser.add_argument('--tol', type=int, default=3, help='Boundary tolerance in pixels for F1')

    args = parser.parse_args()

    rgb_dir = Path(args.rgb_dir)
    gt_dir = Path(args.gt_dir)
    upernet_dir = Path(args.upernet_dir)
    auraseg_dir = Path(args.auraseg_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.csv)
    size = tuple(args.size)
    tol = int(args.tol)

    # build file list
    if args.all:
        # take all image files from GT dir (png/jpg)
        files = sorted([p.name for p in gt_dir.iterdir() if p.suffix.lower() in ('.png', '.jpg', '.jpeg')])
    elif args.files:
        files = args.files
    else:
        print('Error: specify --files or --all')
        sys.exit(1)

    # process all files, create per-image output and metrics CSV
    metrics_list = []
    total = len(files)
    for idx, f in enumerate(files, 1):
        out_path = out_dir / f
        metrics = make_grid_per_image(f, out_path, rgb_dir, gt_dir, upernet_dir, auraseg_dir, size=size, tol=tol)
        metrics_list.append(metrics)
        if idx % 50 == 0 or idx == total:
            print(f'Processed {idx}/{total}')

    # write CSV
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, 'w', newline='') as cf:
        writer = csv.DictWriter(cf, fieldnames=list(metrics_list[0].keys()))
        writer.writeheader()
        for m in metrics_list:
            writer.writerow(m)

    # print summary means
    import statistics
    def mean_or_nan(lst):
        return statistics.mean(lst) if lst else float('nan')

    u_f1 = [m['upernet_f1'] for m in metrics_list]
    a_f1 = [m['auraseg_f1'] for m in metrics_list]
    print('Summary:')
    print(f"UPerNet mean boundary F1: {mean_or_nan(u_f1):.4f}")
    print(f"AURASeg mean boundary F1: {mean_or_nan(a_f1):.4f}")
    print(f"Per-image figures saved to: {out_dir}")
    print(f"CSV metrics saved to: {csv_path}")
