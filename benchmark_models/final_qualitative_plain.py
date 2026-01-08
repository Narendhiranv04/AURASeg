#!/usr/bin/env python3
"""
Generate a plain 3x4 qualitative figure (no metrics) for a small list of scenes.
"""
import argparse
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_and_resize(path: Path, size=(640,360), is_mask=False):
    if not path.exists():
        return None
    if is_mask:
        im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    else:
        im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None:
        return None
    im = cv2.resize(im, (size[0], size[1]), interpolation=cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA)
    if is_mask:
        _, im = cv2.threshold(im, 127, 1, cv2.THRESH_BINARY)
        return im.astype(np.uint8)
    return im


def render_mask(mask: np.ndarray):
    h,w = mask.shape
    panel = np.zeros((h,w,3), dtype=np.uint8)
    panel[mask>0] = [255,255,255]
    return panel


def make_plain_figure(files, out_path, rgb_dir, gt_dir, uper_dir, aura_dir, size=(640,360)):
    n = len(files)
    cols = 4
    plt.rcParams.update({'font.size':12})
    fig, axes = plt.subplots(n, cols, figsize=(14, 3.5*n))
    if n==1:
        axes = axes[np.newaxis,:]

    col_titles = ['RGB','Mask','UPerNet-R50','AURASeg V4-R50']
    for j in range(cols):
        axes[0,j].set_title(col_titles[j], pad=10)

    for i, fname in enumerate(files):
        rgb = load_and_resize(rgb_dir/fname, size=size, is_mask=False)
        gt = load_and_resize(gt_dir/fname, size=size, is_mask=True)
        uper = load_and_resize(uper_dir/fname, size=size, is_mask=True)
        aura = load_and_resize(aura_dir/fname, size=size, is_mask=True)

        if rgb is None:
            rgb = np.full((size[1],size[0],3), 200, dtype=np.uint8)
        else:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        if gt is None:
            gt = np.zeros((size[1],size[0]), dtype=np.uint8)
        if uper is None:
            uper = np.zeros_like(gt)
        if aura is None:
            aura = np.zeros_like(gt)

        panels = [rgb, render_mask(gt), render_mask(uper), render_mask(aura)]
        for j in range(cols):
            ax = axes[i,j]
            ax.imshow(panels[j])
            ax.axis('off')
            for spine in ax.spines.values():
                spine.set_edgecolor('#888')
                spine.set_linewidth(0.6)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved plain figure: {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', nargs='+', required=True)
    parser.add_argument('--rgb_dir', type=str, default='CommonDataset/images/val')
    parser.add_argument('--gt_dir', type=str, default='CommonDataset/labels/val')
    parser.add_argument('--uper_dir', type=str, default='visualization-val/upernet/pred_masks')
    parser.add_argument('--aura_dir', type=str, default='visualization-val/auraseg_v4_r50/pred_masks')
    parser.add_argument('--out', type=str, default='runs/plots/final_qualitative_3_plain.png')
    parser.add_argument('--size', nargs=2, type=int, default=[640,360])
    args = parser.parse_args()

    files = args.files
    rgb_dir = Path(args.rgb_dir)
    gt_dir = Path(args.gt_dir)
    uper_dir = Path(args.uper_dir)
    aura_dir = Path(args.aura_dir)
    out = Path(args.out)
    size = tuple(args.size)

    make_plain_figure(files, out, rgb_dir, gt_dir, uper_dir, aura_dir, size=size)
