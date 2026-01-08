#!/usr/bin/env python3
"""Find images with largest boundary-F1 improvement (AURASeg - UPerNet)
and create a combined figure showing the top-K cases for the paper.

Usage:
  python benchmark_models/top_boundary_diff.py --csv runs/plots/qual_grid_metrics.csv --per_image_dir runs/plots/qual_grid_per_image --top_k 6 --out runs/plots/top_boundary_f1_diff.png
"""
import argparse
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def read_metrics(csv_path):
    rows = []
    with open(csv_path, newline='') as cf:
        reader = csv.DictReader(cf)
        for r in reader:
            # convert fields to float
            try:
                r['upernet_f1'] = float(r.get('upernet_f1', 0.0))
                r['auraseg_f1'] = float(r.get('auraseg_f1', 0.0))
            except Exception:
                r['upernet_f1'] = 0.0
                r['auraseg_f1'] = 0.0
            r['diff'] = r['auraseg_f1'] - r['upernet_f1']
            rows.append(r)
    return rows


def make_combined_figure(rows, per_image_dir: Path, out_path: Path, top_k: int = 6, annotate=True):
    top = rows[:top_k]
    imgs = []
    for r in top:
        fname = r['filename']
        p = per_image_dir / fname
        if not p.exists():
            # try .png/.jpg
            p_png = per_image_dir / (Path(fname).stem + '.png')
            if p_png.exists():
                p = p_png
            else:
                print(f"Warning: per-image file not found: {p}")
                # create blank placeholder
                imgs.append(Image.new('RGB', (1400, 350), (240,240,240)))
                continue
        imgs.append(Image.open(p).convert('RGB'))

    # compute max width
    widths = [im.width for im in imgs]
    heights = [im.height for im in imgs]
    maxw = max(widths)
    total_h = sum(heights)

    # create canvas
    canvas = Image.new('RGB', (maxw, total_h + 120), (255,255,255))
    y = 0
    for i, im in enumerate(imgs):
        # center horizontally
        x = (maxw - im.width)//2
        canvas.paste(im, (x, y))
        # annotate left with diff and F1s
        if annotate:
            r = top[i]
            diff = r['diff']
            ure = r['upernet_f1']
            are = r['auraseg_f1']
            txt = f"{Path(r['filename']).name}  ΔF1={diff:.3f}  AURAF1={are:.3f} UPerF1={ure:.3f}"
            # draw text using PIL ImageDraw
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(canvas)
            try:
                font = ImageFont.truetype("arial.ttf", 18)
            except Exception:
                font = ImageFont.load_default()
            draw.text((10, y+6), txt, fill=(0,0,0), font=font)
        y += im.height

    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, dpi=(300,300))
    print(f"Saved combined figure: {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True)
    parser.add_argument('--per_image_dir', type=str, required=True)
    parser.add_argument('--top_k', type=int, default=6)
    parser.add_argument('--out', type=str, default='runs/plots/top_boundary_f1_diff.png')
    args = parser.parse_args()

    csv_path = Path(args.csv)
    per_dir = Path(args.per_image_dir)
    out_path = Path(args.out)

    rows = read_metrics(csv_path)
    # sort by diff descending (AURASEG - UPERNET)
    rows_sorted = sorted(rows, key=lambda r: r['diff'], reverse=True)

    print('Top images by ΔF1 (AURASeg - UPerNet):')
    for r in rows_sorted[:args.top_k]:
        print(f"{r['filename']}: ΔF1={r['diff']:.4f} (A={r['auraseg_f1']:.4f}, U={r['upernet_f1']:.4f})")

    make_combined_figure(rows_sorted, per_dir, out_path, top_k=args.top_k)
