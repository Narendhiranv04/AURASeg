"""
Cross-Dataset Evaluation (Table IV helper)
=========================================

Evaluates MIX-trained checkpoints on CARL-D (or any other dataset root) and
computes the absolute drop (Δ) from the in-domain score.

Examples (PowerShell):
  # Train MIX (CommonDataset) -> Test CARL-D
  .\\.venv\\Scripts\\python.exe benchmark_models\\eval_cross_dataset.py `
    --runs-dir runs `
    --train-root CommonDataset --train-split val `
    --test-root carl-dataset --test-split test `
    --output runs\\table_iv_mix_to_carl.csv

  # Train CARL-D -> Test MIX (after you train and save checkpoints under runs_carl/)
  .\\.venv\\Scripts\\python.exe benchmark_models\\eval_cross_dataset.py `
    --runs-dir runs_carl `
    --train-root carl-dataset --train-split val `
    --test-root CommonDataset --test-split val `
    --output runs\\table_iv_carl_to_mix.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from unified_dataset import Normalization, UnifiedDrivableAreaDataset
from model_factory import get_benchmark_model


@dataclass
class StreamCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    boundary_iou_sum: float = 0.0
    boundary_precision_sum: float = 0.0
    boundary_recall_sum: float = 0.0
    boundary_f1_sum: float = 0.0
    boundary_count: int = 0

    def update_batch(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        # preds/targets: (B, H, W) with values {0,1}
        preds_fg = preds == 1
        targets_fg = targets == 1

        tp = torch.sum(preds_fg & targets_fg).item()
        fp = torch.sum(preds_fg & (~targets_fg)).item()
        fn = torch.sum((~preds_fg) & targets_fg).item()
        tn = torch.sum((~preds_fg) & (~targets_fg)).item()

        self.tp += int(tp)
        self.fp += int(fp)
        self.fn += int(fn)
        self.tn += int(tn)

    def update_boundary_image(self, pred_bin: np.ndarray, target_bin: np.ndarray, dilation: int) -> None:
        import cv2

        kernel = np.ones((3, 3), np.uint8)
        pred_boundary = cv2.morphologyEx(pred_bin, cv2.MORPH_GRADIENT, kernel)
        target_boundary = cv2.morphologyEx(target_bin, cv2.MORPH_GRADIENT, kernel)

        pred_boundary = cv2.dilate(pred_boundary, kernel, iterations=dilation)
        target_boundary = cv2.dilate(target_boundary, kernel, iterations=dilation)

        tp = int(np.sum((pred_boundary > 0) & (target_boundary > 0)))
        fp = int(np.sum((pred_boundary > 0) & (target_boundary == 0)))
        fn = int(np.sum((pred_boundary == 0) & (target_boundary > 0)))

        boundary_iou = tp / (tp + fp + fn + 1e-6)
        boundary_precision = tp / (tp + fp + 1e-6)
        boundary_recall = tp / (tp + fn + 1e-6)
        boundary_f1 = 2 * boundary_precision * boundary_recall / (boundary_precision + boundary_recall + 1e-6)

        self.boundary_iou_sum += float(boundary_iou)
        self.boundary_precision_sum += float(boundary_precision)
        self.boundary_recall_sum += float(boundary_recall)
        self.boundary_f1_sum += float(boundary_f1)
        self.boundary_count += 1

    def finalize(self) -> Dict[str, float]:
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        total = tp + fp + fn + tn

        iou_drivable = tp / (tp + fp + fn + 1e-6)
        iou_background = tn / (tn + fp + fn + 1e-6)
        miou = (iou_background + iou_drivable) / 2.0

        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        dice = (2 * tp) / (2 * tp + fp + fn + 1e-6)
        accuracy = (tp + tn) / (total + 1e-6)

        if self.boundary_count > 0:
            boundary_iou = self.boundary_iou_sum / self.boundary_count
            boundary_precision = self.boundary_precision_sum / self.boundary_count
            boundary_recall = self.boundary_recall_sum / self.boundary_count
            boundary_f1 = self.boundary_f1_sum / self.boundary_count
        else:
            boundary_iou = boundary_precision = boundary_recall = boundary_f1 = float("nan")

        return {
            "miou": float(miou),
            "iou_drivable": float(iou_drivable),
            "iou_background": float(iou_background),
            "dice": float(dice),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "accuracy": float(accuracy),
            "boundary_iou": float(boundary_iou),
            "boundary_precision": float(boundary_precision),
            "boundary_recall": float(boundary_recall),
            "boundary_f1": float(boundary_f1),
        }


MODEL_DISPLAY_NAMES = {
    "fcn": "FCN-R50",
    "pspnet": "PSPNet-R50",
    "deeplabv3plus": "DeepLabV3+",
    "upernet": "UPerNet-R50",
    "segformer": "SegFormer-B2",
    "mask2former": "FPN-MiTB3",
    "pidnet": "PIDNet-L",
    "auraseg_v4_r50": "AURASeg V4-R50",
}


def _find_checkpoint(runs_dir: Path, model_key: str) -> Path:
    if model_key == "auraseg_v4_r50":
        candidates = [
            runs_dir / "auraseg_v4_resnet50" / "checkpoints" / "best.pth",
            runs_dir / "auraseg_v4_resnet50" / "checkpoints" / "latest.pth",
        ]
    else:
        candidates = [
            runs_dir / f"benchmark_{model_key}" / "checkpoints" / "best.pth",
            runs_dir / f"benchmark_{model_key}" / "checkpoints" / "latest.pth",
        ]

    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Checkpoint not found for {model_key} under {runs_dir}")


def _load_model(model_key: str, checkpoint_path: Path, device: torch.device):
    if model_key == "auraseg_v4_r50":
        from auraseg_v4_resnet import AURASeg_V4_ResNet50

        model = AURASeg_V4_ResNet50(num_classes=2).to(device)
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model, {"name": MODEL_DISPLAY_NAMES[model_key], "output_key": "main"}

    model, info = get_benchmark_model(model_key, num_classes=2, pretrained=False)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, info


def _forward_logits(model_key: str, model, images: torch.Tensor, info: Dict) -> torch.Tensor:
    outputs = model(images) if model_key != "auraseg_v4_r50" else model(images, return_aux=False, return_boundary=False)

    if model_key == "auraseg_v4_r50":
        # AURASeg returns dict with 'main'
        return outputs["main"]

    output_key = info.get("output_key")
    if output_key and isinstance(outputs, dict):
        outputs = outputs[output_key]
    elif isinstance(outputs, dict) and "out" in outputs:
        outputs = outputs["out"]

    if isinstance(outputs, (list, tuple)):
        outputs = outputs[0]
    return outputs


def evaluate_checkpoint_on_dataset(
    model_key: str,
    runs_dir: Path,
    dataset_root: Path,
    split: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    img_size: Tuple[int, int],
    dilation: int,
    max_samples: Optional[int],
) -> Dict[str, float]:
    checkpoint_path = _find_checkpoint(runs_dir, model_key)
    model, info = _load_model(model_key, checkpoint_path, device)

    dataset = UnifiedDrivableAreaDataset(
        dataset_root=dataset_root,
        split=split,
        img_size=img_size,
        transform=False,
        normalization=Normalization(),
        return_names=False,
    )

    if max_samples is not None:
        dataset.images = dataset.images[: max_samples]  # type: ignore[attr-defined]

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    counts = StreamCounts()

    with torch.no_grad():
        for images, masks in tqdm(loader, desc=f"{MODEL_DISPLAY_NAMES.get(model_key, model_key)} [{split}]", leave=False):
            masks_np = masks.numpy().astype(np.uint8)

            images = images.to(device)
            masks = masks.to(device)

            logits = _forward_logits(model_key, model, images, info)
            preds = torch.argmax(logits, dim=1)

            counts.update_batch(preds, masks)

            preds_np = preds.detach().cpu().numpy().astype(np.uint8)

            # Boundary metrics (mean over images)
            for i in range(preds_np.shape[0]):
                counts.update_boundary_image(preds_np[i], masks_np[i], dilation=dilation)

    metrics = counts.finalize()

    # Cleanup to reduce VRAM
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    metrics["checkpoint"] = str(checkpoint_path)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-dataset evaluation helper for Table IV")

    parser.add_argument("--runs-dir", type=str, default="runs", help="Directory containing trained checkpoints")
    parser.add_argument("--models", type=str, nargs="+", default=list(MODEL_DISPLAY_NAMES.keys()))

    parser.add_argument("--train-root", type=str, default="CommonDataset", help="In-domain dataset root (trained on)")
    parser.add_argument("--train-split", type=str, default="val", help="Split to evaluate in-domain")
    parser.add_argument("--test-root", type=str, default="carl-dataset", help="Cross-dataset root (evaluate on)")
    parser.add_argument("--test-split", type=str, default="test", help="Split to evaluate cross-dataset")

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--img-h", type=int, default=384)
    parser.add_argument("--img-w", type=int, default=640)
    parser.add_argument("--boundary-dilation", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=None, help="Debug: limit number of images per split")

    parser.add_argument("--output", type=str, default="runs/table_iv_cross_dataset.csv")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    runs_dir = Path(args.runs_dir)
    train_root = Path(args.train_root)
    test_root = Path(args.test_root)
    img_size = (args.img_h, args.img_w)

    rows = []

    print(f"Device: {device}")
    print(f"Runs dir: {runs_dir.resolve()}")
    print(f"In-domain: {train_root} [{args.train_split}]")
    print(f"Cross-domain: {test_root} [{args.test_split}]")

    for model_key in args.models:
        if model_key not in MODEL_DISPLAY_NAMES:
            raise ValueError(f"Unknown model key '{model_key}'. Allowed: {sorted(MODEL_DISPLAY_NAMES.keys())}")

        print(f"\n=== {MODEL_DISPLAY_NAMES[model_key]} ===")

        in_metrics = evaluate_checkpoint_on_dataset(
            model_key=model_key,
            runs_dir=runs_dir,
            dataset_root=train_root,
            split=args.train_split,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            img_size=img_size,
            dilation=args.boundary_dilation,
            max_samples=args.max_samples,
        )

        out_metrics = evaluate_checkpoint_on_dataset(
            model_key=model_key,
            runs_dir=runs_dir,
            dataset_root=test_root,
            split=args.test_split,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            img_size=img_size,
            dilation=args.boundary_dilation,
            max_samples=args.max_samples,
        )

        row = {
            "model_key": model_key,
            "model": MODEL_DISPLAY_NAMES[model_key],
            "runs_dir": str(runs_dir),
            "train_root": str(train_root),
            "train_split": args.train_split,
            "test_root": str(test_root),
            "test_split": args.test_split,
            # In-domain (Table IV left side)
            "in_miou": in_metrics["miou"],
            "in_precision": in_metrics["precision"],
            "in_recall": in_metrics["recall"],
            "in_f1": in_metrics["f1"],
            "in_biou": in_metrics["boundary_iou"],
            "in_bfi": in_metrics["boundary_f1"],
            # Cross-dataset
            "out_miou": out_metrics["miou"],
            "out_biou": out_metrics["boundary_iou"],
            "out_f1": out_metrics["f1"],
            "out_bfi": out_metrics["boundary_f1"],
            # Absolute drop (delta)
            "delta_miou": in_metrics["miou"] - out_metrics["miou"],
            "delta_f1": in_metrics["f1"] - out_metrics["f1"],
            "delta_bfi": in_metrics["boundary_f1"] - out_metrics["boundary_f1"],
            # Convenience formatting for Table IV cells (score/drop)
            "miou_over_delta": f"{out_metrics['miou']:.4f}/{(in_metrics['miou'] - out_metrics['miou']):.4f}",
            "f1_over_delta": f"{out_metrics['f1']:.4f}/{(in_metrics['f1'] - out_metrics['f1']):.4f}",
            "bfi_over_delta": f"{out_metrics['boundary_f1']:.4f}/{(in_metrics['boundary_f1'] - out_metrics['boundary_f1']):.4f}",
            # Provenance
            "in_checkpoint": in_metrics["checkpoint"],
            "out_checkpoint": out_metrics["checkpoint"],
        }
        rows.append(row)

        print(f"  In-domain   mIoU={row['in_miou']:.4f} F1={row['in_f1']:.4f} BFI={row['in_bfi']:.4f}")
        print(f"  Cross       mIoU={row['out_miou']:.4f} F1={row['out_f1']:.4f} BFI={row['out_bfi']:.4f}")
        print(f"  Drop (delta) mIoU={row['delta_miou']:.4f} F1={row['delta_f1']:.4f} BFI={row['delta_bfi']:.4f}")

    df = pd.DataFrame(rows)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path.resolve()}")


if __name__ == "__main__":
    main()
