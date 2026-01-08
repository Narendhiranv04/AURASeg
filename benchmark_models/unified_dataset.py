"""
Unified Drivable-Area Dataset Loader
===================================

Supports both dataset layouts used in this repo:

1) CommonDataset-style (MIX):
   <root>/images/<split>/*
   <root>/labels/<split>/*

2) CARL-D-style:
   <root>/<split>/images/*
   <root>/<split>/labels/*

Also supports CARL-D "test/test" quirk:
   <root>/test/test/*   (images)
   <root>/test/labels/* (labels)

Mask mapping:
  - CommonDataset: <image_stem>.png (or .jpg)
  - CARL-D: <image_filename>___fuse.png

Mask binarization is robust to arbitrary 2-value encodings (e.g., 21/109).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    _HAS_ALBUMENTATIONS = True
except Exception:  # pragma: no cover
    _HAS_ALBUMENTATIONS = False


def resolve_split_dirs(dataset_root: Path, split: str) -> Tuple[Path, Path]:
    """
    Resolve image/mask directories for a given dataset_root + split.

    Returns:
        (images_dir, labels_dir)
    """
    dataset_root = Path(dataset_root)

    # Layout 1: <root>/images/<split>, <root>/labels/<split>
    images_dir = dataset_root / "images" / split
    labels_dir = dataset_root / "labels" / split
    if images_dir.is_dir() and labels_dir.is_dir():
        return images_dir, labels_dir

    # Layout 2: <root>/<split>/images, <root>/<split>/labels
    images_dir = dataset_root / split / "images"
    labels_dir = dataset_root / split / "labels"
    if images_dir.is_dir() and labels_dir.is_dir():
        return images_dir, labels_dir

    # Layout 3 (CARL-D test quirk): <root>/<split>/<split>, <root>/<split>/labels
    images_dir = dataset_root / split / split
    labels_dir = dataset_root / split / "labels"
    if images_dir.is_dir() and labels_dir.is_dir():
        return images_dir, labels_dir

    raise FileNotFoundError(
        f"Could not resolve dataset dirs for dataset_root={dataset_root} split={split}. "
        "Expected one of: "
        "<root>/images/<split> + <root>/labels/<split>, "
        "<root>/<split>/images + <root>/<split>/labels, "
        "or <root>/<split>/<split> + <root>/<split>/labels."
    )


def _candidate_mask_paths(img_path: Path, labels_dir: Path) -> Iterable[Path]:
    # CommonDataset-style
    yield labels_dir / f"{img_path.stem}.png"
    yield labels_dir / f"{img_path.stem}.jpg"

    # CARL-D-style
    yield labels_dir / f"{img_path.name}___fuse.png"
    yield labels_dir / f"{img_path.name}__fuse.png"
    yield labels_dir / f"{img_path.stem}___fuse.png"

    # Fallbacks used in some datasets
    yield labels_dir / f"{img_path.stem}_mask.png"
    yield labels_dir / f"{img_path.stem}_label.png"


def find_mask_path(img_path: Path, labels_dir: Path) -> Path:
    """Find a corresponding mask path for an image path."""
    for cand in _candidate_mask_paths(img_path, labels_dir):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"No mask found for image: {img_path.name} in {labels_dir}")


def binarize_mask(mask: np.ndarray) -> np.ndarray:
    """
    Convert a grayscale mask to binary {0,1}.

    Strategy:
      - If exactly 2 unique values: map max -> 1, min -> 0
      - Else: map >0 -> 1
    """
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask[..., 0]

    uniq = np.unique(mask)
    if uniq.size == 0:
        return np.zeros_like(mask, dtype=np.uint8)
    if uniq.size == 1:
        return (mask > 0).astype(np.uint8)
    if uniq.size == 2:
        return (mask == uniq.max()).astype(np.uint8)
    return (mask > 0).astype(np.uint8)


@dataclass(frozen=True)
class Normalization:
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)


class UnifiedDrivableAreaDataset(Dataset):
    """
    PyTorch Dataset for binary drivable-area segmentation with robust layout/mask handling.
    """

    def __init__(
        self,
        dataset_root: Path,
        split: str,
        img_size: Tuple[int, int],
        transform: bool = True,
        normalization: Optional[Normalization] = None,
        return_names: bool = False,
    ):
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.img_size = img_size
        self.return_names = return_names
        self.normalization = normalization or Normalization()

        self.images_dir, self.labels_dir = resolve_split_dirs(self.dataset_root, split)

        self.images = sorted(
            list(self.images_dir.glob("*.jpg"))
            + list(self.images_dir.glob("*.jpeg"))
            + list(self.images_dir.glob("*.png"))
        )

        print(f"[{split.upper()}] {self.dataset_root} -> {len(self.images)} images")

        self._transform = self._build_transforms(split, transform)

    def _build_transforms(self, split: str, use_augment: bool):
        if not _HAS_ALBUMENTATIONS:
            return None

        mean = list(self.normalization.mean)
        std = list(self.normalization.std)

        if split == "train" and use_augment:
            return A.Compose(
                [
                    A.Resize(height=self.img_size[0], width=self.img_size[1]),
                    A.HorizontalFlip(p=0.5),
                    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.5),
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
                    A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
                    A.Normalize(mean=mean, std=std),
                    ToTensorV2(),
                ]
            )

        return A.Compose(
            [
                A.Resize(height=self.img_size[0], width=self.img_size[1]),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img_path = self.images[idx]

        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_path = find_mask_path(img_path, self.labels_dir)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Failed to read mask: {mask_path}")
        mask = binarize_mask(mask)

        if self._transform is not None:
            transformed = self._transform(image=image, mask=mask)
            image_t = transformed["image"].float()
            mask_t = transformed["mask"].long()
        else:  # pragma: no cover
            # Minimal fallback without albumentations.
            image = cv2.resize(image, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_NEAREST)

            image = image.astype(np.float32) / 255.0
            mean = np.array(self.normalization.mean, dtype=np.float32)
            std = np.array(self.normalization.std, dtype=np.float32)
            image = (image - mean) / std

            image_t = torch.from_numpy(np.transpose(image, (2, 0, 1))).float()
            mask_t = torch.from_numpy(mask.astype(np.int64))

        if self.return_names:
            return image_t, mask_t, img_path.name
        return image_t, mask_t

