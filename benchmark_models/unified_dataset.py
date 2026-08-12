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

def decode_carl_rgb_mask(mask_path: Path) -> np.ndarray:
    """
    Decodes the official CARL-D ___fuse.png RGB masks before resizing.
    ROAD_RGB (17, 163, 74) -> 1
    BLACK_RGB (0, 0, 0) -> 1
    BACKGROUND_RGB (15, 16, 65) -> 0
    Any other color raises ValueError.
    """
    mask_img = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
    if mask_img is None:
        raise FileNotFoundError(f"Could not load mask: {mask_path}")
    mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB)

    decoded = np.zeros(mask_img.shape[:2], dtype=np.uint8)
    road_mask = (mask_img[:, :, 0] == 17) & (mask_img[:, :, 1] == 163) & (mask_img[:, :, 2] == 74)
    black_mask = (mask_img[:, :, 0] == 0) & (mask_img[:, :, 1] == 0) & (mask_img[:, :, 2] == 0)
    bg_mask = (mask_img[:, :, 0] == 15) & (mask_img[:, :, 1] == 16) & (mask_img[:, :, 2] == 65)

    valid_pixels = road_mask | black_mask | bg_mask
    if not np.all(valid_pixels):
        invalid_count = np.sum(~valid_pixels)
        raise ValueError(f"Found {invalid_count} pixels with unexpected colors in {mask_path}")

    decoded[road_mask] = 1
    decoded[black_mask] = 1
    
    return decoded

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
        aug_params: Optional[dict] = None,
    ):
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.img_size = img_size
        self.return_names = return_names
        self.normalization = normalization or Normalization()
        self.aug_params = aug_params or {
            'shift_limit': 0.1, 'scale_limit': 0.1, 'rotate_limit': 10,
            'brightness_limit': 0.2, 'contrast_limit': 0.2,
            'gauss_var_limit': (10.0, 50.0), 'flip_p': 0.5, 'color_p': 0.3, 'noise_p': 0.2, 'geom_p': 0.5
        }

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
                    A.HorizontalFlip(p=self.aug_params['flip_p']),
                    A.ShiftScaleRotate(
                        shift_limit=self.aug_params['shift_limit'],
                        scale_limit=self.aug_params['scale_limit'],
                        rotate_limit=self.aug_params['rotate_limit'],
                        p=self.aug_params['geom_p']
                    ),
                    A.RandomBrightnessContrast(
                        brightness_limit=self.aug_params['brightness_limit'],
                        contrast_limit=self.aug_params['contrast_limit'],
                        p=self.aug_params['color_p']
                    ),
                    A.GaussNoise(var_limit=self.aug_params['gauss_var_limit'], p=self.aug_params['noise_p']),
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
        
        if "___fuse.png" in mask_path.name:
            mask = decode_carl_rgb_mask(mask_path)
        else:
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

