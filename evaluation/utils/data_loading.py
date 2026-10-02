"""
Data loading utilities for evaluation scripts.

Provides consistent MONAI-based preprocessing for CT volumes that matches
the training pipeline in src/data/dataloading.py.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import pandas as pd
import torch
from monai.data import CacheDataset, DataLoader
from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    ScaleIntensityd,
    Spacingd,
    SpatialPadd,
    ThresholdIntensityd,
    ToTensord,
    Resized,
)


def load_datalist_from_csv(csv_path: str) -> List[dict]:
    """
    Load image paths from a CSV file with an 'image' column.

    Returns a list of dicts with keys 'image' and 'filename', compatible with
    MONAI's CacheDataset. The 'filename' key enables per-sample result tracking.
    """
    df = pd.read_csv(csv_path)
    if "image" not in df.columns:
        raise ValueError(f"CSV at {csv_path} must have an 'image' column.")
    data_dicts = [
        {"image": str(row["image"]), "filename": Path(str(row["image"])).name}
        for _, row in df.iterrows()
    ]
    print(f"Loaded {len(data_dicts)} samples from {csv_path}")
    return data_dicts


def load_datalist_from_folder(
    folder: str,
    extensions: Tuple[str, ...] = (".nii.gz", ".nii", ".mha", ".mhd"),
) -> List[dict]:
    """
    Discover all volumes in a folder with the given extensions.

    Handles the .nii.gz / .nii overlap correctly by checking the full suffix.
    Files are sorted for reproducible ordering.
    """
    folder = Path(folder)
    seen: set = set()
    files: List[Path] = []
    for ext in extensions:
        for p in sorted(folder.glob(f"*{ext}")):
            if str(p) not in seen:
                seen.add(str(p))
                files.append(p)

    data_dicts = [{"image": str(f), "filename": f.name} for f in files]
    print(f"Found {len(data_dicts)} volumes in {folder}")
    return data_dicts


def build_eval_transforms(
    roi_size: Tuple[int, int, int] = (512, 512, 256),
    hu_min: float = -1000.0,
    hu_max: float = 1000.0,
    resample: bool = True,
    pixdim: Tuple[float, float, float] = (0.7, 0.7, 1.25),
) -> Compose:
    """
    Standard CT preprocessing pipeline for evaluation.

    Matches the validation transforms in src/data/dataloading.py:
      1. Load NIfTI / MHA volume
      2. Ensure channel-first layout: (1, D, H, W)
      3. Resample to isotropic spacing (optional)
      4. Clip HU to [hu_min, hu_max]
      5. Normalise to [-1, 1]
      6. Centre-crop then zero-pad to roi_size
    """
    transforms = [
        LoadImaged(keys=["image"], image_only=False),
        EnsureChannelFirstd(keys=["image"]),
    ]
    if resample:
        transforms.append(
            Spacingd(
                keys=["image"],
                pixdim=pixdim,
                mode="trilinear",
                align_corners=True,
            )
        )
    transforms += [
        ThresholdIntensityd(keys=["image"], threshold=hu_max, above=False, cval=hu_max),
        ThresholdIntensityd(keys=["image"], threshold=hu_min, above=True, cval=hu_min),
        ScaleIntensityd(keys=["image"], minv=-1.0, maxv=1.0),
        CenterSpatialCropd(keys=["image"], roi_size=roi_size),
        #SpatialPadd(keys=["image"], spatial_size=roi_size, constant_values=-1.0),
        Resized(keys=["image"],spatial_size=roi_size,mode="trilinear"),
        ToTensord(keys=["image"]),
    ]
    return Compose(transforms)


def get_eval_dataloader(
    data_dicts: List[dict],
    roi_size: Tuple[int, int, int] = (512, 512, 256),
    batch_size: int = 1,
    num_workers: int = 4,
    resample: bool = True,
    hu_min: float = -1000.0,
    hu_max: float = 1000.0,
    cache_rate: float = 0.0,
) -> DataLoader:
    """
    Build a DataLoader with standard CT preprocessing for evaluation.

    Args:
        data_dicts:   Output of load_datalist_from_csv / load_datalist_from_folder.
        roi_size:     Spatial size after centre-crop + pad. Use (512,512,256) for
                      full-volume evaluation or (256,256,128) to match VQ-GAN training.
        batch_size:   Typically 1 for large 3-D volumes.
        num_workers:  DataLoader workers.
        resample:     If True, resample to pixdim=(0.7,0.7,1.25) mm before cropping.
        hu_min/max:   HU clipping range.
        cache_rate:   Fraction of dataset to cache in RAM (0.0 = no caching).
    """
    transforms = build_eval_transforms(roi_size, hu_min, hu_max, resample)
    dataset = CacheDataset(data=data_dicts, transform=transforms, cache_rate=cache_rate)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
