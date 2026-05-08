# data/dataloading.py
import pandas as pd
from pathlib import Path
from typing import Tuple, Union, List
import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
import numpy as np

from monai.data import CacheDataset, PersistentDataset, DataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ThresholdIntensityd,
    ScaleIntensityd,
    RandRotated,
    RandFlipd,
    RandSpatialCropd,
    CenterSpatialCropd,
    ToTensord,
    SpatialPadd
)

def get_datalist(ids_path: str, extended_report: bool = False) -> list[dict]:
    """
    Reads a CSV file with an 'image' column and returns a list of dictionaries
    for MONAI datasets.
    """
    df = pd.read_csv(ids_path, sep=",")
    data_dicts = [{"image": str(row["image"])} for _, row in df.iterrows()]

    if extended_report:
        print("Data dict preview:")
        for i, d in enumerate(data_dicts[:5]):
            print(f"{i}: {d}")

    print(f"Found {len(data_dicts)} subjects.")
    return data_dicts

def get_datalist_cond(ids_path: str, condition_keys: List[str] = [], extended_report: bool = False) -> List[dict]:
    """
    Reads a CSV file with 'image' and condition columns.
    Returns list of dicts suitable for ControlNetDataset.
    """
    df = pd.read_csv(ids_path, sep=",")
    data_dicts = []

    for _, row in df.iterrows():
        entry = {"image": str(row["image"])}
        for key in condition_keys:
            entry[key] = str(row[key])
        data_dicts.append(entry)

    if extended_report:
        print("Data dict preview:", data_dicts[:5])
    print(f"Found {len(data_dicts)} subjects.")
    return data_dicts

def get_dataloader(
    cache_dir: Union[str, Path],
    training_ids: str,
    validation_ids: str,
    batch_size: int,
    num_workers: int = 4,
    use_persistent: bool = False,
    roi_size: Tuple[int, int, int] = (64, 64, 64),
    rank: int = 0,
    world_size: int = 1,
    cache_rate: float = 0.0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Returns training and validation DataLoaders ready for DDP.

    Args:
        cache_dir: directory for persistent dataset cache
        training_ids: CSV file with training image paths
        validation_ids: CSV file with validation image paths
        batch_size: batch size for DataLoader
        num_workers: DataLoader workers
        use_persistent: use PersistentDataset instead of CacheDataset
        roi_size: crop size for random spatial cropping
        rank: rank of current process (DDP)
        world_size: total number of processes (DDP)
    """
    train_files = get_datalist(training_ids)
    val_files = get_datalist(validation_ids)

    train_transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        ThresholdIntensityd(keys=["image"], threshold=1000, above=False, cval=1000),
        ThresholdIntensityd(keys=["image"], threshold=-1000, above=True, cval=-1000),
        ScaleIntensityd(keys=["image"], minv=-1.0, maxv=1.0),
        RandRotated(keys=["image"], range_x=0.0872665, range_y=0.0872665, range_z=0.0872665, prob=0.2),
        RandFlipd(keys=["image"], spatial_axis=1, prob=0.5),
        RandSpatialCropd(keys=["image"], roi_size=roi_size, random_size=False),
        SpatialPadd(keys=["image"], spatial_size=roi_size, constant_values=-1.0),
        ToTensord(keys=["image"])
    ])

    val_transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        ThresholdIntensityd(keys=["image"], threshold=1000, above=False, cval=1000),
        ThresholdIntensityd(keys=["image"], threshold=-1000, above=True, cval=-1000),
        ScaleIntensityd(keys=["image"], minv=-1.0, maxv=1.0),
        CenterSpatialCropd(keys=["image"], roi_size=roi_size),
        SpatialPadd(keys=["image"], spatial_size=roi_size, constant_values=-1.0),
        ToTensord(keys=["image"])
    ])

    if use_persistent:
        train_ds = PersistentDataset(data=train_files, transform=train_transforms, cache_dir=str(Path(cache_dir) / "train"))
        val_ds = PersistentDataset(data=val_files, transform=val_transforms, cache_dir=str(Path(cache_dir) / "val"))
    else:
        train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=cache_rate)
        val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=0.0)

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        sampler=val_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


# -------------------------
# Shared safe loader for .pt latent files
# -------------------------
def _safe_load(path):
    with torch.serialization.safe_globals([
        np.ndarray,
        np.dtype,
        np._core.multiarray._reconstruct
    ]):
        return torch.load(path, weights_only=False)


# -------------------------
# Custom Dataset for precomputed latents
# -------------------------

class LatentDataset(Dataset):
    """
    Dataset for precomputed latent tensors stored as .pt files.
    Compatible with PyTorch 2.6+ safe loading.
    """
    def __init__(self, file_list: List, preload: bool = True):
        self.file_list = [
            f["image"] if isinstance(f, dict) else f
            for f in file_list
        ]
        self.preload = preload

        if preload:
            self.data = [_safe_load(f) for f in self.file_list]
        else:
            self.data = None

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if self.preload:
            return {"image": self.data[idx]}
        else:
            return {"image": _safe_load(self.file_list[idx])}

class ControlNetDataset(Dataset):
    """
    Dataset for latent diffusion ControlNet training.
    Supports precomputed latents for images and raw masks for conditions.
    Uses PyTorch safe-loading for compatibility with PyTorch 2.6+.
    """

    def __init__(self, data_list: list[dict], condition_keys: list[str], preload_latents: bool = True):
        self.data_list = data_list
        self.condition_keys = condition_keys
        self.preload_latents = preload_latents

        if preload_latents:
            self.latents = [_safe_load(entry["image"]) for entry in data_list]
            self.masks = [
                {k: _safe_load(entry[k]) for k in condition_keys}
                for entry in data_list
            ]
        else:
            self.latents = [None] * len(data_list)
            self.masks = [None] * len(data_list)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if self.preload_latents:
            image = self.latents[idx]
            conditions = self.masks[idx]
        else:
            image = _safe_load(self.data_list[idx]["image"])
            conditions = {k: _safe_load(self.data_list[idx][k]) for k in self.condition_keys}

        return {"image": image, **conditions}

# -------------------------
# DiT Dataloader function
# -------------------------
def get_dit_dataloader(
    training_ids: str,
    validation_ids: str,
    batch_size: int,
    num_workers: int = 4,
    rank: int = 0,
    world_size: int = 1,
    roi_size: Tuple[int, int, int] = (64, 64, 64),
    use_precomputed_latents: bool = True,
    preload_latents: bool = True,
    use_persistent: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """
    Returns training and validation DataLoaders for DiT training.
    Supports precomputed latents (.pt) or raw images.
    """
    train_files = get_datalist(training_ids)
    val_files = get_datalist(validation_ids)

    if use_precomputed_latents:
        train_ds = LatentDataset(train_files, preload=preload_latents)
        val_ds = LatentDataset(val_files, preload=preload_latents)
    else:
        train_transforms = Compose([
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            ThresholdIntensityd(keys=["image"], threshold=1000, above=False, cval=1000),
            ThresholdIntensityd(keys=["image"], threshold=-1000, above=True, cval=-1000),
            ScaleIntensityd(keys=["image"], minv=-1.0, maxv=1.0),
            RandRotated(keys=["image"], range_x=0.0872665, range_y=0.0872665, range_z=0.0872665, prob=0.2),
            RandFlipd(keys=["image"], spatial_axis=1, prob=0.5),
            RandSpatialCropd(keys=["image"], roi_size=roi_size, random_size=False),
            ToTensord(keys=["image"])
        ])

        val_transforms = Compose([
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            ThresholdIntensityd(keys=["image"], threshold=1000, above=False, cval=1000),
            ThresholdIntensityd(keys=["image"], threshold=-1000, above=True, cval=-1000),
            ScaleIntensityd(keys=["image"], minv=-1.0, maxv=1.0),
            CenterSpatialCropd(keys=["image"], roi_size=roi_size),
            SpatialPadd(keys=["image"], spatial_size=roi_size),
            ToTensord(keys=["image"])
        ])

        if use_persistent:
            train_ds = PersistentDataset(data=train_files, transform=train_transforms, cache_dir="/tmp/dit_train_cache")
            val_ds = PersistentDataset(data=val_files, transform=val_transforms, cache_dir="/tmp/dit_val_cache")
        else:
            train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=0.5)
            val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=0.0)

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader

def get_controlnet_dataloader(
    training_ids: str,
    validation_ids: str,
    condition_keys: list[str],
    batch_size: int,
    num_workers: int = 4,
    rank: int = 0,
    world_size: int = 1,
    roi_size: Tuple[int, int, int] = (64, 64, 64),
    use_precomputed_latents: bool = True,
    preload_latents: bool = True,
    use_persistent: bool = False,
) -> tuple[DataLoader, DataLoader]:

    train_files = get_datalist_cond(training_ids, condition_keys)
    val_files = get_datalist_cond(validation_ids, condition_keys)

    if use_precomputed_latents:
        train_ds = ControlNetDataset(train_files, condition_keys, preload_latents=preload_latents)
        val_ds = ControlNetDataset(val_files, condition_keys, preload_latents=preload_latents)
    else:
        all_keys = ["image"] + condition_keys
        train_transforms = Compose([
            LoadImaged(keys=all_keys),
            EnsureChannelFirstd(keys=all_keys),
            ThresholdIntensityd(keys=["image"], threshold=1000, above=False, cval=1000),
            ThresholdIntensityd(keys=["image"], threshold=-1000, above=True, cval=-1000),
            ScaleIntensityd(keys=["image"], minv=-1.0, maxv=1.0),
            RandRotated(keys=all_keys, range_x=0.0872665, range_y=0.0872665, range_z=0.0872665, prob=0.2),
            RandFlipd(keys=all_keys, spatial_axis=1, prob=0.5),
            RandSpatialCropd(keys=all_keys, roi_size=roi_size, random_size=False),
            ToTensord(keys=all_keys),
        ])
        val_transforms = Compose([
            LoadImaged(keys=all_keys),
            EnsureChannelFirstd(keys=all_keys),
            ThresholdIntensityd(keys=["image"], threshold=1000, above=False, cval=1000),
            ThresholdIntensityd(keys=["image"], threshold=-1000, above=True, cval=-1000),
            ScaleIntensityd(keys=["image"], minv=-1.0, maxv=1.0),
            CenterSpatialCropd(keys=all_keys, roi_size=roi_size),
            SpatialPadd(keys=all_keys, spatial_size=roi_size),
            ToTensord(keys=all_keys),
        ])

        if use_persistent:
            train_ds = PersistentDataset(data=train_files, transform=train_transforms, cache_dir="/tmp/cnet_train_cache")
            val_ds = PersistentDataset(data=val_files, transform=val_transforms, cache_dir="/tmp/cnet_val_cache")
        else:
            train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=0.5)
            val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=0.0)

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader
