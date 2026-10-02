"""
Feature extraction utilities for generative model evaluation.

Supports:
  - 3D features via MedicalNet (ResNet-50, 23-dataset pretrained)
  - 2.5D features via RadImageNet (ResNet-50, radiological image pretrained)

Both models are loaded from torch.hub and cached locally on first use.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def load_medicalnet(
    device: torch.device,
    model_name: str = "medicalnet_resnet50_23datasets",
) -> nn.Module:
    """
    Load a pretrained MedicalNet (3D ResNet) from torch.hub.

    The model accepts (B, 1, D, H, W) tensors normalised to [-1, 1].
    The classification head is replaced with nn.Identity so the output is
    a (B, feature_dim) feature vector after global average pooling.

    Install deps (first call only):
        torch.hub.load will download weights automatically.
    """
    model = torch.hub.load(
        "Warvito/MedicalNet-models", model_name, verbose=True, trust_repo=True
    )
    # Remove the classification FC — we want penultimate features
    if hasattr(model, "fc"):
        model.fc = nn.Identity()
    return model.to(device).eval()


def load_radimagenet(
    device: torch.device,
    model_name: str = "radimagenet_resnet50",
) -> nn.Module:
    """
    Load a pretrained RadImageNet (2D ResNet-50) from torch.hub.

    Accepts (B, 3, H, W) BGR images normalised with ImageNet statistics.
    Use extract_2p5d_features() to feed 3D volumes slice-by-slice.

    Install deps (first call only):
        torch.hub.load will download weights automatically.
    """
    model = torch.hub.load(
        "Warvito/radimagenet-models", model_name, verbose=True, trust_repo=True
    )
    # The hub model already returns feature vectors (no FC head)
    return model.to(device).eval()


# ---------------------------------------------------------------------------
# 3-D feature extraction
# ---------------------------------------------------------------------------

def extract_3d_features(
    model: nn.Module,
    dataloader,
    device: torch.device,
) -> Tuple[torch.Tensor, List[str]]:
    """
    Extract volumetric features with a 3D CNN (e.g. MedicalNet).

    Volumes should be normalised to [-1, 1] (the training convention in this
    project).  The output is globally average-pooled if the model returns a
    spatial feature map instead of a flat vector.

    Returns:
        features:  (N, feature_dim) float32 CPU tensor
        filenames: list of N filename strings
    """
    features_list: List[torch.Tensor] = []
    filenames: List[str] = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting 3D features"):
            x = batch["image"].to(device)  # (B, 1, D, H, W)

            with torch.amp.autocast("cuda"):
                out = model(x)
                if out.dim() > 2:
                    out = F.adaptive_avg_pool3d(out, (1, 1, 1)).flatten(1)

            features_list.append(out.float().cpu())

            fnames = batch.get("filename", [])
            if isinstance(fnames, torch.Tensor):
                fnames = fnames.tolist()
            elif isinstance(fnames, str):
                fnames = [fnames]
            filenames.extend(fnames)

    return torch.cat(features_list, dim=0), filenames


# ---------------------------------------------------------------------------
# 2.5-D feature extraction
# ---------------------------------------------------------------------------

# ImageNet channel statistics (RGB order), applied after replicating grey -> 3ch
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _radimagenet_normalise(x: torch.Tensor) -> torch.Tensor:
    """
    Normalise a (B, 3, H, W) tensor in [-1, 1] for RadImageNet input.

    Steps:
      1. Rescale [-1, 1] -> [0, 1]
      2. Subtract ImageNet per-channel mean and divide by std
    """
    x = (x + 1.0) / 2.0
    mean = _IMAGENET_MEAN.to(x.device)
    std = _IMAGENET_STD.to(x.device)
    return (x - mean) / std


def extract_2p5d_features(
    model: nn.Module,
    dataloader,
    device: torch.device,
    planes: Tuple[str, ...] = ("xy", "yz", "zx"),
    slice_batch_size: int = 32,
    drop_empty_threshold: float = -0.90,
) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    """
    Extract 2.5D features from a 2D CNN (e.g. RadImageNet).

    For each volume the function:
      1. Slices along each requested orthogonal plane.
      2. Discards near-empty slices (background / air) below the threshold.
      3. Feeds slices through the 2D backbone in mini-batches.
      4. Averages slice features -> one (feature_dim,) vector per plane per volume.

    Volume layout: (B, 1, D, H, W)  [MONAI channel-first convention]
      xy plane = axial    -> slice dim 2 (D), yielding (B, 1, H, W) slices
      yz plane = coronal  -> slice dim 3 (H), yielding (B, 1, D, W) slices
      zx plane = sagittal -> slice dim 4 (W), yielding (B, 1, D, H) slices

    Args:
        model:                  2D CNN returning (B, feature_dim) or (B, C, H', W').
        dataloader:             DataLoader yielding dicts with 'image' and 'filename'.
        device:                 Torch device.
        planes:                 Which planes to extract ('xy', 'yz', 'zx').
        slice_batch_size:       Number of slices processed together (tune to VRAM).
        drop_empty_threshold:   Slices with mean below this value are skipped.
                                In [-1,1] space, -0.90 ~ -900 HU (mostly air).

    Returns:
        features:  dict[plane -> (N, feature_dim) CPU tensor]
        filenames: list of N filename strings
    """
    plane_dims: Dict[str, int] = {"xy": 2, "yz": 3, "zx": 4}
    plane_features: Dict[str, List[torch.Tensor]] = {p: [] for p in planes}
    filenames: List[str] = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting 2.5D features"):
            x = batch["image"].to(device)  # (B, 1, D, H, W)

            for plane in planes:
                dim = plane_dims[plane]
                all_slices = x.unbind(dim=dim)  # tuple of (B, 1, *, *) tensors

                # Filter near-empty slices (air-filled background)
                valid = [s for s in all_slices if s.mean().item() > drop_empty_threshold]
                if not valid:
                    valid = [all_slices[len(all_slices) // 2]]  # fallback: centre slice

                # Process in mini-batches of slices
                slice_feat_chunks: List[torch.Tensor] = []
                for i in range(0, len(valid), slice_batch_size):
                    chunk = torch.cat(valid[i : i + slice_batch_size], dim=0)  # (N, 1, H, W)
                    chunk = chunk.repeat(1, 3, 1, 1)  # (N, 3, H, W)
                    chunk = _radimagenet_normalise(chunk)

                    with torch.amp.autocast("cuda"):
                        feat = model(chunk)
                        if feat.dim() > 2:
                            feat = F.adaptive_avg_pool2d(feat, (1, 1)).flatten(1)

                    slice_feat_chunks.append(feat.float())

                # Keep all slice features: (N_valid_slices, feature_dim) per volume
                vol_feat = torch.cat(slice_feat_chunks, dim=0)
                plane_features[plane].append(vol_feat.cpu())

            fnames = batch.get("filename", [])
            if isinstance(fnames, torch.Tensor):
                fnames = fnames.tolist()
            elif isinstance(fnames, str):
                fnames = [fnames]
            filenames.extend(fnames)

    features = {p: torch.cat(plane_features[p], dim=0) for p in planes}
    return features, filenames


# ---------------------------------------------------------------------------
# Feature file I/O
# ---------------------------------------------------------------------------

def save_features_3d(
    path: str,
    features: torch.Tensor,
    filenames: List[str],
) -> None:
    """Save 3D features to a .npz file."""
    import numpy as np

    np.savez_compressed(
        path,
        features=features.numpy(),
        filenames=np.array(filenames, dtype=object),
    )
    print(f"Saved 3D features {tuple(features.shape)} -> {path}")


def save_features_2p5d(
    path: str,
    features: Dict[str, torch.Tensor],
    filenames: List[str],
) -> None:
    """Save 2.5D features (one array per plane) to a .npz file."""
    import numpy as np

    arrays = {f"features_{p}": features[p].numpy() for p in features}
    arrays["filenames"] = np.array(filenames, dtype=object)
    np.savez_compressed(path, **arrays)
    planes = list(features.keys())
    shapes = {p: tuple(features[p].shape) for p in planes}
    print(f"Saved 2.5D features {shapes} -> {path}")


def load_features(path: str) -> Tuple[Optional[torch.Tensor], Optional[Dict[str, torch.Tensor]], List[str]]:
    """
    Load features from a .npz file saved by save_features_3d or save_features_2p5d.

    Returns:
        features_3d:   (N, D) tensor if file was 3D, else None
        features_2p5d: dict[plane -> (N, D)] if file was 2.5D, else None
        filenames:     list of filename strings
    """
    import numpy as np

    data = np.load(path, allow_pickle=True)
    filenames: List[str] = data["filenames"].tolist() if "filenames" in data else []

    if "features" in data:
        return torch.from_numpy(data["features"]), None, filenames

    # 2.5D: keys are features_xy, features_yz, features_zx
    planes = [k[len("features_"):] for k in data.files if k.startswith("features_")]
    if planes:
        feat_dict = {p: torch.from_numpy(data[f"features_{p}"]) for p in planes}
        return None, feat_dict, filenames

    raise ValueError(f"Unrecognised feature file format at {path}")
