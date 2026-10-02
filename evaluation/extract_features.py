"""
Feature extraction for generative model evaluation.

Pre-extracts and caches feature vectors from a set of volumes (real or synthetic)
so that downstream scripts (compute_fid.py, compute_prdc.py, compute_waddiv.py …)
can load them directly without repeating inference.

Supported backbones
-------------------
  medicalnet   MedicalNet ResNet-50 (3-D, 23-dataset pretrained)
               → 3D features file (.npz with key 'features')
  radimagenet  RadImageNet ResNet-50 (2-D, radiological pretrained)
               → 2.5D features file (.npz with keys 'features_xy/yz/zx')

Usage examples
--------------
# Real volumes from CSV, 3-D features
python evaluation/extract_features.py \\
    --input ids/val.csv \\
    --output outputs/evaluation/features/real_3d.npz \\
    --backbone medicalnet \\
    --roi_size 256 256 128

# Synthetic volumes from folder, 2.5-D features
python evaluation/extract_features.py \\
    --input outputs/samples/dit_l/ \\
    --output outputs/evaluation/features/fake_2p5d.npz \\
    --backbone radimagenet \\
    --roi_size 256 256 128

The .npz files produced here are the inputs to the other evaluation scripts.
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.utils.data_loading import (
    get_eval_dataloader,
    load_datalist_from_csv,
    load_datalist_from_folder,
)
from evaluation.utils.feature_extraction import (
    extract_2p5d_features,
    extract_3d_features,
    load_medicalnet,
    load_radimagenet,
    save_features_2p5d,
    save_features_3d,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract and cache feature vectors from CT volumes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input", required=True,
        help="Path to a CSV file (with 'image' column) OR a folder containing volumes.",
    )
    p.add_argument(
        "--output", required=True,
        help="Output .npz file path (e.g. outputs/evaluation/features/real_3d.npz).",
    )
    p.add_argument(
        "--backbone", default="medicalnet",
        choices=["medicalnet", "radimagenet"],
        help="Feature extractor backbone.",
    )
    p.add_argument(
        "--roi_size", nargs=3, type=int, default=[256, 256, 128], metavar="N",
        help="Spatial crop size applied before feature extraction (D H W).",
    )
    p.add_argument("--batch_size",    type=int, default=1)
    p.add_argument("--num_workers",   type=int, default=4)
    p.add_argument("--device",        default="cuda")
    p.add_argument(
        "--no_resample", action="store_true",
        help="Skip resampling to isotropic spacing (use if volumes are already resampled).",
    )
    p.add_argument(
        "--slice_batch_size", type=int, default=32,
        help="[radimagenet] Number of 2-D slices per forward pass.",
    )
    p.add_argument(
        "--planes", nargs="+", default=["xy", "yz", "zx"],
        choices=["xy", "yz", "zx"],
        help="[radimagenet] Planes to slice along.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Load data list
    # ------------------------------------------------------------------
    input_path = Path(args.input)
    if input_path.is_dir():
        data_dicts = load_datalist_from_folder(str(input_path))
    else:
        data_dicts = load_datalist_from_csv(str(input_path))

    if not data_dicts:
        raise RuntimeError(f"No volumes found at {args.input}")

    loader = get_eval_dataloader(
        data_dicts,
        roi_size=tuple(args.roi_size),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        resample=not args.no_resample,
    )

    # ------------------------------------------------------------------
    # Load backbone and extract features
    # ------------------------------------------------------------------
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.backbone == "medicalnet":
        print("Loading MedicalNet …")
        model = load_medicalnet(device)
        features, filenames = extract_3d_features(model, loader, device)
        save_features_3d(str(output_path), features, filenames)

    else:  # radimagenet
        print("Loading RadImageNet …")
        model = load_radimagenet(device)
        features, filenames = extract_2p5d_features(
            model, loader, device,
            planes=tuple(args.planes),
            slice_batch_size=args.slice_batch_size,
        )
        save_features_2p5d(str(output_path), features, filenames)

    print(f"\nDone. Features saved to {output_path}")


if __name__ == "__main__":
    main()
