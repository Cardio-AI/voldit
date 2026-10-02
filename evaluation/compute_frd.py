"""
Fréchet Radiomics Distance (FRD) for unconditional generation evaluation.

FRD extends the FID concept to radiomic features (shape, texture, intensity,
wavelet descriptors extracted by PyRadiomics) and measures how well the
radiomics distribution of synthetic volumes matches that of real volumes.

Reference implementation: https://github.com/RichardObi/frd-score
Install with:  pip install frd-score

Two evaluation modes
--------------------
  whole-volume    FRD computed on radiomics extracted from the full volume
                  (or a bounding-box crop around non-air voxels).

  segmented       FRD computed separately on each structure identified by
                  TotalSegmentator.  Useful if you want structure-specific
                  radiomics fidelity.  Requires --seg_dir with pre-existing
                  TotalSegmentator outputs (e.g. from compute_volume_distributions.py).

Usage
-----
# Whole-volume FRD
python evaluation/compute_frd.py \\
    --real_dir   data/ct_volumes/real/ \\
    --fake_dir   outputs/samples/dit_l/ \\
    --output_dir outputs/evaluation/frd/

# Segmented FRD (use masks from a previous TotalSegmentator run)
python evaluation/compute_frd.py \\
    --real_dir   data/ct_volumes/real/ \\
    --fake_dir   outputs/samples/dit_l/ \\
    --output_dir outputs/evaluation/frd/ \\
    --seg_dir    outputs/evaluation/volume_distributions/segmentations/ \\
    --structures liver spleen left_kidney right_kidney

Outputs
-------
  frd_results.tsv  – columns: mode, structure, frd_score
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.utils.data_loading import (
    load_datalist_from_csv,
    load_datalist_from_folder,
)


# ---------------------------------------------------------------------------
# Default structures for segmented mode
# ---------------------------------------------------------------------------

DEFAULT_STRUCTURES = [
    "liver", "spleen", "left_kidney", "right_kidney", "pancreas",
    "stomach", "gallbladder",
]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute Fréchet Radiomics Distance (FRD).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    real_g = p.add_mutually_exclusive_group(required=True)
    real_g.add_argument("--real_dir", help="Folder of real CT volumes")
    real_g.add_argument("--real_csv", help="CSV file listing real CT volumes")

    fake_g = p.add_mutually_exclusive_group(required=True)
    fake_g.add_argument("--fake_dir", help="Folder of synthetic CT volumes")
    fake_g.add_argument("--fake_csv", help="CSV file listing synthetic CT volumes")

    p.add_argument("--output_dir", required=True)
    p.add_argument(
        "--mode", nargs="+", default=["whole"],
        choices=["whole", "segmented"],
        help="Evaluation mode: 'whole' for full-volume FRD, "
             "'segmented' for per-structure FRD.",
    )
    p.add_argument(
        "--seg_dir", default=None,
        help="[segmented mode] Root directory with TotalSegmentator outputs. "
             "Expected layout: seg_dir/{real,fake}/<volume_stem>/<structure>.nii.gz",
    )
    p.add_argument(
        "--structures", nargs="+", default=None,
        help="[segmented mode] Structure names to evaluate. "
             "Defaults to a curated subset.",
    )
    p.add_argument("--batch_size", type=int, default=4,
                   help="Batch size for radiomics extraction.")
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()


# ---------------------------------------------------------------------------
# FRD helpers
# ---------------------------------------------------------------------------

def _collect_paths(data_dicts: List[dict]) -> List[str]:
    return [d["image"] for d in data_dicts]


def _compute_frd_whole(
    real_paths: List[str],
    fake_paths: List[str],
) -> float:
    """
    Compute whole-volume FRD using the frd_score package.

    The frd_score package extracts radiomic features via PyRadiomics and then
    computes a Fréchet distance in the radiomic feature space.

    The public API supports directory paths or lists of file paths.
    """
    try:
        # Try the list-of-paths API (frd-score >= 0.1)
        from frd_score import frd
        score = frd.compute_frd(
            path1=real_paths,
            path2=fake_paths,
        )
    except (ImportError, AttributeError, TypeError):
        try:
            # Fallback: directory-based API
            import os, tempfile
            from frd_score import frd

            with tempfile.TemporaryDirectory() as tmp:
                real_dir = Path(tmp) / "real"
                fake_dir = Path(tmp) / "fake"
                real_dir.mkdir()
                fake_dir.mkdir()
                for p in real_paths:
                    os.symlink(p, real_dir / Path(p).name)
                for p in fake_paths:
                    os.symlink(p, fake_dir / Path(p).name)
                score = frd.calculate_frd_given_paths(
                    [str(real_dir), str(fake_dir)]
                )
        except Exception as e:
            raise RuntimeError(
                f"frd_score API call failed: {e}\n"
                "Please check the frd_score installation and API version.\n"
                "Install: pip install frd-score"
            )
    return float(score)


def _compute_frd_segmented(
    real_paths: List[str],
    fake_paths: List[str],
    real_names: List[str],
    fake_names: List[str],
    seg_dir: Path,
    structures: List[str],
) -> List[dict]:
    """
    Compute per-structure FRD using segmentation masks from TotalSegmentator.

    For each structure:
      - Extract the masked sub-volume for each real and fake volume.
      - Compute FRD on those sub-volumes.

    This requires pre-existing TotalSegmentator outputs in seg_dir.
    """
    import nibabel as nib
    import numpy as np
    import tempfile, os
    from frd_score import frd

    rows: list = []
    for struct in structures:
        struct_real: List[str] = []
        struct_fake: List[str] = []

        with tempfile.TemporaryDirectory() as tmp:
            real_masked_dir = Path(tmp) / "real"
            fake_masked_dir = Path(tmp) / "fake"
            real_masked_dir.mkdir()
            fake_masked_dir.mkdir()

            # Apply mask to each real volume
            for vol_path, fname in zip(real_paths, real_names):
                stem = Path(fname).stem.replace(".nii", "").replace(".mha", "")
                mask_path = seg_dir / "real" / stem / f"{struct}.nii.gz"
                out_path = real_masked_dir / f"{stem}_{struct}.nii.gz"
                if _apply_mask_and_save(vol_path, mask_path, out_path):
                    struct_real.append(str(out_path))

            # Apply mask to each fake volume
            for vol_path, fname in zip(fake_paths, fake_names):
                stem = Path(fname).stem.replace(".nii", "").replace(".mha", "")
                mask_path = seg_dir / "fake" / stem / f"{struct}.nii.gz"
                out_path = fake_masked_dir / f"{stem}_{struct}.nii.gz"
                if _apply_mask_and_save(vol_path, mask_path, out_path):
                    struct_fake.append(str(out_path))

            if len(struct_real) < 2 or len(struct_fake) < 2:
                print(f"  [{struct}] Skipping — insufficient volumes with valid masks "
                      f"(real={len(struct_real)}, fake={len(struct_fake)})")
                rows.append({"mode": "segmented", "structure": struct, "frd_score": float("nan")})
                continue

            try:
                score = _compute_frd_whole(struct_real, struct_fake)
                print(f"  [{struct}]  FRD = {score:.4f}  "
                      f"(real={len(struct_real)}, fake={len(struct_fake)})")
                rows.append({"mode": "segmented", "structure": struct, "frd_score": score})
            except Exception as e:
                print(f"  [{struct}] FRD failed: {e}")
                rows.append({"mode": "segmented", "structure": struct, "frd_score": float("nan")})

    return rows


def _apply_mask_and_save(
    vol_path: str,
    mask_path: Path,
    out_path: Path,
) -> bool:
    """
    Zero-out voxels outside the binary mask and save the masked volume.
    Returns True on success, False if the mask file does not exist.
    """
    if not mask_path.exists():
        return False
    try:
        import nibabel as nib
        import numpy as np

        vol_img = nib.load(vol_path)
        mask_img = nib.load(str(mask_path))

        vol_data = vol_img.get_fdata(dtype=np.float32)
        mask_data = mask_img.get_fdata(dtype=np.float32)

        # Resample mask to volume space if shapes differ
        if vol_data.shape != mask_data.shape:
            from scipy.ndimage import zoom
            factors = [v / m for v, m in zip(vol_data.shape, mask_data.shape)]
            mask_data = zoom(mask_data, factors, order=0)

        masked = vol_data * (mask_data > 0.5)
        out_img = nib.Nifti1Image(masked, vol_img.affine, vol_img.header)
        nib.save(out_img, str(out_path))
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    try:
        from frd_score import frd  # noqa: F401
    except ImportError:
        raise ImportError(
            "The 'frd-score' package is required.\n"
            "Install it with:  pip install frd-score\n"
            "Source:  https://github.com/RichardObi/frd-score"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    real_dicts = (
        load_datalist_from_folder(args.real_dir)
        if args.real_dir else load_datalist_from_csv(args.real_csv)
    )
    fake_dicts = (
        load_datalist_from_folder(args.fake_dir)
        if args.fake_dir else load_datalist_from_csv(args.fake_csv)
    )

    real_paths = [d["image"] for d in real_dicts]
    fake_paths = [d["image"] for d in fake_dicts]
    real_names = [d["filename"] for d in real_dicts]
    fake_names = [d["filename"] for d in fake_dicts]

    rows: list = []

    # ------------------------------------------------------------------
    # Whole-volume FRD
    # ------------------------------------------------------------------
    if "whole" in args.mode:
        print(f"\nComputing whole-volume FRD "
              f"(real={len(real_paths)}, fake={len(fake_paths)}) …")
        try:
            score = _compute_frd_whole(real_paths, fake_paths)
            print(f"  Whole-volume FRD = {score:.4f}")
            rows.append({"mode": "whole", "structure": "all", "frd_score": score})
        except Exception as e:
            print(f"  ERROR: {e}")
            rows.append({"mode": "whole", "structure": "all", "frd_score": float("nan")})

    # ------------------------------------------------------------------
    # Segmented FRD
    # ------------------------------------------------------------------
    if "segmented" in args.mode:
        if args.seg_dir is None:
            print("\nWARNING: --seg_dir not provided; skipping segmented FRD. "
                  "Run compute_volume_distributions.py first, then pass --seg_dir.")
        else:
            seg_dir = Path(args.seg_dir)
            structures = args.structures or DEFAULT_STRUCTURES
            print(f"\nComputing segmented FRD for {len(structures)} structures …")
            seg_rows = _compute_frd_segmented(
                real_paths, fake_paths,
                real_names, fake_names,
                seg_dir, structures,
            )
            rows.extend(seg_rows)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    df = pd.DataFrame(rows)
    out_path = output_dir / "frd_results.tsv"
    df.to_csv(out_path, sep="\t", index=False)
    print(f"\nResults → {out_path}")


if __name__ == "__main__":
    main()
