"""
MS-SSIM pairwise diversity for synthetic volume sets.

Diversity is defined as  1 − mean(MS-SSIM)  over random pairs of synthetic
volumes. A score near 1 means high diversity (pairs look nothing alike);
a score near 0 indicates near-identical outputs (mode collapse).

MS-SSIM is computed in 2.5-D (averaged over three orthogonal planes) by
default, which is faster than full 3-D and still captures structural
similarity faithfully for CT data.

Usage
-----
python evaluation/compute_diversity.py \\
    --fake_dir  outputs/samples/dit_l/ \\
    --output_dir outputs/evaluation/diversity/ \\
    --n_pairs 500 \\
    --roi_size 256 256 128

Or from a CSV:
python evaluation/compute_diversity.py \\
    --fake_csv  ids/synthetic_val.csv \\
    --output_dir outputs/evaluation/diversity/ \\
    --n_pairs 500

Optionally pass --real_dir / --real_csv to also report cross-set MS-SSIM
(real vs. synthetic), which indicates how "realistic" the synthesised
volumes look relative to the training distribution.

Outputs
-------
  diversity_results.tsv   – n_pairs rows with per-pair ms_ssim scores
  diversity_summary.tsv   – mean / std / median + diversity score
"""

import argparse
import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.utils.data_loading import (
    build_eval_transforms,
    load_datalist_from_csv,
    load_datalist_from_folder,
)

try:
    from monai.metrics import MultiScaleSSIMMetric
    _HAS_MONAI = True
except ImportError:
    _HAS_MONAI = False


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute MS-SSIM pairwise diversity for a set of synthetic volumes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--fake_dir", help="Folder containing synthetic volumes")
    src.add_argument("--fake_csv", help="CSV file listing synthetic volume paths")

    real_src = p.add_mutually_exclusive_group()
    real_src.add_argument("--real_dir", help="(Optional) folder of real volumes for cross-set comparison")
    real_src.add_argument("--real_csv", help="(Optional) CSV of real volume paths for cross-set comparison")

    p.add_argument("--output_dir",   required=True)
    p.add_argument("--n_pairs",      type=int, default=500,
                   help="Number of random pairs to sample (synthetic–synthetic).")
    p.add_argument("--roi_size",     nargs=3, type=int, default=[256, 256, 128], metavar="N")
    p.add_argument("--batch_size",   type=int, default=1)
    p.add_argument("--num_workers",  type=int, default=4)
    p.add_argument("--device",       default="cuda")
    p.add_argument("--no_resample",  action="store_true")
    p.add_argument("--mode",         default="2p5d", choices=["2p5d", "3d"],
                   help="2p5d averages MS-SSIM over axial/coronal/sagittal slices; "
                        "3d uses the full-volume MONAI metric.")
    p.add_argument("--seed",         type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Volume loading helpers
# ---------------------------------------------------------------------------

def _load_volumes(
    paths: List[dict],
    transforms,
    device: torch.device,
) -> Tuple[List[torch.Tensor], List[str]]:
    """Load and preprocess all volumes into a list of (1,1,D,H,W) tensors."""
    from monai.data import Dataset, DataLoader

    ds = Dataset(data=paths, transform=transforms)
    loader = DataLoader(ds, batch_size=1, num_workers=0, shuffle=False)

    volumes, filenames = [], []
    for batch in tqdm(loader, desc="Loading volumes"):
        volumes.append(batch["image"].cpu())  # keep on CPU; MS-SSIM runs on CPU anyway
        fname = batch.get("filename", [""])
        filenames.append(fname[0] if isinstance(fname, list) else fname)

    return volumes, filenames


# ---------------------------------------------------------------------------
# MS-SSIM helpers
# ---------------------------------------------------------------------------

def _msssim_2p5d(
    a: torch.Tensor,
    b: torch.Tensor,
) -> float:
    """
    Compute 2.5-D MS-SSIM between two (1,1,D,H,W) tensors in [0,1].

    Slices along all three planes, computes 2-D MS-SSIM per slice,
    and returns the mean across all slices and planes.
    """
    # Both inputs should be in [0,1]
    metric = MultiScaleSSIMMetric(
        spatial_dims=2, data_range=1.0, kernel_type="gaussian", kernel_size=7
    )
    values: List[float] = []
    # dims: 2=axial(D), 3=coronal(H), 4=sagittal(W)
    for dim in (2, 3, 4):
        n = a.shape[dim]
        for i in range(n):
            if dim == 2:
                sa, sb = a[:, :, i], b[:, :, i]
            elif dim == 3:
                sa, sb = a[:, :, :, i], b[:, :, :, i]
            else:
                sa, sb = a[:, :, :, :, i], b[:, :, :, :, i]
            try:
                values.append(metric(sa.cpu(), sb.cpu()).mean().item())
            except Exception:
                pass
    return float(np.mean(values)) if values else float("nan")


def _msssim_3d(
    a: torch.Tensor,
    b: torch.Tensor,
) -> float:
    """Full 3-D MS-SSIM between two (1,1,D,H,W) tensors in [0,1]."""
    metric = MultiScaleSSIMMetric(
        spatial_dims=3, data_range=1.0, kernel_type="gaussian"
    )
    return metric(a.cpu(), b.cpu()).mean().item()


def compute_msssim(
    a: torch.Tensor,
    b: torch.Tensor,
    mode: str,
) -> float:
    """
    Compute MS-SSIM between two volumes.  Inputs must be in [-1,1];
    internally converted to [0,1] before the metric.
    """
    a01 = ((a.clamp(-1, 1) + 1.0) / 2.0).float()
    b01 = ((b.clamp(-1, 1) + 1.0) / 2.0).float()
    if mode == "3d":
        return _msssim_3d(a01, b01)
    return _msssim_2p5d(a01, b01)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not _HAS_MONAI:
        raise ImportError("monai is required: pip install monai")

    args = parse_args()
    random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    transforms = build_eval_transforms(
        roi_size=tuple(args.roi_size),
        resample=not args.no_resample,
    )

    # ------------------------------------------------------------------
    # Load synthetic volumes
    # ------------------------------------------------------------------
    if args.fake_dir:
        fake_dicts = load_datalist_from_folder(args.fake_dir)
    else:
        fake_dicts = load_datalist_from_csv(args.fake_csv)

    print(f"Loading {len(fake_dicts)} synthetic volumes …")
    fake_vols, fake_names = _load_volumes(fake_dicts, transforms, device)

    if len(fake_vols) < 2:
        raise ValueError("Need at least 2 synthetic volumes to compute pairwise MS-SSIM.")

    # ------------------------------------------------------------------
    # Synthetic pairwise diversity
    # ------------------------------------------------------------------
    n_pairs = min(args.n_pairs, len(fake_vols) * (len(fake_vols) - 1) // 2)
    pairs = set()
    while len(pairs) < n_pairs:
        i, j = random.sample(range(len(fake_vols)), 2)
        pairs.add((min(i, j), max(i, j)))
    pairs = list(pairs)

    print(f"\nComputing MS-SSIM for {len(pairs)} synthetic–synthetic pairs …")
    records: List[dict] = []
    for i, j in tqdm(pairs):
        val = compute_msssim(fake_vols[i], fake_vols[j], mode=args.mode)
        records.append({
            "file_a": fake_names[i], "file_b": fake_names[j],
            "ms_ssim": val, "set": "fake_vs_fake",
        })
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Optional: real vs. synthetic cross-set MS-SSIM
    # ------------------------------------------------------------------
    if args.real_dir or args.real_csv:
        if args.real_dir:
            real_dicts = load_datalist_from_folder(args.real_dir)
        else:
            real_dicts = load_datalist_from_csv(args.real_csv)

        print(f"Loading {len(real_dicts)} real volumes …")
        real_vols, real_names = _load_volumes(real_dicts, transforms, device)

        n_cross = min(args.n_pairs, len(real_vols) * len(fake_vols))
        cross_pairs = [
            (random.randrange(len(real_vols)), random.randrange(len(fake_vols)))
            for _ in range(n_cross)
        ]
        print(f"Computing MS-SSIM for {len(cross_pairs)} real–synthetic pairs …")
        for ri, fi in tqdm(cross_pairs):
            val = compute_msssim(real_vols[ri], fake_vols[fi], mode=args.mode)
            records.append({
                "file_a": real_names[ri], "file_b": fake_names[fi],
                "ms_ssim": val, "set": "real_vs_fake",
            })
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Save per-pair results
    # ------------------------------------------------------------------
    df = pd.DataFrame(records)
    df.to_csv(output_dir / "diversity_results.tsv", sep="\t", index=False)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    summary_rows: List[dict] = []
    print("\n─── MS-SSIM Diversity ──────────────────────────────────────────")
    for group in df["set"].unique():
        sub = df[df["set"] == group]["ms_ssim"].dropna()
        mean_ms = sub.mean()
        std_ms  = sub.std()
        div     = 1.0 - mean_ms
        print(
            f"  [{group}]  MS-SSIM = {mean_ms:.4f} ± {std_ms:.4f}   "
            f"Diversity = {div:.4f}  (n={len(sub)})"
        )
        summary_rows.extend([
            {"set": group, "metric": "ms_ssim_mean", "value": mean_ms},
            {"set": group, "metric": "ms_ssim_std",  "value": std_ms},
            {"set": group, "metric": "diversity",    "value": div},
            {"set": group, "metric": "n_pairs",      "value": len(sub)},
        ])

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "diversity_summary.tsv"
    summary_df.to_csv(summary_path, sep="\t", index=False)
    print(f"\nResults → {output_dir}")


if __name__ == "__main__":
    main()
