"""
HU intensity histogram Wasserstein distance.

Compares the voxel-level HU distribution of a synthetic set against a real set
by computing the Wasserstein (Earth Mover's) distance between their aggregated
intensity histograms.  Also reports per-volume summary statistics.

Why HU histograms?
------------------
HU distribution captures global contrast, noise level, and tissue-type
balance in a way that feature-based metrics (FID, PRDC) cannot.  A synthetic
CT with incorrect liver/fat/bone HU values will appear unrealistic even if it
scores well on structural metrics.

Two comparison levels
---------------------
  population   One histogram per set (sum over all volumes, normalised).
               Wasserstein distance between the two population histograms.

  per-volume   One histogram per volume -> distributions of per-volume
               summary statistics (mean, std, percentiles).
               Wasserstein distance between the distributions of those stats.

Foreground masking
------------------
Air and padding voxels (-1000 HU) dominate the histogram and can mask
differences in soft tissue / bone.  By default we also report metrics with
voxels below --fg_threshold (default -300 HU) excluded, which roughly keeps
soft tissue and bone but removes large air cavities and background padding.

Usage
-----
python evaluation/compute_hu_distribution.py \\
    --real_dir  data/ct_volumes/real/ \\
    --fake_dir  outputs/samples/dit_l/ \\
    --output_dir outputs/evaluation/hu_distribution/

Or from CSVs:
python evaluation/compute_hu_distribution.py \\
    --real_csv ids/val.csv \\
    --fake_csv ids/synthetic_val.csv \\
    --output_dir outputs/evaluation/hu_distribution/

Outputs
-------
  real_per_volume_stats.tsv     – per-volume HU statistics for real set
  fake_per_volume_stats.tsv     – per-volume HU statistics for synthetic set
  population_histograms.npz     – normalised histogram arrays for both sets
  hu_distribution_summary.tsv  – Wasserstein distances + population stats
  plots/                        – histogram overlay plots (with --plot)
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from monai.data import Dataset, DataLoader
from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    Spacingd,
    SpatialPadd,
    ThresholdIntensityd,
    ToTensord,
)
from scipy.stats import wasserstein_distance
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.utils.data_loading import (
    load_datalist_from_csv,
    load_datalist_from_folder,
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare real vs. synthetic HU intensity distributions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    real_g = p.add_mutually_exclusive_group(required=True)
    real_g.add_argument("--real_dir", help="Folder of real CT volumes")
    real_g.add_argument("--real_csv", help="CSV file listing real CT volumes")

    fake_g = p.add_mutually_exclusive_group(required=True)
    fake_g.add_argument("--fake_dir", help="Folder of synthetic CT volumes")
    fake_g.add_argument("--fake_csv", help="CSV file listing synthetic CT volumes")

    p.add_argument("--output_dir", required=True)
    p.add_argument("--roi_size",   nargs=3, type=int, default=[256, 256, 128], metavar="N",
                   help="Spatial crop applied before histogram extraction (D H W).")
    p.add_argument("--hu_min",     type=float, default=-1000.0)
    p.add_argument("--hu_max",     type=float, default=1000.0)
    p.add_argument("--n_bins",     type=int,   default=200,
                   help="Number of histogram bins over [hu_min, hu_max].")
    p.add_argument("--fg_threshold", type=float, default=-300.0,
                   help="HU threshold for foreground mask.  Voxels below this are "
                        "considered air/background and excluded in the 'fg' metrics.")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--no_resample", action="store_true",
                   help="Skip resampling to isotropic spacing.")
    p.add_argument("--plot", action="store_true",
                   help="Save histogram overlay plots.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading (HU units — no intensity scaling)
# ---------------------------------------------------------------------------

def build_hu_transforms(
    roi_size: Tuple[int, int, int],
    hu_min: float,
    hu_max: float,
    resample: bool,
    pixdim: Tuple[float, float, float] = (0.7, 0.7, 1.25),
) -> Compose:
    """
    Preprocessing pipeline that preserves raw HU values.

    Identical to the standard eval pipeline except there is NO ScaleIntensityd
    step, so the output tensor contains voxel values in the original HU range.
    Padding fill value is hu_min (-1000) to match air HU.
    """
    tfms = [
        LoadImaged(keys=["image"], image_only=False),
        EnsureChannelFirstd(keys=["image"]),
    ]
    if resample:
        tfms.append(
            Spacingd(keys=["image"], pixdim=pixdim, mode="trilinear", align_corners=True)
        )
    tfms += [
        ThresholdIntensityd(keys=["image"], threshold=hu_max, above=False, cval=hu_max),
        ThresholdIntensityd(keys=["image"], threshold=hu_min, above=True,  cval=hu_min),
        # Intentionally no ScaleIntensityd — we want raw HU values
        CenterSpatialCropd(keys=["image"], roi_size=roi_size),
        SpatialPadd(keys=["image"], spatial_size=roi_size, constant_values=hu_min),
        ToTensord(keys=["image"]),
    ]
    return Compose(tfms)


# ---------------------------------------------------------------------------
# Per-volume statistics
# ---------------------------------------------------------------------------

def volume_stats(voxels: np.ndarray, fg_mask: np.ndarray) -> dict:
    """
    Compute summary statistics for all voxels and foreground voxels separately.

    Args:
        voxels:   1-D array of all voxel HU values in the volume.
        fg_mask:  Boolean array selecting foreground (non-air) voxels.
    """
    out: dict = {}
    for tag, vals in [("all", voxels), ("fg", voxels[fg_mask])]:
        if len(vals) == 0:
            for k in ("mean", "std", "p05", "p25", "p50", "p75", "p95"):
                out[f"{tag}_{k}"] = float("nan")
            continue
        out[f"{tag}_mean"] = float(vals.mean())
        out[f"{tag}_std"]  = float(vals.std())
        for q, label in [(5, "p05"), (25, "p25"), (50, "p50"), (75, "p75"), (95, "p95")]:
            out[f"{tag}_{label}"] = float(np.percentile(vals, q))
    out["fg_fraction"] = float(fg_mask.mean())
    return out


# ---------------------------------------------------------------------------
# Histogram helpers
# ---------------------------------------------------------------------------

def build_histogram(
    voxels: np.ndarray,
    bins: np.ndarray,
) -> np.ndarray:
    """Return a normalised count histogram over the supplied bin edges."""
    counts, _ = np.histogram(voxels, bins=bins)
    total = counts.sum()
    return counts / total if total > 0 else counts.astype(float)


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def process_dataset(
    data_dicts: List[dict],
    transforms: Compose,
    bins: np.ndarray,
    fg_threshold: float,
    label: str,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Iterate over all volumes and compute:
      - Per-volume summary statistics
      - Accumulated population histograms (all voxels + foreground only)

    Returns:
        stats_df:         DataFrame with per-volume statistics
        pop_hist_all:     Normalised population histogram (all voxels)
        pop_hist_fg:      Normalised population histogram (foreground voxels only)
    """
    ds = Dataset(data=data_dicts, transform=transforms)
    loader = DataLoader(ds, batch_size=1, num_workers=0, shuffle=False)

    records: List[dict] = []
    accum_all = np.zeros(len(bins) - 1, dtype=np.float64)
    accum_fg  = np.zeros(len(bins) - 1, dtype=np.float64)

    for batch in tqdm(loader, desc=f"Processing [{label}]"):
        vols = batch["image"].squeeze().numpy().ravel()  # (D*H*W,)
        fname = batch.get("filename", ["unknown"])
        fname = fname[0] if isinstance(fname, list) else fname

        fg = vols > fg_threshold
        stats = volume_stats(vols, fg)
        stats["filename"] = fname
        records.append(stats)

        accum_all += np.histogram(vols,      bins=bins)[0]
        accum_fg  += np.histogram(vols[fg],  bins=bins)[0] if fg.any() else 0.0

    stats_df = pd.DataFrame(records)

    pop_hist_all = accum_all / (accum_all.sum() + 1e-12)
    pop_hist_fg  = accum_fg  / (accum_fg.sum()  + 1e-12)

    return stats_df, pop_hist_all, pop_hist_fg


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def save_plots(
    bin_centres: np.ndarray,
    real_hist_all: np.ndarray,  fake_hist_all: np.ndarray,
    real_hist_fg:  np.ndarray,  fake_hist_fg:  np.ndarray,
    output_dir: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plots.")
        return

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    for tag, r, f in [
        ("all_voxels",   real_hist_all, fake_hist_all),
        ("foreground",   real_hist_fg,  fake_hist_fg),
    ]:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(bin_centres, r, label="Real",      color="steelblue",  linewidth=1.2)
        ax.plot(bin_centres, f, label="Synthetic", color="darkorange", linewidth=1.2, linestyle="--")
        ax.set_xlabel("HU")
        ax.set_ylabel("Normalised frequency")
        ax.set_title(f"HU distribution — {tag.replace('_', ' ')}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / f"hu_histogram_{tag}.png", dpi=150)
        plt.close(fig)

    print(f"Plots saved to {plots_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bins = np.linspace(args.hu_min, args.hu_max, args.n_bins + 1)
    bin_centres = (bins[:-1] + bins[1:]) / 2.0

    transforms = build_hu_transforms(
        roi_size=tuple(args.roi_size),
        hu_min=args.hu_min,
        hu_max=args.hu_max,
        resample=not args.no_resample,
    )

    real_dicts = (
        load_datalist_from_folder(args.real_dir) if args.real_dir
        else load_datalist_from_csv(args.real_csv)
    )
    fake_dicts = (
        load_datalist_from_folder(args.fake_dir) if args.fake_dir
        else load_datalist_from_csv(args.fake_csv)
    )

    print(f"\nProcessing {len(real_dicts)} real volumes …")
    real_stats, real_hist_all, real_hist_fg = process_dataset(
        real_dicts, transforms, bins, args.fg_threshold, "real"
    )

    print(f"\nProcessing {len(fake_dicts)} synthetic volumes …")
    fake_stats, fake_hist_all, fake_hist_fg = process_dataset(
        fake_dicts, transforms, bins, args.fg_threshold, "fake"
    )

    # ------------------------------------------------------------------
    # Save per-volume stats
    # ------------------------------------------------------------------
    real_stats.to_csv(output_dir / "real_per_volume_stats.tsv", sep="\t", index=False)
    fake_stats.to_csv(output_dir / "fake_per_volume_stats.tsv", sep="\t", index=False)

    # ------------------------------------------------------------------
    # Save histograms
    # ------------------------------------------------------------------
    np.savez_compressed(
        output_dir / "population_histograms.npz",
        bin_centres=bin_centres,
        real_all=real_hist_all, fake_all=fake_hist_all,
        real_fg=real_hist_fg,   fake_fg=fake_hist_fg,
    )

    # ------------------------------------------------------------------
    # Wasserstein distances (population histograms weighted by bin values)
    # ------------------------------------------------------------------
    wd_all = wasserstein_distance(bin_centres, bin_centres, real_hist_all, fake_hist_all)
    wd_fg  = wasserstein_distance(bin_centres, bin_centres, real_hist_fg,  fake_hist_fg)

    # Wasserstein distance between per-volume statistic distributions
    stat_cols = [c for c in real_stats.columns if c != "filename"]
    stat_rows: List[dict] = []
    for col in stat_cols:
        r_vals = real_stats[col].dropna().values
        f_vals = fake_stats[col].dropna().values
        if len(r_vals) == 0 or len(f_vals) == 0:
            continue
        stat_rows.append({
            "statistic":        col,
            "real_mean":        float(r_vals.mean()),
            "fake_mean":        float(f_vals.mean()),
            "real_std":         float(r_vals.std()),
            "fake_std":         float(f_vals.std()),
            "wasserstein_dist": float(wasserstein_distance(r_vals, f_vals)),
        })

    # ------------------------------------------------------------------
    # Summary report
    # ------------------------------------------------------------------
    summary_rows = [
        {"metric": "wasserstein_population_all_voxels", "value": wd_all},
        {"metric": "wasserstein_population_foreground",  "value": wd_fg},
    ]

    print("\n─── HU Distribution Summary ────────────────────────────────────")
    print(f"  Population Wasserstein (all voxels):   {wd_all:.4f} HU")
    print(f"  Population Wasserstein (foreground):   {wd_fg:.4f} HU")

    if stat_rows:
        stat_df = pd.DataFrame(stat_rows).sort_values("wasserstein_dist", ascending=False)
        print("\n  Per-volume statistic distributions (top-5 by Wasserstein distance):")
        for _, row in stat_df.head(5).iterrows():
            print(
                f"    {row['statistic']:<20}  "
                f"real={row['real_mean']:>8.2f}±{row['real_std']:.2f}  "
                f"fake={row['fake_mean']:>8.2f}±{row['fake_std']:.2f}  "
                f"WD={row['wasserstein_dist']:.4f}"
            )
        stat_df.to_csv(output_dir / "per_volume_stat_comparison.tsv", sep="\t", index=False)
        for _, row in stat_df.iterrows():
            summary_rows.append({
                "metric": f"wasserstein_per_volume_{row['statistic']}",
                "value":  row["wasserstein_dist"],
            })

    pd.DataFrame(summary_rows).to_csv(
        output_dir / "hu_distribution_summary.tsv", sep="\t", index=False
    )

    if args.plot:
        save_plots(
            bin_centres,
            real_hist_all, fake_hist_all,
            real_hist_fg,  fake_hist_fg,
            output_dir,
        )

    print(f"\nAll outputs → {output_dir}")


if __name__ == "__main__":
    main()
