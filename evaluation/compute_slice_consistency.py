"""
Slice-to-slice gradient consistency for 3-D CT volumes.

Evaluates how smoothly the image content transitions between adjacent slices
along each spatial axis.  Real CT data should show gradual, anatomically
consistent transitions; generative models sometimes introduce z-axis
discontinuities (sudden intensity jumps, checkerboard patterns, inconsistent
anatomy) that are invisible to 2-D metrics but detectable here.

Metrics computed per volume
---------------------------
For each axis (axial=D, coronal=H, sagittal=W) the following are computed
between every pair of adjacent slices:

  MAD      Mean Absolute Difference  — raw intensity change per voxel
  NMAD     Normalised MAD = MAD / (intensity range + ε)  — scale-invariant
  MS-SSIM  Multi-Scale SSIM between adjacent slices — structural coherence
           (lower MS-SSIM = more structural change between slices)
  Corr     Pearson correlation — higher = more similar adjacent slices

Only the axial (z) axis is evaluated by default because CT generators most
commonly exhibit z-axis inconsistency.  Pass --axes to change this.

Population comparison
---------------------
For each metric the script reports:
  - Mean ± std for real and synthetic sets
  - Wasserstein distance between the two per-volume distributions

Usage
-----
python evaluation/compute_slice_consistency.py \\
    --real_dir  data/ct_volumes/real/ \\
    --fake_dir  outputs/samples/dit_l/ \\
    --output_dir outputs/evaluation/slice_consistency/ \\
    --roi_size 256 256 128

Or from CSVs:
python evaluation/compute_slice_consistency.py \\
    --real_csv ids/val.csv \\
    --fake_csv ids/synthetic_val.csv \\
    --output_dir outputs/evaluation/slice_consistency/

Outputs
-------
  real_per_volume.tsv    – per-volume metrics for real set
  fake_per_volume.tsv    – per-volume metrics for synthetic set
  consistency_summary.tsv – Wasserstein distances + mean/std per metric
  plots/                  – violin plots (with --plot)
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from monai.data import Dataset, DataLoader
from scipy.stats import wasserstein_distance
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
# Axis helpers
# ---------------------------------------------------------------------------

_AXIS_NAMES = {"axial": 0, "coronal": 1, "sagittal": 2}  # dims in (D, H, W)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate slice-to-slice gradient consistency of CT volumes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    real_g = p.add_mutually_exclusive_group(required=True)
    real_g.add_argument("--real_dir", help="Folder of real CT volumes")
    real_g.add_argument("--real_csv", help="CSV file listing real CT volumes")

    fake_g = p.add_mutually_exclusive_group(required=True)
    fake_g.add_argument("--fake_dir", help="Folder of synthetic CT volumes")
    fake_g.add_argument("--fake_csv", help="CSV file listing synthetic CT volumes")

    p.add_argument("--output_dir", required=True)
    p.add_argument("--roi_size",   nargs=3, type=int, default=[256, 256, 128], metavar="N")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--no_resample", action="store_true")
    p.add_argument(
        "--axes", nargs="+", default=["axial"],
        choices=["axial", "coronal", "sagittal"],
        help="Axes along which to compute inter-slice metrics. "
             "Evaluating all three is thorough but ~3× slower.",
    )
    p.add_argument(
        "--no_msssim", action="store_true",
        help="Skip MS-SSIM computation (much faster; keeps MAD and Pearson r only).",
    )
    p.add_argument(
        "--msssim_every_n", type=int, default=4,
        help="Compute MS-SSIM only for every N-th adjacent pair to reduce runtime. "
             "Set to 1 to evaluate all pairs.",
    )
    p.add_argument("--plot", action="store_true", help="Save violin plots.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Per-volume consistency metrics
# ---------------------------------------------------------------------------

def _pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r between two flat arrays; returns NaN if degenerate."""
    a, b = a.ravel(), b.ravel()
    if a.std() < 1e-6 or b.std() < 1e-6:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _msssim_pair(
    s1: torch.Tensor,
    s2: torch.Tensor,
    metric: "MultiScaleSSIMMetric",
) -> float:
    """MS-SSIM between two 2-D slices (1,1,H,W) in [-1,1] -> [0,1]."""
    a = ((s1.clamp(-1, 1) + 1.0) / 2.0).unsqueeze(0)  # (1,1,H,W)
    b = ((s2.clamp(-1, 1) + 1.0) / 2.0).unsqueeze(0)
    try:
        return float(metric(a, b).mean().item())
    except Exception:
        return float("nan")


def compute_volume_consistency(
    volume: torch.Tensor,
    axes: List[str],
    compute_msssim: bool,
    msssim_every_n: int,
) -> dict:
    """
    Compute inter-slice consistency metrics for a single volume.

    Args:
        volume:         (1, D, H, W) tensor in [-1, 1].
        axes:           Which axes to evaluate.
        compute_msssim: Whether to include MS-SSIM (expensive).
        msssim_every_n: Stride for MS-SSIM slice pairs.

    Returns:
        Flat dict with keys like 'axial_mad', 'axial_nmad', 'axial_corr',
        'axial_msssim' (one value per metric per axis, averaged over all pairs).
    """
    v = volume.squeeze(0)  # (D, H, W)
    intensity_range = float(v.max() - v.min()) + 1e-6
    v_np = v.numpy()

    if compute_msssim and _HAS_MONAI:
        ms_metric = MultiScaleSSIMMetric(
            spatial_dims=2, data_range=1.0, kernel_type="gaussian", kernel_size=7
        )
    else:
        ms_metric = None

    results: dict = {}

    for axis_name in axes:
        dim = _AXIS_NAMES[axis_name]
        n_slices = v_np.shape[dim]

        mads, nmads, corrs, msssims = [], [], [], []

        for i in range(n_slices - 1):
            s1 = np.take(v_np, i,     axis=dim)  # (*, *)
            s2 = np.take(v_np, i + 1, axis=dim)

            diff = np.abs(s1 - s2)
            mads.append(float(diff.mean()))
            nmads.append(float(diff.mean() / intensity_range))
            corrs.append(_pearson_r(s1, s2))

            if ms_metric is not None and i % msssim_every_n == 0:
                t1 = torch.from_numpy(s1).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
                t2 = torch.from_numpy(s2).unsqueeze(0).unsqueeze(0)
                msssims.append(_msssim_pair(t1, t2, ms_metric))

        results[f"{axis_name}_mad"]   = float(np.nanmean(mads))
        results[f"{axis_name}_nmad"]  = float(np.nanmean(nmads))
        results[f"{axis_name}_corr"]  = float(np.nanmean(corrs))

        if msssims:
            results[f"{axis_name}_msssim"] = float(np.nanmean(msssims))

    return results


# ---------------------------------------------------------------------------
# Dataset processing
# ---------------------------------------------------------------------------

def process_dataset(
    data_dicts: List[dict],
    transforms,
    axes: List[str],
    compute_msssim: bool,
    msssim_every_n: int,
    label: str,
) -> pd.DataFrame:
    ds = Dataset(data=data_dicts, transform=transforms)
    loader = DataLoader(ds, batch_size=1, num_workers=0, shuffle=False)

    records: List[dict] = []
    for batch in tqdm(loader, desc=f"Processing [{label}]"):
        x = batch["image"]  # (1, 1, D, H, W)

        fname = batch.get("filename", ["unknown"])
        fname = fname[0] if isinstance(fname, list) else fname

        metrics = compute_volume_consistency(
            x.squeeze(0),  # (1, D, H, W) -> squeeze batch dim
            axes=axes,
            compute_msssim=compute_msssim,
            msssim_every_n=msssim_every_n,
        )
        metrics["filename"] = fname
        records.append(metrics)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def save_violin_plots(
    real_df: pd.DataFrame,
    fake_df: pd.DataFrame,
    metric_cols: List[str],
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

    for col in metric_cols:
        r = real_df[col].dropna().values
        f = fake_df[col].dropna().values
        if len(r) == 0 or len(f) == 0:
            continue

        fig, ax = plt.subplots(figsize=(4, 5))
        parts = ax.violinplot(
            [r, f], positions=[0, 1], showmedians=True, showextrema=True
        )
        # Colour real blue, synthetic orange
        for i, (pc, colour) in enumerate(
            zip(parts["bodies"], ["steelblue", "darkorange"])
        ):
            pc.set_facecolor(colour)
            pc.set_alpha(0.7)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Real", "Synthetic"])
        ax.set_ylabel(col)
        ax.set_title(col.replace("_", " ").title())
        fig.tight_layout()
        fig.savefig(plots_dir / f"{col}.png", dpi=150)
        plt.close(fig)

    print(f"Plots saved to {plots_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    transforms = build_eval_transforms(
        roi_size=tuple(args.roi_size),
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

    compute_msssim = (not args.no_msssim) and _HAS_MONAI
    if not args.no_msssim and not _HAS_MONAI:
        print("WARNING: monai not available — skipping MS-SSIM.")

    print(f"\nAxes to evaluate: {args.axes}")
    print(f"MS-SSIM: {'enabled (every {args.msssim_every_n} pairs)' if compute_msssim else 'disabled'}")

    print(f"\nProcessing {len(real_dicts)} real volumes …")
    real_df = process_dataset(
        real_dicts, transforms, args.axes, compute_msssim, args.msssim_every_n, "real"
    )

    print(f"\nProcessing {len(fake_dicts)} synthetic volumes …")
    fake_df = process_dataset(
        fake_dicts, transforms, args.axes, compute_msssim, args.msssim_every_n, "fake"
    )

    real_df.to_csv(output_dir / "real_per_volume.tsv", sep="\t", index=False)
    fake_df.to_csv(output_dir / "fake_per_volume.tsv", sep="\t", index=False)

    # ------------------------------------------------------------------
    # Population comparison
    # ------------------------------------------------------------------
    metric_cols = [c for c in real_df.columns if c != "filename"]
    summary_rows: List[dict] = []

    print("\n─── Slice Consistency Summary ───────────────────────────────────")
    print(f"  {'Metric':<25}  {'Real mean±std':>20}  {'Fake mean±std':>20}  {'WD':>10}")
    print("  " + "─" * 80)

    for col in metric_cols:
        r_vals = real_df[col].dropna().values
        f_vals = fake_df[col].dropna().values
        if len(r_vals) == 0 or len(f_vals) == 0:
            continue

        wd = float(wasserstein_distance(r_vals, f_vals))
        r_m, r_s = r_vals.mean(), r_vals.std()
        f_m, f_s = f_vals.mean(), f_vals.std()

        print(
            f"  {col:<25}  {r_m:>8.5f}±{r_s:.5f}  "
            f"{f_m:>8.5f}±{f_s:.5f}  {wd:>10.6f}"
        )
        summary_rows.extend([
            {"metric": col, "stat": "real_mean",        "value": r_m},
            {"metric": col, "stat": "real_std",         "value": r_s},
            {"metric": col, "stat": "fake_mean",        "value": f_m},
            {"metric": col, "stat": "fake_std",         "value": f_s},
            {"metric": col, "stat": "wasserstein_dist", "value": wd},
        ])

    pd.DataFrame(summary_rows).to_csv(
        output_dir / "consistency_summary.tsv", sep="\t", index=False
    )

    if args.plot:
        save_violin_plots(real_df, fake_df, metric_cols, output_dir)

    print(f"\nAll outputs → {output_dir}")


if __name__ == "__main__":
    main()
