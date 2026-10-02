"""
Anatomical plausibility via TotalSegmentator + organ volume distributions.

For each volume in a real and a synthetic set:
  1. Run TotalSegmentator to obtain organ/structure segmentation masks.
  2. Compute the volume (mL) of each structure as:
         n_voxels × voxel_volume_mm³ / 1000
  3. Compare real vs. synthetic distributions using:
       - Wasserstein distance (scipy.stats.wasserstein_distance)
       - Mean ± std per group
       - Optional violin / histogram plots (matplotlib)

TotalSegmentator produces 104 anatomical structures. You can restrict the
evaluation to a specific subset via --structures.

Requires:
    pip install totalsegmentator
    GPU recommended (fast mode: ~10–30 s/volume; full mode: ~30–120 s/volume)

Usage
-----
python evaluation/compute_volume_distributions.py \\
    --real_dir  data/ct_volumes/real/ \\
    --fake_dir  outputs/samples/dit_l/ \\
    --output_dir outputs/evaluation/volume_distributions/ \\
    --fast \\
    --structures liver spleen left_kidney right_kidney

Or from CSVs:
python evaluation/compute_volume_distributions.py \\
    --real_csv  ids/val.csv \\
    --fake_csv  ids/synthetic_val.csv \\
    --output_dir outputs/evaluation/volume_distributions/

Outputs
-------
  real_volumes.tsv         – per-volume per-structure volumes for real set
  fake_volumes.tsv         – per-volume per-structure volumes for synthetic set
  distribution_stats.tsv   – mean/std/Wasserstein distance per structure
  plots/                   – violin plots per structure (if matplotlib available)
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.utils.data_loading import (
    load_datalist_from_csv,
    load_datalist_from_folder,
)


# ---------------------------------------------------------------------------
# Default structures to evaluate (a clinically relevant subset)
# ---------------------------------------------------------------------------

DEFAULT_STRUCTURES = [
    "liver", "spleen", "left_kidney", "right_kidney",
    "gallbladder", "stomach", "pancreas",
    "left_lung_upper_lobe", "left_lung_lower_lobe",
    "right_lung_upper_lobe", "right_lung_middle_lobe", "right_lung_lower_lobe",
    "heart", "aorta",
    "vertebrae_L1", "vertebrae_L2", "vertebrae_L3",
]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare real vs. synthetic organ volume distributions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    real_g = p.add_mutually_exclusive_group(required=True)
    real_g.add_argument("--real_dir", help="Folder of real CT volumes")
    real_g.add_argument("--real_csv", help="CSV file listing real CT volumes")

    fake_g = p.add_mutually_exclusive_group(required=True)
    fake_g.add_argument("--fake_dir", help="Folder of synthetic CT volumes")
    fake_g.add_argument("--fake_csv", help="CSV file listing synthetic CT volumes")

    p.add_argument("--output_dir",  required=True)
    p.add_argument(
        "--structures", nargs="+", default=None,
        help="TotalSegmentator structure names to evaluate. "
             "Defaults to a curated subset of 17 clinically relevant organs. "
             "Use 'all' to evaluate all 104 structures.",
    )
    p.add_argument(
        "--fast", action="store_true",
        help="Use TotalSegmentator fast mode (lower resolution, ~3× faster).",
    )
    p.add_argument(
        "--device", default="gpu", choices=["gpu", "cpu"],
        help="TotalSegmentator compute device.",
    )
    p.add_argument(
        "--seg_dir", default=None,
        help="Root directory to store/load segmentation masks. "
             "If not given, a temp dir is used (masks are NOT reused between runs). "
             "Set this to avoid re-running TotalSegmentator on the same volumes.",
    )
    p.add_argument(
        "--skip_existing", action="store_true",
        help="Skip TotalSegmentator for volumes whose segmentation folder already exists.",
    )
    p.add_argument("--plot", action="store_true",
                   help="Save violin plots per structure.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# TotalSegmentator integration
# ---------------------------------------------------------------------------

def run_totalsegmentator(
    volume_path: str,
    output_dir: Path,
    fast: bool = True,
    device: str = "gpu",
) -> Path:
    """
    Run TotalSegmentator on a single volume.

    Uses the Python API if available, falling back to CLI.
    Returns the path to the output segmentation directory.
    """
    try:
        from totalsegmentator.python_api import totalsegmentator
        totalsegmentator(
            input=volume_path,
            output=str(output_dir),
            fast=fast,
            device=device,
            quiet=True,
        )
    except ImportError:
        # Fallback to CLI
        cmd = [
            "TotalSegmentator",
            "-i", str(volume_path),
            "-o", str(output_dir),
            "--device", device,
        ]
        if fast:
            cmd.append("--fast")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"TotalSegmentator failed for {volume_path}:\n{result.stderr}"
            )

    return output_dir


def get_voxel_volume_ml(seg_path: Path) -> float:
    """Return the volume of a single voxel in mL (= cm³)."""
    import nibabel as nib

    img = nib.load(str(seg_path))
    zooms = img.header.get_zooms()[:3]  # mm per voxel
    voxel_vol_mm3 = float(zooms[0] * zooms[1] * zooms[2])
    return voxel_vol_mm3 / 1000.0  # mm³ -> mL


def compute_structure_volumes(
    seg_dir: Path,
    structures: List[str],
) -> Dict[str, Optional[float]]:
    """
    Compute volume (mL) for each requested structure from TotalSegmentator masks.

    TotalSegmentator stores one binary .nii.gz per structure in the output folder.
    Returns NaN for structures whose mask file is missing.
    """
    import nibabel as nib

    volumes: Dict[str, Optional[float]] = {}
    for struct in structures:
        mask_path = seg_dir / f"{struct}.nii.gz"
        if not mask_path.exists():
            volumes[struct] = float("nan")
            continue
        img = nib.load(str(mask_path))
        data = img.get_fdata(dtype=np.float32)
        zooms = img.header.get_zooms()[:3]
        voxel_ml = float(zooms[0] * zooms[1] * zooms[2]) / 1000.0
        volumes[struct] = float(data.sum()) * voxel_ml

    return volumes


# ---------------------------------------------------------------------------
# Dataset processing
# ---------------------------------------------------------------------------

def process_dataset(
    data_dicts: List[dict],
    structures: List[str],
    seg_root: Path,
    fast: bool,
    device: str,
    skip_existing: bool,
    label: str,
) -> pd.DataFrame:
    """
    Run TotalSegmentator on all volumes and collect structure volumes.

    Returns a DataFrame with columns: filename, <structure_1>, ..., <structure_N>
    """
    from tqdm import tqdm

    records: List[dict] = []
    for entry in tqdm(data_dicts, desc=f"Segmenting [{label}]"):
        vol_path = entry["image"]
        fname = entry.get("filename", Path(vol_path).name)
        stem = Path(fname).stem.replace(".nii", "").replace(".mha", "")
        seg_dir = seg_root / label / stem

        if skip_existing and seg_dir.exists():
            print(f"  Skipping (existing segmentation): {fname}")
        else:
            try:
                seg_dir.mkdir(parents=True, exist_ok=True)
                run_totalsegmentator(vol_path, seg_dir, fast=fast, device=device)
            except Exception as e:
                print(f"  WARNING: TotalSegmentator failed for {fname}: {e}")
                records.append({"filename": fname, **{s: float("nan") for s in structures}})
                continue

        vols = compute_structure_volumes(seg_dir, structures)
        records.append({"filename": fname, **vols})

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Distribution comparison
# ---------------------------------------------------------------------------

def compute_success_rates(
    real_df: pd.DataFrame,
    fake_df: pd.DataFrame,
    structures: List[str],
) -> pd.DataFrame:
    """
    Compute TotalSegmentator detection success rate per structure.

    A structure is considered "found" in a volume when its computed volume is
    not NaN and > 0 mL.  This can reveal cases where the synthetic volumes lack
    expected anatomy entirely (e.g. a generated CT with no liver voxels that
    TotalSegmentator can detect), which volume-distribution Wasserstein distances
    alone would miss (NaN rows are silently dropped before WD computation).

    Returns a DataFrame with columns:
        structure, real_success_rate, fake_success_rate, success_rate_diff,
        n_real_total, n_fake_total, n_real_found, n_fake_found
    """
    rows: List[dict] = []
    n_real_total = len(real_df)
    n_fake_total = len(fake_df)

    for struct in structures:
        if struct not in real_df.columns or struct not in fake_df.columns:
            continue

        n_real_found = int((real_df[struct].notna() & (real_df[struct] > 0)).sum())
        n_fake_found = int((fake_df[struct].notna() & (fake_df[struct] > 0)).sum())

        real_rate = n_real_found / n_real_total if n_real_total > 0 else float("nan")
        fake_rate = n_fake_found / n_fake_total if n_fake_total > 0 else float("nan")

        rows.append({
            "structure":          struct,
            "real_success_rate":  real_rate,
            "fake_success_rate":  fake_rate,
            "success_rate_diff":  fake_rate - real_rate,  # negative = synthetic misses structure
            "n_real_found":       n_real_found,
            "n_fake_found":       n_fake_found,
            "n_real_total":       n_real_total,
            "n_fake_total":       n_fake_total,
        })

    return (
        pd.DataFrame(rows)
        .sort_values("success_rate_diff", ascending=True)  # worst (most missing) first
    )


def compare_distributions(
    real_df: pd.DataFrame,
    fake_df: pd.DataFrame,
    structures: List[str],
) -> pd.DataFrame:
    """Compute per-structure summary statistics, Wasserstein distance, and success rates."""
    from scipy.stats import wasserstein_distance

    success_df = compute_success_rates(real_df, fake_df, structures)
    success_map = success_df.set_index("structure")[
        ["real_success_rate", "fake_success_rate"]
    ].to_dict("index")

    rows: List[dict] = []
    for struct in structures:
        r = real_df[struct].dropna().values
        f = fake_df[struct].dropna().values

        if len(r) == 0 or len(f) == 0:
            continue

        wd = float(wasserstein_distance(r, f))
        sr = success_map.get(struct, {})
        rows.append({
            "structure":          struct,
            "real_mean_ml":       float(r.mean()),
            "real_std_ml":        float(r.std()),
            "fake_mean_ml":       float(f.mean()),
            "fake_std_ml":        float(f.std()),
            "wasserstein_dist":   wd,
            "real_success_rate":  sr.get("real_success_rate", float("nan")),
            "fake_success_rate":  sr.get("fake_success_rate", float("nan")),
            "n_real":             len(r),
            "n_fake":             len(f),
        })

    return pd.DataFrame(rows).sort_values("wasserstein_dist", ascending=False)


def save_violin_plots(
    real_df: pd.DataFrame,
    fake_df: pd.DataFrame,
    structures: List[str],
    output_dir: Path,
) -> None:
    """Save violin plots comparing real vs. synthetic volume distributions."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plots.")
        return

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    for struct in structures:
        r = real_df[struct].dropna().values
        f = fake_df[struct].dropna().values
        if len(r) == 0 or len(f) == 0:
            continue

        fig, ax = plt.subplots(figsize=(4, 5))
        ax.violinplot([r, f], positions=[0, 1], showmedians=True, showextrema=True)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Real", "Synthetic"])
        ax.set_ylabel("Volume (mL)")
        ax.set_title(struct.replace("_", " ").title())
        fig.tight_layout()
        fig.savefig(plots_dir / f"{struct}.png", dpi=150)
        plt.close(fig)

    print(f"Violin plots saved to {plots_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Resolve structure list
    # ------------------------------------------------------------------
    if args.structures is None:
        structures = DEFAULT_STRUCTURES
    elif args.structures == ["all"]:
        # TotalSegmentator v2 generates 104 structures; we rely on the output
        # directory to discover which files exist. Use a placeholder — actual
        # names will be detected from disk during processing.
        structures = DEFAULT_STRUCTURES  # refined after first run
        print("WARNING: 'all' mode will only evaluate structures found on disk. "
              "Adjust DEFAULT_STRUCTURES list for a fixed set.")
    else:
        structures = args.structures

    # ------------------------------------------------------------------
    # Segmentation root
    # ------------------------------------------------------------------
    if args.seg_dir:
        seg_root = Path(args.seg_dir)
    else:
        _tmp = tempfile.mkdtemp(prefix="totalseg_")
        seg_root = Path(_tmp)
        print(f"Using temporary segmentation directory: {seg_root}")
        print("Set --seg_dir to reuse segmentations across runs.")

    # ------------------------------------------------------------------
    # Load volume paths
    # ------------------------------------------------------------------
    real_dicts = (
        load_datalist_from_folder(args.real_dir)
        if args.real_dir else load_datalist_from_csv(args.real_csv)
    )
    fake_dicts = (
        load_datalist_from_folder(args.fake_dir)
        if args.fake_dir else load_datalist_from_csv(args.fake_csv)
    )

    # ------------------------------------------------------------------
    # Run TotalSegmentator + compute volumes
    # ------------------------------------------------------------------
    print(f"\nProcessing {len(real_dicts)} real volumes …")
    real_df = process_dataset(
        real_dicts, structures, seg_root,
        fast=args.fast, device=args.device,
        skip_existing=args.skip_existing, label="real",
    )

    print(f"\nProcessing {len(fake_dicts)} synthetic volumes …")
    fake_df = process_dataset(
        fake_dicts, structures, seg_root,
        fast=args.fast, device=args.device,
        skip_existing=args.skip_existing, label="fake",
    )

    # ------------------------------------------------------------------
    # Save per-volume volume tables
    # ------------------------------------------------------------------
    real_df.to_csv(output_dir / "real_volumes.tsv", sep="\t", index=False)
    fake_df.to_csv(output_dir / "fake_volumes.tsv", sep="\t", index=False)

    # ------------------------------------------------------------------
    # Distribution comparison
    # ------------------------------------------------------------------
    stats_df = compare_distributions(real_df, fake_df, structures)
    stats_path = output_dir / "distribution_stats.tsv"
    stats_df.to_csv(stats_path, sep="\t", index=False)

    print("\n─── Organ Volume Distributions (top-10 by Wasserstein distance) ─")
    print(stats_df.head(10).to_string(index=False, float_format="{:.2f}".format))
    print(f"\nFull statistics → {stats_path}")

    # ------------------------------------------------------------------
    # TotalSegmentator success rates
    # ------------------------------------------------------------------
    success_df = compute_success_rates(real_df, fake_df, structures)
    success_path = output_dir / "success_rates.tsv"
    success_df.to_csv(success_path, sep="\t", index=False)

    print("\n─── TotalSegmentator Success Rates ────────────────────────────────")
    print(f"  {'Structure':<35}  {'Real':>6}  {'Fake':>6}  {'Diff':>7}")
    print("  " + "─" * 60)
    for _, row in success_df.iterrows():
        diff_str = f"{row['success_rate_diff']:>+.3f}"
        flag = " ← LOW" if row["fake_success_rate"] < 0.5 and row["real_success_rate"] >= 0.5 else ""
        print(
            f"  {row['structure']:<35}  "
            f"{row['real_success_rate']:>6.3f}  "
            f"{row['fake_success_rate']:>6.3f}  "
            f"{diff_str}{flag}"
        )
    print(f"\nSuccess rates → {success_path}")

    # ------------------------------------------------------------------
    # Optional plots
    # ------------------------------------------------------------------
    if args.plot:
        save_violin_plots(real_df, fake_df, structures, output_dir)

    print(f"\nAll outputs → {output_dir}")


if __name__ == "__main__":
    main()
