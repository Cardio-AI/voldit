"""
Precision, Recall, Density, Coverage (PRDC) for unconditional generation.

These four metrics (Naeem et al., NeurIPS 2020) use k-nearest-neighbour
distances in feature space to decompose image quality (precision / density)
and diversity (recall / coverage):

  Precision   P(synthetic sample is within real data manifold)
  Recall      P(real sample is covered by synthetic manifold)
  Density     Average number of synthetic neighbours per real sphere
  Coverage    Fraction of real samples covered by at least one synthetic

Higher is better for all four metrics.

Requires: pip install prdc

Works on 3-D features (MedicalNet) or averaged 2.5-D features (RadImageNet).
For 2.5-D files the per-plane features are averaged into a single vector
before computing PRDC, giving one set of metrics comparable to the 3-D case.

Usage
-----
python evaluation/compute_prdc.py \\
    --real_features outputs/evaluation/features/real_3d.npz \\
    --fake_features outputs/evaluation/features/fake_3d.npz \\
    --output_dir    outputs/evaluation/prdc/ \\
    --k 5

Outputs
-------
  prdc_results.tsv  – columns: k, metric, value
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.utils.feature_extraction import load_features


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute PRDC metrics from pre-extracted feature files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--real_features", required=True,
                   help=".npz feature file for real volumes")
    p.add_argument("--fake_features", required=True,
                   help=".npz feature file for synthetic volumes")
    p.add_argument("--output_dir", required=True)
    p.add_argument(
        "--k", nargs="+", type=int, default=[3, 5, 10],
        help="Nearest-neighbour k values to evaluate. Multiple values are supported.",
    )
    return p.parse_args()


def load_flat_features(npz_path: str) -> np.ndarray:
    """
    Load features from a .npz file, handling both 3-D and 2.5-D formats.

    For 2.5-D files the features from all planes are averaged into one vector,
    keeping a single consistent representation per volume.
    """
    feat_3d, feat_2p5d, _ = load_features(npz_path)

    if feat_3d is not None:
        return feat_3d.numpy()

    if feat_2p5d is not None:
        # Average across planes: each plane is (N, D); stack -> (num_planes, N, D)
        stacked = np.stack([v.numpy() for v in feat_2p5d.values()], axis=0)
        return stacked.mean(axis=0)  # (N, D)

    raise ValueError(f"Could not read features from {npz_path}")


def main() -> None:
    try:
        from prdc import compute_prdc
    except ImportError:
        raise ImportError(
            "The 'prdc' package is required.  Install it with:\n"
            "    pip install prdc"
        )

    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading real features from  {args.real_features}")
    real = load_flat_features(args.real_features)
    print(f"Loading fake features from  {args.fake_features}")
    fake = load_flat_features(args.fake_features)

    print(f"\nReal: {real.shape}   Fake: {fake.shape}")

    rows: list = []
    print("\n─── PRDC Metrics ───────────────────────────────────────────────")
    print(f"  {'k':>4}   {'Precision':>10}  {'Recall':>10}  {'Density':>10}  {'Coverage':>10}")
    print("  " + "─" * 52)

    for k in sorted(args.k):
        result = compute_prdc(
            real_features=real,
            fake_features=fake,
            nearest_k=k,
        )
        print(
            f"  {k:>4}   "
            f"{result['precision']:>10.4f}  "
            f"{result['recall']:>10.4f}  "
            f"{result['density']:>10.4f}  "
            f"{result['coverage']:>10.4f}"
        )
        for metric, value in result.items():
            rows.append({"k": k, "metric": metric, "value": value})

    df = pd.DataFrame(rows)
    out_path = output_dir / "prdc_results.tsv"
    df.to_csv(out_path, sep="\t", index=False)
    print(f"\nResults → {out_path}")


if __name__ == "__main__":
    main()
