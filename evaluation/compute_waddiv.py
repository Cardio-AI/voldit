"""
WAD-DIV (Wasserstein Distance Diversity) for generative model evaluation.

WAD-Div measures the diversity of a synthetic image set in feature space by
computing the Wasserstein distance between the k-NN distance distribution of
the synthetic set and a reference distribution (zero, exponential, or empirical).

  - wad_div        Raw WAD-Div score (higher = more diverse)
  - wad_div_norm   Normalised to [0,1] against real data diversity (optional)

Reference: WAD-DIV implementation preserved in evaluation/utils/waddiv.py

Three reference modes
---------------------
  zero        Compare to Dirac at 0 (complete mode collapse).
              Useful as an absolute diversity measure without real data.
  exponential Compare to a fitted exponential distribution (soft lower bound).
  empirical   Compare directly against the real data kNN distribution.
              Most informative — requires --real_features.

Usage
-----
# Raw WAD-Div (zero reference)
python evaluation/compute_waddiv.py \\
    --fake_features outputs/evaluation/features/fake_3d.npz \\
    --output_dir    outputs/evaluation/waddiv/

# Normalised WAD-Div (empirical reference, normalised against real)
python evaluation/compute_waddiv.py \\
    --fake_features outputs/evaluation/features/fake_3d.npz \\
    --real_features outputs/evaluation/features/real_3d.npz \\
    --reference empirical --normalize \\
    --output_dir outputs/evaluation/waddiv/

Outputs
-------
  waddiv_results.tsv  – columns: reference, k, wad_div, wad_div_norm (if normalised)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from evaluation.utils.feature_extraction import load_features


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute WAD-DIV diversity from pre-extracted feature files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--fake_features", required=True,
                   help=".npz feature file for synthetic volumes")
    p.add_argument("--real_features", default=None,
                   help=".npz feature file for real volumes (required for "
                        "empirical reference or normalisation)")
    p.add_argument("--output_dir", required=True)
    p.add_argument(
        "--reference", nargs="+",
        default=["zero", "exponential", "empirical"],
        choices=["zero", "exponential", "empirical"],
        help="Reference distribution(s) to use. 'empirical' requires --real_features.",
    )
    p.add_argument(
        "--k", nargs="+", type=int, default=[3, 5, 10],
        help="k values for k-NN distance computation.",
    )
    p.add_argument(
        "--normalize", action="store_true",
        help="Compute normalised WAD-Div score in [0,1]. "
             "Requires --real_features to fit the max-diversity anchor.",
    )
    p.add_argument("--distance_metric", default="euclidean",
                   choices=["euclidean", "cosine"])
    p.add_argument("--ref_percentile", type=float, default=95.0,
                   help="Percentile for exponential reference fitting.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _load_flat(npz_path: str) -> np.ndarray:
    """Load features, averaging over planes for 2.5-D files."""
    feat_3d, feat_2p5d, _ = load_features(npz_path)
    if feat_3d is not None:
        return feat_3d.numpy()
    if feat_2p5d is not None:
        return np.concatenate([v.numpy() for v in feat_2p5d.values()], axis=0)
    raise ValueError(f"Unrecognised feature format: {npz_path}")


def main() -> None:
    try:
        from evaluation.utils.waddiv import WADDiv
    except ImportError:
        raise ImportError(
            "WADDiv not found.  Make sure evaluation/utils/waddiv.py exists."
        )

    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading synthetic features from {args.fake_features}")
    fake = _load_flat(args.fake_features)

    real = None
    if args.real_features:
        print(f"Loading real features from      {args.real_features}")
        real = _load_flat(args.real_features)

    if "empirical" in args.reference and real is None:
        raise ValueError("--real_features is required for empirical reference.")
    if args.normalize and real is None:
        raise ValueError("--real_features is required for --normalize.")

    metric = WADDiv(distance_metric=args.distance_metric)

    rows: list = []
    print(f"\nSynthetic: {fake.shape}{'  Real: ' + str(real.shape) if real is not None else ''}")
    print("\n─── WAD-DIV Results ────────────────────────────────────────────")

    for ref in args.reference:
        if ref == "empirical" and real is None:
            continue

        for k in sorted(args.k):
            # Fit normalisation anchor on real data (once per k)
            if args.normalize:
                metric.fit_max_reference(
                    features=real,
                    k=k,
                    reference="zero",
                    ref_percentile=args.ref_percentile,
                    random_state=args.seed,
                )

            result = metric.compute(
                features=fake,
                k=k,
                reference=ref,
                ref_features=real if ref == "empirical" else None,
                ref_percentile=args.ref_percentile,
                normalize=args.normalize,
                random_state=args.seed,
            )

            wad = result["wad_div"]
            wad_norm = result.get("wad_div_norm")

            norm_str = f"  norm={wad_norm:.4f}" if wad_norm is not None else ""
            print(f"  ref={ref:<12}  k={k:>3}  wad_div={wad:.6f}{norm_str}")

            rows.append({
                "reference":    ref,
                "k":            k,
                "wad_div":      wad,
                "wad_div_norm": wad_norm,
            })

    df = pd.DataFrame(rows)
    out_path = output_dir / "waddiv_results.tsv"
    df.to_csv(out_path, sep="\t", index=False)
    print(f"\nResults → {out_path}")


if __name__ == "__main__":
    main()
