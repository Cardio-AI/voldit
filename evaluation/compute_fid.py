"""
FID and KID computation for unconditional generation evaluation.

Computes:
  FID   Fréchet Inception Distance  — measures quality + diversity together
  KID   Kernel Inception Distance   — unbiased FID variant, better for small N

Both metrics operate on feature vectors pre-extracted by extract_features.py.

Modes
-----
  3-D features (MedicalNet)
    • Single FID/KID score over the full volume embedding.

  2.5-D features (RadImageNet)
    • Per-plane FID/KID (xy, yz, zx).
    • Average across planes reported as the headline number.

Usage
-----
# 3-D FID
python evaluation/compute_fid.py \\
    --real_features  outputs/evaluation/features/real_3d.npz \\
    --fake_features  outputs/evaluation/features/fake_3d.npz \\
    --output_dir     outputs/evaluation/fid/

# 2.5-D FID
python evaluation/compute_fid.py \\
    --real_features  outputs/evaluation/features/real_2p5d.npz \\
    --fake_features  outputs/evaluation/features/fake_2p5d.npz \\
    --output_dir     outputs/evaluation/fid/

Outputs
-------
  fid_results.tsv   – columns: metric, plane, value
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import linalg

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.utils.feature_extraction import load_features


# ---------------------------------------------------------------------------
# FID
# ---------------------------------------------------------------------------

def compute_fid(
    real: np.ndarray,
    fake: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """
    Compute Fréchet Inception Distance.

    FID = ||μ_r - μ_f||² + Tr(Σ_r + Σ_f - 2·sqrt(Σ_r·Σ_f))

    Args:
        real: (N, D) real feature matrix
        fake: (M, D) synthetic feature matrix
        eps:  small value added to diagonal of covariance for numerical stability

    Returns:
        FID score (lower is better)
    """
    mu_r, sigma_r = real.mean(0), np.cov(real, rowvar=False)
    mu_f, sigma_f = fake.mean(0), np.cov(fake, rowvar=False)

    # Add epsilon to diagonal for numerical stability
    sigma_r += np.eye(sigma_r.shape[0]) * eps
    sigma_f += np.eye(sigma_f.shape[0]) * eps

    diff = mu_r - mu_f
    covmean, _ = linalg.sqrtm(sigma_r @ sigma_f, disp=False)

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-2):
            raise RuntimeError(
                f"FID: large imaginary component in sqrt ({np.max(np.abs(covmean.imag)):.4f}). "
                "Covariance matrix may be ill-conditioned (too few samples?)."
            )
        covmean = covmean.real

    fid = float(diff @ diff + np.trace(sigma_r + sigma_f - 2.0 * covmean))
    return fid


# ---------------------------------------------------------------------------
# KID
# ---------------------------------------------------------------------------

def compute_kid(
    real: np.ndarray,
    fake: np.ndarray,
    num_subsets: int = 100,
    subset_size: Optional[int] = None,
    degree: int = 3,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    """
    Compute Kernel Inception Distance using a polynomial kernel.

    KID is an unbiased estimator of the MMD² between two feature distributions.
    Unlike FID it does not require Gaussian assumptions and is unbiased with
    any sample size — preferred when N < 1000.

    k(x, y) = (x·y / d + 1)^degree   (default: degree=3)

    Args:
        real:        (N, D) real feature matrix
        fake:        (M, D) synthetic feature matrix
        num_subsets: Number of random subset evaluations to average over
        subset_size: Size of each subset; defaults to min(N, M, 1000)
        degree:      Polynomial kernel degree
        rng:         Random number generator for reproducibility

    Returns:
        (kid_mean, kid_std)  – KID score and its standard deviation
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_r, n_f = len(real), len(fake)
    m = subset_size or min(n_r, n_f, 1000)
    d = real.shape[1]

    kid_values: list = []
    for _ in range(num_subsets):
        r = real[rng.choice(n_r, m, replace=False)]
        f = fake[rng.choice(n_f, m, replace=False)]

        k_rr = ((r @ r.T) / d + 1.0) ** degree
        k_ff = ((f @ f.T) / d + 1.0) ** degree
        k_rf = ((r @ f.T) / d + 1.0) ** degree

        kid = (
            (k_rr.sum() - np.trace(k_rr)) / (m * (m - 1))
            + (k_ff.sum() - np.trace(k_ff)) / (m * (m - 1))
            - 2.0 * k_rf.mean()
        )
        kid_values.append(float(kid))

    return float(np.mean(kid_values)), float(np.std(kid_values))


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute FID and KID from pre-extracted feature files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--real_features", required=True,
                   help=".npz file with real features (from extract_features.py)")
    p.add_argument("--fake_features", required=True,
                   help=".npz file with synthetic features (from extract_features.py)")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--kid_subsets",     type=int, default=100,
                   help="Number of random subsets for KID estimation")
    p.add_argument("--kid_subset_size", type=int, default=None,
                   help="Subset size for KID; defaults to min(N_real, N_fake, 1000)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    real_3d, real_2p5d, _ = load_features(args.real_features)
    fake_3d, fake_2p5d, _ = load_features(args.fake_features)

    rows: list = []

    def _eval_pair(real_np: np.ndarray, fake_np: np.ndarray, plane: str) -> None:
        print(
            f"\n  [{plane}]  real={real_np.shape}  fake={fake_np.shape}"
        )
        fid_val = compute_fid(real_np, fake_np)
        kid_mean, kid_std = compute_kid(
            real_np, fake_np,
            num_subsets=args.kid_subsets,
            subset_size=args.kid_subset_size,
            rng=rng,
        )
        print(f"    FID = {fid_val:.4f}")
        print(f"    KID = {kid_mean:.6f} ± {kid_std:.6f}")
        rows.append({"plane": plane, "metric": "FID",      "value": fid_val})
        rows.append({"plane": plane, "metric": "KID_mean", "value": kid_mean})
        rows.append({"plane": plane, "metric": "KID_std",  "value": kid_std})

    # ------------------------------------------------------------------
    # 3-D features
    # ------------------------------------------------------------------
    if real_3d is not None and fake_3d is not None:
        print("═══ 3-D Features ═══════════════════════════════════════════")
        _eval_pair(real_3d.numpy(), fake_3d.numpy(), plane="3d")

    # ------------------------------------------------------------------
    # 2.5-D features
    # ------------------------------------------------------------------
    if real_2p5d is not None and fake_2p5d is not None:
        print("═══ 2.5-D Features ═════════════════════════════════════════")
        shared_planes = set(real_2p5d.keys()) & set(fake_2p5d.keys())
        for plane in sorted(shared_planes):
            _eval_pair(
                real_2p5d[plane].numpy(),
                fake_2p5d[plane].numpy(),
                plane=plane,
            )
        # Average FID / KID across planes
        plane_fids = [r["value"] for r in rows if r["metric"] == "FID" and r["plane"] != "3d"]
        plane_kid_means = [r["value"] for r in rows if r["metric"] == "KID_mean" and r["plane"] != "3d"]
        if plane_fids:
            avg_fid = float(np.mean(plane_fids))
            avg_kid = float(np.mean(plane_kid_means))
            print(f"\n  [avg]  FID = {avg_fid:.4f}   KID_mean = {avg_kid:.6f}")
            rows.append({"plane": "avg_2p5d", "metric": "FID",      "value": avg_fid})
            rows.append({"plane": "avg_2p5d", "metric": "KID_mean", "value": avg_kid})

    if not rows:
        raise RuntimeError(
            "No matching feature types found in the two files. "
            "Both must be 3-D or both must be 2.5-D."
        )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    df = pd.DataFrame(rows)
    out_path = output_dir / "fid_results.tsv"
    df.to_csv(out_path, sep="\t", index=False)
    print(f"\nResults → {out_path}")


if __name__ == "__main__":
    main()
