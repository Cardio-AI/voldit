"""
Compute the latent scale factor (1 / std(z)) from precomputed latent files.

Usage:
    python src/scripts/compute_scale_factor.py \
        --latents_csv data/latents/train/latents.csv \
        --limit 200
"""

import argparse
import sys
from pathlib import Path

import torch
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latents_csv", required=True, help="CSV with precomputed latent paths (column: image)")
    parser.add_argument("--limit", type=int, default=None, help="Max number of latents to use (default: all)")
    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.latents_csv)
    paths = df["image"].tolist()

    if args.limit:
        paths = paths[: args.limit]

    print(f"Computing scale factor over {len(paths)} latents...")

    all_latents = []
    for p in paths:
        z = torch.load(p, map_location="cpu", weights_only=False)
        all_latents.append(z.flatten())

    all_latents = torch.cat(all_latents)
    std = all_latents.std().item()
    scale_factor = 1.0 / std

    print(f"\n  std(z)       = {std:.6f}")
    print(f"  scale_factor = {scale_factor:.6f}  (1 / std)")
    print(f"\nAdd to your diffusion config:")
    print(f"  training:")
    print(f"    scale_factor: {scale_factor:.6f}")


if __name__ == "__main__":
    main()
