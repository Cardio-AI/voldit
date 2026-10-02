"""
VQ-GAN reconstruction quality evaluation.

Metrics
-------
Pixel-level (computed on [0,1] range after denormalisation from [-1,1]):
  L1        Mean absolute error between reconstruction and input
  PSNR      Peak signal-to-noise ratio (dB)
  SSIM      Structural similarity index (3-D, then 2-D slice-wise fallback)
  MS-SSIM   Multi-scale SSIM (3-D, then 2-D axial fallback)
  NMSE      Normalised mean squared error = MSE / mean(target^2)

Codebook (accumulated across the full validation/test set):
  Utilisation   Fraction of codebook entries ever selected  [0,1]
  Entropy       Shannon entropy of the code-usage distribution  [bits]
  Perplexity    Effective codebook size = 2^entropy  [1 .. num_embeddings]

Usage
-----
python evaluation/evaluate_vqgan.py \\
    --config   configs/stage1/vqgan_ds8.yaml \\
    --checkpoint outputs/vqgan/best_model.pth \\
    --val_csv  ids/val.csv \\
    --output_dir outputs/evaluation/vqgan/ \\
    --roi_size 256 256 128 \\
    --device cuda

Outputs (written to output_dir):
  per_sample_metrics.tsv  – one row per volume: filename, l1, nmse, psnr, ssim, ms_ssim
  summary_metrics.tsv     – mean / std / median for pixel metrics + codebook stats
"""

import argparse
import math
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm

# Make project root importable regardless of working directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.vqvae import VQVAE
from evaluation.utils.data_loading import get_eval_dataloader, load_datalist_from_csv

# ---------------------------------------------------------------------------
# MONAI metrics (optional imports — we fall back gracefully if unavailable)
# ---------------------------------------------------------------------------
try:
    from monai.metrics import MultiScaleSSIMMetric, SSIMMetric
    _HAS_MONAI_METRICS = True
except ImportError:
    _HAS_MONAI_METRICS = False


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate VQ-GAN reconstruction quality on a validation/test set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config",      required=True, help="Path to YAML model config")
    p.add_argument("--checkpoint",  required=True, help="Path to .pth checkpoint")
    p.add_argument("--val_csv",     required=True, help="CSV with validation image paths")
    p.add_argument("--output_dir",  required=True)
    p.add_argument("--roi_size",    nargs=3, type=int, default=[256, 256, 128],
                   metavar="N", help="Spatial crop size (D H W)")
    p.add_argument("--batch_size",  type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device",      default="cuda")
    p.add_argument("--no_resample", action="store_true",
                   help="Skip resampling to isotropic spacing")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(config_path: str, ckpt_path: str, device: torch.device) -> VQVAE:
    """Load VQVAE from a YAML config and a checkpoint file."""
    cfg = OmegaConf.load(config_path)
    from src.config_utils import get_stage1_params
    model = VQVAE(**get_stage1_params(cfg)).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    # Support several common checkpoint layouts
    state = ckpt.get("state_dict", ckpt.get("model", ckpt))
    model.load_state_dict(state)
    model.eval()
    print(
        f"Loaded VQVAE from {ckpt_path}  "
        f"(num_embeddings={model.quantizer.num_embeddings}, "
        f"embedding_dim={model.quantizer.quantizer.embedding_dim})"
    )
    return model


# ---------------------------------------------------------------------------
# Pixel-level metrics  (inputs expected in [0, 1])
# ---------------------------------------------------------------------------

def compute_l1(pred: torch.Tensor, target: torch.Tensor) -> float:
    return F.l1_loss(pred, target).item()


def compute_nmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Normalised MSE: MSE / E[target^2].  Scale-invariant quality measure."""
    mse = F.mse_loss(pred, target).item()
    norm = target.pow(2).mean().item()
    return mse / (norm + 1e-8)


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    mse = F.mse_loss(pred, target).item()
    if mse < 1e-12:
        return float("inf")
    return 20.0 * math.log10(data_range / math.sqrt(mse))


def _ssim_3d(pred: torch.Tensor, target: torch.Tensor) -> float:
    """3-D SSIM using MONAI, or 2-D slice-wise average as fallback."""
    if _HAS_MONAI_METRICS:
        try:
            metric = SSIMMetric(spatial_dims=3, data_range=1.0, kernel_type="gaussian")
            return metric(pred, target).mean().item()
        except Exception:
            pass
    # Fallback: average over axial slices
    metric2d = SSIMMetric(spatial_dims=2, data_range=1.0, kernel_type="gaussian")
    values = [
        metric2d(pred[:, :, i], target[:, :, i]).mean().item()
        for i in range(pred.shape[2])
    ]
    return float(np.mean(values))


def _msssim_3d(pred: torch.Tensor, target: torch.Tensor) -> float:
    """3-D MS-SSIM using MONAI, or 2-D axial-slice average as fallback."""
    if _HAS_MONAI_METRICS:
        try:
            metric = MultiScaleSSIMMetric(
                spatial_dims=3, data_range=1.0, kernel_type="gaussian"
            )
            return metric(pred, target).mean().item()
        except Exception:
            pass
    # Fallback: 2-D MS-SSIM over axial slices
    # kernel_size must be <= slice H and W; use 7 as a safe default
    metric2d = MultiScaleSSIMMetric(
        spatial_dims=2, data_range=1.0, kernel_type="gaussian", kernel_size=7
    )
    values = []
    for i in range(pred.shape[2]):
        try:
            values.append(metric2d(pred[:, :, i], target[:, :, i]).mean().item())
        except Exception:
            pass
    return float(np.mean(values)) if values else float("nan")


# ---------------------------------------------------------------------------
# Codebook statistics
# ---------------------------------------------------------------------------

def compute_codebook_stats(
    all_indices: torch.Tensor, num_embeddings: int
) -> dict:
    """
    Compute codebook utilisation, entropy, and perplexity from a collection
    of quantizer index tensors gathered over the validation set.

    Args:
        all_indices:    1-D (or any-shape) integer tensor of code indices.
        num_embeddings: Total size of the codebook.

    Returns dict with:
        codebook_utilisation   Fraction of codes used at least once
        codebook_entropy_bits  Shannon entropy of usage distribution [bits]
        codebook_perplexity    2^entropy  (effective codebook size)
        num_used_codes
        total_codes
    """
    flat = all_indices.view(-1)
    counts = torch.bincount(flat, minlength=num_embeddings).float()
    probs = counts / (counts.sum() + 1e-10)

    utilisation = (counts > 0).float().mean().item()
    entropy_bits = float(-torch.sum(probs * torch.log2(probs + 1e-10)))
    perplexity = 2.0 ** entropy_bits

    return {
        "codebook_utilisation":  utilisation,
        "codebook_entropy_bits": entropy_bits,
        "codebook_perplexity":   perplexity,
        "num_used_codes":        int((counts > 0).sum()),
        "total_codes":           num_embeddings,
    }


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.config, args.checkpoint, device)
    num_embeddings = model.quantizer.num_embeddings

    data_dicts = load_datalist_from_csv(args.val_csv)
    loader = get_eval_dataloader(
        data_dicts,
        roi_size=tuple(args.roi_size),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        resample=not args.no_resample,
    )

    per_sample_records: List[dict] = []
    all_indices_list: List[torch.Tensor] = []

    for batch in tqdm(loader, desc="Evaluating VQ-GAN"):
        x = batch["image"].as_tensor().to(device)  # (B, 1, D, H, W) in [-1, 1]

        fnames = batch.get("filename", [])
        if isinstance(fnames, (str, torch.Tensor)):
            fnames = [fnames] if isinstance(fnames, str) else fnames.tolist()

        with torch.no_grad(), torch.amp.autocast("cuda"):
            recon, _, indices = model(x)

        all_indices_list.append(indices.cpu())

        # Denormalise [-1,1] -> [0,1] for metric computation
        x01    = ((x.clamp(-1, 1) + 1.0) / 2.0).float().cpu()
        recon01 = ((recon.clamp(-1, 1) + 1.0) / 2.0).float().cpu()

        for b in range(x.shape[0]):
            xb = x01[b : b + 1]      # (1, 1, D, H, W)
            rb = recon01[b : b + 1]
            fname = fnames[b] if b < len(fnames) else f"sample_{len(per_sample_records)}"

            per_sample_records.append(
                {
                    "filename": fname,
                    "l1":       compute_l1(rb, xb),
                    "nmse":     compute_nmse(rb, xb),
                    "psnr":     compute_psnr(rb, xb),
                    "ssim":     _ssim_3d(rb, xb),
                    "ms_ssim":  _msssim_3d(rb, xb),
                }
            )

        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Save per-sample results
    # ------------------------------------------------------------------
    df = pd.DataFrame(per_sample_records)
    per_sample_path = output_dir / "per_sample_metrics.tsv"
    df.to_csv(per_sample_path, sep="\t", index=False)
    print(f"\nPer-sample metrics → {per_sample_path}")

    # ------------------------------------------------------------------
    # Pixel-metric summary
    # ------------------------------------------------------------------
    pixel_cols = ["l1", "nmse", "psnr", "ssim", "ms_ssim"]
    print("\n─── Reconstruction Metrics (mean ± std)  ───────────────────────")
    summary_rows: List[dict] = []
    for col in pixel_cols:
        m, s, med = df[col].mean(), df[col].std(), df[col].median()
        print(f"  {col:<10}  {m:.4f} ± {s:.4f}   (median {med:.4f})")
        for stat, val in [("mean", m), ("std", s), ("median", med)]:
            summary_rows.append({"metric": col, "stat": stat, "value": val})

    # ------------------------------------------------------------------
    # Codebook statistics
    # ------------------------------------------------------------------
    all_indices = torch.cat(all_indices_list, dim=0)
    cb_stats = compute_codebook_stats(all_indices, num_embeddings)

    print("\n─── Codebook Statistics  ────────────────────────────────────────")
    for k, v in cb_stats.items():
        formatted = f"{v:.4f}" if isinstance(v, float) else str(v)
        print(f"  {k:<30}  {formatted}")
        summary_rows.append({"metric": k, "stat": "value", "value": v})

    # ------------------------------------------------------------------
    # Save summary
    # ------------------------------------------------------------------
    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "summary_metrics.tsv"
    summary_df.to_csv(summary_path, sep="\t", index=False)
    print(f"\nSummary → {summary_path}")


if __name__ == "__main__":
    main()
