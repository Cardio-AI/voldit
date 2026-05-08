"""
Conditional sampling from a trained ControlNet3D + DiT3D.

A CSV file provides the mask paths (one row per subject). The same spatial
transforms used during encoding are applied, then the masks are passed as
conditioning to ControlNet3D at every denoising step. One output volume is
generated per CSV row.

Note: ControlNet3D wraps the frozen DiT base model and produces the final
noise prediction directly — no separate diffusion model is needed at inference.
"""

import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd
import torch
import numpy as np
import nibabel as nib
from omegaconf import OmegaConf
from torch import amp
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    CenterSpatialCropd,
    SpatialPadd,
    ToTensord,
)

from src.models.vqvae import VQVAE
from src.models.dit import DiT3D
from src.models.controlnet_dit import ControlNet3D
from src.models.ddimscheduler import DDIMScheduler
from src.models.ddpmscheduler import DDPMScheduler


def parse_args():
    parser = argparse.ArgumentParser(description="Conditional sampling from a trained ControlNet3D + DiT3D")

    parser.add_argument("--stage1_ckpt", type=str, required=True)
    parser.add_argument("--stage1_cfg", type=str, required=True)
    parser.add_argument("--dit_ckpt", type=str, required=True,
                        help="Path to pretrained DiT3D checkpoint (used to build base model)")
    parser.add_argument("--controlnet_ckpt", type=str, required=True)
    parser.add_argument("--controlnet_cfg", type=str, required=True)

    parser.add_argument("--csv", type=str, required=True,
                        help="CSV file with mask paths, one row per subject. "
                             "Columns must include all condition keys. "
                             "An optional 'image' column is used for output file naming.")
    parser.add_argument("--condition_keys", nargs="+", required=True,
                        help="Column names in the CSV corresponding to the mask conditions")

    parser.add_argument("--output_dir", type=str, default="samples_cond")
    parser.add_argument("--timesteps", type=int, default=300)
    parser.add_argument("--scheduler", type=str, default="ddpm", choices=["ddpm", "ddim"])
    parser.add_argument("--scale_factor", type=float, default=1.0)
    parser.add_argument("--latent_shape", type=int, nargs=3, required=True,
                        metavar=("D", "H", "W"), help="Latent spatial dimensions, must match model input_size e.g. 32 32 32")
    parser.add_argument("--roi_size", type=int, nargs=3, default=[512, 512, 256],
                        metavar=("H", "W", "D"), help="Spatial crop/pad size matching training")
    parser.add_argument("--reference_nii", type=str, default=None,
                        help="Reference .nii.gz to copy affine from")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def load_stage1(cfg_path, ckpt_path, device):
    cfg = OmegaConf.load(cfg_path)
    model = VQVAE(**cfg.model.params)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))
    model_keys = set(model.state_dict().keys())
    filtered = {k: v for k, v in state_dict.items() if k in model_keys}
    skipped = [k for k in state_dict if k not in model_keys]
    if skipped:
        print(f"Warning: skipped {len(skipped)} stage1 keys: {skipped[:5]}{'...' if len(skipped) > 5 else ''}")
    model.load_state_dict(filtered, strict=False)

    return model.to(device).eval().requires_grad_(False)


def load_controlnet(cfg_path, dit_ckpt_path, controlnet_ckpt_path, device):
    cfg = OmegaConf.load(cfg_path)

    # Build base DiT and load its weights
    base_dit = DiT3D(**cfg.pretrained_model.params)
    dit_ckpt = torch.load(dit_ckpt_path, map_location="cpu", weights_only=True)
    # Merge full model state (has pos_embed) with EMA shadow (has trained weights)
    dit_state = dict(dit_ckpt.get("model", dit_ckpt))
    ema_state = dit_ckpt.get("ema")
    if ema_state is not None:
        dit_state.update(ema_state["shadow"])
    base_dit.load_state_dict(dit_state)

    # Build ControlNet3D wrapping the base DiT
    model = ControlNet3D(base_model=base_dit, **cfg.controlnet.params)

    # Load ControlNet weights (EMA preferred)
    cn_ckpt = torch.load(controlnet_ckpt_path, map_location="cpu", weights_only=True)
    ema_state = cn_ckpt.get("ema")
    if ema_state is not None:
        shadow = ema_state["shadow"]
        for name, param in model.named_parameters():
            if name in shadow:
                param.data.copy_(shadow[name])
        print("Loaded ControlNet3D EMA weights.")
    else:
        state_dict = cn_ckpt.get("model", cn_ckpt)
        model.load_state_dict(state_dict)
        print("Loaded ControlNet3D model weights (no EMA found).")

    return model.to(device).eval().requires_grad_(False), cfg


def load_masks(mask_paths_by_key, roi_size, device):
    present_keys = [k for k, p in mask_paths_by_key.items() if p is not None]
    missing_keys = [k for k, p in mask_paths_by_key.items() if p is None]

    if missing_keys:
        print(f"  Warning: missing masks for {missing_keys}, substituting zeros.")

    masks_out = {}

    if present_keys:
        transforms = Compose([
            LoadImaged(keys=present_keys),
            EnsureChannelFirstd(keys=present_keys),
            CenterSpatialCropd(keys=present_keys, roi_size=roi_size),
            SpatialPadd(keys=present_keys, spatial_size=roi_size),
            ToTensord(keys=present_keys),
        ])
        data = transforms({k: mask_paths_by_key[k] for k in present_keys})
        for k in present_keys:
            masks_out[k] = data[k]

    spatial_shape = next(iter(masks_out.values())).shape[1:] if masks_out else tuple(roi_size)
    for k in missing_keys:
        masks_out[k] = torch.zeros(1, *spatial_shape)

    ordered = [masks_out[k] for k in mask_paths_by_key]
    cond = torch.cat(ordered, dim=0).unsqueeze(0)  # [1, n_masks, H, W, D]

    return cond.to(device)


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    affine = None
    if args.reference_nii is not None:
        affine = nib.load(args.reference_nii).affine
        print(f"Using affine from {args.reference_nii}")

    print("Loading VQ-GAN...")
    stage1 = load_stage1(args.stage1_cfg, args.stage1_ckpt, device)

    print("Loading ControlNet3D (with frozen DiT base)...")
    controlnet, controlnet_cfg = load_controlnet(
        args.controlnet_cfg, args.dit_ckpt, args.controlnet_ckpt, device
    )

    # Scheduler
    if args.scheduler == "ddpm":
        scheduler = DDPMScheduler(**controlnet_cfg.ldm.scheduler)
    else:
        scheduler = DDIMScheduler(**controlnet_cfg.ldm.scheduler)
    scheduler.set_timesteps(args.timesteps)

    scale_factor = args.scale_factor
    latent_shape = tuple(args.latent_shape)
    in_channels = controlnet_cfg.pretrained_model.params.in_channels

    df = pd.read_csv(args.csv)
    print(f"Found {len(df)} subject(s) in {args.csv}")
    print(f"Sampling with {args.scheduler.upper()} ({args.timesteps} steps)...")

    for idx, row in df.iterrows():
        mask_paths_by_key = {
            key: str(row[key]) if pd.notna(row[key]) else None
            for key in args.condition_keys
        }

        cond = load_masks(mask_paths_by_key, args.roi_size, device)

        x = torch.randn((1, in_channels, *latent_shape), device=device)

        with torch.no_grad(), amp.autocast(device_type=device.type):
            for t in scheduler.timesteps:
                t_batch = torch.tensor([t], device=device)

                noise_pred = controlnet(
                    x,
                    t=t_batch,
                    y=None,
                    control_input=cond,
                )

                x, _ = scheduler.step(noise_pred, t, x)

            x = x / scale_factor
            recon = stage1.decode_stage_2_outputs(x)

        vol = np.clip(recon[0, 0].float().cpu().numpy(), -1.0, 1.0)
        hu = (((vol + 1.0) * (2000.0 / 2.0)) - 1000.0).astype(np.int16)

        stem = Path(row["image"]).stem if "image" in row else f"{idx:03d}"
        out_path = out_dir / f"{stem}_cond.nii.gz"
        nib.save(nib.Nifti1Image(hu, affine), out_path)
        print(f"  [{idx+1}/{len(df)}] Saved {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
