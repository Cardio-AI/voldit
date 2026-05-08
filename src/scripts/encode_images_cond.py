# encode_controlnet_latents.py
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import torch
import pandas as pd
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    ToTensord,
    ThresholdIntensityd,
    CenterSpatialCropd,
    SpatialPadd,
)
from omegaconf import OmegaConf


def parse_args():
    parser = argparse.ArgumentParser(description="Encode images + masks into VQVAE latents for ControlNet")

    parser.add_argument("--csv", required=True, help="CSV containing image + mask paths")
    parser.add_argument("--output_dir", required=True, help="Directory to store latents and masks")
    parser.add_argument("--vqvae_ckpt", required=True, help="Path to VQVAE checkpoint")
    parser.add_argument("--config", required=True, help="Path to VQVAE config yaml")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of subjects")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--condition_keys", nargs="+", required=True, help="Column names of mask conditions")
    return parser.parse_args()


def load_model(config_path, ckpt_path, device):
    from src.models.vqvae import VQVAE

    config = OmegaConf.load(config_path)
    model = VQVAE(**config.model.params)

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    model_keys = set(model.state_dict().keys())
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
    skipped = [k for k in state_dict if k not in model_keys]
    if skipped:
        print(f"Warning: skipped {len(skipped)} keys not in model: {skipped[:5]}{'...' if len(skipped) > 5 else ''}")
    model.load_state_dict(filtered_state_dict, strict=False)

    model.eval()
    model.requires_grad_(False)
    model = model.to(device)
    return model


def build_transforms(condition_keys, roi_size=(512, 512, 256)):
    all_keys = ["image"] + condition_keys
    image_only = ["image"]

    return Compose([
        LoadImaged(keys=all_keys),
        EnsureChannelFirstd(keys=all_keys),
        ThresholdIntensityd(keys=image_only, threshold=1000, above=False, cval=1000),
        ThresholdIntensityd(keys=image_only, threshold=-1000, above=True, cval=-1000),
        ScaleIntensityd(keys=image_only, minv=-1.0, maxv=1.0),
        CenterSpatialCropd(keys=all_keys, roi_size=roi_size),
        SpatialPadd(keys=all_keys, spatial_size=roi_size),
        ToTensord(keys=all_keys),
    ])


def encode_subject(model, subject_data, transforms, device, output_dir, condition_keys):
    img_path = Path(subject_data["image"])
    latent_path = output_dir / f"{img_path.stem}.pt"
    mask_paths = {key: output_dir / f"{img_path.stem}_{key}.pt" for key in condition_keys}

    if latent_path.exists() and all(p.exists() for p in mask_paths.values()):
        print(f"Skipping {latent_path} (already exists)")
        return {"image": str(latent_path), **{k: str(v) for k, v in mask_paths.items()}}

    data = transforms(subject_data)

    x = data["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        latent = model.encode_stage_2_inputs(x, quantized=True)
        latent = latent.squeeze(0).cpu()
    torch.save(latent, latent_path)

    for key in condition_keys:
        torch.save(data[key].cpu(), mask_paths[key])

    return {"image": str(latent_path), **{k: str(v) for k, v in mask_paths.items()}}


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.config, args.vqvae_ckpt, device)
    transforms = build_transforms(args.condition_keys)

    df = pd.read_csv(args.csv)
    if args.limit:
        df = df.head(args.limit)

    results = []
    for _, row in df.iterrows():
        subject_data = {col: str(row[col]) for col in ["image"] + args.condition_keys}
        paths = encode_subject(model, subject_data, transforms, device, output_dir, args.condition_keys)
        results.append(paths)
        print(f"Processed {paths['image']}")

    df_out = pd.DataFrame(results)
    df_out.to_csv(output_dir / "controlnet_latents.csv", index=False)
    print(f"\nSaved CSV with latent image + mask paths to {output_dir / 'controlnet_latents.csv'}")


if __name__ == "__main__":
    main()
