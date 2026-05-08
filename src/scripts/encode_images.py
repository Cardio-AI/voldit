# encode_latents.py
import argparse
import sys
from pathlib import Path

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
    Resized,
)
from omegaconf import OmegaConf

sys.path.append(str(Path(__file__).resolve().parents[2]))


def parse_args():
    parser = argparse.ArgumentParser(description="Encode images into VQVAE latents")

    parser.add_argument("--csv", required=True, help="CSV containing image paths")
    parser.add_argument("--output_dir", required=True, help="Directory to store latents")
    parser.add_argument("--vqvae_ckpt", required=True, help="Path to VQVAE checkpoint")
    parser.add_argument("--config", required=True, help="Path to VQVAE config yaml")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images")
    parser.add_argument("--device", default="cuda")

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


def build_transforms():
    # Replace with transforms appropriate for your data.
    # Ensure image size is a multiple of 32.
    return Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            ThresholdIntensityd(keys=["image"], threshold=1000, above=False, cval=1000),
            ThresholdIntensityd(keys=["image"], threshold=-1000, above=True, cval=-1000),
            ScaleIntensityd(keys=["image"], minv=-1.0, maxv=1.0),
            Resized(keys=["image"], spatial_size=(512, 512, 256), mode="trilinear"),
            ToTensord(keys=["image"]),
        ]
    )


def encode_images(model, image_paths, transforms, device, output_dir, batch_size):
    latent_paths = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]

        latent_path_candidates = [output_dir / f"{Path(p).stem}.pt" for p in batch_paths]

        to_encode = [(p, lp) for p, lp in zip(batch_paths, latent_path_candidates) if not lp.exists()]
        for _, lp in zip(batch_paths, latent_path_candidates):
            if lp.exists():
                latent_paths.append(str(lp))
                print(f"Skipping {lp} (already exists)")

        if not to_encode:
            continue

        batch_img_paths, batch_latent_paths = zip(*to_encode)

        batch_tensors = [transforms({"image": p})["image"] for p in batch_img_paths]
        x = torch.stack(batch_tensors).to(device)

        with torch.no_grad():
            latents = model.encode_stage_2_inputs(x, quantized=True)
            latents = latents.cpu()

        for latent, latent_path in zip(latents, batch_latent_paths):
            torch.save(latent, latent_path)
            latent_paths.append(str(latent_path))
            print(f"Saved {latent_path}")

    return latent_paths


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "latents.csv"

    model = load_model(args.config, args.vqvae_ckpt, device)

    transforms = build_transforms()

    df = pd.read_csv(args.csv)
    image_paths = [str(row["image"]) for _, row in df.iterrows()]

    if args.limit:
        image_paths = image_paths[: args.limit]

    latent_paths = encode_images(
        model,
        image_paths,
        transforms,
        device,
        output_dir,
        args.batch_size,
    )

    df_latents = pd.DataFrame({"image": latent_paths})
    df_latents.to_csv(output_csv, index=False)

    print(f"\nSaved CSV with latent paths to {output_csv}")


if __name__ == "__main__":
    main()
