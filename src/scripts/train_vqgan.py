""" Training script for VQ-VAE GAN with MONAI losses (DDP compatible) """

import argparse
from pathlib import Path
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import random
import numpy as np
import os
from omegaconf import OmegaConf

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.models.vqvae import VQVAE
from src.models.patchgan_discriminator import PatchDiscriminator
from src.config_utils import get_stage1_params
from src.training.vqgan_trainer import VQGANTrainer
from src.losses.vqgan_loss import VQGANLoss
from src.data.dataloading import get_dataloader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, required=False)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--training_ids", type=str, required=True)
    parser.add_argument("--validation_ids", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=0, help="DDP local rank")
    return parser.parse_args()


def main():
    args = parse_args()

    # -----------------------
    # DDP / device setup
    # -----------------------
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    torch.set_float32_matmul_precision("high")

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        rank = dist.get_rank()
        is_main = rank == 0
    else:
        rank = 0
        is_main = True

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # -----------------------
    # Reproducibility
    # -----------------------
    seed = args.seed + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    # -----------------------
    # Config + run directory
    # -----------------------
    raw_config = OmegaConf.load(args.config)
    stage1_section = raw_config.get("stage1", {})
    defaults = OmegaConf.create({
        "training": {
            "n_epochs": 1300,
            "batch_size": 4,
            "num_workers": 4,
            "eval_freq": 10,
            "roi_size": [512, 512, 256],
            "use_persistent": False,
            "cache_rate": 0.0,
        },
        "optim": {
            "lr_g": stage1_section.get("base_lr", 3.0e-5),
            "lr_d": stage1_section.get("disc_lr", 2.0e-5),
        },
        "losses": {
            "perceptual_weight": stage1_section.get("perceptual_weight", 0.005),
            "adv_weight": stage1_section.get("adv_weight", 0.005),
            "jukebox_weight": 0.0,
            "adv_warmup": 10,
            "perceptual_params": raw_config.get("perceptual_network", {}).get(
                "params", {"spatial_dims": 3, "network_type": "squeeze"}
            ),
            "jukebox_params": {"spatial_dims": 3},
        },
    })
    config = OmegaConf.merge(defaults, raw_config)
    run_dir = Path(args.output_dir) / args.run_name
    if is_main:
        run_dir.mkdir(parents=True, exist_ok=True)

    if world_size > 1:
        dist.barrier()

    writer_train = SummaryWriter(log_dir=run_dir / "logs" / "train") if is_main else None
    writer_val = SummaryWriter(log_dir=run_dir / "logs" / "val") if is_main else None

    # -----------------------
    # Data loaders
    # -----------------------
    train_loader, val_loader = get_dataloader(
        cache_dir=args.cache_dir,
        training_ids=args.training_ids,
        validation_ids=args.validation_ids,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        roi_size=tuple(config.training.roi_size),
        rank=rank,
        world_size=world_size,
        use_persistent=config.training.use_persistent,
        cache_rate=config.training.get("cache_rate", 0.0),
    )

    # -----------------------
    # Models
    # -----------------------
    model = VQVAE(**get_stage1_params(config)).to(device)
    discriminator = PatchDiscriminator(**config.discriminator.params).to(device)

    if world_size > 1:
        discriminator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(discriminator)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        discriminator = DDP(discriminator, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    # -----------------------
    # Loss
    # -----------------------
    loss_fn = VQGANLoss(
        perceptual_weight=config.losses.perceptual_weight,
        jukebox_weight=config.losses.jukebox_weight,
        perceptual_params=config.losses.perceptual_params,
        jukebox_params=config.losses.jukebox_params,
        device=device,
    )

    # -----------------------
    # Optimizers
    # -----------------------
    optimizer_g = optim.AdamW(model.parameters(), lr=config.optim.lr_g)
    optimizer_d = optim.AdamW(discriminator.parameters(), lr=config.optim.lr_d)

    # -----------------------
    # Resume checkpoint
    # -----------------------
    checkpoint_path = run_dir / "last_checkpoint.pth"
    start_epoch = 0
    best_loss = float("inf")

    if checkpoint_path.exists():
        if is_main:
            print(f"Loading checkpoint from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device)

        if isinstance(model, DDP):
            model.module.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint["model"])

        if isinstance(discriminator, DDP):
            discriminator.module.load_state_dict(checkpoint["discriminator"])
        else:
            discriminator.load_state_dict(checkpoint["discriminator"])

        optimizer_g.load_state_dict(checkpoint["optimizer_g"])
        optimizer_d.load_state_dict(checkpoint["optimizer_d"])

        for param_group in optimizer_g.param_groups:
            param_group["lr"] = config.optim.lr_g
        for param_group in optimizer_d.param_groups:
            param_group["lr"] = config.optim.lr_d

        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint.get("best_loss", float("inf"))

        if is_main:
            print(f"Resuming from epoch {start_epoch}")

        if world_size > 1:
            dist.barrier()

    # -----------------------
    # Trainer
    # -----------------------
    trainer = VQGANTrainer(
        model=model,
        discriminator=discriminator,
        loss_fn=loss_fn,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        scheduler_g=None,
        scheduler_d=None,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        run_dir=run_dir,
        config=config,
        writer_train=writer_train,
        writer_val=writer_val,
        is_main=is_main,
        start_epoch=start_epoch,
        best_loss=best_loss,
    )

    trainer.train()

    if is_main:
        writer_train.close()
        writer_val.close()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
