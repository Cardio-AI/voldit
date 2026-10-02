"""
Training script for DiT3D (Diffusion Transformer) in the latent space of the trained VQ-GAN.
"""

import argparse
import math
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import random
import numpy as np
import os
from omegaconf import OmegaConf

from src.models.vqvae import VQVAE
from src.models.dit import DiT3D
from src.models.ddpmscheduler import DDPMScheduler
from src.models.flow_matching_scheduler import FlowMatchingScheduler
from src.config_utils import get_dit_params, get_dit_scheduler, get_stage1_params
from src.training.dit_trainer import DiTTrainer
from src.data.dataloading import get_dit_dataloader


def build_scheduler(config):
    """
    Instantiate the correct scheduler from config.

    ``scheduler_type`` selects "ddpm" (default) or "flow_matching".
    """
    scheduler_type = config.get("scheduler_type", "ddpm")
    if scheduler_type == "flow_matching":
        return FlowMatchingScheduler(**get_dit_scheduler(config))
    if scheduler_type == "ddpm":
        return DDPMScheduler(**get_dit_scheduler(config))
    raise ValueError(f"Unknown scheduler_type: {scheduler_type}")


# ------------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--config_vqvae", type=str, required=False)
    parser.add_argument("--vqvae_ckpt", type=str, required=False)

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--run_name", type=str, required=True)

    parser.add_argument("--training_ids", type=str, required=True)
    parser.add_argument("--validation_ids", type=str, required=True)

    parser.add_argument("--seed", type=int, default=42)

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--use_precomputed_latents", dest="use_precomputed_latents",
                       action="store_true", default=None)
    group.add_argument("--no_precomputed_latents", dest="use_precomputed_latents",
                       action="store_false")

    return parser.parse_args()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    args = parse_args()

    # -----------------------
    # DDP setup
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
    defaults = OmegaConf.create({
        "training": {
            "n_epochs": 1000,
            "eval_freq": 50,
            "batch_size": 2,
            "num_workers": 4,
            "roi_size": [512, 512, 256],
            "use_precomputed_latents": True,
            "scale_factor": 1.0,
            "use_ema": True,
            "ema_decay": 0.9999,
        },
        "optim": {
            "lr": 1.0e-4,
            "lr_gamma": 0.999,
        },
    })
    config = OmegaConf.merge(defaults, OmegaConf.load(args.config))
    if args.use_precomputed_latents is not None:
        config.training.use_precomputed_latents = args.use_precomputed_latents

    run_dir = Path(args.output_dir) / args.run_name
    if is_main:
        run_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(config, run_dir / "config.yaml")

    if world_size > 1:
        dist.barrier()

    writer_train = SummaryWriter(run_dir / "logs" / "train") if is_main else None
    writer_val = SummaryWriter(run_dir / "logs" / "val") if is_main else None

    # -----------------------
    # Data
    # -----------------------
    use_precomputed_latents = config.training.get("use_precomputed_latents", False)

    train_loader, val_loader = get_dit_dataloader(
        training_ids=args.training_ids,
        validation_ids=args.validation_ids,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        rank=rank,
        world_size=world_size,
        roi_size=tuple(config.training.roi_size),
        use_precomputed_latents=use_precomputed_latents,
        preload_latents=True,
        use_persistent=False,
        augment_latents=config.training.get("augment_latents", False),
    )

    # -----------------------
    # Stage 1 (VQ-GAN encoder)
    # -----------------------
    if not use_precomputed_latents:
        if is_main:
            print(f"Loading VQ-GAN from {args.vqvae_ckpt}")
        config_vqvae = OmegaConf.load(args.config_vqvae)
        stage1 = VQVAE(**get_stage1_params(config_vqvae))
        vqvae_ckpt = torch.load(args.vqvae_ckpt, map_location="cpu", weights_only=True)
        vqvae_state = vqvae_ckpt.get("model", vqvae_ckpt.get("state_dict", vqvae_ckpt))
        model_keys = set(stage1.state_dict().keys())
        filtered = {k: v for k, v in vqvae_state.items() if k in model_keys}
        skipped = [k for k in vqvae_state if k not in model_keys]
        if skipped:
            print(f"Warning: skipped {len(skipped)} VQVAE keys: {skipped[:5]}{'...' if len(skipped) > 5 else ''}")
        stage1.load_state_dict(filtered, strict=False)
        stage1.eval()
        stage1.requires_grad_(False)
        stage1 = stage1.to(device)
    else:
        if is_main:
            print("Using precomputed latents — skipping VQ-GAN initialization.")
        stage1 = None

    # -----------------------
    # DiT model
    # -----------------------
    model = DiT3D(**get_dit_params(config)).to(device)
    scheduler = build_scheduler(config)

    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )

    # -----------------------
    # Optimizer + LR scheduler
    # -----------------------

    # Separate parameters: no weight decay on biases and normalisation layers.
    weight_decay = float(config.optim.get("weight_decay", 1e-4))
    no_decay_keywords = {"bias", "norm"}
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if any(kw in name for kw in no_decay_keywords):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params,    "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    if "dit" in config:
        # Preserve the public runner's single optimizer group for exact resume.
        optimizer = optim.AdamW(model.parameters(), lr=config.optim.lr,
                                weight_decay=float(config.optim.get("weight_decay", 0.01)))
    else:
        optimizer = optim.AdamW(param_groups, lr=config.optim.lr)

    # LR scheduler: "cosine_warmup" (default) or "exponential" (legacy).
    scheduler_type = config.optim.get("scheduler", "exponential" if "dit" in config else "cosine_warmup")

    if scheduler_type == "exponential":
        lr_scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=config.optim.lr_gamma
        )
    elif scheduler_type == "cosine_warmup":
        # Cosine annealing with linear warmup.
        # epoch 0            → lr * (1 / warmup_epochs)
        # epoch warmup_epochs → lr * 1.0   (full LR)
        # epoch n_epochs - 1  → lr * min_lr_ratio
        warmup_epochs = int(config.optim.get("warmup_epochs", 100))
        min_lr_ratio  = float(config.optim.get("min_lr_ratio", 0.01))
        total_epochs  = int(config.training.n_epochs)

        def _warmup_cosine(epoch):
            warmup = max(warmup_epochs, 1)
            if epoch < warmup:
                return (epoch + 1) / warmup
            progress = (epoch - warmup) / max(total_epochs - warmup - 1, 1)
            cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        lr_scheduler = LambdaLR(optimizer, lr_lambda=_warmup_cosine)
    else:
        raise ValueError(f"Unknown optimizer scheduler: {scheduler_type}")

    # -----------------------
    # Resume checkpoint
    # -----------------------
    checkpoint_path = run_dir / "last_checkpoint.pth"
    start_epoch = 0
    best_loss = float("inf")
    dit_checkpoint = None

    if checkpoint_path.exists():
        if is_main:
            print(f"Loading checkpoint from {checkpoint_path}")

        dit_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

        raw_state = dit_checkpoint["model"]
        if isinstance(model, DDP):
            model.module.load_state_dict(raw_state)
        else:
            model.load_state_dict(raw_state)

        optimizer.load_state_dict(dit_checkpoint["optimizer"])

        if dit_checkpoint.get("lr_scheduler") is not None:
            lr_scheduler.load_state_dict(dit_checkpoint["lr_scheduler"])

        start_epoch = dit_checkpoint["epoch"] + 1
        best_loss = dit_checkpoint.get("best_loss", float("inf"))

        if world_size > 1:
            dist.barrier()

    # -----------------------
    # Trainer
    # -----------------------
    trainer = DiTTrainer(
        model=model,
        stage1=stage1,
        scheduler=scheduler,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
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

    if dit_checkpoint is not None and dit_checkpoint.get("ema") is not None:
        if is_main:
            print("Restoring EMA state")
        trainer.load_ema_state(dit_checkpoint["ema"])

    if dit_checkpoint is not None:
        if dit_checkpoint.get("scaler") is not None:
            trainer.scaler.load_state_dict(dit_checkpoint["scaler"])
        if dit_checkpoint.get("latent_mean") is not None:
            trainer.latent_mean = dit_checkpoint["latent_mean"].to(device)
            trainer.latent_std = dit_checkpoint["latent_std"].to(device)

    # -----------------------
    # Train
    # -----------------------
    trainer.train()

    # -----------------------
    # Cleanup
    # -----------------------
    if is_main:
        writer_train.close()
        writer_val.close()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
