"""
Unconditional sampling from a trained DiT3D Latent Diffusion Model.
"""

import argparse
import os
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

import torch
import torch.multiprocessing as mp
import numpy as np
import nibabel as nib
from omegaconf import OmegaConf
from torch import amp

from src.models.dit import DiT3D
from src.models.vqvae import VQVAE
from src.models.ddimscheduler import DDIMScheduler
from src.models.ddpmscheduler import DDPMScheduler
from src.config_utils import get_dit_params, get_dit_scheduler, get_stage1_params


def parse_args():
    parser = argparse.ArgumentParser(description="Sample from a trained DiT3D LDM")
    parser.add_argument("--stage1_ckpt", type=str, required=True)
    parser.add_argument("--stage1_cfg", type=str, required=True)
    parser.add_argument("--diff_cfg", type=str, required=True)
    parser.add_argument("--diff_ckpt", type=str, default=None,
                        help="Path to a single DiT checkpoint")
    parser.add_argument("--diff_run_dir", type=str, default=None,
                        help="Directory containing checkpoint_epoch_N.pth files")
    parser.add_argument("--epoch_start", type=int, default=None)
    parser.add_argument("--epoch_end", type=int, default=None)
    parser.add_argument("--epoch_step", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="samples")
    parser.add_argument("--n_samples", type=int, default=4)
    parser.add_argument("--timesteps", type=int, default=300)
    parser.add_argument("--scheduler", type=str, default="ddpm", choices=["ddpm", "ddim"])
    parser.add_argument("--reference_nii", type=str, default=None,
                        help="Reference .nii.gz to copy affine from")
    parser.add_argument("--scale_factor", type=float, default=1.0)
    parser.add_argument("--latent_shape", type=int, nargs=3, required=True,
                        metavar=("D", "H", "W"), help="Latent spatial dimensions, must match model input_size e.g. 32 32 32")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Number of latents to denoise in parallel before decoding one-by-one.")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.diff_ckpt is None and args.diff_run_dir is None:
        parser.error("Specify either --diff_ckpt or --diff_run_dir with --epoch_start/--epoch_end")
    if args.diff_run_dir is not None and (args.epoch_start is None or args.epoch_end is None):
        parser.error("--diff_run_dir requires --epoch_start and --epoch_end")

    return args


def load_stage1(cfg_path, ckpt_path, device):
    cfg = OmegaConf.load(cfg_path)
    model = VQVAE(**get_stage1_params(cfg))

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))
    model_keys = set(model.state_dict().keys())
    filtered = {k: v for k, v in state_dict.items() if k in model_keys}
    skipped = [k for k in state_dict if k not in model_keys]
    if skipped:
        print(f"Warning: skipped {len(skipped)} stage1 keys: {skipped[:5]}{'...' if len(skipped) > 5 else ''}")
    model.load_state_dict(filtered, strict=False)

    return model.to(device).eval().requires_grad_(False)


def load_dit(cfg_path, ckpt_path, device):
    cfg = OmegaConf.load(cfg_path)
    model = DiT3D(**get_dit_params(cfg))

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    # Start with the full model state (includes non-trainable params like pos_embed),
    # then override trainable parameters with EMA shadow weights if available.
    state_dict = ckpt.get("model", ckpt)
    ema = ckpt.get("ema")
    if ema is not None:
        state_dict = dict(state_dict)  # copy so we don't mutate the checkpoint
        state_dict.update(ema["shadow"])
    model.load_state_dict(state_dict)

    return model.to(device).eval().requires_grad_(False), cfg


def _run_diffusion(dit, stage1, scheduler, diff_cfg, indices, out_dir,
                   affine, scale_factor, latent_shape, device, gpu_id=None,
                   batch_size=1):
    in_channels = get_dit_params(diff_cfg).in_channels
    tag = f"[GPU {gpu_id}] " if gpu_id is not None else ""
    out_dir.mkdir(parents=True, exist_ok=True)

    for chunk_start in range(0, len(indices), batch_size):
        chunk = indices[chunk_start : chunk_start + batch_size]
        batch = len(chunk)

        x = torch.randn((batch, in_channels, *latent_shape), device=device)
        with torch.no_grad(), amp.autocast(device_type=device.type):
            for t in scheduler.timesteps:
                t_batch = torch.full((batch,), t, device=device, dtype=torch.long)
                noise_pred = dit(x, t=t_batch, y=None)
                x, _ = scheduler.step(noise_pred, t, x)
            x = x / scale_factor

        latents_cpu = x.float().cpu()
        del x
        torch.cuda.empty_cache()

        for b, i in enumerate(chunk):
            latent = latents_cpu[b : b + 1].to(device)
            with torch.no_grad(), amp.autocast(device_type=device.type):
                recon = stage1.decode_stage_2_outputs(latent)
            del latent

            vol = np.clip(recon[0, 0].float().cpu().numpy(), -1.0, 1.0)
            hu = (((vol + 1.0) * (2000.0 / 2.0)) - 1000.0).astype(np.int16)
            out_path = out_dir / f"sample_{i:03d}.nii.gz"
            nib.save(nib.Nifti1Image(hu, affine), out_path)
            print(f"    {tag}Saved {out_path}", flush=True)

            del recon
            torch.cuda.empty_cache()


def _persistent_worker(rank, gpu_id, stage1_cfg, stage1_ckpt, diff_cfg_path,
                        job_queue, done_queue, scale_factor, latent_shape, batch_size):
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    device = torch.device(f"cuda:{gpu_id}")

    stage1 = load_stage1(stage1_cfg, stage1_ckpt, device)

    diff_cfg = OmegaConf.load(diff_cfg_path)
    dit = DiT3D(**get_dit_params(diff_cfg)).to(device).eval().requires_grad_(False)

    while True:
        job = job_queue.get()
        if job is None:
            break

        ckpt_path, out_dir_str, indices, affine, scheduler_name, timesteps = job

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        state_dict = dict(ckpt.get("model", ckpt))
        ema = ckpt.get("ema")
        if ema is not None:
            state_dict.update(ema["shadow"])
        dit.load_state_dict(state_dict)

        if scheduler_name == "ddpm":
            scheduler = DDPMScheduler(**get_dit_scheduler(diff_cfg))
        else:
            scheduler = DDIMScheduler(**get_dit_scheduler(diff_cfg))
        scheduler.set_timesteps(timesteps)

        _run_diffusion(dit, stage1, scheduler, diff_cfg, indices, Path(out_dir_str),
                       affine, scale_factor, tuple(latent_shape), device, gpu_id=gpu_id,
                       batch_size=batch_size)

        done_queue.put(rank)


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    n_gpus = torch.cuda.device_count()
    use_multi_gpu = n_gpus > 1
    print(f"Detected {n_gpus} GPU(s) — {'multi-GPU persistent workers' if use_multi_gpu else 'single-GPU'}.")

    affine = None
    if args.reference_nii is not None:
        ref = nib.load(args.reference_nii)
        affine = ref.affine
        print(f"Using affine from {args.reference_nii}")

    if args.diff_run_dir is not None:
        run_dir = Path(args.diff_run_dir)
        jobs = []
        for epoch in range(args.epoch_start, args.epoch_end + 1, args.epoch_step):
            ckpt = run_dir / f"checkpoint_epoch_{epoch-1}.pth"
            if not ckpt.exists():
                print(f"Warning: checkpoint not found, skipping: {ckpt}")
                continue
            jobs.append((epoch, ckpt, Path(args.output_dir) / f"epoch_{epoch}"))
    else:
        jobs = [(None, Path(args.diff_ckpt), Path(args.output_dir))]

    # ── Multi-GPU path ────────────────────────────────────────────────────
    if use_multi_gpu:
        nprocs = min(n_gpus, args.n_samples)
        indices_per_rank = [list(range(i, args.n_samples, nprocs)) for i in range(nprocs)]
        active_ranks = [r for r in range(nprocs) if indices_per_rank[r]]

        ctx = mp.get_context("spawn")
        job_queues = [ctx.Queue() for _ in range(nprocs)]
        done_queue = ctx.Queue()

        workers = []
        for rank in active_ranks:
            p = ctx.Process(
                target=_persistent_worker,
                args=(rank, rank, args.stage1_cfg, args.stage1_ckpt, args.diff_cfg,
                      job_queues[rank], done_queue, args.scale_factor, args.latent_shape,
                      args.batch_size),
                daemon=True,
            )
            p.start()
            workers.append(p)

        print(f"Spawned {len(active_ranks)} persistent worker(s) (GPUs {active_ranks}).")

        for epoch, ckpt_path, out_dir in jobs:
            label = f"epoch {epoch}" if epoch is not None else ckpt_path.name
            print(f"\n[{label}] {ckpt_path}")

            for rank in active_ranks:
                job_queues[rank].put((
                    str(ckpt_path), str(out_dir), indices_per_rank[rank],
                    affine, args.scheduler, args.timesteps,
                ))

            for _ in active_ranks:
                while True:
                    try:
                        done_queue.get(timeout=10)
                        break
                    except Exception:
                        # Check if all workers are still alive; raise if any died
                        dead = [p for p in workers if not p.is_alive()]
                        if dead:
                            raise RuntimeError(
                                f"{len(dead)} worker(s) died unexpectedly. "
                                "Check GPU memory and model config."
                            )

        for rank in active_ranks:
            job_queues[rank].put(None)
        for p in workers:
            p.join()

    # ── Single-GPU path ───────────────────────────────────────────────────
    else:
        print("Loading VQ-GAN...")
        stage1 = load_stage1(args.stage1_cfg, args.stage1_ckpt, device)

        diff_cfg_obj = OmegaConf.load(args.diff_cfg)
        dit = DiT3D(**get_dit_params(diff_cfg_obj)).to(device).eval().requires_grad_(False)

        for epoch, ckpt_path, out_dir in jobs:
            label = f"epoch {epoch}" if epoch is not None else ckpt_path.name
            print(f"\n[{label}] {ckpt_path}")

            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            state_dict = dict(ckpt.get("model", ckpt))
            ema = ckpt.get("ema")
            if ema is not None:
                state_dict.update(ema["shadow"])
            dit.load_state_dict(state_dict)

            if args.scheduler == "ddpm":
                scheduler = DDPMScheduler(**get_dit_scheduler(diff_cfg_obj))
            else:
                scheduler = DDIMScheduler(**get_dit_scheduler(diff_cfg_obj))
            scheduler.set_timesteps(args.timesteps)

            indices = list(range(args.n_samples))
            print(f"  Sampling {args.n_samples} volumes ({args.scheduler.upper()}, {args.timesteps} steps)...")
            _run_diffusion(dit, stage1, scheduler, diff_cfg_obj, indices, out_dir,
                           affine, args.scale_factor, tuple(args.latent_shape), device,
                           batch_size=args.batch_size)

    print("\nDone.")


if __name__ == "__main__":
    main()
