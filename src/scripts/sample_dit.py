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
from src.models.flow_matching_scheduler import FlowMatchingScheduler
from src.models.dpm_solver_scheduler import DPMSolverPPScheduler
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
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=4)
    parser.add_argument("--timesteps", type=int, default=300)
    parser.add_argument(
        "--scheduler",
        type=str,
        default="ddpm",
        choices=["ddpm", "ddim", "dpm_pp", "flow_matching"],
        help=(
            "ddpm / ddim : standard DDPM-trained models. "
            "dpm_pp : DPM-Solver++ 2M for DDPM-trained models. "
            "flow_matching : for models trained with FlowMatchingScheduler."
        ),
    )
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


def _build_scheduler(scheduler_name: str, diff_cfg):
    """
    Instantiate the requested inference scheduler from the diffusion config.

    scheduler_name choices
    ----------------------
    "ddpm"           : DDPMScheduler (stochastic, same steps as training)
    "ddim"           : DDIMScheduler (deterministic, subset of steps)
    "dpm_pp"         : DPMSolverPPScheduler (2nd-order multistep)
    "flow_matching"  : FlowMatchingScheduler (for FM-trained models)
    """
    cfg = dict(get_dit_scheduler(diff_cfg))
    trained_with_fm = diff_cfg.get("scheduler_type", "ddpm") == "flow_matching"
    if (scheduler_name == "flow_matching") != trained_with_fm:
        raise ValueError("Sampling objective must match the training config scheduler_type")
    if scheduler_name == "ddpm":
        return DDPMScheduler(**cfg)
    elif scheduler_name == "ddim":
        return DDIMScheduler(**cfg)
    elif scheduler_name == "dpm_pp":
        return DPMSolverPPScheduler(**cfg)
    elif scheduler_name == "flow_matching":
        return FlowMatchingScheduler(**cfg)
    else:
        raise ValueError(f"Unknown scheduler '{scheduler_name}'.")


def _run_diffusion(dit, stage1, scheduler, diff_cfg, indices, out_dir,
                   affine, scale_factor, latent_shape, device, gpu_id=None,
                   batch_size=1, latent_mean=None, latent_std=None):
    in_channels = get_dit_params(diff_cfg).in_channels
    tag = f"[GPU {gpu_id}] " if gpu_id is not None else ""
    out_dir.mkdir(parents=True, exist_ok=True)

    is_flow_matching = isinstance(scheduler, FlowMatchingScheduler)

    for chunk_start in range(0, len(indices), batch_size):
        chunk = indices[chunk_start : chunk_start + batch_size]
        batch = len(chunk)

        x = torch.randn((batch, in_channels, *latent_shape), device=device)
        scheduler.set_timesteps(scheduler.num_inference_steps)
        x_self_cond = None
        with torch.no_grad(), amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            for t in scheduler.timesteps:
                if is_flow_matching:
                    # t is a float in (0, 1]; scale to [0, T] for the embedder
                    t_embed = t.item() * scheduler.num_train_timesteps
                    t_batch = torch.full((batch,), t_embed, device=device, dtype=torch.float32)
                else:
                    t_batch = torch.full((batch,), t, device=device, dtype=torch.long)
                noise_pred = dit(x, t=t_batch, y=None, x_self_cond=x_self_cond)
                x, x0 = scheduler.step(noise_pred, t, x)
                if dit.self_conditioning:
                    x_self_cond = x0.detach()
            x = x / scale_factor
            if latent_mean is not None:
                x = x * (latent_std + 1e-8) + latent_mean

        latents_cpu = x.float().cpu()
        del x
        torch.cuda.empty_cache()

        for b, i in enumerate(chunk):
            latent = latents_cpu[b : b + 1].to(device)
            with torch.no_grad(), amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
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

        latent_mean = ckpt.get("latent_mean")
        latent_std = ckpt.get("latent_std")
        if latent_mean is not None:
            latent_mean = latent_mean.to(device)
            latent_std = latent_std.to(device)
        elif diff_cfg.get("training", {}).get("normalize_latents", False):
            raise RuntimeError(
                f"Config has normalize_latents=true but checkpoint {ckpt_path} "
                "lacks latent_mean/latent_std. Cannot de-normalize — output "
                "will have a constant intensity offset."
            )

        scheduler = _build_scheduler(scheduler_name, diff_cfg)
        scheduler.set_timesteps(timesteps)

        _run_diffusion(dit, stage1, scheduler, diff_cfg, indices, Path(out_dir_str),
                       affine, scale_factor, tuple(latent_shape), device, gpu_id=gpu_id,
                       batch_size=batch_size,
                       latent_mean=latent_mean, latent_std=latent_std)

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

            latent_mean = ckpt.get("latent_mean")
            latent_std = ckpt.get("latent_std")
            if latent_mean is not None:
                latent_mean = latent_mean.to(device)
                latent_std = latent_std.to(device)
            elif diff_cfg_obj.get("training", {}).get("normalize_latents", False):
                raise RuntimeError(
                    f"Config has normalize_latents=true but checkpoint {ckpt_path} "
                    "lacks latent_mean/latent_std. Cannot de-normalize — output "
                    "will have a constant intensity offset."
                )

            scheduler = _build_scheduler(args.scheduler, diff_cfg_obj)
            scheduler.set_timesteps(args.timesteps)

            indices = list(range(args.n_samples))
            print(f"  Sampling {args.n_samples} volumes ({args.scheduler.upper()}, {args.timesteps} steps)...")
            _run_diffusion(dit, stage1, scheduler, diff_cfg_obj, indices, out_dir,
                           affine, args.scale_factor, tuple(args.latent_shape), device,
                           batch_size=args.batch_size,
                           latent_mean=latent_mean, latent_std=latent_std)

    print("\nDone.")


if __name__ == "__main__":
    main()
