import random
import torch
from torch import amp
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from src.models.ema import EMA


def _unsqueeze_right(x, ndim):
    """Add trailing singleton dimensions until ``x`` has ``ndim`` dims."""
    while x.ndim < ndim:
        x = x.unsqueeze(-1)
    return x


class DiTTrainer:

    def __init__(
        self,
        model,
        stage1,
        scheduler,
        optimizer,
        lr_scheduler,
        train_loader,
        val_loader,
        device,
        run_dir: Path,
        config,
        writer_train=None,
        writer_val=None,
        is_main=True,
        start_epoch=0,
        best_loss=float("inf"),
    ):

        self.model = model
        self.stage1 = stage1
        self.scheduler = scheduler

        self.use_ema = config.training.get("use_ema", True)
        self.ema_decay = config.training.get("ema_decay", 0.9999)
        ema_warmup_steps = int(config.training.get("ema_warmup_steps", 0))

        if self.use_ema:
            raw_model = (
                self.model.module if hasattr(self.model, "module") else self.model
            )
            self.ema = EMA(raw_model, decay=self.ema_decay, warmup_steps=ema_warmup_steps)
        else:
            self.ema = None

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.run_dir = run_dir
        self.writer_train = writer_train
        self.writer_val = writer_val
        self.is_main = is_main

        self.n_epochs = config.training.n_epochs
        self.eval_freq = config.training.eval_freq
        self.scale_factor = config.training.get("scale_factor", 1.0)

        # Min-SNR-γ loss weighting (set to 0.0 to disable).
        self.min_snr_gamma = float(config.training.get("min_snr_gamma", 0.0))

        # Offset noise: adds a spatially-uniform per-channel noise component.
        # Helps the model learn global intensity distribution (useful for CT).
        # Only applied during DDPM training; 0.0 disables.
        self.offset_noise_strength = float(
            config.training.get("offset_noise_strength", 0.0)
        )

        # Gradient accumulation: simulates a larger effective batch size.
        self.grad_accum_steps = max(1, int(config.training.get("grad_accum_steps", 1)))

        # Self-conditioning: feed a previous x0 estimate back into the model.
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        self.self_conditioning = config.training.get("self_conditioning", raw_model.self_conditioning)
        if bool(self.self_conditioning) != bool(raw_model.self_conditioning):
            raise ValueError("training.self_conditioning must match model self_conditioning")

        self.start_epoch = start_epoch
        self.best_loss = best_loss

        self.scaler = amp.GradScaler("cuda", enabled=device.type == "cuda")

        # Detect scheduler type once so inner loops stay clean.
        self._is_flow_matching = (
            getattr(self.scheduler, "prediction_type", None) == "flow_matching"
        )

        # Per-channel latent normalisation.
        self.latent_mean = None
        self.latent_std = None
        if config.training.get("normalize_latents", False):
            stats_path = config.training.get("latent_stats_path", None)
            if stats_path and Path(stats_path).exists():
                stats = torch.load(stats_path, map_location="cpu")
                self.latent_mean = stats["mean"].to(self.device)
                self.latent_std = stats["std"].to(self.device)
                if self.is_main:
                    print("Loaded latent stats from", stats_path)
            else:
                if self.is_main:
                    print("Computing per-channel latent statistics (two passes)…")
                self.latent_mean, self.latent_std = self._compute_latent_stats()
                if self.is_main:
                    mean_vals = self.latent_mean.view(-1).tolist()
                    std_vals = self.latent_std.view(-1).tolist()
                    print(f"  channel means : {[f'{v:.4f}' for v in mean_vals]}")
                    print(f"  channel stds  : {[f'{v:.4f}' for v in std_vals]}")

    # ==========================================================
    # PUBLIC TRAIN LOOP
    # ==========================================================

    def train(self):
        for epoch in range(self.start_epoch, self.n_epochs):

            if hasattr(self.train_loader, "sampler") and isinstance(
                self.train_loader.sampler,
                torch.utils.data.DistributedSampler,
            ):
                self.train_loader.sampler.set_epoch(epoch)

            self._train_epoch(epoch)

            if self.lr_scheduler:
                self.lr_scheduler.step()

            if (epoch + 1) % self.eval_freq == 0:

                val_loss = self._validate(epoch)
                if self.is_main:
                    self._save_best_checkpoint(epoch, val_loss)
                    self._save_periodic_checkpoint(epoch)

                if torch.distributed.is_initialized():
                    torch.distributed.barrier()

        if self.is_main:
            self._save_final_model()

    # ==========================================================
    # TRAIN ONE EPOCH
    # ==========================================================

    def _train_epoch(self, epoch):

        self.model.train()

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}", disable=not self.is_main)

        self.optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(pbar):

            x = batch["image"]
            if hasattr(x, "as_tensor"):
                x = x.as_tensor()
            x = x.to(self.device)

            latents = self._encode_inputs(x)
            latents = latents * self.scale_factor

            noise, t, noisy_latents, t_model = self._sample_t_and_noisy(latents)
            target = self._get_target(latents, noise, t)

            # ----------------------------------------------------------
            # Self-conditioning: on ~50 % of steps, run a no-grad forward
            # pass to obtain a rough x0 estimate and feed it back in.
            # ----------------------------------------------------------
            x_self_cond = None
            if self.self_conditioning and random.random() < 0.5:
                with torch.no_grad():
                    with amp.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
                        v_pred = self.model(
                            noisy_latents, t=t_model, y=None, x_self_cond=None
                        )
                    x_self_cond = self._x0_from_pred(
                        v_pred.detach(), noisy_latents, t
                    )

            with amp.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
                noise_pred = self.model(
                    noisy_latents, t=t_model, y=None, x_self_cond=x_self_cond
                )
                # Scale loss for gradient accumulation before backward.
                window_start = (step // self.grad_accum_steps) * self.grad_accum_steps
                window_size = min(self.grad_accum_steps, len(self.train_loader) - window_start)
                loss = self._compute_loss(noise_pred, target, t) / window_size

            self.scaler.scale(loss).backward()

            # Perform an optimiser step every grad_accum_steps, and also on
            # the final (possibly incomplete) accumulation window.
            is_update_step = (
                (step + 1) % self.grad_accum_steps == 0
                or (step + 1) == len(self.train_loader)
            )
            if is_update_step:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                old_scale = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

                if self.ema and self.scaler.get_scale() >= old_scale:
                    self.ema.update()

            if self.is_main:
                # Un-scale for display so the logged value is independent of
                # grad_accum_steps.
                display_loss = loss.item() * window_size
                global_step = epoch * len(self.train_loader) + step
                self.writer_train.add_scalar("loss", display_loss, global_step)
                self.writer_train.add_scalar(
                    "lr",
                    self.optimizer.param_groups[0]["lr"],
                    global_step,
                )
                pbar.set_postfix({"loss": f"{display_loss:.4f}"})

    # ==========================================================
    # VALIDATION
    # ==========================================================

    @torch.no_grad()
    def _validate(self, epoch):

        torch.cuda.empty_cache()
        self.model.eval()
        if self.ema:
            self.ema.apply_shadow()

        try:
            total_loss = 0.0
            total_samples = 0

            for batch in self.val_loader:

                x = batch["image"]
                if hasattr(x, "as_tensor"):
                    x = x.as_tensor()
                x = x.to(self.device)

                latents = self._encode_inputs(x)
                latents = latents * self.scale_factor

                noise, t, noisy_latents, t_model = self._sample_t_and_noisy(latents)
                target = self._get_target(latents, noise, t)

                with amp.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
                    # No self-conditioning during validation (faster, clean loss).
                    noise_pred = self.model(noisy_latents, t=t_model, y=None)
                    loss = self._compute_loss(noise_pred, target, t)

                total_loss += loss.item() * latents.size(0)
                total_samples += latents.size(0)

            totals = torch.tensor([total_loss, total_samples], device=self.device, dtype=torch.float64)
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(totals)
            if totals[1] == 0:
                raise ValueError("Validation dataset is empty")
            total_loss = (totals[0] / totals[1]).item()

            if self.is_main:
                self.writer_val.add_scalar("loss", total_loss, epoch)
                print(f"Validation Loss: {total_loss:.6f}")

        finally:
            if self.ema:
                self.ema.restore()

        return total_loss

    # ==========================================================
    # HELPERS — timestep sampling and noise
    # ==========================================================

    def _sample_t_and_noisy(self, latents):
        """
        Sample timesteps and produce noisy latents.

        Returns
        -------
        noise        : Gaussian noise eps (+ optional offset), same shape as latents.
        t            : timestep tensor (B,).
                       Float in (0,1) for flow matching; Long integer for DDPM.
        noisy_latents: noisy version of latents at t.
        t_model      : what to pass to the DiT timestep embedder.
        """
        B = latents.shape[0]

        if self._is_flow_matching:
            t = self.scheduler.sample_timesteps(B, device=self.device)
            noise = torch.randn_like(latents)
            noisy_latents = self.scheduler.add_noise(latents, noise, t)
            t_model = t * self.scheduler.num_train_timesteps
        else:
            t = torch.randint(
                0,
                self.scheduler.num_train_timesteps,
                (B,),
                device=self.device,
            ).long()

            noise = torch.randn_like(latents)

            # Offset noise: adds a spatially-uniform per-channel perturbation.
            # Prevents the model's high-noise inputs from having near-zero mean,
            # which would make it impossible to learn global CT intensity contrast.
            if self.offset_noise_strength > 0.0:
                offset = torch.randn(
                    B, latents.shape[1], 1, 1, 1, device=self.device
                )
                noise = noise + self.offset_noise_strength * offset

            noisy_latents = self.scheduler.add_noise(
                original_samples=latents,
                noise=noise,
                timesteps=t,
            )
            t_model = t

        return noise, t, noisy_latents, t_model

    # ==========================================================
    # HELPERS — training target
    # ==========================================================

    def _get_target(self, latents, noise, timesteps):
        """
        Return the tensor the model should predict.

        flow_matching : v = eps - x_0  (constant velocity along straight path)
        v_prediction  : v = sqrt(ᾱ) * eps - sqrt(1-ᾱ) * x_0
        epsilon       : eps (the added noise)
        """
        pt = self.scheduler.prediction_type

        if pt == "flow_matching":
            return self.scheduler.get_velocity(latents, noise)

        elif pt == "v_prediction":
            return self.scheduler.get_velocity(latents, noise, timesteps)

        elif pt == "epsilon":
            return noise

        else:
            raise ValueError(f"Unknown prediction_type '{pt}'")

    # ==========================================================
    # HELPERS — x0 estimate from model prediction (self-conditioning)
    # ==========================================================

    def _x0_from_pred(self, pred, x_t, t):
        """
        Derive a clean-latent estimate x0 from the model's prediction.

        Used for self-conditioning.  The formula depends on the prediction type.
        """
        if self._is_flow_matching:
            # x_t = (1-t)*x0 + t*eps  and  v = eps - x0
            # → x0 = x_t - t * v
            t_bc = _unsqueeze_right(t.float(), x_t.ndim)
            return x_t - t_bc * pred

        pt = self.scheduler.prediction_type
        acp = self.scheduler.alphas_cumprod.to(device=x_t.device)

        if pt == "v_prediction":
            # v = sqrt(ᾱ)*eps - sqrt(1-ᾱ)*x0
            # x_t = sqrt(ᾱ)*x0 + sqrt(1-ᾱ)*eps
            # → x0 = sqrt(ᾱ)*x_t - sqrt(1-ᾱ)*v
            sqrt_acp = _unsqueeze_right(acp[t].sqrt(), x_t.ndim)
            sqrt_1m = _unsqueeze_right((1.0 - acp[t]).sqrt(), x_t.ndim)
            return sqrt_acp * x_t - sqrt_1m * pred

        elif pt == "epsilon":
            sqrt_acp = _unsqueeze_right(acp[t].sqrt(), x_t.ndim)
            sqrt_1m = _unsqueeze_right((1.0 - acp[t]).sqrt(), x_t.ndim)
            return (x_t - sqrt_1m * pred) / sqrt_acp.clamp(min=1e-8)

        else:
            return pred  # "sample" prediction — model already predicts x0

    # ==========================================================
    # HELPERS — Min-SNR weighted loss
    # ==========================================================

    def _compute_loss(self, pred, target, t):
        """
        Smooth-L1 loss with optional Min-SNR-γ weighting.

        For flow matching: w(t) = min(SNR, γ) / SNR,  SNR = (1-t)²/t².
        For v-prediction : w(t) = min(SNR, γ) / (SNR + 1).
        For ε-prediction : w(t) = min(SNR, γ) / SNR.
        """
        element_loss = F.smooth_l1_loss(pred.float(), target.float(), reduction="none")
        batch_loss = element_loss.mean(dim=list(range(1, element_loss.ndim)))  # (B,)

        if self.min_snr_gamma > 0.0:
            weights = self._min_snr_weights(t)
            batch_loss = batch_loss * weights

        return batch_loss.mean()

    def _min_snr_weights(self, t):
        """Compute per-sample Min-SNR-γ weights."""
        gamma = self.min_snr_gamma

        if self._is_flow_matching:
            t_safe = t.clamp(min=1e-5)
            snr = ((1.0 - t_safe) / t_safe) ** 2
        else:
            acp = self.scheduler.alphas_cumprod.to(device=t.device)
            snr = acp[t] / (1.0 - acp[t])

        if getattr(self.scheduler, "prediction_type", None) == "v_prediction":
            denom = snr + 1.0
        else:
            denom = snr.clamp(min=1e-8)

        return torch.clamp(snr, max=gamma) / denom.clamp(min=1e-8)

    # ==========================================================
    # HELPERS — encoding
    # ==========================================================

    def _raw_encode(self, x):
        """Encode x through stage1 without applying latent normalisation."""
        if self.stage1 is None:
            return x
        with torch.no_grad():
            return self.stage1.encode_stage_2_inputs(x)

    def _encode_inputs(self, x):
        """Encode and optionally apply per-channel normalisation."""
        latents = self._raw_encode(x)
        if self.latent_mean is not None:
            latents = (latents - self.latent_mean) / (self.latent_std + 1e-8)
        return latents

    def load_ema_state(self, state_dict):
        if self.ema and state_dict is not None:
            self.ema.load_state_dict(state_dict)

    # ==========================================================
    # HELPERS — per-channel latent statistics
    # ==========================================================

    def _compute_latent_stats(self):
        """
        Two-pass computation of per-channel mean and std over the training set.

        Returns (mean, std) each of shape (1, C, 1, 1, 1).
        """
        with torch.no_grad():
            # Pass 1: channel-wise sum and element count.
            chan_sum = None
            n_elements = 0

            for batch in self.train_loader:
                x = batch["image"]
                if hasattr(x, "as_tensor"):
                    x = x.as_tensor()
                x = x.to(self.device)
                latents = self._raw_encode(x).float()
                B, C = latents.shape[:2]
                n_spatial = latents.shape[2] * latents.shape[3] * latents.shape[4]
                flat = latents.view(B, C, n_spatial)

                batch_sum = flat.sum(dim=(0, 2))
                chan_sum = batch_sum if chan_sum is None else chan_sum + batch_sum
                n_elements += B * n_spatial

            count = torch.tensor(float(n_elements), device=self.device, dtype=torch.float64)
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(chan_sum)
                torch.distributed.all_reduce(count)
            if count == 0:
                raise ValueError("Cannot normalize an empty training dataset")
            mean = chan_sum / count  # (C,)

            # Pass 2: channel-wise sum of squared deviations.
            chan_sq = torch.zeros_like(mean)

            for batch in self.train_loader:
                x = batch["image"]
                if hasattr(x, "as_tensor"):
                    x = x.as_tensor()
                x = x.to(self.device)
                latents = self._raw_encode(x).float()
                B, C = latents.shape[:2]
                n_spatial = latents.shape[2] * latents.shape[3] * latents.shape[4]
                flat = latents.view(B, C, n_spatial)
                chan_sq = chan_sq + ((flat - mean.view(1, C, 1)) ** 2).sum(dim=(0, 2))

            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(chan_sq)
            std = (chan_sq / count).sqrt().float()
            mean = mean.float()

        return mean.view(1, -1, 1, 1, 1), std.view(1, -1, 1, 1, 1)

    # ==========================================================
    # CHECKPOINTING
    # ==========================================================

    def _save_best_checkpoint(self, epoch, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            torch.save(self._build_checkpoint(epoch), self.run_dir / "best_model.pth")

    def _save_periodic_checkpoint(self, epoch):
        ckpt = self._build_checkpoint(epoch)
        torch.save(ckpt, self.run_dir / f"checkpoint_epoch_{epoch}.pth")
        torch.save(ckpt, self.run_dir / "last_checkpoint.pth")

    def _save_final_model(self):
        torch.save(self._build_checkpoint(self.n_epochs - 1), self.run_dir / "final_model.pth")

    def _build_checkpoint(self, epoch):
        ckpt = {
            "epoch": epoch,
            "best_loss": self.best_loss,
            "model": self._get_model_state(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            "ema": self.ema.state_dict() if self.ema else None,
        }
        # Save latent normalisation stats so they can be used at inference time.
        if self.latent_mean is not None:
            ckpt["latent_mean"] = self.latent_mean.cpu()
            ckpt["latent_std"] = self.latent_std.cpu()
        return ckpt

    def _get_model_state(self):
        return (
            self.model.module.state_dict()
            if hasattr(self.model, "module")
            else self.model.state_dict()
        )
