import torch
from torch import amp
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from src.models.ema import EMA


class ControlNetDiTTrainer:
    """
    Trainer for ControlNet3D.

    Unlike the UNet-based ControlNetTrainer, ControlNet3D handles the entire
    forward pass internally (including the frozen base DiT), so there is no
    separate diffusion model member here.
    """

    def __init__(
        self,
        controlnet,
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

        self.controlnet = controlnet
        self.stage1 = stage1
        self.scheduler = scheduler

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

        self.start_epoch = start_epoch
        self.best_loss = best_loss

        self.use_precomputed_latents = config.training.get("use_precomputed_latents", False)

        self.condition_keys = config.controlnet.condition_keys

        self.control_dropout = config.training.get("control_dropout", 0.0)
        self.spatial_dropout_prob = config.training.get("spatial_dropout_prob", 0.0)
        self.spatial_dropout_patch_size = config.training.get("spatial_dropout_patch_size", 16)

        self.use_ema = config.training.get("use_ema", True)
        self.ema_decay = config.training.get("ema_decay", 0.9999)

        if self.use_ema:
            raw_model = (
                self.controlnet.module
                if hasattr(self.controlnet, "module")
                else self.controlnet
            )
            self.ema = EMA(raw_model, decay=self.ema_decay)
        else:
            self.ema = None

        self.scaler = amp.GradScaler("cuda")

    # ==========================================================
    # TRAIN LOOP
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

                if self.is_main:
                    val_loss = self._validate(epoch)
                    self._save_best_checkpoint(epoch, val_loss)
                    self._save_periodic_checkpoint(epoch)

                if torch.distributed.is_initialized():
                    torch.distributed.barrier()

        if self.is_main:
            self._save_final_model()

    # ==========================================================
    # TRAIN EPOCH
    # ==========================================================

    def _train_epoch(self, epoch):

        self.controlnet.train()

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}", disable=not self.is_main)

        for step, batch in enumerate(pbar):

            latents = self._get_latents(batch)
            cond = self._get_condition(batch, training=True)

            latents = latents * self.scale_factor

            noise = torch.randn_like(latents)

            timesteps = torch.randint(
                0,
                self.scheduler.num_train_timesteps,
                (latents.shape[0],),
                device=self.device,
            ).long()

            self.optimizer.zero_grad(set_to_none=True)

            with amp.autocast(device_type="cuda"):

                noisy_latents = self.scheduler.add_noise(
                    latents,
                    noise,
                    timesteps,
                )

                noise_pred = self.controlnet(
                    noisy_latents,
                    t=timesteps,
                    y=None,
                    control_input=cond,
                )

                target = self._get_target(latents, noise, timesteps)

                loss = F.smooth_l1_loss(
                    noise_pred.float(),
                    target.float(),
                )

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)

            trainable_params = [p for p in self.controlnet.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.ema:
                self.ema.update()

            if self.is_main:

                global_step = epoch * len(self.train_loader) + step

                self.writer_train.add_scalar("loss", loss.item(), global_step)

                self.writer_train.add_scalar(
                    "lr",
                    self.optimizer.param_groups[0]["lr"],
                    global_step,
                )

                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    # ==========================================================
    # VALIDATION
    # ==========================================================

    @torch.no_grad()
    def _validate(self, epoch):

        torch.cuda.empty_cache()

        self.controlnet.eval()

        if self.ema:
            self.ema.apply_shadow()

        total_loss = 0.0

        try:
            for batch in self.val_loader:

                latents = self._get_latents(batch)
                cond = self._get_condition(batch, training=False)

                latents = latents * self.scale_factor

                noise = torch.randn_like(latents)

                timesteps = torch.randint(
                    0,
                    self.scheduler.num_train_timesteps,
                    (latents.shape[0],),
                    device=self.device,
                ).long()

                with amp.autocast(device_type="cuda"):

                    noisy_latents = self.scheduler.add_noise(
                        latents,
                        noise,
                        timesteps,
                    )

                    noise_pred = self.controlnet(
                        noisy_latents,
                        t=timesteps,
                        y=None,
                        control_input=cond,
                    )

                    target = self._get_target(latents, noise, timesteps)

                    loss = F.smooth_l1_loss(
                        noise_pred.float(),
                        target.float(),
                    )

                total_loss += loss.item() * latents.size(0)

            total_loss /= len(self.val_loader.dataset)

            if self.is_main:
                self.writer_val.add_scalar("loss", total_loss, epoch)
                print(f"Validation Loss: {total_loss:.6f}")

        finally:
            if self.ema:
                self.ema.restore()

        return total_loss

    # ==========================================================
    # HELPERS
    # ==========================================================

    def apply_patch_dropout(self, control, p=0.2, patch_size=16):

        if p <= 0:
            return control

        B, C, H, W, D = control.shape
        mask = torch.rand(B, 1, H//patch_size, W//patch_size, D//patch_size, device=control.device)
        mask = (mask > p).float()
        mask = F.interpolate(mask, size=(H, W, D), mode="nearest")

        return control * mask

    def load_ema_state(self, state_dict):
        if self.ema and state_dict is not None:
            self.ema.load_state_dict(state_dict)

    def _get_latents(self, batch):

        if self.use_precomputed_latents:
            z = batch["image"]
            if hasattr(z, "as_tensor"):
                z = z.as_tensor()
            return z.to(self.device)

        x = batch["image"]
        if hasattr(x, "as_tensor"):
            x = x.as_tensor()
        x = x.to(self.device)

        with torch.no_grad():
            return self.stage1.encode_stage_2_inputs(x)

    def _get_condition(self, batch, training=True):

        conditions = []

        for key in self.condition_keys:

            c = batch[key]

            if hasattr(c, "as_tensor"):
                c = c.as_tensor()

            c = c.float()

            if training and self.control_dropout > 0:
                if torch.rand(1).item() < self.control_dropout:
                    c = torch.zeros_like(c)

            if training and self.spatial_dropout_prob > 0:
                c = self.apply_patch_dropout(
                    c,
                    p=self.spatial_dropout_prob,
                    patch_size=self.spatial_dropout_patch_size,
                )

            conditions.append(c)

        cond = torch.cat(conditions, dim=1)

        return cond.to(self.device)

    def _get_target(self, latents, noise, timesteps):

        if self.scheduler.prediction_type == "epsilon":
            return noise

        elif self.scheduler.prediction_type == "v_prediction":
            return self.scheduler.get_velocity(latents, noise, timesteps)

        else:
            raise ValueError(f"Unknown prediction type {self.scheduler.prediction_type}")

    # ==========================================================
    # CHECKPOINTING
    # ==========================================================

    def _save_best_checkpoint(self, epoch, val_loss):

        if val_loss < self.best_loss:

            self.best_loss = val_loss

            torch.save(
                {
                    "epoch": epoch,
                    "best_loss": self.best_loss,
                    "model": self._get_model_state(),
                    "optimizer": self.optimizer.state_dict(),
                    "lr_scheduler": self.lr_scheduler.state_dict()
                    if self.lr_scheduler
                    else None,
                    "ema": self.ema.state_dict() if self.ema else None,
                },
                self.run_dir / "best_model.pth",
            )

    def _save_periodic_checkpoint(self, epoch):

        checkpoint = {
            "epoch": epoch,
            "best_loss": self.best_loss,
            "model": self._get_model_state(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict()
            if self.lr_scheduler
            else None,
            "ema": self.ema.state_dict() if self.ema else None,
        }

        torch.save(checkpoint, self.run_dir / f"checkpoint_epoch_{epoch}.pth")
        torch.save(checkpoint, self.run_dir / "last_checkpoint.pth")

    def _save_final_model(self):

        torch.save(
            self._get_model_state(),
            self.run_dir / "final_model.pth",
        )

    def _get_model_state(self):

        return (
            self.controlnet.module.state_dict()
            if hasattr(self.controlnet, "module")
            else self.controlnet.state_dict()
        )
