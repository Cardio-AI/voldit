"""
Flow Matching (FM) / Rectified Flow scheduler for training and sampling.

Forward process — linear interpolation (optimal transport path):
    x_t = (1 - t) * x_0 + t * eps,   t in [0, 1]

Model predicts the constant velocity field:
    v_target = eps - x_0              (direction from data to noise)

Inference — Euler ODE solved in reverse (t: 1 → 0):
    x_{t-dt} = x_t - dt * v_pred(x_t, t)

x0 estimate from v_pred (derived from x_t = x_0 + t*v):
    x0_pred = x_t - t * v_pred

Timestep embedding note
-----------------------
The DiT's TimestepEmbedder uses sinusoidal frequencies tuned for
integer values in [0, T].  For FM, training timesteps are floats in
(0, 1).  To keep the embedding well-conditioned, pass
    t_embed = t * num_train_timesteps
to the model, not t itself.  This is handled by DiTTrainer; the
scheduler itself always works with raw t in (0, 1).

References
----------
Liu et al. (2022). Flow Straight and Fast. arXiv:2209.03003
Lipman et al. (2022). Flow Matching for Generative Modeling. arXiv:2210.02747
Esser et al. (2024). Scaling Rectified Flow Transformers (SD3). arXiv:2403.03206
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.utils import unsqueeze_right


class FlowMatchingScheduler(nn.Module):
    """
    Scheduler for Flow Matching / Rectified Flow latent diffusion.

    Replaces DDPMScheduler for training + inference.  The API mirrors
    DDPMScheduler so that DiTTrainer can drive both without structural
    changes to the training loop.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        sample_method: str = "logit_normal",
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
    ) -> None:
        """
        Parameters
        ----------
        num_train_timesteps : int
            Determines the scaling factor applied to t before the sinusoidal
            timestep embedder: t_embed = t * num_train_timesteps.
            Does NOT define a discrete set of training steps — FM training
            timesteps are continuous floats in (0, 1).
        sample_method : str
            How to draw training timesteps t ~ p(t):
            - "logit_normal" : t = sigmoid(N(logit_mean, logit_std^2))
              Concentrates mass near 0.5 (the hardest trajectory region).
              Default training-time distribution for this implementation.
            - "uniform" : t ~ U(0, 1).  Baseline reference.
        logit_mean : float
            Mean of the pre-sigmoid normal for logit-normal sampling.
        logit_std : float
            Std dev of the pre-sigmoid normal for logit-normal sampling.
        """
        super().__init__()
        self.num_train_timesteps = num_train_timesteps
        self.prediction_type = "flow_matching"
        self.sample_method = sample_method
        self.logit_mean = logit_mean
        self.logit_std = logit_std

        # Set by set_timesteps() before inference
        self.num_inference_steps: int | None = None
        self.timesteps: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Training interface
    # ------------------------------------------------------------------

    def sample_timesteps(self, batch_size: int, device: torch.device | str) -> torch.Tensor:
        """
        Sample continuous t in (0, 1) for one training batch.

        logit_normal concentrates training budget near t=0.5 where the
        velocity field is hardest to learn (neither almost-clean nor
        almost-noise), mirroring the Min-SNR rationale for DDPM.

        Returns
        -------
        t : FloatTensor of shape (batch_size,) with values in (0, 1).
        """
        if self.sample_method == "logit_normal":
            u = torch.randn(batch_size, device=device) * self.logit_std + self.logit_mean
            return torch.sigmoid(u)
        elif self.sample_method == "uniform":
            return torch.rand(batch_size, device=device)
        else:
            raise ValueError(
                f"Unknown sample_method '{self.sample_method}'. "
                "Choose 'logit_normal' or 'uniform'."
            )

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Linear interpolation forward process:
            x_t = (1 - t) * x_0 + t * eps

        Parameters
        ----------
        original_samples : (B, C, ...) — clean latents x_0.
        noise            : (B, C, ...) — standard Gaussian noise eps.
        timesteps        : (B,) FloatTensor in (0, 1).

        Returns
        -------
        noisy_samples : (B, C, ...) — interpolated noisy latents at t.
        """
        t = unsqueeze_right(
            timesteps.to(original_samples.device, original_samples.dtype),
            original_samples.ndim,
        )
        return (1.0 - t) * original_samples + t * noise

    def get_velocity(
        self,
        sample: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Constant FM velocity target: v = eps - x_0.

        The velocity is the same at every t along the straight path, so
        ``timesteps`` is not used.  It is accepted for API compatibility
        with DDPMScheduler.get_velocity().
        """
        return noise - sample

    # ------------------------------------------------------------------
    # Inference interface
    # ------------------------------------------------------------------

    def set_timesteps(self, num_inference_steps: int, device=None) -> None:
        """
        Build a uniform Euler grid from t=1 down to t=dt
        where dt = 1 / num_inference_steps.

        The grid intentionally excludes t=0 because the FM model is
        trained to predict the velocity everywhere in (0, 1); at t=0
        the sample is already the clean latent and no step is needed.

        Parameters
        ----------
        num_inference_steps : int
            Number of function evaluations (NFE). Select by measured quality/cost.
        """
        self.num_inference_steps = num_inference_steps
        dt = 1.0 / num_inference_steps
        # [1.0, 1-dt, 1-2*dt, ..., dt] — high noise first
        t_grid = torch.linspace(1.0, dt, num_inference_steps)
        self.timesteps = t_grid.to(device) if device is not None else t_grid

    def step(
        self,
        model_output: torch.Tensor,
        timestep: float | torch.Tensor,
        sample: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Euler ODE step in the reverse (denoising) direction:
            x_{t-dt} = x_t - dt * v_pred

        Parameters
        ----------
        model_output : (B, C, ...) — velocity predicted by the DiT.
        timestep     : scalar float in (0, 1] — current t.
        sample       : (B, C, ...) — current latent x_t.

        Returns
        -------
        prev_sample          : (B, C, ...) — latent at t - dt.
        pred_original_sample : (B, C, ...) — x_0 estimate = x_t - t * v.
        """
        dt = 1.0 / self.num_inference_steps
        t = float(timestep)

        prev_sample = sample - dt * model_output
        # x_0 estimate: rearranging x_t = x_0 + t * v gives x_0 = x_t - t * v
        pred_original_sample = sample - t * model_output

        return prev_sample, pred_original_sample
