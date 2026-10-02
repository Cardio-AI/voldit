"""
DPM-Solver++ 2nd-order multistep (2M) scheduler.

Implements the data-prediction variant of Algorithm 1 from:
    Lu et al. (2022). DPM-Solver++: Fast Solver for Guided Sampling
    of Diffusion Probabilistic Models. arXiv:2211.01095

Drop-in inference replacement for DDIMScheduler — no retraining needed.
Supports epsilon-prediction and v_prediction on any noise schedule.

Compatibility note (discrete vs continuous timesteps)
------------------------------------------------------
DPM-Solver++ was derived for continuous-time diffusion models where the
noise schedule is a smooth analytic function of time.  For discrete-
timestep models (trained with DDPMScheduler using integer indices), this
scheduler operates in log-SNR space using the stored alphas_cumprod
values at the selected inference timesteps — effectively treating the
discrete schedule as samples from the underlying continuous curve.

Select the inference step count empirically for the checkpoint and dataset.
This implementation does not establish a quality-equivalent speedup over DDPM
or DDIM; its discretization and endpoint behavior require matched evaluation.
"""

from __future__ import annotations

import numpy as np
import torch

from .scheduler import Scheduler


class DPMSolverPPScheduler(Scheduler):
    """
    DPM-Solver++ 2M scheduler for fast inference.

    Acts as a drop-in replacement for DDIMScheduler at inference time.
    The same DDPM-trained checkpoint is used without any changes.

    Example
    -------
    >>> scheduler = DPMSolverPPScheduler(**config.scheduler)
    >>> scheduler.set_timesteps(20)           # 20 NFE
    >>> for t in scheduler.timesteps:
    ...     t_batch = torch.full((B,), t, device=device, dtype=torch.long)
    ...     v_pred = dit(x, t=t_batch, y=None)
    ...     x, x0 = scheduler.step(v_pred, t, x)
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        schedule: str = "cosine",
        prediction_type: str = "v_prediction",
        solver_order: int = 2,
        clip_sample: bool = False,
        clip_sample_range: float = 1.0,
        **schedule_args,
    ) -> None:
        """
        Parameters
        ----------
        num_train_timesteps : int
            Number of training timesteps (must match the trained model).
        schedule : str
            Noise schedule name passed to the base Scheduler ("cosine",
            "linear_beta", etc.).  Must match the training schedule.
        prediction_type : str
            "v_prediction" or "epsilon".  Must match the trained model.
        solver_order : int
            1 = first-order Euler (equivalent to DDIM deterministic),
            2 = second-order multistep (default, recommended).
        clip_sample : bool
            Whether to clamp the x0 prediction to [-clip_sample_range,
            +clip_sample_range] after each step.  Can help with latent
            space stability; disabled by default.
        clip_sample_range : float
            Symmetric clamp range when clip_sample=True.
        """
        super().__init__(num_train_timesteps, schedule, **schedule_args)

        if prediction_type not in ("epsilon", "v_prediction"):
            raise ValueError(
                f"DPMSolverPPScheduler only supports 'epsilon' and "
                f"'v_prediction', got '{prediction_type}'."
            )

        self.prediction_type = prediction_type
        self.solver_order = solver_order
        self.clip_sample = clip_sample
        self.clip_sample_range = clip_sample_range

        # Multistep buffer — reset by set_timesteps()
        self._x0_buffer: list[torch.Tensor] = []

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def set_timesteps(self, num_inference_steps: int, device=None) -> None:
        """
        Select evenly-spaced timesteps from the training schedule,
        ordered from most noisy (high t) to least noisy (low t).

        Parameters
        ----------
        num_inference_steps : int
            Number of denoising steps (= number of model evaluations).
            Choose experimentally for the checkpoint.
        """
        if num_inference_steps < 1:
            raise ValueError("num_inference_steps must be positive")
        if num_inference_steps > self.num_train_timesteps:
            raise ValueError(
                f"num_inference_steps ({num_inference_steps}) must be "
                f"<= num_train_timesteps ({self.num_train_timesteps})."
            )
        self.num_inference_steps = num_inference_steps

        step_ratio = self.num_train_timesteps // num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps.copy()).to(device)

        # Reset the x0 buffer so stale state from a previous run is cleared
        self._x0_buffer = []

    # ------------------------------------------------------------------
    # Inference step
    # ------------------------------------------------------------------

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int | torch.Tensor,
        sample: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        One DPM-Solver++ 2M denoising step.

        Parameters
        ----------
        model_output : (B, C, ...) — DiT output (v_pred or eps_pred).
        timestep     : int — current training-scale timestep t_n.
        sample       : (B, C, ...) — current noisy latent x_{t_n}.

        Returns
        -------
        prev_sample          : (B, C, ...) — denoised latent x_{t_{n-1}}.
        pred_original_sample : (B, C, ...) — x_0 estimate at this step.
        """
        t_n = int(timestep)

        # ── 1. Convert model output → x0 prediction ─────────────────────
        x0_pred = self._predict_x0(model_output, t_n, sample)
        if self.clip_sample:
            x0_pred = torch.clamp(x0_pred, -self.clip_sample_range, self.clip_sample_range)

        # ── 2. Update multistep x0 buffer ───────────────────────────────
        self._x0_buffer.append(x0_pred)
        if len(self._x0_buffer) > self.solver_order:
            self._x0_buffer.pop(0)

        # ── 3. Find t_{n-1} in the stored timestep sequence ─────────────
        idx = (self.timesteps == t_n).nonzero(as_tuple=True)[0].item()
        if idx == len(self.timesteps) - 1:
            return x0_pred, x0_pred
        # The next entry in the sequence is the denoising target timestep.
        # At the last step, t_{n-1} = 0.
        t_prev = int(self.timesteps[idx + 1]) if idx + 1 < len(self.timesteps) else 0

        # ── 4. Schedule quantities at t_n and t_{n-1} ───────────────────
        acp = self.alphas_cumprod.to(sample.device)

        alpha_t  = acp[t_n].sqrt()
        sigma_t  = (1.0 - acp[t_n]).sqrt()
        alpha_s  = acp[t_prev].sqrt()
        sigma_s  = (1.0 - acp[t_prev]).sqrt()

        # λ = log(α/σ) — the "log-SNR half" used by DPM-Solver++.
        # λ increases as noise decreases (towards clean data).
        lambda_t = torch.log(alpha_t / sigma_t)
        lambda_s = torch.log(alpha_s / sigma_s)
        h = lambda_s - lambda_t   # > 0 (going to less noise)

        # ── 5. Effective x0 for DPM-Solver++ update ─────────────────────
        if len(self._x0_buffer) == 1 or self.solver_order == 1:
            # First step (or order-1 mode): plain Euler in DPM-Solver++ form
            D = self._x0_buffer[-1]
        else:
            # Second step onward: 2nd-order multistep correction.
            #
            # D = (1 + 1/(2r)) * D0 - (1/(2r)) * D1
            # where D0 = current x0_pred, D1 = previous x0_pred,
            # and r = h_prev / h (ratio of consecutive log-SNR step sizes).
            D0 = self._x0_buffer[-1]   # x0_pred at t_n      (current)
            D1 = self._x0_buffer[-2]   # x0_pred at t_{n-1}  (previous step's start)

            # h_prev: log-SNR step size of the preceding denoising step.
            # The preceding step started at self.timesteps[idx - 1].
            t_n_prev = int(self.timesteps[idx - 1]) if idx > 0 else t_n
            alpha_tn_prev  = acp[t_n_prev].sqrt()
            sigma_tn_prev  = (1.0 - acp[t_n_prev]).sqrt()
            lambda_tn_prev = torch.log(alpha_tn_prev / sigma_tn_prev)
            h_prev = lambda_t - lambda_tn_prev   # > 0

            r = h_prev / h   # step-size ratio in log-SNR space
            D = (1.0 + 1.0 / (2.0 * r)) * D0 - (1.0 / (2.0 * r)) * D1

        # ── 6. DPM-Solver++ 2M update formula ───────────────────────────
        #
        # x_s = (σ_s / σ_t) * x_t  −  α_s * expm1(-h) * D
        #
        # Note: expm1(-h) = exp(-h) - 1 < 0  (h > 0), so
        #       -expm1(-h) > 0  — the D term adds signal, removes noise.
        prev_sample = (sigma_s / sigma_t) * sample - alpha_s * torch.expm1(-h) * D

        return prev_sample, x0_pred

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _predict_x0(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert the DiT's output to a clean-sample (x0) prediction.

        For v_prediction:  x0 = α_t * x_t  − σ_t * v_pred
        For epsilon:        x0 = (x_t − σ_t * ε_pred) / α_t
        """
        acp = self.alphas_cumprod.to(sample.device, sample.dtype)
        alpha_t = acp[timestep].sqrt()
        sigma_t = (1.0 - acp[timestep]).sqrt()

        if self.prediction_type == "v_prediction":
            return alpha_t * sample - sigma_t * model_output
        elif self.prediction_type == "epsilon":
            return (sample - sigma_t * model_output) / alpha_t
        else:
            raise ValueError(f"Unknown prediction_type '{self.prediction_type}'")
