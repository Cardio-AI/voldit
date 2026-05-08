# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import numpy as np
import torch
from monai.utils import StrEnum

from .scheduler import Scheduler


class DDIMPredictionType(StrEnum):
    EPSILON = "epsilon"
    SAMPLE = "sample"
    V_PREDICTION = "v_prediction"


class DDIMScheduler(Scheduler):
    """
    DDIM scheduler — deterministic (eta=0) or stochastic (eta>0) denoising.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        schedule: str = "linear_beta",
        clip_sample: bool = False,
        set_alpha_to_one: bool = True,
        steps_offset: int = 0,
        prediction_type: str = DDIMPredictionType.EPSILON,
        **schedule_args,
    ) -> None:
        super().__init__(num_train_timesteps, schedule, **schedule_args)

        if prediction_type not in DDIMPredictionType.__members__.values():
            raise ValueError("Argument `prediction_type` must be a member of DDIMPredictionType")

        self.prediction_type = prediction_type
        self.final_alpha_cumprod = torch.tensor(1.0) if set_alpha_to_one else self.alphas_cumprod[0]
        self.first_alpha_cumprod = torch.tensor(0.0) if set_alpha_to_one else self.alphas_cumprod[-1]
        self.init_noise_sigma = 1.0
        self.timesteps = torch.from_numpy(np.arange(0, self.num_train_timesteps)[::-1].astype(np.int64))
        self.clip_sample = clip_sample
        self.steps_offset = steps_offset
        self.set_timesteps(self.num_train_timesteps)

    def set_timesteps(self, num_inference_steps: int, device: str | torch.device | None = None) -> None:
        if num_inference_steps > self.num_train_timesteps:
            raise ValueError(
                f"`num_inference_steps`: {num_inference_steps} cannot be larger than `self.num_train_timesteps`:"
                f" {self.num_train_timesteps}."
            )

        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // self.num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps).to(device)
        self.timesteps += self.steps_offset

    def _get_variance(self, timestep: int, prev_timestep: torch.Tensor) -> torch.Tensor:
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        return variance

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        eta: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prev_timestep = timestep - self.num_train_timesteps // self.num_inference_steps

        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t

        if self.prediction_type == DDIMPredictionType.EPSILON:
            pred_original_sample = (sample - (beta_prod_t**0.5) * model_output) / (alpha_prod_t**0.5)
            pred_epsilon = model_output
        elif self.prediction_type == DDIMPredictionType.SAMPLE:
            pred_original_sample = model_output
            pred_epsilon = (sample - (alpha_prod_t**0.5) * pred_original_sample) / (beta_prod_t**0.5)
        elif self.prediction_type == DDIMPredictionType.V_PREDICTION:
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
            pred_epsilon = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample

        if self.clip_sample:
            pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

        variance = self._get_variance(timestep, prev_timestep)
        std_dev_t = eta * variance**0.5
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** 0.5 * pred_epsilon
        pred_prev_sample = alpha_prod_t_prev**0.5 * pred_original_sample + pred_sample_direction

        if eta > 0:
            device = model_output.device if torch.is_tensor(model_output) else "cpu"
            noise = torch.randn(model_output.shape, dtype=model_output.dtype, generator=generator).to(device)
            variance = self._get_variance(timestep, prev_timestep) ** 0.5 * eta * noise
            pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample, pred_original_sample
