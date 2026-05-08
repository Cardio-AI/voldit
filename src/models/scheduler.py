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

import torch
import torch.nn as nn

from src.models.utils import ComponentStore, unsqueeze_right

NoiseSchedules = ComponentStore("NoiseSchedules", "Functions to generate noise schedules")


@NoiseSchedules.add_def("linear_beta", "Linear beta schedule")
def _linear_beta(num_train_timesteps: int, beta_start: float = 1e-4, beta_end: float = 2e-2):
    return torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)


@NoiseSchedules.add_def("scaled_linear_beta", "Scaled linear beta schedule")
def _scaled_linear_beta(num_train_timesteps: int, beta_start: float = 1e-4, beta_end: float = 2e-2):
    return torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2


@NoiseSchedules.add_def("sigmoid_beta", "Sigmoid beta schedule")
def _sigmoid_beta(num_train_timesteps: int, beta_start: float = 1e-4, beta_end: float = 2e-2, sig_range: float = 6):
    betas = torch.linspace(-sig_range, sig_range, num_train_timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


@NoiseSchedules.add_def("cosine", "Cosine schedule")
def _cosine_beta(num_train_timesteps: int, s: float = 8e-3):
    x = torch.linspace(0, num_train_timesteps, num_train_timesteps + 1, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / num_train_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = torch.clip(betas, 0, 0.9999)
    return betas


class Scheduler(nn.Module):
    """
    Base class for schedulers based on a noise schedule function.
    """

    def __init__(self, num_train_timesteps: int = 1000, schedule: str = "linear_beta", **schedule_args) -> None:
        super().__init__()
        schedule_args["num_train_timesteps"] = num_train_timesteps
        noise_sched = NoiseSchedules[schedule](**schedule_args)

        if isinstance(noise_sched, tuple):
            self.betas, self.alphas, self.alphas_cumprod = noise_sched
        else:
            self.betas = noise_sched
            self.alphas = 1.0 - self.betas
            self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        self.num_train_timesteps = num_train_timesteps
        self.one = torch.tensor(1.0)

        self.num_inference_steps = None
        self.timesteps = torch.arange(num_train_timesteps - 1, -1, -1)

    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        self.alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_cumprod = unsqueeze_right(self.alphas_cumprod[timesteps] ** 0.5, original_samples.ndim)
        sqrt_one_minus_alpha_prod = unsqueeze_right((1 - self.alphas_cumprod[timesteps]) ** 0.5, original_samples.ndim)

        noisy_samples = sqrt_alpha_cumprod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def get_velocity(self, sample: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        self.alphas_cumprod = self.alphas_cumprod.to(device=sample.device, dtype=sample.dtype)
        timesteps = timesteps.to(sample.device)

        sqrt_alpha_prod = unsqueeze_right(self.alphas_cumprod[timesteps] ** 0.5, sample.ndim)
        sqrt_one_minus_alpha_prod = unsqueeze_right((1 - self.alphas_cumprod[timesteps]) ** 0.5, sample.ndim)

        velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
        return velocity
