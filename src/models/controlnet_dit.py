# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Sequence, Union, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks import Convolution

from src.models.dit import DiT3D


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class ControlNetConditioningEmbedding(nn.Module):
    """
    Encodes the conditioning input (e.g., segmentation mask) into DiT-compatible latent space.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        num_channels: Sequence[int] = (8, 16, 32, 64, 128),
    ):
        super().__init__()

        self.conv_in = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=num_channels[0],
            strides=1,
            kernel_size=3,
            padding=1,
            conv_only=True,
        )

        self.blocks = nn.ModuleList()
        for i in range(len(num_channels) - 1):
            in_ch = num_channels[i]
            out_ch = num_channels[i + 1]

            self.blocks.append(
                Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=in_ch,
                    out_channels=in_ch,
                    strides=1,
                    kernel_size=3,
                    padding=1,
                    conv_only=True,
                )
            )
            self.blocks.append(
                Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=in_ch,
                    out_channels=out_ch,
                    strides=2,
                    kernel_size=3,
                    padding=1,
                    conv_only=True,
                )
            )

        self.conv_out = zero_module(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=num_channels[-1],
                out_channels=out_channels,
                strides=1,
                kernel_size=3,
                padding=1,
                conv_only=True,
            )
        )

    def forward(self, conditioning):
        x = F.silu(self.conv_in(conditioning))
        for block in self.blocks:
            x = F.silu(block(x))
        x = self.conv_out(x)
        return x


class ControlNet3D(nn.Module):
    """
    ControlNet adapter for DiT3D.

    Unlike the UNet-based ControlNet (which returns residuals), ControlNet3D
    wraps the frozen DiT3D base model and runs the entire forward pass internally,
    returning the final noise prediction directly.

    The base model parameters are frozen by default. Only the control_embedder,
    control_scales, and optionally the last n DiT blocks remain trainable.
    """

    def __init__(
        self,
        base_model: DiT3D,
        control_channels: int,
        inject_layers: Optional[Union[List[int], int]] = None,
        finetune_last_n_blocks: int = 0,
        control_dropout_prob: float = 0.1,
    ):
        super().__init__()
        self.base_model = base_model
        num_blocks = len(base_model.blocks)

        if inject_layers is None:
            self.inject_layers = list(range(num_blocks))
        elif isinstance(inject_layers, int):
            self.inject_layers = list(range(num_blocks - inject_layers, num_blocks))
        else:
            self.inject_layers = inject_layers

        self.control_scales = nn.ParameterList([
            nn.Parameter(torch.ones(1) * 0.1)
            for _ in self.inject_layers
        ])

        self.control_embedder = ControlNetConditioningEmbedding(
            spatial_dims=3,
            in_channels=control_channels,
            out_channels=base_model.blocks[0].attn.qkv.in_features,
            num_channels=(8, 16, 32, 64, 128),
        )

        self.control_time_mlp = zero_module(
            nn.Linear(
                base_model.blocks[0].attn.qkv.in_features,
                1,
            )
        )

        self.control_dropout_prob = control_dropout_prob

        # Freeze base model
        if finetune_last_n_blocks <= 0:
            for p in self.base_model.parameters():
                p.requires_grad = False
            self.base_model.eval()
        else:
            for p in self.base_model.parameters():
                p.requires_grad = False
            start = num_blocks - finetune_last_n_blocks
            for i in range(start, num_blocks):
                for p in self.base_model.blocks[i].parameters():
                    p.requires_grad = True
            for p in self.base_model.final_layer.parameters():
                p.requires_grad = True

    def forward(self, x, t, y, control_input):
        """
        Args:
            x: noisy latent (B, C, D, H, W)
            t: timestep tensor (B,)
            y: class label tensor or None
            control_input: conditioning input (B, control_channels, D, H, W)

        Returns:
            noise_pred: predicted noise (B, C, D, H, W)
        """
        x_embed = self.base_model.x_embedder(x) + self.base_model.pos_embed
        t_embed = self.base_model.t_embedder(t)

        if self.base_model.y_embedder.num_classes > 0 and y is not None:
            y_embed = self.base_model.y_embedder(y, self.training)
            c_embed = t_embed + y_embed
        else:
            c_embed = t_embed

        # Control input augmentation during training
        control_input_mod = control_input
        if self.training:
            if torch.rand(()) < self.control_dropout_prob:
                control_input_mod = torch.zeros_like(control_input)
            else:
                drop_prob = 0.1
                channel_mask = (
                    torch.rand(control_input.shape[1], device=control_input.device) > drop_prob
                )
                control_input_mod = control_input * channel_mask.view(1, -1, 1, 1, 1)

        control = self.control_embedder(control_input_mod)  # (B, hidden, d, h, w)
        control = control.flatten(2).transpose(1, 2)        # (B, T, hidden)

        for idx, block in enumerate(self.base_model.blocks):
            if idx in self.inject_layers:
                scale = 0.1
                control_i = scale * control
            else:
                control_i = None

            x_embed = block(x_embed, c_embed, control=control_i)

        x_out = self.base_model.final_layer(x_embed, c_embed)
        return self.base_model.unpatchify(x_out)
