# vqvae.py
from __future__ import annotations
from collections.abc import Sequence
from torch.nn import functional as F
from typing import Tuple
import torch
import torch.nn as nn
from monai.networks.blocks import Convolution
from monai.networks.layers import Act
from monai.utils import ensure_tuple_rep
from torch import amp

# --- VQ-VAE Residual Block ---
class VQVAEResidualUnit(nn.Module):
    """Residual block used in VQ-VAE encoder/decoder."""
    def __init__(
        self,
        spatial_dims: int,
        num_channels: int,
        num_res_channels: int,
        act=Act.RELU,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.conv1 = Convolution(
            spatial_dims=spatial_dims,
            in_channels=num_channels,
            out_channels=num_res_channels,
            adn_ordering="DA",
            act=act,
            dropout=dropout,
            bias=bias,
        )
        self.conv2 = Convolution(
            spatial_dims=spatial_dims,
            in_channels=num_res_channels,
            out_channels=num_channels,
            conv_only=True,
            bias=bias,
        )

    def forward(self, x):
        return torch.relu(x + self.conv2(self.conv1(x)))

class EMAQuantizer(nn.Module):
    """EMA codebook for VQ-VAE, fixed shape handling and AMP-safe."""
    def __init__(
        self,
        spatial_dims: int,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        ddp_sync: bool = False,
    ):
        super().__init__()
        assert spatial_dims in [2, 3], "Only 2D or 3D inputs supported"

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.requires_grad = False
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        self.ddp_sync = ddp_sync

        # EMA buffers
        self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("ema_w", self.embedding.weight.data.clone())

    @torch.autocast(device_type="cuda", enabled=False)
    def forward(self, inputs: torch.Tensor, batch_size: int = 1024) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        inputs: [B, C, *spatial_dims] -> quantize along last channel
        returns: quantized [B, C, *spatial_dims], loss, encoding indices
        """
        inputs = inputs.float()
        spatial_dims = inputs.shape[2:]
        B = inputs.shape[0]
        inputs_flat = inputs.permute(0, *range(2, inputs.ndim), 1).contiguous().view(-1, self.embedding_dim).float()

        # batch-wise distance computation
        num_batches = (inputs_flat.size(0) + batch_size - 1) // batch_size
        encoding_indices_list = []
        for i in range(num_batches):
            batch_input = inputs_flat[i * batch_size : (i + 1) * batch_size]
            distances = (
                batch_input.pow(2).sum(1, keepdim=True)
                + self.embedding.weight.pow(2).sum(1)
                - 2.0 * batch_input @ self.embedding.weight.t()
            )
            encoding_indices = torch.argmin(distances, dim=1)
            encoding_indices_list.append(encoding_indices)

        encoding_indices_flat = torch.cat(encoding_indices_list)
        encoding_indices = encoding_indices_flat.view(B, *spatial_dims)

        # Quantize
        quantized = self.embedding(encoding_indices_flat)
        quantized = quantized.view(B, *spatial_dims, self.embedding_dim).permute(0, -1, *range(1, 1 + len(spatial_dims))).contiguous()

        # EMA updates
        if self.training:
            with torch.no_grad():
                encodings = F.one_hot(encoding_indices_flat, self.num_embeddings).float()
                encodings_sum = encodings.sum(0)
                dw = encodings.t() @ inputs_flat

                if self.ddp_sync and torch.distributed.is_initialized():
                    torch.distributed.all_reduce(encodings_sum)
                    torch.distributed.all_reduce(dw)

                self.ema_cluster_size.data.mul_(self.decay).add_(encodings_sum, alpha=1 - self.decay)
                self.ema_w.data.mul_(self.decay).add_(dw, alpha=1 - self.decay)

                # Laplace smoothing
                n = self.ema_cluster_size.sum()
                cluster_size = (self.ema_cluster_size + self.epsilon) / (n + self.num_embeddings * self.epsilon) * n
                self.embedding.weight.data.copy_(self.ema_w / cluster_size.unsqueeze(1))

        # Commitment loss
        loss = self.commitment_cost * F.mse_loss(quantized.detach(), inputs)
        quantized = inputs + (quantized - inputs).detach()  # straight-through

        quantized = quantized.float()
        loss = loss.float()
        return quantized, loss, encoding_indices

    def embed(self, indices: torch.Tensor) -> torch.Tensor:
        """Convert indices to embeddings and channel-first format"""
        quantized = self.embedding(indices)
        spatial_dims = indices.shape[1:]
        return quantized.view(indices.shape[0], *spatial_dims, self.embedding_dim)\
                        .permute(0, -1, *range(1, 1 + len(spatial_dims))).contiguous()


class VectorQuantizer(nn.Module):
    """Wrapper around EMAQuantizer, computes perplexity."""
    def __init__(self, quantizer: nn.Module):
        super().__init__()
        self.quantizer = quantizer
        self.num_embeddings = quantizer.num_embeddings
        self.perplexity = torch.tensor(0.0)

    def _compute_stats(self, indices):
        flat = indices.view(-1)

        counts = torch.bincount(
            flat,
            minlength=self.quantizer.num_embeddings
        ).float()

        probs = counts / (counts.sum() + 1e-10)

        perplexity = torch.exp(
            -torch.sum(probs * torch.log(probs + 1e-10))
        )

        self.perplexity = perplexity.detach()

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        quantized, loss, indices = self.quantizer(inputs)
        self._compute_stats(indices)

        return quantized, loss, indices

    def embed(self, indices: torch.Tensor):
        return self.quantizer.embed(indices)

    def quantize(self, inputs: torch.Tensor):
        return self.quantizer(inputs)

# --- Encoder ---
class Encoder(nn.Module):
    """VQ-VAE encoder."""
    def __init__(self, spatial_dims, in_channels, out_channels, num_channels, num_res_layers,
                 num_res_channels, downsample_parameters, dropout, act):
        super().__init__()
        blocks = []
        for i in range(len(num_channels)):
            blocks.append(
                Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=in_channels if i == 0 else num_channels[i-1],
                    out_channels=num_channels[i],
                    strides=downsample_parameters[i][0],
                    kernel_size=downsample_parameters[i][1],
                    adn_ordering="DA",
                    act=act,
                    dropout=None if i == 0 else dropout,
                    dropout_dim=1,
                    dilation=downsample_parameters[i][2],
                    padding=downsample_parameters[i][3],
                )
            )
            for _ in range(num_res_layers):
                blocks.append(VQVAEResidualUnit(
                    spatial_dims=spatial_dims,
                    num_channels=num_channels[i],
                    num_res_channels=num_res_channels[i],
                    act=act,
                    dropout=dropout,
                ))
        blocks.append(Convolution(
            spatial_dims=spatial_dims,
            in_channels=num_channels[-1],
            out_channels=out_channels,
            strides=1,
            kernel_size=3,
            padding=1,
            conv_only=True
        ))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

# --- Decoder ---
class Decoder(nn.Module):
    """VQ-VAE decoder."""
    def __init__(self, spatial_dims, in_channels, out_channels, num_channels, num_res_layers,
                 num_res_channels, upsample_parameters, dropout, act, output_act):
        super().__init__()
        reversed_num_channels = list(reversed(num_channels))
        blocks = [Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=reversed_num_channels[0],
            strides=1,
            kernel_size=3,
            padding=1,
            conv_only=True
        )]
        reversed_res_channels = list(reversed(num_res_channels))
        for i in range(len(num_channels)):
            for _ in range(num_res_layers):
                blocks.append(VQVAEResidualUnit(
                    spatial_dims=spatial_dims,
                    num_channels=reversed_num_channels[i],
                    num_res_channels=reversed_res_channels[i],
                    act=act,
                    dropout=dropout
                ))
            blocks.append(Convolution(
                spatial_dims=spatial_dims,
                in_channels=reversed_num_channels[i],
                out_channels=out_channels if i == len(num_channels)-1 else reversed_num_channels[i+1],
                strides=upsample_parameters[i][0],
                kernel_size=upsample_parameters[i][1],
                adn_ordering="DA",
                act=act,
                dropout=None if i == len(num_channels)-1 else dropout,
                norm=None,
                dilation=upsample_parameters[i][2],
                conv_only=i == len(num_channels)-1,
                is_transposed=True,
                padding=upsample_parameters[i][3],
                output_padding=upsample_parameters[i][4]
            ))
        if output_act:
            blocks.append(Act[output_act]())
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

# --- Full VQVAE ---
class VQVAE(nn.Module):
    def __init__(self, spatial_dims, in_channels, out_channels,
                 num_channels=(96,96,192), num_res_layers=3, num_res_channels=(96,96,192),
                 downsample_parameters=((2,4,1,1),(2,4,1,1),(2,4,1,1)),
                 upsample_parameters=((2,4,1,1,0),(2,4,1,1,0),(2,4,1,1,0)),
                 num_embeddings=32, embedding_dim=64,
                 commitment_cost=0.25, decay=0.99, epsilon=1e-5,
                 dropout=0.0, act=Act.RELU, output_act=None,
                 use_checkpointing=False, ddp_sync=False):
        super().__init__()
        if isinstance(num_res_channels, int):
            num_res_channels = ensure_tuple_rep(num_res_channels, len(num_channels))
        self.encoder = Encoder(spatial_dims, in_channels, embedding_dim,
                               num_channels, num_res_layers, num_res_channels,
                               downsample_parameters, dropout, act)
        self.decoder = Decoder(spatial_dims, embedding_dim, out_channels,
                               num_channels, num_res_layers, num_res_channels,
                               upsample_parameters, dropout, act, output_act)
        self.quantizer = VectorQuantizer(EMAQuantizer(
            spatial_dims=spatial_dims,
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost,
            decay=decay,
            epsilon=epsilon,
            ddp_sync=ddp_sync
        ))
        self.use_checkpointing = use_checkpointing

    def encode(self, x):
        return torch.utils.checkpoint.checkpoint(self.encoder, x, use_reentrant=False) if self.use_checkpointing else self.encoder(x)

    def decode(self, x):
        return torch.utils.checkpoint.checkpoint(self.decoder, x, use_reentrant=False) if self.use_checkpointing else self.decoder(x)

    def quantize(self, z):
        return self.quantizer(z)

    def forward(self, x):
        z = self.encode(x)
        quantized, q_loss, indices = self.quantize(z)
        recon = self.decode(quantized)
        return recon, q_loss, indices

    def encode_stage_2_inputs(self, x: torch.Tensor, quantized: bool = True) -> torch.Tensor:
        """
        Returns latent z (before quantization) or e (quantized).
        """
        z = self.encode(x)
        e, _, _ = self.quantize(z)
        if quantized:
            return e
        return z

    def decode_stage_2_outputs(self, z: torch.Tensor) -> torch.Tensor:
        e, _, _ = self.quantize(z)
        image = self.decode(e)
        return image
