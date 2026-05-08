from __future__ import annotations

import importlib.util
import math
from typing import Optional, Type, Final

import torch
import torch.nn as nn
import torch.nn.functional as F

if importlib.util.find_spec("xformers") is not None:
    import xformers
    import xformers.ops
    import xformers.ops as xops
    has_xformers = True
else:
    xformers = None
    xops = None
    has_xformers = False


class CrossAttention(nn.Module):
    """
    A cross attention layer.

    Args:
        query_dim: number of channels in the query.
        cross_attention_dim: number of channels in the context.
        num_attention_heads: number of heads to use for multi-head attention.
        num_head_channels: number of channels in each head.
        dropout: dropout probability to use.
        upcast_attention: if True, upcast attention operations to full precision.
        use_flash_attention: if True, use flash attention for a memory efficient attention mechanism.
    """

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: int | None = None,
        num_attention_heads: int = 8,
        num_head_channels: int = 64,
        dropout: float = 0.0,
        upcast_attention: bool = False,
        use_flash_attention: bool = True,
    ) -> None:
        super().__init__()
        self.use_flash_attention = use_flash_attention and has_xformers
        inner_dim = num_head_channels * num_attention_heads
        cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim

        self.scale = 1 / math.sqrt(num_head_channels)
        self.num_heads = num_attention_heads

        self.upcast_attention = upcast_attention

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

    def reshape_heads_to_batch_dim(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape
        x = x.reshape(batch_size, seq_len, self.num_heads, dim // self.num_heads)
        x = x.permute(0, 2, 1, 3).reshape(batch_size * self.num_heads, seq_len, dim // self.num_heads)
        return x

    def reshape_batch_dim_to_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape
        x = x.reshape(batch_size // self.num_heads, self.num_heads, seq_len, dim)
        x = x.permute(0, 2, 1, 3).reshape(batch_size // self.num_heads, seq_len, dim * self.num_heads)
        return x

    def reshape_to_4d(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape
        x = x.reshape(batch_size, seq_len, self.num_heads, dim // self.num_heads)
        return x

    def reshape_from_4d(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, num_heads, head_dim = x.shape
        x = x.reshape(batch_size, seq_len, num_heads * head_dim)
        return x

    def _memory_efficient_attention_xformers(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        x = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=None)
        return x

    def _attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        dtype = query.dtype
        if self.upcast_attention:
            query = query.float()
            key = key.float()

        attention_scores = torch.baddbmm(
            torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
            query,
            key.transpose(-1, -2),
            beta=0,
            alpha=self.scale,
        )
        attention_probs = attention_scores.softmax(dim=-1)
        attention_probs = attention_probs.to(dtype=dtype)

        x = torch.bmm(attention_probs, value)
        return x

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        query = self.to_q(x)
        context = context if context is not None else x
        key = self.to_k(context)
        value = self.to_v(context)

        if self.use_flash_attention:
            query = self.reshape_to_4d(query)
            key = self.reshape_to_4d(key)
            value = self.reshape_to_4d(value)

            x = self._memory_efficient_attention_xformers(query.contiguous(), key.contiguous(), value.contiguous())
            x = self.reshape_from_4d(x)
        else:
            query = self.reshape_heads_to_batch_dim(query)
            key = self.reshape_heads_to_batch_dim(key)
            value = self.reshape_heads_to_batch_dim(value)
            x = self._attention(query, key, value)
            x = self.reshape_batch_dim_to_heads(x)

        x = x.to(query.dtype)

        return self.to_out(x)


class AttentionBlock(nn.Module):
    """MONAI-style attention block adapted to handle xformers flash attention."""

    def __init__(
        self,
        spatial_dims: int,
        num_channels: int,
        num_head_channels: int | None = None,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        use_flash_attention: bool = False,
        use_norm: bool = True,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        self.use_flash_attention = use_flash_attention and has_xformers
        self.spatial_dims = spatial_dims
        self.num_channels = num_channels
        self.use_norm = use_norm
        self.use_residual = use_residual

        self.num_heads = num_channels // num_head_channels if num_head_channels is not None else 1
        self.head_dim = num_channels // self.num_heads
        self.scale = 1 / math.sqrt(self.head_dim)

        if self.use_norm:
            self.norm = nn.GroupNorm(num_groups=norm_num_groups, num_channels=num_channels, eps=norm_eps, affine=True)
        else:
            self.norm = nn.Identity()

        self.to_q = nn.Linear(num_channels, num_channels)
        self.to_k = nn.Linear(num_channels, num_channels)
        self.to_v = nn.Linear(num_channels, num_channels)

        self.proj_attn = nn.Linear(num_channels, num_channels)

    def reshape_for_xformers(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.contiguous()

    def reshape_for_manual_attention(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape
        x = x.view(batch_size * self.num_heads, seq_len, self.head_dim)
        return x

    def reshape_from_xformers(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, num_heads, head_dim = x.shape
        x = x.view(batch_size, seq_len, num_heads * head_dim)
        return x

    def reshape_from_manual_attention(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        x = x.view(batch_size, self.num_heads, x.shape[1], self.head_dim)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, x.shape[1], self.num_heads * self.head_dim)
        return x

    def _memory_efficient_attention_xformers(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        x = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=None)
        return x

    def _attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        attention_scores = torch.baddbmm(
            torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
            query,
            key.transpose(-1, -2),
            beta=0,
            alpha=self.scale,
        )
        attention_probs = attention_scores.softmax(dim=-1)
        x = torch.bmm(attention_probs, value)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        if x.ndim == 3:
            batch, seq_len, channel = x.shape
            spatial_shape = None
        elif self.spatial_dims == 2 and x.ndim == 4:
            batch, channel, height, width = x.shape
            spatial_shape = (height, width)
            x = self.norm(x)
            x = x.view(batch, channel, height * width).transpose(1, 2)
        elif self.spatial_dims == 3 and x.ndim == 5:
            batch, channel, height, width, depth = x.shape
            spatial_shape = (height, width, depth)
            x = self.norm(x)
            x = x.view(batch, channel, height * width * depth).transpose(1, 2)
        else:
            raise ValueError(f"Unexpected input shape {x.shape} for spatial_dims={self.spatial_dims}")

        query = self.to_q(x)
        key = self.to_k(x)
        value = self.to_v(x)

        if self.use_flash_attention:
            query = self.reshape_for_xformers(query)
            key = self.reshape_for_xformers(key)
            value = self.reshape_for_xformers(value)
            x = self._memory_efficient_attention_xformers(query, key, value)
            x = self.reshape_from_xformers(x)
        else:
            query = self.reshape_for_manual_attention(query)
            key = self.reshape_for_manual_attention(key)
            value = self.reshape_for_manual_attention(value)
            x = self._attention(query, key, value)
            x = self.reshape_from_manual_attention(x, batch)

        x = self.proj_attn(x)

        if spatial_shape is not None:
            x = x.transpose(1, 2)
            if self.spatial_dims == 2:
                x = x.view(batch, channel, *spatial_shape)
            elif self.spatial_dims == 3:
                x = x.view(batch, channel, *spatial_shape)

        if self.use_residual:
            x = x + residual

        return x


class Attention(nn.Module):
    """Self-attention with optional xformers flash attention or PyTorch SDPA fallback."""

    fused_attn: Final[bool]

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        scale_norm: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        norm_layer: Optional[Type[nn.Module]] = None,
        use_flash_attention: bool = True,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        if qk_norm or scale_norm:
            assert norm_layer is not None, 'norm_layer must be provided if qk_norm or scale_norm is True'

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_flash_attention = use_flash_attention and has_xformers
        self.fused_attn = not self.use_flash_attention and hasattr(F, 'scaled_dot_product_attention')

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.norm = norm_layer(dim) if scale_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)

        q, k = self.q_norm(q), self.k_norm(k)

        if self.use_flash_attention:
            q = q.contiguous()
            k = k.contiguous()
            v = v.contiguous()
            x = xops.memory_efficient_attention(q, k, v, attn_bias=None)
        elif self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))
            if attn_mask is not None:
                attn = attn + attn_mask
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.reshape(B, N, C)
        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
