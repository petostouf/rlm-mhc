"""
Core Transformer Components.

Implements the building blocks for the RLM-mHC Transformer:
- RMSNorm: Root Mean Square Layer Normalization
- SwiGLU FFN: Gated Linear Unit with Swish activation
- Embeddings: Token and position embeddings with RoPE
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Simpler and often more effective than standard LayerNorm.
    RMSNorm(x) = x / RMS(x) * gamma, where RMS(x) = sqrt(mean(x^2))

    Reference: "Root Mean Square Layer Normalization" (Zhang & Sennrich 2019)

    Args:
        dim: Hidden dimension
        eps: Epsilon for numerical stability
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply RMSNorm.

        Args:
            x: Input tensor [*, dim]

        Returns:
            Normalized tensor [*, dim]
        """
        # Compute RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # Normalize and scale
        return (x / rms) * self.weight


class SwiGLU(nn.Module):
    """
    SwiGLU Feed-Forward Network.

    Uses gated linear units with Swish (SiLU) activation for improved
    gradient flow and representation capacity.

    Architecture:
        gate = SiLU(x @ W_gate)
        up = x @ W_up
        output = (gate * up) @ W_down

    The expansion ratio is typically 8/3 to match parameter count with
    standard 4x expansion FFN.

    Reference: "GLU Variants Improve Transformer" (Shazeer 2020)

    Args:
        hidden_dim: Model hidden dimension
        ffn_dim: FFN intermediate dimension (typically 8/3 * hidden_dim)
        dropout: Dropout probability
    """

    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim

        # Gate and up projections (fused for efficiency)
        self.gate_up_proj = nn.Linear(hidden_dim, 2 * ffn_dim, bias=False)

        # Down projection
        self.down_proj = nn.Linear(ffn_dim, hidden_dim, bias=False)

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply SwiGLU transformation.

        Args:
            x: Input tensor [B, S, hidden_dim]

        Returns:
            Output tensor [B, S, hidden_dim]
        """
        # Fused gate and up projection
        gate_up = self.gate_up_proj(x)

        # Split into gate and up
        gate, up = gate_up.chunk(2, dim=-1)

        # Apply SiLU (Swish) to gate and multiply with up
        hidden = F.silu(gate) * up

        # Down projection
        output = self.down_proj(hidden)

        return self.dropout(output)


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).

    Encodes position information by rotating query and key vectors
    in 2D subspaces. Provides relative position awareness without
    explicit position embeddings.

    Reference: "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al. 2021)

    Args:
        dim: Head dimension (must be even)
        max_seq_len: Maximum sequence length
        theta: Base for frequency computation (default: 10000)
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 8192,
        theta: float = 10000.0,
    ):
        super().__init__()

        assert dim % 2 == 0, "RoPE dimension must be even"

        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        # Precompute frequency bands
        # freqs[i] = 1 / (theta^(2i/dim))
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("freqs", freqs, persistent=False)

        # Precompute position indices
        positions = torch.arange(max_seq_len)
        self.register_buffer("positions", positions, persistent=False)

        # Cache for cos/sin values (computed lazily)
        self._cos_cache: Optional[Tensor] = None
        self._sin_cache: Optional[Tensor] = None

    def _compute_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Compute and cache cos/sin values."""
        positions = self.positions[:seq_len].to(device)
        freqs = self.freqs.to(device)

        # Outer product: [seq_len] x [dim/2] -> [seq_len, dim/2]
        angles = torch.outer(positions.float(), freqs)

        # Compute cos and sin
        self._cos_cache = torch.cos(angles).to(dtype)
        self._sin_cache = torch.sin(angles).to(dtype)

        # Store the device to detect cross-device calls
        self._cache_device = device

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        position_offset: int = 0,
    ) -> Tuple[Tensor, Tensor]:
        """
        Apply rotary position embeddings to query and key tensors.

        Args:
            q: Query tensor [B, num_heads, S, head_dim]
            k: Key tensor [B, num_heads, S, head_dim]
            position_offset: Offset for position indices (for kv-cache)

        Returns:
            Tuple of (rotated_q, rotated_k) with same shapes as input
        """
        seq_len = q.shape[2]
        device = q.device
        dtype = q.dtype

        # Ensure cache is computed and on the correct device
        cache_invalid = (
            self._cos_cache is None or
            self._cos_cache.shape[0] < seq_len + position_offset or
            not hasattr(self, '_cache_device') or
            self._cache_device != device
        )
        if cache_invalid:
            self._compute_cache(seq_len + position_offset, device, dtype)

        cos = self._cos_cache[position_offset:position_offset + seq_len]
        sin = self._sin_cache[position_offset:position_offset + seq_len]

        # Apply rotation
        q_rotated = self._apply_rotary(q, cos, sin)
        k_rotated = self._apply_rotary(k, cos, sin)

        return q_rotated, k_rotated

    def _apply_rotary(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        """
        Apply rotary transformation to a tensor.

        Rotation is applied pairwise to consecutive elements:
        [x0, x1] -> [x0*cos - x1*sin, x0*sin + x1*cos]
        """
        # Split into pairs
        x_reshape = x.view(*x.shape[:-1], -1, 2)  # [..., dim/2, 2]

        x0 = x_reshape[..., 0]  # [..., dim/2]
        x1 = x_reshape[..., 1]  # [..., dim/2]

        # Apply rotation
        # cos, sin: [S, dim/2] -> need to broadcast to [..., S, dim/2]
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, S, dim/2]
        sin = sin.unsqueeze(0).unsqueeze(0)

        rotated_0 = x0 * cos - x1 * sin
        rotated_1 = x0 * sin + x1 * cos

        # Interleave back
        rotated = torch.stack([rotated_0, rotated_1], dim=-1)
        return rotated.view_as(x)


class TokenEmbedding(nn.Module):
    """
    Token embedding with optional weight tying.

    Args:
        vocab_size: Size of the vocabulary
        hidden_dim: Embedding dimension
        padding_idx: Index for padding token (optional)
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            hidden_dim,
            padding_idx=padding_idx,
        )
        self.hidden_dim = hidden_dim

    def forward(self, input_ids: Tensor) -> Tensor:
        """
        Embed input tokens.

        Args:
            input_ids: Token indices [B, S]

        Returns:
            Token embeddings [B, S, hidden_dim]
        """
        return self.embedding(input_ids)

    @property
    def weight(self) -> Tensor:
        """Get embedding weights (for weight tying with LM head)."""
        return self.embedding.weight
