"""
Attention Module with Flash Attention 2 Support.

Implements multi-head self-attention with:
- Flash Attention 2 integration (when available)
- Rotary Position Embeddings (RoPE)
- Causal masking for autoregressive generation
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple

from rlm_mhc.model.components import RotaryEmbedding

# Try to import Flash Attention
try:
    from flash_attn import flash_attn_func
    from flash_attn.bert_padding import unpad_input, pad_input
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention with Flash Attention 2.

    Features:
    - Fused QKV projection for efficiency
    - Rotary Position Embeddings (RoPE)
    - Flash Attention 2 when available (falls back to PyTorch)
    - Support for causal and non-causal attention

    Args:
        hidden_dim: Model hidden dimension
        num_heads: Number of attention heads
        head_dim: Dimension per head (default: hidden_dim // num_heads)
        dropout: Attention dropout probability
        rope_theta: Base for RoPE frequency computation
        max_seq_len: Maximum sequence length for RoPE
        use_flash_attn: Whether to use Flash Attention (if available)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: Optional[int] = None,
        dropout: float = 0.0,
        rope_theta: float = 10000.0,
        max_seq_len: int = 8192,
        use_flash_attn: bool = True,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim or (hidden_dim // num_heads)
        self.dropout = dropout
        self.use_flash_attn = use_flash_attn and FLASH_ATTN_AVAILABLE

        assert self.head_dim * num_heads == hidden_dim, (
            f"hidden_dim ({hidden_dim}) must equal num_heads ({num_heads}) * head_dim ({self.head_dim})"
        )

        # Fused QKV projection
        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Rotary embeddings
        self.rotary = RotaryEmbedding(
            dim=self.head_dim,
            max_seq_len=max_seq_len,
            theta=rope_theta,
        )

        # Dropout (only used in non-flash path)
        self.attn_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Scaling factor
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
        is_causal: bool = True,
        position_offset: int = 0,
    ) -> Tensor:
        """
        Apply multi-head self-attention.

        Args:
            x: Input tensor [B, S, hidden_dim]
            attention_mask: Optional attention mask [B, S] or [B, 1, S, S]
            is_causal: Whether to apply causal masking
            position_offset: Position offset for RoPE (for kv-cache)

        Returns:
            Output tensor [B, S, hidden_dim]
        """
        B, S, _ = x.shape

        # Compute Q, K, V
        qkv = self.qkv_proj(x)  # [B, S, 3 * hidden_dim]

        # Reshape to [B, S, 3, num_heads, head_dim]
        qkv = qkv.view(B, S, 3, self.num_heads, self.head_dim)

        # Transpose to [3, B, num_heads, S, head_dim]
        qkv = qkv.permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: [B, num_heads, S, head_dim]

        # Apply RoPE
        q, k = self.rotary(q, k, position_offset=position_offset)

        # Choose attention implementation
        if self.use_flash_attn:
            attn_output = self._flash_attention(q, k, v, attention_mask, is_causal)
        else:
            attn_output = self._standard_attention(q, k, v, attention_mask, is_causal)

        # Reshape: [B, num_heads, S, head_dim] -> [B, S, hidden_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(B, S, self.hidden_dim)

        # Output projection
        output = self.out_proj(attn_output)

        return output

    def _flash_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attention_mask: Optional[Tensor],
        is_causal: bool,
    ) -> Tensor:
        """Apply Flash Attention 2."""
        # Flash Attention expects [B, S, num_heads, head_dim]
        q = q.transpose(1, 2)  # [B, S, num_heads, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Flash Attention call
        dropout_p = self.dropout if self.training else 0.0

        attn_output = flash_attn_func(
            q, k, v,
            dropout_p=dropout_p,
            causal=is_causal,
        )

        # Back to [B, num_heads, S, head_dim]
        return attn_output.transpose(1, 2)

    def _standard_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attention_mask: Optional[Tensor],
        is_causal: bool,
    ) -> Tensor:
        """Standard PyTorch attention (fallback)."""
        B, num_heads, S, head_dim = q.shape

        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, num_heads, S, S]

        # Apply causal mask
        if is_causal:
            causal_mask = torch.triu(
                torch.full((S, S), float('-inf'), device=q.device, dtype=q.dtype),
                diagonal=1
            )
            attn_weights = attn_weights + causal_mask

        # Apply attention mask
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                # [B, S] -> [B, 1, 1, S]
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                attention_mask = attention_mask.masked_fill(
                    attention_mask == 0,
                    float('-inf')
                )
            attn_weights = attn_weights + attention_mask

        # Softmax and dropout
        # Handle case where all attention weights are -inf (produces NaN)
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
        # Replace NaN with zeros (occurs when entire row is masked)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = attn_weights.to(q.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)

        return attn_output


class CausalSelfAttention(MultiHeadAttention):
    """
    Convenience wrapper for causal self-attention.

    Always applies causal masking, suitable for autoregressive models.
    """

    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_offset: int = 0,
    ) -> Tensor:
        return super().forward(
            x,
            attention_mask=attention_mask,
            is_causal=True,
            position_offset=position_offset,
        )
