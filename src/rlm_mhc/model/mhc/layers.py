"""
mHC Layer Implementation.

Manifold-Constrained Hyper-Connection layers that extend the residual
stream to multiple parallel flows with doubly stochastic connection matrices.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple

from rlm_mhc.model.mhc.sinkhorn import sinkhorn_knopp


class mHCLayer(nn.Module):
    """
    Manifold-Constrained Hyper-Connection Layer.

    Extends the residual stream from C dimensions to num_flows × C dimensions
    with 4 parallel flows. Applies learnable doubly stochastic connection
    matrices between flows for stable information mixing.

    Architecture:
        1. Expansion: [B, S, C] → [B, S, num_flows × C] via linear projection
        2. Reshape to flows: [B, S, num_flows, C]
        3. Apply Sinkhorn-Knopp to get doubly stochastic connection matrix P
        4. Mix flows: flow_i_out = Σ_j (P_ij × flow_j)
        5. Contraction: [B, S, num_flows × C] → [B, S, C]

    Args:
        hidden_dim: Model hidden dimension (C)
        num_flows: Number of parallel flows (default: 4)
        sinkhorn_iters: Number of Sinkhorn-Knopp iterations (default: 20)
        dropout: Dropout probability (default: 0.0)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_flows: int = 4,
        sinkhorn_iters: int = 20,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_flows = num_flows
        self.sinkhorn_iters = sinkhorn_iters

        # Expansion projection: C → num_flows × C
        self.expand_proj = nn.Linear(hidden_dim, hidden_dim * num_flows, bias=False)

        # Contraction projection: num_flows × C → C
        self.contract_proj = nn.Linear(hidden_dim * num_flows, hidden_dim, bias=False)

        # Learnable connection weights (will be projected to doubly stochastic)
        self.connection_weights = nn.Parameter(torch.zeros(num_flows, num_flows))

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable training."""
        # Xavier initialization for projections
        nn.init.xavier_uniform_(self.expand_proj.weight)
        nn.init.xavier_uniform_(self.contract_proj.weight)

        # Initialize connection weights close to identity
        # This makes the initial connection matrix close to uniform mixing
        nn.init.zeros_(self.connection_weights)

    def get_connection_matrix(self) -> Tensor:
        """
        Get the current doubly stochastic connection matrix.

        Returns:
            Tensor of shape [num_flows, num_flows] with row and column sums = 1
        """
        return sinkhorn_knopp(self.connection_weights, self.sinkhorn_iters)

    @property
    def amax_gain(self) -> float:
        """
        Compute the Amax Gain metric for monitoring training stability.

        Amax is the maximum absolute value in the connection matrix.
        Values > 2.0 may indicate instability.

        Returns:
            Maximum absolute value in the connection matrix
        """
        P = self.get_connection_matrix()
        return torch.abs(P).max().item()

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply mHC transformation.

        Args:
            x: Input tensor of shape [B, S, C]

        Returns:
            Output tensor of shape [B, S, C]
        """
        B, S, C = x.shape
        assert C == self.hidden_dim, f"Expected hidden_dim {self.hidden_dim}, got {C}"

        # 1. Expansion: [B, S, C] → [B, S, num_flows × C]
        expanded = self.expand_proj(x)

        # 2. Reshape to flows: [B, S, num_flows, C]
        flows = expanded.view(B, S, self.num_flows, self.hidden_dim)

        # 3. Get doubly stochastic connection matrix
        P = self.get_connection_matrix()  # [num_flows, num_flows]

        # 4. Apply connections between flows
        # einsum: 'ij,bsjd->bsid' means flow_i = sum_j(P_ij * flow_j)
        connected = torch.einsum('ij,bsjd->bsid', P, flows)

        # 5. Flatten and contract: [B, S, num_flows × C] → [B, S, C]
        # Use reshape() instead of view() because einsum output may not be contiguous
        connected_flat = connected.reshape(B, S, -1)
        output = self.contract_proj(connected_flat)

        # Apply dropout
        output = self.dropout(output)

        return output

    def extra_repr(self) -> str:
        return (
            f"hidden_dim={self.hidden_dim}, "
            f"num_flows={self.num_flows}, "
            f"sinkhorn_iters={self.sinkhorn_iters}"
        )


class mHCBlock(nn.Module):
    """
    mHC Block combining pre-attention and post-attention mHC layers.

    This block wraps two mHC layers that are placed:
    1. Before the attention layer (pre-attention)
    2. After the attention layer, before FFN (post-attention)

    The block includes layer normalization before each mHC layer
    and uses residual connections.

    Args:
        hidden_dim: Model hidden dimension
        num_flows: Number of parallel flows
        sinkhorn_iters: Number of Sinkhorn iterations
        dropout: Dropout probability
    """

    def __init__(
        self,
        hidden_dim: int,
        num_flows: int = 4,
        sinkhorn_iters: int = 20,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.pre_attention = mHCLayer(
            hidden_dim=hidden_dim,
            num_flows=num_flows,
            sinkhorn_iters=sinkhorn_iters,
            dropout=dropout,
        )

        self.post_attention = mHCLayer(
            hidden_dim=hidden_dim,
            num_flows=num_flows,
            sinkhorn_iters=sinkhorn_iters,
            dropout=dropout,
        )

    def forward_pre(self, x: Tensor, residual: bool = True) -> Tensor:
        """Apply pre-attention mHC transformation."""
        out = self.pre_attention(x)
        return x + out if residual else out

    def forward_post(self, x: Tensor, residual: bool = True) -> Tensor:
        """Apply post-attention mHC transformation."""
        out = self.post_attention(x)
        return x + out if residual else out

    def get_amax_metrics(self) -> Tuple[float, float]:
        """Get Amax metrics for both layers."""
        return (
            self.pre_attention.amax_gain,
            self.post_attention.amax_gain,
        )
