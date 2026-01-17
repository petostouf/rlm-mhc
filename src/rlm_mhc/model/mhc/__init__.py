"""
Manifold-Constrained Hyper-Connections (mHC) Module.

Implements the mHC mechanism from the DeepSeek paper for stabilizing
residual stream dynamics via doubly stochastic connection matrices.
"""

from rlm_mhc.model.mhc.sinkhorn import sinkhorn_knopp, sinkhorn_knopp_log
from rlm_mhc.model.mhc.layers import mHCLayer, mHCBlock

__all__ = [
    "sinkhorn_knopp",
    "sinkhorn_knopp_log",
    "mHCLayer",
    "mHCBlock",
]
