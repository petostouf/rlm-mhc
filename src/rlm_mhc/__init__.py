"""
RLM-mHC: Recursive Language Model with Manifold-Constrained Hyper-Connections.

A novel architecture combining:
- RLM: REPL-based programmatic context exploration
- mHC: Stabilized residual streams via doubly stochastic matrices
"""

__version__ = "0.1.0"

from rlm_mhc.types import ModelConfig, ContextHandle, ModelOutput

__all__ = [
    "__version__",
    "ModelConfig",
    "ContextHandle",
    "ModelOutput",
]
