"""
Shared data classes and type definitions for RLM-mHC.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor


@dataclass
class ModelConfig:
    """Configuration for the RLM-mHC Transformer model."""

    # Model dimensions
    hidden_dim: int = 2048
    num_layers: int = 24
    num_heads: int = 16
    head_dim: int = 128
    ffn_dim: int = 5461  # 8/3 * hidden_dim for SwiGLU
    vocab_size: int = 32000
    max_seq_len: int = 8192

    # Regularization
    dropout: float = 0.0
    attention_dropout: float = 0.0

    # mHC configuration
    mhc_enabled: bool = True
    mhc_flows: int = 4
    mhc_sinkhorn_iters: int = 20

    # Position encoding (RoPE)
    rope_theta: float = 10000.0

    # Training
    gradient_checkpointing: bool = False

    def __post_init__(self):
        """Validate configuration."""
        assert self.hidden_dim % self.num_heads == 0, (
            f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads})"
        )
        assert self.head_dim == self.hidden_dim // self.num_heads, (
            f"head_dim should be {self.hidden_dim // self.num_heads}, got {self.head_dim}"
        )


@dataclass
class ContextHandle:
    """Handle to a loaded context for REPL operations."""

    total_tokens: int
    num_chunks: int
    chunk_size: int
    context_id: Optional[str] = None

    def __repr__(self) -> str:
        return (
            f"ContextHandle(tokens={self.total_tokens}, "
            f"chunks={self.num_chunks}, chunk_size={self.chunk_size})"
        )


@dataclass
class ModelOutput:
    """Standard output from the RLM-mHC model."""

    logits: Tensor  # [B, S, V]
    loss: Optional[Tensor] = None
    hidden_states: Optional[Tuple[Tensor, ...]] = None
    attentions: Optional[Tuple[Tensor, ...]] = None

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


@dataclass
class Chunk:
    """A chunk of tokenized context."""

    id: int
    tokens: List[int]
    start_pos: int
    end_pos: int


@dataclass
class SearchResult:
    """Result from a context search operation."""

    chunk_id: int
    start_pos: int
    end_pos: int
    score: float
    snippet: str


@dataclass
class RecursionCall:
    """Record of a recursive LLM call."""

    id: int
    depth: int
    timestamp: float
    completed: bool = False


@dataclass
class TrainingConfig:
    """Configuration for training."""

    # Optimizer
    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    betas: Tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    max_grad_norm: float = 1.0

    # Scheduler
    scheduler_type: str = "cosine"
    warmup_steps: int = 1000
    min_lr_ratio: float = 0.1

    # Batch
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    max_steps: int = 50000
    num_epochs: int = 1

    # Memory optimization
    gradient_checkpointing: bool = True
    mixed_precision: str = "bf16"

    # Checkpointing
    save_steps: int = 1000
    save_total_limit: int = 5
    output_dir: str = "checkpoints"

    # Logging
    logging_steps: int = 10
    wandb_project: str = "rlm-mhc"
    wandb_entity: Optional[str] = None

    # mHC monitoring
    mhc_amax_log_steps: int = 100


@dataclass
class TrainingResult:
    """Result from a training run."""

    final_loss: float
    total_steps: int
    checkpoints: List[str]
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    """Result from an evaluation run."""

    metrics: Dict[str, float]
    predictions: Optional[List[str]] = None
    references: Optional[List[str]] = None


@dataclass
class RLMExample:
    """A single training example for RLM."""

    context: str
    instruction: str
    response: str
    task_type: str  # 'peek', 'search', 'qa'


@dataclass
class ChunkConfig:
    """Configuration for context chunking."""

    chunk_size: int = 4096
    overlap: int = 256
    max_chunks_in_memory: int = 16
    cache_strategy: str = "lru"


@dataclass
class SessionConfig:
    """Configuration for an RLM session."""

    max_recursion: int = 5
    context_window_size: int = 4096
    chunking: ChunkConfig = field(default_factory=ChunkConfig)
