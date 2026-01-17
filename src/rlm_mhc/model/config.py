"""
Model configuration for RLM-mHC.
"""

from dataclasses import dataclass
from typing import Optional
import yaml
from pathlib import Path


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
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads})"
            )
        expected_head_dim = self.hidden_dim // self.num_heads
        if self.head_dim != expected_head_dim:
            # Auto-correct head_dim
            self.head_dim = expected_head_dim

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ModelConfig":
        """Load configuration from a YAML file."""
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Extract model section if present
        if "model" in config_dict:
            config_dict = config_dict["model"]

        return cls(**config_dict)

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "ModelConfig":
        """Load configuration from a pretrained checkpoint directory."""
        config_path = Path(path) / "config.yaml"
        if config_path.exists():
            return cls.from_yaml(config_path)

        # Try JSON format
        import json
        json_path = Path(path) / "config.json"
        if json_path.exists():
            with open(json_path, "r") as f:
                config_dict = json.load(f)
            return cls(**config_dict)

        raise FileNotFoundError(f"No config file found in {path}")

    def save(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = {
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "ffn_dim": self.ffn_dim,
            "vocab_size": self.vocab_size,
            "max_seq_len": self.max_seq_len,
            "dropout": self.dropout,
            "attention_dropout": self.attention_dropout,
            "mhc_enabled": self.mhc_enabled,
            "mhc_flows": self.mhc_flows,
            "mhc_sinkhorn_iters": self.mhc_sinkhorn_iters,
            "rope_theta": self.rope_theta,
            "gradient_checkpointing": self.gradient_checkpointing,
        }

        with open(path, "w") as f:
            yaml.dump({"model": config_dict}, f, default_flow_style=False)

    def num_parameters(self, include_embeddings: bool = True) -> int:
        """Estimate the number of parameters in the model."""
        # Embeddings
        embed_params = self.vocab_size * self.hidden_dim if include_embeddings else 0

        # Per layer
        # Attention: Q, K, V, O projections
        attn_params = 4 * self.hidden_dim * self.hidden_dim

        # FFN (SwiGLU): gate, up, down
        ffn_params = 3 * self.hidden_dim * self.ffn_dim

        # mHC: expansion + contraction + connection weights
        if self.mhc_enabled:
            mhc_params = 2 * (
                self.hidden_dim * (self.hidden_dim * self.mhc_flows) +  # expansion
                (self.hidden_dim * self.mhc_flows) * self.hidden_dim +  # contraction
                self.mhc_flows * self.mhc_flows  # connection weights
            )
        else:
            mhc_params = 0

        # Layer norms (4 per layer with mHC)
        norm_params = 4 * self.hidden_dim if self.mhc_enabled else 2 * self.hidden_dim

        layer_params = attn_params + ffn_params + mhc_params + norm_params

        # Final layer norm + LM head (tied to embeddings usually)
        final_params = self.hidden_dim

        total = embed_params + self.num_layers * layer_params + final_params
        return total
