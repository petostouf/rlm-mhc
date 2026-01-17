"""
RLM-mHC Transformer Model.

Complete Transformer implementation with integrated mHC layers for
stabilized residual stream dynamics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, List, Union
from pathlib import Path
import json

from rlm_mhc.model.config import ModelConfig
from rlm_mhc.model.components import RMSNorm, SwiGLU, TokenEmbedding
from rlm_mhc.model.attention import CausalSelfAttention
from rlm_mhc.model.mhc import mHCLayer
from rlm_mhc.types import ModelOutput


class TransformerBlock(nn.Module):
    """
    Single Transformer block with mHC integration.

    Architecture (with mHC enabled):
        x → RMSNorm → mHC_pre → (+x) → RMSNorm → Attention → (+x)
          → RMSNorm → mHC_post → (+x) → RMSNorm → FFN → (+x) → output

    Architecture (without mHC):
        x → RMSNorm → Attention → (+x) → RMSNorm → FFN → (+x) → output

    Args:
        config: Model configuration
        layer_idx: Index of this layer (for debugging)
    """

    def __init__(self, config: ModelConfig, layer_idx: int = 0):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx

        # Layer norms
        self.norm1 = RMSNorm(config.hidden_dim)
        self.norm2 = RMSNorm(config.hidden_dim)

        # Attention
        self.attention = CausalSelfAttention(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            head_dim=config.head_dim,
            dropout=config.attention_dropout,
            rope_theta=config.rope_theta,
            max_seq_len=config.max_seq_len,
        )

        # FFN
        self.ffn = SwiGLU(
            hidden_dim=config.hidden_dim,
            ffn_dim=config.ffn_dim,
            dropout=config.dropout,
        )

        # mHC layers (if enabled)
        if config.mhc_enabled:
            self.norm_mhc_pre = RMSNorm(config.hidden_dim)
            self.norm_mhc_post = RMSNorm(config.hidden_dim)

            self.mhc_pre = mHCLayer(
                hidden_dim=config.hidden_dim,
                num_flows=config.mhc_flows,
                sinkhorn_iters=config.mhc_sinkhorn_iters,
                dropout=config.dropout,
            )

            self.mhc_post = mHCLayer(
                hidden_dim=config.hidden_dim,
                num_flows=config.mhc_flows,
                sinkhorn_iters=config.mhc_sinkhorn_iters,
                dropout=config.dropout,
            )
        else:
            self.mhc_pre = None
            self.mhc_post = None

    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_offset: int = 0,
    ) -> Tensor:
        """
        Forward pass through the transformer block.

        Args:
            x: Input tensor [B, S, hidden_dim]
            attention_mask: Optional attention mask [B, S]
            position_offset: Position offset for RoPE

        Returns:
            Output tensor [B, S, hidden_dim]
        """
        # mHC pre-attention (if enabled)
        if self.mhc_pre is not None:
            x = x + self.mhc_pre(self.norm_mhc_pre(x))

        # Attention with residual
        x = x + self.attention(
            self.norm1(x),
            attention_mask=attention_mask,
            position_offset=position_offset,
        )

        # mHC post-attention (if enabled)
        if self.mhc_post is not None:
            x = x + self.mhc_post(self.norm_mhc_post(x))

        # FFN with residual
        x = x + self.ffn(self.norm2(x))

        return x

    def get_mhc_metrics(self) -> Optional[Tuple[float, float]]:
        """Get Amax metrics from mHC layers."""
        if self.mhc_pre is None:
            return None
        return (self.mhc_pre.amax_gain, self.mhc_post.amax_gain)


class RLMModel(nn.Module):
    """
    RLM-mHC: Recursive Language Model with Manifold-Constrained Hyper-Connections.

    A Transformer-based language model with:
    - mHC layers for stabilized training
    - Rotary Position Embeddings (RoPE)
    - Flash Attention 2 support
    - Gradient checkpointing support

    Args:
        config: Model configuration
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config

        # Token embeddings
        self.embeddings = TokenEmbedding(
            vocab_size=config.vocab_size,
            hidden_dim=config.hidden_dim,
        )

        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(config, layer_idx=i)
            for i in range(config.num_layers)
        ])

        # Final layer norm
        self.norm = RMSNorm(config.hidden_dim)

        # LM head (weight tied to embeddings)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.embeddings.weight

        # Gradient checkpointing
        self._gradient_checkpointing = config.gradient_checkpointing

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Initialize weights with scaled initialization."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory efficiency."""
        self._gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self._gradient_checkpointing = False

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        position_offset: int = 0,
        output_hidden_states: bool = False,
    ) -> ModelOutput:
        """
        Forward pass through the model.

        Args:
            input_ids: Token IDs [B, S]
            attention_mask: Attention mask [B, S]
            labels: Target labels for loss computation [B, S]
            position_offset: Position offset for RoPE
            output_hidden_states: Whether to return all hidden states

        Returns:
            ModelOutput with logits, optional loss, and optional hidden states
        """
        B, S = input_ids.shape

        # Truncate if sequence exceeds max_seq_len (RoPE limitation)
        if S > self.config.max_seq_len:
            input_ids = input_ids[:, -self.config.max_seq_len:]
            if attention_mask is not None:
                attention_mask = attention_mask[:, -self.config.max_seq_len:]
            if labels is not None:
                labels = labels[:, -self.config.max_seq_len:]
            S = self.config.max_seq_len

        # Token embeddings
        hidden_states = self.embeddings(input_ids)

        # Collect hidden states if requested
        all_hidden_states = [hidden_states] if output_hidden_states else None

        # Forward through layers
        for layer in self.layers:
            if self._gradient_checkpointing and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    attention_mask,
                    position_offset,
                    use_reentrant=False,
                )
            else:
                hidden_states = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_offset=position_offset,
                )

            if output_hidden_states:
                all_hidden_states.append(hidden_states)

        # Final normalization
        hidden_states = self.norm(hidden_states)

        # LM head
        logits = self.lm_head(hidden_states)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Cross-entropy loss
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return ModelOutput(
            logits=logits,
            loss=loss,
            hidden_states=tuple(all_hidden_states) if all_hidden_states else None,
        )

    @torch.inference_mode()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> Tensor:
        """
        Generate text autoregressively.

        Args:
            input_ids: Input token IDs [B, S]
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            do_sample: Whether to sample (False = greedy)
            pad_token_id: Padding token ID
            eos_token_id: End of sequence token ID

        Returns:
            Generated token IDs [B, S + max_new_tokens]
        """
        self.eval()
        device = input_ids.device

        for _ in range(max_new_tokens):
            # Truncate if needed
            if input_ids.shape[1] > self.config.max_seq_len:
                input_ids = input_ids[:, -self.config.max_seq_len:]

            # Forward pass
            outputs = self.forward(input_ids)
            next_token_logits = outputs.logits[:, -1, :]  # [B, vocab_size]

            # Apply temperature
            # Guard against temperature=0 which causes division by zero
            if temperature == 0.0:
                # temperature=0 means greedy decoding
                do_sample = False
            elif temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Apply top-k filtering
            if top_k is not None and top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')

            # Apply top-p (nucleus) filtering
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = float('-inf')

            # Sample or argmax
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Check for EOS
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return input_ids

    def get_mhc_metrics(self) -> List[Tuple[float, float]]:
        """Get Amax metrics from all mHC layers."""
        metrics = []
        for layer in self.layers:
            layer_metrics = layer.get_mhc_metrics()
            if layer_metrics is not None:
                metrics.append(layer_metrics)
        return metrics

    def num_parameters(self, include_embeddings: bool = True) -> int:
        """Count the number of parameters in the model."""
        total = sum(p.numel() for p in self.parameters())
        if not include_embeddings:
            total -= self.embeddings.weight.numel()
        return total

    @classmethod
    def from_pretrained(cls, path: Union[str, Path]) -> "RLMModel":
        """
        Load a pretrained model from a checkpoint.

        Args:
            path: Path to checkpoint directory

        Returns:
            Loaded model
        """
        path = Path(path)

        # Load config
        config = ModelConfig.from_pretrained(path)

        # Create model
        model = cls(config)

        # Load weights
        weights_path = path / "model.safetensors"
        if weights_path.exists():
            from safetensors.torch import load_file
            state_dict = load_file(weights_path)
        else:
            weights_path = path / "pytorch_model.bin"
            state_dict = torch.load(weights_path, map_location="cpu")

        # Use strict=False because lm_head.weight is tied to embeddings.weight
        # and may not be saved separately in the checkpoint
        model.load_state_dict(state_dict, strict=False)

        return model

    def save_pretrained(self, path: Union[str, Path]):
        """
        Save the model to a checkpoint directory.

        Args:
            path: Path to checkpoint directory
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save config
        self.config.save(path / "config.yaml")

        # Save weights - exclude lm_head.weight since it's tied to embeddings.weight
        # safetensors doesn't support shared tensors, so we filter them out
        from safetensors.torch import save_file
        state_dict = {
            k: v for k, v in self.state_dict().items()
            if k != "lm_head.weight"  # Tied to embeddings.weight
        }
        save_file(state_dict, path / "model.safetensors")
