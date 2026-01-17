"""
Tests for RLM-mHC Transformer model.

Tests:
- Model configuration
- Component shapes (RMSNorm, SwiGLU, Attention)
- Full model forward pass
- Generation
"""

import pytest
import torch

from rlm_mhc.model.config import ModelConfig
from rlm_mhc.model.components import RMSNorm, SwiGLU, RotaryEmbedding, TokenEmbedding
from rlm_mhc.model.attention import MultiHeadAttention, CausalSelfAttention
from rlm_mhc.model.transformer import TransformerBlock, RLMModel


class TestModelConfig:
    """Tests for ModelConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ModelConfig()

        assert config.hidden_dim == 2048
        assert config.num_layers == 24
        assert config.num_heads == 16
        assert config.head_dim == 128
        assert config.mhc_enabled is True

    def test_config_validation(self):
        """Test that invalid configs raise errors."""
        with pytest.raises(ValueError):
            # hidden_dim not divisible by num_heads
            ModelConfig(hidden_dim=100, num_heads=16)

    def test_parameter_count(self):
        """Test parameter count estimation."""
        config = ModelConfig(
            hidden_dim=256,
            num_layers=4,
            num_heads=4,
            vocab_size=1000,
        )

        params = config.num_parameters()
        assert params > 0
        assert isinstance(params, int)

    def test_small_config(self):
        """Test with small config for testing."""
        config = ModelConfig(
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            head_dim=16,
            ffn_dim=171,  # 8/3 * 64
            vocab_size=100,
            max_seq_len=128,
        )

        assert config.head_dim == config.hidden_dim // config.num_heads


class TestRMSNorm:
    """Tests for RMSNorm."""

    def test_output_shape(self):
        """Test RMSNorm output shape."""
        norm = RMSNorm(64)
        x = torch.randn(2, 10, 64)
        output = norm(x)

        assert output.shape == x.shape

    def test_normalization(self):
        """Test that RMSNorm normalizes."""
        norm = RMSNorm(64)
        x = torch.randn(2, 10, 64) * 100  # Large values

        output = norm(x)

        # RMS of output should be approximately 1 (scaled by weight)
        rms = torch.sqrt(torch.mean(output ** 2, dim=-1))
        assert rms.mean() < 10  # Should be normalized

    def test_gradient_flow(self):
        """Test gradient flow through RMSNorm."""
        norm = RMSNorm(64)
        x = torch.randn(2, 10, 64, requires_grad=True)

        output = norm(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None


class TestSwiGLU:
    """Tests for SwiGLU FFN."""

    def test_output_shape(self):
        """Test SwiGLU output shape."""
        ffn = SwiGLU(hidden_dim=64, ffn_dim=171)
        x = torch.randn(2, 10, 64)
        output = ffn(x)

        assert output.shape == x.shape

    def test_gradient_flow(self):
        """Test gradient flow through SwiGLU."""
        ffn = SwiGLU(hidden_dim=64, ffn_dim=171)
        x = torch.randn(2, 10, 64, requires_grad=True)

        output = ffn(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None


class TestRotaryEmbedding:
    """Tests for Rotary Position Embeddings."""

    def test_output_shapes(self):
        """Test RoPE output shapes."""
        rope = RotaryEmbedding(dim=64, max_seq_len=128)

        q = torch.randn(2, 4, 10, 64)  # [B, heads, S, head_dim]
        k = torch.randn(2, 4, 10, 64)

        q_rot, k_rot = rope(q, k)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_position_offset(self):
        """Test RoPE with position offset (for KV cache)."""
        rope = RotaryEmbedding(dim=64, max_seq_len=128)

        q = torch.randn(2, 4, 10, 64)
        k = torch.randn(2, 4, 10, 64)

        q_rot, k_rot = rope(q, k, position_offset=50)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape


class TestMultiHeadAttention:
    """Tests for Multi-Head Attention."""

    @pytest.fixture
    def attention(self):
        """Create test attention layer."""
        return MultiHeadAttention(
            hidden_dim=64,
            num_heads=4,
            head_dim=16,
            use_flash_attn=False,  # Use standard attention for testing
        )

    def test_output_shape(self, attention):
        """Test attention output shape."""
        x = torch.randn(2, 10, 64)
        output = attention(x)

        assert output.shape == x.shape

    def test_causal_attention(self, attention):
        """Test causal masking."""
        x = torch.randn(2, 10, 64)
        output_causal = attention(x, is_causal=True)
        output_non_causal = attention(x, is_causal=False)

        # Outputs should be different with/without causal mask
        assert not torch.allclose(output_causal, output_non_causal)

    def test_attention_mask(self, attention):
        """Test attention with mask."""
        x = torch.randn(2, 10, 64)
        mask = torch.ones(2, 10)
        mask[:, 5:] = 0  # Mask second half

        output = attention(x, attention_mask=mask)

        assert output.shape == x.shape


class TestTransformerBlock:
    """Tests for Transformer Block."""

    @pytest.fixture
    def config(self):
        """Create small test config."""
        return ModelConfig(
            hidden_dim=64,
            num_layers=1,
            num_heads=4,
            head_dim=16,
            ffn_dim=171,
            vocab_size=100,
            max_seq_len=128,
            mhc_enabled=True,
            mhc_flows=4,
        )

    def test_output_shape(self, config):
        """Test block output shape."""
        block = TransformerBlock(config)
        x = torch.randn(2, 10, 64)

        output = block(x)

        assert output.shape == x.shape

    def test_block_with_mhc(self, config):
        """Test block with mHC enabled."""
        config.mhc_enabled = True
        block = TransformerBlock(config)

        assert block.mhc_pre is not None
        assert block.mhc_post is not None

        x = torch.randn(2, 10, 64)
        output = block(x)
        assert output.shape == x.shape

    def test_block_without_mhc(self, config):
        """Test block without mHC."""
        config.mhc_enabled = False
        block = TransformerBlock(config)

        assert block.mhc_pre is None
        assert block.mhc_post is None

        x = torch.randn(2, 10, 64)
        output = block(x)
        assert output.shape == x.shape

    def test_mhc_metrics(self, config):
        """Test mHC metrics retrieval."""
        config.mhc_enabled = True
        block = TransformerBlock(config)

        metrics = block.get_mhc_metrics()

        assert metrics is not None
        assert len(metrics) == 2  # pre and post


class TestRLMModel:
    """Tests for full RLM Model."""

    @pytest.fixture
    def small_config(self):
        """Create small config for testing."""
        return ModelConfig(
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            head_dim=16,
            ffn_dim=171,
            vocab_size=100,
            max_seq_len=128,
            mhc_enabled=True,
            mhc_flows=4,
        )

    @pytest.fixture
    def model(self, small_config):
        """Create small test model."""
        return RLMModel(small_config)

    def test_forward_shape(self, model):
        """Test model forward pass shape."""
        input_ids = torch.randint(0, 100, (2, 10))

        output = model(input_ids)

        assert output.logits.shape == (2, 10, 100)  # [B, S, vocab_size]

    def test_forward_with_labels(self, model):
        """Test forward pass with labels returns loss."""
        input_ids = torch.randint(0, 100, (2, 10))
        labels = torch.randint(0, 100, (2, 10))

        output = model(input_ids, labels=labels)

        assert output.loss is not None
        assert output.loss.dim() == 0  # Scalar

    def test_forward_with_mask(self, model):
        """Test forward pass with attention mask."""
        input_ids = torch.randint(0, 100, (2, 10))
        attention_mask = torch.ones(2, 10)

        output = model(input_ids, attention_mask=attention_mask)

        assert output.logits.shape == (2, 10, 100)

    def test_hidden_states(self, model):
        """Test returning hidden states."""
        input_ids = torch.randint(0, 100, (2, 10))

        output = model(input_ids, output_hidden_states=True)

        assert output.hidden_states is not None
        assert len(output.hidden_states) == 3  # input + 2 layers

    def test_generate(self, model):
        """Test generation."""
        input_ids = torch.randint(0, 100, (1, 5))

        output_ids = model.generate(
            input_ids,
            max_new_tokens=10,
            temperature=1.0,
            do_sample=False,
        )

        assert output_ids.shape[1] == 15  # 5 input + 10 generated

    def test_generate_with_sampling(self, model):
        """Test generation with sampling."""
        input_ids = torch.randint(0, 100, (1, 5))

        output_ids = model.generate(
            input_ids,
            max_new_tokens=10,
            temperature=0.8,
            top_k=10,
            do_sample=True,
        )

        assert output_ids.shape[1] == 15

    def test_parameter_count(self, model):
        """Test parameter counting."""
        params = model.num_parameters()

        assert params > 0
        assert isinstance(params, int)

    def test_gradient_checkpointing(self, model):
        """Test gradient checkpointing toggle."""
        model.gradient_checkpointing_enable()
        assert model._gradient_checkpointing is True

        model.gradient_checkpointing_disable()
        assert model._gradient_checkpointing is False

    def test_mhc_metrics(self, model):
        """Test mHC metrics from full model."""
        metrics = model.get_mhc_metrics()

        assert len(metrics) == 2  # 2 layers
        for pre_amax, post_amax in metrics:
            assert 0 < pre_amax <= 2.0
            assert 0 < post_amax <= 2.0


class TestModelCUDA:
    """Tests for model on CUDA."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_forward(self):
        """Test model forward on CUDA."""
        config = ModelConfig(
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            head_dim=16,
            ffn_dim=171,
            vocab_size=100,
            max_seq_len=128,
        )
        model = RLMModel(config).cuda()
        input_ids = torch.randint(0, 100, (2, 10)).cuda()

        output = model(input_ids)

        assert output.logits.device.type == 'cuda'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_generate(self):
        """Test generation on CUDA."""
        config = ModelConfig(
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            head_dim=16,
            ffn_dim=171,
            vocab_size=100,
            max_seq_len=128,
        )
        model = RLMModel(config).cuda()
        input_ids = torch.randint(0, 100, (1, 5)).cuda()

        output_ids = model.generate(input_ids, max_new_tokens=10)

        assert output_ids.device.type == 'cuda'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
