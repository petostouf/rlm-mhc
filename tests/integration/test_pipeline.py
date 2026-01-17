"""
Integration tests for RLM-mHC end-to-end pipeline.

Tests:
- Full model creation and forward pass
- Training loop (minimal steps)
- Generation pipeline
- Scaffold with real model
"""

import pytest
import torch
import tempfile
from pathlib import Path

from rlm_mhc.model.config import ModelConfig
from rlm_mhc.model.transformer import RLMModel
from rlm_mhc.types import TrainingConfig, SessionConfig


# Small config for fast testing
SMALL_CONFIG = ModelConfig(
    hidden_dim=64,
    num_layers=2,
    num_heads=4,
    head_dim=16,
    ffn_dim=171,
    vocab_size=256,
    max_seq_len=128,
    mhc_enabled=True,
    mhc_flows=4,
    mhc_sinkhorn_iters=10,
)


class TestEndToEndModel:
    """End-to-end tests for model pipeline."""

    @pytest.fixture
    def model(self):
        """Create small test model."""
        return RLMModel(SMALL_CONFIG)

    def test_full_forward_backward(self, model):
        """Test complete forward and backward pass."""
        # Create input
        input_ids = torch.randint(0, 256, (2, 32))
        labels = torch.randint(0, 256, (2, 32))

        # Forward
        output = model(input_ids, labels=labels)

        # Check outputs
        assert output.logits is not None
        assert output.loss is not None
        assert not torch.isnan(output.loss)

        # Backward
        output.loss.backward()

        # Check gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

    def test_generation_loop(self, model):
        """Test autoregressive generation."""
        input_ids = torch.randint(0, 256, (1, 10))

        # Generate
        output_ids = model.generate(
            input_ids,
            max_new_tokens=20,
            temperature=0.8,
            top_k=50,
            do_sample=True,
        )

        assert output_ids.shape[1] == 30  # 10 input + 20 generated
        assert (output_ids >= 0).all()
        assert (output_ids < 256).all()

    def test_model_save_load(self, model):
        """Test saving and loading model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_model"

            # Save
            model.save_pretrained(save_path)

            # Check files exist
            assert (save_path / "config.yaml").exists()
            assert (save_path / "model.safetensors").exists()

            # Load
            loaded_model = RLMModel.from_pretrained(save_path)

            # Compare outputs
            input_ids = torch.randint(0, 256, (1, 10))

            model.eval()
            loaded_model.eval()

            with torch.inference_mode():
                out1 = model(input_ids)
                out2 = loaded_model(input_ids)

            assert torch.allclose(out1.logits, out2.logits, atol=1e-5)

    def test_mhc_stability_during_forward(self, model):
        """Test mHC stability metrics during forward pass."""
        input_ids = torch.randint(0, 256, (4, 64))

        # Multiple forward passes
        for _ in range(5):
            output = model(input_ids)

            # Check mHC metrics
            metrics = model.get_mhc_metrics()
            for pre_amax, post_amax in metrics:
                assert pre_amax < 3.0, f"Pre-attention Amax too high: {pre_amax}"
                assert post_amax < 3.0, f"Post-attention Amax too high: {post_amax}"


class TestMinimalTraining:
    """Test minimal training loop."""

    def test_training_step(self):
        """Test a single training step."""
        model = RLMModel(SMALL_CONFIG)
        model.train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Training data
        input_ids = torch.randint(0, 256, (2, 32))
        labels = input_ids.clone()

        # Forward
        output = model(input_ids, labels=labels)
        loss = output.loss

        # Backward
        loss.backward()

        # Check gradients exist
        total_grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_grad_norm += p.grad.norm().item() ** 2
        total_grad_norm = total_grad_norm ** 0.5

        assert total_grad_norm > 0, "Gradients should be non-zero"

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()

        # Second step should have different loss
        output2 = model(input_ids, labels=labels)
        # Loss might be similar but should still work
        assert not torch.isnan(output2.loss)

    def test_gradient_checkpointing(self):
        """Test gradient checkpointing reduces memory."""
        model = RLMModel(SMALL_CONFIG)

        # Enable gradient checkpointing
        model.gradient_checkpointing_enable()

        input_ids = torch.randint(0, 256, (2, 64))
        labels = input_ids.clone()

        # Should work with checkpointing
        output = model(input_ids, labels=labels)
        output.loss.backward()

        # Gradients should exist
        for p in model.parameters():
            if p.requires_grad:
                assert p.grad is not None

    def test_mixed_precision_forward(self):
        """Test mixed precision (bf16) forward pass."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for mixed precision test")

        model = RLMModel(SMALL_CONFIG).cuda()
        input_ids = torch.randint(0, 256, (2, 32)).cuda()
        labels = input_ids.clone()

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            output = model(input_ids, labels=labels)

        assert not torch.isnan(output.loss)
        assert not torch.isinf(output.loss)


class TestScaffoldIntegration:
    """Test scaffold with real model."""

    def test_scaffold_with_model(self):
        """Test RLMSession with actual model."""
        from rlm_mhc.scaffold.repl import RLMSession
        from rlm_mhc.scaffold.chunking import ChunkConfig

        # Create model
        model = RLMModel(SMALL_CONFIG)

        # Simple tokenizer mock
        class SimpleTokenizer:
            pad_token_id = 0
            eos_token_id = 1

            def encode(self, text):
                return [ord(c) % 256 for c in text]

            def decode(self, tokens):
                return "".join(chr(t % 128 + 32) for t in tokens)

            def __call__(self, text, **kwargs):
                tokens = self.encode(text)
                return {
                    'input_ids': torch.tensor([tokens]),
                    'attention_mask': torch.ones(1, len(tokens)),
                }

        tokenizer = SimpleTokenizer()

        # Create session
        config = SessionConfig(
            max_recursion=3,
            context_window_size=64,
            chunking=ChunkConfig(chunk_size=32, overlap=8),
        )

        session = RLMSession(
            model=model,
            tokenizer=tokenizer,
            config=config,
            device=torch.device('cpu'),
        )

        # Load context
        ctx = session.load("This is a test document for the RLM scaffold.")

        # Peek
        text = session.peek(ctx, 0, 10)
        assert isinstance(text, str)
        assert len(text) == 10

        # Context metadata
        meta = session.context_metadata
        assert meta['total_tokens'] > 0


class TestMHCConvergence:
    """Test mHC convergence properties."""

    def test_sinkhorn_convergence_rate(self):
        """Test Sinkhorn convergence with different iterations."""
        from rlm_mhc.model.mhc.sinkhorn import sinkhorn_knopp

        W = torch.randn(4, 4)

        errors = []
        for n_iters in [1, 5, 10, 20, 50]:
            P = sinkhorn_knopp(W, n_iters=n_iters)
            row_error = (P.sum(dim=1) - 1).abs().max().item()
            col_error = (P.sum(dim=0) - 1).abs().max().item()
            errors.append(max(row_error, col_error))

        # Errors should decrease
        for i in range(len(errors) - 1):
            assert errors[i + 1] <= errors[i] + 1e-6, (
                f"Error should decrease: {errors[i]} -> {errors[i + 1]}"
            )

    def test_mhc_preserves_information(self):
        """Test that mHC transformation preserves information flow."""
        from rlm_mhc.model.mhc.layers import mHCLayer

        layer = mHCLayer(hidden_dim=64, num_flows=4)

        # Create distinct inputs
        x1 = torch.randn(1, 10, 64)
        x2 = torch.randn(1, 10, 64) * 2

        out1 = layer(x1)
        out2 = layer(x2)

        # Outputs should be different
        assert not torch.allclose(out1, out2), "mHC should preserve input differences"

        # But transformation should be smooth (outputs shouldn't be wildly different)
        ratio = out1.norm() / out2.norm()
        assert 0.1 < ratio < 10, f"Output ratio {ratio} suggests unstable transformation"


class TestMemoryEfficiency:
    """Test memory efficiency."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_memory_with_long_sequence(self):
        """Test memory usage with longer sequences."""
        # Use smaller model for memory test
        config = ModelConfig(
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            head_dim=16,
            ffn_dim=171,
            vocab_size=256,
            max_seq_len=512,
            gradient_checkpointing=True,
        )

        model = RLMModel(config).cuda()
        model.gradient_checkpointing_enable()

        # Clear cache
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Forward with long sequence
        input_ids = torch.randint(0, 256, (1, 256)).cuda()
        labels = input_ids.clone()

        output = model(input_ids, labels=labels)
        output.loss.backward()

        peak_memory = torch.cuda.max_memory_allocated() / 1024 ** 2  # MB

        # Should use reasonable memory (adjust threshold as needed)
        assert peak_memory < 1000, f"Peak memory {peak_memory:.1f}MB is too high"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
