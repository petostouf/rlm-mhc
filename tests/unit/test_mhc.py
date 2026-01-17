"""
Tests for mHC (Manifold-Constrained Hyper-Connections) module.

Tests:
- Sinkhorn-Knopp algorithm convergence
- Doubly stochastic matrix properties
- mHCLayer forward pass and shapes
- Amax Gain computation
"""

import pytest
import torch

from rlm_mhc.model.mhc.sinkhorn import sinkhorn_knopp, sinkhorn_knopp_log
from rlm_mhc.model.mhc.layers import mHCLayer, mHCBlock


class TestSinkhornKnopp:
    """Tests for Sinkhorn-Knopp algorithm."""

    def test_doubly_stochastic_rows_sum_to_one(self):
        """Test that rows of output matrix sum to 1."""
        W = torch.randn(4, 4)
        P = sinkhorn_knopp(W, n_iters=20)

        row_sums = P.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(4), atol=1e-5), (
            f"Row sums should be 1, got {row_sums}"
        )

    def test_doubly_stochastic_cols_sum_to_one(self):
        """Test that columns of output matrix sum to 1."""
        W = torch.randn(4, 4)
        P = sinkhorn_knopp(W, n_iters=20)

        col_sums = P.sum(dim=0)
        assert torch.allclose(col_sums, torch.ones(4), atol=1e-5), (
            f"Column sums should be 1, got {col_sums}"
        )

    def test_all_elements_positive(self):
        """Test that all elements are positive."""
        W = torch.randn(4, 4)
        P = sinkhorn_knopp(W, n_iters=20)

        assert (P > 0).all(), "All elements should be positive"

    def test_convergence_with_more_iterations(self):
        """Test that more iterations improve convergence."""
        W = torch.randn(4, 4)

        P_5 = sinkhorn_knopp(W, n_iters=5)
        P_20 = sinkhorn_knopp(W, n_iters=20)

        # P_20 should be closer to doubly stochastic
        error_5 = (P_5.sum(dim=1) - 1).abs().max()
        error_20 = (P_20.sum(dim=1) - 1).abs().max()

        assert error_20 <= error_5, "More iterations should improve convergence"

    def test_gradient_flow(self):
        """Test that gradients flow through Sinkhorn."""
        W = torch.randn(4, 4, requires_grad=True)
        P = sinkhorn_knopp(W, n_iters=20)

        loss = P.sum()
        loss.backward()

        assert W.grad is not None, "Gradients should flow to input"
        assert not torch.isnan(W.grad).any(), "Gradients should not contain NaN"

    def test_different_matrix_sizes(self):
        """Test with different matrix sizes."""
        for size in [2, 4, 8, 16]:
            W = torch.randn(size, size)
            P = sinkhorn_knopp(W, n_iters=20)

            assert P.shape == (size, size)
            assert torch.allclose(P.sum(dim=0), torch.ones(size), atol=1e-4)
            assert torch.allclose(P.sum(dim=1), torch.ones(size), atol=1e-4)

    def test_log_space_version(self):
        """Test log-space Sinkhorn for numerical stability."""
        log_W = torch.randn(4, 4)
        P = sinkhorn_knopp_log(log_W, n_iters=20)

        assert torch.allclose(P.sum(dim=0), torch.ones(4), atol=1e-4)
        assert torch.allclose(P.sum(dim=1), torch.ones(4), atol=1e-4)


class TestMHCLayer:
    """Tests for mHCLayer."""

    @pytest.fixture
    def mhc_layer(self):
        """Create a test mHC layer."""
        return mHCLayer(hidden_dim=64, num_flows=4, sinkhorn_iters=20)

    def test_output_shape(self, mhc_layer):
        """Test that output shape matches input shape."""
        x = torch.randn(2, 10, 64)  # [B, S, C]
        output = mhc_layer(x)

        assert output.shape == x.shape, (
            f"Output shape {output.shape} should match input {x.shape}"
        )

    def test_forward_no_nan(self, mhc_layer):
        """Test that forward pass doesn't produce NaN."""
        x = torch.randn(2, 10, 64)
        output = mhc_layer(x)

        assert not torch.isnan(output).any(), "Output should not contain NaN"

    def test_gradient_flow(self, mhc_layer):
        """Test gradient flow through mHC layer."""
        x = torch.randn(2, 10, 64, requires_grad=True)
        output = mhc_layer(x)

        loss = output.sum()
        loss.backward()

        assert x.grad is not None, "Gradients should flow to input"
        assert not torch.isnan(x.grad).any(), "Gradients should not contain NaN"

    def test_connection_matrix_doubly_stochastic(self, mhc_layer):
        """Test that connection matrix is doubly stochastic."""
        P = mhc_layer.get_connection_matrix()

        assert P.shape == (4, 4)
        assert torch.allclose(P.sum(dim=0), torch.ones(4), atol=1e-4)
        assert torch.allclose(P.sum(dim=1), torch.ones(4), atol=1e-4)

    def test_amax_gain_reasonable(self, mhc_layer):
        """Test that Amax Gain is in reasonable range."""
        amax = mhc_layer.amax_gain

        # For a doubly stochastic 4x4 matrix, max element is typically < 1
        # but can be higher depending on initialization
        assert 0 < amax <= 2.0, f"Amax should be in (0, 2], got {amax}"

    def test_different_batch_sizes(self, mhc_layer):
        """Test with different batch sizes."""
        for batch_size in [1, 2, 4, 8]:
            x = torch.randn(batch_size, 10, 64)
            output = mhc_layer(x)
            assert output.shape == x.shape

    def test_different_sequence_lengths(self, mhc_layer):
        """Test with different sequence lengths."""
        for seq_len in [1, 10, 100, 1000]:
            x = torch.randn(2, seq_len, 64)
            output = mhc_layer(x)
            assert output.shape == x.shape


class TestMHCBlock:
    """Tests for mHCBlock (pre and post attention)."""

    @pytest.fixture
    def mhc_block(self):
        """Create a test mHC block."""
        return mHCBlock(hidden_dim=64, num_flows=4, sinkhorn_iters=20)

    def test_pre_attention_shape(self, mhc_block):
        """Test pre-attention transformation shape."""
        x = torch.randn(2, 10, 64)
        output = mhc_block.forward_pre(x)
        assert output.shape == x.shape

    def test_post_attention_shape(self, mhc_block):
        """Test post-attention transformation shape."""
        x = torch.randn(2, 10, 64)
        output = mhc_block.forward_post(x)
        assert output.shape == x.shape

    def test_residual_connection(self, mhc_block):
        """Test that residual connections are applied."""
        x = torch.randn(2, 10, 64)

        # With residual
        out_with_residual = mhc_block.forward_pre(x, residual=True)

        # Without residual
        out_without_residual = mhc_block.forward_pre(x, residual=False)

        # Output with residual should be different
        assert not torch.allclose(out_with_residual, out_without_residual)

    def test_amax_metrics(self, mhc_block):
        """Test Amax metrics retrieval."""
        pre_amax, post_amax = mhc_block.get_amax_metrics()

        assert isinstance(pre_amax, float)
        assert isinstance(post_amax, float)
        assert 0 < pre_amax <= 2.0
        assert 0 < post_amax <= 2.0


class TestMHCCUDA:
    """Tests for mHC on CUDA (if available)."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_forward(self):
        """Test mHC forward pass on CUDA."""
        layer = mHCLayer(hidden_dim=64, num_flows=4).cuda()
        x = torch.randn(2, 10, 64).cuda()

        output = layer(x)

        assert output.device.type == 'cuda'
        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_backward(self):
        """Test mHC backward pass on CUDA."""
        layer = mHCLayer(hidden_dim=64, num_flows=4).cuda()
        # Create tensor on CUDA first, then set requires_grad to keep it as leaf
        x = torch.randn(2, 10, 64, device='cuda', requires_grad=True)

        output = layer(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.device.type == 'cuda'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
