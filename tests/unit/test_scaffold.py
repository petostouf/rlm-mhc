"""
Tests for RLM Scaffold (REPL) module.

Tests:
- Context chunking and caching
- Context manager operations
- REPL functions (peek, search, llm_query)
- Recursion tracking
"""

import pytest
import torch
from unittest.mock import MagicMock, patch

from rlm_mhc.types import ContextHandle, ChunkConfig, SessionConfig
from rlm_mhc.scaffold.chunking import ChunkStrategy, Chunk, LRUCache
from rlm_mhc.scaffold.context import ContextManager
from rlm_mhc.scaffold.repl import RLMSession, RecursionTracker


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 1

    def encode(self, text):
        """Simple encoding: each character is a token."""
        return [ord(c) for c in text]

    def decode(self, tokens):
        """Simple decoding: each token is a character."""
        return "".join(chr(t) for t in tokens if 32 <= t < 127)

    def __call__(self, text, **kwargs):
        """Tokenizer call interface."""
        tokens = self.encode(text)
        return {
            'input_ids': torch.tensor([tokens]),
            'attention_mask': torch.ones(1, len(tokens)),
        }


class TestLRUCache:
    """Tests for LRU cache."""

    def test_basic_operations(self):
        """Test get, put, contains."""
        cache = LRUCache(maxsize=3)

        cache.put(1, torch.tensor([1, 2, 3]))
        cache.put(2, torch.tensor([4, 5, 6]))

        assert 1 in cache
        assert 2 in cache
        assert 3 not in cache

        result = cache.get(1)
        assert result is not None
        assert torch.equal(result, torch.tensor([1, 2, 3]))

    def test_eviction(self):
        """Test LRU eviction."""
        cache = LRUCache(maxsize=2)

        cache.put(1, torch.tensor([1]))
        cache.put(2, torch.tensor([2]))
        cache.put(3, torch.tensor([3]))  # Should evict 1

        assert 1 not in cache
        assert 2 in cache
        assert 3 in cache

    def test_access_updates_order(self):
        """Test that access updates LRU order."""
        cache = LRUCache(maxsize=2)

        cache.put(1, torch.tensor([1]))
        cache.put(2, torch.tensor([2]))
        cache.get(1)  # Access 1, making 2 the oldest
        cache.put(3, torch.tensor([3]))  # Should evict 2

        assert 1 in cache
        assert 2 not in cache
        assert 3 in cache

    def test_clear(self):
        """Test cache clear."""
        cache = LRUCache(maxsize=3)
        cache.put(1, torch.tensor([1]))
        cache.put(2, torch.tensor([2]))

        cache.clear()

        assert len(cache) == 0
        assert 1 not in cache


class TestChunkStrategy:
    """Tests for chunk strategy."""

    @pytest.fixture
    def tokenizer(self):
        return MockTokenizer()

    @pytest.fixture
    def config(self):
        return ChunkConfig(chunk_size=100, overlap=20, max_chunks_in_memory=4)

    def test_load_short_context(self, tokenizer, config):
        """Test loading a short context (single chunk)."""
        strategy = ChunkStrategy(config, tokenizer)

        handle = strategy.load_context("Hello World!")

        assert handle.total_tokens == 12
        assert handle.num_chunks == 1

    def test_load_long_context(self, tokenizer, config):
        """Test loading a long context (multiple chunks)."""
        strategy = ChunkStrategy(config, tokenizer)

        # Create text longer than chunk_size
        text = "A" * 250

        handle = strategy.load_context(text)

        assert handle.total_tokens == 250
        assert handle.num_chunks > 1

    def test_chunk_overlap(self, tokenizer, config):
        """Test that chunks have proper overlap."""
        strategy = ChunkStrategy(config, tokenizer)

        text = "A" * 200
        strategy.load_context(text)

        # With chunk_size=100, overlap=20, stride=80
        # Chunk 0: 0-100
        # Chunk 1: 80-180
        # Chunk 2: 160-200

        assert len(strategy.chunks) >= 2

        # Check overlap
        if len(strategy.chunks) >= 2:
            chunk0_end = strategy.chunks[0].end_pos
            chunk1_start = strategy.chunks[1].start_pos
            overlap = chunk0_end - chunk1_start
            assert overlap == config.overlap

    def test_get_chunk(self, tokenizer, config):
        """Test getting individual chunks."""
        strategy = ChunkStrategy(config, tokenizer)

        text = "Hello World! " * 20
        strategy.load_context(text)

        chunk0 = strategy.get_chunk(0)

        assert isinstance(chunk0, torch.Tensor)
        assert chunk0.dtype == torch.long

    def test_get_window(self, tokenizer, config):
        """Test getting a window of tokens."""
        strategy = ChunkStrategy(config, tokenizer)

        text = "ABCDEFGHIJ" * 30  # 300 chars
        strategy.load_context(text)

        window = strategy.get_window(10, 50)

        assert len(window) == 40

    def test_cache_behavior(self, tokenizer, config):
        """Test that cache is used."""
        strategy = ChunkStrategy(config, tokenizer)

        text = "A" * 500
        strategy.load_context(text)

        # First access
        chunk0_first = strategy.get_chunk(0)

        # Should be in cache now
        assert 0 in strategy.cache

        # Second access should return cached value
        chunk0_second = strategy.get_chunk(0)

        assert torch.equal(chunk0_first, chunk0_second)


class TestContextManager:
    """Tests for context manager."""

    @pytest.fixture
    def tokenizer(self):
        return MockTokenizer()

    @pytest.fixture
    def manager(self, tokenizer):
        return ContextManager(tokenizer)

    def test_load_string(self, manager):
        """Test loading from string."""
        handle = manager.load("This is a test document.")

        assert handle.total_tokens > 0
        assert manager.current_context is not None

    def test_load_clears_previous(self, manager):
        """Test that loading new context clears previous."""
        manager.load("First document")
        first_tokens = manager.total_tokens

        manager.load("Second longer document with more text")
        second_tokens = manager.total_tokens

        assert second_tokens != first_tokens

    def test_get_tokens(self, manager):
        """Test getting token range."""
        manager.load("Hello World!")

        tokens = manager.get_tokens(0, 5)

        assert isinstance(tokens, torch.Tensor)
        assert len(tokens) == 5

    def test_get_text(self, manager):
        """Test getting decoded text."""
        manager.load("Hello World!")

        text = manager.get_text(0, 5)

        assert isinstance(text, str)
        assert len(text) == 5
        assert text == "Hello"

    def test_metadata(self, manager):
        """Test metadata property."""
        manager.load("Test document")

        meta = manager.metadata

        assert 'total_tokens' in meta
        assert 'num_chunks' in meta
        assert 'chunk_size' in meta

    def test_clear(self, manager):
        """Test clearing context."""
        manager.load("Test")
        manager.clear()

        assert manager.current_context is None
        assert manager.total_tokens == 0


class TestRecursionTracker:
    """Tests for recursion tracker."""

    def test_basic_tracking(self):
        """Test basic recursion tracking."""
        tracker = RecursionTracker(max_depth=5)

        assert tracker.current_depth == 0

        with tracker.track() as depth:
            assert depth == 1
            assert tracker.current_depth == 1

        assert tracker.current_depth == 0

    def test_nested_tracking(self):
        """Test nested recursion tracking."""
        tracker = RecursionTracker(max_depth=5)

        with tracker.track() as depth1:
            assert depth1 == 1

            with tracker.track() as depth2:
                assert depth2 == 2

                with tracker.track() as depth3:
                    assert depth3 == 3

            assert tracker.current_depth == 1

        assert tracker.current_depth == 0

    def test_history(self):
        """Test recursion history."""
        tracker = RecursionTracker(max_depth=5)

        with tracker.track():
            with tracker.track():
                pass

        history = tracker.history

        assert len(history) == 2
        assert history[0].depth == 1
        assert history[1].depth == 2
        assert history[0].completed
        assert history[1].completed

    def test_reset(self):
        """Test tracker reset."""
        tracker = RecursionTracker(max_depth=5)

        with tracker.track():
            pass

        tracker.reset()

        assert tracker.current_depth == 0
        assert len(tracker.history) == 0


class TestRLMSession:
    """Tests for RLM Session."""

    @pytest.fixture
    def mock_model(self):
        """Create mock model."""
        model = MagicMock()
        model.config = MagicMock()
        model.config.hidden_dim = 64

        # Mock forward pass
        def mock_forward(*args, **kwargs):
            result = MagicMock()
            result.logits = torch.randn(1, 10, 100)
            result.hidden_states = (torch.randn(1, 10, 64),)
            return result

        model.side_effect = mock_forward
        model.return_value = mock_forward()

        # Mock generate
        def mock_generate(input_ids, **kwargs):
            max_new = kwargs.get('max_new_tokens', 10)
            return torch.cat([input_ids, torch.randint(0, 100, (1, max_new))], dim=1)

        model.generate = mock_generate

        return model

    @pytest.fixture
    def tokenizer(self):
        return MockTokenizer()

    @pytest.fixture
    def session(self, mock_model, tokenizer):
        """Create test session."""
        return RLMSession(
            model=mock_model,
            tokenizer=tokenizer,
            config=SessionConfig(max_recursion=3),
            device=torch.device('cpu'),
        )

    def test_load_context(self, session):
        """Test loading context."""
        ctx = session.load("This is a test document.")

        assert isinstance(ctx, ContextHandle)
        assert ctx.total_tokens > 0

    def test_peek(self, session):
        """Test peek function."""
        ctx = session.load("Hello World!")

        text = session.peek(ctx, 0, 5)

        assert isinstance(text, str)
        assert len(text) == 5

    def test_peek_tokens(self, session):
        """Test peek returning tokens."""
        ctx = session.load("Hello World!")

        tokens = session.peek(ctx, 0, 5, decode=False)

        assert isinstance(tokens, list)
        assert len(tokens) == 5

    def test_context_metadata(self, session):
        """Test context metadata."""
        session.load("Test document")

        meta = session.context_metadata

        assert 'total_tokens' in meta

    def test_reset(self, session):
        """Test session reset."""
        session.load("Test")
        session.reset()

        assert session.context_manager.current_context is None
        assert session.recursion_tracker.current_depth == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
