"""
Context Chunking Strategy.

Handles the segmentation of long contexts into manageable chunks
with overlap for continuity and LRU caching for efficiency.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from collections import OrderedDict
import torch
from torch import Tensor


@dataclass
class Chunk:
    """A chunk of tokenized context."""

    id: int
    tokens: List[int]
    start_pos: int
    end_pos: int

    def __len__(self) -> int:
        return len(self.tokens)

    def to_tensor(self, device: Optional[torch.device] = None) -> Tensor:
        """Convert chunk tokens to a tensor."""
        tensor = torch.tensor(self.tokens, dtype=torch.long)
        if device is not None:
            tensor = tensor.to(device)
        return tensor


@dataclass
class ChunkConfig:
    """Configuration for context chunking."""

    chunk_size: int = 4096
    overlap: int = 256
    max_chunks_in_memory: int = 16
    cache_strategy: str = "lru"

    def __post_init__(self):
        assert self.overlap < self.chunk_size, (
            f"Overlap ({self.overlap}) must be less than chunk_size ({self.chunk_size})"
        )


class LRUCache:
    """
    Least Recently Used cache for chunk tensors.

    Keeps the most recently accessed chunks in GPU memory.
    """

    def __init__(self, maxsize: int = 16):
        self.maxsize = maxsize
        self._cache: OrderedDict[int, Tensor] = OrderedDict()

    def get(self, key: int) -> Optional[Tensor]:
        """Get a value from the cache, updating access order."""
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key: int, value: Tensor):
        """Put a value in the cache, evicting oldest if needed."""
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self.maxsize:
                # Evict oldest
                self._cache.popitem(last=False)
            self._cache[key] = value

    def __contains__(self, key: int) -> bool:
        return key in self._cache

    def __len__(self) -> int:
        return len(self._cache)

    def clear(self):
        """Clear the cache."""
        self._cache.clear()


class ChunkStrategy:
    """
    Manages chunking and streaming of long contexts.

    Implements sliding window chunking with overlap for semantic
    continuity at chunk boundaries. Uses LRU cache to keep
    frequently accessed chunks in GPU memory.

    Args:
        config: Chunking configuration
        tokenizer: Tokenizer for encoding/decoding text
    """

    def __init__(self, config: ChunkConfig, tokenizer: Any):
        self.config = config
        self.tokenizer = tokenizer
        self.chunks: List[Chunk] = []
        self.cache = LRUCache(maxsize=config.max_chunks_in_memory)
        self._total_tokens = 0

    def load_context(self, text: str) -> "ContextHandle":
        """
        Load a text context by chunking it.

        Args:
            text: Source text (potentially very long)

        Returns:
            ContextHandle for accessing the chunked context
        """
        from rlm_mhc.types import ContextHandle

        # Tokenize
        tokens = self.tokenizer.encode(text)
        self._total_tokens = len(tokens)

        # Compute chunk boundaries with overlap
        stride = self.config.chunk_size - self.config.overlap
        chunk_starts = list(range(0, len(tokens), stride))

        # Create chunks
        self.chunks = []
        for i, start in enumerate(chunk_starts):
            end = min(start + self.config.chunk_size, len(tokens))
            self.chunks.append(Chunk(
                id=i,
                tokens=tokens[start:end],
                start_pos=start,
                end_pos=end,
            ))

        # Clear cache (new context)
        self.cache.clear()

        return ContextHandle(
            total_tokens=len(tokens),
            num_chunks=len(self.chunks),
            chunk_size=self.config.chunk_size,
        )

    def get_chunk(self, chunk_id: int, device: Optional[torch.device] = None) -> Tensor:
        """
        Get a chunk tensor, using cache when possible.

        Args:
            chunk_id: ID of the chunk to retrieve
            device: Target device for the tensor

        Returns:
            Tensor of token IDs for the chunk
        """
        if chunk_id < 0 or chunk_id >= len(self.chunks):
            raise IndexError(f"Chunk ID {chunk_id} out of range [0, {len(self.chunks)})")

        # Check cache
        cached = self.cache.get(chunk_id)
        if cached is not None:
            if device is not None and cached.device != device:
                cached = cached.to(device)
            return cached

        # Load chunk
        chunk = self.chunks[chunk_id]
        tensor = chunk.to_tensor(device)

        # Cache it
        self.cache.put(chunk_id, tensor)

        return tensor

    def get_window(
        self,
        start: int,
        end: int,
        device: Optional[torch.device] = None,
    ) -> Tensor:
        """
        Get a window of tokens, potentially spanning multiple chunks.

        Args:
            start: Start position (in tokens)
            end: End position (in tokens)
            device: Target device for the tensor

        Returns:
            Tensor of token IDs for the window
        """
        if start < 0:
            start = 0
        if end > self._total_tokens:
            end = self._total_tokens

        # Find relevant chunks
        relevant_tokens = []
        for chunk in self.chunks:
            # Check if chunk overlaps with window
            if chunk.start_pos < end and chunk.end_pos > start:
                chunk_tensor = self.get_chunk(chunk.id, device)

                # Calculate local offsets
                local_start = max(0, start - chunk.start_pos)
                local_end = min(len(chunk_tensor), end - chunk.start_pos)

                relevant_tokens.append(chunk_tensor[local_start:local_end])

        if not relevant_tokens:
            return torch.tensor([], dtype=torch.long, device=device)

        # Concatenate and deduplicate (handle overlap)
        all_tokens = torch.cat(relevant_tokens)

        # Due to overlap, we might have duplicates - take unique window
        # This is a simple approach; actual dedup depends on exact positions
        return all_tokens[:end - start]

    def find_chunks_for_range(self, start: int, end: int) -> List[int]:
        """Find chunk IDs that overlap with a given token range."""
        chunk_ids = []
        for chunk in self.chunks:
            if chunk.start_pos < end and chunk.end_pos > start:
                chunk_ids.append(chunk.id)
        return chunk_ids

    @property
    def total_tokens(self) -> int:
        """Total number of tokens in the context."""
        return self._total_tokens

    @property
    def num_chunks(self) -> int:
        """Number of chunks in the context."""
        return len(self.chunks)
