"""
Context Manager for RLM REPL.

Manages the loading and access of external contexts that the model
can programmatically explore.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union, IO
import os

import torch
from torch import Tensor

from rlm_mhc.types import ContextHandle, ChunkConfig
from rlm_mhc.scaffold.chunking import ChunkStrategy


class ContextManager:
    """
    Main interface for context management in RLM.

    Handles loading contexts from various sources (files, strings, streams)
    and provides access to tokens via the chunking strategy.

    Args:
        tokenizer: Tokenizer for encoding/decoding
        config: Chunking configuration
    """

    def __init__(self, tokenizer: Any, config: Optional[ChunkConfig] = None):
        self.tokenizer = tokenizer
        self.config = config or ChunkConfig()
        self.chunk_strategy = ChunkStrategy(self.config, tokenizer)
        self.current_context: Optional[ContextHandle] = None
        self._raw_text: Optional[str] = None

    def load(self, source: Union[str, Path, IO]) -> ContextHandle:
        """
        Load a context from various sources.

        Args:
            source: Can be:
                - A string (treated as path if exists, else as raw text)
                - A Path object
                - A file-like object with read() method

        Returns:
            ContextHandle for accessing the loaded context
        """
        # Determine source type and read text
        if isinstance(source, Path):
            text = source.read_text(encoding='utf-8')
        elif isinstance(source, str):
            if os.path.exists(source):
                text = Path(source).read_text(encoding='utf-8')
            else:
                text = source
        elif hasattr(source, 'read'):
            text = source.read()
            if isinstance(text, bytes):
                text = text.decode('utf-8')
        else:
            raise TypeError(f"Unsupported source type: {type(source)}")

        # Store raw text for reference
        self._raw_text = text

        # Load into chunk strategy
        self.current_context = self.chunk_strategy.load_context(text)

        return self.current_context

    def get_tokens(
        self,
        start: int,
        end: int,
        device: Optional[torch.device] = None,
    ) -> Tensor:
        """
        Get a range of tokens from the context.

        Args:
            start: Start position (in tokens)
            end: End position (in tokens)
            device: Target device

        Returns:
            Tensor of token IDs
        """
        if self.current_context is None:
            raise RuntimeError("No context loaded. Call load() first.")

        return self.chunk_strategy.get_window(start, end, device)

    def get_text(self, start: int, end: int) -> str:
        """
        Get decoded text for a token range.

        Args:
            start: Start position (in tokens)
            end: End position (in tokens)

        Returns:
            Decoded text string
        """
        tokens = self.get_tokens(start, end)
        return self.tokenizer.decode(tokens.tolist())

    @property
    def total_tokens(self) -> int:
        """Total number of tokens in current context."""
        if self.current_context is None:
            return 0
        return self.current_context.total_tokens

    @property
    def metadata(self) -> Dict[str, Any]:
        """Get metadata about the current context."""
        if self.current_context is None:
            return {}

        return {
            'total_tokens': self.current_context.total_tokens,
            'num_chunks': self.current_context.num_chunks,
            'chunk_size': self.current_context.chunk_size,
            'overlap': self.config.overlap,
            'text_length': len(self._raw_text) if self._raw_text else 0,
        }

    def clear(self):
        """Clear the current context and cache."""
        self.current_context = None
        self._raw_text = None
        self.chunk_strategy.cache.clear()
        self.chunk_strategy.chunks = []
