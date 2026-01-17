"""
RLM Session - Main REPL Interface.

Provides the unified API for RLM interaction, combining model,
context management, and exploration functions.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Union
import time

import torch

from rlm_mhc.types import ContextHandle, SessionConfig, SearchResult, ChunkConfig
from rlm_mhc.scaffold.context import ContextManager
from rlm_mhc.scaffold import functions


@dataclass
class RecursionCall:
    """Record of a recursive LLM call."""

    id: int
    depth: int
    timestamp: float
    completed: bool = False


class RecursionTracker:
    """
    Tracks recursion depth for llm_query calls.

    Prevents infinite recursion and provides debugging information
    about the call history.

    Args:
        max_depth: Maximum allowed recursion depth
    """

    def __init__(self, max_depth: int = 5):
        self.max_depth = max_depth
        self._depth = 0
        self._history: List[RecursionCall] = []

    @contextmanager
    def track(self) -> Generator[int, None, None]:
        """
        Context manager for tracking a recursive call.

        Yields:
            Current recursion depth
        """
        self._depth += 1
        call_id = len(self._history)

        self._history.append(RecursionCall(
            id=call_id,
            depth=self._depth,
            timestamp=time.time(),
        ))

        try:
            yield self._depth
        finally:
            self._depth -= 1
            self._history[call_id].completed = True

    @property
    def current_depth(self) -> int:
        """Current recursion depth."""
        return self._depth

    @property
    def history(self) -> List[RecursionCall]:
        """Copy of the call history."""
        return self._history.copy()

    def reset(self):
        """Reset the tracker."""
        self._depth = 0
        self._history.clear()


class RLMSession:
    """
    Main RLM Session for interactive context exploration.

    Provides the unified interface for:
    - Loading and managing contexts
    - Exploring contexts with peek() and search()
    - Querying the model with llm_query()

    Example:
        >>> session = RLMSession.from_pretrained("checkpoints/rlm-mhc-1b")
        >>> ctx = session.load("document.txt")
        >>> print(session.peek(ctx, 0, 100))
        >>> answer = session.llm_query(ctx, "What is the main topic?")

    Args:
        model: RLM model instance
        tokenizer: Tokenizer instance
        config: Session configuration
        device: Target device (default: cuda if available)
    """

    def __init__(
        self,
        model: Any,  # RLMModel
        tokenizer: Any,
        config: Optional[SessionConfig] = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or SessionConfig()

        # Device
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        # Context management
        self.context_manager = ContextManager(
            tokenizer=tokenizer,
            config=self.config.chunking,
        )

        # Recursion tracking
        self.recursion_tracker = RecursionTracker(
            max_depth=self.config.max_recursion
        )

    @classmethod
    def from_pretrained(
        cls,
        path: Union[str, Path],
        device: Optional[torch.device] = None,
        **kwargs,
    ) -> "RLMSession":
        """
        Load a session from a pretrained checkpoint.

        Args:
            path: Path to the checkpoint directory
            device: Target device
            **kwargs: Additional arguments for SessionConfig

        Returns:
            Initialized RLMSession
        """
        from rlm_mhc.model import RLMModel
        from transformers import AutoTokenizer

        path = Path(path)

        # Load model
        model = RLMModel.from_pretrained(path)

        # Load tokenizer
        tokenizer_path = path / "tokenizer"
        if tokenizer_path.exists():
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        else:
            # Try loading from model path
            tokenizer = AutoTokenizer.from_pretrained(path)

        # Create config
        config = SessionConfig(**kwargs)

        return cls(
            model=model,
            tokenizer=tokenizer,
            config=config,
            device=device,
        )

    def load(self, source: Union[str, Path]) -> ContextHandle:
        """
        Load a context from a file or string.

        Args:
            source: File path or text string

        Returns:
            ContextHandle for accessing the context
        """
        return self.context_manager.load(source)

    def peek(
        self,
        context: ContextHandle,
        start: int,
        end: int,
        decode: bool = True,
    ) -> Union[str, List[int]]:
        """
        Examine a portion of the context.

        Args:
            context: Context handle
            start: Start position (tokens)
            end: End position (tokens)
            decode: Return text if True, token IDs if False

        Returns:
            Text string or list of token IDs
        """
        return functions.peek(self, context, start, end, decode)

    def search(
        self,
        context: ContextHandle,
        query: str,
        max_results: int = 10,
        similarity_threshold: float = 0.0,
    ) -> List[SearchResult]:
        """
        Search for relevant sections in the context.

        Args:
            context: Context handle
            query: Search query
            max_results: Maximum results to return
            similarity_threshold: Minimum similarity score

        Returns:
            List of SearchResult objects
        """
        return functions.search(
            self, context, query, max_results, similarity_threshold
        )

    def llm_query(
        self,
        context: ContextHandle,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        context_window: Optional[int] = None,
    ) -> str:
        """
        Query the model with context.

        This is the core RLM function - allows recursive self-calls
        with access to the loaded context.

        Args:
            context: Context handle
            prompt: Query/instruction
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            context_window: Context tokens to include

        Returns:
            Generated response text
        """
        return functions.llm_query(
            self, context, prompt, max_tokens, temperature, context_window
        )

    def llm_batch(
        self,
        context: ContextHandle,
        prompts: List[str],
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> List[str]:
        """
        Batch multiple queries.

        Args:
            context: Context handle
            prompts: List of prompts
            max_tokens: Maximum tokens per response
            temperature: Sampling temperature

        Returns:
            List of responses
        """
        return functions.llm_batch(
            self, context, prompts, max_tokens, temperature
        )

    @property
    def context_metadata(self) -> Dict[str, Any]:
        """Get metadata about the current context."""
        return self.context_manager.metadata

    def reset(self):
        """Reset the session state."""
        self.context_manager.clear()
        self.recursion_tracker.reset()
