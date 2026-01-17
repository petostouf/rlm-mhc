"""
RLM Scaffold Module.

Provides the REPL interface for programmatic context exploration,
implementing the core RLM paradigm where context is an external variable
accessible via specialized functions.
"""

from rlm_mhc.scaffold.repl import RLMSession
from rlm_mhc.scaffold.context import ContextManager
from rlm_mhc.scaffold.chunking import ChunkStrategy, Chunk
from rlm_mhc.scaffold.functions import peek, search, llm_query, llm_batch

__all__ = [
    "RLMSession",
    "ContextManager",
    "ChunkStrategy",
    "Chunk",
    "peek",
    "search",
    "llm_query",
    "llm_batch",
]
