"""
REPL Functions for Context Exploration.

Core functions for programmatically exploring and querying contexts:
- peek(): Examine a portion of the context
- search(): Find relevant sections in the context
- llm_query(): Recursive call to the model with context
- llm_batch(): Batch multiple queries
"""

from typing import List, Optional, Union, TYPE_CHECKING
import torch
import torch.nn.functional as F
from torch import Tensor

from rlm_mhc.types import ContextHandle, SearchResult

if TYPE_CHECKING:
    from rlm_mhc.scaffold.repl import RLMSession


def peek(
    session: "RLMSession",
    context: ContextHandle,
    start: int,
    end: int,
    decode: bool = True,
) -> Union[str, List[int]]:
    """
    Examine a portion of the context.

    This is the primary function for inspecting context content.
    The model can use this to "look at" specific parts of the input.

    Args:
        session: Active RLM session
        context: Context handle
        start: Start position (in tokens)
        end: End position (in tokens)
        decode: If True, return decoded text; else return token IDs

    Returns:
        String (if decode=True) or list of token IDs

    Example:
        >>> text = session.peek(ctx, 0, 100)
        >>> print(text[:50])
        "The quick brown fox jumps over the lazy dog..."
    """
    tokens = session.context_manager.get_tokens(start, end)

    if decode:
        return session.tokenizer.decode(tokens.tolist())
    return tokens.tolist()


def search(
    session: "RLMSession",
    context: ContextHandle,
    query: str,
    max_results: int = 10,
    similarity_threshold: float = 0.0,
) -> List[SearchResult]:
    """
    Search for relevant sections in the context.

    Uses embedding similarity to find chunks most relevant to the query.
    This enables the model to locate information in long contexts.

    Args:
        session: Active RLM session
        context: Context handle
        query: Search query
        max_results: Maximum number of results to return
        similarity_threshold: Minimum similarity score (0-1)

    Returns:
        List of SearchResult objects with positions and scores

    Example:
        >>> results = session.search(ctx, "machine learning")
        >>> for r in results:
        ...     print(f"Score: {r.score:.3f}, Position: {r.start_pos}")
    """
    results = []

    # Encode query
    query_tokens = session.tokenizer.encode(query)
    query_tensor = torch.tensor([query_tokens], device=session.device)

    # Get query embedding from model
    with torch.inference_mode():
        query_output = session.model(query_tensor, output_hidden_states=True)
        # Use last hidden state, mean pooled
        query_emb = query_output.hidden_states[-1].mean(dim=1)  # [1, hidden_dim]

    # Score each chunk
    for chunk in session.context_manager.chunk_strategy.chunks:
        chunk_tensor = chunk.to_tensor(session.device).unsqueeze(0)

        with torch.inference_mode():
            chunk_output = session.model(chunk_tensor, output_hidden_states=True)
            chunk_emb = chunk_output.hidden_states[-1].mean(dim=1)

        # Cosine similarity
        score = F.cosine_similarity(query_emb, chunk_emb, dim=-1).item()

        if score >= similarity_threshold:
            # Get snippet
            snippet_tokens = chunk.tokens[:100]
            snippet = session.tokenizer.decode(snippet_tokens)

            results.append(SearchResult(
                chunk_id=chunk.id,
                start_pos=chunk.start_pos,
                end_pos=chunk.end_pos,
                score=score,
                snippet=snippet,
            ))

    # Sort by score descending
    results.sort(key=lambda x: x.score, reverse=True)

    return results[:max_results]


def llm_query(
    session: "RLMSession",
    context: ContextHandle,
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.7,
    context_window: Optional[int] = None,
) -> str:
    """
    Perform a recursive query to the model with context.

    This is the core RLM function - it allows the model to call itself
    with access to the loaded context. The recursion is tracked to
    prevent infinite loops.

    Args:
        session: Active RLM session
        context: Context handle
        prompt: Query/instruction for the model
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0 = greedy)
        context_window: Number of context tokens to include (None = config default)

    Returns:
        Generated response text

    Example:
        >>> answer = session.llm_query(ctx, "What is the main topic?")
        >>> print(answer)
    """
    with session.recursion_tracker.track() as depth:
        if depth > session.config.max_recursion:
            return "[MAX RECURSION DEPTH REACHED]"

        # Determine context window size
        window_size = context_window or session.config.context_window_size
        window_size = min(window_size, context.total_tokens)

        # Get context window
        context_text = peek(session, context, 0, window_size, decode=True)

        # Build prompt with context
        full_prompt = _build_prompt(context_text, prompt)

        # Tokenize
        input_ids = session.tokenizer.encode(full_prompt)
        input_tensor = torch.tensor([input_ids], device=session.device)

        # Generate
        output_ids = session.model.generate(
            input_tensor,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=session.tokenizer.pad_token_id,
            eos_token_id=session.tokenizer.eos_token_id,
        )

        # Decode response (skip input)
        response_ids = output_ids[0, len(input_ids):].tolist()
        response = session.tokenizer.decode(response_ids)

        return response.strip()


def llm_batch(
    session: "RLMSession",
    context: ContextHandle,
    prompts: List[str],
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> List[str]:
    """
    Batch multiple queries for efficiency.

    Processes multiple prompts in a single batch when possible.

    Args:
        session: Active RLM session
        context: Context handle
        prompts: List of query prompts
        max_tokens: Maximum tokens per response
        temperature: Sampling temperature

    Returns:
        List of responses (same order as prompts)
    """
    # For now, process sequentially
    # TODO: Implement true batching with padding
    responses = []
    for prompt in prompts:
        response = llm_query(
            session, context, prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        responses.append(response)

    return responses


def _build_prompt(context: str, user_prompt: str) -> str:
    """
    Build the full prompt with context for the model.

    Uses a simple template format that clearly delineates context,
    query, and expected response.

    Args:
        context: Context text to include
        user_prompt: User's question/instruction

    Returns:
        Formatted prompt string
    """
    return f"""<|context|>
{context}
<|/context|>

<|query|>
{user_prompt}
<|/query|>

<|response|>
"""
