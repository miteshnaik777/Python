"""
05_generation/prompt_builder.py
─────────────────────────────────
Builds RAG prompts for chat-style LLMs.

Llama-style Chat uses a specific instruction format (also supported by many APIs):
    <s>[INST] <<SYS>>
    {system_message}
    <</SYS>>

    {user_message} [/INST]

The user message is the question + retrieved context (documents).
The system message instructs the model to answer only from context and
to say "I don't know" if the answer is not present — reducing hallucination.

Usage:
    from generation.prompt_builder import build_rag_prompt

    prompt = build_rag_prompt(
        query="What is the refund policy?",
        context_chunks=retrieved_chunks,
    )
"""

import sys
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import cfg

# ── System prompt (exported for external chat APIs) ─────────────────────────────
RAGRAG_SYSTEM_PROMPT = """You are a helpful, accurate, and concise document assistant.
Your job is to answer the user's question ONLY based on the provided document context below.
Rules:
- If the answer is clearly present in the context, answer it directly and cite the source.
- If the answer is partially present, provide what you can and note the limitation.
- If the answer is NOT in the context at all, say: "I could not find that information in the uploaded documents."
- Do NOT make up facts or use knowledge outside the provided context.
- Keep your answer focused and under 300 words unless more detail is needed."""


def build_rag_prompt(
    query: str,
    context_chunks: List[Dict],
    conversation_history: List[Dict] = None,
) -> str:
    """
    Build a Llama 2 Chat-formatted RAG prompt.

    Args:
        query              : The user's current question.
        context_chunks     : Retrieved chunk dicts (must have "text", "source", "page").
        conversation_history: Optional list of {"role": "user"/"assistant", "content": str}
                              for multi-turn chat context.

    Returns:
        A formatted prompt string (Llama-style single string; prefer build_rag_prompt_messages for APIs).
    """
    context_str = _format_context(context_chunks)

    user_message = (
        f"Here are the relevant excerpts from the uploaded documents:\n\n"
        f"{context_str}\n\n"
        f"---\n\n"
        f"Question: {query}"
    )

    # Single-turn format (standard for most deployments)
    if not conversation_history:
        prompt = (
            f"<s>[INST] <<SYS>>\n{RAG_SYSTEM_PROMPT}\n<</SYS>>\n\n"
            f"{user_message} [/INST]"
        )
        return prompt

    # Multi-turn: prepend prior turns
    turns = []
    for turn in conversation_history:
        role = turn.get("role", "")
        content = turn.get("content", "")
        if role == "user":
            turns.append(f"[INST] {content} [/INST]")
        elif role == "assistant":
            turns.append(f"{content} </s>")

    history_str = "\n".join(turns)

    prompt = (
        f"<s>[INST] <<SYS>>\n{RAG_SYSTEM_PROMPT}\n<</SYS>>\n\n"
        f"{history_str}\n"
        f"[INST] {user_message} [/INST]"
    )
    return prompt


def build_rag_prompt_messages(
    query: str,
    context_chunks: List[Dict],
    conversation_history: List[Dict] = None,
) -> List[Dict[str, str]]:
    """
    Build RAG as chat messages for OpenAI-compatible APIs (Groq, OpenAI, Ollama, etc.).

    Returns:
        [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
        or with history: [system, user_turn_1, assistant_turn_1, ..., user_final]
    """
    context_str = _format_context(context_chunks)
    user_content = (
        f"Here are the relevant excerpts from the uploaded documents:\n\n"
        f"{context_str}\n\n---\n\nQuestion: {query}"
    )
    if not conversation_history:
        return [
            {"role": "system", "content": RAGRAG_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
    messages = [{"role": "system", "content": RAGRAG_SYSTEM_PROMPT}]
    for turn in conversation_history:
        role = turn.get("role", "")
        content = turn.get("content", "")
        if role in ("user", "assistant"):
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_content})
    return messages


def _format_context(chunks: List[Dict]) -> str:
    """Format chunk dicts into a numbered, source-annotated context block."""
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.get("source", "unknown")
        page = chunk.get("page", "?")
        text = chunk.get("text", "")
        lines.append(f"[Excerpt {i} — {source}, page {page}]\n{text}")
    return "\n\n".join(lines)


def build_simple_prompt(query: str, context: str) -> str:
    """
    Lightweight helper: build a prompt from a pre-formatted context string.

    Useful for testing or when context is already a plain string.
    """
    user_message = (
        f"Document context:\n{context}\n\n"
        f"Question: {query}"
    )
    return (
        f"<s>[INST] <<SYS>>\n{RAG_SYSTEM_PROMPT}\n<</SYS>>\n\n"
        f"{user_message} [/INST]"
    )
