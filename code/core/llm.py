"""Legacy shim — Phase 1 callers (media.py) keep working unchanged.

Real provider logic moved to core/providers/ (Phase 2). This module re-exports
the same names with the same signatures: chat_text(messages, model,
max_completion_tokens) -> ChatResult. `model` is accepted for back-compat but
providers are fixed per class; the openai provider is the default.
"""

from __future__ import annotations

import warnings

from core.providers import (
    MODEL,
    ChatResult,
    ProviderError,
    get_provider,
    usage,
)

__all__ = ["ChatResult", "MODEL", "ProviderError", "chat_text", "usage"]


def chat_text(
    messages: list[dict],
    model: str = MODEL,
    max_completion_tokens: int = 512,
) -> ChatResult:
    """One chat completion on the primary (openai) provider.

    `model` is accepted for Phase-1 back-compat; when it differs from the
    provider's model we still route through the openai provider (the only
    one Phase 1 used) — media reads and routing share gpt-5.6-luna.
    """
    if model != MODEL:
        warnings.warn(
            f"chat_text: model {model!r} ignored; provider is fixed to {MODEL}",
            stacklevel=2,
        )
    provider = get_provider("openai")
    result = provider.chat(
        messages=messages, max_completion_tokens=max_completion_tokens
    )
    usage.add(result)
    return result
