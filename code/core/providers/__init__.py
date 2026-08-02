"""Provider registry — pick providers by name; shared usage tracker."""

from __future__ import annotations

from functools import lru_cache

from core.providers.base import (
    ChatResult,
    Provider,
    ProviderError,
    ToolCall,
    UsageTracker,
)
from core.providers.openai_provider import (
    MODEL,
    OpenAIProvider,
    get_openai_provider,
)
from core.providers.deepseek_provider import (
    DeepSeekProvider,
    get_deepseek_provider,
)

__all__ = [
    "ChatResult",
    "DeepSeekProvider",
    "MODEL",
    "OpenAIProvider",
    "Provider",
    "ProviderError",
    "ToolCall",
    "UsageTracker",
    "get_provider",
    "usage",
]


@lru_cache(maxsize=1)
def _registry() -> dict[str, Provider]:
    return {
        "openai": get_openai_provider(),
        "deepseek": get_deepseek_provider(),
    }


def get_provider(name: str = "openai") -> Provider:
    try:
        return _registry()[name]
    except KeyError:
        raise ProviderError(f"unknown provider: {name!r} (have: {sorted(_registry())})") from None


# Shared run-wide usage tracker (also imported by the legacy llm shim).
usage = UsageTracker()
