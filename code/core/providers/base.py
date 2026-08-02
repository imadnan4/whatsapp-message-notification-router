"""Phase 2 — provider-agnostic layer (winner pattern #4).

One small interface; each provider isolates its own API quirks:
- openai (gpt-5.6-luna): `max_completion_tokens` (NOT `max_tokens`), no
  `temperature` param (Phase 0 verified), tools via the OpenAI schema.
- deepseek (deepseek-v4-flash): OpenAI-compatible endpoint; maps
  `max_completion_tokens` -> `max_tokens` (the field DeepSeek's API accepts).

Usage tracker is shared so a full run reports real token/cost totals.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Protocol

from dotenv import load_dotenv

from core.data_loader import repo_root


class ProviderError(RuntimeError):
    """Raised when a provider fails (network, auth, schema, empty output...)."""


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON string from the model


@dataclass(frozen=True)
class ChatResult:
    text: str
    tool_calls: tuple[ToolCall, ...]
    prompt_tokens: int
    completion_tokens: int
    model: str
    price_in: float
    price_out: float

    @property
    def cost_usd(self) -> float:
        return (
            self.prompt_tokens / 1e6 * self.price_in
            + self.completion_tokens / 1e6 * self.price_out
        )


class Provider(Protocol):
    name: str
    model: str

    def chat(
        self,
        messages: list[dict],
        max_completion_tokens: int = 512,
        tools: list[dict] | None = None,
        tool_choice: str = "auto",
    ) -> ChatResult: ...


@dataclass
class UsageTracker:
    """Accumulates tokens/cost across provider calls (per run)."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    calls: int = 0
    fallback_calls: int = 0
    per_provider: dict[str, "UsageTracker"] = field(default_factory=dict)

    def add(self, result: ChatResult, fallback: bool = False) -> None:
        self.prompt_tokens += result.prompt_tokens
        self.completion_tokens += result.completion_tokens
        self.cost_usd += result.cost_usd
        self.calls += 1
        if fallback:
            self.fallback_calls += 1
        # Per-model sub-tracker accumulates directly (no recursion — a
        # sub-tracker for the same model must not re-enter `add`).
        sub = self.per_provider.setdefault(result.model, UsageTracker())
        sub.prompt_tokens += result.prompt_tokens
        sub.completion_tokens += result.completion_tokens
        sub.cost_usd += result.cost_usd
        sub.calls += 1

    def summary(self) -> str:
        lines = [
            f"calls={self.calls} (fallback={self.fallback_calls})",
            f"tokens={self.prompt_tokens} in / {self.completion_tokens} out",
            f"cost=${self.cost_usd:.4f}",
        ]
        for model, sub in sorted(self.per_provider.items()):
            lines.append(f"  {model}: {sub.calls} calls, ${sub.cost_usd:.4f}")
        return "\n".join(lines)


def _load_env() -> None:
    load_dotenv(repo_root() / ".env")


def get_env(name: str) -> str | None:
    _load_env()
    return os.getenv(name)


@lru_cache(maxsize=1)
def deepseek_key() -> str:
    """DeepSeek key: DEEPSEEK_API_KEY from .env first, then the local
    auth file (~/.pi/agent/auth.json, dev-machine convenience). The .env
    path is the documented one for public use."""
    env_key = get_env("DEEPSEEK_API_KEY")
    if env_key:
        return env_key
    path = Path.home() / ".pi" / "agent" / "auth.json"
    if not path.exists():
        raise ProviderError("DEEPSEEK_API_KEY not found in .env")
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    key = data.get("deepseek", {}).get("key", "")
    if not key:
        raise ProviderError("deepseek.key missing from auth.json")
    return key
