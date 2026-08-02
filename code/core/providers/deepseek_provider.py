"""DeepSeek provider — fallback only (primary = OpenAI gpt-5.6-luna).

OpenAI-compatible endpoint. Quirk isolated here: DeepSeek's API accepts
`max_tokens`, not `max_completion_tokens`, so the interface field is mapped
at this boundary. Key comes from DEEPSEEK_API_KEY in .env (with a local
dev-machine fallback); never logged.
"""

from __future__ import annotations

from functools import lru_cache

from openai import OpenAI

from core.providers.base import (
    ChatResult,
    ProviderError,
    ToolCall,
    deepseek_key,
)

MODEL = "deepseek-v4-flash"
BASE_URL = "https://api.deepseek.com/v1"
# Approximate DeepSeek pricing (fallback; only used for cost reporting).
PRICE_IN = 0.27
PRICE_OUT = 1.10


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    return OpenAI(api_key=deepseek_key(), base_url=BASE_URL)


class DeepSeekProvider:
    name = "deepseek"
    model = MODEL
    price_in = PRICE_IN
    price_out = PRICE_OUT

    def chat(
        self,
        messages: list[dict],
        max_completion_tokens: int = 512,
        tools: list[dict] | None = None,
        tool_choice: str = "auto",
    ) -> ChatResult:
        try:
            kwargs = dict(
                model=self.model,
                messages=messages,
                # DeepSeek's OpenAI-compatible API uses max_tokens.
                max_tokens=max_completion_tokens,
            )
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = tool_choice
            resp = _client().chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 — provider errors are all non-fatal here
            raise ProviderError(f"deepseek {type(exc).__name__}: {exc}") from exc
        msg = resp.choices[0].message
        tool_calls = tuple(
            ToolCall(id=tc.id, name=tc.function.name, arguments=tc.function.arguments)
            for tc in (msg.tool_calls or [])
        )
        usage = resp.usage
        return ChatResult(
            text=(msg.content or "").strip(),
            tool_calls=tool_calls,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            model=self.model,
            price_in=self.price_in,
            price_out=self.price_out,
        )


def get_deepseek_provider() -> DeepSeekProvider:
    return DeepSeekProvider()
