"""OpenAI provider — gpt-5.6-luna (primary routing model).

API quirks verified by probing (2026-08-02, RESEARCH.md §4):
- `max_completion_tokens`, never `max_tokens` (o-series-style rejection)
- no `temperature` param (default only; 0 is rejected)
- FUNCTION TOOLS are NOT supported on /v1/chat/completions for this model
  unless `reasoning_effort='none'` (400 error; 'low'/'medium' also rejected).
  The supported path is /v1/responses — so tool-calling conversations go
  through the responses API (full reasoning), plain text stays on
  chat.completions. The translation lives HERE so the agent's messages list
  stays provider-agnostic chat format.
"""

from __future__ import annotations

from functools import lru_cache

from openai import OpenAI

from core.providers.base import (
    ChatResult,
    ProviderError,
    ToolCall,
    get_env,
)

MODEL = "gpt-5.6-luna"
# $ per 1M tokens, verified 2026-08-01 (RESEARCH.md §4)
PRICE_IN = 0.20
PRICE_OUT = 1.20


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    key = get_env("OPENAI_API_KEY")
    if not key:
        raise ProviderError("OPENAI_API_KEY not found in repo .env")
    return OpenAI(api_key=key)


class OpenAIProvider:
    name = "openai"
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
        # A transcript that contains tool activity (assistant tool_calls or
        # tool-role results) must continue on the responses path even when
        # this call passes no tools (e.g. the agent's backstop round) —
        # chat.completions is not verified to accept tool-role messages.
        needs_responses = tools or any(
            m.get("role") == "tool" or m.get("tool_calls") for m in messages
        )
        if needs_responses:
            return self._chat_responses(
                messages, max_completion_tokens, tools or [], tool_choice
            )
        return self._chat_completions(messages, max_completion_tokens)

    # -- /v1/chat/completions (plain text, vision) --------------------------

    def _chat_completions(
        self, messages: list[dict], max_completion_tokens: int
    ) -> ChatResult:
        try:
            resp = _client().chat.completions.create(
                model=self.model,
                messages=messages,
                max_completion_tokens=max_completion_tokens,
            )
        except Exception as exc:  # noqa: BLE001 — provider errors are all non-fatal here
            raise ProviderError(f"openai {type(exc).__name__}: {exc}") from exc
        msg = resp.choices[0].message
        usage = resp.usage
        return ChatResult(
            text=(msg.content or "").strip(),
            tool_calls=(),
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            model=self.model,
            price_in=self.price_in,
            price_out=self.price_out,
        )

    # -- /v1/responses (tool-calling conversations) --------------------------

    def _chat_responses(
        self,
        messages: list[dict],
        max_output_tokens: int,
        tools: list[dict],
        tool_choice: str,
    ) -> ChatResult:
        try:
            kwargs = dict(
                model=self.model,
                input=_to_responses_input(messages),
                max_output_tokens=max_output_tokens,
            )
            if tools:
                kwargs["tools"] = _to_responses_tools(tools)
                kwargs["tool_choice"] = tool_choice
            resp = _client().responses.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 — provider errors are all non-fatal here
            raise ProviderError(f"openai/responses {type(exc).__name__}: {exc}") from exc

        text = "".join(
            part.text
            for item in resp.output
            if item.type == "message"
            for part in item.content
            if part.type == "output_text"
        ).strip()
        tool_calls = tuple(
            ToolCall(id=getattr(it, "call_id", None) or it.id, name=it.name, arguments=it.arguments)
            for it in resp.output
            if it.type == "function_call"
        )
        usage = resp.usage
        return ChatResult(
            text=text,
            tool_calls=tool_calls,
            prompt_tokens=usage.input_tokens if usage else 0,
            completion_tokens=usage.output_tokens if usage else 0,
            model=self.model,
            price_in=self.price_in,
            price_out=self.price_out,
        )


def _to_responses_tools(tools: list[dict]) -> list[dict]:
    """chat-format tool schema ({type, function:{name,...}}) -> responses
    format ({type, name, description, parameters} — flat)."""
    out = []
    for t in tools:
        fn = t.get("function", {})
        out.append(
            {
                "type": "function",
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
            }
        )
    return out


def _to_responses_input(messages: list[dict]) -> list[dict]:
    """Translate agent-side chat-format messages to /v1/responses input items.

    Roles: system/user -> input_text items; assistant -> output_text + optional
    function_call items; tool -> function_call_output items (matched by
    call_id).
    """
    items: list[dict] = []
    for m in messages:
        role = m["role"]
        if role in ("system", "user"):
            items.append(
                {
                    "role": role,
                    "content": [{"type": "input_text", "text": m["content"]}],
                }
            )
        elif role == "assistant":
            # responses API requires `content` on assistant input items;
            # empty text is allowed when the turn only carried tool calls.
            items.append(
                {
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": m.get("content") or ""}
                    ],
                }
            )
            # Prior function calls are separate top-level input items, each
            # referenced by call_id from the following function_call_output.
            for c in m.get("tool_calls") or []:
                items.append(
                    {
                        "type": "function_call",
                        # `id` must follow the fc_ prefix convention; `call_id`
                        # is what function_call_output items reference.
                        "id": "fc_" + c["id"],
                        "call_id": c["id"],
                        "name": c["function"]["name"],
                        "arguments": c["function"]["arguments"],
                    }
                )
        elif role == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": m["tool_call_id"],
                    "output": m["content"],
                }
            )
        else:  # pragma: no cover — defensive; agent never sends other roles
            raise ProviderError(f"unsupported message role for responses API: {role!r}")
    return items


def get_openai_provider() -> OpenAIProvider:
    return OpenAIProvider()
