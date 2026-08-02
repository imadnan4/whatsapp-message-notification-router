"""Phase 2 tests: schema matching, provider interface, agent tool loop.

The agent loop is tested with a fake provider (no API calls) so the bounded
loop, tool execution, validation, and fallback paths are covered
deterministically. Media reads are avoided by choosing text-only messages.
"""

from __future__ import annotations

import json

import pytest

from core.agent import RoutingAgent, TOOLS
from core.data_loader import Message, load_dataset
from core.providers import ChatResult, ProviderError, ToolCall
from core.schema import (
    ACTION_VALUES,
    MESSAGE_TYPE_VALUES,
    RoutingOutput,
    build_routing,
    calibrate_confidence,
    extract_json_object,
    match_action,
    match_message_type,
    match_value,
    validate_output,
    validate_reason,
)

DS = load_dataset()


# ---------------------------------------------------------------------------
# Schema: allowed values + three-tier matching
# ---------------------------------------------------------------------------


def test_allowed_value_sets_match_contract():
    assert ACTION_VALUES == ("notify", "digest", "mute")
    assert "business_update" in MESSAGE_TYPE_VALUES
    assert "scam" in MESSAGE_TYPE_VALUES


def test_match_exact_and_normalized():
    assert match_value("notify", ACTION_VALUES) == "notify"
    assert match_value("Notify", ACTION_VALUES) == "notify"
    assert match_value("  digest ", ACTION_VALUES) == "digest"
    assert match_value("business_update", MESSAGE_TYPE_VALUES) == "business_update"
    assert match_value("Business Update", MESSAGE_TYPE_VALUES) == "business_update"
    assert match_value("business-update", MESSAGE_TYPE_VALUES) == "business_update"


def test_match_token_subset_and_failures():
    # tier 3: token-subset, unambiguous
    assert match_value("notify now", ACTION_VALUES) == "notify"
    assert match_value("business update please", MESSAGE_TYPE_VALUES) == "business_update"
    # 'business' alone resolves to business_update (only value containing it)
    assert match_value("business", MESSAGE_TYPE_VALUES) == "business_update"
    # no match -> None
    assert match_value("urgently", MESSAGE_TYPE_VALUES) is None
    assert match_value("payment requested", MESSAGE_TYPE_VALUES) is None
    assert match_value("", ACTION_VALUES) is None
    assert match_value(None, ACTION_VALUES) is None
    assert match_action("notify-now") == "notify"
    # message_type never fatal
    assert match_message_type("something weird") == "unknown"


def test_extract_json_object_tolerates_prose_and_fences():
    assert extract_json_object('{"a": 1}') == {"a": 1}
    assert extract_json_object('here is my answer: {"a": 1} done') == {"a": 1}
    assert extract_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json_object('{"a": 1, "b": [1, 2,]}') == {"a": 1, "b": [1, 2]}
    assert extract_json_object("no json here") is None
    assert extract_json_object('{"nested": {"x": "}"}}') == {"nested": {"x": "}"}}


def test_build_routing_normalizes_and_validates():
    hist_ids = {m.message_id for m in DS.history}
    out, problem = build_routing(
        "msg_001",
        {
            "action": "Notify",
            "message_type": "Business Update",
            "reason": "A verified business sent an order update matching user history.",
            "confidence": "0.87",
            "evidence_message_ids": "message_0004;made_up_id;message_0005",
        },
        known_history_ids=hist_ids,
    )
    assert problem is None
    assert out is not None
    assert out.action == "notify"
    assert out.message_type == "business_update"
    assert out.confidence == calibrate_confidence("notify", 0.87)
    assert out.evidence_message_ids == "message_0004;message_0005"  # invented id dropped

    # confidence clamped (then calibrated)
    out2, _ = build_routing("m", {"action": "mute", "message_type": "scam",
                                  "reason": "This is a fine specific reason.", "confidence": 3.0,
                                  "evidence_message_ids": "none"}, known_history_ids=hist_ids)
    assert out2 is not None and out2.confidence == calibrate_confidence("mute", 1.0)

    # invalid action is strict
    out3, problem3 = build_routing("m", {"action": "maybe", "message_type": "x",
                                         "reason": "This is a fine specific reason.",
                                         "confidence": 0.5, "evidence_message_ids": "none"},
                                   known_history_ids=hist_ids)
    assert out3 is None and "invalid action" in problem3


def test_reason_validation_rejects_generic():
    assert validate_reason("") is not None
    assert validate_reason("short") is not None
    assert validate_reason("important") is not None
    assert validate_reason("The user opted out of this business's promotions.") is None


def test_validate_output_strict_contract():
    bad = {"message_id": "m1", "action": "maybe", "message_type": "nope",
           "reason": "x", "confidence": "abc", "evidence_message_ids": ""}
    assert len(validate_output(bad)) >= 5
    good = {"message_id": "m1", "action": "digest", "message_type": "unknown",
            "reason": "A legitimate but non-urgent business update.",
            "confidence": 0.8, "evidence_message_ids": "none"}
    assert validate_output(good) == []


# ---------------------------------------------------------------------------
# Agent loop with a fake provider (no network)
# ---------------------------------------------------------------------------


class FakeProvider:
    """Scripted provider: sequence of responses, then a valid submission."""

    name = "fake"
    model = "fake-model"

    def __init__(self, script: list[dict], price_in: float = 0.0, price_out: float = 0.0):
        self.script = list(script)
        self.calls = 0
        self.price_in, self.price_out = price_in, price_out

    def chat(self, messages, max_completion_tokens=512, tools=None, tool_choice="auto"):
        self.calls += 1
        step = self.script[min(self.calls - 1, len(self.script) - 1)]
        tcs = tuple(
            ToolCall(id=f"call_{i}", name=t["name"], arguments=json.dumps(t["args"]))
            for i, t in enumerate(step.get("tool_calls", []))
        )
        return ChatResult(
            text=step.get("text", ""),
            tool_calls=tcs,
            prompt_tokens=10,
            completion_tokens=5,
            model=self.model,
            price_in=self.price_in,
            price_out=self.price_out,
        )


def _text_msg(message_id: str = "msg_001") -> Message:
    return next(m for m in DS.incoming if m.message_id == message_id)


def _submit_args(**overrides) -> dict:
    args = {
        "action": "digest",
        "message_type": "greeting",
        "reason": "A harmless greeting that can be read later.",
        "confidence": 0.82,
        "evidence_message_ids": "none",
    }
    args.update(overrides)
    return args


def test_agent_submits_directly():
    agent = RoutingAgent(DS, max_iterations=2)
    agent._provider = FakeProvider([
        {"tool_calls": [{"name": "submit_routing", "args": _submit_args()}]},
    ])
    out = agent.route(_text_msg())
    assert out.action == "digest" and out.message_type == "greeting"
    assert out.message_id == "msg_001"
    assert out.evidence_message_ids == "none"


def test_agent_tool_loop_inspect_then_submit():
    fake = FakeProvider([
        {"tool_calls": [{"name": "inspect_evidence", "args": {"message_id": "message_0001"}}]},
        {"tool_calls": [{"name": "submit_routing", "args": _submit_args(action="notify")}]},
    ])
    agent = RoutingAgent(DS, max_iterations=2)
    agent._provider = fake
    out = agent.route(_text_msg())
    assert out.action == "notify"
    # the tool result must have been fed back to the model
    assert fake.calls == 2


def test_agent_rejected_submission_then_fixed():
    bad = _submit_args(action="maybe")  # invalid action
    fake = FakeProvider([
        {"tool_calls": [{"name": "submit_routing", "args": bad}]},
        {"tool_calls": [{"name": "submit_routing", "args": _submit_args(action="mute")}]},
    ])
    agent = RoutingAgent(DS, max_iterations=3)
    agent._provider = fake
    out = agent.route(_text_msg())
    assert out.action == "mute"
    assert fake.calls == 2


def test_agent_loop_cap_backstop():
    # model keeps inspecting; loop must cap and produce a valid row anyway
    fake = FakeProvider([
        {"tool_calls": [{"name": "inspect_evidence", "args": {"message_id": "message_0001"}}]},
        {"tool_calls": [{"name": "lookup_user_context", "args": {"kind": "user"}}]},
        {"text": '{"action": "digest", "message_type": "unknown", "reason": "No decisive evidence, conservative choice.", "confidence": 0.5, "evidence_message_ids": "none"}'},
    ])
    agent = RoutingAgent(DS, max_iterations=2, max_tool_calls=2)
    agent._provider = fake
    out = agent.route(_text_msg())
    assert out.action == "digest"
    assert fake.calls == 3  # 2 loop rounds + 1 backstop


def test_agent_invalid_text_reply_corrected():
    fake = FakeProvider([
        {"text": "I think this is a greeting."},
        {"tool_calls": [{"name": "submit_routing", "args": _submit_args()}]},
    ])
    agent = RoutingAgent(DS, max_iterations=3)
    agent._provider = fake
    out = agent.route(_text_msg())
    assert out.action == "digest"
    assert fake.calls == 2


def test_agent_fallback_on_provider_error():
    class BoomProvider(FakeProvider):
        def chat(self, messages, max_completion_tokens=512, tools=None, tool_choice="auto"):
            raise RuntimeError("network down")

    agent = RoutingAgent(DS)
    agent._provider = BoomProvider([])
    out = agent.route(_text_msg())
    # hard failure degrades to a flagged conservative row, never raises
    assert out.action == "digest" and out.message_type == "unknown"
    assert "Fallback" in out.reason
    assert validate_output({
        "message_id": out.message_id, "action": out.action,
        "message_type": out.message_type, "reason": out.reason,
        "confidence": out.confidence,
        "evidence_message_ids": out.evidence_message_ids,
    }) == []


def test_agent_evidence_ids_validated_against_history():
    # model cites an invented id -> dropped to none
    fake = FakeProvider([
        {"tool_calls": [{"name": "submit_routing", "args": _submit_args(
            evidence_message_ids="message_9999")}]},
    ])
    agent = RoutingAgent(DS, max_iterations=2)
    agent._provider = fake
    out = agent.route(_text_msg())
    assert out.evidence_message_ids == "none"


def test_build_context_contains_delimiters_and_no_labels():
    from core.agent import build_context

    msg = next(s for s in DS.samples if s.message_id == "sample_msg_001")
    ctx = build_context(DS, msg, None, index=agent_index())
    rendered = ctx.render()
    assert "<message_to_classify>" in rendered
    assert "EVIDENCE CANDIDATES" in rendered
    assert "notify" not in ctx.blocks["MESSAGE"].lower()  # no golden label leak
    assert len(ctx.candidates) <= 6


def agent_index():
    from core.retrieval import build_index

    return build_index(DS)


def test_tool_schemas_wellformed():
    assert len(TOOLS) == 3
    names = {t["function"]["name"] for t in TOOLS}
    assert names == {"inspect_evidence", "lookup_user_context", "submit_routing"}
    for t in TOOLS:
        assert t["type"] == "function"
        assert t["function"]["parameters"]["type"] == "object"


# ---------------------------------------------------------------------------
# Responses-API translation layer (regression tests for the probe findings)
# ---------------------------------------------------------------------------

from core.providers.openai_provider import _to_responses_input, _to_responses_tools


def _transcript():
    return [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_abc",
                    "type": "function",
                    "function": {"name": "inspect_evidence", "arguments": '{"message_id": "message_0001"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_abc", "content": "opened by user"},
    ]


def test_responses_input_translation_shapes():
    items = _to_responses_input(_transcript())
    assert items[0] == {"role": "system", "content": [{"type": "input_text", "text": "sys"}]}
    assert items[1]["role"] == "user"
    # assistant message item: content REQUIRED (even when empty) + separate
    # top-level function_call item with fc_-prefixed id
    assert items[2]["role"] == "assistant"
    assert items[2]["content"] == [{"type": "output_text", "text": ""}]
    assert items[3] == {
        "type": "function_call",
        "id": "fc_call_abc",
        "call_id": "call_abc",
        "name": "inspect_evidence",
        "arguments": '{"message_id": "message_0001"}',
    }
    # tool result matched by call_id
    assert items[4] == {
        "type": "function_call_output",
        "call_id": "call_abc",
        "output": "opened by user",
    }


def test_responses_tools_translation_flat_schema():
    out = _to_responses_tools(TOOLS)
    assert len(out) == 3
    for t in out:
        assert t["type"] == "function"
        assert "name" in t and t["name"]
        assert "parameters" in t
        assert "function" not in t  # flat schema, not chat format


def test_provider_chat_requires_responses_for_tool_transcript():
    # backstop path: transcript with tool messages but no tools -> must NOT
    # go to chat.completions (unverified for tool-role messages)
    from core.providers.openai_provider import OpenAIProvider

    p = OpenAIProvider()
    # monkeypatch the two paths to observe the routing decision
    p._chat_responses = lambda *a, **k: "responses"  # type: ignore[method-assign]
    p._chat_completions = lambda *a, **k: "completions"  # type: ignore[method-assign]
    assert p.chat([{"role": "user", "content": "x"}]) == "completions"
    assert p.chat([{"role": "user", "content": "x"}], tools=TOOLS) == "responses"
    assert p.chat(_transcript()) == "responses"  # tool history, no tools arg


# ---------------------------------------------------------------------------
# Provider fallback + state restore (regression for shadowed-route bug)
# ---------------------------------------------------------------------------


def test_agent_provider_fallback_restores_state(monkeypatch):
    import core.agent as agent_mod

    calls: list[str] = []

    class FlakyOpenAI(FakeProvider):
        def chat(self, messages, max_completion_tokens=512, tools=None, tool_choice="auto"):
            calls.append("openai")
            raise ProviderError("openai down")

    class GoodDeepSeek(FakeProvider):
        def chat(self, messages, max_completion_tokens=512, tools=None, tool_choice="auto"):
            calls.append("deepseek")
            return super().chat(messages, max_completion_tokens, tools, tool_choice)

    fake_registry = {"openai": FlakyOpenAI([
        {"tool_calls": [{"name": "submit_routing", "args": _submit_args(action="notify")}]},
    ]), "deepseek": GoodDeepSeek([
        {"tool_calls": [{"name": "submit_routing", "args": _submit_args()}]},
    ])}
    monkeypatch.setattr(agent_mod, "get_provider", lambda name: fake_registry[name])

    agent = RoutingAgent(DS, max_iterations=2)
    out1 = agent.route(_text_msg())
    assert out1.action == "digest"  # deepseek answer
    assert calls == ["openai", "deepseek"]
    # state restored: provider_name back to openai, provider client cleared
    assert agent.provider_name == "openai"
    assert agent._provider is None
    assert agent._fallback_active is False

    # a second message goes back to the openai provider (no leak)
    out2 = agent.route(_text_msg("msg_002"))
    assert out2.action == "digest"
    assert calls == ["openai", "deepseek", "openai", "deepseek"]


def test_agent_double_provider_failure_falls_back_and_restores(monkeypatch):
    import core.agent as agent_mod

    class AlwaysDown(FakeProvider):
        def chat(self, messages, max_completion_tokens=512, tools=None, tool_choice="auto"):
            raise ProviderError("down")

    monkeypatch.setattr(
        agent_mod, "get_provider", lambda name: AlwaysDown([])
    )
    agent = RoutingAgent(DS, max_iterations=2)
    out = agent.route(_text_msg())
    assert out.action == "digest" and "Fallback" in out.reason
    assert agent.provider_name == "openai"
    assert agent._provider is None
    assert agent._fallback_active is False


def test_schema_extract_json_single_quotes_and_evidence_whitespace():
    # single-quoted JSON fallback
    assert extract_json_object("{'a': 1, 'b': 'x'}") == {"a": 1, "b": "x"}
    # whitespace-separated evidence ids are split, not dropped
    hist_ids = {m.message_id for m in DS.history}
    out, problem = build_routing(
        "m",
        {"action": "mute", "message_type": "scam",
         "reason": "This is a fine specific reason.",
         "confidence": 0.5,
         "evidence_message_ids": "message_0001 message_0002"},
        known_history_ids=hist_ids,
    )
    assert problem is None and out is not None
    assert out.evidence_message_ids == "message_0001;message_0002"
