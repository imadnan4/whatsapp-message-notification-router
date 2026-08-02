"""Phase 2 — the routing agent (winner pattern #1: single agent, bounded loop).

Flow per message:
1. Assemble deterministic context: message + media read (cached) + user/sender/
   group/business features + top evidence candidates. NO model calls here.
2. Tool loop (max MAX_ITERATIONS rounds, MAX_TOOL_CALLS total calls):
   model may call inspect_evidence / lookup_user_context / submit_routing;
   code executes them and feeds results back. Hard caps enforced in code.
3. submit_routing (or a valid JSON text reply) ends the loop; output goes
   through schema validation (three-tier matching, reason checks, evidence
   filtering against real history ids).
4. Provider failure (openai) retries once on deepseek; a still-failing row
   degrades to a clearly-flagged conservative fallback (per-row isolation,
   winner pattern #6) — one bad row never kills a batch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from core.data_loader import Dataset, Message, build_features
from core.media import read_image, transcribe_voice
from core.prompts import SYSTEM_PROMPT, wrap_data_block, wrap_message_text
from core.providers import Provider, ProviderError, get_provider, usage
from core.retrieval import (
    EvidenceCandidate,
    build_index,
    choose_evidence_ids,
    retrieve_evidence,
)
from core.schema import RoutingOutput, build_routing, extract_json_object
MAX_ITERATIONS = 4
MAX_TOOL_CALLS = 6
MAX_OUT_TOKENS = 700
MAX_EVIDENCE_CANDIDATES = 6

# ---------------------------------------------------------------------------
# Tool schemas (OpenAI function-calling format; DeepSeek-compatible)
# ---------------------------------------------------------------------------

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "inspect_evidence",
            "description": (
                "Read one full history message (id from the EVIDENCE CANDIDATES "
                "list) including how the user treated it (opened/replied/dismissed/"
                "muted/reported) to verify a pattern before deciding."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "message_id": {
                        "type": "string",
                        "description": "A message_id from history (e.g. message_0001).",
                    }
                },
                "required": ["message_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_user_context",
            "description": (
                "Pull detail not in the message summary: business (verification, "
                "domain, opt-outs, activity), group (types, roles, mutes), user "
                "(quiet hours, 30d behavior), sender (history stats), or daily_load "
                "(recent notification volume)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": ["business", "group", "user", "sender", "daily_load"],
                    }
                },
                "required": ["kind"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_routing",
            "description": (
                "Your final routing decision as JSON with exactly: action "
                "(notify|digest|mute), message_type, reason (one specific "
                "sentence), confidence (0-1), evidence_message_ids ('id1;id2' "
                "or 'none')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["notify", "digest", "mute"]},
                    "message_type": {
                        "type": "string",
                        "enum": [
                            "personal", "urgent", "event", "payment",
                            "business_update", "promotion", "greeting",
                            "forward", "spam", "scam", "unknown",
                        ],
                    },
                    "reason": {"type": "string", "description": "One specific sentence (>= 12 chars)."},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "evidence_message_ids": {
                        "type": "string",
                        "description": "'id1;id2' from candidates/history, or 'none'.",
                    },
                },
                "required": [
                    "action", "message_type", "reason", "confidence",
                    "evidence_message_ids",
                ],
                "additionalProperties": False,
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Deterministic context bundle (no model calls)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MessageContext:
    """Everything the model sees for one message, rendered to text."""

    blocks: dict[str, str]
    candidates: list[EvidenceCandidate]

    def render(self) -> str:
        parts = []
        for title in (
            "MESSAGE", "RECEIVING USER", "CONVERSATION / SENDER CONTEXT",
            "EVIDENCE CANDIDATES",
        ):
            if title in self.blocks and self.blocks[title]:
                parts.append(f"=== {title} ===\n{self.blocks[title].rstrip()}")
        return "\n\n".join(parts)


def _fmt_evidence(cands: list[EvidenceCandidate]) -> str:
    """Render candidates in two groups: CANONICAL FIRST INSTANCES (earliest
    occurrence of each recurring pattern) first, then later copies as
    behavior context. Measured (Phase 5): the golden evidence is always a
    canonical instance (27/28 are the pattern lead, 28/28 in top-2), so
    making it the first thing the model sees converts near-duplicate-
    preference misses into hits.
    """
    if not cands:
        return "No history exists for this sender/conversation — evidence must be 'none'."
    lead = sorted(
        (c for c in cands if c.pattern_lead),
        # zero-padded ids sort lexicographically == numerically
        key=lambda c: (-c.similarity, c.message_id),
    )
    canon = sorted(
        (c for c in cands if c.is_canonical and not c.pattern_lead),
        key=lambda c: (-c.similarity, c.message_id),
    )
    others = [c for c in cands if not c.is_canonical]
    blocks: list[str] = []
    if lead or canon:
        blocks.append(
            "CANONICAL FIRST INSTANCES (earliest occurrence of each recurring pattern):"
        )
        blocks += [f"  {c.to_prompt_block()}" for c in lead]
        blocks += [f"  {c.to_prompt_block()}" for c in canon]
    if others:
        blocks.append(
            "OTHER INSTANCES (later copies of patterns — behavior context, not primary evidence):"
        )
        blocks += [f"  {c.to_prompt_block()}" for c in others]
    return wrap_data_block("\n".join(blocks), "evidence_candidates")


def build_context(
    ds: Dataset,
    msg: Message,
    media_text: str | None,
    index=None,
    sem=None,
) -> MessageContext:
    """Deterministic per-message context: features + evidence + media read."""
    f = build_features(ds, msg)
    # Voice notes have empty message_text: the transcript is the query text
    # for evidence retrieval, so the pattern-lead logic is not blind.
    cands = retrieve_evidence(
        ds, msg, index=index, k=MAX_EVIDENCE_CANDIDATES,
        query_text=media_text, sem=sem,
    )

    blocks: dict[str, str] = {}

    media_line = ""
    if msg.media_type == "image":
        media_line = f"\n[image read for {msg.media_id}]\n" + wrap_data_block(
            media_text or "(no read)", "image_read"
        )
    elif msg.media_type == "voice":
        media_line = f"\n[voice transcript for {msg.media_id}]\n" + wrap_data_block(
            media_text or "(no transcript)", "voice_transcript"
        )

    blocks["MESSAGE"] = (
        f"id: {msg.message_id}\n"
        f"conversation: {msg.conversation_type} | at: {msg.created_at:%Y-%m-%d %H:%M} "
        f"| forwarded_count: {msg.forwarded_count}\n"
        f"media_type: {msg.media_type or 'text'}{media_line}\n\n"
        f"{wrap_message_text(msg.message_text)}"
    )

    u = f.user
    dnd = u.dnd_raw or "none"
    daily = ""
    if f.daily_latest_sent is not None:
        daily = (
            f"\ndaily notifications: latest {f.daily_latest_sent} sent / "
            f"{f.daily_latest_dismissed or 0} dismissed; 30d avg {f.daily_avg_sent:.0f}"
        )
    blocks["RECEIVING USER"] = (
        f"user_id: {u.user_id} | quiet hours: {dnd} | in quiet hours now: {f.in_quiet_hours}"
        f"\n30d: opened {u.opened_30d}, replied {u.replied_30d}, "
        f"dismissed {u.dismissed_30d}, reported {u.reported_30d}{daily}"
    )

    s = f.sender_stats
    sender_line = (
        f"this sender -> user history: {s.n} msgs, opened {s.opened} ({s.opened_rate:.0%}), "
        f"replied {s.replied} ({s.replied_rate:.0%}), dismissed {s.dismissed}, "
        f"muted_after {s.muted_after}, reported {s.reported}, "
        f"forwarded {s.forwarded}, last contact {s.last_contact_days:.1f}d ago"
        if s.n
        else "this sender -> user history: none (first contact from this sender/conversation)"
    )

    if f.group is not None:
        g, sm, snd = f.group, f.self_membership, f.sender_membership
        blocks["CONVERSATION / SENDER CONTEXT"] = (
            f"group: {g.group_id} '{g.group_name}' | type={g.group_type}, "
            f"members={g.member_count}, admins={g.admin_count}, msgs_30d={g.messages_30d}\n"
            f"you in group: role={sm.role if sm else '?'}, group muted by you: "
            f"{sm.group_muted_by_user if sm else '?'}\n"
            f"sender {msg.sender_user_id}: role={snd.role if snd else '?'}, "
            f"sent_30d={snd.sent_30d if snd else '?'}, read_30d={snd.read_30d if snd else '?'}\n"
            f"{sender_line}"
        )
    elif f.business is not None:
        b, ubh = f.business, f.ubh
        ubh_line = "no user-business history"
        if ubh:
            ubh_line = (
                f"why known: {ubh.why_user_knows_account} | allows_promotions: "
                f"{ubh.allows_promotions} | opted_out_at: "
                f"{ubh.promotions_opted_out_at or 'never'} | activity_180d: "
                f"{ubh.activity_count_180d} | opened/dismissed/replied 30d: "
                f"{ubh.messages_opened_30d}/{ubh.messages_dismissed_30d}/"
                f"{ubh.messages_replied_30d}"
            )
        domain_match = "MATCH" if b.official_domain == b.domain_used_by_sender else "MISMATCH"
        blocks["CONVERSATION / SENDER CONTEXT"] = (
            f"business: {b.business_id} '{b.display_name}' ({b.brand_name}) | "
            f"category={b.category}, verified={b.verified}, account_age_days="
            f"{b.account_age_days}\nofficial_domain={b.official_domain} vs "
            f"sender domain={b.domain_used_by_sender} [{domain_match}], "
            f"sender-domain age={b.domain_used_by_sender_age_days}d, "
            f"msgs_sent_30d={b.messages_sent_30d}, user_reports_30d={b.user_reports_30d}\n"
            f"{ubh_line}"
        )
    else:
        p = f.personal_sender
        blocks["CONVERSATION / SENDER CONTEXT"] = (
            f"personal sender: {msg.sender_user_id}"
            f" | sender 30d: opened {p.opened_30d if p else '?'}, "
            f"replied {p.replied_30d if p else '?'}, dismissed {p.dismissed_30d if p else '?'}, "
            f"reported {p.reported_30d if p else '?'}\n"
            f"{sender_line}"
        )

    blocks["EVIDENCE CANDIDATES"] = _fmt_evidence(cands)
    return MessageContext(blocks=blocks, candidates=cands)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class RoutingAgent:
    """Single agent: bounded tool loop over one message; per-row isolation."""

    def __init__(
        self,
        ds: Dataset,
        provider_name: str = "openai",
        max_iterations: int = MAX_ITERATIONS,
        max_tool_calls: int = MAX_TOOL_CALLS,
        sem_index=None,
    ) -> None:
        self.ds = ds
        self.provider_name = provider_name
        self.max_iterations = max_iterations
        self.max_tool_calls = max_tool_calls
        self._provider: Provider | None = None
        self._history_ids = {m.message_id for m in ds.history}
        self._fallback_active = False  # True while retrying on deepseek
        # TF-IDF index is reusable across all messages (retrieval builds it
        # per call otherwise — wasteful for a 110-row batch).
        self._index = build_index(ds)
        # Semantic index (voice/empty-text fallback) — optional; load-only
        # so tests and keyless environments degrade to pure TF-IDF.
        from core.semantic import SemanticIndex

        self._sem = sem_index if sem_index is not None else SemanticIndex.load()

    # -- public ------------------------------------------------------------

    # -- internals ----------------------------------------------------------

    def _provider_client(self) -> Provider:
        if self._provider is None:
            self._provider = get_provider(self.provider_name)
        return self._provider

    def _chat(self, messages: list[dict], tools: list[dict] | None = None) -> object:
        result = self._provider_client().chat(
            messages=messages,
            max_completion_tokens=MAX_OUT_TOKENS,
            tools=tools,
        )
        usage.add(result, fallback=self._fallback_active)
        return result

    def _route_or_raise(self, msg: Message) -> RoutingOutput:
        media_text = self._media_text(msg)
        ctx = build_context(
            self.ds, msg, media_text, index=self._index, sem=self._sem
        )

        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ctx.render()},
        ]
        tool_calls_made = 0

        for iteration in range(1, self.max_iterations + 1):
            if iteration == self.max_iterations:
                # Near-cap nudge (winner #7): force a decision this round.
                messages.append(
                    {"role": "system", "content": (
                        "This is your FINAL round. Call submit_routing now with "
                        "your decision — do not call other tools."
                    )}
                )
            result = self._chat(messages, tools=TOOLS)

            if not result.tool_calls:
                out, problem = self._parse_submission(result.text)
                if out is not None:
                    return out
                messages.append({
                    "role": "user",
                    "content": f"Invalid routing reply: {problem}. "
                    "Call submit_routing with valid JSON.",
                })
                continue

            assistant_msg = {
                "role": "assistant",
                "content": result.text or None,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": tc.arguments},
                    }
                    for tc in result.tool_calls
                ],
            }
            messages.append(assistant_msg)

            for tc in result.tool_calls:
                tool_calls_made += 1
                if tc.name == "submit_routing":
                    # Valid submission ends the loop; a rejected one is fed
                    # back as the tool result so the model can fix it.
                    out, problem = self._parse_submission(tc.arguments)
                    if out is not None:
                        return out
                    output = f"submit_routing REJECTED: {problem}. Fix it and call submit_routing again."
                elif tool_calls_made > self.max_tool_calls:
                    output = "Tool budget exhausted: stop inspecting and decide now via submit_routing."
                else:
                    output = self._run_tool(tc.name, tc.arguments)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": output})

        # Loop cap hit -> one backstop call with tools removed (winner #7).
        messages.append({
            "role": "user",
            "content": "You have no tool rounds left. Reply with your final routing JSON only "
            "(same schema as submit_routing).",
        })
        result = self._chat(messages, tools=None)
        out, _ = self._parse_submission(result.text)
        if out is not None:
            return out
        return self._fallback(msg, "agent loop exhausted without a valid submission")

    def _media_text(self, msg: Message) -> str | None:
        if msg.media_type == "image" and msg.media_id:
            try:
                return read_image(self.ds, msg.media_id)
            except Exception as exc:  # noqa: BLE001 — per-row isolation
                return f"[image read unavailable: {type(exc).__name__}]"
        if msg.media_type == "voice" and msg.media_id:
            try:
                return transcribe_voice(self.ds, msg.media_id)
            except Exception as exc:  # noqa: BLE001 — per-row isolation
                return f"[voice transcript unavailable: {type(exc).__name__}]"
        return None

    def _parse_submission(self, raw: str) -> tuple[RoutingOutput | None, str | None]:
        obj = extract_json_object(raw)
        if obj is None:
            return None, "no JSON object found in reply"
        return build_routing(
            self._current_message_id,
            obj,
            known_history_ids=self._history_ids,
        )

    # -- tools --------------------------------------------------------------

    def _run_tool(self, name: str, arguments: str) -> str:
        try:
            args = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError:
            return f"tool error: arguments are not valid JSON: {arguments[:120]!r}"
        if not isinstance(args, dict):
            return "tool error: arguments must be a JSON object, not " \
                   f"{type(args).__name__}"
        if name == "inspect_evidence":
            return self._tool_inspect_evidence(args.get("message_id", ""))
        if name == "lookup_user_context":
            return self._tool_lookup_context(args.get("kind", ""))
        return f"unknown tool: {name}"

    def _tool_inspect_evidence(self, message_id: str) -> str:
        m = self.ds.history_by_id.get(message_id)
        if m is None:
            return (
                f"unknown message_id: {message_id!r}. Only ids from the EVIDENCE "
                "CANDIDATES list (or inspectable history) can be cited."
            )
        events = []
        if m.opened is not None:
            events.append(f"opened={m.opened}")
        if m.replied is not None:
            events.append(f"replied={m.replied}")
        if m.dismissed is not None:
            events.append(f"dismissed={m.dismissed}")
        if m.muted_after is not None:
            events.append(f"muted_after={m.muted_after}")
        if m.reported is not None:
            events.append(f"reported={m.reported}")
        if m.reaction_minutes is not None:
            events.append(f"reaction_min={m.reaction_minutes:.0f}")
        return (
            f"[{m.message_id}] {m.created_at:%Y-%m-%d %H:%M} "
            f"conv={m.conversation_type} sender={m.sender_user_id or m.business_id} "
            f"fwd={m.forwarded_count} | {' '.join(events)}\n"
            + wrap_data_block(m.message_text[:300], "history_message")
        )

    def _tool_lookup_context(self, kind: str) -> str:
        f = build_features(self.ds, self._current_message)
        if kind == "business" and f.business is not None:
            b, ubh = f.business, f.ubh
            parts = [
                f"business {b.business_id}: '{b.display_name}' / '{b.brand_name}', "
                f"category={b.category}, verified={b.verified}",
                f"official_domain={b.official_domain}, domain_used_by_sender={b.domain_used_by_sender}",
                f"account_age_days={b.account_age_days}, messages_sent_30d={b.messages_sent_30d}, "
                f"user_reports_30d={b.user_reports_30d}, sender_domain_age_days={b.domain_used_by_sender_age_days}",
            ]
            if ubh:
                parts.append(
                    f"user-business: why_user_knows={ubh.why_user_knows_account}, "
                    f"allows_promotions={ubh.allows_promotions}, "
                    f"promotions_opted_out_at={ubh.promotions_opted_out_at}, "
                    f"last_activity_at={ubh.last_activity_at}, activity_180d={ubh.activity_count_180d}, "
                    f"opened/dismissed/replied_30d={ubh.messages_opened_30d}/"
                    f"{ubh.messages_dismissed_30d}/{ubh.messages_replied_30d}, "
                    f"last_reply_at={ubh.last_reply_at}"
                )
            else:
                parts.append("user-business history: NONE")
            return "\n".join(parts)
        if kind == "group" and f.group is not None:
            g, sm, snd = f.group, f.self_membership, f.sender_membership
            return (
                f"group {g.group_id}: '{g.group_name}', type={g.group_type}, "
                f"members={g.member_count}, admins={g.admin_count}, "
                f"created={g.created_at:%Y-%m-%d}, msgs_30d={g.messages_30d}\n"
                f"you: role={sm.role if sm else '?'}, joined={sm.joined_at if sm else '?'}, "
                f"muted_by_you={sm.group_muted_by_user if sm else '?'}\n"
                f"sender: role={snd.role if snd else '?'}, sent_30d={snd.sent_30d if snd else '?'}, "
                f"read_30d={snd.read_30d if snd else '?'}, replied_30d={snd.replied_30d if snd else '?'}"
            )
        if kind == "user":
            u = f.user
            return (
                f"user {u.user_id}: quiet_hours={u.dnd_raw or 'none'}, "
                f"30d opened={u.opened_30d}, replied={u.replied_30d}, "
                f"dismissed={u.dismissed_30d}, reported={u.reported_30d}"
            )
        if kind == "sender":
            s = f.sender_stats
            return (
                f"sender {self._current_message.sender_user_id or self._current_message.business_id}: "
                f"{s.n} history msgs, opened={s.opened} ({s.opened_rate:.0%}), "
                f"replied={s.replied} ({s.replied_rate:.0%}), dismissed={s.dismissed}, "
                f"muted_after={s.muted_after}, reported={s.reported}, forwarded={s.forwarded}, "
                f"last_contact_days={s.last_contact_days}"
            )
        if kind == "daily_load":
            daily = self.ds.daily.get(self._current_message.user_id, [])
            recent = daily[-7:]
            rows = "\n".join(
                f"  {d.date:%Y-%m-%d}: sent={d.notifications_sent}, dismissed={d.notifications_dismissed}"
                for d in recent
            )
            return f"daily notifications for {self._current_message.user_id} (last 7 days):\n{rows}"
        return f"unknown context kind: {kind!r} (use business|group|user|sender|daily_load)"

    # -- fallback ------------------------------------------------------------

    def _fallback(self, msg: Message, why: str) -> RoutingOutput:
        """Conservative, clearly-flagged row so no message_id is ever dropped."""
        cands = retrieve_evidence(self.ds, msg, index=self._index, k=2)
        return RoutingOutput(
            message_id=msg.message_id,
            action="digest",
            message_type="unknown",
            reason=f"Fallback ({why}): conservative default, no model decision",
            confidence=0.5,
            evidence_message_ids=choose_evidence_ids(cands),
        )

    # -- per-message state (used by tools + submission parsing) ---------------

    @property
    def _current_message(self) -> Message:
        return self._msg

    @property
    def _current_message_id(self) -> str:
        return self._msg.message_id

    def route(self, msg: Message) -> RoutingOutput:
        """Route one message. Never raises: hard failures fall back.

        A provider failure retries the whole message once on the fallback
        provider (deepseek); any other hard failure degrades to a conservative,
        clearly-flagged fallback row so no message_id is ever dropped.
        State is restored in `finally` so a fallback on one row can never
        leak into the next row of a batch.
        """
        self._msg = msg
        original_provider = self.provider_name
        try:
            return self._route_or_raise(msg)
        except ProviderError:
            # Primary provider dead -> try the fallback provider once.
            self._fallback_active = True
            try:
                self._provider = None
                self.provider_name = "deepseek"
                return self._route_or_raise(msg)
            except ProviderError:
                return self._fallback(msg, "provider failure (openai+deepseek)")
            finally:
                self._fallback_active = False
        except Exception as exc:  # noqa: BLE001 — per-row isolation (winner #6)
            return self._fallback(msg, f"internal error: {type(exc).__name__}: {exc}")
        finally:
            self._provider = None
            self.provider_name = original_provider
