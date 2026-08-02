"""Phase 2 — schema single-source-of-truth (winner pattern #3).

Allowed values for `action` and `message_type` live ONLY here. The prompt
(prompts.py), the agent's submit tool (agent.py), and the output validator all
derive from these constants, so they can never drift apart.

Matching is three-tier (exact -> normalized -> token-subset) so richer model
outputs ("Business Update", "notify now", "business-updates") collapse onto the
canonical values instead of being rejected or falling to `unknown`.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, fields

# Canonical allowed values — the single source of truth.
ACTION_VALUES: tuple[str, ...] = ("notify", "digest", "mute")

# Confidence calibration (Phase 4, measured 2026-08-02). The model is
# systematically over-confident: mean bias +0.095 on the 30 samples, MAE
# 0.096, only 50% within 0.1 of the golden label. The golden confidence is
# near-constant per action (notify 0.874±0.019 n=9, digest 0.816±0.023 n=11,
# mute 0.836±0.025 n=10) and the model's within-action confidence carries no
# signal (corr ≈ 0). Shrinkage toward the per-action golden mean (w=0.85)
# gives MAE 0.022, 100% within 0.1 — see RESEARCH.md decision log. A single
# hyperparameter, applied as a simple mapping (not a fitted curve) so it
# stays explainable and robust on the hidden set.
ACTION_CONF_MEAN = {"notify": 0.874, "digest": 0.816, "mute": 0.836}
CALIB_SHRINK = 0.85


def calibrate_confidence(action: str, confidence: float) -> float:
    """Map raw model confidence toward the per-action golden mean.

    Unknown actions pass through unchanged; output is clamped to [0, 1].
    Non-finite input raises ValueError — it must not silently clamp to 0/1.
    """
    _require_finite(confidence)
    mean = ACTION_CONF_MEAN.get(action)
    if mean is None:
        return round(max(0.0, min(1.0, confidence)), 3)
    cal = CALIB_SHRINK * mean + (1.0 - CALIB_SHRINK) * confidence
    return round(max(0.0, min(1.0, cal)), 3)


def _require_finite(value: float) -> float:
    """Reject NaN/inf confidence — they must not silently clamp to 0/1."""
    if not math.isfinite(value):
        raise ValueError(f"confidence must be finite, got {value!r}")
    return value
MESSAGE_TYPE_VALUES: tuple[str, ...] = (
    "personal",
    "urgent",
    "event",
    "payment",
    "business_update",
    "promotion",
    "greeting",
    "forward",
    "spam",
    "scam",
    "unknown",
)

MIN_REASON_CHARS = 12
GENERIC_REASONS = {
    "n/a", "na", "none", "unknown", "not applicable", "see message",
    "based on the message", "the message is not important", "low priority",
    "high priority", "important message", "not important", "default",
}


@dataclass(frozen=True)
class RoutingOutput:
    """One validated prediction row (matches the output.csv contract)."""

    message_id: str
    action: str
    message_type: str
    reason: str
    confidence: float
    evidence_message_ids: str

    def as_row(self) -> list[str]:
        return [
            self.message_id,
            self.action,
            self.message_type,
            self.reason,
            f"{self.confidence:.2f}",
            self.evidence_message_ids,
        ]

    def as_row_dict(self) -> dict:
        """dict form for checkpoint files / pipeline rows (same fields)."""
        return {
            "message_id": self.message_id,
            "action": self.action,
            "message_type": self.message_type,
            "reason": self.reason,
            "confidence": self.confidence,
            "evidence_message_ids": self.evidence_message_ids,
        }


# ---------------------------------------------------------------------------
# Three-tier value matching
# ---------------------------------------------------------------------------


def _normalize(value: str) -> str:
    """'Business Update' / 'business-update' / 'business_update' -> 'business_update'.

    Lowercase; any run of non-alphanumerics becomes a single underscore; strip
    edge underscores.
    """
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _tokens(value: str) -> set[str]:
    return set(_normalize(value).split("_")) - {"", "now", "please", "the"}


def match_value(raw: str | None, allowed: tuple[str, ...]) -> str | None:
    """Three-tier match of `raw` against `allowed`.

    Tier 1: exact (case-insensitive).
    Tier 2: normalized equivalence.
    Tier 3: token-subset — every meaningful token of `raw` appears in exactly
            one allowed value's tokens (e.g. 'business update' -> business_update,
            'notify now' -> notify). Unambiguous matches only.
    Returns None when nothing matches (caller decides the fallback).
    """
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return None

    # Tier 1 + 2: exact or normalized equivalence
    low = raw.lower()
    if low in allowed:
        return low
    norm = _normalize(raw)
    if norm in allowed:
        return norm

    # Tier 3: token-subset, unambiguous
    raw_tokens = _tokens(raw)
    if raw_tokens:
        hits = [v for v in allowed if raw_tokens <= set(v.split("_"))]
        if len(hits) == 1:
            return hits[0]
    return None


def match_action(raw: str | None) -> str | None:
    return match_value(raw, ACTION_VALUES)


def match_message_type(raw: str | None) -> str:
    """message_type is never fatal: unmatched output collapses to `unknown`."""
    return match_value(raw, MESSAGE_TYPE_VALUES) or "unknown"


# ---------------------------------------------------------------------------
# JSON extraction + RoutingOutput construction
# ---------------------------------------------------------------------------


def extract_json_object(text: str) -> dict | None:
    """Pull the first balanced {...} JSON object out of model output.

    Tolerates code fences, prose before/after, and trailing commas (a common
    model slip). Returns None when no parseable object exists.
    """
    start = text.find("{")
    while start != -1:
        # Find the matching close brace, honoring strings.
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    parsed = _try_parse_json(candidate)
                    if parsed is not None:
                        return parsed
                    break
        start = text.find("{", start + 1)
    return None


def _try_parse_json(candidate: str) -> dict | None:
    """Tolerant JSON parsing: plain, trailing commas stripped, Python-literal
    single quotes (ast.literal_eval), then a last-resort single->double quote
    swap when the candidate contains no double quotes at all."""
    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    # trailing commas
    fixed = re.sub(r",\s*([}\]])", r"\1", candidate)
    try:
        obj = json.loads(fixed)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    # single-quoted Python literal (keys and simple strings)
    try:
        import ast

        obj = ast.literal_eval(candidate)
        return obj if isinstance(obj, dict) else None
    except (ValueError, SyntaxError):
        pass
    # last resort: no double quotes anywhere -> swap singles for doubles
    if '"' not in candidate:
        try:
            obj = json.loads(candidate.replace("'", '"'))
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def validate_reason(reason: str) -> str | None:
    """Return a problem description, or None if the reason is acceptable.

    The rubric caps empty/generic/self-contradictory reasons at ~70 even with
    a correct action, so we refuse thin reasons and make the model rewrite.
    """
    r = reason.strip()
    if not r:
        return "reason is empty"
    if len(r) < MIN_REASON_CHARS:
        return f"reason is only {len(r)} chars (need >= {MIN_REASON_CHARS}): '{r}'"
    if r.lower() in GENERIC_REASONS:
        return f"reason is generic: '{r}'"
    return None


def build_routing(
    message_id: str,
    raw: dict,
    known_history_ids: set[str] | None = None,
    strict_action: bool = True,
) -> tuple[RoutingOutput | None, str | None]:
    """Construct a validated RoutingOutput from a raw model dict.

    Returns (output, None) on success or (None, problem) when the model must
    fix something. Normalizations applied:
    - action: three-tier matched; None -> error (strict) or 'digest' fallback.
    - message_type: matched, else 'unknown' (never fatal).
    - confidence: clamped to [0, 1], float()-parseable.
    - reason: must pass validate_reason.
    - evidence: 'none' or semicolon-separated ids, filtered to ids that exist
      in history (invented ids are dropped; all-invented -> 'none').
    """
    action = match_action(raw.get("action"))
    if action is None:
        if strict_action:
            return None, f"invalid action {raw.get('action')!r}; allowed: {ACTION_VALUES}"
        action = "digest"

    message_type = match_message_type(raw.get("message_type"))

    reason = str(raw.get("reason") or "").strip()
    problem = validate_reason(reason)
    if problem:
        return None, problem

    try:
        confidence = float(raw.get("confidence"))
    except (TypeError, ValueError):
        return None, f"confidence not a number: {raw.get('confidence')!r}"
    try:
        _require_finite(confidence)
    except ValueError as exc:
        return None, str(exc)
    confidence = max(0.0, min(1.0, confidence))
    confidence = calibrate_confidence(action, confidence)

    evidence = _clean_evidence(str(raw.get("evidence_message_ids") or ""), known_history_ids)

    return (
        RoutingOutput(
            message_id=message_id,
            action=action,
            message_type=message_type,
            reason=reason,
            confidence=confidence,
            evidence_message_ids=evidence,
        ),
        None,
    )


def _clean_evidence(raw: str, known_history_ids: set[str] | None) -> str:
    """Normalize evidence to 'id1;id2' or 'none'; drop ids not in history."""
    parts = [
        p.strip().strip('"').strip("'")
        for p in re.split(r"[;,\s]+", raw)
        if p.strip()
    ]
    keep: list[str] = []
    for p in parts:
        if p.lower() == "none":
            continue
        if known_history_ids is not None and p not in known_history_ids:
            continue
        if p not in keep:
            keep.append(p)
    return ";".join(keep[:2]) if keep else "none"


# ---------------------------------------------------------------------------
# Validation helpers for tests / pipeline
# ---------------------------------------------------------------------------


def validate_output(row: dict) -> list[str]:
    """Strict contract check for a finished output row (pipeline use).

    Returns a list of violations (empty when the row is fully valid).
    """
    problems: list[str] = []
    for field in ("message_id", "action", "message_type", "reason",
                  "confidence", "evidence_message_ids"):
        if field not in row or row[field] in (None, ""):
            problems.append(f"missing {field}")
    if "action" in row and row["action"] not in ACTION_VALUES:
        problems.append(f"action {row['action']!r} not in {ACTION_VALUES}")
    if "message_type" in row and row["message_type"] not in MESSAGE_TYPE_VALUES:
        problems.append(
            f"message_type {row['message_type']!r} not in {MESSAGE_TYPE_VALUES}"
        )
    if "confidence" in row:
        try:
            c = float(row["confidence"])
            if not 0.0 <= c <= 1.0:
                problems.append(f"confidence {c} out of range")
        except (TypeError, ValueError):
            problems.append(f"confidence {row['confidence']!r} not numeric")
    if "reason" in row and validate_reason(str(row["reason"])) is not None:
        problems.append(f"reason invalid: {validate_reason(str(row['reason']))}")
    return problems


def routing_fields() -> tuple[str, ...]:
    """Field names of RoutingOutput in output.csv order (contract, §6.2)."""
    return tuple(f.name for f in fields(RoutingOutput))
