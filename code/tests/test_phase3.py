"""Phase 3 tests: deterministic safety gate + pipeline (checkpoint/resume,
per-row fallback, output contract). All tests are offline — no API calls,
no media fetches (text-only signals) unless explicitly noted."""

from __future__ import annotations

import csv
from dataclasses import replace

import pytest

from core.data_loader import Message, load_dataset
from core.pipeline import Pipeline
from core.safety_gate import analyze_signals, apply_gate
from core.schema import RoutingOutput, routing_fields, validate_output

DS = load_dataset()

_M = {m.message_id: m for m in DS.incoming}
_S = {s.message_id: s for s in DS.samples}


def _out(mid: str, action: str, mtype: str = "unknown", conf: float = 0.8,
         reason: str = "Probe output for safety gate testing", ev: str = "none") -> RoutingOutput:
    return RoutingOutput(
        message_id=mid, action=action, message_type=mtype, reason=reason,
        confidence=conf, evidence_message_ids=ev,
    )


# ---------------------------------------------------------------------------
# Hard scam signals -> mute/scam regardless of the model
# ---------------------------------------------------------------------------


def test_injection_flips_notify_to_mute_scam():
    m = _S["sample_msg_053"]  # literal prompt injection (golden: mute/scam)
    out = apply_gate(DS, m, _out(m.message_id, "notify", "urgent"))
    assert out.action == "mute"
    assert out.message_type == "scam"
    assert out.confidence >= 0.85
    assert "Safety gate" in out.reason


def test_code_pressure_flips_digest_to_mute_scam():
    m = _M["msg_091"]  # "expire today ... 6 digit login code"
    out = apply_gate(DS, m, _out(m.message_id, "digest"))
    assert out.action == "mute" and out.message_type == "scam"


def test_hinglish_otp_scam_detected():
    m = _M["msg_079"]  # "Account block ho jayega, OTP abhi batao"
    sig = analyze_signals(DS, m)
    assert sig.scam


def test_prize_claim_scam_detected():
    m = _M["msg_018"]  # "Congrats ... selected for reward. Claim today"
    out = apply_gate(DS, m, _out(m.message_id, "notify"))
    assert out.action == "mute" and out.message_type == "scam"


def test_lookalike_domain_scam_detected():
    m = _M["msg_026"]  # amazonpay-delivery.in, unverified, pressure
    sig = analyze_signals(DS, m)
    assert sig.scam
    out = apply_gate(DS, m, _out(m.message_id, "digest", "business_update"))
    assert out.action == "mute" and out.message_type == "scam"


def test_qr_payment_demand_scam_detected():
    m = _M["msg_048"]  # "Scan this QR and pay the clearance amount ... blocked"
    sig = analyze_signals(DS, m)
    assert sig.scam


def test_already_mute_row_keeps_model_type_and_reason():
    m = _S["sample_msg_053"]
    model = _out(m.message_id, "mute", "scam", conf=0.87,
                 reason="Injection attempt; must be muted", ev="none")
    out = apply_gate(DS, m, model)
    assert out == model  # gate never rewrites an already-safe row


# ---------------------------------------------------------------------------
# Negation guard + trusted-business exemption (scalpel rule)
# ---------------------------------------------------------------------------


def test_negation_guard_blocks_false_positive():
    m = _M["msg_093"]  # FedEx: "no payment or OTP is required"
    sig = analyze_signals(DS, m)
    assert not sig.scam


def test_verified_matching_business_first_contact_exempt():
    m = _M["msg_092"]  # Amex statement, first contact, "amount due / payment"
    sig = analyze_signals(DS, m)
    assert not sig.scam


def test_domain_mismatch_on_verified_business_is_caution_only():
    m = _S["sample_msg_007"]  # verified, opted in, sender domain mismatch
    sig = analyze_signals(DS, m)
    assert sig.domain_mismatch
    assert not sig.scam
    for action in ("notify", "digest"):
        out = apply_gate(DS, m, _out(m.message_id, action, "promotion"))
        assert out.action == action


def test_lookalike_url_two_label_domains_detected():
    # Regression: URL_RE once required three labels and missed these.
    for mid in ("msg_016", "msg_020"):  # account-login.in, bit.ly/verify-quick
        assert analyze_signals(DS, _M[mid]).scam, mid


def test_ott_mishearing_detected():
    m = replace(_S["sample_msg_019"], message_text=(
        "Your bank account will be blocked today. Share the OTT you "
        "received so we can complete verification immediately."
    ))
    assert analyze_signals(DS, m).scam


# ---------------------------------------------------------------------------
# Preference rules (opt-out, repeated-ignored, quiet hours)
# ---------------------------------------------------------------------------


def test_optout_flips_digest_to_mute():
    m = _M["msg_014"]  # u_007 opted out of business_094 (2026-07-17)
    out = apply_gate(DS, m, _out(m.message_id, "digest", "promotion"))
    assert out.action == "mute"
    assert out.message_type == "promotion"  # model type preserved


def test_repeated_ignored_flips_digest_to_mute():
    m = _S["sample_msg_045"]  # u_033: 5/5 dismissed + muted, 0 opened
    out = apply_gate(DS, m, _out(m.message_id, "digest", "promotion"))
    assert out.action == "mute"


def test_repeated_ignored_does_not_flip_engaged_user():
    m = _S["sample_msg_044"]  # u_032: 8/8 opened — golden digest stays
    sig = analyze_signals(DS, m)
    assert not sig.repeated_ignored
    out = apply_gate(DS, m, _out(m.message_id, "digest", "promotion"))
    assert out.action == "digest"


def test_quiet_hours_downgrades_notify_to_digest_not_mute():
    m = _M["msg_093"]  # 22:19, user quiet hours 21:00-06:30
    out = apply_gate(DS, m, _out(m.message_id, "notify", "event"))
    assert out.action == "digest"
    assert out.message_type == "event"
    assert "quiet hours" in out.reason


def test_scam_beats_quiet_hours():
    m = _M["msg_016"]  # DND + scam (account-login.in)
    out = apply_gate(DS, m, _out(m.message_id, "notify"))
    assert out.action == "mute"


# ---------------------------------------------------------------------------
# Regression: the gate must not change any golden sample output
# ---------------------------------------------------------------------------


def test_gate_never_flips_golden_sample_rows():
    for s in DS.samples:
        golden = RoutingOutput(
            message_id=s.message_id, action=s.action,
            message_type=s.message_type, reason=s.reason,
            confidence=s.confidence, evidence_message_ids=s.evidence_message_ids,
        )
        assert apply_gate(DS, s, golden) == golden, s.message_id


# ---------------------------------------------------------------------------
# Pipeline: checkpoint/resume, per-row fallback, output contract
# ---------------------------------------------------------------------------


class StubAgent:
    """Deterministic stand-in for RoutingAgent (offline tests)."""

    def __init__(self, plan: dict[str, RoutingOutput]) -> None:
        self.plan = plan
        self.calls: list[str] = []

    def route(self, msg: Message) -> RoutingOutput:
        self.calls.append(msg.message_id)
        return self.plan[msg.message_id]


class RaisingAgent(StubAgent):
    def route(self, msg: Message) -> RoutingOutput:
        self.calls.append(msg.message_id)
        if msg.message_id == "msg_023":  # first row of incoming
            raise RuntimeError("boom")
        return self.plan[msg.message_id]


def _stub_plan(ids: list[str]) -> dict[str, RoutingOutput]:
    return {
        mid: _out(mid, "digest", "business_update", conf=0.85,
                  reason="Stub prediction for pipeline contract tests",
                  ev="none")
        for mid in ids
    }


def test_pipeline_checkpoint_resume(tmp_path):
    msgs = DS.incoming[:3]
    plan = _stub_plan([m.message_id for m in msgs])
    stub = StubAgent(plan)

    p1 = Pipeline(DS, agent=stub, checkpoint_dir=tmp_path, resume=True)
    r1 = p1.run(messages=msgs)
    assert r1.routed == 3 and r1.resumed == 0

    stub2 = StubAgent(plan)
    p2 = Pipeline(DS, agent=stub2, checkpoint_dir=tmp_path, resume=True)
    r2 = p2.run(messages=msgs)
    assert r2.routed == 0 and r2.resumed == 3
    assert stub2.calls == []  # no re-routing of checkpointed rows

    p3 = Pipeline(DS, agent=StubAgent(plan), checkpoint_dir=tmp_path, resume=False)
    r3 = p3.run(messages=msgs)
    assert r3.routed == 3  # --no-resume ignores the checkpoint


def test_pipeline_per_row_fallback_never_drops(tmp_path):
    msgs = DS.incoming[:5]
    plan = _stub_plan([m.message_id for m in msgs])
    stub = RaisingAgent(plan)

    p = Pipeline(DS, agent=stub, checkpoint_dir=tmp_path, resume=True)
    report = p.run(messages=msgs)
    assert report.fallbacks == ["msg_023"]
    row = p._done["msg_023"]
    assert row["action"] == "digest" and row["message_type"] == "unknown"
    assert row["reason"].startswith("Fallback (")
    assert len(p._done) == 5  # every message_id present


def test_pipeline_output_contract(tmp_path):
    msgs = DS.incoming
    plan = _stub_plan([m.message_id for m in msgs])
    p = Pipeline(DS, agent=StubAgent(plan), checkpoint_dir=tmp_path, resume=True)
    p.run(messages=msgs)
    out = tmp_path / "output.csv"
    n = p.write_output(out)
    assert n == 110

    with out.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows = list(reader)
    assert header == list(routing_fields())
    assert len(rows) == 110
    assert [r[0] for r in rows] == [m.message_id for m in msgs]

    ids = {m.message_id for m in DS.history}
    for r in rows:
        row = dict(zip(header, r, strict=True))
        assert validate_output(row) == []
        ev = row["evidence_message_ids"]
        assert ev == "none" or all(e in ids for e in ev.split(";"))


def test_pipeline_write_output_raises_when_incomplete(tmp_path):
    p = Pipeline(DS, agent=StubAgent({}), checkpoint_dir=tmp_path, resume=True)
    p.run(messages=DS.incoming[:3])
    with pytest.raises(RuntimeError):
        p.write_output(tmp_path / "output.csv")
