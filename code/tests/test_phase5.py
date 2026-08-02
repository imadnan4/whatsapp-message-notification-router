"""Phase 5 tests: canonical pattern-clustering + grouped evidence rendering.

All offline — no API calls. Measured baseline (2026-08-02, RESEARCH.md):
- Golden evidence is ALWAYS the min-id member of the incoming message's
  pattern-cluster (pairwise TF-IDF cosine, incoming message joins the
  cluster graph): rank-1 for 27/28 sample rows, top-2 for 28/28 at
  CANONICAL_SIM_THRESHOLD=0.15 (swept 0.10-0.25: 0.10 over-clusters
  sample_msg_047 -> rank 4; 0.20+ under-clusters near-copies).
- The grouped prompt block renders the pattern lead first so the model
  cites the original instead of later near-duplicate copies.
"""

from __future__ import annotations

from dataclasses import fields

from core.agent import build_context
from core.data_loader import Message, load_dataset
from core.pipeline import majority_vote
from core.retrieval import (
    CANONICAL_SIM_THRESHOLD,
    build_index,
    retrieve_evidence,
)

DS = load_dataset()
INDEX = build_index(DS)
_F = {f.name for f in fields(Message)}


def _as_msg(s) -> Message:
    return Message(**{f: getattr(s, f) for f in _F})


def _pattern_lead(s) -> str | None:
    """The pattern-lead candidate id for a sample (canonical of the cluster
    that contains the incoming message)."""
    cands = retrieve_evidence(DS, _as_msg(s), index=INDEX, k=6)
    for c in cands:
        if c.pattern_lead:
            return c.message_id
    return None


def _display_order(s) -> list[str]:
    """The exact order the prompt block renders candidates in (pattern lead
    first, then other canonicals by -sim, then others by score)."""
    cands = retrieve_evidence(DS, _as_msg(s), index=INDEX, k=6)
    lead = sorted(
        (c for c in cands if c.pattern_lead),
        key=lambda c: (-c.similarity, c.message_id),
    )
    canon = sorted(
        (c for c in cands if c.is_canonical and not c.pattern_lead),
        key=lambda c: (-c.similarity, c.message_id),
    )
    rest = [c for c in cands if not c.is_canonical]
    return [c.message_id for c in lead + canon + rest]


def test_threshold_constant_measured():
    # The 0.15 value is a measured sweep result (see module docstring) —
    # pin it so nobody re-tunes without re-measuring.
    assert CANONICAL_SIM_THRESHOLD == 0.15


def test_golden_evidence_is_rank1_in_display_for_27_of_28():
    # Every non-'none' golden evidence id must be the FIRST candidate shown,
    # except sample_msg_042 (semantic cross-conversation health anchor, sim
    # 0.0 — unreachable by content similarity; the prompt topic rule covers
    # it, and it stays in the top-2).
    n_rank1 = 0
    for s in DS.samples:
        gids = [x for x in s.evidence_message_ids.split(";") if x and x != "none"]
        if not gids:
            continue
        order = _display_order(s)
        assert gids[0] in order, f"{s.message_id}: {gids[0]} not listed"
        if order[0] == gids[0]:
            n_rank1 += 1
        else:
            assert order.index(gids[0]) <= 1, f"{s.message_id}: {gids[0]} at {order.index(gids[0])}"
    assert n_rank1 == 27
    assert _pattern_lead(
        next(s for s in DS.samples if s.message_id == "sample_msg_042")
    ) != "message_0047"


def test_golden_evidence_in_top2_of_grouped_display_for_all_rows():
    # The grouped rendering (canonical first) must keep the golden evidence
    # inside the first two listed candidates for every labeled row.
    for s in DS.samples:
        gids = [x for x in s.evidence_message_ids.split(";") if x and x != "none"]
        if not gids:
            continue
        ctx = build_context(DS, _as_msg(s), None, index=INDEX)
        block = ctx.blocks["EVIDENCE CANDIDATES"]
        listed = [
            line.strip().split("[")[1].split("]")[0]
            for line in block.splitlines()
            if line.strip().startswith("[message_")
        ]
        assert gids[0] in listed[:2], f"{s.message_id}: {gids[0]} not in top-2"


def test_near_duplicate_of_incoming_is_not_canonical():
    # sample_msg_002's near-copy message_0133 (same bus notice, higher sim)
    # must be demoted to OTHER INSTANCES; message_0002 is the pattern lead.
    s = next(x for x in DS.samples if x.message_id == "sample_msg_002")
    cands = retrieve_evidence(DS, _as_msg(s), index=INDEX, k=6)
    by_id = {c.message_id: c for c in cands}
    assert by_id["message_0002"].is_canonical
    assert by_id["message_0002"].pattern_lead
    assert not by_id["message_0133"].is_canonical


def test_canonical_marking_deterministic_across_hash_seeds():
    import os
    import subprocess
    import sys

    code = (
        "from dataclasses import fields\n"
        "from core.data_loader import load_dataset\n"
        "from core.retrieval import build_index, retrieve_evidence\n"
        "from core.agent import build_context\n"
        "DS = load_dataset(); IDX = build_index(DS)\n"
        "F = {f.name for f in fields(__import__('core.data_loader', fromlist=['Message']).Message)}\n"
        "out = []\n"
        "for s in DS.samples:\n"
        "    m = __import__('core.data_loader', fromlist=['Message']).Message(**{f: getattr(s, f) for f in F})\n"
        "    ctx = build_context(DS, m, None, index=IDX)\n"
        "    out.append(ctx.blocks['EVIDENCE CANDIDATES'])\n"
        "print('\\n'.join(out))\n"
    )
    env = {**os.environ, "PYTHONHASHSEED": "0"}
    a = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd="."
    )
    env["PYTHONHASHSEED"] = "42"
    b = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd="."
    )
    assert a.returncode == 0 and b.returncode == 0, (a.stderr, b.stderr)
    assert a.stdout == b.stdout


# ---------------------------------------------------------------------------
# Majority voting (Phase 5 robustness candidate)
# ---------------------------------------------------------------------------


def _row(mid, action, mtype, reason, conf, ev):
    return {
        "message_id": mid,
        "action": action,
        "message_type": mtype,
        "reason": reason,
        "confidence": conf,
        "evidence_message_ids": ev,
    }


def test_majority_vote_picks_majority_combo():
    rows = [
        _row("m1", "notify", "event", "r-a", 0.85, "message_0001"),
        _row("m1", "notify", "event", "r-b", 0.84, "message_0001"),
        _row("m1", "digest", "unknown", "r-c", 0.80, "none"),
    ]
    out = majority_vote(rows)
    assert (out["action"], out["message_type"]) == ("notify", "event")
    # fields come from the majority combo's earliest pass row
    assert out["reason"] == "r-a" and out["confidence"] == 0.85


def test_majority_vote_tie_breaks_by_earliest_pass():
    rows = [
        _row("m1", "notify", "event", "r-a", 0.85, "message_0001"),
        _row("m1", "digest", "unknown", "r-c", 0.80, "none"),
    ]
    out = majority_vote(rows)
    assert (out["action"], out["message_type"]) == ("notify", "event")
    assert out["reason"] == "r-a"


def test_majority_vote_requires_rows():
    import pytest as _pytest

    with _pytest.raises(ValueError):
        majority_vote([])


# ---------------------------------------------------------------------------
# Phase 5b — semantic fallback for voice/empty-text queries
# ---------------------------------------------------------------------------


class _StubSem:
    """Duck-typed SemanticIndex: fixed similarities, no API, no cache."""

    def __init__(self, sims: dict[str, float]):
        self._sims = sims

    def similarities(self, query_text: str) -> dict[str, float]:
        return dict(self._sims)


def test_sem_fallback_creates_lead_for_042_without_lexical_signal():
    # sample_msg_042 is a voice note (empty message_text, sim 0 for every
    # candidate). With the semantic fallback the health anchor message_0047
    # (cosine 0.43 >= SEM_EDGE_THRESHOLD) becomes the pattern lead.
    from core.retrieval import SEM_EDGE_THRESHOLD

    s = next(x for x in DS.samples if x.message_id == "sample_msg_042")
    sem = _StubSem({"message_0035": 0.17, "message_0047": 0.43})
    cands = retrieve_evidence(
        DS, _as_msg(s), index=INDEX, k=6,
        query_text="Please call now. Dad is unwell and we are going to the clinic.",
        sem=sem,
    )
    leads = [c for c in cands if c.pattern_lead]
    assert len(leads) == 1
    assert leads[0].message_id == "message_0047"
    assert leads[0].sem_similarity == 0.43
    assert SEM_EDGE_THRESHOLD == 0.40  # measured sweep; see module docstring


def test_sem_fallback_ignored_when_below_threshold():
    # 0.35 < 0.40 -> no edge, no lead (the pre-Phase-5b behavior).
    s = next(x for x in DS.samples if x.message_id == "sample_msg_042")
    sem = _StubSem({"message_0035": 0.17, "message_0047": 0.35})
    cands = retrieve_evidence(
        DS, _as_msg(s), index=INDEX, k=6,
        query_text="Please call now. Dad is unwell and we are going to the clinic.",
        sem=sem,
    )
    assert not any(c.pattern_lead for c in cands)


def test_sem_fallback_never_fires_with_lexical_signal():
    # sample_msg_002 HAS a lexical edge (message_0002 sim 0.26 >= 0.15) —
    # sem must not rewire its cluster even if a different candidate scores
    # high semantically.
    s = next(x for x in DS.samples if x.message_id == "sample_msg_002")
    sem = _StubSem({"message_0002": 0.9, "message_0133": 0.95})
    cands = retrieve_evidence(DS, _as_msg(s), index=INDEX, k=6, sem=sem)
    leads = [c for c in cands if c.pattern_lead]
    assert len(leads) == 1 and leads[0].message_id == "message_0002"


def test_sem_fallback_integration_with_real_cache():
    # Real cached embeddings (built 2026-08-02): 042's golden becomes the
    # pattern lead. Skipped when the cache is absent (fresh clone / offline).
    from core.semantic import SemanticIndex

    sem = SemanticIndex.load()
    if sem is None:
        import pytest

        pytest.skip("semantic cache not built (run main.py once to build)")
    s = next(x for x in DS.samples if x.message_id == "sample_msg_042")
    cands = retrieve_evidence(
        DS, _as_msg(s), index=INDEX, k=6,
        query_text="Please call now. Dad is unwell and we are going to the clinic.",
        sem=sem,
    )
    leads = [c for c in cands if c.pattern_lead]
    assert leads and leads[0].message_id == "message_0047"
