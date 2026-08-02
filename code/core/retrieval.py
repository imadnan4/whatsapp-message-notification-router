"""Phase 1 — evidence retrieval for `evidence_message_ids` (Phase 4 rev).

Deterministic pipeline (no LLM), per RESEARCH.md §5:
1. Scope candidates structurally: same group+sender / same business /
   same personal sender, for the same receiving user.
2. Phase 4 addition: when the conversation scope has history, ALSO surface
   the same user's early history anchors (lowest message ids) — the 30
   golden labels always cite canonical instances from message_0001..0056
   (an id/file-order convention of the generator, not a time convention),
   sometimes from a different conversation of the same user (measured:
   golden-in-top-6 rose 24/28 -> 28/28). Never for first-contact
   conversations (no pattern context).
3. Score with a fixed weighted formula: TF-IDF text similarity (sklearn),
   sender-match bonus, event behavior (reported/dismissed/muted/ignored),
   recency, both-forwarded, and an earliness bonus (first occurrence =
   canonical evidence).
4. Exact-duplicate texts are deduplicated to their earliest instance
   (event tags merged) — the golden convention cites the original, not
   later copies.
5. Return top-k candidates with tags explaining WHY they match, so the
   routing agent can justify `evidence_message_ids` (and the safety gate can
   cite patterns like "repeatedly dismissed").

412 history docs -> TF-IDF build is milliseconds; the index is reusable across
all 110 incoming messages.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from datetime import datetime, timedelta

from core.data_loader import Dataset, HistoryMessage, Message

# Fixed scoring weights (research-backed, see RESEARCH.md §5; tune only with
# measured evidence from the 30 solved samples).
W_SIM = 0.35
W_SAME_SENDER = 0.30
W_REPORTED = 0.15
W_DISMISSED = 0.10
W_MUTED_AFTER = 0.08
W_NOT_OPENED = 0.05
W_RECENT_14D = 0.08
W_RECENT_60D = 0.04
W_BOTH_FORWARDED = 0.05
# Phase 4 (measured 2026-08-02): golden evidence on all 30 samples cites
# canonical instances in message_0001..0056 (id/file order — the generator's
# anchor window, NOT earliest-by-time: only 5/56 of those ids are among the
# 56 earliest messages by created_at). The first occurrence of a recurring
# pattern is the canonical evidence. Earliness bonus + exact-dup dedup
# lifted golden-in-top-6 from 24/28 to 28/28 (top-2 17/28 -> 26/28;
# W_EARLY=0.15 measured cleanly better than 0.10, no swaps).
W_EARLY = 0.15
EARLY_ANCHOR_MAX_ID = 56  # observed canonical-anchor window in the labels
EARLY_DECAY = 400.0  # bonus = W_EARLY * (1 - id/400), ~0.129 for id 56
MIN_EVIDENCE_SCORE = 0.30
# Phase 5 (measured 2026-08-02): the golden evidence is ALWAYS the min-id
# member of the incoming message's pattern-cluster (pairwise TF-IDF cosine,
# with the incoming message itself joining the cluster graph) — golden at
# rank-1 for 27/28 sample rows, top-2 for 28/28 at T=0.15. T swept 0.10-0.25:
# 0.10 over-clusters (sample_msg_047 golden falls to rank 4), 0.20+ misses
# near-copies of the incoming that bridge to the canonical anchor. The
# cluster's earliest instance is tagged canonical and rendered first in the
# prompt block so the model cites the original instead of later near-dups.
CANONICAL_SIM_THRESHOLD = 0.15
# Phase 5b (measured 2026-08-02): voice messages have EMPTY message_text, so
# lexical similarity is 0 for all candidates and no pattern lead exists.
# Semantic fallback: when NO candidate has a lexical edge (max TF-IDF sim <
# CANONICAL_SIM_THRESHOLD), a candidate may join the incoming's cluster via
# embedding cosine >= SEM_EDGE_THRESHOLD. Swept 0.35-0.50: 0.35/0.40 give
# golden rank-1 28/28 + top-2 28/28 (was 27/28), 0.45 loses sample_msg_042
# (sem 0.427). 0.40 chosen (stricter = fewer spurious edges for hidden rows).
SEM_EDGE_THRESHOLD = 0.40


@dataclass(frozen=True)
class EvidenceCandidate:
    message_id: str
    score: float
    similarity: float
    tags: tuple[str, ...]
    text: str
    created_at: datetime
    is_canonical: bool = False  # earliest instance of its pattern-cluster
    pattern_lead: bool = False  # canonical of the cluster containing the incoming message
    sem_similarity: float | None = None  # embedding cosine (voice/empty-text queries)

    def to_prompt_block(self) -> str:
        sem = f" sem={self.sem_similarity:.2f}" if self.sem_similarity is not None else ""
        return (
            f"[{self.message_id}] sim={self.similarity:.2f}{sem} "
            f"tags={','.join(self.tags)} | {self.text[:220]}"
        )


def _norm(text: str) -> str:
    """Lowercase, collapse whitespace; keeps URLs (link patterns are signal)."""
    return re.sub(r"\s+", " ", text.lower()).strip()


class HistoryIndex:
    """Reusable TF-IDF index over all 412 history messages."""

    def __init__(self, ds: Dataset) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.docs = [_norm(m.message_text) for m in ds.history]
        self.ids = [m.message_id for m in ds.history]
        self.idx = {mid: i for i, mid in enumerate(self.ids)}
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), lowercase=False, sublinear_tf=True
        ).fit(self.docs)
        self.matrix = self.vectorizer.transform(self.docs)
        # 412x412 dense pairwise cosine matrix (~1.4 MB) for pattern
        # clustering (canonical first-instance detection).
        self.pairwise = (self.matrix @ self.matrix.T).toarray()

    def pairwise_sim(self, a_id: str, b_id: str) -> float:
        """Cosine similarity between two history messages (pattern clustering)."""
        return self.pairwise[self.idx[a_id], self.idx[b_id]]

    def similarities(self, text: str) -> dict[str, float]:
        """cosine similarity of `text` against every history message."""
        vec = self.vectorizer.transform([_norm(text)])
        scores = (self.matrix @ vec.T).toarray().ravel()
        return dict(zip(self.ids, scores, strict=True))


def build_index(ds: Dataset) -> HistoryIndex:
    return HistoryIndex(ds)


def _scope(ds: Dataset, msg: Message) -> list[HistoryMessage]:
    """Candidate pool: same conversation scope, same receiving user."""
    if msg.conversation_type == "group":
        return [
            m
            for m in ds.history
            if m.group_id == msg.group_id
            and m.user_id == msg.user_id
            and m.sender_user_id == msg.sender_user_id
        ]
    if msg.conversation_type == "business":
        return [
            m
            for m in ds.history
            if m.business_id == msg.business_id and m.user_id == msg.user_id
        ]
    return [
        m
        for m in ds.history
        if m.conversation_type == "personal"
        and m.user_id == msg.user_id
        and m.sender_user_id == msg.sender_user_id
    ]


def retrieve_evidence(
    ds: Dataset,
    msg: Message,
    index: HistoryIndex | None = None,
    k: int = 6,
    query_text: str | None = None,
    sem: object | None = None,
) -> list[EvidenceCandidate]:
    """Top-k evidence candidates for `msg`, scored deterministically.

    An empty list means no history exists for this sender/conversation — the
    agent must then use `none` as evidence.

    Phase 5b: `query_text` is the voice transcript when message_text is
    empty (voice notes have no lexical content of their own); `sem` is an
    optional SemanticIndex used ONLY as a fallback edge for queries with no
    lexical signal at all (measured: rank-1 27/28 -> 28/28, no regressions;
    the unrestricted semantic blend was measured and rejected).
    """
    if index is None:
        index = build_index(ds)
    qtext = (msg.message_text or "").strip() or (query_text or "").strip()
    sims = index.similarities(qtext)
    now = msg.created_at
    cutoff_14 = now - timedelta(days=14)
    cutoff_60 = now - timedelta(days=60)

    scoped = _scope(ds, msg)
    pool = scoped
    if scoped:
        # Same-user early anchors (canonical first occurrences). Only active
        # when the conversation scope has history: a brand-new conversation
        # has no pattern context to anchor on. Sorted for determinism.
        early_ids = {
            m.message_id
            for m in ds.history
            if m.user_id == msg.user_id
            and _id_num(m.message_id) <= EARLY_ANCHOR_MAX_ID
        } - {m.message_id for m in scoped}
        pool = scoped + [ds.history_by_id[i] for i in sorted(early_ids)]

    candidates: list[EvidenceCandidate] = []
    for m in pool:
        sim = sims.get(m.message_id, 0.0)
        # Scope guarantees same sender/conversation for scoped candidates:
        # that evidence is worth W_SAME_SENDER on top of text similarity
        # (was applied as tag only, not score, until 2026-08-01 CodeRabbit
        # review caught the miss). Early anchors are cross-conversation and
        # get no sender bonus — the earliness bonus is their lift. The
        # sender check is conversation-scoped: a same sender in a DIFFERENT
        # group is not the same conversation (Phase 4 review finding 3).
        if msg.conversation_type == "business":
            same_sender = m.business_id == msg.business_id
        elif msg.conversation_type == "group":
            same_sender = (
                m.group_id == msg.group_id
                and m.sender_user_id == msg.sender_user_id
            )
        else:
            same_sender = m.sender_user_id == msg.sender_user_id
        score = sim * W_SIM + (W_SAME_SENDER if same_sender else 0.0)
        tags: list[str] = []

        # Sender match is implicit in the scope; keep the tag for the agent.
        if same_sender:
            tags.append("same_sender" if msg.conversation_type != "business" else "same_business")
        else:
            tags.append("early_anchor")

        if m.reported:
            score += W_REPORTED
            tags.append("reported")
        if m.dismissed:
            score += W_DISMISSED
            tags.append("dismissed")
        if m.muted_after:
            score += W_MUTED_AFTER
            tags.append("muted_after")
        if m.opened is False:
            score += W_NOT_OPENED
            tags.append("not_opened")
        if m.created_at >= cutoff_14:
            score += W_RECENT_14D
            tags.append("recent_14d")
        elif m.created_at >= cutoff_60:
            score += W_RECENT_60D
            tags.append("recent_60d")
        if m.forwarded_count > 0 and msg.forwarded_count > 0:
            score += W_BOTH_FORWARDED
            tags.append("both_forwarded")

        # Phase 4: first occurrence of a pattern is the canonical evidence.
        score += W_EARLY * max(0.0, 1.0 - _id_num(m.message_id) / EARLY_DECAY)

        candidates.append(
            EvidenceCandidate(
                message_id=m.message_id,
                score=score,
                similarity=sim,
                tags=tuple(tags),
                text=m.message_text,
                created_at=m.created_at,
            )
        )

    # Dedup exact-duplicate texts to their earliest instance WITHIN the same
    # conversation (a group copy and a personal copy are different evidence).
    # The kept candidate carries the earliest instance's own score — the
    # golden convention cites the canonical original — but merges event tags
    # (later copies' behavior still informs the decision). Merge is
    # order-independent: all copies are collected first, then the min-id one
    # wins with the union of every copy's tags (Phase 4 review finding 1).
    def _conv_key(c: EvidenceCandidate) -> str:
        m = ds.history_by_id[c.message_id]
        if m.conversation_type == "group":
            return f"group:{m.group_id}:{m.sender_user_id}"
        if m.conversation_type == "business":
            return f"biz:{m.business_id}"
        return f"per:{m.sender_user_id}"

    by_text: dict[tuple[str, str], list[EvidenceCandidate]] = {}
    for c in candidates:
        by_text.setdefault((_norm(c.text), _conv_key(c)), []).append(c)

    merged: list[EvidenceCandidate] = []
    for copies in by_text.values():
        best = min(copies, key=lambda c: _id_num(c.message_id))
        tags = tuple(dict.fromkeys(t for c in copies for t in c.tags))
        merged.append(
            EvidenceCandidate(
                message_id=best.message_id,
                score=best.score,  # canonical instance's own score
                similarity=best.similarity,
                tags=tags,
                text=best.text,
                created_at=best.created_at,
            )
        )

    candidates = sorted(merged, key=lambda c: (c.score, c.similarity), reverse=True)

    # Phase 5: mark the earliest (min-id) member of each pairwise-similarity
    # pattern-cluster as CANONICAL; the cluster that also contains the
    # incoming message yields the PATTERN LEAD. The golden convention always
    # cites these (27/28 at rank-1, 28/28 in top-2, measured); the prompt
    # renders them first so the model prefers the original instance over
    # later near-duplicate copies.
    sem_sims = None
    no_lexical = bool(scoped) and max((c.similarity for c in candidates), default=0.0) < CANONICAL_SIM_THRESHOLD
    if no_lexical and sem is not None and qtext:
        try:
            sem_sims = sem.similarities(qtext)
        except Exception:  # noqa: BLE001 — sem is an enhancement
            sem_sims = None
    candidates = _mark_canonical(candidates, index, sims, sem_sims, no_lexical)
    return candidates[:k]


def _id_num(message_id: str) -> int:
    """Numeric part of a history message id ('message_0047' -> 47)."""
    return int(message_id.rsplit("_", 1)[1])


def _mark_canonical(
    cands: list[EvidenceCandidate],
    index: HistoryIndex,
    sims: dict[str, float],
    sem_sims: dict[str, float] | None = None,
    no_lexical: bool = False,
) -> list[EvidenceCandidate]:
    """Cluster candidates by pairwise text similarity, with the INCOMING
    message as an extra graph node (its similarity to each candidate is the
    edge weight). The min-id member of each cluster is that pattern's
    canonical first instance; the canonical of the cluster containing the
    incoming is the pattern lead. Order-independent (union-find), so
    results are stable across hash seeds.

    Phase 5b: when the query has NO lexical edge to any candidate
    (no_lexical — the voice-message case), an embedding-cosine edge
    (sem_sims >= SEM_EDGE_THRESHOLD) can connect a candidate instead. The
    unrestricted semantic blend was measured and rejected (rank-1 dropped
    27 -> 18-23: embedding closeness is topical, not pattern membership).
    """
    n = len(cands)
    if n == 0:
        return cands
    ids = [c.message_id for c in cands]
    parent = list(range(n + 1))  # last node = the incoming message

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if index.pairwise_sim(ids[i], ids[j]) >= CANONICAL_SIM_THRESHOLD:
                union(i, j)
        if sims.get(ids[i], 0.0) >= CANONICAL_SIM_THRESHOLD:
            union(i, n)  # candidate belongs to the incoming message's pattern
        elif (
            no_lexical
            and sem_sims is not None
            and sem_sims.get(ids[i], 0.0) >= SEM_EDGE_THRESHOLD
        ):
            union(i, n)  # semantic fallback edge (voice/empty-text queries)

    clusters: dict[int, list[int]] = {}
    for i in range(n + 1):
        clusters.setdefault(find(i), []).append(i)

    result: list[EvidenceCandidate] = []
    replacements: dict[str, EvidenceCandidate] = {}
    for members in clusters.values():
        cand_members = [x for x in members if x < n]
        if not cand_members:
            continue
        best = ids[min(cand_members, key=lambda i: _id_num(ids[i]))]
        lead = n in members  # this pattern is the incoming message's own
        for c in cands:
            if c.message_id == best:
                replacements[c.message_id] = replace(
                    c,
                    is_canonical=True,
                    pattern_lead=lead,
                    sem_similarity=(
                        sem_sims.get(c.message_id) if sem_sims is not None else None
                    ),
                )
    return [replacements.get(c.message_id, c) for c in cands]


def choose_evidence_ids(cands: list[EvidenceCandidate], max_ids: int = 2) -> str:
    """Deterministic evidence string: 'id1;id2' or 'none'.

    Used by the pipeline/safety gate when a deterministic answer is needed;
    the agent may override with well-cited choices (validated in phase 3).
    """
    keep = [c.message_id for c in cands if c.score >= MIN_EVIDENCE_SCORE][:max_ids]
    return ";".join(keep) if keep else "none"
