"""Phase 5b — optional semantic (embedding) retrieval layer.

Why: voice messages have EMPTY message_text, so TF-IDF similarity is 0 for
every candidate and the canonical pattern-lead logic has no signal. The
voice transcript carries the content, but for cross-conversation anchors
(e.g. sample_msg_042's health-context evidence) even the transcript has no
lexical overlap — the link is semantic.

Design (measured 2026-08-02, RESEARCH.md §8): text-embedding-3-small cosine
over the 412 history messages. Used ONLY as a fallback edge when the query
has no lexical edge at all (max candidate TF-IDF < CANONICAL_SIM_THRESHOLD)
and the candidate's cosine >= SEM_EDGE_THRESHOLD (0.40, swept 0.35-0.50).
This lifted golden rank-1 from 27/28 to 28/28 with top-2 28/28 unchanged;
the unrestricted blend (sem edges for every query) was measured and REJECTED
— it dropped rank-1 to 18-23/28 because embedding closeness is topical, not
pattern membership, and it poisoned the canonical-lead selection.

Every path degrades gracefully: no cache, no key, or API failure -> None ->
pure TF-IDF retrieval (all offline tests unaffected).
"""

from __future__ import annotations

import json
from pathlib import Path

from core.data_loader import Dataset

MODEL = "text-embedding-3-small"
DIM = 1536
SEM_EDGE_THRESHOLD = 0.40  # imported by retrieval.py for the fallback edge

# Cache layout: <cache>/sem/history.npy (float32 n x 1536), history_ids.json,
# queries.json (query text -> embedding, keyed by exact text).
_HISTORY_VECS = "history.npy"
_HISTORY_IDS = "history_ids.json"
_QUERY_CACHE = "queries.json"


def _sem_dir() -> Path:
    from core.media import cache_root

    return cache_root() / "sem"


def _normalize(vecs):
    import numpy as np

    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return np.divide(vecs, norms, out=np.zeros_like(vecs), where=norms != 0)


class SemanticIndex:
    """Cosine index over history messages; query vectors cached by text."""

    def __init__(self, vectors, ids: list[str]) -> None:
        self.vectors = vectors  # normalized float32 (n, DIM)
        self.idx = {mid: i for i, mid in enumerate(ids)}

    # -- construction ------------------------------------------------------

    @classmethod
    def load(cls) -> "SemanticIndex | None":
        """Load from the cache; None when absent (offline/degraded path)."""
        d = _sem_dir()
        vec_file, ids_file = d / _HISTORY_VECS, d / _HISTORY_IDS
        if not vec_file.exists() or not ids_file.exists():
            return None
        try:
            import numpy as np

            vecs = np.load(vec_file)
            ids = json.loads(ids_file.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 — corrupt cache = degrade, never crash
            return None
        return cls(_normalize(vecs), ids)

    @classmethod
    def ensure(cls, ds: Dataset) -> "SemanticIndex | None":
        """Load, or build the history cache on demand (~$0.001, one API
        call). Returns None on any failure — sem is an enhancement, never a
        hard dependency.
        """
        existing = cls.load()
        if existing is not None:
            return existing
        try:
            from openai import OpenAI

            from core.providers.base import get_env

            client = OpenAI(api_key=get_env("OPENAI_API_KEY"))
            d = _sem_dir()
            d.mkdir(parents=True, exist_ok=True)
            texts = [(m.message_text or "").strip() for m in ds.history]
            ids = [m.message_id for m in ds.history]
            vecs = cls._embed(client, texts, len(ids))
            import numpy as np

            np.save(d / _HISTORY_VECS, vecs)
            (d / _HISTORY_IDS).write_text(
                json.dumps(ids), encoding="utf-8"
            )
            return cls.load()
        except Exception:  # noqa: BLE001 — never crash the pipeline for sem
            return None

    @staticmethod
    def _embed(client, texts: list[str], n: int, batch: int = 200):
        import numpy as np

        out = [None] * n
        todo = [(i, t) for i, t in enumerate(texts) if t]
        for start in range(0, len(todo), batch):
            chunk = todo[start : start + batch]
            resp = client.embeddings.create(
                model=MODEL, input=[t for _, t in chunk]
            )
            for data, (i, _) in zip(resp.data, chunk, strict=True):
                out[i] = data.embedding
        return np.array(
            [v if v is not None else np.zeros(DIM, dtype=np.float32) for v in out],
            dtype=np.float32,
        )

    # -- query ---------------------------------------------------------------

    def similarities(self, query_text: str) -> dict[str, float] | None:
        """Cosine of `query_text` against every history message, or None if
        the query cannot be embedded (empty text, API failure, no key)."""
        q = (query_text or "").strip()
        if not q:
            return None
        vec = self._embed_query(q)
        if vec is None:
            return None
        import numpy as np

        n = np.linalg.norm(vec)
        if n == 0:
            return None
        cos = (self.vectors @ (vec / n)).astype(float)
        return {mid: float(cos[i]) for mid, i in self.idx.items()}

    def _embed_query(self, text: str):
        """Embed one query, cached by exact text in queries.json."""
        d = _sem_dir()
        cache_file = d / _QUERY_CACHE
        cache: dict[str, list[float]] = {}
        if cache_file.exists():
            try:
                cache = json.loads(cache_file.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                cache = {}
        if text in cache:
            return cache[text]
        try:
            from openai import OpenAI

            from core.providers.base import get_env

            client = OpenAI(api_key=get_env("OPENAI_API_KEY"))
            resp = client.embeddings.create(model=MODEL, input=[text])
            vec = resp.data[0].embedding
            cache[text] = vec
            cache_file.write_text(json.dumps(cache), encoding="utf-8")
            return vec
        except Exception:  # noqa: BLE001
            return None
