"""Phase 2 — 30-sample evaluation harness (measure every change).

Runs the RoutingAgent over the 30 solved sample rows (labels stripped, so the
agent never sees the golden answer) and reports per-field accuracy:

- action / message_type: exact match
- reason: normalized token F1 vs the golden reason (specificity proxy)
- confidence: mean absolute error + within-0.1 agreement
- evidence: hit rate (any golden id in our output; 'none' matches 'none')

Writes predictions + a JSON report under .cache/eval/ so prompt iterations can
be compared. Baseline to beat: majority action = 11/30 (36.7%).

Usage: uv run python evaluation/run_samples.py [--limit N] [--provider openai|deepseek]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path

# Script is run as `uv run python evaluation/run_samples.py` from code/ —
# make the `core` package importable before importing it below.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.agent import RoutingAgent
from core.data_loader import Message, SampleMessage, load_dataset
from core.providers import usage
from core.semantic import SemanticIndex


_SAMPLE_FIELDS = {f.name for f in fields(Message)}


def _sample_as_message(s: SampleMessage) -> Message:
    """Rebuild a sample row as an unlabeled incoming Message (labels stripped)."""
    return Message(**{f: getattr(s, f) for f in _SAMPLE_FIELDS})


_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "to", "of", "in", "on", "for",
    "and", "or", "but", "with", "this", "that", "from", "at", "by", "as", "it",
    "its", "your", "you", "user", "message", "sender", "does", "not", "has",
}


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP}


def reason_f1(golden: str, predicted: str) -> float:
    g, p = _tokens(golden), _tokens(predicted)
    if not g or not p:
        return 0.0
    inter = len(g & p)
    if inter == 0:
        return 0.0
    prec = inter / len(p)
    rec = inter / len(g)
    return 2 * prec * rec / (prec + rec)


def _evidence_hit(golden: str, predicted: str) -> bool:
    g = {x for x in golden.split(";") if x and x != "none"}
    p = {x for x in predicted.split(";") if x and x != "none"}
    if not g and not p:
        return True  # both none
    return bool(g & p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="route only the first N samples")
    ap.add_argument("--provider", default="openai", choices=["openai", "deepseek"])
    ap.add_argument("--gate", action="store_true",
                    help="apply the deterministic post-model safety gate")
    ap.add_argument("--out", default="", help="report directory (default .cache/eval)")
    args = ap.parse_args()

    ds = load_dataset()
    sem = SemanticIndex.ensure(ds)  # voice/empty-text evidence fallback
    agent = RoutingAgent(ds, provider_name=args.provider, sem_index=sem)
    samples = ds.samples if not args.limit else ds.samples[: args.limit]

    print(f"routing {len(samples)} samples with provider={args.provider} ...")
    rows, report = [], {}
    t0 = time.time()
    for i, s in enumerate(samples, 1):
        m = _sample_as_message(s)
        pred = agent.route(m)
        if args.gate:
            from core.safety_gate import apply_gate

            pred = apply_gate(ds, m, pred, fetch_media=True)
        rows.append(
            {
                "message_id": s.message_id,
                "golden_action": s.action,
                "golden_type": s.message_type,
                "golden_reason": s.reason,
                "golden_evidence": s.evidence_message_ids,
                "pred_action": pred.action,
                "pred_type": pred.message_type,
                "pred_reason": pred.reason,
                "pred_confidence": round(pred.confidence, 3),
                "pred_evidence": pred.evidence_message_ids,
                "action_ok": pred.action == s.action,
                "type_ok": pred.message_type == s.message_type,
                "reason_f1": round(reason_f1(s.reason, pred.reason), 3),
                "evidence_hit": _evidence_hit(s.evidence_message_ids, pred.evidence_message_ids),
                "conf_mae": round(abs(pred.confidence - s.confidence), 3),
            }
        )
        print(
            f"[{i:02d}/{len(samples)}] {s.message_id}: "
            f"gold={s.action}/{s.message_type} pred={pred.action}/{pred.message_type} "
            f"{'OK' if pred.action == s.action else 'XX'}"
        )

    # ---- summary -----------------------------------------------------------
    n = len(rows)
    action_acc = sum(r["action_ok"] for r in rows) / n
    type_acc = sum(r["type_ok"] for r in rows) / n
    evidence_hit = sum(r["evidence_hit"] for r in rows) / n
    mean_f1 = sum(r["reason_f1"] for r in rows) / n
    conf_mae = sum(r["conf_mae"] for r in rows) / n
    conf_close = sum(1 for r in rows if r["conf_mae"] <= 0.1) / n

    print("\n===== SUMMARY =====")
    print(f"action accuracy:      {action_acc:.1%} ({sum(r['action_ok'] for r in rows)}/{n})")
    print(f"message_type acc:     {type_acc:.1%} ({sum(r['type_ok'] for r in rows)}/{n})")
    print(f"evidence hit rate:    {evidence_hit:.1%} ({sum(r['evidence_hit'] for r in rows)}/{n})")
    print(f"reason token F1:      {mean_f1:.3f}")
    print(f"confidence MAE:       {conf_mae:.3f} (within 0.1: {conf_close:.1%})")
    print(f"actions predicted:    {dict(Counter(r['pred_action'] for r in rows))}")
    print(f"types predicted:      {dict(Counter(r['pred_type'] for r in rows))}")
    print(f"\n{usage.summary()}")
    print(f"wall time: {time.time() - t0:.1f}s")

    # ---- per-row detail for iteration --------------------------------------
    print("\n===== MISMATCHES (action or type) =====")
    for r in rows:
        if not r["action_ok"] or not r["type_ok"]:
            print(
                f"{r['message_id']}: action {r['golden_action']}->{r['pred_action']} | "
                f"type {r['golden_type']}->{r['pred_type']} | ev "
                f"{r['golden_evidence']}->{r['pred_evidence']} | conf "
                f"{r['pred_confidence']} | {r['pred_reason']}"
            )

    report = {
        "provider": args.provider,
        "gate": args.gate,
        "n": n,
        "action_accuracy": action_acc,
        "message_type_accuracy": type_acc,
        "evidence_hit_rate": evidence_hit,
        "reason_f1_mean": mean_f1,
        "confidence_mae": conf_mae,
        "confidence_within_0.1": conf_close,
        "rows": rows,
        "usage": {
            "calls": usage.calls,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "cost_usd": round(usage.cost_usd, 5),
        },
    }

    out_dir = Path(args.out) if args.out else Path(__file__).resolve().parents[2] / ".cache" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (out_dir / f"predictions_{stamp}.csv").write_text(
        _predictions_csv(rows), encoding="utf-8"
    )
    (out_dir / f"report_{stamp}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nreport: {out_dir}/report_{stamp}.json")
    return 0


def _predictions_csv(rows: list[dict]) -> str:
    buf = ["message_id,action,message_type,reason,confidence,evidence_message_ids"]
    for r in rows:
        reason = r["pred_reason"].replace('"', "'").replace("\n", " ")
        buf.append(
            f"{r['message_id']},{r['pred_action']},{r['pred_type']},"
            f'"{reason}",{r["pred_confidence"]:.2f},{r["pred_evidence"]}'
        )
    return "\n".join(buf) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
