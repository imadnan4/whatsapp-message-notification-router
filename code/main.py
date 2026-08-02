"""Phase 3 — terminal entry point.

Usage (from code/):
    uv run python main.py [--dataset-dir ../dataset] [--output ../dataset/output.csv]
                          [--provider openai|deepseek] [--no-resume] [--limit N]

Reads every CSV in dataset/, routes each of the 110 incoming messages through
the agent (bounded tool loop) + the deterministic post-model safety gate,
writes output.csv with EXACTLY one row per message (contract §6.2), and
prints a per-rule gate summary + token/cost usage.

Resume: finished rows live in .cache/checkpoints/routing.jsonl (gitignored);
a re-run skips them. Use --no-resume for a clean full run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.agent import RoutingAgent
from core.data_loader import load_dataset, repo_root
from core.pipeline import Pipeline
from core.providers import usage
from core.semantic import SemanticIndex


def _progress(row: dict, rule, i: int, n: int) -> None:
    gate = f" [gate: {rule}]" if rule else ""
    flag = ""
    if str(row.get("reason", "")).startswith("Fallback ("):
        flag = " [FALLBACK]"
    print(
        f"[{i:03d}/{n}] {row['message_id']} -> {row['action']}/{row['message_type']}"
        f"{gate}{flag}"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Message Notification Router — route dataset/messages.csv to output.csv"
    )
    ap.add_argument("--dataset-dir", default=str(repo_root() / "dataset"))
    ap.add_argument("--output", default=str(repo_root() / "dataset" / "output.csv"))
    ap.add_argument("--provider", default="openai", choices=["openai", "deepseek"])
    ap.add_argument(
        "--no-resume", action="store_true", help="ignore checkpoints (clean run)"
    )
    ap.add_argument("--limit", type=int, default=0, help="route only the first N messages (dev)")
    ap.add_argument("--quiet", action="store_true", help="suppress per-row progress")
    ap.add_argument(
        "--majority",
        type=int,
        default=1,
        help="run N fresh passes and majority-vote action+type for the final "
        "output.csv (default 1 = single pass; ~$0.10 per pass)",
    )
    args = ap.parse_args(argv)

    ds = load_dataset(args.dataset_dir)
    total = len(ds.incoming)
    if args.limit:
        total = min(total, args.limit)
    if args.majority > 1 and args.limit:
        raise SystemExit("--majority is for full runs only (drop --limit)")
    print(f"routing {total} incoming messages (provider={args.provider}, "
          f"resume={'off' if args.no_resume else 'on'}, "
          f"majority={args.majority} pass(es)) ...")

    # Semantic evidence index (voice/empty-text fallback) — builds once on
    # first run (~$0.001), degrades to pure TF-IDF when unavailable.
    sem = SemanticIndex.ensure(ds)
    if sem is None:
        print("note: semantic index unavailable — voice-message evidence "
              "falls back to lexical retrieval only")

    if args.majority > 1:
        return _run_majority(ds, args, sem)

    pipe = Pipeline(
        ds,
        agent=RoutingAgent(ds, provider_name=args.provider, sem_index=sem),
        resume=not args.no_resume,
    )
    report = pipe.run(limit=args.limit, progress=None if args.quiet else _progress)

    if args.limit:
        # Dev mode: only the first N rows are routed; the output contract
        # (exactly 110 rows) only holds for a full run.
        print(f"\n[dev] limited run ({args.limit} rows) — output.csv not written; "
              f"run without --limit for the full submission.")
        n = report.routed + report.resumed
    else:
        n = pipe.write_output(args.output)

    print("\n===== RUN SUMMARY =====")
    if args.limit:
        print(f"rows routed: {n} (dev mode — no output.csv)")
    else:
        print(f"rows written: {n} / {len(ds.incoming)} -> {args.output}")
    print(f"routed fresh: {report.routed} | resumed from checkpoint: {report.resumed}")
    if report.fallbacks:
        print(f"fallback rows: {len(report.fallbacks)} -> {report.fallbacks}")
    if report.gate_flips:
        flips = ", ".join(f"{k}={v}" for k, v in sorted(report.gate_flips.items()))
        print(f"safety-gate flips: {flips}")
    print(f"wall time: {report.wall_seconds:.1f}s")
    print(usage.summary())
    return 0


def _run_majority(ds, args, sem=None) -> int:
    """N fresh passes (resume off) -> majority-voted output.csv."""
    passes: list[dict[str, dict]] = []
    reports = []
    for i in range(args.majority):
        print(f"\n----- majority pass {i + 1}/{args.majority} -----")
        pipe = Pipeline(
            ds,
            agent=RoutingAgent(ds, provider_name=args.provider, sem_index=sem),
            resume=False,
        )
        report = pipe.run(progress=None if args.quiet else _progress)
        reports.append(report)
        passes.append({mid: row for mid, row in pipe._done.items()})

    voted_pipe = Pipeline(ds, resume=False)
    n = voted_pipe.write_voted_output(passes, args.output)

    # per-message agreement across passes (action+type combos)
    n_agree3 = 0
    n_agree2 = 0
    for mid in (m.message_id for m in ds.incoming):
        combos = {(p[mid]["action"], p[mid]["message_type"]) for p in passes}
        n_agree3 += len(combos) == 1
        n_agree2 += len(combos) <= 2

    print("\n===== MAJORITY RUN SUMMARY =====")
    print(f"rows written: {n} / {len(ds.incoming)} -> {args.output}")
    print(f"per-pass fallbacks: {[len(r.fallbacks) for r in reports]}")
    print(f"per-pass gate flips: {[dict(r.gate_flips) for r in reports]}")
    print(f"pass agreement: all 3 passes identical on {n_agree3}/{len(ds.incoming)} "
          f"rows; <=2 distinct combos on {n_agree2}/{len(ds.incoming)}")
    print(f"per-pass wall time: {[f'{r.wall_seconds:.1f}s' for r in reports]}")
    print(usage.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
