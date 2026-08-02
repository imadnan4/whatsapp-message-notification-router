"""Phase 5b — multi-run comparison + bootstrap confidence intervals.

Research basis (Indeed Engineering, "Bootstrap confidence intervals for LLM
evaluation", 2026-07 — cited in RESEARCH.md): LLM outputs are non-deterministic,
so a single-run accuracy number on N inputs has two sources of variance: input
sampling and model non-determinism. The validated recipe is the CLUSTER
bootstrap: resample the N inputs with replacement, carrying all k runs of each
chosen input, and recompute the metric per resample; the 2.5/97.5 percentiles
are a 95% interval for the accuracy of a single stochastic call.

We deploy majority-voted output (main.py --majority 3), which is a DIFFERENT
estimand; for it we report the mode-aggregate bootstrap (resample inputs,
take the mode across runs per input, recompute). The mode-aggregate bootstrap
is NOT a valid CI for single-call accuracy (per the same article) — the two
intervals must not be mixed.

Usage (from code/, after at least 2 eval runs):
    uv run python evaluation/compare_runs.py [report_*.json ...]
    # default: the 3 most recent report JSONs in .cache/eval/
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_BOOTSTRAP = 2000
_SEED = 42


def load_runs(paths: list[str]) -> tuple[list[str], list[dict[str, dict]]]:
    """Load eval report JSONs; returns (message_ids, [ {message_id: row} ])."""
    runs: list[dict[str, dict]] = []
    for p in paths:
        rep = json.loads(Path(p).read_text(encoding="utf-8"))
        rows = {r["message_id"]: r for r in rep["rows"]}
        runs.append(rows)
    mids = list(runs[0].keys())
    assert all(set(r.keys()) == set(mids) for r in runs), "runs have different row sets"
    return mids, runs


def row_metrics(row: dict) -> dict[str, float]:
    def _f(v) -> float:
        if isinstance(v, bool):
            return float(v)
        if isinstance(v, str):
            return float(v.lower() == "true")
        return float(v)

    return {
        "action": _f(row.get("action_ok")),
        "type": _f(row.get("type_ok")),
        "evidence": _f(row.get("evidence_hit")),
        "reason_f1": float(row.get("reason_f1", 0.0)),
        "conf_mae": float(row.get("conf_mae", 0.0)),
    }


def accuracy(mids: list[str], runs: list[dict[str, dict]], metric: str) -> float:
    vals = [row_metrics(r[mid])[metric] for mid in mids for r in runs]
    return sum(vals) / len(vals)


def cluster_bootstrap(
    mids: list[str], runs: list[dict[str, dict]], metric: str
) -> tuple[float, float, float]:
    """95% CI for the accuracy of ONE stochastic call (cluster bootstrap:
    resample inputs, carry all k runs per input)."""
    k = len(runs)
    rng = random.Random(_SEED)
    stats = []
    for _ in range(_BOOTSTRAP):
        sample = [rng.choice(mids) for _ in mids]
        vals = [row_metrics(runs[j][mid])[metric] for mid in sample for j in range(k)]
        stats.append(sum(vals) / len(vals))
    stats.sort()
    lo, hi = stats[int(0.025 * len(stats))], stats[int(0.975 * len(stats))]
    return accuracy(mids, runs, metric), lo, hi


def mode_bootstrap(
    mids: list[str], runs: list[dict[str, dict]], metric: str
) -> tuple[float, float, float]:
    """95% CI for the MAJORITY-VOTED deployment: resample inputs, take the
    (action,type) mode across runs per input (ties -> first run), recompute.
    This estimates the deployed system, not a single call."""
    rng = random.Random(_SEED)
    stats = []
    for _ in range(_BOOTSTRAP):
        sample = [rng.choice(mids) for _ in mids]
        score = 0.0
        for mid in sample:
            # mode over (action, type) with first-run tie-break
            combos = [
                (r[mid]["pred_action"], r[mid]["pred_type"]) for r in runs
            ]
            combo = max(
                set(combos), key=lambda c: (combos.count(c), -combos.index(c))
            )
            idx = combos.index(combo)
            score += row_metrics(runs[idx][mid])[metric]
        stats.append(score / len(sample))
    stats.sort()
    lo, hi = stats[int(0.025 * len(stats))], stats[int(0.975 * len(stats))]
    # point estimate: accuracy of the mode-voted rows
    point = 0.0
    for mid in mids:
        combos = [
            (r[mid]["pred_action"], r[mid]["pred_type"]) for r in runs
        ]
        combo = max(set(combos), key=lambda c: (combos.count(c), -combos.index(c)))
        point += row_metrics(runs[combos.index(combo)][mid])[metric]
    return point / len(mids), lo, hi


def main() -> int:
    global _BOOTSTRAP
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", help="predictions CSVs (default: 3 newest)")
    ap.add_argument("--boot", type=int, default=_BOOTSTRAP)
    args = ap.parse_args()
    _BOOTSTRAP = args.boot

    if args.files:
        paths = args.files
    else:
        base = Path(__file__).resolve().parents[2] / ".cache" / "eval"
        paths = sorted(glob.glob(str(base / "report_*.json")))[-3:]

    mids, runs = load_runs(paths)
    n, k = len(mids), len(runs)
    print(f"inputs: {n}, runs: {k} ({[Path(p).name for p in paths]})")
    print(f"bootstrap resamples: {_BOOTSTRAP} (seed {_SEED})")

    # per-row stability across runs
    print("\nper-row agreement across runs (action/type/evidence):")
    agree = Counter()
    for mid in mids:
        a = len({r[mid]["pred_action"] for r in runs})
        t = len({r[mid]["pred_type"] for r in runs})
        e = len({r[mid]["pred_evidence"] for r in runs})
        agree[(a == 1, t == 1, e == 1)] += 1
    stable = agree[(True, True, True)]
    print(f"  all fields identical on {stable}/{n} rows; "
          f"action identical {sum(c for (a, _, _), c in agree.items() if a)}/{n}; "
          f"type identical {sum(c for (_, t, _), c in agree.items() if t)}/{n}")

    print("\nfield | single-call acc (95% CI, cluster boot) | voted acc (95% CI, mode boot)")
    for metric in ("action", "type", "evidence"):
        pt, lo, hi = cluster_bootstrap(mids, runs, metric)
        vpt, vlo, vhi = mode_bootstrap(mids, runs, metric)
        print(
            f"{metric:<9} | {pt:.1%} ({lo:.1%}-{hi:.1%})"
            f" | {vpt:.1%} ({vlo:.1%}-{vhi:.1%})"
        )
    # reason F1 + confidence MAE (mean across runs; no CI — continuous metrics)
    f1s = [row_metrics(r[mid])["reason_f1"] for mid in mids for r in runs]
    maes = [row_metrics(r[mid])["conf_mae"] for mid in mids for r in runs]
    print(f"\nreason F1 mean: {sum(f1s)/len(f1s):.3f} | confidence MAE mean: {sum(maes)/len(maes):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
