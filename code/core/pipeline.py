"""Phase 3 — per-message orchestration (winner patterns #5 and #6).

Checkpoint/resume: every finished row is appended to a JSONL checkpoint under
<repo>/.cache/checkpoints/ (gitignored). A resumed run skips rows already
completed, so a rate-limit or crash mid-batch never costs the finished rows.
`--no-resume` (fresh checkpoint) forces a clean run.

Per-row isolation: `agent.route` never raises (its own fallback), and the
pipeline wraps the gate + validation in try/except with a clearly-flagged
conservative row as the last line of defense — one bad row never drops a
message_id and never kills the batch.

Output contract (§6.2): exactly one row per incoming message with columns
message_id,action,message_type,reason,confidence,evidence_message_ids,
validated via schema.validate_output plus evidence-id existence (or 'none').
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from core.agent import RoutingAgent
from core.data_loader import Dataset, Message, repo_root
from core.media import read_image, transcribe_voice
from core.safety_gate import analyze_signals, apply_gate
from core.schema import RoutingOutput, routing_fields, validate_output

FALLBACK_ACTION = "digest"
FALLBACK_TYPE = "unknown"
# Deliberately outside the calibrated band (0.78-0.91): a fallback row is a
# flagged failure state — 0.5 makes it visible in output.csv. Never
# calibrated (it does not go through build_routing).
FALLBACK_CONFIDENCE = 0.5

# Bump when checkpoint row semantics change (e.g. confidence calibration);
# resume refuses rows from older versions so calibrated and uncalibrated
# rows can never mix in one output.csv.
CHECKPOINT_VERSION = 2

# Rules reported in gate stats (order of application).
GATE_RULES = ("scam", "opt_out", "repeated_ignored", "quiet_hours")


def majority_vote(pass_rows: list[dict]) -> dict:
    """Vote one message's rows across N independent passes (Phase 5).

    The winning (action, message_type) combo is the majority one; ties are
    broken deterministically by pass order (earliest pass wins). Reason,
    confidence, and evidence come from the winning combo's earliest pass
    row, so every voted field is a row that really was produced (never a
    synthesized blend). All rows are expected to be identical in
    message_id and already gate-applied.
    """
    if not pass_rows:
        raise ValueError("majority_vote needs at least one pass")
    counts: dict[tuple[str, str], int] = {}
    first: dict[tuple[str, str], dict] = {}
    for row in pass_rows:
        combo = (row["action"], row["message_type"])
        counts[combo] = counts.get(combo, 0) + 1
        first.setdefault(combo, row)
    winner = max(counts, key=lambda c: (counts[c], -pass_rows.index(first[c])))
    return dict(first[winner])


@dataclass
class PipelineReport:
    total: int = 0
    routed: int = 0  # freshly routed this run (excluding resumed rows)
    resumed: int = 0
    fallbacks: list[str] = field(default_factory=list)  # message_ids
    gate_flips: dict[str, int] = field(default_factory=dict)  # rule -> count
    wall_seconds: float = 0.0


def _media_text(ds: Dataset, msg: Message) -> str | None:
    """Cached media read/transcript for the gate (agent already fetched it)."""
    try:
        if msg.media_type == "image" and msg.media_id:
            return read_image(ds, msg.media_id)
        if msg.media_type == "voice" and msg.media_id:
            return transcribe_voice(ds, msg.media_id)
    except Exception:  # noqa: BLE001 — gate input is best-effort
        return None
    return None


class Pipeline:
    """Routes every incoming message with checkpoint/resume + per-row isolation."""

    def __init__(
        self,
        ds: Dataset,
        agent: RoutingAgent | None = None,
        checkpoint_dir: str | Path | None = None,
        resume: bool = True,
    ) -> None:
        self.ds = ds
        self.agent = agent or RoutingAgent(ds)
        self.history_ids = {m.message_id for m in ds.history}
        self.checkpoint_path = Path(
            checkpoint_dir or (repo_root() / ".cache" / "checkpoints")
        ) / "routing.jsonl"
        self.resume = resume
        self._done: dict[str, dict] = self._load_checkpoint() if resume else {}
        self.report = PipelineReport()

    # -- checkpoint ---------------------------------------------------------

    def _load_checkpoint(self) -> dict[str, dict]:
        if not self.checkpoint_path.exists():
            return {}
        done: dict[str, dict] = {}
        for line in self.checkpoint_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # torn tail line from a crash — safe to ignore
            if not isinstance(rec, dict):
                continue  # malformed record — safe to ignore
            if rec.get("schema_version") != CHECKPOINT_VERSION:
                continue  # old-format rows (pre-Phase-4 calibration) must not mix
            row = rec.get("row")
            if not isinstance(row, dict):
                continue
            if row.get("message_id") and not self._row_problems(row):
                done[row["message_id"]] = row
        return done

    def _append_checkpoint(self, row: dict) -> None:
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        rec = {"schema_version": CHECKPOINT_VERSION, "row": row}
        with self.checkpoint_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # -- validation -----------------------------------------------------------

    def _row_problems(self, row: dict) -> list[str]:
        problems = validate_output(row)
        evidence = str(row.get("evidence_message_ids") or "none")
        if evidence != "none":
            for eid in evidence.split(";"):
                if eid not in self.history_ids:
                    problems.append(f"evidence id {eid!r} not in history")
        return problems

    # -- per-row ---------------------------------------------------------------

    def _fallback_row(self, msg: Message, why: str) -> dict:
        """Conservative clearly-flagged row: a message_id is never dropped."""
        return RoutingOutput(
            message_id=msg.message_id,
            action=FALLBACK_ACTION,
            message_type=FALLBACK_TYPE,
            reason=f"Fallback (pipeline: {why}): conservative default, no decision",
            confidence=FALLBACK_CONFIDENCE,
            evidence_message_ids="none",
        ).as_row_dict()

    def _route_one(self, msg: Message) -> tuple[dict, str | None]:
        """Route + gate + validate one message. Returns (row, gate_rule|None)."""
        media_text = _media_text(self.ds, msg)

        # Agent: never raises (per-row fallback inside); belt-and-braces wrap
        # for the gate/validation so no hard error can drop a row.
        try:
            out = self.agent.route(msg)
            sig = analyze_signals(self.ds, msg, media_text=media_text)
            final = apply_gate(self.ds, msg, out, media_text=media_text)
            row = final.as_row_dict()
            problems = self._row_problems(row)
            if problems:
                raise ValueError("; ".join(problems))
        except Exception as exc:  # noqa: BLE001 — per-row isolation (winner #6)
            self.report.fallbacks.append(msg.message_id)
            row = self._fallback_row(msg, f"{type(exc).__name__}: {exc}")
            problems = self._row_problems(row)
            if problems:  # fallback row must be valid; last resort below
                row = self._fallback_row(msg, f"unfixable: {'; '.join(problems)}")
            return row, None

        rule = next((r for r in GATE_RULES if getattr(sig, r)), None)
        if rule is not None and final != out:
            self.report.gate_flips[rule] = self.report.gate_flips.get(rule, 0) + 1
        return row, rule

    # -- run ---------------------------------------------------------------------

    def run(
        self,
        messages: list[Message] | None = None,
        limit: int = 0,
        progress=None,
    ) -> PipelineReport:
        """Route `messages` (default: all incoming) with checkpoint/resume.

        `progress(row_dict, rule, i, n)` is called after each message.
        """
        msgs = messages if messages is not None else self.ds.incoming
        if limit:
            msgs = msgs[:limit]

        t0 = time.time()
        for i, msg in enumerate(msgs, 1):
            if msg.message_id in self._done:
                self.report.resumed += 1
                if progress:
                    progress(self._done[msg.message_id], None, i, len(msgs))
                continue
            row, rule = self._route_one(msg)
            self._done[msg.message_id] = row
            self._append_checkpoint(row)
            self.report.routed += 1
            if progress:
                progress(row, rule, i, len(msgs))

        self.report.total = len(msgs)
        self.report.wall_seconds = time.time() - t0
        return self.report

    # -- output -------------------------------------------------------------------

    def write_voted_output(
        self, passes: list[dict[str, dict]], path: str | Path
    ) -> int:
        """Majority-vote N fresh passes (action+type per message), then
        validate and write output.csv (Phase 5 robustness). Each pass maps
        message_id -> gate-applied row; the voted row is a real row from one
        pass (never a blend), so every field stays contract-valid.
        """
        if not passes:
            raise ValueError("write_voted_output needs at least one pass")
        mids = [m.message_id for m in self.ds.incoming]
        for mid in mids:
            for p in passes:
                if mid not in p:
                    raise ValueError(f"pass missing row for {mid}")
        self._done = {
            mid: majority_vote([p[mid] for p in passes]) for mid in mids
        }
        return self.write_output(path)

    def write_output(self, path: str | Path) -> int:
        """Write the validated output.csv (exact columns, one row per message).

        Returns the number of rows written. Raises if any incoming message is
        missing or any row fails the contract — never writes a broken file.
        """
        out_path = Path(path)
        missing = [
            m.message_id
            for m in self.ds.incoming
            if m.message_id not in self._done
        ]
        if missing:
            raise RuntimeError(f"no prediction for: {missing[:5]} ...")
        problems: list[str] = []
        for mid in (m.message_id for m in self.ds.incoming):
            problems.extend(
                f"{mid}: {p}" for p in self._row_problems(self._done[mid])
            )
        if problems:
            raise RuntimeError(f"output would be invalid: {problems[:5]} ...")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh, quoting=csv.QUOTE_MINIMAL)
            writer.writerow(routing_fields())
            for m in self.ds.incoming:
                row = self._done[m.message_id]
                writer.writerow(
                    [
                        row["message_id"],
                        row["action"],
                        row["message_type"],
                        row["reason"],
                        f"{float(row['confidence']):.2f}",
                        row["evidence_message_ids"],
                    ]
                )
        return len(self.ds.incoming)
