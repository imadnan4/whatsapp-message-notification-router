# Message Notification Router - HackerRank Orchestrate (August 2026)

An AI-powered WhatsApp notification router. For every incoming multimodal message
(text, image poster/screenshot, voice note) the system decides `notify` /
`digest` / `mute`, with `message_type`, a one-sentence `reason`, a calibrated
`confidence`, and `evidence_message_ids` citing the user's own history that
justifies the decision - personalized per user, and hard-blocking scam / prompt-
injection attempts regardless of what the message says.

**Architecture at a glance**:

```text
dataset/*.csv + media/
        │
        ▼
core/data_loader.py        deterministic load + joins + features (no LLM)
core/media.py              image → gpt-5.6-luna vision read; voice → faster-whisper
core/retrieval.py          evidence candidates: structured scope + TF-IDF +
                           canonical pattern-clustering (earliest instance first)
        ▼
core/agent.py              single agent, bounded tool loop (max 4 rounds)
                           · inspect_evidence / lookup_user_context / submit_routing
        ▼
core/schema.py             allowed values, three-tier matching, confidence calibration
        ▼
core/safety_gate.py        DETERMINISTIC post-model gate (model cannot override):
                           scam/injection → mute hard block, opt-out/repeated-ignored
                           → mute, quiet hours → digest
        ▼
output.csv                 exactly 110 rows, exact column order
```

Design principle (from the June 2026 #1 winner, same multimodal shape): **the model
only looks and decides; deterministic Python enforces the rules.** Prompt
instructions can be argued away by adversarial content; code cannot.

---

## Setup

Requirements: Python 3.14, [uv](https://docs.astral.sh/uv/).

```bash
cd code

# 1. Install dependencies (use the Tsinghua mirror if the default CDN is slow:
#    UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple uv sync)
uv sync

# 2. Create .env from the template and put your OpenAI API key in it
cp .env.example .env
#    OPENAI_API_KEY=sk-...   (never commit this file; .gitignore covers it)

# 3. Run the router over the full dataset (writes ../dataset/output.csv)
uv run python main.py

# 4. (Recommended for the final submission) majority-vote 3 fresh passes -
#    model temperature is forced to default, so a few marginal rows vary
#    between runs; voting stabilizes them (~$0.10 per pass):
uv run python main.py --majority 3

# 5. Evaluate against the 30 solved samples (needs the same .env):
uv run python evaluation/run_samples.py --gate

# 6. Offline tests (no API calls, no key needed):
uv run pytest
```

Other useful flags: `--no-resume` (ignore checkpoints, clean run),
`--limit N` (dev mode, no output.csv), `--provider deepseek` (fallback provider),
`--dataset-dir / --output` paths.

---

## Output contract

`output.csv` has exactly these columns and exactly one row per `message_id` in
`dataset/messages.csv`:

```text
message_id,action,message_type,reason,confidence,evidence_message_ids
```

- `action` ∈ `notify | digest | mute`
- `message_type` ∈ `personal | urgent | event | payment | business_update |
  promotion | greeting | forward | spam | scam | unknown`
- `confidence` ∈ 0-1, calibrated (measured MAE 0.022 vs the 30 golden samples)
- `evidence_message_ids` = `id1;id2` from `message_history.csv`, or `none`
- `reason` is one specific sentence naming the deciding pattern for THIS user

The output is validated before writing: any missing row, invalid value, or
nonexistent evidence id raises instead of writing a broken file.

---

## Environment variables

| Variable | Required | Source |
| --- | --- | --- |
| `OPENAI_API_KEY` | yes (primary provider) | repo `.env` (gitignored) |
| `DEEPSEEK_API_KEY` | no (fallback provider only) | repo `.env` |
| `MNR_CACHE_DIR` | no | override media/checkpoint cache dir (default repo `.cache/`) |

Secrets are read from the environment only - never from code. The repo `.env`
is gitignored; `.env.example` ships with placeholders.

---

## Cost and runtime (measured, gpt-5.6-luna)

| Run | Cost | Wall time | Calls | Fallbacks |
| --- | --- | --- | --- | --- |
| 30-sample eval + gate | ~$0.028 | ~110 s | ~40 | 0 |
| Full 110-row run | ~$0.103 | ~7 min | ~150 | 0 |
| Final majority run (3 passes, measured 2026-08-02) | $0.3103 | 19.5 min (389s/399s/380s) | 436 | 0 |

Model: `gpt-5.6-luna` ($0.20 / $1.20 per 1M tokens). Media reads and voice
transcripts are cached under `.cache/` (gitignored) - re-runs never re-pay for
them. Checkpoint/resume means an interrupted run only re-does unfinished rows.

---

## Module map

| File | Job |
| --- | --- |
| `main.py` | terminal entry point; `--majority N` votes N fresh passes |
| `core/data_loader.py` | all CSVs → clean structured load + features |
| `core/media.py` | vision reads (gpt-5.6-luna) + faster-whisper transcripts, disk-cached |
| `core/retrieval.py` | evidence candidates: scope + TF-IDF + canonical clustering |
| `core/semantic.py` | optional embedding fallback for voice/empty-text queries (cached) |
| `core/agent.py` | single agent, bounded tool loop, per-row isolation |
| `core/prompts.py` | policy prompt (message text is DATA, never instructions) |
| `core/schema.py` | allowed values, three-tier matching, confidence calibration |
| `core/safety_gate.py` | deterministic scam/opt-out/repeat/quiet-hours rules |
| `core/pipeline.py` | checkpoint/resume, per-row fallback, output validation, voting |
| `core/providers/` | OpenAI (responses API for tools) + DeepSeek fallback, one interface |
| `evaluation/run_samples.py` | per-field accuracy vs the 30 solved samples |
| `evaluation/compare_runs.py` | multi-run comparison + bootstrap 95% CIs (single-call and voted) |
| `tests/` | 88 offline tests (no API calls) |

---

## Measured results (final, 2026-08-02)

On the 30 solved samples (+ gate), across runs (3 runs of the final system,
95% CIs via cluster bootstrap - see `evaluation/compare_runs.py`):

| Field | Baseline (Phase 4) | Final | 95% CI (single call) | Voted |
| --- | --- | --- | --- | --- |
| action | 30/30 | **30/30** | 100-100% | 100% |
| message_type | 29-30/30 | **29-30/30** | 94.4-100% | 100% |
| evidence hit | 70-77% | **93.3%** (28/30) | 83.3-100% | 93.3% |
| reason token F1 | 0.17-0.21 | **0.23-0.24** | - | - |
| confidence MAE | 0.022 | **0.022** | - | 100% within 0.1 |

Evidence residuals (documented, not chased - overfit risk): sample_msg_042 is
a stable model judgment (refuses a cross-conversation business health message
as evidence for a family emergency; the retrieval layer now surfaces it first
via the semantic fallback, and the model still disagrees - defensible);
sample_msg_052 is a golden-label quirk (`none` for a scam that has an exact
history duplicate).

Full 110-row run: exactly 110 valid rows, 0 fallbacks, all adversarial /
injection rows `mute`/`scam`, confidence band 0.82-0.89.

Every material change was measured before/after on the 30 samples (see the
repo-root README, "Measured results") - including changes that were tried
and reverted because they regressed.
