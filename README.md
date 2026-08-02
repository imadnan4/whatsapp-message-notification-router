# WhatsApp Message Notification Router

[![CI](https://github.com/imadnan4/whatsapp-message-notification-router/actions/workflows/ci.yml/badge.svg)](https://github.com/imadnan4/whatsapp-message-notification-router/actions/workflows/ci.yml)

An AI-powered system that routes incoming WhatsApp messages - text, image
posters, and voice notes - into three actions:

| Action | Meaning |
| --- | --- |
| `notify` | Interrupt the user now (important, time-sensitive, action required) |
| `digest` | Useful but can wait for a later summary |
| `mute` | Low value: repetitive, unwanted, opted-out, spam, scam, or unsafe |

Every decision is **personalized per user** - the same offer message can be
`mute` for a user who opted out of promotions and `digest` for one who shops
there - and **safety is enforced in code**: scam attempts (OTP phishing,
account-expiry pressure, prompt injection) are always muted, no matter what
the message says.

Built as a 24-hour hackathon submission for **HackerRank Orchestrate
(August 2026)**.

---

## Highlights

- **Single agent, bounded tool loop** - at most 4 tool rounds per message
  (`inspect_evidence`, `lookup_user_context`, `submit_routing`), hard caps in
  code, one backstop call.
- **Deterministic safety gate after the model** - pure rules the model cannot
  override: scam/injection → `mute` hard block, opt-out → `mute`,
  repeated-ignored sender → `mute`, quiet hours → `digest`.
- **Evidence-grounded routing** - each row cites the historical message(s)
  that justify the decision, found by a canonical pattern-clustering
  retrieval (the earliest instance of the message's recurring pattern), with
  a semantic-embedding fallback for voice notes (whose text field is empty).
- **Calibrated confidence** - model confidence is shrunk toward per-action
  golden means (measured MAE **0.022** on the labeled samples).
- **Multimodal** - images are read directly by a vision-capable LLM; voice
  notes are transcribed locally (faster-whisper), both disk-cached.
- **Engineered for reliability** - checkpoint/resume, per-row fallback (one
  failure never drops a `message_id`), provider fallback (OpenAI primary /
  DeepSeek backup), majority voting across runs for the final output.

---

## Architecture

```text
dataset/*.csv + media/
        |
        v
core/data_loader.py      deterministic load + joins + features (no LLM)
core/media.py            image -> vision LLM read; voice -> faster-whisper
core/retrieval.py        evidence candidates: structured scope + TF-IDF +
                         canonical pattern-clustering
core/semantic.py         embedding fallback for voice / empty-text queries
        |
        v
core/agent.py            single agent, bounded tool loop (max 4 rounds)
                         - inspect_evidence / lookup_user_context / submit_routing
        |
        v
core/schema.py           allowed values, three-tier matching, confidence
                         calibration, output validation
        |
        v
core/safety_gate.py      DETERMINISTIC post-model gate (model cannot override)
                         scam/injection -> mute; opt-out -> mute;
                         repeated-ignored -> mute; quiet hours -> digest
        |
        v
output.csv               exactly 110 rows, exact column order
```

Design principle: **the model only looks and decides; deterministic Python
enforces the rules.** Prompt instructions can be argued away by adversarial
content; code cannot.

---

## Setup

Requirements: Python 3.14, [uv](https://docs.astral.sh/uv/).

```bash
cd code

# 1. Install dependencies
#    (use the Tsinghua mirror if the default CDN is slow on your network:
#     UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple uv sync)
uv sync

# 2. Create .env from the template and add your OpenAI API key
cp .env.example .env
#    OPENAI_API_KEY=sk-...   (never commit this file)

# 3. Route the full dataset (writes ../dataset/output.csv)
uv run python main.py

# 4. (Recommended for the final output) majority-vote 3 fresh passes -
#    the model's temperature is provider-locked, so a few marginal rows vary
#    between runs; voting stabilizes them (~$0.10 per pass):
uv run python main.py --majority 3

# 5. Evaluate against the 30 solved samples (needs the same .env):
uv run python evaluation/run_samples.py --gate

# 6. Offline tests (no API calls, no key needed):
uv run pytest
```

Other flags: `--no-resume` (ignore checkpoints), `--limit N` (dev mode, no
output.csv), `--provider deepseek` (fallback provider), `--dataset-dir` /
`--output` (paths).

---

## Output contract

`output.csv` has exactly these columns and exactly one row per `message_id`
in `dataset/messages.csv`:

```text
message_id,action,message_type,reason,confidence,evidence_message_ids
```

- `action` - `notify` | `digest` | `mute`
- `message_type` - `personal` | `urgent` | `event` | `payment` |
  `business_update` | `promotion` | `greeting` | `forward` | `spam` |
  `scam` | `unknown`
- `confidence` - 0-1, calibrated (measured MAE 0.022 vs the golden samples)
- `evidence_message_ids` - `id1;id2` from `message_history.csv`, or `none`
- `reason` - one specific sentence naming the deciding pattern for THIS user

The output is validated before writing: a missing row, invalid value, or
nonexistent evidence id raises instead of writing a broken file.

---

## Measured results

On the 30 solved samples (+ safety gate), across runs of the final system
(95% confidence intervals via cluster bootstrap,
`evaluation/compare_runs.py`):

| Field | Result | 95% CI | Voted (3-run majority) |
| --- | --- | --- | --- |
| action | 30/30 (100%) | 100-100% | 100% |
| message_type | 29-30/30 (97.8%) | 94.4-100% | 100% |
| evidence hit | 28/30 (93.3%) | 83.3-100% | 93.3% |
| reason token F1 | 0.23-0.24 | - | - |
| confidence MAE | 0.022 | - | 100% within 0.1 |

Full 110-message run (3-pass majority): 110/110 valid rows, **0 fallbacks**,
all 22 scam-signal rows muted, confidence band 0.82-0.89, ~$0.31 total.

---

## Project structure

```text
.
├── code/
│   ├── main.py                  # terminal entry point (--majority N)
│   ├── core/
│   │   ├── agent.py             # single agent, bounded tool loop
│   │   ├── data_loader.py       # CSV loading + features
│   │   ├── media.py             # vision reads + voice transcripts (cached)
│   │   ├── retrieval.py         # evidence retrieval + canonical clustering
│   │   ├── semantic.py          # embedding fallback (voice queries)
│   │   ├── prompts.py           # policy prompt (message text is DATA)
│   │   ├── schema.py            # allowed values, calibration, validation
│   │   ├── safety_gate.py       # deterministic post-model rules
│   │   ├── pipeline.py          # checkpoint/resume, fallback, voting
│   │   └── providers/           # OpenAI (responses API) + DeepSeek fallback
│   ├── evaluation/
│   │   ├── run_samples.py       # per-field accuracy vs the 30 samples
│   │   └── compare_runs.py      # multi-run comparison + bootstrap CIs
│   └── tests/                   # 92 offline tests (no API calls)
└── dataset/                     # input CSVs + media + output.csv
```

---

## Cost and runtime

Measured on `gpt-5.6-luna` ($0.20 / $1.20 per 1M tokens):

| Run | Cost | Wall time | Calls | Fallbacks |
| --- | --- | --- | --- | --- |
| 30-sample eval + gate | ~$0.028 | ~110 s | ~40 | 0 |
| Full 110-row run | ~$0.10 | ~7 min | ~150 | 0 |
| Majority run (3 passes) | ~$0.31 | ~20 min | ~430 | 0 |

Media reads and voice transcripts are cached under `.cache/` (gitignored);
the semantic embedding index builds once on the first run (~$0.001).
Checkpoint/resume means an interrupted run only re-does unfinished rows.

---

## Environment variables

| Variable | Required | Source |
| --- | --- | --- |
| `OPENAI_API_KEY` | yes (primary provider) | `.env` in the repo root (gitignored) |
| `DEEPSEEK_API_KEY` | no (fallback provider only) | `.env` in the repo root |
| `MNR_CACHE_DIR` | no | `.env` or environment (default `<repo>/.cache/`) |

Secrets are read from the environment only, never from code. Copy
`.env.example` to `.env` and fill in your keys; `.env` is gitignored.
