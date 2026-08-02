"""Phase 0 smoke test: verify OpenAI API + gpt-5.6-luna work from this environment.

Usage: uv run python scripts/smoke_api.py
Reads OPENAI_API_KEY from the repo .env (never printed). Prints tokens + cost estimate.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")

MODEL = "gpt-5.6-luna"
# $ per 1M tokens, verified 2026-08-01 (RESEARCH.md §4)
PRICE_IN = 0.20
PRICE_OUT = 1.20


def main() -> int:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        print("ERROR: OPENAI_API_KEY not found in repo .env", file=sys.stderr)
        return 1

    client = OpenAI(api_key=key)
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Reply with exactly: ok"},
            {"role": "user", "content": "Say ok"},
        ],
        max_completion_tokens=16,
    )
    text = (resp.choices[0].message.content or "").strip()
    usage = resp.usage
    cost = (
        usage.prompt_tokens / 1e6 * PRICE_IN
        + usage.completion_tokens / 1e6 * PRICE_OUT
    )
    print(f"model={MODEL} reply={text!r}")
    print(
        f"tokens in={usage.prompt_tokens} out={usage.completion_tokens} "
        f"est_cost=${cost:.6f}"
    )
    return 0 if text == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
