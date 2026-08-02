"""Phase 0 sanity: core deps importable in the uv environment (no API calls)."""

import importlib

DEPENDENCIES = ["openai", "dotenv", "faster_whisper", "sklearn", "PIL"]


def test_phase0_deps_importable():
    for name in DEPENDENCIES:
        assert importlib.import_module(name) is not None, name
