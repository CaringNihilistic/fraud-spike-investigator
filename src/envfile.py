"""Load a local .env into os.environ, if one exists.

Twelve lines instead of a python-dotenv dependency: we need exactly one
behaviour (KEY=value lines, comments, optional quotes) and the rest of this
repo already refuses dependencies it does not need - the dashboard vendors
React rather than pulling npm.

Two rules that matter:
  - A real environment variable ALWAYS wins over the file. Otherwise a stale
    .env silently overrides the key you just exported, and you debug the wrong
    credential for twenty minutes.
  - Never log a value. The whole point of the file is that the secret does not
    appear in a terminal, a screen recording, or a chat window.

.env is gitignored (along with .env.*), so a key pasted there cannot be
committed by accident.
"""
from __future__ import annotations

import os
from pathlib import Path

ENV_PATH = Path(__file__).resolve().parents[1] / ".env"


def load_env(path: Path | None = None) -> list[str]:
    """Read .env into os.environ. Returns the NAMES it set (never values)."""
    p = path or ENV_PATH
    if not p.exists():
        return []
    loaded = []
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        val = val.strip().strip('"').strip("'")
        # An exported variable is the operator's live intent; the file is a
        # default. Never let the file overwrite it.
        if key and val and key not in os.environ:
            os.environ[key] = val
            loaded.append(key)
    return loaded
