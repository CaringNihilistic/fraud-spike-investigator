"""Tests for the .env loader.

Small surface, but two of these are correctness claims worth pinning:
a stale .env must never override an exported variable (or you spend twenty
minutes debugging the wrong credential), and the loader must never return a
VALUE - only names - so a secret cannot reach a log or a screen recording.
"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.envfile import load_env

FAKE = "sk-ant-api03-FAKE-FOR-TESTS"


@pytest.fixture
def envfile(tmp_path):
    return tmp_path / ".env"


@pytest.fixture(autouse=True)
def clean_env():
    """Never let a test leak into the real process environment."""
    saved = {k: os.environ.get(k) for k in ("ANTHROPIC_API_KEY", "FSI_TEST_VAR")}
    for k in saved:
        os.environ.pop(k, None)
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def test_loads_key_and_strips_comments_and_quotes(envfile):
    envfile.write_text('# a comment\n\nANTHROPIC_API_KEY=%s\nFSI_TEST_VAR="quoted"\n' % FAKE,
                       encoding="utf-8")
    names = load_env(envfile)
    assert set(names) == {"ANTHROPIC_API_KEY", "FSI_TEST_VAR"}
    assert os.environ["ANTHROPIC_API_KEY"] == FAKE
    assert os.environ["FSI_TEST_VAR"] == "quoted"


def test_an_exported_variable_always_beats_the_file(envfile):
    """The file is a default; an exported var is the operator's live intent."""
    os.environ["ANTHROPIC_API_KEY"] = "sk-ant-FROM-SHELL"
    envfile.write_text("ANTHROPIC_API_KEY=%s\n" % FAKE, encoding="utf-8")
    assert load_env(envfile) == []
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-FROM-SHELL"


def test_a_blank_value_behaves_as_absent(envfile):
    """An unfilled .env must not set an empty key - that would look like
    credentials to the agent and fail at call time instead of falling back."""
    envfile.write_text("ANTHROPIC_API_KEY=\n", encoding="utf-8")
    assert load_env(envfile) == []
    assert "ANTHROPIC_API_KEY" not in os.environ


def test_missing_file_is_not_an_error(tmp_path):
    """No .env is the normal case - tests and CI run without one."""
    assert load_env(tmp_path / "nope") == []


def test_loader_never_returns_a_secret_value(envfile):
    """Return NAMES only. Anything printed by a caller must be safe."""
    envfile.write_text("ANTHROPIC_API_KEY=%s\n" % FAKE, encoding="utf-8")
    assert FAKE not in "".join(load_env(envfile))


def test_export_prefix_is_tolerated(envfile):
    """People paste `export KEY=...` out of habit from shell docs."""
    envfile.write_text("export ANTHROPIC_API_KEY=%s\n" % FAKE, encoding="utf-8")
    load_env(envfile)
    assert os.environ["ANTHROPIC_API_KEY"] == FAKE
