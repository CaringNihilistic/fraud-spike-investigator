"""The checker that stops a stale headline from shipping - checked itself.

`src/audit/verify_submission.py` closes the process gap failure-log 31 left
open: nothing verified documented English against the artifact it came from.
A checker is only worth having if it actually fires, so these test the LOGIC
against hand-built inputs rather than against today's docs - today's docs pass,
which proves nothing about whether the thing works.

The real regression test is the one that cannot live in pytest: the checker was
pointed at this repository's own docs from commit 5ae3a4a, before failure-31's
corrections, and flagged 10 problems including the retracted "0 in every world"
headline and the calibration figures an external audit had found by hand.

Artifact-dependent checks skip when artifacts_out is absent, because the suite
must run on a fresh clone with no pipeline run and no network.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.audit.verify_submission import (ART, Claim, Retired, check_claims,  # noqa: E402
                                         check_retired)

FILES = ("README.md",)


def _claim(**kw):
    base = dict(name="npv", pattern=r"value ₹([\d.]+)L", expected=8.13,
                tol=0.006, files=FILES)
    base.update(kw)
    return Claim(**base)


def test_flags_a_number_that_disagrees_with_its_artifact():
    texts = {"README.md": "| Net protected value ₹9.99L |"}
    problems, seen, exempt = check_claims(texts, [_claim()])
    assert problems and "9.99" in problems[0]
    assert not exempt


def test_accepts_a_number_that_agrees():
    texts = {"README.md": "| Net protected value ₹8.13L |"}
    problems, seen, exempt = check_claims(texts, [_claim()])
    assert not problems
    assert seen == [("npv", 1)]


def test_a_current_number_on_a_history_line_is_verified_not_skipped():
    """The exemption must apply to MISMATCHES only. A line may discuss the past
    and still state a current figure, and those are the figures most worth
    checking - an earlier draft skipped them and under-counted its own work."""
    texts = {"README.md": "Before the hard negatives this moved; value ₹8.13L today."}
    problems, seen, exempt = check_claims(texts, [_claim()])
    assert not problems
    assert seen == [("npv", 1)], "a matching figure must count as verified"
    assert not exempt


def test_a_superseded_number_is_excused_only_when_the_line_says_so():
    # The real shape: the current figure is stated somewhere, and an older one
    # survives on a line that openly marks itself as history.
    marked = {"README.md": "Net protected value ₹8.13L today.\n"
                           "Cost, as measured at the time: value ₹10.57L."}
    problems, seen, exempt = check_claims(marked, [_claim()])
    assert not problems
    assert seen == [("npv", 1)], "the current figure still counts as verified"
    assert len(exempt) == 1 and "10.57" in exempt[0]


def test_history_only_mention_still_reports_the_current_claim_as_missing():
    """If the ONLY occurrence is historical, the current figure is claimed
    nowhere - which is a real fault, not a pass."""
    only_history = {"README.md": "Cost, as measured at the time: value ₹10.57L."}
    problems, seen, exempt = check_claims(only_history, [_claim()])
    assert any("not found" in p for p in problems)
    assert len(exempt) == 1 and seen == [("npv", 0)]

    unmarked = {"README.md": "Net protected value ₹10.57L."}
    problems, _, exempt = check_claims(unmarked, [_claim()])
    assert problems, "an unlabelled stale number must fail"
    assert not exempt


def test_a_rotted_pattern_is_itself_reported():
    """If a claim stops matching anything, the checker has gone blind and must
    say so rather than passing quietly."""
    texts = {"README.md": "nothing resembling the claim here"}
    problems, seen, _ = check_claims(texts, [_claim()])
    assert any("not found" in p for p in problems)
    assert seen == [("npv", 0)]


def test_catches_the_retracted_headline_in_its_real_table_form():
    """The claim shipped as '| **False alarms** | **0** in every world |', with
    the words in separate table cells. The first pattern required them adjacent
    and silently matched nothing - found only by running against the real
    pre-fix document."""
    r = Retired("zero false alarms",
                r"\*{0,2}0\*{0,2}\s*(?:false alarms?\s*)?in every world",
                "retracted", files=FILES)
    texts = {"README.md": "| **False alarms** | **0** in every world — including a sale |"}
    problems, exempt = check_retired(texts, [r])
    assert problems and "zero false alarms" in problems[0]
    assert not exempt


def test_retired_phrasing_is_allowed_when_the_line_marks_itself_history():
    r = Retired("independent worlds", r"independent (simulated )?worlds", "why",
                files=FILES, exempt=("not five independent",))
    texts = {"README.md": "These are five seeds, not five independent worlds."}
    problems, exempt = check_retired(texts, [r])
    assert not problems
    assert len(exempt) == 1


@pytest.mark.skipif(not (ART / "threshold_sweep_decision.json").exists(),
                    reason="artifacts absent; run the pipeline first")
def test_shipped_cutoffs_match_the_decision_that_adopted_them():
    """Failure-log 31 in one assertion: engine.py shipped STEP_UP_CUT = 20.0
    while the validation sweep had adopted 25. No documentation check could
    have seen it."""
    from src.audit.verify_submission import check_code
    assert check_code() == []


@pytest.mark.skipif(not (ART / "economics.json").exists(),
                    reason="artifacts absent; run the pipeline first")
def test_the_real_docs_currently_pass():
    from src.audit.verify_submission import build_claims, build_retired
    texts = {f: Path(f).read_text(encoding="utf-8")
             for f in ("README.md", "SUBMISSION.md")}
    claim_problems, _, _ = check_claims(texts, build_claims())
    retired_problems, _ = check_retired(texts, build_retired())
    assert not claim_problems + retired_problems, (
        "docs disagree with artifacts:\n" + "\n".join(claim_problems + retired_problems))
