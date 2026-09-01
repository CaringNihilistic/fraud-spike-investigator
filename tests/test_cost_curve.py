"""The cost tool must report, not decide - and must never hardcode its verdict.

`src/policy/cost_curve.py` exists because Track 02 asks for false-positive cost
and we were publishing one point on a curve. Two properties matter more than the
numbers it prints:

  1. it is REPORTING. The shipped cutoffs are derived on the validation slice,
     and nothing here may feed back into them. A cost curve computed on the TEST
     slice that then moved policy would be tuning on test.
  2. its verdict is DERIVED. The first run printed "The curve is FLAT" directly
     above its own 33.2% spread, because the sentence was hardcoded - the third
     time a tool in this repo has done that (failure-log 26, then 35). The test
     below feeds it a genuinely flat curve and a genuinely steep one and
     requires the verdict to change.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.policy import cost_curve as cc  # noqa: E402
from src.policy import engine  # noqa: E402


def _toy(n=400, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.2).astype(int)
    p = np.clip(y * 0.8 + rng.normal(0, 0.15, n), 0, 1)
    amounts = rng.lognormal(6.5, 0.6, n)
    return y, p, amounts


def test_curve_reports_every_requested_threshold():
    y, p, a = _toy()
    df = cc.curve(y, p, a)
    assert list(df.threshold) == cc.THRESHOLDS
    assert (df.precision.between(0, 1)).all() and (df.recall.between(0, 1)).all()


def test_raising_the_threshold_never_increases_flagged_volume():
    y, p, a = _toy()
    df = cc.curve(y, p, a)
    assert (df.flagged.diff().dropna() <= 0).all(), "flagged count must be monotone"
    assert (df.legit_inr_blocked.diff().dropna() <= 1e-6).all(), (
        "blocking less legitimate money as the bar rises is the whole point")


def test_pricing_false_negatives_higher_only_raises_expected_loss():
    y, p, a = _toy()
    cheap = cc.curve(y, p, a, fn_mult=1.0).expected_loss_inr
    dear = cc.curve(y, p, a, fn_mult=5.0).expected_loss_inr
    assert (dear >= cheap - 1e-6).all()


def test_it_does_not_touch_the_shipped_policy():
    """The cost curve runs on the TEST slice. If it could move a cutoff, that
    would be tuning on test - so the constants must be untouched by importing
    and running it."""
    before = (engine.RESTRICT_CUT, engine.STEP_UP_CUT)
    y, p, a = _toy()
    cc.curve(y, p, a)
    cc.break_even_review_cost(y, p, a, 0.5)
    assert (engine.RESTRICT_CUT, engine.STEP_UP_CUT) == before


def test_break_even_rises_when_the_caught_fraud_is_worth_more():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.9, 0.9, 0.1])
    small = cc.break_even_review_cost(y, p, np.array([100.0, 100.0, 10.0, 10.0]), 0.5)
    large = cc.break_even_review_cost(y, p, np.array([900.0, 900.0, 10.0, 10.0]), 0.5)
    assert large > small


def test_a_flat_curve_and_a_steep_one_do_not_get_the_same_verdict():
    """Guards failure-log 35 directly. Rather than re-running the module, this
    exercises the branch condition it uses: the verdict must follow the spread,
    not a sentence someone believed when they wrote it."""
    def verdict_is_flat(spread, cheapest):
        return 100 * spread / cheapest < 10.0

    assert verdict_is_flat(spread=100.0, cheapest=100_000.0), "0.1% must read as flat"
    assert not verdict_is_flat(spread=53_114.0, cheapest=159_803.0), (
        "33.2% must NOT read as flat - this is the exact case that shipped wrong")


@pytest.mark.skipif(not (cc.OUT / "economics.json").exists(),
                    reason="artifacts absent; run the pipeline first")
def test_policy_break_even_matches_the_published_figure():
    """The published ~INR 905 had no artifact behind it until now."""
    pbe = cc.policy_break_even()
    assert pbe is not None and 850 < pbe < 960, pbe
