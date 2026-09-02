"""The growth negatives are only worth running if age cannot be the answer.

`src/models/growth_negatives.py` asks whether a legitimate signup wave fires the
detector. The easy way to build that test is to give the surge aged, obviously
legitimate accounts - it would pass, and it would prove only that the model can
use the 21 features that are not `customer_age_days`.

The version worth running holds account age DEGENERATE between the surge and the
card-testing control, so discrimination has to come from entity structure and the
amount profile instead. That matters specifically because `customer_age_days` has
a documented history of being a label proxy (failure-log 21).

These tests assert the degeneracy rather than trusting it, and they assert the
surge really is entity-free. They deliberately say nothing about whether the
detector fires - that is the measurement, and its criteria were pre-registered in
the module docstring and committed before the first run.
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.growth_negatives import (N_EVENT, _world, cold_start,  # noqa: E402
                                         marketing_surge, positive_control,
                                         warm_twin)
from src.sim import simulator as sim  # noqa: E402


def _event(rows, tag):
    return [r for r in rows if r["scenario"] == tag]


@pytest.fixture(scope="module")
def built():
    w = _world(7)
    return {
        "surge": _event(marketing_surge(w, sim.RNG(7 + 13000), "gn_a"),
                        "gn_marketing_surge"),
        "control": _event(positive_control(w, sim.RNG(7 + 13000), "gn_b"),
                          "gn_control_cardtest"),
        "cold": _event(cold_start(w, sim.RNG(7 + 13000), "gn_c"), "gn_cold_start"),
        "warm": _event(warm_twin(w, sim.RNG(7 + 13000), "gn_d"), "gn_warm_twin"),
    }


def test_account_age_is_degenerate_between_surge_and_card_testing(built):
    """The whole point of the experiment. If the surge's accounts are visibly
    older than the attack's, a pass proves nothing."""
    a = [r["customer_created_day"] for r in built["surge"]]
    b = [r["customer_created_day"] for r in built["control"]]
    ma, mb = statistics.median(a), statistics.median(b)
    assert abs(ma - mb) < 30, (
        f"median created_day differs by {abs(ma - mb):.0f} days "
        f"(surge {ma:.0f} vs card testing {mb:.0f}) - age would be doing the work")
    # and the young tail, which is the part the proxy actually keyed on
    young = lambda xs: sum(1 for x in xs if x > EVENT_DAY_FLOOR) / len(xs)  # noqa: E731
    assert abs(young(a) - young(b)) < 0.25, "young-account share differs too much"


EVENT_DAY_FLOOR = 0   # created_day > 0 means the account was made during the world


def test_the_surge_shares_no_entities_at_all(built):
    """It is a growth test, not a second entity test. Every newcomer must have
    their own device, IP and card - otherwise a firing result is ambiguous."""
    rows = built["surge"]
    for col in ("device_id", "ip", "instrument_id"):
        vals = [r[col] for r in rows]
        top = max(set(vals), key=vals.count)
        assert vals.count(top) <= 3, (
            f"{col} is shared by {vals.count(top)} transactions - this is "
            f"supposed to be the NON-sharing negative")


def test_the_surge_looks_like_card_testing_where_it_should(built):
    """Same volume, same compressed window, one novel instrument per row. If it
    differed on those too, we would not be isolating anything."""
    s, c = built["surge"], built["control"]
    assert len(s) == len(c) == N_EVENT
    span = lambda rows: max(r["ts"] for r in rows) - min(r["ts"] for r in rows)  # noqa: E731
    assert abs(span(s) - span(c)) < 3600
    assert len({r["instrument_id"] for r in s}) > 0.8 * N_EVENT, "instruments must be novel"


def test_the_surge_is_entirely_legitimate(built):
    assert all(r["is_fraud"] == 0 for r in built["surge"])
    assert all(r["is_fraud"] == 1 for r in built["control"])


def test_cold_start_has_no_history_and_its_twin_does(built):
    """The cold-start comparison is only interpretable if the ONLY difference is
    the presence of prior traffic."""
    cold, warm = built["cold"], built["warm"]
    assert len(cold) == len(warm), "opening days must be the same size"
    w = _world(7)
    cold_all = cold_start(w, sim.RNG(7 + 13000), "gn_c")
    warm_all = warm_twin(w, sim.RNG(7 + 13000), "gn_d")
    assert len(cold_all) == len(cold), "cold-start merchant must have NO history"
    assert len(warm_all) > len(warm), "warm twin must carry history"


def test_growth_entities_cannot_collide_with_the_training_world():
    w = _world(7)
    rows = marketing_surge(w, sim.RNG(7 + 13000), "gn_a")
    main = sim.generate(seed=7)
    for col in ("customer_id", "device_id", "ip", "instrument_id"):
        assert not ({r[col] for r in rows} & set(main[col])), f"{col} collides"
