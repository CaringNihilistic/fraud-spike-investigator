"""The topology experiment is only valid if topology is the ONLY thing that moved.

`src/models/topology_generalisation.py` claims that detection survives a change
of attack shape. That claim rests entirely on the known and unseen variants
being matched in every other respect - volume, burst window, amounts, ageing
and fraud prevalence. If the unseen variant were quietly smaller, shorter or
cheaper, a difference in detection would measure that instead.

These guard the CONSTRUCTION. They deliberately assert nothing about whether
the detector fires; that is the measurement, and the module reports it across
five seeds with the known variants as a positive control.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.topology_generalisation import KINDS, _world, attack  # noqa: E402
from src.sim import simulator as sim  # noqa: E402


@pytest.fixture(scope="module")
def variants():
    out = {}
    for kind in KINDS:
        for v in ("known", "unseen"):
            rng = sim.RNG(7 + 11000)          # same stream for both sides
            out[(kind, v)] = attack(_world(7), rng, f"tg_{kind[:4]}_{v[:3]}", kind, v)
    return out


@pytest.mark.parametrize("kind", KINDS)
def test_both_variants_carry_the_same_transaction_volume(variants, kind):
    k, u = variants[(kind, "known")], variants[(kind, "unseen")]
    assert len(k) == len(u), f"{kind}: {len(k)} known vs {len(u)} unseen txns"


@pytest.mark.parametrize("kind", KINDS)
def test_both_variants_occupy_the_same_burst_window(variants, kind):
    k, u = variants[(kind, "known")], variants[(kind, "unseen")]
    span = lambda rows: max(r["ts"] for r in rows) - min(r["ts"] for r in rows)  # noqa: E731
    assert abs(span(k) - span(u)) < 3600, (
        f"{kind}: burst spans differ by more than an hour - a detection "
        f"difference could be timing rather than topology")


@pytest.mark.parametrize("kind", KINDS)
def test_both_variants_are_entirely_fraudulent(variants, kind):
    for v in ("known", "unseen"):
        rows = variants[(kind, v)]
        assert rows and all(r["is_fraud"] == 1 for r in rows), f"{kind}/{v}"


@pytest.mark.parametrize("kind", KINDS)
def test_amounts_are_drawn_the_same_way(variants, kind):
    """Not a distribution test - just enough to catch one variant being made
    systematically cheaper or dearer than its pair."""
    med = lambda rows: sorted(r["amount"] for r in rows)[len(rows) // 2]  # noqa: E731
    k, u = med(variants[(kind, "known")]), med(variants[(kind, "unseen")])
    assert 0.4 < k / u < 2.5, f"{kind}: median amount {k:.0f} vs {u:.0f}"


@pytest.mark.parametrize("kind", KINDS)
def test_the_topology_actually_differs(variants, kind):
    """The experiment is pointless if the 'unseen' graph is the same graph."""
    def shape(rows):
        return (len({r["device_id"] for r in rows}),
                len({r["ip"] for r in rows}),
                len({r["customer_id"] for r in rows}))
    k, u = shape(variants[(kind, "known")]), shape(variants[(kind, "unseen")])
    assert k != u, f"{kind}: identical entity shape {k} - nothing was varied"


def test_unseen_variants_are_not_uniformly_easier():
    """A sanity check on the DESIGN rather than the result: the unseen shapes
    should mostly REDUCE entity concentration, because that is the signal the
    detector leans on. If every unseen variant increased sharing, the
    experiment would be testing an easier problem and a survival result would
    mean nothing."""
    reduced = 0
    for kind in KINDS:
        rng = sim.RNG(7 + 11000)
        k = attack(_world(7), rng, "a", kind, "known")
        rng = sim.RNG(7 + 11000)
        u = attack(_world(7), rng, "a", kind, "unseen")
        per_dev = lambda rows: len(rows) / max(1, len({r["device_id"] for r in rows}))  # noqa: E731
        if per_dev(u) < per_dev(k):
            reduced += 1
    assert reduced >= 3, (
        f"only {reduced}/{len(KINDS)} unseen variants reduced device "
        f"concentration; the suite may be testing an easier problem")


def test_swept_entities_cannot_collide_with_the_training_world():
    rng = sim.RNG(7 + 11000)
    rows = attack(_world(7), rng, "tg_devi_kno", "device_farm", "known")
    main = sim.generate(seed=7)
    for col in ("device_id", "ip", "instrument_id"):
        assert not ({r[col] for r in rows} & set(main[col])), f"{col} collides"
