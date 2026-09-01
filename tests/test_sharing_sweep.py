"""The entity-sharing sweep must be a FAIR test, and these lock that.

`src/models/sharing_sensitivity.py` answers an audit's question: our three
legitimate merchants are three POINTS on what is really a continuum, so what
happens between "shares nothing" and "shares everything"? The sweep builds
legitimate merchants at every sharing level and runs them through the frozen
pipeline.

A sweep like that is easy to fake without meaning to - make the legitimate
merchants a little too clean, or the control a little too obvious, and it will
report whatever you hoped for. These tests guard the CONSTRUCTION so the
result means something:

  - the swept legitimate merchants carry no fraud at all, so any firing is a
    false alarm by definition and not a lucky label
  - sharing is the ONLY thing that varies with `share`; amounts, account ages
    and instruments come from the ordinary customer pool at every level
  - the shared entity really is shared, in the proportion asked for
  - the positive control really is an attack, because a sweep whose control
    cannot fire measures the harness rather than the system (failure-log 24)
  - swept entity ids never collide with the main world's, so the sweep cannot
    contaminate the merchants the headline numbers are computed on

They deliberately do NOT assert what the detector decides. That is the
measurement, and it is reported by the module itself across five seeds.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.sharing_sensitivity import (SHARES, N_EVENT, _sweep_world,  # noqa: E402
                                            build_merchant)
from src.sim import simulator as sim  # noqa: E402


@pytest.fixture(scope="module")
def built():
    """Every swept merchant for one seed, plus the control."""
    w, rng = _sweep_world(7), sim.RNG(7 + 9000)
    out = {}
    for family in ("kiosk", "corporate"):
        for share in SHARES:
            label = "sw_%s_%03d" % (family, int(share * 100))
            out[label] = build_merchant(w, rng, label, share, family)
    out["sw_control_farm"] = build_merchant(w, rng, "sw_control_farm", 1.0,
                                            "kiosk", attack=True)
    return out


def _event(rows):
    return [r for r in rows if r["scenario"].startswith(("sweep_kiosk",
                                                         "sweep_corporate",
                                                         "sweep_control"))]


def test_swept_legitimate_merchants_contain_no_fraud(built):
    """Any spike on these is a false alarm by construction, not a near-miss."""
    for label, rows in built.items():
        if label == "sw_control_farm":
            continue
        assert all(r["is_fraud"] == 0 for r in rows), f"{label} carries fraud"


def test_control_is_actually_an_attack(built):
    """A sweep whose positive control is not an attack proves nothing."""
    ev = _event(built["sw_control_farm"])
    assert ev, "control has no event traffic"
    assert all(r["is_fraud"] == 1 for r in ev)
    assert len({r["device_id"] for r in ev}) == 1, "farm should share ONE device"
    assert len({r["customer_id"] for r in ev}) > 10, "a farm needs many accounts"


def test_sharing_is_the_only_thing_that_varies(built):
    """The shared fraction is honoured, and 0% really shares nothing."""
    for family in ("kiosk", "corporate"):
        for share in SHARES:
            ev = _event(built["sw_%s_%03d" % (family, int(share * 100))])
            assert len(ev) == N_EVENT
            top_ip = max({r["ip"] for r in ev},
                         key=lambda x: sum(1 for r in ev if r["ip"] == x))
            n_on_top = sum(1 for r in ev if r["ip"] == top_ip)
            if share == 0.0:
                # No hub: with 1500 pools and 180 txns, collisions are incidental.
                assert n_on_top < 0.25 * N_EVENT, f"{family} 0% has an IP hub"
            else:
                assert n_on_top >= 0.8 * round(N_EVENT * share), (
                    f"{family} {share:.0%}: only {n_on_top} txns on the shared IP")


def test_kiosk_shares_a_device_and_corporate_does_not(built):
    """The two families must differ in the way the docstring claims."""
    kiosk = _event(built["sw_kiosk_100"])
    corp = _event(built["sw_corporate_100"])
    assert len({r["device_id"] for r in kiosk}) == 1, "kiosk should share ONE device"
    assert len({r["device_id"] for r in corp}) > 10, "corporate shares IP only"


def test_swept_customers_use_their_own_instruments(built):
    """A kiosk queue is many people's own cards through one terminal. If the
    sweep gave them shared instruments it would be building a farm and calling
    it legitimate."""
    ev = _event(built["sw_kiosk_100"])
    assert len({r["instrument_id"] for r in ev}) > 10


def test_swept_accounts_are_ordinary_aged_accounts(built):
    """Account age must not become an accidental second variable - that is the
    leak failure-log 21 was about."""
    for label, rows in built.items():
        if label == "sw_control_farm":
            continue
        ages = [r["customer_created_day"] for r in _event(rows)]
        assert min(ages) < 0, f"{label} has accounts created on the event day"


def test_sweep_entities_never_collide_with_the_main_world():
    """The sweep must not contaminate the merchants the headline is computed on."""
    w, rng = _sweep_world(7), sim.RNG(7 + 9000)
    rows = build_merchant(w, rng, "sw_kiosk_100", 1.0, "kiosk")
    main = sim.generate(seed=7)
    for col in ("customer_id", "device_id", "ip", "instrument_id"):
        overlap = {r[col] for r in rows} & set(main[col])
        assert not overlap, f"{col} collides with the main world: {list(overlap)[:3]}"


def test_swept_merchant_ids_are_distinct_from_the_main_world():
    w, rng = _sweep_world(7), sim.RNG(7 + 9000)
    rows = build_merchant(w, rng, "sw_kiosk_060", 0.6, "kiosk")
    assert {r["merchant_id"] for r in rows}.isdisjoint(set(sim.generate(seed=7).merchant_id))
