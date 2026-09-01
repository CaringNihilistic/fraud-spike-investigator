"""The legitimate merchants that are SUPPOSED to look like attacks.

The flash sale gives every account its own device, IP and instrument, so it
only ever tested "does raw volume fire us" - it never exercised the entity
layer, and m11's entity graph renders empty. These two do exercise it:

  s7_corporate_buyer   40 accounts, ONE office IP, two company cards
                       -> structurally an IP cluster, and legitimate
  s8_shared_kiosk      25 accounts, ONE shared device, own cards, bursty
                       -> structurally a device farm, and legitimate

Both are built from REAL world customers - aged accounts, their own normal
amounts - so entity sharing is the only thing that differs from baseline
traffic. A false alarm on either cannot be blamed on account age or amount.

These tests guard the CONSTRUCTION, not the model's verdict on them. Whether
the detector actually holds is measured by seed_stability.py across five
seeds, and is reported honestly: it holds on four of five (failure-log 29).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.sim.simulator import generate  # noqa: E402


@pytest.fixture(scope="module")
def df():
    return generate()


def test_both_hard_negatives_exist_and_are_legitimate(df):
    for tag in ("s7_corporate_buyer", "s8_shared_kiosk"):
        d = df[df.scenario == tag]
        assert len(d) > 0, f"{tag} missing from the generated world"
        assert d.is_fraud.sum() == 0, f"{tag} must be entirely legitimate"


def test_the_corporate_buyer_really_does_share_an_ip(df):
    """If it did not share, it would not be testing anything."""
    d = df[df.scenario == "s7_corporate_buyer"]
    assert d.ip.nunique() == 1                      # one office
    assert d.customer_id.nunique() >= 20            # across many staff
    assert d.instrument_id.nunique() <= 3           # on company cards
    assert d.device_id.nunique() > 10               # but their own machines


def test_the_kiosk_really_does_share_a_device(df):
    """The device-farm signature, on legitimate traffic."""
    d = df[df.scenario == "s8_shared_kiosk"]
    assert d.device_id.nunique() == 1               # one terminal
    assert d.ip.nunique() == 1
    assert d.customer_id.nunique() >= 15            # many walk-ins
    # each customer pays with their OWN card - this is the only thing
    # separating a kiosk from a device farm, and the model has to use it
    assert d.instrument_id.nunique() == d.customer_id.nunique()


def test_hard_negatives_are_built_from_established_customers(df):
    """Entity sharing must be the ONLY difference from baseline traffic. If
    these accounts were freshly created they would be flagged on age, and the
    experiment would prove nothing about entity correlation."""
    # created_day is negative for accounts that predate the world; attack
    # generators create theirs at or near the attack day. Card testing sits
    # around +13, the device farm around -62; ordinary customers around -199.
    # The hard negatives must sit with the ordinary customers, not the attacks.
    fresh = df[df.scenario == "s1_fraud_spike"].customer_created_day.median()
    for tag in ("s7_corporate_buyer", "s8_shared_kiosk"):
        d = df[df.scenario == tag]
        med = d.customer_created_day.median()
        assert med < -100, f"{tag} accounts are not aged (median {med})"
        assert med < fresh - 100, (
            f"{tag} accounts must be clearly older than attack accounts "
            f"(median {med} vs card testing {fresh})")


def test_the_training_counterpart_is_entity_disjoint_from_the_test_kiosk(df):
    """The model is taught that shared devices can be honest using a DIFFERENT
    merchant on a different day. If the ids overlapped, the held-out kiosk
    would not be held out."""
    hist = df[df.scenario == "s8_hist_kiosk"]
    test = df[df.scenario == "s8_shared_kiosk"]
    assert len(hist) > 0 and len(test) > 0
    assert hist.merchant_id.iloc[0] != test.merchant_id.iloc[0]
    for col in ("device_id", "ip", "customer_id", "instrument_id"):
        assert not (set(hist[col]) & set(test[col])), f"{col} leaks train->test"


def test_the_training_counterpart_lands_before_the_test_period(df):
    """Teaching it must happen in the past, not alongside the held-out case."""
    hist_end = df[df.scenario == "s8_hist_kiosk"].ts.max()
    test_start = df[df.scenario == "s8_shared_kiosk"].ts.min()
    assert hist_end < test_start
