"""Serving-layer tests: the API must not become a second decision-maker.

Uses FastAPI's TestClient against a hand-seeded PipelineState, so these run
in milliseconds with no replay, no model training, and no network.
"""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.policy.engine import ALLOWLIST, Action
from src.serve.api import API_KEY, app
from src.serve.state import STATE, PipelineState


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    """Each test gets a fresh state - the module-level STATE is shared."""
    fresh = PipelineState()
    for attr in ("merchants", "review_queue", "events", "audit"):
        setattr(STATE, attr, getattr(fresh, attr))
    STATE.processed = STATE.total = 0
    STATE._next_case_id = 1
    yield


@pytest.fixture
def client():
    """Authenticated client. Mutating routes require the write key - the
    unauthenticated case is asserted explicitly below, not by accident."""
    return TestClient(app, headers={"X-API-Key": API_KEY})


@pytest.fixture
def anon():
    return TestClient(app)


def _seed_txn(mid="m1", p=0.9, amount=1000.0, action=Action.REVIEW, spiking=False):
    STATE.record_txn(
        {"merchant_id": mid, "ts": 1_760_000_000, "p": p, "amount": amount,
         "customer_id": "c1", "device_id": "d1", "ip": "ip1", "instrument_id": "pi1"},
        risk=90.0, confidence=0.8, action=action, reason="test",
        spiking=spiking, spike_ts=1_760_000_000 if spiking else None,
        baseline_rate=0.01, current_rate=0.5, spike_z=4.2)


def test_health_and_status(client):
    assert client.get("/api/health").json()["ok"] is True
    s = client.get("/api/status").json()
    assert {"processed", "total", "speed_tps", "events"} <= set(s)


def test_merchant_overview_shape(client):
    _seed_txn("m1", spiking=True)
    m = client.get("/api/merchants").json()["merchants"][0]
    for k in ("merchant_id", "risk_score", "exposure_inr", "flagged_count",
              "txn_count", "fraud_rate", "in_spike"):
        assert k in m
    assert m["fraud_rate"]["baseline_rate"] == 0.01
    assert m["in_spike"] is True


def test_unknown_merchant_404s(client):
    assert client.get("/api/merchants/nope/risk").status_code == 404


def test_analyst_cannot_invent_an_action(client):
    """The frozen allowlist binds HUMANS too, not just the LLM. An analyst
    console that accepts arbitrary action strings is the same hole from the
    other side."""
    _seed_txn(action=Action.RESTRICT)
    case_id = client.get("/api/review-queue").json()["cases"][0]["case_id"]
    r = client.post(f"/api/review-queue/{case_id}/decision",
                    json={"action": "delete_merchant_account"})
    assert r.status_code == 400
    for good in sorted(ALLOWLIST):
        assert client.post(f"/api/review-queue/{case_id}/decision",
                           json={"action": good}).status_code == 200


def test_analyst_override_is_recorded_as_an_override(client):
    """An override must be visibly an override - silently accepting it would
    destroy the audit trail the review queue exists to create."""
    _seed_txn(action=Action.RESTRICT)
    case_id = client.get("/api/review-queue").json()["cases"][0]["case_id"]
    out = client.post(f"/api/review-queue/{case_id}/decision",
                      json={"action": "allow", "note": "known good merchant"}).json()
    assert out["system_action"] == "restrict"
    assert out["analyst_action"] == "allow"
    assert out["overridden"] is True


def test_approving_the_system_action_is_not_an_override(client):
    _seed_txn(action=Action.REVIEW)
    case_id = client.get("/api/review-queue").json()["cases"][0]["case_id"]
    out = client.post(f"/api/review-queue/{case_id}/decision",
                      json={"action": "review"}).json()
    assert out["overridden"] is False


def test_only_review_and_restrict_reach_the_queue(client):
    """ALLOW and STEP_UP must not consume analyst time."""
    _seed_txn("m1", action=Action.ALLOW)
    _seed_txn("m2", action=Action.STEP_UP)
    assert client.get("/api/review-queue").json()["cases"] == []
    _seed_txn("m3", action=Action.RESTRICT)
    assert len(client.get("/api/review-queue").json()["cases"]) == 1


def test_score_endpoint_uses_the_real_policy_path(client):
    """POST /transactions must route through fusion + policy, not a shortcut."""
    r = client.post("/api/transactions", json={
        "merchant_id": "m1", "customer_id": "c", "device_id": "d", "ip": "i",
        "instrument_id": "pi", "amount": 100.0, "ts": 1_760_000_000,
        "p_fraud": 0.99, "component_size": 200, "device_account_count": 40,
        "ip_account_count": 40, "instrument_customer_count": 5, "cust_txn_5m": 9}).json()
    assert r["action"] in ALLOWLIST
    assert 0 <= r["risk_score"] <= 100
    assert set(r["components"]) == {"ml", "spike", "graph", "rules"}
    assert r["requires_human"] is True          # high risk always sees a human


def test_score_endpoint_does_not_mutate_replay_state(client):
    """The serving surface must be side-effect free, or a judge poking the API
    during a demo would corrupt the numbers on screen."""
    before = client.get("/api/status").json()["processed"]
    client.post("/api/transactions", json={
        "merchant_id": "m1", "customer_id": "c", "device_id": "d", "ip": "i",
        "instrument_id": "pi", "amount": 100.0, "ts": 1, "p_fraud": 0.99})
    assert client.get("/api/status").json()["processed"] == before
    assert client.get("/api/review-queue").json()["cases"] == []


def test_score_endpoint_rejects_invalid_input(client):
    bad = {"merchant_id": "m1", "customer_id": "c", "device_id": "d", "ip": "i",
           "instrument_id": "pi", "amount": -5.0, "ts": 1, "p_fraud": 0.5}
    assert client.post("/api/transactions", json=bad).status_code == 422
    bad2 = {**bad, "amount": 5.0, "p_fraud": 1.7}
    assert client.post("/api/transactions", json=bad2).status_code == 422


def test_entity_graph_drops_single_account_entities(client):
    """Entities touching one account are the boring case and would swamp the
    picture - legitimate traffic must render as an EMPTY graph."""
    for i in range(6):
        STATE.record_txn(
            {"merchant_id": "legit", "ts": 1 + i, "p": 0.9, "amount": 100.0,
             "customer_id": f"c{i}", "device_id": f"d{i}",     # own device each
             "ip": f"ip{i}", "instrument_id": f"pi{i}"},
            risk=90.0, confidence=0.8, action=Action.REVIEW, reason="t",
            spiking=False, spike_ts=None)
    g = client.get("/api/merchants/legit/entity-graph").json()
    assert g["nodes"] == [] and g["links"] == []


def test_entity_graph_surfaces_a_shared_hub(client):
    for i in range(6):
        STATE.record_txn(
            {"merchant_id": "farm", "ts": 1 + i, "p": 0.9, "amount": 100.0,
             "customer_id": f"c{i}", "device_id": "d_SHARED",   # one device
             "ip": "ip_SHARED", "instrument_id": f"pi{i}"},
            risk=90.0, confidence=0.8, action=Action.REVIEW, reason="t",
            spiking=False, spike_ts=None)
    g = client.get("/api/merchants/farm/entity-graph").json()
    hubs = [n for n in g["nodes"] if n["kind"] == "device"]
    assert hubs and hubs[0]["size"] == 6      # one device, six accounts


def test_investigation_404_before_one_exists(client):
    _seed_txn("m1")
    assert client.get("/api/merchants/m1/investigation").status_code == 404


# ---------------------------------------------------------------- auth gate
def test_write_endpoints_reject_an_unauthenticated_caller(anon):
    """Anything that moves money-affecting state needs the key. A caller who
    can reach the port must not be able to override an analyst."""
    _seed_txn()
    case_id = 1
    writes = [
        ("/api/replay/pause", None),
        ("/api/replay/speed?speed=500", None),
        ("/api/merchants/m1/investigate", None),
        (f"/api/review-queue/{case_id}/decision", {"action": "allow"}),
        ("/api/transactions", {"merchant_id": "m1", "customer_id": "c", "device_id": "d",
                               "ip": "i", "instrument_id": "pi", "amount": 100.0,
                               "ts": 1_760_000_000, "p_fraud": 0.9}),
    ]
    for path, body in writes:
        r = anon.post(path, json=body)
        assert r.status_code == 401, f"{path} answered {r.status_code}, expected 401"


def test_a_wrong_key_is_rejected(anon):
    r = anon.post("/api/replay/pause", headers={"X-API-Key": "not-the-key"})
    assert r.status_code == 401


def test_read_endpoints_stay_open(anon):
    """Views are deliberately unauthenticated - a judge can curl the state."""
    _seed_txn()
    for path in ("/api/health", "/api/status", "/api/merchants",
                 "/api/review-queue", "/api/audit-log"):
        assert anon.get(path).status_code == 200, path


def test_index_hands_the_write_key_to_the_page(anon):
    """The SPA gets the key same-origin so the demo needs no setup step."""
    body = anon.get("/").text
    assert 'name="fsi-key"' in body and API_KEY in body
