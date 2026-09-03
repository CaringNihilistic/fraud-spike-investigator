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


def test_status_reports_real_pipeline_state_not_placeholders(client):
    assert client.get("/api/health").json()["ok"] is True
    s = client.get("/api/status").json()
    assert {"processed", "total", "speed_tps", "events"} <= set(s)


def test_merchant_overview_exposes_every_field_the_dashboard_reads(client):
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


def test_entity_graph_hides_one_account_entities_as_noise(client):
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


# --- the merchant signature -------------------------------------------------
# top_cause comes from the LLM, so on any instance running without one (the
# hosted demo does) the card said "under attack" and nothing said what the
# attack looked like. The signature is COUNTED server-side instead, so it is
# always present - and it must never become a second channel for ground truth.

def test_signature_is_present_without_any_llm(client):
    for i in range(6):
        STATE.record_txn(
            {"merchant_id": "farm", "ts": 1 + i, "p": 0.9, "amount": 100.0,
             "customer_id": f"c{i}", "device_id": "d_SHARED",
             "ip": f"ip{i}", "instrument_id": f"pi{i}"},
            risk=90.0, confidence=0.8, action=Action.RESTRICT, reason="t",
            spiking=False, spike_ts=None)
    m = client.get("/api/merchants").json()["merchants"]
    farm = next(x for x in m if x["merchant_id"] == "farm")
    assert farm["top_cause"] is None            # no agent ran
    assert farm["signature"]["hubs"] == 1       # ...and we still say what we see
    assert farm["signature"]["accounts"] == 6
    assert "1 device shared by 6 accounts" in farm["signature"]["text"]


def test_signature_never_leaks_the_scenario_label(client):
    """The generator's `scenario` is the answer key. It reaches the dashboard
    through no path, including this one."""
    for i in range(4):
        STATE.record_txn(
            {"merchant_id": "leaky", "ts": 1 + i, "p": 0.95, "amount": 50.0,
             "customer_id": f"c{i}", "device_id": "d_FARM_F", "ip": "ip_CLUSTER_I",
             "instrument_id": "pi_STOLEN_9", "scenario": "device_farm",
             "is_fraud": 1},
            risk=99.0, confidence=0.9, action=Action.RESTRICT, reason="t",
            spiking=True, spike_ts=1)
    sig = next(x for x in client.get("/api/merchants").json()["merchants"]
               if x["merchant_id"] == "leaky")["signature"]
    for banned in ("device_farm", "card_testing", "ip_cluster", "fraud_ring",
                   "account_takeover", "scenario", "is_fraud"):
        assert banned not in str(sig).lower()


def test_a_spiking_merchant_with_no_shared_entities_says_so_distinctly(client):
    """Account takeover shares nothing by construction. That absence must not
    render identically to a quiet merchant's - it is the finding, not a blank.
    See failure-log 16."""
    for i in range(5):                                  # spiking, no sharing
        STATE.record_txn(
            {"merchant_id": "ato", "ts": 1 + i, "p": 0.95, "amount": 900.0,
             "customer_id": f"a{i}", "device_id": f"d{i}", "ip": f"ip{i}",
             "instrument_id": f"pi{i}"},
            risk=95.0, confidence=0.9, action=Action.RESTRICT, reason="t",
            spiking=True, spike_ts=1)
    for i in range(5):                                  # quiet, no sharing
        STATE.record_txn(
            {"merchant_id": "calm", "ts": 1 + i, "p": 0.95, "amount": 20.0,
             "customer_id": f"b{i}", "device_id": f"e{i}", "ip": f"jp{i}",
             "instrument_id": f"qi{i}"},
            risk=95.0, confidence=0.9, action=Action.REVIEW, reason="t",
            spiking=False, spike_ts=None)
    ms = {x["merchant_id"]: x for x in client.get("/api/merchants").json()["merchants"]}
    assert ms["ato"]["signature"]["text"] != ms["calm"]["signature"]["text"]
    assert ms["ato"]["signature"]["kind"] == "no_sharing_but_spiking"


def test_a_two_account_coincidence_is_not_reported_as_a_hub(client):
    """Two accounts behind one IP is a consumer NAT, not a cluster. Reporting
    it as "1 IP address shared by 2 accounts" reads identically to a real
    40-account cluster and buries the actual finding."""
    for i in range(6):
        STATE.record_txn(
            {"merchant_id": "nat", "ts": 1 + i, "p": 0.95, "amount": 100.0,
             "customer_id": f"n{i % 2}",            # only TWO accounts
             "device_id": f"nd{i}", "ip": "ip_NAT", "instrument_id": f"np{i}"},
            risk=95.0, confidence=0.9, action=Action.REVIEW, reason="t",
            spiking=False, spike_ts=None)
    sig = next(x for x in client.get("/api/merchants").json()["merchants"]
               if x["merchant_id"] == "nat")["signature"]
    assert sig["hubs"] == 0
    assert "shared by 2 accounts" not in sig["text"]


# --- queue ordering is a policy decision, not a display detail ---------------
# An analyst runs out of time before the queue runs out of cases, so the sort
# key decides how much fraud value a human actually sees. See failure-log 28
# and src/policy/queue_order.py for what arrival order cost.

def test_queue_is_ranked_by_expected_loss_not_arrival(client):
    """A cheap near-certain case must not outrank an expensive likely one."""
    # small amount, very high probability -> low expected loss
    STATE.record_txn(
        {"merchant_id": "q", "ts": 1, "p": 0.99, "amount": 100.0,
         "customer_id": "c1", "device_id": "d1", "ip": "i1", "instrument_id": "p1"},
        risk=99.0, confidence=0.9, action=Action.RESTRICT, reason="t",
        spiking=False, spike_ts=None)
    # large amount, lower probability -> HIGHER expected loss, arrives later
    STATE.record_txn(
        {"merchant_id": "q", "ts": 2, "p": 0.60, "amount": 90_000.0,
         "customer_id": "c2", "device_id": "d2", "ip": "i2", "instrument_id": "p2"},
        risk=88.0, confidence=0.9, action=Action.RESTRICT, reason="t",
        spiking=False, spike_ts=None)

    body = client.get("/api/review-queue").json()
    assert body["ordering"] == "expected_loss_desc"
    cases = [c for c in body["cases"] if c["merchant_id"] == "q"]
    assert cases[0]["amount_inr"] == 90_000.0          # the expensive one leads
    assert cases[0]["expected_loss_inr"] == 54_000.0   # 90,000 x 0.60
    assert cases[1]["expected_loss_inr"] == 99.0       # 100 x 0.99


def test_the_wire_cap_keeps_the_top_cases_not_the_newest(client):
    """The list is capped at 200 for the wire. Capping by arrival would hide
    exactly the cases the ranking exists to surface."""
    for i in range(230):
        STATE.record_txn(
            {"merchant_id": "cap", "ts": 1 + i, "p": 0.9, "amount": float(i + 1),
             "customer_id": f"c{i}", "device_id": f"d{i}", "ip": f"i{i}",
             "instrument_id": f"p{i}"},
            risk=95.0, confidence=0.9, action=Action.REVIEW, reason="t",
            spiking=False, spike_ts=None)
    cases = client.get("/api/review-queue").json()["cases"]
    assert len(cases) == 200
    top = [c for c in cases if c["merchant_id"] == "cap"][0]
    assert top["amount_inr"] == 230.0    # the largest, not the most recent


# ---------------------------------------------------------------------------
# STATE.reset() - what makes the hosted replay loop possible.
#
# The hosted demo runs on a free instance that sleeps after 15 minutes, so
# every visitor used to trigger a cold boot and therefore see a LIVE replay:
# the 86-second wait was the only thing making the demo look live. Keeping the
# instance warm removes the wait and would otherwise leave every visitor
# looking at a FINISHED replay. Hence the loop, and hence this.
# ---------------------------------------------------------------------------

def test_reset_clears_everything_the_replay_accumulated():
    _seed_txn(mid="m1", action=Action.RESTRICT, spiking=True)
    _seed_txn(mid="m2", action=Action.REVIEW)
    STATE.finished = True
    assert STATE.merchants and STATE.review_queue and STATE.processed

    STATE.reset()

    assert STATE.merchants == {}, "a stale merchant would carry a spike into the next pass"
    assert STATE.review_queue == []
    assert STATE.processed == 0
    assert STATE.finished is False
    assert STATE.prevented_inr == 0.0 and STATE.impacted_inr == 0.0
    assert STATE.review_cases == 0
    assert STATE._next_case_id == 1, "case ids must restart or they climb forever"


def test_reset_keeps_the_viewers_own_controls_and_the_slice_size():
    """speed and paused belong to whoever is watching; yanking the slider back
    mid-view is a bug, not a reset. `total` describes the slice, not the pass -
    clearing it makes the progress bar read 0/0."""
    STATE.total = 14160
    STATE.speed = 725.0
    STATE.paused = True

    STATE.reset()

    assert STATE.speed == 725.0
    assert STATE.paused is True
    assert STATE.total == 14160


def test_reset_lets_a_second_pass_produce_the_same_numbers():
    """The point of the loop: pass two must look like pass one, not like pass
    one plus pass two."""
    _seed_txn(mid="m1", amount=5000.0, p=0.9, action=Action.RESTRICT)
    first = (STATE.processed, STATE.prevented_inr, len(STATE.review_queue))
    STATE.reset()
    _seed_txn(mid="m1", amount=5000.0, p=0.9, action=Action.RESTRICT)
    assert (STATE.processed, STATE.prevented_inr, len(STATE.review_queue)) == first


def test_reset_does_not_replace_the_lock():
    """Callers hold a reference to it; swapping it would silently unguard
    every writer that captured the old one."""
    lock = STATE._lock
    STATE.reset()
    assert STATE._lock is lock


def test_a_hosted_deployment_loops_without_anyone_setting_a_flag():
    """The bug this rule exists for: --loop was put in render.yaml, Render
    redeployed the CODE but not the blueprint's start command, and the live
    service served one finished pass. Polling /api/status showed 14160/14160
    at 92s when a restart was due at 42s. Looping is a property of being
    hosted, so derive it from --defer-prepare rather than from a flag someone
    has to remember in a dashboard."""
    from run_demo import should_loop
    assert should_loop(loop=False, no_loop=False, defer_prepare=True), (
        "a hosted deployment must loop even when nobody passed --loop")


def test_a_local_run_still_stops_at_the_end_of_the_slice():
    from run_demo import should_loop
    assert not should_loop(loop=False, no_loop=False, defer_prepare=False)


def test_loop_can_be_forced_on_locally_and_off_when_hosted():
    from run_demo import should_loop
    assert should_loop(loop=True, no_loop=False, defer_prepare=False)
    assert not should_loop(loop=False, no_loop=True, defer_prepare=True), (
        "--no-loop must win, or there is no way back to a single pass")
    assert not should_loop(loop=True, no_loop=True, defer_prepare=True)
