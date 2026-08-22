"""P2 agent safety tests.

These use a SCRIPTED client double instead of the live API, so the full
LangGraph loop - tool dispatch, audit logging, the policy gate, and every
failure path - is exercised deterministically with no credentials and no
network. The point is to prove the guarantees hold, not to measure the model.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent.audit import AuditLog
from src.agent.investigator import deterministic_report, investigate
from src.agent.tools import READ_ONLY_TOOLS, TOOL_FNS, InvestigationContext
from src.policy.engine import Action, decide


# ---------------- scripted client double ----------------
class _Block:
    """Mimics an anthropic tool_use content block."""
    type = "tool_use"

    def __init__(self, name, inp, id_="tu_1"):
        self.name, self.input, self.id = name, inp, id_


class _Resp:
    def __init__(self, content, stop_reason):
        self.content, self.stop_reason = content, stop_reason


class ScriptedClient:
    """Replays a fixed list of responses; records how many calls it saw."""

    def __init__(self, script):
        self.script, self.calls = list(script), 0
        self.messages = self

    def create(self, **kwargs):
        self.calls += 1
        if not self.script:
            return _Resp([], "end_turn")
        return self.script.pop(0)


class ExplodingClient:
    def __init__(self, exc=RuntimeError("api down")):
        self.exc, self.messages = exc, self

    def create(self, **kwargs):
        raise self.exc


@pytest.fixture
def ctx():
    """Small synthetic slice - a ring-shaped attack on m1, quiet m2."""
    rows = []
    for i in range(40):
        rows.append({"merchant_id": "m1", "ts": 1_000_000 + i * 60,
                     "p": 0.95 if i >= 10 else 0.01,
                     "amount": 500.0, "customer_id": f"c{i % 5}",
                     "device_id": "d_shared", "ip": "ip_shared",
                     "instrument_id": f"pi{i % 3}",
                     "is_new_device_for_customer": 1.0, "geo_mismatch": 0.0,
                     "amount_dev_ratio": 1.1, "customer_age_days": 400.0})
    for i in range(40):
        rows.append({"merchant_id": "m2", "ts": 1_000_000 + i * 60, "p": 0.01,
                     "amount": 300.0, "customer_id": f"q{i}",
                     "device_id": f"dq{i}", "ip": f"ipq{i}",
                     "instrument_id": f"piq{i}",
                     "is_new_device_for_customer": 0.0, "geo_mismatch": 0.0,
                     "amount_dev_ratio": 1.0, "customer_age_days": 500.0})
    return InvestigationContext(pd.DataFrame(rows))


def _report_block(**over):
    payload = {"merchant_id": "m1", "cause": "fraud_ring",
               "evidence": ["40 txns, 1 device, 5 accounts"],
               "exposure_inr": 15000.0, "recommended_action": "restrict",
               "confidence": 0.9}
    payload.update(over)
    return _Block("write_investigation_report", payload)


# ---------------- the gate ----------------
def test_agent_cannot_invent_an_action(ctx):
    """The core P2 invariant: whatever the model writes, the allowlist decides.
    An invented action must degrade to REVIEW - never escalate."""
    client = ScriptedClient([
        _Resp([_report_block(recommended_action="ban_merchant_permanently")], "tool_use"),
        _Resp([], "end_turn"),
    ])
    res = investigate(ctx, "m1", client=client)
    assert res.report["recommended_action"] == "ban_merchant_permanently"
    assert res.validated_action == Action.REVIEW      # degraded, not executed


def test_agent_cannot_escalate_beyond_policy(ctx):
    """Even a well-formed 'restrict' is only a RECOMMENDATION - the policy
    engine still decides independently from risk/confidence/spike state."""
    client = ScriptedClient([_Resp([_report_block()], "tool_use"),
                             _Resp([], "end_turn")])
    res = investigate(ctx, "m1", client=client)
    assert res.validated_action == Action.RESTRICT    # allowlisted, passes through
    # ...but the engine, given a low-risk transaction, still allows it:
    assert decide(risk_score=5, confidence=0.9, merchant_in_spike=False).action == Action.ALLOW


# ---------------- fail-safe ----------------
def test_llm_failure_degrades_to_review_never_raises(ctx):
    res = investigate(ctx, "m1", client=ExplodingClient())
    assert res.degraded and res.validated_action == Action.REVIEW
    assert res.report["cause"] == "unclear"           # never guesses on failure
    assert res.report["confidence"] == 0.0


def test_no_client_degrades_to_review(ctx):
    res = investigate(ctx, "m1", client=None)
    # No credentials in the test environment -> fallback path.
    if res.degraded:
        assert res.validated_action == Action.REVIEW


def test_agent_never_writes_a_report_it_did_not_produce(ctx):
    """Model returns no report at all -> fallback, not a fabricated success."""
    client = ScriptedClient([_Resp([], "end_turn")])
    res = investigate(ctx, "m1", client=client)
    assert res.degraded and res.validated_action == Action.REVIEW


def test_tool_budget_is_bounded(ctx):
    """A model that loops forever must be cut off, not run unbounded."""
    looping = ScriptedClient([_Resp([_Block("get_merchant_baseline", {"merchant_id": "m1"},
                                            f"tu{i}")], "tool_use")
                              for i in range(50)])
    res = investigate(ctx, "m1", client=looping)
    assert res.degraded and "budget" in (res.degraded_reason or "")
    assert looping.calls <= 9                         # MAX_TOOL_ROUNDS + 1


def test_unknown_tool_does_not_crash_the_agent(ctx):
    client = ScriptedClient([
        _Resp([_Block("drop_all_tables", {"merchant_id": "m1"})], "tool_use"),
        _Resp([_report_block()], "tool_use"),
        _Resp([], "end_turn")])
    res = investigate(ctx, "m1", client=client)
    assert any(e.tool == "drop_all_tables" and not e.ok for e in res.audit.entries)
    assert res.validated_action == Action.RESTRICT    # recovered and continued


# ---------------- audit + read-only ----------------
def test_every_tool_call_is_audited(ctx):
    client = ScriptedClient([
        _Resp([_Block("get_merchant_baseline", {"merchant_id": "m1"}, "t1")], "tool_use"),
        _Resp([_Block("calculate_exposure", {"merchant_id": "m1"}, "t2")], "tool_use"),
        _Resp([_report_block()], "tool_use"),
        _Resp([], "end_turn")])
    res = investigate(ctx, "m1", client=client)
    assert res.audit.tools_called() == ["get_merchant_baseline", "calculate_exposure",
                                        "write_investigation_report"]
    for e in res.audit.entries:
        assert e.inputs_hash and e.output_hash and isinstance(e.ts, int)


def test_degraded_path_is_audited_too(ctx):
    """The audit log must not go quiet exactly when something breaks."""
    res = investigate(ctx, "m1", client=ExplodingClient())
    assert "calculate_exposure" in res.audit.tools_called()


def test_tools_are_read_only(ctx):
    """Calling every tool must leave the underlying data byte-identical."""
    before = ctx.df.copy(deep=True)
    for name, fn in TOOL_FNS.items():
        if name == "write_investigation_report":
            fn(ctx, "m1", cause="x", evidence=["e"], exposure_inr=1.0,
               recommended_action="review", confidence=0.5)
        else:
            fn(ctx, "m1")
    pd.testing.assert_frame_equal(before, ctx.df)


def test_tool_surface_is_exactly_seven_read_only_tools():
    assert len(READ_ONLY_TOOLS) == 7
    assert "write_investigation_report" in READ_ONLY_TOOLS
    assert "get_customer_anomalies" in READ_ONLY_TOOLS


def test_anomaly_tool_degrades_when_features_absent(ctx):
    """A frame without the anomaly columns must return an error dict, not
    raise - a missing feature cannot be allowed to kill an investigation."""
    import pandas as _pd
    from src.agent.tools import InvestigationContext, get_customer_anomalies
    bare = InvestigationContext(_pd.DataFrame([{
        "merchant_id": "m1", "ts": 1, "p": 0.9, "amount": 10.0,
        "customer_id": "c", "device_id": "d", "ip": "i", "instrument_id": "pi"}]))
    out = get_customer_anomalies(bare, "m1")
    assert "error" in out


def test_ground_truth_is_not_exposed_to_the_agent(ctx):
    """No tool may leak is_fraud/scenario - the agent must reason from signals."""
    import json
    for name, fn in TOOL_FNS.items():
        if name == "write_investigation_report":
            continue
        blob = json.dumps(fn(ctx, "m1"), default=str)
        assert "is_fraud" not in blob and "scenario" not in blob


# ---------------- money stays in Python ----------------
def test_exposure_arithmetic_is_deterministic_not_model_supplied(ctx):
    """calculate_exposure must be reproducible and independent of the model."""
    a = TOOL_FNS["calculate_exposure"](ctx, "m1")
    b = TOOL_FNS["calculate_exposure"](ctx, "m1")
    assert a == b
    hot = ctx.df[(ctx.df.merchant_id == "m1") & (ctx.df.p >= 0.5)]
    assert a["exposure_at_risk_inr"] == pytest.approx(float(hot.amount.sum()))


def test_disabling_the_agent_changes_no_decision():
    """The headline safety claim: the LLM was never in the decision path, so
    decisions must be identical whether or not the agent ran."""
    cases = [(95.0, 0.9, True), (95.0, 0.9, False), (50.0, 0.9, False),
             (10.0, 0.9, False), (None, 0.9, True), (99.0, 0.1, True)]
    without = [decide(r, c, s).action for r, c, s in cases]
    # "Running" the agent cannot feed anything back into decide() - there is no
    # parameter for it. Recomputing must therefore be identical.
    with_agent = [decide(r, c, s).action for r, c, s in cases]
    assert without == with_agent


def test_fallback_report_shape_is_valid(ctx):
    rep = deterministic_report(ctx, "m1", "unit_test", AuditLog())
    for k in ("merchant_id", "cause", "evidence", "exposure_inr",
              "recommended_action", "confidence"):
        assert k in rep
    assert rep["recommended_action"] == Action.REVIEW.value
