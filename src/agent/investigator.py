"""LangGraph investigator: Claude Haiku + 6 read-only tools.

Graph shape (deliberately small - a 3-node loop, not a sprawling DAG):

    call_model ──tool_use──> run_tools ──> call_model ...
         │
         └──report or budget exhausted──> finalize ──> END

Guarantees this module is responsible for, in priority order:
  1. The agent CANNOT decide. `recommended_action` goes through
     policy.validate_recommendation(); anything outside the frozen allowlist
     degrades to REVIEW. It can never escalate to RESTRICT by writing a word.
  2. The agent CANNOT block anyone by failing. Any error - no credentials,
     timeout, API error, malformed output, tool-budget exhaustion - produces
     a deterministic templated report routed to human REVIEW.
  3. Every tool call is audit-logged (tool, inputs hash, output hash, ts).
  4. Tools are read-only by construction (see tools.py).

Model: Claude Haiku 4.5, as specified in CLAUDE.md - this is a bounded,
tool-driven summarization task with a deterministic safety net underneath it,
which is exactly the shape a small fast model handles well.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import TypedDict

from langgraph.graph import END, StateGraph

from src.agent.audit import AuditLog
from src.agent.tools import TOOL_FNS, TOOL_SCHEMAS, InvestigationContext
from src.policy.engine import Action, validate_recommendation

MODEL = "claude-haiku-4-5"
MAX_TOOL_ROUNDS = 8          # hard ceiling on the agent loop
REQUEST_TIMEOUT_S = 60.0

SYSTEM_PROMPT = """You are a fraud-spike investigator for a payment risk team.

A merchant-level spike detector has fired. Your job is to explain WHY, citing
specific evidence, and recommend a bounded action. You are an analyst writing
for a human reviewer - you do not execute anything.

Method:
1. get_merchant_baseline - is this actually abnormal for THIS merchant?
2. get_entity_network and get_velocity_summary - what shape is it?
3. get_flagged_transactions - concrete evidence to cite.
4. calculate_exposure - ALWAYS use this for rupee figures. Never do money
   arithmetic yourself; copy the number the tool returns.
5. write_investigation_report - finish. Call this exactly once, last.

Diagnosis guide:
- card_testing: many NEW instruments, few devices/IPs, small varied amounts
- device_farm: ONE device across many fresh accounts
- ip_cluster: ONE IP across many accounts, each with its own device
- account_takeover: established customers, NEW device and geo, large amounts
- fraud_ring: several accounts densely sharing several devices/IPs/instruments
- legitimate_traffic: volume is up but the flagged RATE is flat and entities
  are diverse (one device and instrument per customer). A sale is not an attack.

Recommended action must be exactly one of: allow, step_up, review, restrict.
Prefer `review` when the evidence is ambiguous or the signal is weak - a human
will see it. Do not recommend `restrict` without concentrated shared-entity
evidence. If you are unsure, say so and set a low confidence: escalating an
uncertain case to a human is correct behaviour, not a failure.
"""


class AgentState(TypedDict):
    merchant_id: str
    messages: list
    report: dict | None
    rounds: int
    error: str | None
    # MUST be declared here. LangGraph discards any key a node returns that is
    # not in the state schema, so an undeclared stop_reason silently vanished
    # and every investigation fell through to the fallback path - which looks
    # exactly like "the model failed" rather than "the graph is misrouting".
    stop_reason: str | None


@dataclass
class InvestigationResult:
    merchant_id: str
    report: dict
    validated_action: Action
    audit: AuditLog
    degraded: bool                  # True = deterministic fallback, not the LLM
    degraded_reason: str | None = None
    tools_called: list = field(default_factory=list)

    def to_record(self) -> dict:
        return {"merchant_id": self.merchant_id,
                "cause": self.report.get("cause"),
                "evidence_count": len(self.report.get("evidence", [])),
                "exposure_inr": self.report.get("exposure_inr"),
                "recommended_action": self.report.get("recommended_action"),
                "validated_action": self.validated_action.value,
                "confidence": self.report.get("confidence"),
                "degraded": self.degraded,
                "degraded_reason": self.degraded_reason,
                "tools_called": ",".join(self.tools_called)}


# --------------------------------------------------------------- fallback
def deterministic_report(ctx: InvestigationContext, merchant_id: str,
                         reason: str, audit: AuditLog | None = None) -> dict:
    """The templated report used whenever the LLM path is unavailable.

    Built from the SAME read-only tools, so it is never empty or speculative -
    it states the measured facts and routes to a human. This is what makes an
    LLM outage a degradation rather than an outage of the product.

    Its tool calls are audit-logged too: "every tool call is recorded" has to
    hold on the degraded path as well, or the audit log silently under-reports
    exactly when something has gone wrong."""
    def _call(name, fn):
        out = fn(ctx, merchant_id)
        if audit is not None:
            audit.record(name, {"merchant_id": merchant_id}, out)
        return out

    from src.agent.tools import (calculate_exposure, get_entity_network,
                                 get_merchant_baseline)
    base = _call("get_merchant_baseline", get_merchant_baseline)
    net = _call("get_entity_network", get_entity_network)
    exp = _call("calculate_exposure", calculate_exposure)
    evidence = [
        f"flagged rate {base.get('recent_flagged_rate')} recently vs "
        f"{base.get('baseline_flagged_rate')} baseline "
        f"({base.get('flagged_rate_multiple')}x)",
        f"{net.get('flagged_count', 0)} flagged txns across "
        f"{net.get('distinct_customers', 0)} customers, "
        f"{net.get('distinct_devices', 0)} devices, {net.get('distinct_ips', 0)} IPs",
        f"exposure at risk INR {exp.get('exposure_at_risk_inr')}",
    ]
    return {
        "merchant_id": merchant_id,
        "cause": "unclear",           # never guess a cause without the analyst layer
        "evidence": evidence,
        "exposure_inr": exp.get("exposure_at_risk_inr", 0.0),
        "recommended_action": Action.REVIEW.value,   # always the safe direction
        "confidence": 0.0,
        "degraded": True,
        "degraded_reason": reason,
    }


# --------------------------------------------------------------- graph
def _anthropic_client():
    """Return a client, or None if no credentials are resolvable.

    Returning None (rather than raising) is deliberate: 'no credentials' is a
    normal operating condition for this system, not an exceptional one."""
    try:
        import anthropic
    except ImportError:
        return None
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        # The SDK can also resolve an `ant auth login` profile; try constructing
        # and let a missing credential surface as an auth error at call time.
        try:
            return anthropic.Anthropic(timeout=REQUEST_TIMEOUT_S)
        except Exception:
            return None
    try:
        return anthropic.Anthropic(timeout=REQUEST_TIMEOUT_S)
    except Exception:
        return None


def build_graph(ctx: InvestigationContext, audit: AuditLog, client):
    """Compile the LangGraph state machine for one investigation."""

    def call_model(state: AgentState) -> dict:
        if state["rounds"] >= MAX_TOOL_ROUNDS:
            return {"error": f"tool budget exhausted after {MAX_TOOL_ROUNDS} rounds"}
        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=2048,
                system=SYSTEM_PROMPT,
                tools=TOOL_SCHEMAS,
                messages=state["messages"],
            )
        except Exception as e:                      # any API failure -> degrade
            return {"error": f"{type(e).__name__}: {e}"}
        return {"messages": state["messages"] + [{"role": "assistant",
                                                   "content": resp.content}],
                "rounds": state["rounds"] + 1,
                "stop_reason": resp.stop_reason}

    def run_tools(state: AgentState) -> dict:
        last = state["messages"][-1]["content"]
        results, report = [], state.get("report")
        for block in last:
            if getattr(block, "type", None) != "tool_use":
                continue
            name, args = block.name, dict(block.input or {})
            args.pop("ctx", None)                    # tools take ctx positionally
            fn = TOOL_FNS.get(name)
            if fn is None:                           # unknown tool -> refuse, don't crash
                out, ok, err = {"error": f"unknown tool {name}"}, False, "unknown_tool"
            else:
                try:
                    out, ok, err = fn(ctx, **args), True, None
                except Exception as e:
                    out, ok, err = {"error": str(e)}, False, type(e).__name__
            audit.record(name, args, out, ok=ok, error=err)
            if name == "write_investigation_report" and ok:
                report = out
            results.append({"type": "tool_result", "tool_use_id": block.id,
                            "content": json.dumps(out, default=str)})
        return {"messages": state["messages"] + [{"role": "user", "content": results}],
                "report": report}

    def route(state: AgentState) -> str:
        if state.get("error"):
            return "finalize"
        if state.get("report"):                      # report written -> done
            return "finalize"
        if state.get("stop_reason") == "tool_use":
            return "run_tools"
        return "finalize"

    def finalize(state: AgentState) -> dict:
        return {}

    g = StateGraph(AgentState)
    g.add_node("call_model", call_model)
    g.add_node("run_tools", run_tools)
    g.add_node("finalize", finalize)
    g.set_entry_point("call_model")
    g.add_conditional_edges("call_model", route,
                            {"run_tools": "run_tools", "finalize": "finalize"})
    g.add_edge("run_tools", "call_model")
    g.add_edge("finalize", END)
    return g.compile()


def investigate(ctx: InvestigationContext, merchant_id: str,
                client=None) -> InvestigationResult:
    """Run one investigation. NEVER raises - a failure degrades to a
    deterministic report routed to human review."""
    audit = AuditLog()
    client = client or _anthropic_client()

    if client is None:
        rep = deterministic_report(ctx, merchant_id, "no_llm_client_available", audit)
        return InvestigationResult(merchant_id, rep, Action.REVIEW, audit,
                                   degraded=True,
                                   degraded_reason="no_llm_client_available",
                                   tools_called=audit.tools_called())

    try:
        graph = build_graph(ctx, audit, client)
        final = graph.invoke({
            "merchant_id": merchant_id,
            "messages": [{"role": "user",
                          "content": f"The spike detector fired for merchant "
                                     f"{merchant_id}. Investigate and file a report."}],
            "report": None, "rounds": 0, "error": None, "stop_reason": None,
        })
    except Exception as e:                            # graph-level failure
        rep = deterministic_report(ctx, merchant_id, f"graph_error:{type(e).__name__}", audit)
        return InvestigationResult(merchant_id, rep, Action.REVIEW, audit,
                                   degraded=True,
                                   degraded_reason=f"graph_error:{type(e).__name__}",
                                   tools_called=audit.tools_called())

    report, err = final.get("report"), final.get("error")
    if not report:
        reason = err or "no_report_written"
        rep = deterministic_report(ctx, merchant_id, reason, audit)
        return InvestigationResult(merchant_id, rep, Action.REVIEW, audit,
                                   degraded=True, degraded_reason=reason,
                                   tools_called=audit.tools_called())

    # THE GATE: whatever the model wrote, the allowlist decides what it means.
    validated = validate_recommendation(report.get("recommended_action", ""))
    return InvestigationResult(merchant_id, report, validated, audit,
                               degraded=False, tools_called=audit.tools_called())
