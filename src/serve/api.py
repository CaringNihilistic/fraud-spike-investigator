"""FastAPI serving layer + static SPA host.

Endpoints are thin: they read PipelineState snapshots and return JSON. All
risk logic lives in src/policy and src/spike - nothing here decides anything.

POST /transactions exists so the API is a real serving surface (score one
transaction through the live pipeline), not just a viewer for the replay.
"""
from __future__ import annotations

import os
import secrets
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.agent.tools import InvestigationContext
from src.policy.engine import ALLOWLIST, Action, decide
from src.policy.fusion import RiskSignals, evaluate_rules, fuse_for_policy
from src.serve.state import STATE

STATIC = Path(__file__).parent / "static"

# ------------------------------------------------------------------ auth
# Every route below that CHANGES something - an analyst decision, a replay
# control, an investigation that spends API credits - requires a shared key.
# Read-only views stay open so a judge can curl the state without ceremony.
#
# Be precise about what this is: a single-tenant gate, NOT identity. It stops
# an unauthenticated caller on the network from overriding an analyst decision
# or draining API credits. It does not tell you WHICH analyst acted, and an
# override with no attributable actor is an audit gap - a real deployment
# needs per-analyst identity (SSO/mTLS) so review_queue decisions carry a
# signer. Stated here rather than left for a judge to notice.
#
# If FSI_API_KEY is unset we mint an ephemeral one and hand it to the page at
# load time, so `python run_demo.py` stays one command with no setup.
API_KEY = os.environ.get("FSI_API_KEY") or secrets.token_urlsafe(18)
KEY_FROM_ENV = "FSI_API_KEY" in os.environ


def require_key(x_api_key: str = Header(default="")):
    """Constant-time compare - a timing oracle on a demo key is silly, but
    getting this wrong by habit is how it gets shipped for real."""
    if not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(401, "missing or invalid X-API-Key header")


WRITE = [Depends(require_key)]

app = FastAPI(title="Fraud Spike Investigator",
              description="Merchant-level fraud-spike detection, entity correlation, "
                          "and policy-gated investigation. Defense-only.",
              version="0.3.0")

# The investigation context is attached at startup by run_demo.py so
# /investigate can run against the same scored slice the replay uses.
_CTX: InvestigationContext | None = None

# Whether the LLM investigator is available on THIS instance. The public
# deployment runs with --no-agent, because an Anthropic key on a public host
# lets any visitor spend credits through /investigate.
AGENT_ENABLED: bool = True


def set_agent_enabled(on: bool):
    global AGENT_ENABLED
    AGENT_ENABLED = bool(on)


def set_context(ctx: InvestigationContext):
    global _CTX
    _CTX = ctx


# ------------------------------------------------------------------ models
class TransactionIn(BaseModel):
    merchant_id: str
    customer_id: str
    device_id: str
    ip: str
    instrument_id: str
    amount: float = Field(gt=0)
    ts: int
    p_fraud: float = Field(ge=0, le=1, description="calibrated ML probability")
    component_size: float = 1.0
    device_account_count: float = 0
    ip_account_count: float = 0
    instrument_customer_count: float = 0
    cust_txn_5m: float = 0


class AnalystDecision(BaseModel):
    action: str
    note: str | None = None


# ------------------------------------------------------------------ routes
@app.get("/api/config")
def config():
    """What this instance can do. The hosted demo runs without the LLM, and
    the dashboard says so rather than showing an empty panel that reads as
    broken."""
    return {"agent_enabled": AGENT_ENABLED, "has_context": _CTX is not None}


@app.get("/api/health")
def health():
    return {"ok": True, "replay": STATE.status()["finished"] and "done" or "running"}


@app.get("/api/status")
def status():
    """Replay progress + the demo narration feed."""
    return STATE.status()


@app.post("/api/replay/speed", dependencies=WRITE)
def set_speed(speed: float):
    if not 1 <= speed <= 20000:
        raise HTTPException(400, "speed must be between 1 and 20000 txns/sec")
    STATE.speed = float(speed)
    # deliberately NOT logged to the event feed: dragging the slider fires many
    # requests and the resulting "speed set to..." spam pushes the actual
    # story (spikes, investigations, the finale) out of the visible window
    return {"speed_tps": STATE.speed}


@app.post("/api/replay/pause", dependencies=WRITE)
def pause(paused: bool = True):
    STATE.paused = bool(paused)
    STATE.log_event("system", "paused" if paused else "resumed")
    return {"paused": STATE.paused}


@app.get("/api/merchants")
def merchants():
    """Overview grid - spiking merchants sort to the top."""
    return {"merchants": STATE.snapshot_merchants()}


@app.get("/api/merchants/{merchant_id}/risk")
def merchant_risk(merchant_id: str):
    m = STATE.snapshot_merchant(merchant_id)
    if m is None:
        raise HTTPException(404, f"unknown merchant {merchant_id}")
    return m


@app.get("/api/merchants/{merchant_id}/entity-graph")
def entity_graph(merchant_id: str, min_accounts: int = 2):
    return STATE.entity_graph(merchant_id, min_accounts=min_accounts)


@app.get("/api/merchants/{merchant_id}/investigation")
def investigation(merchant_id: str):
    with STATE._lock:
        m = STATE.merchants.get(merchant_id)
        rep = m.investigation if m else None
    if rep is None:
        raise HTTPException(404, "no investigation for this merchant yet")
    return rep


@app.post("/api/merchants/{merchant_id}/investigate", dependencies=WRITE)
def run_investigation(merchant_id: str):
    """Trigger an investigation on demand (the replay also fires these on spike)."""
    if _CTX is None:
        raise HTTPException(503, "investigation context not ready")
    if not _CTX.known_merchant(merchant_id):
        # 404 rather than a degraded report on an empty slice - and it keeps a
        # caller-supplied path segment out of the model prompt entirely.
        raise HTTPException(404, "unknown merchant")
    from src.agent.investigator import investigate as _investigate
    res = _investigate(_CTX, merchant_id)
    STATE.set_investigation(merchant_id, res.report, res.audit.to_records(),
                            res.degraded, res.validated_action.value)
    return {**res.report, "degraded": res.degraded,
            "validated_action": res.validated_action.value,
            "audit": res.audit.to_records()}


@app.get("/api/review-queue")
def review_queue(pending_only: bool = False):
    q = STATE.snapshot_queue(pending_only=pending_only)
    # `pending` keeps its original meaning (true pending count) but is now
    # computed over the FULL queue, not the 200-row wire cap - the header and
    # this panel must never quote different numbers.
    return {"cases": q["cases"], "pending": q["pending_total"],
            "total_cases": q["total_cases"]}


@app.post("/api/review-queue/{case_id}/decision", dependencies=WRITE)
def decide_case(case_id: int, body: AnalystDecision):
    """Analyst approve/override. The allowlist binds humans too - an analyst
    cannot invent an action any more than the LLM can."""
    if body.action not in ALLOWLIST:
        raise HTTPException(400, f"action must be one of {sorted(ALLOWLIST)}")
    out = STATE.decide_case(case_id, body.action, body.note)
    if out is None:
        raise HTTPException(404, f"unknown case {case_id}")
    return out


@app.get("/api/audit-log")
def audit_log(limit: int = 200):
    with STATE._lock:
        return {"entries": STATE.audit[-limit:], "total": len(STATE.audit)}


@app.post("/api/transactions", dependencies=WRITE)
def score_transaction(txn: TransactionIn):
    """Score ONE transaction through the live fusion -> policy path.

    Note this does not mutate replay state - it is the serving surface, so a
    caller can ask "what would you do with this?" without perturbing the demo."""
    risk, conf, fused = fuse_for_policy(RiskSignals(
        p_fraud=txn.p_fraud, spike_z=None, component_size=txn.component_size,
        rule_hits=evaluate_rules(
            device_account_count=txn.device_account_count,
            ip_account_count=txn.ip_account_count,
            instrument_customer_count=txn.instrument_customer_count,
            cust_txn_5m=txn.cust_txn_5m)))
    m = STATE.snapshot_merchant(txn.merchant_id)
    spiking = bool(m and m["in_spike"])
    d = decide(risk_score=risk, confidence=conf, merchant_in_spike=spiking)
    return {"merchant_id": txn.merchant_id, "risk_score": fused.risk_score,
            "confidence": fused.confidence, "components": fused.components,
            "fusion_reason": fused.reason, "merchant_in_spike": spiking,
            "action": d.action.value, "reason": d.reason,
            "requires_human": d.requires_human}


# ------------------------------------------------------------------ static
@app.get("/")
def index():
    """Serve the SPA with the write key injected, so the dashboard can act
    without the operator exporting anything. Same-origin only: the key never
    leaves the page it was minted for."""
    doc = (STATIC / "index.html").read_text(encoding="utf-8")
    tag = f'<meta name="fsi-key" content="{API_KEY}">'
    return HTMLResponse(doc.replace("<!--KEY-->", tag))


app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
