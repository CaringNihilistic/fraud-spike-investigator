"""The six READ-ONLY investigation tools.

Design rules, each enforced here rather than by prompt instruction:
  * READ-ONLY. Every tool takes a merchant/window and returns a summary. None
    of them writes to the dataset, the policy engine, or the review queue.
  * calculate_exposure does the RUPEE ARITHMETIC IN PYTHON. The LLM is never
    asked to add up money - it reads the number this tool returns. LLM
    arithmetic is exactly the kind of silent error a risk team cannot audit.
  * write_investigation_report does NOT write anything either: it validates
    and structures the agent's findings. The name is the agent-facing
    contract ("this is how you finish"), not a description of side effects.
  * Tool outputs are plain JSON-able dicts, small enough to sit in context.

Every tool is wrapped by the caller so its (inputs, output) pair lands in the
audit log - see investigator.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Same documented assumptions as the economics loop in train.py. Duplicated
# as an explicit import target rather than re-derived, so the agent's ₹
# figures and the reported economics can never silently diverge.
STEP_UP_FRAUD_BLOCKED = 0.90
REVIEW_COST_INR = 50.0

HIGH_RISK_CUT = 0.5   # a transaction is "flagged" above this calibrated prob


class InvestigationContext:
    """Everything the tools may read: one scored slice of the stream.

    Holds ONLY data that existed at investigation time. `ground truth`
    columns (is_fraud, scenario) are deliberately NOT exposed by any tool -
    the agent must reason from signals, exactly as it would in production.
    """

    def __init__(self, scored: pd.DataFrame):
        required = {"merchant_id", "ts", "p", "amount", "customer_id",
                    "device_id", "ip", "instrument_id"}
        missing = required - set(scored.columns)
        if missing:
            raise ValueError(f"InvestigationContext missing columns: {missing}")
        self.df = scored

    def merchant_slice(self, merchant_id: str) -> pd.DataFrame:
        return self.df[self.df.merchant_id == merchant_id]


# ------------------------------------------------------------------ tools
def get_merchant_baseline(ctx: InvestigationContext, merchant_id: str) -> dict:
    """Normal behaviour for this merchant vs the spike window.

    Baseline = the merchant's earlier hours; spike window = its most recent
    hours. Splitting on the merchant's own history is what lets the agent say
    'this is abnormal FOR THIS MERCHANT' rather than 'this looks big'."""
    g = ctx.merchant_slice(merchant_id)
    if g.empty:
        return {"merchant_id": merchant_id, "error": "no transactions"}
    cut = g.ts.quantile(0.75)
    base, recent = g[g.ts <= cut], g[g.ts > cut]
    return {
        "merchant_id": merchant_id,
        "baseline_txn_count": int(len(base)),
        "baseline_flagged_rate": round(float((base.p >= HIGH_RISK_CUT).mean()), 4),
        "baseline_avg_amount_inr": round(float(base.amount.mean()), 2),
        "recent_txn_count": int(len(recent)),
        "recent_flagged_rate": round(float((recent.p >= HIGH_RISK_CUT).mean()), 4),
        "recent_avg_amount_inr": round(float(recent.amount.mean()), 2),
        "flagged_rate_multiple": round(
            float((recent.p >= HIGH_RISK_CUT).mean() /
                  max(1e-6, (base.p >= HIGH_RISK_CUT).mean())), 2),
    }


def get_flagged_transactions(ctx: InvestigationContext, merchant_id: str,
                             limit: int = 15) -> dict:
    """The highest-risk transactions, as evidence the agent can cite."""
    g = ctx.merchant_slice(merchant_id)
    hot = g[g.p >= HIGH_RISK_CUT].nlargest(min(limit, 50), "p")
    return {
        "merchant_id": merchant_id,
        "flagged_count": int((g.p >= HIGH_RISK_CUT).sum()),
        "total_count": int(len(g)),
        "transactions": [
            {"ts": int(r.ts), "risk": round(float(r.p), 3),
             "amount_inr": round(float(r.amount), 2),
             "customer": r.customer_id, "device": r.device_id,
             "ip": r.ip, "instrument": r.instrument_id}
            for r in hot.itertuples(index=False)
        ],
    }


def get_entity_network(ctx: InvestigationContext, merchant_id: str) -> dict:
    """Shared-entity structure among flagged transactions.

    This is the tool that distinguishes a RING (few entities, many accounts)
    from ordinary traffic (many entities, one account each) - the single most
    diagnostic signal for the attack types this system targets."""
    g = ctx.merchant_slice(merchant_id)
    hot = g[g.p >= HIGH_RISK_CUT]
    if hot.empty:
        return {"merchant_id": merchant_id, "flagged_count": 0,
                "note": "no flagged transactions to correlate"}

    def fanout(entity_col, account_col="customer_id"):
        s = hot.groupby(entity_col)[account_col].nunique().sort_values(ascending=False)
        return [{"entity": str(k), "distinct_accounts": int(v)} for k, v in s.head(5).items()]

    return {
        "merchant_id": merchant_id,
        "flagged_count": int(len(hot)),
        "distinct_customers": int(hot.customer_id.nunique()),
        "distinct_devices": int(hot.device_id.nunique()),
        "distinct_ips": int(hot.ip.nunique()),
        "distinct_instruments": int(hot.instrument_id.nunique()),
        "top_devices_by_account_fanout": fanout("device_id"),
        "top_ips_by_account_fanout": fanout("ip"),
        "instruments_per_customer": round(
            float(hot.instrument_id.nunique() / max(1, hot.customer_id.nunique())), 2),
    }


def get_velocity_summary(ctx: InvestigationContext, merchant_id: str) -> dict:
    """Transaction rate over time - separates a burst from a steady drip."""
    g = ctx.merchant_slice(merchant_id)
    if g.empty:
        return {"merchant_id": merchant_id, "error": "no transactions"}
    hot = g[g.p >= HIGH_RISK_CUT]
    span_h = max(1e-6, (g.ts.max() - g.ts.min()) / 3600)
    hot_span_h = max(1e-6, (hot.ts.max() - hot.ts.min()) / 3600) if len(hot) > 1 else 0.0
    return {
        "merchant_id": merchant_id,
        "window_hours": round(float(span_h), 2),
        "txns_per_hour_overall": round(float(len(g) / span_h), 2),
        "flagged_txns": int(len(hot)),
        "flagged_span_hours": round(float(hot_span_h), 2),
        "flagged_per_hour": round(float(len(hot) / hot_span_h), 2) if hot_span_h > 0 else None,
        "peak_hour_txns": int(g.groupby(g.ts // 3600).size().max()),
    }


def calculate_exposure(ctx: InvestigationContext, merchant_id: str) -> dict:
    """DETERMINISTIC ₹ arithmetic. The LLM never computes money.

    Reports exposure at risk and what the bounded actions would recover,
    using the same assumptions as the reported economics."""
    g = ctx.merchant_slice(merchant_id)
    hot = g[g.p >= HIGH_RISK_CUT]
    at_risk = float(hot.amount.sum())
    # Expected loss weights each flagged amount by its calibrated probability -
    # this is why calibration matters: p is used as a real probability here.
    expected_loss = float((hot.amount * hot.p).sum())
    return {
        "merchant_id": merchant_id,
        "flagged_txn_count": int(len(hot)),
        "exposure_at_risk_inr": round(at_risk, 2),
        "probability_weighted_expected_loss_inr": round(expected_loss, 2),
        "recoverable_via_step_up_inr": round(expected_loss * STEP_UP_FRAUD_BLOCKED, 2),
        "review_cost_if_all_reviewed_inr": round(len(hot) * REVIEW_COST_INR, 2),
        "assumptions": {"step_up_fraud_blocked": STEP_UP_FRAUD_BLOCKED,
                        "review_cost_inr": REVIEW_COST_INR,
                        "note": "computed in Python; the model does not do this arithmetic"},
    }


def write_investigation_report(ctx: InvestigationContext, merchant_id: str,
                               cause: str, evidence: list, exposure_inr: float,
                               recommended_action: str, confidence: float) -> dict:
    """Structure the agent's conclusion. Writes nothing; the policy engine
    still re-validates recommended_action downstream."""
    return {
        "merchant_id": merchant_id,
        "cause": str(cause),
        "evidence": [str(e) for e in (evidence or [])],
        "exposure_inr": float(exposure_inr) if exposure_inr is not None else 0.0,
        "recommended_action": str(recommended_action),
        "confidence": float(np.clip(confidence if confidence is not None else 0.0, 0.0, 1.0)),
    }


# Tool schemas for the Anthropic API. Kept adjacent to the implementations so
# a schema can't drift from the function it describes.
TOOL_SCHEMAS = [
    {"name": "get_merchant_baseline",
     "description": "Normal behaviour for this merchant vs its most recent window. "
                    "Use first to establish whether anything is actually abnormal.",
     "input_schema": {"type": "object", "properties": {
         "merchant_id": {"type": "string"}}, "required": ["merchant_id"]}},
    {"name": "get_flagged_transactions",
     "description": "Highest-risk transactions for this merchant, to cite as evidence.",
     "input_schema": {"type": "object", "properties": {
         "merchant_id": {"type": "string"},
         "limit": {"type": "integer", "description": "max transactions, default 15"}},
         "required": ["merchant_id"]}},
    {"name": "get_entity_network",
     "description": "Shared device/IP/instrument structure among flagged transactions. "
                    "High account-fanout on few entities indicates a ring, farm, or cluster.",
     "input_schema": {"type": "object", "properties": {
         "merchant_id": {"type": "string"}}, "required": ["merchant_id"]}},
    {"name": "get_velocity_summary",
     "description": "Transaction rate and burst span - distinguishes a concentrated "
                    "attack from a steady low-level drip.",
     "input_schema": {"type": "object", "properties": {
         "merchant_id": {"type": "string"}}, "required": ["merchant_id"]}},
    {"name": "calculate_exposure",
     "description": "Deterministic rupee exposure arithmetic. ALWAYS use this for money "
                    "figures - never compute rupee amounts yourself.",
     "input_schema": {"type": "object", "properties": {
         "merchant_id": {"type": "string"}}, "required": ["merchant_id"]}},
    {"name": "write_investigation_report",
     "description": "Finish the investigation. Call exactly once, last.",
     "input_schema": {"type": "object", "properties": {
         "merchant_id": {"type": "string"},
         "cause": {"type": "string",
                   "description": "one of: card_testing, device_farm, ip_cluster, "
                                  "account_takeover, fraud_ring, legitimate_traffic, unclear"},
         "evidence": {"type": "array", "items": {"type": "string"},
                      "description": "specific figures from tool outputs"},
         "exposure_inr": {"type": "number",
                          "description": "copy from calculate_exposure; do not compute"},
         "recommended_action": {"type": "string",
                                "description": "one of: allow, step_up, review, restrict"},
         "confidence": {"type": "number", "description": "0.0-1.0"}},
         "required": ["merchant_id", "cause", "evidence", "exposure_inr",
                      "recommended_action", "confidence"]}},
]

TOOL_FNS = {
    "get_merchant_baseline": get_merchant_baseline,
    "get_flagged_transactions": get_flagged_transactions,
    "get_entity_network": get_entity_network,
    "get_velocity_summary": get_velocity_summary,
    "calculate_exposure": calculate_exposure,
    "write_investigation_report": write_investigation_report,
}

READ_ONLY_TOOLS = frozenset(TOOL_FNS)
