"""The seven READ-ONLY investigation tools.

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
    """Normal behaviour for this merchant, and its worst window.

    Uses the SPIKE DETECTOR'S OWN slow-EWMA baseline - one definition of
    "normal" across the whole system, not a second one invented here.

    The previous version split the merchant's window at the 75th percentile of
    timestamps and called the tail "recent". For an attack that ENDS mid-window
    - which is every attack in this dataset - that reads as
    "baseline 8.99% -> recent 0.35%", i.e. improving, during an active
    incident. The agent then rationalised the contradiction instead of
    distrusting it, once describing a decrease as a "jump". Reporting the PEAK
    window alongside the baseline removes the ambiguity, and every window is
    labelled with explicit bounds so direction cannot be misread."""
    from src.spike.detector import StreamingSpikeDetector

    g = ctx.merchant_slice(merchant_id).sort_values("ts")
    if g.empty:
        return {"merchant_id": merchant_id, "error": "no transactions"}

    det = StreamingSpikeDetector()
    peak_rate, peak_z, peak_ts, fired_at = 0.0, 0.0, None, None
    for r in g.itertuples(index=False):
        z = det.spike_z(merchant_id)
        fire = det.update(merchant_id, int(r.ts), float(r.p))
        if fire is not None and fired_at is None:
            fired_at = int(fire)
        rate = det.hot_rate(merchant_id)
        if rate > peak_rate:
            peak_rate, peak_ts = rate, int(r.ts)
        peak_z = max(peak_z, z)

    return {
        "merchant_id": merchant_id,
        "window_start_ts": int(g.ts.min()), "window_end_ts": int(g.ts.max()),
        "total_txns": int(len(g)),
        # "normal" per the detector's own slow EWMA over the whole window
        "baseline_flagged_rate_ewma": round(float(det.baseline_rate(merchant_id)), 4),
        # worst 30-transaction window observed, and when it happened
        "peak_flagged_rate": round(float(peak_rate), 4),
        "peak_flagged_rate_window_ended_ts": peak_ts,
        "peak_spike_z": round(float(peak_z), 2),
        "peak_vs_baseline_multiple": (
            round(float(peak_rate / det.baseline_rate(merchant_id)), 2)
            if det.baseline_rate(merchant_id) > 0.001 else None),
        # current state at the END of the window (may be calm again post-attack)
        "current_flagged_rate_last_30_txns": round(float(det.hot_rate(merchant_id)), 4),
        "spike_detector_fired": fired_at is not None,
        "spike_fired_at_ts": fired_at,
        "avg_amount_inr": round(float(g.amount.mean()), 2),
        "reading_note": (
            "peak_flagged_rate is the WORST 30-transaction window in this "
            "merchant's history; current_flagged_rate_last_30_txns is only the "
            "state at the end of the window. A finished attack shows a high "
            "peak and a low current value - that is an attack that ENDED, not "
            "an absence of attack."),
    }


def get_flagged_transactions(ctx: InvestigationContext, merchant_id: str,
                             limit: int = 15) -> dict:
    """The highest-risk transactions, as evidence the agent can cite."""
    g = ctx.merchant_slice(merchant_id)
    flagged = g[g.p >= HIGH_RISK_CUT]
    unflagged = g[g.p < HIGH_RISK_CUT]
    hot = flagged.nlargest(min(limit, 50), "p")

    def instr_ratio(d):
        return round(float(d.instrument_id.nunique() / len(d)), 3) if len(d) else None

    return {
        "merchant_id": merchant_id,
        "flagged_count": int((g.p >= HIGH_RISK_CUT).sum()),
        "total_count": int(len(g)),
        # Distinct payment instruments per transaction, with denominators, and
        # the same ratio over this merchant's non-flagged traffic for comparison.
        "flagged_distinct_instruments": int(flagged.instrument_id.nunique()),
        "flagged_distinct_instruments_per_txn": instr_ratio(flagged),
        "non_flagged_count": int(len(unflagged)),
        "non_flagged_distinct_instruments": int(unflagged.instrument_id.nunique()),
        "non_flagged_distinct_instruments_per_txn": instr_ratio(unflagged),
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


def get_customer_anomalies(ctx: InvestigationContext, merchant_id: str,
                           limit: int = 15) -> dict:
    """Per-customer behavioural deviation among flagged transactions.

    Closes a structural blind spot. Every other tool describes ENTITY SHARING
    (one device across many accounts, one IP across many accounts), which is
    the signature of farms, clusters and rings. Account takeover has the
    opposite shape: each victim gets their OWN new device, so entity-sharing
    tools see "29 customers, 29 devices, 29 IPs - no sharing" and conclude the
    traffic is legitimate. It is not; the victims are real customers behaving
    unlike themselves.

    These four signals already existed in the feature builder and drove the ML
    scorer - the agent simply could not see them. Read-only like every other
    tool, and it still never exposes is_fraud or scenario."""
    g = ctx.merchant_slice(merchant_id)
    cols = ["is_new_device_for_customer", "geo_mismatch", "amount_dev_ratio",
            "customer_age_days"]
    missing = [c for c in cols if c not in g.columns]
    if missing:
        return {"merchant_id": merchant_id,
                "error": f"anomaly features unavailable: {missing}"}
    hot = g[g.p >= HIGH_RISK_CUT]
    cold = g[g.p < HIGH_RISK_CUT]
    if hot.empty:
        return {"merchant_id": merchant_id, "flagged_count": 0,
                "note": "no flagged transactions to profile"}

    def profile(d):
        if d.empty:
            return None
        return {
            "n": int(len(d)),
            "share_on_new_device_for_customer": round(float(d.is_new_device_for_customer.mean()), 3),
            "share_with_geo_mismatch": round(float(d.geo_mismatch.mean()), 3),
            "median_amount_vs_customer_own_average": round(float(d.amount_dev_ratio.median()), 2),
            "share_spending_over_3x_own_average": round(float((d.amount_dev_ratio >= 3).mean()), 3),
            "median_customer_age_days": round(float(d.customer_age_days.median()), 1),
            "share_accounts_newer_than_30_days": round(float((d.customer_age_days < 30).mean()), 3),
        }

    f, c = profile(hot), profile(cold)
    lift = {}
    if c:
        for k in ("share_on_new_device_for_customer", "share_with_geo_mismatch",
                  "share_spending_over_3x_own_average", "share_accounts_newer_than_30_days"):
            lift[k] = (round(f[k] / c[k], 2) if c[k] > 0.001 else
                       ("no_baseline_occurrences" if f[k] > 0 else 1.0))

    top = hot.nlargest(min(limit, 50), "p")
    return {
        "merchant_id": merchant_id,
        "flagged_count": int(len(hot)),
        "total_txns": int(len(g)),
        "flagged_share_of_all_txns": round(float(len(hot) / max(1, len(g))), 4),
        # A rate is meaningless without its denominator and a comparison
        # population. Reporting only the flagged profile made 4-of-5 ambient
        # fraud transactions on a quiet merchant look like a coordinated
        # campaign, because "80%" reads as strong until you see n=5 and the
        # merchant's own baseline right beside it.
        "flagged_profile": f,
        "non_flagged_profile_same_merchant": c,
        "flagged_vs_non_flagged_lift": lift,
        "interpretation_note": (
            "Compare flagged_profile against non_flagged_profile_same_merchant "
            "and weigh by n. A share computed over a handful of transactions is "
            "weak evidence however extreme it looks; a modest share over "
            "hundreds is strong. Lift near 1.0 means the flagged transactions "
            "are indistinguishable from this merchant's ordinary traffic on "
            "that dimension."),
        "transactions": [
            {"customer": r.customer_id,
             "on_new_device": bool(r.is_new_device_for_customer),
             "geo_mismatch": bool(r.geo_mismatch),
             "amount_vs_own_avg": round(float(r.amount_dev_ratio), 2),
             "account_age_days": round(float(r.customer_age_days), 1),
             "amount_inr": round(float(r.amount), 2)}
            for r in top.itertuples(index=False)
        ],
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
    {"name": "get_customer_anomalies",
     "description": "Per-customer behavioural deviation among flagged transactions: "
                    "share on a NEW device for that customer, share with a geo "
                    "mismatch, spend vs the customer's OWN historical average, and "
                    "account age. Use when entity-sharing looks absent - established "
                    "customers behaving unlike themselves leave no shared entities.",
     "input_schema": {"type": "object", "properties": {
         "merchant_id": {"type": "string"},
         "limit": {"type": "integer", "description": "max transactions, default 15"}},
         "required": ["merchant_id"]}},
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
    "get_customer_anomalies": get_customer_anomalies,
    "calculate_exposure": calculate_exposure,
    "write_investigation_report": write_investigation_report,
}

READ_ONLY_TOOLS = frozenset(TOOL_FNS)
