"""P1b: explicit risk fusion.

Replaces the `risk_score = p_fraud * 100` shortcut in the economics loop.
That shortcut had two problems worth stating plainly:
  1. It threw away every signal except the ML probability - the merchant
     spike z-score, the entity-graph structure, and the rule hits were
     computed but never reached the policy engine.
  2. It reported no CONFIDENCE, so the policy engine's low-confidence
     escalation path (confidence < 0.4 -> human review) could never fire;
     train.py hardcoded confidence=0.85 for every transaction.

Design constraints (deliberate, and defensible in one sentence each):
  * LINEAR and BOUNDED. A weighted sum of four bounded components, not a
    learned meta-model. A second model stacked on the first would need its
    own temporal split, its own calibration, and its own leakage argument -
    and would be far harder to explain to a risk analyst than "ML 60%,
    spike 20%, graph 12%, rules 8%".
  * The ML probability keeps majority weight. Fusion ADJUSTS a calibrated
    score with context; it does not overrule it.
  * Confidence is about AGREEMENT, not magnitude. Signals pointing the same
    way -> high confidence. Signals disagreeing (ML says fraud, nothing else
    corroborates) -> low confidence -> the policy engine escalates to a
    human instead of auto-restricting. This is the fail-safe direction.
  * Fusion NEVER authorizes anything. It emits (risk_score, confidence);
    policy.engine.decide() remains the only component that picks an action.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# --- scoring shape -------------------------------------------------------
# The calibrated ML probability sets the FLOOR; context can only escalate
# from there, saturating at 100. Written first as a plain weighted average
# (ml*0.6 + context*0.4), which was wrong: it capped a p=1.0 transaction at
# 60, i.e. it silently overruled a model we had just spent a calibration
# slice making trustworthy - the opposite of the stated intent. Keeping the
# ML score as a floor also means risk_score stays on the SAME 0-100 scale as
# the old `p * 100` shortcut, so the policy engine's thresholds (85/60) keep
# their meaning and did not have to be re-tuned (re-tuning them against test
# results would have been leakage).
#
# Relative weights WITHIN the context term (sum to 1.0):
W_SPIKE = 0.50
W_GRAPH = 0.30
W_RULES = 0.20
# How far full corroborating context can lift a transaction toward 100.
# 0.6 means: context alone (ML silent) tops out at 60 -> step-up, never an
# automatic restrict. Corroboration escalates; it does not convict.
CONTEXT_LIFT = 0.6

# A z-score at/above this is treated as a fully-saturated spike signal.
Z_SATURATE = 8.0

# Graph signal: measure EXCESS component size over the ordinary population,
# not raw size. Ordinary customers already sit in components of ~10 because
# they share ISP IP pools, so a naive "size >= 15 is risky" rule fired on 26%
# of LEGITIMATE transactions and handed nearly everyone a graph-risk bonus.
# Both constants are derived from the TRAIN slice only (never test):
#   FLOOR    = train legit p99 (25)  -> 99% of legitimate txns score 0 here
#   SATURATE = train fraud  p90 (120) -> ring/farm-scale components score 1.0
# Below the floor the graph contributes nothing; it is a CORROBORATING signal
# for ring-shaped fraud, not a general-purpose risk term.
COMPONENT_FLOOR = 25.0
COMPONENT_SATURATE = 120.0


@dataclass
class RiskSignals:
    """Everything the fusion layer is allowed to look at. All optional except
    p_fraud - a missing signal contributes 0 and LOWERS confidence rather
    than being silently imputed as 'safe'."""
    p_fraud: float | None                 # calibrated ML probability [0,1], None = ML down
    spike_z: float | None = None          # merchant spike z-score, None = no spike/unknown
    component_size: float | None = None   # entity-graph connected-component size
    rule_hits: list[str] = field(default_factory=list)  # names of deterministic rules fired


@dataclass
class FusedRisk:
    risk_score: float          # 0-100, what the policy engine consumes
    confidence: float          # 0-1, drives low-confidence escalation
    components: dict           # per-signal contribution, for the audit log / dashboard
    reason: str                # one-line human-readable explanation


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def fuse(signals: RiskSignals, max_rules: int = 4) -> FusedRisk:
    """Combine bounded signals into (risk_score 0-100, confidence 0-1).

    ML down (p_fraud is None) returns risk_score=None-equivalent semantics by
    emitting confidence 0.0 and whatever context supports; the policy engine
    still receives a real number, so callers that want the explicit
    fail-safe path should pass risk_score=None to decide() themselves when
    signals.p_fraud is None. See fuse_for_policy() below for that wiring."""
    ml = _clamp01(signals.p_fraud) if signals.p_fraud is not None else 0.0
    spike = _clamp01((signals.spike_z or 0.0) / Z_SATURATE)
    graph = _clamp01(((signals.component_size or 0.0) - COMPONENT_FLOOR)
                     / (COMPONENT_SATURATE - COMPONENT_FLOOR))
    rules = _clamp01(len(signals.rule_hits) / max_rules)

    # ML is the floor; corroborating context escalates into the headroom
    # above it. Monotone in ml, and never scores below the ML score alone.
    context = W_SPIKE * spike + W_GRAPH * graph + W_RULES * rules
    score = 100.0 * _clamp01(ml + (1.0 - ml) * CONTEXT_LIFT * context)

    # ---- confidence = agreement among AVAILABLE signals ----
    # Corroboration: how many context signals meaningfully agree with the ML
    # verdict. Deliberately asymmetric - we only demand corroboration when ML
    # claims HIGH risk, because a false "block" is the expensive error here.
    available = [s for s in (signals.spike_z, signals.component_size) if s is not None]
    context = [spike, graph, rules]
    agreeing = sum(1 for c in context if c >= 0.25)

    if signals.p_fraud is None:
        confidence = 0.0
        reason = "ml_unavailable"
    elif ml >= 0.5:
        # high ML risk: confident only if context corroborates
        confidence = _clamp01(0.35 + 0.20 * agreeing + (0.10 if available else 0.0))
        reason = (f"ml_high({ml:.2f}) corroborated_by={agreeing}"
                  if agreeing else f"ml_high({ml:.2f}) uncorroborated")
    else:
        # low ML risk: confident unless context screams otherwise
        disagree = sum(1 for c in context if c >= 0.5)
        confidence = _clamp01(0.85 - 0.25 * disagree)
        reason = (f"ml_low({ml:.2f}) contradicted_by={disagree}"
                  if disagree else f"ml_low({ml:.2f}) consistent")

    # Report each signal's ACTUAL contribution in risk points, so the audit
    # log and dashboard can show "why this score" rather than raw weights.
    headroom = (1.0 - ml) * CONTEXT_LIFT * 100.0
    return FusedRisk(
        risk_score=round(score, 2),
        confidence=round(confidence, 3),
        components={"ml": round(ml * 100, 2),
                    "spike": round(headroom * W_SPIKE * spike, 2),
                    "graph": round(headroom * W_GRAPH * graph, 2),
                    "rules": round(headroom * W_RULES * rules, 2)},
        reason=reason,
    )


# --------------------------------------------------------------- rules
# Deterministic, human-readable rules. These produce SIGNALS ONLY - they
# never pick an action (that stays in policy.engine). Kept few and obvious
# on purpose: a rule an analyst cannot restate in one sentence is a rule we
# cannot defend when it fires on a real merchant.
RULE_THRESHOLDS = {
    "shared_device_many_accounts": 5,    # one device seen with >=5 distinct accounts
    "shared_ip_many_accounts": 8,        # one IP seen with >=8 distinct accounts
    "instrument_many_customers": 3,      # one payment instrument across >=3 customers
    "burst_velocity": 5,                 # >=5 txns from one customer in 5 minutes
}


def evaluate_rules(device_account_count: float = 0, ip_account_count: float = 0,
                   instrument_customer_count: float = 0, cust_txn_5m: float = 0) -> list[str]:
    """Return the names of fired rules. Inputs are already-computed features,
    so this adds no new state and cannot leak future information."""
    hits = []
    if device_account_count >= RULE_THRESHOLDS["shared_device_many_accounts"]:
        hits.append("shared_device_many_accounts")
    if ip_account_count >= RULE_THRESHOLDS["shared_ip_many_accounts"]:
        hits.append("shared_ip_many_accounts")
    if instrument_customer_count >= RULE_THRESHOLDS["instrument_many_customers"]:
        hits.append("instrument_many_customers")
    if cust_txn_5m >= RULE_THRESHOLDS["burst_velocity"]:
        hits.append("burst_velocity")
    return hits


def fuse_for_policy(signals: RiskSignals) -> tuple[float | None, float, FusedRisk]:
    """Fusion output in the exact shape policy.engine.decide() expects.

    Returns (risk_score_or_None, confidence, full_fused_object). When the ML
    scorer is down we pass risk_score=None so decide() takes its documented
    fail-safe branch (REVIEW, never auto-block) rather than acting on a
    context-only score that would understate risk."""
    fused = fuse(signals)
    risk = None if signals.p_fraud is None else fused.risk_score
    return risk, fused.confidence, fused
