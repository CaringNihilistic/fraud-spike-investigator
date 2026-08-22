"""Deterministic policy engine - the ONLY component that authorizes actions.

Hard rules:
  * Actions come from a frozen allowlist. Nothing else can execute.
  * The LLM investigator may RECOMMEND an allowlisted action; this engine
    re-validates it. An unknown/unauthorized recommendation degrades to REVIEW.
  * Low confidence NEVER auto-restricts - it escalates to a human.
  * If the ML score is missing (model down), decisions fall back to rules
    and everything above ALLOW goes to human review. An LLM failure cannot
    block anyone, because the LLM was never in the decision path.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Action(str, Enum):
    ALLOW = "allow"
    STEP_UP = "step_up"        # additional verification (OTP/3DS)
    REVIEW = "review"          # human review queue
    RESTRICT = "restrict"      # hold high-risk txns for this entity, pending human


ALLOWLIST = frozenset(a.value for a in Action)

# Decision thresholds on the fused 0-100 risk scale. Defaults are the
# original hand-set policy; src/policy/threshold_sweep.py re-derives them on
# the VALIDATION slice under the net-protected-value framework. They are
# parameters of decide() so the sweep can vary them without monkey-patching,
# but the defaults are what production uses.
RESTRICT_CUT = 85.0   # unchanged: validation NPV is flat across 40-80, so the
#                       sweep could not identify a better value; keep the
#                       conservative hand-set cut rather than pick a dead-zone point.
STEP_UP_CUT = 25.0    # was 60.0, hand-set against the old p*100 scale.
#                       Cost-optimized on validation (days 21-23): +8.28% net
#                       protected value. Catches fraud the 60 cut allowed through.
LOW_CONFIDENCE_CUT = 0.4


@dataclass
class Decision:
    action: Action
    reason: str
    requires_human: bool


def decide(risk_score: float | None, confidence: float,
           merchant_in_spike: bool,
           restrict_cut: float = RESTRICT_CUT,
           step_up_cut: float = STEP_UP_CUT) -> Decision:
    """Map (risk_score 0-100 | None, confidence 0-1, spike flag) -> Decision.

    Thresholds are injectable ONLY so the validation sweep can score
    candidate pairs; the safety ordering below is fixed regardless of what
    they are set to - ML-unavailable and low-confidence always short-circuit
    to human review before any threshold is consulted."""
    if risk_score is None:  # ML unavailable -> fail safe, never auto-block
        return Decision(Action.REVIEW, "ml_unavailable_rules_only", True)

    if confidence < LOW_CONFIDENCE_CUT:
        return Decision(Action.REVIEW, "low_confidence_escalation", True)

    if risk_score >= restrict_cut and merchant_in_spike:
        # even the strongest signal only RESTRICTS pending human confirmation
        return Decision(Action.RESTRICT, "high_risk_during_spike", True)
    if risk_score >= restrict_cut:
        return Decision(Action.REVIEW, "high_risk", True)
    if risk_score >= step_up_cut:
        return Decision(Action.STEP_UP, "medium_risk_step_up", False)
    return Decision(Action.ALLOW, "low_risk", False)


def validate_recommendation(recommended: str) -> Action:
    """Re-validate any externally-recommended action (e.g., from the LLM).
    Anything outside the allowlist degrades to REVIEW - never escalates."""
    if recommended in ALLOWLIST:
        return Action(recommended)
    return Action.REVIEW
