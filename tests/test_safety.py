"""Safety-invariant tests. These encode the judging story:
the system must fail SAFE, and the LLM must be unable to escalate."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.policy.engine import Action, decide, validate_recommendation
from src.policy.fusion import (COMPONENT_FLOOR, CONTEXT_LIFT, RiskSignals,
                               evaluate_rules, fuse, fuse_for_policy)
from src.spike.detector import SpikeDetector, StreamingSpikeDetector


# ---------------- policy engine ----------------
def test_ml_unavailable_never_blocks():
    d = decide(risk_score=None, confidence=1.0, merchant_in_spike=True)
    assert d.action == Action.REVIEW and d.requires_human


def test_low_confidence_escalates_not_restricts():
    d = decide(risk_score=99, confidence=0.1, merchant_in_spike=True)
    assert d.action == Action.REVIEW and d.requires_human


def test_restrict_requires_spike_and_human():
    d = decide(risk_score=95, confidence=0.9, merchant_in_spike=True)
    assert d.action == Action.RESTRICT and d.requires_human
    d2 = decide(risk_score=95, confidence=0.9, merchant_in_spike=False)
    assert d2.action == Action.REVIEW  # no auto-restrict outside a spike


def test_llm_cannot_invent_actions():
    # anything outside the allowlist degrades to REVIEW, never escalates
    assert validate_recommendation("ban_customer_forever") == Action.REVIEW
    assert validate_recommendation("drop table users") == Action.REVIEW
    assert validate_recommendation("restrict") == Action.RESTRICT  # allowlisted ok


def test_ordinary_traffic_is_allowed_without_friction():
    assert decide(10, 0.9, False).action == Action.ALLOW


# ---------------- spike detector ----------------
def _feed(det, merchant, hours):
    ev = None
    for i, scores in enumerate(hours):
        ev = det.update_hour(merchant, i * 3600, scores) or ev
    return ev


def test_volume_spike_without_fraud_does_not_fire():
    """Flash-sale invariant: 10x VOLUME with normal score-rate must not fire."""
    det = SpikeDetector()
    quiet = [[0.01] * 15 for _ in range(30)]          # warmup: clean hours
    flash = [[0.01] * 150 for _ in range(3)]           # volume x10, scores normal
    assert _feed(det, "m_flash", quiet + flash) is None


def test_fraud_rate_spike_fires():
    det = SpikeDetector()
    quiet = [[0.01] * 15 for _ in range(30)]
    attack = [[0.9] * 10 + [0.01] * 10 for _ in range(3)]  # half the hour scores hot
    ev = _feed(det, "m_attack", quiet + attack)
    assert ev is not None and ev.z >= det.z_threshold


def test_a_quiet_hour_cannot_false_alarm_on_one_stray_fraud():
    det = SpikeDetector()
    quiet = [[0.01] * 15 for _ in range(30)]
    tiny = [[0.99] * 3]                                # 3 txns, all hot - too few
    assert _feed(det, "m_tiny", quiet + tiny) is None


# ---------------- P1b risk fusion ----------------
def test_fusion_never_scores_below_ml_alone():
    """The floor invariant: fusion may ESCALATE on corroborating context but
    must never overrule a calibrated model downward. This is the bug the
    first implementation had - a weighted average capped p=1.0 at 60."""
    for p in (0.0, 0.1, 0.5, 0.9, 1.0):
        bare = fuse(RiskSignals(p_fraud=p))
        assert bare.risk_score >= p * 100 - 1e-6
        rich = fuse(RiskSignals(p_fraud=p, spike_z=99, component_size=999,
                                rule_hits=["a", "b", "c", "d"]))
        assert rich.risk_score >= bare.risk_score


def test_fusion_never_pulls_a_certain_ml_score_down():
    """A p=1.0 transaction must still read as maximum risk, so the policy
    engine's thresholds keep the meaning they had under the p*100 shortcut."""
    assert fuse(RiskSignals(p_fraud=1.0)).risk_score == 100.0


def test_context_alone_cannot_reach_restrict():
    """Corroboration escalates; it does not convict. With the ML score at
    zero, every context signal saturated must still land below the RESTRICT
    threshold (85) - it may reach step-up territory at most."""
    fused = fuse(RiskSignals(p_fraud=0.0, spike_z=999, component_size=9999,
                             rule_hits=["a", "b", "c", "d"]))
    assert fused.risk_score <= CONTEXT_LIFT * 100 + 1e-6
    assert fused.risk_score < 85
    d = decide(risk_score=fused.risk_score, confidence=fused.confidence,
               merchant_in_spike=True)
    assert d.action != Action.RESTRICT


def test_fusion_ml_unavailable_takes_failsafe_path():
    """ML down must reach decide() as risk_score=None so the documented
    fail-safe branch fires - never a context-only score that understates risk."""
    risk, conf, fused = fuse_for_policy(RiskSignals(p_fraud=None, spike_z=99,
                                                    component_size=999))
    assert risk is None and conf == 0.0
    d = decide(risk_score=risk, confidence=conf, merchant_in_spike=True)
    assert d.action == Action.REVIEW and d.requires_human


def test_uncorroborated_high_ml_lowers_confidence():
    """Confidence is about AGREEMENT. High ML with nothing corroborating must
    be less confident than the same score with corroboration - that is what
    routes ambiguous cases to a human instead of an automatic restrict."""
    alone = fuse(RiskSignals(p_fraud=0.95))
    backed = fuse(RiskSignals(p_fraud=0.95, spike_z=8, component_size=200,
                              rule_hits=["shared_device_many_accounts"]))
    assert alone.confidence < backed.confidence


def test_low_confidence_escalation_branch_is_reachable():
    """The old code hardcoded confidence=0.85, making the policy engine's
    low-confidence branch dead code. Fusion must be able to produce a
    confidence low enough to actually trigger it."""
    fused = fuse(RiskSignals(p_fraud=None))
    assert fused.confidence < 0.4
    assert decide(risk_score=50, confidence=fused.confidence,
                  merchant_in_spike=False).action == Action.REVIEW


def test_ordinary_component_size_contributes_no_graph_risk():
    """Ordinary customers share ISP IP pools and sit in components of ~10.
    They must contribute ZERO graph risk, or every legitimate transaction
    collects a risk bonus (the bug the first COMPONENT_SATURATE=15 had)."""
    for ordinary in (1, 5, 10, COMPONENT_FLOOR):
        f = fuse(RiskSignals(p_fraud=0.2, component_size=ordinary))
        assert f.components["graph"] == 0.0


def test_fusion_outputs_are_bounded():
    for p in (0.0, 0.5, 1.0):
        for z in (0, 50, 1e6):
            f = fuse(RiskSignals(p_fraud=p, spike_z=z, component_size=1e6,
                                 rule_hits=["a"] * 20))
            assert 0.0 <= f.risk_score <= 100.0
            assert 0.0 <= f.confidence <= 1.0


def test_rules_are_signals_not_actions():
    """evaluate_rules must only ever name rules; nothing it returns may be
    an executable action - the policy engine stays the sole authorizer."""
    hits = evaluate_rules(device_account_count=99, ip_account_count=99,
                          instrument_customer_count=99, cust_txn_5m=99)
    assert hits and all(h not in {a.value for a in Action} for h in hits)


def test_streaming_detector_will_not_fire_before_it_has_a_baseline():
    """A brand-new merchant must not manufacture a large z from a few hot
    transactions - otherwise fusion escalates on no evidence."""
    det = StreamingSpikeDetector()
    for i in range(5):
        det.update("m_new", i * 60, 0.99)
    assert det.spike_z("m_new") == 0.0
