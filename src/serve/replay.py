"""Replay driver: streams the test slice through the REAL pipeline.

This is not a canned animation. Every transaction goes through the same
fusion -> policy path as `python -m src.models.train` step 7, so what the
dashboard shows during the demo is what the system actually decided. The only
thing being simulated is the passage of time.

Demo arc (days 24-29 of the test slice, in timestamp order):
  normal traffic -> attack merchants spike -> investigation fires on spike
  -> flash-sale merchant m11 spikes in VOLUME on day 29 and is NOT flagged.

The finale is the point: a 6x legitimate volume spike must produce no spike
state, no restricts, and no investigation.
"""
from __future__ import annotations

import threading
import time

from sklearn.isotonic import IsotonicRegression

from src.agent.tools import InvestigationContext
from src.features.builder import FEATURE_COLS, build_features
from src.models.select_model import (build_gbdt, get_selected_model_name,
                                     pos_weight, temporal_split)
from src.policy.engine import Action, decide
from src.policy.fusion import RiskSignals, evaluate_rules, fuse_for_policy
from src.serve.state import STATE
from src.sim.simulator import generate
from src.spike.detector import StreamingSpikeDetector

# Investigate at most once per merchant, and only on a real spike - an agent
# that re-investigates every transaction would burn tokens and add nothing.
INVESTIGATED: set[str] = set()


def prepare():
    """Train + score the test slice. Returns (scored_test_df, context)."""
    STATE.log_event("system", "training model and scoring test slice...")
    df = build_features(generate(seed=7))
    train, cal, test = temporal_split(df)
    model = build_gbdt(get_selected_model_name(), pos_weight(train.is_fraud))
    model.fit(train[FEATURE_COLS], train.is_fraud)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(model.predict_proba(cal[FEATURE_COLS])[:, 1], cal.is_fraud)
    p = iso.predict(model.predict_proba(test[FEATURE_COLS])[:, 1])
    scored = test.assign(p=p).sort_values("ts").reset_index(drop=True)
    STATE.log_event("system", f"ready: {len(scored):,} test transactions "
                              f"({get_selected_model_name()} scorer)")
    return scored, InvestigationContext(scored)


def run(scored, ctx, investigate_enabled: bool = True):
    """Stream transactions through fusion -> policy at STATE.speed txns/sec."""
    stream = StreamingSpikeDetector()
    in_spike_from: dict[str, int] = {}
    STATE.total = len(scored)
    STATE.started_at = time.time()

    # is_fraud is carried ONLY to cost the decisions after the fact - it is
    # never an input to fusion or to the policy engine. The demo replays a
    # labelled held-out slice, so the running INR figure is a replay of a
    # measured result rather than a live prediction.
    cols = ["merchant_id", "ts", "p", "amount", "customer_id", "device_id",
            "ip", "instrument_id", "component_size", "device_account_count",
            "ip_account_count", "instrument_customer_count", "cust_txn_5m",
            "is_fraud"]
    records = scored[cols].to_dict("records")

    batch_sleep, since_sleep = 0.0, 0
    for row in records:
        while STATE.paused:
            time.sleep(0.05)

        mid = row["merchant_id"]
        z_before = stream.spike_z(mid)
        fire = stream.update(mid, int(row["ts"]), float(row["p"]))
        if fire is not None:
            in_spike_from.setdefault(mid, fire)
        spiking = mid in in_spike_from

        risk, conf, _fused = fuse_for_policy(RiskSignals(
            p_fraud=float(row["p"]), spike_z=z_before,
            component_size=float(row["component_size"]),
            rule_hits=evaluate_rules(
                device_account_count=float(row["device_account_count"]),
                ip_account_count=float(row["ip_account_count"]),
                instrument_customer_count=float(row["instrument_customer_count"]),
                cust_txn_5m=float(row["cust_txn_5m"]))))
        d = decide(risk_score=risk, confidence=conf, merchant_in_spike=spiking)
        STATE.record_txn(row, risk, conf, d.action, d.reason, spiking,
                         in_spike_from.get(mid),
                         baseline_rate=stream.baseline_rate(mid),
                         current_rate=stream.hot_rate(mid),
                         spike_z=stream.spike_z(mid))

        # Investigation is triggered BY THE SPIKE, not by a timer - same
        # trigger condition the architecture specifies.
        if investigate_enabled and spiking and mid not in INVESTIGATED:
            INVESTIGATED.add(mid)
            threading.Thread(target=_investigate_async, args=(ctx, mid),
                             daemon=True).start()

        # Pace the stream. Sleeping per-transaction would cap us at ~60/s on
        # Windows' timer granularity, so sleep once per batch instead.
        since_sleep += 1
        batch = max(1, int(STATE.speed / 20))
        if since_sleep >= batch:
            batch_sleep = batch / max(1.0, STATE.speed)
            time.sleep(batch_sleep)
            since_sleep = 0

    STATE.finished = True
    STATE.log_event("system", "replay complete")
    _finale_check()


def _investigate_async(ctx: InvestigationContext, mid: str):
    """Run the agent off the hot path. A slow or failing investigation must
    never stall transaction processing - the stream keeps flowing regardless."""
    # Imported lazily, not at module scope: a serve-only deployment running
    # with --no-agent must not need langgraph or the anthropic SDK installed
    # at all. That is what makes requirements-serve.txt genuinely slim.
    from src.agent.investigator import investigate
    STATE.log_event("investigation", f"{mid}: spike detected, investigating...",
                    merchant_id=mid)
    try:
        res = investigate(ctx, mid)
        STATE.set_investigation(mid, res.report, res.audit.to_records(),
                                res.degraded, res.validated_action.value)
    except Exception as e:      # belt and braces; investigate() already fail-safes
        STATE.log_event("investigation", f"{mid}: investigation error {type(e).__name__}",
                        merchant_id=mid)


def _finale_check():
    """State the flash-sale result explicitly in the event feed - it is the
    single most important thing the demo has to show."""
    m11 = STATE.merchants.get("m11")
    if not m11:
        return
    verdict = "NOT flagged" if not m11.in_spike else "FLAGGED (regression!)"
    STATE.log_event(
        "finale",
        f"FLASH SALE m11: {m11.txn_count} txns, "
        f"{m11.action_mix.get('restrict', 0)} restricts -> {verdict}",
        merchant_id="m11")


def start_background(investigate_enabled: bool = True):
    """Prepare data, then replay on a daemon thread."""
    scored, ctx = prepare()
    t = threading.Thread(target=run, args=(scored, ctx, investigate_enabled),
                         daemon=True)
    t.start()
    return t
