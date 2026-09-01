"""What order should the review queue be worked in? (failure 28)

A review queue only matters if the analyst runs out of time before it runs
out of cases - and ours produces 578 cases on a 13,782-transaction slice, so
it always will. Under a capacity constraint, the ORDER is a policy decision
worth as much as the threshold, and ours was never chosen: the serving layer
appended cases as they arrived and the dashboard showed the newest first.
Arrival order has no relationship to money at all.

This module measures what that costs, by rebuilding the review queue through
exactly the same fusion -> policy path train.py step 7 uses, then asking one
question: if an analyst can only work the first N cases, how much fraud value
does each ordering put in front of them?

Three orderings:
  arrival        - what we shipped: newest first
  risk           - by fused risk score, the obvious fix
  expected_loss  - amount x calibrated probability, i.e. rupees at stake

`expected_loss` deliberately uses the CALIBRATED probability `p`, not the
fused risk score. Fusion's risk_score is an escalation scale, not a
probability - multiplying rupees by it would be a category error, and this
project is careful about that distinction elsewhere.

GROUND TRUTH IS READ HERE AND NOWHERE NEAR THE SERVING LAYER. is_fraud scores
the orderings after the fact; ReviewCase never carries it, and no API route
can reach it.

Run: python -m src.policy.queue_order
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.features.builder import FEATURE_COLS, build_features  # noqa: E402
from src.models.ablation import _fit_eval  # noqa: E402
from src.models.select_model import get_selected_model_name, temporal_split  # noqa: E402
from src.policy.engine import Action, decide  # noqa: E402
from src.policy.fusion import RiskSignals, evaluate_rules, fuse_for_policy  # noqa: E402
from src.sim.simulator import generate  # noqa: E402
from src.spike.detector import StreamingSpikeDetector  # noqa: E402

OUT = Path("artifacts_out")
OUT.mkdir(exist_ok=True)

# Analyst capacity, stated so it can be argued with. 30 cases per analyst-hour
# (two minutes each) x an 8-hour shift = 240 cases a day, the same assumption
# the dashboard's capacity panel already uses.
CAPACITY_POINTS = [50, 100, 240, 400]


def build_queue() -> pd.DataFrame:
    """Rebuild the review queue over the test slice, same path as train.py."""
    df = build_features(generate())
    train, cal, test = temporal_split(df)
    _, p_test = _fit_eval(FEATURE_COLS, train, cal, test, get_selected_model_name())

    stream = StreamingSpikeDetector()
    in_spike: dict[str, int] = {}
    rows = []
    for r in test.assign(p=p_test).itertuples(index=False):
        z_before = stream.spike_z(r.merchant_id)
        fire = stream.update(r.merchant_id, int(r.ts), float(r.p))
        if fire is not None:
            in_spike.setdefault(r.merchant_id, fire)
        signals = RiskSignals(
            p_fraud=float(r.p), spike_z=z_before,
            component_size=float(r.component_size),
            rule_hits=evaluate_rules(
                device_account_count=float(r.device_account_count),
                ip_account_count=float(r.ip_account_count),
                instrument_customer_count=float(r.instrument_customer_count),
                cust_txn_5m=float(r.cust_txn_5m)),
        )
        risk, conf, _ = fuse_for_policy(signals)
        d = decide(risk_score=risk, confidence=conf,
                   merchant_in_spike=r.merchant_id in in_spike)
        if d.action in (Action.REVIEW, Action.RESTRICT):
            rows.append({
                "ts": int(r.ts), "merchant_id": r.merchant_id,
                "amount_inr": float(r.amount), "risk_score": float(risk),
                "p": float(r.p), "expected_loss_inr": float(r.amount) * float(r.p),
                "is_fraud": int(r.is_fraud),
            })
    return pd.DataFrame(rows)


def main():
    q = build_queue()
    n, total_fraud = len(q), float(q.loc[q.is_fraud == 1, "amount_inr"].sum())
    print("=== review-queue ordering ===")
    print(f"{n:,} cases | fraud value in the queue: INR {total_fraud:,.2f}\n")

    orderings = {
        # what we shipped: newest first
        "arrival": q.sort_values("ts", ascending=False),
        "risk": q.sort_values("risk_score", ascending=False),
        "expected_loss": q.sort_values("expected_loss_inr", ascending=False),
    }

    rows = []
    header = "  cases worked  " + "".join(f"{k:>18}" for k in orderings)
    print(header)
    for k in CAPACITY_POINTS:
        if k > n:
            continue
        caught = {name: float(o.head(k).query("is_fraud == 1").amount_inr.sum())
                  for name, o in orderings.items()}
        rows.append({"cases_worked": k,
                     **{f"{name}_fraud_inr": round(v, 2) for name, v in caught.items()},
                     "expected_loss_vs_arrival_inr": round(
                         caught["expected_loss"] - caught["arrival"], 2)})
        print(f"  {k:<14}" + "".join(f"{v:>18,.0f}" for v in caught.values()))

    print("\n--- share of the queue's fraud value captured ---")
    print(header)
    for k in CAPACITY_POINTS:
        if k > n:
            continue
        pct = {name: 100.0 * float(o.head(k).query("is_fraud == 1").amount_inr.sum())
               / total_fraud for name, o in orderings.items()}
        print(f"  {k:<14}" + "".join(f"{v:>17.1f}%" for v in pct.values()))

    # Derived verdict: never hardcode a conclusion the numbers might contradict.
    ref = next(r for r in rows if r["cases_worked"] == 240)
    gain = ref["expected_loss_vs_arrival_inr"]
    gain_pct = 100.0 * gain / ref["arrival_fraud_inr"] if ref["arrival_fraud_inr"] else float("inf")
    beats_risk = ref["expected_loss_fraud_inr"] >= ref["risk_fraud_inr"]
    verdict = (
        f"At one analyst-day (240 cases), ordering by expected loss puts "
        f"INR {ref['expected_loss_fraud_inr']:,.0f} of fraud in front of the analyst "
        f"against INR {ref['arrival_fraud_inr']:,.0f} for the arrival order we shipped "
        f"- a difference of INR {gain:,.0f} ({gain_pct:+.1f}%). "
        + ("Expected loss also beats ranking by risk alone, which is the point: "
           "the queue is denominated in rupees and its sort key should be too."
           if beats_risk else
           "NOTE: ranking by risk alone did better here, so the rupee weighting is "
           "not paying for itself on this slice - report that, do not bury it."))
    print(f"\n=== verdict ===\n{verdict}")

    pd.DataFrame(rows).to_csv(OUT / "queue_order.csv", index=False)
    json.dump({"queue_cases": n, "queue_fraud_inr": round(total_fraud, 2),
               "capacity_points": CAPACITY_POINTS, "rows": rows,
               "verdict": verdict},
              open(OUT / "queue_order.json", "w"), indent=2)
    print(f"\nwrote {OUT / 'queue_order.csv'}, {OUT / 'queue_order.json'}")


if __name__ == "__main__":
    main()
