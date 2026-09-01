"""Does the rejected configuration's NPV advantage survive averaging?

Failure-log 29 rejects "config 4" - teaching the model that a shared IP can be
honest as well as a shared device. Config 4 removes the corporate-buyer false
alarm and scored a HIGHER net protected value than the configuration we
shipped: INR 9.44L against INR 8.11L.

That comparison was on ONE seed, and the reason we rejected config 4 - it loses
card-testing detection entirely - is on a DIFFERENT seed. Comparing a win on
seed 7 against a failure on seed 101 is not a comparison. An outside reviewer
caught that, correctly, as the thinnest part of the argument.

So: run both configurations across all five seeds and report NPV for each. If
config 4's advantage does not survive averaging, the rejection is on firmer
ground than "we preferred the other failure mode". If it DOES survive, we say
so - we rejected a configuration that was economically better, on the grounds
that it silently loses an attack class we cannot explain, and that trade should
be visible rather than buried.

Ground truth is read here to score outcomes and reaches no serving component.

Run: python -m src.policy.config4_npv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.features.builder import FEATURE_COLS, build_features  # noqa: E402
from src.models.ablation import ATTACK_MERCHANTS, _fit_eval  # noqa: E402
from src.models.select_model import get_selected_model_name, temporal_split  # noqa: E402
from src.models.train import (REVIEW_COST_INR, STEP_UP_ABANDON,  # noqa: E402
                              STEP_UP_FRAUD_BLOCKED)
from src.policy.engine import Action, decide  # noqa: E402
from src.policy.fusion import RiskSignals, evaluate_rules, fuse_for_policy  # noqa: E402
from src.sim import simulator as sim  # noqa: E402
from src.spike.detector import StreamingSpikeDetector  # noqa: E402

OUT = Path("artifacts_out")
OUT.mkdir(exist_ok=True)
SEEDS = [7, 11, 23, 42, 101]


def run_one(seed: int, model_name: str) -> dict:
    """One world, scored through the same fusion -> policy path as train.py."""
    df = build_features(sim.generate(seed=seed))
    train, cal, test = temporal_split(df)
    _, p_test = _fit_eval(FEATURE_COLS, train, cal, test, model_name)

    stream = StreamingSpikeDetector()
    in_spike: dict[str, int] = {}
    prevented = impacted = 0.0
    reviews = 0
    for r in test.assign(p=p_test).itertuples(index=False):
        z = stream.spike_z(r.merchant_id)
        fire = stream.update(r.merchant_id, int(r.ts), float(r.p))
        if fire is not None:
            in_spike.setdefault(r.merchant_id, fire)
        signals = RiskSignals(
            p_fraud=float(r.p), spike_z=z, component_size=float(r.component_size),
            rule_hits=evaluate_rules(
                device_account_count=float(r.device_account_count),
                ip_account_count=float(r.ip_account_count),
                instrument_customer_count=float(r.instrument_customer_count),
                cust_txn_5m=float(r.cust_txn_5m)))
        risk, conf, _ = fuse_for_policy(signals)
        d = decide(risk_score=risk, confidence=conf,
                   merchant_in_spike=r.merchant_id in in_spike)
        amt, fraud = float(r.amount), bool(r.is_fraud)
        if d.action is Action.RESTRICT:
            prevented += amt if fraud else 0.0
            impacted += 0.0 if fraud else amt
        elif d.action is Action.REVIEW:
            prevented += amt if fraud else 0.0
        elif d.action is Action.STEP_UP:
            prevented += amt * STEP_UP_FRAUD_BLOCKED if fraud else 0.0
            impacted += 0.0 if fraud else amt * STEP_UP_ABANDON
        if d.action in (Action.REVIEW, Action.RESTRICT):
            reviews += 1

    caught = sum(1 for m in ATTACK_MERCHANTS.values() if m in in_spike)
    false_alarms = sorted(set(in_spike) - set(ATTACK_MERCHANTS.values())
                          - {"m11", "m4", "m10"})
    legit_spikes_fired = sorted(m for m in ("m4", "m10", "m11") if m in in_spike)
    return {
        "seed": seed,
        "net_protected_value_inr": round(prevented - impacted - reviews * REVIEW_COST_INR, 2),
        "reviews": reviews,
        "attacks_caught": f"{caught}/{len(ATTACK_MERCHANTS)}",
        "legit_spikes_fired": ",".join(legit_spikes_fired) or "none",
        "other_false_alarms": ",".join(false_alarms) or "none",
    }


def main():
    model_name = get_selected_model_name()
    print("=== config 3 (shipped) vs config 4 (rejected), all 5 seeds ===")
    print(f"model family: {model_name}\n")

    results = {}
    for label, flag in (("config3_shipped", False), ("config4_rejected", True)):
        sim.HIST_CORPORATE_BUYER = flag
        rows = [run_one(s, model_name) for s in SEEDS]
        results[label] = rows
        mean = sum(r["net_protected_value_inr"] for r in rows) / len(rows)
        print(f"--- {label} (HIST_CORPORATE_BUYER={flag}) ---")
        for r in rows:
            print("  seed %-4d NPV INR %12s  attacks %-4s  legit spikes fired: %s"
                  % (r["seed"], format(r["net_protected_value_inr"], ",.0f"),
                     r["attacks_caught"], r["legit_spikes_fired"]))
        print("  MEAN NPV INR %s\n" % format(mean, ",.2f"))
        results[label + "_mean_npv"] = round(mean, 2)
    sim.HIST_CORPORATE_BUYER = False

    m3, m4 = results["config3_shipped_mean_npv"], results["config4_rejected_mean_npv"]
    a3 = sum(int(r["attacks_caught"].split("/")[0]) for r in results["config3_shipped"])
    a4 = sum(int(r["attacks_caught"].split("/")[0]) for r in results["config4_rejected"])
    verdict = (
        f"Across 5 seeds: config 3 mean NPV INR {m3:,.0f} and {a3}/25 attacks; "
        f"config 4 mean NPV INR {m4:,.0f} and {a4}/25 attacks. "
        + (f"Config 4's single-seed NPV advantage does NOT survive averaging "
           f"(it is {m4 - m3:+,.0f}), so the rejection does not rest on trading "
           f"money for a failure mode we happened to prefer."
           if m4 <= m3 else
           f"Config 4 IS economically better on average ({m4 - m3:+,.0f}), and we "
           f"rejected it anyway because it loses {25 - a4} of 25 attacks. That is a "
           f"real trade against our own declared cost rule and it is reported as one, "
           f"not buried."))
    print("=== verdict ===\n" + verdict)

    pd.DataFrame(results["config3_shipped"]).assign(config="3_shipped").to_csv(
        OUT / "config4_npv_config3.csv", index=False)
    pd.DataFrame(results["config4_rejected"]).assign(config="4_rejected").to_csv(
        OUT / "config4_npv_config4.csv", index=False)
    json.dump({"seeds": SEEDS, "verdict": verdict, **results},
              open(OUT / "config4_npv.json", "w"), indent=2)
    print(f"\nwrote {OUT / 'config4_npv.json'}")


if __name__ == "__main__":
    main()
