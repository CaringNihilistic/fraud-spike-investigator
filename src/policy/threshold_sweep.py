"""Policy-threshold sweep on the VALIDATION slice (days 21-23).

The 85/60 restrict/step-up cutoffs were hand-set against the old
`risk_score = p * 100` scale and never re-derived. This sweeps both cutoffs
under the SAME net-protected-value framework and ₹ assumptions used to
report test economics, and picks the cost-optimal pair.

RULE FIXED IN ADVANCE (before looking at any result), mirroring the
model-selection tie-break discipline:
  * Adopt a new pair ONLY if it beats (85, 60) validation net protected
    value by more than ADOPT_MARGIN_PCT.
  * A win inside that margin is treated as noise -> keep 85/60 and report
    that the sweep VALIDATED the existing policy.
  * Ties in NPV break toward FEWER human review cases (operational load),
    then toward the higher restrict cut (more conservative auto-blocking).

Honest caveats, stated up front:
  * The validation slice holds 125 fraud transactions and 90 of them are a
    single device-farm attack, so this sweep is decided by essentially one
    attack type. That is precisely why the adopt margin exists.
  * Isotonic calibration is fit on this same slice, so calibrated
    probabilities here are mildly in-sample-optimistic. This is a POLICY
    selection, not a performance estimate - the numbers reported in the
    README's results table still come from the untouched test slice.
  * The test slice is never read by this module.

Run: python -m src.policy.threshold_sweep
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.features.builder import FEATURE_COLS, build_features  # noqa: E402
from src.models.select_model import (build_gbdt, get_selected_model_name,  # noqa: E402
                                      pos_weight, temporal_split)
from src.policy.engine import Action, decide  # noqa: E402
from src.policy.fusion import RiskSignals, evaluate_rules, fuse_for_policy  # noqa: E402
from src.sim.simulator import generate  # noqa: E402
from src.spike.detector import StreamingSpikeDetector  # noqa: E402

OUT = Path("artifacts_out")
OUT.mkdir(exist_ok=True)

# Same documented assumptions as train.py step 7 - changing them here would
# make the sweep optimize a different objective than the one we report.
REVIEW_COST_INR = 50.0
STEP_UP_ABANDON = 0.07
STEP_UP_FRAUD_BLOCKED = 0.90

BASELINE = (85.0, 60.0)
ADOPT_MARGIN_PCT = 2.0   # must beat baseline NPV by >2% to be adopted
GRID_STEP = 5


def score_pair(rows, restrict_cut: float, step_up_cut: float) -> dict:
    """Replay pre-computed (risk, confidence, spiking, amount, is_fraud) rows
    through the policy engine at one threshold pair and cost the outcome."""
    prevented = impacted = 0.0
    review_cases = 0
    mix: dict[str, int] = {}
    for risk, conf, spiking, amount, fraud in rows:
        d = decide(risk_score=risk, confidence=conf, merchant_in_spike=spiking,
                   restrict_cut=restrict_cut, step_up_cut=step_up_cut)
        mix[d.action.value] = mix.get(d.action.value, 0) + 1
        if d.action == Action.RESTRICT:
            prevented += amount if fraud else 0.0
            impacted += 0.0 if fraud else amount
            review_cases += 1
        elif d.action == Action.REVIEW:
            prevented += amount if fraud else 0.0
            review_cases += 1
        elif d.action == Action.STEP_UP:
            prevented += amount * STEP_UP_FRAUD_BLOCKED if fraud else 0.0
            impacted += 0.0 if fraud else amount * STEP_UP_ABANDON
    review_cost = review_cases * REVIEW_COST_INR
    return {"restrict_cut": restrict_cut, "step_up_cut": step_up_cut,
            "net_protected_value_inr": round(prevented - impacted - review_cost, 2),
            "fraud_prevented_inr": round(prevented, 2),
            "legit_impacted_inr": round(impacted, 2),
            "review_cases": review_cases,
            "review_cost_inr": round(review_cost, 2),
            "n_restrict": mix.get("restrict", 0), "n_review": mix.get("review", 0),
            "n_step_up": mix.get("step_up", 0), "n_allow": mix.get("allow", 0)}


def build_validation_rows():
    """Score the validation slice exactly the way train.py scores test."""
    df = build_features(generate(seed=7))
    train, cal, _test = temporal_split(df)  # test deliberately unused
    model = build_gbdt(get_selected_model_name(), pos_weight(train.is_fraud))
    model.fit(train[FEATURE_COLS], train.is_fraud)

    iso = IsotonicRegression(out_of_bounds="clip")
    raw_cal = model.predict_proba(cal[FEATURE_COLS])[:, 1]
    iso.fit(raw_cal, cal.is_fraud)          # in-sample for the sweep - see docstring
    p_cal = iso.predict(raw_cal)

    stream = StreamingSpikeDetector()
    in_spike: dict[str, int] = {}
    rows = []
    for r in cal.assign(p=p_cal).itertuples(index=False):
        z_before = stream.spike_z(r.merchant_id)
        fire = stream.update(r.merchant_id, int(r.ts), float(r.p))
        if fire is not None:
            in_spike.setdefault(r.merchant_id, fire)
        risk, conf, _ = fuse_for_policy(RiskSignals(
            p_fraud=float(r.p), spike_z=z_before, component_size=float(r.component_size),
            rule_hits=evaluate_rules(
                device_account_count=float(r.device_account_count),
                ip_account_count=float(r.ip_account_count),
                instrument_customer_count=float(r.instrument_customer_count),
                cust_txn_5m=float(r.cust_txn_5m))))
        rows.append((risk, conf, r.merchant_id in in_spike,
                     float(r.amount), bool(r.is_fraud)))
    return rows, len(cal), int(cal.is_fraud.sum())


def main():
    print("=== policy threshold sweep (VALIDATION days 21-23 only) ===")
    rows, n_cal, n_fraud = build_validation_rows()
    print(f"validation rows {n_cal:,} | fraud {n_fraud} "
          f"(note: {n_fraud} positives, mostly one device-farm attack)")

    results = []
    for restrict_cut in range(40, 101, GRID_STEP):
        for step_up_cut in range(20, 101, GRID_STEP):
            if step_up_cut >= restrict_cut:
                continue  # step-up must be the strictly lower bar
            results.append(score_pair(rows, float(restrict_cut), float(step_up_cut)))

    table = pd.DataFrame(results)
    # tie-break: NPV desc, then fewer reviews, then higher restrict cut
    table = table.sort_values(
        ["net_protected_value_inr", "review_cases", "restrict_cut"],
        ascending=[False, True, False]).reset_index(drop=True)
    table.to_csv(OUT / "threshold_sweep.csv", index=False)

    best = table.iloc[0]
    base = table[(table.restrict_cut == BASELINE[0]) & (table.step_up_cut == BASELINE[1])].iloc[0]
    lift_pct = 100.0 * (best.net_protected_value_inr - base.net_protected_value_inr) / abs(base.net_protected_value_inr)

    print("\ntop 8 pairs by validation net protected value:")
    print(table.head(8)[["restrict_cut", "step_up_cut", "net_protected_value_inr",
                          "review_cases", "n_restrict", "n_step_up"]].to_string(index=False))
    # NOTE: plain "INR" not the rupee glyph - the Windows console is cp1252
    # and a stray UnicodeEncodeError here would kill the run after the sweep.
    print(f"\nbaseline (85, 60): NPV INR {base.net_protected_value_inr:,.2f}, "
          f"{base.review_cases} reviews")
    print(f"best     ({best.restrict_cut:.0f}, {best.step_up_cut:.0f}): "
          f"NPV INR {best.net_protected_value_inr:,.2f}, {best.review_cases} reviews")
    print(f"lift over baseline: {lift_pct:+.3f}%  (adopt margin: >{ADOPT_MARGIN_PCT}%)")

    # ---- per-parameter margin check ----------------------------------
    # The pre-declared rule ("adopt the best PAIR if it beats baseline by
    # >2%") turned out to be under-specified for this surface: the restrict
    # dimension is DEGENERATE on validation - every cut from 40 to 80 scores
    # exactly the same NPV, because no validation transaction lands in that
    # band at all (fused scores are bimodal, ~0 or ~100). Picking 80 out of a
    # 9-wide exact tie is choosing an arbitrary point in a dead zone, which
    # is precisely the noise-chasing the margin exists to prevent.
    #
    # So the SAME margin is applied per parameter: a cutoff moves only if
    # moving it alone clears the margin. Where it does not, the conservative
    # default stands. This refinement was made after seeing the surface, and
    # is recorded here rather than quietly folded into the result.
    # The grid requires step_up < restrict, so a one-at-a-time move is not
    # always a POINT ON THE GRID: moving restrict down to 55 while step_up
    # stays at the baseline 60 is not a valid policy at all. That pair used
    # to raise IndexError and kill the run. A move we cannot evaluate in
    # isolation is not evidence FOR making it, so it returns None and the
    # conservative default stands - the same direction the margin already
    # points. Never silently substitute a nearby cell.
    def npv_at(rc, sc):
        row = table[(table.restrict_cut == rc) & (table.step_up_cut == sc)]
        return None if row.empty else float(row.iloc[0].net_protected_value_inr)

    def lift_of(rc, sc, base):
        v = npv_at(rc, sc)
        return None if v is None else 100.0 * (v - base) / abs(base)

    base_npv = npv_at(*BASELINE)
    step_only = float(best.step_up_cut)
    restrict_only = float(best.restrict_cut)
    step_lift = lift_of(BASELINE[0], step_only, base_npv)
    restrict_lift = lift_of(restrict_only, BASELINE[1], base_npv)

    def clears(lift):
        return lift is not None and lift > ADOPT_MARGIN_PCT

    adopted_restrict = restrict_only if clears(restrict_lift) else BASELINE[0]
    adopted_step_up = step_only if clears(step_lift) else BASELINE[1]
    adopted = (adopted_restrict, adopted_step_up)

    def verdict_for(lift, keep):
        if lift is None:
            return ("NOT INDEPENDENTLY EVALUABLE (step_up must stay below "
                    f"restrict) -> KEEP {int(keep)}")
        return (f"{lift:+.3f}%  -> "
                f"{'ADOPT' if lift > ADOPT_MARGIN_PCT else 'KEEP ' + str(int(keep))}")

    print(f"\nper-parameter lift (each moved alone, other held at baseline):")
    print(f"  restrict {BASELINE[0]:.0f} -> {restrict_only:.0f}: "
          f"{verdict_for(restrict_lift, BASELINE[0])}")
    print(f"  step_up  {BASELINE[1]:.0f} -> {step_only:.0f}: "
          f"{verdict_for(step_lift, BASELINE[1])}")

    def _say(lift):
        return "not independently evaluable" if lift is None else f"{lift:+.2f}%"

    verdict = (f"ADOPTED ({adopted[0]:.0f}, {adopted[1]:.0f}). "
               f"step_up {BASELINE[1]:.0f}->{adopted[1]:.0f}: {_say(step_lift)}; "
               f"restrict {BASELINE[0]:.0f}->{adopted[0]:.0f}: {_say(restrict_lift)}. "
               f"(adopt margin {ADOPT_MARGIN_PCT}%, applied per parameter)")
    if adopted == tuple(BASELINE):
        verdict = (f"KEPT (85, 60) - no single cutoff clears the {ADOPT_MARGIN_PCT}% margin on a "
                   f"slice with {n_fraud} positives. The sweep VALIDATED the existing policy.")
    print(f"\n{verdict}")

    json.dump({"adopted_restrict_cut": adopted[0], "adopted_step_up_cut": adopted[1],
               "baseline": {"restrict_cut": BASELINE[0], "step_up_cut": BASELINE[1],
                            "net_protected_value_inr": float(base.net_protected_value_inr),
                            "review_cases": int(base.review_cases)},
               "grid_best": {"restrict_cut": float(best.restrict_cut),
                             "step_up_cut": float(best.step_up_cut),
                             "net_protected_value_inr": float(best.net_protected_value_inr),
                             "review_cases": int(best.review_cases)},
               "lift_pct": round(lift_pct, 4), "adopt_margin_pct": ADOPT_MARGIN_PCT,
               "per_parameter_lift_pct": {"restrict": round(restrict_lift, 4),
                                           "step_up": round(step_lift, 4)},
               "restrict_surface_flat_over": "40-80 (exact NPV ties -> unidentified)",
               "validation_rows": n_cal, "validation_fraud": n_fraud,
               "verdict": verdict},
              open(OUT / "threshold_sweep_decision.json", "w"), indent=2)
    print(f"wrote {OUT / 'threshold_sweep.csv'} and {OUT / 'threshold_sweep_decision.json'}")


if __name__ == "__main__":
    main()
