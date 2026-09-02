"""Legitimate merchants that grow suddenly - the negatives we never built.

Every legitimate merchant in this project except the flash sale is an
ENTITY-SHARING case: a corporate office on one IP, a kiosk on one device, and a
0-100% sweep of the same axis. So we have tested "shared infrastructure looks
like fraud" thoroughly and "sudden legitimate growth looks like fraud" not at
all - and the second is the more common real-world false alarm.

Two scenarios, and the first is the one that matters:

  A. MARKETING SURGE   an established merchant runs a campaign and 150 NEW
                       customers buy for the first time in one afternoon.
  B. COLD-START        a merchant appears for the first time and has a busy
                       opening day, with no history for anything to compare to.

WHY A IS DANGEROUS, mechanically. Reading builder.py, a brand-new customer
emits: geo_mismatch 0, is_new_device_for_customer 0, amount_dev_ratio 1.0 - all
neutral, which is good design. But a brand-new device and IP emit
is_first_seen_device = 1 and is_first_seen_ip = 1. A signup wave therefore
carries both flags on nearly every transaction, which is exactly what a
card-testing wave carries. The features that must do the discriminating are
entity fan-out and the amount profile.

BUILT SO customer_age_days CANNOT BE THE ANSWER. The easy version of this test
gives the surge aged accounts, proves the model can use the 21 other features,
and teaches nothing. Here the surge draws its account ages from the SAME
mixture as card testing (sim._attack_created_day at AGED_SHARE_CARD_TESTING), so
age is statistically degenerate between them and the discrimination has to come
from somewhere else. A test asserts that degeneracy rather than trusting it -
this matters because customer_age_days is the feature with a proven history of
being a label proxy (failure-log 21).

============================ PRE-REGISTRATION ============================
Written and committed BEFORE the first run, because this is the experiment most
likely to produce a mild ambiguous result close to a deadline - the condition
under which "that's within tolerance" gets tempting.

FIRING is defined exactly as everywhere else in this repo: the shipped
StreamingSpikeDetector returns a fire timestamp for that merchant. Not a
z-score, not a flagged rate, not a judgement call.

  - CONTROL FAILS (the real attacks do not fire 5/5 through this harness)
        -> VERDICT IS VOID. The run measures the harness, not the system.
           Publish nothing from it. Same rule as the sharing sweep.

  - SURGE FIRES on >= 1 of 5 seeds
        -> A REAL FALSE-ALARM MODE, and it gets published as one, with the seed
           named. We do NOT fix it by adding a marketing surge to training under
           deadline: that is the failure-log 29 move, it invalidates the frozen
           evaluation, and it needs a fresh held-out set to mean anything. It
           becomes a logged limitation and a v2 item.

  - SURGE NEVER FIRES and the control fires 5/5
        -> Reported as a genuine strengthening of the failure-21 fix: with age
           held degenerate, entity structure carried the discrimination. Stated
           as ONE scenario at one intensity, not as "we handle legitimate
           growth".

  - COLD-START merchant scores systematically HIGHER than matched warm
    merchants (mean calibrated p, same traffic)
        -> An ARCHITECTURAL gap, not a scenario gap. The honest answer is a
           cold-start rule - route a new merchant's first N transactions to
           review regardless of score - and we say so rather than patching the
           training distribution. We do not ship that rule in this run; the
           finding is the deliverable.

Nothing here is retrained, tuned, or re-thresholded. Shipped model, shipped
calibration, shipped cutoffs, 5 seeds.
==========================================================================

Run: python -m src.models.growth_negatives
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.features.builder import FEATURE_COLS, build_features  # noqa: E402
from src.models.select_model import (build_gbdt, get_selected_model_name,  # noqa: E402
                                     pos_weight, temporal_split)
from src.policy.engine import Action, decide  # noqa: E402
from src.policy.fusion import RiskSignals, evaluate_rules, fuse_for_policy  # noqa: E402
from src.sim import simulator as sim  # noqa: E402
from src.spike.detector import StreamingSpikeDetector  # noqa: E402

OUT = Path("artifacts_out")
OUT.mkdir(exist_ok=True)

SEEDS = [7, 11, 23, 42, 101]
EVENT_DAY = 26
HIST_DAYS = range(20, 30)
DAILY = 200
N_EVENT = 180          # matched to the card-testing wave this is compared against


def _world(seed: int) -> sim.World:
    return sim.World(rng=sim.RNG(seed + 3000))


def _txn(w, rng, ts, label, cid, fresh=False, rng_age=None) -> dict:
    """One transaction. `fresh` means a customer who has never bought anywhere:
    a brand-new account with a brand-new device, IP and card - which is what a
    signup wave actually looks like, and what makes is_first_seen_device and
    is_first_seen_ip fire."""
    c = w.customers[cid]
    amount = max(20.0, rng.normal(c["avg_amount"], c["avg_amount"] * 0.35))
    tag = "new" if fresh else "reg"
    return {
        "ts": int(ts), "merchant_id": label,
        "customer_id": sim.oid("c", f"gn{tag}{cid}"),
        "device_id": sim.oid("d", f"gn{tag}{cid}"),
        "ip": sim.oid("ip", f"gnpool{cid % 1500}" if not fresh else f"gnnewip{cid}"),
        "instrument_id": sim.oid("pi", f"gn{tag}{cid}"),
        "geo": c["home_geo"], "amount": round(float(amount), 2),
        "payment_method": str(rng.choice(["card", "upi", "netbanking", "wallet"],
                                         p=[0.35, 0.45, 0.1, 0.1])),
        # Age comes from the SAME mixture card testing uses, so the two are
        # statistically indistinguishable on the feature with a history of
        # being a label proxy. tests/test_growth_negatives.py asserts it.
        "customer_created_day": (sim._attack_created_day(
            rng_age or rng, EVENT_DAY, sim.AGED_SHARE_CARD_TESTING)
            if fresh else c["created_day"]),
        "is_fraud": 0, "scenario": "gn_baseline",
    }


def _history(w, rng, label: str, days) -> list[dict]:
    rows = []
    for day in days:
        if day == EVENT_DAY:
            continue
        start = sim.START_TS + day * sim.DAY
        for t in sim._poisson_times(rng, start, int(rng.normal(DAILY, 25))):
            rows.append(_txn(w, rng, t, label, int(rng.integers(0, w.n_customers))))
    return rows


def marketing_surge(w, rng, label: str) -> list[dict]:
    """150 brand-new customers buying for the first time in one afternoon.
    No entity sharing whatsoever - every account has its own device, IP and
    card. What it DOES share with card testing: first-seen device and IP on
    nearly every row, one novel instrument per transaction, a compressed
    window, and the same account-age mixture."""
    rows = _history(w, rng, label, HIST_DAYS)
    start = sim.START_TS + EVENT_DAY * sim.DAY + 13 * 3600
    for i, t in enumerate(np.sort(start + rng.uniform(0, 4 * 3600, N_EVENT))):
        cid = 100000 + i % 150          # 150 distinct newcomers, ~1.2 txns each
        r = _txn(w, rng, t, label, int(rng.integers(0, w.n_customers)), fresh=True)
        r.update(customer_id=sim.oid("c", f"gnsurge{cid}"),
                 device_id=sim.oid("d", f"gnsurge{cid}"),
                 ip=sim.oid("ip", f"gnsurge{cid}"),
                 instrument_id=sim.oid("pi", f"gnsurge{cid}"),
                 scenario="gn_marketing_surge")
        rows.append(r)
    return rows


def cold_start(w, rng, label: str) -> list[dict]:
    """A merchant that did not exist until its opening day. Same busy afternoon
    as the surge, but with NO prior history for anything to compare against."""
    rows = []
    start = sim.START_TS + EVENT_DAY * sim.DAY + 10 * 3600
    for i, t in enumerate(np.sort(start + rng.uniform(0, 6 * 3600, N_EVENT + 120))):
        fresh = i % 3 == 0
        r = _txn(w, rng, t, label, int(rng.integers(0, w.n_customers)), fresh=fresh)
        r["scenario"] = "gn_cold_start"
        rows.append(r)
    return rows


def warm_twin(w, rng, label: str) -> list[dict]:
    """The cold-start merchant's control: identical opening-day traffic, but on
    a merchant with six days of ordinary history behind it. Any score gap
    between the two is attributable to having no history, which is the only
    thing that differs."""
    rows = _history(w, rng, label, range(20, 26))
    start = sim.START_TS + EVENT_DAY * sim.DAY + 10 * 3600
    for i, t in enumerate(np.sort(start + rng.uniform(0, 6 * 3600, N_EVENT + 120))):
        fresh = i % 3 == 0
        r = _txn(w, rng, t, label, int(rng.integers(0, w.n_customers)), fresh=fresh)
        r["scenario"] = "gn_warm_twin"
        rows.append(r)
    return rows


def positive_control(w, rng, label: str) -> list[dict]:
    """A real card-testing wave at the same volume and window as the surge. If
    this does not fire, the harness cannot detect an attack it was handed and
    the whole run is void."""
    rows = _history(w, rng, label, HIST_DAYS)
    devs = [sim.oid("d", f"gnct{label}{i}") for i in range(3)]
    ips = [sim.oid("ip", f"gnct{label}{i}") for i in range(2)]
    start = sim.START_TS + EVENT_DAY * sim.DAY + 13 * 3600
    for i, t in enumerate(np.sort(start + rng.uniform(0, 4 * 3600, N_EVENT))):
        r = _txn(w, rng, t, label, int(rng.integers(0, w.n_customers)))
        r.update(customer_id=sim.oid("c", f"gnct{label}{i % 60}"),
                 device_id=str(rng.choice(devs)), ip=str(rng.choice(ips)),
                 instrument_id=sim.oid("pi", f"gnctpi{label}{i}"),
                 customer_created_day=sim._attack_created_day(
                     rng, EVENT_DAY, sim.AGED_SHARE_CARD_TESTING),
                 payment_method="card",
                 amount=round(float(rng.choice([10, 25, 49, 99]) * rng.uniform(1, 30)), 2),
                 is_fraud=1, scenario="gn_control_cardtest")
        rows.append(r)
    return rows


SCENARIOS = [("marketing_surge", marketing_surge, False),
             ("cold_start", cold_start, False),
             ("warm_twin", warm_twin, False),
             ("control_cardtest", positive_control, True)]


def evaluate(scored: pd.DataFrame, labels: list[str]) -> dict[str, dict]:
    out = {}
    for label in labels:
        d = scored[scored.merchant_id == label].sort_values("ts")
        stream = StreamingSpikeDetector()
        fired, peak_z, restricted, impacted = False, 0.0, 0, 0.0
        for r in d.itertuples(index=False):
            z = stream.spike_z(r.merchant_id)
            peak_z = max(peak_z, z)
            if stream.update(r.merchant_id, int(r.ts), float(r.p)) is not None:
                fired = True
            sg = RiskSignals(
                p_fraud=float(r.p), spike_z=z, component_size=float(r.component_size),
                rule_hits=evaluate_rules(
                    device_account_count=float(r.device_account_count),
                    ip_account_count=float(r.ip_account_count),
                    instrument_customer_count=float(r.instrument_customer_count),
                    cust_txn_5m=float(r.cust_txn_5m)))
            risk, conf, _ = fuse_for_policy(sg)
            dec = decide(risk_score=risk, confidence=conf, merchant_in_spike=fired)
            if dec.action is Action.RESTRICT:
                restricted += 1
                if not bool(r.is_fraud):
                    impacted += float(r.amount)
        ev = d[d.scenario.str.startswith("gn_") & (d.scenario != "gn_baseline")]
        out[label] = {
            "fired": fired, "peak_z": round(peak_z, 2),
            "event_txns": int(len(ev)),
            "mean_p_event": round(float(ev.p.mean()), 4) if len(ev) else 0.0,
            "flagged_rate_event": round(float((ev.p >= 0.5).mean()), 4) if len(ev) else 0.0,
            "restricted": restricted, "legit_inr_impacted": round(impacted, 2),
            "mean_first_seen_device": round(float(ev.is_first_seen_device.mean()), 3) if len(ev) else 0.0,
            "median_age_days": round(float(ev.customer_age_days.median()), 1) if len(ev) else 0.0,
        }
    return out


def run_seed(seed: int, model_name: str) -> pd.DataFrame:
    std = build_features(sim.generate(seed=seed))
    train, cal, _ = temporal_split(std)
    model = build_gbdt(model_name, pos_weight(train.is_fraud))
    model.fit(train[FEATURE_COLS], train.is_fraud)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(model.predict_proba(cal[FEATURE_COLS])[:, 1], cal.is_fraud)

    w, rng = _world(seed), sim.RNG(seed + 13000)
    rows, meta = [], []
    for name, fn, is_attack in SCENARIOS:
        label = f"gn_{name[:9]}"
        rows += fn(w, rng, label)
        meta.append({"merchant": label, "scenario": name, "is_attack": is_attack})

    labels = [m["merchant"] for m in meta]
    combined = pd.concat([sim.generate(seed=seed), pd.DataFrame(rows)],
                         ignore_index=True).sort_values("ts").reset_index(drop=True)
    sub = build_features(combined)
    sub = sub[sub.merchant_id.isin(labels)].copy()
    sub["p"] = iso.predict(model.predict_proba(sub[FEATURE_COLS])[:, 1])
    res = evaluate(sub, labels)
    return pd.DataFrame([{"seed": seed, **m, **res[m["merchant"]]} for m in meta])


def main():
    model_name = get_selected_model_name()
    print("=== do legitimate GROWTH bursts false-alarm us? ===")
    print(f"model {model_name} | shipped cutoffs | no retraining | seeds {SEEDS}")
    print("Criteria were pre-registered in this module's docstring and committed")
    print("before the first run.\n")

    tab = pd.concat([run_seed(s, model_name) for s in SEEDS], ignore_index=True)

    print("  scenario           fired   mean p   flagged   peak z   first-seen dev   median age")
    for name, _, _ in SCENARIOS:
        r = tab[tab.scenario == name]
        print("  %-17s %d / %-3d  %6.3f   %6.3f   %6.2f   %14.2f   %10.1f"
              % (name, int(r.fired.sum()), len(r), r.mean_p_event.mean(),
                 r.flagged_rate_event.mean(), r.peak_z.max(),
                 r.mean_first_seen_device.mean(), r.median_age_days.median()))
    print()

    ctl = tab[tab.is_attack]
    surge = tab[tab.scenario == "marketing_surge"]
    cold = tab[tab.scenario == "cold_start"]
    warm = tab[tab.scenario == "warm_twin"]
    n_ctl = int(ctl.fired.sum())
    gap = float(cold.mean_p_event.mean() - warm.mean_p_event.mean())

    print("--- cold-start penalty: is a merchant punished for having no history? ---")
    print("  cold-start mean p %.4f  vs  warm twin mean p %.4f  ->  gap %+.4f"
          % (cold.mean_p_event.mean(), warm.mean_p_event.mean(), gap))
    print("  (identical opening-day traffic; only the presence of history differs)\n")

    if n_ctl < len(ctl):
        verdict = ("VOID. The positive control - a real card-testing wave at the same "
                   "volume and window - fired on only %d of %d seeds, so this harness "
                   "cannot detect an attack it was handed and nothing below it is "
                   "evidence. Pre-registered outcome: publish nothing from this run."
                   % (n_ctl, len(ctl)))
    elif surge.fired.any():
        seeds_hit = sorted(int(s) for s in surge[surge.fired].seed)
        verdict = (
            "A LEGITIMATE MARKETING SURGE FIRES THE DETECTOR, on %d of %d seeds (%s). "
            "This is a real false-alarm mode and the pre-registered response is to "
            "publish it as one, not to patch it: adding a marketing surge to training "
            "under deadline is the failure-log 29 move, it invalidates the frozen "
            "evaluation, and it needs a fresh held-out set to mean anything. Account "
            "age was held DEGENERATE against card testing by construction (median %.1f "
            "vs %.1f days), so this is not the label proxy from failure-log 21 - a wave "
            "of first-seen devices and novel instruments is enough on its own. "
            "Legitimate INR restricted: %s."
            % (len(seeds_hit), len(surge), ", ".join("seed %d" % s for s in seeds_hit),
               surge.median_age_days.median(), ctl.median_age_days.median(),
               format(surge.legit_inr_impacted.sum(), ",.0f")))
    else:
        verdict = (
            "THE SURGE DOES NOT FIRE, and age was not what saved it. Across %d seeds a "
            "legitimate wave of 150 brand-new customers - first-seen device and IP on "
            "%.0f%% of its transactions, one novel instrument per transaction, the same "
            "compressed window as card testing, and account ages drawn from the SAME "
            "mixture (median %.1f vs the control's %.1f days) - never entered spike "
            "state, while the control fired %d/%d at a mean score of %.3f against the "
            "surge's %.3f. With age held degenerate the discrimination had to come from "
            "entity structure, and it did. SCOPE: one scenario at one intensity on our "
            "own generator. It says the failure-21 fix holds under a targeted probe, "
            "NOT that we handle legitimate growth in general."
            % (len(surge), 100 * surge.mean_first_seen_device.mean(),
               surge.median_age_days.median(), ctl.median_age_days.median(),
               n_ctl, len(ctl), ctl.mean_p_event.mean(), surge.mean_p_event.mean()))

    cold_note = (
        "COLD START: a brand-new merchant scores %+.4f against an identical warm twin. %s"
        % (gap,
           ("That is an architectural gap rather than a scenario gap - the pre-registered "
            "response is a cold-start rule (route a new merchant's first N transactions "
            "to review regardless of score), stated as a finding rather than shipped "
            "under deadline."
            if gap > 0.02 else
            "No systematic penalty: the cold-start defaults in builder.py are neutral "
            "where it matters (geo_mismatch 0, is_new_device_for_customer 0, "
            "amount_dev_ratio 1.0 with no history), and only the first-seen device/IP "
            "flags fire - which the warm twin carries too.")))

    print("=== verdict ===\n" + verdict + "\n\n" + cold_note)
    tab.to_csv(OUT / "growth_negatives.csv", index=False)
    json.dump({"seeds": SEEDS, "control_fired": n_ctl,
               "surge_fired": int(surge.fired.sum()),
               "cold_start_score_gap": round(gap, 4),
               "verdict": verdict, "cold_start_note": cold_note,
               "rows": tab.to_dict("records")},
              open(OUT / "growth_negatives.json", "w"), indent=2)
    print(f"\nwrote {OUT / 'growth_negatives.csv'}")


if __name__ == "__main__":
    main()
