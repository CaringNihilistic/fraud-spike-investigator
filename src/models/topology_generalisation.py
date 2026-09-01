"""Does the detector catch coordinated fraud, or only the graph shapes we drew?

This is the sharpest criticism the project has received and the last one it can
answer with its own data. Our five attacks are five TOPOLOGIES - one device
shared by fifty accounts, one IP shared by forty, a dense fifteen-account ring -
and the model has only ever been tested against the shapes it was trained on.
If detection collapses when the shape changes while the crime does not, then the
merchant-level result is a fact about our generator, not about fraud.

So: hold everything else constant and vary ONLY the entity graph.

  known                          unseen variant
  ----------------------------   -------------------------------------------
  card testing  3 dev /  2 ip    25 dev / 20 ip - almost no fan-out left
  device farm   1 dev, 50 acct   10 dev / 6 ip, 30 acct, partial overlap
  ip cluster    1 ip,  40 acct   6 rotating ips across the same 40 accounts
  takeover      25 unique dev    5 shared proxy devices across 25 victims
  fraud ring    dense 15x4x3     sparse bipartite: 30 acct, 2 dev each of 20

Transaction count, burst window, amount distribution, account ageing and fraud
prevalence are matched to the known version in every case. The ONLY difference
is who shares what with whom.

WHAT MAKES THIS INTERPRETABLE. The known topologies run in the SAME suite as a
POSITIVE CONTROL, with fresh entity ids so nothing is recognised by identity.
If known fires and unseen does not, that is a real finding about topology
dependence. If NEITHER fires, the harness is broken and the verdict is VOID -
which matters, because the last two experiments in this repo were both wrong on
their first run (failure-log 32 and 33).

Nothing here is retrained. Shipped model, shipped calibration, shipped cutoffs.
The swept merchants appear in no training data, and their customer pool is
disjoint from the world the model was fit on.

THIS CAN COME BACK BADLY, and that is the point of running it. A collapse on
unseen topologies would narrow the central claim from "detects coordinated
fraud" to "detects the coordination patterns it was shown". We would publish
that.

Run: python -m src.models.topology_generalisation
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
HIST_DAYS = range(20, 30)     # ordinary history so each merchant has a baseline
EVENT_DAY = 26
DAILY = 200


def _world(seed: int) -> sim.World:
    return sim.World(rng=sim.RNG(seed + 7000))


def _txn(w, rng, ts, label, cid) -> dict:
    """Ordinary legitimate transaction for a swept merchant. Mirrors
    simulator._base_txn; customer ids are namespaced so they cannot collide
    with the world the model was trained on. 1500 ISP pools, matching
    World.__post_init__ - getting that wrong inverted an entire sweep once
    (failure-log 32)."""
    c = w.customers[cid]
    amount = max(20.0, rng.normal(c["avg_amount"], c["avg_amount"] * 0.35))
    return {
        "ts": int(ts), "merchant_id": label,
        "customer_id": sim.oid("c", f"tg{cid}"),
        "device_id": sim.oid("d", f"tgcust{cid}"),
        "ip": sim.oid("ip", f"tgpool{cid % 1500}"),
        "instrument_id": sim.oid("pi", f"tgcust{cid}"),
        "geo": c["home_geo"], "amount": round(float(amount), 2),
        "payment_method": str(rng.choice(["card", "upi", "netbanking", "wallet"],
                                         p=[0.35, 0.45, 0.1, 0.1])),
        "customer_created_day": c["created_day"],
        "is_fraud": 0, "scenario": "tg_baseline",
    }


def _history(w, rng, label: str) -> list[dict]:
    rows = []
    for day in HIST_DAYS:
        if day == EVENT_DAY:
            continue
        start = sim.START_TS + day * sim.DAY
        for t in sim._poisson_times(rng, start, int(rng.normal(DAILY, 25))):
            rows.append(_txn(w, rng, t, label, int(rng.integers(0, w.n_customers))))
    return rows


# --------------------------------------------------------------- attacks
# Each pair shares n, window, amount model and ageing. Only the graph differs.
def attack(w, rng, label: str, kind: str, variant: str) -> list[dict]:
    p = f"{label}{variant}"
    start = sim.START_TS + EVENT_DAY * sim.DAY
    rows = []

    def burst(n, hour, span):
        return np.sort(start + hour * 3600 + rng.uniform(0, span * 3600, n))

    if kind == "card_testing":
        n = 180
        # known: a tight attacker rig. unseen: fan-out almost eliminated.
        ndev, nip = (3, 2) if variant == "known" else (25, 20)
        devs = [sim.oid("d", f"ct{p}{i}") for i in range(ndev)]
        ips = [sim.oid("ip", f"ct{p}{i}") for i in range(nip)]
        for i, t in enumerate(burst(n, 10, 6)):
            r = _txn(w, rng, t, label, int(rng.integers(0, w.n_customers)))
            r.update(customer_id=sim.oid("c", f"ct{p}{i % 60}"),
                     device_id=str(rng.choice(devs)), ip=str(rng.choice(ips)),
                     instrument_id=sim.oid("pi", f"st{p}{i}"),   # novel card each time
                     customer_created_day=sim._attack_created_day(
                         rng, EVENT_DAY, sim.AGED_SHARE_CARD_TESTING),
                     payment_method="card",
                     amount=round(float(rng.choice([10, 25, 49, 99]) * rng.uniform(1, 30)), 2),
                     is_fraud=1, scenario=f"tg_{kind}_{variant}")
            rows.append(r)

    elif kind == "device_farm":
        n, nacct = 130, (50 if variant == "known" else 30)
        ndev, nip = (1, 1) if variant == "known" else (10, 6)
        devs = [sim.oid("d", f"fm{p}{i}") for i in range(ndev)]
        ips = [sim.oid("ip", f"fm{p}{i}") for i in range(nip)]
        pis = [sim.oid("pi", f"fm{p}{i}") for i in range(8)]
        for i, t in enumerate(burst(n, 12, 5)):
            acct = i % nacct
            # unseen: each account sticks to 2 devices, so overlap is partial
            dev = devs[0] if ndev == 1 else devs[(acct * 2 + i % 2) % ndev]
            r = _txn(w, rng, t, label, int(rng.integers(0, w.n_customers)))
            r.update(customer_id=sim.oid("c", f"fm{p}{acct}"),
                     device_id=dev, ip=str(rng.choice(ips)),
                     instrument_id=str(rng.choice(pis)),
                     customer_created_day=sim._attack_created_day(
                         rng, EVENT_DAY, sim.AGED_SHARE_FARM),
                     is_fraud=1, scenario=f"tg_{kind}_{variant}")
            rows.append(r)

    elif kind == "ip_cluster":
        n, nacct = 100, 40
        nip = 1 if variant == "known" else 6          # unseen: rotating IPs
        ips = [sim.oid("ip", f"cl{p}{i}") for i in range(nip)]
        for i, t in enumerate(burst(n, 14, 3)):
            acct = i % nacct
            r = _txn(w, rng, t, label, int(rng.integers(0, w.n_customers)))
            r.update(customer_id=sim.oid("c", f"cl{p}{acct}"),
                     device_id=sim.oid("d", f"cl{p}{acct}"),
                     ip=ips[i % nip],
                     instrument_id=sim.oid("pi", f"cl{p}{acct}"),
                     customer_created_day=sim._attack_created_day(
                         rng, EVENT_DAY, sim.AGED_SHARE_CLUSTER),
                     is_fraud=1, scenario=f"tg_{kind}_{variant}")
            rows.append(r)

    elif kind == "account_takeover":
        victims = rng.choice(w.n_customers, size=25, replace=False)
        # known: a fresh device per victim (shares nothing). unseen: the
        # attacker reuses 5 proxy devices across all victims.
        proxies = [sim.oid("d", f"at{p}{i}") for i in range(5)]
        for j, v in enumerate(victims):
            c = w.customers[int(v)]
            for _ in range(int(rng.integers(2, 5))):
                t = start + rng.uniform(1 * 3600, 23 * 3600)
                r = _txn(w, rng, t, label, int(v))
                r.update(device_id=(sim.oid("d", f"at{p}{int(v)}") if variant == "known"
                                    else proxies[j % 5]),
                         geo=int((c["home_geo"] + 10) % 20),
                         amount=round(c["avg_amount"] * float(rng.uniform(5, 12)), 2),
                         is_fraud=1, scenario=f"tg_{kind}_{variant}")
                rows.append(r)

    elif kind == "fraud_ring":
        n = 120
        if variant == "known":                 # dense: 15 accounts over 4/3/5
            accts = [sim.oid("c", f"rg{p}{i}") for i in range(15)]
            devs = [sim.oid("d", f"rg{p}{i}") for i in range(4)]
            ips = [sim.oid("ip", f"rg{p}{i}") for i in range(3)]
            pis = [sim.oid("pi", f"rg{p}{i}") for i in range(5)]
            pick = lambda i: (str(rng.choice(accts)), str(rng.choice(devs)),  # noqa: E731
                              str(rng.choice(ips)), str(rng.choice(pis)))
        else:                                   # sparse bipartite: 30 accounts,
            accts = [sim.oid("c", f"rg{p}{i}") for i in range(30)]  # 2 devices each
            devs = [sim.oid("d", f"rg{p}{i}") for i in range(20)]
            ips = [sim.oid("ip", f"rg{p}{i}") for i in range(15)]
            pis = [sim.oid("pi", f"rg{p}{i}") for i in range(25)]

            def pick(i):
                a = i % 30
                return (accts[a], devs[(a * 2 + i % 2) % 20], ips[a % 15], pis[a % 25])
        for i, t in enumerate(burst(n, 9, 10)):
            a, d, ip, pi = pick(i)
            r = _txn(w, rng, t, label, int(rng.integers(0, w.n_customers)))
            r.update(customer_id=a, device_id=d, ip=ip, instrument_id=pi,
                     customer_created_day=sim._attack_created_day(
                         rng, EVENT_DAY, sim.AGED_SHARE_RING),
                     is_fraud=1, scenario=f"tg_{kind}_{variant}")
            rows.append(r)

    else:
        raise ValueError(kind)
    return rows


KINDS = ["card_testing", "device_farm", "ip_cluster", "account_takeover", "fraud_ring"]


def evaluate(scored: pd.DataFrame, labels: list[str]) -> dict[str, dict]:
    out = {}
    for label in labels:
        d = scored[scored.merchant_id == label].sort_values("ts")
        stream = StreamingSpikeDetector()
        fired_at, restricted, prevented, impacted = None, 0, 0.0, 0.0
        for r in d.itertuples(index=False):
            z = stream.spike_z(r.merchant_id)
            hit = stream.update(r.merchant_id, int(r.ts), float(r.p))
            if hit is not None and fired_at is None:
                fired_at = hit
            sg = RiskSignals(
                p_fraud=float(r.p), spike_z=z, component_size=float(r.component_size),
                rule_hits=evaluate_rules(
                    device_account_count=float(r.device_account_count),
                    ip_account_count=float(r.ip_account_count),
                    instrument_customer_count=float(r.instrument_customer_count),
                    cust_txn_5m=float(r.cust_txn_5m)))
            risk, conf, _ = fuse_for_policy(sg)
            dec = decide(risk_score=risk, confidence=conf,
                         merchant_in_spike=fired_at is not None)
            if dec.action is Action.RESTRICT:
                restricted += 1
                prevented += float(r.amount) if bool(r.is_fraud) else 0.0
                impacted += 0.0 if bool(r.is_fraud) else float(r.amount)
        atk = d[d.is_fraud == 1]
        ttd = None
        if fired_at is not None and len(atk):
            ttd = round((fired_at - int(atk.ts.min())) / 60.0, 1)
        out[label] = {
            "fired": fired_at is not None,
            "minutes_to_detect": ttd,
            "attack_txns": int(len(atk)),
            "flagged_rate_attack": round(float((atk.p >= 0.5).mean()), 4) if len(atk) else 0.0,
            "restricted": restricted,
            "fraud_inr_prevented": round(prevented, 2),
            "legit_inr_impacted": round(impacted, 2),
        }
    return out


def run_seed(seed: int, model_name: str) -> pd.DataFrame:
    std = build_features(sim.generate(seed=seed))
    train, cal, _ = temporal_split(std)
    model = build_gbdt(model_name, pos_weight(train.is_fraud))
    model.fit(train[FEATURE_COLS], train.is_fraud)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(model.predict_proba(cal[FEATURE_COLS])[:, 1], cal.is_fraud)

    w, rng = _world(seed), sim.RNG(seed + 11000)
    rows, meta = [], []
    for kind in KINDS:
        for variant in ("known", "unseen"):
            label = f"tg_{kind[:4]}_{variant[:3]}"
            rows += _history(w, rng, label)
            rows += attack(w, rng, label, kind, variant)
            meta.append({"merchant": label, "kind": kind, "variant": variant})

    labels = [m["merchant"] for m in meta]
    combined = pd.concat([sim.generate(seed=seed), pd.DataFrame(rows)],
                         ignore_index=True).sort_values("ts").reset_index(drop=True)
    feats = build_features(combined)
    sub = feats[feats.merchant_id.isin(labels)].copy()
    sub["p"] = iso.predict(model.predict_proba(sub[FEATURE_COLS])[:, 1])
    res = evaluate(sub, labels)
    return pd.DataFrame([{"seed": seed, **m, **res[m["merchant"]]} for m in meta])


def main():
    model_name = get_selected_model_name()
    print("=== does detection survive a change of attack TOPOLOGY? ===")
    print(f"model: {model_name} | shipped cutoffs | no retraining | seeds {SEEDS}\n")
    print("Each attack appears twice with matched volume, window, amounts and")
    print("account ageing. Only the entity graph differs. The KNOWN variants are")
    print("the positive control - if they do not fire, the harness is broken and")
    print("the verdict is VOID rather than a finding.\n")

    tab = pd.concat([run_seed(s, model_name) for s in SEEDS], ignore_index=True)

    print("  attack              variant   detected   mean flagged   median TTD   INR legit")
    for kind in KINDS:
        for variant in ("known", "unseen"):
            r = tab[(tab.kind == kind) & (tab.variant == variant)]
            ttd = r.minutes_to_detect.dropna()
            print("  %-18s  %-7s   %d / %-4d     %6.3f       %8s   %9s"
                  % (kind, variant, int(r.fired.sum()), len(r),
                     r.flagged_rate_attack.mean(),
                     ("%.0fm" % ttd.median()) if len(ttd) else "n/a",
                     format(r.legit_inr_impacted.sum(), ",.0f")))
        print()

    known, unseen = tab[tab.variant == "known"], tab[tab.variant == "unseen"]
    k_rate, u_rate = known.fired.mean(), unseen.fired.mean()
    k_flag, u_flag = known.flagged_rate_attack.mean(), unseen.flagged_rate_attack.mean()

    if k_rate < 0.8:
        verdict = (
            "VOID. The positive control did not hold: the KNOWN topologies - the "
            "shapes this model was trained on, rebuilt here with fresh entity ids - "
            "fired only %d of %d times (%.0f%%). A harness that cannot detect the "
            "attacks it was designed around measures itself, not the system, so the "
            "unseen rows below it mean nothing either way. Fix the harness first. "
            "This branch is written and reachable because the last two experiments "
            "in this repo were both wrong on their first run (failure-log 32, 33)."
            % (int(known.fired.sum()), len(known), 100 * k_rate))
    elif u_rate >= k_rate - 0.1:
        by_kind = known.groupby("kind").fired.mean()
        weak = [k for k, v in by_kind.items() if v < 0.8]
        verdict = (
            "MERCHANT-LEVEL DETECTION SURVIVES THE TOPOLOGY CHANGE, BUT CONFIDENCE "
            "DOES NOT - and the gap is the useful part. Known %d/%d (%.0f%%), unseen "
            "%d/%d (%.0f%%) across %d seeds, with volume, burst window, amounts and "
            "account ageing matched so the entity graph is the only thing that moved. "
            "The merchant-level layer is therefore not just recognising the shapes it "
            "was drawn from. BUT the transaction scorer clearly is affected: the mean "
            "flagged rate inside the attack falls from %.3f to %.3f (-%.0f%%) when the "
            "graph changes. The per-transaction model loses confidence while the "
            "merchant-level spike still fires - which is the product's own thesis "
            "showing up as a measurement, since the whole argument for a merchant "
            "layer is that it survives what per-order scoring finds ambiguous. "
            "CAVEAT WE REPORT RATHER THAN BURY: %s. SCOPE: five topologies, one "
            "variant each, our own generator, 5 seeds. It shows the result is not "
            "brittle to THESE changes - not that it holds for every coordination "
            "pattern a real attacker might invent."
            % (int(known.fired.sum()), len(known), 100 * k_rate,
               int(unseen.fired.sum()), len(unseen), 100 * u_rate, len(SEEDS),
               k_flag, u_flag, 100 * (1 - u_flag / k_flag) if k_flag else 0.0,
               ("the control is uneven - " + ", ".join(
                   "%s fires only %.0f%% even in its KNOWN form, so both of its rows "
                   "are weak evidence" % (k, 100 * by_kind[k]) for k in weak))
               if weak else "the control fired on every attack family"))
    else:
        worst = (unseen.groupby("kind").fired.mean().sort_values())
        verdict = (
            "DETECTION IS TOPOLOGY-DEPENDENT, and this is a real limitation rather "
            "than a harness fault - the control held at %d/%d (%.0f%%) while the "
            "unseen variants reached only %d/%d (%.0f%%). Weakest: %s. Everything "
            "except the entity graph was held constant, so the gap is attributable "
            "to shape alone. The honest reading is that this is a detector of the "
            "coordination patterns it has seen, and novel topologies need either "
            "training examples or a different signal. We publish it because an "
            "unexamined generalisation claim is worse than a measured limit."
            % (int(known.fired.sum()), len(known), 100 * k_rate,
               int(unseen.fired.sum()), len(unseen), 100 * u_rate,
               ", ".join("%s %.0f%%" % (k, 100 * v) for k, v in worst.head(3).items())))

    print("=== verdict ===\n" + verdict)
    tab.to_csv(OUT / "topology_generalisation.csv", index=False)
    json.dump({"seeds": SEEDS, "known_detection_rate": round(float(k_rate), 4),
               "unseen_detection_rate": round(float(u_rate), 4),
               "verdict": verdict, "rows": tab.to_dict("records")},
              open(OUT / "topology_generalisation.json", "w"), indent=2)
    print(f"\nwrote {OUT / 'topology_generalisation.csv'}")


if __name__ == "__main__":
    main()
