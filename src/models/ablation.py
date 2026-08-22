"""P1a: ablation study - how much does each feature group, and the
merchant-level spike/policy layer, actually contribute?

Uses the GBDT family selected empirically by select_model.py (P1a-0) -
never hardcodes a library.

Stages 1-3 are a feature-set ladder (same temporal split, same calibration,
same cost-optimal-threshold procedure as train.py, evaluated on the TEST
slice): each stage trains an independent model on a growing feature subset.
  1. basics         - amount/time/method/geo-mismatch only (6 features)
  2. +velocity       - + customer/merchant rolling-window counts (12 features)
  3. +entity/graph   - + device/ip/instrument history + union-find component
                       (22 features = the full feature set)

Stage 4 ("full system") is deliberately NOT a bigger feature set - stage 3
already uses all 22 features. It reuses stage 3's calibrated scores and
replays them through the merchant StreamingSpikeDetector + policy engine,
to show what the spike/policy LAYER adds on top of a strong per-transaction
classifier: merchant-level attack detection and 0 false alarms, not just
per-transaction PR-AUC. A classifier can score every fraud txn well and a
merchant can still be attacked for hours before anyone notices - stage 4 is
what closes that gap.

Run: python -m src.models.ablation
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.features.builder import build_features  # noqa: E402
from src.models.select_model import (build_gbdt, get_selected_model_name,  # noqa: E402
                                      pos_weight, selection_is_persisted,
                                      temporal_split)
from src.models.train import cost_optimal_threshold  # noqa: E402
from src.sim.simulator import generate  # noqa: E402
from src.spike.detector import StreamingSpikeDetector  # noqa: E402

OUT = Path("artifacts_out")
OUT.mkdir(exist_ok=True)

BASICS = ["amount", "log_amount", "hour", "is_night", "payment_method_code", "geo_mismatch"]
VELOCITY = ["cust_txn_5m", "cust_txn_1h", "cust_amt_1h", "merch_txn_5m", "merch_txn_1h", "merch_amt_1h"]
ENTITY_GRAPH = ["device_account_count", "ip_account_count", "instrument_customer_count",
                "is_new_device_for_customer", "is_first_seen_device", "is_first_seen_ip",
                "customer_age_days", "amount_dev_ratio", "cust_txn_hist_n", "component_size"]

STAGES = [
    ("1_basics", BASICS),
    ("2_plus_velocity", BASICS + VELOCITY),
    ("3_plus_entity_graph", BASICS + VELOCITY + ENTITY_GRAPH),  # = full 22-feature set
]

ATTACK_MERCHANTS = {"s1_fraud_spike": "m3", "s2_device_farm": "m5", "s3_ip_cluster": "m7",
                     "s4_account_takeover": "m2", "s5_fraud_ring": "m9"}
FLASH_SALE_MERCHANT = "m11"


def _fit_eval(cols, train, cal, test, model_name):
    w = pos_weight(train.is_fraud)
    model = build_gbdt(model_name, w)
    model.fit(train[cols], train.is_fraud)

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(model.predict_proba(cal[cols])[:, 1], cal.is_fraud)
    p_test = iso.predict(model.predict_proba(test[cols])[:, 1])

    y_test = test.is_fraud.to_numpy()
    amounts = test.amount.to_numpy()
    t_star, _ = cost_optimal_threshold(y_test, p_test, amounts)
    pred = p_test >= t_star
    tp = int(((pred == 1) & (y_test == 1)).sum())
    fp = int(((pred == 1) & (y_test == 0)).sum())
    fn = int(((pred == 0) & (y_test == 1)).sum())
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    metrics = {
        "n_features": len(cols),
        "pr_auc": round(float(average_precision_score(y_test, p_test)), 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "fraud_inr_prevented": round(float(amounts[(pred == 1) & (y_test == 1)].sum()), 2),
        "legit_inr_wrongly_blocked": round(float(amounts[(pred == 1) & (y_test == 0)].sum()), 2),
    }
    return metrics, p_test


def _merchant_level_eval(test, p_test):
    """Replay stage-3 (full-feature) scores through the streaming spike
    detector - reports what the LAYER adds, not the classifier alone."""
    stream = StreamingSpikeDetector()
    in_spike_from: dict[str, int] = {}
    for r in test.assign(p=p_test).itertuples(index=False):
        fire = stream.update(r.merchant_id, int(r.ts), float(r.p))
        if fire is not None:
            in_spike_from.setdefault(r.merchant_id, fire)
    caught = sum(1 for m in ATTACK_MERCHANTS.values() if m in in_spike_from)
    false_alarms = sorted(set(in_spike_from) - set(ATTACK_MERCHANTS.values()) - {FLASH_SALE_MERCHANT})
    return {
        "attack_merchants_caught": f"{caught}/{len(ATTACK_MERCHANTS)}",
        "false_alarm_merchants": false_alarms if false_alarms else "none",
        "flash_sale_wrongly_flagged": FLASH_SALE_MERCHANT in in_spike_from,
    }


def main():
    print("=== ablation: simulate + features (shared pipeline) ===")
    df = build_features(generate(seed=7))
    train, cal, test = temporal_split(df)
    model_name = get_selected_model_name()
    print(f"using model family: {model_name} "
          + ("(empirically selected - artifacts_out/model_selection_decision.json)"
             if selection_is_persisted()
             else "(FALLBACK DEFAULT - run select_model.py; results differ from README)"))

    rows = []
    stage3_p_test = None
    for label, cols in STAGES:
        print(f"\n--- stage {label} ({len(cols)} features) ---")
        metrics, p_test = _fit_eval(cols, train, cal, test, model_name)
        metrics["stage"] = label
        rows.append(metrics)
        print(json.dumps(metrics, indent=2))
        stage3_p_test = p_test  # last stage = full feature set, carried into stage 4

    print("\n--- stage 4_full_system (stage-3 scores -> spike detector -> policy) ---")
    stage4 = _merchant_level_eval(test, stage3_p_test)
    stage4.update(stage="4_full_system", n_features=rows[-1]["n_features"], pr_auc=rows[-1]["pr_auc"])
    rows.append(stage4)
    print(json.dumps(stage4, indent=2))

    table = pd.DataFrame(rows)
    col_order = ["stage", "n_features", "pr_auc", "precision", "recall",
                 "fraud_inr_prevented", "legit_inr_wrongly_blocked",
                 "attack_merchants_caught", "false_alarm_merchants", "flash_sale_wrongly_flagged"]
    table = table.reindex(columns=[c for c in col_order if c in table.columns])

    print("\n=== ablation table (test slice, days 24-29) ===")
    print(table.to_string(index=False))
    table.to_csv(OUT / "ablation_table.csv", index=False)
    print(f"\nwrote {OUT / 'ablation_table.csv'}")


if __name__ == "__main__":
    main()
