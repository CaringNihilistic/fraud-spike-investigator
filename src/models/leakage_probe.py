"""P0: adversarial self-audit of our OWN evaluation.

WHY THIS EXISTS
---------------
Every number in this repo is measured on data we generated. That makes one
failure mode structurally likely: the simulator can encode the label into a
feature, and the model then scores well by reading our own answer key rather
than by detecting fraud.

We already found and published one instance of this at the AGENT layer
(failure-log 19: entity IDs were self-labelling, `pi_STOLEN_*`; de-labelling
them cost us 10/10 -> 5/10 correct-cause). This module runs the same attack
one layer down, against the ML evaluation, and it FINDS SOMETHING. See the
README "Leakage self-audit" section and failure-log 21.

WHAT IT MEASURES
----------------
  1. Single-feature PR-AUC for all 22 features, same recipe as the ablation
     (train -> isotonic on the calibration slice -> score the test slice).
     A single feature scoring near the full pipeline is a label proxy.
  2. Feature-SET comparisons: full 22 vs the two suspected proxies alone vs
     the 22 minus those two vs pure entity-sharing.
  3. The generator-side evidence: distribution of the suspect features by
     scenario. If attack rows differ from legitimate rows by construction
     rather than by behaviour, this is where it shows.

It deliberately reuses ablation._fit_eval rather than defining a second
training recipe - two definitions of "the same measurement" is how we got
failure-log 14.

Run: python -m src.models.leakage_probe
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
from src.sim.simulator import generate  # noqa: E402

OUT = Path("artifacts_out")
OUT.mkdir(exist_ok=True)

# The two features this probe puts on trial. Named by what we SUSPECT them of,
# not by what bucket the ablation happens to file them under.
LABEL_PROXY_SUSPECTS = ["customer_age_days", "amount_dev_ratio"]

# Entity CORRELATION proper: "this device is shared across N accounts". This is
# the signal the README's headline claim is actually about, isolated from the
# per-customer profile features that share its ablation bucket.
ENTITY_SHARING = ["device_account_count", "ip_account_count",
                  "instrument_customer_count", "component_size"]
FIRST_SEEN = ["is_new_device_for_customer", "is_first_seen_device", "is_first_seen_ip"]

FEATURE_SETS = [
    ("full_22", FEATURE_COLS),
    ("label_proxy_suspects_only", LABEL_PROXY_SUSPECTS),
    ("full_22_minus_suspects", [c for c in FEATURE_COLS if c not in LABEL_PROXY_SUSPECTS]),
    ("entity_sharing_only", ENTITY_SHARING),
    ("entity_sharing_plus_first_seen", ENTITY_SHARING + FIRST_SEEN),
    ("suspects_plus_entity_sharing", LABEL_PROXY_SUSPECTS + ENTITY_SHARING),
]


def main():
    print("=== leakage probe: simulate + features (same pipeline as train/ablation) ===")
    df = build_features(generate(seed=7))
    train, cal, test = temporal_split(df)
    model_name = get_selected_model_name()
    prevalence = float(test.is_fraud.mean())
    print(f"model family: {model_name} | test rows {len(test)} | "
          f"test fraud prevalence {prevalence:.4%} "
          f"(random-baseline PR-AUC = {prevalence:.4f})")

    # ---------------------------------------------------- 1. single features
    print("\n--- 1. single-feature PR-AUC (all 22, same train/calibrate/test recipe) ---")
    singles = []
    for col in FEATURE_COLS:
        m, _ = _fit_eval([col], train, cal, test, model_name)
        singles.append({"feature": col, "pr_auc": m["pr_auc"]})
    singles_df = pd.DataFrame(singles).sort_values("pr_auc", ascending=False)
    print(singles_df.to_string(index=False))

    # ---------------------------------------------------- 2. feature sets
    print("\n--- 2. feature-SET PR-AUC ---")
    sets = []
    for label, cols in FEATURE_SETS:
        m, _ = _fit_eval(cols, train, cal, test, model_name)
        sets.append({"feature_set": label, "n_features": len(cols), "pr_auc": m["pr_auc"],
                     "precision": m["precision"], "recall": m["recall"]})
    sets_df = pd.DataFrame(sets)
    print(sets_df.to_string(index=False))

    # ---------------------------------------------------- 3. generator evidence
    print("\n--- 3. generator-side evidence: suspect features by scenario ---")
    by_scen = (df.groupby("scenario")[LABEL_PROXY_SUSPECTS]
                 .median().round(2).sort_values("customer_age_days"))
    by_scen["n_rows"] = df.groupby("scenario").size()
    print(by_scen.to_string())

    # ---------------------------------------------------- verdict
    full = next(r["pr_auc"] for r in sets if r["feature_set"] == "full_22")
    proxy = next(r["pr_auc"] for r in sets if r["feature_set"] == "label_proxy_suspects_only")
    minus = next(r["pr_auc"] for r in sets if r["feature_set"] == "full_22_minus_suspects")
    sharing = next(r["pr_auc"] for r in sets if r["feature_set"] == "entity_sharing_only")
    both = next(r["pr_auc"] for r in sets if r["feature_set"] == "suspects_plus_entity_sharing")

    verdict = {
        "model_family": model_name,
        "test_fraud_prevalence": round(prevalence, 6),
        "pr_auc_full_22": full,
        "pr_auc_two_suspect_features_only": proxy,
        "pr_auc_full_minus_suspects": minus,
        "pr_auc_entity_sharing_only": sharing,
        "gap_full_vs_two_features": round(full - proxy, 4),
        "marginal_gain_of_sharing_over_suspects": round(both - proxy, 4),
        "two_features_reproduce_headline": bool(abs(full - proxy) < 0.02),
        "conclusion": (
            "Two features reproduce the 22-feature headline. The simulator sets attack "
            "accounts' creation date to the attack day, so customer_age_days is close to a "
            "direct label encoding; ambient fraud is generated as a multiple of a legitimate "
            "amount, so amount_dev_ratio is a second proxy. Entity-SHARING features carry "
            "real structure on their own but add almost nothing on top of the proxies, so the "
            "ablation's stage-2 -> stage-3 jump must NOT be read as 'entity correlation is "
            "where the lift comes from'."),
    }
    print("\n=== verdict ===")
    print(json.dumps(verdict, indent=2))

    singles_df.to_csv(OUT / "leakage_probe_single_features.csv", index=False)
    sets_df.to_csv(OUT / "leakage_probe_feature_sets.csv", index=False)
    by_scen.to_csv(OUT / "leakage_probe_by_scenario.csv")
    json.dump(verdict, open(OUT / "leakage_probe.json", "w"), indent=2)
    print(f"\nwrote {OUT / 'leakage_probe_single_features.csv'}, "
          f"{OUT / 'leakage_probe_feature_sets.csv'}, "
          f"{OUT / 'leakage_probe_by_scenario.csv'}, {OUT / 'leakage_probe.json'}")


if __name__ == "__main__":
    main()
