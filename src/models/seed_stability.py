"""P2: is the headline a result, or one lucky world?

Every number in the README comes from seed 7 - a single simulated world,
n=1, no interval. On 712 test positives that is not enough to distinguish
PR-AUC 0.93 from 0.91, and we were quoting four decimal places.

This re-runs the ENTIRE pipeline (simulate -> features -> temporal split ->
train -> isotonic -> cost-optimal threshold -> merchant-level replay) across
several independent worlds. The scenario-to-merchant assignment is fixed by
construction (m3 card-testing, m5 device farm, m7 IP cluster, m2 ATO,
m9 fraud ring, m11 flash sale); only the random draws differ, so each seed is
a fresh sample of the same generative process.

The merchant-level rows are the point. "5/5 attacks caught, 0 false alarms,
flash sale never flagged" is a much stronger claim across five worlds than in
one, and unlike PR-AUC it is not affected by the label proxies documented in
failure-log 21 - the flash sale is legitimate traffic in every world.

Seeds are fixed here, in advance, and reported in full: no dropping a world
because it came out badly.

Run: python -m src.models.seed_stability
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.features.builder import FEATURE_COLS, build_features  # noqa: E402
from src.models.ablation import _fit_eval, _merchant_level_eval  # noqa: E402
from src.models.select_model import get_selected_model_name, temporal_split  # noqa: E402
from src.sim.simulator import generate  # noqa: E402

OUT = Path("artifacts_out")
OUT.mkdir(exist_ok=True)

# Fixed in advance. Seed 7 is the README's world and is included so the run
# doubles as a reproducibility check on the headline number.
SEEDS = [7, 11, 23, 42, 101]


def main():
    model_name = get_selected_model_name()
    print(f"=== seed stability: {len(SEEDS)} independent worlds, model {model_name} ===")
    print(f"seeds (fixed in advance): {SEEDS}")

    rows = []
    for seed in SEEDS:
        df = build_features(generate(seed=seed))
        train, cal, test = temporal_split(df)
        metrics, p_test = _fit_eval(FEATURE_COLS, train, cal, test, model_name)
        merch = _merchant_level_eval(test, p_test)
        row = {
            "seed": seed,
            "test_rows": len(test),
            "test_fraud_prevalence": round(float(test.is_fraud.mean()), 5),
            "pr_auc": metrics["pr_auc"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "legit_inr_wrongly_blocked": metrics["legit_inr_wrongly_blocked"],
            "attacks_caught": merch["attack_merchants_caught"],
            "false_alarm_merchants": ("none" if merch["false_alarm_merchants"] == "none"
                                       else ",".join(merch["false_alarm_merchants"])),
            "flash_sale_wrongly_flagged": merch["flash_sale_wrongly_flagged"],
        }
        rows.append(row)
        print(f"  seed {seed:>4}: PR-AUC {row['pr_auc']:.4f} | P {row['precision']:.3f} "
              f"| R {row['recall']:.3f} | attacks {row['attacks_caught']} "
              f"| false alarms {row['false_alarm_merchants']} "
              f"| flash flagged {row['flash_sale_wrongly_flagged']}")

    table = pd.DataFrame(rows)
    print()
    print(table.to_string(index=False))

    pr = table.pr_auc
    summary = {
        "seeds": SEEDS,
        "n_worlds": len(SEEDS),
        "model_family": model_name,
        "pr_auc_mean": round(float(pr.mean()), 4),
        "pr_auc_std": round(float(pr.std(ddof=1)), 4),
        "pr_auc_min": round(float(pr.min()), 4),
        "pr_auc_max": round(float(pr.max()), 4),
        "precision_mean": round(float(table.precision.mean()), 4),
        "recall_mean": round(float(table.recall.mean()), 4),
        # The claims that actually matter, and the only ones we should state
        # without an interval attached:
        "worlds_with_all_5_attacks_caught": int((table.attacks_caught == "5/5").sum()),
        "worlds_with_zero_false_alarms": int((table.false_alarm_merchants == "none").sum()),
        "worlds_where_flash_sale_was_flagged": int(table.flash_sale_wrongly_flagged.sum()),
    }
    print()
    print("=== summary ===")
    print(json.dumps(summary, indent=2))
    print()
    print(f"Report PR-AUC as {summary['pr_auc_mean']:.3f} "
          f"[{summary['pr_auc_min']:.3f}-{summary['pr_auc_max']:.3f}] over "
          f"{len(SEEDS)} worlds, not as a single 4-decimal number.")

    table.to_csv(OUT / "seed_stability.csv", index=False)
    json.dump(summary, open(OUT / "seed_stability.json", "w"), indent=2)
    print(f"wrote {OUT / 'seed_stability.csv'}, {OUT / 'seed_stability.json'}")


if __name__ == "__main__":
    main()
