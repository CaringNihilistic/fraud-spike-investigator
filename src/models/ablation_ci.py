"""Paired bootstrap confidence intervals for the ablation ladder.

The ablation table reports point estimates: basics -> +velocity -> +entity/graph.
Two of those deltas are enormous and one is not, and until now the table gave a
reader no way to tell which is which. Our own Honest-limitations section says
differences below roughly +/-0.02 should not be treated as real - so the
+velocity step, which lands just above that line, was exactly the row that
needed an uncertainty statement and did not have one.

METHOD. Each stage is fit ONCE on the real temporal split (train -> isotonic on
the calibration slice -> score the test slice), exactly as ablation.py does.
Then the *test rows* are resampled with replacement, and every variant is
scored on the SAME resampled rows - a paired bootstrap. Pairing matters: the
variants share a test set, so their errors are correlated, and an unpaired
interval on the difference would be far too wide.

What this measures and what it does not: this is SAMPLING uncertainty over one
test slice. It is not generative variance - that is what
src/models/seed_stability.py measures by regenerating the world five times, and
the two are complementary. Neither is real-world variance.

N_BOOTSTRAP and the interval width were fixed before looking at any result.

Run: python -m src.models.ablation_ci
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.features.builder import build_features  # noqa: E402
from src.models.ablation import STAGES, _fit_eval  # noqa: E402
from src.models.select_model import get_selected_model_name, temporal_split  # noqa: E402
from src.sim.simulator import generate  # noqa: E402

OUT = Path("artifacts_out")
OUT.mkdir(exist_ok=True)

N_BOOTSTRAP = 2000
CI = 95.0
SEED = 7

# The claims the table makes, as explicit comparisons.
COMPARISONS = [
    ("1_basics", "2_plus_velocity"),
    ("2_plus_velocity", "3_plus_entity_graph"),
    ("1_basics", "3_plus_entity_graph"),
]


def main():
    model_name = get_selected_model_name()
    print("=== paired bootstrap CIs on the ablation ladder ===")
    print(f"model family: {model_name} | {N_BOOTSTRAP:,} resamples | {CI:.0f}% percentile CI")
    print("Fixed in advance; the test slice is resampled, the models are not refit.\n")

    df = build_features(generate())
    train, cal, test = temporal_split(df)
    y = test.is_fraud.to_numpy()

    # Score every stage once on the real split.
    scores: dict[str, np.ndarray] = {}
    for label, cols in STAGES:
        metrics, p_test = _fit_eval(cols, train, cal, test, model_name)
        scores[label] = np.asarray(p_test, dtype=float)
        print(f"  {label:<22} n={metrics['n_features']:<3} pr_auc={metrics['pr_auc']}")

    rng = np.random.default_rng(SEED)
    n = len(y)
    lo_q, hi_q = (100 - CI) / 2, 100 - (100 - CI) / 2

    # One resample of ROW INDICES per iteration, shared by every variant.
    boot: dict[str, list[float]] = {k: [] for k in scores}
    kept = 0
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue          # a degenerate resample has no defined AP
        kept += 1
        for label, p in scores.items():
            boot[label].append(float(average_precision_score(yb, p[idx])))

    rows = []
    print(f"\n--- deltas ({kept:,} usable resamples) ---")
    for a, b in COMPARISONS:
        pa, pb = np.array(boot[a]), np.array(boot[b])
        d = pb - pa
        lo, hi = np.percentile(d, [lo_q, hi_q])
        point = float(average_precision_score(y, scores[b])) - \
            float(average_precision_score(y, scores[a]))
        significant = bool(lo > 0 or hi < 0)
        rows.append({
            "from": a, "to": b,
            "delta_pr_auc": round(point, 4),
            "ci_low": round(float(lo), 4), "ci_high": round(float(hi), 4),
            "excludes_zero": significant,
        })
        verdict = "SIGNIFICANT" if significant else "not distinguishable from zero"
        print(f"  {a:<22} -> {b:<22} {point:+.4f}  "
              f"[{lo:+.4f}, {hi:+.4f}]  {verdict}")

    # Per-stage intervals, for the table itself.
    stage_rows = []
    print("\n--- per-stage AUC-PR ---")
    for label, p in scores.items():
        arr = np.array(boot[label])
        lo, hi = np.percentile(arr, [lo_q, hi_q])
        point = float(average_precision_score(y, p))
        stage_rows.append({"stage": label, "pr_auc": round(point, 4),
                           "ci_low": round(float(lo), 4), "ci_high": round(float(hi), 4)})
        print(f"  {label:<22} {point:.4f}  [{lo:.4f}, {hi:.4f}]")

    pd.DataFrame(stage_rows).to_csv(OUT / "ablation_ci_stages.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT / "ablation_ci_deltas.csv", index=False)
    json.dump({"n_bootstrap": N_BOOTSTRAP, "usable_resamples": kept,
               "ci_pct": CI, "seed": SEED, "test_rows": int(n),
               "test_positives": int(y.sum()),
               "stages": stage_rows, "deltas": rows},
              open(OUT / "ablation_ci.json", "w"), indent=2)
    print(f"\nwrote {OUT / 'ablation_ci_stages.csv'}, {OUT / 'ablation_ci_deltas.csv'}, "
          f"{OUT / 'ablation_ci.json'}")


if __name__ == "__main__":
    main()
