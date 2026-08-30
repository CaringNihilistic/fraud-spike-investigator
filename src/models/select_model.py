"""P1a-0: empirical model selection - run BEFORE the ablation study.

Trains four classifiers on the train slice with identical class weighting
and DEFAULT hyperparameters (no per-model tuning): LogisticRegression
(scaled features), XGBoost, LightGBM, CatBoost. Compares PR-AUC and
train+inference time on the VALIDATION slice (days 21-23) ONLY - the test
slice is never touched here, so selection cannot leak into the reported
test-set metrics.

Selection rule (fixed in advance, before looking at results):
  highest validation PR-AUC wins; if the top GBDTs are within
  PR_AUC_TIE_MARGIN of each other, pick by speed + maintainability and say
  so explicitly.

The winner is persisted to artifacts_out/model_selection_decision.json.
train.py and ablation.py read that file (via get_selected_model_name) and
build the SAME family with the project's tuned hyperparameters through
build_gbdt() - this module is the only place that knows how to construct
each library's estimator, so the rest of the codebase stays library-agnostic
("GBDT primary model, selected empirically", not a hardcoded import).

Run: python -m src.models.select_model
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.features.builder import FEATURE_COLS, build_features  # noqa: E402
from src.sim.simulator import generate  # noqa: E402

OUT = Path("artifacts_out")
OUT.mkdir(exist_ok=True)
DECISION_FILE = OUT / "model_selection_decision.json"

PR_AUC_TIE_MARGIN = 0.02  # fixed in advance: within this margin among GBDTs -> speed/maintainability tie-break


def temporal_split(df: pd.DataFrame):
    """Duplicated from train.py deliberately (not imported) to keep this
    module import-cycle-free: train.py imports FROM here for the winner."""
    day = (df.ts - df.ts.min()) // 86400
    return df[day <= 20], df[(day >= 21) & (day <= 23)], df[day >= 24]


def pos_weight(y) -> float:
    return (y == 0).sum() / max(1, (y == 1).sum())


# --------------------------------------------------------------- factory
def build_gbdt(name: str, weight: float, n_estimators: int = 400,
                learning_rate: float = 0.06, random_state: int = 7):
    """Construct the WINNING family with the project's tuned hyperparameters
    (used by train.py / ablation.py for the real run). Kept separate from
    the default-hyperparameter estimators below, which exist only to make
    the selection comparison fair (no per-model tuning)."""
    # A missing library fails LOUDLY and says what to do. It must never
    # silently substitute a different family: the whole point of P1a-0 is that
    # the model was chosen empirically, and quietly training a different one
    # would invalidate every number downstream while looking like it worked.
    def _need(pkg: str, err: Exception):
        raise ModuleNotFoundError(
            f"build_gbdt('{name}') needs '{pkg}', which is not installed.\n"
            f"  If you are deploying: requirements-serve.txt is deliberately slim and\n"
            f"  only carries the SELECTED family. Ship\n"
            f"  artifacts_out/model_selection_decision.json so the winner is known,\n"
            f"  or install the full requirements.txt.\n"
            f"  Original error: {err}") from err

    if name == "LightGBM":
        try:
            import lightgbm as lgb
        except ImportError as e:
            _need("lightgbm", e)
        return lgb.LGBMClassifier(n_estimators=n_estimators, learning_rate=learning_rate,
                                   num_leaves=63, scale_pos_weight=weight,
                                   random_state=random_state, verbose=-1)
    if name == "XGBoost":
        try:
            from xgboost import XGBClassifier
        except ImportError as e:
            _need("xgboost", e)
        return XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate,
                              max_depth=6, scale_pos_weight=weight,
                              random_state=random_state, eval_metric="aucpr", verbosity=0)
    if name == "CatBoost":
        try:
            from catboost import CatBoostClassifier
        except ImportError as e:
            _need("catboost", e)
        return CatBoostClassifier(iterations=n_estimators, learning_rate=learning_rate,
                                   depth=6, scale_pos_weight=weight,
                                   random_state=random_state, verbose=False)
    if name == "LogisticRegression":
        return LogisticRegression(class_weight="balanced", max_iter=1000, random_state=random_state)
    raise ValueError(f"unknown model family: {name}")


def selection_is_persisted() -> bool:
    """Whether an actual empirical selection exists on disk."""
    return DECISION_FILE.exists()


def get_selected_model_name(default: str = "LightGBM") -> str:
    """Read the persisted winner. Falls back to `default` with a loud
    warning if select_model.py has never been run - training must never
    hard-fail just because the selection artifact is missing.

    Callers that PRINT the model name must check selection_is_persisted()
    and say which case they are in: announcing a fallback default as
    "selected empirically" is a lie that would send a reader looking for an
    XGBoost result while the run actually used LightGBM."""
    if DECISION_FILE.exists():
        return json.load(open(DECISION_FILE))["winner"]
    print(f"\n{'!' * 70}\n"
          f"WARNING: {DECISION_FILE} not found.\n"
          f"  Falling back to default={default} - this is NOT the empirically\n"
          f"  selected model, and results will NOT match the README (which\n"
          f"  reports XGBoost). Run `python -m src.models.select_model` first.\n"
          f"{'!' * 70}\n")
    return default


# --------------------------------------------------------------- runners
# Every runner uses DEFAULT hyperparameters (constructor called with only
# class-weighting args) - the point of this comparison is "which family is
# strong out of the box", not "which family did we tune the hardest".
def _time_fit_predict(fit_fn, predict_fn):
    t0 = time.perf_counter()
    fit_fn()
    t_fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    p = predict_fn()
    t_pred = time.perf_counter() - t0
    return p, t_fit, t_pred


def run_logreg(train, cal):
    scaler = StandardScaler()
    x_tr = scaler.fit_transform(train[FEATURE_COLS])
    x_cal = scaler.transform(cal[FEATURE_COLS])
    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=7)
    return _time_fit_predict(lambda: model.fit(x_tr, train.is_fraud),
                              lambda: model.predict_proba(x_cal)[:, 1])


def run_xgboost(train, cal):
    from xgboost import XGBClassifier
    w = pos_weight(train.is_fraud)
    model = XGBClassifier(scale_pos_weight=w, random_state=7, eval_metric="aucpr", verbosity=0)
    return _time_fit_predict(lambda: model.fit(train[FEATURE_COLS], train.is_fraud),
                              lambda: model.predict_proba(cal[FEATURE_COLS])[:, 1])


def run_lightgbm(train, cal):
    import lightgbm as lgb
    w = pos_weight(train.is_fraud)
    model = lgb.LGBMClassifier(scale_pos_weight=w, random_state=7, verbose=-1)
    return _time_fit_predict(lambda: model.fit(train[FEATURE_COLS], train.is_fraud),
                              lambda: model.predict_proba(cal[FEATURE_COLS])[:, 1])


def run_catboost(train, cal):
    from catboost import CatBoostClassifier
    w = pos_weight(train.is_fraud)
    model = CatBoostClassifier(scale_pos_weight=w, random_state=7, verbose=False)
    return _time_fit_predict(lambda: model.fit(train[FEATURE_COLS], train.is_fraud),
                              lambda: model.predict_proba(cal[FEATURE_COLS])[:, 1])


RUNNERS = {
    "LogisticRegression": run_logreg,
    "XGBoost": run_xgboost,
    "LightGBM": run_lightgbm,
    "CatBoost": run_catboost,
}
FAMILY = {"LogisticRegression": "linear", "XGBoost": "GBDT", "LightGBM": "GBDT", "CatBoost": "GBDT"}


def main():
    print("=== model selection: simulate + features (shared pipeline) ===")
    df = build_features(generate(seed=7))
    train, cal, _test = temporal_split(df)  # test slice untouched - selection uses validation only
    y_cal = cal.is_fraud.to_numpy()
    print(f"train {len(train):,} | validation(cal, days 21-23) {len(cal):,} "
          f"(val fraud {cal.is_fraud.mean():.3%})")

    rows = []
    for name, fn in RUNNERS.items():
        print(f"--- {name} (default hyperparams) ---")
        p, t_fit, t_pred = fn(train, cal)
        pr_auc = average_precision_score(y_cal, p)
        rows.append({"model": name, "family": FAMILY[name],
                      "val_pr_auc": round(float(pr_auc), 4),
                      "train_time_s": round(t_fit, 3),
                      "inference_time_s": round(t_pred, 4)})
        print(f"  val PR-AUC={pr_auc:.4f}  train={t_fit:.2f}s  infer={t_pred:.4f}s")

    result = pd.DataFrame(rows).sort_values("val_pr_auc", ascending=False).reset_index(drop=True)

    # ---------------------------------------------------- selection rule
    notes = []
    gbdt = result[result.family == "GBDT"].sort_values("val_pr_auc", ascending=False).reset_index(drop=True)
    best_overall = result.iloc[0]

    if best_overall["family"] != "GBDT":
        notes.append(
            f"{best_overall.model} (non-GBDT) leads on raw validation PR-AUC "
            f"({best_overall.val_pr_auc}), but the architecture requires a GBDT "
            f"(SHAP explainability, native missing-value handling, calibration "
            f"behavior documented in CLAUDE.md) - selecting among GBDTs instead.")

    top_gbdt = gbdt.iloc[0]
    within_margin = gbdt[gbdt.val_pr_auc >= top_gbdt.val_pr_auc - PR_AUC_TIE_MARGIN]
    if len(within_margin) > 1:
        fastest = within_margin.sort_values("train_time_s").iloc[0]
        winner = fastest["model"]
        notes.append(
            f"Top GBDTs are within {PR_AUC_TIE_MARGIN} PR-AUC of each other "
            f"({', '.join(f'{r.model}={r.val_pr_auc}' for _, r in within_margin.iterrows())}) "
            f"-> tie-break rule applies: selected by speed + maintainability -> {winner} "
            f"(fastest train time, single-file model artifact, no external compiler toolchain).")
    else:
        winner = top_gbdt["model"]
        notes.append(f"{winner} has the highest validation PR-AUC among GBDTs "
                      f"({top_gbdt.val_pr_auc}) - no tie-break needed.")

    print("\n=== validation results (days 21-23; test slice untouched) ===")
    print(result.to_string(index=False))
    for n in notes:
        print(f"\n{n}")
    print(f"\nWINNER: {winner}")

    result.to_csv(OUT / "model_selection.csv", index=False)
    json.dump({"winner": winner, "tie_margin": PR_AUC_TIE_MARGIN, "notes": notes,
               "table": rows},
              open(DECISION_FILE, "w"), indent=2)
    print(f"\nwrote {OUT / 'model_selection.csv'} and {DECISION_FILE}")
    return winner


if __name__ == "__main__":
    main()
