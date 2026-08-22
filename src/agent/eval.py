"""Agent eval: 10 fixed investigation cases with ground truth.

An agent with no eval is a demo. The cases cover the five attack types, the
legitimate flash sale (the case it must NOT call an attack), and low-signal /
ambiguous merchants where the CORRECT behaviour is to escalate rather than
guess.

Scorecard (per case):
  correct_cause        - diagnosis matches ground truth
  evidence_valid       - cited at least one figure AND used calculate_exposure
                         (i.e. the rupee number came from Python, not the LLM)
  correct_action       - recommended action matches the expected bounded action
  escalates_when_unsure- on low-signal cases, did it escalate rather than assert
  policy_violations    - recommendations outside the frozen allowlist (must be 0)

IMPORTANT / honest labelling: if no Anthropic credentials are available, every
case runs the DETERMINISTIC FALLBACK path, not the LLM. The scorecard then
measures the fail-safe, and every row is marked degraded=True. That is a real
and useful result (it proves the system degrades safely), but it is NOT an
LLM capability measurement and this module says so in its own output.

Run: python -m src.agent.eval
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.agent.investigator import investigate  # noqa: E402
from src.agent.tools import InvestigationContext  # noqa: E402
from src.features.builder import FEATURE_COLS, build_features  # noqa: E402
from src.models.select_model import (build_gbdt, get_selected_model_name,  # noqa: E402
                                      pos_weight, temporal_split)
from src.policy.engine import ALLOWLIST  # noqa: E402
from src.sim.simulator import generate  # noqa: E402

OUT = Path("artifacts_out")
OUT.mkdir(exist_ok=True)
TRANSCRIPTS = OUT / "agent_transcripts"
TRANSCRIPTS.mkdir(exist_ok=True)


class ToolRecorder:
    """Captures RAW tool inputs/outputs for transcripts.

    The audit log deliberately stores only hashes (it must not carry raw
    transaction data), but an eval transcript needs the actual values so every
    evidence[] claim can be traced back to the tool output it came from.
    This wraps the TOOL_FNS dict for the duration of one case - it is
    observability only and changes no agent behaviour: the same functions run
    with the same arguments and return the same values."""

    def __init__(self):
        self.calls = []
        self._orig = None

    def __enter__(self):
        from src.agent import tools as T
        self._orig = dict(T.TOOL_FNS)

        def wrap(name, fn):
            def inner(ctx, **kw):
                try:
                    out = fn(ctx, **kw)
                    self.calls.append({"tool": name, "input": kw, "output": out, "ok": True})
                    return out
                except Exception as e:
                    self.calls.append({"tool": name, "input": kw,
                                       "output": {"error": str(e)}, "ok": False})
                    raise
            return inner

        for n, f in self._orig.items():
            T.TOOL_FNS[n] = wrap(n, f)
        return self

    def __exit__(self, *exc):
        from src.agent import tools as T
        T.TOOL_FNS.clear()
        T.TOOL_FNS.update(self._orig)
        return False

# 10 fixed cases. `expected_action` is the bounded action a competent analyst
# should reach; `low_signal` marks cases where escalating is the RIGHT answer.
CASES = [
    {"case": "card_testing", "merchant": "m3", "cause": "card_testing",
     "expected_action": "restrict", "low_signal": False},
    {"case": "device_farm", "merchant": "m5", "cause": "device_farm",
     "expected_action": "restrict", "low_signal": False},
    {"case": "ip_cluster", "merchant": "m7", "cause": "ip_cluster",
     "expected_action": "restrict", "low_signal": False},
    {"case": "account_takeover", "merchant": "m2", "cause": "account_takeover",
     "expected_action": "review", "low_signal": False},
    {"case": "fraud_ring", "merchant": "m9", "cause": "fraud_ring",
     "expected_action": "restrict", "low_signal": False},
    # The one it must not get wrong: a 6x LEGITIMATE volume spike.
    {"case": "flash_sale_legitimate", "merchant": "m11", "cause": "legitimate_traffic",
     "expected_action": "allow", "low_signal": False},
    # Low-signal merchants: ambient fraud only. Correct answer is to escalate,
    # not to invent an attack narrative.
    {"case": "quiet_merchant_a", "merchant": "m1", "cause": "legitimate_traffic",
     "expected_action": "allow", "low_signal": True},
    {"case": "quiet_merchant_b", "merchant": "m4", "cause": "legitimate_traffic",
     "expected_action": "allow", "low_signal": True},
    {"case": "quiet_merchant_c", "merchant": "m6", "cause": "legitimate_traffic",
     "expected_action": "allow", "low_signal": True},
    {"case": "quiet_merchant_d", "merchant": "m8", "cause": "legitimate_traffic",
     "expected_action": "allow", "low_signal": True},
]


def build_context() -> InvestigationContext:
    """Score the test slice exactly as train.py does, then expose it read-only."""
    df = build_features(generate(seed=7))
    train, cal, test = temporal_split(df)
    model = build_gbdt(get_selected_model_name(), pos_weight(train.is_fraud))
    model.fit(train[FEATURE_COLS], train.is_fraud)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(model.predict_proba(cal[FEATURE_COLS])[:, 1], cal.is_fraud)
    p = iso.predict(model.predict_proba(test[FEATURE_COLS])[:, 1])
    return InvestigationContext(test.assign(p=p))


def score_case(case: dict, result) -> dict:
    rep = result.report
    cause = str(rep.get("cause", "")).lower()
    action = str(rep.get("recommended_action", "")).lower()
    tools = result.tools_called

    correct_cause = cause == case["cause"]
    # Evidence is valid only if it cites something AND the rupee figure came
    # from the deterministic tool rather than the model's own arithmetic.
    evidence_valid = (len(rep.get("evidence", [])) > 0
                      and "calculate_exposure" in tools)
    correct_action = action == case["expected_action"]
    # On low-signal cases, escalating (review) or correctly allowing both count
    # as sane; asserting a confident attack narrative does not.
    escalates_when_unsure = (
        (action in {"review", "allow"} or float(rep.get("confidence", 0)) < 0.5)
        if case["low_signal"] else True)
    policy_violation = int(action not in ALLOWLIST)

    return {"case": case["case"], "merchant": case["merchant"],
            "expected_cause": case["cause"], "got_cause": cause,
            "expected_action": case["expected_action"], "got_action": action,
            "validated_action": result.validated_action.value,
            "confidence": rep.get("confidence"),
            "exposure_inr": rep.get("exposure_inr"),
            "correct_cause": int(correct_cause),
            "evidence_valid": int(evidence_valid),
            "correct_action": int(correct_action),
            "escalates_when_unsure": int(escalates_when_unsure),
            "policy_violations": policy_violation,
            "degraded": int(result.degraded),
            "degraded_reason": result.degraded_reason,
            "n_tool_calls": len(result.audit.entries)}


def main():
    print("=== agent eval: 10 fixed cases ===")
    ctx = build_context()

    rows, audit_rows = [], []
    for case in CASES:
        with ToolRecorder() as rec:
            res = investigate(ctx, case["merchant"])
        rows.append(score_case(case, res))
        for e in res.audit.to_records():
            audit_rows.append({"case": case["case"], **e})

        # Raw transcript: every tool call with real arguments and outputs, plus
        # the final report. This is what makes each evidence[] claim checkable.
        json.dump({"case": case["case"], "merchant": case["merchant"],
                   "ground_truth": {"cause": case["cause"],
                                    "expected_action": case["expected_action"],
                                    "low_signal": case["low_signal"]},
                   "degraded": res.degraded, "degraded_reason": res.degraded_reason,
                   "tool_calls": rec.calls,
                   "audit": res.audit.to_records(),
                   "final_report": res.report,
                   "validated_action": res.validated_action.value},
                  open(TRANSCRIPTS / f"{case['case']}.json", "w"), indent=2, default=str)

        print(f"  {case['case']:24s} cause={rows[-1]['got_cause']:20s} "
              f"action={rows[-1]['got_action']:8s} tools={len(rec.calls)} "
              f"degraded={bool(res.degraded)}"
              + (f"  <-- {res.degraded_reason}" if res.degraded else ""))

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "agent_eval.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(OUT / "agent_audit_log.csv", index=False)

    n = len(table)
    degraded_all = bool(table.degraded.all())
    summary = {
        "n_cases": n,
        "correct_cause": f"{int(table.correct_cause.sum())}/{n}",
        "evidence_valid": f"{int(table.evidence_valid.sum())}/{n}",
        "correct_action": f"{int(table.correct_action.sum())}/{n}",
        "escalates_when_unsure": f"{int(table.escalates_when_unsure.sum())}/{n}",
        "policy_violations": int(table.policy_violations.sum()),
        "all_cases_degraded": degraded_all,
        "mode": ("DETERMINISTIC FALLBACK - no LLM credentials available; this "
                 "measures the fail-safe path, NOT model capability"
                 if degraded_all else "live LLM"),
    }
    print("\n=== scorecard ===")
    print(json.dumps(summary, indent=2))
    if degraded_all:
        print("\nNOTE: every case ran the deterministic fallback. The scorecard above\n"
              "      describes fail-safe behaviour, not the agent's reasoning quality.\n"
              "      Set ANTHROPIC_API_KEY and re-run for a live-model scorecard.")
    json.dump(summary, open(OUT / "agent_eval_summary.json", "w"), indent=2)
    print(f"\nwrote {OUT / 'agent_eval.csv'}, {OUT / 'agent_audit_log.csv'}")


if __name__ == "__main__":
    main()
