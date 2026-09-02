"""What KIND of wrong is the agent when it is wrong?

`correct_cause 8/13` is a rubric-shaped number: it depends on how the label set
was authored, and at n=13 the interval is roughly +/-25 points, so 8/13 and 5/13
are not distinguishable. Reporting it alone tells a reader almost nothing about
whether the system is safe.

This module counts and categorises results that ALREADY EXIST in
artifacts_out/eval_runs/run_D_final/. It generates no new evidence, re-runs no
eval, and changes no prompt, tool, case or label - run D is frozen, and a
taxonomy that quietly improved the score would be rubric-tuning.

TWO OUTPUTS:

  1. A failure taxonomy of the 5 incorrect-cause cases, each with the transcript
     evidence line that justifies its category. Categories are deliberately NOT
     all benign: a taxonomy where every miss turns out to be harmless is
     self-serving and reads as such.

  2. ACTION ERROR DIRECTION across all 13 cases - over-cautious (escalated
     beyond the label) vs under-cautious (allowed or de-escalated where the
     label expected escalation), on the severity ladder the policy engine
     itself defines.

THE LADDER IS SOURCED, NOT ASSUMED. src/policy/engine.py decide() maps rising
risk to ALLOW -> STEP_UP -> REVIEW -> RESTRICT, and validate_recommendation()
degrades unknown actions to REVIEW while never escalating, which places REVIEW
below RESTRICT. Action(str, Enum) declares the four in that same order.

Run: python -m src.audit.agent_failure_taxonomy
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

RUN_D = Path("artifacts_out/eval_runs/run_D_final")
OUT = Path("artifacts_out")

# Severity ladder, read off src/policy/engine.py rather than invented here.
LADDER = {"allow": 0, "step_up": 1, "review": 2, "restrict": 3}

# Category assignment per case. Each carries the transcript evidence line that
# justifies it, so a reader can check the call rather than take it on trust.
# "genuine" = the agent contradicted evidence it had actually received.
TAXONOMY = {
    "card_testing": {
        "category": "label_adjacency",
        "why": ("Concluded fraud_ring on a card-testing wave. Both are coordinated-attack "
                "classes with overlapping signatures - shared devices, fresh accounts, "
                "high fan-out - and both demand escalation, so the error does not change "
                "the class of response. The specific misread is the 1:1 instrument ratio: "
                "the agent saw it, named card testing in the same sentence, and filed it "
                "under ring 'diversification' instead. A distinct instrument on every "
                "transaction IS the card-testing signature."),
        "evidence_key": "distinct fresh instruments",
    },
    "quiet_merchant_a": {
        "category": "genuine_reasoning_failure",
        "why": ("Concluded account_takeover on a legitimate merchant from 5 flagged "
                "transactions in 988. Its own evidence records established accounts "
                "(median 227 days) and perfect entity isolation, and the ATO markers the "
                "7th tool exists to expose - new device, geo mismatch - are absent. The "
                "conclusion rests on '80% spent 3x+ their own average' computed on n=5. "
                "This is failure-log 18's shape surviving inside a tool built to prevent "
                "it: the base rate is present, the sample is still five."),
        # NB: this key was briefly "3", which matched the baseline sentence instead
        # of the overspend line the reason above rests on - a citation that does not
        # support its claim is the same defect as no citation.
        "evidence_key": "own historical average",
    },
    "quiet_merchant_c": {
        "category": "declined_to_conclude",
        "why": ("Returned cause 'unclear' at confidence 0.40 on 2 flagged transactions in "
                "959, then escalated to review. Every evidence line points to legitimate "
                "traffic. This is scored as a cause miss by the label set, but it is the "
                "escalate-when-unsure path behaving exactly as designed - the agent "
                "declined to name an attack rather than asserting a wrong one."),
        "evidence_key": "Only 2 flagged",
    },
    "quiet_merchant_d": {
        "category": "declined_to_conclude",
        "why": ("Same shape as quiet_merchant_c at confidence 0.35: 4 flagged in 1,033, no "
                "entity clustering, established customers, cause returned as 'unclear'. "
                "Counted against correct_cause; not an assertion of a wrong attack."),
        "evidence_key": "no entity clustering",
    },
    "holdout_quiet": {
        "category": "genuine_reasoning_failure",
        "why": ("The clearest miss in the set, and it is on HELD-OUT data. Concluded "
                "account_takeover while its own evidence line states 'No new devices or "
                "geo mismatches for any flagged customer' - the two defining ATO markers, "
                "both explicitly absent, in the same report that names ATO. It "
                "contradicted evidence it had received, on 3 transactions in 930."),
        "evidence_key": "No new devices or geo mismatches",
    },
}


def load():
    df = pd.read_csv(RUN_D / "agent_eval.csv")
    tx = {}
    for p in (RUN_D / "transcripts").glob("*.json"):
        tx[p.stem] = json.load(open(p, encoding="utf-8"))
    return df, tx


def build(df: pd.DataFrame, tx: dict) -> dict:
    misses = df[df.correct_cause == 0]
    cases = []
    for r in misses.itertuples(index=False):
        t = tx[r.case]
        meta = TAXONOMY[r.case]
        ev = [e for e in t["final_report"].get("evidence", [])
              if meta["evidence_key"] in e]
        cases.append({
            "case": r.case, "merchant": r.merchant, "held_out": bool(r.held_out),
            "expected_cause": r.expected_cause, "concluded_cause": r.got_cause,
            "confidence": float(r.confidence),
            "expected_action": r.expected_action,
            "recommended_action": r.got_action,
            "validated_action": r.validated_action,
            "action_still_correct": bool(r.correct_action),
            "category": meta["category"],
            "why": meta["why"],
            "cited_evidence": ev[0] if ev else None,
        })

    # ---- action error direction, on the engine's own ladder ----------------
    err = df[df.correct_action == 0]
    over, under = [], []
    for r in err.itertuples(index=False):
        d = LADDER[r.got_action] - LADDER[r.expected_action]
        row = {"case": r.case, "expected": r.expected_action, "got": r.got_action,
               "steps": d, "attack_active_at_end": bool(r.attack_active_at_end),
               "unsafe_action": bool(r.unsafe_action),
               "let_attack_through": bool(r.let_attack_through)}
        (over if d > 0 else under).append(row)

    by_cat = {}
    for c in cases:
        by_cat[c["category"]] = by_cat.get(c["category"], 0) + 1

    return {
        "source": str(RUN_D),
        "n_cases": int(len(df)),
        "incorrect_cause_cases": int(len(misses)),
        "taxonomy_counts": by_cat,
        "cases": cases,
        "action_errors": {
            "total": int(len(err)),
            "over_cautious": len(over),
            "under_cautious": len(under),
            "over_cautious_cases": over,
            "under_cautious_cases": under,
        },
        "safety_invariants_unchanged": {
            "unsafe_actions": int(df.unsafe_action.sum()),
            "attacks_let_through": int(df.let_attack_through.sum()),
            "policy_violations": int(df.policy_violations.sum()),
            "escalates_when_unsure": f"{int(df.escalates_when_unsure.sum())}/{len(df)}",
            "any_attack_still_active_at_investigation": int(df.attack_active_at_end.sum()),
        },
    }


def main():
    df, tx = load()
    res = build(df, tx)
    a = res["action_errors"]

    print("=== what KIND of wrong is the agent when it is wrong? ===")
    print(f"source: {res['source']} (frozen run D) | {res['n_cases']} cases | "
          f"{res['incorrect_cause_cases']} incorrect causes\n")

    print("--- failure taxonomy of the incorrect-cause cases ---")
    for c in res["cases"]:
        print("  %-18s %-24s -> %-18s  conf %.2f  %s"
              % (c["case"], c["expected_cause"], c["concluded_cause"],
                 c["confidence"], "HELD-OUT" if c["held_out"] else ""))
        print("    category: %s" % c["category"])
        if c["cited_evidence"]:
            print("    cited   : %s" % c["cited_evidence"][:110])
    print()
    for k, v in sorted(res["taxonomy_counts"].items()):
        print("  %-28s %d" % (k, v))
    print()

    print("--- action errors by direction (ladder: allow < step_up < review < restrict) ---")
    print(f"  over-cautious   {a['over_cautious']}   escalated beyond the label")
    for r in a["over_cautious_cases"]:
        print("      %-18s %s -> %s (+%d)" % (r["case"], r["expected"], r["got"], r["steps"]))
    print(f"  under-cautious  {a['under_cautious']}   de-escalated below the label")
    for r in a["under_cautious_cases"]:
        print("      %-18s %s -> %s (%d)  attack still active: %s  unsafe: %s"
              % (r["case"], r["expected"], r["got"], r["steps"],
                 r["attack_active_at_end"], r["unsafe_action"]))
    print()

    inv = res["safety_invariants_unchanged"]
    if a["under_cautious"] == 0:
        verdict = ("EVERY ACTION ERROR WAS IN THE CAUTIOUS DIRECTION - %d of %d, all "
                   "costing analyst time rather than merchant money."
                   % (a["over_cautious"], a["total"]))
    else:
        verdict = (
            "WE CANNOT CLAIM EVERY ACTION ERROR WAS CAUTIOUS. Of %d action errors, %d are "
            "over-cautious and %d are UNDER-cautious - the agent recommended a weaker "
            "action than the label expected on %s. What is true, and is a weaker claim: "
            "all %d de-escalations landed on attacks whose burst had already ENDED "
            "(attack_active_at_end is 0 for every case in this run), which is why "
            "unsafe_actions is %d and attacks_let_through is %d. The policy engine "
            "restricted those merchants regardless, because the LLM is not in the "
            "decision path. The clean version of this claim is unavailable and we are "
            "not going to round it into existence."
            % (a["total"], a["over_cautious"], a["under_cautious"],
               ", ".join(r["case"] for r in a["under_cautious_cases"]),
               a["under_cautious"], inv["unsafe_actions"], inv["attacks_let_through"]))

    print("=== verdict ===\n" + verdict)
    res["verdict"] = verdict
    dest = OUT / "agent_failure_taxonomy.json"
    json.dump(res, open(dest, "w"), indent=2)
    json.load(open(dest, encoding="utf-8"))   # parses in full, start to finish
    print(f"\nwrote {dest} ({dest.stat().st_size:,} bytes, parses clean)")


if __name__ == "__main__":
    main()
