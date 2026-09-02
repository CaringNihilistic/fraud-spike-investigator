"""Does every headline number in the docs still match the artifact it came from?

This exists because the same defect has now happened five times. We re-measure,
we update the tables, and the numbers living in other sentences go stale
silently, because nothing type-checks English. Failure-log 26 diagnosed it,
failure-log 31 shipped a policy constant because of it, and two external audits
found instances we had missed. Our own docs carried it as an OPEN process gap.

This closes it. `python -m src.audit.verify_submission` exits non-zero if a
documented headline disagrees with the artifact that produced it.

WHY THIS IS A REGISTRY AND NOT A NUMBER SCRAPER. The obvious design - find every
number in the docs, compare it to every number in the artifacts - is wrong HERE
specifically. This repository deliberately contains ~22 superseded figures
(0.9344, INR 10.57L, 0.945, "0 false alarms") because retracted results are kept
visible with their history. A scraper would flag all of them and create pressure
to delete exactly the honesty the project is built on. So each check is declared
explicitly, and the things that must NOT appear are declared separately with
visible exemptions.

THREE KINDS OF CHECK:

  1. CLAIMS      a documented number must equal its artifact field, wherever it
                 is claimed. Catches the stale-table defect.
  2. RETIRED     a retracted phrasing must not appear unless the line marks
                 itself as history. Exemptions are PRINTED, never silent - a
                 hidden exemption would reintroduce the bug it prevents.
  3. CODE        a shipped constant - in Python OR in the dashboard's JS -
                 must equal the decision artifact that
                 derived it. Catches failure-log 31, which no documentation
                 check could have seen: engine.py shipped STEP_UP_CUT = 20.0
                 while the validation sweep adopted 25.

Scope, stated so this is not oversold: it verifies the HEADLINE numbers - the
ones a judge reads and the ones that have actually gone stale. It is not a proof
that every sentence is current.

Run: python -m src.audit.verify_submission
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ART = Path("artifacts_out")
DOCS = ("README.md", "SUBMISSION.md")
# The live dashboard states headline numbers too, and a judge reads it before
# either markdown file. It went stale once while both docs were clean.
DASHBOARD = "src/serve/static/app.js"
SCANNED = DOCS + (DASHBOARD,)


def art(name: str) -> dict:
    p = ART / name
    if not p.exists():
        raise FileNotFoundError(
            f"{p} missing - run the pipeline first (see CLAUDE.md). This check "
            f"compares docs against artifacts, so it cannot run without them.")
    return json.load(open(p, encoding="utf-8"))


@dataclass
class Claim:
    """One documented number, and where its truth lives."""
    name: str
    pattern: str            # capture group `group` holds the value as written
    expected: float
    tol: float = 0.005
    files: tuple = DOCS
    required: bool = True   # must appear at least once somewhere
    group: int = 1
    # Skip a match whose LINE contains any of these. Cheaper and far clearer
    # than encoding "not the uncalibrated row" into the regex itself.
    skip_lines_with: tuple = ()


@dataclass
class Retired:
    """A phrasing that must not appear unless the line marks itself history."""
    name: str
    pattern: str
    why: str
    files: tuple = DOCS
    exempt: tuple = ()      # extra per-claim history markers


# Lines that are explicitly talking about the past are allowed to quote the
# past. Keeping retracted numbers visible is the point of the failure log.
HISTORY_MARKERS = (
    "pre-fix", "PRE-FIX", "as measured at that time", "one generation later",
    # "failure 3" used to live here and was far too loose: it matched any line
    # citing failure-log 3x, so a line saying "[failure 37]" auto-exempted itself.
    # A tampering test caught it silently exempting the very claim failure-log 37
    # retracts. "failure-log" alone is the marker; the bare citation is not.
    "one dataset generation", "failure-log", "historical",
    "earlier versions", "used to", "we retract", "RETRACT", "was wrong",
    "previously", "before the hard negatives", "old generator", "at that time",
    "as measured at the time", "the fix cost", "have since moved", "FIXED",
    "since moved again", "\u2192",   # an arrow is an old->new transition by construction
)


def build_claims() -> list[Claim]:
    e, m, c = art("economics.json"), art("metrics.json"), art("config4_npv.json")
    cc = art("cost_curve.json")
    ft = art("agent_failure_taxonomy.json")
    s = art("seed_stability.json")
    lakh = lambda v: v / 1e5  # noqa: E731
    return [
        Claim("net protected value (L)",
              r"[Nn]et protected value\*{0,2}\s*\|?[^|\n]*?₹([\d.]+)L",
              lakh(e["net_protected_value_inr"]), 0.006),
        Claim("fraud exposure prevented (L)",
              r"[Ee]xposure prevented[^|\n]*?\|\s*\*{0,2}₹([\d.]+)L",
              lakh(e["fraud_exposure_prevented_inr"]), 0.006),
        Claim("legitimate revenue impacted",
              r"[Ll]egitimate revenue impacted\s*\|\s*₹([\d,]+)",
              e["legit_revenue_impacted_inr"], 1.0),
        Claim("human review load",
              r"[Hh]uman review load\s*\|\s*([\d.]+) cases per 1,000",
              e["reviews_per_1000_txns"], 0.05),
        Claim("precision",
              r"[Pp]recision\s*/\s*[Rr]ecall[^\n]*?(\d\.\d+)\s*/\s*\d\.\d+",
              m["precision"], 0.0006),
        Claim("recall",
              r"[Pp]recision\s*/\s*[Rr]ecall[^\n]*?\d\.\d+\s*/\s*(\d\.\d+)",
              m["recall"], 0.0006),
        Claim("brier (calibrated)", r"[Bb]rier[^\n]*?\|\s*\*{0,2}(\d\.\d+)",
              m["brier_score"], 0.0002,
              skip_lines_with=("raw", "uncalibrated")),
        Claim("5-seed PR-AUC mean",
              r"PR-AUC[^\n]{0,14}?(\d\.\d+)\s*±\s*\d\.\d+",
              s["pr_auc_mean"], 0.0006),
        Claim("5-seed PR-AUC std",
              r"PR-AUC[^\n]{0,14}?\d\.\d+\s*±\s*(\d\.\d+)",
              s["pr_auc_std"], 0.0006),
        Claim("config3 mean NPV",
              r"shipped configuration\*{0,2}\s*\|\s*\*{0,2}₹([\d,]+)",
              c["config3_shipped_mean_npv"], 1.0, files=("README.md",)),
        # Counted in failure-log 37, after the docs asserted for weeks that EVERY
        # action error was cautious without anyone running the count.
        Claim("action errors over-cautious",
              r"(\d+) over-cautious",
              ft["action_errors"]["over_cautious"], 0.1),
        Claim("action errors under-cautious",
              r"(\d+) under-cautious",
              ft["action_errors"]["under_cautious"], 0.1),
        Claim("genuine reasoning failures",
              r"\*\*(\d+) are genuine reasoning failures\*\*",
              ft["taxonomy_counts"]["genuine_reasoning_failure"], 0.1,
              files=("SUBMISSION.md",)),
        Claim("dashboard net protected value (L)",
              r'k="net protected value" v="\u20b9([\d.]+)L"',
              lakh(e["net_protected_value_inr"]), 0.006, files=(DASHBOARD,)),
        Claim("dashboard break-even",
              r"positive up to\$\{' '\}<b>\u20b9([\d,]+) per review",
              cc["break_even_review_cost_policy_inr"], 1.5, files=(DASHBOARD,)),
        Claim("break-even review cost",
              r"break-even at \*{0,2}₹([\d,]+)/review",
              cc["break_even_review_cost_policy_inr"], 1.5),
        Claim("config4 mean NPV",
              r"rejected configuration\s*\|\s*₹([\d,]+)",
              c["config4_rejected_mean_npv"], 1.0, files=("README.md",)),
    ]


def build_retired() -> list[Retired]:
    return [
        Retired("zero false alarms", r"\*{0,2}0\*{0,2}\s*(?:false alarms?\s*)?in every world",
                "retracted: the current figure is 1 in 35 non-attack merchant-windows"),
        Retired("independent worlds", r"independent (simulated )?worlds",
                "they are 5 seeds of ONE generator, not independent worlds",
                exempt=("not five independent", "not independent", "seeds, not")),
        Retired("old step-up cutoff", r"\(85,\s*20\)",
                "the validation sweep adopts (85, 25); 20 is below our own 2% margin"),
        # The claim retracted in failure-log 37. It survived three external audits
        # and this checker, because it carried no number for the checker to verify.
        Retired("every action error was cautious",
                r"[Ee]very action error was in the cautious direction",
                "counted in failure-log 37: 4 over-cautious, 3 UNDER-cautious",
                exempt=("retracted", "was wrong")),
    ]


def check_claims(texts: dict[str, str], claims: list[Claim]):
    """A documented number must equal its artifact - UNLESS the line is openly
    talking about the past. Keeping superseded figures visible is the point of
    the failure log, so a checker that forbade them would push us to delete our
    own history. Every exemption is reported, never applied silently."""
    problems, seen, exemptions = [], [], []
    for cl in claims:
        found = 0
        for f in cl.files:
            for mo in re.finditer(cl.pattern, texts[f]):
                lo = texts[f].rfind("\n", 0, mo.start()) + 1
                hi = texts[f].find("\n", mo.end())
                line = texts[f][lo:hi if hi != -1 else None]
                n = texts[f][:mo.start()].count("\n") + 1
                if any(x in line for x in cl.skip_lines_with):
                    continue
                raw = mo.group(cl.group).replace(",", "")
                try:
                    got = float(raw)
                except ValueError:
                    problems.append(f"{f}: {cl.name}: cannot parse '{mo.group(cl.group)}'")
                    continue
                if abs(got - cl.expected) <= cl.tol:
                    found += 1          # agrees with the artifact: verified
                    continue
                # Disagrees. Only an explicit history marker may excuse it, and
                # the excuse is always printed.
                mk = next((x for x in HISTORY_MARKERS if x in line), None)
                if mk:
                    exemptions.append(
                        f"{f}:{n}  {cl.name} = {mo.group(cl.group)} "
                        f"(superseded, line marks itself: '{mk}')")
                else:
                    problems.append(
                        f"{f}:{n}  {cl.name}\n"
                        f"      doc says  {mo.group(cl.group)}\n"
                        f"      artifact  {cl.expected:.4f}")
        if cl.required and found == 0 and not any(cl.name in p for p in problems):
            problems.append(f"{cl.name}: claim not found in {', '.join(cl.files)} "
                            f"- pattern may have rotted, which is itself a failure")
        seen.append((cl.name, found))
    return problems, seen, exemptions


def check_retired(texts: dict[str, str], retired: list[Retired]) -> tuple[list[str], list[str]]:
    problems, exemptions = [], []
    for r in retired:
        for f in r.files:
            for i, line in enumerate(texts[f].splitlines(), 1):
                if not re.search(r.pattern, line):
                    continue
                markers = HISTORY_MARKERS + r.exempt
                hit = next((mk for mk in markers if mk in line), None)
                if hit:
                    exemptions.append(f"{f}:{i}  {r.name}  (allowed: '{hit}')")
                else:
                    problems.append(f"{f}:{i}  {r.name} appears unlabelled\n"
                                    f"      {r.why}\n"
                                    f"      {line.strip()[:110]}")
    return problems, exemptions


def check_code() -> list[str]:
    """The check no documentation scan could have made: does the shipped
    constant still equal the decision that derived it? This is failure-log 31."""
    from src.policy import engine
    d = art("threshold_sweep_decision.json")
    problems = []
    pairs = [("STEP_UP_CUT", engine.STEP_UP_CUT, d["adopted_step_up_cut"]),
             ("RESTRICT_CUT", engine.RESTRICT_CUT, d["adopted_restrict_cut"])]
    for name, shipped, adopted in pairs:
        if float(shipped) != float(adopted):
            problems.append(
                f"src/policy/engine.py  {name} = {shipped}\n"
                f"      threshold_sweep_decision.json adopted {adopted}\n"
                f"      the shipped policy does not match the rule that chose it "
                f"(this is exactly failure-log 31)")

    # The dashboard draws its own "adopted" marker on the cost panel, and it
    # went a whole dataset generation stale - showing step-up 20, the value
    # failure-log 31 established our rule does not authorise, on the page a
    # judge sees first. Neither doc check could see it: the number lives in a
    # JS comparison, not in prose.
    js = Path(DASHBOARD).read_text(encoding="utf-8")
    m = re.search(r"const on = d\.cut === (\d+)", js)
    if not m:
        problems.append(
            f"{DASHBOARD}: the cost panel's adopted-cutoff marker was not found "
            f"- the pattern has rotted, which is itself a failure")
    elif float(m.group(1)) != float(d["adopted_step_up_cut"]):
        problems.append(
            f"{DASHBOARD}  cost panel marks step-up {m.group(1)} as adopted\n"
            f"      threshold_sweep_decision.json adopted {d['adopted_step_up_cut']}\n"
            f"      the running demo is telling a judge we shipped a cutoff we did not")
    return problems


def main() -> int:
    # Windows consoles default to cp1252 and these docs are full of INR
    # signs and arrows. Never let the checker die on its own output.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    texts = {f: Path(f).read_text(encoding="utf-8") for f in SCANNED}
    print("=== verify_submission: do the docs still match the artifacts? ===\n")

    claim_problems, seen, claim_exempt = check_claims(texts, build_claims())
    retired_problems, exemptions = check_retired(texts, build_retired())
    exemptions = claim_exempt + exemptions
    code_problems = check_code()

    print("--- headline claims checked ---")
    for name, n in seen:
        print(f"  {'ok ' if n else '!! '} {name:<28} {n} occurrence(s)")
    print()

    if exemptions:
        print("--- superseded figures, allowed because the line marks itself history ---")
        print("    (printed, never silent: a hidden exemption reintroduces the bug)")
        for e in exemptions:
            print("  " + e)
        print()

    problems = claim_problems + retired_problems + code_problems
    total = len(seen) + len(build_retired()) + 2
    if not problems:
        print(f"PASS - {sum(n for _, n in seen)} documented headline figures across "
              f"{len(SCANNED)} files (docs + the live dashboard) match their artifacts, "
              f"no retracted phrasing is "
              f"unlabelled, and the shipped policy cutoffs equal the decision that "
              f"adopted them.")
        return 0

    print("FAIL\n")
    for p in problems:
        print("  x " + p + "\n")
    print(f"{len(problems)} problem(s). Every one is a number that would have "
          f"shipped disagreeing with our own machine output.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
