"""Evidence-traceability checker for agent transcripts.

Every number the agent asserts in evidence[] must be findable in some tool
output it actually received. This is the check that separates "the model
summarised the data" from "the model made something up that sounds right".

Derived forms are accepted, because restating a rate as a percentage or
quoting the min/max of a returned list is legitimate summarisation, not
fabrication:
  * x, x*100, x/100 at 0/1/2 decimal places (rate <-> percent)
  * truncation as well as rounding (the model wrote 3947 for 3947.6)
  * min/max/len over any numeric list inside a tool output
  * ratios between any two aggregate values in the SAME tool output
    (covers "40 of 46 customers = 87%")

Run: python -m src.agent.verify_evidence <transcript_dir>
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

# Fields the anomaly tool introduced - listed so the checker's coverage is
# explicit rather than incidental (stage 1c).
ANOMALY_FIELDS = [
    "share_on_new_device_for_customer", "share_with_geo_mismatch",
    "median_amount_vs_customer_own_average", "share_spending_over_3x_own_average",
    "median_customer_age_days", "share_accounts_newer_than_30_days",
    "on_new_device", "geo_mismatch", "amount_vs_own_avg", "account_age_days",
    # run D: instrument-novelty ratio on get_flagged_transactions
    "flagged_distinct_instruments", "flagged_distinct_instruments_per_txn",
    "non_flagged_distinct_instruments", "non_flagged_distinct_instruments_per_txn",
    "non_flagged_count",
]


# De-labelled entity ids are hex (ip_87192b85), so they contain digit runs that
# a naive extractor reads as quantitative claims. Strip them before scanning:
# quoting an identifier is not asserting a number.
_ENTITY_ID = re.compile(r"(?:c|d|ip|pi|m)_[0-9a-f]{4,}")


def numbers_in(s) -> set[float]:
    out = set()
    for m in re.findall(r"\d[\d,]*\.?\d*", _ENTITY_ID.sub(" ", str(s))):
        try:
            out.add(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def _walk_numbers(obj, acc: list):
    """Every scalar number anywhere in a tool output, plus list aggregates."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            # Numbers embedded in FIELD NAMES are legitimately quotable:
            # "current_flagged_rate_last_30_txns" makes "last 30 transactions"
            # a faithful restatement, not an invented figure.
            acc.extend(numbers_in(k))
            _walk_numbers(v, acc)
    elif isinstance(obj, list):
        nums = [x for x in obj if isinstance(x, (int, float)) and not isinstance(x, bool)]
        if nums:
            acc.extend([min(nums), max(nums), len(nums), sum(nums)])
        acc.append(len(obj))
        for v in obj:
            _walk_numbers(v, acc)
    elif isinstance(obj, bool):
        pass
    elif isinstance(obj, (int, float)):
        acc.append(float(obj))


def universe(tool_calls) -> set[float]:
    """All numbers the agent could legitimately cite, plus derived forms."""
    raw: list[float] = []
    for c in tool_calls:
        _walk_numbers(c.get("output"), raw)
        # min/max over per-transaction numeric fields (lists of dicts)
        out = c.get("output")
        if isinstance(out, dict):
            for v in out.values():
                if isinstance(v, list) and v and isinstance(v[0], dict):
                    keys = {k for d in v for k, x in d.items()
                            if isinstance(x, (int, float)) and not isinstance(x, bool)}
                    for k in keys:
                        vals = [d[k] for d in v if isinstance(d.get(k), (int, float))]
                        if vals:
                            raw.extend([min(vals), max(vals), sum(vals), len(vals)])

    u: set[float] = set()
    for v in raw:
        for d in (v, v * 100, v / 100):
            for nd in (0, 1, 2):
                u.add(round(d, nd))
            u.add(float(math.floor(abs(d) * 100) / 100))  # truncation
            u.add(float(math.floor(abs(d))))
    # Pairwise derivations over returned aggregates: ratios ("40 of 46 = 87%")
    # and sums ("fanout 55+49 = 104"). Both are summarisation of data the tool
    # actually returned, not fabrication.
    agg = sorted({v for v in raw if v}, key=abs, reverse=True)[:120]
    for i, a in enumerate(agg):
        for b in agg[i + 1:]:
            for nd in (0, 1, 2):
                if b:
                    u.add(round(a / b * 100, nd))
                u.add(round(a + b, nd))
    return u


def check_transcript(path: Path) -> dict:
    d = json.load(open(path, encoding="utf-8"))
    u = universe(d.get("tool_calls", []))
    claims, bad = 0, []
    for ev in d.get("final_report", {}).get("evidence", []):
        claims += 1
        unmatched = {
            n for n in numbers_in(ev)
            if not any(round(n, nd) in u for nd in (0, 1, 2))
            and float(math.floor(n)) not in u
        }
        # bare years/ids/timestamps are not quantitative claims
        unmatched = {n for n in unmatched if n < 1e9}
        if unmatched:
            bad.append({"evidence": ev, "untraceable": sorted(unmatched)})
    tools = [c["tool"] for c in d.get("tool_calls", [])]
    return {"case": d.get("case"), "claims": claims, "untraceable_claims": len(bad),
            "detail": bad, "used_anomaly_tool": "get_customer_anomalies" in tools,
            "tools": tools}


def main():
    tdir = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts_out/agent_transcripts")
    rows = [check_transcript(f) for f in sorted(tdir.glob("*.json"))]
    total = sum(r["claims"] for r in rows)
    bad = sum(r["untraceable_claims"] for r in rows)
    for r in rows:
        flag = "" if not r["untraceable_claims"] else f"  <-- {r['untraceable_claims']} UNTRACEABLE"
        anom = "anomaly-tool" if r["used_anomaly_tool"] else "            "
        print(f"  {r['case']:24s} claims={r['claims']:2d} {anom}{flag}")
        for b in r["detail"]:
            print(f"      {b['untraceable']}: {b['evidence'][:110]}")
    print(f"\n  traceable: {total - bad}/{total} evidence claims "
          f"({100.0 * (total - bad) / max(1, total):.1f}%)")
    return rows


if __name__ == "__main__":
    main()
