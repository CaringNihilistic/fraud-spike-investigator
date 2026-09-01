"""Can ANY reachable public dataset evaluate a MERCHANT-LEVEL burst detector?

Our standing limitation is that the merchant-level layer - the part the product
actually is - is validated only on data we generated. The old phrasing was "the
public datasets we evaluated have no merchant column", which is an argument
from absence and invites the obvious reply: go and find one that does.

So we did, twice, and this module holds both answers under ONE definition of
the criterion so the two cannot drift apart (failure-log 14's lesson: exactly
one definition of "the same measurement").

  kartik2112/fraud-detection (Sparkov)      1.3M txns,  693 merchants
  ealtman2019/credit-card-transactions      24M txns,  IBM/TabFormer

Why only these two: Amazon Science's fraud-dataset-benchmark - the field's
curated standard - contains NINE datasets, and exactly one of them (Sparkov)
carries a merchant identifier. IEEE-CIS, ULB, Fraud-ecommerce and the rest do
not. TabFormer is not in that benchmark and is IBM-generated rather than real,
but it is not OURS, which is the half of "self-authored synthetic world" it can
actually remove. Datasets carrying only an MCC are excluded on sight: a merchant
CATEGORY is not a merchant, and mistaking one for the other is exactly the
DeviceInfo error in failure-log 24.

WHAT A MERCHANT-LEVEL DETECTOR NEEDS TO BE TESTABLE. Two things, and the second
is the one everybody forgets:

  1. CONCENTRATION - some merchant window must actually be mostly fraud. Ours
     fires on the fraud RATE inside a window, so if fraud is spread thinly
     across every merchant there is no burst to detect.
  2. REACHABILITY - the shipped StreamingSpikeDetector only fires when
     `window` transactions land within `max_span_s`. A dataset whose merchants
     are too slow to ever pack 30 transactions into 6 hours cannot make our
     detector fire NO MATTER WHAT THE FRAUD LOOKS LIKE. Running it there would
     return "0 false alarms" that measures our own guard.

Checking (2) is what stops this from becoming the invalid-null trap that
failure-log 24 documents for IEEE-CIS. Both checks run BEFORE any modelling
code, deliberately: discovering afterwards that the test could not have worked
is how you publish a null in the pessimistic direction.

Data (gitignored, never committed - Kaggle terms; we publish metrics, not data):
    kaggle.api.dataset_download_files('kartik2112/fraud-detection',
                                      path='data/sparkov', unzip=True)
    kaggle.api.dataset_download_file('ealtman2019/credit-card-transactions',
                                     'credit_card_transactions-ibm_v2.csv',
                                     path='data/tabformer')

Run: python -m src.models.merchant_data_check
"""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.spike.detector import StreamingSpikeDetector  # noqa: E402

OUT = Path("artifacts_out")
OUT.mkdir(exist_ok=True)

# The detector's own volume guard: below this, a window is not evaluable.
MIN_TXNS_PER_WINDOW = 10
# What our synthetic attacks reach in a 30-txn window, for scale. Measured from
# the live demo's merchant cards (73-93% flagged), not asserted.
OUR_ATTACK_RATE_LOW = 0.70
# Read the firing preconditions off the shipped detector rather than restating
# them, so this check can never drift from the thing it is checking.
_D = StreamingSpikeDetector()
FIRE_WINDOW, FIRE_SPAN_S = _D.window, _D.max_span_s


# ------------------------------------------------------------------ loaders
def load_sparkov() -> pd.DataFrame | None:
    p = Path("data/sparkov/fraudTrain.csv")
    if not p.exists():
        return None
    df = pd.read_csv(p, usecols=["unix_time", "cc_num", "merchant", "amt", "is_fraud"])
    return df.rename(columns={"amt": "amount"})


def load_tabformer() -> pd.DataFrame | None:
    """IBM TabFormer. Timestamp is split across Year/Month/Day/Time, the label
    is a Yes/No string, and Amount carries a currency symbol."""
    d = Path("data/tabformer")
    p = d / "credit_card_transactions-ibm_v2.csv"
    if not p.exists():
        z = d / "credit_card_transactions-ibm_v2.csv.zip"
        if z.exists():
            with zipfile.ZipFile(z) as f:
                f.extractall(d)
        if not p.exists():
            return None
    # 24M rows: read the merchant id as int64 rather than str, and build the
    # timestamp from integer parts. String ids and string concatenation here
    # cost several GB and buy nothing.
    df = pd.read_csv(p, usecols=["User", "Card", "Year", "Month", "Day", "Time",
                                 "Amount", "Merchant Name", "Is Fraud?"],
                     dtype={"Merchant Name": "int64", "Year": "int16",
                            "Month": "int8", "Day": "int8", "User": "int32",
                            "Card": "int8"})
    hm = df.Time.str.split(":", n=1, expand=True)
    ts = pd.to_datetime(dict(year=df.Year, month=df.Month, day=df.Day,
                             hour=pd.to_numeric(hm[0], errors="coerce"),
                             minute=pd.to_numeric(hm[1], errors="coerce")),
                        errors="coerce")
    df = df.assign(
        # pandas 3 returns datetime64[us] here, NOT [ns]. Dividing the raw
        # int64 by 1e9 therefore compressed 17 years into 11 days and made
        # every merchant look like a burst - see failure-log 33. Convert to
        # seconds explicitly instead of assuming the unit.
        unix_time=ts.astype("datetime64[s]").astype("int64"),
        cc_num=df.User.astype("int64") * 100 + df.Card,
        merchant=df["Merchant Name"],
        amount=pd.to_numeric(df.Amount.str.replace("$", "", regex=False),
                             errors="coerce", downcast="float"),
        is_fraud=(df["Is Fraud?"] == "Yes").astype("int8"),
    )
    return df[ts.notna()][["unix_time", "cc_num", "merchant", "amount", "is_fraud"]]


DATASETS = [
    ("Sparkov (kartik2112/fraud-detection)", load_sparkov),
    ("IBM TabFormer (ealtman2019/credit-card-transactions)", load_tabformer),
]


# ------------------------------------------------------------------ measure
def summarise(df: pd.DataFrame) -> dict:
    span_days = (df.unix_time.max() - df.unix_time.min()) / 86400
    out = {
        "rows": int(len(df)),
        "merchants": int(df.merchant.nunique()),
        "cards": int(df.cc_num.nunique()),
        "fraud_rate": round(float(df.is_fraud.mean()), 5),
        "span_days": round(float(span_days), 1),
    }

    g = df.groupby("merchant").agg(n=("is_fraud", "size"), f=("is_fraud", "sum"))
    rate = g.f / g.n
    out["merchant_fraud_rate_median"] = round(float(rate.median()), 5)
    out["merchant_fraud_rate_max"] = round(float(rate.max()), 5)
    out["merchants_with_zero_fraud"] = int((g.f == 0).sum())

    # (1) CONCENTRATION: does any evaluable window reach an attack-like rate?
    for unit, div in (("hour", 3600), ("day", 86400)):
        w = (df.assign(_w=(df.unix_time // div).astype("int64"))
               .groupby(["merchant", "_w"])
               .agg(n=("is_fraud", "size"), f=("is_fraud", "sum")))
        ok = w[w.n >= MIN_TXNS_PER_WINDOW]
        out[f"evaluable_merchant_{unit}_windows"] = int(len(ok))
        if len(ok):
            r = ok.f / ok.n
            out[f"max_fraud_rate_per_merchant_{unit}"] = round(float(r.max()), 4)
            out[f"windows_at_attack_like_rate_{unit}"] = int((r >= OUR_ATTACK_RATE_LOW).sum())
        else:
            out[f"max_fraud_rate_per_merchant_{unit}"] = None
            out[f"windows_at_attack_like_rate_{unit}"] = 0

    # (2) REACHABILITY: can the SHIPPED detector's span guard ever be satisfied?
    tightest = []
    for _, gg in df.groupby("merchant", sort=False):
        t = np.sort(gg.unix_time.values)
        if len(t) < FIRE_WINDOW:
            continue
        tightest.append(int((t[FIRE_WINDOW - 1:] - t[:len(t) - FIRE_WINDOW + 1]).min()))
    s = pd.Series(tightest, dtype="float64")
    out["merchants_with_enough_txns"] = int(len(s))
    out["tightest_window_median_days"] = round(float(s.median() / 86400), 2) if len(s) else None
    out["tightest_window_min_hours"] = round(float(s.min() / 3600), 2) if len(s) else None
    out["merchants_that_can_satisfy_span_guard"] = int((s <= FIRE_SPAN_S).sum()) if len(s) else 0

    # The structural tell: per-CARD phenomenon or per-MERCHANT one?
    fr = df[df.is_fraud == 1]
    out["fraud_txns"] = int(len(fr))
    out["fraud_per_compromised_card"] = round(len(fr) / max(1, fr.cc_num.nunique()), 2)
    out["fraud_per_affected_merchant"] = round(len(fr) / max(1, fr.merchant.nunique()), 2)
    return out


def verdict_for(name: str, s: dict) -> tuple[bool, str]:
    """Derived, never hardcoded. Testable requires BOTH concentration and
    reachability - either one missing makes a null uninterpretable."""
    attack_like = (s["windows_at_attack_like_rate_hour"]
                   + s["windows_at_attack_like_rate_day"])
    reachable = s["merchants_that_can_satisfy_span_guard"]
    testable = attack_like > 0 and reachable > 0
    if testable:
        return True, (
            f"{name}: TESTABLE. {attack_like} merchant windows reach an attack-like "
            f"fraud rate and {reachable} merchants can satisfy the detector's "
            f"{FIRE_SPAN_S // 3600}h span guard, so the merchant layer can be "
            f"evaluated here and SHOULD be.")
    reasons = []
    if attack_like == 0:
        reasons.append(
            f"NO CONCENTRATION - the highest fraud rate in any evaluable "
            f"merchant-day is {s['max_fraud_rate_per_merchant_day']}, against the "
            f"{OUR_ATTACK_RATE_LOW:.0%}-93% our attacks reach; nothing here is a burst")
    if reachable == 0:
        reasons.append(
            f"NOT REACHABLE - 0 of {s['merchants_with_enough_txns']} merchants can "
            f"pack {FIRE_WINDOW} transactions into the detector's "
            f"{FIRE_SPAN_S // 3600}h span guard (fastest anywhere: "
            f"{s['tightest_window_min_hours']}h, median {s['tightest_window_median_days']} "
            f"days), so the shipped detector CANNOT FIRE here whatever the fraud does")
    return False, (
        f"{name}: NOT TESTABLE, structurally rather than for want of data. "
        + "; ".join(reasons)
        + f". The tell is the last block: fraud averages "
          f"{s['fraud_per_compromised_card']} txns per compromised CARD against "
          f"{s['fraud_per_affected_merchant']} per merchant across {s['merchants']} "
          f"merchants. This models stolen cards spent across many merchants, not "
          f"merchants under coordinated attack - different loss classes. Running "
          f"our detector here would produce a null that measures the dataset (or "
          f"our own guard), not the detector. See failure-log 24.")


def main():
    print("=== can public data evaluate a MERCHANT-LEVEL burst detector? ===")
    print(f"criterion: a window must be >= {MIN_TXNS_PER_WINDOW} txns AND reach a "
          f">= {OUR_ATTACK_RATE_LOW:.0%} fraud rate,")
    print(f"and some merchant must be able to pack {FIRE_WINDOW} txns into the "
          f"detector's {FIRE_SPAN_S // 3600}h span guard.\n")

    results, verdicts = {}, []
    for name, loader in DATASETS:
        df = loader()
        if df is None:
            print(f"--- {name}: not on disk, skipped (see module docstring) ---\n")
            continue
        s = summarise(df)
        ok, v = verdict_for(name, s)
        results[name] = {**s, "testable": ok, "verdict": v}
        verdicts.append(v)

        print(f"--- {name} ---")
        print("  {rows:,} txns | {merchants:,} merchants | {cards:,} cards | "
              "{span_days:.0f} days | fraud {fraud_rate:.3%}".format(**s))
        print("  merchant fraud rate: median {merchant_fraud_rate_median:.4f}  "
              "max {merchant_fraud_rate_max:.4f}  |  zero-fraud merchants "
              "{merchants_with_zero_fraud}".format(**s))
        for unit in ("hour", "day"):
            mx = s[f"max_fraud_rate_per_merchant_{unit}"]
            print(f"  merchant-{unit:<5} evaluable windows "
                  f"{s[f'evaluable_merchant_{unit}_windows']:>9,}   max rate "
                  f"{('n/a' if mx is None else format(mx, '.3f')):>6}   "
                  f">= {OUR_ATTACK_RATE_LOW:.2f}: {s[f'windows_at_attack_like_rate_{unit}']}")
        print(f"  span-guard reachability: {s['merchants_that_can_satisfy_span_guard']} of "
              f"{s['merchants_with_enough_txns']} merchants can pack {FIRE_WINDOW} txns "
              f"into {FIRE_SPAN_S // 3600}h")
        print(f"    tightest window: median {s['tightest_window_median_days']} days, "
              f"fastest {s['tightest_window_min_hours']}h")
        print("  {fraud_txns:,} fraud txns | {fraud_per_compromised_card:.1f} per "
              "compromised card | {fraud_per_affected_merchant:.1f} per merchant\n"
              .format(**s))

    if not results:
        print("No datasets on disk. Download commands are in the module docstring.")
        print("data/ is gitignored and never committed - Kaggle terms.")
        return

    any_testable = any(r["testable"] for r in results.values())
    print("=== verdict ===")
    for v in verdicts:
        print("  " + v + "\n")
    summary = (
        "At least one reachable public dataset CAN evaluate the merchant layer - "
        "build the evaluation." if any_testable else
        f"None of the {len(results)} merchant-bearing datasets we can reach is able to "
        f"evaluate this layer. Amazon Science's fraud-dataset-benchmark carries NINE "
        f"datasets and exactly one (Sparkov) has a merchant identifier at all; we "
        f"tested that one and IBM's 24M-transaction set as well. The limitation is not "
        f"that we did not look - it is that merchants-under-coordinated-attack is a "
        f"loss class public card-fraud data does not contain, and closing it needs a "
        f"PSP's own traffic.")
    print("  SUMMARY: " + summary)

    json.dump({"min_txns_per_window": MIN_TXNS_PER_WINDOW,
               "our_attack_rate_floor": OUR_ATTACK_RATE_LOW,
               "fire_window": FIRE_WINDOW, "fire_span_s": FIRE_SPAN_S,
               "any_testable": any_testable, "summary": summary,
               "datasets": results},
              open(OUT / "merchant_data_check.json", "w"), indent=2)
    print(f"\nwrote {OUT / 'merchant_data_check.json'}")


if __name__ == "__main__":
    main()
