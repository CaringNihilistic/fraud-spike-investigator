"""Can any public dataset actually evaluate a MERCHANT-LEVEL burst detector?

Our honest limitation has always been that the merchant-level layer - the part
the product is actually about - is validated only on data we generated. The
stated reason was "the public datasets we evaluated have no merchant column",
which is an argument from absence and invites the obvious reply: then go and
find one that does.

So we did. `kartik2112/fraud-detection` (the Sparkov generator's output, 1.3M
transactions over 538 days) has a `merchant` column, card ids and timestamps -
everything the spike detector needs. This module asks whether it can actually
TEST that detector, and answers with numbers rather than an opinion.

WHAT A MERCHANT-LEVEL DETECTOR NEEDS TO BE TESTABLE. Ours fires on the fraud
RATE inside a merchant window, with a volume guard so a single stray fraud on a
quiet merchant cannot raise an alarm (failure-log 2). So the dataset must
contain merchant windows that are (a) busy enough to evaluate and (b) actually
concentrated in fraud. If the fraud is spread thinly across every merchant,
there is no burst to detect and a null result would measure nothing - which is
exactly the trap failure-log 24 documents for IEEE-CIS.

This check is deliberately run BEFORE any modelling. Discovering afterwards
that the test could not have worked is how you end up publishing an invalid
null in the pessimistic direction.

Data (gitignored, never committed - Kaggle terms; we publish metrics, not data):
    python -c "import kaggle;kaggle.api.authenticate();
               kaggle.api.dataset_download_files('kartik2112/fraud-detection',
                                                 path='data/sparkov', unzip=True)"

Run: python -m src.models.merchant_data_check
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

OUT = Path("artifacts_out")
OUT.mkdir(exist_ok=True)
SPARKOV = Path("data/sparkov/fraudTrain.csv")

# The detector's own volume guard: below this, a window is not evaluable.
MIN_TXNS_PER_WINDOW = 10
# What our synthetic attacks reach, for scale. Measured, not asserted:
# see the merchant cards in the live demo (73-93% flagged in a 30-txn window).
OUR_ATTACK_RATE_LOW = 0.70


def summarise(df: pd.DataFrame) -> dict:
    span_days = (df.unix_time.max() - df.unix_time.min()) / 86400
    out = {
        "rows": int(len(df)),
        "merchants": int(df.merchant.nunique()),
        "cards": int(df.cc_num.nunique()),
        "fraud_rate": round(float(df.is_fraud.mean()), 5),
        "span_days": round(float(span_days), 1),
    }

    # Is fraud concentrated in a few merchants, or spread across all of them?
    g = df.groupby("merchant").agg(n=("is_fraud", "size"), f=("is_fraud", "sum"))
    rate = g.f / g.n
    out["merchant_fraud_rate_median"] = round(float(rate.median()), 5)
    out["merchant_fraud_rate_p99"] = round(float(rate.quantile(0.99)), 5)
    out["merchant_fraud_rate_max"] = round(float(rate.max()), 5)
    out["merchants_with_zero_fraud"] = int((g.f == 0).sum())

    # The decisive question: does any evaluable window reach an attack-like rate?
    for unit, div in (("hour", 3600), ("day", 86400)):
        w = df.assign(_w=(df.unix_time // div).astype(int)) \
              .groupby(["merchant", "_w"]).agg(n=("is_fraud", "size"),
                                               f=("is_fraud", "sum"))
        ok = w[w.n >= MIN_TXNS_PER_WINDOW]
        out[f"evaluable_merchant_{unit}_windows"] = int(len(ok))
        if len(ok):
            r = ok.f / ok.n
            out[f"max_fraud_rate_per_merchant_{unit}"] = round(float(r.max()), 4)
            out[f"windows_at_attack_like_rate_{unit}"] = int((r >= OUR_ATTACK_RATE_LOW).sum())
        else:
            out[f"max_fraud_rate_per_merchant_{unit}"] = None
            out[f"windows_at_attack_like_rate_{unit}"] = 0

    # The structural tell: is this a per-CARD phenomenon or a per-MERCHANT one?
    fr = df[df.is_fraud == 1]
    out["fraud_txns"] = int(len(fr))
    out["fraud_per_compromised_card"] = round(len(fr) / max(1, fr.cc_num.nunique()), 2)
    out["fraud_per_affected_merchant"] = round(len(fr) / max(1, fr.merchant.nunique()), 2)
    return out


def main():
    if not SPARKOV.exists():
        print(f"{SPARKOV} not found. Download command is in this module's docstring.")
        print("data/ is gitignored and never committed - Kaggle terms.")
        return

    df = pd.read_csv(SPARKOV, usecols=["unix_time", "cc_num", "merchant",
                                       "amt", "is_fraud"])
    s = summarise(df)

    print("=== can public data evaluate a merchant-level burst detector? ===")
    print("dataset: kartik2112/fraud-detection (Sparkov), the one public set we")
    print("found that HAS a merchant column.\n")
    print("  {rows:,} transactions | {merchants:,} merchants | {cards:,} cards"
          .format(**s))
    print("  {span_days:.0f} days | fraud {fraud_rate:.3%}\n".format(**s))

    print("--- is fraud concentrated in some merchants? ---")
    print("  merchant fraud rate: median {merchant_fraud_rate_median:.4f}  "
          "p99 {merchant_fraud_rate_p99:.4f}  max {merchant_fraud_rate_max:.4f}"
          .format(**s))
    print("  merchants with zero fraud: {merchants_with_zero_fraud} of {merchants}\n"
          .format(**s))

    print("--- are there windows a rate-based detector could fire on? ---")
    print(f"  (volume guard: >= {MIN_TXNS_PER_WINDOW} txns, as the detector uses)")
    for unit in ("hour", "day"):
        n = s[f"evaluable_merchant_{unit}_windows"]
        mx = s[f"max_fraud_rate_per_merchant_{unit}"]
        hit = s[f"windows_at_attack_like_rate_{unit}"]
        print(f"  merchant-{unit:<5} evaluable windows {n:>7,}   "
              f"max fraud rate {('n/a' if mx is None else format(mx, '.3f')):>6}   "
              f"windows >= {OUR_ATTACK_RATE_LOW:.2f}: {hit}")
    print()

    print("--- per-card or per-merchant phenomenon? ---")
    print("  {fraud_txns:,} fraud transactions".format(**s))
    print("  {fraud_per_compromised_card:.1f} per compromised card | "
          "{fraud_per_affected_merchant:.1f} per affected merchant\n".format(**s))

    # ---- derived verdict; never hardcode a conclusion the numbers may contradict
    hourly_evaluable = s["evaluable_merchant_hour_windows"]
    attack_like = (s["windows_at_attack_like_rate_hour"]
                   + s["windows_at_attack_like_rate_day"])
    max_day = s["max_fraud_rate_per_merchant_day"] or 0.0
    testable = attack_like > 0

    if testable:
        verdict = (
            f"TESTABLE. This dataset contains {attack_like} merchant windows at an "
            f"attack-like fraud rate, so a merchant-level burst detector can be "
            f"evaluated on it and SHOULD be.")
    else:
        verdict = (
            f"NOT TESTABLE, and the reason is structural rather than a shortage of "
            f"data. There are {hourly_evaluable} evaluable merchant-HOUR windows, and "
            f"across merchant-days the highest fraud rate anywhere is {max_day:.3f} - "
            f"against the {OUR_ATTACK_RATE_LOW:.0%}-93% our attacks reach in a 30-txn "
            f"window. Nothing here is a burst. The tell is in the last block: fraud "
            f"averages {s['fraud_per_compromised_card']:.1f} transactions per "
            f"compromised CARD and only {s['fraud_per_affected_merchant']:.1f} per "
            f"merchant across {s['merchants']} merchants, of which just "
            f"{s['merchants_with_zero_fraud']} escape entirely. This models stolen "
            f"cards spent across many merchants - not merchants under coordinated "
            f"attack. Those are different loss classes, and running our detector here "
            f"would produce a null that measures the dataset, not the detector. "
            f"See failure-log 24 for the same trap on IEEE-CIS.")

    print("=== verdict ===\n" + verdict)
    json.dump({**s, "min_txns_per_window": MIN_TXNS_PER_WINDOW,
               "our_attack_rate_floor": OUR_ATTACK_RATE_LOW,
               "merchant_level_testable": testable, "verdict": verdict},
              open(OUT / "merchant_data_check.json", "w"), indent=2)
    print(f"\nwrote {OUT / 'merchant_data_check.json'}")


if __name__ == "__main__":
    main()
