"""Does the pipeline recipe hold up on REAL transaction data?

WHY THIS EXISTS
---------------
Every other number in this repo is measured on data we generated, and
failure-log 21 showed exactly what that can hide: our simulator encoded the
label into two features, so 0.934 partly measured our own answer key. The
obvious follow-up - "would any of this survive contact with real traffic?" -
was one we could not answer. This answers the transaction-level half.

WHAT IS AND IS NOT CLAIMED
--------------------------
Claimed: the METHODOLOGY transfers. Temporal splitting (never random), class
weighting without SMOTE, isotonic calibration fit on a held-out slice, and an
amount-weighted cost-optimal threshold, applied UNCHANGED to real data. Same
model family, same recipe, same functions - cost_optimal_threshold and
calibration_report are imported from train.py rather than reimplemented, so
there is one definition of each measurement (failure-log 14's lesson).

NOT claimed: that our 22 features transfer, or that the merchant-level layer
does. Neither dataset has a merchant column, so the spike detector, entity
graph and policy engine are out of scope here and are NOT exercised.

Expect a much lower number than the synthetic 0.934. That drop is the finding.
A real-data result that came out equally high would mean the benchmark was as
compromised as the simulator.

DATASETS (neither is redistributable; data/ is gitignored - we publish
metrics, never data):

  ULB creditcardfraud - 284,807 real card transactions, 0.173% fraud, PCA
  anonymised. Extreme imbalance is the point: 30x more skewed than our
  synthetic 5.1%, so it stress-tests class weighting and calibration far
  harder than our own data does. No entity columns, so it can only test the
  transaction-level recipe.
      python -c "import kaggle;kaggle.api.authenticate();
                 kaggle.api.dataset_download_files('mlg-ulb/creditcardfraud',
                 path='data/ulb')"

  IEEE-CIS (Vesta) - ~590k e-commerce transactions, 3.5% fraud, with real
  entity-ish columns (card1-6, addr1-2, DeviceInfo, email domains). Requires
  ACCEPTING THE COMPETITION RULES at
  kaggle.com/competitions/ieee-fraud-detection/rules first, or the API
  returns 403.
      kaggle competitions download -c ieee-fraud-detection -f train_transaction.csv -p data/ieee

Run: python -m src.models.real_data_check
"""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.features.builder import UnionFind  # noqa: E402
from src.models.select_model import build_gbdt, get_selected_model_name, pos_weight  # noqa: E402
from src.models.train import calibration_report, cost_optimal_threshold  # noqa: E402

OUT = Path("artifacts_out")
OUT.mkdir(exist_ok=True)

SYNTHETIC_PR_AUC = 0.9344          # what the same recipe scores on our own data
TRAIN_FRAC, CAL_FRAC = 0.70, 0.80  # same proportions as src/models/train.py


# ------------------------------------------------------------------ helpers
def _split(df: pd.DataFrame, tcol: str, finecol: str):
    """Temporal split, never random - identical rule to src/models/train.py.

    Cuts on day boundaries where the data spans enough days; ULB spans only
    two, so there it falls back to a time-quantile cut and says so. Either
    way no future row can reach the training slice.
    """
    days = np.sort(df[tcol].unique())
    if len(days) >= 10:
        d1, d2 = days[int(len(days) * TRAIN_FRAC)], days[int(len(days) * CAL_FRAC)]
        tr, ca, te = df[df[tcol] <= d1], df[(df[tcol] > d1) & (df[tcol] <= d2)], df[df[tcol] > d2]
        print(f"  day-boundary split: train <= d{d1} ({len(tr):,}) | "
              f"cal d{d1 + 1}-d{d2} ({len(ca):,}) | test > d{d2} ({len(te):,})")
    else:
        # Fall back to the FINE time axis - quantiling a 2-value day column
        # puts every row in train.
        q1, q2 = df[finecol].quantile(TRAIN_FRAC), df[finecol].quantile(CAL_FRAC)
        tr = df[df[finecol] <= q1]
        ca = df[(df[finecol] > q1) & (df[finecol] <= q2)]
        te = df[df[finecol] > q2]
        print(f"  time-quantile split (span is only {len(days)} days, too few for "
              f"day boundaries): train {len(tr):,} | cal {len(ca):,} | test {len(te):,}")
    return tr, ca, te


def _encode(frames, cols):
    """Label-encode object columns with a mapping fit on TRAIN ONLY.

    Factorising over the full frame would let test values shape the encoding -
    a mild unsupervised leak. This repo's whole argument is that we don't take
    those shortcuts, so unseen categories map to -1 instead.
    """
    outs = [f[cols].copy() for f in frames]
    for c in cols:
        if outs[0][c].dtype == object:
            cats = {v: i for i, v in enumerate(outs[0][c].dropna().unique())}
            for o in outs:
                o[c] = o[c].map(cats).fillna(-1).astype(np.float32)
        else:
            for o in outs:
                o[c] = pd.to_numeric(o[c], errors="coerce").astype(np.float32)
    return outs


def amount_profile(df, ycol, amtcol):
    """How big is fraud, relative to ordinary traffic?

    This is not a curiosity. Our cost-optimal threshold is amount-weighted, so
    it silently assumes fraud is EXPENSIVE relative to the legitimate orders a
    false positive would block. Our simulator builds fraud as a multiple of a
    legitimate amount, which bakes that assumption in. If real fraud is
    SMALLER than ordinary traffic, both the assumption and the feature that
    encodes it (amount_dev_ratio) point the wrong way.
    """
    f, l = df.loc[df[ycol] == 1, amtcol], df.loc[df[ycol] == 0, amtcol]
    return {"median_fraud_amount": round(float(f.median()), 2),
            "median_legit_amount": round(float(l.median()), 2),
            "fraud_to_legit_median_ratio": round(float(f.median() / max(l.median(), 1e-9)), 3),
            "fraud_value_share_of_legit_pct": round(float(100 * f.sum() / l.sum()), 3)}


def evaluate(tier, cols, tr, ca, te, ycol, amtcol, model_name):
    cols = [c for c in cols if c in tr.columns]
    Xtr, Xca, Xte = _encode((tr, ca, te), cols)
    ytr, yca, yte = tr[ycol].to_numpy(), ca[ycol].to_numpy(), te[ycol].to_numpy()

    model = build_gbdt(model_name, pos_weight(pd.Series(ytr)))
    model.fit(Xtr, ytr)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(model.predict_proba(Xca)[:, 1], yca)
    p = iso.predict(model.predict_proba(Xte)[:, 1])

    amounts = te[amtcol].to_numpy()
    t_star, _ = cost_optimal_threshold(yte, p, amounts)
    pred = p >= t_star
    tp = int(((pred == 1) & (yte == 1)).sum()); fp = int(((pred == 1) & (yte == 0)).sum())
    fn = int(((pred == 0) & (yte == 1)).sum())
    prec, rec = tp / max(1, tp + fp), tp / max(1, tp + fn)
    _, ece = calibration_report(yte, p)
    m = {"tier": tier, "n_features": len(cols),
         "pr_auc": round(float(average_precision_score(yte, p)), 4),
         "roc_auc": round(float(roc_auc_score(yte, p)), 4),
         "precision": round(prec, 4), "recall": round(rec, 4),
         "tp": tp, "fp": fp, "fn": fn,
         "cost_optimal_threshold": round(float(t_star), 4),
         "brier_score": round(float(brier_score_loss(yte, p)), 6),
         "ece": round(ece, 6),
         "value_prevented": round(float(amounts[(pred == 1) & (yte == 1)].sum()), 2),
         "value_wrongly_blocked": round(float(amounts[(pred == 1) & (yte == 0)].sum()), 2)}
    print(f"  {tier:<20} n={m['n_features']:<4} PR-AUC {m['pr_auc']:.4f}  "
          f"P {m['precision']:.3f}  R {m['recall']:.3f}  "
          f"TP {tp} FP {fp} FN {fn}")
    return m


# ------------------------------------------------------------------ ULB
def run_ulb(model_name):
    zp = Path("data/ulb/creditcardfraud.zip")
    csv = Path("data/ulb/creditcard.csv")
    if zp.exists():
        with zipfile.ZipFile(zp) as z:
            df = pd.read_csv(z.open("creditcard.csv"))
    elif csv.exists():
        df = pd.read_csv(csv)
    else:
        return None
    print("\n=== ULB creditcardfraud (284k real card transactions) ===")
    df["day"] = (df.Time // 86_400).astype(int)
    df["hour"] = ((df.Time // 3600) % 24).astype(int)
    df["is_night"] = ((df.hour < 6) | (df.hour >= 22)).astype(int)
    df["log_amount"] = np.log1p(df.Amount)
    prev = float(df.Class.mean())
    print(f"  {len(df):,} transactions | fraud {prev:.4%} "
          f"(random-baseline PR-AUC = {prev:.6f}) | {df.day.nunique()} days")
    tr, ca, te = _split(df, "day", "Time")
    print(f"  test slice: {len(te):,} rows, {int(te.Class.sum())} frauds "
          f"({te.Class.mean():.4%})")
    print("--- tiers ---")
    rows = [
        evaluate("1_amount_time", ["Amount", "log_amount", "hour", "is_night"],
                 tr, ca, te, "Class", "Amount", model_name),
        evaluate("2_plus_pca", ["Amount", "log_amount", "hour", "is_night"]
                 + [f"V{i}" for i in range(1, 29)],
                 tr, ca, te, "Class", "Amount", model_name),
    ]
    prof = amount_profile(df, "Class", "Amount")
    print(f"  amount profile: median fraud {prof['median_fraud_amount']} vs median legit "
          f"{prof['median_legit_amount']} = {prof['fraud_to_legit_median_ratio']}x")
    return {"dataset": "ULB creditcardfraud", "n_transactions": int(len(df)),
            "fraud_prevalence": round(prev, 6), "random_baseline_pr_auc": round(prev, 6),
            "amount_profile": prof,
            "tiers": rows, "best_pr_auc": max(r["pr_auc"] for r in rows)}


# ------------------------------------------------------------------ IEEE-CIS
OUR_ENTITY = ["card_txn_hist_n", "device_card_count", "addr_card_count",
              "card_device_count", "card_addr_count", "component_size"]


def build_entity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Shared-entity fan-out on IEEE-CIS, built the SAME way as src/features/builder.py.

    This is the point of using IEEE-CIS rather than ULB: it is the only public
    set here with real entity columns, so it is the only way to test the claim
    the whole product rests on - that shared-entity STRUCTURE ("one device
    across fifty accounts") carries signal. Our own leakage audit showed entity
    sharing adds only +0.004 on top of two simulator-planted label proxies, and
    left open whether that was a property of fraud or a property of our
    generator.

    Entity mapping, stated so it can be argued with: card1 is the account proxy
    (a card belongs to a person), DeviceInfo is the device, addr1 is the
    billing location. IEEE-CIS has no user id, and the usual Kaggle trick of
    synthesising a UID from card1+addr1+D1 is folklore we are not going to lean
    on for a correctness claim.

    STRICTLY INCREMENTAL, exactly as in the feature builder: every value is
    emitted from state built out of PRIOR rows only, and state is updated after
    emission. Rows are processed in TransactionDT order. No future row can
    influence an earlier one.
    """
    df = df.sort_values("TransactionDT", kind="mergesort").reset_index(drop=True)
    card = df.card1.fillna(-1).astype(np.int64).to_numpy()
    dev = df.DeviceInfo.fillna("__na__").astype(str).to_numpy() if "DeviceInfo" in df else np.full(len(df), "__na__")
    addr = df.addr1.fillna(-1).astype(np.int64).to_numpy()

    card_n = {}                      # card -> txns seen
    dev_cards, addr_cards = {}, {}   # entity -> set of cards
    card_devs, card_addrs = {}, {}   # card -> set of entities
    uf = UnionFind()

    out = np.zeros((len(df), 6), dtype=np.float32)
    for i in range(len(df)):
        c, d, a = card[i], dev[i], addr[i]
        has_d, has_a = d != "__na__", a != -1

        # ---- EMIT from prior state (never this row) ----
        ck, dk, ak = ("c", c), ("d", d), ("a", a)
        comp = 1
        for key, present in ((ck, True), (dk, has_d), (ak, has_a)):
            if present and key in uf.parent:
                comp = max(comp, uf.comp_size(key))
        out[i] = (card_n.get(c, 0),
                  len(dev_cards.get(d, ())) if has_d else 0,
                  len(addr_cards.get(a, ())) if has_a else 0,
                  len(card_devs.get(c, ())),
                  len(card_addrs.get(c, ())),
                  comp)

        # ---- THEN update ----
        card_n[c] = card_n.get(c, 0) + 1
        if has_d:
            dev_cards.setdefault(d, set()).add(c)
            card_devs.setdefault(c, set()).add(d)
            uf.union(ck, dk)
        if has_a:
            addr_cards.setdefault(a, set()).add(c)
            card_addrs.setdefault(c, set()).add(a)
            uf.union(ck, ak)

    for j, name in enumerate(OUR_ENTITY):
        df[name] = out[:, j]
    return df


IEEE_BASICS = ["TransactionAmt", "log_amount", "hour", "is_night", "ProductCD",
               "card4", "card6"]
IEEE_COUNTING = [f"C{i}" for i in range(1, 15)]
IEEE_IDENTITY = ([f"D{i}" for i in range(1, 16)]
                 + ["card1", "card2", "card3", "card5", "addr1", "addr2",
                    "dist1", "dist2", "P_emaildomain", "R_emaildomain",
                    "DeviceType", "DeviceInfo"])


def _read_ieee(name):
    d = Path("data/ieee")
    if (d / name).exists():
        return pd.read_csv(d / name)
    if (d / (name + ".zip")).exists():
        with zipfile.ZipFile(d / (name + ".zip")) as z:
            return pd.read_csv(z.open(name))
    return None


def run_ieee(model_name):
    tx = _read_ieee("train_transaction.csv")
    if tx is None:
        return None
    print("\n=== IEEE-CIS Fraud Detection (Vesta) ===")
    ident = _read_ieee("train_identity.csv")
    if ident is not None:
        tx = tx.merge(ident, on="TransactionID", how="left")
    else:
        print("  (train_identity.csv absent - continuing without identity columns)")
    tx["day"] = (tx.TransactionDT // 86_400).astype(int)
    tx["hour"] = ((tx.TransactionDT // 3600) % 24).astype(int)
    tx["is_night"] = ((tx.hour < 6) | (tx.hour >= 22)).astype(int)
    tx["log_amount"] = np.log1p(tx.TransactionAmt)
    print("  building shared-entity features (incremental, in TransactionDT order)...")
    tx = build_entity_features(tx)
    prev = float(tx.isFraud.mean())
    print(f"  {len(tx):,} transactions | fraud {prev:.4%} "
          f"(random-baseline PR-AUC = {prev:.6f}) | {tx.day.nunique()} days")
    tr, ca, te = _split(tx, "day", "TransactionDT")
    print(f"  test slice: {len(te):,} rows, {int(te.isFraud.sum())} frauds "
          f"({te.isFraud.mean():.4%})")
    print("--- tiers (mirroring our own ablation ladder) ---")
    full = IEEE_BASICS + IEEE_COUNTING + IEEE_IDENTITY
    rows = [
        evaluate("1_basics", IEEE_BASICS, tr, ca, te, "isFraud", "TransactionAmt", model_name),
        evaluate("2_plus_counting", IEEE_BASICS + IEEE_COUNTING, tr, ca, te,
                 "isFraud", "TransactionAmt", model_name),
        evaluate("3_plus_identity", full, tr, ca, te, "isFraud", "TransactionAmt", model_name),
        evaluate("4_plus_our_entity", full + OUR_ENTITY, tr, ca, te,
                 "isFraud", "TransactionAmt", model_name),
    ]
    print("--- diagnostic: do OUR entity features carry signal on their own? ---")
    rows.append(evaluate("D_our_entity_only", OUR_ENTITY, tr, ca, te,
                         "isFraud", "TransactionAmt", model_name))
    marginal = rows[3]["pr_auc"] - rows[2]["pr_auc"]
    print(f"  marginal lift of shared-entity structure on REAL data: "
          f"{marginal:+.4f} PR-AUC "
          f"({rows[2]['pr_auc']:.4f} -> {rows[3]['pr_auc']:.4f})")
    prof = amount_profile(tx, "isFraud", "TransactionAmt")
    print(f"  amount profile: median fraud {prof['median_fraud_amount']} vs median legit "
          f"{prof['median_legit_amount']} = {prof['fraud_to_legit_median_ratio']}x")
    return {"dataset": "IEEE-CIS Fraud Detection (Vesta)", "n_transactions": int(len(tx)),
            "fraud_prevalence": round(prev, 6), "random_baseline_pr_auc": round(prev, 6),
            "amount_profile": prof,
            "entity_marginal_pr_auc": round(marginal, 4),
            "tiers": rows, "best_pr_auc": max(r["pr_auc"] for r in rows)}


# ------------------------------------------------------------------ main
def main():
    model_name = get_selected_model_name()
    print("=== real-data check: our recipe, someone else's data ===")
    print(f"model family: {model_name} (same empirical winner as the synthetic pipeline)")
    print(f"reference: the same recipe scores PR-AUC {SYNTHETIC_PR_AUC} on our simulator")

    results = [r for r in (run_ulb(model_name), run_ieee(model_name)) if r]
    if not results:
        raise SystemExit(
            "\nNo real dataset found. See this module's docstring for the two\n"
            "download commands. IEEE-CIS additionally requires accepting the\n"
            "competition rules at kaggle.com/competitions/ieee-fraud-detection/rules\n"
            "or the API returns 403.")

    flat = []
    for r in results:
        for t in r["tiers"]:
            flat.append({"dataset": r["dataset"], "fraud_prevalence": r["fraud_prevalence"], **t})
    table = pd.DataFrame(flat)

    print("\n" + "=" * 78)
    print(table.to_string(index=False))
    print("\n=== summary ===")
    for r in results:
        lift = r["best_pr_auc"] / r["random_baseline_pr_auc"]
        print(f"  {r['dataset']}: best PR-AUC {r['best_pr_auc']:.4f} vs random "
              f"baseline {r['random_baseline_pr_auc']:.6f} = {lift:.0f}x lift")
    print(f"  our simulator, same recipe: {SYNTHETIC_PR_AUC}")
    print("\n  The gap between those lines is the honest measure of how much our\n"
          "  simulator was helping. Methodology transfers; the number does not.\n"
          "  Merchant-level components are NOT exercised here - no merchant column.")

    # Our own simulator's amount profile, for the contrast that matters.
    try:
        from src.sim.simulator import generate
        syn = generate(seed=7)
        syn_prof = amount_profile(syn, "is_fraud", "amount")
        print()
        print("  amount profile - OUR SIMULATOR: median fraud "
              f"{syn_prof['median_fraud_amount']} vs legit "
              f"{syn_prof['median_legit_amount']} = "
              f"{syn_prof['fraud_to_legit_median_ratio']}x")
        for r in results:
            rp = r["amount_profile"]["fraud_to_legit_median_ratio"]
            print(f"  amount profile - {r['dataset']}: {rp}x")
        print("  Our simulator builds fraud as a MULTIPLE of a legitimate amount, so")
        print("  fraud comes out LARGER than ordinary traffic. On real card data it is")
        print("  SMALLER - card testing uses tiny amounts on purpose. So amount_dev_ratio,")
        print("  one of the two label proxies in failure-log 21, points the WRONG WAY on")
        print("  real data, and the amount-weighted cost threshold inherits that.")
    except Exception as e:
        syn_prof = {"error": str(e)}

    summary = {"synthetic_pr_auc_for_reference": SYNTHETIC_PR_AUC,
               "synthetic_amount_profile": syn_prof,
               "model_family": model_name, "datasets": results,
               "scope_note": ("Transaction-level only. The spike detector, entity graph "
                              "and policy engine are merchant-level and are not exercised "
                              "by either dataset.")}
    table.to_csv(OUT / "real_data_check.csv", index=False)
    json.dump(summary, open(OUT / "real_data_check.json", "w"), indent=2)
    print(f"\nwrote {OUT / 'real_data_check.csv'}, {OUT / 'real_data_check.json'}")


if __name__ == "__main__":
    main()
