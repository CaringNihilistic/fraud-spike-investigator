"""Synthetic transaction simulator for Fraud Spike Investigator.

Generates a 30-day, multi-merchant transaction stream with SIX labeled
scenarios injected into known time windows on known merchants:

  S1 fraud_spike        - baseline ~0.7% fraud jumps to ~5% on one merchant
  S2 device_farm        - one device -> ~50 accounts -> multiple instruments
  S3 ip_cluster         - one IP -> many accounts, abnormal velocity
  S4 account_takeover   - old customer, sudden new device+geo+big amount
  S5 fraud_ring         - accounts sharing devices/IPs/instruments
  S6 flash_sale         - LEGITIMATE volume spike; must NOT be flagged

Design note (research-grounded): fraud rings only exist in a graph if
shared entities are EXPLICITLY wired across accounts - row-independent
sampling cannot produce them. All scenario generators below reuse
device/ip/instrument ids deliberately.

Ground truth columns: is_fraud (label), scenario (provenance).
The scenario column is for evaluation ONLY - never a model feature.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from dataclasses import dataclass, field

RNG = np.random.default_rng

DAY = 24 * 3600
START_TS = 1_760_000_000  # arbitrary epoch anchor
N_DAYS = 30

# ---------------------------------------------------------------- entity ids
# Entity IDs used to be semantic: d_FARM_F, pi_STOLEN_S3, ip_CLUSTER_I,
# d_ATO_A17. That made the eval easier than reality - an investigator reading
# "pi_STOLEN_*" can name the attack without doing any analysis, and in a real
# stream no identifier announces its own intent.
#
# Every ID - legitimate AND attack - now runs through the same deterministic
# hash, so all IDs are drawn from one indistinguishable format and carry zero
# information about their origin. Ground truth lives ONLY in the `scenario`
# column, which no agent tool exposes.
#
# This is a pure relabelling: the hash consumes no RNG draws, so the random
# stream is untouched, and every feature counts entity SETS rather than parsing
# ID strings - so model metrics must come out bit-identical. train.py asserts it.
ID_SALT = "fsi-v1"


def oid(kind: str, key) -> str:
    """Opaque, stable, collision-resistant id. Same shape for every entity."""
    h = hashlib.sha1(f"{ID_SALT}|{kind}|{key}".encode()).hexdigest()[:8]
    return f"{kind}_{h}"

# ---------------------------------------------------------------- entities
@dataclass
class World:
    rng: np.random.Generator
    n_merchants: int = 12
    n_customers: int = 4000
    customers: dict = field(default_factory=dict)  # cid -> profile

    def __post_init__(self):
        for cid in range(self.n_customers):
            self.customers[cid] = {
                "home_geo": int(self.rng.integers(0, 20)),
                "device": oid("d", f"cust{cid}"),        # personal device
                "ip": oid("ip", f"pool{int(self.rng.integers(0, 1500))}"),  # shared ISP pool
                "instrument": oid("pi", f"cust{cid}"),
                "avg_amount": float(self.rng.lognormal(6.5, 0.6)),  # ~INR 665 median
                "created_day": int(self.rng.integers(-400, 0)),
            }


def _base_txn(w: World, rng, ts, merchant_id, cid=None) -> dict:
    if cid is None:
        cid = int(rng.integers(0, w.n_customers))
    c = w.customers[cid]
    amount = max(20.0, rng.normal(c["avg_amount"], c["avg_amount"] * 0.35))
    return {
        "ts": int(ts),
        "merchant_id": f"m{merchant_id}",
        "customer_id": oid("c", cid),
        "device_id": c["device"],
        "ip": c["ip"],
        "instrument_id": c["instrument"],
        "geo": c["home_geo"],
        "amount": round(float(amount), 2),
        "payment_method": rng.choice(["card", "upi", "netbanking", "wallet"], p=[0.35, 0.45, 0.1, 0.1]),
        "customer_created_day": c["created_day"],
        "is_fraud": 0,
        "scenario": "baseline",
    }


def _poisson_times(rng, day_start, n, hour_bias=True):
    ts = day_start + rng.uniform(0, DAY, size=n)
    if hour_bias:  # more traffic 10:00-22:00
        hours = ((ts - day_start) / 3600) % 24
        keep = (hours > 8) | (rng.random(n) < 0.3)
        ts = ts[keep]
    return np.sort(ts)


# ---------------------------------------------------------------- baseline
def gen_baseline(w: World, rng) -> list[dict]:
    rows = []
    for day in range(N_DAYS):
        day_start = START_TS + day * DAY
        for m in range(w.n_merchants):
            daily = int(rng.normal(220, 30))
            for t in _poisson_times(rng, day_start, daily):
                r = _base_txn(w, rng, t, m)
                if rng.random() < 0.007:  # ambient ~0.7% fraud
                    r["is_fraud"] = 1
                    r["scenario"] = "baseline_fraud"
                    # Fraud amount is FRAUD-TYPE dependent, and we used to model
                    # only the expensive case (uniform 1.5-4.0x), which made
                    # amount_dev_ratio the second label proxy. Real card testing
                    # uses TINY amounts on purpose - ULB's real fraud runs 0.42x
                    # the median legitimate amount, while IEEE-CIS e-commerce
                    # fraud runs 1.10x (failure-log 23/24). Generate both, so the
                    # ratio is no longer monotonic in the label.
                    if AMBIENT_AMOUNT_MIXTURE:
                        mult = (float(rng.uniform(0.08, 0.6)) if rng.random() < 0.4
                                else float(rng.uniform(1.2, 3.5)))
                    else:
                        # The ORIGINAL broken model, kept reachable ONLY so the
                        # sensitivity sweep has a true control. Never shipped.
                        mult = float(rng.uniform(1.5, 4.0))
                    r["amount"] = round(max(1.0, r["amount"] * mult), 2)
                rows.append(r)
    return rows


# ------------------------------------------------- attack account ageing
# Real fraudsters BUY AGED ACCOUNTS. Our first generator created every attack
# account on the attack day itself, which made customer_age_days a near-perfect
# label proxy: median attack age 0.98-5.65 days against a legitimate 215.76, so
# TWO features scored what all twenty-two did (failure-log 21).
#
# The fix is not to make attackers look old - it is to stop making them
# uniformly young. Draw from a MIXTURE: `aged_share` of attack accounts are
# aged exactly like the legitimate population (bought, farmed or compromised
# long before the attack), and the rest are genuinely new, because throwaway
# guest checkout is also real.
#
# The shares below were fixed on this rationale BEFORE measuring, and are not
# tuned afterwards: rings buy aged accounts and are the most patient (0.8);
# farms and IP clusters mix bought with fresh (0.6); card testing runs largely
# on disposable guest checkouts, so it stays the youngest (0.5).
# Whether ambient fraud amounts use the two-mode mixture (shipped) or the
# original uniform(1.5, 4.0) (broken). This exists so
# aged_share_sensitivity.py can build a TRUE control that reverts BOTH
# generator fixes at once. Reverting only one produced a control that did
# not reproduce the leak, and the sweep correctly flagged itself as
# untrustworthy rather than reporting a clean pass (failure 26).
AMBIENT_AMOUNT_MIXTURE = True

# Whether to also teach the model that a shared IP can be honest. OFF.
# It removes the corporate-buyer false alarm and raises NPV on seed 7, but on
# seed 101 it drops card-testing detection to mean p=0.004 and loses the attack
# entirely - and we cannot explain why. Missing a whole attack class is worse
# than one false alarm on a legitimate merchant, and shipping an unexplained
# change is worse than shipping a characterised limitation. Left switchable so
# the rejected configuration stays measurable: see failure-log 29 and
# src/policy/config4_npv.py.
HIST_CORPORATE_BUYER = False

AGED_SHARE_RING = 0.8
AGED_SHARE_FARM = 0.6
AGED_SHARE_CLUSTER = 0.6
AGED_SHARE_CARD_TESTING = 0.5


def _attack_created_day(rng, day: int, aged_share: float) -> int:
    """created_day for one attack account. Same draw as a real customer with
    probability `aged_share`, otherwise genuinely recent."""
    if float(rng.random()) < aged_share:
        return int(rng.integers(-400, 0))      # indistinguishable from legit
    return int(day - rng.integers(0, 12))      # genuinely new


# ---------------------------------------------------------------- scenarios
def s1_fraud_spike(w, rng, merchant=3, day=24, tag="s1_fraud_spike", n=180, pfx="S"):
    """Card-testing wave: attacker cycles MANY stolen card numbers through a
    SMALL pool of attacker-controlled devices/IPs on guest-checkout accounts.
    Signals this creates: first-seen instruments, device/IP fan-out, velocity."""
    rows, day_start = [], START_TS + day * DAY
    atk_devices = [oid("d", f"ct{pfx}{i}") for i in range(3)]
    atk_ips = [oid("ip", f"ct{pfx}{i}") for i in range(2)]
    times = _poisson_times(rng, day_start + 10 * 3600, n, hour_bias=False)
    for i, t in enumerate(times):
        r = _base_txn(w, rng, t, merchant)
        r.update(is_fraud=1, scenario=tag,
                 customer_id=oid("c", f"ct{pfx}{i % 60}"),      # throwaway guest accounts
                 device_id=str(rng.choice(atk_devices)),
                 ip=str(rng.choice(atk_ips)),
                 instrument_id=oid("pi", f"stolen{pfx}{i}"),    # a NEW stolen card each time
                 customer_created_day=_attack_created_day(
                     rng, day, AGED_SHARE_CARD_TESTING),
                 payment_method="card",
                 amount=round(float(rng.choice([10, 25, 49, 99]) * rng.uniform(1, 30)), 2))
        rows.append(r)
    return rows


def s2_device_farm(w, rng, merchant=5, day=25, tag="s2_device_farm", pfx="F", n=130):
    """ONE device shared by ~50 fresh accounts using several instruments."""
    rows, day_start = [], START_TS + day * DAY
    farm_device, farm_ip = oid("d", f"farm{pfx}"), oid("ip", f"farm{pfx}")
    instruments = [oid("pi", f"farm{pfx}{i}") for i in range(8)]
    for i, t in enumerate(np.sort(day_start + 12 * 3600 + rng.uniform(0, 5 * 3600, n))):
        acct = oid("c", f"farm{pfx}{i % 50}")  # 50 synthetic accounts
        r = {**_base_txn(w, rng, t, merchant),
             "customer_id": acct, "device_id": farm_device, "ip": farm_ip,
             "instrument_id": str(rng.choice(instruments)),
             "customer_created_day": _attack_created_day(rng, day, AGED_SHARE_FARM),
             "is_fraud": 1, "scenario": tag}
        rows.append(r)
    return rows


def s3_ip_cluster(w, rng, merchant=7, day=26, tag="s3_ip_cluster", pfx="I", n=100):
    """One IP, many accounts, abnormal velocity (each acct has own device)."""
    rows, day_start = [], START_TS + day * DAY
    for i, t in enumerate(np.sort(day_start + 14 * 3600 + rng.uniform(0, 3 * 3600, n))):
        acct = oid("c", f"clu{pfx}{i % 40}")
        r = {**_base_txn(w, rng, t, merchant),
             "customer_id": acct, "device_id": oid("d", f"clu{pfx}{i % 40}"),
             "ip": oid("ip", f"clu{pfx}"),
             "instrument_id": oid("pi", f"clu{pfx}{i % 40}"),
             "customer_created_day": _attack_created_day(rng, day, AGED_SHARE_CLUSTER),
             "is_fraud": 1, "scenario": tag}
        rows.append(r)
    return rows


def s4_account_takeover(w, rng, merchant=2, day=27, n_victims=25, tag="s4_account_takeover", pfx="A"):
    """Established customers suddenly on a new device, new geo, big amounts."""
    rows, day_start = [], START_TS + day * DAY
    victims = rng.choice(w.n_customers, size=n_victims, replace=False)
    for v in victims:
        c = w.customers[int(v)]
        for k in range(int(rng.integers(2, 5))):
            t = day_start + rng.uniform(1 * 3600, 23 * 3600)
            r = _base_txn(w, rng, t, merchant, int(v))
            r.update(device_id=oid("d", f"ato{pfx}{int(v)}"),  # NEW device
                     geo=int((c["home_geo"] + 10) % 20),     # NEW location
                     amount=round(c["avg_amount"] * float(rng.uniform(5, 12)), 2),
                     is_fraud=1, scenario=tag)
            rows.append(r)
    return rows


def s5_fraud_ring(w, rng, merchant=9, day=28, tag="s5_fraud_ring", pfx="R", n=120):
    """15 accounts densely sharing 4 devices, 3 IPs, 5 instruments."""
    rows, day_start = [], START_TS + day * DAY
    devices = [oid("d", f"ring{pfx}{i}") for i in range(4)]
    ips = [oid("ip", f"ring{pfx}{i}") for i in range(3)]
    instruments = [oid("pi", f"ring{pfx}{i}") for i in range(5)]
    accounts = [oid("c", f"ring{pfx}{i}") for i in range(15)]
    for t in np.sort(day_start + 9 * 3600 + rng.uniform(0, 10 * 3600, n)):
        r = {**_base_txn(w, rng, t, merchant),
             "customer_id": str(rng.choice(accounts)),
             "device_id": str(rng.choice(devices)),
             "ip": str(rng.choice(ips)),
             "instrument_id": str(rng.choice(instruments)),
             "customer_created_day": _attack_created_day(rng, day, AGED_SHARE_RING),
             "is_fraud": 1, "scenario": tag}
        rows.append(r)
    return rows


def s7_corporate_buyer(w, rng, merchant=4, day=25, tag="s7_corporate_buyer", n=200):
    """LEGITIMATE shared-entity traffic. A corporate account: ~40 employees
    buying from one office IP on two company cards.

    THIS IS THE HARD NEGATIVE THE FLASH SALE NEVER WAS. Every m11 account has
    its own device, IP and instrument by construction, so the flash sale only
    ever tested "does raw volume fire us" - it never exercised the entity
    layer at all, and m11's entity graph renders empty. This merchant tests
    the thing the product is actually built on: ip_account_count around 40 and
    a large shared component is exactly the signature we escalate on, and here
    it is completely legitimate.

    Built from REAL world customers - aged accounts, their own normal amounts,
    their own devices - so the ONLY thing differing from baseline traffic is
    the shared IP and card. One variable. A false alarm here cannot be blamed
    on account age or amount.
    """
    rows, day_start = [], START_TS + day * DAY
    office_ip = oid("ip", f"corp{merchant}{day}")
    corp_cards = [oid("pi", f"corp{merchant}{day}{i}") for i in range(2)]
    staff = rng.choice(w.n_customers, size=40, replace=False)
    # A working day, not a burst: 09:00-18:00.
    for i, t in enumerate(np.sort(day_start + 9 * 3600 + rng.uniform(0, 9 * 3600, n))):
        cid = int(staff[i % len(staff)])
        r = _base_txn(w, rng, t, merchant, cid)
        r.update(ip=office_ip,
                 instrument_id=str(rng.choice(corp_cards)),
                 scenario=tag)
        rows.append(r)
    return rows


def s8_shared_kiosk_burst(w, rng, merchant=10, day=26, tag="s8_shared_kiosk", n=180):
    """LEGITIMATE, and built to break us. A shared payment terminal - a travel
    desk, campus counter or ticketing kiosk - where many different customers
    transact through ONE device and ONE IP inside a two-hour window.

    Strictly harder than s7: it presents the device-farm signature (one device,
    many accounts) AND compresses into a burst, so if the model scores this
    traffic as risky the merchant's flagged RATE spikes and the detector fires.
    If anything in this project produces a false alarm, it should be this.

    Again built from real customers with their own instruments and amounts, so
    the shared device and IP are the only difference from ordinary traffic.
    """
    rows, day_start = [], START_TS + day * DAY
    kiosk_device = oid("d", f"kiosk{merchant}{day}")
    kiosk_ip = oid("ip", f"kiosk{merchant}{day}")
    walkins = rng.choice(w.n_customers, size=25, replace=False)
    for i, t in enumerate(np.sort(day_start + 13 * 3600 + rng.uniform(0, 2 * 3600, n))):
        cid = int(walkins[i % len(walkins)])
        r = _base_txn(w, rng, t, merchant, cid)
        r.update(device_id=kiosk_device, ip=kiosk_ip, scenario=tag)
        rows.append(r)
    return rows


def s6_flash_sale(w, rng, merchant=11, day=29):
    """LEGITIMATE 6x volume spike: real customers, own devices, normal amounts.
    Ground truth is_fraud=0. The system must NOT flag this merchant."""
    rows, day_start = [], START_TS + day * DAY
    n = 1300  # ~6x normal daily volume
    for t in np.sort(day_start + 11 * 3600 + rng.uniform(0, 6 * 3600, n)):
        cid = int(rng.integers(0, w.n_customers))
        r = _base_txn(w, rng, t, merchant, cid)
        r["amount"] = round(r["amount"] * 0.7, 2)  # discounted prices
        r["scenario"] = "s6_flash_sale"
        if rng.random() < 0.007:  # ambient fraud rate unchanged
            r["is_fraud"] = 1
        rows.append(r)
    return rows


# ---------------------------------------------------------------- entry
def generate(seed: int = 7) -> pd.DataFrame:
    """Historical attacks land in the TRAINING period (days 6-18) on some
    merchants with their own entity ids; the held-out attacks land in the
    TEST period (days 24-29) on DIFFERENT merchants with FRESH entity ids.
    The model learns attack *patterns* from history and must generalize to
    unseen attacks - honest temporal evaluation, no entity leakage."""
    rng = RNG(seed)
    w = World(rng=rng)
    rows = gen_baseline(w, rng)

    # -- historical (train-period) attacks: smaller, different merchants --
    rows += s1_fraud_spike(w, rng, merchant=1, day=6, tag="s1_hist", n=120)
    rows += s2_device_farm(w, rng, merchant=4, day=8, tag="s2_hist", pfx="FH", n=90)
    rows += s3_ip_cluster(w, rng, merchant=6, day=10, tag="s3_hist", pfx="IH", n=70)
    rows += s4_account_takeover(w, rng, merchant=8, day=12, n_victims=18, tag="s4_hist", pfx="AH")
    rows += s5_fraud_ring(w, rng, merchant=10, day=14, tag="s5_hist", pfx="RH", n=80)
    rows += s5_fraud_ring(w, rng, merchant=3, day=18, tag="s5_hist_b", pfx="RH2", n=80)

    # Day 22 lands in the CALIBRATION/VALIDATION slice (days 21-23) on purpose.
    # Without it, validation contained only ambient ~0.6% fraud and no attack at
    # all, so validation PR-AUC scored "can you rank ambient fraud" - not "can
    # you catch an attack", which is what the model is selected to do. Model
    # selection needs at least one attack in the slice it selects on.
    rows += s2_device_farm(w, rng, merchant=0, day=22, tag="s2_hist_b", pfx="FH2", n=90)

    # A LEGITIMATE shared-device merchant in the TRAINING period. Without it the
    # training distribution contains shared-device fraud and no shared-device
    # honest traffic, so the model learns "shared device = fraud" - because in
    # this world it always was. That gap is what made a legitimate kiosk score
    # 0.972 against a device farm's 0.985 (failure-log 29). Same discipline as
    # the historical attacks above: different merchant, different day, entity
    # ids disjoint from the held-out kiosk at m10/day 26.
    rows += s8_shared_kiosk_burst(w, rng, merchant=6, day=16,
                                  tag="s8_hist_kiosk", n=140)
    # The shared-IP counterpart is REACHABLE but OFF by default - see
    # HIST_CORPORATE_BUYER below for why it is rejected. It is left switchable
    # so the rejected configuration can be measured by anyone, rather than
    # asserted to be worse.
    if HIST_CORPORATE_BUYER:
        rows += s7_corporate_buyer(w, rng, merchant=1, day=19,
                                   tag="s7_hist_corporate", n=160)

    # NOT ENABLED, deliberately: the shared-IP counterpart. Training contains
    # shared-IP fraud and no shared-IP honest traffic, which is why a legitimate
    # corporate buyer still false-alarms on seed 11 - the symmetric fix. We
    # measured it: adding s7_corporate_buyer to training does remove that false
    # alarm and raises NPV to INR 9.44L, but on seed 101 it drops card-testing
    # detection to mean p=0.004 (0% flagged) and loses the attack entirely.
    # Missing a whole attack class is a worse failure than one false alarm on a
    # legitimate merchant, and we cannot explain WHY the corporate-buyer
    # examples destabilise card testing. Shipping an unexplained change is worse
    # than shipping a characterised limitation. See failure-log 29.

    # -- held-out (test-period) attacks: novel entities, novel merchants --
    rows += s1_fraud_spike(w, rng)      # m3,  day 24
    rows += s2_device_farm(w, rng)      # m5,  day 25
    rows += s3_ip_cluster(w, rng)       # m7,  day 26
    rows += s4_account_takeover(w, rng) # m2,  day 27
    rows += s5_fraud_ring(w, rng)       # m9,  day 28
    rows += s6_flash_sale(w, rng)       # m11, day 29 (legit - must NOT flag)

    # -- LEGITIMATE shared-entity traffic: the hard negatives (must NOT flag) --
    # The flash sale gives every account its own entities, so it never tested
    # the entity layer. These two do, and s8 is deliberately built to break it.
    rows += s7_corporate_buyer(w, rng)     # m4,  day 25 (legit, shared IP + card)
    rows += s8_shared_kiosk_burst(w, rng)  # m10, day 26 (legit, shared device, bursty)

    df = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
    return df


def generate_holdout(seed: int = 99) -> pd.DataFrame:
    """A SECOND world for held-out agent evaluation.

    Different RNG seed, different merchants, different days from generate().
    Used only to score the investigator on cases that influenced no design
    decision - the ten original cases were looked at repeatedly while building
    tools, so they cannot measure generalisation any more.

    Deliberately NOT used for model training or any reported ML metric."""
    rng = RNG(seed)
    w = World(rng=rng)
    rows = gen_baseline(w, rng)

    # historical attacks so the period looks lived-in
    rows += s2_device_farm(w, rng, merchant=7, day=9, tag="s2_hist_ho", pfx="HOH", n=90)
    rows += s5_fraud_ring(w, rng, merchant=2, day=13, tag="s5_hist_ho", pfx="ROH", n=80)

    # held-out TEST-period attacks on merchants unused by the 10 original cases
    rows += s2_device_farm(w, rng, merchant=0, day=25, tag="s2_device_farm",
                            pfx="HO_F", n=130)
    rows += s5_fraud_ring(w, rng, merchant=4, day=27, tag="s5_fraud_ring",
                            pfx="HO_R", n=120)
    # merchant 10 gets no injected attack - the quiet held-out case
    return pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)


if __name__ == "__main__":
    df = generate()
    df.to_csv("data/transactions.csv", index=False)
    print(f"{len(df):,} transactions | fraud rate {df.is_fraud.mean():.3%}")
    print(df.groupby("scenario").agg(n=("ts", "size"), fraud=("is_fraud", "mean")))
