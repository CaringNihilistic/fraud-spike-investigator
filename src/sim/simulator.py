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

import numpy as np
import pandas as pd
from dataclasses import dataclass, field

RNG = np.random.default_rng

DAY = 24 * 3600
START_TS = 1_760_000_000  # arbitrary epoch anchor
N_DAYS = 30

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
                "device": f"d{cid}",             # personal device
                "ip": f"ip{int(self.rng.integers(0, 1500))}",  # shared ISP pool
                "instrument": f"pi{cid}",
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
        "customer_id": f"c{cid}",
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
                    r["amount"] = round(r["amount"] * float(rng.uniform(1.5, 4.0)), 2)
                rows.append(r)
    return rows


# ---------------------------------------------------------------- scenarios
def s1_fraud_spike(w, rng, merchant=3, day=24, tag="s1_fraud_spike", n=180, pfx="S"):
    """Card-testing wave: attacker cycles MANY stolen card numbers through a
    SMALL pool of attacker-controlled devices/IPs on guest-checkout accounts.
    Signals this creates: first-seen instruments, device/IP fan-out, velocity."""
    rows, day_start = [], START_TS + day * DAY
    atk_devices = [f"d_CT_{pfx}{i}" for i in range(3)]
    atk_ips = [f"ip_CT_{pfx}{i}" for i in range(2)]
    times = _poisson_times(rng, day_start + 10 * 3600, n, hour_bias=False)
    for i, t in enumerate(times):
        r = _base_txn(w, rng, t, merchant)
        r.update(is_fraud=1, scenario=tag,
                 customer_id=f"c_CT_{pfx}{i % 60}",            # throwaway guest accounts
                 device_id=str(rng.choice(atk_devices)),
                 ip=str(rng.choice(atk_ips)),
                 instrument_id=f"pi_STOLEN_{pfx}{i}",          # a NEW stolen card each time
                 customer_created_day=int(day),
                 payment_method="card",
                 amount=round(float(rng.choice([10, 25, 49, 99]) * rng.uniform(1, 30)), 2))
        rows.append(r)
    return rows


def s2_device_farm(w, rng, merchant=5, day=25, tag="s2_device_farm", pfx="F", n=130):
    """ONE device shared by ~50 fresh accounts using several instruments."""
    rows, day_start = [], START_TS + day * DAY
    farm_device, farm_ip = f"d_FARM_{pfx}", f"ip_FARM_{pfx}"
    instruments = [f"pi_FARM_{pfx}{i}" for i in range(8)]
    for i, t in enumerate(np.sort(day_start + 12 * 3600 + rng.uniform(0, 5 * 3600, n))):
        acct = f"c{pfx}{i % 50}"  # 50 synthetic accounts
        r = {**_base_txn(w, rng, t, merchant),
             "customer_id": acct, "device_id": farm_device, "ip": farm_ip,
             "instrument_id": str(rng.choice(instruments)),
             "customer_created_day": int(day - rng.integers(0, 3)),  # brand-new accounts
             "is_fraud": 1, "scenario": tag}
        rows.append(r)
    return rows


def s3_ip_cluster(w, rng, merchant=7, day=26, tag="s3_ip_cluster", pfx="I", n=100):
    """One IP, many accounts, abnormal velocity (each acct has own device)."""
    rows, day_start = [], START_TS + day * DAY
    for i, t in enumerate(np.sort(day_start + 14 * 3600 + rng.uniform(0, 3 * 3600, n))):
        acct = f"c{pfx}{i % 40}"
        r = {**_base_txn(w, rng, t, merchant),
             "customer_id": acct, "device_id": f"d{pfx}{i % 40}", "ip": f"ip_CLUSTER_{pfx}",
             "instrument_id": f"pi{pfx}{i % 40}",
             "customer_created_day": int(day - rng.integers(0, 5)),
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
            r.update(device_id=f"d_ATO_{pfx}{int(v)}",       # NEW device
                     geo=int((c["home_geo"] + 10) % 20),     # NEW location
                     amount=round(c["avg_amount"] * float(rng.uniform(5, 12)), 2),
                     is_fraud=1, scenario=tag)
            rows.append(r)
    return rows


def s5_fraud_ring(w, rng, merchant=9, day=28, tag="s5_fraud_ring", pfx="R", n=120):
    """15 accounts densely sharing 4 devices, 3 IPs, 5 instruments."""
    rows, day_start = [], START_TS + day * DAY
    devices = [f"d_RING_{pfx}{i}" for i in range(4)]
    ips = [f"ip_RING_{pfx}{i}" for i in range(3)]
    instruments = [f"pi_RING_{pfx}{i}" for i in range(5)]
    accounts = [f"c{pfx}{i}" for i in range(15)]
    for t in np.sort(day_start + 9 * 3600 + rng.uniform(0, 10 * 3600, n)):
        r = {**_base_txn(w, rng, t, merchant),
             "customer_id": str(rng.choice(accounts)),
             "device_id": str(rng.choice(devices)),
             "ip": str(rng.choice(ips)),
             "instrument_id": str(rng.choice(instruments)),
             "customer_created_day": int(day - rng.integers(0, 10)),
             "is_fraud": 1, "scenario": tag}
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

    # -- held-out (test-period) attacks: novel entities, novel merchants --
    rows += s1_fraud_spike(w, rng)      # m3,  day 24
    rows += s2_device_farm(w, rng)      # m5,  day 25
    rows += s3_ip_cluster(w, rng)       # m7,  day 26
    rows += s4_account_takeover(w, rng) # m2,  day 27
    rows += s5_fraud_ring(w, rng)       # m9,  day 28
    rows += s6_flash_sale(w, rng)       # m11, day 29 (legit - must NOT flag)

    df = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = generate()
    df.to_csv("data/transactions.csv", index=False)
    print(f"{len(df):,} transactions | fraud rate {df.is_fraud.mean():.3%}")
    print(df.groupby("scenario").agg(n=("ts", "size"), fraud=("is_fraud", "mean")))
