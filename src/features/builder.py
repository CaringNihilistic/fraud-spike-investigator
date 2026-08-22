"""Leakage-safe feature builder.

Iterates transactions in STRICT chronological order. For each transaction,
features are computed from state accumulated from PRIOR transactions only,
then state is updated. This guarantees no future information leaks into
any feature - the property temporal holdout evaluation depends on.

Feature groups (~22 features):
  basics        - amount, log-amount, hour, night flag, payment method, geo-mismatch
  velocity      - customer & merchant counts/amounts in 5m / 1h windows
  entity history- accounts-per-device, accounts-per-IP, customers-per-instrument,
                  first-seen flags, customer account age & amount deviation
  graph-lite    - union-find connected-component size over shared
                  device/IP/instrument links (incremental, O(alpha) per edge)
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

PM_CODES = {"card": 0, "upi": 1, "netbanking": 2, "wallet": 3}


class UnionFind:
    def __init__(self):
        self.parent: dict = {}
        self.size: dict = {}

    def find(self, x):
        p = self.parent.setdefault(x, x)
        self.size.setdefault(x, 1)
        while p != self.parent[p]:
            self.parent[p] = self.parent[self.parent[p]]
            p = self.parent[p]
        self.parent[x] = p
        return p

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]

    def comp_size(self, x) -> int:
        return self.size[self.find(x)]


class _Window:
    """Sliding time-window counter for (count, amount_sum)."""

    def __init__(self, seconds: int):
        self.seconds = seconds
        self.q: dict[str, deque] = defaultdict(deque)
        self.amt: dict[str, float] = defaultdict(float)

    def query(self, key, now):
        q = self.q[key]
        while q and q[0][0] <= now - self.seconds:
            ts, a = q.popleft()
            self.amt[key] -= a
        return len(q), self.amt[key]

    def add(self, key, now, amount):
        self.q[key].append((now, amount))
        self.amt[key] += amount


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("ts").reset_index(drop=True)

    cust_5m, cust_1h = _Window(300), _Window(3600)
    merch_5m, merch_1h = _Window(300), _Window(3600)

    device_accounts = defaultdict(set)
    ip_accounts = defaultdict(set)
    instrument_customers = defaultdict(set)
    customer_devices = defaultdict(set)
    customer_geos = defaultdict(set)
    cust_amount_hist = defaultdict(lambda: (0, 0.0))  # n, mean (Welford-lite)
    uf = UnionFind()

    out = np.zeros((len(df), 22), dtype=np.float64)
    cols = [
        "amount", "log_amount", "hour", "is_night", "payment_method_code", "geo_mismatch",
        "cust_txn_5m", "cust_txn_1h", "cust_amt_1h",
        "merch_txn_5m", "merch_txn_1h", "merch_amt_1h",
        "device_account_count", "ip_account_count", "instrument_customer_count",
        "is_new_device_for_customer", "is_first_seen_device", "is_first_seen_ip",
        "customer_age_days", "amount_dev_ratio", "cust_txn_hist_n", "component_size",
    ]

    ts0 = int(df["ts"].iloc[0])
    for i, r in enumerate(df.itertuples(index=False)):
        now, cust, merch = r.ts, r.customer_id, r.merchant_id
        dev, ip, pi = r.device_id, r.ip, r.instrument_id

        hour = ((now - ts0) / 3600) % 24
        n_hist, mean_hist = cust_amount_hist[cust]
        c5n, _ = cust_5m.query(cust, now)
        c1n, c1a = cust_1h.query(cust, now)
        m5n, _ = merch_5m.query(merch, now)
        m1n, m1a = merch_1h.query(merch, now)

        # graph-lite: component size BEFORE adding this txn's edges
        comp = max(uf.comp_size(("d", dev)) if ("d", dev) in uf.parent else 1,
                   uf.comp_size(("ip", ip)) if ("ip", ip) in uf.parent else 1,
                   uf.comp_size(("pi", pi)) if ("pi", pi) in uf.parent else 1)

        out[i] = (
            r.amount, np.log1p(r.amount), hour, float(hour < 6 or hour > 23),
            PM_CODES.get(r.payment_method, 0),
            float(len(customer_geos[cust]) > 0 and r.geo not in customer_geos[cust]),
            c5n, c1n, c1a, m5n, m1n, m1a,
            len(device_accounts[dev]), len(ip_accounts[ip]), len(instrument_customers[pi]),
            float(len(customer_devices[cust]) > 0 and dev not in customer_devices[cust]),
            float(len(device_accounts[dev]) == 0), float(len(ip_accounts[ip]) == 0),
            max(0.0, (now - 1_760_000_000) / 86400 - r.customer_created_day),
            (r.amount / mean_hist) if n_hist >= 3 and mean_hist > 0 else 1.0,
            n_hist, comp,
        )

        # ---- state update (AFTER feature emission) ----
        cust_5m.add(cust, now, r.amount); cust_1h.add(cust, now, r.amount)
        merch_5m.add(merch, now, r.amount); merch_1h.add(merch, now, r.amount)
        device_accounts[dev].add(cust); ip_accounts[ip].add(cust)
        instrument_customers[pi].add(cust); customer_devices[cust].add(dev)
        customer_geos[cust].add(r.geo)
        cust_amount_hist[cust] = (n_hist + 1, (mean_hist * n_hist + r.amount) / (n_hist + 1))
        uf.union(("c", cust), ("d", dev)); uf.union(("c", cust), ("ip", ip))
        uf.union(("c", cust), ("pi", pi))

    feats = pd.DataFrame(out, columns=cols)
    merged = pd.concat([df.reset_index(drop=True), feats], axis=1)
    return merged.loc[:, ~merged.columns.duplicated(keep="last")]


FEATURE_COLS = [
    "amount", "log_amount", "hour", "is_night", "payment_method_code", "geo_mismatch",
    "cust_txn_5m", "cust_txn_1h", "cust_amt_1h",
    "merch_txn_5m", "merch_txn_1h", "merch_amt_1h",
    "device_account_count", "ip_account_count", "instrument_customer_count",
    "is_new_device_for_customer", "is_first_seen_device", "is_first_seen_ip",
    "customer_age_days", "amount_dev_ratio", "cust_txn_hist_n", "component_size",
]
