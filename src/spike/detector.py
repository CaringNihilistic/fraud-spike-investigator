"""Merchant-level fraud-spike detector: EWMA + rolling z-score change-point.

Operates ONLINE on hourly buckets of per-transaction model scores.
For each merchant-hour it tracks:
  score_rate = share of transactions with p_fraud above a reference cut
It flags a spike when the EWMA of score_rate exceeds the merchant's own
rolling baseline by z_threshold standard deviations AND absolute volume
is non-trivial (min_txn) - so a 3-txn hour can't fire the alarm.

Crucially this fires on the FRAUD-SCORE rate, not raw volume, so a
legitimate flash sale (volume up, score rate flat) does NOT trigger.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field


@dataclass
class MerchantState:
    ewma: float = 0.0
    initialized: bool = False
    history: deque = field(default_factory=lambda: deque(maxlen=72))  # 3 days of hours


@dataclass
class StreamState:
    window: deque = field(default_factory=lambda: deque(maxlen=30))  # (ts, score)
    fired_at: int | None = None
    # Slow-moving baseline of this merchant's hot-rate, for the z signal that
    # feeds risk fusion. Deliberately NOT used by the firing rule - firing
    # stays the simple, explainable "k of last n within a bounded span".
    baseline: float = 0.0
    baseline_var: float = 0.0
    baseline_n: int = 0


class StreamingSpikeDetector:
    """Fast-path, event-driven detector for minutes-level time-to-detect.

    Fires for a merchant the moment >= k_hot of its last `window` transactions
    score >= score_cut AND those transactions span <= max_span_s (so stale
    history on a quiet merchant can never assemble an alarm).

    Explainable in one sentence: "12 of this merchant's last 30 transactions
    were high-risk within a short span." Volume-adaptive by construction:
    high-volume merchants accumulate evidence in minutes.
    Complements the hourly EWMA detector (slow path / baseline drift).
    """

    def __init__(self, window: int = 30, k_hot: int = 8, min_rate: float = 0.25,
                 score_cut: float = 0.5, max_span_s: int = 6 * 3600,
                 baseline_alpha: float = 0.01, z_noise_floor: float = 0.03):
        self.window, self.k_hot, self.min_rate = window, k_hot, min_rate
        self.score_cut, self.max_span_s = score_cut, max_span_s
        # Slow alpha: the baseline must represent long-run normal, not chase
        # the attack it is supposed to stand out against.
        self.baseline_alpha, self.z_noise_floor = baseline_alpha, z_noise_floor
        self.state: dict[str, StreamState] = defaultdict(StreamState)

    def hot_rate(self, merchant_id: str) -> float:
        """Share of the merchant's current window scoring above score_cut."""
        st = self.state[merchant_id]
        if not st.window:
            return 0.0
        return sum(s >= self.score_cut for _, s in st.window) / len(st.window)

    def baseline_rate(self, merchant_id: str) -> float:
        """This merchant's slow-EWMA normal flagged-rate. Read-only; exposed so
        the dashboard shows the SAME baseline the detector reasons about rather
        than inventing a second, contradictory one."""
        return self.state[merchant_id].baseline

    def spike_z(self, merchant_id: str) -> float:
        """Standardized deviation of the current hot-rate from this merchant's
        own slow baseline. Read-only signal for risk fusion - never consulted
        by the firing rule. Returns 0.0 before the baseline has warmed up, so
        a brand-new merchant cannot manufacture a large z from one hot txn."""
        st = self.state[merchant_id]
        if st.baseline_n < self.window:
            return 0.0
        std = max(st.baseline_var ** 0.5, self.z_noise_floor)
        return max(0.0, (self.hot_rate(merchant_id) - st.baseline) / std)

    def update(self, merchant_id: str, ts: int, score: float) -> int | None:
        """Feed one transaction. Returns fire timestamp on the txn that
        first crosses the bar (once per merchant), else None.

        Fires when the last <=window txns contain >= k_hot high-risk scores
        AND the high-risk RATE within the window is >= min_rate AND the
        window spans <= max_span_s. The rate + span guards are what a raw
        flag-counter lacks: ambient fraud drip-accumulating over days can
        never assemble an alarm here."""
        st = self.state[merchant_id]
        st.window.append((ts, score))
        span = st.window[-1][0] - st.window[0][0]
        hot = sum(s >= self.score_cut for _, s in st.window)
        rate = hot / len(st.window)

        # Baseline update happens on EVERY txn, including after firing, and
        # AFTER the z-signal for this txn would have been read - same
        # "decide, then update state" discipline as the rest of the pipeline.
        a = self.baseline_alpha
        if st.baseline_n == 0:
            st.baseline = rate
        else:
            dev = rate - st.baseline
            st.baseline += a * dev
            st.baseline_var = (1 - a) * (st.baseline_var + a * dev * dev)
        st.baseline_n += 1

        if st.fired_at is not None:
            return None
        if (hot >= self.k_hot and rate >= self.min_rate
                and span <= self.max_span_s):
            st.fired_at = ts
            return ts
        return None


@dataclass
class SpikeEvent:
    merchant_id: str
    hour_ts: int
    score_rate: float
    baseline: float
    z: float
    n_txn: int


class SpikeDetector:
    def __init__(self, alpha: float = 0.35, z_threshold: float = 4.0,
                 score_cut: float = 0.5, min_txn: int = 10, warmup_hours: int = 24):
        self.alpha = alpha
        self.z_threshold = z_threshold
        self.score_cut = score_cut
        self.min_txn = min_txn
        self.warmup_hours = warmup_hours
        self.state: dict[str, MerchantState] = defaultdict(MerchantState)

    def update_hour(self, merchant_id: str, hour_ts: int,
                    scores: list[float]) -> SpikeEvent | None:
        st = self.state[merchant_id]
        n = len(scores)
        rate = (sum(s >= self.score_cut for s in scores) / n) if n else 0.0

        # baseline stats from history BEFORE this hour
        hist = list(st.history)
        spike = None
        if len(hist) >= self.warmup_hours and n >= self.min_txn:
            mean = sum(hist) / len(hist)
            var = sum((h - mean) ** 2 for h in hist) / len(hist)
            std = max(var ** 0.5, 0.03)  # noise floor: 1-2 stray fraud txns in a
            #                              low-volume hour must not fire the alarm
            ewma_now = self.alpha * rate + (1 - self.alpha) * (st.ewma if st.initialized else rate)
            z = (ewma_now - mean) / std
            if z >= self.z_threshold:
                spike = SpikeEvent(merchant_id, hour_ts, rate, mean, round(z, 2), n)

        # state update AFTER decision
        st.ewma = self.alpha * rate + (1 - self.alpha) * (st.ewma if st.initialized else rate)
        st.initialized = True
        st.history.append(rate)
        return spike
