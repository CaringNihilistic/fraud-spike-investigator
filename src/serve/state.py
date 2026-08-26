"""In-memory pipeline state shared by the replay driver and the API.

Single process, single lock, plain dicts - no Redis, no DB, no queue. The
whole system is replayable from the transaction stream, so durable state
would be ceremony: if you want the state back, replay it.

Everything here is written by ONE writer (the replay thread) and read by the
API request handlers, guarded by a single RLock. Snapshots are deep-ish copies
so a handler can never observe a half-updated merchant.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field

from src.policy.engine import Action

HIGH_RISK_CUT = 0.5


@dataclass
class MerchantState:
    merchant_id: str
    txn_count: int = 0
    flagged_count: int = 0
    exposure_inr: float = 0.0          # ₹ of flagged transactions
    in_spike: bool = False
    spike_started_ts: int | None = None
    risk_score: float = 0.0            # ROLLING peak, not the last txn - see below
    confidence: float = 0.0
    last_ts: int = 0
    # Baseline and current rate come from the SPIKE DETECTOR's own slow EWMA
    # rather than a second definition invented here. A naive "first 200 txns"
    # baseline is contaminated whenever the attack starts early in the slice
    # (m3's card-testing wave is on day 24, the first test day), which made the
    # dashboard read "0.3 -> 0.0" - i.e. improving - during an active attack.
    baseline_rate: float = 0.0
    current_rate: float = 0.0
    spike_z: float = 0.0
    # High-water marks. The rolling gauge cools once the burst leaves the
    # window, which is right for a live board but erases the story for anyone
    # who looks after the fact - so keep what the peak actually was.
    # NOTE: peak *risk* is deliberately NOT tracked. Every merchant has at least
    # one ambient-fraud transaction scoring ~100, so peak single-txn risk came
    # out 100 for all 12 merchants including the flash sale - it discriminates
    # nothing. Peak flagged-RATE and peak z do (93% / z=5.8 for an attack vs
    # 3% / z=1.1 for a quiet merchant), because they are merchant-level.
    peak_rate_ever: float = 0.0
    peak_z_ever: float = 0.0
    # Rolling risk window: the gauge shows how hot this merchant is NOW, not
    # whatever its most recent single transaction happened to score.
    recent_risk: deque = field(default_factory=lambda: deque(maxlen=50))
    action_mix: dict = field(default_factory=lambda: defaultdict(int))
    investigation: dict | None = None
    entities: dict = field(default_factory=lambda: {
        "device": defaultdict(set), "ip": defaultdict(set), "instrument": defaultdict(set)})

    def fraud_rate_delta(self) -> dict:
        base, recent = self.baseline_rate, self.current_rate
        return {"baseline_rate": round(base, 4), "current_rate": round(recent, 4),
                "delta_multiple": (round(recent / base, 2)
                                   if base > 0.001 and recent > base else None),
                "delta_points": round((recent - base) * 100, 2),
                "spike_z": round(self.spike_z, 2)}

    def peak_risk(self) -> float:
        """Rolling peak over the recent window. A merchant under attack stays
        visibly hot for as long as the attack is in its recent history, then
        cools as the window rolls past it - which is the behaviour an analyst
        watching a live board expects."""
        return max(self.recent_risk) if self.recent_risk else 0.0

    def to_summary(self) -> dict:
        d = self.fraud_rate_delta()
        return {"merchant_id": self.merchant_id, "txn_count": self.txn_count,
                "flagged_count": self.flagged_count,
                "exposure_inr": round(self.exposure_inr, 2),
                "in_spike": self.in_spike, "spike_started_ts": self.spike_started_ts,
                "risk_score": round(self.peak_risk(), 1),
                "last_txn_risk": round(self.risk_score, 1),
                "peak_rate_ever": round(self.peak_rate_ever, 4),
                "peak_z_ever": round(self.peak_z_ever, 2),
                "confidence": round(self.confidence, 3),
                "last_ts": self.last_ts, "fraud_rate": d,
                "action_mix": dict(self.action_mix),
                "top_cause": (self.investigation or {}).get("cause"),
                "has_investigation": self.investigation is not None}


@dataclass
class ReviewCase:
    case_id: int
    merchant_id: str
    ts: int
    amount_inr: float
    risk_score: float
    confidence: float
    system_action: str
    reason: str
    customer_id: str
    device_id: str
    # analyst decision, None until a human acts
    analyst_action: str | None = None
    analyst_note: str | None = None

    def to_dict(self) -> dict:
        return {"case_id": self.case_id, "merchant_id": self.merchant_id, "ts": self.ts,
                "amount_inr": round(self.amount_inr, 2),
                "risk_score": round(self.risk_score, 1), "confidence": self.confidence,
                "system_action": self.system_action, "reason": self.reason,
                "customer_id": self.customer_id, "device_id": self.device_id,
                "analyst_action": self.analyst_action, "analyst_note": self.analyst_note,
                "overridden": (self.analyst_action is not None
                               and self.analyst_action != self.system_action)}


class PipelineState:
    def __init__(self):
        self._lock = threading.RLock()
        self.merchants: dict[str, MerchantState] = {}
        self.review_queue: list[ReviewCase] = []
        self.events: deque = deque(maxlen=300)      # demo narration feed
        self.audit: list[dict] = []
        self.processed = 0
        self.total = 0
        self.started_at: float | None = None
        self.speed = 200.0                          # txns per second
        self.paused = False
        self.finished = False
        self._next_case_id = 1

    # ---------------------------------------------------------- writes
    def merchant(self, mid: str) -> MerchantState:
        if mid not in self.merchants:
            self.merchants[mid] = MerchantState(merchant_id=mid)
        return self.merchants[mid]

    def record_txn(self, row, risk: float, confidence: float, action: Action,
                   reason: str, spiking: bool, spike_ts: int | None,
                   baseline_rate: float = 0.0, current_rate: float = 0.0,
                   spike_z: float = 0.0):
        """One transaction through the pipeline. Called only by the replay thread."""
        with self._lock:
            m = self.merchant(row["merchant_id"])
            flagged = float(row["p"]) >= HIGH_RISK_CUT
            m.txn_count += 1
            m.last_ts = int(row["ts"])
            m.risk_score, m.confidence = risk or 0.0, confidence
            m.recent_risk.append(risk or 0.0)
            m.baseline_rate, m.current_rate, m.spike_z = baseline_rate, current_rate, spike_z
            m.peak_rate_ever = max(m.peak_rate_ever, current_rate)
            m.peak_z_ever = max(m.peak_z_ever, spike_z)
            m.action_mix[action.value] += 1
            if flagged:
                m.flagged_count += 1
                m.exposure_inr += float(row["amount"])
                for kind, col in (("device", "device_id"), ("ip", "ip"),
                                  ("instrument", "instrument_id")):
                    m.entities[kind][row[col]].add(row["customer_id"])
            if spiking and not m.in_spike:
                m.in_spike, m.spike_started_ts = True, spike_ts
                self.log_event("spike", f"{m.merchant_id} entered spike state",
                               merchant_id=m.merchant_id, ts=int(row["ts"]))
            if action in (Action.REVIEW, Action.RESTRICT):
                self.review_queue.append(ReviewCase(
                    case_id=self._next_case_id, merchant_id=row["merchant_id"],
                    ts=int(row["ts"]), amount_inr=float(row["amount"]),
                    risk_score=risk or 0.0, confidence=confidence,
                    system_action=action.value, reason=reason,
                    customer_id=row["customer_id"], device_id=row["device_id"]))
                self._next_case_id += 1
            self.processed += 1

    def log_event(self, kind: str, message: str, **extra):
        with self._lock:
            self.events.append({"kind": kind, "message": message,
                                "at": round(time.time(), 3), **extra})

    def set_investigation(self, mid: str, report: dict, audit_entries: list,
                          degraded: bool, validated_action: str):
        with self._lock:
            m = self.merchant(mid)
            m.investigation = {**report, "degraded": degraded,
                               "validated_action": validated_action,
                               "audit": audit_entries}
            self.audit.extend([{"merchant_id": mid, **e} for e in audit_entries])
            self.log_event("investigation",
                           f"{mid}: {report.get('cause')} -> {validated_action}"
                           + (" (degraded)" if degraded else ""),
                           merchant_id=mid)

    def decide_case(self, case_id: int, analyst_action: str, note: str | None):
        with self._lock:
            for c in self.review_queue:
                if c.case_id == case_id:
                    c.analyst_action, c.analyst_note = analyst_action, note
                    self.log_event("analyst",
                                   f"case {case_id} ({c.merchant_id}): analyst chose "
                                   f"{analyst_action}"
                                   + (" [OVERRIDE]" if analyst_action != c.system_action else ""),
                                   merchant_id=c.merchant_id)
                    return c.to_dict()
            return None

    # ---------------------------------------------------------- reads
    def snapshot_merchants(self) -> list[dict]:
        with self._lock:
            return sorted((m.to_summary() for m in self.merchants.values()),
                          key=lambda d: (-d["risk_score"] if d["in_spike"] else 0,
                                         -d["exposure_inr"]))

    def snapshot_merchant(self, mid: str) -> dict | None:
        with self._lock:
            m = self.merchants.get(mid)
            return m.to_summary() if m else None

    def entity_graph(self, mid: str, min_accounts: int = 2, max_nodes: int = 60) -> dict:
        """Nodes/links for the force layout: entities that FAN OUT across
        several accounts, which is the shape that indicates a ring or farm.
        Entities touching one account are dropped - they are the boring case
        and would swamp the picture."""
        with self._lock:
            m = self.merchants.get(mid)
            if not m:
                return {"nodes": [], "links": []}
            nodes, links, seen = [], [], set()

            def add(nid, kind, label, size):
                if nid not in seen:
                    seen.add(nid)
                    nodes.append({"id": nid, "kind": kind, "label": label, "size": size})

            for kind in ("device", "ip", "instrument"):
                for ent, accounts in m.entities[kind].items():
                    if len(accounts) < min_accounts or len(nodes) > max_nodes:
                        continue
                    eid = f"{kind}:{ent}"
                    add(eid, kind, str(ent), len(accounts))
                    for acct in list(accounts)[:12]:
                        aid = f"customer:{acct}"
                        add(aid, "customer", str(acct), 1)
                        links.append({"source": eid, "target": aid})
            return {"merchant_id": mid, "nodes": nodes, "links": links,
                    "note": f"entities shared by >={min_accounts} accounts among "
                            f"flagged transactions"}

    def snapshot_queue(self, pending_only: bool = False) -> dict:
        """Recent cases plus TRUE totals. The row list is capped for the wire,
        but the counts must come from the full queue — the header and the
        queue panel quoting two different numbers reads as a bug on screen."""
        with self._lock:
            q = [c.to_dict() for c in self.review_queue]
            pending_total = sum(1 for c in q if c["analyst_action"] is None)
            if pending_only:
                q = [c for c in q if c["analyst_action"] is None]
            return {"cases": q[-200:], "total_cases": len(self.review_queue),
                    "pending_total": pending_total}

    def status(self) -> dict:
        with self._lock:
            elapsed = (time.time() - self.started_at) if self.started_at else 0.0
            return {"processed": self.processed, "total": self.total,
                    "pct": round(100.0 * self.processed / max(1, self.total), 2),
                    "speed_tps": self.speed, "paused": self.paused,
                    "finished": self.finished, "elapsed_s": round(elapsed, 1),
                    "merchants_in_spike": sum(1 for m in self.merchants.values() if m.in_spike),
                    "review_pending": sum(1 for c in self.review_queue
                                          if c.analyst_action is None),
                    "events": list(self.events)[-40:]}


STATE = PipelineState()
