"""Append-only audit log for every agent tool call.

Records (tool, inputs hash, output hash, ts) - hashes rather than payloads so
the log stays small and contains no raw transaction data, while still proving
after the fact that a given call returned a given result. An investigation
that cannot be replayed and checked is not defensible to a risk team.

The log is also how we prove the agent stayed READ-ONLY: every tool it is
allowed to call appears here, and none of them mutate state.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field


def _hash(obj) -> str:
    """Stable content hash. sort_keys so key order can never change the hash;
    default=str so numpy/pandas scalars don't blow up serialization."""
    payload = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass
class AuditEntry:
    ts: int              # logical step counter, not wall-clock: deterministic replay
    tool: str
    inputs_hash: str
    output_hash: str
    ok: bool
    error: str | None = None


@dataclass
class AuditLog:
    entries: list[AuditEntry] = field(default_factory=list)

    def record(self, tool: str, inputs, output, ok: bool = True,
               error: str | None = None) -> AuditEntry:
        e = AuditEntry(ts=len(self.entries), tool=tool,
                       inputs_hash=_hash(inputs), output_hash=_hash(output),
                       ok=ok, error=error)
        self.entries.append(e)
        return e

    def tools_called(self) -> list[str]:
        return [e.tool for e in self.entries]

    def to_records(self) -> list[dict]:
        return [asdict(e) for e in self.entries]
