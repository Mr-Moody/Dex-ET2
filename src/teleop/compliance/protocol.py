"""Compliance event contract (spec 6.1). T-160 implements a Solana sink against this.

Do not change these signatures without updating every sink: the coordinator, HUD and the
Solana teammate track all depend on them.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

ComplianceKind = Literal[
    "STALL_ALERT",
    "RESCUE_LOGGED",
    "VALIDATION_GATE",
    "SKILL_DEPLOYED",
    "OOD_REFUSAL",
]


@dataclass(frozen=True, slots=True)
class ComplianceEvent:
    kind: ComplianceKind
    bay_id: int
    part_sku_id: str
    timestamp_ns: int
    payload_sha256: str
    summary: str
    metadata: Mapping[str, Any] = field(default_factory=dict[str, Any])

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "bay_id": self.bay_id,
            "part_sku_id": self.part_sku_id,
            "timestamp_ns": self.timestamp_ns,
            "payload_sha256": self.payload_sha256,
            "summary": self.summary,
            "metadata": dict(self.metadata),
        }


@runtime_checkable
class ComplianceSink(Protocol):
    def emit(self, event: ComplianceEvent) -> None:
        """Record an event. Must return in well under a millisecond and never raise."""
        ...

    def close(self) -> None: ...


def payload_digest(
    kind: ComplianceKind,
    bay_id: int,
    part_sku_id: str,
    timestamp_ns: int,
    summary: str,
    metadata: Mapping[str, Any],
) -> str:
    """SHA-256 over canonical JSON (sorted keys, no whitespace) of everything but the digest."""
    canonical = json.dumps(
        {
            "kind": kind,
            "bay_id": bay_id,
            "part_sku_id": part_sku_id,
            "timestamp_ns": timestamp_ns,
            "summary": summary,
            "metadata": dict(metadata),
        },
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def make_event(
    kind: ComplianceKind,
    bay_id: int,
    part_sku_id: str,
    summary: str,
    metadata: Mapping[str, Any] | None = None,
    timestamp_ns: int | None = None,
) -> ComplianceEvent:
    """Build an event with its payload digest; ``timestamp_ns`` defaults to ``time.time_ns()``."""
    meta: dict[str, Any] = dict(metadata or {})
    ts = time.time_ns() if timestamp_ns is None else timestamp_ns
    digest = payload_digest(kind, bay_id, part_sku_id, ts, summary, meta)
    return ComplianceEvent(kind, bay_id, part_sku_id, ts, digest, summary, meta)
