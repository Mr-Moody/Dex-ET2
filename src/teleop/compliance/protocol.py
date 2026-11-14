"""Compliance event contract (spec 2026-11-14 section 6.1). Owned by T-100; do not change."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
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
    metadata: Mapping[str, Any]


@runtime_checkable
class ComplianceSink(Protocol):
    def emit(self, event: ComplianceEvent) -> None: ...
    def close(self) -> None: ...
