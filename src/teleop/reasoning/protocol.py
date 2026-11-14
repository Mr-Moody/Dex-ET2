"""Fleet reasoning contract (spec 6.2). T-161 implements a Verda Nemotron reasoner against this.

Do not change these signatures without updating every reasoner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from teleop.compliance.protocol import ComplianceKind


@dataclass(frozen=True, slots=True)
class ReasoningContext:
    event_kind: ComplianceKind
    bay_id: int
    part_sku_id: str
    confidence: float
    x_bolt_m: float
    tau_ext_inf_nm: float
    matched_signature_id: str | None
    mahalanobis_distance: float | None
    is_ood: bool
    rollout_passed: int | None = None
    rollout_total: int | None = None


@dataclass(frozen=True, slots=True)
class ReasoningVerdict:
    headline: str
    operator_guidance: str
    compliance_rationale: str
    model: str
    latency_ms: float


@runtime_checkable
class FleetReasoner(Protocol):
    def submit(self, ctx: ReasoningContext) -> None:
        """Queue a context for triage. Must not block the 60 Hz loop and never raise."""
        ...

    def latest_verdict(self, bay_id: int) -> ReasoningVerdict | None: ...

    def close(self) -> None: ...
