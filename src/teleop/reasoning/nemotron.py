"""Verda Nemotron fleet reasoner. T-100 placeholder; T-161 (Verda teammate) owns this file.

``VerdaNemotronPlaceholder`` makes no network calls: ``submit()`` formats a deterministic
template verdict in microseconds and stores it per bay. T-161 keeps it as the offline fallback.
"""

from __future__ import annotations

import os
import threading
import time

from teleop.reasoning.protocol import ReasoningContext, ReasoningVerdict

OOD_THRESHOLD = 3.0  # gamma_ood, spec 4.2; quoted in rationales only


def _cm(x_m: float) -> str:
    return f"{x_m * 100.0:.1f} cm"


def _distance(ctx: ReasoningContext) -> str:
    d = ctx.mahalanobis_distance
    return "n/a" if d is None else f"{d:.2f}"


def triage(ctx: ReasoningContext) -> tuple[str, str, str]:
    """Deterministic (headline, operator_guidance, compliance_rationale) for a context."""
    bay = f"Bay {ctx.bay_id}"
    stall = (
        f"x_bolt={_cm(ctx.x_bolt_m)}, |tau_ext|={ctx.tau_ext_inf_nm:.1f} Nm, C={ctx.confidence:.2f}"
    )
    match ctx.event_kind:
        case "STALL_ALERT":
            if ctx.is_ood or ctx.matched_signature_id is None:
                return (
                    f"{bay}: unseen contact stall at {_cm(ctx.x_bolt_m)}",
                    "Select the bay, hold the clutch, depress the stopper flap with the thumb "
                    "and slide the bolt past 5.2 cm with the index finger.",
                    f"No validated skill for {ctx.part_sku_id} matches this stall ({stall}); "
                    "the bay retracted to a safe hover and was queued for a human rescue.",
                )
            return (
                f"{bay}: known stall {ctx.matched_signature_id} at {_cm(ctx.x_bolt_m)}",
                "No action needed: the fleet skill is replaying inside the torque tube.",
                f"Stall matched {ctx.matched_signature_id} (d_M={_distance(ctx)} <= "
                f"{OOD_THRESHOLD:.1f}); autonomous replay authorised ({stall}).",
            )
        case "OOD_REFUSAL":
            return (
                f"{bay}: out-of-distribution stall at {_cm(ctx.x_bolt_m)} refused",
                "Inspect the latch for a pinned or damaged bolt before any rescue attempt.",
                f"Mahalanobis distance {_distance(ctx)} exceeds gamma_ood={OOD_THRESHOLD:.1f}; "
                f"refusing skill transfer and escalating to a human ({stall}).",
            )
        case "VALIDATION_GATE":
            passed = ctx.rollout_passed or 0
            total = ctx.rollout_total or 0
            ok = total > 0 and passed == total
            return (
                f"{bay}: skill validation {passed}/{total} {'PASS' if ok else 'FAIL'}",
                "Skill promoted to the fleet library."
                if ok
                else "Skill rejected; repeat the rescue with a cleaner motion.",
                f"Headless sim gate cleared the bolt in {passed} of {total} perturbed "
                f"rollouts for {ctx.part_sku_id}; "
                + ("deployment approved." if ok else "deployment withheld."),
            )
        case "RESCUE_LOGGED":
            return (
                f"{bay}: operator rescue logged at {_cm(ctx.x_bolt_m)}",
                "Release the clutch; the demonstration is being fitted and validated.",
                f"Human rescue trajectory recorded for {ctx.part_sku_id} ({stall}).",
            )
        case "SKILL_DEPLOYED":
            return (
                f"{bay}: skill {ctx.matched_signature_id or 'new'} deployed to fleet",
                "No action needed.",
                f"Validated skill for {ctx.part_sku_id} broadcast to all bays.",
            )


class VerdaNemotronPlaceholder:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model = f"placeholder:{model or 'offline'}"
        self._lock = threading.Lock()
        self._verdicts: dict[int, ReasoningVerdict] = {}

    @classmethod
    def from_env(cls) -> VerdaNemotronPlaceholder:
        return cls(
            base_url=os.environ.get("VERDA_BASE_URL") or None,
            api_key=os.environ.get("VERDA_API_KEY") or None,
            model=os.environ.get("NEMOTRON_MODEL") or None,
        )

    def submit(self, ctx: ReasoningContext) -> None:
        t0 = time.perf_counter()
        headline, guidance, rationale = triage(ctx)
        verdict = ReasoningVerdict(
            headline=headline,
            operator_guidance=guidance,
            compliance_rationale=rationale,
            model=self.model,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
        )
        with self._lock:
            self._verdicts[ctx.bay_id] = verdict

    def latest_verdict(self, bay_id: int) -> ReasoningVerdict | None:
        with self._lock:
            return self._verdicts.get(bay_id)

    def close(self) -> None:
        pass
