"""Deterministic offline fleet reasoner: rule-based triage text for the HUD and console."""

from __future__ import annotations

import math
import threading
import time

from teleop.reasoning.protocol import ReasoningContext, ReasoningVerdict

OOD_THRESHOLD = 3.0  # gamma_ood, spec 4.2
BOLT_CLEARED_CM = 5.2  # x_bolt threshold in cm (spec 2.1)


def _cm(x_m: float) -> str:
    if not math.isfinite(x_m):
        return "n/a"
    return f"{x_m * 100.0:.1f} cm"


def _fmt_float(val: float, fmt: str) -> str:
    if not math.isfinite(val):
        return "inf" if val > 0 else ("-inf" if val < 0 else "n/a")
    return format(val, fmt)


def _distance(ctx: ReasoningContext) -> str:
    d = ctx.mahalanobis_distance
    if d is None:
        return "n/a"
    return _fmt_float(d, ".2f")


def triage(ctx: ReasoningContext) -> tuple[str, str, str]:
    """Deterministic (headline, operator_guidance, compliance_rationale) for a context."""
    bay = f"Bay {ctx.bay_id}"
    tau_s = _fmt_float(ctx.tau_ext_inf_nm, ".1f")
    conf_s = _fmt_float(ctx.confidence, ".2f")
    stall = f"x_bolt={_cm(ctx.x_bolt_m)}, |tau_ext|={tau_s} Nm, C={conf_s}"
    match ctx.event_kind:
        case "STALL_ALERT":
            if ctx.is_ood or ctx.matched_signature_id is None:
                return (
                    f"{bay}: unseen contact stall at {_cm(ctx.x_bolt_m)}",
                    "Select the bay, hold the clutch, depress the stopper flap with the thumb "
                    f"and slide the bolt past {BOLT_CLEARED_CM:.1f} cm with the index finger.",
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
            need = total if ctx.rollout_threshold is None else ctx.rollout_threshold
            ok = total > 0 and passed >= need
            return (
                f"{bay}: skill validation {passed}/{total} {'PASS' if ok else 'FAIL'}",
                "Skill promoted to the fleet library."
                if ok
                else "Skill rejected; repeat the rescue with a cleaner motion.",
                f"Headless sim gate cleared the bolt in {passed} of {total} perturbed "
                f"randomized rollouts (pass at {need}) for {ctx.part_sku_id}; "
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
        case _:
            return (
                f"{bay}: event {ctx.event_kind} at {_cm(ctx.x_bolt_m)}",
                "Monitor bay telemetry.",
                f"Recorded {ctx.event_kind} for {ctx.part_sku_id} ({stall}).",
            )


class TriageReasoner:
    """Zero-network deterministic ``FleetReasoner``: one ``triage(ctx)`` verdict per bay."""

    mode = "offline"

    def __init__(self) -> None:
        self.model = "triage"
        self._lock = threading.Lock()
        self._verdicts: dict[int, ReasoningVerdict] = {}

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
