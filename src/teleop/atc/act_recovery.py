"""Live ACT recovery: the gated release policy drives a stalled bay (ACT recovery spec 1, 5).

:class:`ActRecoveryFactory` is the coordinator's ``RecoveryFactory`` for ``name = "act"``. It
covers a stall only if its fingerprint is near a prototype of a fault that passed the offline
gate (``gate.json``), and is healthy while its :class:`InferenceWorker` is.

:class:`ActRecovery` runs one bay's recovery in three phases:

1. retract 3 cm along latch -X (the hover a human rescue starts from, so the live start state
   matches the training episodes);
2. hold for ``settle_s``;
3. policy: on its first tick, then every ``query_every`` sim ticks (aligned across bays so they
   share one batch), post ``observe(rig)`` and the bay's camera images to the
   worker; each tick play row ``(tick - obs_tick) // 2`` of the newest chunk (30 fps rows at
   60 Hz), clamped to its last row. Until the first chunk arrives the bay holds its pose.

Safety is checked after physics, in this order: worker unhealthy, no chunk within the worker's
timeout of a submit, hand torque at the hard limit, no progress (confidence below the stall
threshold for the stall detector's 300 ms), and the recovery timeout. The policy never sees
``bolt_x``, ``flap_hinge`` or fault flags; bolt clearance stays in the coordinator.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from teleop.atc.recovery import BaySignals, RecoveryCommand, SmoothMove
from teleop.atc.types import StallSignature
from teleop.atc.validator import (
    CONTROL_DT,
    TAU_HARD_MAX_NM,
    BayRig,
    local_axis_world,
    translate_pose,
)
from teleop.policy.act_runtime import InferenceWorker, load_act_policy, read_policy_config
from teleop.policy.manifest import GateManifest, covers
from teleop.policy.schema import command_from_action, observe

logger = logging.getLogger(__name__)

GATE_FILE = "gate.json"
ROWS_PER_TICK = 2  # dataset rows are 30 fps, the control loop is 60 Hz


class CameraRenderer(Protocol):
    def render(self, data: Any, bay_id: int) -> dict[str, NDArray[np.uint8]]: ...
    def close(self) -> None: ...


class ActRecovery:
    """One bay's ACT recovery: retract, settle, then play the policy's action chunks."""

    name = "act"

    def __init__(
        self,
        rig: BayRig,
        worker: InferenceWorker,
        renderer: CameraRenderer,
        data: Any,
        *,
        query_every: int,
        retract_m: float,
        retract_s: float,
        settle_s: float,
        timeout_s: float,
        tau_hard_max: float,
        stall_confidence: float,
        stall_ticks: int,
    ) -> None:
        self.bay_id = rig.bay_id
        self.worker, self.renderer, self.data = worker, renderer, data
        self.query_every = max(int(query_every), 1)
        self.settle_s, self.timeout_s = settle_s, timeout_s
        self.tau_hard_max = tau_hard_max
        self.stall_confidence, self.stall_ticks = stall_confidence, stall_ticks
        wrist = rig.wrist_pose()
        ax = local_axis_world(rig.world_from_latch(), 0)
        self.hover = translate_pose(wrist, -retract_m * ax)
        self.retract: SmoothMove | None = SmoothMove(wrist, self.hover, retract_s)
        self.settle_t = 0.0
        self.tick = 0  # policy-phase ticks
        self.low_ticks = 0
        self.t = 0.0
        self._waiting: tuple[int, int] | None = None  # (tick, monotonic ns) of an unserved submit
        worker.forget(self.bay_id)

    @property
    def phase(self) -> str:
        if self.retract is not None:
            return "retract"
        return "settle" if self.settle_t < self.settle_s else "policy"

    def command(self, rig: BayRig, s: BaySignals) -> RecoveryCommand | None:
        self.t += CONTROL_DT
        if self.retract is not None:
            pose = self.retract.step()
            if self.retract.done:
                self.retract = None
            return RecoveryCommand(pose)
        if self.settle_t < self.settle_s:
            self.settle_t += CONTROL_DT
            return RecoveryCommand(self.hover, v_cmd=0.0)
        if self.tick == 0 or self._fleet_tick(rig) % self.query_every == 0:
            self._submit(rig)
        self.tick += 1
        got = self.worker.latest(self.bay_id)
        if got is None:
            return None
        obs_tick, chunk = got
        row = min(max(self.tick - 1 - obs_tick, 0) // ROWS_PER_TICK, len(chunk) - 1)
        wrist, fingers = command_from_action(rig.world_from_latch(), chunk[row])
        return RecoveryCommand(wrist, fingers)

    def _fleet_tick(self, rig: BayRig) -> int:
        """The shared sim tick, so every bay in ACT posts on the same ticks (one batch)."""
        data = getattr(rig, "data", None)
        if data is None:
            return self.tick
        return round(float(data.time) / CONTROL_DT)

    def _submit(self, rig: BayRig) -> None:
        images = self.renderer.render(self.data, self.bay_id)
        self.worker.submit(self.bay_id, observe(rig), images, self.tick)
        if self._waiting is None:
            self._waiting = (self.tick, time.monotonic_ns())

    def check(self, rig: BayRig, s: BaySignals) -> str | None:
        if not self.worker.healthy():
            return "inference unavailable"
        if self._waiting is not None:
            got = self.worker.latest(self.bay_id)
            if got is not None and got[0] >= self._waiting[0]:
                self._waiting = None
            elif (time.monotonic_ns() - self._waiting[1]) * 1e-6 > self.worker.timeout_ms:
                return "inference timeout"
        if s.tau_hand_inf >= self.tau_hard_max:
            return "hand torque limit"
        if self.phase == "policy":
            self.low_ticks = self.low_ticks + 1 if s.confidence < self.stall_confidence else 0
            if self.low_ticks >= self.stall_ticks:
                return "no progress"
        if self.t > self.timeout_s:
            return "act timeout"
        return None

    def close(self) -> None:
        self.worker.forget(self.bay_id)


class ActRecoveryFactory:
    """``RecoveryFactory`` for the gated ACT release (``name = "act"``)."""

    name = "act"

    def __init__(
        self,
        manifest: GateManifest,
        worker: InferenceWorker,
        renderer: CameraRenderer,
        data: Any,
        *,
        query_every: int = 5,
        retract_m: float = 0.03,
        retract_s: float = 0.5,
        settle_s: float = 0.7,
        timeout_s: float = 8.0,
        tau_hard_max: float = TAU_HARD_MAX_NM,
        stall_confidence: float = 0.3,
        stall_ticks: int = 18,
    ) -> None:
        self.manifest = manifest
        self.worker = worker
        self.renderer = renderer
        self.data = data
        self._kw: dict[str, Any] = {
            "query_every": query_every,
            "retract_m": retract_m,
            "retract_s": retract_s,
            "settle_s": settle_s,
            "timeout_s": timeout_s,
            "tau_hard_max": tau_hard_max,
            "stall_confidence": stall_confidence,
            "stall_ticks": stall_ticks,
        }
        self.created: list[ActRecovery] = []

    @property
    def unhealthy_reason(self) -> str:
        return self.worker.unhealthy_reason

    def covers(self, z: StallSignature) -> bool:
        return covers(self.manifest, z) is not None

    def healthy(self) -> bool:
        return self.worker.healthy()

    def create(self, rig: BayRig, z: StallSignature) -> ActRecovery:
        rec = ActRecovery(rig, self.worker, self.renderer, self.data, **self._kw)
        self.created.append(rec)
        return rec

    def close(self) -> None:
        self.worker.close()
        self.renderer.close()


def load_act_factory(checkpoint: Path, device: str, model: Any, data: Any) -> ActRecoveryFactory:
    """The ``--act-checkpoint`` hook: gate manifest, ACT policy, worker and camera renderer.

    Raises ``ManifestError`` if ``gate.json`` is missing or for another schema, or if the
    checkpoint's features don't match the policy schema; ``ImportError``/``RuntimeError`` if
    torch, LeRobot or the device is unavailable. ``teleop atc`` then falls back to ``dmp``.
    """
    from teleop.policy.cameras import BayCameraRenderer

    checkpoint = Path(checkpoint)
    manifest = GateManifest.load(checkpoint / GATE_FILE)
    if not manifest.passed_faults():
        logger.warning("act: %s passes no fault type; ACT will cover no stall", checkpoint)
    _, image_hw = read_policy_config(checkpoint)
    worker = InferenceWorker(load_act_policy(checkpoint, device))
    return ActRecoveryFactory(manifest, worker, BayCameraRenderer(model, image_hw), data)
