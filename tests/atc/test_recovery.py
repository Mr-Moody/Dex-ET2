"""Recovery backend seam, DMP byte-for-byte golden guard, and live ACT recovery execution."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from conftest import NullReasoner, RecordingComplianceSink, models_present, needs_models
from teleop.atc.act_recovery import ActRecovery, ActRecoveryFactory, load_act_factory
from teleop.atc.coordinator import AtcCoordinator
from teleop.atc.recovery import RecoveryBackend, choose_recovery, next_backend
from teleop.atc.stage import (
    KEY_SPACE,
    EventTap,
    ReasonerTap,
    ScriptedGlove,
    StageController,
    StageDirector,
)
from teleop.atc.types import BOLT_CLEARED_M, BayState
from teleop.commands.atc import DEFAULT_CONFIG, load_config, resolve_seed
from teleop.compliance.protocol import ComplianceEvent
from teleop.policy.act_runtime import InferenceWorker, ReplayPolicy
from teleop.policy.manifest import FaultResult, GateManifest, ManifestError
from teleop.policy.schema import (
    ACTION_NAMES,
    CAMERAS,
    IMAGE_HW,
    SCHEMA_VERSION,
    action_from_command,
)
from teleop.unoq.client import UnoQClient
from teleop.unoq.server import UnoQServer

# --------------------------------------------------------------------------- DMP golden guard

GOLDEN_EVENTS: list[tuple[str, int]] = [
    ("STALL_ALERT", 1),
    ("RESCUE_LOGGED", 1),
    ("VALIDATION_GATE", 1),
    ("SKILL_DEPLOYED", 1),
    ("RESCUE_LOGGED", 2),
    ("OOD_REFUSAL", 2),
]
GOLDEN_FILE = Path(__file__).parent / "golden" / "dmp_stage.json"
UPDATE_HINT = (
    "if this show change is intentional, recapture with "
    "`UPDATE_GOLDEN=1 uv run pytest tests/atc/test_recovery.py -k dmp_mode_script_unchanged`"
)


def _load_golden() -> dict[str, Any]:
    if not GOLDEN_FILE.is_file():
        return {}
    return json.loads(GOLDEN_FILE.read_text(encoding="utf-8"))


def _save_golden(platform: str, ticks: int, wrist: list[list[float]]) -> None:
    golden = _load_golden()
    golden[platform] = {"ticks": ticks, "wrist": wrist}
    GOLDEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_FILE.write_text(
        json.dumps(dict(sorted(golden.items())), indent=2) + "\n", encoding="utf-8"
    )


@needs_models
def test_dmp_mode_script_unchanged(unoq_client: UnoQClient) -> None:
    sink = RecordingComplianceSink()
    cfg = load_config(DEFAULT_CONFIG, {"script": True})
    controller = StageController.build(
        unoq_client,
        sink,
        NullReasoner(),
        cycle_dwell_s=cfg.cycle_dwell_s,
        include_background_bays=cfg.background_bays,
        seed=resolve_seed(cfg),
        realism=cfg.realism,
        gate_pass_min=cfg.gate_pass_min,
    )
    try:
        director = StageDirector(controller)
        ticks = 0
        while not director.done and ticks < 12_000:
            controller.tick(director.step())
            ticks += 1
        assert director.ok, director.results
        assert [(e.kind, e.bay_id) for e in sink.events] == GOLDEN_EVENTS
        wrist = []
        for b in (1, 2):
            pos = director.co.rig(b).wrist_pose().position
            assert pos is not None
            wrist.append([round(float(v), 9) for v in pos])
    finally:
        controller.close()

    if os.environ.get("UPDATE_GOLDEN"):
        _save_golden(sys.platform, ticks, wrist)
        return
    golden = _load_golden().get(sys.platform)
    if golden is None:
        return
    assert ticks == golden["ticks"], f"{sys.platform} tick count drifted; {UPDATE_HINT}"
    assert wrist == golden["wrist"], f"{sys.platform} wrist positions drifted; {UPDATE_HINT}"


# --------------------------------------------------------------------------- backend routing


@pytest.mark.parametrize(
    ("backend", "act_ok", "dmp_available", "expected"),
    [
        (RecoveryBackend.DMP, True, True, "dmp"),
        (RecoveryBackend.DMP, True, False, None),
        (RecoveryBackend.HYBRID, True, True, "act"),
        (RecoveryBackend.HYBRID, False, True, "dmp"),
        (RecoveryBackend.ACT, True, False, "act"),
        (RecoveryBackend.ACT, False, True, None),
    ],
)
def test_choose_recovery_and_backend_cycle(
    backend: RecoveryBackend, act_ok: bool, dmp_available: bool, expected: str | None
) -> None:
    assert (
        choose_recovery(backend, act_covers=act_ok, act_healthy=True, dmp_available=dmp_available)
        == expected
    )
    assert [
        next_backend(b) for b in (RecoveryBackend.DMP, RecoveryBackend.HYBRID, RecoveryBackend.ACT)
    ] == [
        RecoveryBackend.HYBRID,
        RecoveryBackend.ACT,
        RecoveryBackend.DMP,
    ]


class _HoldController:
    name = "act"

    def __init__(self, abort: str | None = None, raise_on_cmd: bool = False) -> None:
        self.abort = abort
        self.raise_on_cmd = raise_on_cmd
        self.closed = False

    def command(self, rig: Any, s: Any) -> Any:
        if self.raise_on_cmd:
            raise RuntimeError("policy bug")
        return None

    def check(self, rig: Any, s: Any) -> str | None:
        return self.abort

    def close(self) -> None:
        self.closed = True


class _StubActFactory:
    name = "act"

    def __init__(
        self, healthy: bool = True, abort: str | None = None, raise_on_cmd: bool = False
    ) -> None:
        self._healthy = healthy
        self.abort = abort
        self.raise_on_cmd = raise_on_cmd
        self.unhealthy_reason = "" if healthy else "worker crashed"
        self.created: list[_HoldController] = []

    def covers(self, z: Any) -> bool:
        return True

    def healthy(self) -> bool:
        return self._healthy

    def create(self, rig: Any, z: Any) -> _HoldController:
        c = _HoldController(self.abort, self.raise_on_cmd)
        self.created.append(c)
        return c

    def close(self) -> None:
        return None


def _build_controller(
    unoq: UnoQClient, **kw: Any
) -> tuple[StageController, RecordingComplianceSink]:
    sink = RecordingComplianceSink()
    events = EventTap(sink)
    co = AtcCoordinator.build(unoq, events, NullReasoner(), **kw)
    return StageController(co, events, ReasonerTap(NullReasoner()), cycle_dwell_s=0.5), sink


def _tick_until(
    controller: StageController, pred: Any, op: Any = None, max_ticks: int = 3000
) -> None:
    for _ in range(max_ticks):
        controller.tick(op.latest() if op is not None else None)
        if pred():
            return
    raise AssertionError("condition never reached")


@needs_models
def test_mid_recovery_switch_downgrade_and_exception_isolation(unoq_client: UnoQClient) -> None:
    # 1. Mid-recovery switch finishes in-flight bay on old backend.
    ctl, _ = _build_controller(
        unoq_client, recovery_backend=RecoveryBackend.HYBRID, act=_StubActFactory()
    )
    try:
        co = ctl.coordinator
        co.start_cycle(1, flap_jammed=True)
        _tick_until(ctl, lambda: co.recovery_name(1) == "act")
        co.set_recovery_backend(RecoveryBackend.DMP)
        for _ in range(30):
            ctl.tick(None)
        assert co.recovery_name(1) == "act" and co.recovery_backend == RecoveryBackend.DMP
    finally:
        ctl.close()

    # 2. Unhealthy ACT downgrades coordinator back to DMP.
    ctl2, _ = _build_controller(
        unoq_client, recovery_backend=RecoveryBackend.HYBRID, act=_StubActFactory(healthy=False)
    )
    try:
        co2 = ctl2.coordinator
        co2.start_cycle(1, flap_jammed=True)
        _tick_until(ctl2, lambda: co2.state(1) != BayState.NOMINAL)
        assert co2.recovery_backend == RecoveryBackend.DMP and "worker crashed" in co2.backend_note
    finally:
        ctl2.close()

    # 3. Exception inside ACT controller aborts cleanly to operator queue instead of crashing.
    fac3 = _StubActFactory(raise_on_cmd=True)
    ctl3, _ = _build_controller(unoq_client, recovery_backend=RecoveryBackend.HYBRID, act=fac3)
    try:
        co3 = ctl3.coordinator
        co3.start_cycle(1, flap_jammed=True)
        _tick_until(ctl3, lambda: co3.state(1) in (BayState.QUEUED, BayState.RESCUE))
        assert "policy bug" in co3.outcome(1) and fac3.created[0].closed
    finally:
        ctl3.close()


# --------------------------------------------------------------------------- live ACT replay


class _ZeroRenderer:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def render(self, data: Any, bay_id: int) -> dict[str, NDArray[np.uint8]]:
        self.calls.append(bay_id)
        return {c: np.zeros((*IMAGE_HW, 3), dtype=np.uint8) for c in CAMERAS}

    def close(self) -> None:
        return None


class _ActionRecorder:
    def __init__(self) -> None:
        self.bay: int | None = None
        self.rows: list[NDArray[np.float64]] = []

    def on_episode_start(self, bay_id: int, source: str, signature_id: Any, signature: Any) -> None:
        return None

    def on_tick(
        self, bay_id: int, rig: Any, wrist_world: Any, fingers: Mapping[str, float]
    ) -> None:
        if bay_id == self.bay:
            self.rows.append(action_from_command(rig.world_from_latch(), wrist_world, fingers))

    def on_episode_end(self, bay_id: int, success: bool, aborted_reason: str | None) -> None:
        return None


@dataclass(frozen=True)
class _Demo:
    actions: NDArray[np.float32]
    signatures: dict[int, tuple[float, float, float, float]]


@pytest.fixture(scope="module")
def act_demo() -> _Demo:
    if not models_present():
        pytest.skip("Panda/Inspire MJCFs missing; run `uv run python scripts/fetch_models.py`")
    server = UnoQServer(host="127.0.0.1", udp_port=0, tcp_port=0, name="demo-node").start()
    client = UnoQClient(
        "127.0.0.1", server.udp_port, server.tcp_port, local_server=server
    ).connect()
    rec = _ActionRecorder()
    ctl, _ = _build_controller(client, observer=rec)
    co = ctl.coordinator
    try:
        co.start_cycle(1, flap_jammed=True)
        co.start_cycle(2, flap_jammed=True)
        _tick_until(
            ctl,
            lambda: (
                ctl.rescue_bay() is not None
                and all(co.last_signature(b) is not None for b in (1, 2))
            ),
        )
        bay = ctl.rescue_bay()
        assert bay is not None
        rec.bay = bay
        glove = ScriptedGlove(ctl)
        ctl.on_key(ord(str(bay)))
        ctl.on_key(KEY_SPACE)
        rig = co.rig(bay)
        _tick_until(ctl, lambda: rig.bolt_x() >= BOLT_CLEARED_M + 0.002, glove)
        for _ in range(15):
            ctl.tick(glove.latest())
        sigs: dict[int, tuple[float, float, float, float]] = {}
        for b in (1, 2):
            z = co.last_signature(b)
            assert z is not None
            a, bb, c, d = (float(v) for v in z.as_vector())
            sigs[b] = (a, bb, c, d)
    finally:
        ctl.close()
        client.close()
    rows = np.asarray(rec.rows[::2], dtype=np.float32)
    assert len(rows) > 60 and rows.shape[1] == len(ACTION_NAMES)
    return _Demo(rows, sigs)


def _make_act_factory(demo: _Demo, cls: Any = ActRecoveryFactory, **kw: Any) -> Any:
    manifest = GateManifest(
        schema_version=SCHEMA_VERSION,
        checkpoint_sha256="replay",
        bar=(9, 10),
        results={"flap": FaultResult("flap", 10, 10, 3.0, 4.0, None)},
        prototypes={"flap": list(demo.signatures.values())},
        inference_p95_ms=None,
    )
    worker = InferenceWorker(ReplayPolicy(demo.actions))
    return cls(manifest, worker, _ZeroRenderer(), None, **kw)


@needs_models
def test_live_act_recovery_clears_flap_batches_two_bays_and_isolates_abort(
    unoq_client: UnoQClient, act_demo: _Demo
) -> None:
    class Bay1Fragile(ActRecoveryFactory):
        def create(self, rig: Any, z: Any) -> ActRecovery:
            rec = super().create(rig, z)
            if rig.bay_id == 1:
                rec.tau_hard_max = 0.0
            return rec

    factory = _make_act_factory(act_demo, Bay1Fragile)
    ctl, sink = _build_controller(unoq_client, recovery_backend=RecoveryBackend.HYBRID, act=factory)
    co = ctl.coordinator
    try:
        co.start_cycle(1, flap_jammed=True)
        co.start_cycle(2, flap_jammed=True)
        _tick_until(ctl, lambda: co.recovery_name(1) == "act" and co.recovery_name(2) == "act")
        _tick_until(ctl, lambda: co.state(1) != BayState.AUTONOMOUS_SKILL)
        assert "hand torque limit" in co.outcome(1)
        assert co.state(1) in (BayState.QUEUED, BayState.RESCUE)
        assert co.recovery_name(2) == "act"

        _tick_until(ctl, lambda: co.state(2) != BayState.AUTONOMOUS_SKILL, max_ticks=900)
        assert co.outcome(2) == "cleared autonomously"
        logged: list[ComplianceEvent] = [
            e for e in sink.events if e.kind == "RESCUE_LOGGED" and e.bay_id == 2
        ]
        assert logged and logged[-1].metadata.get("backend") == "act"
        assert factory.worker.healthy()
    finally:
        ctl.close()
        factory.close()


def test_load_act_factory_validates_gate_and_policy_schema(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="gate.json"):
        load_act_factory(tmp_path, "cpu", None, None)

    gate = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_sha256": "x",
        "bar": [9, 10],
        "results": {
            "flap": {
                "fault": "flap",
                "rollouts": 10,
                "passed_rollouts": 10,
                "mean_time_to_clear_s": 3.0,
                "max_tau_hand_inf": 4.0,
                "dmp_passed_rollouts": None,
                "note": "",
            }
        },
        "prototypes": {"flap": [[0.03, 1.0, 2.0, 0.3]]},
        "inference_p95_ms": None,
    }
    (tmp_path / "gate.json").write_text(json.dumps(gate), encoding="utf-8")
    model_dir = tmp_path / "checkpoints" / "last" / "pretrained_model"
    model_dir.mkdir(parents=True)
    bad_cfg = {
        "input_features": {"observation.state": {"type": "STATE", "shape": [20]}},
        "output_features": {"action": {"type": "ACTION", "shape": [12]}},
    }
    (model_dir / "config.json").write_text(json.dumps(bad_cfg), encoding="utf-8")
    with pytest.raises(ManifestError, match="observation.state"):
        load_act_factory(tmp_path, "cpu", None, None)
