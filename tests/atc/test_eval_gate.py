"""T-196: offline ACT evaluation gate (``teleop eval``) and ``validate_controller``."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from teleop.atc.coordinator import DEFAULT_REALISM, ROUTINE, AtcCoordinator, OperatorInput
from teleop.atc.recovery import BaySignals, RecoveryCommand
from teleop.atc.stage import RESCUE_PRESS, RESCUE_WAYPOINTS
from teleop.atc.types import BOLT_CLEARED_M, BayState, ConsoleInputState
from teleop.atc.validator import CONTROL_DT, BayRig, SkillValidator
from teleop.compliance.protocol import ComplianceEvent
from teleop.policy.act_runtime import ReplayPolicy
from teleop.policy.manifest import GateManifest
from teleop.policy.schema import (
    CAMERAS,
    FPS,
    EpisodeMeta,
    action_from_command,
    lerobot_features,
    observe,
)
from teleop.reasoning.protocol import ReasoningContext, ReasoningVerdict
from teleop.sim.bays import PANDA_XML
from teleop.sim.hands import HandSpec, load_hand
from teleop.types import Pose
from teleop.unoq.client import UnoQClient
from teleop.unoq.dmp import quat_to_mat
from teleop.unoq.server import UnoQServer

HW = (32, 32)

# Captured on main (2136247) before the validate_controller refactor.
EXPECTED_DMP_PASSED = 19
EXPECTED_DMP_MAX_TAU: tuple[float, ...] = (
    1.2347783465622229,
    1.601130144260064,
    1.2976389381018814,
    1.2663415431811123,
    1.2345507669357823,
    1.2288316538660504,
    1.7265918713327866,
    2.3345881516796063,
    1.2681589725923104,
    1.210370884724036,
    1.261216001212686,
    1.4101188520068444,
    1.2248335273916635,
    1.1946272048761908,
    1.2315917737229516,
    1.220959590604384,
    1.2214651312238856,
    1.2321467472680754,
    1.2265235567524704,
    1.2007754171473142,
)


def _models_present() -> bool:
    try:
        right = load_hand("inspire").mjcf["right"]
    except FileNotFoundError:
        return False
    return PANDA_XML.is_file() and right.is_file()


pytestmark = pytest.mark.skipif(
    not _models_present(), reason="Panda/Inspire MJCFs missing; run scripts/fetch_models.py"
)


class _QuietSink:
    def emit(self, event: ComplianceEvent) -> None:
        pass

    def close(self) -> None:
        pass


class _QuietReasoner:
    def submit(self, ctx: ReasoningContext) -> None:
        pass

    def latest_verdict(self, bay_id: int) -> ReasoningVerdict | None:
        return None

    def close(self) -> None:
        pass


class _InProcessUnoQ(UnoQClient):
    """Transport-free UnoQClient backed directly by ``UnoQServer.handle``."""

    def __init__(self) -> None:
        server = UnoQServer(host="127.0.0.1", udp_port=0, tcp_port=0, name="test-node")
        super().__init__("127.0.0.1", local_server=server)
        self._node = server

    def request(self, message: Mapping[str, Any], timeout_s: float | None = None) -> dict[str, Any]:
        reply = self._node.handle(message)
        if not reply.get("ok"):
            raise RuntimeError(str(reply.get("error", "request failed")))
        return reply

    def close(self) -> None:
        pass


@dataclass(frozen=True)
class FlapDemo:
    states: NDArray[np.float32]  # (T, 29) 30 fps
    actions: NDArray[np.float32]  # (T, 12) 30 fps
    signature: tuple[float, float, float, float]
    skill: Any


class _DemoObserver:
    def __init__(self, bay_id: int) -> None:
        self.bay_id = bay_id
        self.states: list[NDArray[np.float64]] = []
        self.actions: list[NDArray[np.float64]] = []

    def on_episode_start(self, bay_id: int, source: str, signature_id: Any, signature: Any) -> None:
        pass

    def on_tick(
        self, bay_id: int, rig: BayRig, wrist_world: Any, fingers: Mapping[str, float]
    ) -> None:
        if bay_id == self.bay_id:
            self.states.append(observe(rig))
            self.actions.append(action_from_command(rig.world_from_latch(), wrist_world, fingers))

    def on_episode_end(self, bay_id: int, success: bool, aborted_reason: str | None) -> None:
        pass


@pytest.fixture(scope="module")
def hand_spec() -> HandSpec:
    return load_hand("inspire")


@pytest.fixture(scope="module")
def validator(hand_spec: HandSpec) -> SkillValidator:
    return SkillValidator(hand_spec, realism=DEFAULT_REALISM)


@pytest.fixture(scope="module")
def flap_demo(hand_spec: HandSpec, validator: SkillValidator) -> FlapDemo:
    """Run the canonical Bay 1 flap rescue; capture the fitted DMP skill and 30 fps episode rows."""
    unoq = _InProcessUnoQ()
    obs = _DemoObserver(bay_id=1)
    co = AtcCoordinator.build(
        unoq,
        _QuietSink(),
        _QuietReasoner(),
        hand_spec=hand_spec,
        validator=validator,
        observer=obs,
    )
    try:
        co.start_cycle(1, flap_jammed=True)
        for _ in range(1500):
            if co.tick()[0].state in {BayState.RESCUE, BayState.OOD_ESCALATED, BayState.QUEUED}:
                break
        for _ in range(40):
            co.tick()
        bay_rig = co.rig(1)
        hover = bay_rig.wrist_pose()
        assert hover.position is not None
        r_world = quat_to_mat(bay_rig.world_from_latch().orientation)
        origin = np.asarray(hover.position)
        base = dict(ROUTINE.fingers)
        console = ConsoleInputState(selected_bay=1, clutch=True)
        done = False
        for i in range(1, len(RESCUE_WAYPOINTS)):
            if done:
                break
            dur, p1, f1 = RESCUE_WAYPOINTS[i]
            _, p0, f0 = RESCUE_WAYPOINTS[i - 1]
            n = max(int(round(dur / CONTROL_DT)), 1)
            for k in range(1, n + 1):
                a = k / n
                delta = np.asarray(p0) + a * (np.asarray(p1) - np.asarray(p0))
                f = f0 + a * (f1 - f0)
                q = {name: base[name] + f * v for name, v in RESCUE_PRESS.items()}
                q_full = {**base, **q}
                pos = origin + r_world @ delta
                sample = OperatorInput(
                    Pose(orientation=hover.orientation, position=(pos[0], pos[1], pos[2])),
                    q_full,
                )
                snaps = co.tick(console, sample)
                if snaps[0].x_bolt_m >= BOLT_CLEARED_M + 0.0015:
                    done = True
                    break
        released = ConsoleInputState(selected_bay=1, clutch=False)
        max_extra = round(co.config.gate_deadline_s / CONTROL_DT) + 60
        for _ in range(max_extra):
            co.tick(released, None)
            if co.state(1) == BayState.NOMINAL:
                break
        skill = next(iter(co.library.values()))
        sig = co.last_signature(1)
        assert sig is not None
        vec = tuple(float(v) for v in sig.as_vector())
        assert len(vec) == 4
        signature: tuple[float, float, float, float] = (vec[0], vec[1], vec[2], vec[3])
    finally:
        co.close()
        unoq.close()
    states = np.asarray(obs.states[::2], dtype=np.float32)
    actions = np.asarray(obs.actions[::2], dtype=np.float32)
    return FlapDemo(states=states, actions=actions, signature=signature, skill=skill)


def _write_flap_dataset(root: Path, demo: FlapDemo) -> Path:
    pytest.importorskip("lerobot")
    from teleop.record.recorder import EpisodeRecorder

    rec = EpisodeRecorder(f"local/{root.name}", root, FPS, lerobot_features(HW), "panda_inspire")
    imgs = {cam: np.zeros((*HW, 3), dtype=np.uint8) for cam in CAMERAS}
    rec.start_episode("recover latch: flap")
    for st, act in zip(demo.states, demo.actions, strict=True):
        rec.add_frame(st, act, imgs)
    meta = EpisodeMeta(
        source="scripted",
        fault="flap",
        bay=1,
        signature_id="Jam_Sig_1",
        signature=demo.signature,
        latch_pose={"x": 0.5, "y": 0.0, "z": 0.0, "yaw_rad": 0.0},
        success=True,
        bolt_x_final=0.054,
        max_tau_hand_inf=1.6,
        max_arm_ratio=0.35,
        aborted_reason=None,
    )
    rec.end_episode(meta.to_json())
    rec.close()
    return root


@pytest.fixture(scope="module")
def flap_dataset(tmp_path_factory: pytest.TempPathFactory, flap_demo: FlapDemo) -> Path:
    return _write_flap_dataset(tmp_path_factory.mktemp("ds") / "flap", flap_demo)


def test_validate_unchanged_for_dmp(validator: SkillValidator, flap_demo: FlapDemo) -> None:
    report = validator.validate(flap_demo.skill)
    assert report.n_passed == EXPECTED_DMP_PASSED
    taus = [r.max_tau_ext_inf_nm for r in report.results]
    assert taus == pytest.approx(EXPECTED_DMP_MAX_TAU, abs=1e-9)


def test_pin_fault_uses_bolt_pinned(
    validator: SkillValidator, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[bool, bool]] = []
    orig_reset = validator.rig.reset_latch

    def spy_reset(flap_jammed: bool = False, bolt_pinned: bool = False) -> None:
        calls.append((flap_jammed, bolt_pinned))
        orig_reset(flap_jammed=flap_jammed, bolt_pinned=bolt_pinned)

    monkeypatch.setattr(validator.rig, "reset_latch", spy_reset)

    class _ImmediateAbort:
        name = "test"

        def command(self, rig: BayRig, s: BaySignals) -> RecoveryCommand | None:
            return None

        def check(self, rig: BayRig, s: BaySignals) -> str | None:
            return "stop"

        def close(self) -> None:
            pass

    conds = validator.draw_conditions("pin")[:1]
    report = validator.validate_controller(
        lambda _rig: _ImmediateAbort(), fault="pin", conditions=conds
    )
    assert report.total == 1
    assert calls == [(False, True)]


def test_eval_writes_manifest_with_passed_fault(
    tmp_path: Path,
    flap_demo: FlapDemo,
    flap_dataset: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("lerobot")
    from teleop.commands import eval as eval_cmd

    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    monkeypatch.setattr(eval_cmd, "_load_policy", lambda _p, _d: ReplayPolicy(flap_demo.actions))

    rc = eval_cmd.main(
        [
            "--checkpoint",
            str(ckpt),
            "--dataset",
            str(flap_dataset),
            "--faults",
            "flap",
            "--rollouts",
            "4",
            "--bar",
            "3",
        ]
    )
    assert rc == 0
    manifest = GateManifest.load(ckpt / "gate.json")
    assert "flap" in manifest.passed_faults()
    assert manifest.bar == (3, 4)
    res = manifest.results["flap"]
    assert res.rollouts == 4
    assert res.passed_rollouts >= 3
    assert res.mean_time_to_clear_s is not None and res.mean_time_to_clear_s > 0.0
    assert res.dmp_passed_rollouts is not None and 0 <= res.dmp_passed_rollouts <= 4
    assert manifest.prototypes.get("flap") == [flap_demo.signature]


def test_eval_fault_below_bar_not_passed(
    tmp_path: Path,
    flap_demo: FlapDemo,
    flap_dataset: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("lerobot")
    from teleop.commands import eval as eval_cmd

    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    idle = np.repeat(flap_demo.actions[:1], 40, axis=0)
    monkeypatch.setattr(eval_cmd, "_load_policy", lambda _p, _d: ReplayPolicy(idle))

    rc = eval_cmd.main(
        [
            "--checkpoint",
            str(ckpt),
            "--dataset",
            str(flap_dataset),
            "--faults",
            "flap",
            "--rollouts",
            "4",
            "--bar",
            "3",
        ]
    )
    assert rc == 1
    manifest = GateManifest.load(ckpt / "gate.json")
    assert "flap" not in manifest.passed_faults()
    assert manifest.results["flap"].passed_rollouts == 0
    assert not manifest.prototypes.get("flap")


def test_eval_no_data_fault(
    tmp_path: Path,
    flap_demo: FlapDemo,
    flap_dataset: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("lerobot")
    from teleop.commands import eval as eval_cmd

    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    monkeypatch.setattr(eval_cmd, "_load_policy", lambda _p, _d: ReplayPolicy(flap_demo.actions))

    rc = eval_cmd.main(
        [
            "--checkpoint",
            str(ckpt),
            "--dataset",
            str(flap_dataset),
            "--faults",
            "flap,pin",
            "--rollouts",
            "4",
            "--bar",
            "3",
        ]
    )
    assert rc == 0
    manifest = GateManifest.load(ckpt / "gate.json")
    assert "flap" in manifest.passed_faults()
    assert "pin" not in manifest.passed_faults()
    assert manifest.results["pin"].note == "no data"
    assert manifest.results["pin"].rollouts == 0
    assert manifest.results["pin"].passed_rollouts == 0
    assert "pin" not in manifest.prototypes
