"""T-196: the offline ACT gate (``teleop eval``) and the controller-generic validator.

No torch, LeRobot or GPU: ``_load_policy`` is monkeypatched to a ``ReplayPolicy`` of a scripted
flap rescue, the camera renderer to zeros, and the dataset reader to in-memory episodes.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco as _mujoco  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
import pytest
from numpy.typing import NDArray

from teleop.compliance.protocol import ComplianceEvent
from teleop.policy.schema import CAMERAS, IMAGE_HW, SCHEMA_VERSION
from teleop.reasoning.protocol import ReasoningContext, ReasoningVerdict
from teleop.sim.bays import PANDA_XML
from teleop.sim.hands import load_hand


def _models_present() -> bool:
    try:
        right = load_hand("inspire").mjcf["right"]
    except FileNotFoundError:
        return False
    return PANDA_XML.is_file() and right.is_file()


mj: Any = _mujoco

needs_models = pytest.mark.skipif(
    not _models_present(), reason="Panda/Inspire MJCFs missing; run scripts/fetch_models.py"
)

# ``validate`` on the scripted flap rescue's DMP (SkillValidator(hand_spec,
# realism=DEFAULT_REALISM), seed 0, 20 rollouts). Recaptured after #92's clutch yield: the
# scripted rescue's arm torque ratio spikes past teleop_yield_ratio, so its demo (and the DMP
# fitted from it) changed; the validator itself did not.
GOLDEN_DMP_N_PASSED = 20
GOLDEN_DMP_MAX_TAU_NM = 1.3064834304972683


class _Sink:
    def emit(self, event: ComplianceEvent) -> None:
        pass

    def close(self) -> None:
        pass


class _Quiet:
    def submit(self, ctx: ReasoningContext) -> None:
        pass

    def latest_verdict(self, bay_id: int) -> ReasoningVerdict | None:
        return None

    def close(self) -> None:
        pass


class ZeroRenderer:
    def __init__(self) -> None:
        self.calls = 0

    def render(self, data: Any, bay_id: int) -> dict[str, NDArray[np.uint8]]:
        self.calls += 1
        return {c: np.zeros((*IMAGE_HW, 3), dtype=np.uint8) for c in CAMERAS}

    def close(self) -> None:
        pass


class _ActionRecorder:
    """Observer: the latch-frame actions of one bay's operator rescue, one row per 60 Hz tick."""

    def __init__(self) -> None:
        self.bay: int | None = None
        self.rows: list[NDArray[np.float64]] = []

    def on_episode_start(self, bay_id: int, source: str, signature_id: Any, signature: Any) -> None:
        pass

    def on_tick(
        self, bay_id: int, rig: Any, wrist_world: Any, fingers: Mapping[str, float]
    ) -> None:
        from teleop.policy.schema import action_from_command

        if bay_id == self.bay:
            self.rows.append(action_from_command(rig.world_from_latch(), wrist_world, fingers))

    def on_episode_end(self, bay_id: int, success: bool, aborted_reason: str | None) -> None:
        pass


@dataclass(frozen=True)
class Demo:
    actions: NDArray[np.float32]  # 30 fps latch-frame rows of one scripted flap rescue
    signatures: dict[int, tuple[float, float, float, float]]


def build_demo() -> Demo:
    """Jam both flaps and rescue the first queued bay with the scripted glove, recorded."""
    from teleop.atc.coordinator import AtcCoordinator
    from teleop.atc.stage import KEY_SPACE, EventTap, ReasonerTap, ScriptedGlove, StageController
    from teleop.atc.types import BOLT_CLEARED_M
    from teleop.unoq.client import UnoQClient
    from teleop.unoq.server import UnoQServer

    server = UnoQServer(host="127.0.0.1", udp_port=0, tcp_port=0, name="test-node").start()
    unoq = UnoQClient("127.0.0.1", server.udp_port, server.tcp_port, local_server=server).connect()
    recorder = _ActionRecorder()
    events = EventTap(_Sink())
    co = AtcCoordinator.build(unoq, events, _Quiet(), observer=recorder)
    controller = StageController(co, events, ReasonerTap(_Quiet()), cycle_dwell_s=0.5)

    def tick_until(pred: Any, op: Any = None, max_ticks: int = 3000) -> None:
        for _ in range(max_ticks):
            controller.tick(op.latest() if op is not None else None)
            if pred():
                return
        raise AssertionError("condition never reached")

    try:
        co.start_cycle(1, flap_jammed=True)
        co.start_cycle(2, flap_jammed=True)
        tick_until(
            lambda: (
                controller.rescue_bay() is not None
                and all(co.last_signature(b) is not None for b in (1, 2))
            )
        )
        bay = controller.rescue_bay()
        assert bay is not None
        recorder.bay = bay
        glove = ScriptedGlove(controller)
        controller.on_key(ord(str(bay)))
        controller.on_key(KEY_SPACE)
        rig = co.rig(bay)
        tick_until(lambda: rig.bolt_x() >= BOLT_CLEARED_M + 0.002, glove)
        for _ in range(15):
            controller.tick(glove.latest())
        sigs: dict[int, tuple[float, float, float, float]] = {}
        for b in (1, 2):
            z = co.last_signature(b)
            assert z is not None
            a, bb, c, d = (float(v) for v in z.as_vector())
            sigs[b] = (a, bb, c, d)
    finally:
        controller.close()
        unoq.close()
    rows = np.asarray(recorder.rows[::2], dtype=np.float32)  # 60 Hz ticks -> 30 fps rows
    assert len(rows) > 60
    return Demo(rows, sigs)


def episode_of(demo: Demo, *, fault: str = "flap", source: str = "scripted") -> Any:
    """The demo as an in-memory eval episode (what ``read_eval_episodes`` yields for a dataset)."""
    from teleop.commands.eval import EvalEpisode
    from teleop.policy.schema import _reference_rotation  # pyright: ignore[reportPrivateUsage]
    from teleop.unoq.dmp import mat_to_quat, so3_exp

    a = demo.actions.astype(np.float64)
    ref = _reference_rotation()
    wrist = np.array([[*r[:3], *mat_to_quat(ref @ so3_exp(r[3:6]))] for r in a])
    n = len(a)
    return EvalEpisode(
        fault=fault,
        source=source,
        success=True,
        signature=demo.signatures[1],
        t_s=np.arange(n, dtype=np.float64) / 30.0,
        wrist_latch=wrist,
        hand=a[:, 6:12],
        tau_hand_inf=np.full(n, 0.2),
    )


@pytest.fixture(scope="module")
def demo() -> Demo:
    if not _models_present():
        pytest.skip("Panda/Inspire MJCFs missing; run scripts/fetch_models.py")
    return build_demo()


@pytest.fixture(scope="module")
def validator() -> Iterator[Any]:
    from teleop.atc.coordinator import DEFAULT_REALISM
    from teleop.atc.validator import SkillValidator

    if not _models_present():
        pytest.skip("Panda/Inspire MJCFs missing; run scripts/fetch_models.py")
    yield SkillValidator(load_hand("inspire"), realism=DEFAULT_REALISM)


def write_checkpoint(path: Path) -> Path:
    """A schema-correct ``config.json`` and stand-in weights (the policy itself is stubbed)."""
    from teleop.policy.act_runtime import ACTION_KEY, IMAGE_KEY, STATE_KEY
    from teleop.policy.schema import ACTION_NAMES, STATE_NAMES

    path.mkdir(parents=True, exist_ok=True)
    feats: dict[str, Any] = {STATE_KEY: {"type": "STATE", "shape": [len(STATE_NAMES)]}}
    for cam in CAMERAS:
        feats[IMAGE_KEY.format(cam)] = {"type": "VISUAL", "shape": [3, *IMAGE_HW]}
    config = {
        "input_features": feats,
        "output_features": {ACTION_KEY: {"type": "ACTION", "shape": [len(ACTION_NAMES)]}},
    }
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"weights")
    return path


# --------------------------------------------------------------------------- baseline


@needs_models
def test_validate_unchanged_for_dmp(validator: Any, demo: Demo) -> None:
    from teleop.commands.eval import fit_dmp_for

    skill = fit_dmp_for("flap", [episode_of(demo)])
    assert skill is not None
    report = validator.validate(skill)
    assert report.n_passed == GOLDEN_DMP_N_PASSED
    worst = max(r.max_tau_ext_inf_nm for r in report.results)
    assert worst == pytest.approx(GOLDEN_DMP_MAX_TAU_NM, abs=1e-9)


# --------------------------------------------------------------------------- teleop eval


def _run_eval(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    actions: NDArray[np.float32],
    episodes: list[Any],
    *extra: str,
) -> tuple[int, Path]:
    from teleop.commands import eval as ev
    from teleop.policy.act_runtime import ReplayPolicy

    ckpt = write_checkpoint(tmp_path / "ckpt")
    monkeypatch.setattr(ev, "_load_policy", lambda _c, _d: ReplayPolicy(actions))
    monkeypatch.setattr(ev, "_make_renderer", lambda _m, _hw: ZeroRenderer())
    monkeypatch.setattr(ev, "read_eval_episodes", lambda _root: episodes)
    code = ev.main(
        ["--checkpoint", str(ckpt), "--dataset", str(tmp_path / "ds"), "--device", "cpu", *extra]
    )
    return code, ckpt / "gate.json"


@needs_models
def test_eval_writes_manifest_with_passed_fault(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, demo: Demo
) -> None:
    from teleop.policy.manifest import GateManifest

    code, gate = _run_eval(
        monkeypatch,
        tmp_path,
        demo.actions,
        [episode_of(demo)],
        "--faults",
        "flap",
        "--rollouts",
        "4",
        "--bar",
        "3",
    )
    manifest = GateManifest.load(gate)
    assert code == 0
    assert manifest.schema_version == SCHEMA_VERSION
    assert manifest.bar == (3, 4)
    assert manifest.passed_faults() == ("flap",)
    assert manifest.prototypes["flap"] == [demo.signatures[1]]
    flap = manifest.results["flap"]
    assert flap.rollouts == 4 and flap.passed_rollouts >= 3
    assert flap.dmp_passed_rollouts is not None  # the DMP ran the same conditions
    assert flap.mean_time_to_clear_s is not None and flap.mean_time_to_clear_s > 0.0
    assert len(manifest.checkpoint_sha256) == 64


@needs_models
def test_eval_fault_below_bar_not_passed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, demo: Demo
) -> None:
    from teleop.policy.manifest import GateManifest

    idle = np.repeat(demo.actions[:1], len(demo.actions), axis=0)
    code, gate = _run_eval(
        monkeypatch,
        tmp_path,
        idle,
        [episode_of(demo)],
        "--faults",
        "flap",
        "--rollouts",
        "2",
        "--bar",
        "2",
    )
    manifest = GateManifest.load(gate)
    assert code == 1
    assert "flap" not in manifest.passed_faults()
    assert "flap" not in manifest.prototypes  # prototypes exist only for passed faults
    assert manifest.results["flap"].passed_rollouts < 2


@needs_models
def test_eval_no_data_fault(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, demo: Demo) -> None:
    from teleop.policy.manifest import GateManifest

    idle = np.repeat(demo.actions[:1], len(demo.actions), axis=0)
    _, gate = _run_eval(
        monkeypatch,
        tmp_path,
        idle,
        [episode_of(demo)],
        "--faults",
        "flap,pin",
        "--rollouts",
        "2",
        "--bar",
        "2",
    )
    manifest = GateManifest.load(gate)
    pin = manifest.results["pin"]
    assert pin.note == "no data" and pin.rollouts == 0
    assert "pin" not in manifest.passed_faults() and "pin" not in manifest.prototypes


@needs_models
def test_pin_fault_uses_bolt_pinned(validator: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from teleop.atc.recovery import BaySignals, RecoveryCommand

    class Stop:
        name = "stop"

        def command(self, rig: Any, s: BaySignals) -> RecoveryCommand | None:
            return None

        def check(self, rig: Any, s: BaySignals) -> str | None:
            return "stop"

        def close(self) -> None:
            pass

    calls: list[dict[str, bool]] = []
    real = validator.rig.reset_latch

    def spy(**kw: bool) -> None:
        calls.append(kw)
        real(**kw)

    monkeypatch.setattr(validator.rig, "reset_latch", spy)
    cond = validator.draw_conditions("eval/pin")[:1]
    for fault, expected in (
        ("pin", {"flap_jammed": False, "bolt_pinned": True}),
        ("flap", {"flap_jammed": True, "bolt_pinned": False}),
    ):
        calls.clear()
        report = validator.validate_controller(lambda _rig: Stop(), fault=fault, conditions=cond)
        assert calls == [expected]
        assert report.results[0].abort_reason == "stop"
        assert not report.results[0].passed


@needs_models
def test_controller_rollouts_restore_model_parameters(validator: Any) -> None:
    from teleop.atc.recovery import BaySignals, RecoveryCommand

    class Stop:
        name = "stop"

        def command(self, rig: Any, s: BaySignals) -> RecoveryCommand | None:
            return None

        def check(self, rig: Any, s: BaySignals) -> str | None:
            return "stop"

        def close(self) -> None:
            pass

    m = validator.model
    before = m.geom_friction.copy(), m.jnt_stiffness.copy(), m.qpos_spring.copy()
    validator.validate_controller(
        lambda _rig: Stop(), conditions=validator.draw_conditions("x")[:2]
    )
    np.testing.assert_array_equal(m.geom_friction, before[0])
    np.testing.assert_array_equal(m.jnt_stiffness, before[1])
    np.testing.assert_array_equal(m.qpos_spring, before[2])


def test_eval_refuses_checkpoint_of_another_schema(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from teleop.commands import eval as ev

    ckpt = tmp_path / "old"
    ckpt.mkdir()
    (ckpt / "config.json").write_text(json.dumps({"input_features": {}, "output_features": {}}))
    monkeypatch.setattr(ev, "read_eval_episodes", lambda _r: [])
    code = ev.main(["--checkpoint", str(ckpt), "--dataset", str(tmp_path), "--device", "cpu"])
    assert code == 2
    assert "does not match the policy schema" in capsys.readouterr().err
    assert not (ckpt / "gate.json").exists()


@pytest.mark.parametrize(
    "argv",
    [["--faults", "wiggle"], ["--bar", "11"], ["--faults", ""], ["--rollouts", "3", "--bar", "0"]],
)
def test_eval_rejects_bad_arguments(argv: list[str], tmp_path: Path) -> None:
    from teleop.commands import eval as ev

    assert ev.main(["--checkpoint", str(tmp_path), "--dataset", str(tmp_path), *argv]) == 2


def test_mirror_copies_joints_and_latch_placement() -> None:
    from teleop.commands.eval import mirror_bay_state
    from teleop.sim.bays import CANONICAL_LATCH, build_four_bay_scene, build_single_bay_scene

    if not _models_present():
        pytest.skip("Panda/Inspire MJCFs missing; run scripts/fetch_models.py")
    src, dst = build_single_bay_scene(CANONICAL_LATCH), build_four_bay_scene()
    sd, dd = mj.MjData(src), mj.MjData(dst)
    j1 = src.joint("bay1/joint1")
    sd.qpos[int(j1.qposadr[0])] = 0.37
    latch = src.body("bay1/latch_base")
    src.body_pos[int(latch.id)] += (0.01, -0.02, 0.0)
    mirror_bay_state(src, sd, dst, dd)
    assert dd.qpos[int(dst.joint("bay1/joint1").qposadr[0])] == pytest.approx(0.37)
    np.testing.assert_allclose(
        dst.body_pos[int(dst.body("bay1/latch_base").id)], src.body_pos[int(latch.id)]
    )
