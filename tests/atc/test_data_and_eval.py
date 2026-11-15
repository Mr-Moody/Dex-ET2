"""ATC episode recording, scripted generation, dataset QC, and offline eval gate."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco as _mujoco  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
import pytest
from numpy.typing import NDArray

from conftest import NullReasoner, RecordingComplianceSink, models_present, needs_models
from teleop.atc.generate import GeneratorConfig, latch_perturbation, run_generation
from teleop.policy.schema import (
    ACTION_NAMES,
    CAMERAS,
    FPS,
    IMAGE_HW,
    SCHEMA_VERSION,
    STATE_NAMES,
    action_from_command,
    command_from_action,
    lerobot_features,
)
from teleop.types import Pose
from teleop.unoq.client import UnoQClient
from teleop.unoq.dmp import HAND_ACTUATORS
from teleop.unoq.server import UnoQServer

mj: Any = _mujoco

GOLDEN_DMP_N_PASSED = 20
IDENTITY_POSE = Pose(orientation=(0.0, 0.0, 0.0, 1.0), position=(0.0, 0.0, 0.0))


@pytest.fixture(autouse=True)
def _offline_hf(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")


class ZeroRenderer:
    """Headless policy-camera stand-in when GL rendering is not the subject under test."""

    def __init__(self, hw: tuple[int, int] = IMAGE_HW) -> None:
        self.hw = hw

    def render(self, data: Any, bay_id: int) -> dict[str, NDArray[np.uint8]]:
        return {cam: np.zeros((*self.hw, 3), dtype=np.uint8) for cam in CAMERAS}

    def close(self) -> None:
        return None


class MemoryRecorder:
    """In-memory EpisodeRecorder stand-in with optional encoding delay."""

    def __init__(self, save_delay_s: float = 0.0) -> None:
        self.save_delay_s = save_delay_s
        self.episodes: list[tuple[str, int, dict[str, Any]]] = []
        self._task: str | None = None
        self._frames = 0

    def start_episode(self, task: str) -> None:
        self._task, self._frames = task, 0

    def add_frame(self, state: Any, action: Any, images: Mapping[str, Any]) -> None:
        self._frames += 1

    def end_episode(self, meta: Mapping[str, Any]) -> int:
        if self.save_delay_s:
            time.sleep(self.save_delay_s)
        assert self._task is not None
        self.episodes.append((self._task, self._frames, dict(meta)))
        return len(self.episodes) - 1

    def discard_episode(self) -> None:
        self._task = None

    def close(self) -> None:
        return None


def _build_stage(unoq: UnoQClient) -> tuple[Any, RecordingComplianceSink]:
    from teleop.atc.coordinator import AtcCoordinator
    from teleop.atc.stage import EventTap, ReasonerTap, StageController

    sink = RecordingComplianceSink()
    events = EventTap(sink)
    reasoner = NullReasoner()
    co = AtcCoordinator.build(unoq, events, reasoner)
    return StageController(co, events, ReasonerTap(reasoner), cycle_dwell_s=0.5), sink


def _tick_until(
    controller: Any, pred: Callable[[], bool], op: Any = None, max_ticks: int = 6000
) -> None:
    for _ in range(max_ticks):
        controller.tick(op.latest() if op is not None else None)
        if pred():
            return
    raise AssertionError("condition never reached")


# ------------------------------------------------------------------ generation & EpisodeSink


def test_perturbation_bounds_and_seed_determinism() -> None:
    cfg = GeneratorConfig(perturb_xy_m=0.05, perturb_yaw_deg=15.0)
    draws_a = [latch_perturbation(np.random.default_rng(7), cfg) for _ in range(50)]
    draws_b = [latch_perturbation(np.random.default_rng(7), cfg) for _ in range(50)]
    assert draws_a == draws_b
    assert all(
        abs(dx) <= 0.05 and abs(dy) <= 0.05 and abs(dyaw) <= 15.0 for dx, dy, dyaw in draws_a
    )
    assert draws_a[0] != latch_perturbation(np.random.default_rng(8), cfg)


@needs_models
def test_generator_produces_successes_and_stops_on_unrecoverable_fault(
    unoq_client: UnoQClient, tmp_path: Path
) -> None:
    from teleop.atc.episodes import EpisodeSink, fault_lookup
    from teleop.atc.stage import StageController

    for idx, (target, cfg, expect_ok) in enumerate(
        [
            (2, GeneratorConfig(), True),
            (1, GeneratorConfig(faults=("bay2_pin",), max_sim_s_per_episode=12.0), False),
        ]
    ):
        controller = StageController.build(
            unoq_client, RecordingComplianceSink(), NullReasoner(), seed=0, cycle_dwell_s=0.5
        )
        rec = MemoryRecorder()
        sink = EpisodeSink(
            tmp_path / f"gen_{idx}",
            controller.coordinator.model,
            controller.coordinator.data,
            fault_lookup(controller),
            "scripted",
            renderer=ZeroRenderer(),
            recorder=rec,
        )
        n = run_generation(controller, sink, target, seed=0, config=cfg)
        sink.close()
        controller.close()
        if expect_ok:
            assert n >= target
            assert sum(1 for _, _, m in rec.episodes if m["success"]) >= target
        else:
            assert n == 0 and not any(m["success"] for _, _, m in rec.episodes)


@needs_models
def test_episode_sink_nonblocking_and_two_bay_stage_metadata(
    unoq_client: UnoQClient, tmp_path: Path
) -> None:
    from teleop.atc.episodes import EpisodeSink, fault_lookup
    from teleop.atc.stage import StageDirector

    controller, events = _build_stage(unoq_client)
    rec = MemoryRecorder(save_delay_s=0.25)
    sink = EpisodeSink(
        tmp_path / "nonblock",
        controller.coordinator.model,
        controller.coordinator.data,
        fault_lookup(controller),
        "glove",
        renderer=ZeroRenderer(),
        recorder=rec,
    )
    rig = controller.coordinator.rig(1)
    wrist = rig.wrist_pose()
    durations: list[float] = []
    for i in range(90):
        if i % 30 == 0:
            sink.on_episode_start(1, "operator", None, None)
        t0 = time.perf_counter()
        sink.on_tick(1, rig, wrist, {})
        durations.append(time.perf_counter() - t0)
        if i % 30 == 29:
            sink.on_episode_end(1, True, None)
    assert float(np.percentile(durations, 99)) < 0.005
    # Empty episode without frames is dropped cleanly.
    sink.on_episode_start(1, "operator", None, None)
    sink.on_episode_end(1, False, "aborted")
    sink.close()
    assert len(rec.episodes) == 3

    # Full Bay 1 glove rescue -> Bay 2 DMP clear records both episodes with signatures.
    rec_stage = MemoryRecorder()
    sink_stage = EpisodeSink(
        tmp_path / "stage",
        controller.coordinator.model,
        controller.coordinator.data,
        fault_lookup(controller),
        "glove",
        renderer=ZeroRenderer(),
        recorder=rec_stage,
    )
    controller.coordinator.set_observer(sink_stage)
    director = StageDirector(controller)
    for _ in range(8000):
        controller.tick(director.step())
        if any(e.kind == "RESCUE_LOGGED" and e.bay_id == 2 for e in events.events):
            break
    sink_stage.close()
    controller.close()
    metas = [m for _, _, m in rec_stage.episodes]
    assert sorted((m["bay"], m["source"]) for m in metas) == [(1, "glove"), (2, "dmp")]
    assert all(m["success"] and m["fault"] == "flap" for m in metas)


@needs_models
def test_episode_sink_tags_a_queued_episode_into_its_meta(
    unoq_client: UnoQClient, tmp_path: Path
) -> None:
    from teleop.atc.episodes import EpisodeSink, fault_lookup

    controller, _ = _build_stage(unoq_client)
    rec = MemoryRecorder(save_delay_s=0.3)
    sink = EpisodeSink(
        tmp_path / "tag",
        controller.coordinator.model,
        controller.coordinator.data,
        fault_lookup(controller),
        "glove",
        renderer=ZeroRenderer(),
        recorder=rec,
    )
    rig = controller.coordinator.rig(1)
    wrist = rig.wrist_pose()

    def record_one() -> str:
        sink.on_episode_start(1, "operator", None, None)
        sink.on_tick(1, rig, wrist, {})
        sink.on_episode_end(1, True, None)
        uid = sink.latest_episode_uid(1)
        assert isinstance(uid, str)
        return uid

    record_one()  # occupies the writer
    queued = record_one()
    assert sink.tag_episode(queued, dataset_tag="flag")
    assert sink.tag_episode("bay9-nope", dataset_tag="flag") is False
    sink.close()
    controller.close()
    assert [m.get("dataset_tag") for _, _, m in rec.episodes] == [None, "flag"]


# ------------------------------------------------------------------ LeRobot dataset & QC


@pytest.mark.lerobot
@needs_models
def test_lerobot_episode_recording_and_eval_reader_roundtrip(
    unoq_client: UnoQClient, tmp_path: Path
) -> None:
    pytest.importorskip("lerobot")
    from teleop.atc.episodes import EpisodeSink, fault_lookup
    from teleop.atc.stage import KEY_SPACE, ScriptedGlove
    from teleop.atc.types import BOLT_CLEARED_M
    from teleop.commands.eval import read_eval_episodes
    from teleop.process.qc import open_dataset

    controller, _ = _build_stage(unoq_client)
    root = tmp_path / "ds"
    sink = EpisodeSink(
        root,
        controller.coordinator.model,
        controller.coordinator.data,
        fault_lookup(controller),
        "scripted",
        renderer=ZeroRenderer(),
    )
    controller.coordinator.set_observer(sink)
    glove = ScriptedGlove(controller)
    controller.on_key(ord("F"))
    _tick_until(controller, lambda: controller.rescue_bay() == 1)
    controller.on_key(ord("1"))
    controller.on_key(KEY_SPACE)
    rig = controller.coordinator.rig(1)
    _tick_until(controller, lambda: rig.bolt_x() >= BOLT_CLEARED_M + 0.002, glove)
    for _ in range(15):
        controller.tick(glove.latest())
    controller.on_key(KEY_SPACE)
    for _ in range(3):
        controller.tick(glove.latest())
    sink.close()
    controller.close()

    ds = open_dataset(sink.root)
    assert ds.num_episodes == 1
    assert ds.meta.features["observation.state"]["names"] == list(STATE_NAMES)
    assert ds.meta.features["action"]["names"] == list(ACTION_NAMES)
    (ep,) = read_eval_episodes(sink.root)
    assert (ep.fault, ep.source, ep.success) == ("flap", "scripted", True)
    assert ep.hand.shape[1] == len(HAND_ACTUATORS) and ep.wrist_latch.shape[1] == 7


@pytest.mark.lerobot
def test_atc_qc_filters_trims_and_writes_clean_dataset(tmp_path: Path) -> None:
    pytest.importorskip("lerobot")
    from teleop.process.atc_qc import AtcQCThresholds, atc_trim, run_atc_qc
    from teleop.process.qc import open_dataset, read_sidecar, write_clean
    from teleop.record.recorder import EpisodeRecorder

    hw = (32, 32)
    ranges = {
        "hand/pinky": (0.0, 1.57),
        "hand/ring": (0.0, 1.57),
        "hand/middle": (0.0, 1.5),
        "hand/index": (0.0, 1.5),
        "hand/thumb_proximal": (0.1, 0.57),
        "hand/thumb_yaw": (0.0, 1.308),
    }

    def make_actions(idle_head: int, moving: int, idle_tail: int) -> NDArray[np.float32]:
        n = idle_head + moving + idle_tail
        a = np.zeros((n, len(ACTION_NAMES)), np.float32)
        a[:, 6:] = 0.3
        for k in range(n):
            a[k, 0] = min(max(k - idle_head + 1, 0), moving) * 0.002
        return a

    def make_meta(**over: Any) -> dict[str, Any]:
        base: dict[str, Any] = {
            "source": "glove",
            "fault": "pin",
            "bay": 1,
            "signature_id": "Jam_Sig_1",
            "signature": [0.035, 1.2, 2.4, 0.3],
            "latch_pose": {"x": 0.5},
            "success": True,
            "bolt_x_final": 0.054,
            "max_tau_hand_inf": 2.0,
            "max_arm_ratio": 0.4,
            "aborted_reason": None,
        }
        base.update(over)
        return base

    assert atc_trim(make_actions(8, 30, 6), 1e-3) == (8, 38)
    out_of_range = make_actions(5, 30, 5)
    out_of_range[15, 6] = 1.65
    cases = [
        (make_actions(6, 30, 6), make_meta(), []),
        (make_actions(5, 30, 5), make_meta(success=False), ["not successful"]),
        (make_actions(5, 30, 5), make_meta(source="act"), ["source excluded"]),
        (make_actions(5, 10, 5), make_meta(), ["too short"]),
        (make_actions(5, 30, 5), make_meta(max_tau_hand_inf=12.0), ["torque breach"]),
        (out_of_range, make_meta(), ["action out of range"]),
        (make_actions(6, 30, 6), make_meta(dataset_tag="flag"), ["copilot flagged"]),
        (make_actions(6, 30, 6), make_meta(dataset_tag="approve"), []),
    ]
    raw_root = tmp_path / "raw"
    rec = EpisodeRecorder("local/raw", raw_root, FPS, lerobot_features(hw), "panda_inspire")
    rng = np.random.default_rng(0)
    for actions, meta, _ in cases:
        rec.start_episode("recover latch: pin")
        for act in actions:
            rec.add_frame(
                np.zeros(len(STATE_NAMES), np.float32),
                act,
                {cam: rng.integers(0, 255, (*hw, 3), dtype=np.uint8) for cam in CAMERAS},
            )
        rec.end_episode(meta)
    rec.close()

    results = run_atc_qc(
        raw_root, AtcQCThresholds(), frozenset({"glove", "scripted", "dmp"}), ranges=ranges
    )
    assert [r.reasons for r in results] == [expected for _, _, expected in cases]

    clean_root = tmp_path / "clean"
    write_clean(raw_root, results, clean_root)
    rows = read_sidecar(clean_root)
    assert len(rows) == 2 and rows[0]["fault"] == "pin" and rows[0]["source"] == "glove"
    ds = open_dataset(clean_root)
    assert int(ds.meta.total_episodes) == 1 and len(ds) == 30


# ------------------------------------------------------------------ offline ACT eval gate


@dataclass(frozen=True)
class _Demo:
    actions: NDArray[np.float32]
    signature: tuple[float, float, float, float]


class _ActionObserver:
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


@pytest.fixture(scope="module")
def flap_demo() -> _Demo:
    if not models_present():
        pytest.skip("Panda/Inspire MJCFs missing; run `uv run python scripts/fetch_models.py`")
    from teleop.atc.coordinator import AtcCoordinator
    from teleop.atc.stage import KEY_SPACE, EventTap, ReasonerTap, ScriptedGlove, StageController
    from teleop.atc.types import BOLT_CLEARED_M

    server = UnoQServer(host="127.0.0.1", udp_port=0, tcp_port=0, name="eval-demo").start()
    unoq = UnoQClient("127.0.0.1", server.udp_port, server.tcp_port, local_server=server).connect()
    obs = _ActionObserver()
    events = EventTap(RecordingComplianceSink())
    co = AtcCoordinator.build(unoq, events, NullReasoner(), observer=obs)
    controller = StageController(co, events, ReasonerTap(NullReasoner()), cycle_dwell_s=0.5)
    try:
        co.start_cycle(1, flap_jammed=True)
        co.start_cycle(2, flap_jammed=True)
        _tick_until(
            controller,
            lambda: (
                controller.rescue_bay() is not None
                and all(co.last_signature(b) is not None for b in (1, 2))
            ),
            max_ticks=3000,
        )
        bay = controller.rescue_bay()
        assert bay is not None
        obs.bay = bay
        glove = ScriptedGlove(controller)
        controller.on_key(ord(str(bay)))
        controller.on_key(KEY_SPACE)
        rig = co.rig(bay)
        _tick_until(
            controller, lambda: rig.bolt_x() >= BOLT_CLEARED_M + 0.002, glove, max_ticks=3000
        )
        for _ in range(15):
            controller.tick(glove.latest())
        sig = co.last_signature(1)
        assert sig is not None
        a, b, c, d = (float(v) for v in sig.as_vector())
        return _Demo(np.asarray(obs.rows[::2], dtype=np.float32), (a, b, c, d))
    finally:
        controller.close()
        unoq.close()


def _eval_episode(demo: _Demo, *, fault: str = "flap", source: str = "scripted") -> Any:
    from teleop.commands.eval import EvalEpisode

    a = demo.actions.astype(np.float64)
    wrist_rows: list[list[float]] = []
    for row in a:
        rel_pose, _ = command_from_action(IDENTITY_POSE, row)
        assert rel_pose.position is not None
        wrist_rows.append([*rel_pose.position, *rel_pose.orientation])
    n = len(a)
    return EvalEpisode(
        fault=fault,
        source=source,
        success=True,
        signature=demo.signature,
        t_s=np.arange(n, dtype=np.float64) / 30.0,
        wrist_latch=np.asarray(wrist_rows, dtype=np.float64),
        hand=a[:, 6:12],
        tau_hand_inf=np.full(n, 0.2),
    )


def _write_checkpoint(path: Path) -> Path:
    from teleop.policy.act_runtime import ACTION_KEY, IMAGE_KEY, STATE_KEY

    path.mkdir(parents=True, exist_ok=True)
    feats: dict[str, Any] = {STATE_KEY: {"type": "STATE", "shape": [len(STATE_NAMES)]}}
    for cam in CAMERAS:
        feats[IMAGE_KEY.format(cam)] = {"type": "VISUAL", "shape": [3, *IMAGE_HW]}
    cfg = {
        "input_features": feats,
        "output_features": {ACTION_KEY: {"type": "ACTION", "shape": [len(ACTION_NAMES)]}},
    }
    (path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"weights")
    return path


@needs_models
def test_validate_dmp_golden_and_eval_gate_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, flap_demo: _Demo
) -> None:
    from teleop.atc.coordinator import DEFAULT_REALISM
    from teleop.atc.validator import SkillValidator
    from teleop.commands import eval as ev
    from teleop.policy.act_runtime import ReplayPolicy
    from teleop.sim.hands import load_hand

    ep = _eval_episode(flap_demo)
    skill = ev.fit_dmp_for("flap", [ep])
    assert skill is not None
    validator = SkillValidator(load_hand("inspire"), realism=DEFAULT_REALISM)
    report = validator.validate(skill)
    assert report.passed and report.n_passed == GOLDEN_DMP_N_PASSED
    assert 0.5 < max(r.max_tau_ext_inf_nm for r in report.results) < 3.0

    monkeypatch.setattr(ev, "_make_renderer", lambda _m, _hw: ZeroRenderer())
    ckpt = _write_checkpoint(tmp_path / "ckpt_pass")
    manifest = ev.evaluate(
        ckpt,
        [ep],
        faults=("flap", "pin"),
        rollouts=4,
        bar=3,
        device="cpu",
        seed=0,
        realism=DEFAULT_REALISM,
        policy=ReplayPolicy(flap_demo.actions),
    )
    assert manifest.schema_version == SCHEMA_VERSION
    assert manifest.passed_faults() == ("flap",)
    assert manifest.prototypes["flap"] == [flap_demo.signature]
    assert manifest.results["flap"].dmp_passed_rollouts is not None
    assert manifest.results["pin"].note == "no data"

    # Reproducibility check + idle policy below bar check.
    manifest_again = ev.evaluate(
        ckpt,
        [ep],
        faults=("flap",),
        rollouts=2,
        bar=2,
        device="cpu",
        seed=0,
        realism=DEFAULT_REALISM,
        policy=ReplayPolicy(flap_demo.actions),
    )
    manifest_repeat = ev.evaluate(
        ckpt,
        [ep],
        faults=("flap",),
        rollouts=2,
        bar=2,
        device="cpu",
        seed=0,
        realism=DEFAULT_REALISM,
        policy=ReplayPolicy(flap_demo.actions),
    )
    assert manifest_again.results["flap"] == manifest_repeat.results["flap"]

    idle = np.repeat(flap_demo.actions[:1], len(flap_demo.actions), axis=0)
    failed = ev.evaluate(
        ckpt,
        [ep],
        faults=("flap",),
        rollouts=2,
        bar=2,
        device="cpu",
        seed=0,
        realism=DEFAULT_REALISM,
        policy=ReplayPolicy(idle),
    )
    assert failed.passed_faults() == () and "flap" not in failed.prototypes

    # CLI validation rejects bad schema and invalid flags with exit code 2.
    bad_ckpt = tmp_path / "bad_ckpt"
    bad_ckpt.mkdir()
    (bad_ckpt / "config.json").write_text('{"input_features":{},"output_features":{}}')
    monkeypatch.setattr(ev, "read_eval_episodes", lambda _r: [])
    assert ev.main(["--checkpoint", str(bad_ckpt), "--dataset", str(tmp_path)]) == 2
    assert ev.main(["--checkpoint", str(ckpt), "--dataset", str(tmp_path), "--faults", "nope"]) == 2
    assert ev.main(["--checkpoint", str(ckpt), "--dataset", str(tmp_path), "--bar", "99"]) == 2


@needs_models
def test_mirror_renderer_tracks_moved_latch(offscreen_render_ok: None) -> None:
    from teleop.commands.eval import MirrorRenderer
    from teleop.retarget.clutch import rescue_camera_names
    from teleop.sim.bays import CANONICAL_LATCH, build_single_bay_scene

    src = build_single_bay_scene(CANONICAL_LATCH)
    sd = mj.MjData(src)
    mirror = MirrorRenderer(src, (64, 64))
    try:
        top = int(mirror.model.camera(rescue_camera_names("bay1/")[0]).id)
        latch = int(src.body("bay1/latch_base").id)

        def latch_in_camera() -> NDArray[np.float64]:
            mirror.render(sd, 1)
            d = mirror.data
            body = int(mirror.model.body("bay1/latch_base").id)
            rot = np.asarray(d.cam_xmat[top], dtype=np.float64).reshape(3, 3)
            return rot.T @ (np.asarray(d.xpos[body]) - np.asarray(d.cam_xpos[top]))

        mj.mj_forward(src, sd)
        nominal = latch_in_camera()
        yaw = math.radians(30.0)
        src.body_pos[latch] += (0.08, 0.06, 0.0)
        w, x, y, z = (float(v) for v in src.body_quat[latch])
        c, s = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
        src.body_quat[latch] = (c * w - s * z, c * x - s * y, c * y + s * x, c * z + s * w)
        mj.mj_forward(src, sd)
        moved = latch_in_camera()
    finally:
        mirror.close()
    np.testing.assert_allclose(moved, nominal, atol=1e-3)
