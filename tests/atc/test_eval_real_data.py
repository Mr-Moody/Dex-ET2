"""``teleop eval`` on a real recorded LeRobot dataset, with the real camera mirror.

``test_eval_gate.py`` covers the gate logic with in-memory episodes and a zero renderer. These
tests record a scripted rescue of a jammed flap into a LeRobot v3 dataset, the way T-191 does,
and run the real ``read_eval_episodes``, the DMP fit and the ``MirrorRenderer`` on it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import mujoco as _mujoco  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
import pytest
from numpy.typing import NDArray

from teleop.sim.bays import PANDA_XML
from teleop.sim.hands import load_hand

mj: Any = _mujoco


def _models_present() -> bool:
    try:
        right = load_hand("inspire").mjcf["right"]
    except FileNotFoundError:
        return False
    return PANDA_XML.is_file() and right.is_file()


needs_models = pytest.mark.skipif(
    not _models_present(), reason="Panda/Inspire MJCFs missing; run scripts/fetch_models.py"
)


class _Sink:
    def emit(self, event: Any) -> None:
        pass

    def close(self) -> None:
        pass


class _Quiet:
    def submit(self, ctx: Any) -> None:
        pass

    def latest_verdict(self, bay_id: int) -> None:
        return None

    def close(self) -> None:
        pass


class _Capture:
    """Observer: one bay's operator rescue as (state, latch-frame action) rows per 60 Hz tick."""

    def __init__(self) -> None:
        self.bay: int | None = None
        self.states: list[Any] = []
        self.actions: list[Any] = []

    def on_episode_start(self, bay_id: int, source: str, signature_id: Any, signature: Any) -> None:
        pass

    def on_tick(self, bay_id: int, rig: Any, wrist_world: Any, fingers: Any) -> None:
        from teleop.policy.schema import action_from_command, observe

        if bay_id == self.bay:
            self.states.append(observe(rig))
            self.actions.append(action_from_command(rig.world_from_latch(), wrist_world, fingers))

    def on_episode_end(self, bay_id: int, success: bool, aborted_reason: str | None) -> None:
        pass


class Recorded:
    def __init__(self, root: Path, states: NDArray[np.float32], actions: NDArray[np.float32]):
        self.root, self.states, self.actions = root, states, actions


@pytest.fixture(scope="module")
def recorded(tmp_path_factory: pytest.TempPathFactory) -> Recorded:
    """A one-episode ATC dataset: a scripted-glove rescue of a jammed flap at 30 fps."""
    if not _models_present():
        pytest.skip("Panda/Inspire MJCFs missing; run scripts/fetch_models.py")
    pytest.importorskip("lerobot")
    from teleop.atc.coordinator import AtcCoordinator
    from teleop.atc.stage import KEY_SPACE, EventTap, ReasonerTap, ScriptedGlove, StageController
    from teleop.atc.types import BOLT_CLEARED_M
    from teleop.policy.schema import CAMERAS, FPS, lerobot_features
    from teleop.record.recorder import EpisodeRecorder
    from teleop.unoq.client import UnoQClient
    from teleop.unoq.server import UnoQServer

    server = UnoQServer(host="127.0.0.1", udp_port=0, tcp_port=0, name="test-node").start()
    unoq = UnoQClient("127.0.0.1", server.udp_port, server.tcp_port, local_server=server).connect()
    cap = _Capture()
    events = EventTap(_Sink())
    co = AtcCoordinator.build(unoq, events, _Quiet(), observer=cap)
    controller = StageController(co, events, ReasonerTap(_Quiet()), cycle_dwell_s=0.5)

    def tick_until(pred: Any, op: Any = None, max_ticks: int = 3000) -> None:
        for _ in range(max_ticks):
            controller.tick(op.latest() if op is not None else None)
            if pred():
                return
        raise AssertionError("condition never reached")

    try:
        co.start_cycle(1, flap_jammed=True)
        tick_until(lambda: controller.rescue_bay() is not None and co.last_signature(1) is not None)
        cap.bay = 1
        glove = ScriptedGlove(controller)
        controller.on_key(ord("1"))
        controller.on_key(KEY_SPACE)
        rig = co.rig(1)
        tick_until(lambda: rig.bolt_x() >= BOLT_CLEARED_M + 0.002, glove)
        for _ in range(15):
            controller.tick(glove.latest())
        z = co.last_signature(1)
        assert z is not None
        bolt_final = rig.bolt_x()
    finally:
        controller.close()
        unoq.close()
    states = np.asarray(cap.states[::2], dtype=np.float32)  # 60 Hz ticks -> 30 fps rows
    actions = np.asarray(cap.actions[::2], dtype=np.float32)
    assert len(actions) > 60
    root = tmp_path_factory.mktemp("eval") / "scripted"
    hw = (32, 32)  # the AV1 encoder crashes on 16 x 16 frames
    rec = EpisodeRecorder(f"local/{root.name}", root, FPS, lerobot_features(hw), "panda_inspire")
    rec.start_episode("recover latch: flap")
    for s, a in zip(states, actions, strict=True):
        rec.add_frame(s, a, {c: np.zeros((*hw, 3), np.uint8) for c in CAMERAS})
    rec.end_episode(
        {
            "source": "scripted",
            "fault": "flap",
            "bay": 1,
            "signature_id": "Jam_Sig_1",
            "signature": list(z.as_vector()),
            "latch_pose": {},
            "success": True,
            "bolt_x_final": bolt_final,
            "max_tau_hand_inf": 0.0,
            "max_arm_ratio": 0.0,
            "aborted_reason": None,
        }
    )
    rec.close()
    return Recorded(root, states, actions)


def _write_checkpoint(path: Path) -> Path:
    """A schema-correct ``config.json`` and stand-in weights (the policy itself is stubbed)."""
    from teleop.policy.act_runtime import ACTION_KEY, IMAGE_KEY, STATE_KEY
    from teleop.policy.schema import ACTION_NAMES, CAMERAS, IMAGE_HW, STATE_NAMES

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


class _ImageSpy:
    """Wraps a policy: keeps the camera images of every batch it is asked about."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.images: list[dict[str, NDArray[np.uint8]]] = []

    def predict_chunk(self, batch: Any) -> NDArray[np.float32]:
        self.images.append({k: v.copy() for k, v in batch.images.items()})
        return self.inner.predict_chunk(batch)


def _run(
    monkeypatch: pytest.MonkeyPatch, ckpt: Path, recorded: Recorded, policy: Any, *extra: str
) -> tuple[int, Any]:
    from teleop.commands import eval as ev
    from teleop.policy.manifest import GateManifest

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(ev, "_load_policy", lambda _c, _d: policy)
    code = ev.main(
        ["--checkpoint", str(ckpt), "--dataset", str(recorded.root), "--device", "cpu", *extra]
    )
    gate = ckpt / "gate.json"
    return code, GateManifest.load(gate) if gate.is_file() else None


@needs_models
def test_read_eval_episodes_from_recorded_dataset(recorded: Recorded) -> None:
    """Every hand column and every hand torque column comes through, by name."""
    from teleop.commands.eval import read_eval_episodes
    from teleop.policy.schema import STATE_NAMES
    from teleop.unoq.dmp import HAND_ACTUATORS

    (ep,) = read_eval_episodes(recorded.root)
    assert (ep.fault, ep.source, ep.success) == ("flap", "scripted", True)
    n = len(recorded.states)
    hand = [STATE_NAMES.index(f"hand/{a}") for a in HAND_ACTUATORS]
    tau = [STATE_NAMES.index(f"tau_hand/{a}") for a in HAND_ACTUATORS]
    wrist = [STATE_NAMES.index(f"wrist/{f}") for f in ("x", "y", "z", "qx", "qy", "qz", "qw")]
    assert ep.hand.shape == (n, len(HAND_ACTUATORS))
    np.testing.assert_allclose(ep.hand, recorded.states[:, hand], rtol=0, atol=1e-6)
    np.testing.assert_allclose(
        ep.tau_hand_inf, np.abs(recorded.states[:, tau]).max(axis=1), rtol=0, atol=1e-6
    )
    np.testing.assert_allclose(ep.wrist_latch, recorded.states[:, wrist], rtol=0, atol=1e-6)


@needs_models
def test_eval_end_to_end_on_recorded_dataset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, recorded: Recorded
) -> None:
    """Real dataset, DMP fit and camera mirror; only the policy is a replay stub."""
    from teleop.policy.act_runtime import ReplayPolicy
    from teleop.policy.schema import CAMERAS, IMAGE_HW

    spy = _ImageSpy(ReplayPolicy(recorded.actions))
    code, gate = _run(
        monkeypatch,
        _write_checkpoint(tmp_path / "ckpt"),
        recorded,
        spy,
        "--faults",
        "flap",
        "--rollouts",
        "2",
        "--bar",
        "1",
    )
    assert gate is not None and code in (0, 1)
    r = gate.results["flap"]
    assert r.rollouts == 2 and r.note == ""
    assert r.dmp_passed_rollouts is not None  # the DMP was fitted from the recorded episode
    assert spy.images, "the policy was never queried"
    for cam in CAMERAS:
        img = spy.images[0][cam]
        assert img.shape == (1, *IMAGE_HW, 3)
        assert int(img.max()) > int(img.min())  # a real render, not a blank frame


@needs_models
def test_eval_is_reproducible(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, recorded: Recorded
) -> None:
    """Same checkpoint, dataset and seed: the same gate results, run after run."""
    from teleop.policy.act_runtime import ReplayPolicy

    runs = []
    for i in range(2):
        _, gate = _run(
            monkeypatch,
            _write_checkpoint(tmp_path / f"ckpt{i}"),
            recorded,
            ReplayPolicy(recorded.actions),
            "--faults",
            "flap",
            "--rollouts",
            "2",
            "--bar",
            "1",
        )
        assert gate is not None
        r = gate.results["flap"]
        runs.append((r.passed_rollouts, r.mean_time_to_clear_s, r.max_tau_hand_inf))
    assert runs[0] == runs[1]


@needs_models
def test_mirror_keeps_top_camera_on_the_latch() -> None:
    """A gate rollout moves the latch (up to 8 cm and 30 deg); the top camera moves with it."""
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
            rel = np.asarray(d.xpos[body]) - np.asarray(d.cam_xpos[top])
            return rot.T @ rel

        mj.mj_forward(src, sd)
        nominal = latch_in_camera()
        yaw = math.radians(30.0)
        src.body_pos[latch] += (0.08, 0.06, 0.0)
        w, x, y, z = (float(v) for v in src.body_quat[latch])
        dq = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
        src.body_quat[latch] = (
            dq[0] * w - dq[3] * z,
            dq[0] * x - dq[3] * y,
            dq[0] * y + dq[3] * x,
            dq[0] * z + dq[3] * w,
        )
        mj.mj_forward(src, sd)
        moved = latch_in_camera()
    finally:
        mirror.close()
    np.testing.assert_allclose(moved, nominal, atol=1e-3)
