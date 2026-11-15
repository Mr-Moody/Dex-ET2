"""ACT recovery policy schema, gate manifest, batched inference worker, and camera rendering."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import mujoco as _mujoco  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
import pytest
from numpy.typing import NDArray

from conftest import needs_models
from teleop.atc.types import StallSignature
from teleop.policy.act_runtime import (
    WRIST_POS,
    InferenceWorker,
    ObsBatch,
    ReplayPolicy,
    load_act_policy,
    pretrained_dir,
)
from teleop.policy.manifest import FaultResult, GateManifest, ManifestError, covers
from teleop.policy.schema import (
    ACTION_NAMES,
    CAMERAS,
    FPS,
    IMAGE_HW,
    SCHEMA_VERSION,
    STATE_NAMES,
    EpisodeMeta,
    action_from_command,
    camera_name,
    command_from_action,
    lerobot_features,
    observe,
)
from teleop.types import Pose
from teleop.unoq.dmp import HAND_ACTUATORS, compose_pose, mat_to_quat, relative_pose, so3_exp

mj: Any = _mujoco
H, W = IMAGE_HW
CHUNK = 30


def _random_pose(rng: np.random.Generator) -> Pose:
    rot = so3_exp(rng.normal(size=3))
    pos = rng.normal(size=3)
    return Pose(
        orientation=mat_to_quat(rot), position=(float(pos[0]), float(pos[1]), float(pos[2]))
    )


def _quat_close(a: Any, b: Any, tol: float = 1e-9) -> bool:
    qa, qb = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return bool(min(np.max(np.abs(qa - qb)), np.max(np.abs(qa + qb))) < tol)


def _wait(pred: Any, timeout_s: float = 5.0) -> None:
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        if pred():
            return
        time.sleep(0.002)
    raise AssertionError("condition never reached")


# ------------------------------------------------------------------ policy schema


def test_schema_names_and_latch_relative_action_roundtrip() -> None:
    assert len(STATE_NAMES) == 29 and len(set(STATE_NAMES)) == 29
    assert len(ACTION_NAMES) == 12 and len(set(ACTION_NAMES)) == 12
    assert (SCHEMA_VERSION, FPS, IMAGE_HW, CAMERAS) == (1, 30, (128, 128), ("wrist", "top"))
    assert camera_name(2, "wrist") == "bay2/wrist_cam"
    assert camera_name(1, "top").startswith("bay1/")
    with pytest.raises(ValueError):
        camera_name(1, "side")

    feats = lerobot_features()
    assert feats["observation.state"]["shape"] == (29,)
    assert feats["action"]["shape"] == (12,)
    assert feats["observation.images.wrist"]["shape"] == (128, 128, 3)

    meta = EpisodeMeta(
        source="glove",
        fault="flap",
        bay=1,
        signature_id="Jam_Sig_1",
        signature=(0.035, 1.2, 2.4, 0.3),
        latch_pose={"x": 0.5, "yaw": 0.0},
        success=True,
        bolt_x_final=0.054,
        max_tau_hand_inf=2.1,
        max_arm_ratio=0.42,
        aborted_reason=None,
    )
    assert EpisodeMeta.from_json(meta.to_json()) == meta

    rng = np.random.default_rng(7)
    fingers = {a: float(i) * 0.1 for i, a in enumerate(HAND_ACTUATORS)}
    for _ in range(15):
        wfl, wrist, other = _random_pose(rng), _random_pose(rng), _random_pose(rng)
        action = action_from_command(wfl, wrist, fingers)
        back, back_fingers = command_from_action(wfl, action)
        assert back.position is not None and wrist.position is not None
        assert np.allclose(back.position, wrist.position, atol=1e-9)
        assert _quat_close(back.orientation, wrist.orientation)
        assert back_fingers == pytest.approx(fingers)
        moved = compose_pose(other, relative_pose(wfl, wrist))
        assert np.allclose(action, action_from_command(other, moved, fingers))


@needs_models
def test_observe_reads_state_by_joint_name() -> None:
    from teleop.atc.validator import BayRig
    from teleop.sim.bays import build_four_bay_scene, reset_robot_home

    model = build_four_bay_scene(include_background_bays=False)
    data = mj.MjData(model)
    for b in (1, 2):
        reset_robot_home(model, data, b)
    data.qpos[model.joint("bay1/joint4").qposadr[0]] -= 0.1
    mj.mj_forward(model, data)
    rig = BayRig(model, data, 1)
    obs = observe(rig)
    assert obs.shape == (29,) and all(math.isfinite(float(v)) for v in obs)
    assert obs[STATE_NAMES.index("arm/joint4")] == pytest.approx(
        float(data.qpos[model.joint("bay1/joint4").qposadr[0]])
    )
    rel = relative_pose(rig.world_from_latch(), rig.wrist_pose())
    assert rel.position is not None
    assert np.allclose(obs[13:16], rel.position) and _quat_close(obs[16:20], rel.orientation)
    assert np.allclose(obs[-3:], rig.tau_ext()[4:7])


# ------------------------------------------------------------------ gate manifest


def test_gate_manifest_roundtrip_schema_and_coverage(tmp_path: Path) -> None:
    flap_protos = [(0.035, 1.2, 2.4, 0.3), (0.036, 1.3, 2.5, 0.31), (0.034, 1.1, 2.3, 0.29)]
    pin_protos = [(0.012, 4.0, 5.5, 1.1), (0.013, 4.1, 5.4, 1.0)]
    manifest = GateManifest(
        schema_version=SCHEMA_VERSION,
        checkpoint_sha256="ab" * 32,
        bar=(9, 10),
        results={
            "flap": FaultResult("flap", 10, 10, 1.8, 2.2, 3),
            "pin": FaultResult("pin", 10, 4, None, 6.1, 0),
        },
        prototypes={"flap": flap_protos, "pin": pin_protos},
        inference_p95_ms=12.5,
    )
    path = tmp_path / "gate.json"
    manifest.save(path)
    assert GateManifest.load(path) == manifest
    assert manifest.passed_faults() == ("flap",)
    assert covers(manifest, StallSignature(*flap_protos[0])) == "flap"
    assert covers(manifest, StallSignature(0.035, 11.2, 12.4, 5.3)) is None
    assert covers(manifest, StallSignature(*pin_protos[0])) is None  # failed bar -> ignored

    with pytest.raises(ManifestError, match="gate.json"):
        GateManifest.load(tmp_path / "missing" / "gate.json")
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["schema_version"] = SCHEMA_VERSION + 1
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ManifestError, match="schema"):
        GateManifest.load(path)


# ------------------------------------------------------------------ inference worker & runtime


class _CountingPolicy:
    def __init__(self, sleep_s: float = 0.0, fail: bool = False, bad_dim: bool = False) -> None:
        self.sleep_s = sleep_s
        self.fail = fail
        self.bad_dim = bad_dim
        self.batches: list[int] = []

    def predict_chunk(self, batch: ObsBatch) -> NDArray[np.float32]:
        self.batches.append(batch.size)
        if self.sleep_s:
            time.sleep(self.sleep_s)
        if self.fail:
            raise RuntimeError("cuda device lost")
        dim = 7 if self.bad_dim else len(ACTION_NAMES)
        out = np.zeros((batch.size, CHUNK, dim), dtype=np.float32)
        out[:, :, 0] = batch.state[:, WRIST_POS.start, None]
        return out


def _zero_imgs() -> dict[str, NDArray[np.uint8]]:
    return {c: np.zeros((H, W, 3), dtype=np.uint8) for c in CAMERAS}


def _wrist_state(x: float = 0.0) -> NDArray[np.float64]:
    s = np.zeros(len(STATE_NAMES))
    s[WRIST_POS.start] = x
    return s


def test_inference_worker_batches_latest_wins_and_health_trips() -> None:
    # 2-bay batching + latest-wins overwrite per bay.
    policy = _CountingPolicy()
    worker = InferenceWorker(policy, autostart=False)
    try:
        worker.submit(1, _wrist_state(0.05), _zero_imgs(), tick=0)
        worker.submit(1, _wrist_state(0.15), _zero_imgs(), tick=3)
        worker.submit(2, _wrist_state(0.25), _zero_imgs(), tick=5)
        worker.start()
        _wait(lambda: worker.latest(1) is not None and worker.latest(2) is not None)
        assert policy.batches == [2]
        one, two = worker.latest(1), worker.latest(2)
        assert one is not None and two is not None
        assert one[0] == 3 and one[1][0, 0] == pytest.approx(0.15)
        assert two[0] == 5 and two[1][0, 0] == pytest.approx(0.25)
        worker.forget(1)
        assert worker.latest(1) is None and worker.healthy()
    finally:
        worker.close()

    # Single-call timeout, exception, and bad chunk shape each trip worker unhealthy cleanly.
    for kw, expected in (
        ({"policy": _CountingPolicy(sleep_s=0.06), "timeout_ms": 40.0}, "timeout"),
        ({"policy": _CountingPolicy(fail=True)}, "cuda device lost"),
        ({"policy": _CountingPolicy(bad_dim=True)}, "shape"),
    ):
        w = InferenceWorker(**kw)  # type: ignore[arg-type]
        try:
            w.submit(1, _wrist_state(), _zero_imgs(), tick=0)
            _wait(lambda w=w: not w.healthy())
            assert expected in w.unhealthy_reason
        finally:
            w.close()


def test_replay_policy_and_checkpoint_schema_validation(tmp_path: Path) -> None:
    line = np.zeros((30, len(ACTION_NAMES)), dtype=np.float32)
    line[8:, 0] = np.linspace(0.0, 0.044, 22)
    line[:, 6] = np.arange(30)
    stub = ReplayPolicy(line, chunk=6)
    assert stub.start_index(np.zeros(3)) >= 8
    batch = ObsBatch(
        np.stack([_wrist_state(0.0), _wrist_state(0.10)]).astype(np.float32),
        {c: np.zeros((2, H, W, 3), dtype=np.uint8) for c in CAMERAS},
    )
    out = stub.predict_chunk(batch)
    assert out.shape == (2, 6, len(ACTION_NAMES)) and out[1, -1, 6] == 29

    run = tmp_path / "run"
    for step in ("000500", "001000"):
        d = run / "checkpoints" / step / "pretrained_model"
        d.mkdir(parents=True)
        (d / "config.json").write_text("{}", encoding="utf-8")
    assert pretrained_dir(run) == run / "checkpoints" / "001000" / "pretrained_model"

    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "config.json").write_text(
        json.dumps(
            {
                "input_features": {"observation.state": {"type": "STATE", "shape": [20]}},
                "output_features": {"action": {"type": "ACTION", "shape": [12]}},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ManifestError, match="observation.state"):
        load_act_policy(bad, "cpu")


# ------------------------------------------------------------------ camera rendering


@needs_models
def test_bay_camera_renderer_shapes_and_near_clip(offscreen_render_ok: None) -> None:
    from teleop.atc.coordinator import ROUTINE
    from teleop.atc.validator import BayRig, step_physics
    from teleop.policy.cameras import BayCameraRenderer, policy_clip
    from teleop.sim.bays import build_four_bay_scene, reset_robot_home

    model: Any = build_four_bay_scene(include_background_bays=False)
    data: Any = mj.MjData(model)
    reset_robot_home(model, data, 1)
    rig = BayRig(model, data, 1)
    start = compose_pose(rig.world_from_latch(), ROUTINE.pose_at(0.0))
    for _ in range(180):
        rig.command(start, ROUTINE.fingers)
        step_physics(model, data)

    znear_before = float(model.vis.map.znear)
    renderer = BayCameraRenderer(model)
    try:
        images = renderer.render(data, 1)
    finally:
        renderer.close()
    assert float(model.vis.map.znear) == znear_before
    assert set(images) == set(CAMERAS)
    for img in images.values():
        assert img.shape == (*IMAGE_HW, 3) and img.dtype == np.uint8
    assert int(images["wrist"].max()) > int(images["wrist"].min())

    seg_renderer = mj.Renderer(model, 128, 128)
    try:
        seg_renderer.enable_segmentation_rendering()
        cam = camera_name(1, "wrist")
        with policy_clip(model, cam):
            seg_renderer.update_scene(data, cam)
            seg = seg_renderer.render()
    finally:
        seg_renderer.close()
    ids, types = seg[..., 0], seg[..., 1]
    geoms = ids[(types == int(mj.mjtObj.mjOBJ_GEOM)) & (ids >= 0)]
    counts = np.bincount(model.geom_bodyid[geoms], minlength=model.nbody)
    seen = {str(model.body(int(b)).name) for b in np.flatnonzero(counts >= 20)}
    assert "bay1/index_intermediate" in seen and "bay1/slide_bolt" in seen
