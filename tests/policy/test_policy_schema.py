"""T-190: ACT policy schema (state/action names by name, latch-frame actions, episode meta)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

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
from teleop.sim.bays import PANDA_XML
from teleop.sim.hands import load_hand
from teleop.types import Pose
from teleop.unoq.dmp import HAND_ACTUATORS, mat_to_quat, relative_pose, so3_exp


def _models_present() -> bool:
    try:
        right = load_hand("inspire").mjcf["right"]
    except FileNotFoundError:
        return False
    return PANDA_XML.is_file() and right.is_file()


needs_models = pytest.mark.skipif(
    not _models_present(), reason="Panda/Inspire MJCFs missing; run scripts/fetch_models.py"
)


def _pose(rng: np.random.Generator) -> Pose:
    rot = so3_exp(rng.normal(size=3))
    pos = rng.normal(size=3)
    return Pose(
        orientation=mat_to_quat(rot), position=(float(pos[0]), float(pos[1]), float(pos[2]))
    )


def _quat_close(a: Any, b: Any, tol: float) -> bool:
    qa, qb = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return bool(min(np.max(np.abs(qa - qb)), np.max(np.abs(qa + qb))) < tol)


def test_names_lengths() -> None:
    assert len(STATE_NAMES) == 29
    assert len(ACTION_NAMES) == 12
    assert len(set(STATE_NAMES)) == 29 and len(set(ACTION_NAMES)) == 12
    assert STATE_NAMES[:7] == tuple(f"arm/joint{i}" for i in range(1, 8))
    assert STATE_NAMES[7:13] == tuple(f"hand/{a}" for a in HAND_ACTUATORS)
    assert STATE_NAMES[-3:] == ("tau_wrist/joint5", "tau_wrist/joint6", "tau_wrist/joint7")
    assert ACTION_NAMES[:6] == ("wrist/x", "wrist/y", "wrist/z", "wrist/rx", "wrist/ry", "wrist/rz")
    assert (SCHEMA_VERSION, FPS, IMAGE_HW, CAMERAS) == (1, 30, (128, 128), ("wrist", "top"))


def test_action_roundtrip() -> None:
    rng = np.random.default_rng(7)
    fingers = {a: float(i) * 0.1 for i, a in enumerate(HAND_ACTUATORS)}
    for _ in range(20):
        wfl, wrist = _pose(rng), _pose(rng)
        action = action_from_command(wfl, wrist, fingers)
        assert action.shape == (12,)
        back, back_fingers = command_from_action(wfl, action)
        assert back.position is not None and wrist.position is not None
        assert np.allclose(back.position, wrist.position, atol=1e-9)
        assert _quat_close(back.orientation, wrist.orientation, 1e-9)
        assert back_fingers == pytest.approx(fingers)


def test_action_is_latch_relative() -> None:
    """Moving latch and wrist together leaves the action unchanged (one policy, every bay)."""
    rng = np.random.default_rng(3)
    fingers = dict.fromkeys(HAND_ACTUATORS, 0.2)
    wfl, wrist = _pose(rng), _pose(rng)
    rel = relative_pose(wfl, wrist)
    other = _pose(rng)
    from teleop.unoq.dmp import compose_pose

    moved = compose_pose(other, rel)
    assert np.allclose(
        action_from_command(wfl, wrist, fingers), action_from_command(other, moved, fingers)
    )


def test_camera_names() -> None:
    assert camera_name(2, "wrist") == "bay2/wrist_cam"
    assert camera_name(1, "top").startswith("bay1/")
    with pytest.raises(ValueError):
        camera_name(1, "side")


def test_lerobot_features() -> None:
    feats = lerobot_features()
    assert feats["observation.state"]["shape"] == (29,)
    assert feats["observation.state"]["names"] == list(STATE_NAMES)
    assert feats["action"]["shape"] == (12,)
    assert feats["observation.images.wrist"]["shape"] == (128, 128, 3)
    assert feats["observation.images.top"]["dtype"] == "video"


def test_episode_meta_roundtrip() -> None:
    meta = EpisodeMeta(
        source="glove",
        fault="flap",
        bay=1,
        signature_id="Jam_Sig_1",
        signature=(0.035, 1.2, 2.4, 0.3),
        latch_pose={"x": 0.5, "y": 0.0, "z": 0.02, "yaw": 0.0},
        success=True,
        bolt_x_final=0.054,
        max_tau_hand_inf=2.1,
        max_arm_ratio=0.42,
        aborted_reason=None,
    )
    assert EpisodeMeta.from_json(meta.to_json()) == meta
    assert meta.to_json()["source"] == "glove"


@needs_models
def test_observe_by_name() -> None:
    import mujoco as _mujoco

    from teleop.atc.validator import BayRig
    from teleop.sim.bays import build_four_bay_scene, reset_robot_home

    model = build_four_bay_scene(include_background_bays=False)
    mj: Any = _mujoco
    data = mj.MjData(model)
    for b in (1, 2):
        reset_robot_home(model, data, b)
    data.qpos[model.joint("bay1/joint4").qposadr[0]] -= 0.1
    mj.mj_forward(model, data)
    rig = BayRig(model, data, 1)
    obs = observe(rig)
    assert obs.shape == (29,)
    assert obs[STATE_NAMES.index("arm/joint4")] == pytest.approx(
        float(data.qpos[model.joint("bay1/joint4").qposadr[0]])
    )
    assert np.allclose(obs[7:13], rig.hand_q())
    rel = relative_pose(rig.world_from_latch(), rig.wrist_pose())
    assert rel.position is not None
    assert np.allclose(obs[13:16], rel.position)
    assert _quat_close(obs[16:20], rel.orientation, 1e-9)
    tau = rig.tau_ext()
    assert np.allclose(obs[-3:], tau[4:7])
    assert all(math.isfinite(float(v)) for v in obs)
