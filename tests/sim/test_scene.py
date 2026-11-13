from typing import Any

import mujoco as _mujoco  # pyright: ignore[reportMissingTypeStubs]
import numpy as np

from teleop.sim.hands import HandSpec

mj: Any = _mujoco


def test_scene_has_both_hands_prefixed(scene, hand: HandSpec):
    model, _, _ = scene
    names = [str(model.actuator(i).name) for i in range(model.nu)]
    assert any(n.startswith("left/") for n in names)
    assert any(n.startswith("right/") for n in names)
    for side in ("left", "right"):
        assert model.body(f"{side}/mocap").mocapid[0] >= 0
        assert model.body(f"{side}/{hand.base_body}").id > 0
    assert model.camera("front").id >= 0


def test_scene_is_stable_at_home(scene):
    model, data, _ = scene
    for _ in range(300):
        mj.mj_step(model, data)
    assert np.all(np.isfinite(data.qpos))
    assert abs(data.qpos[model.joint("cube_free").qposadr[0] + 2] - 0.02) < 2e-3


def test_hand_base_tracks_mocap(scene, hand):
    model, data, _ = scene
    mid = model.body("right/mocap").mocapid[0]
    data.mocap_pos[mid] += (0.0, 0.05, 0.0)
    for _ in range(300):  # 0.5 s at 600 Hz
        mj.mj_step(model, data)
    base = data.body(f"right/{hand.base_body}").xpos
    assert np.linalg.norm(base - data.mocap_pos[mid]) < 0.005


def test_no_hand_contacts_at_home(scene):
    model, data, _ = scene
    for _ in range(300):
        mj.mj_step(model, data)
        for c in data.contact[: data.ncon]:
            for g in (c.geom1, c.geom2):
                body = model.body(model.geom_bodyid[g]).name
                assert not body.startswith(("left/", "right/")), body
