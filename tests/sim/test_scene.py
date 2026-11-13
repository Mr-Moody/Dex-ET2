import mujoco

# pyright: reportUnknownMemberType=false, reportAttributeAccessIssue=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
import numpy as np

from teleop.sim.hands import HandSpec


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
        mujoco.mj_step(model, data)
    assert np.all(np.isfinite(data.qpos))
    assert abs(data.qpos[model.joint("cube_free").qposadr[0] + 2] - 0.02) < 2e-3
