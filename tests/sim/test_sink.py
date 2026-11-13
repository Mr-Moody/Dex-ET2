import math
from typing import Any

import mujoco as _mujoco  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
import pytest

from teleop.sim.sink import SimSink
from teleop.types import JointTargets, Pose, WristTarget

mj: Any = _mujoco


def _first_actuator(model, side="right"):
    for i in range(model.nu):
        name = str(model.actuator(i).name)
        if name.startswith(f"{side}/"):
            return i, name.split("/", 1)[1]
    raise AssertionError("no actuator")


def test_sink_maps_by_name_and_clips(scene):
    model, data, _ = scene
    sink = SimSink(model, data, 600, 60)
    i, name = _first_actuator(model)
    sink.apply([JointTargets("right", 0, {name: 1e6})], [])
    assert data.ctrl[i] == model.actuator_ctrlrange[i][1]


def test_sink_ignores_unknown_actuator(scene):
    model, data, _ = scene
    sink = SimSink(model, data, 600, 60)
    sink.apply([JointTargets("right", 0, {"nope": 1.0})], [])
    assert sink.unknown_names == 1
    assert np.all(data.ctrl == 0)


def test_wrist_mocap_quat_converted(scene):
    model, data, _ = scene
    sink = SimSink(model, data, 600, 60)
    # asymmetric normalised xyzw; wxyz must be a distinct permutation
    quat = (0.1, 0.2, 0.3, math.sqrt(1 - 0.01 - 0.04 - 0.09))
    pose = Pose(orientation=quat, position=(0.1, 0.2, 0.3))
    sink.apply([], [WristTarget("left", pose)])
    mid = model.body("left/mocap").mocapid[0]
    assert np.allclose(data.mocap_quat[mid], (quat[3], quat[0], quat[1], quat[2]))
    assert np.allclose(data.mocap_pos[mid], (0.1, 0.2, 0.3))
    st = sink.state()
    assert np.allclose(st.wrist["left"].orientation, quat)
    assert all(k.startswith(("left/", "right/")) for k in st.qpos)


def test_substeps(scene):
    model, data, _ = scene
    sink = SimSink(model, data, 600, 60)
    t0 = data.time
    sink.apply([], [])
    assert abs((data.time - t0) - 1 / 60) < 1e-9
    assert mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "left/mocap") >= 0


def test_unlimited_actuator_not_clipped_to_zero(scene):
    model, data, _ = scene
    i, name = _first_actuator(model)
    model.actuator_ctrllimited[i] = 0
    sink = SimSink(model, data, 600, 60)
    sink.apply([JointTargets("right", 0, {name: 0.7})], [])
    assert data.ctrl[i] == 0.7


def test_nonfinite_targets_dropped(scene):
    model, data, _ = scene
    sink = SimSink(model, data, 600, 60)
    i, name = _first_actuator(model)
    sink.apply([JointTargets("right", 0, {name: float("nan")})], [])
    assert sink.nonfinite == 1 and data.ctrl[i] == 0


def test_bad_rate_ratio_rejected(scene):
    model, data, _ = scene
    with pytest.raises(ValueError):
        SimSink(model, data, 500, 60)
