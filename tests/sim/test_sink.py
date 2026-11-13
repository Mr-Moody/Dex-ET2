import mujoco

# pyright: reportUnknownMemberType=false, reportAttributeAccessIssue=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
import numpy as np

from teleop.sim.sink import SimSink
from teleop.types import JointTargets, Pose, WristTarget


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
    pose = Pose(orientation=(0.0, 0.0, 0.0, 1.0), position=(0.1, 0.2, 0.3))
    sink.apply([], [WristTarget("left", pose)])
    mid = model.body("left/mocap").mocapid[0]
    assert list(data.mocap_quat[mid]) == [1, 0, 0, 0]
    assert np.allclose(data.mocap_pos[mid], (0.1, 0.2, 0.3))
    st = sink.state()
    assert st.wrist["left"].orientation == (0.0, 0.0, 0.0, 1.0)
    assert all(k.startswith(("left/", "right/")) for k in st.qpos)


def test_substeps(scene):
    model, data, _ = scene
    sink = SimSink(model, data, 600, 60)
    t0 = data.time
    sink.apply([], [])
    assert abs((data.time - t0) - 1 / 60) < 1e-9
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left/mocap") >= 0
