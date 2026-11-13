from typing import Any

import mujoco as _mujoco  # pyright: ignore[reportMissingTypeStubs]

from teleop.sim.tasks.lift_cube import LiftCube

mj: Any = _mujoco


def test_lift_cube_success_when_lifted(scene):
    model, data, task = scene
    adr = model.joint("cube_free").qposadr[0]
    assert not task.score(model, data).success
    for _ in range(40):  # 40 * 1/60 s > 0.5 s of sim time
        data.qpos[adr + 2] = 0.2
        data.qvel[:] = 0
        for _ in range(10):
            data.qpos[adr + 2] = 0.2
            mj.mj_step(model, data)
        res = task.score(model, data)
    assert res.success
    assert res.score >= 19.0


def test_reset_randomizes_cube_by_seed(scene):
    model, data, task = scene
    adr = model.joint("cube_free").qposadr[0]
    task.reset(model, data, 1)
    a = data.qpos[adr : adr + 2].copy()
    task.reset(model, data, 1)
    assert (data.qpos[adr : adr + 2] == a).all()
    assert abs(a[0]) <= 0.03 and abs(a[1] - 0.1) <= 0.03
    assert isinstance(task, LiftCube)


def test_reset_clears_latched_success(scene):
    model, data, task = scene
    adr = model.joint("cube_free").qposadr[0]
    for _ in range(40):
        data.qpos[adr + 2] = 0.2
        for _ in range(10):
            data.qpos[adr + 2] = 0.2
            mj.mj_step(model, data)
        res = task.score(model, data)
    assert res.success
    task.reset(model, data, 0)
    res = task.score(model, data)
    assert not res.success
    assert res.score < 5.0
