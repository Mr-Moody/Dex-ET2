"""Sprung slide-bolt latch physics, hand MJCF specs, 4-bay scene, DLS diff-IK, and tau_ext."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import mujoco as _mujoco  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
import pytest

from conftest import models_present, needs_models
from teleop.atc.types import BOLT_CLEARED_M, LatchPose
from teleop.sim.bays import (
    BACKGROUND_BAYS,
    BAY2_LATCH,
    CANONICAL_LATCH,
    build_four_bay_scene,
    build_single_bay_scene,
    latch_world_pose,
    reset_robot_home,
    wrist_pose_over_latch,
)
from teleop.sim.hands import actuator_names, joint_names, load_hand
from teleop.sim.ik import PandaDiffIK
from teleop.sim.latch import LatchController, add_latch_to_spec
from teleop.sim.torques import ExternalTorqueEstimator
from teleop.types import OPENXR_JOINTS, SIDES

mj: Any = _mujoco

HZ = 600
CTRL_EVERY = 10  # 60 Hz control on 600 Hz physics
FINGER_SPEED = 0.05  # m/s nominal bolt speed


@dataclass
class LatchRig:
    model: Any
    data: Any
    ctl: LatchController
    bolt_body: int
    bolt_dof: int
    flap_qadr: int
    flap_dof: int
    slide_dir: np.ndarray

    def flap_deg(self) -> float:
        return math.degrees(float(self.data.qpos[self.flap_qadr]))

    def run(
        self, seconds: float, push_n: float = 0.0, thumb_deg: float | None = None
    ) -> Iterator[None]:
        d = self.data
        finger = self.ctl.bolt_x()
        for _ in range(int(seconds * HZ)):
            if push_n > 0.0:
                finger += FINGER_SPEED / HZ
                lag = finger - self.ctl.bolt_x()
                force = min(max(2000.0 * lag - 20.0 * float(d.qvel[self.bolt_dof]), 0.0), push_n)
            else:
                force = 0.0
            d.xfrc_applied[self.bolt_body, :3] = self.slide_dir * force
            if thumb_deg is None:
                d.qfrc_applied[self.flap_dof] = 0.0
            else:
                err = float(d.qpos[self.flap_qadr]) - math.radians(thumb_deg)
                d.qfrc_applied[self.flap_dof] = -40.0 * err - 0.05 * float(d.qvel[self.flap_dof])
            mj.mj_step(self.model, d)
            yield

    def step(self, seconds: float, push_n: float = 0.0, thumb_deg: float | None = None) -> None:
        for _ in self.run(seconds, push_n, thumb_deg):
            pass


@pytest.fixture
def rig() -> LatchRig:
    spec: Any = mj.MjSpec()
    spec.compiler.degree = False
    spec.option.timestep = 1.0 / HZ
    spec.option.integrator = mj.mjtIntegrator.mjINT_IMPLICITFAST
    add_latch_to_spec(spec, "bay1", LatchPose(position=(0.3, -0.2, 0.02), yaw_rad=0.6))
    model = spec.compile()
    data = mj.MjData(model)
    flap = model.joint("bay1/flap_hinge")
    mj.mj_forward(model, data)
    xmat = np.asarray(data.xmat[model.body("bay1/latch_base").id], dtype=float).reshape(3, 3)
    return LatchRig(
        model=model,
        data=data,
        ctl=LatchController(model, data, "bay1"),
        bolt_body=int(model.body("bay1/slide_bolt").id),
        bolt_dof=int(model.joint("bay1/bolt_slide").dofadr[0]),
        flap_qadr=int(flap.qposadr[0]),
        flap_dof=int(flap.dofadr[0]),
        slide_dir=xmat[:, 0].copy(),
    )


@pytest.mark.parametrize("hand_name", ["inspire", "leap"])
def test_hand_specs_map_openxr_joints_and_compile(hand_name: str) -> None:
    spec = load_hand(hand_name)
    for fm in spec.flex:
        assert fm.axis in ("x", "y", "z") and fm.joints
        for j in fm.joints:
            assert j in OPENXR_JOINTS
    if hand_name == "leap":
        assert len(spec.flex) == 16 == len({fm.actuator for fm in spec.flex})
    if not all(p.is_file() for p in spec.mjcf.values()):
        pytest.skip("hand MJCFs missing; run `uv run python scripts/fetch_models.py`")
    for side in SIDES:
        model = mj.MjModel.from_xml_path(str(spec.mjcf[side]))
        names = actuator_names(spec, side)
        assert len(names) == model.nu and len(joint_names(spec, side)) == model.njnt
        for fm in spec.flex:
            assert fm.actuator in names


def test_latch_collision_boxes_and_visual_meshes_preserve_physics(rig: LatchRig) -> None:
    m = rig.model
    bodies = {str(m.body(i).name) for i in range(int(m.nbody))} - {"world"}
    assert bodies == {"bay1/latch_base", "bay1/slide_bolt", "bay1/stopper_flap"}
    for g in range(int(m.ngeom)):
        if m.geom_contype[g] or m.geom_conaffinity[g]:
            assert m.geom_type[g] == mj.mjtGeom.mjGEOM_BOX
    hinge = m.joint("bay1/flap_hinge")
    assert float(m.qpos_spring[int(hinge.qposadr[0])]) == pytest.approx(math.radians(35.0))
    assert float(m.jnt_stiffness[int(hinge.id)]) == pytest.approx(0.45)
    assert np.allclose(m.jnt_range[int(m.joint("bay1/bolt_slide").id)], [0.0, 0.06])


def test_routine_flap_flat_allows_one_finger_bolt_slide(rig: LatchRig) -> None:
    rig.ctl.reset()
    rig.step(0.3)
    assert abs(rig.flap_deg()) < 2.0
    rig.step(1.5, push_n=4.0)
    assert rig.ctl.is_cleared() and rig.ctl.bolt_x() >= BOLT_CLEARED_M
    assert abs(rig.flap_deg()) < 2.0


def test_sprung_flap_blocks_bolt_around_35mm_unless_depressed(rig: LatchRig) -> None:
    rig.ctl.reset(flap_jammed=True)
    rig.step(0.3)
    assert rig.flap_deg() > 30.0
    rig.step(1.5, push_n=30.0)
    assert 0.030 <= rig.ctl.bolt_x() <= 0.040 and not rig.ctl.is_cleared()

    rig.ctl.reset(flap_jammed=True)
    rig.step(0.5, thumb_deg=6.0)
    rig.step(1.5, push_n=4.0, thumb_deg=6.0)
    assert rig.ctl.is_cleared()


def test_arm_flap_fault_is_one_way_latched_until_reset(rig: LatchRig) -> None:
    ctl = rig.ctl
    ctl.reset()
    rig.step(0.3)
    assert ctl.arm_flap_fault() is True
    rig.step(0.5)
    assert rig.flap_deg() > 30.0
    assert ctl.arm_flap_fault() is False
    ctl.reset()
    rig.step(0.5)
    assert not ctl.flap_fault_armed and abs(rig.flap_deg()) < 2.0
    assert ctl.arm_flap_fault() is True


def test_pinned_bolt_fault_locks_bolt_at_zero(rig: LatchRig) -> None:
    rig.ctl.reset(bolt_pinned=True)
    peak = max(abs(rig.ctl.bolt_x()) for _ in rig.run(1.5, push_n=30.0))
    assert peak < 0.002 and not rig.ctl.is_cleared()
    rig.ctl.reset()
    rig.step(1.5, push_n=4.0)
    assert rig.ctl.is_cleared()


@pytest.fixture(scope="module")
def single_bay() -> Any:
    if not models_present():
        pytest.skip("Panda/Inspire MJCFs missing; run `uv run python scripts/fetch_models.py`")
    return build_single_bay_scene(CANONICAL_LATCH)


def _drive(
    model: Any,
    data: Any,
    ik: PandaDiffIK,
    target: Callable[[], Any],
    seconds: float,
    each_tick: Callable[[], None] | None = None,
) -> None:
    dt = CTRL_EVERY / HZ
    for i in range(int(seconds * HZ)):
        if i % CTRL_EVERY == 0:
            ik.apply(data, ik.solve(data, target(), dt))
            if each_tick is not None:
                each_tick()
        mj.mj_step(model, data)


@needs_models
@pytest.mark.parametrize(
    ("dx", "dy", "dyaw_deg"),
    [(0.0, 0.0, 0.0), (0.08, 0.09, 35.0), (-0.08, -0.09, -35.0)],
)
def test_panda_dls_ik_tracks_rotated_and_shifted_latch_pose(
    single_bay: Any, dx: float, dy: float, dyaw_deg: float
) -> None:
    model = single_bay
    data = mj.MjData(model)
    ik = PandaDiffIK(model, "bay1")
    local = CANONICAL_LATCH.offset(dx, dy, math.radians(dyaw_deg))
    target = wrist_pose_over_latch(latch_world_pose(1, local), height=0.20)

    reset_robot_home(model, data, 1)
    ik.reset()
    _drive(model, data, ik, lambda: target, seconds=3.0)
    err = ik.error(data, target)
    assert np.linalg.norm(err[:3]) < 5e-3
    assert math.degrees(float(np.linalg.norm(err[3:]))) < 2.0
    q = np.asarray(data.qpos, dtype=float)[ik.qpos_adr]
    assert np.all(q >= ik.q_min - 1e-3) and np.all(q <= ik.q_max + 1e-3)


@needs_models
def test_external_torque_spike_detected_on_flap_collision(single_bay: Any) -> None:
    model = single_bay
    data = mj.MjData(model)
    ik = PandaDiffIK(model, "bay1")
    est = ExternalTorqueEstimator(model, "bay1")
    ctl = LatchController(model, data, "bay1")
    world_latch = latch_world_pose(1, CANONICAL_LATCH)
    flap_geom = int(model.geom("bay1/flap_paddle").id)

    reset_robot_home(model, data, 1)
    ctl.reset(flap_jammed=True)
    ik.reset()
    hover = wrist_pose_over_latch(world_latch, height=0.20, along_x=0.035)
    _drive(model, data, ik, lambda: hover, seconds=2.5)
    free: list[float] = []
    _drive(model, data, ik, lambda: hover, 0.5, lambda: free.append(est.estimate_inf(data)))
    baseline = max(free)
    assert baseline < 0.5

    press = wrist_pose_over_latch(world_latch, height=0.06, along_x=0.035)
    pressed: list[float] = []
    touched = False

    def tick() -> None:
        nonlocal touched
        pressed.append(est.estimate_inf(data))
        for c in data.contact[: int(data.ncon)]:
            if flap_geom in (int(c.geom1), int(c.geom2)):
                touched = True

    _drive(model, data, ik, lambda: press, 1.5, tick)
    assert touched and max(pressed) > max(2.0, 10.0 * max(baseline, 0.05))


@needs_models
def test_four_bay_scene_has_two_active_and_two_background_bays() -> None:
    model = build_four_bay_scene(CANONICAL_LATCH, BAY2_LATCH)
    data = mj.MjData(model)
    for bay in (1, 2, 3, 4):
        model.site(f"bay{bay}/wrist_site")
        reset_robot_home(model, data, bay)
    assert int(model.nu) == 4 * (7 + 6)

    background = {f"bay{b}/" for b in BACKGROUND_BAYS}
    for g in range(int(model.ngeom)):
        body = str(model.body(int(model.geom_bodyid[g])).name)
        if body[:5] in background:
            assert model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0, body

    mj.mj_forward(model, data)
    w2 = latch_world_pose(2, BAY2_LATCH)
    site = model.site("bay2/latch_frame").id
    assert np.allclose(data.site_xpos[site], w2.position, atol=1e-9)
    x_axis = np.asarray(data.site_xmat[site], dtype=float).reshape(3, 3)[:, 0]
    assert math.degrees(math.atan2(x_axis[1], x_axis[0])) == pytest.approx(35.0)
