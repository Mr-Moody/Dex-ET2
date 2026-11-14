"""T-110: sprung slide-bolt latch, 4-bay Panda+LEAP scene, DLS diff-IK and tau_ext estimator.

Latch tests build a bare latch-only model (no third-party assets). The robot tests need the
Menagerie Panda and LEAP models and skip when they have not been fetched.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import mujoco as _mujoco  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
import pytest

from teleop.atc.types import BOLT_CLEARED_M, LatchPose
from teleop.sim.bays import (
    BACKGROUND_BAYS,
    BAY2_LATCH,
    CANONICAL_LATCH,
    PANDA_XML,
    build_four_bay_scene,
    build_single_bay_scene,
    latch_world_pose,
    reset_robot_home,
    wrist_pose_over_latch,
)
from teleop.sim.hands import load_hand
from teleop.sim.ik import PandaDiffIK
from teleop.sim.latch import LatchController, add_latch_to_spec
from teleop.sim.torques import ExternalTorqueEstimator

mj: Any = _mujoco

HZ = 600
CTRL_EVERY = 10  # 60 Hz control on 600 Hz physics
FINGER_SPEED = 0.05  # m/s, the spec's nominal bolt speed v_nom


# --------------------------------------------------------------------------- latch-only rig


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
        """Step physics with an index finger on the hook and optionally a thumb on the flap.

        The finger advances along latch +X at ``FINGER_SPEED`` and presses the hook with a
        spring force capped at ``push_n``, like a fingertip driven by a speed-limited arm.
        """
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
            else:  # a stiff thumb (PD on the paddle) holding the flap at thumb_deg
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
    # A yawed, offset latch so the tests exercise the latch frame, not just world X.
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


def test_latch_is_three_box_bodies(rig: LatchRig) -> None:
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
    assert abs(rig.flap_deg()) < 2.0  # flap_catch holds it flat against the spring
    rig.step(1.5, push_n=4.0)
    assert rig.ctl.is_cleared()
    assert rig.ctl.bolt_x() >= BOLT_CLEARED_M
    assert abs(rig.flap_deg()) < 2.0


def test_sprung_flap_blocks_bolt_around_35mm_unless_depressed(rig: LatchRig) -> None:
    rig.ctl.reset(flap_jammed=True)
    rig.step(0.3)
    assert rig.flap_deg() > 30.0
    rig.step(1.5, push_n=30.0)  # a hard shove does not get past a sprung flap
    assert 0.030 <= rig.ctl.bolt_x() <= 0.040
    assert not rig.ctl.is_cleared()

    # A thumb holding the paddle down (<= 8 deg) lets the index slide the bolt clear.
    rig.ctl.reset(flap_jammed=True)
    rig.step(0.5, thumb_deg=6.0)
    rig.step(1.5, push_n=4.0, thumb_deg=6.0)
    assert rig.ctl.is_cleared()


def test_arm_flap_fault_is_one_way_latched_until_reset(rig: LatchRig) -> None:
    ctl = rig.ctl
    ctl.reset()
    rig.step(0.3)
    assert not ctl.flap_fault_armed
    assert abs(rig.flap_deg()) < 2.0

    assert ctl.arm_flap_fault() is True
    rig.step(0.5)
    assert rig.flap_deg() > 30.0
    # Further rig toggles are ignored: the flap stays up, the fault stays armed.
    assert ctl.arm_flap_fault() is False
    assert ctl.arm_flap_fault() is False
    rig.step(0.3)
    assert ctl.flap_fault_armed
    assert rig.flap_deg() > 30.0

    ctl.reset()
    rig.step(0.5)
    assert not ctl.flap_fault_armed
    assert abs(rig.flap_deg()) < 2.0
    assert ctl.arm_flap_fault() is True  # re-armable once per cycle


def test_pinned_bolt_fault_locks_bolt_at_zero(rig: LatchRig) -> None:
    rig.ctl.reset(bolt_pinned=True)
    assert rig.ctl.bolt_pinned
    peak = 0.0
    for _ in rig.run(1.5, push_n=30.0):
        peak = max(peak, abs(rig.ctl.bolt_x()))
    assert peak < 0.002
    assert not rig.ctl.is_cleared()

    rig.ctl.reset()  # pin released on the next cycle
    rig.step(1.5, push_n=4.0)
    assert rig.ctl.is_cleared()


# --------------------------------------------------------------------------- Panda + LEAP


def _models_present() -> bool:
    try:
        right = load_hand("inspire").mjcf["right"]
    except FileNotFoundError:
        return False
    return PANDA_XML.is_file() and right.is_file()


needs_models = pytest.mark.skipif(
    not _models_present(), reason="Panda/Inspire MJCFs missing; run scripts/fetch_models.py"
)


@pytest.fixture(scope="module")
def single_bay() -> Any:
    if not _models_present():
        pytest.skip("Panda/Inspire MJCFs missing; run `uv run python scripts/fetch_models.py`")
    return build_single_bay_scene(CANONICAL_LATCH)


def _rot_err_deg(err6: np.ndarray) -> float:
    return math.degrees(float(np.linalg.norm(err6[3:])))


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


PERTURBATIONS = [
    (0.0, 0.0, 0.0),
    (0.08, 0.09, 35.0),  # Bay 2's demo jam
    (-0.08, -0.09, -35.0),
    (0.12, 0.0, 35.0),
    (0.0, -0.12, -35.0),
]


@needs_models
@pytest.mark.parametrize(("dx", "dy", "dyaw_deg"), PERTURBATIONS)
def test_panda_dls_ik_tracks_rotated_and_shifted_latch_pose(
    single_bay: Any, dx: float, dy: float, dyaw_deg: float
) -> None:
    model = single_bay
    data = mj.MjData(model)
    ik = PandaDiffIK(model, "bay1")
    local = CANONICAL_LATCH.offset(dx, dy, math.radians(dyaw_deg))
    target = wrist_pose_over_latch(latch_world_pose(1, local), height=0.20)

    # Kinematic convergence of the DLS iteration itself.
    reset_robot_home(model, data, 1)
    ik.reset()
    for _ in range(300):
        data.qpos[ik.qpos_adr] = ik.solve(data, target, 1.0 / 60)
        mj.mj_kinematics(model, data)
    err = ik.error(data, target)
    assert np.linalg.norm(err[:3]) < 1e-3
    assert _rot_err_deg(err) < 0.5

    # Closed-loop tracking through the Panda position servos at 60 Hz.
    reset_robot_home(model, data, 1)
    ik.reset()
    _drive(model, data, ik, lambda: target, seconds=3.0)
    err = ik.error(data, target)
    assert np.linalg.norm(err[:3]) < 5e-3
    assert _rot_err_deg(err) < 2.0
    q = np.asarray(data.qpos, dtype=float)[ik.qpos_adr]
    assert np.all(q >= ik.q_min - 1e-3)
    assert np.all(q <= ik.q_max + 1e-3)


@needs_models
def test_external_torque_spike_detected_on_flap_collision(single_bay: Any) -> None:
    model = single_bay
    data = mj.MjData(model)
    ik = PandaDiffIK(model, "bay1")
    est = ExternalTorqueEstimator(model, "bay1")
    assert len(est.joint_names) == 23  # 7 arm + 16 LEAP
    ctl = LatchController(model, data, "bay1")
    world_latch = latch_world_pose(1, CANONICAL_LATCH)
    flap_geom = int(model.geom("bay1/flap_paddle").id)

    reset_robot_home(model, data, 1)
    ctl.reset(flap_jammed=True)
    ik.reset()
    # Hover over the raised paddle (hinge at x = 0.0565, free end towards -X).
    hover = wrist_pose_over_latch(world_latch, height=0.20, along_x=0.035)
    _drive(model, data, ik, lambda: hover, seconds=2.5)
    free: list[float] = []
    _drive(model, data, ik, lambda: hover, 0.5, lambda: free.append(est.estimate_inf(data)))
    baseline = max(free)
    assert baseline < 0.5  # inverse dynamics explains free motion: no phantom external torque

    # Lower the hand onto the flap.
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
    assert touched, "hand never reached the stopper flap"
    assert max(pressed) > 2.0
    assert max(pressed) > 10.0 * max(baseline, 0.05)


@needs_models
def test_four_bay_scene_has_two_active_and_two_background_bays() -> None:
    model = build_four_bay_scene(CANONICAL_LATCH, BAY2_LATCH)
    data = mj.MjData(model)
    for bay in (1, 2, 3, 4):
        for name in ("joint1", "joint7", "if_mcp", "th_ipl", "bolt_slide", "flap_hinge"):
            model.joint(f"bay{bay}/{name}")
        for name in ("actuator1", "actuator7", "if_mcp_act", "th_axl_act", "rf_rot_act"):
            model.actuator(f"bay{bay}/{name}")
        model.site(f"bay{bay}/wrist_site")
        reset_robot_home(model, data, bay)
    assert int(model.nu) == 4 * 23

    background = {f"bay{b}/" for b in BACKGROUND_BAYS}
    for g in range(int(model.ngeom)):
        body = str(model.body(int(model.geom_bodyid[g])).name)
        if body[:5] in background:
            assert model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0, body

    # Bay 2's latch frame sits at its perturbed bay-local pose.
    mj.mj_forward(model, data)
    w2 = latch_world_pose(2, BAY2_LATCH)
    site = model.site("bay2/latch_frame").id
    assert np.allclose(data.site_xpos[site], w2.position, atol=1e-9)
    x_axis = np.asarray(data.site_xmat[site], dtype=float).reshape(3, 3)[:, 0]
    assert math.degrees(math.atan2(x_axis[1], x_axis[0])) == pytest.approx(35.0)
