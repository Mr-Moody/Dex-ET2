"""Relative-clutch SE(3) indexing, quaternion flexion/calibration, and wrist Coil/IMU fallback."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import mujoco as _mujoco  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from conftest import needs_models
from teleop.atc.types import StallSignature
from teleop.config import WristConfig
from teleop.retarget.calibration import Calibration, calibrate
from teleop.retarget.clutch import (
    BAY_ACTUATORS,
    RelativeClutchIndexer,
    RescueTrajectoryBuffer,
    bleed_off_targets,
    read_hand_q,
)
from teleop.retarget.direct import DirectRetargeter
from teleop.retarget.flexion import flex_value, twist_angle
from teleop.sim.bays import (
    BAY2_LATCH,
    CANONICAL_LATCH,
    build_single_bay_scene,
    latch_world_pose,
    reset_robot_home,
)
from teleop.sim.hands import FlexMap, HandSpec
from teleop.types import HandFrame, Pose, Quat, Side
from teleop.unoq.dmp import HAND_ACTUATORS, compose_pose, mat_to_quat, quat_to_mat, so3_exp, so3_log
from teleop.unoq.server import UnoQServer
from teleop.wrist.auto import AutoWrist
from teleop.wrist.coil import CoilWrist
from teleop.wrist.imu import ImuWrist

mj: Any = _mujoco
IDENT: Quat = (0.0, 0.0, 0.0, 1.0)


def _pose(x: float, y: float, z: float, rotvec: tuple[float, float, float] = (0, 0, 0)) -> Pose:
    return Pose(orientation=mat_to_quat(so3_exp(rotvec)), position=(x, y, z))


def _pos_err(a: Pose, b: Pose) -> float:
    assert a.position is not None and b.position is not None
    return float(np.linalg.norm(np.subtract(a.position, b.position)))


def _rot_deg(a: Pose, b: Pose) -> float:
    r = quat_to_mat(a.orientation).T @ quat_to_mat(b.orientation)
    return math.degrees(float(np.linalg.norm(so3_log(r))))


def _qaxis(axis: str, a: float) -> Quat:
    s, c = math.sin(a / 2), math.cos(a / 2)
    return {"x": (s, 0.0, 0.0, c), "y": (0.0, s, 0.0, c), "z": (0.0, 0.0, s, c)}[axis]


# --------------------------------------------------------------------------- clutch + rescue buffer


def test_clutch_engage_and_relative_reclutch_have_zero_pose_jump() -> None:
    clutch = RelativeClutchIndexer()
    sim0 = _pose(0.5, 0.0, 0.2, (0.0, math.pi / 2, 0.0))
    glove0 = _pose(0.1, 0.2, 1.0, (0.0, 0.0, 0.5))

    assert clutch.update(False, glove0, sim0) is None
    assert clutch.update(True, glove0, sim0) == sim0

    turn = so3_exp((0.0, 0.0, math.radians(20.0)))
    glove1 = Pose(
        orientation=mat_to_quat(turn @ quat_to_mat(glove0.orientation)), position=(0.15, 0.2, 1.0)
    )
    cmd1 = clutch.update(True, glove1, sim0)
    assert cmd1 is not None and cmd1.position == pytest.approx((0.55, 0.0, 0.2))
    expected_r1 = turn @ quat_to_mat(sim0.orientation)
    assert _rot_deg(cmd1, Pose(orientation=mat_to_quat(expected_r1))) < 1e-6

    # Releasing and re-engaging at original hand location ratchets forward without jumping back.
    assert clutch.update(False, glove0, cmd1) is None
    assert clutch.update(True, glove0, cmd1) == cmd1
    cmd2 = clutch.update(
        True, Pose(orientation=glove0.orientation, position=(0.15, 0.2, 1.0)), cmd1
    )
    assert cmd2 is not None and cmd2.position == pytest.approx((0.60, 0.0, 0.2))


def test_hand_relative_clutch_tracks_hand_heading() -> None:
    clutch = RelativeClutchIndexer(hand_relative=True)
    # Hand pitched 45 deg (palm facing down/forward) and yawed +90 deg (facing world +Y).
    r_sim = so3_exp((0.0, 0.0, math.pi / 2)) @ so3_exp((0.0, -math.pi / 4, 0.0))
    sim0 = Pose(orientation=mat_to_quat(r_sim), position=(0.5, 0.0, 0.2))
    glove0 = _pose(0.0, 0.0, 1.0)

    assert clutch.update(True, glove0, sim0) == sim0
    # Operator moves forward (+Y) by 0.1 m -> moves along hand heading (+Y_world).
    cmd_fwd = clutch.update(True, _pose(0.0, 0.1, 1.0), sim0)
    assert cmd_fwd is not None
    assert cmd_fwd.position == pytest.approx((0.5, 0.1, 0.2), abs=1e-9)
    # Operator moves right (+X) by 0.1 m -> moves right of hand (+X_world when facing +Y).
    cmd_right = clutch.update(True, _pose(0.1, 0.1, 1.0), sim0)
    assert cmd_right is not None
    assert cmd_right.position == pytest.approx((0.6, 0.1, 0.2), abs=1e-9)
    # Operator moves up (+Z) by 0.05 m -> moves straight up in world Z.
    cmd_up = clutch.update(True, _pose(0.1, 0.1, 1.05), sim0)
    assert cmd_up is not None
    assert cmd_up.position == pytest.approx((0.6, 0.1, 0.25), abs=1e-9)


@needs_models
def test_clutch_release_bleeds_off_servo_force_and_discards_incomplete_rescue() -> None:
    model = build_single_bay_scene(CANONICAL_LATCH)
    data = mj.MjData(model)
    reset_robot_home(model, data, 1)
    ids = [int(model.actuator(f"bay1/{n}").id) for n in BAY_ACTUATORS]
    rng = np.random.default_rng(3)
    for i in ids:
        lo, hi = model.actuator_ctrlrange[i]
        data.ctrl[i] = float(np.clip(data.ctrl[i] + rng.uniform(-0.3, 0.3), lo, hi))
    for _ in range(5):
        mj.mj_step(model, data)
    before = float(np.abs(np.asarray(data.actuator_force)[ids[:7]]).max())
    assert before > 20.0

    q_target, _ = bleed_off_targets(model, data, "bay1")
    assert q_target.shape == (7 + len(HAND_ACTUATORS),)
    for _ in range(120):
        mj.mj_step(model, data)
    after = float(np.abs(np.asarray(data.actuator_force)[ids[:7]]).max())
    assert after < 0.1 * before

    buf = RescueTrajectoryBuffer(latch_world_pose(1, CANONICAL_LATCH))
    q_hand = read_hand_q(model, data, "bay1")
    for k in range(5):
        assert buf.record(True, k / 60, _pose(0.05, 0.0, 0.2), q_hand, 0.1)
    assert buf.finish(bolt_cleared=False) is None and len(buf) == 0


def test_successful_rescue_transforms_wrist_into_latch_frame_and_fits_dmp() -> None:
    world_latch = latch_world_pose(2, BAY2_LATCH)
    buf = RescueTrajectoryBuffer(world_latch)
    n = 60
    local_truth: list[Pose] = []
    for k in range(n):
        u = k / (n - 1)
        local = _pose(
            0.03 - 0.05 * u, 0.01 * math.sin(6 * u), 0.15 - 0.1 * u, (0.0, 0.4 * u, 0.2 * u)
        )
        local_truth.append(local)
        world = compose_pose(world_latch.as_pose(), local)
        q = 0.5 * u + 0.01 * np.arange(len(HAND_ACTUATORS))
        assert buf.record(
            True, 100.0 + k / 60, world, dict(zip(HAND_ACTUATORS, q, strict=True)), 0.5
        )

    traj = buf.finish(bolt_cleared=True)
    assert traj is not None and len(traj) == n
    for k in (0, n // 2, n - 1):
        row = traj.wrist_latch_se3[k]
        got = Pose(
            orientation=(float(row[3]), float(row[4]), float(row[5]), float(row[6])),
            position=(float(row[0]), float(row[1]), float(row[2])),
        )
        assert _pos_err(got, local_truth[k]) < 1e-9 and _rot_deg(got, local_truth[k]) < 1e-6

    reply = UnoQServer().handle(
        traj.fit_dmp_request("ev_latch_v1", StallSignature(0.035, 2.1, 3.4, 1.2))
    )
    assert reply["ok"] and reply["fit_ms"] < 20.0


# --------------------------------------------------------------------------- flexion + retargeting


@given(st.floats(min_value=-math.pi + 1e-3, max_value=math.pi - 1e-3))
def test_twist_angle_extracts_signed_axis_rotation_and_ignores_hemisphere_flip(a: float) -> None:
    for axis in ("x", "y", "z"):
        q = _qaxis(axis, a)
        neg = (-q[0], -q[1], -q[2], -q[3])
        assert abs(twist_angle(q, axis) - a) < 1e-6  # type: ignore[arg-type]
        assert abs(twist_angle(neg, axis) - a) < 1e-6  # type: ignore[arg-type]


def test_flex_calibration_and_direct_retargeter_roundtrip(tmp_path: Path) -> None:
    spec = HandSpec(
        "syn",
        {"left": Path("l.xml"), "right": Path("r.xml")},
        False,
        "base",
        [FlexMap("idx", ["a"], "x"), FlexMap("rot", ["b"], "z", range=[-0.5, 0.5])],
    )
    ctrl_ranges: Any = {s: {"idx": (0.0, 1.0), "rot": (-0.5, 0.5)} for s in ("left", "right")}

    def frame(a_rad: float | None, b_rad: float = 0.0) -> HandFrame:
        joints: dict[str, Quat] = {"b": _qaxis("z", b_rad)}
        if a_rad is not None:
            joints["a"] = _qaxis("x", a_rad)
        return HandFrame("right", 7, 0, "t", joints)

    assert flex_value(HandFrame("right", 0, 0, "t", {}), spec.flex[0]) is None

    calib = calibrate(spec, [frame(0.0), frame(0.0)], [frame(1.0), frame(1.0)])
    path = tmp_path / "calib.json"
    calib.to_json(path)
    assert Calibration.from_json(path) == calib
    assert calib.ranges["right"]["rot"] == (-0.5, 0.5)  # fixed-range actuator preserved

    ret = DirectRetargeter(spec, ctrl_ranges, calib)
    assert ret.retarget(frame(0.5, 0.25)).positions == pytest.approx({"idx": 0.5, "rot": 0.25})
    assert ret.retarget(frame(2.0, -2.0)).positions == pytest.approx({"idx": 1.0, "rot": -0.5})
    assert ret.retarget(frame(None, 0.0)).positions["idx"] == pytest.approx(1.0)  # holds last


# --------------------------------------------------------------------------- wrist sources


def _wrist_frame(
    side: Side = "right", pos: tuple[float, float, float] | None = None, wrist: bool = True
) -> HandFrame:
    return HandFrame(
        side=side,
        host_time_ns=0,
        seq=0,
        source="test",
        joints={},
        wrist=Pose(orientation=IDENT, position=pos) if wrist else None,
    )


def test_coil_imu_and_auto_wrist_switch_without_pose_jump() -> None:
    cfg = WristConfig(studio_to_sim_scale=2.0, studio_to_sim_offset=(0.0, 0.0, -0.75))
    coil = CoilWrist(cfg)
    with pytest.raises(ValueError):
        coil.wrist(_wrist_frame(wrist=False))

    imu = ImuWrist(cfg)
    imu.nudge("left", 0.02, 0.0, 0.0)
    assert imu.wrist(_wrist_frame("left", wrist=False)).pose.position == pytest.approx(
        (cfg.home_left[0] + 0.02, cfg.home_left[1], cfg.home_left[2])
    )

    now_ns = 0
    auto = AutoWrist(cfg, hold_s=0.5, clock=lambda: now_ns)
    last_coil = auto.wrist(_wrist_frame(pos=(0.1, 0.2, 1.0))).pose.position
    assert auto.mode == "coil" and last_coil is not None

    now_ns = int(0.6 * 1e9)
    first_imu = auto.wrist(_wrist_frame(wrist=False)).pose.position
    assert auto.mode == "imu" and first_imu == pytest.approx(last_coil)

    back_coil = auto.wrist(_wrist_frame(pos=(0.5, 0.3, 1.4))).pose.position
    assert auto.mode == "coil" and back_coil == pytest.approx(first_imu)
