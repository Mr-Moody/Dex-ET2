"""T-120: relative-clutch pose indexing, force bleed-off, rescue recording and split cameras."""

from __future__ import annotations

import math
from typing import Any

import mujoco as _mujoco  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
import pytest

from teleop.atc.types import LatchPose, StallSignature
from teleop.retarget.clutch import (
    BAY_ACTUATORS,
    SIDE_LIFT_M,
    TOP_TILT_RAD,
    RelativeClutchIndexer,
    RescueTrajectoryBuffer,
    bleed_off_targets,
    compose_split_view,
    read_hand_q,
    rescue_camera_names,
    rescue_view_option,
)
from teleop.sim.bays import (
    BAY2_LATCH,
    CANONICAL_LATCH,
    PANDA_XML,
    build_four_bay_spec,
    build_single_bay_scene,
    latch_world_pose,
    reset_robot_home,
)
from teleop.sim.hands import load_hand
from teleop.types import Pose
from teleop.unoq.dmp import (
    HAND_ACTUATORS,
    compose_pose,
    mat_to_quat,
    quat_to_mat,
    so3_exp,
    so3_log,
)
from teleop.unoq.server import UnoQServer

mj: Any = _mujoco


def _pose(x: float, y: float, z: float, rotvec: tuple[float, float, float] = (0, 0, 0)) -> Pose:
    return Pose(orientation=mat_to_quat(so3_exp(rotvec)), position=(x, y, z))


def _pos_err(a: Pose, b: Pose) -> float:
    assert a.position is not None and b.position is not None
    return float(np.linalg.norm(np.subtract(a.position, b.position)))


def _rot_deg(a: Pose, b: Pose) -> float:
    r = quat_to_mat(a.orientation).T @ quat_to_mat(b.orientation)
    return math.degrees(float(np.linalg.norm(so3_log(r))))


def _models_present() -> bool:
    try:
        right = load_hand("inspire").mjcf["right"]
    except FileNotFoundError:
        return False
    return PANDA_XML.is_file() and right.is_file()


needs_models = pytest.mark.skipif(
    not _models_present(), reason="Panda/Inspire MJCFs missing; run scripts/fetch_models.py"
)


# --------------------------------------------------------------------------- clutch indexing


def test_clutch_engage_has_zero_pose_jump() -> None:
    clutch = RelativeClutchIndexer()
    sim = _pose(0.55, -0.1, 0.22, (0.3, -1.2, 0.4))
    glove = _pose(-0.2, 0.4, 1.1, (1.0, 0.2, -2.0))  # far away and rotated: must not matter

    assert clutch.update(False, glove, sim) is None
    assert not clutch.engaged

    first = clutch.update(True, glove, sim)
    assert first == sim
    assert clutch.engaged

    # Glove held still on the next ticks: the command stays exactly on the anchor.
    for _ in range(3):
        held = clutch.update(True, glove, _pose(9.0, 9.0, 9.0))  # live sim wrist is ignored
        assert held is not None
        assert _pos_err(held, sim) < 1e-12
        assert _rot_deg(held, sim) < 1e-6


def test_relative_translation_and_rotation_track_after_re_clutch() -> None:
    clutch = RelativeClutchIndexer()
    sim0 = _pose(0.5, 0.0, 0.2, (0.0, math.pi / 2, 0.0))
    glove0 = _pose(0.1, 0.2, 1.0, (0.0, 0.0, 0.5))

    clutch.update(True, glove0, sim0)
    # Glove moves +5 cm in X and turns 20 deg about world Z.
    turn = so3_exp((0.0, 0.0, math.radians(20.0)))
    glove1 = Pose(
        orientation=mat_to_quat(turn @ quat_to_mat(glove0.orientation)), position=(0.15, 0.2, 1.0)
    )
    cmd1 = clutch.update(True, glove1, sim0)
    assert cmd1 is not None
    assert cmd1.position == pytest.approx((0.55, 0.0, 0.2))
    expected_r1 = turn @ quat_to_mat(sim0.orientation)  # same world-frame turn on the robot
    assert _rot_deg(cmd1, Pose(orientation=mat_to_quat(expected_r1))) < 1e-6

    # Release, carry the glove back to where it started, re-clutch: no jump, motion resumes.
    assert clutch.update(False, glove0, cmd1) is None
    assert clutch.update(True, glove0, cmd1) == cmd1
    glove2 = Pose(orientation=glove0.orientation, position=(0.15, 0.2, 1.0))
    cmd2 = clutch.update(True, glove2, cmd1)
    assert cmd2 is not None
    assert cmd2.position == pytest.approx((0.60, 0.0, 0.2))  # ratcheted a further 5 cm
    assert _rot_deg(cmd2, cmd1) < 1e-6

    # A desk rotated 90 deg and half-scale travel: glove +X becomes robot +Y at half the distance.
    desk = RelativeClutchIndexer.with_desk_yaw(math.radians(90.0), pos_scale=0.5)
    desk.update(True, glove0, sim0)
    cmd = desk.update(True, Pose(orientation=glove0.orientation, position=(0.3, 0.2, 1.0)), sim0)
    assert cmd is not None
    assert cmd.position == pytest.approx((0.5, 0.1, 0.2), abs=1e-12)


# --------------------------------------------------------------------------- bleed-off + buffer


@needs_models
def test_clutch_release_sets_qtarget_to_qactual_and_discards_incomplete_buffer() -> None:
    model = build_single_bay_scene(CANONICAL_LATCH)
    data = mj.MjData(model)
    reset_robot_home(model, data, 1)
    ids = [int(model.actuator(f"bay1/{n}").id) for n in BAY_ACTUATORS]
    joint_qadr = [int(model.jnt_qposadr[int(model.actuator_trnid[i][0])]) for i in ids]

    # Mid-rescue: every servo is chasing a target away from where its joint is.
    rng = np.random.default_rng(3)
    for i in ids:
        lo, hi = model.actuator_ctrlrange[i]
        data.ctrl[i] = float(np.clip(data.ctrl[i] + rng.uniform(-0.3, 0.3), lo, hi))
    for _ in range(5):
        mj.mj_step(model, data)
    forces_before = np.abs(np.asarray(data.actuator_force)[ids])
    assert forces_before[:7].max() > 20.0  # the arm is pushing hard

    q_target, named = bleed_off_targets(model, data, "bay1")
    assert q_target.shape == (7 + len(HAND_ACTUATORS),)
    assert set(named) == {f"bay1/{n}" for n in BAY_ACTUATORS}
    np.testing.assert_allclose(np.asarray(data.ctrl)[ids], q_target)
    # Targets are the measured positions clipped to each servo's range (a coupled Inspire
    # joint can sit a hair past its servo floor).
    lo, hi = np.asarray(model.actuator_ctrlrange)[ids].T
    np.testing.assert_allclose(q_target, np.clip(np.asarray(data.qpos)[joint_qadr], lo, hi))
    np.testing.assert_allclose(
        q_target[7:], np.clip(read_hand_q(model, data, "bay1"), lo[7:], hi[7:])
    )

    # Force bleeds off: with targets on the measured joints, the arm stops instead of lunging.
    q_hold = q_target.copy()
    for _ in range(120):
        mj.mj_step(model, data)
    drift = np.abs(np.asarray(data.qpos)[joint_qadr][:7] - q_hold[:7]).max()
    assert drift < 0.05
    forces_after = np.abs(np.asarray(data.actuator_force)[ids])
    assert forces_after[:7].max() < 0.1 * forces_before[:7].max()
    assert (
        np.abs(
            np.asarray(data.qvel)[
                [int(model.joint(f"bay1/joint{k}").dofadr[0]) for k in range(1, 8)]
            ]
        ).max()
        < 0.05
    )

    # A rescue released mid-jam is thrown away.
    buf = RescueTrajectoryBuffer(latch_world_pose(1, CANONICAL_LATCH))
    wrist = _pose(0.05, 0.0, 0.2)
    q_hand = read_hand_q(model, data, "bay1")
    assert not buf.record(False, 0.0, wrist, q_hand, 0.1)  # clutch not held: nothing recorded
    for k in range(10):
        assert buf.record(True, k / 60, wrist, q_hand, 0.1)
    assert len(buf) == 10
    assert buf.finish(bolt_cleared=False) is None
    assert len(buf) == 0
    assert buf.finish(bolt_cleared=True) is None  # nothing left to keep


def test_new_clutch_hold_without_finish_drops_the_stale_hold() -> None:
    buf = RescueTrajectoryBuffer(LatchPose(position=(0.0, 0.0, 0.0)))
    wrist = _pose(0.1, 0.0, 0.0)
    q_hand = [0.0] * len(HAND_ACTUATORS)
    for t in (0.0, 0.1):
        assert buf.record(True, t, wrist, q_hand, 0.0)
    assert not buf.record(False, 0.2, wrist, q_hand, 0.0)  # released, but finish() never called
    for t in (5.0, 5.1, 5.2):
        assert buf.record(True, t, wrist, q_hand, 0.0)

    traj = buf.finish(bolt_cleared=True)
    assert traj is not None
    np.testing.assert_allclose(traj.timestamps_s, [0.0, 0.1, 0.2])  # second hold only


def test_finish_after_release_tick_keeps_the_hold() -> None:
    buf = RescueTrajectoryBuffer(LatchPose(position=(0.0, 0.0, 0.0)))
    wrist = _pose(0.1, 0.0, 0.0)
    for t in (0.0, 0.1, 0.2):
        assert buf.record(True, t, wrist, [0.0] * len(HAND_ACTUATORS), 0.0)
    assert not buf.record(False, 0.3, wrist, [0.0] * len(HAND_ACTUATORS), 0.0)
    traj = buf.finish(bolt_cleared=True)
    assert traj is not None
    assert len(traj) == 3


def test_successful_clutch_rescue_transforms_wrist_into_local_latch_frame() -> None:
    world_latch = latch_world_pose(2, BAY2_LATCH)  # 35 deg yaw, shifted
    buf = RescueTrajectoryBuffer(world_latch)

    n = 120
    local_truth: list[Pose] = []
    q_truth = np.zeros((n, len(HAND_ACTUATORS)))
    t0 = 1234.5  # clock time at clutch press
    for k in range(n):
        u = k / (n - 1)
        local = _pose(
            0.03 - 0.05 * u, 0.01 * math.sin(6 * u), 0.15 - 0.1 * u, (0.0, 0.4 * u, 0.2 * u)
        )
        local_truth.append(local)
        q_truth[k] = 0.5 * u + 0.01 * np.arange(len(HAND_ACTUATORS))
        world = compose_pose(world_latch.as_pose(), local)
        joints = dict(zip(HAND_ACTUATORS, q_truth[k], strict=True))
        assert buf.record(True, t0 + k / 60, world, joints, 0.2 + 3.0 * (0.4 < u < 0.7))
    assert not buf.record(
        True, t0 + (n - 1) / 60, local_truth[-1], q_truth[-1], 0.0
    )  # no time step

    traj = buf.finish(bolt_cleared=True)
    assert traj is not None
    assert len(traj) == n
    assert len(buf) == 0
    assert traj.timestamps_s[0] == 0.0
    assert traj.duration_s == pytest.approx((n - 1) / 60)
    np.testing.assert_allclose(traj.fingers_q, q_truth)
    for k in range(n):
        got = Pose(
            orientation=(
                float(traj.wrist_latch_se3[k, 3]),
                float(traj.wrist_latch_se3[k, 4]),
                float(traj.wrist_latch_se3[k, 5]),
                float(traj.wrist_latch_se3[k, 6]),
            ),
            position=(
                float(traj.wrist_latch_se3[k, 0]),
                float(traj.wrist_latch_se3[k, 1]),
                float(traj.wrist_latch_se3[k, 2]),
            ),
        )
        assert _pos_err(got, local_truth[k]) < 1e-9
        assert _rot_deg(got, local_truth[k]) < 1e-6
        assert (
            _pos_err(buf.world_wrist(got), compose_pose(world_latch.as_pose(), local_truth[k]))
            < 1e-9
        )

    # The payload is exactly what the UNO Q fit_dmp op expects.
    reply = UnoQServer().handle(
        traj.fit_dmp_request("ev_latch_v1", StallSignature(0.035, 2.1, 3.4, 1.2))
    )
    assert reply["ok"], reply
    assert reply["skill"]["signature_id"] == "Jam_Sig_1"
    assert reply["fit_ms"] < 20.0


# --------------------------------------------------------------------------- split cameras


@needs_models
def test_split_rescue_cameras_are_orthographic_over_each_active_latch() -> None:
    # The fleet scene adds both active bays' rescue cameras itself (``build_four_bay_spec``).
    spec = build_four_bay_spec(CANONICAL_LATCH, BAY2_LATCH, include_background_bays=False)
    model = spec.compile()
    data = mj.MjData(model)
    mj.mj_forward(model, data)

    for bay, local in ((1, CANONICAL_LATCH), (2, BAY2_LATCH)):
        w = latch_world_pose(bay, local)
        latch_x = np.array([math.cos(w.yaw_rad), math.sin(w.yaw_rad), 0.0])
        latch_y = np.array([-math.sin(w.yaw_rad), math.cos(w.yaw_rad), 0.0])
        top, side = rescue_camera_names(f"bay{bay}")
        for name in (top, side):
            cam = model.camera(name)
            assert int(model.cam_projection[cam.id]) == int(mj.mjtProjection.mjPROJ_ORTHOGRAPHIC)
        top_mat = np.asarray(data.cam_xmat[model.camera(top).id]).reshape(3, 3)
        side_mat = np.asarray(data.cam_xmat[model.camera(side).id]).reshape(3, 3)
        # Cameras look along their -Z: top down (tilted across the slide axis), side along +Y.
        tilt = math.cos(TOP_TILT_RAD) * np.array([0.0, 0.0, 1.0]) + math.sin(TOP_TILT_RAD) * latch_y
        np.testing.assert_allclose(top_mat[:, 2], tilt, atol=1e-9)
        assert -top_mat[2, 2] < -0.9  # mostly looking down
        np.testing.assert_allclose(top_mat[:, 0], latch_x, atol=1e-9)  # bolt slides to the right
        np.testing.assert_allclose(side_mat[:, 0], latch_x, atol=1e-9)
        np.testing.assert_allclose(side_mat[:, 1], [0, 0, 1], atol=1e-9)
        origin = np.asarray(w.position)
        aims = ((top, top_mat, origin), (side, side_mat, origin + [0.0, 0.0, SIDE_LIFT_M]))
        for name, mat, aim in aims:  # each view is centred on the latch
            to_aim = aim - np.asarray(data.cam_xpos[model.camera(name).id])
            assert float(-mat[:, 2] @ to_aim) / float(np.linalg.norm(to_aim)) > 0.999


@needs_models
def test_split_view_renders_orange_flap_in_top_inset(offscreen_render_ok: None) -> None:
    spec = build_four_bay_spec(include_background_bays=False)
    top, side = rescue_camera_names("bay1")
    model = spec.compile()
    data = mj.MjData(model)
    reset_robot_home(model, data, 1)
    renderer = mj.Renderer(model, 120, 160)
    images = []
    for cam in ("overview", top, side):
        opt = None if cam == "overview" else rescue_view_option()
        renderer.update_scene(data, cam, scene_option=opt)
        images.append(renderer.render().copy())
    renderer.close()
    main, top_img, side_img = images

    rgb = top_img.astype(int)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    orange = (r > 180) & (r - b > 100) & (g < r)  # the lit paddle reads orange to yellow-orange
    assert orange.sum() > 20  # the stopper paddle is visible, not hidden by the arm

    out = compose_split_view(main, top_img, side_img)
    assert out.shape == main.shape
    assert not np.array_equal(out, main)


def test_compose_split_view_insets_top_right() -> None:
    main = np.zeros((90, 160, 3), dtype=np.uint8)
    top = np.full((40, 40, 3), 100, dtype=np.uint8)
    side = np.full((40, 80, 3), 200, dtype=np.uint8)
    out = compose_split_view(main, top, side, border_px=2, corner="top_right")
    assert out.shape == main.shape
    assert out[2:32, 128:158].min() == 100  # top inset, 30 px tall, top-right corner
    assert out[36:66, 98:158].min() == 200  # side inset below it, aspect kept
    assert out[80:, :].max() == 0  # rest of the overview untouched
    assert out[:, :90].max() == 0

    low = compose_split_view(main, top, side, border_px=2)  # default: bottom-right
    assert low[22:52, 128:158].min() == 100
    assert low[56:86, 98:158].min() == 200
    assert low[:18, :].max() == 0
    with pytest.raises(ValueError):
        compose_split_view(main[..., 0], top, side)


def test_latch_pose_buffer_accepts_pose_or_latch_pose() -> None:
    lp = LatchPose(position=(0.1, 0.2, 0.02), yaw_rad=0.3)
    a = RescueTrajectoryBuffer(lp)
    b = RescueTrajectoryBuffer(lp.as_pose())
    assert a.world_from_latch == b.world_from_latch
    with pytest.raises(ValueError):
        a.record(True, 0.0, _pose(0, 0, 0), [0.0] * 15, 0.0)
