"""Wrist sources: Coil Pro, IMU contingency and auto-switching."""

from __future__ import annotations

import pytest

from teleop.config import WristConfig
from teleop.types import HandFrame, Pose, Side
from teleop.wrist.auto import AutoWrist
from teleop.wrist.coil import CoilWrist
from teleop.wrist.imu import ImuWrist

IDENT = (0.0, 0.0, 0.0, 1.0)


def make_frame(
    side: Side = "right",
    pos: tuple[float, float, float] | None = None,
    wrist: bool = True,
    joints: dict | None = None,
    orient=IDENT,
) -> HandFrame:
    return HandFrame(
        side=side,
        host_time_ns=0,
        seq=0,
        source="test",
        joints=joints or {},
        wrist=Pose(orientation=orient, position=pos) if wrist else None,
    )


class FakeClock:
    def __init__(self) -> None:
        self.t = 0

    def __call__(self) -> int:
        return self.t

    def advance(self, s: float) -> None:
        self.t += int(s * 1e9)


def approx(a, b):
    return a == pytest.approx(b, abs=1e-9)


def test_coil_applies_scale_offset() -> None:
    cfg = WristConfig(studio_to_sim_scale=2.0, studio_to_sim_offset=(0.1, 0.2, -1.0))
    out = CoilWrist(cfg).wrist(make_frame(pos=(0.1, 0.2, 1.0)))
    assert out.side == "right"
    assert approx(out.pose.position, (0.3, 0.6, 1.0))
    assert out.pose.orientation == IDENT


def test_coil_without_position_raises() -> None:
    coil = CoilWrist(WristConfig())
    with pytest.raises(ValueError):
        coil.wrist(make_frame(wrist=False))
    with pytest.raises(ValueError):
        coil.wrist(make_frame(pos=None))


def test_coil_recenter_maps_current_to_home() -> None:
    cfg = WristConfig(studio_to_sim_scale=2.0, studio_to_sim_offset=(0.5, 0.0, 0.0))
    coil = CoilWrist(cfg)
    fl = make_frame("left", (-0.3, 0.1, 1.0))
    fr = make_frame("right", (0.4, -0.1, 1.2))
    coil.recenter(fl, fr)
    assert approx(coil.wrist(fl).pose.position, cfg.home_left)
    assert approx(coil.wrist(fr).pose.position, cfg.home_right)
    moved = make_frame("right", (0.5, -0.1, 1.2))
    assert approx(coil.wrist(moved).pose.position, (0.25 + 0.2, 0.0, 0.25))


def test_imu_uses_home_plus_nudge() -> None:
    cfg = WristConfig()
    imu = ImuWrist(cfg)
    f = make_frame("left", wrist=False)
    assert approx(imu.wrist(f).pose.position, cfg.home_left)
    imu.nudge("left", 0.01, 0.0, 0.02)
    imu.nudge("left", 0.01, 0.0, 0.0)
    assert approx(imu.wrist(f).pose.position, (-0.23, 0.0, 0.27))
    assert approx(imu.wrist(make_frame("right", wrist=False)).pose.position, cfg.home_right)
    imu.reset_nudge()
    assert approx(imu.wrist(f).pose.position, cfg.home_left)


def test_imu_orientation_fallback() -> None:
    imu = ImuWrist(WristConfig())
    q1 = (0.0, 0.0, 0.7071067811865476, 0.7071067811865476)
    q2 = (0.7071067811865476, 0.0, 0.0, 0.7071067811865476)
    assert imu.wrist(make_frame(orient=q1, joints={"wrist": q2})).pose.orientation == q1
    assert imu.wrist(make_frame(wrist=False, joints={"wrist": q2})).pose.orientation == q2
    assert imu.wrist(make_frame(wrist=False)).pose.orientation == IDENT


def test_auto_switches_to_imu_after_hold() -> None:
    clock = FakeClock()
    auto = AutoWrist(WristConfig(), hold_s=0.5, clock=clock)
    auto.wrist(make_frame(pos=(0.0, 0.0, 1.0)))
    assert auto.mode == "coil"
    clock.advance(0.3)
    auto.wrist(make_frame(wrist=False))
    assert auto.mode == "coil"
    clock.advance(0.3)
    auto.wrist(make_frame(wrist=False))
    assert auto.mode == "imu"


def test_auto_no_jump_on_switch() -> None:
    clock = FakeClock()
    cfg = WristConfig(studio_to_sim_offset=(0.0, 0.0, -0.75))
    auto = AutoWrist(cfg, hold_s=0.5, clock=clock)
    last = auto.wrist(make_frame(pos=(0.1, 0.2, 1.0))).pose.position
    clock.advance(0.6)
    first_imu = auto.wrist(make_frame(wrist=False)).pose.position
    assert auto.mode == "imu"
    assert approx(first_imu, last)
    # Per side: the other hand never saw Coil and sits at its home.
    assert approx(auto.wrist(make_frame("left", wrist=False)).pose.position, cfg.home_left)
