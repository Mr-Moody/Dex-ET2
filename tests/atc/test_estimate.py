from __future__ import annotations

import math

import numpy as np
import pytest

from teleop.atc.estimate import (
    HOOK_BACK_X_M,
    INDEX_PAD_RADIUS_M,
    TOUCH_SIGMA_M,
    LatchEstimate,
    LatchEstimator,
    TouchDetector,
    contact_offset,
    touch_correct,
)
from teleop.atc.types import LatchPose
from teleop.sim.bays import BAY2_LATCH, CANONICAL_LATCH


def axis_x(pose: LatchPose) -> np.ndarray:
    return np.array([math.cos(pose.yaw_rad), math.sin(pose.yaw_rad), 0.0])


def local_point(pose: LatchPose, x: float, y: float, z: float) -> np.ndarray:
    """Latch-frame coordinates to world coordinates."""
    c, s = math.cos(pose.yaw_rad), math.sin(pose.yaw_rad)
    px, py, pz = pose.position
    return np.array([px + c * x - s * y, py + s * x + c * y, pz + z])


def test_sample_is_seeded_per_bay_and_cycle():
    e = LatchEstimator(seed=7)
    a, b = e.sample(CANONICAL_LATCH, 1, 3), LatchEstimator(seed=7).sample(CANONICAL_LATCH, 1, 3)
    assert a == b
    assert e.sample(CANONICAL_LATCH, 1, 4) != a and e.sample(CANONICAL_LATCH, 2, 3) != a


def test_sample_error_statistics_match_sigma():
    e = LatchEstimator(seed=0)
    errs = np.array(
        [
            np.subtract(
                e.sample(CANONICAL_LATCH, 1, k).pose.position[:2], CANONICAL_LATCH.position[:2]
            )
            for k in range(2000)
        ]
    )
    np.testing.assert_allclose(errs.std(axis=0), [0.003, 0.003], rtol=0.1)


def test_zero_sigma_returns_truth():
    est = LatchEstimator(sigma_xy_m=0.0, sigma_yaw_rad=0.0).sample(BAY2_LATCH, 2, 0)
    assert est.pose == BAY2_LATCH


def test_touch_moves_estimate_along_push_axis_only():
    est = LatchEstimate(LatchPose((0.5, 0.01, 0.02), 0.3), 0.003, 0.003, 0.026)
    # A tip exactly where the estimate expects the hook (x_bolt = 0.01) gives no shift.
    expect = local_point(est.pose, HOOK_BACK_X_M + 0.01 - INDEX_PAD_RADIUS_M, 0.0, 0.05)
    assert touch_correct(est, expect, 0.01).last_correction_m == pytest.approx(0.0, abs=1e-9)
    moved = touch_correct(est, expect + 0.004 * axis_x(est.pose), 0.01)
    assert moved.last_correction_m == pytest.approx(0.004)
    assert moved.pose.yaw_rad == est.pose.yaw_rad and moved.sigma_x_m == TOUCH_SIGMA_M
    assert moved.touch_corrected


def test_outlier_touch_is_rejected():
    est = LatchEstimate(CANONICAL_LATCH, 0.003, 0.003, 0.026)
    tip = local_point(est.pose, HOOK_BACK_X_M - INDEX_PAD_RADIUS_M + 0.013, 0.0, 0.05)
    out = touch_correct(est, tip, 0.0)
    assert out.pose == est.pose and out.rejected_correction_m == pytest.approx(0.013)


def test_outlier_gate_is_floored_at_the_touch_sigma():
    # An exact placement (sigma 0) still accepts a touch inside 4 touch sigmas (4 mm).
    est = LatchEstimate(CANONICAL_LATCH, 0.0, 0.0, 0.0)
    tip = local_point(est.pose, HOOK_BACK_X_M - INDEX_PAD_RADIUS_M + 0.0007, 0.0, 0.05)
    moved = touch_correct(est, tip, 0.0)
    assert moved.touch_corrected and moved.last_correction_m == pytest.approx(0.0007)
    far = local_point(est.pose, HOOK_BACK_X_M - INDEX_PAD_RADIUS_M + 0.0045, 0.0, 0.05)
    out = touch_correct(est, far, 0.0)
    assert out.pose == est.pose and out.rejected_correction_m == pytest.approx(0.0045)


def test_skill_contact_offset_convention_and_sign():
    truth = LatchPose((0.5, 0.01, 0.02), 0.3)
    tip = local_point(truth, HOOK_BACK_X_M + 0.03 - 0.009, 0.004, 0.05)  # a 9 mm "pad"
    exact = LatchEstimate(truth, 0.003, 0.001, 0.026)
    off = contact_offset(exact, tip, 0.03)
    assert off == pytest.approx(-0.009)  # offset = -pad
    # The default pad offset is today's INDEX_PAD_RADIUS_M behaviour.
    assert touch_correct(exact, tip, 0.03).last_correction_m == pytest.approx(
        touch_correct(exact, tip, 0.03, pad_offset_m=-INDEX_PAD_RADIUS_M).last_correction_m
    )
    # The demo's own offset makes a truthful estimate a zero correction ...
    assert touch_correct(exact, tip, 0.03, pad_offset_m=off).last_correction_m == pytest.approx(
        0.0, abs=1e-12
    )
    # ... and an estimate 1 mm too far along +X is corrected by -1 mm.
    ahead = LatchEstimate(
        LatchPose(tuple(np.add(truth.position, 0.001 * axis_x(truth))), truth.yaw_rad),
        0.003,
        0.001,
        0.026,
    )
    fixed = touch_correct(ahead, tip, 0.03, pad_offset_m=off)
    assert fixed.last_correction_m == pytest.approx(-0.001)
    np.testing.assert_allclose(fixed.pose.position, truth.position, atol=1e-12)


def test_touch_detector_encoder_only_ignores_the_wrist():
    det = TouchDetector()
    det.reset()
    assert not any(det.update(0.03, None) for _ in range(10))  # wrist path disabled
    assert det.update(0.0306, None)  # encoder still fires
    det.reset()
    assert not det.update(0.03, 5.0) and det.update(0.03, 5.0)  # with a reading, 2 ticks
