"""Noisy latch pose estimate with a touch correction along the push axis.

Pure module: no MuJoCo state. The estimate perturbs x, y and yaw of the true latch pose
(z is assumed known); a fingertip contact on the hook's back face then pins the push axis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import ArrayLike

from teleop.atc.types import LatchPose
from teleop.sim.latch import HOOK_HALF, HOOK_POS

POSE_SIGMA_XY_M = 0.003
POSE_SIGMA_YAW_RAD = math.radians(1.5)
TOUCH_SIGMA_M = 0.001
MAX_TOUCH_SIGMAS = 4.0
HOOK_BACK_X_M = HOOK_POS[0] - HOOK_HALF[0]  # hook back face in the latch frame at x_bolt = 0
INDEX_PAD_RADIUS_M = 0.0144  # index tip site to its hook contact along latch +X (tuned)


@dataclass(frozen=True, slots=True)
class LatchEstimate:
    pose: LatchPose
    sigma_xy_m: float
    sigma_x_m: float  # push axis; equals sigma_xy_m until touched
    sigma_yaw_rad: float
    touch_corrected: bool = False
    last_correction_m: float | None = None
    rejected_correction_m: float | None = None


class LatchEstimator:
    def __init__(
        self,
        seed: int = 0,
        sigma_xy_m: float = POSE_SIGMA_XY_M,
        sigma_yaw_rad: float = POSE_SIGMA_YAW_RAD,
    ) -> None:
        self._seed = seed
        self._sigma_xy_m = sigma_xy_m
        self._sigma_yaw_rad = sigma_yaw_rad

    def sample(self, true_pose: LatchPose, bay_id: int, cycle: int) -> LatchEstimate:
        """One draw per (seed, bay, cycle); independent of call order."""
        rng = np.random.default_rng((self._seed, bay_id, cycle))
        dx, dy = rng.normal(0.0, self._sigma_xy_m, size=2)
        dyaw = rng.normal(0.0, self._sigma_yaw_rad)
        pose = true_pose.offset(dx=float(dx), dy=float(dy), dyaw_rad=float(dyaw))
        return LatchEstimate(pose, self._sigma_xy_m, self._sigma_xy_m, self._sigma_yaw_rad)


def touch_correct(est: LatchEstimate, index_tip_world: ArrayLike, x_bolt: float) -> LatchEstimate:
    """Shift the latch origin along its own +X so the hook back face meets the fingertip."""
    axis = np.array([math.cos(est.pose.yaw_rad), math.sin(est.pose.yaw_rad), 0.0])
    origin = np.asarray(est.pose.position, dtype=float)
    tip = np.asarray(index_tip_world, dtype=float)
    shift = float(np.dot(tip - origin, axis) + INDEX_PAD_RADIUS_M - HOOK_BACK_X_M - x_bolt)
    # Floored at the touch sigma: an exact placement (sigma 0) still takes a mm-level touch.
    if abs(shift) > MAX_TOUCH_SIGMAS * max(est.sigma_x_m, TOUCH_SIGMA_M):
        return replace(est, rejected_correction_m=shift)
    x, y, z = est.pose.position
    pose = LatchPose(
        position=(x + shift * float(axis[0]), y + shift * float(axis[1]), z),
        yaw_rad=est.pose.yaw_rad,
    )
    return replace(
        est,
        pose=pose,
        sigma_x_m=TOUCH_SIGMA_M,
        touch_corrected=True,
        last_correction_m=shift,
        rejected_correction_m=None,
    )
