"""Noisy latch pose estimate, the fingertip touch detector and the push-axis touch correction.

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


def contact_offset(est: LatchEstimate, index_tip_world: ArrayLike, x_bolt: float) -> float:
    """``(tip - origin)·x - (HOOK_BACK_X_M + x_bolt)`` in the estimate's latch frame, m.

    At a true touch with an exact estimate this is the finger's contact offset: ``-pad``, where
    ``pad`` is how far the hook face sits ahead of the index tip site along latch +X. A skill
    calibrates it from its own demonstration (``SkillPrimitive.touch_offset_m``).
    """
    axis = np.array([math.cos(est.pose.yaw_rad), math.sin(est.pose.yaw_rad), 0.0])
    origin = np.asarray(est.pose.position, dtype=float)
    tip = np.asarray(index_tip_world, dtype=float)
    return float(np.dot(tip - origin, axis) - HOOK_BACK_X_M - x_bolt)


def touch_correct(
    est: LatchEstimate,
    index_tip_world: ArrayLike,
    x_bolt: float,
    pad_offset_m: float | None = None,
) -> LatchEstimate:
    """Shift the latch origin along its own +X so the hook back face meets the fingertip.

    ``pad_offset_m`` is the expected ``contact_offset`` at a true touch (``-pad``). The default
    ``-INDEX_PAD_RADIUS_M`` is the routine posture; a skill replay passes its demo's offset.
    """
    expected = -INDEX_PAD_RADIUS_M if pad_offset_m is None else pad_offset_m
    axis = np.array([math.cos(est.pose.yaw_rad), math.sin(est.pose.yaw_rad), 0.0])
    shift = contact_offset(est, index_tip_world, x_bolt) - expected
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


ENCODER_TOUCH_M = 0.0005  # bolt encoder travel that counts as a touch
WRIST_TOUCH_NM = 0.15  # compliant-wrist push torque that counts as a touch (tuned)


class TouchDetector:
    """Fingertip touch on the bolt hook from the bolt encoder or the compliant-wrist springs.

    Fires once per arming (``reset``): on the first tick the bolt has moved more than
    ``encoder_m`` from where it was when armed, or after ``ticks`` consecutive ticks of wrist
    push torque above ``wrist_nm``. A freely sliding bolt barely loads the wrist, so the encoder
    catches the clean-cycle touch and the wrist catches a blocked (pinned) bolt. Only sensors:
    the encoder ``x_bolt`` and the rig's own wrist spring deflection.
    """

    def __init__(
        self, encoder_m: float = ENCODER_TOUCH_M, wrist_nm: float = WRIST_TOUCH_NM, ticks: int = 2
    ) -> None:
        self.encoder_m = encoder_m
        self.wrist_nm = wrist_nm
        self.ticks = ticks
        self.armed = True
        self._x0: float | None = None
        self._high = 0

    def reset(self) -> None:
        """Arm: the next update sets the encoder baseline."""
        self.armed = True
        self._x0 = None
        self._high = 0

    def disarm(self) -> None:
        self.armed = False
        self._x0 = None
        self._high = 0

    def update(self, x_bolt: float, wrist_push_nm: float | None) -> bool:
        """One tick. ``wrist_push_nm=None`` is encoder-only (skill replay: a thumb pressing the
        flap loads the wrist without the index touching the hook)."""
        if not self.armed:
            return False
        if self._x0 is None:
            self._x0 = x_bolt
        wrist_high = wrist_push_nm is not None and wrist_push_nm > self.wrist_nm
        self._high = self._high + 1 if wrist_high else 0
        if x_bolt - self._x0 > self.encoder_m or self._high >= self.ticks:
            self.disarm()
            return True
        return False
