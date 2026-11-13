"""Auto wrist: Coil Pro while positions keep arriving, IMU contingency otherwise."""

from __future__ import annotations

import time
from collections.abc import Callable

from teleop.config import WristConfig
from teleop.types import SIDES, HandFrame, Pose, Side, Vec3, WristTarget
from teleop.wrist.coil import CoilWrist
from teleop.wrist.imu import ImuWrist


class AutoWrist:
    def __init__(
        self,
        cfg: WristConfig,
        hold_s: float = 0.5,
        clock: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self._coil = CoilWrist(cfg)
        self._imu = ImuWrist(cfg)
        self._hold_ns = int(hold_s * 1e9)
        self._clock = clock
        self._last_seen: dict[Side, int | None] = {s: None for s in SIDES}
        self._last_pos: dict[Side, Vec3 | None] = {s: None for s in SIDES}
        self._handover: dict[Side, bool] = {s: False for s in SIDES}
        self.mode = "imu"

    @property
    def coil(self) -> CoilWrist:
        return self._coil

    @property
    def imu(self) -> ImuWrist:
        return self._imu

    def wrist(self, frame: HandFrame) -> WristTarget:
        side = frame.side
        now = self._clock()
        if frame.wrist is not None and frame.wrist.position is not None:
            target = self._coil.wrist(frame)
            self._last_seen[side] = now
            self._last_pos[side] = target.pose.position
            self._handover[side] = True
            self.mode = "coil"
            return target
        seen = self._last_seen[side]
        last = self._last_pos[side]
        if seen is not None and last is not None and now - seen <= self._hold_ns:
            # Brief dropout: hold the last Coil position with the frame's orientation.
            self.mode = "coil"
            quat = self._imu.wrist(frame).pose.orientation
            return WristTarget(side=side, pose=Pose(orientation=quat, position=last))
        if self._handover[side] and last is not None:
            self._imu.set_home(side, last)
            self._handover[side] = False
        self.mode = "imu"
        return self._imu.wrist(frame)
