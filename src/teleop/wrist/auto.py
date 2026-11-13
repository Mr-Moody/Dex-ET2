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
        self._active: set[Side] = set()
        self.modes: dict[Side, str] = {s: "imu" for s in SIDES}

    @property
    def mode(self) -> str:
        """Common mode of the sides seen so far, or "mixed" when they differ."""
        seen = {self.modes[s] for s in SIDES if s in self._active} or {"imu"}
        return seen.pop() if len(seen) == 1 else "mixed"

    def mode_for(self, side: Side) -> str:
        return self.modes[side]

    @property
    def coil(self) -> CoilWrist:
        return self._coil

    @property
    def imu(self) -> ImuWrist:
        return self._imu

    def wrist(self, frame: HandFrame) -> WristTarget:
        side = frame.side
        self._active.add(side)
        now = self._clock()
        if frame.wrist is not None and frame.wrist.position is not None:
            if self.modes[side] == "imu" and self._last_pos[side] is not None:
                # Returning from IMU: keep the hand where it is, no jump.
                current = self._imu.wrist(frame).pose.position
                if current is not None:
                    self._coil.recenter_to(side, frame, current)
            target = self._coil.wrist(frame)
            self._last_seen[side] = now
            self._last_pos[side] = target.pose.position
            self._handover[side] = True
            self.modes[side] = "coil"
            return target
        seen = self._last_seen[side]
        last = self._last_pos[side]
        if seen is not None and last is not None and now - seen <= self._hold_ns:
            # Brief dropout: hold the last Coil position with the frame's orientation.
            self.modes[side] = "coil"
            quat = self._imu.orientation(frame)
            return WristTarget(side=side, pose=Pose(orientation=quat, position=last))
        if self._handover[side] and last is not None:
            self._imu.set_home(side, last)
            self._handover[side] = False
        self.modes[side] = "imu"
        return self._imu.wrist(frame)
