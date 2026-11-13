"""IMU contingency wrist: orientation from the glove, position from home plus nudges."""

from __future__ import annotations

from teleop.config import WristConfig
from teleop.types import SIDES, HandFrame, Pose, Quat, Side, Vec3, WristTarget

_IDENTITY: Quat = (0.0, 0.0, 0.0, 1.0)


class ImuWrist:
    mode = "imu"

    def __init__(self, cfg: WristConfig) -> None:
        self._home: dict[Side, Vec3] = {"left": cfg.home_left, "right": cfg.home_right}
        self._nudge: dict[Side, Vec3] = {s: (0.0, 0.0, 0.0) for s in SIDES}

    def nudge(self, side: Side, dx: float, dy: float, dz: float) -> None:
        n = self._nudge[side]
        self._nudge[side] = (n[0] + dx, n[1] + dy, n[2] + dz)

    def reset_nudge(self) -> None:
        self._nudge = {s: (0.0, 0.0, 0.0) for s in SIDES}

    def set_home(self, side: Side, position: Vec3) -> None:
        """Rebase a side on a new home and clear its nudge (used by AutoWrist)."""
        self._home[side] = position
        self._nudge[side] = (0.0, 0.0, 0.0)

    @staticmethod
    def orientation(frame: HandFrame) -> Quat:
        if frame.wrist is not None:
            return frame.wrist.orientation
        return frame.joints.get("wrist", _IDENTITY)

    def wrist(self, frame: HandFrame) -> WristTarget:
        h = self._home[frame.side]
        n = self._nudge[frame.side]
        pos = (h[0] + n[0], h[1] + n[1], h[2] + n[2])
        return WristTarget(
            side=frame.side, pose=Pose(orientation=self.orientation(frame), position=pos)
        )
