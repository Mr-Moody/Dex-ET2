"""Wrist source backed by Rokoko Coil Pro world positions."""

from __future__ import annotations

from teleop.config import WristConfig
from teleop.types import SIDES, HandFrame, Pose, Side, Vec3, WristTarget


class CoilWrist:
    mode = "coil"

    def __init__(self, cfg: WristConfig) -> None:
        self._cfg = cfg
        self._recenter: dict[Side, Vec3] = {s: (0.0, 0.0, 0.0) for s in SIDES}

    def _home(self, side: Side) -> Vec3:
        return self._cfg.home_left if side == "left" else self._cfg.home_right

    def _base(self, frame: HandFrame) -> tuple[Pose, Vec3]:
        if frame.wrist is None or frame.wrist.position is None:
            raise ValueError("CoilWrist needs frame.wrist.position (Coil Pro not tracking)")
        p = frame.wrist.position
        k = self._cfg.studio_to_sim_scale
        o = self._cfg.studio_to_sim_offset
        return frame.wrist, (p[0] * k + o[0], p[1] * k + o[1], p[2] * k + o[2])

    def wrist(self, frame: HandFrame) -> WristTarget:
        pose, base = self._base(frame)
        r = self._recenter[frame.side]
        pos = (base[0] + r[0], base[1] + r[1], base[2] + r[2])
        return WristTarget(side=frame.side, pose=Pose(orientation=pose.orientation, position=pos))

    def recenter_to(self, side: Side, frame: HandFrame, target: Vec3) -> None:
        """Map the frame's current Coil position onto `target` (sim coordinates).

        Applied on top of the config scale and offset. Raises ValueError if the frame
        has no position or belongs to the other side.
        """
        if frame.side != side:
            raise ValueError(f"frame is for {frame.side}, expected {side}")
        _, base = self._base(frame)
        self._recenter[side] = (target[0] - base[0], target[1] - base[1], target[2] - base[2])

    def recenter(self, frame_left: HandFrame | None, frame_right: HandFrame | None) -> None:
        """Map each given side's current Coil position onto its home pose.

        Raises ValueError for a given frame without a position.
        """
        for frame in (frame_left, frame_right):
            if frame is not None:
                self.recenter_to(frame.side, frame, self._home(frame.side))
