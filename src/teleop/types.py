"""Shared value types for the teleop pipeline. Quaternions are xyzw, positions in metres."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

Side = Literal["left", "right"]
SIDES: tuple[Side, Side] = ("left", "right")
Quat = tuple[float, float, float, float]  # xyzw
Vec3 = tuple[float, float, float]

_FINGERS = ("thumb", "index", "middle", "ring", "little")
_THUMB_PARTS = ("metacarpal", "proximal", "distal", "tip")
_FINGER_PARTS = ("metacarpal", "proximal", "intermediate", "distal", "tip")

# 26 names in XrHandJointEXT order.
OPENXR_JOINTS: tuple[str, ...] = ("palm", "wrist") + tuple(
    f"{finger}_{part}"
    for finger in _FINGERS
    for part in (_THUMB_PARTS if finger == "thumb" else _FINGER_PARTS)
)


@dataclass(frozen=True, slots=True)
class Pose:
    orientation: Quat
    position: Vec3 | None = None


@dataclass(frozen=True, slots=True)
class HandFrame:
    side: Side
    host_time_ns: int
    seq: int
    source: str
    joints: Mapping[str, Quat]  # local rotations, keys subset of OPENXR_JOINTS
    device_time_us: int | None = None
    wrist: Pose | None = None  # world pose if the source provides it (Coil Pro)


@dataclass(frozen=True, slots=True)
class BimanualFrame:
    left: HandFrame | None
    right: HandFrame | None


@dataclass(frozen=True, slots=True)
class SourceStatus:
    connected: bool
    last_rx_ns: int | None
    frames: int
    dropped: int
    timestamp_anomalies: int


@dataclass(frozen=True, slots=True)
class JointTargets:
    side: Side
    host_time_ns: int
    positions: Mapping[str, float]  # unprefixed actuator names


@dataclass(frozen=True, slots=True)
class WristTarget:
    side: Side
    pose: Pose  # pose.position required


@dataclass(frozen=True, slots=True)
class RobotState:
    qpos: Mapping[str, float]  # prefixed joint names
    wrist: Mapping[Side, Pose]


@dataclass(frozen=True, slots=True)
class TaskScore:
    success: bool
    score: float
    elapsed_s: float
