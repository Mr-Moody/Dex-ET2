"""LeRobot dataset feature schema (plain dicts)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from teleop.config import CameraConfig
from teleop.types import SIDES, Side

WRIST_FIELDS = ("x", "y", "z", "qx", "qy", "qz", "qw")


def _names(per_side: Mapping[Side, Sequence[str]]) -> list[str]:
    return [f"{side}/{n}" for side in SIDES for n in per_side[side]] + [
        f"{side}/wrist.{f}" for side in SIDES for f in WRIST_FIELDS
    ]


def state_names(hand_joint_names: Mapping[Side, Sequence[str]]) -> list[str]:
    return _names(hand_joint_names)


def action_names(hand_actuator_names: Mapping[Side, Sequence[str]]) -> list[str]:
    return _names(hand_actuator_names)


def build_features(
    state: list[str], action: list[str], cameras: Sequence[CameraConfig]
) -> dict[str, dict[str, Any]]:
    feats: dict[str, dict[str, Any]] = {
        "observation.state": {"dtype": "float32", "shape": (len(state),), "names": state},
        "action": {"dtype": "float32", "shape": (len(action),), "names": action},
    }
    for c in cameras:
        feats[f"observation.images.{c.name}"] = {
            "dtype": "video",
            "shape": (c.height, c.width, 3),
            "names": ["height", "width", "channels"],
        }
    return feats
