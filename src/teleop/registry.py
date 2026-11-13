"""Lazy ``"module:attr"`` registries so parallel tasks never edit this file."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import cast

SOURCES: dict[str, str] = {
    "synthetic": "teleop.input.synthetic:SyntheticSource",
    "replay": "teleop.input.replay:ReplaySource",
    "studio": "teleop.input.studio:StudioSource",
    "rokoko-sdk": "teleop.input.rgmp:SdkSource",
}
WRISTS: dict[str, str] = {
    "auto": "teleop.wrist.auto:AutoWrist",
    "coil": "teleop.wrist.coil:CoilWrist",
    "imu": "teleop.wrist.imu:ImuWrist",
}
RETARGETERS: dict[str, str] = {"direct": "teleop.retarget.direct:DirectRetargeter"}
TASKS: dict[str, str] = {"lift_cube": "teleop.sim.tasks.lift_cube:LiftCube"}
SINKS: dict[str, str] = {"sim": "teleop.sim.sink:SimSink"}


def resolve(table: Mapping[str, str], name: str) -> type:
    """Import and return the class registered under ``name``; KeyError lists valid names."""
    if name not in table:
        raise KeyError(f"unknown name {name!r}; valid: {sorted(table)}")
    module, attr = table[name].split(":")
    return cast("type", getattr(importlib.import_module(module), attr))
