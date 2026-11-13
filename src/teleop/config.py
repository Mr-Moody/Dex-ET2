"""Draccus-loadable configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from teleop.types import Vec3


@dataclass
class StudioConfig:
    host: str = "0.0.0.0"
    port: int = 14043
    actor: str | None = None
    tee_raw: Path | None = None


@dataclass
class WristConfig:
    mode: str = "auto"
    home_left: Vec3 = (-0.25, 0.0, 0.25)
    home_right: Vec3 = (0.25, 0.0, 0.25)
    studio_to_sim_offset: Vec3 = (0.0, 0.0, 0.0)
    studio_to_sim_scale: float = 1.0
    nudge_step_m: float = 0.01


@dataclass
class CameraConfig:
    name: str = "front"
    width: int = 224
    height: int = 224


@dataclass
class SimConfig:
    source: str = "synthetic"
    hand: str = "inspire"
    task: str = "lift_cube"
    sink: str = "sim"
    retargeter: str = "direct"
    control_hz: int = 60
    physics_hz: int = 600
    record_fps: int = 30
    headless: bool = False
    steps: int | None = None
    calibration: Path | None = None
    replay_file: Path | None = None
    studio: StudioConfig = field(default_factory=StudioConfig)
    wrist: WristConfig = field(default_factory=WristConfig)
    cameras: list[CameraConfig] = field(default_factory=lambda: [CameraConfig()])
