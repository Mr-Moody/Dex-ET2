"""Structural interfaces implemented by the parallel M1 modules."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from teleop.types import (
    BimanualFrame,
    HandFrame,
    JointTargets,
    RobotState,
    SourceStatus,
    TaskScore,
    WristTarget,
)


@runtime_checkable
class InputSource(Protocol):
    name: str

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def latest(self) -> BimanualFrame: ...
    def status(self) -> SourceStatus: ...


@runtime_checkable
class PacketParser(Protocol):
    def parse(self, payload: bytes, host_time_ns: int) -> list[HandFrame]: ...


@runtime_checkable
class WristSource(Protocol):
    mode: str

    def wrist(self, frame: HandFrame) -> WristTarget: ...


@runtime_checkable
class Retargeter(Protocol):
    def retarget(self, frame: HandFrame) -> JointTargets: ...


@runtime_checkable
class RobotSink(Protocol):
    def apply(self, targets: Sequence[JointTargets], wrists: Sequence[WristTarget]) -> None: ...
    def state(self) -> RobotState: ...


@runtime_checkable
class Task(Protocol):
    name: str
    instruction: str

    def add_to_spec(self, spec: Any) -> None: ...  # mujoco.MjSpec
    def reset(self, model: Any, data: Any, seed: int) -> None: ...
    def score(self, model: Any, data: Any) -> TaskScore: ...


@runtime_checkable
class AuxSensorSource(Protocol):
    name: str

    def features(self) -> dict[str, dict[str, Any]]: ...
    def read(self) -> Mapping[str, Any]: ...
