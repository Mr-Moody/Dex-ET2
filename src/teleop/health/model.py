"""Health value types shared by every hardware check."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class Level(StrEnum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"
    OFF = "off"


@dataclass(frozen=True)
class Health:
    name: str
    level: Level
    summary: str
    detail: str = ""
    can_reconnect: bool = False
    busy: bool = False


class HealthCheck(Protocol):
    """One hardware check. ``status()`` must not block; ``reconnect()`` may."""

    name: str

    def status(self) -> Health: ...

    def reconnect(self) -> None: ...


def health_to_dict(h: Health) -> dict[str, object]:
    return {
        "name": h.name,
        "level": h.level.value,
        "summary": h.summary,
        "detail": h.detail,
        "can_reconnect": h.can_reconnect,
        "busy": h.busy,
    }
