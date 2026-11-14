"""Which camera feeds the console shows, in which slot, at what size.

Grid mode: one CCTV tile per bay present. Rescue mode: the rescued bay's top and side
orthographic views as the two main feeds, then every bay's CCTV as a small strip. Slots are
the byte-0 ids of the binary frame messages; the telemetry ``feeds`` list maps them to roles.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

FeedRole = Literal["tile", "main", "strip"]

CCTV_SUFFIX = "cctv"  # matches ``teleop.sim.cells.cctv_camera_name``
TOP_SUFFIX = "top_ortho"  # matches ``teleop.retarget.clutch.rescue_camera_names``
SIDE_SUFFIX = "side_ortho"

TILE = (640, 360)
MAIN = (960, 540)
STRIP = (384, 216)


@dataclass(frozen=True, slots=True)
class FeedSpec:
    slot: int
    camera: str
    bay: int
    role: FeedRole
    width: int
    height: int


def _feed(slot: int, bay: int, suffix: str, role: FeedRole, size: tuple[int, int]) -> FeedSpec:
    return FeedSpec(slot, f"bay{bay}/{suffix}", bay, role, size[0], size[1])


def feed_layout(bays: Sequence[int], rescue_bay: int | None) -> tuple[FeedSpec, ...]:
    """Feeds for the bays present; ``rescue_bay`` (if present) switches to rescue mode."""
    ordered = sorted(bays)
    if rescue_bay is None or rescue_bay not in ordered:
        return tuple(_feed(i, b, CCTV_SUFFIX, "tile", TILE) for i, b in enumerate(ordered))
    mains = (
        _feed(0, rescue_bay, TOP_SUFFIX, "main", MAIN),
        _feed(1, rescue_bay, SIDE_SUFFIX, "main", MAIN),
    )
    strips = tuple(_feed(2 + i, b, CCTV_SUFFIX, "strip", STRIP) for i, b in enumerate(ordered))
    return mains + strips
