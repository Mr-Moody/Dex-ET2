"""Which camera feeds the console shows, in which slot, at what size.

Grid mode: one CCTV tile per bay present. Rescue mode: the rescued bay's top and side
orthographic views as the two main feeds, then every bay's CCTV as a small strip. Slots are
the byte-0 ids of the binary frame messages; the telemetry ``feeds`` list maps them to roles.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

FeedRole = Literal["tile", "main", "strip"]
LayoutMode = Literal["auto", "grid", "pin"]

CCTV_SUFFIX = "cctv"  # matches ``teleop.sim.cells.cctv_camera_name``
WRIST_SUFFIX = "wrist_cam"  # matches ``teleop.sim.bays.wrist_camera_name`` (and T-191)
TOP_SUFFIX = "top_ortho"  # matches ``teleop.retarget.clutch.rescue_camera_names``
SIDE_SUFFIX = "side_ortho"
SPLIT_VIEW = "split"
WRIST_SPLIT_VIEW = "wrist_split"
VALID_SINGLE_CAMERAS = frozenset({CCTV_SUFFIX, WRIST_SUFFIX, TOP_SUFFIX, SIDE_SUFFIX})
VALID_VIEWS = frozenset(
    {CCTV_SUFFIX, WRIST_SUFFIX, TOP_SUFFIX, SIDE_SUFFIX, SPLIT_VIEW, WRIST_SPLIT_VIEW}
)

TILE = (640, 360)
MAIN = (960, 540)
STRIP = (384, 216)
ZOOM_MIN, ZOOM_MAX = 0.5, 4.0
PAN_LIMIT = 1.0


@dataclass(frozen=True, slots=True)
class FeedSpec:
    slot: int
    camera: str
    bay: int
    role: FeedRole
    width: int
    height: int
    zoom: float = 1.0
    pan_x: float = 0.0
    pan_y: float = 0.0


@dataclass(frozen=True, slots=True)
class ViewState:
    """Operator-chosen layout, pinned bay, per-bay camera angles, tile order and zoom/pan."""

    mode: LayoutMode = "auto"
    pinned_bay: int | None = None
    cameras: Mapping[int, str] = field(default_factory=dict[int, str])
    bay_order: tuple[int, ...] = (1, 2, 3, 4)
    zooms: Mapping[str, tuple[float, float, float]] = field(
        default_factory=dict[str, tuple[float, float, float]]
    )

    def camera_for(self, bay: int, default: str = CCTV_SUFFIX) -> str:
        cam = self.cameras.get(bay, default)
        return cam if cam in VALID_VIEWS else default

    def zoom_for(self, camera_name: str) -> tuple[float, float, float]:
        z, px, py = self.zooms.get(camera_name, (1.0, 0.0, 0.0))
        return (
            min(ZOOM_MAX, max(ZOOM_MIN, float(z))),
            min(PAN_LIMIT, max(-PAN_LIMIT, float(px))),
            min(PAN_LIMIT, max(-PAN_LIMIT, float(py))),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "pinned_bay": self.pinned_bay,
            "cameras": {str(k): v for k, v in self.cameras.items() if v in VALID_VIEWS},
            "bay_order": list(self.bay_order),
            "zooms": {
                k: [round(v[0], 3), round(v[1], 3), round(v[2], 3)]
                for k, v in self.zooms.items()
                if abs(v[0] - 1.0) > 1e-3 or abs(v[1]) > 1e-3 or abs(v[2]) > 1e-3
            },
        }


def _feed(
    slot: int,
    bay: int,
    suffix: str,
    role: FeedRole,
    size: tuple[int, int],
    state: ViewState | None = None,
) -> FeedSpec:
    cam_name = f"bay{bay}/{suffix}"
    zoom, px, py = (
        state.zoom_for(cam_name) if state is not None and role != "strip" else (1.0, 0.0, 0.0)
    )
    return FeedSpec(slot, cam_name, bay, role, size[0], size[1], zoom, px, py)


def _ordered_bays(bays: Sequence[int], state: ViewState | None) -> list[int]:
    present = sorted(set(bays))
    if state is None:
        return present
    seen: list[int] = [b for b in state.bay_order if b in present]
    for b in present:
        if b not in seen:
            seen.append(b)
    return seen


def feed_layout(
    bays: Sequence[int],
    rescue_bay: int | None,
    view_state: ViewState | None = None,
) -> tuple[FeedSpec, ...]:
    """Feeds for the bays present; supports equal grid, auto-rescue focus, and manual pinning."""
    ordered = _ordered_bays(bays, view_state)
    mode: LayoutMode = view_state.mode if view_state is not None else "auto"

    focus_bay: int | None = None
    if mode == "grid":
        focus_bay = None
    elif mode == "pin":
        cand = view_state.pinned_bay if view_state is not None else None
        focus_bay = cand if cand in ordered else None
    else:
        focus_bay = rescue_bay if (rescue_bay is not None and rescue_bay in ordered) else None

    if focus_bay is None:
        tiles: list[FeedSpec] = []
        for i, b in enumerate(ordered):
            cam = view_state.camera_for(b, CCTV_SUFFIX) if view_state is not None else CCTV_SUFFIX
            if cam not in VALID_SINGLE_CAMERAS:
                cam = CCTV_SUFFIX
            tiles.append(_feed(i, b, cam, "tile", TILE, view_state))
        return tuple(tiles)

    default_focus_cam = SPLIT_VIEW if focus_bay == rescue_bay else CCTV_SUFFIX
    focus_cam = (
        view_state.camera_for(focus_bay, default_focus_cam)
        if view_state is not None
        else default_focus_cam
    )
    if focus_cam == SPLIT_VIEW:
        mains: tuple[FeedSpec, ...] = (
            _feed(0, focus_bay, TOP_SUFFIX, "main", MAIN, view_state),
            _feed(1, focus_bay, SIDE_SUFFIX, "main", MAIN, view_state),
        )
    elif focus_cam == WRIST_SPLIT_VIEW:
        mains = (
            _feed(0, focus_bay, WRIST_SUFFIX, "main", MAIN, view_state),
            _feed(1, focus_bay, TOP_SUFFIX, "main", MAIN, view_state),
        )
    else:
        mains = (_feed(0, focus_bay, focus_cam, "main", MAIN, view_state),)

    base_slot = len(mains)
    strips = tuple(
        _feed(base_slot + i, b, CCTV_SUFFIX, "strip", STRIP, view_state)
        for i, b in enumerate(ordered)
    )
    return mains + strips
