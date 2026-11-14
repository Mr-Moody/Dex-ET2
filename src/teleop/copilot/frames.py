"""Latest camera frame per bay and camera, plus a short history of each rescue for reviews.

While a bay is in ``RESCUE`` with the clutch held, its frames also go into a ring buffer: at most
12 frames per bay, sampled at most every ``sample_interval_s`` per camera so the buffer spans the
whole rescue rather than its last second. The buffer is kept after the clutch is released (a
review needs it then) and cleared when that bay's next stall begins.
"""

from __future__ import annotations

import hashlib
import io
import threading
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, cast

from teleop.copilot.feed import CameraFrame

RING_FRAMES = 12
SAMPLE_INTERVAL_S = 0.5
MAX_SIDE_PX = 512
RESCUE = "RESCUE"


@dataclass(frozen=True, slots=True)
class StoredFrame:
    bay: int
    camera: str
    jpeg: bytes = field(repr=False)
    t_ns: int
    sim_t: float | None  # the sim time of the latest telemetry when the frame arrived

    @property
    def sha256(self) -> str:
        """A short hash to log in place of the image (logs never hold raw frames)."""
        return hashlib.sha256(self.jpeg).hexdigest()[:16]


def downscale_jpeg(jpeg: bytes, max_side_px: int = MAX_SIDE_PX, quality: int = 85) -> bytes:
    """The same JPEG if it already fits in ``max_side_px``, else a smaller re-encode."""
    from PIL import Image

    with Image.open(io.BytesIO(jpeg)) as img:
        if max(img.size) <= max_side_px:
            return jpeg
        small = img.convert("RGB")
        small.thumbnail((max_side_px, max_side_px))
        buf = io.BytesIO()
        small.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()


class FrameStore:
    """Thread-safe store. Feed it telemetry and frames from :class:`FeedClient` callbacks."""

    def __init__(
        self,
        ring_frames: int = RING_FRAMES,
        sample_interval_s: float = SAMPLE_INTERVAL_S,
    ) -> None:
        self._lock = threading.Lock()
        self._latest: dict[tuple[int, str], StoredFrame] = {}
        self._ring: dict[int, deque[StoredFrame]] = {}
        self._ring_frames = ring_frames
        self._interval_ns = round(sample_interval_s * 1e9)
        self._last_sample: dict[tuple[int, str], int] = {}
        self._recording: set[int] = set()
        self._states: dict[int, str] = {}
        self._sim_t: float | None = None

    def on_telemetry(self, msg: Mapping[str, Any]) -> None:
        """Track which bay is being rescued with the clutch held, and new stalls."""
        rescue = msg.get("rescue")
        held: int | None = None
        if isinstance(rescue, dict):
            r = cast(dict[str, Any], rescue)
            bay = r.get("bay")
            if isinstance(bay, int) and r.get("clutch") is True:
                held = bay
        sim_t = msg.get("sim_t")
        with self._lock:
            self._sim_t = float(sim_t) if isinstance(sim_t, int | float) else None
            states = _states(msg)
            for bay, state in states.items():
                if state == RESCUE and self._states.get(bay) not in (RESCUE, None):
                    self._ring.pop(bay, None)  # a new stall: drop the last rescue's frames
                    self._last_sample = {k: v for k, v in self._last_sample.items() if k[0] != bay}
            self._states = states
            self._recording = {held} if held is not None and states.get(held) == RESCUE else set()

    def add(self, frame: CameraFrame) -> None:
        with self._lock:
            stored = StoredFrame(frame.bay, frame.camera, frame.jpeg, frame.t_ns, self._sim_t)
            key = (frame.bay, frame.camera)
            self._latest[key] = stored
            if frame.bay not in self._recording:
                return
            last = self._last_sample.get(key)
            if last is not None and frame.t_ns - last < self._interval_ns:
                return
            self._last_sample[key] = frame.t_ns
            ring = self._ring.setdefault(frame.bay, deque(maxlen=self._ring_frames))
            ring.append(stored)

    def latest(self, bay: int, camera: str | None = None) -> StoredFrame | None:
        """The newest frame of ``bay``: from ``camera`` if given, else from any camera."""
        with self._lock:
            if camera is not None:
                return self._latest.get((bay, camera))
            frames = [f for (b, _), f in self._latest.items() if b == bay]
        return max(frames, key=lambda f: f.t_ns, default=None)

    def rescue_frames(self, bay: int) -> list[StoredFrame]:
        """The sampled frames of the bay's latest clutch hold, oldest first."""
        with self._lock:
            return list(self._ring.get(bay, ()))

    def is_recording(self, bay: int) -> bool:
        with self._lock:
            return bay in self._recording


def _states(msg: Mapping[str, Any]) -> dict[int, str]:
    out: dict[int, str] = {}
    bays = msg.get("bays")
    if not isinstance(bays, list):
        return out
    for item in cast(list[object], bays):
        if isinstance(item, dict):
            entry = cast(dict[str, Any], item)
            bay, state = entry.get("id"), entry.get("state")
            if isinstance(bay, int) and isinstance(state, str):
                out[bay] = state
    return out
