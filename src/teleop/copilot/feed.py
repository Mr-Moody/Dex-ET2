"""Read-only client for the fleet ops console stream (``ws://127.0.0.1:8765/ws``).

The console sends two kinds of message:

- **Text:** telemetry JSON (``"type": "telemetry"``, about 10 Hz).
- **Binary:** one camera frame. Byte 0 is the feed slot, byte 1 is ``layout_id % 256``, and the
  rest is one JPEG. The slot maps to a camera through the latest telemetry's ``feeds``. A frame
  whose byte 1 does not match the current layout is from an old layout and is dropped.

The client **never sends** anything: the copilot is advisory only. It reconnects with backoff
(1 s, doubling, at most 5 s) when the console stops or restarts.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterable, Awaitable, Callable, Mapping
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Any, cast

from websockets.exceptions import WebSocketException

logger = logging.getLogger(__name__)

JPEG_MAGIC = b"\xff\xd8"
MIN_BACKOFF_S = 1.0
MAX_BACKOFF_S = 5.0
MAX_MESSAGE_BYTES = 16 * 1024 * 1024

Telemetry = dict[str, Any]
Connection = AsyncIterable[str | bytes]
Connect = Callable[[str], AbstractAsyncContextManager[Connection]]


@dataclass(frozen=True, slots=True)
class FeedInfo:
    """One console feed slot: which camera of which bay it shows, and its role in the layout."""

    slot: int
    camera: str
    bay: int
    role: str


@dataclass(frozen=True, slots=True)
class CameraFrame:
    """One JPEG from the console, already mapped to its bay and camera."""

    bay: int
    camera: str
    role: str
    jpeg: bytes
    t_ns: int  # time.monotonic_ns() on arrival


def parse_telemetry(text: str) -> Telemetry | None:
    """The telemetry dict, or ``None`` for anything that is not a telemetry message."""
    try:
        value: object = json.loads(text)
    except ValueError:
        return None
    if not isinstance(value, dict):
        return None
    msg = cast(Telemetry, value)
    return msg if msg.get("type") == "telemetry" else None


def parse_feeds(msg: Mapping[str, Any]) -> dict[int, FeedInfo]:
    """``slot -> FeedInfo`` from a telemetry message's ``feeds``; bad entries are skipped."""
    feeds: dict[int, FeedInfo] = {}
    raw = msg.get("feeds")
    if not isinstance(raw, list):
        return feeds
    for item in cast(list[object], raw):
        if not isinstance(item, dict):
            continue
        entry = cast(dict[str, Any], item)
        slot, camera, bay, role = (entry.get(k) for k in ("slot", "camera", "bay", "role"))
        if isinstance(slot, int) and isinstance(camera, str) and isinstance(bay, int):
            feeds[slot] = FeedInfo(slot, camera, bay, role if isinstance(role, str) else "")
    return feeds


def split_frame(data: bytes) -> tuple[int, int, bytes] | None:
    """``(slot, layout_byte, jpeg)`` from a binary message, or ``None`` if it is not a frame."""
    if len(data) < 4 or data[2:4] != JPEG_MAGIC:
        return None
    return data[0], data[1], bytes(data[2:])


class FeedMapper:
    """Turns raw console messages into telemetry and mapped frames. Pure, no I/O.

    Frames that arrive before the first telemetry, or that carry another layout's byte, are
    dropped and counted.
    """

    def __init__(self, clock: Callable[[], int] = time.monotonic_ns) -> None:
        self._clock = clock
        self._layout_byte: int | None = None
        self._feeds: dict[int, FeedInfo] = {}
        self.dropped_frames = 0

    def reset(self) -> None:
        """Forget the layout (after a reconnect, the console may have restarted)."""
        self._layout_byte = None
        self._feeds = {}

    def on_text(self, text: str) -> Telemetry | None:
        msg = parse_telemetry(text)
        if msg is None:
            return None
        layout = msg.get("layout_id")
        if isinstance(layout, int):
            self._layout_byte = layout % 256
            self._feeds = parse_feeds(msg)
        return msg

    def on_bytes(self, data: bytes) -> CameraFrame | None:
        parts = split_frame(data)
        if parts is None:
            self.dropped_frames += 1
            return None
        slot, layout_byte, jpeg = parts
        feed = self._feeds.get(slot)
        if layout_byte != self._layout_byte or feed is None:
            self.dropped_frames += 1
            return None
        return CameraFrame(feed.bay, feed.camera, feed.role, jpeg, self._clock())


def default_connect(url: str) -> AbstractAsyncContextManager[Connection]:
    from websockets.asyncio.client import connect

    conn: Any = connect(url, max_size=MAX_MESSAGE_BYTES, open_timeout=5, ping_interval=10)
    return cast(AbstractAsyncContextManager[Connection], conn)


class FeedClient:
    """Watches the console and hands every telemetry message and camera frame to callbacks.

    Callbacks run on the event loop and must be quick (they should only store or queue).
    """

    def __init__(
        self,
        url: str,
        on_telemetry: Callable[[Telemetry], None],
        on_frame: Callable[[CameraFrame], None] | None = None,
        *,
        connect: Connect = default_connect,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        mapper: FeedMapper | None = None,
        min_backoff_s: float = MIN_BACKOFF_S,
        max_backoff_s: float = MAX_BACKOFF_S,
    ) -> None:
        self.url = url
        self._on_telemetry = on_telemetry
        self._on_frame = on_frame
        self._connect = connect
        self._sleep = sleep
        self.mapper = mapper if mapper is not None else FeedMapper()
        self._min_backoff = min_backoff_s
        self._max_backoff = max_backoff_s
        self._stopping = False
        self.connected = False
        self.sessions = 0
        self.telemetry_count = 0

    def stop(self) -> None:
        """Ask :meth:`run` to return after the current connection attempt or message."""
        self._stopping = True

    async def run(self) -> None:
        """Connect, read until the connection ends, back off, repeat until :meth:`stop`."""
        delay = self._min_backoff
        while not self._stopping:
            try:
                async with self._connect(self.url) as conn:
                    self.connected = True
                    self.sessions += 1
                    delay = self._min_backoff  # a successful connect resets the backoff
                    logger.info("copilot feed connected to %s", self.url)
                    await self._read(conn)
            except (OSError, TimeoutError, WebSocketException) as exc:
                logger.debug("copilot feed: %s", exc)
            finally:
                if self.connected:
                    logger.info("copilot feed disconnected; reconnecting")
                self.connected = False
                self.mapper.reset()
            if self._stopping:
                break
            await self._sleep(delay)
            delay = min(delay * 2.0, self._max_backoff)

    async def _read(self, conn: Connection) -> None:
        async for message in conn:
            if self._stopping:
                return
            if isinstance(message, str):
                msg = self.mapper.on_text(message)
                if msg is not None:
                    self.telemetry_count += 1
                    self._on_telemetry(msg)
            else:
                frame = self.mapper.on_bytes(message)
                if frame is not None and self._on_frame is not None:
                    self._on_frame(frame)
