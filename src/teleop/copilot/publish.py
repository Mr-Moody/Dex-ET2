"""Posts the copilot's call-outs to the console page's Copilot panel (``T-215``).

This is the only module in ``teleop.copilot`` that writes to the console connection, and the only
thing it writes is ``{"type": "copilot", ...}``: text for the page to display. It cannot carry a
key press or any other control message, and the advisory guard test checks that this file stays
that narrow. The console server validates every field again before it rebroadcasts.

``DisplayOutbox.post`` may be called from any thread (call-outs come from worker threads and
timers). It never blocks and never raises: with no console connection the messages wait, at most
``MAX_PENDING`` of them, and the oldest are dropped.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import threading
from collections import deque
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:  # callouts -> frames -> feed -> publish: import for typing only
    from teleop.copilot.callouts import Callout

MAX_PENDING = 20
MESSAGE_TYPE = "copilot"


class Writer(Protocol):
    async def send(self, message: str) -> None: ...


def callout_message(callout: Callout) -> dict[str, Any]:
    """The display message for one call-out. Frames and keys never go in it."""
    return {
        "type": MESSAGE_TYPE,
        "bay": callout.bay,
        "severity": callout.severity,
        "say": callout.say,
        "detail": callout.detail,
        "source": callout.source,
    }


class DisplayOutbox:
    """Thread-safe, bounded queue of display messages, drained onto the console connection."""

    def __init__(self, max_pending: int = MAX_PENDING) -> None:
        self._pending: deque[dict[str, Any]] = deque(maxlen=max_pending)
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._wake: asyncio.Event | None = None
        self.sent = 0

    def post_callout(self, callout: Callout) -> None:
        self.post(callout_message(callout))

    def post(self, message: dict[str, Any]) -> None:
        if message.get("type") != MESSAGE_TYPE:
            raise ValueError("the outbox only carries copilot display messages")
        with self._lock:
            self._pending.append(message)
            loop, wake = self._loop, self._wake
        if loop is not None and wake is not None and not loop.is_closed():
            with contextlib.suppress(RuntimeError):  # the loop closed after the check
                loop.call_soon_threadsafe(wake.set)

    def pending(self) -> int:
        with self._lock:
            return len(self._pending)

    async def pump(self, conn: Writer) -> None:
        """Send queued messages until cancelled or the connection fails."""
        wake = asyncio.Event()
        with self._lock:
            self._wake, self._loop = wake, asyncio.get_running_loop()
            if self._pending:
                wake.set()
        try:
            while True:
                await wake.wait()
                wake.clear()
                while True:
                    with self._lock:
                        message = self._pending.popleft() if self._pending else None
                    if message is None:
                        break
                    await conn.send(json.dumps(message, separators=(",", ":")))
                    self.sent += 1
        finally:
            with self._lock:
                self._wake = self._loop = None
