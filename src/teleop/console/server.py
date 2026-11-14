"""Local WebSocket + static HTTP server for the fleet ops console.

One port serves the page (``GET /`` and the flat files in ``web/``) and the ``/ws`` stream.
Server -> client: telemetry JSON (text) and camera frames (binary: byte 0 = feed slot,
byte 1 = ``layout_id % 256``, then one JPEG). Client -> server: ``{"type": "key", "code": n}``
(GLFW key codes), handed to ``on_key``; ``{"type": "reconnect", "target": t}`` for ``t`` in
``RECONNECT_TARGETS``, handed to ``on_reconnect`` (which must not block).

Publishing is thread-safe and never blocks the caller. The server keeps the latest telemetry
and the latest frame per slot: a new client (or a page reload) gets them immediately, and a
slow client holds at most one pending message per slot plus one telemetry message, so frames
are dropped, not queued. With no clients, publishing only updates the latest-state slots.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import mimetypes
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast
from urllib.parse import unquote, urlsplit

from websockets.asyncio.server import Server, ServerConnection, serve
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Request, Response

logger = logging.getLogger("teleop.console.server")

WEB_DIR = Path(__file__).parent / "web"
WS_PATH = "/ws"
ALLOWED_SUFFIXES = frozenset({".html", ".css", ".js", ".svg", ".png", ".woff2"})
KEY_MIN, KEY_MAX = 32, 348  # printable GLFW keys .. GLFW_KEY_LAST
RECONNECT_TARGETS = frozenset({"hotspot", "gloves", "panel"})
START_TIMEOUT_S = 5.0

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".woff2": "font/woff2",
}


class _Client:
    """Per-connection latest-value mailbox drained by one sender task."""

    def __init__(self, ws: ServerConnection) -> None:
        self.ws = ws
        self.text: str | None = None
        self.frames: dict[int, bytes] = {}
        self.wake = asyncio.Event()

    def put_text(self, text: str) -> None:
        self.text = text
        self.wake.set()

    def put_frame(self, slot: int, payload: bytes) -> None:
        self.frames[slot] = payload
        self.wake.set()

    async def run(self) -> None:
        while True:
            await self.wake.wait()
            self.wake.clear()
            text, self.text = self.text, None
            frames, self.frames = self.frames, {}
            if text is not None:
                await self.ws.send(text)
            for slot in sorted(frames):
                await self.ws.send(frames[slot])


class ConsoleServer:
    def __init__(
        self,
        web_dir: Path = WEB_DIR,
        host: str = "127.0.0.1",
        port: int = 8765,
        on_key: Callable[[int], None] | None = None,
        on_reconnect: Callable[[str], None] | None = None,
    ) -> None:
        self.web_dir = web_dir.resolve()
        self.host = host
        self._requested_port = port
        self.on_key = on_key
        self.on_reconnect = on_reconnect
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._server: Server | None = None
        self._stop: asyncio.Event | None = None
        self._clients: set[_Client] = set()
        self._lock = threading.Lock()
        self._latest_text: str | None = None
        self._latest_frames: dict[int, bytes] = {}
        self._port: int | None = None

    # ------------------------------------------------------------------ lifecycle

    @property
    def port(self) -> int:
        if self._port is None:
            raise RuntimeError("server not started")
        return self._port

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    @property
    def client_count(self) -> int:
        return len(self._clients)

    def start(self) -> None:
        """Bind and serve on a daemon thread. Raises ``OSError`` if the port is taken."""
        ready = threading.Event()
        failure: list[BaseException] = []

        def run() -> None:
            loop = asyncio.new_event_loop()
            self._loop = loop
            try:
                loop.run_until_complete(self._main(ready))
            except BaseException as exc:  # noqa: BLE001 - surfaced to start()
                failure.append(exc)
                ready.set()
            finally:
                loop.close()

        self._thread = threading.Thread(target=run, name="console-server", daemon=True)
        self._thread.start()
        if not ready.wait(START_TIMEOUT_S):
            raise OSError("console server did not start in time")
        if failure:
            self._thread.join(timeout=START_TIMEOUT_S)
            exc = failure[0]
            raise exc if isinstance(exc, OSError) else OSError(str(exc))
        logger.info("console: serving %s", self.url)

    async def _main(self, ready: threading.Event) -> None:
        self._stop = asyncio.Event()
        async with serve(
            self._handle,
            self.host,
            self._requested_port,
            process_request=self._process_request,
            compression=None,  # JPEG frames don't deflate; skip the CPU cost
            max_size=64 * 1024,
        ) as server:
            self._server = server
            self._port = int(next(iter(server.sockets)).getsockname()[1])
            ready.set()
            await self._stop.wait()

    def stop(self) -> None:
        loop, stop = self._loop, self._stop
        if loop is not None and stop is not None and not loop.is_closed():
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(stop.set)
        if self._thread is not None:
            self._thread.join(timeout=START_TIMEOUT_S)
        self._thread = None

    # ------------------------------------------------------------------ publish (any thread)

    def publish_telemetry(self, msg: dict[str, Any]) -> None:
        text = json.dumps(msg, separators=(",", ":"))
        with self._lock:
            self._latest_text = text
        self._fanout(lambda c: c.put_text(text))

    def publish_frame(self, slot: int, layout_id: int, jpeg: bytes) -> None:
        payload = bytes((slot & 0xFF, layout_id % 256)) + jpeg
        with self._lock:
            self._latest_frames[slot] = payload
        self._fanout(lambda c: c.put_frame(slot, payload))

    def _fanout(self, put: Callable[[_Client], None]) -> None:
        loop = self._loop
        if not self._clients or loop is None or loop.is_closed():
            return

        def deliver() -> None:
            for client in list(self._clients):
                put(client)

        with contextlib.suppress(RuntimeError):  # loop closed between the check and the call
            loop.call_soon_threadsafe(deliver)

    # ------------------------------------------------------------------ connections

    async def _handle(self, ws: ServerConnection) -> None:
        client = _Client(ws)
        with self._lock:
            if self._latest_text is not None:
                client.put_text(self._latest_text)
            for slot, payload in self._latest_frames.items():
                client.put_frame(slot, payload)
        self._clients.add(client)
        sender = asyncio.create_task(client.run())
        try:
            async for message in ws:
                if isinstance(message, str):
                    self._on_message(message)
        except ConnectionClosed:
            pass
        finally:
            self._clients.discard(client)
            sender.cancel()
            with contextlib.suppress(asyncio.CancelledError, ConnectionClosed):
                await sender

    def _on_message(self, message: str) -> None:
        try:
            raw: Any = json.loads(message)
        except ValueError:
            logger.debug("console: ignoring non-JSON message")
            return
        if not isinstance(raw, dict):
            return
        msg = cast("dict[str, Any]", raw)
        if msg.get("type") == "reconnect":
            self._on_reconnect_message(msg)
            return
        code = msg.get("code")
        if msg.get("type") != "key" or not isinstance(code, int) or isinstance(code, bool):
            return
        if not KEY_MIN <= code <= KEY_MAX:
            return
        if self.on_key is not None:
            try:
                self.on_key(code)
            except Exception:
                logger.exception("console: on_key failed")

    def _on_reconnect_message(self, msg: dict[str, Any]) -> None:
        target = msg.get("target")
        if not isinstance(target, str) or target not in RECONNECT_TARGETS:
            return
        if self.on_reconnect is not None:
            try:
                self.on_reconnect(target)  # non-blocking by contract; runs on the asyncio thread
            except Exception:
                logger.exception("console: on_reconnect failed")

    # ------------------------------------------------------------------ static files

    def _process_request(self, connection: ServerConnection, request: Request) -> Response | None:
        path = urlsplit(request.path).path
        if path == WS_PATH:
            return None  # continue with the WebSocket handshake
        return self._static(path)

    def _static(self, path: str) -> Response:
        name = unquote(path).lstrip("/") or "index.html"
        suffix = Path(name).suffix.lower()
        bad = "/" in name or "\\" in name or ".." in name or suffix not in ALLOWED_SUFFIXES
        target = self.web_dir / name
        if bad or not target.is_file() or target.resolve().parent != self.web_dir:
            return _response(404, "Not Found", b"not found", "text/plain; charset=utf-8")
        ctype = _CONTENT_TYPES.get(suffix) or mimetypes.guess_type(name)[0] or "text/plain"
        return _response(200, "OK", target.read_bytes(), ctype)


def _response(status: int, reason: str, body: bytes, ctype: str) -> Response:
    headers = Headers(
        [
            ("Content-Type", ctype),
            ("Content-Length", str(len(body))),
            ("Cache-Control", "no-store"),
        ]
    )
    return Response(status, reason, headers, body)
