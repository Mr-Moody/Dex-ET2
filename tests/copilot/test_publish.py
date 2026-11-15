"""T-215: the copilot posts display-only call-outs to the console, and nothing else."""

from __future__ import annotations

import ast
import asyncio
import json
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest

from teleop.copilot import publish
from teleop.copilot.callouts import Callout
from teleop.copilot.feed import Connection, FeedClient
from teleop.copilot.publish import MAX_PENDING, DisplayOutbox, callout_message


def _callout(say: str = "Bay 1 stalled on a flap jam.", **kw: Any) -> Callout:
    base: dict[str, Any] = {
        "bay": 1,
        "event": "stall",
        "say": say,
        "detail": "C(t) 0.21",
        "severity": "alarm",
        "source": "template",
        "frame_sha256": "abc123",
        "model": "m",
    }
    return Callout(**(base | kw))


class Socket:
    """Fake connection: yields ``incoming`` then waits to be stopped; records what is sent."""

    def __init__(self, incoming: list[str], on_sent: Any = None) -> None:
        self.incoming = incoming
        self.sent: list[str] = []
        self.on_sent = on_sent
        self.closed = asyncio.Event()

    async def send(self, message: str) -> None:
        self.sent.append(message)
        if self.on_sent is not None:
            self.on_sent(self)

    def __aiter__(self) -> AsyncIterator[str | bytes]:
        return self._gen()

    async def _gen(self) -> AsyncIterator[str | bytes]:
        for m in self.incoming:
            yield m
        await self.closed.wait()  # like a socket the console closed


def test_message_has_display_fields_only() -> None:
    msg = callout_message(_callout())
    assert msg == {
        "type": "copilot",
        "bay": 1,
        "severity": "alarm",
        "say": "Bay 1 stalled on a flap jam.",
        "detail": "C(t) 0.21",
        "source": "template",
    }
    assert "frame_sha256" not in msg and "model" not in msg


def test_outbox_rejects_anything_but_copilot_messages() -> None:
    box = DisplayOutbox()
    for bad in ({"type": "key", "code": 32}, {"code": 32}, {"type": "view"}):
        with pytest.raises(ValueError):
            box.post(bad)
    assert box.pending() == 0


def test_outbox_is_bounded_and_drops_the_oldest() -> None:
    box = DisplayOutbox()
    for i in range(MAX_PENDING + 5):
        box.post_callout(_callout(say=f"line {i}"))
    assert box.pending() == MAX_PENDING

    async def drain() -> list[str]:
        sock = Socket([])
        task = asyncio.create_task(box.pump(sock))
        for _ in range(50):
            await asyncio.sleep(0)
        task.cancel()
        return sock.sent

    sent = [json.loads(s)["say"] for s in asyncio.run(drain())]
    assert sent == [f"line {i}" for i in range(5, MAX_PENDING + 5)]


def test_pump_sends_messages_posted_from_other_threads_while_running() -> None:
    box = DisplayOutbox()

    async def go() -> list[str]:
        sock = Socket([])
        task = asyncio.create_task(box.pump(sock))
        await asyncio.sleep(0)
        threading.Thread(target=lambda: box.post_callout(_callout(say="late"))).start()
        for _ in range(200):
            await asyncio.sleep(0.005)
            if sock.sent:
                break
        task.cancel()
        return sock.sent

    assert [json.loads(s)["say"] for s in asyncio.run(go())] == ["late"]
    box.post_callout(_callout())  # after the pump ended: queued, no error
    assert box.pending() == 1


def test_feed_client_drains_the_outbox_onto_the_console_connection() -> None:
    box = DisplayOutbox()
    box.post_callout(_callout(say="waiting before connect"))
    sockets: list[Socket] = []
    client: list[FeedClient] = []

    def stop_when_sent(sock: Socket) -> None:
        if len(sock.sent) == 2:
            client[0].stop()
            sock.closed.set()

    def connect(url: str) -> Any:
        @asynccontextmanager
        async def cm() -> AsyncIterator[Connection]:
            sock = Socket([], on_sent=stop_when_sent)
            sockets.append(sock)
            box.post_callout(_callout(say="after connect"))
            yield sock

        return cm()

    async def sleep(_: float) -> None:
        return None

    fc = FeedClient("ws://test/ws", lambda m: None, connect=connect, sleep=sleep, outbox=box)
    client.append(fc)
    asyncio.run(asyncio.wait_for(fc.run(), timeout=5))
    assert [json.loads(s)["say"] for s in sockets[0].sent] == [
        "waiting before connect",
        "after connect",
    ]
    assert {json.loads(s)["type"] for s in sockets[0].sent} == {"copilot"}


def test_publish_module_can_only_write_copilot_messages() -> None:
    """The one module that writes to the console never builds a ``key`` message."""
    path = Path(publish.__file__)
    tree = ast.parse(path.read_text(), filename=str(path))
    strings = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant)}
    assert "key" not in strings and "copilot" in strings
    sends = [n for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr == "send"]
    assert len(sends) == 1  # the pump's single send of an already validated copilot message
    imports = [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert not any(m.startswith("teleop.atc") for m in imports)


def test_callout_reaches_a_console_page_through_the_real_server() -> None:
    """Copilot FeedClient -> ConsoleServer -> another client, over real WebSockets."""
    from websockets.sync.client import connect

    from teleop.console.server import ConsoleServer

    srv = ConsoleServer(port=0)
    srv.start()
    box = DisplayOutbox()
    fc = FeedClient(f"ws://127.0.0.1:{srv.port}/ws", lambda m: None, outbox=box)
    thread = threading.Thread(target=lambda: asyncio.run(fc.run()), daemon=True)
    try:
        with connect(f"ws://127.0.0.1:{srv.port}/ws", open_timeout=5) as page:
            thread.start()
            box.post_callout(_callout(say="Bay 2 stalled", bay=2, severity="attention"))
            msg = json.loads(page.recv(timeout=5))
        assert msg["type"] == "copilot" and msg["bay"] == 2 and msg["say"] == "Bay 2 stalled"
        assert msg["severity"] == "attention" and msg["seq"] == 1
    finally:
        fc.stop()
        srv.stop()
