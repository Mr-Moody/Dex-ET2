"""Console HTTP/WebSocket server and HudView -> JSON telemetry serialization."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from websockets.sync.client import ClientConnection, connect

from teleop.atc.hud import GateStatus, HudView, RescueStatus
from teleop.atc.types import BaySnapshot, BayState
from teleop.console.layout import feed_layout
from teleop.console.server import ConsoleServer
from teleop.console.telemetry import TelemetryAdapter


@pytest.fixture
def web_root(tmp_path: Path) -> Path:
    d = tmp_path / "web"
    d.mkdir()
    (d / "index.html").write_text("<!doctype html><title>ops</title>", encoding="utf-8")
    (d / "console.css").write_text("body{}", encoding="utf-8")
    (d / "console.js").write_text("// js", encoding="utf-8")
    (d / "secret.py").write_text("x = 1", encoding="utf-8")
    (tmp_path / "outside.txt").write_text("nope", encoding="utf-8")
    return d


@pytest.fixture
def server_events() -> dict[str, list[Any]]:
    return {"keys": [], "reconnects": [], "client_msgs": []}


@pytest.fixture
def server(web_root: Path, server_events: dict[str, list[Any]]) -> Iterator[ConsoleServer]:
    srv = ConsoleServer(
        web_dir=web_root,
        port=0,
        on_key=server_events["keys"].append,
        on_reconnect=server_events["reconnects"].append,
        on_client_message=server_events["client_msgs"].append,
    )
    srv.start()
    yield srv
    srv.stop()


def _http_get(srv: ConsoleServer, path: str) -> tuple[int, str, bytes]:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{srv.port}{path}", timeout=5) as r:
            return r.status, r.headers.get("Content-Type", ""), r.read()
    except urllib.error.HTTPError as exc:
        return exc.code, "", b""


def _ws(srv: ConsoleServer, *, max_queue: int | None = None) -> ClientConnection:
    return connect(
        f"ws://127.0.0.1:{srv.port}/ws", max_size=None, max_queue=max_queue, open_timeout=5
    )


def _wait(cond: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return
        time.sleep(0.01)
    raise AssertionError("condition not met in time")


def test_http_serves_whitelisted_assets_and_blocks_traversal(server: ConsoleServer) -> None:
    status, ctype, body = _http_get(server, "/")
    assert status == 200 and ctype.startswith("text/html") and b"<title>ops" in body
    assert _http_get(server, "/console.css")[0] == 200
    assert _http_get(server, "/console.js")[0] == 200
    for bad in ("/secret.py", "/../outside.txt", "/%2e%2e/outside.txt", "/sub/x.css", "/nope.js"):
        assert _http_get(server, bad)[0] == 404


def test_ws_replays_latest_state_and_streams_slot_headers(server: ConsoleServer) -> None:
    server.publish_telemetry({"type": "telemetry", "layout_id": 1, "n": 1})
    server.publish_telemetry({"type": "telemetry", "layout_id": 1, "n": 2})
    server.publish_frame(0, 1, b"\xff\xd8old")
    server.publish_frame(0, 1, b"\xff\xd8new")
    server.publish_frame(1, 258, b"\xff\xd8slot1")
    with _ws(server) as ws:
        got = [ws.recv(timeout=5) for _ in range(3)]
        texts = [json.loads(m) for m in got if isinstance(m, str)]
        frames = sorted(m for m in got if isinstance(m, bytes))
        assert texts == [{"type": "telemetry", "layout_id": 1, "n": 2}]
        assert frames == sorted([bytes([0, 1]) + b"\xff\xd8new", bytes([1, 2]) + b"\xff\xd8slot1"])


def test_ws_dispatches_keys_reconnect_view_and_copilot_messages(
    server: ConsoleServer, server_events: dict[str, list[Any]]
) -> None:
    with _ws(server) as sender, _ws(server) as viewer:
        _wait(lambda: server.client_count == 2)
        # Malformed payloads are ignored; valid key, reconnect, and view messages reach callbacks.
        sender.send("not json")
        sender.send(json.dumps({"type": "key", "code": 9999}))
        sender.send(json.dumps({"type": "key", "code": 49}))
        sender.send(json.dumps({"type": "reconnect", "target": "toaster"}))
        sender.send(json.dumps({"type": "reconnect", "target": "gloves"}))
        sender.send(json.dumps({"type": "view", "mode": "pin", "pinned_bay": 2}))
        _wait(
            lambda: (
                server_events["keys"] == [49]
                and server_events["reconnects"] == ["gloves"]
                and len(server_events["client_msgs"]) == 1
            )
        )
        assert server_events["client_msgs"][0]["pinned_bay"] == 2

        # Malformed copilot message dropped; valid copilot message stamped & broadcast to others.
        sender.send(json.dumps({"type": "copilot", "bay": 99, "say": "bad bay"}))
        sender.send(
            json.dumps(
                {
                    "type": "copilot",
                    "bay": 1,
                    "severity": "alarm",
                    "say": "x" * 500,
                    "verdict": "flag",
                    "extra": "ignored",
                }
            )
        )
        raw_note = viewer.recv(timeout=5)
        assert isinstance(raw_note, str)
        note = json.loads(raw_note)
        assert note["type"] == "copilot" and note["seq"] == 1 and note["bay"] == 1
        assert len(note["say"]) == 240 and "extra" not in note
        assert server_events["keys"] == [49]  # copilot messages never trigger key callbacks


def test_slow_client_drops_stale_frames_and_keeps_newest(server: ConsoleServer) -> None:
    pad = bytes(200_000)
    with _ws(server, max_queue=1) as ws:
        _wait(lambda: server.client_count == 1)
        for i in range(400):
            server.publish_frame(0, 1, i.to_bytes(4, "big") + pad)
        time.sleep(0.4)
        received: list[bytes] = []
        while True:
            try:
                msg = ws.recv(timeout=0.8)
            except TimeoutError:
                break
            assert isinstance(msg, bytes)
            received.append(msg)
    assert received and int.from_bytes(received[-1][2:6], "big") == 399
    assert len(received) <= 40


def test_telemetry_adapter_serializes_hud_view_without_nan() -> None:
    adapter = TelemetryAdapter(
        (1, 2, 3, 4), (3, 4), clock=lambda: datetime(2026, 11, 15, 10, 2, 31)
    )
    grid = feed_layout([1, 2, 3, 4], None)
    snap1 = BaySnapshot(1, BayState.NOMINAL, 0.95, 0.0, 0.5, "EV-LATCH-01")
    snap2 = BaySnapshot(2, BayState.NOMINAL, 0.95, 0.0, 0.5, "EV-LATCH-01")
    base = HudView(
        bays=(snap1, snap2),
        queue=(),
        background_bays=(3, 4),
        compliance="SOLANA  no events yet",
        reasoning="TRIAGE  idle",
    )
    adapter.build(base, 1.0, grid, {3: 0.91, 4: 0.97})

    rescue_view = replace(
        base,
        bays=(BaySnapshot(1, BayState.RESCUE, 0.12, 0.035, 0.5, "EV-LATCH-01"), snap2),
        selected_bay=1,
        rescue=RescueStatus(1, True, 3.2),
        gate=GateStatus(12.0, 20, 20, float("nan"), True),
    )
    msg = adapter.build(
        rescue_view,
        float("nan"),
        feed_layout([1, 2, 3, 4], 1),
        {3: float("nan"), 4: float("inf")},
    )
    encoded = json.dumps(msg, allow_nan=False)
    assert encoded and msg["mode"] == "rescue"
    assert msg["rescue"] == {"bay": 1, "clutch": True, "elapsed_s": 3.2}
    assert msg["gate"]["elapsed_s"] is None  # type: ignore[index]
