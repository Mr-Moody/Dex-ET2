"""Console WebSocket/HTTP server: static files, latest-state replay, bounded buffering, keys."""

from __future__ import annotations

import json
import socket
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import pytest
from websockets.sync.client import ClientConnection, connect

from teleop.console.server import ConsoleServer


@pytest.fixture
def web_dir(tmp_path: Path) -> Path:
    d = tmp_path / "web"
    d.mkdir()
    (d / "index.html").write_text("<!doctype html><title>ops</title>", encoding="utf-8")
    (d / "console.css").write_text("body{}", encoding="utf-8")
    (d / "console.js").write_text("// js", encoding="utf-8")
    (d / "secret.py").write_text("x = 1", encoding="utf-8")
    (tmp_path / "outside.txt").write_text("nope", encoding="utf-8")
    return d


@pytest.fixture
def keys() -> list[int]:
    return []


@pytest.fixture
def server(web_dir: Path, keys: list[int]) -> Iterator[ConsoleServer]:
    srv = ConsoleServer(web_dir=web_dir, port=0, on_key=keys.append)
    srv.start()
    yield srv
    srv.stop()


def _get(srv: ConsoleServer, path: str) -> tuple[int, str, bytes]:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{srv.port}{path}", timeout=5) as r:
            return r.status, r.headers.get("Content-Type", ""), r.read()
    except urllib.error.HTTPError as exc:
        return exc.code, "", b""


def _ws(srv: ConsoleServer) -> ClientConnection:
    return connect(f"ws://127.0.0.1:{srv.port}/ws", max_size=None, max_queue=None, open_timeout=5)


def _wait(cond: object, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():  # type: ignore[operator]
            return
        time.sleep(0.01)
    raise AssertionError("condition not met in time")


def test_serves_index_and_assets(server: ConsoleServer) -> None:
    status, ctype, body = _get(server, "/")
    assert status == 200 and ctype.startswith("text/html") and b"<title>ops" in body
    status, ctype, _ = _get(server, "/console.css")
    assert status == 200 and ctype.startswith("text/css")
    status, ctype, _ = _get(server, "/console.js")
    assert status == 200 and "javascript" in ctype
    assert server.url == f"http://127.0.0.1:{server.port}/"


@pytest.mark.parametrize(
    "path", ["/secret.py", "/../outside.txt", "/%2e%2e/outside.txt", "/sub/x.css", "/missing.js"]
)
def test_rejects_other_paths(server: ConsoleServer, path: str) -> None:
    assert _get(server, path)[0] == 404


def test_new_client_gets_latest_state(server: ConsoleServer) -> None:
    server.publish_telemetry({"type": "telemetry", "layout_id": 1, "n": 1})
    server.publish_telemetry({"type": "telemetry", "layout_id": 1, "n": 2})
    server.publish_frame(0, 1, b"\xff\xd8old")
    server.publish_frame(0, 1, b"\xff\xd8new")
    server.publish_frame(1, 1, b"\xff\xd8one")
    with _ws(server) as ws:
        got = [ws.recv(timeout=5) for _ in range(3)]
    texts = [json.loads(m) for m in got if isinstance(m, str)]
    frames = sorted(m for m in got if isinstance(m, bytes))
    assert texts == [{"type": "telemetry", "layout_id": 1, "n": 2}]
    assert frames == sorted([bytes([0, 1]) + b"\xff\xd8new", bytes([1, 1]) + b"\xff\xd8one"])


def test_live_frames_carry_slot_and_layout_id(server: ConsoleServer) -> None:
    with _ws(server) as ws:
        _wait(lambda: server.client_count == 1)
        server.publish_frame(5, 258, b"\xff\xd8x")
        assert ws.recv(timeout=5) == bytes([5, 2]) + b"\xff\xd8x"


def test_key_messages_reach_callback(server: ConsoleServer, keys: list[int]) -> None:
    with _ws(server) as ws:
        ws.send("not json")
        ws.send(json.dumps({"type": "key", "code": 9999}))
        ws.send(json.dumps({"type": "key", "code": "49"}))
        ws.send(json.dumps({"type": "other", "code": 50}))
        ws.send(json.dumps({"type": "key", "code": 49}))
        ws.send(json.dumps({"type": "key", "code": 263}))  # GLFW Left arrow moves one panel left
        _wait(lambda: keys == [49, 263])
        time.sleep(0.1)
    assert keys == [49, 263]


def test_publish_without_clients_is_noop(server: ConsoleServer) -> None:
    for i in range(10_000):
        server.publish_frame(i % 4, 1, b"\xff\xd8" + bytes(100))
        server.publish_telemetry({"type": "telemetry", "i": i})
    assert server.client_count == 0


def test_slow_client_keeps_only_latest(server: ConsoleServer) -> None:
    frame = bytes(200_000)
    # max_queue=1: the sync client otherwise pre-reads 16 messages on its own thread.
    url = f"ws://127.0.0.1:{server.port}/ws"
    with connect(url, max_size=None, max_queue=1, open_timeout=5) as ws:
        _wait(lambda: server.client_count == 1)
        for i in range(500):
            server.publish_frame(0, 1, i.to_bytes(4, "big") + frame)
        time.sleep(0.5)
        received: list[bytes] = []
        while True:
            try:
                msg = ws.recv(timeout=1.0)
            except TimeoutError:
                break
            assert isinstance(msg, bytes)
            received.append(msg)
    assert received, "no frames arrived"
    assert int.from_bytes(received[-1][2:6], "big") == 499  # the newest frame always lands
    # Only what fits in the OS socket buffers plus one pending frame: dropped, not queued.
    assert len(received) <= 40


def test_disconnect_is_clean(server: ConsoleServer) -> None:
    with _ws(server):
        _wait(lambda: server.client_count == 1)
    _wait(lambda: server.client_count == 0)
    server.publish_frame(0, 1, b"\xff\xd8")


def test_port_in_use_raises(web_dir: Path) -> None:
    blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    blocker.bind(("127.0.0.1", 0))
    blocker.listen()
    port = blocker.getsockname()[1]
    try:
        srv = ConsoleServer(web_dir=web_dir, port=port)
        with pytest.raises(OSError):
            srv.start()
        assert not any(t.name == "console-server" and t.is_alive() for t in threading.enumerate())
    finally:
        blocker.close()


def test_publish_from_many_threads(server: ConsoleServer) -> None:
    with _ws(server) as ws:
        _wait(lambda: server.client_count == 1)

        def spam(slot: int) -> None:
            for _ in range(200):
                server.publish_frame(slot, 1, b"\xff\xd8" + bytes([slot]))

        threads = [threading.Thread(target=spam, args=(s,)) for s in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        seen: set[int] = set()
        deadline = time.monotonic() + 5
        while len(seen) < 4 and time.monotonic() < deadline:
            msg = ws.recv(timeout=5)
            assert isinstance(msg, bytes)
            seen.add(msg[0])
    assert seen == {0, 1, 2, 3}


def test_reconnect_message_validation(web_dir: Path, keys: list[int]) -> None:
    got: list[str] = []
    srv = ConsoleServer(web_dir=web_dir, port=0, on_key=keys.append, on_reconnect=got.append)
    srv._on_message(json.dumps({"type": "reconnect", "target": "gloves"}))
    assert got == ["gloves"]
    for bad in (
        {"type": "reconnect", "target": "toaster"},
        {"type": "reconnect", "target": 5},
        {"type": "reconnect", "target": ["gloves"]},
        {"type": "reconnect"},
    ):
        srv._on_message(json.dumps(bad))
    assert got == ["gloves"]
    srv._on_message(json.dumps({"type": "key", "code": 49}))
    assert keys == [49]
    for target in ("hotspot", "panel"):
        srv._on_message(json.dumps({"type": "reconnect", "target": target}))
    assert got == ["gloves", "hotspot", "panel"]

    plain = ConsoleServer(web_dir=web_dir, port=0)
    plain._on_message(json.dumps({"type": "reconnect", "target": "gloves"}))  # no callback: no-op


def test_client_view_and_dummy_messages_reach_callback(web_dir: Path) -> None:
    got: list[dict[str, object]] = []
    srv = ConsoleServer(web_dir=web_dir, port=0, on_client_message=got.append)
    srv.start()
    try:
        with _ws(srv) as ws:
            ws.send(json.dumps({"type": "view", "mode": "pin", "pinned_bay": 2}))
            ws.send(json.dumps({"type": "dummy", "mode": "auto"}))
            ws.send(json.dumps({"type": "unknown", "x": 1}))
            _wait(lambda: len(got) == 2)
        assert got[0]["type"] == "view" and got[0]["pinned_bay"] == 2
        assert got[1]["type"] == "dummy" and got[1]["mode"] == "auto"
    finally:
        srv.stop()


# -- copilot display messages (T-215) -------------------------------------------------------------


def _note(**kw: object) -> str:
    return json.dumps({"type": "copilot", "bay": 1, "say": "Bay 1 stalled", **kw})


def _recv_json(ws: ClientConnection) -> dict[str, object]:
    msg = ws.recv(timeout=5)
    assert isinstance(msg, str)
    return json.loads(msg)


def test_copilot_message_is_stamped_and_rebroadcast_to_others_only(
    server: ConsoleServer, keys: list[int]
) -> None:
    with _ws(server) as copilot, _ws(server) as page:
        _wait(lambda: server.client_count == 2)
        copilot.send(_note(severity="alarm", detail="flap jam", verdict="flag", source="model"))
        got = _recv_json(page)
        assert got["type"] == "copilot" and got["seq"] == 1
        assert got["bay"] == 1 and got["severity"] == "alarm" and got["say"] == "Bay 1 stalled"
        assert got["detail"] == "flap jam" and got["verdict"] == "flag"
        assert got["source"] == "model" and isinstance(got["wall"], str)
        # the sender does not get its own message back
        server.publish_telemetry({"type": "telemetry", "n": 1})
        assert _recv_json(copilot) == {"type": "telemetry", "n": 1}
    assert keys == []  # a copilot message is display only: it never reaches the key handler


def test_copilot_messages_are_replayed_to_a_new_client_and_capped(server: ConsoleServer) -> None:
    with _ws(server) as copilot:
        for i in range(25):
            copilot.send(_note(say=f"line {i}"))
        _wait(lambda: server._copilot_seq == 25)  # pyright: ignore[reportPrivateUsage]
    with _ws(server) as page:
        got = [_recv_json(page) for _ in range(20)]
    assert [m["say"] for m in got] == [f"line {i}" for i in range(5, 25)]
    assert [m["seq"] for m in got] == list(range(6, 26))


@pytest.mark.parametrize(
    "bad",
    [
        {"say": ""},
        {"say": 5},
        {"bay": 0},
        {"bay": 5},
        {"bay": "1"},
        {"bay": True},
        {"severity": "panic"},
        {"severity": ["info"]},
    ],
)
def test_malformed_copilot_messages_are_dropped(web_dir: Path, bad: dict[str, object]) -> None:
    srv = ConsoleServer(web_dir=web_dir, port=0)
    srv._on_message(json.dumps({"type": "copilot", "bay": 1, "say": "ok"} | bad))  # pyright: ignore[reportPrivateUsage]
    assert srv._copilot_seq == 0 and not srv._copilot  # pyright: ignore[reportPrivateUsage]


def test_copilot_fields_are_whitelisted_and_bounded(web_dir: Path) -> None:
    srv = ConsoleServer(web_dir=web_dir, port=0)
    srv._on_message(  # pyright: ignore[reportPrivateUsage]
        _note(
            say="x" * 1000,
            detail="y" * 5000,
            verdict="maybe",
            code=65,
            extra="<script>",
            bay=None,
        )
    )
    (text,) = srv._copilot  # pyright: ignore[reportPrivateUsage]
    note = json.loads(text)
    assert len(note["say"]) == 240 and len(note["detail"]) == 600
    assert note["verdict"] is None and note["bay"] is None
    assert set(note) == {
        "type", "bay", "severity", "say", "detail", "source", "verdict", "seq", "wall"
    }  # fmt: skip
