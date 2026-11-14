"""The console page ships self-contained and speaks the plan's wire protocol."""

from __future__ import annotations

import re
import urllib.request
from collections.abc import Iterator

import pytest

from teleop.console.server import WEB_DIR, ConsoleServer


def _read(name: str) -> str:
    return (WEB_DIR / name).read_text(encoding="utf-8")


def test_index_references_only_local_assets() -> None:
    html = _read("index.html")
    assert 'href="console.css"' in html and 'src="console.js"' in html
    for name in ("index.html", "console.css", "console.js"):
        assert not re.search(r"https?://", _read(name)), f"{name} references the network"


def test_js_speaks_protocol() -> None:
    js = _read("console.js")
    for token in ('"/ws"', "layout_id", '"key"', "Signal lost", "arraybuffer"):
        assert token in js, token


def test_js_maps_keys_to_glfw_codes() -> None:
    js = _read("console.js")
    # digits 48-57, letters upper-case 65-90, space 32; no auto-repeat
    assert "event.repeat" in js
    assert "toUpperCase" in js and "32" in js


@pytest.fixture
def server() -> Iterator[ConsoleServer]:
    srv = ConsoleServer(port=0)
    srv.start()
    yield srv
    srv.stop()


@pytest.mark.parametrize(
    ("path", "ctype"),
    [("/", "text/html"), ("/console.css", "text/css"), ("/console.js", "javascript")],
)
def test_server_serves_assets(server: ConsoleServer, path: str, ctype: str) -> None:
    with urllib.request.urlopen(f"{server.url.rstrip('/')}{path}", timeout=5) as r:
        assert r.status == 200
        assert ctype in r.headers.get("Content-Type", "")
