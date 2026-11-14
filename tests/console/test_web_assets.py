"""The console page ships self-contained and speaks the plan's wire protocol."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import pytest

from teleop.console.server import WEB_DIR, ConsoleServer


def _read(name: str) -> str:
    return (WEB_DIR / name).read_text(encoding="utf-8")


def test_index_references_only_local_assets() -> None:
    html = _read("index.html")
    assert 'href="console.css"' in html and 'src="console.js"' in html
    assert 'id="view-overview"' in html and "4 segmented conveyors" in html
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


_FRAME_URLS_HARNESS = """
const revoked = [];
const revoke = (u) => revoked.push(u);
const s = { img: {}, url: null, pending: null };
showFrame(s, "u1", revoke); s.img.onload();          // u1 shown
showFrame(s, "u2", revoke); const lateU2 = s.img.onload;
showFrame(s, "u3", revoke);                          // u2 superseded before it loaded
s.img.onload();                                      // u3 shown, u1 freed
lateU2();                                            // a stale callback must be a no-op
showFrame(s, "u4", revoke); s.img.onerror();         // a broken frame is freed too
console.log(JSON.stringify({ revoked, url: s.url, pending: s.pending }));
"""


def test_frame_urls_are_always_revoked(tmp_path: Path) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    js = _read("console.js")
    m = re.search(r"// <frame-urls>(.*?)// </frame-urls>", js, re.DOTALL)
    assert m, "console.js must keep the frame URL helper between <frame-urls> markers"
    script = tmp_path / "harness.js"
    script.write_text(m.group(1) + _FRAME_URLS_HARNESS, encoding="utf-8")
    out = subprocess.run([node, str(script)], capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    result = json.loads(out.stdout)
    assert sorted(result["revoked"]) == ["u1", "u2", "u4"]  # every superseded URL is freed
    assert result["url"] == "u3" and result["pending"] is None
