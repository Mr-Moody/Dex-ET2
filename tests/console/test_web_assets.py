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


def test_health_strip_assets() -> None:
    html, js = _read("index.html"), _read("health.js")
    assert 'src="health.js"' in html and 'id="health"' in html
    for token in ('"reconnect"', '"hotspot"', '"gloves"', '"panel"'):
        assert token in js, token
    assert not re.search(r"https?://", js)
    assert "/* health strip */" in _read("console.css")


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


def test_health_reconnect_button_never_keeps_focus() -> None:
    js = _read("health.js")
    assert '"mousedown"' in js and "preventDefault" in js


# A minimal DOM: enough of Element for health.js to build pills and move nodes around.
_FAKE_DOM = """
class El {
  constructor(tag) {
    this.tag = tag; this.children = []; this.parent = null; this.dataset = {};
    this.listeners = {}; this.hidden = false; this.disabled = false; this.title = "";
    this._text = ""; this.builds = 0;
  }
  get textContent() { return this._text; }
  set textContent(v) { this._text = String(v); this.children = []; }
  _adopt(k) { if (typeof k === "string") { const t = new El("#text"); t._text = k; k = t; }
    if (k.parent) k.remove(); k.parent = this; return k; }
  append(...ks) { for (const k of ks) this.children.push(this._adopt(k)); }
  prepend(...ks) { for (const k of ks.reverse()) this.children.unshift(this._adopt(k)); }
  remove() {
    if (!this.parent) return;
    const c = this.parent.children;
    c.splice(c.indexOf(this), 1);
    this.parent = null;
  }
  after(k) {
    const p = this.parent;
    k = p._adopt(k);
    p.children.splice(p.children.indexOf(this) + 1, 0, k);
  }
  replaceChildren(...ks) { this.builds++; for (const c of this.children) c.parent = null;
    this.children = []; this.append(...ks); }
  setAttribute() {}
  addEventListener(t, f) { (this.listeners[t] = this.listeners[t] || []).push(f); }
  find(cls) {
    if (this.className === cls) return this;
    for (const c of this.children) { const f = c.find(cls); if (f) return f; }
    return null;
  }
}
const strip = new El("section"); strip.hidden = true;
const handlers = [];
globalThis.document = {
  getElementById: () => strip,
  addEventListener: (_, f) => handlers.push(f),
  createElement: (tag) => new El(tag),
};
const toasts = [];
globalThis.toast = (k, t) => toasts.push(t);
const sent = [];
globalThis.WebSocket = { OPEN: 1 };
globalThis.ws = { readyState: 1, send: (m) => sent.push(JSON.parse(m)) };
"""

_HEALTH_HARNESS = (
    _FAKE_DOM
    + """
%s
const mk = (level) => [
  { name: "gloves", level, summary: "s", detail: "", can_reconnect: true, busy: false },
];
const send = (health) => handlers[0]({ detail: { health } });
send(mk("ok")); send(mk("ok")); send(mk("ok"));
const same = strip.builds;
send(mk("fail")); send(mk("fail"));
send(undefined);
const r = { same, after: strip.builds, toasts: toasts.length, hidden: strip.hidden };
console.log(JSON.stringify(r));
"""
)


def _node_json(script: str) -> object:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    out = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def test_health_strip_rebuilds_only_on_change() -> None:
    got = _node_json(_HEALTH_HARNESS % _read("health.js"))
    assert got == {"same": 1, "after": 1, "toasts": 1, "hidden": True}


_VOLATILE_HARNESS = (
    _FAKE_DOM
    + """
%s
const gloves = (n, extra) => ({ name: "gloves", level: "warn", summary: `no packets for ${n} s`,
  detail: `streaming ${n} Hz`, can_reconnect: true, busy: false, ...extra });
const panel = (n) => ({ name: "panel", level: "ok", summary: "UNO Q, buttons live",
  detail: `MCU heartbeat ${n} ms ago`, can_reconnect: false, busy: false });
const send = (health) => handlers[0]({ detail: { health } });
send([gloves(1), panel(10)]);
const pill = strip.children[0];
const btn = pill.find("hpill__btn");
const detail = pill.find("hpill__detail");
const builds = strip.builds;
send([gloves(2), panel(20)]); send([gloves(3), panel(30)]);
const r = {
  builds: strip.builds - builds,
  samePill: strip.children[0] === pill,
  sameBtn: pill.find("hpill__btn") === btn,
  sameDetail: pill.find("hpill__detail") === detail,
  summary: pill.find("hpill__sum").textContent,
  detail: detail.textContent,
  panelTitle: strip.children[1].title,
};
let prevented = false;
for (const f of btn.listeners.mousedown || []) f({ preventDefault: () => { prevented = true; } });
r.prevented = prevented;
for (const f of btn.listeners.click || []) f({});
r.sent = sent;
send([gloves(3, { busy: true }), panel(40)]);
r.busySameBtn = pill.find("hpill__btn") === btn;
r.busyDisabled = btn.disabled;
r.busySpin = !!btn.find("hpill__spin");
send([gloves(4), panel(50)]);
r.idleSpin = !!btn.find("hpill__spin");
send([gloves(4, { level: "ok" }), panel(60)]);
r.okBtn = !!pill.find("hpill__btn");
r.okDetail = !!pill.find("hpill__detail");
send([panel(70)]);
r.after = strip.children.length;
console.log(JSON.stringify(r));
"""
)


def test_health_strip_updates_pills_in_place() -> None:
    got = _node_json(_VOLATILE_HARNESS % _read("health.js"))
    assert got == {
        "builds": 0,  # volatile numbers in summary/detail never rebuild the strip
        "samePill": True,
        "sameBtn": True,  # a click mid-update still lands on the live button
        "sameDetail": True,
        "summary": "no packets for 3 s",
        "detail": "streaming 3 Hz",
        "panelTitle": "MCU heartbeat 30 ms ago",
        "prevented": True,
        "sent": [{"type": "reconnect", "target": "gloves"}],
        "busySameBtn": True,
        "busyDisabled": True,
        "busySpin": True,
        "idleSpin": False,
        "okBtn": False,
        "okDetail": False,
        "after": 1,
    }
