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
    # arrows move between panels and must not scroll the page or steal arrows from sliders/inputs
    assert "ArrowLeft" in js and "k in ARROW_CODES) event.preventDefault()" in js
    assert "ownsArrowKeys(event.target)" in js and '"INPUT"' in js


def _marked(js: str, name: str) -> str:
    m = re.search(rf"// <{name}>(.*?)// </{name}>", js, re.DOTALL)
    assert m, f"console.js must keep the {name} block between <{name}> markers"
    return m.group(1)


_BAY_KEYS_HARNESS = """
const BAYS = [1, 2, 3, 4];
%s
const arrows = Object.fromEntries(Object.entries(ARROW_CODES));
const bays = Object.fromEntries(BAYS.map((b) => [b, bayKey(b).code]));
console.log(JSON.stringify({ arrows, bays }));
"""


def test_arrow_keys_match_the_stage_keymap() -> None:
    from teleop.atc.stage import BAY_KEYS, KEYMAP, StageAction

    got = _node_json(_BAY_KEYS_HARNESS % _marked(_read("console.js"), "bay-keys"))
    assert isinstance(got, dict)
    # GLFW codes the stage maps to panel moves; bays are selected with their digit keys
    assert got["arrows"] == {"ArrowLeft": 263, "ArrowRight": 262, "ArrowUp": 265, "ArrowDown": 264}
    for key, code in got["arrows"].items():
        assert KEYMAP[code] is StageAction(f"move_{key.removeprefix('Arrow').lower()}")
    assert {int(b): c for b, c in got["bays"].items()} == BAY_KEYS
    for bay, code in BAY_KEYS.items():
        assert KEYMAP[code] is StageAction(f"select_bay_{bay}")


# Fake elements whose closest() matches selectors from the element's own ancestor chain.
_DBLCLICK_HARNESS = """
class Element {
  constructor(...chain) { this.chain = chain; }
  closest(sel) {
    const parts = sel.split(",").map((p) => p.trim());
    return this.chain.some((c) => parts.includes(c)) ? this : null;
  }
}
%s
const pic = new Element(".feed__pic");
const cases = {
  picture: pic,
  image: new Element(".feed__pic"),
  zoomButton: new Element("button", ".feed__tools", ".feed__pic"),
  toolsGap: new Element(".feed__tools", ".feed__pic"),
  zoomBadge: new Element("button", ".feed__top"),
  name: new Element(".feed__top"),
  bottomLabel: new Element(".feed__bottom > *", ".feed__pic"),
  signalLost: new Element(".feed__lost", ".feed__pic"),
  powerCut: new Element(".feed__power", ".feed__pic"),
  notElement: {},
};
console.log(JSON.stringify(Object.fromEntries(
  Object.entries(cases).map(([k, el]) => [k, onBarePicture(el)]))));
"""


def test_only_the_bare_picture_pins_on_double_click() -> None:
    got = _node_json(_DBLCLICK_HARNESS % _marked(_read("console.js"), "feed-dblclick"))
    assert isinstance(got, dict)
    assert {k for k, v in got.items() if v} == {"picture", "image"}


def test_feed_overlays_do_not_select_text_and_buttons_are_named() -> None:
    css, js = _read("console.css"), _read("console.js")
    assert "user-select: none" in css and "touch-action: manipulation" in css
    assert 'btn.setAttribute("aria-label", tip)' in js


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


def test_glove_calibration_panel() -> None:
    html, js = _read("index.html"), _read("console.js")
    for token in ('id="panel-calib"', 'id="calib-go"', 'id="calib-flat"', 'id="calib-fist"'):
        assert token in html, token
    assert html.count('class="calib__icon"') == 2  # inline stroke SVGs, no emoji or icon CDN
    for token in ('type: "calib"', '"start"', '"cancel"', "t.calib"):
        assert token in js, token
    assert (
        "/* ---------------------------------------------------------------- glove calibration */"
        in _read("console.css")
    )


def test_data_page_assets() -> None:
    html, js = _read("index.html"), _read("data.js")
    assert 'src="data.js"' in html
    for token in ('id="open-data"', 'id="page-data"', 'id="data-rec"', 'id="data-list"'):
        assert token in html, token
    for token in ('type: "record"', '"start"', '"stop"', '"delete"', '"telemetry"', "#data"):
        assert token in js, token
    assert not re.search(r"https?://", js)
    assert "data page */" in _read("console.css")


_COPILOT_HARNESS = """
%s
const m = (seq, bay, extra = {}) => ({ seq, bay, say: `s${seq}`, severity: "info", ...extra });
let h = [];
for (let i = 1; i <= 25; i++) h = copilotAdd(h, m(i, 1));
const mixed = [m(1, 1), m(2, 2), m(3, 1, { verdict: "flag" }), m(4, 1), m(5, null), m(6, 2)];
console.log(JSON.stringify({
  kept: h.map((x) => x.seq),
  rows: copilotRows(mixed).map((x) => x.seq),
  empty: copilotRows([]),
}));
"""


def test_copilot_panel_assets_and_row_selection() -> None:
    html, js, css = _read("index.html"), _read("console.js"), _read("console.css")
    assert 'id="panel-copilot"' in html and 'id="copilot-list"' in html
    assert 'msg.type === "copilot"' in js and "updateCopilot()" in js
    assert ".cop " in css and ".cop__verdict" in css
    # server text is shown as text, never parsed as markup
    block = js[js.index("function updateCopilot") : js.index("function syncViewBar")]
    assert "innerHTML" not in block
    got = _node_json(_COPILOT_HARNESS % _marked(js, "copilot"))
    assert isinstance(got, dict)
    assert got["kept"] == list(range(6, 26))  # the last 20
    # newest first; per bay one call-out and one review (seq 3 is Bay 1's review)
    assert got["rows"] == [6, 5, 4, 3] and got["empty"] == []
