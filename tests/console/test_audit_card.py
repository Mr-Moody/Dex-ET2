"""Second-opinion audit card (audit.js) and the gate's VLM verdict line (console.js updateGate)."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections.abc import Iterator
from typing import Any

import pytest
from websockets.sync.client import connect

from teleop.atc.hud import GateStatus, HudView
from teleop.console.server import WEB_DIR, ConsoleServer
from teleop.console.telemetry import TelemetryAdapter


def _read(name: str) -> str:
    return (WEB_DIR / name).read_text(encoding="utf-8")


# Enough of Element for audit.js and updateGate: nested text, dataset, buttons, rebuild counts.
_DOM = r"""
class El {
  constructor(tag) {
    this.tag = tag; this.children = []; this.parent = null; this.dataset = {};
    this.listeners = {}; this.hidden = false; this._text = ""; this.builds = 0;
    this.tabIndex = tag === "button" ? 0 : -1; this.className = "";
  }
  get textContent() {
    return this.children.length ? this.children.map((c) => c.textContent).join(" ") : this._text;
  }
  set textContent(v) { this._text = String(v); this.children = []; }
  _adopt(k) { if (typeof k === "string") { const t = new El("#text"); t._text = k; k = t; }
    k.parent = this; return k; }
  append(...ks) { for (const k of ks) this.children.push(this._adopt(k)); }
  replaceChildren(...ks) { this.builds++; this.children = []; this.append(...ks); }
  setAttribute() {}
  addEventListener(t, f) { (this.listeners[t] = this.listeners[t] || []).push(f); }
  walk(ok) { return this.children.flatMap((c) => (ok(c) ? [c] : []).concat(c.walk(ok))); }
  all(tag) { return this.walk((c) => c.tag === tag); }
  byClass(cls) { return this.walk((c) => c.className === cls); }
}
const els = {
  "audit-panel": new El("section"), gate: new El("div"), "panel-gate": new El("section"),
};
els["audit-panel"].hidden = true;
const handlers = [];
globalThis.window = globalThis;
globalThis.document = {
  getElementById: (id) => els[id] || null,
  addEventListener: (_, f) => handlers.push(f),
  createElement: (tag) => new El(tag),
};
globalThis.$ = (id) => els[id];
const send = (audit) => handlers.forEach((f) => f({ detail: { audit } }));
const panel = els["audit-panel"];
"""

_AUDIT_HARNESS = (
    _DOM
    + """
%s
const base = {
  audit_id: "bay1-1", status: "pending", bay: 1, skill_id: "skill_007", episode_uid: "ep-1",
  verdict: null, dataset_tag: null, confidence: null, reasons: [], operator_note: "",
  model: "gemini-4-argon", frame_count: 6, latency_ms: null, wall: "19:00:01",
};
const out = {};
send(null); out.nullHidden = panel.hidden;
send(base); out.pendingHidden = panel.hidden; out.pending = panel.textContent;
const builds = panel.builds; send(base); send(base); out.stable = panel.builds === builds;
send({ ...base, status: "complete", verdict: "approve", dataset_tag: "approve", confidence: 0.94,
  reasons: ["Flap depressed cleanly", "Bolt cleared 5.6 cm"], operator_note: "Keep for ACT.",
  latency_ms: 8100 });
out.approve = panel.textContent; out.tone = panel.dataset.tone;
out.reasons = panel.byClass("argon__reason").length;
const btns = panel.all("button");
out.buttons = btns.map((b) => {
  let prevented = false;
  (b.listeners.mousedown || []).forEach((f) => f({ preventDefault: () => { prevented = true; } }));
  return { tab: b.tabIndex, prevented };
});
btns[0].listeners.click.forEach((f) => f());
out.dismissed = panel.hidden;
send({ ...base, status: "complete", verdict: "approve", dataset_tag: "approve", confidence: 0.94 });
out.staysDismissed = panel.hidden;
send({ ...base, audit_id: "bay1-2", status: "complete", verdict: "flag", dataset_tag: "flag",
  confidence: 0.9, reasons: ["Thumb near -X HV terminal"], operator_note: "Exclude." });
out.flagShown = !panel.hidden; out.flag = panel.textContent; out.flagTone = panel.dataset.tone;
send({ ...base, audit_id: "e", status: "error", operator_note: "Audit request timed out" });
out.error = panel.textContent;
send({ ...base, audit_id: "s", status: "skipped", frame_count: 0,
  operator_note: "Hourly review budget reached" });
out.skipped = panel.textContent;
send(undefined); out.hiddenAgain = panel.hidden;
console.log(JSON.stringify(out));
"""
)


def _node(script: str) -> dict[str, Any]:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    out = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_index_mounts_the_audit_card_and_script() -> None:
    html = _read("index.html")
    assert 'id="audit-panel"' in html and 'src="audit.js"' in html
    assert html.index('src="console.js"') < html.index('src="audit.js"')
    for name in ("audit.js", "console.js", "console.css", "index.html"):
        assert not re.search(r"https?://", _read(name)), f"{name} references the network"


def test_audit_card_lifecycle() -> None:
    got = _node(_AUDIT_HARNESS % _read("audit.js"))
    assert got["nullHidden"] is True
    assert got["pendingHidden"] is False and "Reviewing 6 frames" in got["pending"]
    assert "Bay 1" in got["pending"] and "skill_007" in got["pending"]
    assert got["stable"] is True, "identical telemetry must not rebuild the card"
    assert "Approved for ACT" in got["approve"] and "94%" in got["approve"]
    assert "Flap depressed cleanly" in got["approve"] and "8.1 s" in got["approve"]
    assert "dataset: approve" in got["approve"] and "gemini-4-argon" in got["approve"]
    assert got["tone"] == "approve" and got["reasons"] == 2
    assert got["buttons"] == [{"tab": -1, "prevented": True}]
    assert got["dismissed"] is True and got["staysDismissed"] is True
    assert got["flagShown"] is True and got["flagTone"] == "flag"
    assert "Flagged" in got["flag"] and "HV terminal" in got["flag"]
    assert "failed" in got["error"] and "timed out" in got["error"]
    assert "skipped" in got["skipped"] and "budget" in got["skipped"]
    assert got["hiddenAgain"] is True


_GATE_HARNESS = (
    _DOM
    + """
const m = %s.match(/function updateGate\\(t\\)\\s*\\{[\\s\\S]*?\\n\\}/);
const updateGate = new Function("t", m[0] + "\\nreturn updateGate(t);");
const g = { fit_ms: 12, passed: 20, total: 20, elapsed_s: 3.4, ok: true };
const row = () => els.gate.byClass("gate__vlm")[0] || null;
const out = {};
updateGate({ gate: g }); out.plain = row() === null;
updateGate({ gate: { ...g, vlm_verdict: "approve", vlm_reason: "Clean bolt hook",
  vlm_model: "gemini-4-hydrogen", vlm_latency_ms: 3400 } });
out.approve = row().textContent; out.approveKind = row().dataset.verdict;
updateGate({ gate: { ...g, ok: false, passed: 18, worst_failure: "Thumb near HV",
  vlm_verdict: "reject", vlm_reason: "Thumb near HV", vlm_model: "gemini-4-hydrogen" } });
out.reject = els.gate.textContent; out.rejectKind = row().dataset.verdict;
out.reasonOnce = els.gate.textContent.split("Thumb near HV").length - 1;
updateGate({ gate: { ...g, vlm_verdict: "physics_only", vlm_model: "physics_only" } });
out.physics = row().textContent; out.physicsKind = row().dataset.verdict;
console.log(JSON.stringify(out));
"""
)


def test_gate_shows_the_vlm_verdict_beside_the_sim_score() -> None:
    got = _node(_GATE_HARNESS % json.dumps(_read("console.js")))
    assert got["plain"] is True
    assert got["approveKind"] == "approve"
    assert "VLM hydrogen" in got["approve"] and "Grasp approved · 3.4 s" in got["approve"]
    assert "Clean bolt hook" in got["approve"]
    assert got["rejectKind"] == "reject" and "Grasp rejected" in got["reject"]
    assert "18/20" in got["reject"] and got["reasonOnce"] == 1
    assert got["physicsKind"] == "physics_only" and "Physics only" in got["physics"]


def test_gate_vlm_fields_reach_telemetry() -> None:
    gate = GateStatus(
        12.0, 20, 20, 3.4, True, vlm_verdict="approve", vlm_reason="ok", vlm_model="m"
    )
    msg = TelemetryAdapter(bays_present=(1, 2), background=()).build(
        HudView(bays=(), queue=(), background_bays=(), gate=gate), 2.0, (), {}
    )
    assert msg["gate"]["vlm_verdict"] == "approve"
    assert msg["gate"]["vlm_model"] == "m" and "vlm_source" not in msg["gate"]


@pytest.fixture
def server() -> Iterator[ConsoleServer]:
    srv = ConsoleServer(port=0)
    srv.start()
    yield srv
    srv.stop()


def test_copilot_audit_ingress_reaches_browsers_and_replays(server: ConsoleServer) -> None:
    audit = {"audit_id": "a1", "status": "complete", "verdict": "flag", "confidence": float("nan")}
    url = f"ws://127.0.0.1:{server.port}/ws"
    with connect(url) as browser, connect(url) as copilot:
        copilot.send(json.dumps({"type": "copilot_audit", "audit": audit}))
        got = json.loads(browser.recv(timeout=5.0))
        assert got["audit"] == {**audit, "confidence": None}
    server.publish_telemetry({"type": "telemetry", "layout_id": 1, "audit": None})
    with connect(url) as late:
        assert json.loads(late.recv(timeout=5.0))["audit"]["audit_id"] == "a1"
