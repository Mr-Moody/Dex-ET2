"""T-197: recovery backend, ACT health and the M toggle on the console and the viewer HUD."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections.abc import Iterator
from dataclasses import replace
from datetime import datetime
from typing import Any

import pytest

from teleop.atc.hud import (
    HudView,
    RecoveryStatus,
    format_recovery,
    hud_lines,
)
from teleop.atc.recovery import RecoveryBackend
from teleop.atc.stage import StageController
from teleop.atc.types import BaySnapshot, BayState
from teleop.compliance.protocol import ComplianceEvent
from teleop.console.layout import feed_layout
from teleop.console.server import WEB_DIR
from teleop.console.telemetry import TelemetryAdapter
from teleop.reasoning.protocol import ReasoningContext, ReasoningVerdict
from teleop.sim.bays import PANDA_XML
from teleop.sim.hands import load_hand

SKU = "EV-LATCH-01"
GRID = feed_layout([1, 2, 3, 4], None)


def _models_present() -> bool:
    try:
        right = load_hand("leap").mjcf["right"]
    except FileNotFoundError:
        return False
    return PANDA_XML.is_file() and right.is_file()


needs_models = pytest.mark.skipif(
    not _models_present(), reason="Panda/LEAP MJCFs missing; run scripts/fetch_models.py"
)

# --------------------------------------------------------------------------- telemetry


def _snap(bay: int, state: BayState = BayState.NOMINAL) -> BaySnapshot:
    return BaySnapshot(bay, state, 0.9, 0.0, 0.5, SKU)


def _view(**kw: Any) -> HudView:
    base = HudView(
        bays=(_snap(1), _snap(2)),
        queue=(),
        background_bays=(3, 4),
        compliance="SOLANA  no events yet",
        reasoning="TRIAGE  idle",
    )
    return replace(base, **kw)


def _build(view: HudView) -> dict[str, Any]:
    adapter = TelemetryAdapter((1, 2, 3, 4), (3, 4), clock=lambda: datetime(2026, 11, 15, 1, 0, 0))
    return adapter.build(view, 1.0, GRID, {3: 0.9, 4: 0.9})


def test_telemetry_has_recovery_block() -> None:
    msg = _build(_view())
    assert msg["recovery"] == {
        "backend": "dmp",
        "note": "",
        "act": {"loaded": False, "healthy": False, "p95_ms": None, "faults": []},
    }
    json.dumps(msg, allow_nan=False)


def test_note_change_reaches_telemetry() -> None:
    note = "ACT unavailable (worker crashed): back to dmp"
    msg = _build(_view(recovery=RecoveryStatus(backend="dmp", note=note, act_loaded=True)))
    assert msg["recovery"]["note"] == note
    assert msg["recovery"]["act"]["loaded"] is True and msg["recovery"]["act"]["healthy"] is False


def test_act_health_reaches_telemetry() -> None:
    rec = RecoveryStatus("hybrid", "", True, True, 12.4, ("flap", "pin"))
    act = _build(_view(recovery=rec))["recovery"]["act"]
    assert act == {"loaded": True, "healthy": True, "p95_ms": 12.4, "faults": ["flap", "pin"]}


def test_bay_recovery_tag() -> None:
    msg = _build(_view(recovery_bays=((2, "act"),)))
    tags = {b["id"]: b["recovery"] for b in msg["bays"]}
    assert tags == {1: None, 2: "act", 3: None, 4: None}


# --------------------------------------------------------------------------- HUD


def test_hud_recovery_line() -> None:
    assert any(line.startswith("RECOVERY dmp") for line in hud_lines(_view()))


def test_hud_recovery_line_formats() -> None:
    assert format_recovery(RecoveryStatus()) == "RECOVERY dmp | ACT off"
    ok = RecoveryStatus("hybrid", "", True, True, 12.0, ("flap",))
    assert format_recovery(ok) == "RECOVERY hybrid | ACT ok p95 12 ms | flap"
    assert format_recovery(replace(ok, act_healthy=False)) == "RECOVERY hybrid | ACT down | flap"
    assert format_recovery(replace(ok, p95_ms=None, faults=())) == "RECOVERY hybrid | ACT ok"


# --------------------------------------------------------------------------- stage -> HudView


class _NullSink:
    def emit(self, event: ComplianceEvent) -> None:
        return None

    def close(self) -> None:
        return None


class _NullReasoner:
    def submit(self, ctx: ReasoningContext) -> None:
        return None

    def latest_verdict(self, bay_id: int) -> ReasoningVerdict | None:
        return None

    def close(self) -> None:
        return None


class _FakeAct:
    name = "act"
    unhealthy_reason = ""

    def __init__(self, healthy: bool = True) -> None:
        self._healthy = healthy

    def covers(self, z: Any) -> bool:
        return True

    def healthy(self) -> bool:
        return self._healthy

    def create(self, rig: Any, z: Any) -> Any:
        raise AssertionError("not used")

    def close(self) -> None:
        return None


@pytest.fixture
def controller() -> Iterator[StageController]:
    from teleop.unoq.client import UnoQClient
    from teleop.unoq.server import UnoQServer

    server = UnoQServer(host="127.0.0.1", udp_port=0, tcp_port=0, name="test-node").start()
    client = UnoQClient("127.0.0.1", server.udp_port, server.tcp_port, local_server=server)
    client.connect()
    ctl = StageController.build(client, _NullSink(), _NullReasoner(), cycle_dwell_s=0.2)  # type: ignore[arg-type]
    yield ctl
    ctl.close()
    client.close()


@needs_models
def test_stage_default_recovery_is_dmp(controller: StageController) -> None:
    view = controller.hud_view()
    assert view.recovery == RecoveryStatus()
    assert view.recovery_bays == ()
    assert any(line.startswith("RECOVERY dmp") for line in hud_lines(view))


@needs_models
def test_stage_reports_backend_switch_and_act_health(controller: StageController) -> None:
    co = controller.coordinator
    co.set_act(_FakeAct())
    co.set_recovery_backend(RecoveryBackend.HYBRID)
    rec = controller.hud_view().recovery
    assert (rec.backend, rec.act_loaded, rec.act_healthy) == ("hybrid", True, True)
    assert "hybrid" in rec.note
    co.set_act(_FakeAct(healthy=False))
    assert controller.hud_view().recovery.act_healthy is False


# --------------------------------------------------------------------------- web assets


def _read(name: str) -> str:
    return (WEB_DIR / name).read_text(encoding="utf-8")


def _marked(js: str, name: str) -> str:
    m = re.search(rf"// <{name}>(.*?)// </{name}>", js, re.DOTALL)
    assert m, f"console.js must keep the {name} block between <{name}> markers"
    return m.group(1)


def _node_json(script: str) -> Any:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    out = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def test_assets_wire_up_the_recovery_chip_and_m_key() -> None:
    html, js, css = _read("index.html"), _read("console.js"), _read("console.css")
    assert 'id="recovery-chip"' in html and 'id="recovery-name"' in html
    assert "t.recovery" in js and "b.recovery" in js and "updateRecovery(t)" in js
    assert "codes: [77]" in js and "Switch recovery" in js
    assert ".rchip" in css and ".spark__tag" in css


_RECOVERY_HARNESS = """
%s
const SWITCH = "recovery backend hybrid (from the next stall)";
const DOWNGRADE = "ACT unavailable (worker crashed): back to dmp";
const act = (o) => ({ loaded: true, healthy: true, p95_ms: 12.4, faults: ["flap"], ...o });
const rec = (backend, o = {}) => ({ backend, note: "", act: act(o) });
console.log(JSON.stringify({
  cycle: ["dmp", "hybrid", "act", "dmp"].map(nextBackend),
  dmp: recoveryChip({ backend: "dmp", note: "", act: act({ loaded: false }) }),
  hybrid: recoveryChip(rec("hybrid")),
  down: recoveryChip(rec("hybrid", { healthy: false })),
  missing: recoveryChip({ backend: "act", note: "", act: act({ loaded: false }) }),
  first: recoveryToast(null, { ...rec("dmp"), note: "x" }),
  same: recoveryToast("x", { ...rec("dmp"), note: "x" }),
  cleared: recoveryToast("x", rec("dmp")),
  switched: recoveryToast("", { ...rec("hybrid"), note: SWITCH }),
  downgrade: recoveryToast("", { ...rec("dmp"), note: DOWNGRADE }),
}));
"""


def test_recovery_chip_and_toast_logic() -> None:
    got = _node_json(_RECOVERY_HARNESS % _marked(_read("console.js"), "recovery"))
    assert got["cycle"] == ["hybrid", "act", "dmp", "hybrid"]
    # the same cycle as the stage's M key
    from teleop.atc.recovery import next_backend

    assert got["cycle"][:3] == [
        next_backend(RecoveryBackend(b)).value for b in ("dmp", "hybrid", "act")
    ]
    assert got["dmp"]["name"] == "DMP" and got["dmp"]["actText"] == ""
    assert got["dmp"]["tone"] == "dmp" and got["dmp"]["hint"] == "Switch recovery: hybrid"
    assert got["hybrid"]["name"] == "HYBRID" and got["hybrid"]["actText"] == "ACT ok 12 ms"
    assert got["hybrid"]["tone"] == "ok" and "flap" in got["hybrid"]["title"]
    assert got["down"]["actText"] == "ACT down" and got["down"]["tone"] == "warn"
    # backend ACT: the name already says ACT
    assert got["missing"]["name"] == "ACT" and got["missing"]["actText"] == "not loaded"
    assert got["missing"]["tone"] == "warn"
    # no replay on the first frame, none while the note is unchanged or cleared
    assert got["first"] is None and got["same"] is None and got["cleared"] is None
    # a plain switch is the M key's own toast; a downgrade is a warning
    assert got["switched"] is None
    assert got["downgrade"] == {
        "text": "ACT unavailable (worker crashed): back to dmp",
        "level": "warn",
    }
