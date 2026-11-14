"""T-150: ``teleop atc`` stage controls, 4-bay HUD, audio cues and the scripted 6-beat run."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from teleop.atc.hud import (
    GateStatus,
    HudView,
    RescueStatus,
    confidence_bar,
    format_bay_table,
    format_gate,
    format_rescue,
    hud_lines,
    hud_overlays,
)
from teleop.atc.stage import (
    KEYMAP,
    AudioCues,
    StageAction,
    merge_console,
)
from teleop.atc.types import BaySnapshot, BayState, ConsoleInputState, StallMatch
from teleop.compliance.protocol import ComplianceEvent, make_event
from teleop.reasoning.protocol import ReasoningContext, ReasoningVerdict
from teleop.sim.bays import PANDA_XML
from teleop.sim.hands import load_hand

SKU = "ev_latch_v1"


def _models_present() -> bool:
    try:
        right = load_hand("inspire").mjcf["right"]
    except FileNotFoundError:
        return False
    return PANDA_XML.is_file() and right.is_file()


needs_models = pytest.mark.skipif(
    not _models_present(), reason="Panda/Inspire MJCFs missing; run scripts/fetch_models.py"
)


# --------------------------------------------------------------------------- fakes


class RecordingSink:
    def __init__(self) -> None:
        self.events: list[ComplianceEvent] = []

    def emit(self, event: ComplianceEvent) -> None:
        self.events.append(event)

    def close(self) -> None:
        pass


class RecordingReasoner:
    def __init__(self) -> None:
        self.contexts: list[ReasoningContext] = []

    def submit(self, ctx: ReasoningContext) -> None:
        self.contexts.append(ctx)

    def latest_verdict(self, bay_id: int) -> ReasoningVerdict | None:
        if not any(c.bay_id == bay_id for c in self.contexts):
            return None
        return ReasoningVerdict(f"bay {bay_id} triaged", "hold", "logged", "fake", 0.1)

    def close(self) -> None:
        pass


class FakeSpeaker:
    def __init__(self) -> None:
        self.said: list[str] = []

    def say(self, text: str) -> None:
        self.said.append(text)

    def close(self) -> None:
        pass


# --------------------------------------------------------------------------- HUD formatting


def _view() -> HudView:
    bays = (
        BaySnapshot(1, BayState.RESCUE, 0.12, 0.034, 5.2, SKU),
        BaySnapshot(
            2,
            BayState.QUEUED,
            0.25,
            0.0352,
            4.8,
            SKU,
            queue_position=0,
            last_match=StallMatch(None, float("inf"), True),
        ),
    )
    return HudView(
        bays=bays,
        background_bays=(3, 4),
        queue=(2,),
        selected_bay=1,
        rescue=RescueStatus(bay_id=1, clutch_engaged=True, elapsed_s=7.25),
        gate=GateStatus(fit_ms=11.4, passed=3, total=3, elapsed_s=0.28, ok=True),
        compliance="SOLANA VALIDATION_GATE bay 1 #3f2a9c1e0b7d [anchored]",
        reasoning="NEMOTRON bay 1: Novel jam, operator rescue requested",
        beat="3 Operator rescue",
        unoq="UNO Q loopback",
    )


def test_hud_formats_bay_states_confidence_and_fifo_queue() -> None:
    assert confidence_bar(0.0) == "[..........] 0.00"
    assert confidence_bar(0.87) == "[|||||||||.] 0.87"
    assert confidence_bar(1.0) == "[||||||||||] 1.00"
    assert confidence_bar(1.7) == "[||||||||||] 1.00"  # clamped

    view = _view()
    labels, values = format_bay_table(view)
    label_rows, value_rows = labels.split("\n"), values.split("\n")
    assert len(label_rows) == len(value_rows)
    rows = dict(zip(label_rows, value_rows, strict=True))
    assert "RESCUE" in rows["BAY 1"] and "0.12" in rows["BAY 1"] and "3.4 cm" in rows["BAY 1"]
    assert "QUEUED #1" in rows["BAY 2"] and "0.25" in rows["BAY 2"] and "OOD" in rows["BAY 2"]
    assert "BACKGROUND" in rows["BAY 3"] and "BACKGROUND" in rows["BAY 4"]
    assert rows["QUEUE"] == "2"
    assert rows["SELECTED"] == "BAY 1"

    assert format_rescue(view.rescue) == "RESCUE BAY 1 | ENGAGED | 00:07.2"
    assert format_rescue(RescueStatus(2, False, 61.0)) == "RESCUE BAY 2 | HOLD CLUTCH | 01:01.0"
    assert format_rescue(RescueStatus(None, False, None)) == "RESCUE  none"
    assert format_gate(view.gate) == "UNO Q DMP fit: 11.4 ms | Sim Gate: 3/3 PASS (0.28 s)"
    assert format_gate(GateStatus(9.0, 1, 3, 0.31, False)) == (
        "UNO Q DMP fit: 9.0 ms | Sim Gate: 1/3 FAIL (0.31 s)"
    )
    assert format_gate(GateStatus()) == "UNO Q DMP fit: -- | Sim Gate: --"

    text = "\n".join(hud_lines(view))
    for needle in (
        "BAY 1",
        "QUEUE",
        "UNO Q DMP fit: 11.4 ms",
        "SOLANA VALIDATION_GATE",
        "NEMOTRON bay 1",
        "3 Operator rescue",
    ):
        assert needle in text

    overlays = hud_overlays(view)
    assert len(overlays) >= 3
    assert all(isinstance(font, int) and isinstance(pos, int) for font, pos, _, _ in overlays)
    assert len({pos for _, pos, _, _ in overlays}) == len(overlays)  # one panel per corner
    assert any(left and "BAY 1" in left for _, _, left, _ in overlays)

    empty = HudView(bays=(), queue=())
    assert "QUEUE" in "\n".join(hud_lines(empty))  # renders before the first tick


# --------------------------------------------------------------------------- stage controls


def test_keymap_and_console_merge_mirror_hardware() -> None:
    assert KEYMAP[ord("1")] is StageAction.SELECT_BAY_1
    assert KEYMAP[ord("4")] is StageAction.SELECT_BAY_4
    assert KEYMAP[32] is StageAction.CLUTCH
    assert KEYMAP[ord("F")] is StageAction.ARM_BAY1_FLAP
    assert KEYMAP[ord("G")] is StageAction.ARM_BAY2_FLAP
    assert KEYMAP[ord("P")] is StageAction.ARM_BAY2_PIN
    assert KEYMAP[ord("R")] is StageAction.RESET

    keyboard = ConsoleInputState(selected_bay=1, clutch=False, timestamp_ns=100)
    hardware = ConsoleInputState(selected_bay=2, clutch=True, timestamp_ns=200)
    merged = merge_console([keyboard, hardware])
    assert merged.selected_bay == 2  # most recent selection wins
    assert merged.clutch
    estop = ConsoleInputState(selected_bay=None, estop=True, timestamp_ns=50)
    assert not merge_console([hardware, estop]).clutch  # any E-stop drops the clutch
    assert merge_console([]) == ConsoleInputState()


def test_audio_cues_follow_compliance_events_and_stay_silent_when_disabled() -> None:
    speaker = FakeSpeaker()
    cues = AudioCues(speaker)
    stall = make_event("STALL_ALERT", 1, SKU, "stall", {"is_ood": True})
    gate = make_event("VALIDATION_GATE", 1, SKU, "gate", {"passed": True})
    rejected = make_event("VALIDATION_GATE", 1, SKU, "gate", {"passed": False})
    ood = make_event("OOD_REFUSAL", 2, SKU, "ood", {})
    auto = make_event("RESCUE_LOGGED", 2, SKU, "auto", {"autonomous": True})
    manual = make_event("RESCUE_LOGGED", 1, SKU, "manual", {"autonomous": False})
    for event in (stall, manual, gate, rejected, auto, ood):
        cues.on_event(event)
    assert speaker.said == [
        "Bay 1 stalled. Queued for rescue.",
        "Skill validated. Broadcasting to fleet.",
        "Skill rejected by the validation gate.",
        "Bay 2 cleared autonomously with a fleet skill.",
        "Out of distribution fault. Refusing skill transfer.",
    ]

    muted = FakeSpeaker()
    off = AudioCues(muted, enabled=False)
    off.on_event(stall)
    assert muted.said == []


# --------------------------------------------------------------------------- full stage


@pytest.fixture
def unoq() -> Iterator[object]:
    from teleop.unoq.client import UnoQClient
    from teleop.unoq.server import UnoQServer

    server = UnoQServer(host="127.0.0.1", udp_port=0, tcp_port=0, name="test-node").start()
    client = UnoQClient(
        "127.0.0.1", server.udp_port, server.tcp_port, local_server=server
    ).connect()
    yield client
    client.close()


@needs_models
def test_atc_headless_scripted_6_beat_stage_run_completes(unoq: object) -> None:
    from teleop.atc.stage import BEATS, STAGE_SEED, StageController, StageDirector
    from teleop.unoq.client import UnoQClient

    assert isinstance(unoq, UnoQClient)
    sink, reasoner, speaker = RecordingSink(), RecordingReasoner(), FakeSpeaker()
    controller = StageController.build(
        unoq, sink, reasoner, audio=AudioCues(speaker), cycle_dwell_s=0.5, seed=STAGE_SEED
    )
    director = StageDirector(controller)
    for _ in range(12_000):
        if director.done:
            break
        controller.tick(director.step())
    assert director.done, director.results
    assert [r.name for r in director.results] == list(BEATS)
    assert all(r.ok for r in director.results), director.results

    co = controller.coordinator
    # Beat 1-4: Bay 1 stalled on an unseen jam, was rescued and its skill passed the gate.
    kinds = [e.kind for e in sink.events]
    assert kinds[:4] == ["STALL_ALERT", "RESCUE_LOGGED", "VALIDATION_GATE", "SKILL_DEPLOYED"]
    assert len(co.library) == 1
    view = controller.hud_view()
    assert view.gate.ok and (view.gate.passed or 0) >= 18 and view.gate.total == 20
    assert view.gate.fit_ms is not None
    # Beat 5: Bay 2 cleared its rotated/shifted jam autonomously.
    autonomous = [e for e in sink.events if e.kind == "RESCUE_LOGGED" and e.bay_id == 2]
    assert autonomous and autonomous[0].metadata["autonomous"] is True
    # Beat 6: the pinned bolt was refused and handed to the operator.
    assert kinds[-1] == "OOD_REFUSAL"
    assert co.state(2) == BayState.OOD_ESCALATED
    assert view.rescue.bay_id == 2 and not view.rescue.clutch_engaged
    assert "OOD_REFUSAL" in view.compliance and "bay 2" in view.reasoning
    # The HUD mirrors the stage and the cues tracked every beat.
    assert controller.cleared_cycles[1] >= 1 and controller.cleared_cycles[2] >= 1
    assert speaker.said[0] == "Bay 1 stalled. Queued for rescue."
    assert "Skill validated. Broadcasting to fleet." in speaker.said
    assert speaker.said[-1] == "Out of distribution fault. Refusing skill transfer."
    # Reset (R) returns every active bay to a fresh routine cycle and keeps the library.
    controller.on_key(ord("R"))
    controller.tick(None)
    assert co.state(2) == BayState.NOMINAL and co.queue == ()
    assert len(co.library) == 1


@needs_models
def test_atc_command_headless_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from teleop.commands import atc

    monkeypatch.chdir(tmp_path)  # compliance audit log lands in tmp_path/outputs
    for var in ("SOLANA_RPC_URL", "SOLANA_KEYPAIR_PATH", "VERDA_API_KEY", "UNOQ_HOST"):
        monkeypatch.delenv(var, raising=False)
    rc = atc.main(
        [
            "--headless",
            "--steps",
            "30",
            "--source",
            "synthetic",
            "--unoq-host",
            "127.0.0.1",
            "--unoq-tcp-port",
            "1",  # nothing listens: falls back to the local loopback node
            "--env-file",
            str(tmp_path / "missing.env"),
        ]
    )
    assert rc == 0
