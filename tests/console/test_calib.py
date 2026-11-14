"""Console glove calibration wizard: fake clock and fake source, no thread, no hardware."""

from __future__ import annotations

import json
import math
from collections.abc import Collection
from pathlib import Path

import pytest

from teleop.console.calib import GloveCalibrator
from teleop.retarget.calibration import Calibration
from teleop.retarget.direct import DirectRetargeter
from teleop.sim.hands import FlexMap, HandSpec
from teleop.types import BimanualFrame, HandFrame, Quat, Side, SourceStatus

SPEC = HandSpec(
    "syn",
    {"left": Path("l.xml"), "right": Path("r.xml")},
    False,
    "base",
    [FlexMap("idx", ["a", "b"], "x"), FlexMap("mid", ["c"], "x")],
)


def _qx(a: float) -> Quat:
    return (math.sin(a / 2), 0.0, 0.0, math.cos(a / 2))


class FakeSource:
    """Every ``latest`` call yields a fresh right-hand frame at the current curl angle."""

    name = "fake"

    def __init__(self) -> None:
        self.angle = 0.0
        self.mid_angle: float | None = None
        self.on = True
        self._seq = 0

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def status(self) -> SourceStatus:
        return SourceStatus(True, None, self._seq, 0, 0)

    def latest(self) -> BimanualFrame:
        if not self.on:
            return BimanualFrame(None, None)
        self._seq += 1
        mid = self.angle if self.mid_angle is None else self.mid_angle
        joints = {"a": _qx(self.angle), "b": _qx(self.angle), "c": _qx(mid)}
        return BimanualFrame(None, HandFrame("right", self._seq, self._seq, "fake", joints))


class Applied:
    def __init__(self) -> None:
        self.calls: list[Calibration] = []

    def __call__(self, calib: Calibration) -> Collection[tuple[Side, str]]:
        self.calls.append(calib)
        r = DirectRetargeter(SPEC, {"right": {"idx": (0.0, 1.0), "mid": (0.0, 1.0)}}, calib)
        return r.degenerate


def _make(tmp_path: Path, src: FakeSource, apply: Applied, **kw: object) -> GloveCalibrator:
    return GloveCalibrator(
        SPEC,
        src,
        apply,
        persist_path=kw.pop("persist_path", tmp_path / "calib" / "atc.json"),  # type: ignore[arg-type]
        prep_s=1.0,
        hold_s=1.0,
        clock=lambda: 0.0,
    )


def _run(cal: GloveCalibrator, src: FakeSource, flat: float, fist: float) -> None:
    """Drive the whole wizard at 10 Hz of fake time, curling the hand per phase."""
    t = 0.0
    assert cal.begin(t)
    while True:
        st = cal.status_dict()
        src.angle = fist if st["pose"] == "fist" else flat
        if not cal.step(t):
            return
        t += 0.1


def test_full_capture_applies_live_and_persists(tmp_path: Path) -> None:
    src, apply = FakeSource(), Applied()
    cal = _make(tmp_path, src, apply)
    assert cal.status_dict()["origin"] == "defaults"
    _run(cal, src, flat=0.1, fist=1.2)
    assert len(apply.calls) == 1
    o, c = apply.calls[0].ranges["right"]["idx"]
    assert o == pytest.approx(0.2) and c == pytest.approx(2.4)
    st = cal.status_dict()
    assert st["state"] == "idle" and st["result"] == "applied" and st["seq"] == 1
    assert st["origin"] == "live" and st["flat_done"] and st["fist_done"]
    saved = json.loads((tmp_path / "calib" / "atc.json").read_text(encoding="utf-8"))
    assert saved["right"]["idx"] == pytest.approx([0.2, 2.4])


def test_reversed_curl_is_applied_as_is(tmp_path: Path) -> None:
    """Studio twists that curl the other way give closed < open; the retargeter handles it."""
    src, apply = FakeSource(), Applied()
    cal = _make(tmp_path, src, apply)
    _run(cal, src, flat=0.0, fist=-1.0)
    o, c = apply.calls[0].ranges["right"]["idx"]
    assert c < o
    assert cal.status_dict()["result"] == "applied"


def test_degenerate_actuator_reports_partial(tmp_path: Path) -> None:
    src, apply = FakeSource(), Applied()
    src.mid_angle = 0.3  # the middle finger never moves
    cal = _make(tmp_path, src, apply)
    _run(cal, src, flat=0.0, fist=1.0)
    st = cal.status_dict()
    assert st["result"] == "partial" and "mid" in st["detail"]
    assert len(apply.calls) == 1


def test_phases_and_countdown(tmp_path: Path) -> None:
    src, apply = FakeSource(), Applied()
    cal = _make(tmp_path, src, apply)
    cal.begin(0.0)
    cal.step(0.4)
    st = cal.status_dict()
    assert (st["state"], st["pose"], st["remaining_s"]) == ("prep", "flat", 0.6)
    assert st["frames"]["flat"] == 0  # nothing is collected while getting ready
    cal.step(1.0)
    cal.step(1.5)
    st = cal.status_dict()
    assert (st["state"], st["pose"]) == ("hold", "flat") and st["frames"]["flat"] == 1
    cal.step(2.0)
    st = cal.status_dict()
    assert (st["state"], st["pose"]) == ("prep", "fist") and st["flat_done"]
    assert not st["fist_done"]


def test_cancel_leaves_retargeter_alone(tmp_path: Path) -> None:
    src, apply = FakeSource(), Applied()
    cal = _make(tmp_path, src, apply)
    cal.begin(0.0)
    for t in (0.5, 1.2, 1.6, 2.3):
        cal.step(t)
    cal.cancel()
    assert not cal.step(5.0)
    st = cal.status_dict()
    assert st["result"] == "cancelled" and st["state"] == "idle" and st["origin"] == "defaults"
    assert not st["flat_done"] and not st["fist_done"]
    assert apply.calls == []
    assert not (tmp_path / "calib" / "atc.json").exists()


def test_no_frames_does_not_apply(tmp_path: Path) -> None:
    src, apply = FakeSource(), Applied()
    src.on = False
    cal = _make(tmp_path, src, apply)
    _run(cal, src, flat=0.0, fist=1.0)
    assert cal.status_dict()["result"] == "no_frames"
    assert apply.calls == []


def test_second_start_while_running_is_refused(tmp_path: Path) -> None:
    src, apply = FakeSource(), Applied()
    cal = _make(tmp_path, src, apply)
    assert cal.begin(0.0)
    assert not cal.begin(0.5)
    assert cal.status_dict()["state"] == "prep"  # still running the first capture


def test_save_failure_keeps_live_apply(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x", encoding="utf-8")
    src, apply = FakeSource(), Applied()
    cal = _make(tmp_path, src, apply, persist_path=blocker / "atc.json")
    _run(cal, src, flat=0.0, fist=1.0)
    st = cal.status_dict()
    assert st["result"] == "applied" and "not saved" in st["detail"]
    assert len(apply.calls) == 1


def test_messages_and_rerun(tmp_path: Path) -> None:
    src, apply = FakeSource(), Applied()
    cal = _make(tmp_path, src, apply)
    _run(cal, src, flat=0.0, fist=1.0)
    _run(cal, src, flat=0.0, fist=0.8)  # drift fix: runs again on top of a live calibration
    assert len(apply.calls) == 2 and cal.status_dict()["seq"] == 2
    cal.begin(0.0)
    cal.handle_message({"type": "calib", "action": "cancel"})
    assert cal.status_dict()["result"] == "cancelled"
    cal.handle_message({"type": "calib", "action": "bogus"})
    assert cal.status_dict()["state"] == "idle"


def test_glove_operator_recalibrate_swaps_retargeter() -> None:
    from teleop.commands.atc import GloveOperator

    ranges = {"idx": (0.0, 1.0), "mid": (0.0, 1.0)}
    old = DirectRetargeter(SPEC, {"right": ranges}, Calibration.default(SPEC))
    op = GloveOperator(FakeSource(), object(), old, hand=SPEC, ctrlrange=ranges)  # type: ignore[arg-type]
    calib = Calibration.default(SPEC)
    calib.ranges["right"]["idx"] = (0.0, -1.0)
    calib.ranges["right"]["mid"] = (0.2, 0.21)
    assert op.recalibrate(calib) == {("right", "mid")}
    assert op.retargeter is not old
    frame = FakeSource()
    frame.angle = -0.25  # a = b = -0.25 -> idx flex -0.5: half way to the reversed fist
    targets = op.retargeter.retarget(frame.latest().right).positions  # type: ignore[arg-type]
    assert targets["idx"] == pytest.approx(0.5)


def test_glove_operator_without_spec_refuses() -> None:
    from teleop.commands.atc import GloveOperator

    op = GloveOperator(FakeSource(), object(), object())  # type: ignore[arg-type]
    with pytest.raises(RuntimeError):
        op.recalibrate(Calibration.default(SPEC))
