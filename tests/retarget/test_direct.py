import math
from pathlib import Path

from teleop.retarget.calibration import Calibration, calibrate
from teleop.retarget.direct import DirectRetargeter
from teleop.sim.hands import FlexMap, HandSpec
from teleop.types import HandFrame, Quat


def qaxis(axis: str, a: float) -> Quat:
    s, c = math.sin(a / 2), math.cos(a / 2)
    return {"x": (s, 0.0, 0.0, c), "y": (0.0, s, 0.0, c), "z": (0.0, 0.0, s, c)}[axis]


SPEC = HandSpec(
    "syn",
    {"left": Path("l.xml"), "right": Path("r.xml")},
    False,
    "base",
    [FlexMap("idx", ["a"], "x"), FlexMap("mid", ["b"], "x")],
)
CR = {s: {"idx": (0.0, 1.0), "mid": (-1.0, 1.0)} for s in ("left", "right")}


def fr(a: float | None, b: float | None = 0.0, t: int = 7) -> HandFrame:
    j = {}
    if a is not None:
        j["a"] = qaxis("x", a)
    if b is not None:
        j["b"] = qaxis("x", b)
    return HandFrame("right", t, 0, "t", j)


def make(open_: float = 0.0, closed: float = 1.0) -> DirectRetargeter:
    calib = calibrate(SPEC, [fr(open_, open_)], [fr(closed, closed)])
    return DirectRetargeter(SPEC, CR, calib)  # type: ignore[arg-type]


def test_retarget_open_maps_lo_closed_maps_hi():
    r = make()
    t = r.retarget(fr(0.0, 0.0))
    assert t.side == "right" and t.host_time_ns == 7
    assert abs(t.positions["idx"] - 0.0) < 1e-9 and abs(t.positions["mid"] + 1.0) < 1e-9
    t = r.retarget(fr(1.0, 1.0))
    assert abs(t.positions["idx"] - 1.0) < 1e-9 and abs(t.positions["mid"] - 1.0) < 1e-9
    t = r.retarget(fr(0.5, 0.5))
    assert abs(t.positions["idx"] - 0.5) < 1e-9


def test_retarget_clips_beyond_range():
    r = make()
    assert r.retarget(fr(2.0, 2.0)).positions["idx"] == 1.0
    assert r.retarget(fr(-1.0, -1.0)).positions["mid"] == -1.0


def test_retarget_handles_reversed_span():
    r = make(open_=0.0, closed=-1.0)
    assert not r.degenerate
    t = r.retarget(fr(0.0, 0.0))
    assert abs(t.positions["idx"]) < 1e-9
    t = r.retarget(fr(-1.0, -1.0))
    assert abs(t.positions["idx"] - 1.0) < 1e-9 and abs(t.positions["mid"] - 1.0) < 1e-9
    assert abs(r.retarget(fr(-0.5, -0.5)).positions["idx"] - 0.5) < 1e-9


def test_degenerate_range_falls_back():
    calib = calibrate(SPEC, [fr(0.3, 0.3)], [fr(0.3, 0.3)])
    r = DirectRetargeter(SPEC, CR, calib)  # type: ignore[arg-type]
    assert r.degenerate == {("right", "idx"), ("right", "mid")}
    t = r.retarget(fr(0.0, 0.0))  # default range (0, 1.4)
    assert t.positions["idx"] == 0.0


def test_missing_joint_holds_previous():
    r = make()
    r.retarget(fr(0.5, 0.5))
    t = r.retarget(fr(None, 1.0))
    assert abs(t.positions["idx"] - 0.5) < 1e-9
    assert abs(t.positions["mid"] - 1.0) < 1e-9
    # first frame with nothing present starts at lo
    t2 = make().retarget(fr(None, None))
    assert t2.positions == {"idx": 0.0, "mid": -1.0}


def test_default_calibration_used_when_absent():
    r = DirectRetargeter(SPEC, CR, Calibration.default(SPEC))  # type: ignore[arg-type]
    assert abs(r.retarget(fr(1.4, 0.0)).positions["idx"] - 1.0) < 1e-9
