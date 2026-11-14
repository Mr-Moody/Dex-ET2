import math
from pathlib import Path

from teleop.retarget.calibration import Calibration, calibrate
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
    [FlexMap("idx", ["a", "b"], "x"), FlexMap("w", ["c"], "x", weights=[2.0])],
)


def fr(angle: float, side="right") -> HandFrame:
    j = {n: qaxis("x", angle) for n in ("a", "b", "c")}
    return HandFrame(side, 0, 0, "t", j)  # type: ignore[arg-type]


def test_default():
    c = Calibration.default(SPEC)
    assert c.ranges["left"]["idx"] == (0.0, 2.8)
    assert c.ranges["right"]["w"] == (0.0, 2.8)


def test_calibrate_median():
    opens = [fr(0.0), fr(0.1), fr(5.0 * 0 + 0.0)]
    closed = [fr(0.5), fr(0.6), fr(0.7)]
    c = calibrate(SPEC, opens, closed)
    o, cl = c.ranges["right"]["idx"]
    assert abs(o - 0.0) < 1e-9 and abs(cl - 1.2) < 1e-9
    assert abs(c.ranges["right"]["w"][1] - 1.2) < 1e-9
    assert c.ranges["left"]["idx"] == (0.0, 2.8)  # no left data -> default


def test_fixed_range_skips_calibration():
    spec = HandSpec(
        "syn",
        {"left": Path("l.xml"), "right": Path("r.xml")},
        False,
        "base",
        [FlexMap("rot", ["a"], "x", range=[-0.5, 0.5])],
    )
    assert Calibration.default(spec).ranges["right"]["rot"] == (-0.5, 0.5)
    c = calibrate(spec, [fr(0.0)], [fr(0.3)])
    assert c.ranges["right"]["rot"] == (-0.5, 0.5)


def test_calibration_json_roundtrip(tmp_path: Path):
    c = calibrate(SPEC, [fr(0.0)], [fr(0.5)])
    p = tmp_path / "c.json"
    c.to_json(p)
    assert Calibration.from_json(p) == c
