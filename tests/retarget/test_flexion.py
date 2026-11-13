import math

from hypothesis import given
from hypothesis import strategies as st

from teleop.retarget.flexion import flex_value, twist_angle
from teleop.sim.hands import FlexMap
from teleop.types import HandFrame, Quat


def qaxis(axis: str, a: float) -> Quat:
    s, c = math.sin(a / 2), math.cos(a / 2)
    return {"x": (s, 0.0, 0.0, c), "y": (0.0, s, 0.0, c), "z": (0.0, 0.0, s, c)}[axis]


def qmul(a: Quat, b: Quat) -> Quat:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def test_twist_angle_pure_axis():
    assert abs(twist_angle(qaxis("x", 0.7), "x") - 0.7) < 1e-6


def test_twist_ignores_swing():
    # swing * twist: twist about x applied first, then a swing about y
    q2 = qmul(qaxis("x", 0.7), qaxis("y", 0.5))
    assert abs(twist_angle(q2, "x") - 0.7) < 1e-6


@given(st.floats(min_value=-math.pi + 1e-3, max_value=math.pi - 1e-3))
def test_twist_sign_and_wrap(a: float):
    for axis in ("x", "y", "z"):
        assert abs(twist_angle(qaxis(axis, a), axis) - a) < 1e-6  # type: ignore[arg-type]


def test_twist_pure_swing_and_negated_quat():
    assert twist_angle(qaxis("y", math.pi), "x") == 0.0
    q = qaxis("x", 0.7)
    nq = (-q[0], -q[1], -q[2], -q[3])
    assert abs(twist_angle(nq, "x") - 0.7) < 1e-6


def _frame(joints: dict[str, Quat]) -> HandFrame:
    return HandFrame("right", 0, 0, "t", joints)


def test_flex_value_weights_invert_missing():
    fm = FlexMap("a", ["j1", "j2"], "x", invert=True, weights=[1.0, 2.0])
    f = _frame({"j1": qaxis("x", 0.2), "j2": qaxis("x", 0.1)})
    v = flex_value(f, fm)
    assert v is not None and abs(v + 0.4) < 1e-9
    only = flex_value(_frame({"j1": qaxis("x", 0.2)}), fm)
    assert only is not None and abs(only + 0.2) < 1e-9
    assert flex_value(_frame({}), fm) is None
