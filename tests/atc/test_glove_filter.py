from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from teleop.atc.glove_filter import GloveInputFilter, frame_time_s
from teleop.types import BimanualFrame, HandFrame, Pose

IDENTITY = (0.0, 0.0, 0.0, 1.0)
DT_US = 12_000  # 83.3 Hz glove stream


def _frame(seq: int, t_us: int | None, host_ns: int | None = None) -> HandFrame:
    return HandFrame(
        side="right",
        host_time_ns=seq * 1_000_000 if host_ns is None else host_ns,
        seq=seq,
        source="test",
        joints={},
        device_time_us=t_us,
    )


def _pose(x: float) -> Pose:
    return Pose(orientation=IDENTITY, position=(x, 0.0, 0.0))


def _x(pose: Pose) -> float:
    assert pose.position is not None
    return pose.position[0]


def test_frame_time_prefers_device_clock() -> None:
    assert frame_time_s(_frame(1, 2_000_000)) == pytest.approx(2.0)
    assert frame_time_s(_frame(1, None, host_ns=3_000_000_000)) == pytest.approx(3.0)


def test_first_sample_passes_through() -> None:
    f = GloveInputFilter()
    wrist, targets = f.step(_frame(1, 0), _pose(0.3), {"idx": 0.7})
    assert _x(wrist) == pytest.approx(0.3)
    assert targets == {"idx": pytest.approx(0.7)}


def test_still_hand_jitter_is_attenuated() -> None:
    rng = np.random.default_rng(0)
    f = GloveInputFilter()
    raw: list[float] = []
    out: list[float] = []
    fingers: list[float] = []
    for i in range(400):
        x = 0.2 + float(rng.normal(0.0, 0.002))  # 2 mm sensor noise
        q = 0.5 + float(rng.normal(0.0, 0.02))
        wrist, targets = f.step(_frame(i, i * DT_US), _pose(x), {"idx": q})
        raw.append(x)
        out.append(_x(wrist))
        fingers.append(targets["idx"])
    assert np.std(out[100:]) < 0.35 * np.std(raw[100:])
    assert np.mean(out[100:]) == pytest.approx(0.2, abs=1e-3)
    assert np.std(fingers[100:]) < 0.5 * 0.02


def test_fast_motion_tracks_with_small_lag() -> None:
    f = GloveInputFilter()
    last = 0.0
    for i in range(100):  # 0.5 m/s sweep
        last = _x(f.step(_frame(i, i * DT_US), _pose(0.5 * i * DT_US * 1e-6), {})[0])
    assert 0.5 * 99 * DT_US * 1e-6 - last < 0.02


def test_orientation_stays_unit_and_smoothed() -> None:
    f = GloveInputFilter()
    q: Any = None
    for i in range(50):
        a = 0.02 * (-1) ** i  # +-1 degree wobble about Z
        pose = Pose(orientation=(0.0, 0.0, math.sin(a / 2), math.cos(a / 2)), position=None)
        q = f.step(_frame(i, i * DT_US), pose, {})[0].orientation
    assert float(np.linalg.norm(q)) == pytest.approx(1.0)
    assert abs(2.0 * math.asin(q[2])) < 0.01


def test_repeated_frame_is_filtered_once() -> None:
    f = GloveInputFilter()
    f.step(_frame(1, 0), _pose(0.0), {})
    first = _x(f.step(_frame(2, DT_US), _pose(0.1), {})[0])
    for _ in range(10):  # the 60 Hz control loop reads the same glove frame again
        assert _x(f.step(_frame(2, DT_US), _pose(0.1), {})[0]) == first


def test_stream_gap_restarts_filter() -> None:
    f = GloveInputFilter()
    for i in range(20):
        f.step(_frame(i, i * DT_US), _pose(0.0), {"idx": 0.0})
    wrist, targets = f.step(_frame(20, 20 * DT_US + 500_000), _pose(0.3), {"idx": 1.0})
    assert _x(wrist) == pytest.approx(0.3)
    assert targets["idx"] == pytest.approx(1.0)


def test_clock_going_backwards_restarts_filter() -> None:
    f = GloveInputFilter()
    for i in range(20):
        f.step(_frame(i, 10_000_000 + i * DT_US), _pose(0.0), {})
    assert _x(f.step(_frame(0, 0), _pose(0.3), {})[0]) == pytest.approx(0.3)


def test_gap_must_be_positive() -> None:
    with pytest.raises(ValueError):
        GloveInputFilter(gap_reset_s=0.0)


# --------------------------------------------------------------------------- GloveOperator


class _Source:
    def __init__(self) -> None:
        self.frame: HandFrame | None = None

    def latest(self) -> BimanualFrame:
        return BimanualFrame(left=None, right=self.frame)


class _Wrist:
    def __init__(self) -> None:
        self.x = 0.0

    def wrist(self, frame: HandFrame) -> Any:
        class _T:
            pose = _pose(self.x)

        return _T()


class _Retarget:
    def __init__(self, value: float) -> None:
        self.value = value

    def retarget(self, frame: HandFrame) -> Any:
        class _J:
            positions = {"idx": self.value}

        return _J()


def test_glove_operator_smooths_and_resets_on_recalibration() -> None:
    from teleop.commands.atc import GloveOperator

    src, wrist = _Source(), _Wrist()
    op = GloveOperator(src, wrist, _Retarget(0.0), smoothing=GloveInputFilter())  # type: ignore[arg-type]
    src.frame = _frame(1, 0)
    op.latest()
    src.frame, wrist.x = _frame(2, DT_US), 0.1
    out = op.latest()
    assert out is not None
    assert 0.0 < _x(out.glove_wrist) < 0.1  # smoothed, not the raw jump

    op.retargeter = _Retarget(1.0)  # type: ignore[assignment]  # as recalibrate() swaps it
    src.frame = _frame(3, 2 * DT_US)
    out = op.latest()
    assert out is not None
    assert out.hand_targets["idx"] == pytest.approx(1.0)  # filters restarted, no glide


def test_glove_operator_without_smoothing_is_raw() -> None:
    from teleop.commands.atc import GloveOperator

    src, wrist = _Source(), _Wrist()
    op = GloveOperator(src, wrist, _Retarget(0.4))  # type: ignore[arg-type]
    src.frame, wrist.x = _frame(1, 0), 0.25
    out = op.latest()
    assert out is not None
    assert _x(out.glove_wrist) == 0.25
    assert out.hand_targets == {"idx": 0.4}


def test_glove_filter_config_key(tmp_path: Any) -> None:
    from teleop.commands.atc import DEFAULT_CONFIG, load_config

    assert load_config(DEFAULT_CONFIG, {}).glove_filter is True
    cfg = tmp_path / "atc.yaml"
    cfg.write_text("glove_filter: false\n", encoding="utf-8")
    assert load_config(cfg, {}).glove_filter is False
