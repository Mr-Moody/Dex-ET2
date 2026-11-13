"""Headless integration tests for the sim / calibrate / record wiring."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from teleop.config import CameraConfig, SimConfig
from teleop.sim.hands import load_hand
from teleop.types import OPENXR_JOINTS, BimanualFrame, HandFrame, Pose, Side, SourceStatus

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "studio" / "coil_on.jsonl"


def _leap_available() -> bool:
    try:
        spec = load_hand("leap")
    except FileNotFoundError:
        return False
    return all(p.is_file() for p in spec.mjcf.values())


needs_leap = pytest.mark.skipif(
    not _leap_available(), reason="LEAP MJCF missing; run scripts/fetch_models.py"
)


def _cfg(**kw: object) -> SimConfig:
    cfg = SimConfig(hand="leap", headless=True, cameras=[CameraConfig("front", 32, 32)])
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def _frame(side: Side, i: int) -> HandFrame:
    angle = 0.7 * (1.0 - math.cos(2.0 * math.pi * i / 20.0))
    q = (math.sin(angle / 2), 0.0, 0.0, math.cos(angle / 2))
    home = (-0.25, 0.0, 0.25) if side == "left" else (0.25, 0.0, 0.25)
    return HandFrame(
        side=side,
        host_time_ns=i + 1,
        seq=i,
        source="fake",
        joints={j: q for j in OPENXR_JOINTS},
        wrist=Pose(orientation=(0.0, 0.0, 0.0, 1.0), position=home),
    )


class DropoutSource:
    """Frames for the first ``n_live`` ``latest()`` calls, then nothing."""

    name = "dropout"

    def __init__(self, n_live: int) -> None:
        self.n_live = n_live
        self.calls = 0

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def latest(self) -> BimanualFrame:
        i = self.calls
        self.calls += 1
        if i < self.n_live:
            return BimanualFrame(_frame("left", i), _frame("right", i))
        return BimanualFrame(None, None)

    def status(self) -> SourceStatus:
        live = self.calls <= self.n_live
        return SourceStatus(live, 1 if live else None, min(self.calls, self.n_live), 0, 0)


def _side_ctrl(app: object, side: str) -> list[int]:
    model = app.model  # type: ignore[attr-defined]
    return [i for i in range(model.nu) if str(model.actuator(i).name).startswith(f"{side}/")]


@needs_leap
def test_sim_synthetic_moves_actuators() -> None:
    from teleop.app import TeleopApp

    app = TeleopApp(_cfg(source="synthetic", steps=120))
    history: list[np.ndarray] = []
    app.on_tick = lambda _i: history.append(np.array(app.data.ctrl, copy=True))
    app.run()
    ctrl = np.stack(history)
    assert ctrl.shape[0] == 120
    for side in ("left", "right"):
        idx = _side_ctrl(app, side)
        assert idx
        assert float(ctrl[:, idx].var(axis=0).max()) > 0.0, side
    assert app.latency.summary()["n"] > 0


@needs_leap
def test_sim_survives_source_dropout() -> None:
    from teleop.app import TeleopApp

    src = DropoutSource(n_live=30)
    app = TeleopApp(_cfg(steps=120), source=src)
    at_30: list[np.ndarray] = []
    after: list[np.ndarray] = []

    def tick(i: int) -> None:
        if i == 29:
            at_30.append(np.array(app.data.ctrl, copy=True))
        elif i >= 30:
            after.append(np.array(app.data.ctrl, copy=True))

    app.on_tick = tick
    stats = app.run()
    assert stats.steps == 120
    assert len(after) == 90
    assert float(np.abs(at_30[0]).max()) > 0.0
    for c in after:
        np.testing.assert_array_equal(c, at_30[0])
    assert app.source.status().connected is False


@needs_leap
def test_max_gap_counts_open_dropout() -> None:
    import dataclasses

    from teleop.app import TeleopApp

    now = [0]
    period = 16_666_667

    class Stamped(DropoutSource):
        def latest(self) -> BimanualFrame:
            f = super().latest()
            if f.left is None or f.right is None:
                return f
            return BimanualFrame(
                dataclasses.replace(f.left, host_time_ns=now[0]),
                dataclasses.replace(f.right, host_time_ns=now[0]),
            )

    app = TeleopApp(_cfg(), source=Stamped(n_live=30), clock=lambda: now[0])
    app.reset_gaps()
    for i in range(60):
        now[0] = i * period
        app.step(i)
    # frames at ticks 0..29, silent since: the open gap is 30 periods (0.5 s)
    assert app.max_gap_s() == pytest.approx(30 * period / 1e9, abs=1e-6)


@needs_leap
def test_keys_nudge_imu_and_queue_reset() -> None:
    from teleop.app import KEY, TeleopApp

    cfg = _cfg(steps=1)
    cfg.wrist.mode = "imu"
    app = TeleopApp(cfg, source=DropoutSource(n_live=5))
    app.on_key(KEY["UP"])
    app.on_key(KEY["W"])
    app.on_key(KEY["R"])
    app.step(0)
    step = cfg.wrist.nudge_step_m
    right = app.wrist.wrist(_frame("right", 0)).pose.position
    left = app.wrist.wrist(_frame("left", 0)).pose.position
    assert right == pytest.approx((0.25, step, 0.25))
    assert left == pytest.approx((-0.25, step, 0.25))
    assert app.resets == 1


@needs_leap
def test_first_coil_frames_recenter_to_home() -> None:
    from teleop.app import TeleopApp

    class FarSource(DropoutSource):
        def latest(self) -> BimanualFrame:
            f = super().latest()
            out: list[HandFrame | None] = []
            for h in (f.left, f.right):
                if h is None:
                    out.append(None)
                    continue
                p = h.wrist.position if h.wrist and h.wrist.position else (0.0, 0.0, 0.0)
                far = Pose(orientation=(0.0, 0.0, 0.0, 1.0), position=(p[0], p[1] + 2, p[2] + 1))
                out.append(HandFrame(h.side, h.host_time_ns, h.seq, h.source, h.joints, wrist=far))
            return BimanualFrame(out[0], out[1])

    app = TeleopApp(_cfg(steps=3), source=FarSource(n_live=10))
    app.run()
    pos = app.sink.state().wrist
    assert pos["left"].position == pytest.approx((-0.25, 0.0, 0.25))
    assert pos["right"].position == pytest.approx((0.25, 0.0, 0.25))


def test_parse_sim_args_accepts_hyphens_and_bare_bools() -> None:
    from teleop.app import parse_sim_config

    cfg = parse_sim_config(
        ["--source", "replay", "--replay-file", "x.jsonl", "--headless", "--steps", "5"]
    )
    assert cfg.source == "replay"
    assert cfg.replay_file == Path("x.jsonl")
    assert cfg.headless is True
    assert cfg.steps == 5


def test_calibrate_from_replay_writes_ranges(tmp_path: Path, capsys) -> None:
    from teleop.commands import calibrate
    from teleop.retarget.calibration import Calibration

    out = tmp_path / "calib" / "t.json"
    rc = calibrate.main(
        [
            "--source",
            "replay",
            "--replay-file",
            str(FIXTURE),
            "--hand",
            "leap",
            "--out",
            str(out),
            "--seconds",
            "0.5",
            "--prep-seconds",
            "0",
        ]
    )
    assert rc == 0
    text = capsys.readouterr().out
    assert "FLAT HAND" in text and "FIST" in text
    calib = Calibration.from_json(out)
    assert set(calib.ranges) == {"left", "right"}
    assert "if_mcp_act" in calib.ranges["right"]


@needs_leap
@pytest.mark.lerobot
def test_record_two_short_episodes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("lerobot")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from teleop.commands import record

    root = tmp_path / "ds"
    rc = record.main(
        [
            "--episodes",
            "2",
            "--max-seconds",
            "1",
            "--root",
            str(root),
            "--task",
            "pick up the cube",
            "--hand",
            "leap",
            "--source",
            "synthetic",
            "--cameras",
            '[{"name": "front", "width": 32, "height": 32}]',
        ]
    )
    assert rc == 0
    ds = LeRobotDataset("local/teleop", root=root)
    assert ds.num_episodes == 2
    item = ds[0]
    assert "observation.images.front" in item
    rows = [
        json.loads(line)
        for line in (root / "meta" / "teleop_episodes.jsonl").read_text().splitlines()
    ]
    assert [r["episode_index"] for r in rows] == [0, 1]
    for key in ("score", "success", "calibration", "wrist_mode", "source_status", "latency"):
        assert key in rows[0]
    for row in rows:
        status = row["source_status"]
        assert status["connected"] is True
        # loop-observed gap between new frames; bounded by the 1 s episode cap
        assert 0.0 < status["max_gap_s"] < 1.0
        assert row["max_cube_z"] == pytest.approx(0.02, abs=0.01)
        assert len(row["score_curve"]) >= 1
