import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("lerobot")

from teleop.policy.schema import (  # noqa: E402
    ACTION_NAMES,
    CAMERAS,
    FPS,
    STATE_NAMES,
    lerobot_features,
)
from teleop.record.recorder import EpisodeRecorder  # noqa: E402

pytestmark = pytest.mark.lerobot

HW = (32, 32)  # the AV1 video encoder crashes the process on 16 x 16 frames
HAND = 0.3  # inside every Inspire ctrlrange
RANGES = {
    "hand/pinky": (0.0, 1.57),
    "hand/ring": (0.0, 1.57),
    "hand/middle": (0.0, 1.5),
    "hand/index": (0.0, 1.5),
    "hand/thumb_proximal": (0.1, 0.57),
    "hand/thumb_yaw": (0.0, 1.308),
}
ALL_SOURCES = frozenset({"glove", "scripted", "dmp"})


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")


def _actions(idle_head: int, moving: int, idle_tail: int) -> np.ndarray:
    """Wrist x still for ``idle_head`` frames, moving 2 mm per frame, then still again."""
    n = idle_head + moving + idle_tail
    a = np.zeros((n, len(ACTION_NAMES)), np.float32)
    a[:, 6:] = HAND
    for k in range(n):
        a[k, 0] = min(max(k - idle_head + 1, 0), moving) * 0.002
    return a


def _meta(**over) -> dict:
    meta = {
        "source": "scripted",
        "fault": "flap",
        "bay": 1,
        "signature_id": "Jam_Sig_1",
        "signature": [0.035, 1.2, 2.4, 0.3],
        "latch_pose": {"x": 0.5},
        "success": True,
        "bolt_x_final": 0.054,
        "max_tau_hand_inf": 2.0,
        "max_arm_ratio": 0.4,
        "aborted_reason": None,
    }
    meta.update(over)
    return meta


def _good(**over) -> tuple[np.ndarray, dict]:
    return _actions(5, 30, 5), _meta(**over)


def _build(root: Path, episodes: list[tuple[np.ndarray, dict]]) -> Path:
    rec = EpisodeRecorder(f"local/{root.name}", root, FPS, lerobot_features(HW), "panda_inspire")
    rng = np.random.default_rng(0)
    for actions, meta in episodes:
        rec.start_episode(f"recover latch: {meta.get('fault')}")
        for act in actions:
            rec.add_frame(
                np.zeros(len(STATE_NAMES), np.float32),
                act,
                {cam: rng.integers(0, 255, (*HW, 3), dtype=np.uint8) for cam in CAMERAS},
            )
        rec.end_episode(meta)
    rec.close()
    return root


def _build_legacy(root: Path) -> Path:
    from teleop.config import CameraConfig
    from teleop.record.schema import build_features

    feats = build_features(["a"], ["x"], [CameraConfig(name="front", width=HW[1], height=HW[0])])
    rec = EpisodeRecorder(f"local/{root.name}", root, 10, feats, "test_robot")
    rec.start_episode("lift")
    for i in range(30):
        rec.add_frame(
            np.zeros(1, np.float32),
            np.array([i * 0.1], np.float32),
            {"front": np.zeros((*HW, 3), np.uint8)},
        )
    rec.end_episode({"score": 5.0})
    rec.close()
    return root


def _qc(root: Path, sources: frozenset[str] = ALL_SOURCES, **kw):
    from teleop.process.atc_qc import AtcQCThresholds, run_atc_qc

    return run_atc_qc(root, AtcQCThresholds(**kw), sources, ranges=RANGES)


def test_drops_each_reason(tmp_path):
    out_of_range = _actions(5, 30, 5)
    out_of_range[20, 6] = 1.6  # hand/pinky above 1.57
    cases = [
        (_good(), None),
        (_good(success=False), "not successful"),
        (_good(source="act"), "source excluded"),
        ((_actions(5, 10, 5), _meta()), "too short"),
        (_good(max_gap_s=0.5), "timestamp gap"),
        (_good(max_tau_hand_inf=12.0), "torque breach"),
        (_good(max_arm_ratio=0.8), "torque breach"),
        ((out_of_range, _meta()), "action out of range"),
    ]
    root = _build(tmp_path / "a", [ep for ep, _ in cases])
    results = _qc(root)
    assert [r.keep for r in results] == [True] + [False] * (len(cases) - 1)
    for r, (_, reason) in zip(results, cases, strict=True):
        assert r.reasons == ([] if reason is None else [reason])


def test_trims_idle_head_and_tail(tmp_path):
    from teleop.process.atc_qc import atc_trim

    assert atc_trim(_actions(10, 30, 10), 1e-3) == (10, 40)
    assert atc_trim(np.zeros((20, len(ACTION_NAMES)), np.float32), 1e-3) == (0, 0)
    root = _build(tmp_path / "a", [(_actions(10, 30, 10), _meta())])
    (r,) = _qc(root)
    assert r.keep and r.trim == (10, 40)


def test_idle_only_episode_is_too_short(tmp_path):
    root = _build(tmp_path / "a", [(_actions(30, 0, 0), _meta())])
    (r,) = _qc(root)
    assert r.reasons == ["too short"]


def test_sources_filter(tmp_path):
    root = _build(tmp_path / "a", [_good(source="scripted"), _good(source="glove")])
    results = _qc(root, sources=frozenset({"glove"}))
    assert [r.reasons for r in results] == [["source excluded"], []]


def test_gap_from_source_status_and_missing_gap_warned(tmp_path):
    root = _build(
        tmp_path / "a",
        [
            _good(source_status={"connected": True, "max_gap_s": 0.3}),
            _good(source_status={"connected": True, "max_gap_s": 0.1}),
            _good(),
        ],
    )
    results = _qc(root)
    assert [r.reasons for r in results] == [["timestamp gap"], [], []]
    assert [r.warnings for r in results] == [[], [], ["no_max_gap_s"]]


def test_is_atc_dataset(tmp_path):
    from teleop.process.atc_qc import is_atc_dataset

    assert is_atc_dataset(_build(tmp_path / "atc", [_good()]))
    assert not is_atc_dataset(_build_legacy(tmp_path / "legacy"))
    assert not is_atc_dataset(tmp_path / "missing")


def test_clean_sidecar_carries_labels(tmp_path):
    from teleop.process.qc import open_dataset, read_sidecar, write_clean

    root = _build(
        tmp_path / "a",
        [_good(success=False), (_actions(10, 30, 10), _meta(fault="pin", source="glove"))],
    )
    out = tmp_path / "clean"
    write_clean(root, _qc(root), out)
    rows = read_sidecar(out)
    assert list(rows) == [0]
    row = rows[0]
    assert row["fault"] == "pin" and row["source"] == "glove"
    assert row["signature_id"] == "Jam_Sig_1" and row["signature"] == [0.035, 1.2, 2.4, 0.3]
    assert row["success"] is True and row["bay"] == 1
    assert row["source_episode_index"] == 1
    ds = open_dataset(out)
    assert int(ds.meta.total_episodes) == 1 and len(ds) == 30
    assert ds[0]["task"] == "recover latch: pin"


def test_summary_counts(tmp_path):
    from teleop.process.atc_qc import summarize
    from teleop.process.qc import read_sidecar

    root = _build(
        tmp_path / "a",
        [
            _good(),
            _good(success=False),
            _good(fault="pin"),
            _good(fault="pin", source="glove"),
        ],
    )
    results = _qc(root)
    text = summarize(results, {root: read_sidecar(root)})
    lines = {line.split()[0]: line.split()[1:] for line in text.splitlines() if line.strip()}
    assert lines["fault"] == ["glove", "scripted", "total"]
    assert lines["flap"] == ["-", "1/2", "1/2"]
    assert lines["pin"] == ["1/1", "1/1", "2/2"]
    assert lines["total"] == ["1/1", "2/3", "3/4"]


def test_mixed_datasets_exit_2(tmp_path, capsys):
    from teleop.commands.process import main

    atc = _build(tmp_path / "atc", [_good()])
    legacy = _build_legacy(tmp_path / "legacy")
    out = tmp_path / "clean"
    rc = main(["--in", str(atc), "--in", str(legacy), "--out", str(out)])
    assert rc == 2
    assert "teleop process: cannot mix ATC and legacy datasets" in capsys.readouterr().err
    assert not out.exists()


def test_unknown_source_exits_2(tmp_path, capsys):
    from teleop.commands.process import main

    atc = _build(tmp_path / "atc", [_good()])
    rc = main(["--in", str(atc), "--out", str(tmp_path / "clean"), "--sources", "glove,robot"])
    assert rc == 2
    assert "robot" in capsys.readouterr().err


def test_cli_atc_merges_sessions_and_prints_summary(tmp_path, capsys):
    from teleop.commands.process import main
    from teleop.process.qc import read_sidecar

    a = _build(tmp_path / "a", [_good(), _good(source="dmp")])
    b = _build(tmp_path / "b", [_good(source="glove", fault="pin")])
    out = tmp_path / "clean"
    rc = main(["--in", str(a), "--in", str(b), "--out", str(out), "--sources", "glove,scripted"])
    assert rc == 0
    printed = capsys.readouterr().out
    assert "kept 2/3" in printed and "flap" in printed and "pin" in printed
    assert [r["source"] for r in read_sidecar(out).values()] == ["scripted", "glove"]
    report = json.loads((out / "qc_report.json").read_text(encoding="utf-8"))
    assert [e["reasons"] for e in report["episodes"]] == [[], ["source excluded"], []]
