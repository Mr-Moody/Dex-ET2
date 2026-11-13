import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("lerobot")

from teleop.config import CameraConfig  # noqa: E402
from teleop.record.recorder import EpisodeRecorder  # noqa: E402
from teleop.record.schema import build_features  # noqa: E402

pytestmark = pytest.mark.lerobot

FPS = 10


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")


def _ramp(n: int, start: int = 0, stop: int | None = None) -> np.ndarray:
    """Actions idle until ``start``, then moving, then idle again from ``stop``."""
    stop = n if stop is None else stop
    a = np.zeros((n, 3), np.float32)
    for i in range(n):
        a[i] = (min(max(i - start + 1, 0), stop - start)) * 0.1
    return a


def _build(root: Path, episodes: list[tuple[np.ndarray, dict]]) -> Path:
    cam = CameraConfig(name="front", width=32, height=32)
    feats = build_features(["a", "b", "c", "d"], ["x", "y", "z"], [cam])
    rec = EpisodeRecorder(f"local/{root.name}", root, FPS, feats, "test_robot")
    rng = np.random.default_rng(0)
    for actions, meta in episodes:
        rec.start_episode("lift")
        for act in actions:
            rec.add_frame(
                np.zeros(4, np.float32),
                act,
                {"front": rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)},
            )
        rec.end_episode(meta)
    rec.close()
    return root


def _good(n: int = 40, **meta) -> tuple[np.ndarray, dict]:
    return _ramp(n, 5, n - 5), {"score": 5.0, "success": True, **meta}


def _qc(root, **kw):
    from teleop.process.qc import QCThresholds, run_qc

    return run_qc(root, QCThresholds(**kw))


def test_too_short_rejected(tmp_path):
    root = _build(tmp_path / "a", [_good(12), _good(40)])
    r = _qc(root)
    assert not r[0].keep and "too_short" in r[0].reasons
    assert r[1].keep and r[1].reasons == []


def test_dropout_rejected(tmp_path):
    root = _build(
        tmp_path / "a",
        [
            _good(source_status={"connected": False, "frames": 10}),
            _good(source_status={"connected": True, "max_gap_s": 0.9}),
            _good(source_status={"connected": True, "max_gap_s": 0.1}),
        ],
    )
    r = _qc(root)
    assert [x.keep for x in r] == [False, False, True]
    assert "input_dropout" in r[0].reasons and "input_dropout" in r[1].reasons


def test_hotspot_and_no_progress_and_missing_meta(tmp_path):
    root = _build(
        tmp_path / "a",
        [
            _good(hotspot_events=[[1, "restart"]]),
            (_ramp(40, 5, 35), {"score": 1.0}),
            (_ramp(40, 5, 35), {"score": 1.0, "max_cube_z": 0.08}),
            (_ramp(40, 5, 35), {}),
        ],
    )
    r = _qc(root)
    assert r[0].reasons == ["hotspot_event"]
    assert r[1].reasons == ["no_progress"]
    assert r[2].keep
    assert r[3].keep  # all meta missing: rules that need it are skipped


def test_out_of_range(tmp_path):
    from teleop.process.qc import QCThresholds, run_qc

    root = _build(tmp_path / "a", [_good()])
    ranges = {"x": (0.0, 10.0), "y": (0.0, 10.0), "z": (0.0, 10.0)}
    assert run_qc(root, QCThresholds(), ranges=ranges)[0].reasons == []
    tight = {**ranges, "x": (0.0, 0.5)}
    assert run_qc(root, QCThresholds(), ranges=tight)[0].reasons == ["out_of_range"]
    assert run_qc(root, QCThresholds())[0].keep


def test_trim_idle_ends(tmp_path):
    root = _build(tmp_path / "a", [(_ramp(40, 10, 30), {"score": 5.0})])
    (res,) = _qc(root)
    assert res.keep and res.trim == (9, 30)


def test_trim_keeps_one_second_and_score_frames(tmp_path):
    from teleop.process.qc import compute_trim

    a = _ramp(40, 20, 24)  # short motion
    s, e = compute_trim(a, FPS, 1e-3, None)
    assert e - s >= FPS and s >= 0 and e <= 40
    s, e = compute_trim(_ramp(40, 20, 30), FPS, 1e-3, [3, 38])
    assert s <= 3 and e > 38


def test_empty_dataset_report(tmp_path):
    from teleop.process.qc import QCThresholds, run_qc, write_clean

    root = tmp_path / "empty"
    EpisodeRecorder(f"local/{root.name}", root, FPS, build_features(["a"], ["x"], []), "r").close()
    results = run_qc(root, QCThresholds())
    assert results == []
    out = tmp_path / "out"
    write_clean(root, results, out)
    report = json.loads((out / "qc_report.json").read_text())
    assert report["episodes"] == [] and report["totals"]["kept"] == 0
    assert "no episodes" in report["empty_reason"]
    assert "no episodes" in (out / "qc_report.md").read_text()
    # a root that does not exist at all behaves the same
    assert run_qc(tmp_path / "missing", QCThresholds()) == []


def _two_sessions(tmp_path):
    a = _build(tmp_path / "a", [_good(), _good(source_status={"connected": False})])
    b = _build(tmp_path / "b", [(_ramp(40, 10, 30), {"score": 4.0, "success": False})])
    return a, b


def test_merge_two_sessions(tmp_path):
    from teleop.process.qc import QCThresholds, run_qc, write_clean

    a, b = _two_sessions(tmp_path)
    results = run_qc(a, QCThresholds()) + run_qc(b, QCThresholds())
    out = tmp_path / "clean"
    write_clean(a, results, out)
    rows = [
        json.loads(line)
        for line in (out / "meta" / "teleop_episodes.jsonl").read_text().splitlines()
    ]
    assert [r["episode_index"] for r in rows] == [0, 1]
    assert [r["source_root"] for r in rows] == [str(a), str(b)]
    assert [r["source_episode_index"] for r in rows] == [0, 0]
    assert rows[1]["score"] == 4.0
    report = json.loads((out / "qc_report.json").read_text())
    frames = sum(r.trim[1] - r.trim[0] for r in results if r.keep)
    assert report["totals"] == {"episodes": 3, "kept": 2, "rejected": 1, "frames_written": frames}
    md = (out / "qc_report.md").read_text()
    assert "input_dropout" in md and "| episode |" in md


def test_clean_dataset_loads(tmp_path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from teleop.process.annotate import RuleAnnotator
    from teleop.process.qc import QCThresholds, run_qc, write_clean

    a, b = _two_sessions(tmp_path)
    results = run_qc(a, QCThresholds()) + run_qc(b, QCThresholds())
    out = tmp_path / "clean"
    write_clean(a, results, out, annotator=RuleAnnotator())
    ds = LeRobotDataset("local/clean", root=out)
    assert ds.num_episodes == 2
    assert len(ds) == sum(r.trim[1] - r.trim[0] for r in results if r.keep)
    item = ds[0]
    assert tuple(item["observation.images.front"].shape) == (3, 32, 32)
    assert tuple(item["action"].shape) == (3,)
    rows = [
        json.loads(line)
        for line in (out / "meta" / "teleop_episodes.jsonl").read_text().splitlines()
    ]
    assert rows[0]["annotation"]["label"] == "success"
    assert rows[1]["annotation"]["label"] == "failure"


def test_rule_annotator():
    from teleop.process.annotate import Annotation, RuleAnnotator

    ann = RuleAnnotator()
    assert ann.annotate({}, {"success": True, "score": 6}) == Annotation("success", True, "score=6")
    assert ann.annotate({}, {"score": 1.0}).label == "failure"
    assert ann.annotate({}, {}).success is None


def test_cli_process(tmp_path, capsys):
    from teleop.cli import main

    a, b = _two_sessions(tmp_path)
    out = tmp_path / "clean"
    rc = main(["process", "--in", str(a), "--in", str(b), "--out", str(out), "--annotator", "rule"])
    assert rc == 0
    assert (out / "qc_report.json").exists()


def test_missing_max_gap_no_dropout_but_warned(tmp_path):
    from teleop.process.qc import write_clean

    root = _build(tmp_path / "a", [_good(source_status={"connected": True}), _good()])
    r = _qc(root)
    assert all(x.keep for x in r)
    assert r[0].warnings == ["no_max_gap_s"] and r[1].warnings == ["no_source_status"]
    out = tmp_path / "out"
    write_clean(root, r, out)
    report = json.loads((out / "qc_report.json").read_text())
    assert len(report["qc_warnings"]) == 2


def test_merge_mismatched_schema_raises_before_writing(tmp_path):
    from teleop.process.qc import QCThresholds, run_qc, write_clean

    a = _build(tmp_path / "a", [_good()])
    cam = CameraConfig(name="front", width=32, height=32)
    feats = build_features(["a", "b", "c", "d"], ["x", "y", "z"], [cam])
    rec = EpisodeRecorder("local/b", tmp_path / "b", 20, feats, "test_robot")
    rec.start_episode("lift")
    for act in _ramp(60, 5, 55):
        rec.add_frame(np.zeros(4, np.float32), act, {"front": np.zeros((32, 32, 3), np.uint8)})
    rec.end_episode({"score": 5.0})
    rec.close()
    results = run_qc(a, QCThresholds()) + run_qc(tmp_path / "b", QCThresholds())
    assert all(r.keep for r in results)
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="different fps or schema"):
        write_clean(a, results, out)
    assert not out.exists()


def test_all_rejected(tmp_path):
    from teleop.process.qc import write_clean

    root = _build(tmp_path / "a", [_good(12), _good(source_status={"connected": False})])
    r = _qc(root)
    out = tmp_path / "out"
    write_clean(root, r, out)
    report = json.loads((out / "qc_report.json").read_text())
    assert report["totals"]["kept"] == 0 and "all 2 episodes" in report["empty_reason"]
    assert not (out / "meta").exists()


def test_annotator_gets_all_cameras(tmp_path):
    from teleop.process.annotate import Annotation
    from teleop.process.qc import write_clean

    seen = {}

    class Spy:
        def annotate(self, episode_frames, meta):
            seen.update({k: len(v) for k, v in episode_frames.items()})
            return Annotation("x", None, "")

    root = _build(tmp_path / "a", [_good()])
    r = _qc(root)
    write_clean(root, r, tmp_path / "out", annotator=Spy())
    assert seen == {"front": r[0].trim[1] - r[0].trim[0]}


def test_cli_ranges_from_missing_hand_warns(tmp_path, capsys):
    from teleop.cli import main

    root = _build(tmp_path / "a", [_good()])
    argv = ["process", "--in", str(root), "--out", str(tmp_path / "o")]
    assert main([*argv, "--ranges-from-hand", "nope"]) == 0
    assert "warning" in capsys.readouterr().err


def test_hand_ranges_from_mjcf():
    from teleop.commands.process import _hand_ranges

    ranges = _hand_ranges("inspire")
    if ranges is None:
        pytest.skip("hand MJCF not fetched (third_party)")
    assert any(k.startswith("left/") for k in ranges) and any(
        k.startswith("right/") for k in ranges
    )
    assert all(lo < hi for lo, hi in ranges.values())
