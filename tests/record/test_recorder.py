import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("lerobot")

from teleop.config import CameraConfig  # noqa: E402
from teleop.record.recorder import EpisodeRecorder  # noqa: E402
from teleop.record.schema import build_features  # noqa: E402

pytestmark = pytest.mark.lerobot

REPO = "local/test"


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")


def _features():
    cam = CameraConfig(name="front", width=32, height=32)
    return build_features(["a", "b", "c", "d"], ["x", "y", "z"], [cam])


def _rec(tmp_path: Path) -> EpisodeRecorder:
    return EpisodeRecorder(REPO, tmp_path / "ds", 10, _features(), "test_robot")


def _episode(rec: EpisodeRecorder, n: int) -> None:
    rec.start_episode("pick")
    rng = np.random.default_rng(0)
    for _ in range(n):
        rec.add_frame(
            np.zeros(4, np.float32),
            np.zeros(3, np.float32),
            {"front": rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)},
        )


def _two(tmp_path: Path) -> EpisodeRecorder:
    rec = _rec(tmp_path)
    for i in range(2):
        _episode(rec, 12)
        assert rec.end_episode({"score": i}) == i
    rec.close()
    return rec


def test_roundtrip_two_episodes(tmp_path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    _two(tmp_path)
    ds = LeRobotDataset(REPO, root=tmp_path / "ds")
    assert ds.num_episodes == 2
    assert len(ds) == 24
    item = ds[0]
    for k in ("observation.state", "action", "observation.images.front"):
        assert k in item


def test_teleop_meta_sidecar(tmp_path):
    _two(tmp_path)
    lines = (tmp_path / "ds" / "meta" / "teleop_episodes.jsonl").read_text().splitlines()
    rows = [json.loads(line) for line in lines]
    assert [r["episode_index"] for r in rows] == [0, 1]
    assert rows[1]["score"] == 1


def test_interrupt_mid_episode_dataset_loads(tmp_path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    with pytest.raises(KeyboardInterrupt), _rec(tmp_path) as rec:
        _episode(rec, 12)
        rec.end_episode({})
        _episode(rec, 5)
        raise KeyboardInterrupt
    ds = LeRobotDataset(REPO, root=tmp_path / "ds")
    assert ds.num_episodes == 1
    assert len(ds) == 12


def test_close_idempotent(tmp_path):
    rec = _two(tmp_path)
    rec.close()
    rec.close()


def test_existing_root_raises_clear_error(tmp_path):
    from teleop.record.recorder import RecorderRootExistsError

    root = tmp_path / "ds"
    root.mkdir()
    (root / "x").write_text("y")
    with pytest.raises(RecorderRootExistsError, match="fresh session"):
        EpisodeRecorder(REPO, root, 10, _features(), "test_robot")


def test_discard_then_new_episode_indexes_correctly(tmp_path):
    rec = _rec(tmp_path)
    _episode(rec, 6)
    rec.discard_episode()
    _episode(rec, 6)
    assert rec.end_episode({}) == 0
    _episode(rec, 6)
    assert rec.end_episode({}) == 1
    rec.close()


def test_close_with_zero_episodes_does_not_raise(tmp_path):
    rec = _rec(tmp_path)
    rec.close()
    rec2 = _rec(tmp_path / "b")
    rec2.start_episode("t")
    rec2.discard_episode()
    rec2.close()
