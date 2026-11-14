"""T-195: batched ACT inference worker, replay stub and checkpoint schema checks."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from teleop.policy.act_runtime import (
    WRIST_POS,
    InferenceWorker,
    ObsBatch,
    ReplayPolicy,
    load_act_policy,
    pretrained_dir,
)
from teleop.policy.manifest import ManifestError
from teleop.policy.schema import ACTION_NAMES, CAMERAS, IMAGE_HW, STATE_NAMES

H, W = IMAGE_HW
CHUNK = 30


def _images() -> dict[str, NDArray[np.uint8]]:
    return {c: np.zeros((H, W, 3), dtype=np.uint8) for c in CAMERAS}


def _state(x: float = 0.0) -> NDArray[np.float64]:
    s = np.zeros(len(STATE_NAMES))
    s[WRIST_POS.start] = x
    return s


class CountingPolicy:
    def __init__(self, sleep_s: float = 0.0, fail: bool = False) -> None:
        self.sleep_s = sleep_s
        self.fail = fail
        self.batches: list[int] = []

    def predict_chunk(self, batch: ObsBatch) -> NDArray[np.float32]:
        self.batches.append(batch.size)
        assert batch.state.shape == (batch.size, len(STATE_NAMES))
        assert batch.state.dtype == np.float32
        for cam in CAMERAS:
            assert batch.images[cam].shape == (batch.size, H, W, 3)
        if self.sleep_s:
            time.sleep(self.sleep_s)
        if self.fail:
            raise RuntimeError("cuda fell over")
        out = np.zeros((batch.size, CHUNK, len(ACTION_NAMES)), dtype=np.float32)
        out[:, :, 0] = batch.state[:, WRIST_POS.start, None]  # echo the bay's own wrist x
        return out


def _wait(pred: Any, timeout_s: float = 5.0) -> None:
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        if pred():
            return
        time.sleep(0.002)
    raise AssertionError("condition never reached")


# --------------------------------------------------------------------------- worker


def test_worker_batches_two_bays() -> None:
    policy = CountingPolicy()
    worker = InferenceWorker(policy, autostart=False)
    try:
        worker.submit(1, _state(0.1), _images(), tick=0)
        worker.submit(2, _state(0.2), _images(), tick=5)
        worker.start()
        _wait(lambda: worker.latest(1) is not None and worker.latest(2) is not None)
        assert policy.batches == [2]
        one, two = worker.latest(1), worker.latest(2)
        assert one is not None and two is not None
        assert one[0] == 0 and two[0] == 5
        assert one[1].shape == (CHUNK, len(ACTION_NAMES))
        assert one[1][0, 0] == pytest.approx(0.1) and two[1][0, 0] == pytest.approx(0.2)
        assert worker.healthy() and worker.p95_ms() is not None
    finally:
        worker.close()


def test_worker_latest_wins_per_bay() -> None:
    policy = CountingPolicy()
    worker = InferenceWorker(policy, autostart=False)
    try:
        worker.submit(1, _state(0.1), _images(), tick=0)
        worker.submit(1, _state(0.3), _images(), tick=5)
        worker.start()
        _wait(lambda: worker.latest(1) is not None)
        got = worker.latest(1)
        assert got is not None and got[0] == 5 and got[1][0, 0] == pytest.approx(0.3)
        assert policy.batches == [1]
    finally:
        worker.close()


def test_worker_submit_does_not_block() -> None:
    worker = InferenceWorker(CountingPolicy(sleep_s=0.2))
    try:
        worker.submit(1, _state(), _images(), tick=0)
        t0 = time.perf_counter()
        for k in range(20):
            worker.submit(1, _state(), _images(), tick=k)
        assert (time.perf_counter() - t0) < 0.05
    finally:
        worker.close()


def test_worker_over_budget_turns_unhealthy() -> None:
    worker = InferenceWorker(CountingPolicy(sleep_s=0.04))
    try:
        for k in range(40):
            if not worker.healthy():
                break
            worker.submit(1, _state(), _images(), tick=k)
            _wait(
                lambda k=k: (
                    (got := worker.latest(1)) is not None and got[0] == k or not worker.healthy()
                )
            )
        assert not worker.healthy()
        assert worker.calls == 10
        assert "budget" in worker.unhealthy_reason
    finally:
        worker.close()


def test_worker_over_budget_streak_resets() -> None:
    times = iter([0.04] * 9 + [0.0] + [0.04] * 9)

    class Varying(CountingPolicy):
        def predict_chunk(self, batch: ObsBatch) -> NDArray[np.float32]:
            self.sleep_s = next(times)
            return super().predict_chunk(batch)

    worker = InferenceWorker(Varying())
    try:
        for k in range(19):
            worker.submit(1, _state(), _images(), tick=k)
            _wait(lambda k=k: (got := worker.latest(1)) is not None and got[0] == k)
        assert worker.healthy()
    finally:
        worker.close()


def test_worker_single_call_over_timeout_turns_unhealthy() -> None:
    worker = InferenceWorker(CountingPolicy(sleep_s=0.06), timeout_ms=50.0)
    try:
        worker.submit(1, _state(), _images(), tick=0)
        _wait(lambda: not worker.healthy())
        assert "timeout" in worker.unhealthy_reason
    finally:
        worker.close()


def test_worker_exception_turns_unhealthy() -> None:
    worker = InferenceWorker(CountingPolicy(fail=True))
    try:
        worker.submit(1, _state(), _images(), tick=0)
        _wait(lambda: not worker.healthy())
        assert "cuda fell over" in worker.unhealthy_reason
        assert worker.latest(1) is None
        worker.submit(1, _state(), _images(), tick=1)  # ignored, never raises
    finally:
        worker.close()


def test_worker_bad_chunk_shape_turns_unhealthy() -> None:
    class Wrong:
        def predict_chunk(self, batch: ObsBatch) -> NDArray[np.float32]:
            return np.zeros((batch.size, CHUNK, 7), dtype=np.float32)

    worker = InferenceWorker(Wrong())
    try:
        worker.submit(1, _state(), _images(), tick=0)
        _wait(lambda: not worker.healthy())
        assert "shape" in worker.unhealthy_reason
    finally:
        worker.close()


def test_worker_close_and_forget() -> None:
    worker = InferenceWorker(CountingPolicy())
    worker.submit(1, _state(), _images(), tick=0)
    _wait(lambda: worker.latest(1) is not None)
    worker.forget(1)
    assert worker.latest(1) is None
    worker.close()
    assert not worker.healthy() and "closed" in worker.unhealthy_reason


# --------------------------------------------------------------------------- replay stub


def _line(n: int = 50) -> NDArray[np.float32]:
    a = np.zeros((n, len(ACTION_NAMES)), dtype=np.float32)
    a[:, 0] = np.linspace(0.0, 0.049, n)
    a[:, 6] = np.arange(n)  # a marker per row
    return a


def test_replay_policy_nearest_row_and_padding() -> None:
    a = _line()
    stub = ReplayPolicy(a, chunk=10, skip_m=0.0)
    batch = ObsBatch(
        np.stack([_state(0.0), _state(0.02), _state(0.06)]).astype(np.float32),
        {c: np.zeros((3, H, W, 3), dtype=np.uint8) for c in CAMERAS},
    )
    out = stub.predict_chunk(batch)
    assert out.shape == (3, 10, len(ACTION_NAMES))
    assert out[0, 0, 6] == 0 and out[0, 9, 6] == 9
    assert out[1, 0, 6] == 20
    assert out[2, 0, 6] == 49 and out[2, 9, 6] == 49  # past the end: hold the last row


def test_replay_policy_skips_a_dwell() -> None:
    a = np.zeros((30, len(ACTION_NAMES)), dtype=np.float32)
    a[10:, 0] = np.linspace(0.0, 0.04, 20)  # 10 rows standing still, then a push
    a[:, 6] = np.arange(30)
    assert ReplayPolicy(a, chunk=5).start_index(np.zeros(3)) >= 10


def test_replay_policy_rejects_bad_shape() -> None:
    with pytest.raises(ValueError):
        ReplayPolicy(np.zeros((5, 3), dtype=np.float32))


# --------------------------------------------------------------------------- checkpoint


def _features(
    state: int = len(STATE_NAMES), action: int = len(ACTION_NAMES), cams: tuple[str, ...] = CAMERAS
) -> dict[str, Any]:
    ins: dict[str, Any] = {"observation.state": {"type": "STATE", "shape": [state]}}
    for cam in cams:
        ins[f"observation.images.{cam}"] = {"type": "VISUAL", "shape": [3, H, W]}
    return {
        "type": "act",
        "input_features": ins,
        "output_features": {"action": {"type": "ACTION", "shape": [action]}},
    }


def write_checkpoint(root: Path, **kw: Any) -> Path:
    model = root / "checkpoints" / "last" / "pretrained_model"
    model.mkdir(parents=True)
    (model / "config.json").write_text(json.dumps(_features(**kw)), encoding="utf-8")
    return root


@pytest.mark.parametrize(
    ("kw", "needle"),
    [
        ({"state": 20}, "observation.state"),
        ({"action": 14}, "action"),
        ({"cams": ("wrist",)}, "observation.images.top"),
        ({"cams": ("wrist", "top", "side")}, "unexpected cameras"),
    ],
)
def test_load_rejects_schema_mismatch(tmp_path: Path, kw: dict[str, Any], needle: str) -> None:
    write_checkpoint(tmp_path, **kw)
    with pytest.raises(ManifestError, match=needle):
        load_act_policy(tmp_path, "cpu")


def test_load_rejects_missing_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="no ACT checkpoint"):
        load_act_policy(tmp_path, "cpu")


def test_pretrained_dir_layouts(tmp_path: Path) -> None:
    flat = tmp_path / "flat"
    flat.mkdir()
    (flat / "config.json").write_text("{}", encoding="utf-8")
    assert pretrained_dir(flat) == flat
    run = tmp_path / "run"
    for step in ("000500", "001000"):
        d = run / "checkpoints" / step / "pretrained_model"
        d.mkdir(parents=True)
        (d / "config.json").write_text("{}", encoding="utf-8")
    assert pretrained_dir(run) == run / "checkpoints" / "001000" / "pretrained_model"


# --------------------------------------------------------------------------- real ACT (lerobot)


def _save_act(root: Path, *, tiny: bool) -> Path:
    """A randomly initialised ACT checkpoint with this schema's features and fitted processors."""
    torch: Any = pytest.importorskip("torch")
    pytest.importorskip("lerobot")
    from lerobot.configs.types import FeatureType, PolicyFeature  # pyright: ignore
    from lerobot.policies.act.configuration_act import ACTConfig  # pyright: ignore
    from lerobot.policies.act.modeling_act import ACTPolicy  # pyright: ignore
    from lerobot.policies.factory import make_pre_post_processors  # pyright: ignore

    ins: dict[str, Any] = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(len(STATE_NAMES),))
    }
    for cam in CAMERAS:
        ins[f"observation.images.{cam}"] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, H, W))
    outs = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(len(ACTION_NAMES),))}
    small: dict[str, Any] = (
        {"dim_model": 64, "n_heads": 2, "dim_feedforward": 128, "n_encoder_layers": 1}
        if tiny
        else {}
    )
    cfg: Any = ACTConfig(
        input_features=ins,
        output_features=outs,
        chunk_size=CHUNK,
        n_action_steps=CHUNK,
        pretrained_backbone_weights=None,  # no download
        device="cpu",
        **small,
    )
    torch.manual_seed(0)
    policy: Any = ACTPolicy(cfg)
    stats: dict[str, dict[str, Any]] = {
        "observation.state": {"mean": np.zeros(len(STATE_NAMES)), "std": np.ones(len(STATE_NAMES))},
        "action": {"mean": np.zeros(len(ACTION_NAMES)), "std": np.ones(len(ACTION_NAMES))},
    }
    for cam in CAMERAS:
        stats[f"observation.images.{cam}"] = {
            "mean": np.full((3, 1, 1), 0.5),
            "std": np.full((3, 1, 1), 0.25),
        }
    pre, post = make_pre_post_processors(cfg, dataset_stats=stats)
    out = root / "pretrained_model"
    policy.save_pretrained(out)
    pre.save_pretrained(out)
    post.save_pretrained(out)
    return out


@pytest.mark.lerobot
def test_load_real_act_checkpoint_predicts_a_chunk(tmp_path: Path) -> None:
    path = _save_act(tmp_path, tiny=True)
    policy = load_act_policy(path, "cpu")
    batch = ObsBatch(
        np.zeros((2, len(STATE_NAMES)), dtype=np.float32),
        {c: np.zeros((2, H, W, 3), dtype=np.uint8) for c in CAMERAS},
    )
    out = policy.predict_chunk(batch)
    assert out.shape == (2, CHUNK, len(ACTION_NAMES)) and out.dtype == np.float32
    assert np.isfinite(out).all()


def _p95_ms(policy: Any, n: int = 60) -> float:
    rng = np.random.default_rng(0)
    batch = ObsBatch(
        rng.standard_normal((2, len(STATE_NAMES))).astype(np.float32),
        {c: rng.integers(0, 255, (2, H, W, 3), dtype=np.uint8) for c in CAMERAS},
    )
    for _ in range(5):  # warm-up
        policy.predict_chunk(batch)
    ms: list[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        policy.predict_chunk(batch)
        ms.append((time.perf_counter() - t0) * 1e3)
    return float(np.percentile(ms, 95))


@pytest.mark.hardware
def test_act_p95_latency_b2(tmp_path: Path) -> None:
    """Laptop latency check (B=2). ``ACT_CHECKPOINT``/``ACT_DEVICE`` pick a real checkpoint;
    otherwise a randomly initialised full-size ACT (ResNet-18) is timed on ``ACT_DEVICE``."""
    device = os.environ.get("ACT_DEVICE", "cuda")
    ckpt = os.environ.get("ACT_CHECKPOINT")
    path = Path(ckpt) if ckpt else _save_act(tmp_path, tiny=False)
    p95 = _p95_ms(load_act_policy(path, device))
    print(f"\nACT p95 B=2 on {device}: {p95:.1f} ms")
    assert p95 <= 30.0


def test_worker_threads_are_daemons() -> None:
    worker = InferenceWorker(CountingPolicy())
    try:
        names = {t.name: t.daemon for t in threading.enumerate()}
        assert names.get("act-inference") is True
    finally:
        worker.close()
