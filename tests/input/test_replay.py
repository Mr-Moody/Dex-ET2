import base64
import json
import time
from pathlib import Path

import pytest

from teleop.input.replay import ReplaySource
from teleop.types import HandFrame


class FakeParser:
    def parse(self, payload: bytes, host_time_ns: int) -> list[HandFrame]:
        return [HandFrame(side="right", host_time_ns=host_time_ns, seq=0, source="fake", joints={})]


def test_replay_uses_injected_parser(tmp_path: Path):
    p = tmp_path / "cap.jsonl"
    b64 = base64.b64encode(b"x").decode()
    lines = [
        json.dumps({"t_ns": i * 10_000_000, "src": "studio", "payload_b64": b64}) for i in range(3)
    ]
    p.write_text("\n".join(lines) + "\n")
    src = ReplaySource(p, parser=FakeParser(), speed=100)
    src.start()
    try:
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and src.status().frames < 3:
            time.sleep(0.01)
        assert src.status().frames >= 3
        assert src.latest().right is not None
    finally:
        src.stop()


def test_replay_missing_file_raises_filenotfound(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        ReplaySource(tmp_path / "nope.jsonl")


def _line(t_ns: int, src: str = "studio") -> str:
    return json.dumps({"t_ns": t_ns, "src": src, "payload_b64": base64.b64encode(b"x").decode()})


def test_replay_unknown_src_raises_at_construction(tmp_path: Path):
    p = tmp_path / "cap.jsonl"
    p.write_text(_line(0, "mystery") + "\n")
    with pytest.raises(ValueError, match="mystery"):
        ReplaySource(p)


def test_replay_skips_malformed_lines(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    p = tmp_path / "cap.jsonl"
    p.write_text(
        "\n".join([_line(0), "{not json", '{"src": "studio"}', _line(1_000_000), _line(2_000_000)])
        + "\n"
    )
    src = ReplaySource(p, parser=FakeParser(), speed=100, loop=False)
    with caplog.at_level("WARNING"):
        src.start()
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and src.status().frames < 3:
            time.sleep(0.01)
        log_deadline = time.monotonic() + 2.0
        while time.monotonic() < log_deadline and not any(
            "malformed" in r.getMessage() for r in caplog.records
        ):
            time.sleep(0.01)
    src.stop()
    assert src.status().frames == 3
    assert src.last_error is None
    assert any("2 malformed" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize(("speed", "max_frames"), [(1.0, 40), (100.0, 600)])
def test_replay_loop_zero_delta_rate_bounded(tmp_path: Path, speed: float, max_frames: int):
    p = tmp_path / "cap.jsonl"
    p.write_text(_line(0) + "\n")
    src = ReplaySource(p, parser=FakeParser(), speed=speed, loop=True)
    src.start()
    time.sleep(0.5)
    src.stop()
    assert 1 <= src.status().frames <= max_frames
