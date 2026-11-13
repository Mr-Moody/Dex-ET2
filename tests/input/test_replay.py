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
