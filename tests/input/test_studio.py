"""Rokoko Studio JSON v3 UDP/LZ4 parser, StudioSource, LatestBuffer, and ReplaySource."""

from __future__ import annotations

import base64
import json
import math
import socket
import time
from pathlib import Path
from typing import Any

import lz4.block
import lz4.frame
import pytest
from hypothesis import given
from hypothesis import strategies as st

from teleop.config import StudioConfig
from teleop.input.buffer import LatestBuffer
from teleop.input.replay import ReplaySource
from teleop.input.studio import StudioParser, StudioSource
from teleop.types import OPENXR_JOINTS, HandFrame, Side

FIXTURES = Path(__file__).parent.parent / "fixtures" / "studio"


def _load_payloads(name: str) -> list[bytes]:
    path = FIXTURES / f"{name}.jsonl"
    if not path.is_file():
        pytest.skip(f"fixture {path.name} not captured yet")
    return [
        base64.b64decode(json.loads(line)["payload_b64"])
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def _norm(q: tuple[float, float, float, float]) -> float:
    return math.sqrt(sum(c * c for c in q))


def _payload(hand_pos: dict[str, float], bones: dict[str, Any] | None = None) -> bytes:
    ident = {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}

    def bone(pos: dict[str, float] | None = None, rot: dict[str, float] | None = None) -> Any:
        return {"position": pos or {"x": 0.0, "y": 0.0, "z": 0.0}, "rotation": rot or ident}

    body: dict[str, Any] = {
        "rightLowerArm": bone(),
        "rightHand": bone(hand_pos),
        "rightIndexProximal": bone(),
        "rightIndexMedial": bone(rot={"x": 0.0, "y": 0.0, "z": math.sin(0.2), "w": math.cos(0.2)}),
    }
    body.update(bones or {})
    actor = {"name": "a", "meta": {"hasGloves": True, "hasRightGlove": True}, "body": body}
    return json.dumps({"scene": {"timestamp": 1.5, "actors": [actor]}}).encode()


def test_parses_both_hands_and_lz4_from_coil_on_fixture() -> None:
    parser = StudioParser()
    payloads = _load_payloads("coil_on")
    frames = [f for i, p in enumerate(payloads) for f in parser.parse(p, 1000 + i)]
    for side in ("left", "right"):
        sided = [f for f in frames if f.side == side]
        assert sided
        for f in sided:
            assert f.source == "studio" and len(f.joints) >= 15
            assert set(f.joints) <= set(OPENXR_JOINTS)
            assert all(abs(_norm(q) - 1.0) < 1e-3 for q in f.joints.values())
            assert f.wrist is not None and f.wrist.position is not None

    expected = StudioParser().parse(payloads[0], 5)
    for packed in (lz4.frame.compress(payloads[0]), lz4.block.compress(payloads[0])):
        assert StudioParser().parse(packed, 5) == expected


def test_y_up_to_z_up_wrist_position_and_orientation() -> None:
    f = StudioParser().parse(_payload({"x": 1.0, "y": 2.0, "z": 3.0}), 1)[0]
    assert f.wrist is not None and f.wrist.position == pytest.approx((1.0, -3.0, 2.0))
    assert f.device_time_us == 1_500_000
    assert f.joints["index_intermediate"] == pytest.approx((0.0, 0.0, math.sin(0.2), math.cos(0.2)))

    s = math.sin(math.pi / 4)
    hand = {
        "position": {"x": 0.0, "y": 0.0, "z": 0.0},
        "rotation": {"x": 0.0, "y": s, "z": 0.0, "w": s},
    }
    f_rot = StudioParser().parse(_payload({"x": 0.0, "y": 0.0, "z": 0.0}, {"rightHand": hand}), 1)[
        0
    ]
    assert f_rot.wrist is not None
    assert f_rot.wrist.orientation == pytest.approx((0.0, 0.0, s, s), abs=1e-9)


@given(st.binary())
def test_garbage_payload_never_raises(data: bytes) -> None:
    out = StudioParser().parse(data, 1)
    assert out == [] or all(f.source == "studio" for f in out)


def test_studio_source_receives_udp_and_reports_liveness() -> None:
    payload = _load_payloads("coil_on")[0]
    src = StudioSource(StudioConfig(host="127.0.0.1", port=0))
    src.start()
    try:
        tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        tx.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, max(len(payload) + 1024, 65536))
        try:
            deadline = time.monotonic() + 0.8
            while time.monotonic() < deadline and src.latest().right is None:
                tx.sendto(payload, ("127.0.0.1", src.bound_port()))
                time.sleep(0.02)
        finally:
            tx.close()
        assert src.latest().right is not None and src.latest().left is not None
        assert src.liveness().bound is True and src.liveness().last_right_frame_ns is not None
    finally:
        src.stop()


def test_latest_buffer_keeps_newest_and_tracks_staleness() -> None:
    def mk(side: Side, t_ns: int, dev: int | None = None, seq: int = 0) -> HandFrame:
        return HandFrame(
            side=side, host_time_ns=t_ns, seq=seq, source="t", joints={}, device_time_us=dev
        )

    buf = LatestBuffer()
    buf.put(mk("left", 1_000_000_000, dev=20, seq=1))
    buf.put(mk("left", 1_050_000_000, dev=15, seq=2))
    buf.put(mk("right", 1_050_000_000, dev=25, seq=3))
    got = buf.get()
    assert got.left is not None and got.left.seq == 2
    assert got.right is not None and got.right.seq == 3
    assert buf.status(now_ns=1_100_000_000).connected is True
    assert buf.status(now_ns=1_400_000_000).connected is False
    assert buf.status(now_ns=1_100_000_000).timestamp_anomalies == 1


def test_replay_source_streams_recorded_jsonl_and_skips_malformed_lines(tmp_path: Path) -> None:
    raw_b64 = base64.b64encode(_payload({"x": 0.1, "y": 0.2, "z": 0.3})).decode()
    rec = lambda t_ns: json.dumps({"t_ns": t_ns, "src": "studio", "payload_b64": raw_b64})  # noqa: E731
    cap = tmp_path / "cap.jsonl"
    cap.write_text("\n".join([rec(0), "{bad json", rec(5_000_000), rec(10_000_000)]) + "\n")

    src = ReplaySource(cap, speed=100.0, loop=False)
    src.start()
    try:
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and src.status().frames < 3:
            time.sleep(0.01)
        assert src.status().frames == 3 and src.latest().right is not None
        assert src.last_error is None
    finally:
        src.stop()
