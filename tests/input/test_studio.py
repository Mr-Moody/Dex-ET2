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
from teleop.input.studio import StudioParser, StudioSource, quat_conjugate, quat_multiply
from teleop.protocols import PacketParser
from teleop.types import OPENXR_JOINTS

FIXTURES = Path(__file__).parent.parent / "fixtures" / "studio"


def load_payloads(name: str) -> list[bytes]:
    path = FIXTURES / f"{name}.jsonl"
    if not path.is_file():
        pytest.skip(f"fixture {path.name} not captured yet")
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            out.append(base64.b64decode(json.loads(line)["payload_b64"]))
    return out


def norm(q: tuple[float, float, float, float]) -> float:
    return math.sqrt(sum(c * c for c in q))


def test_is_packet_parser():
    assert isinstance(StudioParser(), PacketParser)


def test_parses_both_hands_from_coil_on():
    parser = StudioParser()
    frames = []
    for i, payload in enumerate(load_payloads("coil_on")):
        frames.extend(parser.parse(payload, 1000 + i))
    for side in ("left", "right"):
        sided = [f for f in frames if f.side == side]
        assert sided
        for f in sided:
            assert f.source == "studio"
            assert len(f.joints) >= 15
            assert set(f.joints) <= set(OPENXR_JOINTS)
            for q in f.joints.values():
                assert abs(norm(q) - 1.0) < 1e-3
            assert f.wrist is not None
            assert f.wrist.position is not None
            assert abs(norm(f.wrist.orientation) - 1.0) < 1e-3
            assert f.device_time_us is not None
    seqs = [f.seq for f in frames]
    assert len(set(seqs)) == len(seqs)


def test_coil_off_has_no_wrist_position():
    parser = StudioParser()
    frames = [f for p in load_payloads("coil_off") for f in parser.parse(p, 1)]
    assert frames
    assert all(f.wrist is None or f.wrist.position is None for f in frames)


def test_lz4_payload_parses_same_as_plain():
    plain = load_payloads("coil_on")[0]
    expected = StudioParser().parse(plain, 5)
    assert len(expected) == 2
    for packed in (lz4.frame.compress(plain), lz4.block.compress(plain)):
        assert StudioParser().parse(packed, 5) == expected


def test_lz4_fixture_parses():
    parser = StudioParser()
    frames = [f for p in load_payloads("lz4") for f in parser.parse(p, 1)]
    assert {f.side for f in frames} == {"left", "right"}


def test_parser_tolerates_timestamp_jump():
    plain = load_payloads("coil_on")
    a, b = json.loads(plain[1]), json.loads(plain[0])
    a["scene"]["timestamp"] = 100.0
    b["scene"]["timestamp"] = 5.0  # clock stepped backwards
    parser = StudioParser()
    fa = parser.parse(json.dumps(a).encode(), 111)
    fb = parser.parse(json.dumps(b).encode(), 222)
    assert fa and fb
    assert {f.host_time_ns for f in fa} == {111}
    assert {f.host_time_ns for f in fb} == {222}
    assert fa[0].device_time_us == 100_000_000
    assert fb[0].device_time_us == 5_000_000


@given(st.binary())
def test_garbage_payload_returns_empty(data: bytes):
    out = StudioParser().parse(data, 1)
    assert out == [] or all(f.source == "studio" for f in out)


@pytest.mark.parametrize(
    "payload",
    [
        b"{}",
        b'{"scene": 3}',
        b'{"scene": {"actors": "x"}}',
        b'{"scene": {"actors": [1, null, {"body": 5}]}}',
        b'{"scene": {"actors": [{"meta": {"hasGloves": true}, "body": {"leftHand": {}}}]}}',
        b"[1,2,3]",
        b"\xff\xfe",
    ],
)
def test_malformed_structures_return_empty(payload: bytes):
    assert StudioParser().parse(payload, 1) == []


def test_actor_selection_by_name():
    payload = json.loads(load_payloads("coil_on")[0])
    actor = payload["scene"]["actors"][0]
    other = json.loads(json.dumps(actor))
    other["name"] = "other"
    payload["scene"]["actors"] = [other, actor]
    raw = json.dumps(payload).encode()
    assert StudioParser(actor="newton").parse(raw, 1)
    assert StudioParser(actor="missing").parse(raw, 1) == []
    assert StudioParser().parse(raw, 1)  # first actor with gloves


def test_actor_without_gloves_skipped():
    payload = json.loads(load_payloads("coil_on")[0])
    payload["scene"]["actors"][0]["meta"]["hasGloves"] = False
    assert StudioParser().parse(json.dumps(payload).encode(), 1) == []


def test_single_glove_gives_one_frame():
    payload = json.loads(load_payloads("coil_on")[0])
    payload["scene"]["actors"][0]["meta"]["hasLeftGlove"] = False
    frames = StudioParser().parse(json.dumps(payload).encode(), 1)
    assert [f.side for f in frames] == ["right"]


def test_local_rotation_inverts_parent():
    """child_world == parent_world * local, so local = conj(parent) * child."""
    parent = (0.0, 0.0, math.sin(math.pi / 8), math.cos(math.pi / 8))  # 45 deg about z
    local = (math.sin(math.pi / 12), 0.0, 0.0, math.cos(math.pi / 12))  # 30 deg about x
    child = quat_multiply(parent, local)
    got = quat_multiply(quat_conjugate(parent), child)
    assert got == pytest.approx(local, abs=1e-9)


def _payload(hand_pos: dict[str, float], bones: dict[str, Any] | None = None) -> bytes:
    ident = {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}

    def bone(pos=None, rot=None):
        return {"position": pos or {"x": 0, "y": 0, "z": 0}, "rotation": rot or ident}

    body: dict[str, Any] = {
        "rightLowerArm": bone(),
        "rightHand": bone(hand_pos),
        "rightIndexProximal": bone(),
        "rightIndexMedial": bone(rot={"x": 0.0, "y": 0.0, "z": math.sin(0.2), "w": math.cos(0.2)}),
    }
    body.update(bones or {})
    actor = {"name": "a", "meta": {"hasGloves": True, "hasRightGlove": True}, "body": body}
    return json.dumps({"scene": {"timestamp": 1.5, "actors": [actor]}}).encode()


def test_y_up_to_z_up_wrist_position():
    frames = StudioParser().parse(_payload({"x": 1.0, "y": 2.0, "z": 3.0}), 1)
    assert len(frames) == 1
    f = frames[0]
    assert f.wrist is not None
    assert f.wrist.position == pytest.approx((1.0, -3.0, 2.0))
    assert f.device_time_us == 1_500_000
    assert f.joints["index_intermediate"] == pytest.approx((0.0, 0.0, math.sin(0.2), math.cos(0.2)))
    assert f.joints["index_proximal"] == pytest.approx((0.0, 0.0, 0.0, 1.0))


def test_wrist_orientation_converted_to_z_up():
    # 90 deg about Studio +Y (up) must become 90 deg about world +Z
    s = math.sin(math.pi / 4)
    hand = {"position": {"x": 0, "y": 0, "z": 0}, "rotation": {"x": 0, "y": s, "z": 0, "w": s}}
    f = StudioParser().parse(_payload({"x": 0, "y": 0, "z": 0}, {"rightHand": hand}), 1)[0]
    assert f.wrist is not None
    assert f.wrist.orientation == pytest.approx((0.0, 0.0, s, s), abs=1e-9)


def test_missing_position_gives_none():
    raw = json.loads(_payload({"x": 0, "y": 0, "z": 0}))
    del raw["scene"]["actors"][0]["body"]["rightHand"]["position"]
    f = StudioParser().parse(json.dumps(raw).encode(), 1)[0]
    assert f.wrist is not None and f.wrist.position is None


def _udp_sender(payload: bytes) -> socket.socket:
    """UDP socket able to send ``payload`` in one datagram.

    macOS caps outgoing datagrams at the socket send buffer (default 9216 bytes,
    ``net.inet.udp.maxdgram``), smaller than a captured Studio frame; raise it.
    """
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    tx.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, max(len(payload) + 1024, 65536))
    return tx


def _send_some(port: int, payload: bytes, n: int = 3) -> None:
    tx = _udp_sender(payload)
    try:
        for _ in range(n):
            tx.sendto(payload, ("127.0.0.1", port))
            time.sleep(0.02)
    finally:
        tx.close()


def test_studio_source_receives_udp():
    payload = load_payloads("coil_on")[0]
    src = StudioSource(StudioConfig(host="127.0.0.1", port=0))
    src.start()
    try:
        port = src.bound_port()
        tx = _udp_sender(payload)
        try:
            deadline = time.monotonic() + 0.5
            while time.monotonic() < deadline and src.latest().right is None:
                tx.sendto(payload, ("127.0.0.1", port))
                time.sleep(0.02)
        finally:
            tx.close()
        assert src.latest().right is not None
        assert src.latest().left is not None
        assert src.status().frames >= 2
    finally:
        src.stop()


def test_studio_source_tees_raw(tmp_path: Path):
    payload = load_payloads("coil_on")[0]
    out = tmp_path / "tee.jsonl"
    src = StudioSource(StudioConfig(host="127.0.0.1", port=0, tee_raw=out))
    src.start()
    try:
        port = src.bound_port()
        _send_some(port, payload, 3)
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and src.status().frames < 6:
            time.sleep(0.01)
    finally:
        src.stop()
    recs = [json.loads(x) for x in out.read_text().splitlines()]
    assert len(recs) == 3
    assert recs[0]["src"] == "studio"
    assert base64.b64decode(recs[0]["payload_b64"]) == payload
    assert isinstance(recs[0]["t_ns"], int)


def test_nan_timestamp_keeps_frames():
    base = json.loads(load_payloads("coil_on")[0])
    for bad in (float("nan"), float("inf"), float("-inf"), True, "NaN"):
        base["scene"]["timestamp"] = bad
        raw = json.dumps(base).encode()
        if isinstance(bad, float):
            assert b'"timestamp": ' + json.dumps(bad).encode() in raw  # real non-finite literal
        frames = StudioParser().parse(raw, 1)
        assert len(frames) == 2
        assert all(f.device_time_us is None for f in frames)


class _FailingTee:
    def write(self, _s: str) -> int:
        raise OSError("disk full")

    def flush(self) -> None: ...

    def close(self) -> None: ...


def test_tee_write_error_does_not_kill_source(tmp_path: Path):
    payload = load_payloads("coil_on")[0]
    src = StudioSource(StudioConfig(host="127.0.0.1", port=0, tee_raw=tmp_path / "t.jsonl"))
    src.start()
    try:
        real_tee = src._tee  # pyright: ignore[reportPrivateUsage]
        assert real_tee is not None
        real_tee.close()  # release the Windows file lock
        src._tee = _FailingTee()  # type: ignore[assignment]  # pyright: ignore[reportPrivateUsage]
        _send_some(src.bound_port(), payload, 5)
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and src.status().frames < 2:
            time.sleep(0.01)
        assert src.latest().right is not None
        assert src.last_error is None
        assert src._thread is not None and src._thread.is_alive()  # pyright: ignore[reportPrivateUsage]
    finally:
        src.stop()


def test_start_failure_does_not_leak_socket(tmp_path: Path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    src = StudioSource(StudioConfig(host="127.0.0.1", port=0, tee_raw=blocker / "tee.jsonl"))
    with pytest.raises(OSError):
        src.start()
    assert src._sock is None  # pyright: ignore[reportPrivateUsage]


def test_bind_failure_does_not_truncate_tee(tmp_path: Path):
    tee_path = tmp_path / "keep.jsonl"
    tee_path.write_text("previous recording\n")
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as holder:
        holder.bind(("127.0.0.1", 0))
        port = holder.getsockname()[1]
        src = StudioSource(StudioConfig(host="127.0.0.1", port=port, tee_raw=tee_path))
        with pytest.raises(OSError):
            src.start()
    assert tee_path.read_text() == "previous recording\n"
