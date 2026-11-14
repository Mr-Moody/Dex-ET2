from __future__ import annotations

import json
import math
import socket
import struct
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from teleop import registry
from teleop.app import make_source
from teleop.config import SimConfig, StudioConfig
from teleop.input.buffer import LatestBuffer
from teleop.input.replay import ReplaySource
from teleop.input.rgmp import (
    DEFAULT_SOLVER_PORT,
    MSG_PREFIX_DATA,
    MSG_PREFIX_DEFINITION,
    MSG_PREFIX_DISCONNECT,
    RCSP_MARKER,
    RcspPayloadType,
    RgmpSession,
    SdkSource,
    StreamDefinition,
    decode_data,
    decode_disconnect,
    encode_frame,
    parse_frames,
    sensor_to_zup_quat,
    sensor_to_zup_vec,
    subscribe_usb_devices,
)
from teleop.protocols import InputSource
from teleop.types import OPENXR_JOINTS


def _rcsp_frame(ptype: int, obj: dict[str, Any]) -> bytes:
    raw = json.dumps(obj).encode("utf-8")
    return bytes((RCSP_MARKER, 1, 8, ptype)) + len(raw).to_bytes(4, "little") + raw


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise EOFError("unexpected EOF")
        buf.extend(chunk)
    return bytes(buf)


def test_split_frames_across_reads() -> None:
    defn_bytes = b'{"protocol_name":"RGMP","device_id":7,"groups":[]}'
    data_bytes = struct.pack("<IIQff", 7, 0, 123_456, 0.5, -1.25)
    disc_bytes = struct.pack("<I", 7)
    partial_tail = struct.pack("<II", MSG_PREFIX_DATA, 16) + b"\x01\x02\x03"

    wire = (
        encode_frame(MSG_PREFIX_DEFINITION, defn_bytes)
        + encode_frame(MSG_PREFIX_DATA, data_bytes)
        + encode_frame(MSG_PREFIX_DISCONNECT, disc_bytes)
        + partial_tail
    )

    buf = bytearray()
    got: list[tuple[int, bytes]] = []
    for byte in wire:
        buf.append(byte)
        got.extend(parse_frames(buf))

    assert got == [
        (MSG_PREFIX_DEFINITION, defn_bytes),
        (MSG_PREFIX_DATA, data_bytes),
        (MSG_PREFIX_DISCONNECT, disc_bytes),
    ]
    assert bytes(buf) == partial_tail


def test_decode_packed_unaligned_floats() -> None:
    # A single 4-byte FLOAT places the following DOUBLE, INT64, and UINT64 at offsets
    # 20, 28, and 36 inside the frame payload (4-byte aligned, NOT 8-byte aligned).
    defn_json = json.dumps(
        {
            "protocol_name": "RGMP",
            "protocol_version": "2.0.0",
            "device_id": 42,
            "device_type": "solved_hand",
            "timestamp_epoch": "device_boot",
            "device_info": {"hand": "right"},
            "groups": [
                {
                    "name": "mixed_unaligned",
                    "expected_rate_hz": 83.33,
                    "streams": [
                        {
                            "target_frame": "lead_f32",
                            "measure_type": "CUSTOM",
                            "custom_label": "lead",
                            "data_type": "FLOAT",
                        },
                        {
                            "target_frame": "unaligned_f64",
                            "measure_type": "CUSTOM",
                            "custom_label": "dbl",
                            "data_type": "DOUBLE",
                        },
                        {
                            "target_frame": "unaligned_i64",
                            "measure_type": "CUSTOM",
                            "custom_label": "i64",
                            "data_type": "INT64",
                        },
                        {
                            "target_frame": "unaligned_u64",
                            "measure_type": "CUSTOM",
                            "custom_label": "u64",
                            "data_type": "UINT64",
                        },
                        {
                            "target_frame": "quat_xyzw",
                            "measure_type": "ORIENTATION",
                            "data_type": "FLOAT[4]",
                        },
                        {
                            "target_frame": "mat_2x3",
                            "measure_type": "CUSTOM",
                            "custom_label": "mat",
                            "data_type": "FLOAT[2, 3]",
                        },
                    ],
                }
            ],
        }
    )
    defn = StreamDefinition.from_json(defn_json)
    assert defn.device_id == 42
    assert defn.timestamp_epoch == "device_boot"
    assert defn.groups[0].name == "mixed_unaligned"

    packed = struct.pack(
        "<IIQfdqQ4f6f",
        42,
        0,
        987_654_321,
        1.5,
        math.pi,
        -9_000_000_000_123,
        18_000_000_000_456,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
    )
    unaligned_view = memoryview(b"pad" + packed)[3:]
    dev_id, group_id, ts_us, values = decode_data(defn, unaligned_view)

    assert dev_id == 42
    assert group_id == 0
    assert ts_us == 987_654_321
    assert values["lead_f32"] == pytest.approx((1.5,))
    assert values["unaligned_f64"] == pytest.approx((math.pi,))
    assert values["unaligned_i64"] == (-9_000_000_000_123,)
    assert values["unaligned_u64"] == (18_000_000_000_456,)
    assert values["quat_xyzw"] == pytest.approx((0.0, 0.0, 0.0, 1.0))
    assert values["mat_2x3"] == pytest.approx((1.0, 2.0, 3.0, 4.0, 5.0, 6.0))


def test_disconnect_prefix_3() -> None:
    session = RgmpSession()
    defn = {
        "protocol_name": "RGMP",
        "protocol_version": "2.0.0",
        "device_id": 11,
        "device_type": "solved_hand",
        "timestamp_epoch": "device_boot",
        "device_info": {"hand": "left"},
        "groups": [
            {
                "name": "joints_local",
                "expected_rate_hz": 83.33,
                "streams": [
                    {
                        "target_frame": j,
                        "measure_type": "ORIENTATION",
                        "data_type": "FLOAT[4]",
                    }
                    for j in OPENXR_JOINTS
                ],
            }
        ],
    }
    session.feed(encode_frame(MSG_PREFIX_DEFINITION, json.dumps(defn).encode("utf-8")), 100)
    assert 11 in session.definitions

    quats = [c for _ in OPENXR_JOINTS for c in (0.0, 0.0, 0.0, 1.0)]
    data_payload = struct.pack(f"<IIQ{4 * len(OPENXR_JOINTS)}f", 11, 0, 5000, *quats)
    frames = session.feed(encode_frame(MSG_PREFIX_DATA, data_payload), 200)
    assert len(frames) == 1
    assert frames[0].side == "left"

    disc_payload = struct.pack("<I", 11)
    assert decode_disconnect(disc_payload) == 11
    after_disc = session.feed(encode_frame(MSG_PREFIX_DISCONNECT, disc_payload), 300)
    assert after_disc == []
    assert 11 not in session.definitions

    stale = session.feed(encode_frame(MSG_PREFIX_DATA, data_payload), 400)
    assert stale == []


@given(st.lists(st.binary(max_size=128), max_size=20))
def test_parse_frames_fuzz_never_raises(chunks: list[bytes]) -> None:
    buf = bytearray()
    for chunk in chunks:
        buf.extend(chunk)
        frames = parse_frames(buf)
        for prefix, payload in frames:
            assert isinstance(prefix, int)
            assert 0 <= prefix < (1 << 32)
            assert isinstance(payload, bytes)


def test_rcsp_subscribes_usb_serial_devices_only_and_skips_events() -> None:
    received: list[dict[str, Any]] = []

    def serve(srv: socket.socket) -> None:
        conn, _ = srv.accept()
        with conn:
            conn.settimeout(2.0)
            hdr = _recv_exact(conn, 8)
            assert hdr[:4] == bytes((RCSP_MARKER, 1, 8, RcspPayloadType.COMMAND))
            plen = int.from_bytes(hdr[4:8], "little")
            cmd1 = json.loads(_recv_exact(conn, plen).decode("utf-8"))
            received.append(cmd1)

            conn.sendall(
                _rcsp_frame(
                    RcspPayloadType.EVENT,
                    {"Publisher": "Logs", "Topic": "Warning", "Data": "test"},
                )
            )
            conn.sendall(
                _rcsp_frame(
                    RcspPayloadType.RESPONSE_OK,
                    {
                        "Status": "Ok",
                        "TrackId": cmd1["TrackId"],
                        "Response": {
                            "Devices": [
                                {
                                    "DeviceId": 101,
                                    "DeviceType": "Smartgloves",
                                    "Connections": ["Serial", "TCP"],
                                    "Updatable": True,
                                    "IsBootloader": False,
                                },
                                {
                                    "DeviceId": 102,
                                    "DeviceType": "Smartgloves",
                                    "Connections": ["TCP"],
                                    "Updatable": True,
                                    "IsBootloader": False,
                                },
                                {
                                    "DeviceId": 103,
                                    "DeviceType": "Smartgloves",
                                    "Connections": ["Serial"],
                                    "Updatable": True,
                                    "IsBootloader": True,
                                },
                            ]
                        },
                    },
                )
            )

            hdr2 = _recv_exact(conn, 8)
            plen2 = int.from_bytes(hdr2[4:8], "little")
            cmd2 = json.loads(_recv_exact(conn, plen2).decode("utf-8"))
            received.append(cmd2)
            conn.sendall(
                _rcsp_frame(
                    RcspPayloadType.RESPONSE_OK,
                    {"Status": "Ok", "TrackId": cmd2["TrackId"]},
                )
            )

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        port = int(srv.getsockname()[1])
        t = threading.Thread(target=serve, args=(srv,), daemon=True)
        t.start()
        subscribed = subscribe_usb_devices("127.0.0.1", port, timeout_s=2.0)
        t.join(timeout=2.0)

    assert subscribed == [101]
    assert [c["Command"] for c in received] == ["ListDevices", "SubscribeToData"]
    assert received[1]["Arguments"] == {"DeviceId": 101, "Connection": "Serial"}


def test_sdk_source_streams_openxr_and_local_hands_and_resolves_in_registry() -> None:
    cls = registry.resolve(registry.SOURCES, "rokoko-sdk")
    assert cls is SdkSource
    built = make_source(SimConfig(source="rokoko-sdk"))
    assert isinstance(built, SdkSource)
    assert isinstance(built, InputSource)
    assert built.port == DEFAULT_SOLVER_PORT

    s45 = math.sin(math.pi / 4.0)
    c45 = math.cos(math.pi / 4.0)

    defn_left = {
        "protocol_name": "RGMP",
        "protocol_version": "2.0.0",
        "device_id": 1,
        "device_type": "solved_hand",
        "timestamp_epoch": "unix_epoch",
        "device_info": {"hand": "left", "friendly_name": "Smartgloves II (left)"},
        "groups": [
            {
                "name": "joints_openxr",
                "expected_rate_hz": 83.33,
                "streams": [
                    {
                        "target_frame": j,
                        "reference_frame": "xr_base",
                        "measure_type": "TRANSFORM",
                        "data_type": "FLOAT[7]",
                    }
                    for j in OPENXR_JOINTS
                ],
            }
        ],
    }
    defn_right = {
        "protocol_name": "RGMP",
        "protocol_version": "2.0.0",
        "device_id": 2,
        "device_type": "solved_hand",
        "timestamp_epoch": "unix_epoch",
        "device_info": {"hand": "right", "friendly_name": "Smartgloves II (right)"},
        "groups": [
            {
                "name": "joints_local",
                "expected_rate_hz": 83.33,
                "streams": [
                    {
                        "target_frame": "wrist",
                        "measure_type": "TRANSFORM",
                        "data_type": "FLOAT[7]",
                    },
                    *[
                        {
                            "target_frame": j,
                            "measure_type": "ORIENTATION",
                            "data_type": "FLOAT[4]",
                        }
                        for j in OPENXR_JOINTS
                        if j != "wrist"
                    ],
                ],
            }
        ],
    }

    left_vals: list[float] = []
    for j in OPENXR_JOINTS:
        q = (s45, 0.0, 0.0, c45) if j == "index_proximal" else (0.0, 0.0, 0.0, 1.0)
        left_vals.extend((0.0, 0.0, 0.0, *q))
    left_data = struct.pack(f"<IIQ{7 * len(OPENXR_JOINTS)}f", 1, 0, 10_000, *left_vals)

    right_vals: list[float] = [0.2, 0.3, -0.4, 0.0, 0.0, 0.0, 1.0]
    for j in OPENXR_JOINTS:
        if j == "wrist":
            continue
        q = (s45, 0.0, 0.0, c45) if j == "index_proximal" else (0.0, 0.0, 0.0, 1.0)
        right_vals.extend(q)
    right_data = struct.pack(f"<IIQ7f{4 * (len(OPENXR_JOINTS) - 1)}f", 2, 0, 12_000, *right_vals)

    wire = b"".join(
        [
            encode_frame(MSG_PREFIX_DEFINITION, json.dumps(defn_left).encode("utf-8")),
            encode_frame(MSG_PREFIX_DEFINITION, json.dumps(defn_right).encode("utf-8")),
            encode_frame(MSG_PREFIX_DATA, left_data),
            encode_frame(MSG_PREFIX_DATA, right_data),
        ]
    )

    stop_srv = threading.Event()

    def serve_rgmp(srv: socket.socket) -> None:
        conn, _ = srv.accept()
        with conn:
            try:
                for i in range(0, len(wire), 257):
                    conn.sendall(wire[i : i + 257])
                    time.sleep(0.001)
                stop_srv.wait(timeout=2.0)
            except (BrokenPipeError, ConnectionResetError):
                pass

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        solver_port = int(srv.getsockname()[1])
        t = threading.Thread(target=serve_rgmp, args=(srv,), daemon=True)
        t.start()

        src = SdkSource(StudioConfig(host="127.0.0.1", port=solver_port), rcsp_port=None)
        src.start()
        try:
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                bi = src.latest()
                if bi.left is not None and bi.right is not None:
                    break
                time.sleep(0.01)
            bi = src.latest()
            assert bi.left is not None
            assert bi.right is not None
            assert bi.left.device_time_us == 10_000
            assert bi.right.device_time_us == 12_000
            assert set(bi.left.joints).issubset(OPENXR_JOINTS)
            assert len(bi.left.joints) == len(OPENXR_JOINTS)
            assert bi.left.wrist is not None and bi.left.wrist.position is None
            assert bi.right.wrist is not None and bi.right.wrist.position is not None
            for side_frame in (bi.left, bi.right):
                assert side_frame.joints["index_proximal"] == pytest.approx(
                    (s45, 0.0, 0.0, c45), abs=1e-5
                )
                for q in side_frame.joints.values():
                    assert math.hypot(*q) == pytest.approx(1.0, abs=1e-4)
        finally:
            stop_srv.set()
            src.stop()
            t.join(timeout=2.0)


def test_sensor_to_zup_quat_preserves_identity_and_conjugates_axes() -> None:
    assert sensor_to_zup_quat((0.0, 0.0, 0.0, 1.0)) == (0.0, 0.0, 0.0, 1.0)
    assert sensor_to_zup_vec((0.1, 0.2, -0.3)) == pytest.approx((0.1, -0.2, 0.3))
    s = math.sin(0.3)
    c = math.cos(0.3)
    assert sensor_to_zup_quat((0.0, s, 0.0, c)) == pytest.approx((0.0, -s, 0.0, c))
    assert sensor_to_zup_quat((0.0, 0.0, s, c)) == pytest.approx((0.0, 0.0, -s, c))


def test_multi_group_emit_deduplicates_per_tick_and_carries_forearm_coil_pos() -> None:
    native_bones = [
        "hand",
        "thumb_metacarpal",
        "thumb_proximal",
        "thumb_distal",
        "thumb_tip",
        *(
            f"{f}_{p}"
            for f in ("index", "middle", "ring", "little")
            for p in ("metacarpal", "proximal", "medial", "distal", "tip")
        ),
    ]
    defn = {
        "protocol_name": "RGMP",
        "protocol_version": "2.0.0",
        "device_id": 9,
        "device_type": "solved_hand",
        "timestamp_epoch": "unix_epoch",
        "device_info": {"hand": "right"},
        "groups": [
            {
                "name": "joints_local",
                "expected_rate_hz": 83.33,
                "streams": [
                    {
                        "target_frame": "forearm",
                        "measure_type": "TRANSFORM",
                        "data_type": "FLOAT[7]",
                    },
                    *[
                        {
                            "target_frame": b,
                            "measure_type": "ORIENTATION",
                            "data_type": "FLOAT[4]",
                        }
                        for b in native_bones
                    ],
                ],
            },
            {
                "name": "joints_openxr",
                "expected_rate_hz": 83.33,
                "streams": [
                    {
                        "target_frame": j,
                        "reference_frame": "xr_base",
                        "measure_type": "TRANSFORM",
                        "data_type": "FLOAT[7]",
                    }
                    for j in OPENXR_JOINTS
                ],
            },
        ],
    }

    session = RgmpSession()
    buf = LatestBuffer()
    session.feed(encode_frame(MSG_PREFIX_DEFINITION, json.dumps(defn).encode("utf-8")), 100)

    g0_vals = [0.15, -0.25, -0.35, 0.0, 0.0, 0.0, 1.0] + [0.0, 0.0, 0.0, 1.0] * len(native_bones)
    g0_pkt = struct.pack(f"<IIQ7f{4 * len(native_bones)}f", 9, 0, 7777, *g0_vals)
    g1_vals = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] * len(OPENXR_JOINTS)
    g1_pkt = struct.pack(f"<IIQ{7 * len(OPENXR_JOINTS)}f", 9, 1, 7777, *g1_vals)

    emitted = session.feed(
        encode_frame(MSG_PREFIX_DATA, g0_pkt) + encode_frame(MSG_PREFIX_DATA, g1_pkt), 200
    )
    for f in emitted:
        buf.put(f)

    assert len(emitted) == 1
    assert buf.status(now_ns=200).timestamp_anomalies == 0
    assert emitted[0].wrist is not None
    assert emitted[0].wrist.position == pytest.approx((0.15, 0.25, 0.35))
    assert emitted[0].wrist.orientation == pytest.approx((0.0, 0.0, 0.0, 1.0))

    defn_local_only = {**defn, "device_id": 10, "groups": [defn["groups"][0]]}
    session.feed(
        encode_frame(MSG_PREFIX_DEFINITION, json.dumps(defn_local_only).encode("utf-8")), 300
    )
    local_pkt = struct.pack(f"<IIQ7f{4 * len(native_bones)}f", 10, 0, 8888, *g0_vals)
    (local_frame,) = session.feed(encode_frame(MSG_PREFIX_DATA, local_pkt), 400)
    assert len(local_frame.joints) == len(OPENXR_JOINTS)
    assert "index_intermediate" in local_frame.joints
    assert "palm" in local_frame.joints
    assert local_frame.wrist is not None
    assert local_frame.wrist.orientation == pytest.approx((0.0, 0.0, 0.0, 1.0))
    assert local_frame.wrist.position == pytest.approx((0.15, 0.25, 0.35))


def test_sdk_source_tee_raw_and_replay_roundtrip(tmp_path: Path) -> None:
    tee_file = tmp_path / "rgmp_capture.jsonl"
    defn = {
        "protocol_name": "RGMP",
        "protocol_version": "2.0.0",
        "device_id": 3,
        "device_type": "solved_hand",
        "timestamp_epoch": "unix_epoch",
        "device_info": {"hand": "right"},
        "groups": [
            {
                "name": "joints_local",
                "expected_rate_hz": 83.33,
                "streams": [
                    {
                        "target_frame": j,
                        "measure_type": "ORIENTATION",
                        "data_type": "FLOAT[4]",
                    }
                    for j in OPENXR_JOINTS
                ],
            }
        ],
    }
    quats = [c for _ in OPENXR_JOINTS for c in (0.0, 0.0, 0.0, 1.0)]
    pkt = struct.pack(f"<IIQ{4 * len(OPENXR_JOINTS)}f", 3, 0, 50_000, *quats)
    wire = encode_frame(MSG_PREFIX_DEFINITION, json.dumps(defn).encode("utf-8")) + encode_frame(
        MSG_PREFIX_DATA, pkt
    )

    stop_srv = threading.Event()

    def serve(srv: socket.socket) -> None:
        conn, _ = srv.accept()
        with conn:
            conn.sendall(wire)
            stop_srv.wait(timeout=2.0)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        port = int(srv.getsockname()[1])
        t = threading.Thread(target=serve, args=(srv,), daemon=True)
        t.start()

        src = SdkSource(StudioConfig(host="127.0.0.1", port=port, tee_raw=tee_file), rcsp_port=None)
        src.start()
        try:
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline and src.latest().right is None:
                time.sleep(0.01)
            assert src.latest().right is not None
        finally:
            stop_srv.set()
            src.stop()
            t.join(timeout=2.0)

    assert tee_file.is_file()
    replay = ReplaySource(tee_file, parser=RgmpSession(), speed=100.0, loop=False)
    replay.start()
    try:
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and replay.latest().right is None:
            time.sleep(0.01)
        frame = replay.latest().right
        assert frame is not None
        assert frame.source == "rokoko-sdk"
        assert frame.device_time_us == 50_000
    finally:
        replay.stop()
