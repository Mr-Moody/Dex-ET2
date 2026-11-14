"""Tests for the temporary Rokoko Studio JSON v3 dummy streamer (TODO(rokoko-gloves-swap))."""

from __future__ import annotations

import time

import pytest

from teleop.config import StudioConfig, WristConfig
from teleop.input.dummy_glove import (
    INSPIRE_CTRL_RANGES,
    DummyControlClient,
    DummyGloveUdpServer,
    actuator_targets_to_fractions,
    build_studio_v3_packet,
)
from teleop.input.studio import StudioParser, StudioSource
from teleop.retarget.calibration import Calibration
from teleop.retarget.direct import DirectRetargeter
from teleop.sim.hands import load_hand
from teleop.wrist.auto import AutoWrist


def test_studio_packet_roundtrips_wrist_and_inspire_fingers() -> None:
    desired_rad = {
        "pinky": 0.4,
        "ring": 0.4,
        "middle": 0.6,
        "index": 0.75,
        "thumb_proximal": 0.35,
        "thumb_yaw": 0.6,
    }
    fracs = actuator_targets_to_fractions(desired_rad)
    pkt = build_studio_v3_packet(
        1.25,
        wrist_pos_zup=(0.04, -0.02, 0.015),
        wrist_quat_zup=(0.0, 0.0, 0.0, 1.0),
        finger_fractions=fracs,
    )
    frames = StudioParser().parse(pkt, host_time_ns=1_000_000)
    assert len(frames) == 1
    frame = frames[0]
    assert frame.side == "right"
    assert frame.wrist is not None and frame.wrist.position is not None
    assert frame.wrist.position == pytest.approx((0.04, -0.02, 0.015), abs=1e-6)

    wrist = AutoWrist(WristConfig(mode="auto"))
    wt = wrist.wrist(frame)
    assert wt.pose.position == pytest.approx((0.04, -0.02, 0.015), abs=1e-6)

    spec = load_hand("inspire")
    ret = DirectRetargeter(spec, {"right": INSPIRE_CTRL_RANGES}, Calibration.default(spec))
    out = ret.retarget(frame).positions
    for name, expected in desired_rad.items():
        assert out[name] == pytest.approx(expected, abs=1e-5)


def test_dummy_udp_server_streams_to_studio_source() -> None:
    studio = StudioSource(StudioConfig(host="127.0.0.1", port=0))
    studio.start()
    assert studio._sock is not None
    studio_port = int(studio._sock.getsockname()[1])
    dummy = DummyGloveUdpServer(
        studio_host="127.0.0.1",
        studio_port=studio_port,
        control_host="127.0.0.1",
        control_port=0,
        hz=60.0,
    ).start()
    client = DummyControlClient("127.0.0.1", dummy.control_port)
    try:
        client.send(
            active=True,
            wrist_pos=(0.02, 0.01, -0.01),
            wrist_quat=(0.0, 0.0, 0.0, 1.0),
            fingers={"thumb_yaw": 0.5, "index": 0.4},
        )
        deadline = time.monotonic() + 2.0
        got = None
        while time.monotonic() < deadline:
            r = studio.latest().right
            if (
                r is not None
                and r.wrist is not None
                and r.wrist.position is not None
                and abs(r.wrist.position[0] - 0.02) < 1e-4
            ):
                got = r
                break
            time.sleep(0.02)
        assert got is not None and got.wrist is not None
        assert got.wrist.position == pytest.approx((0.02, 0.01, -0.01), abs=1e-4)
    finally:
        client.close()
        dummy.stop()
        studio.stop()
