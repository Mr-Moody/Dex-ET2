"""UNO Q edge compute: 1-Euro filters, Mahalanobis OOD gate, RBF-DMP, and loopback RPC."""

from __future__ import annotations

import dataclasses
import math
import socket
import time
from typing import Any

import numpy as np
import pytest

from teleop.atc.types import ConsoleInputState, SkillPrimitive, StallSignature
from teleop.types import Pose
from teleop.unoq import protocol as P
from teleop.unoq.classifier import MahalanobisStallClassifier
from teleop.unoq.client import UnoQClient, UnoQError
from teleop.unoq.dmp import (
    HAND_ACTUATORS,
    compose_pose,
    evaluate_latch_frame,
    evaluate_rbf_dmp,
    fit_rbf_dmp,
    mat_to_quat,
    phase,
    pose_to_vec7,
    quat_to_mat,
    relative_pose,
    so3_exp,
    so3_log,
)
from teleop.unoq.filter import OneEuroFilter, PoseOneEuroFilter
from teleop.unoq.server import UnoQServer

HZ = 83.3
SKU = "ev_latch_v1"
JAM_SIG_1 = (0.035, 2.1, 3.4, 1.2)
PINNED_BOLT = (0.0, 2.3, 3.6, 1.1)


def _rot_deg(a: Pose, b: Pose) -> float:
    r = quat_to_mat(a.orientation).T @ quat_to_mat(b.orientation)
    return math.degrees(float(np.linalg.norm(so3_log(r))))


def _pos_err(a: Pose, b: Pose) -> float:
    assert a.position is not None and b.position is not None
    return float(np.linalg.norm(np.subtract(a.position, b.position)))


def _vec7_pose(v: Any) -> Pose:
    f = [float(x) for x in v]
    return Pose(orientation=(f[3], f[4], f[5], f[6]), position=(f[0], f[1], f[2]))


def _yaw_pose(x: float, y: float, z: float, yaw_deg: float) -> Pose:
    h = math.radians(yaw_deg) / 2
    return Pose(orientation=(0.0, 0.0, math.sin(h), math.cos(h)), position=(x, y, z))


def _synthetic_rescue(duration_s: float = 4.0, hz: float = 60.0) -> dict[str, Any]:
    t = np.arange(0.0, duration_s + 1e-9, 1.0 / hz)
    u = t / duration_s
    pos = np.stack(
        [
            0.02 - 0.06 * np.sin(math.pi * u / 2),
            0.01 * np.sin(2 * math.pi * u),
            0.20 - 0.12 * (3 * u**2 - 2 * u**3),
        ],
        axis=1,
    )
    quats = [
        mat_to_quat(
            so3_exp([0.0, 0.0, 0.6 * ui]) @ so3_exp([0.0, 0.3 * math.sin(math.pi * ui), 0.0])
        )
        for ui in u
    ]
    wrist = np.hstack([pos, np.asarray(quats)])
    n = len(HAND_ACTUATORS)
    fingers = np.stack([0.8 * np.sin(math.pi * u * (1 + i / n)) for i in range(n)], axis=1)
    tau = np.where((u > 0.4) & (u < 0.7), 3.0, 0.2)
    return {"t": t, "wrist": wrist, "fingers": fingers, "tau": tau}


def test_one_euro_and_pose_filters_reduce_jitter_and_keep_quaternion_hemisphere() -> None:
    rng = np.random.default_rng(7)
    f = OneEuroFilter()
    raw = 0.30 + rng.normal(0.0, 0.002, int(4 * HZ))
    out = np.array([float(f(x)) for x in raw])
    settle = int(HZ)
    assert np.std(raw[settle:]) / np.std(out[settle:]) > 3.0

    f = OneEuroFilter()
    for _ in range(80):
        f(rng.normal(0.0, 0.002))
    ys = [float(f(0.05 + rng.normal(0.0, 0.002))) for _ in range(40)]
    first = next(i for i, y in enumerate(ys) if y >= 0.045)
    assert (first + 1) / HZ < 0.100

    pf = PoseOneEuroFilter()
    q = (0.0, 0.0, math.sin(0.3), math.cos(0.3))
    neg = (-q[0], -q[1], -q[2], -q[3])
    for p in (pf(Pose(orientation=o, position=(0.1, 0.0, 0.2))) for o in (q, neg, q, neg)):
        assert abs(float(np.linalg.norm(p.orientation)) - 1.0) < 1e-9
        assert _rot_deg(p, Pose(orientation=q)) < 1e-6


def test_rbf_dmp_fits_under_20ms_reconstructs_and_roundtrips_wire_dict() -> None:
    demo = _synthetic_rescue()
    times: list[float] = []
    fit_ms: list[float] = []
    skill = fit_rbf_dmp(demo["t"], demo["wrist"], demo["fingers"], demo["tau"])
    for _ in range(15):
        t0 = time.perf_counter()
        skill = fit_rbf_dmp(demo["t"], demo["wrist"], demo["fingers"], demo["tau"])
        times.append(time.perf_counter() - t0)
        fit_ms.append(skill.fit_ms)
    assert min(times) < 0.020 and min(fit_ms) < 20.0
    assert skill.n_kernels == 30 and skill.n_dims == 6 + len(HAND_ACTUATORS)

    duration = float(demo["t"][-1])
    for k in range(0, len(demo["t"]), 8):
        s = float(phase(demo["t"][k], duration))
        wrist, fingers, _ = evaluate_latch_frame(skill, s)
        truth = _vec7_pose(demo["wrist"][k])
        assert _pos_err(wrist, truth) < 0.002 and _rot_deg(wrist, truth) < 0.5
        q = np.array([fingers[n] for n in HAND_ACTUATORS])
        assert np.max(np.abs(q - demo["fingers"][k])) < 0.03

    with_offset = dataclasses.replace(skill, touch_offset_m=-0.0123)
    assert SkillPrimitive.from_dict(with_offset.to_dict()) == with_offset


def test_rbf_dmp_se3_warp_rotates_35deg_and_shifts_12cm_accurately() -> None:
    bay1 = _yaw_pose(0.05, 0.0, 0.02, 0.0)
    bay2 = _yaw_pose(1.03, 0.09, 0.02, 35.0)
    demo = _synthetic_rescue()
    world_bay1 = [compose_pose(bay1, _vec7_pose(w)) for w in demo["wrist"]]
    local = np.array([pose_to_vec7(relative_pose(bay1, p)) for p in world_bay1])
    skill = fit_rbf_dmp(demo["t"], local, demo["fingers"], demo["tau"])

    duration = float(demo["t"][-1])
    for k in range(0, len(demo["t"]), 10):
        s = float(phase(demo["t"][k], duration))
        got1, _, _ = evaluate_rbf_dmp(skill, s, bay1)
        got2, _, _ = evaluate_rbf_dmp(skill, s, bay2)
        expect2 = compose_pose(bay2, relative_pose(bay1, world_bay1[k]))
        assert _pos_err(got2, expect2) < 0.002 and _rot_deg(got2, expect2) < 0.5
        assert _pos_err(relative_pose(bay2, got2), relative_pose(bay1, got1)) < 1e-9

    g1, _, _ = evaluate_rbf_dmp(skill, 0.0, bay1)
    g2, _, _ = evaluate_rbf_dmp(skill, 0.0, bay2)
    assert _rot_deg(g1, g2) == pytest.approx(35.0, abs=0.1)


def test_mahalanobis_matches_flap_jam_and_rejects_pinned_bolt_as_ood() -> None:
    clf = MahalanobisStallClassifier()
    assert clf.classify(SKU, JAM_SIG_1).is_ood

    clf.register_prototype(SKU, "Jam_Sig_1", np.array([JAM_SIG_1]))
    match = clf.classify(SKU, (0.0352, 2.3, 3.1, 1.4))
    assert not match.is_ood and match.signature_id == "Jam_Sig_1" and match.distance < 1.5

    pinned = clf.classify(SKU, PINNED_BOLT)
    assert pinned.is_ood and pinned.signature_id is None and pinned.distance >= 7.0
    assert clf.classify("other_sku", JAM_SIG_1).is_ood


def test_protocol_encodes_lines_and_unoq_loopback_roundtrips() -> None:
    line = P.encode({"ok": True, "distance": float("inf"), "z": [1.0, float("nan")]})
    assert P.decode(line) == {"ok": True, "distance": None, "z": [1.0, None]}
    with pytest.raises(P.ProtocolError):
        P.decode_request(b'{"no_op": 1}')

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        dead_port = int(s.getsockname()[1])

    with UnoQClient.connect_or_mock("127.0.0.1", udp_port=dead_port, tcp_port=dead_port) as client:
        assert client.is_mock and client.ping() == "unoq-loopback"
        wrist = _yaw_pose(0.1, -0.2, 0.3, 20.0)
        out_wrist, out_joints = client.filter_step(wrist, {"index_proximal": 0.4}, t_ns=0)
        assert out_wrist is not None and _pos_err(out_wrist, wrist) < 1e-9
        assert out_joints == pytest.approx({"index_proximal": 0.4})

        demo = _synthetic_rescue()
        skill = client.fit_dmp(
            SKU, StallSignature(*JAM_SIG_1), demo["t"], demo["wrist"], demo["fingers"], demo["tau"]
        )
        assert client.classify_stall(SKU, JAM_SIG_1)[0].is_ood
        assert client.register_skill(skill) == 1
        match, skill_id = client.classify_stall(SKU, (0.0352, 2.3, 3.1, 1.4))
        assert not match.is_ood and skill_id == skill.skill_id
        assert client.classify_stall(SKU, PINNED_BOLT)[0].is_ood
        with pytest.raises(UnoQError):
            client.request({"op": "no_such_op"})

    with UnoQServer(host="127.0.0.1", udp_port=0, tcp_port=0, name="board") as server:
        server.set_console_state(ConsoleInputState(selected_bay=2, clutch=True))
        with UnoQClient.connect_or_mock("127.0.0.1", server.udp_port, server.tcp_port) as live:
            assert not live.is_mock and live.ping() == "board"
            assert live.console_state() == ConsoleInputState(selected_bay=2, clutch=True)
