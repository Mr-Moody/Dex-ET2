"""T-130: UNO Q edge node: 1-Euro filters, Mahalanobis stall gate, RBF-DMP and loopback service."""

from __future__ import annotations

import ast
import logging
import math
import socket
import time
from pathlib import Path
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
UNOQ_SRC = Path(__file__).resolve().parents[2] / "src" / "teleop" / "unoq"

# Stall vectors z = (x_bolt, mean |tau|_inf, max |tau|_inf, |d tau_wrist|_2), spec 4.2.
JAM_SIG_1 = (0.035, 2.1, 3.4, 1.2)
PINNED_BOLT = (0.0, 2.3, 3.6, 1.1)


def _rot_deg(a: Pose, b: Pose) -> float:
    r = quat_to_mat(a.orientation).T @ quat_to_mat(b.orientation)
    return math.degrees(float(np.linalg.norm(so3_log(r))))


def _pos_err(a: Pose, b: Pose) -> float:
    assert a.position is not None and b.position is not None
    return float(np.linalg.norm(np.subtract(a.position, b.position)))


def _vec7_pose(v: Any) -> Pose:
    """``[x, y, z, qx, qy, qz, qw]`` -> Pose."""
    f = [float(x) for x in v]
    return Pose(orientation=(f[3], f[4], f[5], f[6]), position=(f[0], f[1], f[2]))


def _yaw_pose(x: float, y: float, z: float, yaw_deg: float) -> Pose:
    h = math.radians(yaw_deg) / 2
    return Pose(orientation=(0.0, 0.0, math.sin(h), math.cos(h)), position=(x, y, z))


def synthetic_rescue(duration_s: float = 4.0, hz: float = 60.0) -> dict[str, Any]:
    """A smooth thumb-hold + index-slide rescue in the latch frame (wrist, hand joints, tau)."""
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


# --------------------------------------------------------------------------- package rules


def test_unoq_package_imports_no_mujoco_or_torch() -> None:
    for path in UNOQ_SRC.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                assert name.split(".")[0] not in {"mujoco", "torch"}, f"{path.name}: {name}"


# --------------------------------------------------------------------------- 1-Euro


def test_one_euro_filter_reduces_jitter_with_low_step_lag() -> None:
    rng = np.random.default_rng(7)

    # Still hand with 2 mm sensor jitter at 83.3 Hz.
    f = OneEuroFilter()
    raw = 0.30 + rng.normal(0.0, 0.002, int(4 * HZ))
    out = np.array([float(f(x)) for x in raw])
    settle = int(HZ)
    assert np.std(raw[settle:]) / np.std(out[settle:]) > 3.0

    # A 5 cm step is tracked to 90 % within 100 ms (a fixed 1 Hz low-pass needs ~430 ms).
    f = OneEuroFilter()
    for _ in range(80):
        f(rng.normal(0.0, 0.002))
    ys = [float(f(0.05 + rng.normal(0.0, 0.002))) for _ in range(40)]
    first = next(i for i, y in enumerate(ys) if y >= 0.045)
    assert (first + 1) / HZ < 0.100

    # A 0.5 m/s sweep lags by under 1 cm once the filter has caught up.
    f = OneEuroFilter()
    ramp = 0.5 * np.arange(160) / HZ
    lag = np.abs(np.array([float(f(x)) for x in ramp]) - ramp)
    assert lag[40:].max() < 0.010

    # Timestamps override the nominal period; vectors filter element-wise.
    f = OneEuroFilter()
    v = f(np.array([1.0, 2.0, 3.0]), t_s=0.0)
    v = f(np.array([1.0, 2.0, 3.0]), t_s=0.012)
    assert v.shape == (3,)
    np.testing.assert_allclose(v, [1.0, 2.0, 3.0])


def test_pose_one_euro_filter_keeps_quaternion_hemisphere() -> None:
    f = PoseOneEuroFilter()
    q = (0.0, 0.0, math.sin(0.3), math.cos(0.3))
    neg = (-q[0], -q[1], -q[2], -q[3])
    out = [f(Pose(orientation=o, position=(0.1, 0.0, 0.2))) for o in (q, neg, q, neg)]
    for p in out:
        assert abs(float(np.linalg.norm(p.orientation)) - 1.0) < 1e-9
        assert _rot_deg(p, Pose(orientation=q)) < 1e-6  # sign flips are not rotations


# --------------------------------------------------------------------------- RBF-DMP


def test_rbf_dmp_fit_completes_under_20ms_and_reconstructs_trajectory() -> None:
    demo = synthetic_rescue()
    skill = fit_rbf_dmp(demo["t"], demo["wrist"], demo["fingers"], demo["tau"])
    times: list[float] = []
    for _ in range(5):
        start = time.perf_counter()
        skill = fit_rbf_dmp(demo["t"], demo["wrist"], demo["fingers"], demo["tau"])
        times.append(time.perf_counter() - start)
    assert min(times) < 0.020
    assert skill.fit_ms < 20.0
    assert skill.n_kernels == 30
    assert skill.n_dims == 6 + len(HAND_ACTUATORS)
    assert skill.actuator_names == HAND_ACTUATORS

    duration = float(demo["t"][-1])
    for k in range(0, len(demo["t"]), 5):
        s = float(phase(demo["t"][k], duration))
        wrist, fingers, _ = evaluate_latch_frame(skill, s)
        truth = _vec7_pose(demo["wrist"][k])
        assert _pos_err(wrist, truth) < 0.002
        assert _rot_deg(wrist, truth) < 0.5
        q = np.array([fingers[n] for n in HAND_ACTUATORS])
        assert np.max(np.abs(q - demo["fingers"][k])) < 0.03

    # Nominal torque profile follows the demo's contact window.
    _, _, tau_contact = evaluate_latch_frame(skill, float(phase(0.55 * duration, duration)))
    _, _, tau_free = evaluate_latch_frame(skill, 1.0)
    assert tau_contact == pytest.approx(3.0)
    assert tau_free == pytest.approx(0.2)

    # The skill survives the wire format.
    assert SkillPrimitive.from_dict(skill.to_dict()) == skill


def test_rbf_dmp_se3_warp_rotates_35deg_and_shifts_12cm_accurately() -> None:
    # Record in Bay 1's latch frame (from world poses), replay in Bay 2's.
    bay1 = _yaw_pose(0.05, 0.0, 0.02, 0.0)
    bay2 = _yaw_pose(0.45 + 0.58, 0.09, 0.02, 35.0)  # +8 cm, +9 cm (~12 cm) and +35 deg
    demo = synthetic_rescue()
    world_bay1 = [compose_pose(bay1, _vec7_pose(w)) for w in demo["wrist"]]
    local = np.array([pose_to_vec7(relative_pose(bay1, p)) for p in world_bay1])
    skill = fit_rbf_dmp(demo["t"], local, demo["fingers"], demo["tau"])

    duration = float(demo["t"][-1])
    for k in range(0, len(demo["t"]), 7):
        s = float(phase(demo["t"][k], duration))
        got1, _, _ = evaluate_rbf_dmp(skill, s, bay1)
        got2, fingers2, _ = evaluate_rbf_dmp(skill, s, bay2)
        expect2 = compose_pose(bay2, relative_pose(bay1, world_bay1[k]))
        assert _pos_err(got1, world_bay1[k]) < 0.002
        assert _pos_err(got2, expect2) < 0.002
        assert _rot_deg(got2, expect2) < 0.5
        # Same motion relative to each latch: the warp is exactly the bay-to-bay transform.
        assert _pos_err(relative_pose(bay2, got2), relative_pose(bay1, got1)) < 1e-9
        assert _rot_deg(relative_pose(bay2, got2), relative_pose(bay1, got1)) < 1e-6
        np.testing.assert_allclose(
            [fingers2[n] for n in HAND_ACTUATORS], demo["fingers"][k], atol=0.03
        )

    # The goal wrist heading turns by the latch yaw.
    g1, _, _ = evaluate_rbf_dmp(skill, 0.0, bay1)
    g2, _, _ = evaluate_rbf_dmp(skill, 0.0, bay2)
    assert _rot_deg(g1, g2) == pytest.approx(35.0, abs=0.1)


# --------------------------------------------------------------------------- classifier


def test_mahalanobis_matches_flap_jam_at_35mm_and_rejects_pinned_bolt_at_0mm_as_ood() -> None:
    clf = MahalanobisStallClassifier()
    assert clf.classify(SKU, JAM_SIG_1).is_ood  # nothing learned yet for this SKU

    clf.register_prototype(SKU, "Jam_Sig_1", np.array([JAM_SIG_1]))  # one 1x4 rescue
    bay2_stall = (0.0352, 2.3, 3.1, 1.4)  # same jam, perturbed bay
    match = clf.classify(SKU, bay2_stall)
    assert not match.is_ood
    assert match.signature_id == "Jam_Sig_1"
    assert match.distance < 1.5

    pinned = clf.classify(SKU, PINNED_BOLT)
    assert pinned.is_ood
    assert pinned.signature_id is None
    assert pinned.distance >= 7.0

    assert clf.classify("other_sku", JAM_SIG_1).is_ood  # prototypes are per SKU

    # More rescues refine the covariance without losing the match.
    clf.register_prototype(SKU, "Jam_Sig_1", [(0.034, 2.0, 3.6, 1.0), (0.036, 2.4, 3.2, 1.3)])
    assert clf.sample_count(SKU, "Jam_Sig_1") == 3
    assert clf.classify(SKU, bay2_stall).signature_id == "Jam_Sig_1"
    assert clf.classify(SKU, PINNED_BOLT).is_ood

    with pytest.raises(ValueError):
        clf.register_prototype(SKU, "bad", [1.0, 2.0])


# --------------------------------------------------------------------------- protocol + service


def test_protocol_encodes_lines_and_nulls_non_finite_floats() -> None:
    line = P.encode({"ok": True, "distance": float("inf"), "z": [1.0, float("nan")]})
    assert line.endswith(b"\n") and line.count(b"\n") == 1
    assert P.decode(line) == {"ok": True, "distance": None, "z": [1.0, None]}
    with pytest.raises(P.ProtocolError):
        P.decode_request(b'{"no_op": 1}')
    with pytest.raises(P.ProtocolError):
        P.decode(b"[1, 2]")


def _dead_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def test_unoq_client_roundtrips_over_local_loopback_server(
    caplog: pytest.LogCaptureFixture,
) -> None:
    port = _dead_port()
    start = time.perf_counter()
    with caplog.at_level(logging.WARNING, logger="teleop.unoq.client"):
        client = UnoQClient.connect_or_mock("127.0.0.1", udp_port=port, tcp_port=port)
    assert time.perf_counter() - start < 1.0
    assert "unoq: using local loopback node" in caplog.text
    with client:
        assert client.is_mock
        assert client.ping() == "unoq-loopback"

        # Fast UDP stream: a still wrist comes back unchanged, joints round-trip by name.
        wrist = _yaw_pose(0.1, -0.2, 0.3, 20.0)
        out_wrist, out_joints = client.filter_step(wrist, {"index_proximal": 0.4}, t_ns=0)
        assert out_wrist is not None
        assert _pos_err(out_wrist, wrist) < 1e-9
        assert out_joints == pytest.approx({"index_proximal": 0.4})

        # Rescue -> fit -> (validated) register -> later stalls match it; Negative Control OOD.
        match, skill_id = client.classify_stall(SKU, JAM_SIG_1)
        assert match.is_ood and skill_id is None
        demo = synthetic_rescue()
        skill = client.fit_dmp(
            SKU, StallSignature(*JAM_SIG_1), demo["t"], demo["wrist"], demo["fingers"], demo["tau"]
        )
        assert skill.signature_id == "Jam_Sig_1"
        assert skill.prototype.as_vector() == JAM_SIG_1
        assert skill.fit_ms < 20.0
        assert client.classify_stall(SKU, JAM_SIG_1)[0].is_ood  # not deployed until registered
        assert client.register_skill(skill) == 1
        match, skill_id = client.classify_stall(SKU, (0.0352, 2.3, 3.1, 1.4))
        assert match.signature_id == "Jam_Sig_1" and not match.is_ood
        assert skill_id == skill.skill_id
        match, skill_id = client.classify_stall(SKU, PINNED_BOLT)
        assert match.is_ood and skill_id is None

        assert client.console_state() == ConsoleInputState()
        with pytest.raises(UnoQError):
            client.request({"op": "no_such_op"})
    assert not client.is_mock  # close() stopped the loopback node


def test_connect_or_mock_prefers_a_live_node() -> None:
    with UnoQServer(host="127.0.0.1", udp_port=0, tcp_port=0, name="board") as server:
        server.set_console_state(ConsoleInputState(selected_bay=2, clutch=True))
        with UnoQClient.connect_or_mock("127.0.0.1", server.udp_port, server.tcp_port) as client:
            assert not client.is_mock
            assert client.ping() == "board"
            assert client.console_state() == ConsoleInputState(selected_bay=2, clutch=True)


def test_unoq_command_probes_and_falls_back(capsys: pytest.CaptureFixture[str]) -> None:
    from teleop.commands.unoq import main

    port = _dead_port()
    assert main(["--host", "127.0.0.1", "--udp-port", str(port), "--tcp-port", str(port)]) == 0
    assert "unoq-loopback answered via local loopback" in capsys.readouterr().out


def test_unoq_serve_survives_console_attach_failure(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import teleop.commands.unoq as cmd

    def router_not_ready(*_: object) -> Any:
        raise ConnectionRefusedError("arduino-router not ready")

    monkeypatch.setattr(cmd, "open_console", router_not_ready)
    server = UnoQServer(host="127.0.0.1", udp_port=0, tcp_port=0)
    with caplog.at_level(logging.WARNING, logger="teleop.commands.unoq"):
        attach: Any = cmd._attach_console  # pyright: ignore[reportPrivateUsage]
        assert attach(server) is None
    assert "arduino-router not ready" in caplog.text


def test_skill_ids_are_content_addressed() -> None:
    demo = synthetic_rescue()
    a = fit_rbf_dmp(demo["t"], demo["wrist"], demo["fingers"], demo["tau"])
    b = fit_rbf_dmp(demo["t"], demo["wrist"], demo["fingers"], demo["tau"])
    assert a.skill_id == b.skill_id  # same demonstration, same id (seeds the gate)
    prefix, suffix = a.skill_id.rsplit("/", 1)
    assert prefix == "ev_latch_v1/Jam_Sig_1" and len(suffix) == 8
    other = fit_rbf_dmp(demo["t"], demo["wrist"] + 0.001, demo["fingers"], demo["tau"])
    assert other.skill_id != a.skill_id
    assert (
        fit_rbf_dmp(
            demo["t"], demo["wrist"], demo["fingers"], demo["tau"], skill_id="given"
        ).skill_id
        == "given"
    )


def test_skill_touch_offset_round_trips_and_old_dicts_load() -> None:
    import dataclasses

    demo = synthetic_rescue()
    skill = fit_rbf_dmp(demo["t"], demo["wrist"], demo["fingers"], demo["tau"])
    assert skill.touch_offset_m is None
    with_offset = dataclasses.replace(skill, touch_offset_m=-0.0123)
    assert SkillPrimitive.from_dict(with_offset.to_dict()) == with_offset
    legacy = skill.to_dict()
    del legacy["touch_offset_m"]
    assert SkillPrimitive.from_dict(legacy).touch_offset_m is None
