"""Temporary dummy Rokoko glove streamer and operator co-pilot.

TODO(rokoko-gloves-swap): TEMPORARY MODULE while physical Rokoko Smartgloves are
disconnected. See ``docs/DUMMY_GLOVE_SWAP.md`` for the swap-out checklist.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import random
import socket
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import numpy as np

from teleop.atc.coordinator import ROUTINE, AtcCoordinator, OperatorInput
from teleop.atc.stage import CONTROL_DT, RESCUE_PRESS, RESCUE_WAYPOINTS, StageController
from teleop.atc.types import BOLT_CLEARED_M
from teleop.input.studio import quat_multiply, quat_normalize
from teleop.retarget.calibration import DEFAULT_CLOSED_PER_JOINT
from teleop.types import Pose, Quat, Vec3
from teleop.unoq.dmp import HAND_ACTUATORS, quat_to_mat

logger = logging.getLogger("teleop.dummy_glove")

DEFAULT_CONTROL_PORT = 14044
DEFAULT_STREAM_HZ = 60.0
DummyMode = Literal["manual", "auto", "off"]

INSPIRE_CTRL_RANGES: dict[str, tuple[float, float]] = {
    "pinky": (0.0, 1.57),
    "ring": (0.0, 1.57),
    "middle": (0.0, 1.50),
    "index": (0.0, 1.50),
    "thumb_proximal": (0.10, 0.57),
    "thumb_yaw": (0.0, 1.308),
}

_IDENTITY_Q: Quat = (0.0, 0.0, 0.0, 1.0)
_STUDIO_CHAINS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Thumb", ("Proximal", "Medial", "Distal", "Tip")),
    ("Index", ("Proximal", "Medial", "Distal", "Tip")),
    ("Middle", ("Proximal", "Medial", "Distal", "Tip")),
    ("Ring", ("Proximal", "Medial", "Distal", "Tip")),
    ("Little", ("Proximal", "Medial", "Distal", "Tip")),
)


def _clamp(value: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, float(value)))


def _rx_quat(angle_rad: float) -> Quat:
    h = 0.5 * angle_rad
    return (math.sin(h), 0.0, 0.0, math.cos(h))


def zup_to_yup_vec(v: Vec3) -> dict[str, float]:
    return {"x": float(v[0]), "y": float(v[2]), "z": float(-v[1])}


def zup_to_yup_quat(q: Quat) -> dict[str, float]:
    qn = quat_normalize(q)
    return {"x": float(qn[0]), "y": float(qn[2]), "z": float(-qn[1]), "w": float(qn[3])}


def _quat_dict(q: Quat) -> dict[str, float]:
    qn = quat_normalize(q)
    return {"x": float(qn[0]), "y": float(qn[1]), "z": float(qn[2]), "w": float(qn[3])}


def actuator_targets_to_fractions(targets: Mapping[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, (lo, hi) in INSPIRE_CTRL_RANGES.items():
        rad = float(targets.get(name, ROUTINE.fingers.get(name, lo)))
        span = hi - lo
        out[name] = _clamp((rad - lo) / span, 0.0, 1.0) if span > 1e-6 else 0.0
    return out


def fractions_to_actuator_targets(fractions: Mapping[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, (lo, hi) in INSPIRE_CTRL_RANGES.items():
        frac = _clamp(float(fractions.get(name, 0.0)), 0.0, 1.0)
        out[name] = lo + frac * (hi - lo)
    return out


def routine_fractions() -> dict[str, float]:
    return actuator_targets_to_fractions(ROUTINE.fingers)


def build_studio_v3_packet(
    timestamp_s: float,
    wrist_pos_zup: Vec3 = (0.0, 0.0, 0.0),
    wrist_quat_zup: Quat = _IDENTITY_Q,
    finger_fractions: Mapping[str, float] | None = None,
    *,
    actor_name: str = "dummy_operator",
) -> bytes:
    """Build a valid Rokoko Studio JSON v3 UDP datagram matching StudioParser."""
    fracs = dict(routine_fractions())
    if finger_fractions is not None:
        for k, v in finger_fractions.items():
            if k in fracs:
                fracs[k] = _clamp(float(v), 0.0, 1.0)

    hq_d = zup_to_yup_quat(wrist_quat_zup)
    hand_q: Quat = (hq_d["x"], hq_d["y"], hq_d["z"], hq_d["w"])
    zero = {"x": 0.0, "y": 0.0, "z": 0.0}
    body: dict[str, Any] = {
        "rightLowerArm": {"position": zero, "rotation": _quat_dict(hand_q)},
        "rightHand": {"position": zup_to_yup_vec(wrist_pos_zup), "rotation": _quat_dict(hand_q)},
    }
    c = DEFAULT_CLOSED_PER_JOINT
    angles: dict[tuple[str, str], float] = {
        ("Thumb", "Proximal"): c * fracs["thumb_yaw"],
        ("Thumb", "Medial"): c * fracs["thumb_proximal"],
        ("Thumb", "Distal"): c * fracs["thumb_proximal"],
        ("Index", "Proximal"): c * fracs["index"],
        ("Index", "Medial"): c * fracs["index"],
        ("Middle", "Proximal"): c * fracs["middle"],
        ("Middle", "Medial"): c * fracs["middle"],
        ("Ring", "Proximal"): c * fracs["ring"],
        ("Ring", "Medial"): c * fracs["ring"],
        ("Little", "Proximal"): c * fracs["pinky"],
        ("Little", "Medial"): c * fracs["pinky"],
    }
    for fname, suffixes in _STUDIO_CHAINS:
        pq = hand_q
        for sfx in suffixes:
            qw = quat_normalize(quat_multiply(pq, _rx_quat(angles.get((fname, sfx), 0.0))))
            body[f"right{fname}{sfx}"] = {"position": zero, "rotation": _quat_dict(qw)}
            pq = qw

    pkt = {
        "version": "3,0",
        "fps": 30.0,
        "scene": {
            "timestamp": round(float(timestamp_s), 6),
            "actors": [
                {
                    "name": actor_name,
                    "meta": {
                        "hasGloves": True,
                        "hasLeftGlove": False,
                        "hasRightGlove": True,
                        "hasBody": False,
                    },
                    "body": body,
                }
            ],
            "newtons": [],
            "props": [],
            "characters": [],
        },
    }
    return json.dumps(pkt, separators=(",", ":")).encode("utf-8")


@dataclass
class DummyStreamTarget:
    active: bool = True
    wrist_pos_zup: Vec3 = (0.0, 0.0, 0.0)
    wrist_quat_zup: Quat = _IDENTITY_Q
    finger_fractions: dict[str, float] = field(default_factory=routine_fractions)


class DummyGloveUdpServer:
    """Standalone UDP streamer (Studio JSON v3 -> StudioSource) + localhost control socket."""

    def __init__(
        self,
        studio_host: str = "127.0.0.1",
        studio_port: int = 14043,
        control_host: str = "127.0.0.1",
        control_port: int = DEFAULT_CONTROL_PORT,
        hz: float = DEFAULT_STREAM_HZ,
    ) -> None:
        self.studio_addr = (studio_host, int(studio_port))
        self.control_host = control_host
        self._req_control_port = int(control_port)
        self.hz = max(1.0, float(hz))
        self._lock = threading.Lock()
        self._target = DummyStreamTarget()
        self._stop = threading.Event()
        self._ctrl_sock: socket.socket | None = None
        self._tx_sock: socket.socket | None = None
        self._threads: list[threading.Thread] = []
        self.packets_sent = 0

    @property
    def control_port(self) -> int:
        if self._ctrl_sock is None:
            return self._req_control_port
        return int(self._ctrl_sock.getsockname()[1])

    def start(self) -> DummyGloveUdpServer:
        ctrl = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        ctrl.bind((self.control_host, self._req_control_port))
        ctrl.settimeout(0.1)
        tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._ctrl_sock = ctrl
        self._tx_sock = tx
        self._stop.clear()
        for name, fn in (
            ("dummy-glove-ctrl", self._control_loop),
            ("dummy-glove-tx", self._stream_loop),
        ):
            th = threading.Thread(target=fn, name=name, daemon=True)
            th.start()
            self._threads.append(th)
        return self

    def stop(self) -> None:
        self._stop.set()
        for th in self._threads:
            th.join(timeout=2.0)
        self._threads.clear()
        for sock in (self._ctrl_sock, self._tx_sock):
            if sock is not None:
                sock.close()
        self._ctrl_sock = None
        self._tx_sock = None

    def apply_command(self, cmd: Mapping[str, Any]) -> None:
        with self._lock:
            if "active" in cmd:
                self._target.active = bool(cmd["active"])
            pos_raw = cmd.get("wrist_pos")
            if isinstance(pos_raw, list | tuple):
                pos = cast("list[Any] | tuple[Any, ...]", pos_raw)
                if len(pos) == 3:
                    self._target.wrist_pos_zup = (float(pos[0]), float(pos[1]), float(pos[2]))
            quat_raw = cmd.get("wrist_quat")
            if isinstance(quat_raw, list | tuple):
                quat = cast("list[Any] | tuple[Any, ...]", quat_raw)
                if len(quat) == 4:
                    self._target.wrist_quat_zup = (
                        float(quat[0]),
                        float(quat[1]),
                        float(quat[2]),
                        float(quat[3]),
                    )
            fracs_raw = cmd.get("fingers")
            if isinstance(fracs_raw, Mapping):
                fracs = cast("Mapping[Any, Any]", fracs_raw)
                for k, v in fracs.items():
                    ks = str(k)
                    if ks in INSPIRE_CTRL_RANGES:
                        self._target.finger_fractions[ks] = _clamp(float(v), 0.0, 1.0)

    def _control_loop(self) -> None:
        assert self._ctrl_sock is not None
        while not self._stop.is_set():
            try:
                raw, addr = self._ctrl_sock.recvfrom(8192)
            except TimeoutError:
                continue
            except OSError:
                break
            try:
                decoded: Any = json.loads(raw.decode("utf-8"))
            except ValueError:
                continue
            if not isinstance(decoded, dict):
                continue
            msg = cast("dict[str, Any]", decoded)
            if msg.get("op") == "ping":
                reply = json.dumps({"ok": True, "role": "dummy_glove"}).encode("utf-8")
                with contextlib.suppress(OSError):
                    self._ctrl_sock.sendto(reply, addr)
                continue
            self.apply_command(msg)

    def _stream_loop(self) -> None:
        assert self._tx_sock is not None
        period = 1.0 / self.hz
        t0 = time.monotonic()
        while not self._stop.is_set():
            now = time.monotonic()
            with self._lock:
                active = self._target.active
                pos = self._target.wrist_pos_zup
                quat = self._target.wrist_quat_zup
                fracs = dict(self._target.finger_fractions)
            if active:
                payload = build_studio_v3_packet(now - t0, pos, quat, fracs)
                try:
                    self._tx_sock.sendto(payload, self.studio_addr)
                    self.packets_sent += 1
                except OSError:
                    pass
            elapsed = time.monotonic() - now
            if period - elapsed > 0:
                self._stop.wait(period - elapsed)


class DummyControlClient:
    """Non-blocking UDP sender from ``teleop atc`` to ``scripts/dummy_glove_process.py``."""

    def __init__(self, host: str = "127.0.0.1", port: int = DEFAULT_CONTROL_PORT) -> None:
        self.addr = (host, int(port))
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setblocking(False)

    def send(
        self,
        *,
        active: bool,
        wrist_pos: Vec3,
        wrist_quat: Quat,
        fingers: Mapping[str, float],
    ) -> None:
        msg = json.dumps(
            {
                "active": active,
                "wrist_pos": list(wrist_pos),
                "wrist_quat": list(wrist_quat),
                "fingers": dict(fingers),
            },
            separators=(",", ":"),
        ).encode("utf-8")
        with contextlib.suppress(OSError):
            self._sock.sendto(msg, self.addr)

    def close(self) -> None:
        self._sock.close()


def _waypoint_state(elapsed_s: float) -> tuple[Vec3, float]:
    """Interpolate latch-frame (dx, dy, dz) and thumb press along RESCUE_WAYPOINTS."""
    cursor = 0.0
    for (_, p0, f0), (dur, p1, f1) in zip(RESCUE_WAYPOINTS, RESCUE_WAYPOINTS[1:], strict=False):
        seg = max(float(dur), CONTROL_DT)
        if elapsed_s <= cursor + seg:
            a = _clamp((elapsed_s - cursor) / seg, 0.0, 1.0)
            dx = p0[0] + a * (p1[0] - p0[0])
            dy = p0[1] + a * (p1[1] - p0[1])
            dz = p0[2] + a * (p1[2] - p0[2])
            return (dx, dy, dz), f0 + a * (f1 - f0)
        cursor += seg
    last = RESCUE_WAYPOINTS[-1]
    return last[1], last[2]


_TOTAL_SOLVE_S = sum(w[0] for w in RESCUE_WAYPOINTS[1:])


class DummyOperatorSupervisor:
    """Temporary stand-in operator + manual UI override (TODO(rokoko-gloves-swap))."""

    SETTLE_S = 0.7
    CLEAR_MARGIN_M = 0.0015
    AUTO_FAULT_COOLDOWN_S = 6.0

    def __init__(
        self,
        controller: StageController,
        *,
        udp_client: DummyControlClient | None = None,
        process_pid: int | None = None,
        mode: DummyMode = "manual",
        auto_faults: bool = False,
        rng_seed: int | None = None,
    ) -> None:
        self.controller = controller
        self.udp_client = udp_client
        self.process_pid = process_pid
        self._lock = threading.Lock()
        self._mode: DummyMode = mode
        self._auto_faults = auto_faults
        base = routine_fractions()
        self._dx_m = 0.0
        self._dy_m = 0.0
        self._dz_m = 0.0
        self._thumb = 0.0
        self._index = base["index"]
        self._grip = base["middle"]
        self._guided_solve = False
        self._solve_elapsed_s = 0.0
        self._was_clutched = False
        self._auto_hold_ticks = 0
        self._next_fault_sim_t = 3.5
        self._rng = random.Random(rng_seed)

    def reset_manual_pose(self) -> None:
        base = routine_fractions()
        with self._lock:
            self._dx_m = self._dy_m = self._dz_m = 0.0
            self._thumb = 0.0
            self._index = base["index"]
            self._grip = base["middle"]
            self._guided_solve = False
            self._solve_elapsed_s = 0.0

    def apply_ui_message(self, msg: Mapping[str, Any]) -> None:
        action = str(msg.get("action", "set"))
        if action == "reset_pose":
            self.reset_manual_pose()
            return
        with self._lock:
            mode = msg.get("mode")
            if mode in ("manual", "auto", "off"):
                self._mode = mode
                if mode != "manual":
                    self._guided_solve = False
            if "auto_faults" in msg:
                self._auto_faults = bool(msg["auto_faults"])
            if action == "solve":
                self._guided_solve = True
                self._solve_elapsed_s = 0.0
            elif action == "stop_solve":
                self._guided_solve = False
            for key, attr, lo, hi in (
                ("dx_cm", "_dx_m", -0.03, 0.07),
                ("dy_cm", "_dy_m", -0.03, 0.03),
                ("dz_cm", "_dz_m", -0.03, 0.04),
            ):
                if key in msg:
                    setattr(self, attr, _clamp(float(msg[key]) / 100.0, lo, hi))
                    self._guided_solve = False
            nudge_raw = msg.get("nudge")
            if isinstance(nudge_raw, Mapping):
                nudge = cast("Mapping[str, Any]", nudge_raw)
                dx_n = float(nudge.get("dx_cm", 0.0)) / 100.0
                dy_n = float(nudge.get("dy_cm", 0.0)) / 100.0
                dz_n = float(nudge.get("dz_cm", 0.0)) / 100.0
                self._dx_m = _clamp(self._dx_m + dx_n, -0.03, 0.07)
                self._dy_m = _clamp(self._dy_m + dy_n, -0.03, 0.03)
                self._dz_m = _clamp(self._dz_m + dz_n, -0.03, 0.04)
                self._guided_solve = False
            for fkey, fattr in (("thumb", "_thumb"), ("index", "_index"), ("grip", "_grip")):
                if fkey in msg:
                    setattr(self, fattr, _clamp(float(msg[fkey]), 0.0, 1.0))
                    self._guided_solve = False

    def status_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "temporary": True,
                "transport": "studio_udp" if self.udp_client is not None else "in_process",
                "pid": self.process_pid,
                "mode": self._mode,
                "auto_faults": self._auto_faults,
                "solving": self._guided_solve,
                "dx_cm": round(self._dx_m * 100.0, 2),
                "dy_cm": round(self._dy_m * 100.0, 2),
                "dz_cm": round(self._dz_m * 100.0, 2),
                "thumb": round(self._thumb, 3),
                "index": round(self._index, 3),
                "grip": round(self._grip, 3),
            }

    def _finger_fractions(self, thumb_override: float | None = None) -> dict[str, float]:
        base = routine_fractions()
        with self._lock:
            thumb = self._thumb if thumb_override is None else thumb_override
            idx = self._index
            grip = self._grip
        return {
            "index": idx,
            "middle": grip,
            "ring": _clamp(grip * 0.65, 0.0, 1.0),
            "pinky": _clamp(grip * 0.65, 0.0, 1.0),
            "thumb_proximal": _clamp(base["thumb_proximal"] + 0.35 * thumb, 0.0, 1.0),
            "thumb_yaw": thumb,
        }

    def _maybe_inject_auto_fault(self, co: AtcCoordinator) -> None:
        ctl = self.controller
        if co.time_s < self._next_fault_sim_t:
            return
        if ctl.rescue_bay() is not None or co.queue or ctl.armed or ctl.active_faults:
            return
        candidates = list(ctl.active_bays)
        if not candidates:
            return
        bay = self._rng.choice(candidates)
        pin = bool(co.library) and bay == 2 and self._rng.random() < 0.25
        ctl.arm(bay, flap_jammed=not pin, bolt_pinned=pin)
        self._next_fault_sim_t = co.time_s + self.AUTO_FAULT_COOLDOWN_S

    def step(self, downstream: OperatorInput | None = None) -> OperatorInput | None:
        ctl = self.controller
        co = ctl.coordinator
        with self._lock:
            mode = self._mode
            auto_faults = self._auto_faults

        if auto_faults:
            self._maybe_inject_auto_fault(co)

        if mode == "off":
            if self.udp_client is not None:
                self.udp_client.send(
                    active=False,
                    wrist_pos=(0.0, 0.0, 0.0),
                    wrist_quat=_IDENTITY_Q,
                    fingers=routine_fractions(),
                )
            return downstream

        rescue_bay = ctl.rescue_bay()
        selected = ctl.console.selected_bay
        clutched = bool(ctl.console.clutch and rescue_bay is not None and selected == rescue_bay)

        if mode == "auto" and rescue_bay is not None:
            refused = co.state(rescue_bay).value == "OOD_ESCALATED"
            if refused and ctl.rescue_elapsed_s >= self.SETTLE_S + 0.8:
                if ctl.console.clutch:
                    ctl.on_key(32)
                ctl.on_key(ord("R"))
                self._next_fault_sim_t = co.time_s + self.AUTO_FAULT_COOLDOWN_S
            elif not refused and ctl.rescue_elapsed_s >= self.SETTLE_S:
                if selected != rescue_bay:
                    ctl.on_key(48 + rescue_bay)
                elif not ctl.console.clutch and self._auto_hold_ticks == 0:
                    self.reset_manual_pose()
                    with self._lock:
                        self._guided_solve = True
                        self._solve_elapsed_s = 0.0
                    ctl.on_key(32)
                    self._auto_hold_ticks = 1
        if rescue_bay is None:
            self._auto_hold_ticks = 0

        if clutched and not self._was_clutched:
            with self._lock:
                self._dx_m = self._dy_m = self._dz_m = 0.0
                if mode == "auto":
                    self._guided_solve = True
                    self._solve_elapsed_s = 0.0
        elif not clutched and self._was_clutched:
            with self._lock:
                self._dx_m = self._dy_m = self._dz_m = 0.0
                self._guided_solve = False
                self._solve_elapsed_s = 0.0
        self._was_clutched = clutched

        if not clutched or rescue_bay is None:
            fracs = self._finger_fractions(0.0 if mode == "auto" else None)
            if self.udp_client is not None:
                self.udp_client.send(
                    active=True,
                    wrist_pos=(0.0, 0.0, 0.0),
                    wrist_quat=_IDENTITY_Q,
                    fingers=fracs,
                )
            return None if self.udp_client is None else downstream

        rig = co.rig(rescue_bay)
        with self._lock:
            solving = self._guided_solve
        if solving:
            if rig.bolt_x() < BOLT_CLEARED_M + self.CLEAR_MARGIN_M:
                with self._lock:
                    self._solve_elapsed_s += CONTROL_DT
                    elapsed = self._solve_elapsed_s
                (dx, dy, dz), press = _waypoint_state(elapsed)
                thumb_frac = _clamp(
                    (press * RESCUE_PRESS["thumb_yaw"]) / INSPIRE_CTRL_RANGES["thumb_yaw"][1],
                    0.0,
                    1.0,
                )
                with self._lock:
                    self._dx_m, self._dy_m, self._dz_m = dx, dy, dz
                    self._thumb = thumb_frac
                    if elapsed >= _TOTAL_SOLVE_S + 0.5 and mode == "auto":
                        self._guided_solve = False
                        ctl.on_key(32)
            elif mode == "auto":
                with self._lock:
                    self._guided_solve = False
                ctl.on_key(32)

        with self._lock:
            local_delta = np.array([self._dx_m, self._dy_m, self._dz_m], dtype=np.float64)
        world_delta = quat_to_mat(rig.world_from_latch().orientation) @ local_delta
        pos_zup: Vec3 = (float(world_delta[0]), float(world_delta[1]), float(world_delta[2]))
        fracs = self._finger_fractions()

        if self.udp_client is not None:
            self.udp_client.send(
                active=True,
                wrist_pos=pos_zup,
                wrist_quat=_IDENTITY_Q,
                fingers=fracs,
            )
            if downstream is not None:
                return downstream

        targets = fractions_to_actuator_targets(fracs)
        full_targets = {name: targets.get(name, 0.0) for name in HAND_ACTUATORS}
        return OperatorInput(Pose(orientation=_IDENTITY_Q, position=pos_zup), full_targets)

    def close(self) -> None:
        if self.udp_client is not None:
            self.udp_client.close()
