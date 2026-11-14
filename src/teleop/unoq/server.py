"""UNO Q edge service: UDP ``filter_step`` stream plus TCP request/response ops (spec 5.1).

Runs on the board (``uv run teleop unoq --serve``) or in-process on the laptop as the loopback
fallback. ``handle()`` is the transport-free dispatcher; the socket loops only frame lines.
"""

from __future__ import annotations

import contextlib
import logging
import socket
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any, cast

import numpy as np

from teleop.atc.types import BayState, ConsoleInputState, SkillPrimitive
from teleop.types import Pose
from teleop.unoq import protocol as P
from teleop.unoq.classifier import MahalanobisStallClassifier
from teleop.unoq.dmp import HAND_ACTUATORS, fit_rbf_dmp, pose_to_vec7
from teleop.unoq.filter import JointsOneEuroFilter, PoseOneEuroFilter
from teleop.unoq.mcu_bridge import (
    McuConsoleBridge,
    console_state_from_dict,
    console_state_to_dict,
)

logger = logging.getLogger(__name__)

NODE_NAME = "unoq-qrb2210"
CONSOLE_SYNC_S = 0.05  # MCU bridge -> console_state copy period
POLL_S = 0.2  # how often blocking socket loops check for stop()


def console_to_dict(state: ConsoleInputState) -> dict[str, Any]:
    return console_state_to_dict(state)


def console_from_dict(raw: Mapping[str, Any]) -> ConsoleInputState:
    return console_state_from_dict(raw)


def _vec7_to_pose(raw: Any) -> Pose:
    v = [float(x) for x in raw]
    if len(v) != 7:
        raise ValueError("wrist must be [x, y, z, qx, qy, qz, qw]")
    return Pose(orientation=(v[3], v[4], v[5], v[6]), position=(v[0], v[1], v[2]))


class UnoQServer:
    """Edge node state (filters, stall classifier, skill library) behind UDP and TCP sockets.

    Port 0 binds an ephemeral port; read the real ones from ``udp_port``/``tcp_port`` after
    ``start()``.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        udp_port: int = P.DEFAULT_UDP_PORT,
        tcp_port: int = P.DEFAULT_TCP_PORT,
        classifier: MahalanobisStallClassifier | None = None,
        name: str = NODE_NAME,
    ) -> None:
        self.host = host
        self.name = name
        self._req_udp_port = udp_port
        self._req_tcp_port = tcp_port
        self.classifier = classifier or MahalanobisStallClassifier()
        self.wrist_filter = PoseOneEuroFilter()
        self.joint_filter = JointsOneEuroFilter()
        self.skills: dict[str, SkillPrimitive] = {}
        self._console = ConsoleInputState()
        self._bridge: McuConsoleBridge | None = None  # set once, before serving
        self._last_heard_ns: int | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._conns: set[socket.socket] = set()
        self._udp: socket.socket | None = None
        self._tcp: socket.socket | None = None
        self._handlers: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
            P.OP_PING: self._ping,
            P.OP_FILTER_STEP: self._filter_step,
            P.OP_CLASSIFY_STALL: self._classify_stall,
            P.OP_FIT_DMP: self._fit_dmp,
            P.OP_REGISTER_SKILL: self._register_skill,
            P.OP_CLEAR_SKILLS: self._clear_skills,
            P.OP_CONSOLE_STATE: self._console_state,
            P.OP_SET_BAY_STATE: self._set_bay_state,
            P.OP_JOYSTICK: self._joystick,
        }

    # ------------------------------------------------------------------ lifecycle

    @property
    def udp_port(self) -> int:
        return int(self._udp.getsockname()[1]) if self._udp else self._req_udp_port

    @property
    def tcp_port(self) -> int:
        return int(self._tcp.getsockname()[1]) if self._tcp else self._req_tcp_port

    def start(self) -> UnoQServer:
        udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp.bind((self.host, self._req_udp_port))
        udp.settimeout(POLL_S)
        tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        tcp.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        tcp.bind((self.host, self._req_tcp_port))
        tcp.listen()
        tcp.settimeout(POLL_S)
        self._udp, self._tcp = udp, tcp
        self._stop.clear()
        for target, label in ((self._udp_loop, "udp"), (self._accept_loop, "tcp")):
            th = threading.Thread(target=target, name=f"unoq-{label}", daemon=True)
            th.start()
            self._threads.append(th)
        logger.info("unoq: serving on udp %d / tcp %d", self.udp_port, self.tcp_port)
        return self

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            conns = list(self._conns)
        for conn in conns:
            with contextlib.suppress(OSError):
                conn.shutdown(socket.SHUT_RDWR)
            conn.close()
        for th in self._threads:
            th.join(timeout=2.0)
        self._threads.clear()
        for sock in (self._udp, self._tcp):
            if sock is not None:
                sock.close()
        self._udp = self._tcp = None

    def __enter__(self) -> UnoQServer:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def set_console_state(self, state: ConsoleInputState) -> None:
        """Called by the MCU serial bridge whenever the operator console changes."""
        with self._lock:
            self._console = state

    def attach_console(self, bridge: McuConsoleBridge) -> None:
        """Mirror the MCU console into ``console_state`` and forward ``set_bay_state`` to it.

        The caller owns the bridge's reader (``bridge.start()``); this only copies its state
        every 50 ms on a daemon thread.
        """
        self._bridge = bridge
        th = threading.Thread(target=self._console_sync, args=(bridge,), name="unoq-console")
        th.daemon = True
        th.start()
        self._threads.append(th)

    def _console_sync(self, bridge: McuConsoleBridge) -> None:
        while not self._stop.is_set():
            heard = bridge.last_heard_ns
            state = bridge.state()
            with self._lock:
                self._console = state
                self._last_heard_ns = heard
            self._stop.wait(CONSOLE_SYNC_S)

    # ------------------------------------------------------------------ dispatch

    def handle(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Run one request; never raises (failures come back as ``{"ok": false, ...}``)."""
        op = request.get("op")
        handler = self._handlers.get(op) if isinstance(op, str) else None
        if handler is None:
            return P.error(f"unknown op {op!r}")
        try:
            if op in (P.OP_SET_BAY_STATE, P.OP_JOYSTICK):
                return handler(dict(request))  # MCU write may block: never under the lock
            with self._lock:
                return handler(dict(request))
        except (KeyError, TypeError, ValueError, OSError) as exc:
            return P.error(f"{op}: {type(exc).__name__}: {exc}")

    def _ping(self, _: dict[str, Any]) -> dict[str, Any]:
        return P.ok(node=self.name)

    def _filter_step(self, req: dict[str, Any]) -> dict[str, Any]:
        t_ns = req.get("t_ns")
        t_s = None if t_ns is None else int(t_ns) * 1e-9
        reply = P.ok()
        if req.get("wrist") is not None:
            pose = self.wrist_filter.filter(_vec7_to_pose(req["wrist"]), t_s)
            reply["wrist"] = pose_to_vec7(pose)
        joints = cast(Mapping[Any, Any], req.get("joints") or {})
        reply["joints"] = self.joint_filter.filter(
            {str(k): float(v) for k, v in joints.items()}, t_s
        )
        return reply

    def _classify_stall(self, req: dict[str, Any]) -> dict[str, Any]:
        sku = str(req["part_sku_id"])
        match = self.classifier.classify(sku, req["z"])
        skill_id = None
        if match.signature_id is not None:
            for skill in self.skills.values():
                if skill.part_sku_id == sku and skill.signature_id == match.signature_id:
                    skill_id = skill.skill_id  # latest registered wins
        return P.ok(
            matched_signature_id=match.signature_id,
            distance=match.distance,
            is_ood=match.is_ood,
            skill_id=skill_id,
        )

    def _fit_dmp(self, req: dict[str, Any]) -> dict[str, Any]:
        sku = str(req["part_sku_id"])
        signature_id = req.get("signature_id")
        if signature_id is None:
            signature_id = f"Jam_Sig_{len(self.classifier.signatures(sku)) + 1}"
        names = req.get("actuator_names") or HAND_ACTUATORS
        skill = fit_rbf_dmp(
            req["timestamps_s"],
            req["wrist_latch_se3"],
            req["fingers_q"],
            req["tau_ext_inf"],
            n_kernels=int(req.get("n_kernels", 30)),
            part_sku_id=sku,
            signature_id=str(signature_id),
            prototype=[float(v) for v in req["z"]],
            actuator_names=[str(n) for n in names],
        )
        return P.ok(fit_ms=skill.fit_ms, skill=skill.to_dict())

    def _register_skill(self, req: dict[str, Any]) -> dict[str, Any]:
        """Add a gate-validated skill to the library; its stall prototype becomes matchable.

        Idempotent per ``skill_id``: a replayed skill (laptop swap back to this node) adds no
        duplicate prototype sample.
        """
        skill = SkillPrimitive.from_dict(req["skill"])
        if skill.skill_id in self.skills:
            n = self.classifier.sample_count(skill.part_sku_id, skill.signature_id)
            return P.ok(skill_id=skill.skill_id, signature_samples=n)
        self.skills[skill.skill_id] = skill
        z = np.asarray(skill.prototype.as_vector(), dtype=np.float64)
        self.classifier.register_prototype(skill.part_sku_id, skill.signature_id, z)
        n = self.classifier.sample_count(skill.part_sku_id, skill.signature_id)
        return P.ok(skill_id=skill.skill_id, signature_samples=n)

    def _clear_skills(self, req: dict[str, Any]) -> dict[str, Any]:
        """Forget one SKU's skills and stall prototypes (signature numbering restarts)."""
        sku = str(req["part_sku_id"])
        stale = [sid for sid, s in self.skills.items() if s.part_sku_id == sku]
        for sid in stale:
            del self.skills[sid]
        self.classifier.clear(sku)
        return P.ok(removed=len(stale))

    def _console_state(self, _: dict[str, Any]) -> dict[str, Any]:
        heard = self._last_heard_ns
        age = None if heard is None else max(0, (time.monotonic_ns() - heard) // 1_000_000)
        return P.ok(console=console_to_dict(self._console), mcu_heartbeat_age_ms=age)

    def _set_bay_state(self, req: dict[str, Any]) -> dict[str, Any]:
        bay_id = int(req["bay_id"])
        state = BayState[str(req["state"])]  # KeyError -> error reply
        if self._bridge is not None:
            self._bridge.set_bay_state(bay_id, state)
        return P.ok()

    def _joystick(self, req: dict[str, Any]) -> dict[str, Any]:
        """Joystick map and calibration on the attached MCU console."""
        bridge = self._bridge
        if bridge is None:
            return P.error("joystick: no MCU console attached")
        action = str(req.get("action", "status"))
        if action == "calibrate":
            bridge.start_joystick_calibration()
        elif action == "abort":
            bridge.abort_joystick_calibration()
        elif action == "set":
            bridge.set_joy_map(str(req["map"]))  # ValueError -> error reply
        elif action == "raw":
            bridge.stream_joystick_raw(bool(req.get("on", True)))
        elif action != "status":
            return P.error(f"joystick: unknown action {action!r}")
        return P.ok(**bridge.joystick_status())

    # ------------------------------------------------------------------ sockets

    def _udp_loop(self) -> None:
        sock = self._udp
        assert sock is not None
        while not self._stop.is_set():
            try:
                data, addr = sock.recvfrom(P.MAX_DATAGRAM)
            except TimeoutError:
                continue
            except OSError:
                break
            try:
                req = P.decode_request(data)
                if req["op"] not in P.UDP_OPS:
                    reply = P.error(f"op {req['op']!r} is TCP-only")
                else:
                    reply = self.handle(req)
                    if "id" in req:
                        reply["id"] = req["id"]
            except P.ProtocolError as exc:
                reply = P.error(str(exc))
            try:
                sock.sendto(P.encode(reply), addr)
            except OSError as exc:
                logger.warning("unoq: udp reply to %s failed: %s", addr, exc)

    def _accept_loop(self) -> None:
        sock = self._tcp
        assert sock is not None
        while not self._stop.is_set():
            try:
                conn, _ = sock.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            conn.settimeout(None)
            with self._lock:
                self._conns.add(conn)
            th = threading.Thread(target=self._serve_conn, args=(conn,), daemon=True)
            th.start()

    def _serve_conn(self, conn: socket.socket) -> None:
        try:
            with conn.makefile("rb") as reader:
                for line in reader:
                    if not line.strip():
                        continue
                    try:
                        reply = self.handle(P.decode_request(line))
                    except P.ProtocolError as exc:
                        reply = P.error(str(exc))
                    conn.sendall(P.encode(reply))
        except OSError:
            pass
        finally:
            with self._lock:
                self._conns.discard(conn)
            conn.close()
