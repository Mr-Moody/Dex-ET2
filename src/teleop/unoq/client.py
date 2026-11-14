"""Laptop-side UNO Q client with automatic in-process loopback fallback (spec 5.1).

``UnoQClient.connect_or_mock(host)`` tries the board for 150 ms. If nothing answers it starts an
``UnoQServer`` thread on 127.0.0.1 (ephemeral ports), logs ``"unoq: using local loopback
node"`` and talks to that instead, so the demo runs identically with the board unplugged.
"""

from __future__ import annotations

import contextlib
import itertools
import logging
import socket
import threading
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from teleop.atc.types import BayState, ConsoleInputState, SkillPrimitive, StallMatch, StallSignature
from teleop.types import Pose
from teleop.unoq import protocol as P
from teleop.unoq.dmp import pose_to_vec7
from teleop.unoq.server import UnoQServer, console_from_dict

logger = logging.getLogger(__name__)

CONNECT_TIMEOUT_S = 0.15
REQUEST_TIMEOUT_S = 2.0
UDP_TIMEOUT_S = 0.05


class UnoQError(RuntimeError):
    """The node answered ``{"ok": false}`` or the connection failed."""


class UnoQTransportError(UnoQError):
    """The connection failed (not connected, socket error, closed): the node may be gone."""


def _tolist(a: ArrayLike) -> list[Any]:
    return np.asarray(a, dtype=np.float64).tolist()


class UnoQClient:
    def __init__(
        self,
        host: str,
        udp_port: int = P.DEFAULT_UDP_PORT,
        tcp_port: int = P.DEFAULT_TCP_PORT,
        timeout_s: float = REQUEST_TIMEOUT_S,
        udp_timeout_s: float = UDP_TIMEOUT_S,
        *,
        local_server: UnoQServer | None = None,
    ) -> None:
        self.host = host
        self.udp_port = udp_port
        self.tcp_port = tcp_port
        self.timeout_s = timeout_s
        self.udp_timeout_s = udp_timeout_s
        self._server = local_server
        self._tcp: socket.socket | None = None
        self._reader: Any = None
        self._udp: socket.socket | None = None
        self._lock = threading.Lock()
        self._udp_ids = itertools.count(1)

    # ------------------------------------------------------------------ connection

    @property
    def is_mock(self) -> bool:
        """True when talking to the in-process loopback node instead of a board."""
        return self._server is not None

    def connect(self, timeout_s: float | None = None) -> UnoQClient:
        t = self.timeout_s if timeout_s is None else timeout_s
        sock = socket.create_connection((self.host, self.tcp_port), timeout=t)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._tcp = sock
        self._reader = sock.makefile("rb")
        udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp.settimeout(self.udp_timeout_s)
        self._udp = udp
        return self

    @classmethod
    def connect_or_mock(
        cls,
        host: str,
        udp_port: int = P.DEFAULT_UDP_PORT,
        tcp_port: int = P.DEFAULT_TCP_PORT,
        connect_timeout_s: float = CONNECT_TIMEOUT_S,
    ) -> UnoQClient:
        """Connect to a live UNO Q (ping within ``connect_timeout_s``) or start a local node."""
        client = cls(host, udp_port, tcp_port)
        try:
            client.connect(connect_timeout_s)
            client.request({"op": P.OP_PING}, timeout_s=connect_timeout_s)
            client.settimeout(client.timeout_s)
            return client
        except (OSError, UnoQError) as exc:
            client.close()
            logger.debug("unoq: %s:%d unreachable (%s)", host, tcp_port, exc)
        server = UnoQServer(host="127.0.0.1", udp_port=0, tcp_port=0, name="unoq-loopback")
        server.start()
        logger.warning("unoq: using local loopback node")
        local = cls("127.0.0.1", server.udp_port, server.tcp_port, local_server=server)
        try:
            return local.connect()
        except OSError:
            server.stop()
            raise

    def settimeout(self, timeout_s: float) -> None:
        if self._tcp is not None:
            self._tcp.settimeout(timeout_s)

    def close(self) -> None:
        for closer in (self._reader, self._tcp, self._udp):
            if closer is not None:
                with contextlib.suppress(OSError):
                    closer.close()
        self._reader = self._tcp = self._udp = None
        if self._server is not None:
            self._server.stop()
            self._server = None

    def __enter__(self) -> UnoQClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ transport

    def request(self, message: Mapping[str, Any], timeout_s: float | None = None) -> dict[str, Any]:
        """One TCP request/response round trip. Raises ``UnoQError`` on ``ok: false``."""
        if self._tcp is None or self._reader is None:
            raise UnoQTransportError("not connected")
        with self._lock:
            try:
                if timeout_s is not None:
                    self._tcp.settimeout(timeout_s)
                self._tcp.sendall(P.encode(dict(message)))
                line = self._reader.readline()
            except OSError as exc:
                raise UnoQTransportError(f"{message.get('op')}: {exc}") from exc
            finally:
                if timeout_s is not None:
                    self._tcp.settimeout(self.timeout_s)
        if not line:
            raise UnoQTransportError(f"{message.get('op')}: connection closed")
        reply = P.decode(line)
        if not reply.get("ok"):
            raise UnoQError(str(reply.get("error", "request failed")))
        return reply

    # ------------------------------------------------------------------ ops

    def ping(self) -> str:
        return str(self.request({"op": P.OP_PING})["node"])

    def filter_step(
        self,
        wrist: Pose | None,
        joints: Mapping[str, float] | None = None,
        t_ns: int | None = None,
    ) -> tuple[Pose | None, dict[str, float]]:
        """Smooth one glove sample over UDP. On a lost or late datagram, returns the input."""
        raw_joints = {str(k): float(v) for k, v in (joints or {}).items()}
        if self._udp is None:
            raise UnoQTransportError("not connected")
        rid = next(self._udp_ids)
        msg: dict[str, Any] = {
            "op": P.OP_FILTER_STEP,
            "id": rid,
            "t_ns": time.monotonic_ns() if t_ns is None else t_ns,
            "joints": raw_joints,
        }
        if wrist is not None:
            msg["wrist"] = pose_to_vec7(wrist)
        try:
            self._udp.sendto(P.encode(msg), (self.host, self.udp_port))
            while True:
                data = self._udp.recv(P.MAX_DATAGRAM)
                reply = P.decode(data)
                if reply.get("id") == rid:
                    break  # drop stale replies to earlier, timed-out requests
        except (OSError, P.ProtocolError):
            return wrist, raw_joints
        if not reply.get("ok"):
            return wrist, raw_joints
        out_wrist = wrist
        w = reply.get("wrist")
        if w is not None:
            v = [float(x) for x in w]
            out_wrist = Pose(orientation=(v[3], v[4], v[5], v[6]), position=(v[0], v[1], v[2]))
        out_joints = {str(k): float(v) for k, v in dict(reply.get("joints") or {}).items()}
        return out_wrist, out_joints

    def classify_stall(
        self, part_sku_id: str, z: StallSignature | Sequence[float]
    ) -> tuple[StallMatch, str | None]:
        """Nearest stall prototype for this SKU, plus the deployed skill for it (if any)."""
        vec = z.as_vector() if isinstance(z, StallSignature) else tuple(float(v) for v in z)
        reply = self.request({"op": P.OP_CLASSIFY_STALL, "part_sku_id": part_sku_id, "z": vec})
        distance = reply.get("distance")
        match = StallMatch(
            signature_id=reply.get("matched_signature_id"),
            distance=float("inf") if distance is None else float(distance),
            is_ood=bool(reply["is_ood"]),
        )
        skill_id = reply.get("skill_id")
        return match, None if skill_id is None else str(skill_id)

    def fit_dmp(
        self,
        part_sku_id: str,
        z: StallSignature | Sequence[float],
        timestamps_s: ArrayLike,
        wrist_latch_se3: ArrayLike,
        fingers_q: ArrayLike,
        tau_ext_inf: ArrayLike,
        signature_id: str | None = None,
    ) -> SkillPrimitive:
        vec = z.as_vector() if isinstance(z, StallSignature) else tuple(float(v) for v in z)
        msg: dict[str, Any] = {
            "op": P.OP_FIT_DMP,
            "part_sku_id": part_sku_id,
            "z": list(vec),
            "timestamps_s": _tolist(timestamps_s),
            "wrist_latch_se3": _tolist(wrist_latch_se3),
            "fingers_q": _tolist(fingers_q),
            "tau_ext_inf": _tolist(tau_ext_inf),
        }
        if signature_id is not None:
            msg["signature_id"] = signature_id
        return SkillPrimitive.from_dict(self.request(msg)["skill"])

    def register_skill(self, skill: SkillPrimitive) -> int:
        """Deploy a validated skill; returns how many rescues back its stall prototype."""
        reply = self.request({"op": P.OP_REGISTER_SKILL, "skill": skill.to_dict()})
        return int(reply["signature_samples"])

    def console_state(self) -> ConsoleInputState:
        return self.console_status()[0]

    def console_status(self) -> tuple[ConsoleInputState, int | None]:
        """Console state plus the MCU heartbeat age in ms (None: no MCU attached or heard)."""
        reply = self.request({"op": P.OP_CONSOLE_STATE})
        age = reply.get("mcu_heartbeat_age_ms")
        return console_from_dict(reply["console"]), None if age is None else int(age)

    def set_bay_state(self, bay_id: int, state: BayState) -> None:
        """Set one bay's LED state on the board's MCU console (no-op without a console)."""
        self.request({"op": P.OP_SET_BAY_STATE, "bay_id": bay_id, "state": state.name})
