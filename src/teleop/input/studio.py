"""Rokoko Studio Custom Streaming (JSON v3) parser and UDP input source.

Studio sends per-bone *world* rotations; ``HandFrame.joints`` holds *local* rotations
(``conj(parent_world) * child_world``). See ``docs/studio-json-v3.md`` for the format notes.
"""

from __future__ import annotations

import base64
import contextlib
import importlib
import json
import logging
import math
import socket
import threading
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import IO, Any, cast

from teleop.config import StudioConfig
from teleop.input.buffer import ThreadedSource
from teleop.types import HandFrame, Pose, Quat, Side, Vec3

logger = logging.getLogger(__name__)

# lz4 ships no type stubs; load dynamically so strict pyright stays quiet.
_lz4_frame: Any = importlib.import_module("lz4.frame")
_lz4_block: Any = importlib.import_module("lz4.block")

# Studio finger bone suffix -> OpenXR joint part. Finger "Medial" is the OpenXR "intermediate".
# Studio has no finger metacarpal. The thumb is shifted by one: Studio ThumbProximal sits at the
# carpometacarpal joint (it is the first bone below the hand), i.e. OpenXR thumb_metacarpal, so
# Medial -> thumb_proximal, Distal -> thumb_distal and the end site Tip -> thumb_tip.
_FINGER_PARTS: Mapping[str, str] = {
    "Proximal": "proximal",
    "Medial": "intermediate",
    "Distal": "distal",
    "Tip": "tip",
}
_THUMB_PARTS: Mapping[str, str] = {
    "Proximal": "metacarpal",
    "Medial": "proximal",
    "Distal": "distal",
    "Tip": "tip",
}
_STUDIO_FINGERS: Mapping[str, str] = {
    "Thumb": "thumb",
    "Index": "index",
    "Middle": "middle",
    "Ring": "ring",
    "Little": "little",
}
_MAX_DECOMPRESSED = 1 << 22
_CHAIN = ("Proximal", "Medial", "Distal", "Tip")


def _build_joint_map(side: Side) -> tuple[tuple[str, str, str], ...]:
    """(openxr name, studio bone, studio parent bone) for every mapped finger joint."""
    hand = f"{side}Hand"
    rows: list[tuple[str, str, str]] = []
    for studio_finger, xr_finger in _STUDIO_FINGERS.items():
        parts = _THUMB_PARTS if studio_finger == "Thumb" else _FINGER_PARTS
        parent = hand
        for suffix in _CHAIN:
            bone = f"{side}{studio_finger}{suffix}"
            rows.append((f"{xr_finger}_{parts[suffix]}", bone, parent))
            parent = bone
    return tuple(rows)


# Explicit name map; anything not listed here is ignored.
JOINT_MAP: Mapping[Side, tuple[tuple[str, str, str], ...]] = {
    "left": _build_joint_map("left"),
    "right": _build_joint_map("right"),
}


def quat_multiply(a: Quat, b: Quat) -> Quat:
    """Hamilton product a * b for xyzw quaternions."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def quat_conjugate(q: Quat) -> Quat:
    return (-q[0], -q[1], -q[2], q[3])


def quat_normalize(q: Quat) -> Quat:
    n = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    if not math.isfinite(n) or n < 1e-9:
        raise ValueError("degenerate quaternion")
    return (q[0] / n, q[1] / n, q[2] / n, q[3] / n)


def yup_to_zup_vec(v: Vec3) -> Vec3:
    """Studio Y-up -> MuJoCo Z-up: a +90 degree rotation about X, (x, y, z) -> (x, -z, y)."""
    return (v[0], -v[2], v[1])


def yup_to_zup_quat(q: Quat) -> Quat:
    """Same basis change on an orientation (conjugation by the +90 deg X rotation)."""
    return (q[0], -q[2], q[1], q[3])


def _decompress(payload: bytes) -> bytes:
    """Return plain JSON bytes; anything not starting with ``{`` is tried as LZ4."""
    if payload[:1] == b"{":
        return payload
    try:
        return cast("bytes", _lz4_frame.decompress(payload))
    except Exception:
        pass
    # 4-byte LE size prefix (python-lz4 default); cap it so garbage cannot force a huge alloc
    if len(payload) > 4 and int.from_bytes(payload[:4], "little") <= _MAX_DECOMPRESSED:
        try:
            return cast("bytes", _lz4_block.decompress(payload))
        except Exception:
            pass
    for size in (1 << 16, 1 << 18, _MAX_DECOMPRESSED):  # raw block without a size prefix
        try:
            return cast("bytes", _lz4_block.decompress(payload, uncompressed_size=size))
        except Exception:
            continue
    raise ValueError("unrecognised payload framing")


def _as_dict(x: Any) -> dict[str, Any]:
    return cast("dict[str, Any]", x) if isinstance(x, dict) else {}


def _vec(d: Any) -> Vec3 | None:
    try:
        v = (float(d["x"]), float(d["y"]), float(d["z"]))
    except (KeyError, TypeError, ValueError):
        return None
    return v if all(math.isfinite(c) for c in v) else None


def _rot(bone: Any) -> Quat | None:
    try:
        r = bone["rotation"]
        q = (float(r["x"]), float(r["y"]), float(r["z"]), float(r["w"]))
        return quat_normalize(q)
    except (KeyError, TypeError, ValueError):
        return None


class StudioParser:
    """Parses Studio JSON v3 datagrams into per-hand ``HandFrame`` s."""

    def __init__(self, actor: str | None = None) -> None:
        self._actor = actor
        self._seq = 0

    def parse(self, payload: bytes, host_time_ns: int) -> list[HandFrame]:
        try:
            return self._parse(payload, host_time_ns)
        except Exception:
            logger.debug("dropping unparseable Studio payload", exc_info=True)
            return []

    def _select_actor(self, actors: Any) -> dict[str, Any] | None:
        if not isinstance(actors, list):
            return None
        for item in cast("list[Any]", actors):
            if not isinstance(item, dict):
                continue
            actor = cast("dict[str, Any]", item)
            if self._actor is not None:
                if actor.get("name") == self._actor:
                    return actor
            elif _as_dict(actor.get("meta")).get("hasGloves"):
                return actor
        return None

    def _parse(self, payload: bytes, host_time_ns: int) -> list[HandFrame]:
        data = _as_dict(json.loads(_decompress(payload)))
        scene = _as_dict(data.get("scene"))
        actor = self._select_actor(scene.get("actors"))
        if actor is None or not isinstance(actor.get("body"), dict):
            return []
        meta = _as_dict(actor.get("meta"))
        body = _as_dict(actor["body"])
        ts = scene.get("timestamp")
        device_us: int | None = None
        if isinstance(ts, int | float) and not isinstance(ts, bool):
            try:
                device_us = int(float(ts) * 1e6)
            except (ValueError, OverflowError):  # NaN / inf
                device_us = None
        frames: list[HandFrame] = []
        sides: tuple[tuple[Side, str], ...] = (("left", "hasLeftGlove"), ("right", "hasRightGlove"))
        for side, flag in sides:
            if meta.get("hasGloves") and flag in meta and not meta[flag]:
                continue
            frame = self._hand(side, body, host_time_ns, device_us)
            if frame is not None:
                frames.append(frame)
        return frames

    def _hand(
        self,
        side: Side,
        body: Mapping[str, Any],
        host_time_ns: int,
        device_us: int | None,
    ) -> HandFrame | None:
        hand_bone = body.get(f"{side}Hand")
        hand_q = _rot(hand_bone)
        if hand_q is None:
            return None
        world: dict[str, Quat] = {f"{side}Hand": hand_q}
        joints: dict[str, Quat] = {}
        for xr_name, bone, parent in JOINT_MAP[side]:
            q = _rot(body.get(bone))
            if q is None:
                continue
            world[bone] = q
            parent_q = world.get(parent)
            if parent_q is None:
                continue
            joints[xr_name] = quat_normalize(quat_multiply(quat_conjugate(parent_q), q))
        arm_q = _rot(body.get(f"{side}LowerArm"))
        if arm_q is not None:
            wrist_local = quat_normalize(quat_multiply(quat_conjugate(arm_q), hand_q))
            joints["wrist"] = wrist_local
            joints["palm"] = wrist_local
        pos = _vec(_as_dict(hand_bone).get("position"))
        wrist = Pose(
            orientation=quat_normalize(yup_to_zup_quat(hand_q)),
            position=yup_to_zup_vec(pos) if pos is not None else None,
        )
        self._seq += 1
        return HandFrame(
            side=side,
            host_time_ns=host_time_ns,
            seq=self._seq,
            source="studio",
            joints=joints,
            device_time_us=device_us,
            wrist=wrist,
        )


@dataclass(frozen=True)
class StudioLiveness:
    """Snapshot of Studio datagram arrival, for health display (monotonic ns timestamps)."""

    bound: bool
    port: int | None
    bind_error: str | None
    last_datagram_ns: int | None
    last_right_frame_ns: int | None
    datagram_hz: float


_HZ_WINDOW_NS = 1_000_000_000


class StudioSource(ThreadedSource):
    """Binds a UDP socket, parses Studio datagrams and feeds the latest-value buffer."""

    def __init__(self, cfg: StudioConfig, *, tolerate_bind_error: bool = False) -> None:
        super().__init__("studio")
        self._cfg = cfg
        self._tolerate_bind_error = tolerate_bind_error
        self._live_lock = threading.Lock()
        self._bind_error: str | None = None
        self._last_datagram_ns: int | None = None
        self._last_right_ns: int | None = None
        self._arrivals: deque[int] = deque()
        self._parser = StudioParser(cfg.actor)
        self._sock: socket.socket | None = None
        self._tee: IO[str] | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        tee: IO[str] | None = None
        try:
            sock.bind((self._cfg.host, self._cfg.port))
            sock.settimeout(0.1)
            if self._cfg.tee_raw is not None:  # only after a successful bind (open truncates)
                self._cfg.tee_raw.parent.mkdir(parents=True, exist_ok=True)
                tee = self._cfg.tee_raw.open("w", encoding="utf-8", newline="\n")
        except OSError as exc:
            sock.close()
            if not self._tolerate_bind_error:
                raise
            with self._live_lock:
                self._bind_error = f"{self._cfg.host}:{self._cfg.port}: {exc}"
            logger.warning("studio bind failed: %s", self._bind_error)
            return
        with self._live_lock:
            self._bind_error = None
        self._sock = sock
        self._tee = tee
        super().start()

    def liveness(self) -> StudioLiveness:
        """Thread-safe, non-blocking snapshot of datagram arrival."""
        now = time.monotonic_ns()
        with self._live_lock:
            self._prune_arrivals(now)
            hz = float(len(self._arrivals))
            sock = self._sock
            port: int | None = None
            if sock is not None:
                try:
                    port = int(sock.getsockname()[1])
                except OSError:
                    port = None
            return StudioLiveness(
                bound=sock is not None,
                port=port,
                bind_error=self._bind_error,
                last_datagram_ns=self._last_datagram_ns,
                last_right_frame_ns=self._last_right_ns,
                datagram_hz=hz,
            )

    def _prune_arrivals(self, now: int) -> None:
        """Drop arrivals older than the rate window; caller holds ``_live_lock``."""
        while self._arrivals and now - self._arrivals[0] > _HZ_WINDOW_NS:
            self._arrivals.popleft()

    def bound_port(self) -> int:
        if self._sock is None:
            raise RuntimeError("source not started")
        return int(self._sock.getsockname()[1])

    def stop(self) -> None:
        super().stop()
        if self._sock is not None:
            self._sock.close()
            self._sock = None
        if self._tee is not None:
            self._tee.close()
            self._tee = None

    def _run(self, stop: threading.Event) -> None:
        sock = self._sock
        if sock is None:
            return
        while not stop.is_set():
            try:
                data, _addr = sock.recvfrom(65535)
            except TimeoutError:
                continue
            except ConnectionResetError:  # Windows ICMP reset noise
                continue
            now = time.monotonic_ns()
            with self._live_lock:
                self._last_datagram_ns = now
                self._arrivals.append(now)
                self._prune_arrivals(now)
            tee = self._tee
            if tee is not None:
                rec = {
                    "t_ns": now,
                    "src": "studio",
                    "payload_b64": base64.b64encode(data).decode("ascii"),
                }
                try:
                    tee.write(json.dumps(rec, separators=(",", ":")) + "\n")
                    tee.flush()
                except (OSError, ValueError):
                    logger.exception("studio tee write failed; disabling raw tee")
                    self._tee = None
                    with contextlib.suppress(OSError, ValueError):
                        tee.close()
            for frame in self._parser.parse(data, now):
                self._buffer.put(frame)
                if frame.side == "right":
                    with self._live_lock:
                        self._last_right_ns = now
