"""Rokoko Device SDK RGMP v2 stream parser, RCSP v1 USB subscriber and ``SdkSource``.

Connects to ``rkk-hand-solver`` (default ``127.0.0.1:12277``) over TCP RGMP and issues
best-effort RCSP ``SubscribeToData`` on ``127.0.0.1:45451`` for USB-attached Smartgloves II.
"""

from __future__ import annotations

import base64
import contextlib
import enum
import itertools
import json
import logging
import math
import re
import socket
import struct
import threading
import time
from collections.abc import Buffer, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, cast

from teleop.config import StudioConfig
from teleop.input.buffer import ThreadedSource
from teleop.input.studio import (
    quat_conjugate,
    quat_multiply,
    quat_normalize,
    yup_to_zup_quat,
    yup_to_zup_vec,
)
from teleop.types import OPENXR_JOINTS, HandFrame, Pose, Quat, Side, Vec3

logger = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_DRIVER_PORT = 12276
DEFAULT_SOLVER_PORT = 12277
DEFAULT_RCSP_PORT = 45451
STUDIO_DEFAULT_PORT = 14043

MSG_PREFIX_DEFINITION = 1
MSG_PREFIX_DATA = 2
MSG_PREFIX_DISCONNECT = 3

OUTER_HEADER_FMT = "<II"
OUTER_HEADER_SIZE = struct.calcsize(OUTER_HEADER_FMT)
DATA_HEADER_FMT = "<IIQ"
DATA_HEADER_SIZE = struct.calcsize(DATA_HEADER_FMT)
DISCONNECT_FMT = "<I"
DISCONNECT_SIZE = struct.calcsize(DISCONNECT_FMT)

RCSP_MARKER = 0xDC
RCSP_HEADER_VERSION = 1
RCSP_HEADER_SIZE = 8
MAX_STREAM_BUFFER = 1 << 22
MAX_RCSP_PAYLOAD = 1 << 20


class RcspPayloadType(enum.IntEnum):
    COMMAND = 1
    RESPONSE_OK = 2
    RESPONSE_ERROR = 3
    EVENT = 4


_BASE_TYPES: Mapping[str, tuple[str, int]] = {
    "INT32": ("i", 4),
    "UINT32": ("I", 4),
    "INT64": ("q", 8),
    "UINT64": ("Q", 8),
    "FLOAT": ("f", 4),
    "DOUBLE": ("d", 8),
}
_DATA_TYPE_RE = re.compile(
    r"^(INT32|UINT32|INT64|UINT64|FLOAT|DOUBLE)(?:\[(\d+)(?:,\s*(\d+))?\])?$"
)


def parse_data_type(data_type: str) -> tuple[str, int]:
    """Return ``(struct_fmt, byte_size)`` for an RGMP v2 ``data_type`` string."""
    m = _DATA_TYPE_RE.match(data_type.strip())
    if m is None:
        raise ValueError(f"unrecognized RGMP data_type: {data_type!r}")
    fmt_char, elem_bytes = _BASE_TYPES[m.group(1)]
    cols = int(m.group(2)) if m.group(2) is not None else 1
    rows = int(m.group(3)) if m.group(3) is not None else 1
    if cols < 1 or rows < 1:
        raise ValueError(f"RGMP array dimensions must be >= 1: {data_type!r}")
    count = cols * rows
    return f"<{count}{fmt_char}", elem_bytes * count


def encode_frame(prefix: int, payload: Buffer) -> bytes:
    """Pack a single ``[u32 prefix][u32 len][payload]`` RGMP frame."""
    raw = bytes(payload)
    return struct.pack(OUTER_HEADER_FMT, prefix, len(raw)) + raw


def parse_frames(buf: bytearray) -> list[tuple[int, bytes]]:
    """Consume complete ``[u32 prefix][u32 len][payload]`` frames; leave partial tail in ``buf``."""
    out: list[tuple[int, bytes]] = []
    offset = 0
    total = len(buf)
    while total - offset >= OUTER_HEADER_SIZE:
        hdr = struct.unpack_from(OUTER_HEADER_FMT, buf, offset)
        prefix, length = int(hdr[0]), int(hdr[1])
        end = offset + OUTER_HEADER_SIZE + length
        if end > total:
            break
        out.append((prefix, bytes(buf[offset + OUTER_HEADER_SIZE : end])))
        offset = end
    if offset:
        del buf[:offset]
    return out


@dataclass(frozen=True, slots=True)
class StreamField:
    key: str
    offset: int
    struct_fmt: str
    byte_size: int
    data_type: str
    measure_type: str
    target_frame: str
    reference_frame: str
    custom_label: str | None = None


@dataclass(frozen=True, slots=True)
class StreamGroup:
    name: str
    expected_rate_hz: float
    streams: tuple[StreamField, ...]
    byte_size: int


def _parse_group(raw_obj: Any) -> StreamGroup:
    if not isinstance(raw_obj, dict):
        raise ValueError("RGMP group entry must be an object")
    grp = cast("dict[str, Any]", raw_obj)
    name = str(grp.get("name", ""))
    rate = float(grp.get("expected_rate_hz", 0.0))
    streams_raw: Any = grp.get("streams") or []
    if not isinstance(streams_raw, list):
        raise ValueError(f"RGMP group {name!r} streams must be a list")

    raw_list = cast("list[Any]", streams_raw)
    entries = [cast("dict[str, Any]", s) for s in raw_list if isinstance(s, dict)]
    if len(entries) != len(raw_list):
        raise ValueError(f"RGMP group {name!r} has non-object stream entry")

    target_counts: dict[str, int] = {}
    for entry in entries:
        tf = str(entry.get("target_frame") or entry.get("name") or "")
        target_counts[tf] = target_counts.get(tf, 0) + 1

    used_keys: set[str] = set()
    fields: list[StreamField] = []
    offset = 0
    for idx, entry in enumerate(entries):
        data_type = str(entry["data_type"])
        struct_fmt, byte_size = parse_data_type(data_type)
        measure_type = str(entry.get("measure_type", ""))
        target_frame = str(entry.get("target_frame") or entry.get("name") or f"stream_{idx}")
        ref_raw = entry.get("reference_frame")
        reference_frame = target_frame if ref_raw is None else str(ref_raw)
        label_raw = entry.get("custom_label")
        custom_label = None if label_raw is None else str(label_raw)

        explicit_name = entry.get("name")
        if explicit_name is not None:
            candidate = str(explicit_name)
        elif target_counts.get(target_frame, 0) <= 1:
            candidate = target_frame
        else:
            suffix = custom_label or measure_type or str(idx)
            candidate = f"{target_frame}:{suffix}"
        key = f"{candidate}:{idx}" if candidate in used_keys else candidate
        used_keys.add(key)

        fields.append(
            StreamField(
                key=key,
                offset=offset,
                struct_fmt=struct_fmt,
                byte_size=byte_size,
                data_type=data_type,
                measure_type=measure_type,
                target_frame=target_frame,
                reference_frame=reference_frame,
                custom_label=custom_label,
            )
        )
        offset += byte_size
    return StreamGroup(name=name, expected_rate_hz=rate, streams=tuple(fields), byte_size=offset)


@dataclass(frozen=True, slots=True)
class StreamDefinition:
    device_id: int
    device_type: str
    timestamp_epoch: str
    device_info: Mapping[str, Any]
    groups: tuple[StreamGroup, ...]
    static_data: tuple[Mapping[str, Any], ...] = ()

    @classmethod
    def from_json(cls, payload: bytes | str) -> StreamDefinition:
        text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
        loaded: Any = json.loads(text)
        if not isinstance(loaded, dict):
            raise ValueError("RGMP stream definition must be a JSON object")
        raw = cast("dict[str, Any]", loaded)
        device_id = int(raw["device_id"])
        if not (0 <= device_id < (1 << 32)):
            raise ValueError(f"device_id out of uint32 range: {device_id}")
        device_type = str(raw.get("device_type", ""))
        timestamp_epoch = str(raw.get("timestamp_epoch", ""))
        info_raw: Any = raw.get("device_info") or {}
        device_info: dict[str, Any] = (
            dict(cast("dict[str, Any]", info_raw)) if isinstance(info_raw, dict) else {}
        )
        static_raw: Any = raw.get("static_data") or []
        static_list: list[Any] = (
            cast("list[Any]", static_raw) if isinstance(static_raw, list) else []
        )
        static_data: list[Mapping[str, Any]] = [
            dict(cast("dict[str, Any]", item)) for item in static_list if isinstance(item, dict)
        ]
        groups_raw: Any = raw.get("groups") or []
        if not isinstance(groups_raw, list):
            raise ValueError("RGMP groups must be a list")
        groups = tuple(_parse_group(g) for g in cast("list[Any]", groups_raw))
        return cls(
            device_id=device_id,
            device_type=device_type,
            timestamp_epoch=timestamp_epoch,
            device_info=device_info,
            groups=groups,
            static_data=tuple(static_data),
        )

    def hand_side(self) -> Side | None:
        for k in ("hand", "side", "handedness"):
            v = str(self.device_info.get(k, "")).strip().lower()
            if v == "left":
                return "left"
            if v == "right":
                return "right"
        for k in ("is_left", "IsLeft"):
            flag = self.device_info.get(k)
            if isinstance(flag, bool):
                return "left" if flag else "right"
        friendly = str(self.device_info.get("friendly_name", "")).lower()
        if "left" in friendly and "right" not in friendly:
            return "left"
        if "right" in friendly and "left" not in friendly:
            return "right"
        return None

    def preferred_group_id(self) -> int | None:
        """Pick one solver group so ``--emit both`` / ``--emit-openxr`` never double-emits."""
        by_name = {grp.name: idx for idx, grp in enumerate(self.groups)}
        for candidate in ("joints_openxr", "joints_local", "joints_world", "default", ""):
            if candidate in by_name:
                return by_name[candidate]
        return None


def decode_data(
    defn: StreamDefinition | Mapping[int, StreamDefinition],
    payload: Buffer,
) -> tuple[int, int, int, dict[str, tuple[int | float, ...]]]:
    """Decode a prefix-2 RGMP Data Frame via ``struct.unpack_from`` (no alignment assumptions)."""
    mv = memoryview(payload)
    if mv.nbytes < DATA_HEADER_SIZE:
        raise ValueError(f"RGMP data payload too short: {mv.nbytes} < {DATA_HEADER_SIZE}")
    hdr = struct.unpack_from(DATA_HEADER_FMT, mv, 0)
    device_id, group_id, timestamp_us = int(hdr[0]), int(hdr[1]), int(hdr[2])
    if isinstance(defn, StreamDefinition):
        if defn.device_id != device_id:
            raise ValueError(
                f"RGMP device_id mismatch: payload={device_id}, definition={defn.device_id}"
            )
        active = defn
    else:
        if device_id not in defn:
            raise KeyError(f"unknown RGMP device_id: {device_id}")
        active = defn[device_id]
    if group_id >= len(active.groups):
        raise ValueError(f"RGMP group_id {group_id} out of range ({len(active.groups)} groups)")
    group = active.groups[group_id]
    expected_bytes = DATA_HEADER_SIZE + group.byte_size
    if mv.nbytes < expected_bytes:
        raise ValueError(
            f"RGMP group {group.name!r} payload too short: {mv.nbytes} < {expected_bytes}"
        )
    values: dict[str, tuple[int | float, ...]] = {}
    for stream in group.streams:
        unpacked = cast(
            "tuple[int | float, ...]",
            struct.unpack_from(stream.struct_fmt, mv, DATA_HEADER_SIZE + stream.offset),
        )
        values[stream.key] = unpacked
    return device_id, group_id, timestamp_us, values


def decode_disconnect(payload: Buffer) -> int:
    """Decode a prefix-3 RGMP Device Disconnect Frame and return ``device_id``."""
    mv = memoryview(payload)
    if mv.nbytes < DISCONNECT_SIZE:
        raise ValueError(f"RGMP disconnect payload too short: {mv.nbytes} < {DISCONNECT_SIZE}")
    (raw_dev,) = struct.unpack_from(DISCONNECT_FMT, mv, 0)
    return int(raw_dev)


def _build_openxr_parents() -> Mapping[str, str | None]:
    parents: dict[str, str | None] = {"wrist": None, "palm": "wrist"}
    for finger in ("thumb", "index", "middle", "ring", "little"):
        parts = (
            ("metacarpal", "proximal", "distal", "tip")
            if finger == "thumb"
            else ("metacarpal", "proximal", "intermediate", "distal", "tip")
        )
        prev = "wrist"
        for part in parts:
            name = f"{finger}_{part}"
            parents[name] = prev
            prev = name
    return parents


OPENXR_PARENTS: Mapping[str, str | None] = _build_openxr_parents()
_JOINT_ALIASES: Mapping[str, str] = {
    "hand": "wrist",
    "index_medial": "index_intermediate",
    "middle_medial": "middle_intermediate",
    "ring_medial": "ring_intermediate",
    "little_medial": "little_intermediate",
}
_OPENXR_SET = frozenset(OPENXR_JOINTS)


# Glove sensor frame (+X right, +Y toward wrist, +Z down) -> MuJoCo Z-up: +180 deg about X.
# Basis conjugation by R_x(pi) = (1, 0, 0, 0) maps (x, y, z) -> (x, -y, -z) and
# q = (qx, qy, qz, qw) -> (qx, -qy, -qz, qw), preserving identity (0, 0, 0, 1).
def sensor_to_zup_vec(v: Vec3) -> Vec3:
    return (v[0], -v[1], -v[2])


def sensor_to_zup_quat(q: Quat) -> Quat:
    return (q[0], -q[1], -q[2], q[3])


def _extract_pose_or_quat(vals: tuple[int | float, ...]) -> tuple[Vec3 | None, Quat | None]:
    if len(vals) == 7:
        px, py, pz, qx, qy, qz, qw = (float(x) for x in vals)
        pos: Vec3 | None = (
            (px, py, pz)
            if (math.isfinite(px) and math.isfinite(py) and math.isfinite(pz))
            else None
        )
        return pos, quat_normalize((qx, qy, qz, qw))
    if len(vals) == 4:
        qx, qy, qz, qw = (float(x) for x in vals)
        return None, quat_normalize((qx, qy, qz, qw))
    return None, None


def _nonzero_pos(pos: Vec3 | None, eps: float = 1e-7) -> Vec3 | None:
    if pos is None:
        return None
    return pos if (abs(pos[0]) > eps or abs(pos[1]) > eps or abs(pos[2]) > eps) else None


class RgmpSession:
    """Stateful RGMP byte-stream decoder producing ``HandFrame`` items."""

    def __init__(self) -> None:
        self.definitions: dict[int, StreamDefinition] = {}
        self._world_pos_zup: dict[int, tuple[int, Vec3]] = {}
        self._buf = bytearray()
        self._seq = 0

    def reset(self) -> None:
        self.definitions.clear()
        self._world_pos_zup.clear()
        self._buf.clear()

    def parse(self, payload: bytes, host_time_ns: int) -> list[HandFrame]:
        return self.feed(payload, host_time_ns)

    def drain_wire_frames(self, chunk: Buffer) -> list[tuple[int, bytes]]:
        self._buf.extend(memoryview(chunk))
        raw_frames = parse_frames(self._buf)
        if len(self._buf) > MAX_STREAM_BUFFER:
            logger.warning(
                "RGMP stream buffer exceeded %d bytes; resetting session", MAX_STREAM_BUFFER
            )
            self.reset()
        return raw_frames

    def handle_frame(self, prefix: int, payload: bytes, host_time_ns: int) -> HandFrame | None:
        if prefix == MSG_PREFIX_DEFINITION:
            try:
                defn = StreamDefinition.from_json(payload)
            except (UnicodeDecodeError, ValueError, KeyError, TypeError):
                logger.warning("ignoring malformed RGMP definition frame")
                return None
            self.definitions[defn.device_id] = defn
            self._world_pos_zup.pop(defn.device_id, None)
            return None
        if prefix == MSG_PREFIX_DISCONNECT:
            with contextlib.suppress(ValueError):
                dev_id = decode_disconnect(payload)
                self.definitions.pop(dev_id, None)
                self._world_pos_zup.pop(dev_id, None)
            return None
        if prefix == MSG_PREFIX_DATA:
            return self._decode_hand_frame(payload, host_time_ns)
        return None

    def feed(self, chunk: Buffer, host_time_ns: int) -> list[HandFrame]:
        frames: list[HandFrame] = []
        for prefix, payload in self.drain_wire_frames(chunk):
            frame = self.handle_frame(prefix, payload, host_time_ns)
            if frame is not None:
                frames.append(frame)
        return frames

    def _cache_sensor_root_pos(
        self,
        dev_id: int,
        ts_us: int,
        group: StreamGroup,
        values: Mapping[str, tuple[int | float, ...]],
    ) -> None:
        for stream in group.streams:
            if stream.target_frame not in ("wrist", "hand", "forearm"):
                continue
            vals = values.get(stream.key)
            if vals is None:
                continue
            with contextlib.suppress(ValueError):
                pos, _ = _extract_pose_or_quat(vals)
                nz = _nonzero_pos(pos)
                if nz is not None:
                    self._world_pos_zup[dev_id] = (ts_us, sensor_to_zup_vec(nz))
                    return

    def _decode_hand_frame(self, payload: bytes, host_time_ns: int) -> HandFrame | None:
        try:
            dev_id, group_id, ts_us, values = decode_data(self.definitions, payload)
        except (KeyError, ValueError):
            return None
        defn = self.definitions[dev_id]
        side = defn.hand_side()
        if side is None:
            return None
        pref = defn.preferred_group_id()
        if pref is None:
            return None
        group = defn.groups[group_id]
        if group_id != pref:
            if group.name in ("joints_local", "joints_world"):
                self._cache_sensor_root_pos(dev_id, ts_us, group, values)
            return None
        if group.name in ("joints_openxr", "joints_world"):
            return self._from_world_group(
                dev_id,
                side,
                ts_us,
                host_time_ns,
                group,
                values,
                openxr=(group.name == "joints_openxr"),
            )
        return self._from_local_group(side, ts_us, host_time_ns, group, values)

    def _from_world_group(
        self,
        dev_id: int,
        side: Side,
        ts_us: int,
        host_time_ns: int,
        group: StreamGroup,
        values: Mapping[str, tuple[int | float, ...]],
        *,
        openxr: bool,
    ) -> HandFrame | None:
        world_quats: dict[str, Quat] = {}
        forearm_q: Quat | None = None
        wrist_pos_raw: Vec3 | None = None
        forearm_pos_raw: Vec3 | None = None
        for stream in group.streams:
            vals = values.get(stream.key)
            if vals is None:
                continue
            try:
                pos, quat = _extract_pose_or_quat(vals)
            except ValueError:
                return None
            if stream.target_frame == "forearm":
                forearm_q = quat
                forearm_pos_raw = pos
                continue
            joint = _JOINT_ALIASES.get(stream.target_frame, stream.target_frame)
            if joint not in _OPENXR_SET:
                continue
            if quat is not None:
                world_quats[joint] = quat
            if joint == "wrist" and pos is not None:
                wrist_pos_raw = pos
        if not world_quats:
            return None
        joints: dict[str, Quat] = {}
        for joint, q_world in world_quats.items():
            parent = OPENXR_PARENTS.get(joint)
            if parent is None:
                joints[joint] = (
                    quat_normalize(quat_multiply(quat_conjugate(forearm_q), q_world))
                    if forearm_q is not None
                    else q_world
                )
            elif parent in world_quats:
                joints[joint] = quat_normalize(
                    quat_multiply(quat_conjugate(world_quats[parent]), q_world)
                )
        if "palm" not in joints and "wrist" in joints:
            joints["palm"] = joints["wrist"]

        wrist_world = world_quats.get("wrist")
        nz_pos = _nonzero_pos(wrist_pos_raw) or _nonzero_pos(forearm_pos_raw)
        to_quat = yup_to_zup_quat if openxr else sensor_to_zup_quat
        to_vec = yup_to_zup_vec if openxr else sensor_to_zup_vec
        pos_zup: Vec3 | None = to_vec(nz_pos) if nz_pos is not None else None
        if pos_zup is None and openxr:
            cached = self._world_pos_zup.get(dev_id)
            if cached is not None and cached[0] == ts_us:
                pos_zup = cached[1]
        wrist = (
            Pose(orientation=quat_normalize(to_quat(wrist_world)), position=pos_zup)
            if wrist_world is not None
            else None
        )
        self._seq += 1
        return HandFrame(
            side=side,
            host_time_ns=host_time_ns,
            seq=self._seq,
            source="rokoko-sdk",
            joints=joints,
            device_time_us=ts_us,
            wrist=wrist,
        )

    def _from_local_group(
        self,
        side: Side,
        ts_us: int,
        host_time_ns: int,
        group: StreamGroup,
        values: Mapping[str, tuple[int | float, ...]],
    ) -> HandFrame | None:
        if group.name not in ("joints_local", "default", ""):
            return None
        joints: dict[str, Quat] = {}
        wrist_pos_raw: Vec3 | None = None
        forearm_pos_raw: Vec3 | None = None
        wrist_q_local: Quat | None = None
        forearm_q_world: Quat | None = None
        for stream in group.streams:
            vals = values.get(stream.key)
            if vals is None:
                continue
            try:
                pos, quat = _extract_pose_or_quat(vals)
            except ValueError:
                return None
            if stream.target_frame == "forearm":
                forearm_q_world = quat
                forearm_pos_raw = pos
                continue
            joint = _JOINT_ALIASES.get(stream.target_frame, stream.target_frame)
            if joint not in _OPENXR_SET:
                continue
            if quat is not None:
                joints[joint] = quat
                if joint == "wrist":
                    wrist_q_local = quat
            if joint == "wrist" and pos is not None:
                wrist_pos_raw = pos
        if not joints:
            return None
        if "palm" not in joints and "wrist" in joints:
            joints["palm"] = joints["wrist"]

        if forearm_q_world is not None and wrist_q_local is not None:
            wrist_world_q: Quat | None = quat_normalize(
                quat_multiply(forearm_q_world, wrist_q_local)
            )
        else:
            wrist_world_q = wrist_q_local if wrist_q_local is not None else forearm_q_world
        nz_pos = _nonzero_pos(wrist_pos_raw) or _nonzero_pos(forearm_pos_raw)
        wrist = (
            Pose(
                orientation=quat_normalize(sensor_to_zup_quat(wrist_world_q)),
                position=sensor_to_zup_vec(nz_pos) if nz_pos is not None else None,
            )
            if wrist_world_q is not None
            else None
        )
        self._seq += 1
        return HandFrame(
            side=side,
            host_time_ns=host_time_ns,
            seq=self._seq,
            source="rokoko-sdk",
            joints=joints,
            device_time_us=ts_us,
            wrist=wrist,
        )


def _rcsp_recv_exact(sock: socket.socket, size: int) -> bytes:
    buf = bytearray()
    while len(buf) < size:
        chunk = sock.recv(size - len(buf))
        if not chunk:
            raise EOFError("RCSP connection closed prematurely")
        buf.extend(chunk)
    return bytes(buf)


def _rcsp_send_cmd(
    sock: socket.socket,
    track_id: str,
    command: str,
    arguments: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    body_obj: dict[str, Any] = {"Command": command, "TrackId": track_id, "Version": 1}
    if arguments is not None:
        body_obj["Arguments"] = dict(arguments)
    body = json.dumps(body_obj, separators=(",", ":")).encode("utf-8")
    header = bytes((RCSP_MARKER, RCSP_HEADER_VERSION, RCSP_HEADER_SIZE, RcspPayloadType.COMMAND))
    sock.sendall(header + len(body).to_bytes(4, "little") + body)

    while True:
        hdr = _rcsp_recv_exact(sock, RCSP_HEADER_SIZE)
        if hdr[0] != RCSP_MARKER:
            raise ValueError(f"invalid RCSP marker: 0x{hdr[0]:02x}")
        ptype = RcspPayloadType(hdr[3])
        plen = int.from_bytes(hdr[4:8], "little")
        if plen > MAX_RCSP_PAYLOAD:
            raise ValueError(f"RCSP payload too large: {plen} > {MAX_RCSP_PAYLOAD}")
        raw_payload: Any = json.loads(_rcsp_recv_exact(sock, plen).decode("utf-8"))
        payload = cast("dict[str, Any]", raw_payload) if isinstance(raw_payload, dict) else {}
        if ptype == RcspPayloadType.EVENT:
            continue
        if ptype == RcspPayloadType.RESPONSE_ERROR:
            err_raw: Any = payload.get("Error") or {}
            err = cast("dict[str, Any]", err_raw) if isinstance(err_raw, dict) else {}
            raise RuntimeError(f"RCSP {command} failed: {err.get('Code')}: {err.get('Message')}")
        if ptype == RcspPayloadType.RESPONSE_OK:
            resp: Any = payload.get("Response") or {}
            return cast("dict[str, Any]", resp) if isinstance(resp, dict) else {}
        raise ValueError(f"unexpected RCSP payload type: {ptype!r}")


def subscribe_usb_devices(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_RCSP_PORT,
    timeout_s: float = 1.0,
) -> list[int]:
    """List RCSP devices on ``host:port`` and send ``SubscribeToData`` for USB (``Serial``) ones."""
    counter = itertools.count(1)
    subscribed: list[int] = []
    with socket.create_connection((host, port), timeout=timeout_s) as sock:
        sock.settimeout(timeout_s)
        resp = _rcsp_send_cmd(sock, str(next(counter)), "ListDevices")
        devices_raw: Any = resp.get("Devices") or []
        if not isinstance(devices_raw, list):
            return subscribed
        for item in cast("list[Any]", devices_raw):
            if not isinstance(item, dict):
                continue
            dev = cast("dict[str, Any]", item)
            if bool(dev.get("IsBootloader", False)):
                continue
            conns_raw: Any = dev.get("Connections") or []
            conns = (
                [str(c) for c in cast("list[Any]", conns_raw)]
                if isinstance(conns_raw, list)
                else []
            )
            if "Serial" not in conns:
                continue
            dev_id = int(dev["DeviceId"])
            _rcsp_send_cmd(
                sock,
                str(next(counter)),
                "SubscribeToData",
                {"DeviceId": dev_id, "Connection": "Serial"},
            )
            subscribed.append(dev_id)
    return subscribed


_RCSP_RETRY_S = 2.0


class SdkSource(ThreadedSource):
    """Connects to ``rkk-hand-solver`` RGMP TCP output and subscribes USB Smartgloves via RCSP."""

    def __init__(
        self,
        cfg: StudioConfig | None = None,
        *,
        host: str | None = None,
        port: int | None = None,
        rcsp_port: int | None = DEFAULT_RCSP_PORT,
        reconnect_s: float = 0.25,
    ) -> None:
        super().__init__("rokoko-sdk")
        if cfg is not None:
            cfg_host = DEFAULT_HOST if cfg.host in ("0.0.0.0", "") else cfg.host
            cfg_port = DEFAULT_SOLVER_PORT if cfg.port == STUDIO_DEFAULT_PORT else cfg.port
            tee_raw: Path | None = cfg.tee_raw
        else:
            cfg_host = DEFAULT_HOST
            cfg_port = DEFAULT_SOLVER_PORT
            tee_raw = None
        self.host = cfg_host if host is None else host
        self.port = cfg_port if port is None else port
        self.rcsp_port = rcsp_port
        self._tee_path = tee_raw
        self._tee: IO[str] | None = None
        self._reconnect_s = max(0.02, reconnect_s)
        self._sock: socket.socket | None = None
        self._sock_lock = threading.Lock()
        self._last_rcsp_try = -math.inf

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        if self._tee_path is not None:
            self._tee_path.parent.mkdir(parents=True, exist_ok=True)
            self._tee = self._tee_path.open("w", encoding="utf-8", newline="\n")
        self._last_rcsp_try = -math.inf
        super().start()

    def stop(self) -> None:
        self._stop_event.set()
        with self._sock_lock:
            sock = self._sock
            if sock is not None:
                with contextlib.suppress(OSError):
                    sock.shutdown(socket.SHUT_RDWR)
        super().stop()
        if self._tee is not None:
            with contextlib.suppress(OSError, ValueError):
                self._tee.close()
            self._tee = None

    def _maybe_subscribe_usb(self, *, force: bool = False) -> None:
        if self.rcsp_port is None:
            return
        now = time.monotonic()
        if not force and (now - self._last_rcsp_try) < _RCSP_RETRY_S:
            return
        self._last_rcsp_try = now
        try:
            devs = subscribe_usb_devices(self.host, self.rcsp_port, timeout_s=0.5)
            if devs:
                logger.info("RCSP subscribed USB Smartgloves device(s): %s", devs)
        except (OSError, ValueError, RuntimeError) as exc:
            logger.debug("RCSP USB subscribe skipped (%s:%s): %s", self.host, self.rcsp_port, exc)

    def _write_tee(self, prefix: int, payload: bytes, now_ns: int) -> None:
        tee = self._tee
        if tee is None:
            return
        framed = encode_frame(prefix, payload)
        rec = {
            "t_ns": now_ns,
            "src": "rgmp",
            "payload_b64": base64.b64encode(framed).decode("ascii"),
        }
        try:
            tee.write(json.dumps(rec, separators=(",", ":")) + "\n")
            tee.flush()
        except (OSError, ValueError):
            logger.exception("RGMP tee write failed; disabling raw tee")
            self._tee = None
            with contextlib.suppress(OSError, ValueError):
                tee.close()

    def _run(self, stop: threading.Event) -> None:
        session = RgmpSession()
        while not stop.is_set():
            self._maybe_subscribe_usb()
            sock: socket.socket | None = None
            try:
                sock = socket.create_connection((self.host, self.port), timeout=0.5)
                sock.settimeout(0.1)
                with self._sock_lock:
                    self._sock = sock
                session.reset()
                while not stop.is_set():
                    try:
                        chunk = sock.recv(65536)
                    except TimeoutError:
                        continue
                    if not chunk:
                        break
                    now = time.monotonic_ns()
                    for prefix, payload in session.drain_wire_frames(chunk):
                        self._write_tee(prefix, payload, now)
                        frame = session.handle_frame(prefix, payload, now)
                        if frame is not None:
                            self._buffer.put(frame)
            except OSError:
                pass
            finally:
                with self._sock_lock:
                    self._sock = None
                if sock is not None:
                    with contextlib.suppress(OSError):
                        sock.close()
            if not stop.is_set() and stop.wait(self._reconnect_s):
                return
