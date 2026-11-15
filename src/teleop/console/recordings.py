"""Start / stop / delete episode recordings from the console's Data page.

One **recording** is one Start -> Stop window. Start opens a fresh LeRobot dataset folder
``<root>/<YYYYmmdd-HHMMSS>/`` and hangs an ``EpisodeSink`` on the coordinator, so every rescue or
autonomous recovery that happens until Stop becomes an episode in it. Stop unhooks the sink and
finalises the dataset on a background thread (encoding takes a while). Delete removes a whole
recording folder: LeRobot v3 packs episodes into shared files, so a single episode can't be cut
out cleanly, and ``teleop process`` already drops failed ones at QC.

Browser messages: ``{"type": "record", "action": "start" | "stop"}`` and
``{"type": "record", "action": "delete", "id": "<folder name>"}``. They arrive on the WebSocket
thread; the observer swap is queued and applied by ``pump()`` on the control thread, so the 60 Hz
loop never sees a half-closed sink. Folder scans run on a scanner thread; ``status()`` only
reads the cached list, so telemetry never touches the disk.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
import shutil
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, cast

logger = logging.getLogger("teleop.console.recordings")

SIDECAR = Path("meta") / "teleop_episodes.jsonl"
SCAN_INTERVAL_S = 3.0
MAX_LISTED = 50
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_STAMP_RE = re.compile(r"^(\d{8}-\d{6})")


class SinkLike(Protocol):
    @property
    def written(self) -> int: ...

    def close(self) -> None: ...


@dataclass
class _Active:
    id: str
    path: Path
    sink: SinkLike
    started_at: str


@dataclass
class _Desk:
    """State shared between the WebSocket, control, closer and scanner threads."""

    active: _Active | None = None
    saving: dict[str, SinkLike] = field(default_factory=dict[str, SinkLike])
    recent: list[SinkLike] = field(default_factory=list[SinkLike])
    closing: set[str] = field(default_factory=set[str])  # saving ids with a closer running
    deleting: set[str] = field(default_factory=set[str])
    pending_observer: list[SinkLike | None] = field(default_factory=list[SinkLike | None])
    listing: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    note: str = ""


def _read_episodes(sidecar: Path) -> list[dict[str, Any]]:
    """Sidecar lines as small dicts for the page; bad or half-written lines are skipped."""
    out: list[dict[str, Any]] = []
    try:
        text = sidecar.read_text(encoding="utf-8")
    except OSError:
        return out
    for line in text.splitlines():
        try:
            raw: Any = json.loads(line)
        except ValueError:
            continue
        if not isinstance(raw, dict):
            continue
        d = cast("dict[str, Any]", raw)
        audit_raw = d.get("audit")
        audit_note: str | None = None
        if isinstance(audit_raw, dict):
            audit_map = cast("dict[str, Any]", audit_raw)
            note_val = audit_map.get("operator_note")
            if isinstance(note_val, str) and note_val.strip():
                audit_note = note_val.strip()
        tag_raw = d.get("dataset_tag")
        dataset_tag = (
            tag_raw.strip().lower() if isinstance(tag_raw, str) and tag_raw.strip() else None
        )
        out.append(
            {
                "index": d.get("episode_index"),
                "at": d.get("recorded_at"),
                "bay": d.get("bay"),
                "source": d.get("source"),
                "fault": d.get("fault"),
                "success": bool(d.get("success")),
                "frames": d.get("frames"),
                "reason": d.get("aborted_reason"),
                "dataset_tag": dataset_tag,
                "audit_note": audit_note,
            }
        )
    return out


def _started_at(name: str, folder: Path) -> str:
    m = _STAMP_RE.match(name)
    if m is not None:
        with contextlib.suppress(ValueError):
            return datetime.strptime(m.group(1), "%Y%m%d-%H%M%S").isoformat(timespec="seconds")
    try:
        return datetime.fromtimestamp(folder.stat().st_mtime).isoformat(timespec="seconds")
    except OSError:
        return ""


def scan(root: Path) -> list[dict[str, Any]]:
    """Every recording folder under ``root`` (one with a sidecar), newest first."""
    rows: list[dict[str, Any]] = []
    try:
        children = [p for p in root.iterdir() if p.is_dir()]
    except OSError:
        return rows
    for folder in children:
        sidecar = folder / SIDECAR
        if not sidecar.is_file():
            continue
        ended = ""
        with contextlib.suppress(OSError):
            ended = datetime.fromtimestamp(sidecar.stat().st_mtime).isoformat(timespec="seconds")
        rows.append(
            {
                "id": folder.name,
                "started_at": _started_at(folder.name, folder),
                "ended_at": ended,
                "episodes": _read_episodes(sidecar),
            }
        )
    rows.sort(key=lambda r: (r["started_at"], r["id"]), reverse=True)
    return rows[:MAX_LISTED]


class RecordingDesk:
    def __init__(
        self,
        root: Path,
        open_sink: Callable[[Path], SinkLike],
        set_observer: Callable[[Any], None],
        *,
        unavailable: str | None = None,
        clock: Callable[[], datetime] = datetime.now,
        scan_interval_s: float = SCAN_INTERVAL_S,
    ) -> None:
        self.root = Path(root)
        self.display_root = _display(self.root)
        self.open_sink = open_sink
        self.set_observer = set_observer
        self.unavailable = unavailable
        self.clock = clock
        self.scan_interval_s = scan_interval_s
        self._lock = threading.Lock()
        self._desk = _Desk()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._closers: list[threading.Thread] = []
        self._scanner: threading.Thread | None = None

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> None:
        self.rescan()
        self._scanner = threading.Thread(target=self._scan_loop, name="rec-scan", daemon=True)
        self._scanner.start()

    def close(self) -> None:
        """Stop an active recording and wait for every dataset to be finalised."""
        self.stop_recording()
        with self._lock:
            self._desk.pending_observer.clear()
        self.set_observer(None)  # the control loop has stopped by now
        self._close_stopped(keep=None)
        for t in list(self._closers):
            t.join()
        self._stop.set()
        self._wake.set()
        if self._scanner is not None:
            self._scanner.join(timeout=5)

    # ------------------------------------------------------------------ browser messages

    def handle(self, msg: dict[str, Any]) -> None:
        action = msg.get("action")
        if action == "start":
            self.start_recording()
        elif action == "stop":
            self.stop_recording()
        elif action == "delete":
            rid = msg.get("id")
            if isinstance(rid, str):
                self.delete(rid)

    def start_recording(self) -> bool:
        if self.unavailable is not None:
            self._note(self.unavailable)
            return False
        with self._lock:
            if self._desk.active is not None:
                return False
            folder = self._new_folder()
            try:
                sink = self.open_sink(folder)
            except Exception as exc:
                logger.exception("console: could not start recording in %s", folder)
                self._desk.note = f"Could not start recording: {exc}"
                return False
            self._desk.active = _Active(
                folder.name, folder, sink, self.clock().isoformat(timespec="seconds")
            )
            self._desk.pending_observer.append(sink)
            self._desk.note = ""
        logger.info("console: recording episodes to %s", folder)
        return True

    def stop_recording(self) -> bool:
        with self._lock:
            active, self._desk.active = self._desk.active, None
            if active is None:
                return False
            self._desk.saving[active.id] = active.sink
            self._desk.pending_observer.append(None)
        return True

    def delete(self, rid: str) -> bool:
        if not _NAME_RE.match(rid) or ".." in rid:
            return False
        target = self.root / rid
        with self._lock:
            busy = (
                self._desk.active is not None and self._desk.active.id == rid
            ) or rid in self._desk.saving
            if busy:
                self._desk.note = "Stop the recording and let it finish saving before deleting it"
                return False
            if not (target / SIDECAR).is_file() or target.resolve().parent != self.root.resolve():
                return False
            self._desk.deleting.add(rid)

        def remove() -> None:
            try:
                shutil.rmtree(target)
                logger.info("console: deleted recording %s", target)
            except OSError as exc:
                logger.warning("console: could not delete %s (%s)", target, exc)
                self._note(f"Could not delete {rid}: {exc.strerror or exc}")
            finally:
                self.rescan()
                with self._lock:
                    self._desk.deleting.discard(rid)

        threading.Thread(target=remove, name="rec-delete", daemon=True).start()
        return True

    # ------------------------------------------------------------------ control thread

    def pump(self) -> None:
        """Apply a queued observer swap; call once per control tick."""
        with self._lock:
            if not self._desk.pending_observer:
                return
            observer = self._desk.pending_observer[-1]
            self._desk.pending_observer.clear()
        self.set_observer(observer)
        # Stopped sinks are off the coordinator now: finalise them off the control thread.
        self._close_stopped(keep=observer)

    def _close_stopped(self, keep: SinkLike | None) -> None:
        with self._lock:
            todo = [
                (rid, sink)
                for rid, sink in self._desk.saving.items()
                if sink is not keep and rid not in self._desk.closing
            ]
            self._desk.closing.update(rid for rid, _ in todo)
        for rid, sink in todo:
            t = threading.Thread(
                target=self._finalise, args=(rid, sink), name="rec-close", daemon=True
            )
            self._closers.append(t)
            t.start()

    def _finalise(self, rid: str, sink: SinkLike) -> None:
        try:
            sink.close()
        except Exception:
            logger.exception("console: finalising recording %s failed", rid)
        finally:
            with self._lock:
                self._desk.recent.append(sink)
                if len(self._desk.recent) > 8:
                    del self._desk.recent[:-8]
            self.rescan()  # list it as saved before dropping "saving": it never blinks out
            with self._lock:
                self._desk.saving.pop(rid, None)
                self._desk.closing.discard(rid)

    def tag_episode(
        self, uid: str, *, dataset_tag: str, audit: Mapping[str, Any] | None = None
    ) -> bool:
        """Forward a Tier 3 Argon ``dataset_tag`` to the active, saving, or recently saved sink."""
        with self._lock:
            candidates: list[SinkLike] = []
            if self._desk.active is not None:
                candidates.append(self._desk.active.sink)
            candidates.extend(self._desk.saving.values())
            candidates.extend(reversed(self._desk.recent))
        for sink in candidates:
            tagger = getattr(sink, "tag_episode", None)
            if callable(tagger):
                try:
                    if bool(tagger(uid, dataset_tag=dataset_tag, audit=audit)):
                        self.rescan()
                        return True
                except Exception:
                    logger.exception("console: tagging episode %s failed", uid)
        return False

    # ------------------------------------------------------------------ scanning

    def rescan(self) -> None:
        rows = scan(self.root)
        with self._lock:
            self._desk.listing = rows
        self._wake.set()

    def _scan_loop(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(self.scan_interval_s)
            self._wake.clear()
            if self._stop.is_set():
                return
            rows = scan(self.root)
            with self._lock:
                self._desk.listing = rows

    # ------------------------------------------------------------------ telemetry

    def status(self) -> dict[str, Any]:
        with self._lock:
            d = self._desk
            active = d.active
            listed = {r["id"] for r in d.listing}
            rows: list[dict[str, Any]] = []
            if active is not None and active.id not in listed:
                rows.append(
                    {"id": active.id, "started_at": active.started_at, "ended_at": "",
                     "episodes": []}
                )  # fmt: skip
            rows.extend(d.listing)
            out_rows: list[dict[str, Any]] = []
            for r in rows:
                state = "saved"
                if active is not None and r["id"] == active.id:
                    state = "recording"
                elif r["id"] in d.saving:
                    state = "saving"
                elif r["id"] in d.deleting:
                    state = "deleting"
                out_rows.append({**r, "state": state})
            return {
                "available": self.unavailable is None,
                "unavailable": self.unavailable,
                "root": self.display_root,
                "recording": None
                if active is None
                else {
                    "id": active.id,
                    "started_at": active.started_at,
                    "episodes": _written(active.sink),
                },
                "saving": len(d.saving),
                "note": d.note,
                "recordings": out_rows,
            }

    def _note(self, text: str) -> None:
        with self._lock:
            self._desk.note = text

    def _new_folder(self) -> Path:
        stamp = self.clock().strftime("%Y%m%d-%H%M%S")
        candidate = self.root / stamp
        n = 1
        while candidate.exists() or (
            self._desk.active is not None and self._desk.active.id == candidate.name
        ):
            candidate = self.root / f"{stamp}-{n}"
            n += 1
        return candidate


def _display(root: Path) -> str:
    """``root`` relative to the working directory when it is inside it, with ``/`` separators."""
    try:
        return root.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except (OSError, ValueError):
        return root.as_posix()


def _written(sink: SinkLike) -> int:
    try:
        return int(sink.written)
    except Exception:  # noqa: BLE001 - a status read must never fail telemetry
        return 0
