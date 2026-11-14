"""T-231: the deployed DMP skill library on disk (``data/skills/library.json``).

The coordinator's ``library`` and the UNO Q classifier are lost on exit. ``SkillStore`` keeps a
JSON copy: ``on_deployed`` (a coordinator ``skill_listeners`` callback, so it runs on the gate-apply
path) only appends to an in-memory list and wakes a daemon writer thread, which rewrites the whole
file atomically (temp file, then ``os.replace``). ``load`` runs once at startup; ``reset`` backs
the file up and clears the board and the laptop library (the console's "Start fresh").

Loaded skills are trusted: they passed the gate when saved and are not re-gated. File format:
``{"version": 1, "skills": [{"skill": <to_dict>, "learned_at": ISO, "session": str | null}]}``.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from teleop.atc.types import SkillPrimitive
from teleop.unoq.client import UnoQError
from teleop.unoq.dmp import HAND_ACTUATORS

if TYPE_CHECKING:
    from teleop.atc.coordinator import AtcCoordinator

logger = logging.getLogger(__name__)

FORMAT_VERSION = 1


@dataclass(frozen=True, slots=True)
class SkillEntry:
    skill_id: str
    signature_id: str
    learned_at: str
    session: str | None


@dataclass(frozen=True, slots=True)
class LoadReport:
    loaded: int
    skipped: list[str]


@dataclass(frozen=True, slots=True)
class ResetReport:
    board_cleared: bool
    backup: Path | None


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


class SkillStore:
    def __init__(self, path: Path, *, session: Callable[[], str | None] = lambda: None) -> None:
        self.path = path
        self._session = session
        self._lock = threading.Lock()  # guards the record lists below
        self._io = threading.Lock()  # serializes file writes against reset's backup
        self._records: list[dict[str, Any]] = []  # valid entries, as written to the file
        self._kept: list[dict[str, Any]] = []  # entries load skipped: preserved, never dropped
        self._unreadable = False  # the file on disk could not be parsed: keep it aside first
        # Set by load/reset: lets on_deployed drop a deploy that straddled a reset (its skill is
        # no longer in the coordinator's library, so it must not reach the file either).
        self._coordinator: AtcCoordinator | None = None
        self._wake: queue.Queue[bool] | None = None
        self._writer: threading.Thread | None = None

    # ------------------------------------------------------------------ startup

    def load(self, coordinator: AtcCoordinator) -> LoadReport:
        """Clear the board and library, then restore every valid saved skill into both."""
        self._coordinator = coordinator
        coordinator.reset_library()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if raw["version"] != FORMAT_VERSION or not isinstance(raw["skills"], list):
                raise ValueError(f"unsupported library version {raw['version']!r}")
            items: list[Any] = raw["skills"]
        except FileNotFoundError:
            return LoadReport(0, [])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logger.warning("skills: %s is unreadable (%s); starting empty", self.path, exc)
            self._unreadable = True
            return LoadReport(0, [f"{self.path.name}: {exc}"])
        loaded: list[dict[str, Any]] = []
        kept: list[dict[str, Any]] = []
        skipped: list[str] = []
        library: dict[str, SkillPrimitive] = {}
        for i, item in enumerate(items):
            name = f"entry {i}"
            try:
                if not isinstance(item["skill"], dict):
                    raise TypeError("entry is not a skill object")
                name = str(item["skill"].get("skill_id", name))
                skill = SkillPrimitive.from_dict(item["skill"])
                if skill.actuator_names != HAND_ACTUATORS:
                    raise ValueError("actuator_names do not match the current hand")
                coordinator.unoq.register_skill(skill)
            except (KeyError, TypeError, ValueError, UnoQError) as exc:
                logger.warning("skills: skipping %s (%s)", name, exc)
                skipped.append(f"{name}: {exc}")
                kept.append(item)
                continue
            library[skill.skill_id] = skill
            loaded.append(item)
        coordinator.library.update(library)
        with self._lock:
            self._records = loaded
            self._kept = kept
        return LoadReport(len(loaded), skipped)

    # ------------------------------------------------------------------ save

    def on_deployed(self, skill: SkillPrimitive) -> None:
        """Queue a rewrite with ``skill`` added. Never touches the disk on the caller's thread."""
        co = self._coordinator
        if co is not None and skill.skill_id not in co.library:
            return  # a reset cleared the library between the deploy and this call
        record = {
            "skill": skill.to_dict(),
            "learned_at": datetime.now().isoformat(timespec="seconds"),
            "session": self._session(),
        }
        with self._lock:
            self._records = [r for r in self._records if r["skill"]["skill_id"] != skill.skill_id]
            self._records.append(record)
            wake = self._ensure_writer()
        wake.put(True)

    def _ensure_writer(self) -> queue.Queue[bool]:
        # Caller holds self._lock.
        if self._wake is None or self._writer is None or not self._writer.is_alive():
            self._wake = queue.Queue()
            self._writer = threading.Thread(
                target=self._run, args=(self._wake,), name="skill-store", daemon=True
            )
            self._writer.start()
        return self._wake

    def _run(self, wake: queue.Queue[bool]) -> None:
        stop = False
        while not stop and wake.get():
            while True:  # coalesce a burst of deploys into one rewrite
                try:
                    stop = not wake.get_nowait()
                except queue.Empty:
                    break
                if stop:
                    break
            self._write()

    def _write(self) -> None:
        with self._io:
            with self._lock:
                skills = [*self._kept, *self._records]
                fresh = bool(self._records)
            if not fresh:  # nothing deployed since start or reset: leave the file alone
                return
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                if self._unreadable and self.path.exists():
                    aside = self.path.with_name(f"{self.path.stem}-{_stamp()}.json.corrupt")
                    os.replace(self.path, aside)
                    logger.warning("skills: kept the unreadable library as %s", aside)
                self._unreadable = False
                tmp = self.path.with_suffix(".json.tmp")
                tmp.write_text(
                    json.dumps({"version": FORMAT_VERSION, "skills": skills}), encoding="utf-8"
                )
                os.replace(tmp, self.path)
            except OSError:
                logger.exception("skills: could not save %s", self.path)

    def close(self) -> None:
        """Flush pending writes and stop the writer. A later ``on_deployed`` restarts it."""
        with self._lock:
            wake, writer = self._wake, self._writer
            self._wake = self._writer = None
        if wake is not None and writer is not None:
            wake.put(False)
            writer.join()

    # ------------------------------------------------------------------ reset + inspection

    def reset(self, coordinator: AtcCoordinator) -> ResetReport:
        """Back the file up, clear the library and the board. Skills mid-replay run to the end."""
        self._coordinator = coordinator
        board_cleared = coordinator.reset_library()  # first: the board and library never outlive it
        with self._io:
            with self._lock:
                self._records = []
                self._kept = []
            backup: Path | None = None
            try:
                if self.path.exists():
                    backup = self.path.with_name(f"{self.path.stem}-{_stamp()}.json.bak")
                    n = 1
                    while backup.exists():
                        backup = backup.with_name(f"{self.path.stem}-{_stamp()}-{n}.json.bak")
                        n += 1
                    os.replace(self.path, backup)
            except OSError:
                logger.exception("skills: could not back up %s", self.path)
                backup = None
            self._unreadable = False
        return ResetReport(board_cleared=board_cleared, backup=backup)

    def entries(self) -> list[SkillEntry]:
        with self._lock:
            return [
                SkillEntry(
                    r["skill"]["skill_id"],
                    r["skill"]["signature_id"],
                    r["learned_at"],
                    r["session"],
                )
                for r in self._records
            ]
