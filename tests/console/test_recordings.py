"""Data page recording desk: fake sinks and a temp folder, no lerobot, no MuJoCo."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from teleop.console.recordings import SIDECAR, RecordingDesk, scan


class FakeSink:
    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self.closed = False
        self.written = 0

    def add(self, **meta: Any) -> None:
        """Write one sidecar line, as ``EpisodeRecorder.end_episode`` would."""
        path = self.folder / SIDECAR
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"episode_index": self.written, **meta}) + "\n")
        self.written += 1

    def close(self) -> None:
        self.closed = True


class Harness:
    def __init__(self, root: Path, *, unavailable: str | None = None) -> None:
        self.sinks: list[FakeSink] = []
        self.observer: Any = "unset"
        self.ticks = iter(datetime(2026, 11, 15, 10, 0, s) for s in range(60))
        self.desk = RecordingDesk(
            root,
            self._open,
            self._set_observer,
            unavailable=unavailable,
            clock=lambda: next(self.ticks),
            scan_interval_s=0.05,
        )

    def _open(self, folder: Path) -> FakeSink:
        sink = FakeSink(folder)
        self.sinks.append(sink)
        return sink

    def _set_observer(self, obs: Any) -> None:
        self.observer = obs


def _wait(cond: Any, timeout: float = 3.0) -> None:
    end = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < end, "timed out"
        time.sleep(0.01)


def test_start_hooks_sink_only_on_pump(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    assert h.desk.start_recording()
    assert h.observer == "unset"  # the control thread swaps it, not the WebSocket thread
    h.desk.pump()
    assert h.observer is h.sinks[0]
    st = h.desk.status()
    assert st["recording"]["id"] == "20261115-100000"
    assert st["recordings"][0]["state"] == "recording"
    assert not h.desk.start_recording()  # one at a time


def test_stop_unhooks_then_finalises_off_thread(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    h.desk.start_recording()
    h.desk.pump()
    h.sinks[0].add(bay=2, source="glove", success=True, recorded_at="2026-11-15T10:00:05")
    assert h.desk.stop_recording()
    assert not h.sinks[0].closed  # nothing closes until the observer is off the coordinator
    h.desk.pump()
    assert h.observer is None
    _wait(lambda: h.sinks[0].closed and h.desk.status()["saving"] == 0)
    rows = h.desk.status()["recordings"]
    assert [r["state"] for r in rows] == ["saved"]
    assert rows[0]["episodes"][0]["bay"] == 2 and rows[0]["episodes"][0]["success"]


def test_stop_then_start_before_pump(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    h.desk.start_recording()
    h.desk.pump()
    h.desk.stop_recording()
    h.desk.start_recording()
    h.desk.pump()
    assert h.observer is h.sinks[1]
    _wait(lambda: h.sinks[0].closed)
    assert not h.sinks[1].closed


def test_delete_saved_recording(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    h.desk.start_recording()
    h.desk.pump()
    h.sinks[0].add(bay=1, source="dmp", success=False)
    h.desk.stop_recording()
    h.desk.pump()
    _wait(lambda: h.desk.status()["saving"] == 0)
    rid = h.desk.status()["recordings"][0]["id"]
    h.desk.handle({"type": "record", "action": "delete", "id": rid})
    _wait(lambda: not (tmp_path / rid).exists())
    _wait(lambda: h.desk.status()["recordings"] == [])


def test_delete_refuses_active_and_bad_ids(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    (outside / SIDECAR).parent.mkdir(parents=True)
    (outside / SIDECAR).write_text("{}\n", encoding="utf-8")
    h.desk.start_recording()
    h.desk.pump()
    h.sinks[0].add(bay=1, source="glove", success=True)
    active = h.sinks[0].folder.name
    assert not h.desk.delete(active)
    assert "Stop the recording" in h.desk.status()["note"]
    for bad in ("..", f"../{outside.name}", "a/b", "", "missing"):
        assert not h.desk.delete(bad)
    assert (tmp_path / active).exists() and outside.exists()


def test_unavailable_never_starts(tmp_path: Path) -> None:
    h = Harness(tmp_path, unavailable="Recording needs lerobot")
    h.desk.handle({"type": "record", "action": "start"})
    st = h.desk.status()
    assert not st["available"] and st["recording"] is None and h.sinks == []


def test_close_finalises_an_active_recording(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    h.desk.start()
    h.desk.start_recording()
    h.desk.pump()
    h.desk.close()
    assert h.sinks[0].closed and h.observer is None


def test_scan_lists_newest_first_and_skips_junk(tmp_path: Path) -> None:
    for name, lines in (
        ("20261114-090000", ['{"bay": 1, "success": true}']),
        ("20261115-090000", ['{"bay": 2, "success": false}', '{"bay": 3, "succ']),
    ):
        p = tmp_path / name / SIDECAR
        p.parent.mkdir(parents=True)
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (tmp_path / "no-sidecar").mkdir()
    (tmp_path / "stray.txt").write_text("x", encoding="utf-8")
    rows = scan(tmp_path)
    assert [r["id"] for r in rows] == ["20261115-090000", "20261114-090000"]
    assert [e["bay"] for e in rows[0]["episodes"]] == [2]  # half-written line skipped
    assert rows[0]["started_at"] == "2026-11-15T09:00:00"
    assert scan(tmp_path / "missing") == []
