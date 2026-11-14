"""Polls health checks at a fixed rate and runs reconnects off the caller thread."""

from __future__ import annotations

import dataclasses
import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

from teleop.health.model import Health, HealthCheck, Level, health_to_dict


class HealthMonitor:
    def __init__(self, checks: Sequence[HealthCheck], period_s: float = 0.5) -> None:
        self._checks = list(checks)
        self._period_s = period_s
        self._lock = threading.Lock()
        self._cache: dict[str, Health] = {}
        self._busy: set[str] = set()
        self._executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="health-reconnect")
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="health-poll", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._executor.shutdown(wait=False)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.poll_once()
            self._stop.wait(self._period_s)

    def poll_once(self) -> None:
        for check in self._checks:
            try:
                health = check.status()
            except Exception as exc:
                health = Health(
                    name=check.name, level=Level.FAIL, summary="check error", detail=str(exc)
                )
            with self._lock:
                self._cache[check.name] = health

    def snapshot(self) -> list[Health]:
        with self._lock:
            return [
                dataclasses.replace(self._cache[c.name], busy=c.name in self._busy)
                for c in self._checks
                if c.name in self._cache
            ]

    def snapshot_dicts(self) -> list[dict[str, object]]:
        return [health_to_dict(h) for h in self.snapshot()]

    def request_reconnect(self, name: str) -> bool:
        check = next((c for c in self._checks if c.name == name), None)
        if check is None:
            return False
        with self._lock:
            if name in self._busy:
                return False
            self._busy.add(name)
        try:
            self._executor.submit(self._reconnect, check)
        except RuntimeError:
            with self._lock:
                self._busy.discard(name)
            return False
        return True

    def _reconnect(self, check: HealthCheck) -> None:
        try:
            check.reconnect()
        except Exception:
            pass  # the next poll reports the real state
        finally:
            with self._lock:
                self._busy.discard(check.name)
