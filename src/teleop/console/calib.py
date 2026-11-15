"""In-console glove calibration: flat -> fist capture that hot-swaps the live retargeter.

The browser sends ``{"type": "calib", "action": "start" | "cancel"}``; the wizard runs on its own
thread (never the 60 Hz control loop): prep, hold the flat hand, prep, hold a fist, then
``calibrate()`` the frames, hand the result to ``apply`` (which swaps the running retargeter in
one reference assignment) and best-effort persist it as JSON. ``status_dict`` feeds the
``calib`` field of the console telemetry.

Timing mirrors ``teleop calibrate`` (2 s prep + 2 s hold per pose). ``begin`` / ``step`` take an
explicit ``now`` so tests drive the wizard with a fake clock and no thread.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Collection
from pathlib import Path
from typing import Any, Literal

from teleop.protocols import InputSource
from teleop.retarget.calibration import Calibration, calibrate, take_new_frames
from teleop.sim.hands import HandSpec
from teleop.types import HandFrame, Side

logger = logging.getLogger("teleop.console.calib")

PREP_S = 2.0
HOLD_S = 2.0
POLL_HZ = 120.0

Phase = Literal["idle", "prep_flat", "hold_flat", "prep_fist", "hold_fist"]
Result = Literal["applied", "partial", "cancelled", "no_frames", "busy", "error"]
# ``apply`` installs a calibration and returns the actuators it had to fall back to defaults on.
ApplyFn = Callable[[Calibration], Collection[tuple[Side, str]]]

_NEXT: dict[Phase, Phase] = {
    "prep_flat": "hold_flat",
    "hold_flat": "prep_fist",
    "prep_fist": "hold_fist",
}
_RESULT_TEXT: dict[Result, str] = {
    "applied": "Applied",
    "partial": "Applied (partial)",
    "cancelled": "Cancelled",
    "no_frames": "No glove frames",
    "busy": "Already calibrating",
    "error": "Calibration failed",
}


class GloveCalibrator:
    def __init__(
        self,
        spec: HandSpec,
        source: InputSource,
        apply: ApplyFn,
        *,
        persist_path: Path | None = None,
        loaded_from: Path | None = None,
        side: Side = "right",
        prep_s: float = PREP_S,
        hold_s: float = HOLD_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.spec = spec
        self.source = source
        self.apply = apply
        self.persist_path = persist_path
        self.side: Side = side
        self.prep_s = prep_s
        self.hold_s = hold_s
        self.clock = clock
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._phase: Phase = "idle"
        self._phase_end = 0.0
        self._remaining = 0.0
        self._seen: set[tuple[Side, int, int]] = set()
        self._flat: list[HandFrame] = []
        self._fist: list[HandFrame] = []
        self._result: Result | None = None
        self._detail = ""
        self._seq = 0  # bumps on every result, so the page toasts each one once
        self._origin: Literal["defaults", "file", "live"] = (
            "file" if loaded_from is not None else "defaults"
        )
        self._path: str | None = str(loaded_from) if loaded_from is not None else None
        self._last_done = (False, False)  # (flat, fist) tiles of the last success

    # ------------------------------------------------------------------ browser messages

    def handle_message(self, msg: dict[str, Any]) -> None:
        action = msg.get("action")
        if action == "start":
            self.request_start()
        elif action == "cancel":
            self.cancel()

    def request_start(self) -> bool:
        """Begin a capture on a worker thread; ``False`` (and a ``busy`` result) if running."""
        if not self.begin(self.clock()):
            return False
        self._thread = threading.Thread(target=self._run, name="console-calib", daemon=True)
        self._thread.start()
        return True

    def cancel(self) -> None:
        with self._lock:
            if self._phase != "idle":
                self._finish("cancelled", "")

    def _run(self) -> None:
        while self.step(self.clock()):
            time.sleep(1.0 / POLL_HZ)

    # ------------------------------------------------------------------ state machine

    def begin(self, now: float) -> bool:
        with self._lock:
            if self._phase != "idle":
                self._seq += 1
                self._result, self._detail = "busy", ""
                return False
            self._flat, self._fist = [], []
            self._seen = set()
            self._result, self._detail = None, ""
            self._enter("prep_flat", now)
            return True

    def step(self, now: float) -> bool:
        """Advance the wizard; ``True`` while it is still running."""
        with self._lock:
            if self._phase == "idle":
                return False
            if self._phase in ("hold_flat", "hold_fist"):
                out = self._flat if self._phase == "hold_flat" else self._fist
                take_new_frames(self.source.latest(), self._seen, out)
            self._remaining = max(0.0, self._phase_end - now)
            if now < self._phase_end:
                return True
            if self._phase != "hold_fist":
                self._enter(_NEXT[self._phase], now)
                return True
        self._complete()  # outside the lock: apply may take the control thread's lock
        return False

    def _enter(self, phase: Phase, now: float) -> None:
        self._phase = phase
        self._phase_end = now + (self.prep_s if phase.startswith("prep") else self.hold_s)
        self._remaining = self._phase_end - now

    def _complete(self) -> None:
        with self._lock:
            if self._phase != "hold_fist":
                return  # cancelled between the last step and here
            flat, fist = list(self._flat), list(self._fist)
        n_flat = sum(f.side == self.side for f in flat)
        n_fist = sum(f.side == self.side for f in fist)
        if n_flat == 0 or n_fist == 0:
            with self._lock:
                self._finish("no_frames", f"flat {n_flat}, fist {n_fist} {self.side} frames")
            return
        try:
            calib = calibrate(self.spec, flat, fist)
            fallback = sorted(a for s, a in self.apply(calib) if s == self.side)
        except Exception as exc:
            logger.exception("console calib: apply failed")
            with self._lock:
                self._finish("error", str(exc))
            return
        detail = f"defaults kept for {', '.join(fallback)}" if fallback else ""
        saved = self._persist(calib)
        if saved is not None:
            detail = f"{detail}; {saved}" if detail else saved
        with self._lock:  # a cancel that raced the apply is overridden: the swap happened
            self._origin = "live"
            self._path = str(self.persist_path) if self.persist_path is not None else None
            self._last_done = (True, True)
            self._finish("partial" if fallback else "applied", detail)

    def _persist(self, calib: Calibration) -> str | None:
        """Write the JSON; a failure is reported but never undoes the live apply."""
        if self.persist_path is None:
            return None
        try:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            calib.to_json(self.persist_path)
        except OSError as exc:
            logger.warning("console calib: could not save %s (%s)", self.persist_path, exc)
            return f"not saved ({exc.strerror or exc})"
        return None

    def _finish(self, result: Result, detail: str) -> None:
        self._phase = "idle"
        self._remaining = 0.0
        self._result = result
        self._detail = detail
        self._seq += 1
        logger.info("console calib: %s %s", result, detail)

    # ------------------------------------------------------------------ telemetry

    def status_dict(self) -> dict[str, Any]:
        with self._lock:
            phase = self._phase
            running = phase != "idle"
            flat_done = phase in ("prep_fist", "hold_fist") or (not running and self._last_done[0])
            fist_done = not running and self._last_done[1]
            return {
                "state": "idle" if not running else phase.split("_")[0],
                "pose": None if not running else phase.split("_")[1],
                "remaining_s": round(self._remaining, 1),
                "flat_done": flat_done,
                "fist_done": fist_done,
                "frames": {
                    "flat": sum(f.side == self.side for f in self._flat),
                    "fist": sum(f.side == self.side for f in self._fist),
                },
                "origin": self._origin,
                "path": self._path,
                "result": self._result,
                "result_text": _RESULT_TEXT[self._result] if self._result else None,
                "detail": self._detail,
                "seq": self._seq,
            }
