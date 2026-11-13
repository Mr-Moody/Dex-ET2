"""Replay a raw capture (JSON Lines, one datagram per line) as an input source."""

from __future__ import annotations

import base64
import importlib
import json
import logging
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

from teleop.input.buffer import ThreadedSource
from teleop.protocols import PacketParser

logger = logging.getLogger(__name__)
_LOOP_GAP_S = 1.0 / 60.0  # nominal frame period between loop passes


class ReplaySource(ThreadedSource):
    def __init__(
        self,
        path: Path,
        parser: PacketParser | None = None,
        speed: float = 1.0,
        loop: bool = True,
    ) -> None:
        super().__init__("replay")
        self._path = Path(path)
        if not self._path.is_file():
            raise FileNotFoundError(f"capture file not found: {self._path}")
        self._parser = parser
        self._speed = speed if speed > 0 else 1.0
        self._loop = loop
        if parser is None:
            self._parser = self._default_parser_if_needed()

    def _default_parser_if_needed(self) -> PacketParser | None:
        """Validate capture sources up front; build StudioParser lazily if all are studio."""
        srcs: set[str] = set()
        for rec in self._records():
            srcs.add(str(rec.get("src", "")))
        bad = sorted(srcs - {"studio"})
        if bad:
            raise ValueError(f"no parser for capture source(s) {bad}; pass a parser explicitly")
        if not srcs:
            return None
        # lazy: StudioParser lives in a sibling module
        studio = importlib.import_module("teleop.input.studio")
        return cast("PacketParser", studio.StudioParser())

    def _records(self) -> Iterator[dict[str, Any]]:
        """Yield well-formed records; malformed lines are skipped (see ``_run`` for counting)."""
        with self._path.open("r", encoding="utf-8") as fh:
            for line in fh:
                rec = self._decode(line)
                if rec is not None:
                    yield rec

    @staticmethod
    def _decode(line: str) -> dict[str, Any] | None:
        line = line.strip()
        if not line:
            return None
        try:
            rec = json.loads(line)
            if not isinstance(rec, dict):
                return None
            rec = cast("dict[str, Any]", rec)
            int(rec["t_ns"])
            base64.b64decode(rec["payload_b64"], validate=True)
        except (ValueError, KeyError, TypeError):
            return None
        return rec

    def _run(self, stop: threading.Event) -> None:
        parser = self._parser
        if parser is None:
            return
        warned = False
        while not stop.is_set():
            prev_t: int | None = None
            skipped = 0
            with self._path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if stop.is_set():
                        return
                    if not line.strip():
                        continue
                    rec = self._decode(line)
                    if rec is None:
                        skipped += 1
                        continue
                    t_ns = int(rec["t_ns"])
                    delay = 0.0 if prev_t is None else max(t_ns - prev_t, 0) / 1e9 / self._speed
                    if delay > 0 and stop.wait(delay):
                        return
                    prev_t = t_ns
                    payload = base64.b64decode(rec["payload_b64"])
                    for frame in parser.parse(payload, time.monotonic_ns()):
                        self._buffer.put(frame)
            if skipped and not warned:
                warned = True
                logger.warning("replay %s: skipped %d malformed line(s)", self._path, skipped)
            if not self._loop:
                return
            if stop.wait(_LOOP_GAP_S / self._speed):  # bound rate on tiny/zero-delta files
                return
