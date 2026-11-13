"""Replay a raw capture (JSON Lines, one datagram per line) as an input source."""

from __future__ import annotations

import base64
import importlib
import json
import threading
import time
from pathlib import Path
from typing import cast

from teleop.input.buffer import ThreadedSource
from teleop.protocols import PacketParser


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

    def _parser_for(self, src: str) -> PacketParser:
        if self._parser is None:
            if src != "studio":
                raise ValueError(f"no parser for capture source {src!r}; pass one explicitly")
            # lazy: StudioParser lives in a sibling module
            studio = importlib.import_module("teleop.input.studio")
            self._parser = cast("PacketParser", studio.StudioParser())
        return self._parser

    def _run(self, stop: threading.Event) -> None:
        while not stop.is_set():
            prev_t: int | None = None
            with self._path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if stop.is_set():
                        return
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    t_ns = int(rec["t_ns"])
                    delay = 0.0 if prev_t is None else max(t_ns - prev_t, 0) / 1e9 / self._speed
                    if delay > 0 and stop.wait(delay):
                        return
                    prev_t = t_ns
                    payload = base64.b64decode(rec["payload_b64"])
                    parser = self._parser_for(str(rec.get("src", "")))
                    for frame in parser.parse(payload, time.monotonic_ns()):
                        self._buffer.put(frame)
            if not self._loop:
                return
