"""``uv run python -m teleop.copilot watch``: print copilot events from the live console.

A manual check of the feed, frame store and event detector (no Gemini calls, no speech). Start
the demo (``uv run teleop atc --source synthetic``), run this in a second terminal, then press
``F`` in the console to jam Bay 1. Advisory only: it never sends anything to the console.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import sys
from collections.abc import Sequence
from typing import Any, TextIO

from teleop.copilot.config import CopilotConfig
from teleop.copilot.events import CopilotEvent, EventDetector, EventKind
from teleop.copilot.feed import CameraFrame, FeedClient
from teleop.copilot.frames import FrameStore


def describe(event: CopilotEvent, store: FrameStore) -> str:
    bay: dict[str, Any] = event.telemetry.get("bay", {})
    conf = bay.get("confidence")
    bolt = bay.get("x_bolt_cm")
    bits = [f"{event.previous_state} -> {event.state}"]
    if isinstance(conf, int | float):
        bits.append(f"C={conf:.2f}")
    if isinstance(bolt, int | float):
        bits.append(f"bolt {bolt:.1f} cm")
    if event.kind is EventKind.GATE_RESULT and event.gate_ok is not None:
        gate: dict[str, Any] = event.telemetry.get("gate") or {}
        bits.append(
            f"gate {'PASS' if event.gate_ok else 'FAIL'} {gate.get('passed')}/{gate.get('total')}"
        )
    frame = store.latest(event.bay)
    bits.append(f"frame {frame.camera} {frame.sha256}" if frame else "no frame yet")
    rescue = store.rescue_frames(event.bay)
    if rescue:
        bits.append(f"{len(rescue)} rescue frames")
    return f"[{event.wall}] BAY {event.bay} {event.kind.value.upper():<20} " + ", ".join(bits)


async def watch(url: str, out: TextIO) -> None:
    store = FrameStore()
    detector = EventDetector()

    def on_telemetry(msg: dict[str, Any]) -> None:
        store.on_telemetry(msg)
        for event in detector.feed(msg):
            out.write(describe(event, store) + "\n")
            out.flush()

    def on_frame(frame: CameraFrame) -> None:
        store.add(frame)

    client = FeedClient(url, on_telemetry, on_frame)
    out.write(f"watching {url} (Ctrl+C to stop)\n")
    out.flush()
    await client.run()


def watch_main(argv: Sequence[str] | None = None, out: TextIO | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m teleop.copilot watch", description=__doc__)
    parser.add_argument("--console", help="console WebSocket URL (default COPILOT_CONSOLE_URL)")
    args = parser.parse_args(argv)
    url = args.console or CopilotConfig.from_env().console_url
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(watch(url, out if out is not None else sys.stdout))
    return 0
