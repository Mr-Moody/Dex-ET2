"""``uv run python -m teleop.copilot run``: live spoken call-outs for the fleet ops console.

Wires the console feed, frame store and event detector (``T-211``) to the call-out service and
the speaker, and prints and logs every call-out to ``runs/copilot/<date>.jsonl``. Start the demo
first (``uv run teleop atc --source synthetic``), then run this in a second terminal.

Call-outs are spoken with the system voice straight away. ``--gemini-tts`` tries Gemini TTS first
(the system voice takes over after 1.5 s); live it was slower than that on every line, so it is
off by default. Without ``GEMINI_API_KEY`` it still runs, with template call-outs only. Advisory
only: it never sends anything to the console.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import dataclasses
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TextIO

from teleop.copilot.callouts import CalloutService, CopilotOutputs
from teleop.copilot.config import CopilotConfig
from teleop.copilot.events import EventDetector
from teleop.copilot.feed import CameraFrame, FeedClient
from teleop.copilot.frames import FrameStore
from teleop.copilot.gemini import GeminiClient
from teleop.copilot.speech import Speaker

DEFAULT_LOG_DIR = Path("runs") / "copilot"


async def run(
    cfg: CopilotConfig,
    outputs: CopilotOutputs,
    client: GeminiClient | None,
    speaker: Speaker,
) -> None:
    store = FrameStore()
    detector = EventDetector()
    service = CalloutService(
        client,
        model=cfg.fast_model,
        thinking_level=cfg.fast_thinking,
        sink=outputs.callout,
        speaker=speaker,
        frame_for=store.latest,
        no_images=cfg.no_images,
    )

    def on_telemetry(msg: dict[str, Any]) -> None:
        store.on_telemetry(msg)
        for event in detector.feed(msg):
            outputs.event(event, store.latest(event.bay))
            service.handle(event)

    def on_frame(frame: CameraFrame) -> None:
        store.add(frame)

    feed = FeedClient(cfg.console_url, on_telemetry, on_frame)
    try:
        await feed.run()
    finally:
        feed.stop()
        service.close()


def run_main(argv: Sequence[str] | None = None, out: TextIO | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m teleop.copilot run", description=__doc__)
    parser.add_argument("--console", help="console WebSocket URL (default COPILOT_CONSOLE_URL)")
    parser.add_argument("--no-images", action="store_true", help="send text only, no frames")
    parser.add_argument("--mute", action="store_true", help="print and log, but do not speak")
    parser.add_argument(
        "--gemini-tts",
        action="store_true",
        help="speak with Gemini TTS (system voice after 1.5 s); default: system voice at once",
    )
    parser.add_argument("--env-file", help="read this .env instead of discovering one")
    parser.add_argument(
        "--log-dir", default=str(DEFAULT_LOG_DIR), help="JSONL log directory (default runs/copilot)"
    )
    args = parser.parse_args(argv)
    stdout = out if out is not None else sys.stdout
    try:
        cfg = CopilotConfig.from_env(
            env_file=args.env_file, no_images=args.no_images, mute=args.mute
        )
    except ValueError as exc:
        print(f"copilot: {exc}", file=sys.stderr)
        return 2
    url = args.console or cfg.console_url
    cfg = dataclasses.replace(cfg, console_url=url)
    client: GeminiClient | None = None
    if cfg.gemini_api_key:
        client = GeminiClient(cfg.gemini_api_key, no_images=cfg.no_images)
    else:
        print(
            "copilot: GEMINI_API_KEY is not set; speaking template call-outs with the system "
            "voice only (see docs/copilot.md).",
            file=sys.stderr,
        )
    outputs = CopilotOutputs(stdout, Path(args.log_dir))
    speaker = Speaker(
        client if args.gemini_tts else None,
        tts_model=cfg.tts_model,
        voice=cfg.voice,
        mute=cfg.mute,
        on_spoken=outputs.spoken,
    )
    outputs.note(
        f"copilot watching {url}: call-outs {cfg.fast_model if client else 'templates only'}, "
        f"speech {_speech_label(cfg, client is not None and args.gemini_tts)}, "
        f"images {'off' if cfg.no_images else 'on'}, log {outputs.log_path()} (Ctrl+C to stop)"
    )
    try:
        with contextlib.suppress(KeyboardInterrupt):
            asyncio.run(run(cfg, outputs, client, speaker))
    finally:
        speaker.close()
    return 0


def _speech_label(cfg: CopilotConfig, gemini_tts: bool) -> str:
    if cfg.mute:
        return "muted"
    return f"{cfg.tts_model}/{cfg.voice}" if gemini_tts else "system voice"
