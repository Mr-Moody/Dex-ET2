"""``uv run python -m teleop.copilot run``: live spoken call-outs for the fleet ops console.

Wires the console feed, frame store and event detector (``T-211``) to the call-out service and
the speaker, and prints and logs every call-out to ``runs/copilot/<date>.jsonl``. Note that
``uv run teleop atc`` now embeds live Copilot call-outs in-process by default; use this standalone
runner when attaching to a ``--no-gemini`` or remote console.

Call-outs stream over ``GeminiStreamingTTS`` SSE (``Kore`` voice) by default when
``GEMINI_API_KEY`` is set, falling back to ``espeak-ng`` / the OS system voice within ~1.8 s or
when ``--no-gemini-tts`` is passed. Advisory only: the only thing it posts to the console is
``{"type": "copilot", ...}`` display messages, never a key or a control.
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

from teleop.copilot.callouts import Callout, CalloutService, CopilotOutputs
from teleop.copilot.config import CopilotConfig
from teleop.copilot.events import EventDetector
from teleop.copilot.feed import CameraFrame, FeedClient
from teleop.copilot.frames import FrameStore
from teleop.copilot.gemini import GeminiClient
from teleop.copilot.local import LocalModelClient
from teleop.copilot.publish import DisplayOutbox
from teleop.copilot.speech import Speaker
from teleop.copilot.tts_stream import GeminiStreamingTTS

DEFAULT_LOG_DIR = Path("runs") / "copilot"


async def run(
    cfg: CopilotConfig,
    outputs: CopilotOutputs,
    client: GeminiClient | None,
    speaker: Speaker,
    local_client: LocalModelClient | None = None,
    outbox: DisplayOutbox | None = None,
) -> None:
    store = FrameStore()
    detector = EventDetector()

    def show(callout: Callout) -> None:
        outputs.callout(callout)
        if outbox is not None:
            outbox.post_callout(callout)

    service = CalloutService(
        client,
        model=cfg.fast_model,
        local_client=local_client,
        local_model=cfg.local_model,
        thinking_level=cfg.fast_thinking,
        sink=show,
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

    feed = FeedClient(cfg.console_url, on_telemetry, on_frame, outbox=outbox)
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
        action=argparse.BooleanOptionalAction,
        default=None,
        help="stream speech with Gemini SSE TTS (default: on when GEMINI_API_KEY is set)",
    )
    parser.add_argument(
        "--no-panel", action="store_true", help="do not show call-outs in the console page"
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
    try:
        local_client = LocalModelClient.from_config(cfg)
    except ValueError as exc:
        print(f"copilot: {exc}", file=sys.stderr)
        return 2
    client: GeminiClient | None = None
    if cfg.gemini_api_key:
        client = GeminiClient(cfg.gemini_api_key, no_images=cfg.no_images)
    elif local_client is not None:
        print(
            f"copilot: GEMINI_API_KEY is not set; using local fallback {cfg.local_model} "
            "(reviews unavailable; see docs/copilot.md).",
            file=sys.stderr,
        )
    else:
        print(
            "copilot: GEMINI_API_KEY is not set; speaking template call-outs with the system "
            "voice only (see docs/copilot.md).",
            file=sys.stderr,
        )
    outputs = CopilotOutputs(stdout, Path(args.log_dir))
    outbox = None if args.no_panel else DisplayOutbox()
    use_gemini_tts = (
        bool(cfg.gemini_api_key)
        if args.gemini_tts is None
        else (args.gemini_tts and bool(cfg.gemini_api_key))
    )
    streamer: GeminiStreamingTTS | None = None
    if use_gemini_tts and cfg.gemini_api_key:
        streamer = GeminiStreamingTTS(cfg.gemini_api_key, model=cfg.tts_model, voice=cfg.voice)
    speaker = Speaker(
        streamer,
        tts_model=cfg.tts_model,
        voice=cfg.voice,
        mute=cfg.mute,
        on_spoken=outputs.spoken,
    )
    speaker.precache_cues_async()
    outputs.note(
        f"copilot watching {url}: call-outs {_callout_label(cfg, client, local_client)}, "
        f"speech {_speech_label(cfg, streamer is not None)}, "
        f"panel {'off' if outbox is None else 'on'}, "
        f"images {'off' if cfg.no_images else 'on'}, log {outputs.log_path()} (Ctrl+C to stop)"
    )
    try:
        with contextlib.suppress(KeyboardInterrupt):
            asyncio.run(run(cfg, outputs, client, speaker, local_client, outbox))
    finally:
        speaker.close()
        if streamer is not None:
            streamer.close()
    return 0


def _callout_label(
    cfg: CopilotConfig,
    client: GeminiClient | None,
    local_client: LocalModelClient | None,
) -> str:
    if client is not None and local_client is not None:
        return f"{cfg.fast_model} (fallback {cfg.local_model})"
    if client is not None:
        return cfg.fast_model
    if local_client is not None:
        return f"local {cfg.local_model}"
    return "templates only"


def _speech_label(cfg: CopilotConfig, gemini_tts: bool) -> str:
    if cfg.mute:
        return "muted"
    return f"{cfg.tts_model}/{cfg.voice}" if gemini_tts else "system voice"
