"""``uv run python -m teleop.copilot probe``: a live check of the Gemini key, models and latency.

The probe lists the models the key can use, then times one call per copilot role:

- a call-out (telemetry plus one camera JPEG -> JSON) on the fast model,
- a grasp review (telemetry plus three JPEGs -> JSON) on the review model,
- one speech synthesis (text -> WAV) on the TTS model.

It prints a Markdown report to paste into ``docs/copilot.md``, plus the env values it
recommends. It never prints the API key. It is a manual command: ``pytest`` never runs it live.
"""

from __future__ import annotations

import argparse
import dataclasses
import io
import json
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TextIO

from teleop.copilot.config import CopilotConfig
from teleop.copilot.gemini import (
    THINKING_LEVELS,
    GeminiClient,
    GeminiError,
    ImagePart,
    ModelInfo,
    Part,
    TextPart,
    ThinkingLevel,
    wav_duration_ms,
)

PROBE_MODELS = (
    "gemini-4-hydrogen",
    "gemini-4-argon",
    "gemini-3.8-flash",
    "gemini-3.8-flash-lite-tts",
    "gemini-3.8-flash-tts",
    "gemini-3.1-flash-tts-preview",
)
FAST_COMPARISON_MODEL = "gemini-3.8-flash"
TTS_FALLBACKS = ("gemini-3.8-flash-tts", "gemini-3.1-flash-tts-preview")

CALLOUT_TIMEOUT_S = 8.0
CALLOUT_TARGET_MS = 3_000.0
REVIEW_TIMEOUT_S = 90.0
TTS_TIMEOUT_S = 30.0  # generous on purpose: the probe measures; the 4 s cut-off is T-212's
TTS_TARGET_MS = 4_000.0
LIST_TIMEOUT_S = 10.0

CALLOUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "say": {"type": "string", "description": "Spoken line, at most 15 words."},
        "detail": {"type": "string", "description": "At most 40 words."},
        "severity": {"type": "string", "enum": ["info", "attention", "alarm"]},
    },
    "required": ["say", "detail", "severity"],
    "additionalProperties": False,
}

REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["approve", "flag", "insufficient_evidence"]},
        "reasons": {"type": "array", "items": {"type": "string"}},
        "operator_note": {"type": "string", "description": "At most 60 words."},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["verdict", "reasons", "operator_note", "confidence"],
    "additionalProperties": False,
}

PROBE_SYSTEM = (
    "You assist the human operator of a robot work cell. Each bay has a robot arm that opens a "
    "sprung latch: a stopper flap must be held down while a bolt slides at least 5.2 cm clear. A "
    "bay stalls when its contact confidence C(t) stays below 0.3. Use only the telemetry and "
    "images given. Say 'unclear' rather than guess. Never suggest bypassing a safety stop or "
    "changing settings. This is a connectivity probe: the images are synthetic test patterns."
)

PROBE_TELEMETRY: dict[str, Any] = {
    "event": "stall",
    "bay": {
        "id": 1,
        "state": "STALLED",
        "confidence": 0.21,
        "history": [0.93, 0.88, 0.52, 0.27, 0.21],
        "x_bolt_cm": 1.4,
        "fault": {"kind": "flap", "phase": "armed"},
    },
    "bolt_clear_cm": 5.2,
    "threshold": 0.3,
}

TTS_LINE = "Bay one stalled on a flap jam. Rescue needed."

Clock = Callable[[], int]


@dataclass(frozen=True, slots=True)
class ProbeResult:
    role: str
    model: str
    ok: bool
    ms: float | None
    note: str


def sample_jpeg(seed: int = 0, size: int = 64) -> bytes:
    """A small synthetic JPEG (no real camera frame is ever sent by the probe)."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (size, size), (40, 44, 52))
    draw = ImageDraw.Draw(img)
    off = (seed * 9) % (size // 2)
    draw.rectangle((8 + off, 20, 8 + off + size // 3, 44), fill=(230, 120, 30))
    draw.line((0, size - 10, size, size - 10), fill=(200, 200, 200), width=2)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return buf.getvalue()


def _timed(clock: Clock, fn: Callable[[], Any]) -> tuple[Any, float]:
    t0 = clock()
    value = fn()
    return value, (clock() - t0) / 1e6


def _missing(answer: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    required: list[str] = schema["required"]
    return [k for k in required if k not in answer]


def _json_call(
    client: GeminiClient,
    clock: Clock,
    role: str,
    model: str,
    parts: Sequence[Part],
    schema: dict[str, Any],
    timeout_s: float,
    thinking: ThinkingLevel,
    target_ms: float | None = None,
) -> ProbeResult:
    try:
        answer, ms = _timed(
            clock,
            lambda: client.generate_json(
                model, PROBE_SYSTEM, parts, schema, timeout_s, thinking_level=thinking
            ),
        )
    except (GeminiError, ValueError) as exc:
        return ProbeResult(role, model, False, None, f"thinking {thinking}: {exc}")
    missing = _missing(answer, schema)
    note = f"thinking {thinking}; answer {json.dumps(answer, ensure_ascii=False)}"
    if missing:
        note += f"; missing fields {missing}"
    if target_ms is not None and ms > target_ms:
        note += f"; slower than the {target_ms:.0f} ms target"
    return ProbeResult(role, model, not missing, ms, note)


def _tts_call(client: GeminiClient, clock: Clock, model: str, voice: str) -> ProbeResult:
    try:
        wav, ms = _timed(clock, lambda: client.synthesize(TTS_LINE, model, voice, TTS_TIMEOUT_S))
        duration = wav_duration_ms(wav)
    except (GeminiError, ValueError, EOFError) as exc:
        return ProbeResult("speech", model, False, None, f"voice {voice}: {exc}")
    note = f"voice {voice}; WAV {len(wav)} bytes, {duration:.0f} ms of audio"
    if ms > TTS_TARGET_MS:
        note += f"; slower than the {TTS_TARGET_MS:.0f} ms fallback cut-off"
    return ProbeResult("speech", model, True, ms, note)


def run_probe(
    cfg: CopilotConfig,
    client: GeminiClient,
    out: TextIO,
    clock: Clock = time.monotonic_ns,
) -> int:
    """Run every check, print the report, return 0 when all three roles work."""
    images = "off (text only)" if cfg.no_images else "on (synthetic 64x64 JPEGs)"
    out.write("## Probe results\n\n")
    out.write(f"Images: {images}. Times are wall-clock ms from `time.monotonic_ns()`.\n\n")

    out.write("### Models the key can use\n\n")
    listed: dict[str, ModelInfo] = {}
    try:
        models, list_ms = _timed(clock, lambda: client.list_models(LIST_TIMEOUT_S))
        listed = {m.name: m for m in models}
        generating = sorted(m.name for m in models if m.can_generate)
        out.write(
            f"`GET /v1beta/models` returned {len(models)} models in {list_ms:.0f} ms; "
            f"{len(generating)} support `generateContent`.\n\n"
        )
    except GeminiError as exc:
        out.write(f"Listing failed: {exc}\n\n")
    wanted = dict.fromkeys(
        (*PROBE_MODELS, cfg.fast_model, cfg.review_model, cfg.tts_model)
    )  # ordered, no duplicates
    out.write("| Model | Listed | generateContent |\n|---|---|---|\n")
    for name in wanted:
        info = listed.get(name)
        out.write(
            f"| `{name}` | {'yes' if info else 'no'} | "
            f"{'yes' if info and info.can_generate else 'no'} |\n"
        )
    out.write("\n")

    frames: list[Part] = [ImagePart(sample_jpeg(i)) for i in range(3)]
    telemetry = TextPart("Telemetry:\n" + json.dumps(PROBE_TELEMETRY, indent=1))
    callout_parts: list[Part] = [telemetry, frames[0]]
    review_parts: list[Part] = [
        TextPart("Review this rescue. Frames: at the stall, during the clutch hold, after."),
        telemetry,
        *frames,
    ]

    results: list[ProbeResult] = [
        _json_call(
            client,
            clock,
            "call-out",
            cfg.fast_model,
            callout_parts,
            CALLOUT_SCHEMA,
            CALLOUT_TIMEOUT_S,
            cfg.fast_thinking,
            CALLOUT_TARGET_MS,
        )
    ]
    if cfg.fast_model != FAST_COMPARISON_MODEL and (not listed or FAST_COMPARISON_MODEL in listed):
        results.append(
            _json_call(
                client,
                clock,
                "call-out (comparison)",
                FAST_COMPARISON_MODEL,
                callout_parts,
                CALLOUT_SCHEMA,
                CALLOUT_TIMEOUT_S,
                cfg.fast_thinking,
                CALLOUT_TARGET_MS,
            )
        )
    review = _json_call(
        client,
        clock,
        "review",
        cfg.review_model,
        review_parts,
        REVIEW_SCHEMA,
        REVIEW_TIMEOUT_S,
        cfg.review_thinking,
    )
    results.append(review)
    tts = _tts_call(client, clock, cfg.tts_model, cfg.voice)
    results.append(tts)
    if not tts.ok:
        for alt in TTS_FALLBACKS:
            if alt != cfg.tts_model and (not listed or alt in listed):
                alt_result = _tts_call(client, clock, alt, cfg.voice)
                results.append(dataclasses.replace(alt_result, role="speech (fallback)"))
                if alt_result.ok:
                    break

    out.write("### Timed calls\n\n| Role | Model | Result | ms | Notes |\n|---|---|---|---|---|\n")
    for r in results:
        ms = f"{r.ms:.0f}" if r.ms is not None else "-"
        note = r.note.replace("|", "\\|").replace("\n", " ")
        out.write(f"| {r.role} | `{r.model}` | {'ok' if r.ok else 'FAILED'} | {ms} | {note} |\n")
    out.write("\n")

    out.write("### Recommended `.env` values\n\n```dotenv\n")
    for line in _recommend(cfg, results):
        out.write(line + "\n")
    out.write("```\n")
    ok = {r.role.split(" ")[0] for r in results if r.ok}
    return 0 if {"call-out", "review", "speech"} <= ok else 1


def _recommend(cfg: CopilotConfig, results: Sequence[ProbeResult]) -> list[str]:
    def first_ok(*roles: str) -> ProbeResult | None:
        return next((r for r in results if r.role in roles and r.ok), None)

    fast = first_ok("call-out")
    if fast is None:
        fast = first_ok("call-out (comparison)")
    speech = first_ok("speech", "speech (fallback)")
    review = first_ok("review")
    lines: list[str] = []
    if fast is not None:
        lines.append(f"COPILOT_FAST_MODEL={fast.model}")
    else:
        lines.append(f"# COPILOT_FAST_MODEL: no call-out model worked (tried {cfg.fast_model})")
    if review is not None:
        lines.append(f"COPILOT_REVIEW_MODEL={review.model}")
    elif fast is not None:
        lines.append(
            f"COPILOT_REVIEW_MODEL={fast.model}  # {cfg.review_model} failed; "
            "use the fast model as the reviewer"
        )
    else:
        lines.append(f"# COPILOT_REVIEW_MODEL: {cfg.review_model} failed")
    if speech is not None:
        lines.append(f"COPILOT_TTS_MODEL={speech.model}")
        lines.append(f"COPILOT_VOICE={cfg.voice}")
    else:
        lines.append(f"# COPILOT_TTS_MODEL: no TTS model worked (tried {cfg.tts_model})")
    return lines


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m teleop.copilot", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    probe = sub.add_parser("probe", help="live check of the Gemini key, models and latency")
    probe.add_argument("--no-images", action="store_true", help="send text only, no JPEGs")
    probe.add_argument("--fast-model", help="override COPILOT_FAST_MODEL")
    probe.add_argument("--review-model", help="override COPILOT_REVIEW_MODEL")
    probe.add_argument("--tts-model", help="override COPILOT_TTS_MODEL")
    probe.add_argument("--voice", help="override COPILOT_VOICE")
    for role in ("fast", "review"):
        probe.add_argument(
            f"--{role}-thinking",
            choices=THINKING_LEVELS,
            help=f"override COPILOT_{role.upper()}_THINKING",
        )
    probe.add_argument("--env-file", help="read this .env instead of discovering one")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    client_factory: Callable[[CopilotConfig], GeminiClient] | None = None,
    out: TextIO | None = None,
) -> int:
    args = _parser().parse_args(argv)
    stdout = out if out is not None else sys.stdout
    try:
        cfg = CopilotConfig.from_env(env_file=args.env_file, no_images=args.no_images)
    except ValueError as exc:
        print(f"copilot: {exc}", file=sys.stderr)
        return 2
    overrides = {
        k: v
        for k, v in {
            "fast_model": args.fast_model,
            "review_model": args.review_model,
            "tts_model": args.tts_model,
            "voice": args.voice,
            "fast_thinking": args.fast_thinking,
            "review_thinking": args.review_thinking,
        }.items()
        if v
    }
    cfg = dataclasses.replace(cfg, **overrides)
    if not cfg.has_key:
        print(
            "copilot: GEMINI_API_KEY is not set. Add it to the repo's .env (see docs/copilot.md).",
            file=sys.stderr,
        )
        return 2
    if client_factory is not None:
        client = client_factory(cfg)
    else:
        assert cfg.gemini_api_key is not None
        client = GeminiClient(cfg.gemini_api_key, no_images=cfg.no_images)
    return run_probe(cfg, client, stdout)


if __name__ == "__main__":
    if sys.argv[1:2] == ["watch"]:  # T-211: live event printout (teleop.copilot.watch)
        from teleop.copilot.watch import watch_main

        raise SystemExit(watch_main(sys.argv[2:]))
    raise SystemExit(main())
