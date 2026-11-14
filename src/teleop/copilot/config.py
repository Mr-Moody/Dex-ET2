"""Copilot settings from the environment and ``.env`` (``GEMINI_API_KEY`` and ``COPILOT_*``).

Precedence, highest first: the process environment, the ``.env`` of the current worktree, then
the ``.env`` at the main checkout's root (found with ``git rev-parse --git-common-dir``). The last
one means a key saved once in the main checkout also works inside every git worktree, so it never
has to be copied around. Passing ``env=`` reads that mapping only and never touches the disk.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from dotenv import dotenv_values

from teleop.copilot.gemini import THINKING_LEVELS, ThinkingLevel

DEFAULT_FAST_MODEL = "gemini-4-hydrogen"
DEFAULT_REVIEW_MODEL = "gemini-4-argon"
DEFAULT_TTS_MODEL = "gemini-3.8-flash-lite-tts"
DEFAULT_VOICE = "Kore"
DEFAULT_LOCAL_MODEL = "gemma4:e4b"
DEFAULT_CONSOLE_URL = "ws://127.0.0.1:8765/ws"
DEFAULT_MAX_REVIEWS_PER_HOUR = 20
# Lowest effort Gemini 4 accepts ("minimal" is HTTP 400). Measured faster; see docs/copilot.md.
DEFAULT_FAST_THINKING: ThinkingLevel = "low"
DEFAULT_REVIEW_THINKING: ThinkingLevel = "low"

ENV_VARS = (
    "GEMINI_API_KEY",
    "COPILOT_FAST_MODEL",
    "COPILOT_REVIEW_MODEL",
    "COPILOT_TTS_MODEL",
    "COPILOT_VOICE",
    "COPILOT_LOCAL_URL",
    "COPILOT_LOCAL_MODEL",
    "COPILOT_CONSOLE_URL",
    "COPILOT_MAX_REVIEWS_PER_HOUR",
    "COPILOT_FAST_THINKING",
    "COPILOT_REVIEW_THINKING",
)


@dataclass(frozen=True, slots=True)
class CopilotConfig:
    """Everything the copilot needs to start. ``repr`` never shows the API key."""

    gemini_api_key: str | None = field(default=None, repr=False)
    fast_model: str = DEFAULT_FAST_MODEL
    review_model: str = DEFAULT_REVIEW_MODEL
    tts_model: str = DEFAULT_TTS_MODEL
    voice: str = DEFAULT_VOICE
    local_url: str | None = None
    local_model: str = DEFAULT_LOCAL_MODEL
    console_url: str = DEFAULT_CONSOLE_URL
    max_reviews_per_hour: int = DEFAULT_MAX_REVIEWS_PER_HOUR
    fast_thinking: ThinkingLevel = DEFAULT_FAST_THINKING
    review_thinking: ThinkingLevel = DEFAULT_REVIEW_THINKING
    no_images: bool = False
    mute: bool = False

    def __post_init__(self) -> None:
        if self.max_reviews_per_hour < 1:
            raise ValueError(
                f"COPILOT_MAX_REVIEWS_PER_HOUR must be at least 1, got {self.max_reviews_per_hour}"
            )
        for name, level in (
            ("COPILOT_FAST_THINKING", self.fast_thinking),
            ("COPILOT_REVIEW_THINKING", self.review_thinking),
        ):
            if level not in THINKING_LEVELS:
                raise ValueError(
                    f"{name} must be one of {', '.join(THINKING_LEVELS)}, got {level!r}"
                )

    @property
    def has_key(self) -> bool:
        return bool(self.gemini_api_key)

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        *,
        env_file: str | Path | None = None,
        no_images: bool = False,
        mute: bool = False,
    ) -> CopilotConfig:
        """Build the config. ``env`` replaces the environment and disk (for tests).

        ``env_file`` names one ``.env`` to read instead of the automatic discovery.
        """
        values = dict(env) if env is not None else _layered_env(env_file)

        def get(name: str) -> str | None:
            raw = values.get(name)
            if raw is None:
                return None
            raw = raw.strip()
            return raw or None  # an empty ``KEY=`` line means unset

        max_reviews_raw = get("COPILOT_MAX_REVIEWS_PER_HOUR")
        try:
            max_reviews = int(max_reviews_raw) if max_reviews_raw else DEFAULT_MAX_REVIEWS_PER_HOUR
        except ValueError:
            raise ValueError(
                f"COPILOT_MAX_REVIEWS_PER_HOUR must be an integer, got {max_reviews_raw!r}"
            ) from None
        return cls(
            gemini_api_key=get("GEMINI_API_KEY"),
            fast_model=get("COPILOT_FAST_MODEL") or DEFAULT_FAST_MODEL,
            review_model=get("COPILOT_REVIEW_MODEL") or DEFAULT_REVIEW_MODEL,
            tts_model=get("COPILOT_TTS_MODEL") or DEFAULT_TTS_MODEL,
            voice=get("COPILOT_VOICE") or DEFAULT_VOICE,
            local_url=get("COPILOT_LOCAL_URL"),
            local_model=get("COPILOT_LOCAL_MODEL") or DEFAULT_LOCAL_MODEL,
            console_url=get("COPILOT_CONSOLE_URL") or DEFAULT_CONSOLE_URL,
            max_reviews_per_hour=max_reviews,
            fast_thinking=_thinking(get("COPILOT_FAST_THINKING"), DEFAULT_FAST_THINKING),
            review_thinking=_thinking(get("COPILOT_REVIEW_THINKING"), DEFAULT_REVIEW_THINKING),
            no_images=no_images,
            mute=mute,
        )


def _thinking(raw: str | None, default: ThinkingLevel) -> ThinkingLevel:
    if raw is None:
        return default
    level = raw.lower()
    for known in THINKING_LEVELS:
        if level == known:
            return known
    return cast(ThinkingLevel, level)  # rejected with a clear message in __post_init__


def _layered_env(env_file: str | Path | None) -> dict[str, str]:
    files = [Path(env_file)] if env_file is not None else _discover_env_files()
    merged: dict[str, str] = {}
    for path in reversed(files):  # lowest precedence first, so later files win
        if path.is_file():
            merged.update({k: v for k, v in dotenv_values(path).items() if v is not None})
    merged.update({k: v for k, v in os.environ.items() if k in ENV_VARS})
    return merged


def _discover_env_files(cwd: Path | None = None) -> list[Path]:
    """``.env`` files to read, highest precedence first: this worktree, then the main checkout."""
    here = (cwd or Path.cwd()).resolve()
    files: list[Path] = []
    top = _git(here, "--show-toplevel")
    files.append((top or here) / ".env")
    common = _git(here, "--git-common-dir")
    if common is not None:
        main_root = common.parent if common.name == ".git" else None
        if main_root is not None and main_root / ".env" not in files:
            files.append(main_root / ".env")
    return files


def _git(cwd: Path, flag: str) -> Path | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", flag],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    line = out.stdout.strip()
    return Path(line) if out.returncode == 0 and line else None
