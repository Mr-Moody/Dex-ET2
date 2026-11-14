"""Operator copilot: an advisory intermediary that watches the fleet ops console.

It runs as its own process on the operator's computer, connects to the console as one more
viewer, and helps the human with spoken call-outs and second-opinion reviews. It is **advisory
only**: nothing in this package sends key presses or any other control message to the console,
and nothing imports :mod:`teleop.atc.coordinator`. The human and the existing gates decide.

Start it with ``uv run python -m teleop.copilot …`` (see ``docs/copilot.md``).
"""

from teleop.copilot.config import CopilotConfig
from teleop.copilot.gemini import (
    GeminiClient,
    GeminiError,
    GeminiHttpError,
    GeminiMalformedJsonError,
    GeminiTimeoutError,
    ImagePart,
    ModelInfo,
    TextPart,
    ThinkingLevel,
    Transport,
)

__all__ = [
    "CopilotConfig",
    "GeminiClient",
    "GeminiError",
    "GeminiHttpError",
    "GeminiMalformedJsonError",
    "GeminiTimeoutError",
    "ImagePart",
    "ModelInfo",
    "TextPart",
    "ThinkingLevel",
    "Transport",
]
