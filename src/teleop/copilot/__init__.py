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
from teleop.copilot.local import (
    REVIEW_UNAVAILABLE,
    CopilotModels,
    LocalModelClient,
    ReviewUnavailableError,
)

__all__ = [
    "REVIEW_UNAVAILABLE",
    "CopilotConfig",
    "CopilotModels",
    "GeminiClient",
    "GeminiError",
    "GeminiHttpError",
    "GeminiMalformedJsonError",
    "GeminiTimeoutError",
    "ImagePart",
    "LocalModelClient",
    "ModelInfo",
    "ReviewUnavailableError",
    "TextPart",
    "ThinkingLevel",
    "Transport",
]
