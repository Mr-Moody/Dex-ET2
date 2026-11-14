"""Fleet reasoning co-pilot hooks (Verda Nemotron track, T-161)."""

from teleop.reasoning.nemotron import VerdaNemotronPlaceholder
from teleop.reasoning.protocol import FleetReasoner, ReasoningContext, ReasoningVerdict

__all__ = ["FleetReasoner", "ReasoningContext", "ReasoningVerdict", "VerdaNemotronPlaceholder"]
