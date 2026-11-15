"""Shared fixtures and protocol doubles for the test suite."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterator

import pytest

from teleop.compliance.protocol import ComplianceEvent
from teleop.reasoning.protocol import ReasoningContext, ReasoningVerdict
from teleop.sim.bays import PANDA_XML
from teleop.sim.hands import HandSpec, load_hand
from teleop.unoq.client import UnoQClient
from teleop.unoq.server import UnoQServer

# Builds a tiny model and renders one offscreen frame. Run in a subprocess because a broken
# GL backend (e.g. EGL without a GPU) can kill the interpreter instead of raising.
_RENDER_PROBE = (
    "import mujoco; "
    "m=mujoco.MjModel.from_xml_string("
    "'<mujoco><worldbody><geom size=\"1\"/></worldbody></mujoco>'); "
    "r=mujoco.Renderer(m,8,8); "
    "mujoco.mj_forward(m, d:=mujoco.MjData(m)); "
    "r.update_scene(d); r.render()"
)


def models_present() -> bool:
    """Return True when the fetched Panda and Inspire MJCFs exist on disk."""
    try:
        right = load_hand("inspire").mjcf["right"]
    except FileNotFoundError:
        return False
    return PANDA_XML.is_file() and right.is_file()


needs_models = pytest.mark.skipif(
    not models_present(),
    reason="Panda/Inspire MJCFs missing; run `uv run python scripts/fetch_models.py`",
)


class RecordingComplianceSink:
    """In-memory ComplianceSink double shared across ATC tests."""

    def __init__(self) -> None:
        self.events: list[ComplianceEvent] = []

    def emit(self, event: ComplianceEvent) -> None:
        self.events.append(event)

    def close(self) -> None:
        return None

    def kinds(self) -> list[str]:
        return [e.kind for e in self.events]


class NullReasoner:
    """In-memory FleetReasoner double shared across ATC tests."""

    def __init__(self) -> None:
        self.contexts: list[ReasoningContext] = []

    def submit(self, ctx: ReasoningContext) -> None:
        self.contexts.append(ctx)

    def latest_verdict(self, bay_id: int) -> ReasoningVerdict | None:
        return None

    def close(self) -> None:
        return None


@pytest.fixture(scope="session")
def offscreen_render_ok() -> None:
    """Skip the requesting test if MuJoCo offscreen rendering does not work here."""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _RENDER_PROBE], capture_output=True, text=True, timeout=60
        )
    except subprocess.TimeoutExpired:
        pytest.skip("mujoco offscreen render probe timed out")
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-1:] or ["no output"]
        pytest.skip(f"mujoco offscreen rendering unavailable (rc={proc.returncode}): {tail[0]}")


@pytest.fixture(scope="session")
def inspire_spec() -> HandSpec:
    if not models_present():
        pytest.skip("Panda/Inspire MJCFs missing; run `uv run python scripts/fetch_models.py`")
    return load_hand("inspire")


@pytest.fixture
def unoq_client() -> Iterator[UnoQClient]:
    """Start a fresh local loopback UnoQServer + connected UnoQClient on ephemeral ports."""
    server = UnoQServer(host="127.0.0.1", udp_port=0, tcp_port=0, name="test-node").start()
    client = UnoQClient(
        "127.0.0.1", server.udp_port, server.tcp_port, local_server=server
    ).connect()
    try:
        yield client
    finally:
        client.close()
