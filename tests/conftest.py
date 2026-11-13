"""Shared fixtures for the whole test suite."""

from __future__ import annotations

import subprocess
import sys

import pytest

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
