"""Run ``teleop atc`` through the fleet ops console instead of the MuJoCo viewer.

The main thread keeps the 60 Hz stage loop and, after every tick, drops a ``SimSnapshot`` in
a latest-value slot. The render worker draws the camera feeds from it on its own thread and the
console server streams them, plus telemetry, to the browser. Browser key presses go into the
existing ``StageController.on_key`` queue; the glove and hardware console are unchanged.
"""

from __future__ import annotations

import logging
import webbrowser
from collections.abc import Callable
from typing import Any

from teleop.atc.coordinator import OperatorInput
from teleop.atc.stage import StageController
from teleop.atc.types import N_BAYS
from teleop.console.background import load_cycle
from teleop.console.render import RenderWorker
from teleop.console.server import ConsoleServer
from teleop.console.snapshot import LatestSlot, SimSnapshot
from teleop.console.telemetry import TelemetryAdapter
from teleop.sim.cells import cctv_camera_name
from teleop.sim.loop import FixedRateLoop, LoopStats

logger = logging.getLogger("teleop.console")

CONTROL_HZ = 60


class ConsolePortInUse(OSError):
    def __init__(self, port: int, cause: OSError) -> None:
        super().__init__(f"console port {port} in use ({cause})")
        self.port = port


def bays_in_model(model: Any) -> tuple[int, ...]:
    """Bays whose CCTV camera exists in the compiled scene."""
    out: list[int] = []
    for k in range(1, N_BAYS + 1):
        try:
            model.camera(cctv_camera_name(k))
        except KeyError:
            continue
        out.append(k)
    return tuple(out)


def run_console(
    controller: StageController,
    operator: Callable[[], OperatorInput | None],
    *,
    port: int,
    fps: int,
    open_browser: bool,
    steps: int | None = None,
) -> LoopStats:
    co = controller.coordinator
    bays = bays_in_model(co.model)
    server = ConsoleServer(port=port, on_key=controller.on_key)
    try:
        server.start()
    except OSError as exc:
        raise ConsolePortInUse(port, exc) from exc

    snapshots: LatestSlot[SimSnapshot] = LatestSlot()
    snapshots.put(SimSnapshot.capture(co.data, controller.hud_view(), co.time_s))
    background = load_cycle() if controller.background_bays else None
    adapter = TelemetryAdapter(bays, controller.background_bays)
    worker = RenderWorker(co.model, snapshots, server, adapter, bays, background, fps=fps)
    worker.start()
    print(f"teleop atc: fleet ops console at {server.url}", flush=True)
    if open_browser:
        webbrowser.open(server.url)

    def step(_: int) -> bool:
        controller.tick(operator())
        snapshots.put(SimSnapshot.capture(co.data, controller.hud_view(), co.time_s))
        return True

    try:
        stats = FixedRateLoop(CONTROL_HZ).run(step, steps)
    finally:
        worker.stop()
        worker.join(timeout=5)
        server.stop()
    logger.info(
        "atc console: %d steps, %d loop overruns; %d frame sets at %.1f ms mean, %d render errors",
        stats.steps,
        stats.overruns,
        worker.stats.frames,
        worker.stats.mean_frame_ms,
        worker.stats.errors,
    )
    return stats
