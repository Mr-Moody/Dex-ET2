"""Run ``teleop atc`` through the fleet ops console instead of the MuJoCo viewer.

The main thread keeps the 60 Hz stage loop and, after every tick, drops a ``SimSnapshot`` in
a latest-value slot. The render worker draws the camera feeds from it on its own thread and the
console server streams them, plus telemetry, to the browser. Browser key presses go into the
existing ``StageController.on_key`` queue; the glove and hardware console are unchanged.
"""

from __future__ import annotations

import logging
import threading
import webbrowser
from collections.abc import Callable, Mapping
from typing import Any, cast

from teleop.atc.coordinator import OperatorInput
from teleop.atc.stage import StageController
from teleop.atc.types import N_BAYS
from teleop.console.background import load_cycle
from teleop.console.health_publish import HealthPublisher
from teleop.console.layout import (
    PAN_LIMIT,
    ROT_LIMIT,
    VALID_VIEWS,
    ZOOM_MAX,
    ZOOM_MIN,
    LayoutMode,
    ViewState,
)
from teleop.console.render import RenderWorker
from teleop.console.server import ConsoleServer
from teleop.console.snapshot import LatestSlot, SimSnapshot
from teleop.console.telemetry import TelemetryAdapter
from teleop.health.monitor import HealthMonitor
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


def _next_view_state(prev: ViewState, msg: Mapping[str, Any]) -> ViewState:
    mode_raw = msg.get("mode", prev.mode)
    mode: LayoutMode = mode_raw if mode_raw in ("auto", "grid", "pin", "overview") else prev.mode
    pinned = prev.pinned_bay
    if "pinned_bay" in msg:
        pb = msg["pinned_bay"]
        valid_pin = isinstance(pb, int) and not isinstance(pb, bool) and 1 <= pb <= N_BAYS
        pinned = int(pb) if valid_pin else None
    cameras = dict(prev.cameras)
    bay_cam = msg.get("camera")
    bay_id = msg.get("bay")
    if (
        isinstance(bay_id, int)
        and not isinstance(bay_id, bool)
        and 1 <= bay_id <= N_BAYS
        and isinstance(bay_cam, str)
        and bay_cam in VALID_VIEWS
    ):
        cameras[bay_id] = bay_cam
    order = prev.bay_order
    raw_order = msg.get("bay_order")
    if isinstance(raw_order, list):
        items = cast("list[Any]", raw_order)
        parsed = [
            int(x)
            for x in items
            if isinstance(x, int) and not isinstance(x, bool) and 1 <= x <= N_BAYS
        ]
        dedup: list[int] = []
        for x in parsed:
            if x not in dedup:
                dedup.append(x)
        for k in range(1, N_BAYS + 1):
            if k not in dedup:
                dedup.append(k)
        order = tuple(dedup)
    zooms = dict(prev.zooms)
    zcam = msg.get("zoom_cam")
    if isinstance(zcam, str) and ("/" in zcam or zcam == "overview"):
        if msg.get("reset_zoom"):
            zooms.pop(zcam, None)
        else:
            cur_z, cur_px, cur_py, cur_rx, cur_ry = zooms.get(zcam, (1.0, 0.0, 0.0, 0.0, 0.0))
            z_raw = msg.get("zoom", cur_z)
            px_raw = msg.get("pan_x", cur_px)
            py_raw = msg.get("pan_y", cur_py)
            rx_raw = msg.get("rot_x", cur_rx)
            ry_raw = msg.get("rot_y", cur_ry)
            if isinstance(z_raw, int | float) and not isinstance(z_raw, bool):
                cur_z = min(ZOOM_MAX, max(ZOOM_MIN, float(z_raw)))
            if isinstance(px_raw, int | float) and not isinstance(px_raw, bool):
                cur_px = min(PAN_LIMIT, max(-PAN_LIMIT, float(px_raw)))
            if isinstance(py_raw, int | float) and not isinstance(py_raw, bool):
                cur_py = min(PAN_LIMIT, max(-PAN_LIMIT, float(py_raw)))
            if isinstance(rx_raw, int | float) and not isinstance(rx_raw, bool):
                cur_rx = min(ROT_LIMIT, max(-ROT_LIMIT, float(rx_raw)))
            if isinstance(ry_raw, int | float) and not isinstance(ry_raw, bool):
                cur_ry = min(ROT_LIMIT, max(-ROT_LIMIT, float(ry_raw)))
            zooms[zcam] = (cur_z, cur_px, cur_py, cur_rx, cur_ry)
    return ViewState(mode=mode, pinned_bay=pinned, cameras=cameras, bay_order=order, zooms=zooms)


def run_console(
    controller: StageController,
    operator: Callable[[], OperatorInput | None],
    *,
    port: int,
    fps: int,
    open_browser: bool,
    steps: int | None = None,
    health: HealthMonitor | None = None,
    dummy: Any = None,
) -> LoopStats:
    co = controller.coordinator
    bays = bays_in_model(co.model)
    view_lock = threading.Lock()
    view_box: list[ViewState] = [ViewState()]
    fault_queue: list[tuple[int, str]] = []

    def on_client_message(msg: dict[str, Any]) -> None:
        mtype = msg.get("type")
        if mtype == "view":
            with view_lock:
                view_box[0] = _next_view_state(view_box[0], msg)
            worker.set_view_state(view_box[0])
        elif mtype == "fault":
            bay = msg.get("bay")
            kind = msg.get("kind")
            if (
                isinstance(bay, int)
                and not isinstance(bay, bool)
                and bay in controller.active_bays
                and kind in ("flap", "pin", "clear")
            ):
                with view_lock:
                    fault_queue.append((bay, str(kind)))
        elif mtype == "dummy" and dummy is not None:
            dummy.apply_ui_message(msg)

    def reconnect(name: str) -> None:
        if health is not None:
            health.request_reconnect(name)  # non-blocking; the bool is not needed here

    server = ConsoleServer(
        port=port,
        on_key=controller.on_key,
        on_reconnect=reconnect if health is not None else None,
        on_client_message=on_client_message,
    )
    try:
        server.start()
    except OSError as exc:
        raise ConsolePortInUse(port, exc) from exc

    snapshots: LatestSlot[SimSnapshot] = LatestSlot()
    snapshots.put(SimSnapshot.capture(co.data, controller.hud_view(), co.time_s))
    background = load_cycle() if controller.background_bays else None
    adapter = TelemetryAdapter(bays, controller.background_bays)
    sink = HealthPublisher(server, health) if health is not None else server
    worker = RenderWorker(
        co.model,
        snapshots,
        sink,
        adapter,
        bays,
        background,
        fps=fps,
        dummy_status=dummy.status_dict if dummy is not None else None,
    )
    worker.set_view_state(view_box[0])
    worker.start()
    print(f"teleop atc: fleet ops console at {server.url}", flush=True)
    if open_browser:
        webbrowser.open(server.url)

    def step(_: int) -> bool:
        with view_lock:
            pending_faults = list(fault_queue)
            fault_queue.clear()
        for bay, kind in pending_faults:
            if kind == "clear":
                controller.armed.pop(bay, None)
            elif kind == "flap":
                controller.arm(bay, flap_jammed=True)
            elif kind == "pin":
                controller.arm(bay, bolt_pinned=True)
        op_in = operator()
        if dummy is not None:
            op_in = dummy.step(op_in)
        controller.tick(op_in)
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
