"""Fleet ops console preview: synthetic telemetry and placeholder frames, no simulation.

    uv run --with pillow python scripts/console_preview.py [--port 8765] [--rescue]

Walks a fake stage script (routine, Bay 1 stall, queue, rescue, skill gate, Bay 2 auto skill,
Bay 2 unseen jam) through the real ``TelemetryAdapter`` and ``ConsoleServer``, with labelled
placeholder JPEG frames, so the page can be built and screenshotted without MuJoCo.
``--rescue`` holds the rescue view. ``--no-bay4`` shows a STANDBY tile.
"""

from __future__ import annotations

import argparse
import io
import math
import time
from collections.abc import Sequence

from PIL import Image, ImageDraw  # pyright: ignore[reportMissingImports]  (Pillow arrives with T-173)

from teleop.atc.hud import GateStatus, HudView, RescueStatus
from teleop.atc.types import BaySnapshot, BayState
from teleop.console.layout import FeedSpec, ViewState, feed_layout
from teleop.console.run import _next_view_state
from teleop.console.server import ConsoleServer
from teleop.console.telemetry import TelemetryAdapter

SKU = "EV-LATCH-01"
LOOP_S = 44.0
FPS = 10.0
TELEMETRY_HZ = 10.0


def _snap(bay: int, state: BayState, c: float, **kw: object) -> BaySnapshot:
    return BaySnapshot(bay, state, max(0.0, min(1.0, c)), 0.0, 0.5, SKU, **kw)  # type: ignore[arg-type]


def scene(t: float, hold_rescue: bool) -> HudView:
    """The fake stage at script time ``t`` (seconds into the loop)."""
    if hold_rescue:
        t = 12.0
    wobble = 0.03 * math.sin(t * 2.1)
    b1 = _snap(1, BayState.NOMINAL, 0.95 + wobble)
    b2 = _snap(2, BayState.NOMINAL, 0.94 - wobble)
    queue: tuple[int, ...] = ()
    rescue = RescueStatus()
    gate = GateStatus()
    compliance = "Solana devnet: no events yet"
    reasoning = "Fleet advisor idle"
    selected: int | None = None
    if 6 <= t < 7:
        b1 = _snap(1, BayState.STALLED, 0.21)
        compliance = "Solana devnet: STALL_ALERT bay 1, tx 4f2a…9c1e"
    elif 7 <= t < 9:
        b1 = _snap(1, BayState.QUEUED, 0.18, queue_position=0)
        queue = (1,)
        compliance = "Solana devnet: STALL_ALERT bay 1, tx 4f2a…9c1e"
        reasoning = "Bay 1: raised stopper flap, likely a sprung latch. Recommend rescue."
    elif 9 <= t < 19:
        b1 = _snap(1, BayState.RESCUE, 0.35 + 0.05 * math.sin(t * 3))
        selected = 1
        rescue = RescueStatus(1, t >= 10, t - 9)
        compliance = "Solana devnet: STALL_ALERT bay 1, tx 4f2a…9c1e"
        reasoning = "Bay 1: raised stopper flap, likely a sprung latch. Recommend rescue."
    elif 19 <= t < 20.5:
        b1 = _snap(1, BayState.VALIDATING, 0.9)
        gate = GateStatus(fit_ms=14.0)
        compliance = "Solana devnet: RESCUE_LOGGED bay 1, tx 8b03…77fd"
    if t >= 20.5:
        gate = GateStatus(14.0, 3, 3, 0.41, True)
        compliance = "Solana devnet: SKILL_DEPLOYED skill-1, tx 1d9e…a402"
    if 26 <= t < 27:
        b2 = _snap(2, BayState.STALLED, 0.24)
    elif 27 <= t < 31:
        b2 = _snap(2, BayState.AUTONOMOUS_SKILL, 0.88, active_skill_id="skill-1")
    elif 34 <= t < LOOP_S:
        b2 = _snap(2, BayState.OOD_ESCALATED, 0.12)
        compliance = "Solana devnet: OOD_REFUSAL bay 2, tx 6c55…02be"
        reasoning = "Bay 2: bolt pinned at 0 cm, unlike the learned jam. Skill transfer refused."
    return HudView(
        bays=(b1, b2),
        queue=queue,
        background_bays=(3, 4),
        selected_bay=selected,
        rescue=rescue,
        gate=gate,
        compliance=compliance,
        reasoning=reasoning,
        beat="",
        unoq="UNO Q local loopback | compliance devnet | reasoner offline (preview)",
    )


def fake_health(now: float) -> list[dict[str, object]]:
    """Hotspot ok, gloves failing (reconnectable; busy for 3 s of each 10 s), panel warn."""
    busy = now % 10.0 < 3.0
    return [
        {
            "name": "hotspot",
            "level": "ok",
            "summary": "Up, 2 clients",
            "detail": "SSID booth",
            "can_reconnect": False,
            "busy": False,
        },
        {
            "name": "gloves",
            "level": "fail",
            "summary": "No data",
            "detail": "no datagram for 2.4 s",
            "can_reconnect": True,
            "busy": busy,
        },
        {
            "name": "panel",
            "level": "warn",
            "summary": "Loopback",
            "detail": "board unreachable",
            "can_reconnect": True,
            "busy": False,
        },
    ]


def placeholder_jpeg(feed: FeedSpec, t: float) -> bytes:
    img = Image.new("RGB", (feed.width, feed.height), (29, 44, 38))
    draw = ImageDraw.Draw(img)
    # Sustainable warehouse clerestory band + conveyor belt + moving EV battery packs.
    draw.rectangle([0, 0, feed.width, int(feed.height * 0.28)], fill=(44, 68, 58))
    belt_y0, belt_y1 = int(feed.height * 0.52), int(feed.height * 0.68)
    draw.rectangle([0, belt_y0 - 4, feed.width, belt_y1 + 4], fill=(42, 118, 82))
    draw.rectangle([0, belt_y0, feed.width, belt_y1], fill=(28, 32, 34))
    for i in range(-1, feed.width // 60 + 2):
        cx = int((i * 60 + t * 50) % (feed.width + 60)) - 30
        draw.line([(cx, belt_y0), (cx, belt_y1)], fill=(60, 68, 70), width=2)
    x = int((t * 50) % feed.width)
    draw.rectangle(
        [x, belt_y0 + 4, x + 64, belt_y1 - 4], fill=(38, 128, 112), outline=(242, 183, 5)
    )
    draw.text(
        (16, int(feed.height * 0.36)),
        f"{feed.camera} · EV Battery Recycling Conveyor ({feed.role})",
        fill=(218, 232, 224),
    )
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=75)
    return buf.getvalue()


def run(port: int, hold_rescue: bool, bays: Sequence[int]) -> None:
    vstate = [ViewState()]

    def on_msg(msg: dict[str, object]) -> None:
        if msg.get("type") == "view":
            vstate[0] = _next_view_state(vstate[0], msg)

    server = ConsoleServer(port=port, on_client_message=on_msg)
    server.start()
    print(f"console preview: {server.url}  (Ctrl+C to stop)", flush=True)
    adapter = TelemetryAdapter(bays, [b for b in (3, 4) if b in bays])
    t0 = time.monotonic()
    next_tel = next_frame = 0.0
    try:
        while True:
            now = time.monotonic() - t0
            t = now % LOOP_S
            view = scene(t, hold_rescue)
            feeds = feed_layout(bays, view.rescue.bay_id, vstate[0])
            if now >= next_tel:
                bg = {3: 0.96 + 0.03 * math.sin(now), 4: 0.95 + 0.03 * math.cos(now * 1.3)}
                msg = adapter.build(view, now, feeds, bg, view_state=vstate[0])
                msg["health"] = fake_health(now)
                server.publish_telemetry(msg)
                next_tel += 1.0 / TELEMETRY_HZ
            if now >= next_frame:
                for f in feeds:
                    server.publish_frame(f.slot, adapter.layout_id, placeholder_jpeg(f, now))
                next_frame += 1.0 / FPS
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


def main() -> int:
    ap = argparse.ArgumentParser(prog="console_preview")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--rescue", action="store_true", help="hold the rescue view")
    ap.add_argument("--no-bay4", action="store_true", help="show Bay 4 as STANDBY")
    ns = ap.parse_args()
    run(ns.port, ns.rescue, (1, 2, 3) if ns.no_bay4 else (1, 2, 3, 4))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
