"""``teleop unoq``: run the UNO Q edge service, or probe one from the laptop.

    teleop unoq --serve [--no-console] [--host 0.0.0.0] [--udp-port 15051] [--tcp-port 15052]
    teleop unoq [--host unoq.local] [--udp-port 15051] [--tcp-port 15052]
    teleop unoq --host <board> --calibrate-joystick | --joy-map 2431 | --joy-raw

``--serve`` attaches the MCU operator console (``open_console()``) unless ``--no-console`` is
given, no MCU is found or opening it fails (the service still serves). The calibrated joystick
map is kept in ``~/.config/teleop/unoq_joystick_map`` (``UNOQ_JOYMAP_FILE``) on the board and
sent to the MCU at start and after every MCU reboot.

Without ``--serve`` it connects (or falls back to the local loopback node), pings it and
prints which node answered. ``--calibrate-joystick`` walks you through mapping the console
joystick: push it towards where you want Bay 1, 2, 3 and 4 when each bay's LED blinks amber.
"""

from __future__ import annotations

import argparse
import logging
import os
import threading
import time
from typing import Any

from teleop.unoq.mcu_bridge import (
    McuConsoleBridge,
    is_joy_map,
    load_joy_map,
    open_console,
    save_joy_map,
)
from teleop.unoq.protocol import DEFAULT_TCP_PORT, DEFAULT_UDP_PORT
from teleop.unoq.server import UnoQServer

logger = logging.getLogger(__name__)

CALIBRATION_TIMEOUT_S = 90.0
RAW_STREAM_S = 15.0
_DIRECTIONS = ("X+", "X-", "Y+", "Y-")


def _attach_console(server: UnoQServer) -> McuConsoleBridge | None:
    try:
        transport, kind = open_console()
    except Exception as exc:  # e.g. arduino-router not ready yet at boot
        logger.warning("unoq: MCU console unavailable (%s); serving without one", exc)
        return None
    if kind == "software":
        logger.warning("unoq: no MCU console found; serving without one")
        return None
    joy_map = load_joy_map()
    bridge = McuConsoleBridge(transport, joy_map=joy_map, on_joy_map=save_joy_map)
    bridge.start()
    server.attach_console(bridge)
    logger.info("unoq: MCU console attached (%s), joystick map %s", kind, joy_map or "default")
    return bridge


def describe_joy_map(code: str) -> str:
    """``X+ -> Bay 2, X- -> Bay 4, ...`` for a joystick map."""
    return ", ".join(f"{d} -> Bay {b}" for d, b in zip(_DIRECTIONS, code, strict=True))


def _calibration_message(cal: dict[str, Any]) -> str:
    phase, bay = cal.get("phase"), cal.get("bay")
    if phase == "start":
        return "Let go of the stick: capturing its centre (all four LEDs amber)..."
    if phase == "step":
        return f"Push the stick towards where you want Bay {bay} (its LED blinks amber)"
    if phase == "dup":
        return f"That direction is already taken. Push another one for Bay {bay}"
    if phase == "ok":
        return f"Calibrated and saved on the board: {describe_joy_map(str(cal.get('map')))}"
    return f"Calibration failed: {cal.get('reason')}"


def _calibrate(client: Any) -> int:
    client.joystick("calibrate")
    print("Joystick calibration started on the console. Ctrl-C aborts.", flush=True)
    last: dict[str, Any] | None = None
    deadline = time.monotonic() + CALIBRATION_TIMEOUT_S
    try:
        while time.monotonic() < deadline:
            cal = client.joystick("status").get("calibration")
            if cal is not None and cal != last:
                last = cal
                print(_calibration_message(cal), flush=True)
                if cal.get("phase") == "ok":
                    return 0
                if cal.get("phase") == "fail":
                    return 1
            time.sleep(0.1)
    except KeyboardInterrupt:
        client.joystick("abort")
        print("aborted", flush=True)
        return 1
    client.joystick("abort")
    print("No answer from the console: is the UNO Q running the console sketch?", flush=True)
    return 1


def _stream_raw(client: Any) -> int:
    client.joystick("raw", on=True)
    print(f"Raw joystick ADC (x, y) for {RAW_STREAM_S:.0f} s. Push each way and note it.")
    seen: list[int] | None = None
    deadline = time.monotonic() + RAW_STREAM_S
    try:
        while time.monotonic() < deadline:
            raw = client.joystick("status").get("raw")
            if raw is not None and raw != seen:
                seen = raw
                print(f"  x={raw[0]:4d}  y={raw[1]:4d}", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        client.joystick("raw", on=False)
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="teleop unoq")
    ap.add_argument("--serve", action="store_true", help="run the edge service (on the UNO Q)")
    ap.add_argument("--host", default=None, help="bind address (--serve) or UNO Q host")
    ap.add_argument("--udp-port", type=int, default=DEFAULT_UDP_PORT)
    ap.add_argument("--tcp-port", type=int, default=DEFAULT_TCP_PORT)
    ap.add_argument("--no-console", action="store_true", help="--serve without the MCU console")
    joy = ap.add_mutually_exclusive_group()
    joy.add_argument(
        "--calibrate-joystick", action="store_true", help="map the stick directions to bays"
    )
    joy.add_argument("--joy-map", help="set the map directly: bays for stick X+, X-, Y+, Y-")
    joy.add_argument("--joy-raw", action="store_true", help="print raw stick values for a while")
    ns = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if ns.serve:
        server = UnoQServer(host=ns.host or "0.0.0.0", udp_port=ns.udp_port, tcp_port=ns.tcp_port)
        server.start()  # logs "unoq: serving on udp ... / tcp ..."
        bridge = None if ns.no_console else _attach_console(server)
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            pass
        finally:
            server.stop()
            if bridge is not None:
                bridge.close()
        return 0

    if ns.joy_map is not None and not is_joy_map(ns.joy_map):
        ap.error("--joy-map needs four digits using each of 1-4 once, e.g. 2431")

    from teleop.unoq.client import UnoQClient

    host = ns.host or os.environ.get("UNOQ_HOST", "unoq.local")
    joystick = ns.calibrate_joystick or ns.joy_map is not None or ns.joy_raw
    if joystick:
        try:
            client = UnoQClient(host, ns.udp_port, ns.tcp_port).connect()
        except OSError as exc:
            print(f"unoq: cannot reach {host}:{ns.tcp_port} ({exc})", flush=True)
            return 1
        with client:
            if ns.calibrate_joystick:
                return _calibrate(client)
            if ns.joy_raw:
                return _stream_raw(client)
            code = str(ns.joy_map)
            client.joystick("set", map=code)
            print(f"unoq: joystick map set: {describe_joy_map(code)}", flush=True)
            return 0

    with UnoQClient.connect_or_mock(host, ns.udp_port, ns.tcp_port) as client:
        where = "local loopback" if client.is_mock else f"{host}:{ns.tcp_port}"
        print(f"unoq: {client.ping()} answered via {where}", flush=True)
    return 0
