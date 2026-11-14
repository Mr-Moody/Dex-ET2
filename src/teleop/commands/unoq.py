"""``teleop unoq``: run the UNO Q edge service, or probe one from the laptop.

    teleop unoq --serve [--no-console] [--host 0.0.0.0] [--udp-port 15051] [--tcp-port 15052]
    teleop unoq [--host unoq.local] [--udp-port 15051] [--tcp-port 15052]

``--serve`` attaches the MCU operator console (``open_console()``) unless ``--no-console`` is
given, no MCU is found or opening it fails (the service still serves).

Without ``--serve`` it connects (or falls back to the local loopback node), pings it and
prints which node answered.
"""

from __future__ import annotations

import argparse
import logging
import os
import threading

from teleop.unoq.mcu_bridge import McuConsoleBridge, open_console
from teleop.unoq.protocol import DEFAULT_TCP_PORT, DEFAULT_UDP_PORT
from teleop.unoq.server import UnoQServer

logger = logging.getLogger(__name__)


def _attach_console(server: UnoQServer) -> McuConsoleBridge | None:
    try:
        transport, kind = open_console()
    except Exception as exc:  # e.g. arduino-router not ready yet at boot
        logger.warning("unoq: MCU console unavailable (%s); serving without one", exc)
        return None
    if kind == "software":
        logger.warning("unoq: no MCU console found; serving without one")
        return None
    bridge = McuConsoleBridge(transport)
    bridge.start()
    server.attach_console(bridge)
    logger.info("unoq: MCU console attached (%s)", kind)
    return bridge


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="teleop unoq")
    ap.add_argument("--serve", action="store_true", help="run the edge service (on the UNO Q)")
    ap.add_argument("--host", default=None, help="bind address (--serve) or UNO Q host")
    ap.add_argument("--udp-port", type=int, default=DEFAULT_UDP_PORT)
    ap.add_argument("--tcp-port", type=int, default=DEFAULT_TCP_PORT)
    ap.add_argument("--no-console", action="store_true", help="--serve without the MCU console")
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

    from teleop.unoq.client import UnoQClient

    host = ns.host or os.environ.get("UNOQ_HOST", "unoq.local")
    with UnoQClient.connect_or_mock(host, ns.udp_port, ns.tcp_port) as client:
        where = "local loopback" if client.is_mock else f"{host}:{ns.tcp_port}"
        print(f"unoq: {client.ping()} answered via {where}", flush=True)
    return 0
