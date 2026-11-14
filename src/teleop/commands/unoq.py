"""``teleop unoq``: run the UNO Q edge service, or probe one from the laptop.

    teleop unoq --serve [--host 0.0.0.0] [--udp-port 15051] [--tcp-port 15052]
    teleop unoq [--host unoq.local] [--udp-port 15051] [--tcp-port 15052]

Without ``--serve`` it connects (or falls back to the local loopback node), pings it and
prints which node answered.
"""

from __future__ import annotations

import argparse
import logging
import os
import threading

from teleop.unoq.protocol import DEFAULT_TCP_PORT, DEFAULT_UDP_PORT


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="teleop unoq")
    ap.add_argument("--serve", action="store_true", help="run the edge service (on the UNO Q)")
    ap.add_argument("--host", default=None, help="bind address (--serve) or UNO Q host")
    ap.add_argument("--udp-port", type=int, default=DEFAULT_UDP_PORT)
    ap.add_argument("--tcp-port", type=int, default=DEFAULT_TCP_PORT)
    ns = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if ns.serve:
        from teleop.unoq.server import UnoQServer

        server = UnoQServer(host=ns.host or "0.0.0.0", udp_port=ns.udp_port, tcp_port=ns.tcp_port)
        server.start()  # logs "unoq: serving on udp ... / tcp ..."
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            pass
        finally:
            server.stop()
        return 0

    from teleop.unoq.client import UnoQClient

    host = ns.host or os.environ.get("UNOQ_HOST", "unoq.local")
    with UnoQClient.connect_or_mock(host, ns.udp_port, ns.tcp_port) as client:
        where = "local loopback" if client.is_mock else f"{host}:{ns.tcp_port}"
        print(f"unoq: {client.ping()} answered via {where}", flush=True)
    return 0
