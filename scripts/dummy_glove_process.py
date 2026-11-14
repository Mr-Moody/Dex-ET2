"""Standalone temporary Rokoko Studio JSON v3 UDP streamer process.

TODO(rokoko-gloves-swap): TEMPORARY PROCESS while physical Rokoko Smartgloves are
disconnected. Run alongside ``uv run teleop atc --source studio`` (or pass ``--dummy-glove``
to ``teleop atc`` to spawn it automatically). Stop/remove once real Rokoko Studio streaming
is active on UDP port 14043. See ``docs/DUMMY_GLOVE_SWAP.md``.
"""

from __future__ import annotations

import argparse
import logging
import signal
import threading

from teleop.input.dummy_glove import (
    DEFAULT_CONTROL_PORT,
    DEFAULT_STREAM_HZ,
    DummyGloveUdpServer,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dummy_glove_process")
    ap.add_argument("--studio-host", default="127.0.0.1", help="Target StudioSource UDP host")
    ap.add_argument("--studio-port", type=int, default=14043, help="Target StudioSource UDP port")
    ap.add_argument("--control-host", default="127.0.0.1", help="Local UDP control bind host")
    ap.add_argument(
        "--control-port",
        type=int,
        default=DEFAULT_CONTROL_PORT,
        help="Local UDP control bind port",
    )
    ap.add_argument("--hz", type=float, default=DEFAULT_STREAM_HZ, help="UDP stream rate in Hz")
    ns = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    done = threading.Event()

    def _handle_sig(_sig: int, _frame: object) -> None:
        done.set()

    signal.signal(signal.SIGINT, _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)

    srv = DummyGloveUdpServer(
        studio_host=ns.studio_host,
        studio_port=ns.studio_port,
        control_host=ns.control_host,
        control_port=ns.control_port,
        hz=ns.hz,
    ).start()
    logging.info(
        "DUMMY GLOVE PROCESS (TEMP): streaming Studio JSON v3 -> %s:%d at %.0f Hz (ctrl %s:%d)",
        ns.studio_host,
        ns.studio_port,
        ns.hz,
        ns.control_host,
        srv.control_port,
    )
    try:
        while not done.is_set():
            done.wait(0.5)
    finally:
        srv.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
