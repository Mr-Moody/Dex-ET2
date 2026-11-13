"""``teleop replay --file PATH [--speed X]``: replay a raw capture and print its status."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from teleop.input.replay import ReplaySource


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="teleop replay")
    ap.add_argument("--file", type=Path, required=True)
    ap.add_argument("--speed", type=float, default=1.0)
    ns = ap.parse_args(argv)
    source = ReplaySource(ns.file, speed=ns.speed)
    source.start()
    try:
        while True:
            print(source.status(), flush=True)
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        source.stop()
    return 0
