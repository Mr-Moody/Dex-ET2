"""``teleop sim [--source ...] [--hand ...] [--calibration F] [--headless --steps N] [--hotspot]``.

Any ``SimConfig`` field is accepted as a flag (hyphens or underscores). Opens the MuJoCo passive
viewer unless ``--headless``. Viewer keys: arrows/PgUp/PgDn nudge the right wrist, WASD/Q/E the
left one (IMU mode), R resets the task, C recenters the Coil Pro onto the home poses.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from teleop.app import TeleopApp, parse_sim_config, start_hotspot_watchdog
from teleop.net.hotspot import HotspotConfigError, HotspotWatchdog

logger = logging.getLogger("teleop.sim")


def setup_logging() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def add_hotspot_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--hotspot", action="store_true", help="keep the booth hotspot up")
    ap.add_argument("--env-file", type=Path, default=Path(".env"))


def maybe_hotspot(ns: argparse.Namespace) -> HotspotWatchdog | None:
    """Start the hotspot watchdog if ``--hotspot``; raises HotspotConfigError on a bad .env."""
    if not ns.hotspot:
        return None
    return start_hotspot_watchdog(ns.env_file)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="teleop sim", allow_abbrev=False)
    add_hotspot_args(ap)
    ns, rest = ap.parse_known_args(argv)
    setup_logging()
    cfg = parse_sim_config(rest)
    try:
        watchdog = maybe_hotspot(ns)
    except (HotspotConfigError, RuntimeError) as exc:
        print(f"teleop sim: {exc}")
        return 1
    try:
        app = TeleopApp(cfg, hotspot=watchdog)
        stats = app.run()
        logger.info("done: %d steps, %d loop overruns", stats.steps, stats.overruns)
    except KeyboardInterrupt:
        pass
    finally:
        if watchdog is not None:
            watchdog.stop()
    return 0
