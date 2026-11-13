"""``teleop calibrate [--source studio] [--hand H] --out data/calib/<name>.json``.

Console prompts: "FLAT HAND — hold still" then "FIST", collecting frames for ``--seconds``
each, then writes a per-actuator (open, closed) ``Calibration``. ``--source replay
--replay-file F`` runs it headless from a capture.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import time
from pathlib import Path
from typing import Any

from teleop.app import make_source, parse_sim_config
from teleop.protocols import InputSource
from teleop.retarget.calibration import Calibration, calibrate
from teleop.retarget.direct import MIN_SPAN
from teleop.sim.hands import HandSpec, load_hand
from teleop.types import SIDES, HandFrame, Side

POLL_HZ = 120.0


def collect(source: InputSource, seconds: float) -> list[HandFrame]:
    """New frames (per side, deduplicated by host time and seq) seen during ``seconds``."""
    frames: list[HandFrame] = []
    seen: set[tuple[Side, int, int]] = set()
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        latest = source.latest()
        for f in (latest.left, latest.right):
            if f is not None and (key := (f.side, f.host_time_ns, f.seq)) not in seen:
                seen.add(key)
                frames.append(f)
        time.sleep(1.0 / POLL_HZ)
    return frames


def _phase(source: InputSource, prompt: str, seconds: float, prep_s: float) -> list[HandFrame]:
    if prep_s > 0:
        print(f"next: {prompt} (in {prep_s:.0f} s)", flush=True)
        time.sleep(prep_s)
    print(f"{prompt} ... ({seconds:.1f} s)", flush=True)
    frames = collect(source, seconds)
    counts = {s: sum(f.side == s for f in frames) for s in SIDES}
    print(f"  collected left={counts['left']} right={counts['right']} frames", flush=True)
    return frames


def report(
    spec: HandSpec, calib: Calibration, open_f: list[HandFrame], closed_f: list[HandFrame]
) -> list[str]:
    """Print the per-actuator ranges; return the flagged ``side/actuator`` names."""
    flagged: list[str] = []
    for side in SIDES:
        has_data = any(f.side == side for f in open_f) and any(f.side == side for f in closed_f)
        print(f"{side}:{'' if has_data else '  (no frames: defaults)'}")
        for fm in spec.flex:
            o, c = calib.ranges[side][fm.actuator]
            flag = ""
            if not has_data:
                flag = "  DEFAULT"
            elif abs(c - o) < MIN_SPAN:
                flag = f"  DEGENERATE (|span| < {MIN_SPAN} rad; retargeter falls back to default)"
                flagged.append(f"{side}/{fm.actuator}")
            print(f"  {fm.actuator:<18} open {o:+.3f}  closed {c:+.3f}  span {c - o:+.3f}{flag}")
    return flagged


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="teleop calibrate", allow_abbrev=False)
    ap.add_argument("--out", type=Path, default=Path("data/calib/default.json"))
    ap.add_argument("--source", default="studio")
    ap.add_argument("--seconds", type=float, default=2.0, help="collection time per pose")
    ap.add_argument("--prep-seconds", type=float, default=2.0, help="pause before each pose")
    ns, rest = ap.parse_known_args(argv)
    reconfigure: Any = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:  # non-UTF-8 pipes: replace the em dash rather than crash
        with contextlib.suppress(ValueError):
            reconfigure(errors="replace")
    cfg = parse_sim_config(rest)
    cfg.source = ns.source
    spec = load_hand(cfg.hand)
    source = make_source(cfg)
    source.start()
    try:
        open_f = _phase(source, "FLAT HAND — hold still", ns.seconds, ns.prep_seconds)
        closed_f = _phase(source, "FIST", ns.seconds, ns.prep_seconds)
    except KeyboardInterrupt:
        print("calibration aborted; nothing written")
        return 130
    finally:
        source.stop()
    if not open_f or not closed_f:
        print(f"teleop calibrate: no frames from source {cfg.source!r}; nothing written")
        return 1
    calib = calibrate(spec, open_f, closed_f)
    flagged = report(spec, calib, open_f, closed_f)
    ns.out.parent.mkdir(parents=True, exist_ok=True)
    calib.to_json(ns.out)
    print(
        f"wrote {ns.out}" + (f"; {len(flagged)} degenerate: {', '.join(flagged)}" * bool(flagged))
    )
    return 0
