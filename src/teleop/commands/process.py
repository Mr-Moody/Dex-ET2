"""``teleop process --in A [--in B ...] --out OUT [--annotator rule]``: QC, trim, merge sessions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from teleop.process.annotate import Annotator, RuleAnnotator
from teleop.process.qc import QCResult, QCThresholds, run_qc, write_clean


def _hand_ranges(name: str) -> dict[str, tuple[float, float]] | None:
    """Action name (left/<act>, right/<act>) -> ctrlrange; None (with a warning) on failure."""
    try:
        from teleop.sim.hands import actuator_ranges, load_hand

        spec = load_hand(name)
        out: dict[str, tuple[float, float]] = {}
        for side in ("left", "right"):
            for act, rng in actuator_ranges(spec, side).items():
                out[f"{side}/{act}"] = rng
        return out
    except Exception as exc:  # noqa: BLE001 - QC must still run without ranges
        print(f"teleop process: warning: cannot load hand {name!r} ranges: {exc}", file=sys.stderr)
        return None


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="teleop process")
    ap.add_argument("--in", dest="inputs", type=Path, action="append", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ranges-from-hand", metavar="NAME", default=None)
    ap.add_argument("--annotator", choices=["rule"], default=None)
    ns = ap.parse_args(argv)

    annotator: Annotator | None = RuleAnnotator() if ns.annotator == "rule" else None
    ranges = _hand_ranges(ns.ranges_from_hand) if ns.ranges_from_hand else None
    results: list[QCResult] = []
    for root in ns.inputs:
        results += run_qc(root, QCThresholds(), ranges=ranges)
    write_clean(ns.inputs[0], results, ns.out, annotator=annotator)
    kept = sum(r.keep for r in results)
    print(f"teleop process: kept {kept}/{len(results)} episodes -> {ns.out}")
    return 0
