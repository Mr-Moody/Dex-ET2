"""``teleop process --in A [--in B ...] --out OUT [--annotator rule]``: QC, trim, merge sessions.

ATC recovery datasets (sidecar lines carry ``source``) are detected automatically and go through
:mod:`teleop.process.atc_qc`; ``--sources`` picks which episode sources go in. One call takes
either ATC or legacy datasets, never both.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from teleop.process.annotate import Annotator, RuleAnnotator
from teleop.process.qc import (
    QCResult,
    QCThresholds,
    num_episodes,
    read_sidecar,
    run_qc,
    write_clean,
)


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


def _process_atc(ns: argparse.Namespace, annotator: Annotator | None) -> int:
    from teleop.process.atc_qc import (
        SOURCES,
        AtcQCThresholds,
        atc_action_ranges,
        run_atc_qc,
        summarize,
    )

    sources = frozenset(s.strip() for s in str(ns.sources).split(",") if s.strip())
    unknown = sorted(sources - set(SOURCES))
    if unknown or not sources:
        print(
            f"teleop process: unknown --sources {', '.join(unknown) or '(empty)'}; "
            f"expected a comma-separated subset of {','.join(SOURCES)}",
            file=sys.stderr,
        )
        return 2
    # Loaded once for every input; {} (load failed, already warned) skips the range rule.
    ranges = atc_action_ranges(ns.ranges_from_hand or "inspire") or {}
    results: list[QCResult] = []
    for root in ns.inputs:
        results += run_atc_qc(root, AtcQCThresholds(), sources, ranges=ranges)
    write_clean(ns.inputs[0], results, ns.out, annotator=annotator)
    kept = sum(r.keep for r in results)
    print(f"teleop process: kept {kept}/{len(results)} ATC episodes -> {ns.out}")
    print(summarize(results, {Path(root): read_sidecar(root) for root in ns.inputs}))
    no_gap = sum("no_max_gap_s" in r.warnings for r in results)
    if no_gap:
        print(
            f"teleop process: warning: {no_gap} episode(s) had no max_gap_s; gap check skipped",
            file=sys.stderr,
        )
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="teleop process")
    ap.add_argument("--in", dest="inputs", type=Path, action="append", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ranges-from-hand", metavar="NAME", default=None)
    ap.add_argument("--annotator", choices=["rule"], default=None)
    ap.add_argument(
        "--sources",
        default="glove,scripted,dmp",
        help="ATC datasets only: comma-separated episode sources to keep",
    )
    ns = ap.parse_args(argv)

    from teleop.process.atc_qc import is_atc_dataset

    kinds = {is_atc_dataset(root) for root in ns.inputs if num_episodes(root) > 0}
    if len(kinds) > 1:
        print("teleop process: cannot mix ATC and legacy datasets", file=sys.stderr)
        return 2

    annotator: Annotator | None = RuleAnnotator() if ns.annotator == "rule" else None
    if kinds == {True}:
        return _process_atc(ns, annotator)
    ranges = _hand_ranges(ns.ranges_from_hand) if ns.ranges_from_hand else None
    results: list[QCResult] = []
    for root in ns.inputs:
        results += run_qc(root, QCThresholds(), ranges=ranges)
    write_clean(ns.inputs[0], results, ns.out, annotator=annotator)
    kept = sum(r.keep for r in results)
    print(f"teleop process: kept {kept}/{len(results)} episodes -> {ns.out}")
    return 0
