"""``teleop process --in A [--in B ...] --out OUT [--annotator rule]``: QC, trim, merge sessions."""

from __future__ import annotations

import argparse
from pathlib import Path

from teleop.process.annotate import Annotator, RuleAnnotator
from teleop.process.qc import QCResult, QCThresholds, run_qc, write_clean


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="teleop process")
    ap.add_argument("--in", dest="inputs", type=Path, action="append", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--annotator", choices=["rule"], default=None)
    ns = ap.parse_args(argv)

    annotator: Annotator | None = RuleAnnotator() if ns.annotator == "rule" else None
    results: list[QCResult] = []
    for root in ns.inputs:
        results += run_qc(root, QCThresholds())
    write_clean(ns.inputs[0], results, ns.out, annotator=annotator)
    kept = sum(r.keep for r in results)
    print(f"teleop process: kept {kept}/{len(results)} episodes -> {ns.out}")
    return 0
