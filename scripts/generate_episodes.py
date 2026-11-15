"""Generate scripted ATC rescue episodes in parallel (ACT training data).

Runs several ``teleop atc --headless --generate`` workers, each with its own seed and output
folder, then prints the ``teleop process`` command that cleans and merges them. Saving episodes
(camera rendering and video encoding) is the slow part, so extra workers scale almost linearly
until you run out of CPU cores.

    uv sync --group train
    uv run python scripts/fetch_models.py          # once, for the Panda and Inspire models
    uv run python scripts/generate_episodes.py --total 80 --workers 4 --out data/episodes/gen-1

Each worker makes ``ceil(total / workers)`` successful episodes. Use a different ``--seed-base``
(or ``--out``) per teammate so nobody generates the same data twice.
"""

from __future__ import annotations

import argparse
import math
import shlex
import subprocess
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument(
        "--total", type=int, default=60, help="successful episodes to make (all workers)"
    )
    ap.add_argument("--workers", type=int, default=3, help="parallel generator processes")
    ap.add_argument("--out", type=Path, default=Path("data/episodes/gen"), help="output folder")
    ap.add_argument("--seed-base", type=int, default=1000, help="worker i uses seed-base + i")
    args = ap.parse_args(argv)
    if args.total < 1 or args.workers < 1:
        ap.error("--total and --workers must be at least 1")

    per_worker = math.ceil(args.total / args.workers)
    args.out.mkdir(parents=True, exist_ok=True)
    procs: list[tuple[int, Path, subprocess.Popen[bytes]]] = []
    for i in range(args.workers):
        seed = args.seed_base + i
        root = args.out / f"seed{seed}"
        log = args.out / f"seed{seed}.log"
        cmd = [
            "uv", "run", "teleop", "atc", "--headless", "--no-audio",
            "--generate", str(per_worker), "--seed", str(seed),
            "--record-episodes", str(root),
        ]  # fmt: skip
        print(f"worker {i}: seed {seed}, {per_worker} episodes -> {root} (log: {log})")
        procs.append(
            (seed, root, subprocess.Popen(cmd, stdout=log.open("wb"), stderr=subprocess.STDOUT))
        )

    failed = 0
    for seed, _, proc in procs:
        code = proc.wait()
        print(f"worker seed {seed} exited with {code}")
        failed += code != 0  # exit 1 means fewer episodes than asked, not a crash
    # A fresh folder is the dataset itself; a reused one gets a timestamped session inside it.
    found = {
        p.parent.parent
        for pat in ("seed*/meta", "seed*/*/meta")
        for p in args.out.glob(pat + "/teleop_episodes.jsonl")
    }
    sessions = sorted(found)
    print(f"\n{len(sessions)} dataset folder(s) written under {args.out}. Clean and merge with:")
    flags = " ".join(f"--in {shlex.quote(str(s))}" for s in sessions)
    print(f"  uv run teleop process {flags} --out {shlex.quote(str(args.out / 'clean'))}")
    return 1 if failed == len(procs) else 0


if __name__ == "__main__":
    sys.exit(main())
