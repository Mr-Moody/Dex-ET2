"""Fetch third-party hand and arm models into ``third_party/`` (idempotent, commit-pinned).

Needs only ``git``. Run from anywhere: ``uv run python scripts/fetch_models.py``.
Sources, commits and licences are recorded in ``third_party/README.md``.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY = ROOT / "third_party"


@dataclass(frozen=True)
class Source:
    dest: str
    url: str
    commit: str
    sparse_dirs: tuple[str, ...]
    expect: tuple[str, ...]  # files that must exist when the fetch is complete


SOURCES = (
    Source(
        dest="menagerie",
        url="https://github.com/google-deepmind/mujoco_menagerie",
        commit="370c43965323c0944b433b202aac5058b5f907a9",
        sparse_dirs=("leap_hand", "franka_emika_panda"),
        expect=(
            "leap_hand/left_hand.xml",
            "leap_hand/right_hand.xml",
            "franka_emika_panda/panda_nohand.xml",
        ),
    ),
    # Inspire RH56DFX MJCF (Correll Lab, arXiv 2603.08988), submodule pin of rh56_controller.
    Source(
        dest="inspire",
        url="https://github.com/correlllab/h1_mujoco",
        commit="fa8795a1ce1c2be418e8d87b72761c3a452efcd6",
        sparse_dirs=("inspire/assets/collision", "inspire/assets/visual"),
        expect=("inspire/inspire_left.xml", "inspire/inspire_right.xml"),
    ),
)


def git(*args: str, cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True, timeout=300
    )
    return result.stdout.strip()


def fetch(src: Source) -> None:
    dest = THIRD_PARTY / src.dest
    if (dest / ".git").exists():
        try:
            if git("rev-parse", "HEAD", cwd=dest) == src.commit and all(
                (dest / f).is_file() for f in src.expect
            ):
                print(f"{src.dest}: up to date at {src.commit[:9]}")
                return
        except subprocess.CalledProcessError:
            pass
    else:
        dest.mkdir(parents=True, exist_ok=True)
        git("init", "-q", cwd=dest)
        git("remote", "add", "origin", src.url, cwd=dest)
    git("remote", "set-url", "origin", src.url, cwd=dest)
    git("sparse-checkout", "set", "--cone", *src.sparse_dirs, cwd=dest)
    git("fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", src.commit, cwd=dest)
    git("checkout", "-q", "--force", src.commit, cwd=dest)
    missing = [f for f in src.expect if not (dest / f).is_file()]
    if missing:
        raise SystemExit(f"{src.dest}: missing after fetch: {missing}")
    print(f"{src.dest}: fetched {src.commit[:9]} from {src.url}")


def main() -> int:
    THIRD_PARTY.mkdir(exist_ok=True)
    try:
        for src in SOURCES:
            fetch(src)
    except subprocess.TimeoutExpired as exc:
        print(f"git timed out: {exc.cmd}", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as exc:
        print(f"git failed: {exc.cmd}\n{exc.stderr}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
