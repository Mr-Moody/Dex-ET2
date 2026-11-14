"""Regenerate the canned background-bay cycle for the fleet ops console.

    uv run python scripts/record_background_cycle.py

Runs Bay 1 headless through one routine unlatch cycle and writes
``src/teleop/console/data/background_cycle.npz``.
"""

from __future__ import annotations

from teleop.console.background import DEFAULT_CYCLE_PATH, record_cycle, save_cycle


def main() -> int:
    cycle = record_cycle()
    save_cycle(cycle, DEFAULT_CYCLE_PATH)
    size_kb = DEFAULT_CYCLE_PATH.stat().st_size / 1024
    print(
        f"wrote {DEFAULT_CYCLE_PATH} ({size_kb:.0f} KB): {len(cycle.t)} frames, "
        f"{cycle.duration_s:.2f} s, {len(cycle.joint_names)} joints, "
        f"min C(t) {float(cycle.confidence.min()):.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
