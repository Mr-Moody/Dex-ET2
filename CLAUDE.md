# CLAUDE.md

Rokoko gloves -> MuJoCo teleop -> LeRobot datasets. Hackathon project.

- Spec: `docs/superpowers/specs/2026-11-13-glove-teleop-booth-design.md`
- Plan (Global Constraints live here, do not copy them): `docs/superpowers/plans/2026-11-13-m0-m1-glove-to-sim.md`
- Contribution rules: `CONTRIBUTING.md`

## Workflow

1. Run `uv run task list --ready` first.
2. Never work without `uv run task claim`. One active claim per agent.
3. One git worktree per task, branch `<type>/T-NNN-<kebab>`.
4. Follow the Global Constraints in the plan (Python 3.12, uv only, xyzw quaternions, joints by
   name, no hardware in tests, and so on).
5. Verify before any PR:
   `uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest`
6. Conventional Commits `type(scope): summary`. Never commit as co-author (`Co-Authored-By`
   trailers or "Generated with" lines are forbidden for all agents). Never push to `main`.

## Hardware

This laptop is the hardware host (gloves, Studio, hotspot). Hardware-touching tasks run here;
hardware-only tests use `@pytest.mark.hardware`.

## Layout

`src/teleop/cli.py` holds the pre-populated `COMMANDS` dict; add command modules under
`src/teleop/commands/` exposing `main(argv: list[str]) -> int` and do not edit `cli.py`.

## Active Pivot: Air Traffic Control for Robot Hands (`T-100+`)

Read these documents first before starting any new work:

- Project Brief: `PROJECT_BRIEF.md`
- Engineering Spec: `docs/superpowers/specs/2026-11-14-atc-fleet-rescue-design.md`
- Implementation Plan & File-Ownership Rules: `docs/superpowers/plans/2026-11-14-atc-implementation-plan.md`
- ADR: `docs/adr/0002-atc-fleet-rescue-pivot.md`

Key rules for parallel agents on the `T-100`–`T-161` track:

1. Claim tasks from the **`T-100+`** series (`uv run task list --ready`). Legacy M2/M3 booth tasks (`T-013`, `T-060`, `T-070`, `T-080`, `T-081`) are superseded by ADR 0002.
2. **Only `T-100`** edits shared entrypoints (`src/teleop/cli.py`, `scripts/fetch_models.py`, `configs/hands/leap.yaml`, `.env.example`, `.pre-commit-config.yaml`). Tasks `T-110`–`T-161` must only create/edit their own owned modules listed in `2026-11-14-atc-implementation-plan.md`.
3. `T-160` (**Solana compliance & alert chain**, `src/teleop/compliance/solana.py`) and `T-161` (**Verda Nemotron reasoning**, `src/teleop/reasoning/nemotron.py`) are reserved for their respective teammates; `T-100` creates non-blocking placeholder implementations for both.
4. Never commit as co-author (`Co-Authored-By` trailers or "Generated with" lines are forbidden for all agents).
