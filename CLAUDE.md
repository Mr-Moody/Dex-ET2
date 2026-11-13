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
6. Conventional Commits `type(scope): summary`; trailer
   `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Never push to `main`.

## Hardware

This laptop is the hardware host (gloves, Studio, hotspot). Hardware-touching tasks run here;
hardware-only tests use `@pytest.mark.hardware`.

## Layout

`src/teleop/cli.py` holds the pre-populated `COMMANDS` dict; add command modules under
`src/teleop/commands/` exposing `main(argv: list[str]) -> int` and do not edit `cli.py`.
