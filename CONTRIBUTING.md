# Contributing

Design: `docs/superpowers/specs/2026-11-13-glove-teleop-booth-design.md`.
Plan: `docs/superpowers/plans/2026-11-13-m0-m1-glove-to-sim.md`.

## Rules

- `main` is protected. No direct pushes. Every change lands via a PR into `main`, squash-merged.
- Branch names: `<type>/<task-id>-<kebab-summary>`, e.g. `feat/T-007-rgmp-parser`.
- Commit messages and PR titles follow Conventional Commits 1.0: `type(scope): imperative summary`.
  - Types: `feat`, `fix`, `chore`, `docs`, `refactor`, `test`, `perf`, `build`, `ci`, `style`, `revert`.
  - Scopes: `input`, `sim`, `retarget`, `record`, `train`, `cli`, `infra`, `docs`, `tasks`, `net`.
  - Breaking changes: `!` after type/scope and a `BREAKING CHANGE:` footer.
- Rebase on `origin/main` before opening a PR. No merge commits on feature branches.
  Keep PRs small: one task per PR.
- Use the PR template: task ID, spec and plan links, summary, test evidence (command plus
  output), checklist, and a `Closes T-xxx` line.
- Never commit secrets, licence keys, raw recordings or datasets. `data/`, `outputs/` and `.env`
  are gitignored. Small trimmed captures under `tests/fixtures/` (each under 200 KB) are the exception.
- Add an ADR to `docs/adr/` for any stack change or cross-cutting decision.
- Use uv only (`uv add`, `uv run`, `uv sync`), never pip. A task that needs a new dependency
  says so in its PR, because `uv.lock` conflicts are expensive.

## Verification before any PR

```
uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest
```

Paste the output in the PR. No test or CI job may need gloves, Studio, the hotspot, a GPU or a
display; hardware-only checks use `@pytest.mark.hardware` and are deselected by default.

## Weekend relaxation (spec section 6)

The separate reviewer-agent pass per PR is dropped. A PR is squash-merged once CI is green and
the lead has done a quick check. Required review and branch protection are not enforced on
GitHub for the weekend. Repo visibility, licence and Hub pushes: ask the user first.

## Pre-commit

`uv run pre-commit install --hook-type pre-commit --hook-type commit-msg`
