---
id: T-100
title: ATC shared contracts, CLI registration, Panda fetch, 16-DoF LEAP & placeholders
status: in_review
owner: lead-laptop
branch: feat/atc-shared-contracts
depends_on: [T-050]
plan: docs/superpowers/plans/2026-11-14-atc-implementation-plan.md#task-t-100-atc-shared-contracts-cli-registration-panda-fetch-16-dof-leap--placeholders
claimed_at: '2026-11-14T00:45:41+00:00'
---

**Sequential foundation task for the ATC pivot (`PROJECT_BRIEF.md`). Only `T-100` may edit shared entrypoints (`cli.py`, `scripts/fetch_models.py`, `configs/hands/leap.yaml`, `.env.example`, `.pre-commit-config.yaml`).**

Acceptance (from the plan):

- [ ] Rebase onto latest `origin/main` first; mark any still-`todo` legacy booth tasks (`T-013`, `T-060`, `T-070`, `T-080`, `T-081`) as `blocked` via `uv run task status <ID> blocked`
- [ ] Extend `scripts/fetch_models.py` (`menagerie`) to fetch `franka_emika_panda` (`panda_nohand.xml`) alongside `leap_hand` and document in `third_party/README.md`
- [ ] Extend `configs/hands/leap.yaml` to map all 16 LEAP actuators (`th_axl_act`, `if_rot_act`, `mf_rot_act`, `rf_rot_act` added)
- [ ] Register `"atc": "teleop.commands.atc"` and `"unoq": "teleop.commands.unoq"` in `src/teleop/cli.py` and add `.env.example` keys (`UNOQ_*`, `SOLANA_*`, `VERDA_*`, `NEMOTRON_MODEL`)
- [ ] Create `src/teleop/atc/types.py`, `src/teleop/compliance/{protocol,solana}.py` (`SolanaCompliancePlaceholder`), and `src/teleop/reasoning/{protocol,nemotron}.py` (`VerdaNemotronPlaceholder`)
- [ ] `tests/atc/test_types_and_placeholders.py` and full verification (`ruff`, `pyright`, `pytest`) pass
