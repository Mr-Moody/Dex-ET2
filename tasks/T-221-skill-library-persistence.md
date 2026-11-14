---
id: T-221
title: Persist the DMP skill library across runs
status: todo
owner: null
branch: null
depends_on: [T-220]
plan: docs/superpowers/plans/2026-11-14-console-pages-data-collection.md
claimed_at: null
---

- [ ] `atc/skill_store.py`: load, atomic save on deploy, reset with backup
- [ ] Coordinator `skill_listeners` and `reset_library`
- [ ] `teleop atc --skills` / `--fresh-skills`
- [ ] Full verify passes; DMP stage golden unchanged

Spec: `docs/superpowers/specs/2026-11-14-console-pages-and-data-collection-design.md`. Follow the
plan's Global Constraints and touch only the files the plan's File Ownership table lists for this
task.
