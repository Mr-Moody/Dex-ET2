---
id: T-161
title: Verda Nemotron fleet reasoning co-pilot
status: todo
owner: null
branch: null
depends_on: [T-100]
plan: docs/superpowers/plans/2026-11-14-atc-implementation-plan.md#task-t-161-verda-nemotron-fleet-reasoning-co-pilot-teammate-task
claimed_at: null
---

**Reserved for Verda/Nemotron teammate. Implement inside `src/teleop/reasoning/nemotron.py` and `tests/reasoning/test_nemotron.py` without changing `FleetReasoner` / `ReasoningContext` / `ReasoningVerdict` signatures in `protocol.py`.**

Acceptance (from the plan):

- [ ] Async background worker calls Verda-hosted Nemotron (`VERDA_BASE_URL`, `VERDA_API_KEY`, `NEMOTRON_MODEL`) on stall alerts, 3-rollout gates, and OOD refusals
- [ ] Produces structured `ReasoningVerdict` (`headline`, `operator_guidance`, `compliance_rationale`) without blocking the 60 Hz loop
- [ ] Retains deterministic offline fallback when `VERDA_API_KEY` is unset or requests time out
- [ ] Unit tests in `tests/reasoning/test_nemotron.py` pass with mocked HTTP responses
