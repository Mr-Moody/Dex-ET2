---
id: T-162
title: Switch Verda fleet reasoner to Kimi K3.1 Pro
status: in_review
owner: ethan
branch: feat/verda-kimi-reasoner
depends_on: [T-161]
plan: docs/superpowers/plans/2026-11-14-atc-implementation-plan.md#task-t-161-verda-nemotron-fleet-reasoning-co-pilot-teammate-task
claimed_at: '2026-11-14T04:41:12+00:00'
---

Point the Verda reasoner (`src/teleop/reasoning/nemotron.py`) at Verda's Inference API model
`kimi-k3-1-pro` instead of Nemotron, without changing `protocol.py` signatures.

Acceptance:

- [ ] Default model is `kimi-k3-1-pro` on `https://inference.verda.com/v1`; `NEMOTRON_MODEL` still overrides
- [ ] Sends `reasoning_effort` (default `none`) so the 4 s budget is not eaten by reasoning tokens
- [ ] Drops the Nemotron-only `detailed thinking off` prompt line
- [ ] Ignores `reasoning_content`; empty `content` (finish_reason `length`) falls back to offline triage
- [ ] Unit tests with mocked HTTP cover the new payload and fallback
