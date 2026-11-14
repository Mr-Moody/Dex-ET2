---
id: T-214
title: Copilot local Gemma fallback
status: in_review
owner: cline
branch: feat/copilot-local-gemma
depends_on: [T-212]
plan: docs/superpowers/plans/2026-11-14-operator-copilot-plan.md#t-214-local-gemma-fallback
claimed_at: '2026-11-14T22:28:34+00:00'
---

OpenAI-compatible multimodal `LocalModelClient` (`POST {local_url}/chat/completions`, `COPILOT_LOCAL_URL`, `COPILOT_LOCAL_MODEL="gemma4:e4b"`) implementing the structured JSON VLM fallback when `GEMINI_API_KEY` is unset or `--no-gemini` is passed:

- **Fallback hierarchy**: Live Gemini (`gemini-4-hydrogen` / `gemini-4-argon`) -> `LocalModelClient` (`gemma4:e4b` when `COPILOT_LOCAL_URL` is configured) -> 100% deterministic physics-only (`SkillValidator` + deterministic template cues, zero network calls) when neither is configured or `--headless` is set.

Owned files: `src/teleop/copilot/local.py`, `tests/copilot/test_local.py`, `docs/copilot.md`.
