---
id: T-210
title: Operator copilot foundation, persistent IPv4 Gemini client and model probe
status: done
owner: null
branch: null
depends_on: [T-175]
plan: docs/superpowers/plans/2026-11-14-operator-copilot-plan.md#t-210-copilot-foundation-gemini-client-and-model-probe
claimed_at: null
---

Config (`CopilotConfig`), stdlib Gemini REST + SSE transport (`gemini.py`, `tts_stream.py` persistent IPv4 `IPv4HTTPSConnection` with `AF_INET`, `TCP_NODELAY`, `SO_KEEPALIVE`, TLS SNI, and `Connection: keep-alive`), and `python -m teleop.copilot probe` to confirm model IDs (`gemini-4-hydrogen`, `gemini-4-argon`, `gemini-3.8-flash-lite-tts` voice `Kore`) and latency.

Owned files: `src/teleop/copilot/{__init__,__main__,config,gemini,tts_stream}.py`, `docs/copilot.md`, `tests/copilot/test_gemini.py`.
Safety boundary: active grasp gating (`GraspGate`) and console/dataset audit hooks are permitted, while `tests/copilot/test_gemini.py` enforces that the copilot never injects raw operator `"key"` / `.on_key` control messages or blocks the 60 Hz physics loop.
