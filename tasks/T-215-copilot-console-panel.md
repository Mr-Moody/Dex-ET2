---
id: T-215
title: Show copilot output in the fleet ops console (call-outs panel, Argon audit card, gate badge)
status: claimed
owner: claude-code
branch: feat/copilot-console-panel
depends_on: [T-212]
plan: docs/superpowers/plans/2026-11-14-operator-copilot-plan.md#t-215-show-copilot-output-in-the-console-optional-coordinate-first
claimed_at: '2026-11-14T23:40:31+00:00'
---

The call-outs panel (`{"type": "copilot"}`) landed in #96. This branch adds the rest: it surfaces Tier 2 `GraspGate` status and Tier 3 silent `gemini-4-argon` engineering audits in the Fleet Ops Console without any external URLs (`http://` / `https://` forbidden in `src/teleop/console/web/*`):

- **Gate badge (`#gate`)**: Shows parallel `SkillValidator` rollout score alongside the `gemini-4-hydrogen` physical grasp verdict (`approve` / `reject` / `physics_only`).
- **Audit second-opinion card (`#audit-panel`, `web/audit.js`)**: Renders `status` (`pending | complete | skipped | error`), `verdict` (`approve | flag`), `dataset_tag` (`approve | flag`), `confidence`, engineering `reasons`, `operator_note`, `model`, `frame_count`, and `latency_ms`, fed both in-process via `telemetry.py` (`msg["audit"]`) and over WebSocket (`type: "copilot_audit"`) on `ConsoleServer` (never `"key"` messages).

Owned files: `src/teleop/console/{telemetry,server,run}.py`, `src/teleop/console/web/{index.html,console.css,console.js,audit.js}`, `tests/console/*`.
