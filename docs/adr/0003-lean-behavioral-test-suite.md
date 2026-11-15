# ADR 0003: Consolidate Test Suite into a Lean Behavioral Contract Budget

Status: accepted (2026-11-15). Supersedes the "keep all M1 unit tests intact" clause in ADR 0002 Decision 1.

## Context

Parallel wave execution across `T-001`–`T-215` produced one new `test_*.py` file per task, accumulating **81 test files, 809 test functions, and 19,365 LOC** against ~18,900 LOC of `src/teleop` (>1:1 test-to-source ratio). Much of that footprint consisted of:

- Source-file/AST/HTML/CSS/JS string scraping (`.read_text()` + regexes and `// <section>` comment markers embedded in `src/teleop/console/web/console.js`).
- Mock-heavy subprocess/HTTP/Prompt-string assertions (`nmcli`, `ip`, `tts` CLI argv, Gemini prompt wording) and private-attribute pokes (`# pyright: ignore[reportPrivateUsage]`, `cls.__new__(cls)`).
- Legacy M1/M2 walk-up `lift_cube` bimanual booth tests (`test_lift_cube.py`, `test_dummy_glove.py`, `test_rgmp.py`, `test_app.py`) retained solely under ADR 0002 Decision 1's transition clause after the ATC pivot had already settled.

## Decisions

1. **Supersede ADR 0002 Decision 1's M1 Test-Freeze Clause:**
   Keep M1 runtime modules that power the ATC stack (`StudioSource`, `ReplaySource`, `LatestBuffer`, `DirectRetargeter`, `Calibration`, `CoilWrist`/`ImuWrist`/`AutoWrist`, `load_hand`), while retiring standalone walk-up booth tests and collapsing M1 primitive coverage into compact behavioral tests.
2. **Enforce a Hard 12-File / `< 4,000` LOC Suite Budget:**
   Consolidate `tests/` into 12 subsystem-level `test_*.py` modules plus `tests/conftest.py`:
   - `tests/sim/test_latch_panda.py` — MuJoCo MJCF/hand compilation, 3-box latch physics, 7-DoF DLS IK, external torque estimation, 4-bay scene.
   - `tests/test_retarget_and_wrist.py` — relative-clutch indexing, rescue buffer, `twist_angle` Hypothesis properties, retargeting, wrist trackers.
   - `tests/input/test_studio.py` — Rokoko Studio JSON v3 / LZ4 / Y-up→Z-up parser, Hypothesis fuzzing, UDP source, `LatestBuffer`, `ReplaySource`.
   - `tests/unoq/test_edge_dmp.py` — $1\text{€}$ filter, 30-kernel RBF-DMP fit & $SE(3)$ warp, Mahalanobis OOD gate, `unoq` wire protocol & loopback server.
   - `tests/unoq/test_mcu_bridge.py` — STM32U585 serial line parser, one-way fault latch, E-stop/clutch safety invariants, `SwitchableUnoQ` failover.
   - `tests/atc/test_coordinator.py` — touch pose update, `GloveInputFilter`, $C(t)$ stall detection, FIFO queue, Bay 1 rescue → 20-rollout gate → Bay 2 clear, OOD negative control.
   - `tests/atc/test_recovery.py` — byte-for-byte DMP golden guard (`tests/atc/golden/dmp_stage.json`), recovery backend selection/downgrade, batched live ACT replay & abort isolation.
   - `tests/atc/test_data_and_eval.py` — deterministic fault generator, non-blocking `EpisodeSink` & LeRobot v3 sidecar, `atc_qc`, offline `teleop eval` gate & `MirrorRenderer`.
   - `tests/policy/test_policy.py` — latch-relative `STATE_NAMES`/`ACTION_NAMES` roundtrip, `GateManifest`, `InferenceWorker` batching/health trips, `BayCameraRenderer`.
   - `tests/compliance/test_solana.py` — canonical SHA-256 JSON, Ed25519 hash-chain & crash repair, `< 0.5\text{ ms}` non-blocking emit, SPL Memo anchoring & tamper detection.
   - `tests/console/test_server.py` — HTTP whitelist/traversal guard, WebSocket state replay, binary frame headers, message routing, slow-client backpressure, `HudView` JSON serialization.
   - `tests/test_cli_and_devtools.py` — `teleop` CLI subcommand importability/dispatch and `uv run task` atomic git-ref claim/race/release lifecycle.
3. **Ban Low-Signal Test Patterns & Test-Only Hooks in `src/`:**
   - Never add test-only comment markers, CLI flags, or module-global monkeypatch hooks to `src/` when dependency injection suffices.
   - Never scrape source files, HTML/CSS/JS assets, or Python ASTs from `tests/`.
   - Never bypass constructors with `__new__`, poke private `_` attributes (`reportPrivateUsage`), or write unit tests for stdlib dataclasses/enums/constants.

## Consequences

- `tests/` shrinks by >80% (~19.4k LOC → ~3.3k LOC across 12 `test_*.py` files) while keeping every physics, math, wire-protocol, safety, DMP golden, ACT gate, and Solana invariant green.
- Future tasks must extend one of the 12 domain test modules (and delete redundant assertions if needed) rather than creating per-task `test_*.py` files.
