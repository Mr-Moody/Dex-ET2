# Console Glove Calibration Panel (Live Apply)

Date: 2026-11-14. Builds on the fleet ops console
(`docs/superpowers/specs/2026-11-14-fleet-ops-console-design.md`) and the existing CLI
`teleop calibrate` / `DirectRetargeter` pipeline.

## Intent

Without a per-user calibration file, ATC falls back to `Calibration.default` (open = 0,
closed = +1.4 rad). Rokoko Studio finger twists often curl the **opposite** way (and a flat
hand is not identity), so fist maps to “still open” and hyperextension past flat starts to
close the sim hand.

Operators need a **one-click, in-console** flat → fist capture that **hot-swaps** the live
`DirectRetargeter` so mapping can be fixed or re-run for drift without restarting `teleop atc`
or leaving the dashboard.

## Goals

- A **Glove calibration** panel on the console Setup & health page, in its `#calib-slot` (layout C; see `2026-11-14-console-pages-and-data-collection-design.md`): two pose tiles (Flat /
  Fist) with **stroke SVG** hand icons (no emoji), plus a primary **Calibrate gloves** action
  when idle.
- Server-driven wizard: prep delay → collect flat → prep delay → collect fist → build
  `Calibration` → **replace the running retargeter immediately** → optionally persist JSON.
- Status is glanceable on the pose tiles (pending / get-ready / hold countdown / done) with
  **Cancel** anytime before apply.
- Re-runnable any time during a session (drift, bad capture, glove re-seat).
- Degenerate spans (`|closed − open| < MIN_SPAN`) fall back per actuator as today’s
  `DirectRetargeter` already does; UI reports partial vs full success.

## Non-goals

- Changing swing-twist / `flex_value` math or adding `invert` flags in hand YAML (calibration
  already handles reversed spans).
- Booth / visitor attract-mode calibration (`teleop booth`); this is ATC console only.
- Left-hand calibration UX (ATC `GloveOperator` is right-hand only today; collect whatever
  sides Studio sends, but apply ranges for the side(s) the retargeter uses).
- Redesigning the Dummy glove TEMP panel (stays separate; real calib is for Studio gloves).

## Anti-goals

- Blocking the 60 Hz control loop or stalling rescue teleop while waiting on UI.
- Applying a half-finished capture (must Cancel cleanly or only swap after both poses).
- Requiring a terminal / CLI for day-of demo fixes.

## Constraints

- Global Constraints from the M0/M1 plan still apply (Python 3.12, uv, joints by name, xyzw,
  no hardware in unit tests).
- Prefer editing console-owned surfaces (`src/teleop/console/**`, web assets) and
  `GloveOperator` in `commands/atc.py` for hot-swap; reuse `teleop.retarget.calibration`
  / `calibrate()` / collect timing from `commands/calibrate.py` (extract shared helpers if
  needed rather than duplicating).
- Default timing mirrors CLI: **2 s prep** + **2 s hold** per pose (configurable constants OK).
- Persist path default: `data/calib/atc.json` (create parents); failure to write must not
  undo a successful live apply (warn in UI / toast).
- WebSocket: extend client→server messages (e.g. `type: "calib"`, `action: "start"|"cancel"`)
  and telemetry with a small `calib` status object for the panel. No new npm deps; inline SVG
  in the web assets (or a tiny static SVG file under `console/web/`).
- Tests: pure unit tests with a fake clock / fake `InputSource`; no Studio hardware, no
  browser required in CI.

## Approaches

| | Approach | Trade-off |
|---|----------|-----------|
| **A (chosen)** | Console-driven wizard on the ATC/server side; rail panel C with SVG pose tiles; live retargeter swap | Best for drift; timing owned in one place |
| B | Browser-timed open/close windows; server only records & applies | UI flexible; clock splits / WS jitter risk |
| C | Button shells out to `teleop calibrate` and reloads JSON | Weak mid-session apply; fights the dashboard |

**Chosen:** A + panel layout C (two pose cards, stroke SVGs). Mockups lived under
`.superpowers/brainstorm/` (gitignored).

## UI (layout C)

**Idle**

- Rail section titled **Glove calibration**.
- Meta line: e.g. `Idle · defaults` or `Idle · live` / path hint when a calib is loaded.
- Two dashed/muted tiles: Flat (open-hand SVG) and Fist (closed-hand SVG).
- Full-width accent button: **Calibrate gloves**.

**Running**

- Meta: `Step 1 · Flat` / `Step 2 · Fist` (and prep vs hold).
- Active tile: safety-yellow border/fill; hint `Get ready · Ns` then `Hold · Ns`.
- Completed tile: nominal green + check-ish hint `Done`.
- Pending tile: muted.
- **Cancel** (danger outline) replaces the primary button until finished.

**Done / error**

- Toast + meta: `Applied` or `Applied (partial)` / `Cancelled` / `No glove frames`.
- Return to idle with both tiles reflecting last success (optional subtle “live” badge).

Icons: custom **stroke SVGs** matching graphite console (currentColor; yellow/green via CSS),
not emoji or external icon CDNs.

## Behaviour

1. Client sends `calib` / `start`.
2. Server refuses if already running or no glove operator / source; else enters prep for flat.
3. Collect right (and left if present) frames for hold window; same for fist after prep.
4. `calibrate(inspire_spec, open_frames, closed_frames)` → new `DirectRetargeter` with existing
   ctrlranges → assign onto `GloveOperator.retargeter` atomically from the control thread’s
   perspective (queue the swap onto the tick, or lock briefly).
5. Best-effort `to_json` persist; telemetry `calib` state → idle.
6. `cancel` aborts to idle with no retargeter change.

## Builder calls (defaults unless overridden)

- Prep/hold: 2.0 s each (same as CLI).
- Persist: on by default to `data/calib/atc.json`.
- Skip / “use last good”: **out of scope** for v1 (Cancel only); last successful live calib
  remains until overwritten.
- Only right-hand ranges must be non-degenerate for ATC success messaging; left may be
  defaulted if no frames.

## What’s left to the implementer

- Exact WS schema field names and telemetry shape.
- Whether collect helpers live in `retarget/calibration.py` vs a small `console/calib.py`.
- Threading detail for safe retargeter swap vs FixedRateLoop.
- Pixel polish of SVG paths (readable at ~56×56 in the rail).

## Approval

Approve this spec to proceed to an implementation plan, then code.
