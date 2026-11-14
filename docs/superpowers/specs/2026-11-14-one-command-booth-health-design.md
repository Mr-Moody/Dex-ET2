# One-command booth launch with live hardware health

Date: 2026-11-14. Status: design for review.

## Intent

Running the booth today takes several commands and flags (`--source studio`, `--hotspot`,
`--unoq-host`, a separate `teleop unoq --serve` on the board) and the operator cannot see from
the console whether the hardware is actually connected. The operator wants **one command**,
`uv run teleop atc`, that brings everything up, and a console that shows at a glance whether the
**hotspot**, the **Rokoko gloves** and the **control panel (Arduino UNO Q)** are connected, with a
**Reconnect** button for each one that is not.

The demo must never be blocked by hardware. Anything that is down falls back to what works
today (keyboard, local UNO Q loopback, scripted rescue), and picks the real device back up when it
returns.

## Goals

1. `uv run teleop atc` with no flags: hotspot up, Studio glove listener bound, UNO Q found or
   mocked, browser console open.
2. A health strip in the ops console with three lights (Hotspot, Gloves, Control panel), each
   with a one-line reason and, when not green, a Reconnect button.
3. Hot-swap: a device that comes back (or is reconnected) takes over without restarting the
   process. A device that drops degrades to the fallback without stopping the 60 Hz loop.
4. The UNO Q auto-runs its edge service on boot (App Lab app) and waits for the laptop. Its
   physical console (joystick, clutch, E-stop, bay LEDs, LED matrix) works end to end through
   that service.

## Non-goals

- Starting or configuring Rokoko Studio from the laptop. The glove check diagnoses; the fix in
  Studio is the operator's.
- SSH or remote process management of the UNO Q.
- A pre-flight gate page (option B, rejected below).
- Health for compliance (Solana) and reasoner (Verda). They already show their mode in the
  header label and are not in scope.
- Non-Windows hotspot support.

## Anti-goals (would make this a failure even if it works)

- A reconnect or health poll that blocks or stutters the 60 Hz control loop or the render
  thread.
- A green light that lies: "Control panel" green when the board answers pings but the MCU
  buttons are not heard. The light must reflect what the operator actually relies on.
- Changing behaviour of `--source synthetic`, `--headless`, `--script` runs or the CI headless
  check (`uv run teleop atc --source synthetic --headless --script --no-audio` must still exit 0
  with no hardware, no hotspot attempt and no network wait beyond today's).
- Switching glove input mid-rescue (a glove dropout must not hand the bay to the scripted
  rescue halfway through a human rescue).

## Constraints

- Global Constraints of `docs/superpowers/plans/2026-11-13-m0-m1-glove-to-sim.md` (Python 3.12,
  uv only, no hardware in tests, `@pytest.mark.hardware` for hardware-only tests, strict
  pyright).
- File ownership (`docs/superpowers/plans/2026-11-14-act-recovery-plan.md`):
  `commands/atc.py` and `configs/atc.yaml` belong to T-190 (in review); `console/telemetry.py`
  and `console/web/*` belong to T-197 (todo); `atc/stage.py` to T-190/T-197. This work must
  land **after T-190 merges**, keep its logic in new modules, and touch those files only at small
  hook points, so it rebases cleanly against T-197. `src/teleop/cli.py` is not touched (no new
  command).
- The UNO Q side (`teleop.unoq.server`, `mcu_bridge`) runs on the board's Linux; `mcu_bridge`
  is stdlib-only and must stay so.
- Hotspot code only exists on Windows (`teleop.net.winrt_backend`).

## Approaches

- **A. Health strip inside the existing ops console (chosen).** The sim and console start
  immediately; a strip at the top of the console shows the three lights with reasons and
  Reconnect buttons; missing devices fall back and are hot-swapped back in. Chosen because the
  demo is never stuck waiting, and status plus fixes live on the page the operator is already
  presenting from.
- **B. Pre-flight page before the console.** Blocks on three checks (or "Start anyway") and only
  then starts the sim. Clearer "ready" moment, but a mid-demo glove dropout still needs
  in-console status, so it means building both. Rejected.
- **C. Terminal `teleop doctor`.** Rejected: the operator asked for the checks in the dashboard.

## Design

### 1. One command (config defaults)

New `configs/atc.yaml` keys (and matching `AtcConfig` fields), with `source` switched to
`studio`:

```yaml
source: studio          # studio falls back to the scripted rescue when no glove packets
hotspot: true           # keep the booth hotspot up (Windows; ignored with --headless)
```

- `--no-hotspot` flag disables it; the existing `--hotspot` flag stays and forces it on.
- `--headless` implies no hotspot and never waits on hardware.
- Tests, README and the scripted demo keep passing `--source synthetic` explicitly.
- A missing `HOTSPOT_PASSWORD` or non-Windows host is **not** fatal any more under the config
  default: the Hotspot light shows "not configured: set HOTSPOT_PASSWORD in .env" and the rest
  starts. (`--hotspot` given explicitly keeps today's fail-fast exit 1.)
- `unoq_host: unoq.local` stays the default; nothing new to configure for the board.

### 2. Health model (new package `src/teleop/health/`)

```python
class Level(StrEnum):
    OK, WARN, FAIL, OFF  # OFF = disabled/not configured, grey


@dataclass(frozen=True)
class Health:
    name: str  # "hotspot" | "gloves" | "panel"
    level: Level
    summary: str  # short, e.g. "streaming 83 Hz"
    detail: str  # one line, what to do, e.g. "no packets for 4 s: is Studio streaming to <laptop-ip>:14043?"
    can_reconnect: bool
    busy: bool  # a reconnect is in progress


class HealthCheck(Protocol):
    def status(self) -> Health: ...  # cheap, non-blocking, called at ~2 Hz
    def reconnect(self) -> None: ...  # may block for seconds; never called on the loop thread
```

A `HealthMonitor` owns the three checks, runs `reconnect()` on a single worker thread (one at a
time per check; a second click while busy is ignored), and exposes `snapshot() -> list[Health]`.

#### Hotspot check

- Wraps the existing `Hotspot` / `HotspotWatchdog` (`teleop.net.hotspot`), using
  `Hotspot.status()`.
- OK: state on, configured SSID, `client_count >= 1` (summary "up, N clients").
  WARN: on with 0 clients ("up, no clients: are the gloves powered and on this hotspot?").
  Clients are the gloves and the UNO Q; Rokoko Studio runs on this laptop (CLAUDE.md: the
  laptop hosts gloves, Studio and hotspot), so Studio is not a client.
  FAIL: off or last watchdog event is an error. OFF: disabled, headless, non-Windows, or no
  password (detail says which).
- Reconnect: `Hotspot.ensure_up()` (the watchdog keeps running).

#### Gloves check

- `StudioSource` gains liveness counters, updated on its receive thread: last datagram time,
  last parsed right-hand frame time, datagram rate over the last second, sender address. Exposed
  as a read-only `liveness()` method. (`teleop.input.studio` is not owned by an active task.)
- OK: right-hand frames within the last 0.5 s ("streaming 83 Hz").
  WARN: datagrams arriving but no right-hand frames ("packets but no right hand: check the actor
  or glove assignment in Studio"). FAIL: socket bound, nothing for > 2 s ("no packets for N s: is Studio
  running with custom streaming to 127.0.0.1:14043, and are the gloves connected in Studio?"); or bind failed ("port 14043 in use").
  OFF: `--source synthetic` or `replay`.
- Reconnect: stop and restart the source (closes and rebinds the UDP socket).
- Studio runs on this laptop, so no IP discovery is needed; the hint names the configured
  `studio.port`. The sender address is logged, not shown.

#### Glove fallback (no mid-rescue switch)

The operator input is chosen **when a rescue starts** (clutch engaged on the queued bay): if the
Gloves check is OK, the gloves drive that rescue; otherwise the scripted rescue (today's
`--source synthetic` behaviour) plays, and the console shows a toast "gloves offline: playing
scripted rescue". The choice holds until the clutch releases. A dropout during a glove rescue
holds the last pose (today's behaviour), never switches to scripted.

**Episode provenance.** When `--record-episodes` is on, each recorded rescue must carry the
input that actually drove it (`"glove"` or `"scripted"`), not the startup-wide label `main()`
computes today. Scripted fallback rescues must never be saved as human glove episodes, because
they feed ACT training (T-191..T-198). This is an interface requirement on T-191's episode sink
(`atc/episodes.py`): a per-episode source label passed at rescue start. Coordinate with T-191
before it lands; if T-191 has already landed, add the parameter there in this work.

#### Control panel check (UNO Q)

Two things are checked, and the light reflects both:

- **Board link**: the laptop's UNO Q client answers `ping`.
- **Buttons**: the board reports a fresh MCU heartbeat (new field, see §4).

Levels: OK: board linked and MCU heartbeat < 0.5 s old ("UNO Q 192.168.137.5, buttons live").
WARN: board linked, MCU silent ("board up but buttons not heard: is the console sketch
running?"). FAIL: no board, running on the local loopback node ("UNO Q not found at unoq.local:
using local mock; keyboard works"). The fallback is today's: the loopback node for classify and
fit, keyboard for console input.

Reconnect (also attempted automatically every 10 s while on the loopback node): try
`unoq_host`; on success, swap the live client in (see §3).

### 3. UNO Q hot-swap

**Loop-thread budget.** `classify_stall`, `fit_dmp` and `register_skill` stay synchronous on the
loop thread (moving them off it would restructure T-190's coordinator). To keep that from
stuttering the loop:

- The panel check pings the board on its **own** connection every 1 s (timeout 0.3 s). Two
  missed pings mark the board FAIL and swap to loopback **proactively**, so the coordinator's
  next request already goes to the local node. In practice a board that is unplugged or rebooted
  is detected between stalls, not during one.
- Only a board that dies in the exact window of a loop-thread request costs one hitch of up to
  `REQUEST_TIMEOUT_S` (2 s) plus the retry on loopback. This is accepted as rare; request
  timeouts are not shortened (a legitimate `fit_dmp` on the board must not time out).
- Skill replay to a new node happens **outside** the swap lock, on the monitor's worker thread,
  against the new client before it is published. The lock covers only the pointer swap, so a
  loop-thread request never waits behind a replay. Skills registered during the replay are
  replayed again before publishing (loop until the recorded list is unchanged).


The coordinator, `UnoQConsolePoller` and the stage hold one `UnoQClient` today. Introduce a
`SwitchableUnoQ` (new, in `src/teleop/health/` or `src/teleop/unoq/switch.py`) with the same
methods the coordinator calls (`classify_stall`, `fit_dmp`, `register_skill`, `filter_step`,
`console_state`, `ping`, `is_mock`, `close`), forwarding to the current client under a lock.

- `swap(new_client)`: before switching, re-register every skill this process has registered
  (the switch remembers `register_skill` calls in order) on the new node, so the learned fleet
  library survives a mock -> board swap (and board reboot). Then switch and close the old client
  (stopping the loopback server if it was the mock).
- A request that fails on the live board with `OSError`/`UnoQError` marks the panel FAIL, swaps
  to a fresh loopback node (with skills replayed) and retries once there, so a board dropping
  mid-classify does not break the rescue flow.
- Never swap while a request is in flight (the lock covers this).
- The `UnoQConsolePoller` uses the switch's current host and is restarted on swap. Today the
  stage's `hardware` console list is fixed at build time; it needs a way to add or replace the
  poller after a mock -> board swap (a small hook on `StageController`).
- The header `unoq_label` is refreshed on swap.

**Laptop serial console.** The current sketch talks only through `arduino-router`, so
`--serial-port` is not a booth configuration. The path stays as-is for other firmware; when
`serial_port` is set, the panel light uses the local `McuConsoleBridge` heartbeat for "buttons"
and the board ping for "board link", and Reconnect reopens the serial port.

### 4. Board side: auto-running edge service with the physical console

New App Lab app under `firmware/unoq_app/` (sketch: existing `firmware/unoq_console/`
contents; Python: a small `main.py`) set to run at startup. It:

1. Starts `UnoQServer` on `0.0.0.0:15051/15052`.
2. Opens the MCU console with `open_console()` (on the board this resolves to
   `RouterBridgeTransport`), wraps it in `McuConsoleBridge`, and on every console state change
   calls `server.set_console_state(...)`.
3. Adds the MCU heartbeat age to the `console_state` reply (`mcu_heartbeat_age_ms`, `null` when
   never heard) so the laptop can tell "board up" from "buttons live".
4. New TCP op `set_bay_state {bay_id, state}` forwarding to the bridge as
   `CMD:BAY_STATE:<n>:<TOKEN>` so the bay LEDs and the matrix mirror the fleet. The laptop sends
   it from the same place the stage mirrors bay states today (`console.set_bay_state`), via the
   `UnoQConsolePoller`, only on change.
5. Waits forever for connections; no laptop needed to boot it.

`teleop unoq --serve` gains the same bridge wiring (a `--console` flag, default on when
`open_console()` returns the router transport) so the App Lab `main.py` is a thin call into it.

### 5. Console: health strip and reconnect messages

- Telemetry JSON gets a top-level `health` list (the `Health` fields above), merged in at the
  publish site in `console/run.py` / the render worker, not inside `TelemetryAdapter` (T-197's
  file). Published at ~2 Hz, independent of frame rate.
- New client -> server message `{"type": "reconnect", "target": "hotspot"|"gloves"|"panel"}`.
  `ConsoleServer` gains an `on_reconnect` callback alongside `on_key`, validated against the
  three names, and hands it to `HealthMonitor` (worker thread, never the asyncio loop).
- Web: a strip above the camera wall, one pill per check:

  ```
  ┌ Hotspot ● up, 2 clients ┐ ┌ Gloves ● no packets 4 s  [Reconnect] ┐ ┌ Panel ● UNO Q 192.168.137.5, buttons live ┐
  ```

  Green / amber / red / grey dot; summary always visible; detail as the pill's tooltip and shown
  inline when not green; Reconnect visible when `can_reconnect` and level is not OK, disabled
  with a spinner while `busy`. A change to FAIL also raises a toast through the existing toast
  path. Keep it in a separate `health.js` / section of CSS where possible to minimise overlap
  with T-197's edits.
- MuJoCo viewer mode: one extra HUD overlay line with the three levels (no buttons).

## Decided by the operator

- Option A (health strip in the live console, fall back and hot-swap).
- UNO Q auto-runs its service on boot and waits for the laptop to connect.
- Single command is `uv run teleop atc`.

## Calls I made (correct any)

- `source: studio` and `hotspot: true` become the config defaults; synthetic runs pass flags.
- Missing hotspot password is no longer fatal under the default (only under explicit
  `--hotspot`).
- Glove fallback decided at rescue start, never mid-rescue.
- A board dying mid-request may cost one ~2 s loop hitch; proactive 1 s pings make that rare.
  Classify/fit stay on the loop thread.
- Each recorded episode is labelled glove or scripted by what actually drove it.
- Studio runs on the laptop, so the glove hint points at 127.0.0.1 and no IP discovery is built.
- `--serial-port` kept but not a booth setup; when used, the panel light reads its heartbeat.
- Thresholds: gloves OK < 0.5 s, FAIL > 2 s; MCU heartbeat OK < 0.5 s; auto board retry every
  10 s; health published at 2 Hz.
- Skill library replayed onto the new node on every UNO Q swap.
- The panel light requires the MCU heartbeat to be green, not just the board ping.

## Left to the builder

- Exact App Lab app layout (`app.yaml`, how the repo's `teleop` package is installed or vendored
  on the board, run-at-startup setting): research against current App Lab docs and the board.
- Whether `SwitchableUnoQ` lives in `teleop.unoq` or `teleop.health`.
- Pill styling within the existing console look.
- Task breakdown and IDs in `tasks/`, sequenced after T-190, and coordination notes for T-197,
  T-191 (episode label) and T-198 (which also edits `configs/atc.yaml` defaults).
