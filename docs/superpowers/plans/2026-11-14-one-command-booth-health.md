# One-Command Booth Launch with Live Hardware Health: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `uv run teleop atc` with no flags brings up the hotspot, the Studio glove listener, the UNO Q
(or its loopback mock) and the browser console. The console shows Hotspot, Gloves and Control panel
health with Reconnect buttons, and hot-swaps devices in and out without stopping the 60 Hz loop.

**Architecture:** A new `teleop.health` package holds three `HealthCheck`s and a `HealthMonitor`
that polls them at 2 Hz on its own thread and runs reconnects on a worker thread. The console
publishes the monitor's cached snapshot inside the telemetry JSON and forwards
`{"type":"reconnect"}` messages to it. A `SwitchableUnoQ` proxy lets the coordinator's UNO Q client
be swapped between the board and the loopback node. The board runs an App Lab app that serves the
edge service with the MCU console bridge attached.

**Tech Stack:** Python 3.12, uv, pytest, pyright strict, MuJoCo, `websockets`, plain HTML/CSS/JS,
Arduino App Lab (UNO Q).

**Spec:** `docs/superpowers/specs/2026-11-14-one-command-booth-health-design.md`

## Global Constraints

- Everything in the Global Constraints of `docs/superpowers/plans/2026-11-13-m0-m1-glove-to-sim.md`
  (Python 3.12, uv only, xyzw quaternions, joints by name, no hardware in tests,
  `@pytest.mark.hardware` for hardware-only tests).
- Verify before every PR: `uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest`.
- No reconnect, health poll or UNO Q swap may run on the 60 Hz loop thread or the console's asyncio
  loop. `HealthCheck.status()` must not block (no I/O). `reconnect()` may block.
- `uv run teleop atc --source synthetic --headless --script --no-audio` must still exit 0 with no
  hardware, no hotspot attempt and no extra network wait.
- `teleop.unoq.mcu_bridge` stays stdlib-only.
- Do not edit `src/teleop/cli.py`. Edit `console/telemetry.py`, `console/render.py` and
  `atc/coordinator.py` not at all; edit `console/web/*` and `atc/stage.py` only where a task says
  (T-197 owns them next).
- Thresholds: gloves OK if a right-hand frame arrived within 0.5 s, FAIL after 2.0 s with no
  datagram; MCU heartbeat OK within 0.5 s; board ping every 1.0 s with 0.3 s timeout, 2 misses =
  FAIL and proactive swap to loopback; automatic board retry every 10 s while on loopback; health
  poll at 2 Hz.
- Health JSON shape (one object per check, in this order: `hotspot`, `gloves`, `panel`):
  `{"name": str, "level": "ok"|"warn"|"fail"|"off", "summary": str, "detail": str, "can_reconnect": bool, "busy": bool}`.
- Conventional Commits. No `Co-Authored-By` or "Generated with" lines.

## Review Focus

1. **Reconnect clicked repeatedly or for a check that is busy.** The second click is ignored, with
   no second thread and no exception. Pinned in Task 1 (`test_reconnect_ignored_while_busy`).
2. **Board unplugged between stalls.** The next `classify_stall` goes to loopback with no 2 s
   hitch, and learned skills are still matchable. Pinned in Task 5
   (`test_board_loss_swaps_before_next_request`).
3. **`HOTSPOT_PASSWORD` missing on a default run.** The process starts and the Hotspot pill is grey
   with the fix in its detail. With explicit `--hotspot` it exits 1 as today. Pinned in Task 8
   (`test_default_run_without_password_starts`).
4. **Gloves drop mid-rescue.** The rescue keeps glove input (holding the last pose) and does not
   switch to scripted. Pinned in Task 6 (`test_glove_dropout_mid_rescue_holds`).
5. **Malformed reconnect messages from the page** (unknown target, non-string, missing field). They
   are ignored and logged at debug, never raise. Pinned in Task 7
   (`test_reconnect_message_validation`).

---

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `src/teleop/health/__init__.py` | Re-exports | 1 |
| `src/teleop/health/model.py` | `Level`, `Health`, `HealthCheck` protocol, `health_to_dict` | 1 |
| `src/teleop/health/monitor.py` | `HealthMonitor`: 2 Hz poll thread, cached snapshot, reconnect worker | 1 |
| `src/teleop/input/studio.py` | Add `StudioLiveness` and `StudioSource.liveness()` | 2 |
| `src/teleop/health/gloves.py` | `GloveCheck` | 2 |
| `src/teleop/health/hotspot.py` | `HotspotCheck`, `build_hotspot_check` | 3 |
| `src/teleop/unoq/protocol.py` | `OP_SET_BAY_STATE`, `mcu_heartbeat_age_ms` in console dict | 4 |
| `src/teleop/unoq/server.py` | `attach_console(bridge)`, `set_bay_state` op | 4 |
| `src/teleop/unoq/client.py` | `set_bay_state()`; `console_state()` also returns heartbeat age | 4 |
| `src/teleop/commands/unoq.py` | `--serve` attaches the MCU console (`--no-console` to skip) | 4 |
| `firmware/unoq_app/` | App Lab app (auto-run) wrapping the sketch and `teleop unoq --serve` | 4 |
| `src/teleop/unoq/switch.py` | `SwitchableUnoQ` | 5 |
| `src/teleop/health/panel.py` | `PanelCheck` (ping thread, proactive swap, auto retry) | 5 |
| `src/teleop/atc/stage.py` | `UnoQConsolePoller` forwards bay states; `StageController.replace_hardware()` | 5 |
| `src/teleop/atc/glove_select.py` | `SelectingGlove`: glove vs scripted chosen per rescue | 6 |
| `src/teleop/console/server.py` | `on_reconnect` callback and message validation | 7 |
| `src/teleop/console/health_publish.py` | `HealthPublisher` wrapper adding `health` to telemetry | 7 |
| `src/teleop/console/run.py` | `run_console(..., health=None)` | 7 |
| `src/teleop/console/web/health.js`, `index.html`, `console.css` | Health strip | 7 |
| `src/teleop/atc/hud.py` | Viewer overlay health line | 7 |
| `src/teleop/commands/atc.py`, `configs/atc.yaml` | Wiring, defaults, `--no-hotspot` | 8 |
| `README.md` | One-command run, health strip, UNO Q app | 8 |

---

### Task 1: Health model and monitor

**Files:**
- Create: `src/teleop/health/__init__.py`, `src/teleop/health/model.py`, `src/teleop/health/monitor.py`
- Test: `tests/health/__init__.py`, `tests/health/test_monitor.py`

**Interfaces:**
- Produces:
  - `class Level(StrEnum): OK = "ok"; WARN = "warn"; FAIL = "fail"; OFF = "off"`
  - `@dataclass(frozen=True) class Health: name: str; level: Level; summary: str; detail: str = ""; can_reconnect: bool = False; busy: bool = False`
  - `class HealthCheck(Protocol): name: str; def status(self) -> Health: ...; def reconnect(self) -> None: ...`
  - `def health_to_dict(h: Health) -> dict[str, object]` (the shape in Global Constraints)
  - `class HealthMonitor: __init__(self, checks: Sequence[HealthCheck], period_s: float = 0.5); start(); stop(); poll_once() -> None; snapshot() -> list[Health]; snapshot_dicts() -> list[dict[str, object]]; request_reconnect(name: str) -> bool` (False when the name is unknown or that check is busy)

- [ ] **Step 1: Write the failing tests** in `tests/health/test_monitor.py`, using a `FakeCheck` whose `reconnect()` blocks on a `threading.Event`:
  - `test_snapshot_reflects_last_poll`: after `poll_once()`, `snapshot()` equals the fakes' `status()` values in order.
  - `test_reconnect_runs_off_caller_thread`: `request_reconnect("gloves")` returns True immediately while the fake is still blocked; the fake records a thread name that is not the caller's.
  - `test_busy_is_reported_while_reconnecting`: while blocked, `poll_once()` then `snapshot()` shows `busy=True` for that check only; after release and another poll, `busy=False`.
  - `test_reconnect_ignored_while_busy`: a second `request_reconnect("gloves")` while blocked returns False and the fake's call count stays 1.
  - `test_unknown_name_rejected`: `request_reconnect("nope") is False`.
  - `test_status_exception_becomes_fail`: a check whose `status()` raises yields `Health(level=FAIL, summary="check error", detail=<exception text>)`, and the monitor keeps polling.
  - `test_health_to_dict_shape`: the keys and value types exactly match Global Constraints.
- [ ] **Step 2: Run** `uv run pytest tests/health/test_monitor.py -v`. Expected: FAIL (module not found).
- [ ] **Step 3: Implement** `model.py` and `monitor.py`. `HealthMonitor` overrides `busy` on the cached `Health` with `dataclasses.replace`. Reconnects run on a `concurrent.futures.ThreadPoolExecutor(max_workers=3, thread_name_prefix="health-reconnect")`, at most one in flight per check. The poll thread is a daemon and polls once immediately on `start()`.
- [ ] **Step 4: Run** the tests. Expected: PASS. Run `uv run pyright src/teleop/health`. Expected: 0 errors.
- [ ] **Step 5: Commit** `feat(health): health model and monitor with off-thread reconnect`.

### Task 2: Glove liveness and GloveCheck

**Files:**
- Modify: `src/teleop/input/studio.py` (`StudioSource`)
- Create: `src/teleop/health/gloves.py`
- Test: `tests/input/test_studio.py` (add), `tests/health/test_gloves.py`

**Interfaces:**
- Consumes: Task 1 `Health`, `Level`.
- Produces:
  - `@dataclass(frozen=True) class StudioLiveness: bound: bool; port: int | None; bind_error: str | None; last_datagram_ns: int | None; last_right_frame_ns: int | None; datagram_hz: float`
  - `StudioSource.liveness(self) -> StudioLiveness` (thread-safe, non-blocking). `start()` failing to bind records `bind_error` instead of raising, **only** when constructed with `tolerate_bind_error=True` (new keyword, default False, so today's callers keep their behaviour).
  - `class GloveCheck: name = "gloves"; __init__(self, source: StudioSource | None, port: int, clock: Callable[[], int] = time.monotonic_ns)`. `source=None` means not using Studio (OFF).

- [ ] **Step 1: Write the failing tests.**
  - In `tests/input/test_studio.py`: `test_liveness_counts_datagrams`. Start a `StudioSource` on `127.0.0.1` port 0, send three datagrams built with the existing `_payload(...)` helper, and poll up to 1 s: `last_right_frame_ns` is not None and `datagram_hz > 0`. `test_liveness_bind_error_tolerated`: bind a socket to a port, then start a second source with `tolerate_bind_error=True` on that port. `liveness().bound is False` and `bind_error` mentions the port.
  - In `tests/health/test_gloves.py` (a fake source returning fixed `StudioLiveness`, a fake clock):
    - `test_ok_when_recent_right_frame`: frame 0.2 s ago, `datagram_hz=83.0` → `OK`, summary `"streaming 83 Hz"`.
    - `test_warn_packets_without_right_hand`: datagram 0.1 s ago, frame None → `WARN`, detail contains `"no right hand"`.
    - `test_fail_no_packets`: bound, last datagram 4.2 s ago → `FAIL`, summary `"no packets for 4 s"`, detail contains `"127.0.0.1:14043"`, `can_reconnect=True`.
    - `test_fail_never_received`: bound, nothing ever → `FAIL`, summary `"waiting for Studio"`.
    - `test_fail_bind_error`: detail contains `"port 14043 in use"`.
    - `test_off_without_source`: `GloveCheck(None, 14043).status().level is OFF`, summary `"synthetic glove"`, `can_reconnect=False`.
    - `test_reconnect_restarts_source`: `reconnect()` calls `stop()` then `start()` on the fake.
- [ ] **Step 2: Run** `uv run pytest tests/input/test_studio.py tests/health/test_gloves.py -v`. Expected: the new tests FAIL.
- [ ] **Step 3: Implement.** Liveness fields are updated in `_run` under a small lock. `datagram_hz` is the count over a rolling 1 s window (a `deque` of timestamps). The OK/WARN boundary is a right-hand frame within 0.5 s. FAIL needs > 2.0 s since the last datagram. Between 0.5 s and 2.0 s, report `WARN` "glove stream stalling".
- [ ] **Step 4: Run** the tests. Expected: PASS.
- [ ] **Step 5: Commit** `feat(health): Studio glove liveness and glove health check`.

### Task 3: Hotspot check

**Files:**
- Create: `src/teleop/health/hotspot.py`
- Test: `tests/health/test_hotspot_check.py`

**Interfaces:**
- Consumes: `teleop.net.hotspot.Hotspot`, `HotspotWatchdog`, `load_password`, `HotspotConfigError`; `teleop.app.start_hotspot_watchdog`.
- Produces:
  - `class HotspotCheck: name = "hotspot"; __init__(self, hotspot: Hotspot | None, watchdog: HotspotWatchdog | None, off_reason: str = "")`
  - `def build_hotspot_check(enabled: bool, env_file: Path, *, strict: bool) -> tuple[HotspotCheck, HotspotWatchdog | None]`. When `enabled` is False, it returns OFF `"disabled"`. On a non-Windows platform it returns OFF `"Windows only"`. When `load_password` raises `HotspotConfigError`, it re-raises if `strict`, otherwise returns OFF with detail `"set HOTSPOT_PASSWORD in .env"`. Otherwise it starts the watchdog (bring-up runs on the watchdog thread, not inline) and returns a live check.
- `start_hotspot_watchdog` in `teleop.app` must expose its `Hotspot`. Add `HotspotWatchdog.hotspot` as a read-only property in `net/hotspot.py` rather than changing the function's return type.

- [ ] **Step 1: Write the failing tests** with the existing fake tethering backend pattern from `tests/net/test_hotspot.py`:
  - `test_ok_with_clients`: state on, configured SSID, 2 clients → `OK`, summary `"up, 2 clients"`.
  - `test_warn_no_clients`: on, 0 clients → `WARN`, detail contains `"gloves"` and `"UNO Q"`.
  - `test_fail_off`: state off → `FAIL`, `can_reconnect=True`.
  - `test_fail_on_last_watchdog_error`: watchdog `events[-1]` is `"error: ..."` → `FAIL`, detail is that text.
  - `test_reconnect_calls_ensure_up`: the fake backend's `start` is called.
  - `test_status_does_not_call_backend`: `status()` reads a value cached by the watchdog poll, so backend call counts don't change. This keeps WinRT calls off the console thread. Cache `Hotspot.status()` in `HotspotWatchdog.poll_once()` and expose it as `HotspotWatchdog.last_status`.
  - `test_build_without_password_not_strict_is_off` and `test_build_without_password_strict_raises` (monkeypatch `sys.platform` to `"win32"` and the env file to an empty temp file).
- [ ] **Step 2: Run** `uv run pytest tests/health/test_hotspot_check.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `hotspot.py`, plus the `hotspot` and `last_status` additions on `HotspotWatchdog`.
- [ ] **Step 4: Run** `uv run pytest tests/health tests/net -v`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(health): hotspot health check over the watchdog`.

### Task 4: Board-side console bridge, bay-state op and App Lab app

**Files:**
- Modify: `src/teleop/unoq/protocol.py`, `src/teleop/unoq/server.py`, `src/teleop/unoq/client.py`, `src/teleop/commands/unoq.py`
- Create: `firmware/unoq_app/` (App Lab layout; see Step 5), `firmware/unoq_app/README.md`
- Test: `tests/unoq/test_server_console.py`

**Interfaces:**
- Consumes: `McuConsoleBridge`, `SoftwareMcu`, `open_console` (`teleop.unoq.mcu_bridge`).
- Produces:
  - `protocol.OP_SET_BAY_STATE = "set_bay_state"`, added to `TCP_OPS`. Request `{"op": "set_bay_state", "bay_id": int, "state": str}`, where `state` is a `BayState` value name.
  - The `console_state` reply gains `"mcu_heartbeat_age_ms": int | None`.
  - `UnoQServer.attach_console(self, bridge: McuConsoleBridge) -> None`. It starts a daemon thread that copies `bridge.state()` into `set_console_state` every 50 ms and records the bridge's last-heard time for the heartbeat age.
  - `UnoQClient.set_bay_state(self, bay_id: int, state: BayState) -> None` and `UnoQClient.console_status(self) -> tuple[ConsoleInputState, int | None]` (state plus heartbeat age; `console_state()` is kept).
  - `teleop unoq --serve [--no-console]`: by default it calls `open_console()` and attaches the bridge unless the result kind is `"software"`. In that case it logs `"no MCU console found"` and serves without one.

- [ ] **Step 1: Write the failing tests** (`UnoQServer` on `127.0.0.1` port 0, `McuConsoleBridge(SoftwareMcu())`, a real `UnoQClient`):
  - `test_console_state_reflects_mcu_buttons`: `mcu.press_bay(2); mcu.set_clutch(True)`. Within 0.5 s, `client.console_status()[0]` has `selected_bay == 2` and `clutch is True`.
  - `test_heartbeat_age_reported`: with the bridge attached and the MCU pumping, the age is `< 500`. With no bridge attached, it is `None`.
  - `test_set_bay_state_reaches_mcu`: `client.set_bay_state(1, BayState.STALLED)` writes `CMD:BAY_STATE:1:STALLED` to the `SoftwareMcu` (assert on what it received).
  - `test_set_bay_state_without_console_is_ok`: no bridge → reply ok, no error.
  - `test_set_bay_state_rejects_bad_state`: `{"state": "BOGUS"}` → error reply, server keeps serving.
- [ ] **Step 2: Run** `uv run pytest tests/unoq/test_server_console.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** the protocol, server, client and command changes. `mcu_bridge.py` must stay stdlib-only. If the bridge lacks a last-heard accessor, add `McuConsoleBridge.last_heard_ns` there.
- [ ] **Step 4: Run** `uv run pytest tests/unoq -v`. Expected: PASS.
- [ ] **Step 5: Create the App Lab app** in `firmware/unoq_app/`. First check the current Arduino App Lab app layout (manifest file name, `python/` and `sketch/` folders, how Python dependencies are declared, and the run-at-startup setting) against Arduino's UNO Q documentation. The sketch is a copy of `firmware/unoq_console/unoq_console.ino`. `python/main.py` calls `teleop.commands.unoq.main(["--serve"])`. `README.md` explains how `teleop` gets onto the board, how to deploy the app, how to set it to run at startup, and how to check it from the laptop with `uv run teleop unoq --host <board-ip>`. Mark the on-board check as manual (hardware).
- [ ] **Step 6: Commit** `feat(unoq): serve the MCU console and bay states from the edge service, App Lab auto-run app`.

### Task 5: UNO Q hot-swap and Control panel check

**Files:**
- Create: `src/teleop/unoq/switch.py`, `src/teleop/health/panel.py`
- Modify: `src/teleop/atc/stage.py` (`UnoQConsolePoller` only, plus a new `StageController.replace_hardware` method)
- Test: `tests/unoq/test_switch.py`, `tests/health/test_panel.py`

**Interfaces:**
- Consumes: Task 4 `UnoQClient.set_bay_state`, `UnoQClient.console_status`; Task 1 `Health`.
- Produces:
  - `class SwitchableUnoQ`. Same public methods the coordinator and stage use: `filter_step`, `classify_stall`, `fit_dmp`, `register_skill`, `console_state`, `ping`, `is_mock` (property), `close`. Plus:
    - `host: str` (property)
    - `current: UnoQClient` (property)
    - `swap(self, new: UnoQClient) -> None`: replays the remembered `register_skill` calls on `new` outside the lock, repeating until no new skills arrived during the replay, then swaps under the lock and closes the old client.
    - `swap_to_loopback(self) -> None`
    - `on_swap: list[Callable[[SwitchableUnoQ], None]]`
  - On `OSError`/`UnoQError` from the live board inside `classify_stall`, `fit_dmp` or `register_skill`: call `swap_to_loopback()` and retry once on loopback. A failure on loopback propagates.
  - `class PanelCheck: name = "panel"; __init__(self, unoq: SwitchableUnoQ, host: str, udp_port: int, tcp_port: int, *, ping_period_s: float = 1.0, ping_timeout_s: float = 0.3, retry_period_s: float = 10.0, clock=time.monotonic_ns)`, plus `start()` and `stop()`. Its own thread does two jobs. While on the board, it pings on a separate `UnoQClient` connection; after 2 consecutive misses it calls `unoq.swap_to_loopback()`. While on loopback, it tries `UnoQClient(host, ...).connect()` plus a ping every `retry_period_s` and calls `unoq.swap(client)` on success. `reconnect()` runs that same board attempt immediately.
  - `PanelCheck(..., local_bridge: McuConsoleBridge | None = None)`. When given (`--serial-port` set), "buttons" reads the age of `local_bridge.last_heard_ns` instead of the board-reported heartbeat. `reconnect()` also closes the serial transport and reopens it with `open_console(serial_port)` (pass `serial_port: str | None = None` too). Test: `test_panel_uses_local_bridge_heartbeat` (a `SoftwareMcu`-backed bridge, a board linked over loopback servers, `OK` while the MCU pumps).
  - `UnoQConsolePoller.set_bay_state` forwards to `client.set_bay_state` only when the state changed for that bay. Failures are logged at debug and dropped. The poller stores the heartbeat age from `console_status()`; `heartbeat_age_ms` is a property.
  - `StageController.replace_hardware(self, consoles: Sequence[ConsoleSource]) -> None`: closes the old hardware consoles and installs the new list. It is called from the `on_swap` callback; the list assignment is atomic and the tick reads `self.consoles` once per tick.

- [ ] **Step 1: Write the failing tests.** Use two real `UnoQServer`s on `127.0.0.1` port 0, one as the "board" and one as loopback, so no hardware is needed.
  - `test_swap_replays_skills`: register a skill through the switch on server A, `swap()` to a client of server B, and `classify_stall` on B matches that skill.
  - `test_failover_retries_on_loopback`: stop server A mid-test. `classify_stall` returns the loopback answer (no exception), and `is_mock` is True afterwards.
  - `test_board_loss_swaps_before_next_request`: a `PanelCheck` with a fake clock and a stopped server A, plus two `poll()` steps (expose `PanelCheck.step()` for tests). Then `unoq.is_mock` is True, and the next `classify_stall` completes in < 0.1 s.
  - `test_auto_retry_returns_to_board`: on loopback, server A comes back, and after `retry_period_s` of fake time `step()` swaps to it (`is_mock` False, `host` is A's).
  - `test_panel_levels`: board linked with heartbeat 120 ms → `OK`, summary contains `"buttons live"`. Board linked, heartbeat None → `WARN`, detail contains `"console sketch"`. Loopback → `FAIL`, summary contains `"not found"`, detail contains `"keyboard works"`, `can_reconnect=True`.
  - `test_poller_forwards_bay_state_on_change_only`: two identical `set_bay_state(1, STALLED)` calls produce one request on the server.
- [ ] **Step 2: Run** `uv run pytest tests/unoq/test_switch.py tests/health/test_panel.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.** In `switch.py`, the lock guards only the client pointer and the in-flight request. Replay uses the new client directly before publication. Type the coordinator parameter so pyright accepts `SwitchableUnoQ` without editing `coordinator.py`: if `AtcCoordinator.build` is annotated `UnoQClient`, make `SwitchableUnoQ` structurally compatible and cast at the single call site in `commands/atc.py` (Task 8), with a comment.
- [ ] **Step 4: Run** `uv run pytest tests/unoq tests/health tests/atc -v`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(unoq): hot-swappable UNO Q client and control panel health check`.

### Task 6: Glove or scripted rescue chosen per rescue

**Files:**
- Create: `src/teleop/atc/glove_select.py`
- Test: `tests/atc/test_glove_select.py`

**Interfaces:**
- Consumes: `GloveOperator` (`commands/atc.py`, `latest() -> OperatorInput | None`, `running()` context manager), `ScriptedGlove` (`atc/stage.py`), Task 1 `Level`, Task 2 `GloveCheck`.
- Produces:
  - `class SelectingGlove: __init__(self, controller: StageController, glove: GloveOperatorLike, scripted: ScriptedGlove, glove_ok: Callable[[], bool])`, plus `latest() -> OperatorInput | None` and `running()`. `GloveOperatorLike` is a Protocol with `latest()` and `running()`.
  - `rescue_input: Literal["glove", "scripted"] | None`, which is None when no rescue is engaged. The episode recorder (T-191) reads this attribute per episode.
  - On the engage edge (clutch on and selected bay equal to `controller.rescue_bay()`) it picks glove if `glove_ok()`, otherwise scripted. In the scripted case it emits a stage action note through the existing action log with the text `"gloves offline: playing scripted rescue"`. The pick holds until the engage condition goes false.

- [ ] **Step 1: Write the failing tests** with a fake controller (`rescue_bay()`, `console`, action log) and fake glove and scripted objects returning sentinels:
  - `test_glove_used_when_ok_at_engage`
  - `test_scripted_used_when_glove_down_at_engage`: also asserts the action note text.
  - `test_glove_dropout_mid_rescue_holds`: engage with OK, then `glove_ok` goes False; `latest()` keeps returning the glove's value and `rescue_input == "glove"`.
  - `test_choice_resets_on_release`: release, glove still down, then re-engage → scripted.
  - `test_not_engaged_returns_none`: returns None from both paths and `rescue_input is None`.
- [ ] **Step 2: Run** `uv run pytest tests/atc/test_glove_select.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.** If `StageController` has no public way to add an action note, add a one-line `note(text: str, level: str = "warn")` method in `stage.py` that uses the existing `ActionNote` path. That is the only `stage.py` edit in this task.
- [ ] **Step 4: Run** the tests. Expected: PASS.
- [ ] **Step 5: Commit** `feat(atc): choose glove or scripted rescue at clutch engage`.

### Task 7: Console health strip and reconnect messages

**Files:**
- Modify: `src/teleop/console/server.py`, `src/teleop/console/run.py`, `src/teleop/console/web/index.html`, `src/teleop/console/web/console.css` (append a `/* health strip */` section only), `src/teleop/atc/hud.py`
- Create: `src/teleop/console/health_publish.py`, `src/teleop/console/web/health.js`
- Test: `tests/console/test_server.py` (add), `tests/console/test_health_publish.py`, `tests/console/test_web_assets.py` (add), `tests/atc/test_hud.py` (add, or the existing HUD test file)

**Interfaces:**
- Consumes: Task 1 `HealthMonitor.snapshot_dicts()`, `request_reconnect(name)`.
- Produces:
  - `ConsoleServer(..., on_reconnect: Callable[[str], None] | None = None)`. It accepts `{"type": "reconnect", "target": t}` only when `t` is in `{"hotspot", "gloves", "panel"}`. The callback is invoked on the asyncio thread and must be non-blocking; `request_reconnect` is.
  - `class HealthPublisher`: wraps an object with `publish_telemetry` and `publish_frame`. On `publish_telemetry(msg)` it adds `msg["health"] = monitor.snapshot_dicts()` and forwards. `publish_frame` passes through.
  - `run_console(..., health: HealthMonitor | None = None)`. When given, it passes `HealthPublisher(server, health)` to `RenderWorker` in place of `server`, and passes `on_reconnect=health.request_reconnect` (ignoring the bool) to `ConsoleServer`.
  - `hud_overlays` / `hud_lines` gain an optional `health: Sequence[Health] | None` and add one line like `HOTSPOT ok | GLOVES fail | PANEL warn`.

- [ ] **Step 1: Write the failing tests.**
  - `test_reconnect_message_validation`: `_on_message` with a valid target calls `on_reconnect("gloves")`. Unknown target, non-string target, missing target, and `on_reconnect=None` do nothing and raise nothing. Key messages still work.
  - `test_health_publisher_adds_health`: a fake inner server receives `msg` with `"health"` equal to the monitor's dicts. Frames pass through untouched.
  - `test_run_console_without_health_unchanged`: the existing `test_run.py` path still passes with `health=None`, and the telemetry has no `health` key.
  - Web assets: `index.html` includes `health.js` and a `#health` container. `health.js` references `"reconnect"`, `"hotspot"`, `"gloves"` and `"panel"`, and contains no CDN URLs.
  - HUD: with three `Health` values the overlay contains `"GLOVES fail"`.
- [ ] **Step 2: Run** `uv run pytest tests/console tests/atc -v`. Expected: the new tests FAIL.
- [ ] **Step 3: Implement the server, publisher, run and HUD changes.**
- [ ] **Step 4: Implement the web strip.** `health.js` registers a telemetry listener. Hook it with a minimal change in `console.js`: one exported callback list, or a `CustomEvent("telemetry", {detail: msg})` dispatched where telemetry is parsed. The listener renders one pill per entry into `#health`, placed above the camera wall: a dot coloured by `level` (green, amber, red, grey), the summary, the detail inline when level is not ok (tooltip otherwise), and a Reconnect button when `can_reconnect && level !== "ok"`. The button is disabled with a spinner while `busy` and sends `{"type":"reconnect","target":name}` on the existing socket. A transition to `fail` calls the existing `toast()`. With no `health` key in telemetry the strip stays hidden. The strip must not push the camera wall below the fold at 1366×768.
- [ ] **Step 5: Run** `uv run pytest tests/console tests/atc -v` and `uv run python scripts/console_preview.py`. Check the strip renders with fake data; if the preview has no health data, add three fake entries to its telemetry, including one `fail` with `can_reconnect`.
- [ ] **Step 6: Commit** `feat(console): hardware health strip with reconnect buttons`.

### Task 8: One-command wiring, defaults and README

**Files:**
- Modify: `src/teleop/commands/atc.py`, `configs/atc.yaml`, `README.md`
- Test: `tests/commands/test_atc_health.py` (or the existing `teleop atc` command test module)

**Interfaces:**
- Consumes: Tasks 1–7.
- Produces:
  - `AtcConfig.hotspot: bool = True`. YAML: `hotspot: true` and `source: studio`.
  - CLI: `--no-hotspot` (sets False). The existing `--hotspot` forces True and makes a missing password fatal (`strict=True`), keeping today's exit 1.
  - `--headless` implies no hotspot, no `PanelCheck` thread, and no `HealthMonitor`. The UNO Q uses today's single `connect_or_mock` path, wrapped in `SwitchableUnoQ`.
  - Non-headless wiring:
    - `build_hotspot_check(cfg.hotspot, env_file, strict=ns.hotspot)`.
    - `StudioSource(..., tolerate_bind_error=True)` inside `GloveOperator` when `source == "studio"`, with `GloveCheck(source, cfg.studio.port)`.
    - `SwitchableUnoQ` passed to `StageController.build`, with `PanelCheck(...)` started.
    - On `on_swap`: `controller.replace_hardware(_hardware_consoles(cfg, unoq))` and refresh `controller.unoq_label`.
    - `SelectingGlove(controller, glove_operator, ScriptedGlove(controller), lambda: glove_check.status().level is Level.OK)` as the operator when `source == "studio"` and not `--script`.
    - `HealthMonitor([hotspot_check, glove_check, panel_check])` started and passed to `run_console(health=...)`. In viewer mode, health goes to the HUD overlay.
  - Episode label: when `--record-episodes` is on and the operator is a `SelectingGlove`, pass the hook a callable source label, `lambda: selecting.rescue_input or "glove"`, instead of the startup string. If T-191's hook signature only accepts `str`, leave the string path in place and add a `TODO(T-191)` comment plus a note in `tasks/T-191-*.md` describing the per-episode label requirement from the spec.

- [ ] **Step 1: Write the failing tests** (no hardware: monkeypatch `UnoQClient.connect_or_mock` and the hotspot builder):
  - `test_default_config_is_one_command`: `load_config(DEFAULT_CONFIG, {})` has `source == "studio"` and `hotspot is True`.
  - `test_headless_skips_hotspot_and_health`: with `--source synthetic --headless --script --no-audio`, the hotspot builder is not called, `PanelCheck` is not started, and the exit code is 0.
  - `test_default_run_without_password_starts`: non-strict build with no password → OFF check, and `main` proceeds (stub `run_console` to return immediately).
  - `test_explicit_hotspot_without_password_exits_1`.
  - `test_no_hotspot_flag_disables`.
- [ ] **Step 2: Run** `uv run pytest tests/commands -v`. Expected: the new tests FAIL.
- [ ] **Step 3: Implement the wiring** in `commands/atc.py` and update `configs/atc.yaml`.
- [ ] **Step 4: Update `README.md`.**
  - "Run the demo": lead with `uv run teleop atc` as the booth command.
  - Health strip: a short section on what each light means and what Reconnect does.
  - Mark `--hotspot` and `--unoq-host` as optional overrides.
  - Link to `firmware/unoq_app/README.md` for the board.
  - Keep the synthetic, scripted and headless rows (they now need `--source synthetic` explicitly; check every example).
- [ ] **Step 5: Full verify:** `uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest`. Expected: all pass. Then run `uv run teleop atc --source synthetic --headless --script --no-audio`. Expected: exit 0, all 6 beats ok.
- [ ] **Step 6: Manual hardware check (this laptop):**
  - `uv run teleop atc` opens the console with three pills.
  - Stop Studio streaming → Gloves turns red within about 2 s. Restart it and click Reconnect → green.
  - Unplug the UNO Q → Panel turns red and the keyboard still works. Plug it back → green within 10 s, and the bay LEDs follow the fleet.
- [ ] **Step 7: Commit** `feat(atc): one-command booth launch with live hardware health`.

---

## Task dependencies

Task 1 comes first. Tasks 2, 3 and 4 are independent of each other. Task 5 needs 4. Task 6 needs 2.
Task 7 needs 1. Task 8 needs all of them.
