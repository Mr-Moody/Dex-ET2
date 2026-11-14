"""Stage controls for ``teleop atc``: keyboard console mirror, pallet cycling, audio cues, script.

:class:`StageController` wraps an :class:`~teleop.atc.coordinator.AtcCoordinator` and runs one
60 Hz tick at a time. It merges the keyboard mirror (a :class:`~teleop.unoq.mcu_bridge.SoftwareMcu`
behind the same :class:`~teleop.unoq.mcu_bridge.McuConsoleBridge` the real board uses) with any
hardware console, restarts finished pallet cycles with whatever fault is armed, mirrors bay states
to the console LEDs, and taps compliance events and reasoning submissions for the HUD and the
audio cues.

Keys (GLFW codes from the passive viewer, mirrors the hardware console):

=========  ===============================================================================
``1..4``   select a bay (the operator takes the queued bay that holds the rescue slot)
``SPACE``  toggle the software dead-man clutch
``F``      Judge Fault Rig: arm Bay 1's sprung flap for its next cycle (beat 2)
``G``      arm Bay 2's sprung flap (35 deg / 12 cm latch) for its next cycle (beat 5)
``P``      arm the Negative Control pinned bolt (x = 0) on Bay 2's next cycle (beat 6)
``R``      reset: every active bay starts a clean cycle, faults disarmed, rig re-armed
=========  ===============================================================================

:class:`StageDirector` plays the 2.5-minute stage script (six beats) by pressing those same keys
and driving a scripted glove rescue, for headless tests and unattended rehearsals.
"""

from __future__ import annotations

import contextlib
import logging
import math
import platform
import queue
import shutil
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable, Generator, Iterable, Iterator, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, Protocol

import numpy as np

from teleop.atc.coordinator import (
    DEFAULT_REALISM,
    ROUTINE,
    AtcCoordinator,
    CoordinatorConfig,
    OperatorInput,
)
from teleop.atc.hud import ActionNote, GateStatus, HudView, RescueStatus
from teleop.atc.recovery import next_backend
from teleop.atc.types import BOLT_CLEARED_M, BaySnapshot, BayState, ConsoleInputState
from teleop.atc.validator import GATE_PASS_MIN, ValidationReport
from teleop.compliance.protocol import ComplianceEvent, ComplianceSink
from teleop.reasoning.protocol import FleetReasoner, ReasoningContext, ReasoningVerdict
from teleop.sim.bays import ACTIVE_BAYS, BACKGROUND_BAYS
from teleop.types import Pose
from teleop.unoq.client import UnoQClient
from teleop.unoq.dmp import quat_to_mat
from teleop.unoq.mcu_bridge import McuConsoleBridge, SoftwareMcu

logger = logging.getLogger(__name__)

CONTROL_DT = 1.0 / 60.0


# --------------------------------------------------------------------------- keys


class StageAction(StrEnum):
    SELECT_BAY_1 = "select_bay_1"
    SELECT_BAY_2 = "select_bay_2"
    SELECT_BAY_3 = "select_bay_3"
    SELECT_BAY_4 = "select_bay_4"
    CLUTCH = "clutch"
    ARM_BAY1_FLAP = "arm_bay1_flap"
    ARM_BAY2_FLAP = "arm_bay2_flap"
    ARM_BAY2_PIN = "arm_bay2_pin"
    RESET = "reset"
    CYCLE_BACKEND = "cycle_backend"


KEY_SPACE = 32
KEYMAP: dict[int, StageAction] = {
    ord("1"): StageAction.SELECT_BAY_1,
    ord("2"): StageAction.SELECT_BAY_2,
    ord("3"): StageAction.SELECT_BAY_3,
    ord("4"): StageAction.SELECT_BAY_4,
    KEY_SPACE: StageAction.CLUTCH,
    ord("F"): StageAction.ARM_BAY1_FLAP,
    ord("G"): StageAction.ARM_BAY2_FLAP,
    ord("P"): StageAction.ARM_BAY2_PIN,
    ord("R"): StageAction.RESET,
    ord("M"): StageAction.CYCLE_BACKEND,
}
ACTION_LOG_LEN = 20
CONTROL_KEYS_HINT = "Use 1-4, SPACE, F, G, P, R or M"


def key_name(code: int) -> str:
    """Readable name for a GLFW key code: ``SPACE``, ``F``, ``1``."""
    if code == KEY_SPACE:
        return "SPACE"
    if 32 < code < 127:
        return chr(code)
    return f"key {code}"


_SELECT = {
    StageAction.SELECT_BAY_1: 1,
    StageAction.SELECT_BAY_2: 2,
    StageAction.SELECT_BAY_3: 3,
    StageAction.SELECT_BAY_4: 4,
}


# --------------------------------------------------------------------------- consoles


class ConsoleSource(Protocol):
    """An operator console: the keyboard mirror, a serial MCU bridge or a UNO Q poller."""

    def state(self) -> ConsoleInputState: ...
    def take_fault_rig_edge(self) -> bool: ...
    def set_bay_state(self, bay_id: int, state: BayState) -> None: ...
    def reset_rig(self) -> None: ...
    def close(self) -> None: ...


def merge_console(states: Iterable[ConsoleInputState]) -> ConsoleInputState:
    """Combine consoles: latest bay selection wins, any clutch holds, any E-stop releases."""
    items = list(states)
    if not items:
        return ConsoleInputState()
    selected = [s for s in items if s.selected_bay is not None]
    bay = max(selected, key=lambda s: s.timestamp_ns).selected_bay if selected else None
    estop = any(s.estop for s in items)
    return ConsoleInputState(
        selected_bay=bay,
        clutch=any(s.clutch for s in items) and not estop,
        estop=estop,
        fault_rig_latched=any(s.fault_rig_latched for s in items),
        timestamp_ns=max(s.timestamp_ns for s in items),
    )


class UnoQConsolePoller:
    """Console state from a remote UNO Q (``OP_CONSOLE_STATE``), polled on its own thread.

    Uses its own client connection so polling never interleaves with the coordinator's
    classify/fit requests. The clutch is dropped when no poll has succeeded for
    ``timeout_s`` (same dead-man rule as the serial bridge). Bay states for the board LEDs are
    queued by :meth:`set_bay_state` (only on change) and sent from the polling thread, so the
    control loop never waits on the network.
    """

    def __init__(
        self,
        client: UnoQClient,
        period_s: float = 0.05,
        timeout_s: float = 0.25,
        clock: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self._client = client
        self._period_s = period_s
        self._timeout_ns = int(timeout_s * 1e9)
        self._clock = clock
        self._lock = threading.Lock()
        self._state = ConsoleInputState()
        self._last_ok_ns: int | None = None
        self._heartbeat_age_ms: int | None = None
        self._edge = False
        self._bay_states: dict[int, BayState] = {}  # last state requested per bay
        self._pending: dict[int, BayState] = {}  # changed, not yet sent
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="unoq-console", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(self._period_s):
            self.poll_once()

    def poll_once(self) -> None:
        """Send queued bay states, then poll the console once (the thread's body)."""
        with self._lock:
            pending, self._pending = self._pending, {}
        for bay_id, bay_state in pending.items():
            try:
                self._client.set_bay_state(bay_id, bay_state)
            except Exception as exc:
                logger.debug("unoq bay state %d dropped: %s", bay_id, exc)
        try:
            s, age = self._client.console_status()
        except Exception as exc:
            logger.debug("unoq console poll failed: %s", exc)
            return
        with self._lock:
            if s.fault_rig_latched and not self._state.fault_rig_latched:
                self._edge = True
            self._state = s
            self._heartbeat_age_ms = age
            self._last_ok_ns = self._clock()

    @property
    def heartbeat_age_ms(self) -> int | None:
        """MCU heartbeat age the board reported on the last successful poll (None: never)."""
        with self._lock:
            return self._heartbeat_age_ms

    @property
    def last_ok_ns(self) -> int | None:
        """Clock time of the last successful poll (None: never); the panel check reads it."""
        with self._lock:
            return self._last_ok_ns

    def state(self) -> ConsoleInputState:
        with self._lock:
            s, last = self._state, self._last_ok_ns
        stale = last is None or self._clock() - last > self._timeout_ns
        return replace(s, clutch=False) if s.clutch and stale else s

    def take_fault_rig_edge(self) -> bool:
        with self._lock:
            edge, self._edge = self._edge, False
        return edge

    def set_bay_state(self, bay_id: int, state: BayState) -> None:
        with self._lock:
            if self._bay_states.get(bay_id) == state:
                return
            self._bay_states[bay_id] = state
            self._pending[bay_id] = state

    def reset_rig(self) -> None:
        pass

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._client.close()


# --------------------------------------------------------------------------- audio


class Speaker(Protocol):
    def say(self, text: str) -> None: ...
    def close(self) -> None: ...


class NullSpeaker:
    def say(self, text: str) -> None:
        pass

    def close(self) -> None:
        pass


def _speech_command(text: str) -> list[str] | None:
    """Platform TTS command line for ``text``, or None when no synthesizer is available."""
    system = platform.system()
    if system == "Windows":
        quoted = text.replace("'", "''")
        script = (
            "Add-Type -AssemblyName System.Speech; "
            "(New-Object System.Speech.Synthesis.SpeechSynthesizer)"
            f".Speak('{quoted}')"
        )
        return ["powershell", "-NoProfile", "-NonInteractive", "-Command", script]
    if system == "Darwin":
        return ["say", text]
    for tool in ("spd-say", "espeak-ng", "espeak"):
        if shutil.which(tool):
            return [tool, text]
    return None


class SystemSpeaker:
    """Speaks cues with the OS synthesizer on a worker thread; never blocks the control loop.

    Cues are played in order; if more than ``backlog`` are waiting the oldest are dropped, so
    the voice never lags the stage by more than a couple of lines.
    """

    def __init__(self, backlog: int = 2) -> None:
        self._q: queue.Queue[str | None] = queue.Queue(maxsize=backlog)
        self._thread = threading.Thread(target=self._loop, name="atc-audio", daemon=True)
        self._thread.start()

    def say(self, text: str) -> None:
        while True:
            try:
                self._q.put_nowait(text)
                return
            except queue.Full:
                with contextlib.suppress(queue.Empty):
                    self._q.get_nowait()

    def _loop(self) -> None:
        while (text := self._q.get()) is not None:
            cmd = _speech_command(text)
            if cmd is None:
                logger.info("audio cue: %s", text)
                continue
            try:
                subprocess.run(cmd, check=False, capture_output=True, timeout=15)
            except (OSError, subprocess.SubprocessError) as exc:
                logger.warning("audio cue failed (%s): %s", exc, text)

    def close(self) -> None:
        with contextlib.suppress(queue.Full):
            self._q.put_nowait(None)


class AudioCues:
    """Maps compliance events to short spoken status prompts."""

    def __init__(self, speaker: Speaker, enabled: bool = True) -> None:
        self.speaker = speaker
        self.enabled = enabled

    @staticmethod
    def text_for(event: ComplianceEvent) -> str | None:
        meta = event.metadata
        match event.kind:
            case "STALL_ALERT":
                return f"Bay {event.bay_id} stalled. Queued for rescue."
            case "OOD_REFUSAL":
                return "Out of distribution fault. Refusing skill transfer."
            case "VALIDATION_GATE":
                if meta.get("passed"):
                    return "Skill validated. Broadcasting to fleet."
                return "Skill rejected by the validation gate."
            case "RESCUE_LOGGED" if meta.get("autonomous"):
                return f"Bay {event.bay_id} cleared autonomously with a fleet skill."
            case _:
                return None

    def on_event(self, event: ComplianceEvent) -> str | None:
        text = self.text_for(event)
        if text is not None and self.enabled:
            self.speaker.say(text)
        return text if self.enabled else None

    def close(self) -> None:
        self.speaker.close()


# --------------------------------------------------------------------------- hook taps


class EventTap:
    """``ComplianceSink`` that forwards to the real sink and keeps events for the HUD/cues."""

    def __init__(self, inner: ComplianceSink, history: int = 64) -> None:
        self.inner = inner
        self.history: deque[ComplianceEvent] = deque(maxlen=history)
        self._pending: list[ComplianceEvent] = []

    def emit(self, event: ComplianceEvent) -> None:
        self.history.append(event)
        self._pending.append(event)
        try:
            self.inner.emit(event)
        except Exception:
            logger.exception("compliance sink failed")

    def drain(self) -> list[ComplianceEvent]:
        events, self._pending = self._pending, []
        return events

    def latest(self) -> ComplianceEvent | None:
        return self.history[-1] if self.history else None

    def status_line(self) -> str:
        event = self.latest()
        if event is None:
            return "SOLANA  no events yet"
        line = f"SOLANA {event.kind} bay {event.bay_id} #{event.payload_sha256[:12]}"
        latest_status: Any = getattr(self.inner, "latest_status", None)
        if callable(latest_status):
            try:
                status: Any = latest_status()
            except Exception:
                status = None
            if status is not None and getattr(status, "digest", None):
                line += f" [{getattr(status, 'state', '?')}]"
        mode = getattr(self.inner, "mode", None)
        if isinstance(mode, str):
            line += f" ({mode})"
        return line

    def close(self) -> None:
        self.inner.close()


class ReasonerTap:
    """``FleetReasoner`` that forwards and remembers which bay was triaged last."""

    def __init__(self, inner: FleetReasoner) -> None:
        self.inner = inner
        self.last_bay: int | None = None

    def submit(self, ctx: ReasoningContext) -> None:
        self.last_bay = ctx.bay_id
        try:
            self.inner.submit(ctx)
        except Exception:
            logger.exception("fleet reasoner failed")

    def latest_verdict(self, bay_id: int) -> ReasoningVerdict | None:
        return self.inner.latest_verdict(bay_id)

    def status_line(self) -> str:
        if self.last_bay is None:
            return "NEMOTRON  idle"
        try:
            verdict = self.inner.latest_verdict(self.last_bay)
        except Exception:
            verdict = None
        if verdict is None:
            return f"NEMOTRON bay {self.last_bay}: triaging..."
        return f"NEMOTRON bay {self.last_bay}: {verdict.headline}"

    def close(self) -> None:
        self.inner.close()


# --------------------------------------------------------------------------- controller


GATE_CONDITIONS = "randomized pose, friction, flap spring and rest angle"

# Half-range of each randomized condition, to rank which failing rollout condition was the most
# extreme: friction +-25%, flap stiffness +-20%, springref +-3 deg, y error 3 mm, yaw 1.5 deg.
_COND_SCALE = (0.25, 0.20, 3.0, 3.0, 1.5)


def worst_failure(
    report: ValidationReport | None, nominal_springref_deg: float = 0.0
) -> str | None:
    """Plain-words most extreme condition across the failed rollouts, or ``None`` if none failed.

    Deterministic: ranks each condition by its value over the randomized half-range and takes
    the largest (first one wins a tie), e.g. ``"failed at +23% friction, 2.8 mm y error"``. The
    flap rest angle is ranked by its offset from ``nominal_springref_deg`` (the model's rest angle).
    """
    if report is None:
        return None
    best: tuple[float, str] | None = None
    for r in report.results:
        if r.passed:
            continue
        c = r.conditions
        parts = (
            (c.friction_scale - 1.0) * 100.0,
            (c.flap_stiffness_scale - 1.0) * 100.0,
            c.flap_springref_deg - nominal_springref_deg,
            abs(c.est_error[1]) * 1e3,
            abs(math.degrees(c.est_error[2])),
        )
        sev = (
            abs(parts[0]) / 100.0 / _COND_SCALE[0],
            abs(parts[1]) / 100.0 / _COND_SCALE[1],
            abs(parts[2]) / _COND_SCALE[2],
            parts[3] / _COND_SCALE[3],
            parts[4] / _COND_SCALE[4],
        )
        i = max(range(5), key=lambda k: sev[k])
        text = (
            f"{parts[0]:+.0f}% friction",
            f"{parts[1]:+.0f}% flap stiffness",
            f"{parts[2]:+.1f} deg flap rest angle",
            f"{parts[3]:.1f} mm y error",
            f"{parts[4]:.1f} deg yaw error",
        )[i]
        if best is None or sev[i] > best[0]:
            best = (sev[i], text)
    return None if best is None else f"failed at {best[1]}"


class StageController:
    """One stage tick: keys -> console -> pallet cycling -> coordinator -> LEDs, HUD, cues."""

    def __init__(
        self,
        coordinator: AtcCoordinator,
        events: EventTap,
        reasoning: ReasonerTap,
        *,
        audio: AudioCues | None = None,
        hardware: Sequence[ConsoleSource] = (),
        cycle_dwell_s: float = 1.0,
        background_bays: tuple[int, ...] = (),
        unoq_label: str = "",
    ) -> None:
        self.coordinator = coordinator
        self.active_bays = tuple(s.bay_id for s in coordinator.snapshots()) or ACTIVE_BAYS
        self.events = events
        self.reasoning = reasoning
        self.audio = audio
        self.cycle_dwell_s = cycle_dwell_s
        self.background_bays = background_bays
        self.unoq_label = unoq_label
        self.mcu = SoftwareMcu()
        self.keyboard = McuConsoleBridge(self.mcu)
        self.hardware = list(hardware)
        self.console = ConsoleInputState()
        self.beat = ""
        self.armed: dict[int, tuple[bool, bool]] = {}  # bay -> (flap_jammed, bolt_pinned)
        self.active_faults: dict[int, str] = {}  # bay -> "flap" | "pin" on its current cycle
        self.cleared_cycles: dict[int, int] = dict.fromkeys(self.active_bays, 0)
        self.gate = GateStatus()
        self._keys: deque[int] = deque()
        self.actions: deque[ActionNote] = deque(maxlen=ACTION_LOG_LEN)
        self._action_seq = 0
        self._done_since: dict[int, float] = {}
        self._rig_spent = False
        self._rescue_bay: int | None = None
        self._rescue_t0 = 0.0

    @classmethod
    def build(
        cls,
        unoq: UnoQClient,
        sink: ComplianceSink,
        reasoner: FleetReasoner,
        *,
        audio: AudioCues | None = None,
        hardware: Sequence[ConsoleSource] = (),
        cycle_dwell_s: float = 1.0,
        include_background_bays: bool = False,
        unoq_label: str = "",
        seed: int = 0,
        realism: float = DEFAULT_REALISM,
        gate_pass_min: int = GATE_PASS_MIN,
    ) -> StageController:
        """``seed`` seeds the pose estimator and the skill gate (``CoordinatorConfig.seed``).

        ``realism`` scales the estimate error and the gate randomization (``CoordinatorConfig``).
        """
        events, reasoning = EventTap(sink), ReasonerTap(reasoner)
        coordinator = AtcCoordinator.build(
            unoq,
            events,
            reasoning,
            include_background_bays=include_background_bays,
            config=CoordinatorConfig(seed=seed, realism=realism, gate_pass_min=gate_pass_min),
        )
        return cls(
            coordinator,
            events,
            reasoning,
            audio=audio,
            hardware=hardware,
            cycle_dwell_s=cycle_dwell_s,
            background_bays=BACKGROUND_BAYS if include_background_bays else (),
            unoq_label=unoq_label,
        )

    @property
    def consoles(self) -> list[ConsoleSource]:
        return [self.keyboard, *self.hardware]

    def replace_hardware(self, consoles: Sequence[ConsoleSource]) -> None:
        """Install new hardware consoles and close the old ones (e.g. after a UNO Q swap).

        Called from the swap thread: the list assignment is atomic and the tick snapshots
        ``consoles`` once for its edge, merge and LED passes. A closed console tolerates a
        late call (the poller drops it).
        """
        old, self.hardware = self.hardware, list(consoles)
        for console in old:
            if any(console is c for c in self.hardware):
                continue
            try:
                console.close()
            except Exception:
                logger.exception("console close failed")

    # ------------------------------------------------------------------ keys

    def on_key(self, keycode: int) -> None:
        """Passive-viewer ``key_callback``; queued and handled on the control thread."""
        self._keys.append(int(keycode))

    def _handle_keys(self) -> None:
        while self._keys:
            code = self._keys.popleft()
            action = KEYMAP.get(code)
            if action is None:
                self._note(key_name(code), f"Not a control key. {CONTROL_KEYS_HINT}", "warn")
                continue
            warning = self._precheck(action)
            if warning is not None:
                self._note(key_name(code), warning, "warn")
                continue
            self.apply(action)
            text, level = self._outcome(action)
            self._note(key_name(code), text, level)

    def _note(self, key: str, text: str, level: str) -> None:
        self._action_seq += 1
        self.actions.append(ActionNote(self._action_seq, key, text, level))
        logger.info("key %s: %s", key, text)

    def note(self, text: str, level: str = "warn") -> None:
        """Add a system line to the action log (no key behind it)."""
        self._note("sys", text, level)

    @property
    def rescue_elapsed_s(self) -> float:
        """Seconds since the current bay took the rescue slot (0 when none holds it)."""
        return 0.0 if self._rescue_bay is None else self.coordinator.time_s - self._rescue_t0

    def rescue_bay(self) -> int | None:
        """The bay waiting for, or under, an operator rescue (not one being validated)."""
        co = self.coordinator
        for bay in co.rescue_slot_holders():
            if co.state(bay) in (BayState.RESCUE, BayState.OOD_ESCALATED):
                return bay
        return None

    def _precheck(self, action: StageAction) -> str | None:
        """Why ``action`` would do nothing right now, or ``None`` if it applies."""
        if action in _SELECT and _SELECT[action] not in self.active_bays:
            bay = _SELECT[action]
            if bay in self.background_bays:
                return f"Bay {bay} replays a recorded cycle and cannot be selected"
            return f"Bay {bay} is not running in this simulation"
        if action is StageAction.ARM_BAY1_FLAP:
            if 1 not in self.active_bays:
                return "Bay 1 is not running in this simulation"
            if self.armed.get(1, (False, False))[0]:
                return "Bay 1 flap jam is already armed for its next cycle"
            if self._rig_spent:
                return "Bay 1 is already running its jammed cycle. Press R to reset the rig"
        if (
            action in (StageAction.ARM_BAY2_FLAP, StageAction.ARM_BAY2_PIN)
            and 2 not in self.active_bays
        ):
            return "Bay 2 is not running in this simulation"
        return None

    def _outcome(self, action: StageAction) -> tuple[str, str]:
        """What ``action`` just did, as ``(text, level)`` for the operator displays."""
        rescue = self.rescue_bay()
        if action in _SELECT:
            bay = _SELECT[action]
            if rescue == bay:
                return f"Bay {bay} selected. Engage the clutch (SPACE) to drive it", "ok"
            if rescue is not None:
                return f"Bay {bay} selected, but Bay {rescue} is the one waiting for rescue", "warn"
            return f"Bay {bay} selected", "ok"
        if action is StageAction.CLUTCH:
            selected = self.mcu.selected_bay
            if self.mcu.clutch:
                if rescue is None:
                    return "Clutch engaged, but no bay is waiting for rescue", "warn"
                if selected != rescue:
                    shown = f"Bay {selected}" if selected is not None else "no bay"
                    return (
                        f"Clutch engaged, but {shown} is selected. Press {rescue} to drive "
                        f"Bay {rescue}",
                        "warn",
                    )
                return f"Clutch engaged: your gloves now drive Bay {rescue}", "ok"
            if rescue is not None and selected == rescue:
                return f"Clutch released on Bay {rescue}: checking whether the bolt cleared", "info"
            return "Clutch released", "info"
        if action is StageAction.ARM_BAY1_FLAP:
            return "Bay 1 flap jam armed: Bay 1 stalls on its next cycle", "ok"
        if action is StageAction.ARM_BAY2_FLAP:
            return "Bay 2 flap jam armed: Bay 2 stalls on its next cycle", "ok"
        if action is StageAction.ARM_BAY2_PIN:
            return (
                "Bay 2 bolt pinned for its next cycle: expect a refused stall (negative control)",
                "ok",
            )
        if action is StageAction.CYCLE_BACKEND:
            backend = self.coordinator.recovery_backend.value
            return f"Recovery backend: {backend} (from the next stall)", "ok"
        return "Reset: faults cleared and all bays restarted", "ok"

    def apply(self, action: StageAction) -> None:
        if action in _SELECT:
            self.mcu.press_bay(_SELECT[action])
        elif action is StageAction.CLUTCH:
            self.mcu.toggle_clutch()
        elif action is StageAction.ARM_BAY1_FLAP:
            self.mcu.trigger_fault_rig()  # same path as the judge's physical rig
        elif action is StageAction.ARM_BAY2_FLAP:
            self.arm(2, flap_jammed=True)
        elif action is StageAction.ARM_BAY2_PIN:
            self.arm(2, bolt_pinned=True)
        elif action is StageAction.RESET:
            self.reset()
        elif action is StageAction.CYCLE_BACKEND:
            co = self.coordinator
            co.set_recovery_backend(next_backend(co.recovery_backend))

    def arm(self, bay_id: int, flap_jammed: bool = False, bolt_pinned: bool = False) -> None:
        """Apply a rig fault to the bay's next pallet cycle."""
        if bay_id not in self.active_bays:
            logger.warning("bay %d is not active; fault ignored", bay_id)
            return
        self.armed[bay_id] = (flap_jammed, bolt_pinned)
        logger.info("bay %d armed: flap_jammed=%s bolt_pinned=%s", bay_id, flap_jammed, bolt_pinned)

    def reset(self) -> None:
        self.armed.clear()
        self.active_faults.clear()
        self._done_since.clear()
        self._rig_spent = False
        for console in self.consoles:
            console.reset_rig()
        self.mcu.set_clutch(False)
        for bay in self.active_bays:
            self.coordinator.start_cycle(bay)
        logger.info("stage reset: all active bays restarted")

    # ------------------------------------------------------------------ tick

    def tick(self, operator: OperatorInput | None = None) -> tuple[BaySnapshot, ...]:
        self._handle_keys()
        self.keyboard.pump()
        consoles = self.consoles  # one snapshot per tick: replace_hardware may run concurrently
        for console in consoles:
            if console.take_fault_rig_edge():
                self.arm(1, flap_jammed=True)
        self.console = merge_console(c.state() for c in consoles)
        self._cycle_pallets()
        snaps = self.coordinator.tick(self.console, operator)
        self._after_tick(snaps, consoles)
        return snaps

    def _cycle_pallets(self) -> None:
        """Start the next pallet cycle on a bay whose cycle finished ``cycle_dwell_s`` ago."""
        co = self.coordinator
        now = co.time_s
        for bay in self.active_bays:
            finished = (
                co.state(bay) == BayState.NOMINAL
                and bool(co.outcome(bay))
                and not co.in_routine(bay)
            )
            if not finished:
                self._done_since.pop(bay, None)
                continue
            if bay not in self._done_since:
                self._done_since[bay] = now
                if co.outcome(bay) == "cleared":
                    self.cleared_cycles[bay] += 1
                continue
            if now - self._done_since[bay] < self.cycle_dwell_s:
                continue
            del self._done_since[bay]
            jam, pin = self.armed.pop(bay, (False, False))
            if bay == 1 and self._rig_spent and not jam:
                for console in self.consoles:  # the faulted cycle is over: re-arm the rig
                    console.reset_rig()
                self._rig_spent = False
            if bay == 1 and jam:
                self._rig_spent = True
            if pin or jam:
                self.active_faults[bay] = "pin" if pin else "flap"
            else:
                self.active_faults.pop(bay, None)
            co.start_cycle(bay, flap_jammed=jam, bolt_pinned=pin)

    def _after_tick(
        self, snaps: tuple[BaySnapshot, ...], consoles: Sequence[ConsoleSource]
    ) -> None:
        co = self.coordinator
        holders = co.rescue_slot_holders()
        holder = holders[0] if holders else None
        if holder != self._rescue_bay:
            self._rescue_bay = holder
            self._rescue_t0 = co.time_s
        for snap in snaps:
            for console in consoles:
                console.set_bay_state(snap.bay_id, snap.state)
        for event in self.events.drain():
            self._on_event(event)

    def _on_event(self, event: ComplianceEvent) -> None:
        meta = event.metadata
        if event.kind == "RESCUE_LOGGED" and not meta.get("autonomous"):
            fit = meta.get("fit_ms")
            self.gate = replace(self.gate, fit_ms=float(fit) if fit is not None else None)
        elif event.kind == "VALIDATION_GATE":
            self.gate = replace(
                self.gate,
                passed=int(meta.get("rollouts_passed", 0)),
                total=int(meta.get("rollouts_total", 0)),
                elapsed_s=float(meta.get("elapsed_s", math.nan)),
                ok=bool(meta.get("passed")),
                conditions=GATE_CONDITIONS,
                worst_failure=worst_failure(
                    self.coordinator.last_report, self.coordinator.validator.nominal_springref_deg
                ),
            )
        if self.audio is not None:
            self.audio.on_event(event)

    # ------------------------------------------------------------------ HUD

    def hud_view(self) -> HudView:
        co = self.coordinator
        rescue = RescueStatus()
        if self._rescue_bay is not None:
            engaged = self.console.clutch and self.console.selected_bay == self._rescue_bay
            rescue = RescueStatus(self._rescue_bay, engaged, co.time_s - self._rescue_t0)
        return HudView(
            bays=co.snapshots(),
            queue=co.queue,
            background_bays=self.background_bays,
            selected_bay=self.console.selected_bay,
            rescue=rescue,
            gate=self.gate,
            compliance=self.events.status_line(),
            reasoning=self.reasoning.status_line(),
            beat=self.beat,
            unoq=self.unoq_label,
            actions=tuple(self.actions),
            faults=self._faults_view(),
        )

    def _faults_view(self) -> tuple[tuple[int, str, str], ...]:
        """Faults per bay: armed for the next cycle wins over the one on the current cycle."""
        out: dict[int, tuple[int, str, str]] = {
            bay: (bay, kind, "active") for bay, kind in self.active_faults.items()
        }
        for bay, (jam, pin) in self.armed.items():
            if pin or jam:
                out[bay] = (bay, "pin" if pin else "flap", "armed")
        return tuple(out[bay] for bay in sorted(out))

    def close(self) -> None:
        self.coordinator.close()
        for console in self.consoles:
            try:
                console.close()
            except Exception:
                logger.exception("console close failed")
        if self.audio is not None:
            self.audio.close()


# --------------------------------------------------------------------------- scripted operator

# Scripted glove rescue (duration s, wrist delta from the safe hover in the latch frame in
# metres, finger press fraction): the wrist drops so the index sits low behind the bolt hook,
# the thumb swings under the palm and presses the raised stopper flap flat at its free end,
# then the wrist pushes the bolt clear. Same demonstration the T-140 tests use.
# Kept as tuned on main: a retune for the randomized gate (x 0, y -9 mm, a 2.2 s push) never
# cleared the bolt on an exact latch placement (realism 0), on Windows or Linux.
RESCUE_PRESS = {"thumb_yaw": 0.6}
RESCUE_WAYPOINTS: tuple[tuple[float, tuple[float, float, float], float], ...] = (
    (0.0, (0.0, 0.0, 0.0), 0.0),
    (0.4981, (0.0055, -0.004, -0.012), 0.0),
    (0.8199, (0.0055, -0.004, -0.012), 1.0),
    (0.3333, (0.0055, -0.004, -0.012), 1.0),
    (1.4385, (0.0055 + 0.0458, -0.004, -0.012), 1.0),
)


def scripted_rescue(coordinator: AtcCoordinator, bay_id: int) -> Iterator[OperatorInput]:
    """Glove samples (one per 60 Hz tick) of a flap-hold rescue, anchored at the current hover."""
    rig = coordinator.rig(bay_id)
    hover = rig.wrist_pose()
    if hover.position is None:
        raise ValueError("wrist pose has no position")
    r_world = quat_to_mat(rig.world_from_latch().orientation)
    origin = np.asarray(hover.position, dtype=np.float64)
    base = dict(ROUTINE.fingers)
    for (_, p0, f0), (dur, p1, f1) in zip(RESCUE_WAYPOINTS, RESCUE_WAYPOINTS[1:], strict=False):
        n = max(round(dur / CONTROL_DT), 1)
        for k in range(1, n + 1):
            a = k / n
            delta = np.asarray(p0) + a * (np.asarray(p1) - np.asarray(p0))
            press = f0 + a * (f1 - f0)
            fingers = dict(base)
            for name, v in RESCUE_PRESS.items():
                fingers[name] = base.get(name, 0.0) + press * v
            pos = origin + r_world @ delta
            wrist = Pose(orientation=hover.orientation, position=(pos[0], pos[1], pos[2]))
            yield OperatorInput(wrist, fingers)


class ScriptedGlove:
    """Stand-in glove for ``--source synthetic``: performs the demo rescue on the clutch.

    While the operator holds the clutch on the bay waiting for rescue, each ``latest()`` call
    (one per 60 Hz tick) returns the next sample of :func:`scripted_rescue`, anchored at that
    bay's hover pose. Like the stage director it waits for the 3 cm retract to settle first and
    stops advancing once the bolt is clear, then holds that pose. Releasing the clutch resets
    it, so the next engagement starts a fresh rescue. Otherwise it returns ``None`` and the hand
    holds still (the raw synthetic source just opens and closes every finger).
    """

    SETTLE_S = 0.7  # same hover settle the stage director allows before its scripted rescue
    CLEAR_MARGIN_M = 0.0015

    def __init__(self, controller: StageController) -> None:
        self.controller = controller
        self._bay: int | None = None
        self._samples: Iterator[OperatorInput] | None = None
        self._last: OperatorInput | None = None

    def latest(self) -> OperatorInput | None:
        ctl = self.controller
        bay = ctl.rescue_bay()
        engaged = bay is not None and ctl.console.clutch and ctl.console.selected_bay == bay
        if not engaged:
            self._bay, self._samples, self._last = None, None, None
            return None
        assert bay is not None
        if self._samples is None or self._bay != bay:
            if ctl.rescue_elapsed_s < self.SETTLE_S:
                return None  # still retracting into the hover: hold until it settles
            self._bay = bay
            self._samples = scripted_rescue(ctl.coordinator, bay)
        cleared = ctl.coordinator.rig(bay).bolt_x() >= BOLT_CLEARED_M + self.CLEAR_MARGIN_M
        if not cleared or self._last is None:
            sample = next(self._samples, None)
            if sample is not None:
                self._last = sample
        return self._last

    @contextlib.contextmanager
    def running(self) -> Generator[ScriptedGlove]:
        yield self


# --------------------------------------------------------------------------- 6-beat director

# Run seed for the scripted stage show. The spec allows fixed confirmed seeds for the stage.
# Pinned against the show's own config (cycle dwell 1.0 s, background bays on, DEFAULT_REALISM,
# as `teleop atc --script` runs): at 0.25 seeds 0-7 all pass on Windows and Linux (CI). Seed 7
# also passed at realism 0.5 on both, with the gate at 19/20 (a margin if realism is raised).
STAGE_SEED = 7

BEATS: tuple[str, ...] = (
    "1 Routine autonomy",
    "2 Fault rig & contact stall",
    "3 Operator rescue",
    "4 Edge fit & sim gate",
    "5 Bay 2 pose generalization",
    "6 Negative control",
)


@dataclass
class BeatResult:
    name: str
    start_s: float
    end_s: float | None = None
    ok: bool = False


Script = Generator[OperatorInput | None, None, None]


class StageDirector:
    """Plays the six stage beats through the same keys an operator would press.

    Call :meth:`step` once per tick and pass its result to ``controller.tick``. Each beat waits
    for its observable outcome (bay states, coordinator outcomes, gate result) with a
    ``beat_timeout_s`` budget of sim time; a beat that times out ends the run as failed.
    """

    def __init__(self, controller: StageController, beat_timeout_s: float = 30.0) -> None:
        self.controller = controller
        self.co = controller.coordinator
        self.beat_timeout_s = beat_timeout_s
        self.results: list[BeatResult] = []
        self.done = False
        self._script = self._run()

    @property
    def ok(self) -> bool:
        return self.done and len(self.results) == len(BEATS) and all(r.ok for r in self.results)

    def step(self) -> OperatorInput | None:
        if self.done:
            return None
        try:
            return next(self._script)
        except StopIteration:
            self.done = True
            self.controller.beat = "stage complete" if self.ok else "stage FAILED"
            return None

    # ------------------------------------------------------------------ helpers

    def _press(self, key: int | str) -> None:
        self.controller.on_key(ord(key) if isinstance(key, str) else key)

    def _wait(self, pred: Callable[[], bool], budget_s: float | None = None) -> Generator[None]:
        """Yield idle ticks until ``pred`` holds; raises TimeoutError past the budget."""
        t_end = self.co.time_s + (budget_s if budget_s is not None else self.beat_timeout_s)
        while not pred():
            if self.co.time_s > t_end:
                raise TimeoutError
            yield None

    def _idle(self, seconds: float) -> Generator[None]:
        for _ in range(max(round(seconds / CONTROL_DT), 1)):
            yield None

    def _begin(self, index: int) -> BeatResult:
        result = BeatResult(BEATS[index], self.co.time_s)
        self.results.append(result)
        self.controller.beat = result.name
        logger.info("stage beat %s", result.name)
        return result

    def _end(self, result: BeatResult, ok: bool) -> None:
        result.end_s, result.ok = self.co.time_s, ok
        logger.info(
            "stage beat %s: %s in %.1f s",
            result.name,
            "ok" if ok else "FAILED",
            result.end_s - result.start_s,
        )

    # ------------------------------------------------------------------ script

    def _run(self) -> Script:
        co, ctl = self.co, self.controller
        try:
            beat = self._begin(0)  # routine autonomy: every active bay clears a clean cycle
            yield from self._wait(lambda: all(n >= 1 for n in ctl.cleared_cycles.values()))
            self._end(beat, True)

            beat = self._begin(1)  # judge flips the fault rig; Bay 1 stalls on contact
            self._press("F")
            yield from self._wait(lambda: co.state(1) == BayState.RESCUE)
            yield from self._idle(0.7)  # let the 3 cm retract settle into the hover
            self._end(beat, True)

            beat = self._begin(2)  # operator selects Bay 1, holds the clutch, rescues
            self._press("1")
            self._press(KEY_SPACE)
            for sample in scripted_rescue(co, 1):
                yield sample
                if co.rig(1).bolt_x() >= BOLT_CLEARED_M + 0.0015:
                    break
            self._press(KEY_SPACE)
            self._end(beat, co.rig(1).bolt_x() >= BOLT_CLEARED_M)
            if not beat.ok:
                return

            beat = self._begin(3)  # UNO Q fit + 20-rollout gate, skill deployed
            yield from self._wait(
                lambda: co.state(1) == BayState.NOMINAL,
                self.beat_timeout_s + co.config.gate_deadline_s,  # gate applies at its deadline
            )
            self._end(beat, bool(ctl.gate.ok) and len(co.library) >= 1)
            if not beat.ok:
                return

            beat = self._begin(4)  # Bay 2's rotated/shifted jam cleared autonomously
            self._press("G")
            yield from self._wait(
                lambda: (
                    co.outcome(2) == "cleared autonomously"
                    or co.state(2) in (BayState.QUEUED, BayState.RESCUE)
                )
            )
            self._end(beat, co.outcome(2) == "cleared autonomously")
            if not beat.ok:
                return

            beat = self._begin(5)  # pinned bolt: OOD, refused, handed to the human
            self._press("P")
            yield from self._wait(
                lambda: co.state(2) in (BayState.OOD_ESCALATED, BayState.AUTONOMOUS_SKILL)
            )
            self._end(beat, co.state(2) == BayState.OOD_ESCALATED)
        except TimeoutError:
            if self.results and self.results[-1].end_s is None:
                self._end(self.results[-1], False)


__all__ = [
    "BEATS",
    "KEYMAP",
    "STAGE_SEED",
    "ScriptedGlove",
    "key_name",
    "AudioCues",
    "BeatResult",
    "ConsoleSource",
    "EventTap",
    "NullSpeaker",
    "ReasonerTap",
    "StageAction",
    "StageController",
    "StageDirector",
    "SystemSpeaker",
    "UnoQConsolePoller",
    "merge_console",
    "scripted_rescue",
    "worst_failure",
]
