"""``teleop atc``: 4-bay fleet supervisor with HUD, keyboard stage controls and audio cues.

    teleop atc [--source studio|synthetic|replay] [--unoq-host HOST] [--headless --steps N]
               [--script] [--seed N] [--hotspot | --no-hotspot] [--config configs/atc.yaml]
               [--demo-rescue [--demo-shake X]]

The default (``uv run teleop atc``) is the booth: it keeps the hotspot up, listens for the Studio
gloves, uses the UNO Q board (or a local loopback node, hot-swapped either way) and shows a live
health strip with Reconnect buttons. A rescue uses the gloves when they are healthy at clutch
engage, else the scripted rescue. ``--headless`` skips the hotspot and the health checks.

Opens the fleet ops console in the browser unless ``--headless``: a CCTV feed per bay plus
fleet telemetry, served on ``--console-port`` (``--no-browser`` only prints the URL).
``--display viewer`` opens the MuJoCo passive viewer with the text HUD instead. Keys (in the
console page or the viewer): ``1..4`` select
a bay, ``SPACE`` toggles the clutch, ``F``/``G``/``P`` arm the Bay 1 flap, Bay 2 flap and Bay 2
pinned-bolt faults, ``R`` resets (see ``teleop.atc.stage``). ``--script`` plays the six-beat
stage script with a scripted glove; headless it exits 0 only if every beat passed.
``--demo-rescue`` plays one hard-coded take: Bay 1 jams, then a shaky "human" glove presses the
flap down and clears the bolt (``--demo-shake`` scales the hand shake, 0 for none).

Recovery after a matched stall: ``--recovery-backend dmp`` (default, the learned DMP skills),
``hybrid`` (a gated ACT checkpoint from ``--act-checkpoint``, DMP behind it) or ``act``; ``M``
cycles it live. ``--record-episodes DIR`` saves every recovery as a LeRobot episode and
``--headless --generate N`` produces N scripted rescue episodes. These load their modules only
when used, so the default run needs no torch.

Hooks come from the environment (``.env``): ``SOLANA_RPC_URL``/``SOLANA_KEYPAIR_PATH`` for the
compliance chain, falling back to a local audit log. The fleet reasoner is the offline triage.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import importlib
import importlib.util
import logging
import subprocess
import threading
from collections.abc import Callable, Generator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import yaml

from teleop.app import make_source
from teleop.atc.coordinator import DEFAULT_REALISM, OperatorInput
from teleop.atc.demo_rescue import DemoRescueDirector, Shake
from teleop.atc.glove_filter import GloveInputFilter
from teleop.atc.glove_select import SelectingGlove
from teleop.atc.hud import hud_lines, hud_overlays
from teleop.atc.recovery import RecoveryBackend
from teleop.atc.stage import (
    STAGE_SEED,
    AudioCues,
    ConsoleSource,
    NullSpeaker,
    ScriptedGlove,
    StageController,
    StageDirector,
    SystemSpeaker,
    UnoQConsolePoller,
)
from teleop.atc.validator import GATE_PASS_MIN
from teleop.commands.sim import add_hotspot_args, maybe_hotspot, setup_logging
from teleop.config import SimConfig, StudioConfig, WristConfig
from teleop.health.gloves import GloveCheck
from teleop.health.hotspot import HotspotCheck, build_hotspot_check
from teleop.health.model import Level
from teleop.health.monitor import HealthMonitor
from teleop.health.panel import PanelCheck
from teleop.input.studio import StudioSource
from teleop.net.hotspot import HotspotConfigError
from teleop.protocols import InputSource, Retargeter, WristSource
from teleop.registry import RETARGETERS, WRISTS, resolve
from teleop.retarget.calibration import Calibration
from teleop.retarget.direct import DirectRetargeter
from teleop.sim.hands import REPO_ROOT, HandSpec, load_hand
from teleop.sim.loop import FixedRateLoop
from teleop.types import Side
from teleop.unoq.client import UnoQClient
from teleop.unoq.dmp import HAND_ACTUATORS
from teleop.unoq.mcu_bridge import McuConsoleBridge
from teleop.unoq.protocol import DEFAULT_TCP_PORT, DEFAULT_UDP_PORT
from teleop.unoq.switch import SwitchableUnoQ

logger = logging.getLogger("teleop.atc")

DEFAULT_CONFIG = REPO_ROOT / "configs" / "atc.yaml"
CONTROL_HZ = 60
HUD_OVERLAY_HZ = 10.0  # viewer text refresh; each set_texts waits for a rendered frame


@dataclass
class AtcConfig:
    source: str = "studio"
    replay_file: Path | None = None
    studio: StudioConfig = field(default_factory=StudioConfig)
    wrist_mode: str = "auto"
    calibration: Path | None = None
    unoq_host: str = "unoq.local"
    unoq_udp_port: int = DEFAULT_UDP_PORT
    unoq_tcp_port: int = DEFAULT_TCP_PORT
    serial_port: str | None = None
    hotspot: bool = True  # keep the booth hotspot up (Windows; never with --headless)
    background_bays: bool = True
    cycle_dwell_s: float = 1.0
    jam_every: int = 0  # every Nth pallet arrives with a raised flap (0: only on a jam press)
    audio: bool = True
    script: bool = False
    demo_rescue: bool = False  # hard-coded jam + human-like operator rescue on Bay 1
    demo_shake: float = 1.0  # hand-shake scale for --demo-rescue (0: steady)
    hud_log_period_s: float = 5.0
    headless: bool = False
    steps: int | None = None
    display: str = "console"  # console (browser CCTV wall) | viewer (MuJoCo passive viewer)
    console_port: int = 8765
    console_fps: int = 30
    open_browser: bool = True
    recovery_backend: str = "dmp"  # dmp | hybrid | act
    act_checkpoint: Path | None = None  # models/act/<run>/ with gate.json
    act_device: str = "cuda"
    record_episodes: Path | None = None  # save recoveries as LeRobot episodes here
    generate: int | None = None  # headless: generate N scripted rescue episodes
    seed: int | None = None  # None: STAGE_SEED with --script, else 0
    realism: float = DEFAULT_REALISM  # latch-pose error and gate randomization scale (0: off)
    gate_pass_min: int = GATE_PASS_MIN  # randomized skill gate: deploy at >= N of 20
    gemini: bool = True  # Tier 2 grasp gate on Gemini when GEMINI_API_KEY is set (--no-gemini)
    dummy_glove: bool = False  # TODO(rokoko-gloves-swap): temporary dummy UDP/UI operator
    glove_filter: bool = True  # 1-Euro smoothing of the glove wrist pose and finger targets


_PATH_FIELDS = {"replay_file", "calibration", "act_checkpoint", "record_episodes"}
DISPLAYS = ("console", "viewer")
BACKENDS = tuple(b.value for b in RecoveryBackend)

# Lazy hooks into the ACT recovery track: flag -> (module, attribute). Imported only when the
# flag is used, so the default ``dmp`` run never imports torch or lerobot.
HOOKS: dict[str, tuple[str, str]] = {
    "--act-checkpoint": ("teleop.atc.act_recovery", "load_act_factory"),
    "--record-episodes": ("teleop.atc.episodes", "open_episode_sink"),
    "--generate": ("teleop.atc.generate", "run_generation"),
}


class HookMissing(Exception):
    def __init__(self, flag: str, module: str) -> None:
        super().__init__(f"{flag} is not implemented yet ({module})")


def load_hook(flag: str) -> Any:
    module, attr = HOOKS[flag]
    try:
        mod = importlib.import_module(module)
    except ModuleNotFoundError as exc:
        if exc.name is not None and (module == exc.name or module.startswith(exc.name + ".")):
            raise HookMissing(flag, module) from exc
        raise
    return getattr(mod, attr)


def resolve_seed(cfg: AtcConfig) -> int:
    """Run seed: an explicit one always wins; else the scripted show uses ``STAGE_SEED``."""
    if cfg.seed is not None:
        return cfg.seed
    return STAGE_SEED if (cfg.script or cfg.demo_rescue) else 0


def load_config(path: Path | None, overrides: Mapping[str, Any]) -> AtcConfig:
    """Dataclass defaults < YAML file < command-line overrides (``None`` means not given)."""
    raw: dict[str, Any] = {}
    if path is not None and path.is_file():
        loaded: Any = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"{path}: expected a mapping")
        raw = cast("dict[str, Any]", loaded)
    known = {f.name for f in dataclasses.fields(AtcConfig)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)}")
    raw.update({k: v for k, v in overrides.items() if v is not None})
    studio_raw: Any = raw.pop("studio", None)
    cfg = AtcConfig(**{k: v for k, v in raw.items() if k not in _PATH_FIELDS})
    for name in _PATH_FIELDS:
        if raw.get(name) is not None:
            setattr(cfg, name, Path(str(raw[name])))
    if isinstance(studio_raw, StudioConfig):
        cfg.studio = studio_raw
    elif isinstance(studio_raw, dict):
        cfg.studio = StudioConfig(**cast("dict[str, Any]", studio_raw))
    if cfg.display not in DISPLAYS:
        raise ValueError(f"display must be one of {DISPLAYS}, got {cfg.display!r}")
    if cfg.recovery_backend not in BACKENDS:
        raise ValueError(
            f"recovery_backend must be one of {BACKENDS}, got {cfg.recovery_backend!r}"
        )
    return cfg


# --------------------------------------------------------------------------- glove operator


class GloveOperator:
    """Right-hand glove frames -> ``OperatorInput`` (wrist pose + hand targets by name)."""

    def __init__(
        self,
        source: InputSource,
        wrist: WristSource,
        retargeter: Retargeter,
        *,
        hand: HandSpec | None = None,
        ctrlrange: Mapping[str, tuple[float, float]] | None = None,
        smoothing: GloveInputFilter | None = None,
    ) -> None:
        self.source = source
        self.wrist = wrist
        self.retargeter = retargeter
        self.hand = hand
        self.ctrlrange = dict(ctrlrange) if ctrlrange is not None else None
        self.smoothing = smoothing
        self._smoothed_for: Retargeter | None = None

    def recalibrate(self, calib: Calibration) -> set[tuple[Side, str]]:
        """Swap in a retargeter built from ``calib``; returns the actuators left on defaults.

        One reference assignment, so the control thread's next ``latest`` uses either the old
        or the new retargeter, never a half-built one.
        """
        if self.hand is None or self.ctrlrange is None:
            raise RuntimeError("glove operator was built without its hand spec and ctrlrange")
        retargeter = DirectRetargeter(self.hand, {"right": self.ctrlrange}, calib)
        self.retargeter = cast("Retargeter", retargeter)
        return set(retargeter.degenerate)

    @classmethod
    def build(
        cls, cfg: AtcConfig, model: Any, *, tolerate_bind_error: bool = False
    ) -> GloveOperator:
        """``tolerate_bind_error``: a busy Studio port shows on the Gloves light, not a crash."""
        hand = load_hand("inspire")
        ranges: dict[str, tuple[float, float]] = {}
        for name in HAND_ACTUATORS:  # both active bays share one hand model: read Bay 1's
            act = model.actuator(f"bay1/{name}")
            if bool(model.actuator_ctrllimited[act.id]):
                lo, hi = (float(v) for v in model.actuator_ctrlrange[act.id])
                ranges[name] = (lo, hi)
        calib = (
            Calibration.from_json(cfg.calibration)
            if cfg.calibration is not None
            else Calibration.default(hand)
        )
        retarget_cls: Any = resolve(RETARGETERS, "direct")
        wrist_cls: Any = resolve(WRISTS, cfg.wrist_mode)
        sim_cfg = SimConfig(source=cfg.source, replay_file=cfg.replay_file, studio=cfg.studio)
        source: InputSource = (
            StudioSource(cfg.studio, tolerate_bind_error=True)
            if tolerate_bind_error and cfg.source == "studio"
            else make_source(sim_cfg)
        )
        return cls(
            source,
            cast("WristSource", wrist_cls(WristConfig(mode=cfg.wrist_mode))),
            cast("Retargeter", retarget_cls(hand, {"right": ranges}, calib)),
            hand=hand,
            ctrlrange=ranges,
            smoothing=GloveInputFilter() if cfg.glove_filter else None,
        )

    def latest(self) -> OperatorInput | None:
        frame = self.source.latest().right
        if frame is None:
            return None
        retargeter = self.retargeter
        targets = retargeter.retarget(frame).positions
        wrist = self.wrist.wrist(frame).pose
        if self.smoothing is None:
            return OperatorInput(wrist, dict(targets))
        if retargeter is not self._smoothed_for:
            # A recalibration swapped the retargeter (from the console thread): the finger
            # targets jump to the new ranges, so restart the filters rather than glide there.
            self.smoothing.reset()
            self._smoothed_for = retargeter
        wrist, smoothed = self.smoothing.step(frame, wrist, targets)
        return OperatorInput(wrist, smoothed)

    @contextlib.contextmanager
    def running(self) -> Generator[GloveOperator]:
        self.source.start()
        try:
            yield self
        finally:
            self.source.stop()


# --------------------------------------------------------------------------- wiring


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="teleop atc", allow_abbrev=False)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--source", choices=("studio", "synthetic", "replay"), default=None)
    ap.add_argument("--replay-file", type=Path, default=None)
    ap.add_argument("--calibration", type=Path, default=None)
    ap.add_argument("--unoq-host", default=None)
    ap.add_argument("--unoq-udp-port", type=int, default=None)
    ap.add_argument("--unoq-tcp-port", type=int, default=None)
    ap.add_argument("--serial-port", default=None, help="UNO Q MCU console serial port")
    ap.add_argument("--headless", action="store_true", default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--script", action="store_true", default=None, help="play the 6-beat script")
    ap.add_argument(
        "--demo-rescue",
        action="store_true",
        default=None,
        help="jam Bay 1, then rescue it with a scripted shaky human glove",
    )
    ap.add_argument(
        "--demo-shake", type=float, default=None, help="--demo-rescue hand-shake scale (0: none)"
    )
    ap.add_argument("--no-audio", dest="audio", action="store_false", default=None)
    ap.add_argument("--no-background", dest="background_bays", action="store_false", default=None)
    ap.add_argument("--display", choices=DISPLAYS, default=None, help="console (default) or viewer")
    ap.add_argument("--seed", type=int, default=None, help="run seed (stage script: STAGE_SEED)")
    ap.add_argument(
        "--realism", type=float, default=None, help="pose error / gate randomization scale, 0-1"
    )
    ap.add_argument("--gate-pass-min", type=int, default=None, help="gate passes needed of 20")
    ap.add_argument(
        "--jam-every", type=int, default=None, help="every Nth pallet has a raised flap"
    )
    ap.add_argument("--console-port", type=int, default=None)
    ap.add_argument("--console-fps", type=int, default=None)
    ap.add_argument("--no-browser", dest="open_browser", action="store_false", default=None)
    ap.add_argument("--recovery-backend", choices=BACKENDS, default=None)
    ap.add_argument("--act-checkpoint", type=Path, default=None, help="ACT run dir with gate.json")
    ap.add_argument("--act-device", default=None, help="torch device for ACT (default cuda)")
    ap.add_argument("--record-episodes", type=Path, default=None, help="LeRobot dataset dir")
    ap.add_argument("--generate", type=int, default=None, help="headless: N scripted episodes")
    ap.add_argument(
        "--dummy-glove",
        action="store_true",
        default=None,
        help="TEMP: spawn standalone dummy Studio UDP streamer + UI teleop override",
    )
    add_hotspot_args(ap)
    ap.add_argument("--no-hotspot", action="store_true", help="do not keep the booth hotspot up")
    ap.add_argument(
        "--no-gemini",
        dest="gemini",
        action="store_false",
        default=None,
        help="no Gemini grasp gate (local Gemma if COPILOT_LOCAL_URL is set, else physics only)",
    )
    return ap


def _hardware_consoles(cfg: AtcConfig, unoq: SwitchableUnoQ) -> list[ConsoleSource]:
    if cfg.serial_port:
        from teleop.unoq.mcu_bridge import load_joy_map, open_console, save_joy_map

        transport, kind = open_console(cfg.serial_port)
        bridge = McuConsoleBridge(transport, joy_map=load_joy_map(), on_joy_map=save_joy_map)
        bridge.start()
        logger.info("console: %s", kind)
        return [bridge]
    if unoq.is_mock:
        return []
    try:
        poll_client = UnoQClient(cfg.unoq_host, cfg.unoq_udp_port, cfg.unoq_tcp_port).connect()
    except OSError as exc:
        logger.warning("console: cannot open a second UNO Q connection (%s)", exc)
        return []
    logger.info("console: polling UNO Q %s", cfg.unoq_host)
    return [UnoQConsolePoller(poll_client)]


def _unoq_label(cfg: AtcConfig, unoq: SwitchableUnoQ) -> str:
    return "UNO Q local loopback" if unoq.is_mock else f"UNO Q {cfg.unoq_host}"


def _console_rebuilder(
    cfg: AtcConfig, controller: StageController, unoq: SwitchableUnoQ
) -> Callable[[], None]:
    """Reinstall the board console poller for the current link (may block on connect).

    Shared by the swap listener and the panel check, which can run on different threads; the
    lock keeps two rebuilds from installing over each other (and leaking a poller).
    """
    lock = threading.Lock()

    def rebuild() -> None:
        with lock:
            controller.replace_hardware(_hardware_consoles(cfg, unoq))

    return rebuild


def _button_poller(controller: StageController) -> Callable[[], UnoQConsolePoller | None]:
    """The installed board console poller (None when none is): what the laptop hears."""
    return lambda: next((c for c in controller.hardware if isinstance(c, UnoQConsolePoller)), None)


def _swap_listener(
    cfg: AtcConfig,
    controller: StageController,
    suffix: str,
    rebuild: Callable[[], None] | None = None,
) -> Callable[[SwitchableUnoQ], None]:
    """``SwitchableUnoQ.on_swap``: restart the board console poller and refresh the header.

    Runs on the swapping thread. A laptop serial console does not depend on the board link, so
    it stays installed (the panel check's Reconnect reopens it).
    """

    def on_swap(unoq: SwitchableUnoQ) -> None:
        if not cfg.serial_port:
            if rebuild is not None:
                rebuild()
            else:
                controller.replace_hardware(_hardware_consoles(cfg, unoq))
        controller.unoq_label = _unoq_label(cfg, unoq) + suffix
        logger.info("atc: %s", controller.unoq_label)

    return on_swap


def main(argv: list[str]) -> int:
    ns = _parser().parse_args(argv)
    setup_logging()
    if ns.hotspot and ns.no_hotspot:
        print("teleop atc: --hotspot and --no-hotspot are mutually exclusive")
        return 2
    skip = ("config", "hotspot", "no_hotspot", "env_file")
    overrides = {k: v for k, v in vars(ns).items() if k not in skip}
    overrides["hotspot"] = True if ns.hotspot else (False if ns.no_hotspot else None)
    try:
        cfg = load_config(ns.config, overrides)
    except (OSError, ValueError, TypeError) as exc:
        print(f"teleop atc: bad config: {exc}")
        return 2
    env_file = _resolve_env_file(ns.env_file)
    if env_file is not None:
        from dotenv import load_dotenv

        load_dotenv(env_file)
    if cfg.demo_rescue and (cfg.script or cfg.generate is not None):
        print("teleop atc: --demo-rescue cannot be combined with --script or --generate")
        return 2
    if cfg.generate is not None and (not cfg.headless or cfg.record_episodes is None):
        print("teleop atc: --generate needs --headless and --record-episodes")
        return 2
    used = {
        "--act-checkpoint": cfg.act_checkpoint is not None,
        "--record-episodes": cfg.record_episodes is not None,
        "--generate": cfg.generate is not None,
    }
    try:
        hooks = {flag: load_hook(flag) for flag, on in used.items() if on}
    except HookMissing as exc:
        print(f"teleop atc: {exc}")
        return 3
    with contextlib.ExitStack() as stack:
        try:
            hotspot_check = _start_hotspot(cfg, ns, stack)
        except (HotspotConfigError, RuntimeError) as exc:
            print(f"teleop atc: {exc}")
            return 1
        try:
            return _run_booth(cfg, hooks, hotspot_check, stack)
        except KeyboardInterrupt:
            return 0


def _start_hotspot(
    cfg: AtcConfig, ns: argparse.Namespace, stack: contextlib.ExitStack
) -> HotspotCheck | None:
    """The hotspot health check (None when headless); its watchdog stops with ``stack``.

    A missing password only turns the light off, unless ``--hotspot`` was given explicitly (then
    it raises ``HotspotConfigError``, today's exit 1). Headless runs build no health checks and
    keep the hotspot down unless ``--hotspot`` asks for today's plain watchdog.
    """
    if cfg.headless:
        check, watchdog = None, maybe_hotspot(ns)
    else:
        check, watchdog = build_hotspot_check(cfg.hotspot, ns.env_file, strict=ns.hotspot)
    if watchdog is not None:
        stack.callback(watchdog.stop)
    return check


def _resolve_env_file(env_file: Path) -> Path | None:
    """The ``.env`` to load. A git worktree has none of its own, so the default ``.env`` falls
    back to the main checkout's. Without this the Solana keys are missing there and the audit
    trail silently drops to local hashing with no explorer link. An explicit path never falls back.
    """
    if env_file.is_file():
        return env_file
    if env_file != Path(".env"):
        return None
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    common = Path(out.stdout.strip()) if out.returncode == 0 and out.stdout.strip() else None
    if common is None or common.name != ".git":
        return None
    main_env = common.parent / ".env"
    return main_env if main_env.is_file() else None


def _run_booth(
    cfg: AtcConfig,
    hooks: Mapping[str, Any],
    hotspot_check: HotspotCheck | None,
    stack: contextlib.ExitStack,
) -> int:
    """Build the stage, the operator and (unless headless) the health checks, then run."""
    from teleop.compliance.solana import SolanaComplianceSink
    from teleop.reasoning.triage import TriageReasoner

    unoq = SwitchableUnoQ(
        UnoQClient.connect_or_mock(cfg.unoq_host, cfg.unoq_udp_port, cfg.unoq_tcp_port)
    )
    stack.callback(unoq.close)
    sink = SolanaComplianceSink.from_env()
    reasoner = TriageReasoner()
    speak = cfg.audio and not cfg.headless
    hardware = _hardware_consoles(cfg, unoq)
    suffix = f" | compliance {sink.mode} | reasoner {reasoner.mode}"
    controller = StageController.build(
        cast("UnoQClient", unoq),  # same request surface; the coordinator only forwards calls
        sink,
        reasoner,
        audio=AudioCues(NullSpeaker(), enabled=False),
        hardware=hardware,
        cycle_dwell_s=cfg.cycle_dwell_s,
        auto_jam_every=0
        if (cfg.script or cfg.demo_rescue or cfg.generate is not None)
        else cfg.jam_every,
        include_background_bays=cfg.background_bays,
        seed=resolve_seed(cfg),
        realism=cfg.realism,
        gate_pass_min=cfg.gate_pass_min,
        unoq_label=_unoq_label(cfg, unoq) + suffix,
    )
    stack.callback(controller.close)
    logger.info("atc: %s", controller.unoq_label)
    stack.callback(_close_act, controller)
    _setup_recovery(cfg, controller, hooks.get("--act-checkpoint"))
    sink_handle: Any = None
    tag_router = _EpisodeTagRouter()
    operator_box: list[Any] = []  # filled once the operator is built (after the sink)
    if "--record-episodes" in hooks:
        scripted = cfg.generate is not None or cfg.source == "synthetic"
        # With gloves, SelectingGlove may play the scripted rescue instead (gloves unhealthy at
        # clutch engage), so the label is read per rescue, never fixed at startup.
        source: str | Callable[[], str] = "scripted" if scripted else rescue_label(operator_box)
        try:
            sink_handle = hooks["--record-episodes"](cfg.record_episodes, controller, source)
        except (RuntimeError, OSError) as exc:
            print(f"teleop atc: {exc}")
            return 1
        tag_router.add(sink_handle)
        controller.coordinator.set_observer(sink_handle)
        stack.callback(_close_episodes, controller, sink_handle)
    copilot_attach: Any = None
    if not cfg.headless:  # headless runs (CI, --generate) stay physics-only and deterministic
        copilot_attach = _setup_copilot(cfg, controller, stack, tag_router, speak=speak)
    if "--generate" in hooks:
        assert cfg.generate is not None
        n = int(hooks["--generate"](controller, sink_handle, cfg.generate, resolve_seed(cfg)))
        print(f"teleop atc: generated {n} successful episodes -> {cfg.record_episodes}")
        return 0 if n >= cfg.generate else 1

    glove, studio = _build_glove(cfg, controller)
    health: HealthMonitor | None = None
    if hotspot_check is not None:  # not headless
        glove_check = GloveCheck(studio, cfg.studio.port)
        if isinstance(glove, GloveOperator) and studio is not None:
            glove = SelectingGlove(
                controller,
                glove,
                ScriptedGlove(controller),
                lambda: glove_check.status().level is Level.OK,
            )
        bridge = next((c for c in hardware if isinstance(c, McuConsoleBridge)), None)
        board_console = not cfg.serial_port  # buttons arrive over a second board connection
        rebuild = _console_rebuilder(cfg, controller, unoq) if board_console else None
        panel = PanelCheck(
            unoq,
            cfg.unoq_host,
            cfg.unoq_udp_port,
            cfg.unoq_tcp_port,
            serial_port=cfg.serial_port,
            local_bridge=bridge,
            buttons=_button_poller(controller) if board_console else None,
            rebuild_consoles=rebuild,
        )
        unoq.on_swap.append(_swap_listener(cfg, controller, suffix, rebuild))
        stack.callback(unoq.on_swap.clear)
        panel.start()
        stack.callback(panel.stop)
        health = HealthMonitor([hotspot_check, glove_check, panel])
        health.start()
        stack.callback(health.stop)
    operator_box.append(glove)
    return _run(
        cfg,
        controller,
        glove,
        health,
        tag_router=tag_router,
        copilot_attach=copilot_attach,
    )


class _EpisodeTagRouter:
    """Routes Tier 3 Argon ``tag_episode`` calls to CLI and/or UI ``RecordingDesk``."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sinks: list[Any] = []

    def add(self, sink: Any) -> None:
        if sink is None:
            return
        with self._lock:
            if sink not in self._sinks:
                self._sinks.append(sink)

    def tag_episode(
        self, uid: str, *, dataset_tag: str, audit: Mapping[str, Any] | None = None
    ) -> bool:
        with self._lock:
            sinks = list(self._sinks)
        matched = False
        for sink in sinks:
            tagger = getattr(sink, "tag_episode", None)
            if callable(tagger):
                try:
                    if bool(tagger(uid, dataset_tag=dataset_tag, audit=audit)):
                        matched = True
                except Exception:
                    logger.exception("atc: episode tag router failed")
        return matched


class _StageSpeakerAdapter:
    """Adapts ``copilot.speech.Speaker`` to ``stage.Speaker`` without double-closing."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def say(self, text: str) -> None:
        self._inner.say(text)

    def close(self) -> None:
        pass  # ExitStack closes the shared copilot Speaker once


def _setup_copilot(
    cfg: AtcConfig,
    controller: StageController,
    stack: contextlib.ExitStack,
    episode_sink: Any = None,
    *,
    speak: bool = False,
) -> Any:
    """Attach unified Tier 1 voice + callouts, Tier 2 grasp gate, and silent Tier 3 Argon audit.

    Advisory side effects only: the gate can withhold a skill from the Fleet Library and speaks
    its verdict; it never sends console keys or commands a robot. With Gemini, the gate also
    hands each judged rescue (and OOD refusal) to the silent Tier 3 audit, whose report feeds the
    console card (``controller.audit``) and tags recorded episodes for ``teleop process``.
    """
    from teleop.copilot.config import CopilotConfig
    from teleop.copilot.gate import build_grasp_gate

    try:
        ccfg = CopilotConfig.from_env(mute=not speak)
    except ValueError as exc:
        logger.warning("atc: copilot config invalid (%s); falling back to offline audio", exc)
        if speak:
            controller.audio = AudioCues(SystemSpeaker(), enabled=True)
        return None
    use_gemini = cfg.gemini and ccfg.has_key
    speaker, streamer = _copilot_speaker(ccfg, use_gemini=use_gemini) if speak else (None, None)
    if streamer is not None:
        stack.callback(streamer.close)
    if speaker is not None:
        stack.callback(speaker.close)
        speaker.precache_cues_async()

    has_vlm = use_gemini or bool(ccfg.local_url)
    use_console_callouts = cfg.display == "console"
    if speaker is not None and not use_console_callouts:
        # In --display viewer mode, wire unified Copilot Speaker into StageController.audio
        # while suppressing generic VALIDATION_GATE speech when GraspGate speaks verdict.say.
        controller.audio = AudioCues(
            _StageSpeakerAdapter(speaker),
            enabled=True,
            skip_validation_gate=has_vlm,
        )

    copilot_attach: Any = None
    if use_console_callouts:
        copilot_attach = _build_console_copilot_attach(
            ccfg,
            use_gemini=use_gemini,
            speaker=speaker,
            gate_speaks=has_vlm,
        )

    if not has_vlm:
        logger.info("atc: grasp gate physics only (no GEMINI_API_KEY / COPILOT_LOCAL_URL)")
        return copilot_attach

    reviewer = _audit_reviewer(ccfg, controller, episode_sink) if use_gemini else None
    if reviewer is not None:
        stack.callback(reviewer.close)  # after the gate (LIFO): the gate submits to it

    def _on_gate_verdict(verdict: Any, bay_id: int) -> None:
        controller.on_grasp_verdict(verdict)
        if copilot_attach is not None:
            publish_verdict = getattr(copilot_attach, "publish_verdict", None)
            if callable(publish_verdict):
                publish_verdict(verdict, bay_id)

    gate = build_grasp_gate(
        ccfg,
        use_gemini=use_gemini,
        speaker=speaker,
        audit_reviewer=reviewer,
        on_verdict=_on_gate_verdict,
    )
    stack.callback(gate.close)
    gate.warmup(controller.coordinator.model)
    controller.coordinator.grasp_gate = gate
    controller.coordinator.nonblocking_gate = not cfg.script and not cfg.headless
    logger.info("atc: grasp gate %s (%s)", gate.source, gate.model)
    return copilot_attach


def _build_console_copilot_attach(
    ccfg: Any,
    *,
    use_gemini: bool,
    speaker: Any,
    gate_speaks: bool,
) -> Any:
    """Build an in-process callout bridge for ``run_console`` (populates ``#panel-copilot``)."""
    import time

    from teleop.copilot.callouts import Callout, CalloutService
    from teleop.copilot.events import EventDetector, EventKind
    from teleop.copilot.feed import CameraFrame, parse_feeds
    from teleop.copilot.frames import FrameStore
    from teleop.copilot.gemini import GeminiClient
    from teleop.copilot.local import LocalModelClient
    from teleop.copilot.publish import callout_message

    server_box: list[Any] = []

    def attach(server: Any) -> tuple[Any, Any, Any]:
        server_box.append(server)
        store = FrameStore()
        detector = EventDetector()
        feeds_lock = threading.Lock()
        feeds_state: list[tuple[int, dict[int, Any]]] = [(-1, {})]

        client = (
            GeminiClient(ccfg.gemini_api_key, no_images=ccfg.no_images)
            if use_gemini and ccfg.gemini_api_key
            else None
        )
        local_client = LocalModelClient.from_config(ccfg)

        def show(callout: Callout) -> None:
            server.publish_copilot(callout_message(callout))

        service = CalloutService(
            client,
            model=ccfg.fast_model,
            local_client=local_client,
            local_model=ccfg.local_model,
            thinking_level=ccfg.fast_thinking,
            sink=show,
            speaker=speaker,
            frame_for=store.latest,
            no_images=ccfg.no_images,
        )

        def on_telemetry(msg: dict[str, Any]) -> None:
            lid = msg.get("layout_id")
            if isinstance(lid, int) and not isinstance(lid, bool):
                with feeds_lock:
                    feeds_state[0] = (lid % 256, parse_feeds(msg))
            store.on_telemetry(msg)
            for event in detector.feed(msg):
                if gate_speaks and event.kind is EventKind.GATE_RESULT:
                    # GraspGate already spoke & posted the verdict; avoid duplicate TTS.
                    continue
                service.handle(event)

        def on_frame(slot: int, layout_id: int, jpeg: bytes) -> None:
            with feeds_lock:
                cur_lid, fmap = feeds_state[0]
            if layout_id % 256 != cur_lid:
                return
            info = fmap.get(slot)
            if info is None:
                return
            store.add(
                CameraFrame(
                    bay=info.bay,
                    camera=info.camera,
                    role=info.role,
                    jpeg=jpeg,
                    t_ns=time.monotonic_ns(),
                )
            )

        return on_telemetry, on_frame, service.close

    def publish_verdict(verdict: Any, bay_id: int) -> None:
        if not server_box:
            return
        say = getattr(verdict, "say", "") or ""
        reason = getattr(verdict, "reason", "") or ""
        source = getattr(verdict, "source", "") or ""
        if not say.strip() or source in ("physics_only", "error"):
            return
        approved = bool(getattr(verdict, "approved", True))
        server_box[0].publish_copilot(
            {
                "type": "copilot",
                "bay": int(bay_id),
                "severity": "info" if approved else "alarm",
                "say": str(say).strip(),
                "detail": str(reason).strip(),
                "source": "local" if source == "local" else "model",
            }
        )

    attach.publish_verdict = publish_verdict  # type: ignore[attr-defined]
    return attach


def _audit_reviewer(ccfg: Any, controller: StageController, episode_sink: Any) -> Any:
    """The Tier 3 Argon audit (Gemini only; never spoken). ``episode_sink`` may be None."""
    from teleop.copilot.gemini import GeminiClient
    from teleop.copilot.review import AuditReport, AuditReviewer

    def on_update(report: AuditReport) -> None:
        controller.audit = report.to_dict()  # read by the HUD view -> console telemetry

    client = GeminiClient(ccfg.gemini_api_key, no_images=ccfg.no_images)
    return AuditReviewer(client, cfg=ccfg, on_update=on_update, episode_sink=episode_sink)


def _copilot_speaker(ccfg: Any, *, use_gemini: bool = True) -> tuple[Any, Any]:
    """Unified voice: Gemini SSE TTS when enabled, fast ``espeak-ng`` system fallback otherwise."""
    from teleop.copilot.speech import Speaker
    from teleop.copilot.tts_stream import GeminiStreamingTTS

    streamer = (
        GeminiStreamingTTS(ccfg.gemini_api_key, model=ccfg.tts_model, voice=ccfg.voice)
        if use_gemini and ccfg.gemini_api_key
        else None
    )
    speaker = Speaker(streamer, tts_model=ccfg.tts_model, voice=ccfg.voice, mute=ccfg.mute)
    return speaker, streamer


def rescue_label(operator_box: list[Any]) -> Callable[[], str]:
    """Episode label for a human rescue, read when it starts: what really drove the hand.

    ``operator_box`` holds the operator once built. A ``SelectingGlove`` reports the input it
    chose at clutch engage (``"scripted"`` when the gloves were unhealthy); anything else is the
    glove.
    """

    def label() -> str:
        op = operator_box[0] if operator_box else None
        if isinstance(op, SelectingGlove) and op.rescue_input is not None:
            return op.rescue_input
        return "glove"

    return label


def _build_glove(
    cfg: AtcConfig, controller: StageController
) -> tuple[GloveOperator | ScriptedGlove | None, StudioSource | None]:
    """The operator input (None under ``--script``) and its Studio listener, if any."""
    if cfg.script or cfg.demo_rescue:
        return None, None
    if cfg.source == "synthetic":
        # Without real gloves, a scripted rescue plays while the clutch is held.
        return ScriptedGlove(controller), None
    glove = GloveOperator.build(
        cfg, controller.coordinator.model, tolerate_bind_error=not cfg.headless
    )
    studio = glove.source if isinstance(glove.source, StudioSource) else None
    return glove, studio


def _close_act(controller: StageController) -> None:
    act = controller.coordinator.act
    if act is not None:
        act.close()


def _close_episodes(controller: StageController, sink_handle: Any) -> None:
    controller.coordinator.set_observer(None)
    sink_handle.close()


def _setup_recovery(cfg: AtcConfig, controller: StageController, load_act: Any) -> None:
    """Set the recovery backend and load the ACT release; any ACT failure falls back to dmp."""
    from teleop.policy.manifest import ManifestError

    co = controller.coordinator
    backend = RecoveryBackend(cfg.recovery_backend)
    if load_act is not None and cfg.act_checkpoint is not None:
        try:
            co.set_act(load_act(cfg.act_checkpoint, cfg.act_device, co.model, co.data))
        except (ManifestError, OSError, ImportError, RuntimeError, ValueError) as exc:
            # ImportError: the `act` dependency group (torch) is not installed; RuntimeError: no
            # CUDA device. Either way the demo keeps running on the DMP.
            logger.warning("atc: ACT checkpoint not loaded (%s); recovery backend dmp", exc)
            co.backend_note = f"ACT unavailable ({exc}): back to dmp"
            backend = RecoveryBackend.DMP
    co.set_recovery_backend(backend)
    logger.info("atc: recovery backend %s", co.recovery_backend.value)


def _run(
    cfg: AtcConfig,
    controller: StageController,
    glove: GloveOperator | ScriptedGlove | SelectingGlove | None,
    health: HealthMonitor | None = None,
    *,
    tag_router: _EpisodeTagRouter | None = None,
    copilot_attach: Any = None,
) -> int:
    import subprocess
    import sys

    director: StageDirector | DemoRescueDirector | None = None
    if cfg.script:
        director = StageDirector(controller)
    elif cfg.demo_rescue:
        director = DemoRescueDirector(
            controller, shake=Shake().scaled(max(cfg.demo_shake, 0.0)), seed=resolve_seed(cfg)
        )
    dummy_proc: subprocess.Popen[bytes] | None = None
    dummy: Any = None
    # TODO(rokoko-gloves-swap): Temporary dummy UDP streamer + UI override while physical
    # Rokoko Smartgloves are disconnected. Remove/disable once real Studio UDP is connected.
    use_console_dummy = (
        director is None
        and not cfg.headless
        and cfg.display == "console"
        and (cfg.dummy_glove or cfg.source == "synthetic")
    )
    if director is None:
        if cfg.dummy_glove and cfg.source == "studio":
            script_path = REPO_ROOT / "scripts" / "dummy_glove_process.py"
            dummy_proc = subprocess.Popen(
                [
                    sys.executable,
                    str(script_path),
                    "--studio-host",
                    "127.0.0.1",
                    "--studio-port",
                    str(cfg.studio.port),
                ],
                stdin=subprocess.DEVNULL,
            )
        if cfg.source == "synthetic" and use_console_dummy:
            glove = None  # the console dummy operator drives the rescue instead
        if use_console_dummy:
            from teleop.input.dummy_glove import DummyControlClient, DummyOperatorSupervisor

            udp_cli = DummyControlClient() if dummy_proc is not None else None
            dummy = DummyOperatorSupervisor(
                controller,
                udp_client=udp_cli,
                process_pid=dummy_proc.pid if dummy_proc is not None else None,
            )
    co = controller.coordinator
    next_log = 0.0

    def operator() -> OperatorInput | None:
        if director is not None:
            return director.step()
        return glove.latest() if glove is not None else None

    def log_hud() -> None:
        nonlocal next_log
        if cfg.hud_log_period_s <= 0 or co.time_s < next_log:
            return
        next_log = co.time_s + cfg.hud_log_period_s
        logger.info("HUD t=%.1fs\n  %s", co.time_s, "\n  ".join(hud_lines(controller.hud_view())))

    try:
        with glove.running() if glove is not None else contextlib.nullcontext():
            if cfg.headless:
                steps = 0
                while cfg.steps is None or steps < cfg.steps:
                    controller.tick(operator())
                    steps += 1
                    log_hud()
                    if director is not None and director.done:
                        break
                logger.info("atc: %d steps, %.1f s sim time", steps, co.time_s)
            elif cfg.display == "viewer":
                _run_viewer(cfg, controller, operator, health=health)
            else:
                from teleop.console.run import ConsolePortInUse, run_console

                recordings = _recording_desk(cfg, controller, glove)
                if tag_router is not None:
                    tag_router.add(recordings)
                recordings.start()
                try:
                    run_console(
                        controller,
                        operator,
                        port=cfg.console_port,
                        fps=cfg.console_fps,
                        open_browser=cfg.open_browser,
                        steps=cfg.steps,
                        health=health,
                        dummy=dummy,
                        calib=_console_calibrator(cfg, glove),
                        recordings=recordings,
                        copilot_attach=copilot_attach,
                    )
                except ConsolePortInUse as exc:
                    print(f"teleop atc: console port {exc.port} in use (try --console-port)")
                    return 1
                finally:
                    recordings.close()  # finalises a recording left running at quit
    finally:
        if dummy is not None:
            dummy.close()
        if dummy_proc is not None:
            dummy_proc.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                dummy_proc.wait(timeout=3)

    if director is None:
        return 0
    if isinstance(director, DemoRescueDirector):
        took = f"{director.rescue_s:.1f} s" if director.rescue_s is not None else "--"
        gate = "deployed" if director.deployed else "not deployed"
        print(f"demo rescue: {'ok' if director.ok else 'FAILED'} (rescue {took}, skill {gate})")
        return 0 if director.ok else 1
    for r in director.results:
        took = f"{r.end_s - r.start_s:5.1f} s" if r.end_s is not None else "   --  "
        print(f"beat {r.name:<30} {'ok' if r.ok else 'FAILED':<6} {took}", flush=True)
    return 0 if director.ok else 1


CONSOLE_CALIB_PATH = REPO_ROOT / "data" / "calib" / "atc.json"
CONSOLE_EPISODES_ROOT = REPO_ROOT / "data" / "episodes"


def _recording_desk(
    cfg: AtcConfig,
    controller: StageController,
    glove: GloveOperator | ScriptedGlove | SelectingGlove | None,
) -> Any:
    """Start/stop/delete episode recordings from the console's Data page."""
    from teleop.console.recordings import RecordingDesk

    co = controller.coordinator
    unavailable: str | None = None
    if cfg.record_episodes is not None:
        unavailable = f"Recording every rescue to {cfg.record_episodes} (--record-episodes)"
    elif importlib.util.find_spec("lerobot") is None:
        unavailable = "Recording needs lerobot: run `uv sync --group train`"
    label: str | Callable[[], str] = (
        "glove" if isinstance(glove, GloveOperator) else rescue_label([glove])
    )
    if glove is None or isinstance(glove, ScriptedGlove):
        label = "scripted"

    def open_sink(folder: Path) -> Any:
        from teleop.atc.episodes import EpisodeSink, fault_lookup

        return EpisodeSink(folder, co.model, co.data, fault_lookup(controller), label)

    return RecordingDesk(CONSOLE_EPISODES_ROOT, open_sink, co.set_observer, unavailable=unavailable)


def _console_calibrator(
    cfg: AtcConfig, glove: GloveOperator | ScriptedGlove | SelectingGlove | None
) -> Any:
    """The console's glove calibration wizard, or ``None`` without a real glove operator."""
    inner = glove.glove if isinstance(glove, SelectingGlove) else glove
    if not isinstance(inner, GloveOperator) or inner.hand is None:
        return None
    from teleop.console.calib import GloveCalibrator

    return GloveCalibrator(
        inner.hand,
        inner.source,
        inner.recalibrate,
        persist_path=CONSOLE_CALIB_PATH,
        loaded_from=cfg.calibration,
    )


def _run_viewer(
    cfg: AtcConfig,
    controller: StageController,
    operator: Any,
    launch: Any = None,
    health: HealthMonitor | None = None,
) -> None:
    """Drive the passive viewer at ``CONTROL_HZ``.

    ``viewer.set_texts`` must never run under ``viewer.lock()``: MuJoCo's ``set_texts`` spins
    until the render thread has consumed the previous text request, and the render thread needs
    that lock, so calling it under the lock freezes the app at 100% CPU. Each call also waits
    for a rendered frame, so the overlay is refreshed at ``HUD_OVERLAY_HZ`` and only on change.
    """
    if launch is None:
        import mujoco.viewer  # pyright: ignore[reportMissingTypeStubs]

        launch = mujoco.viewer.launch_passive  # pyright: ignore
    co = controller.coordinator
    bg_player = None
    if getattr(controller, "background_bays", ()):
        from teleop.console.background import BackgroundPlayer, load_cycle

        bg_player = BackgroundPlayer(load_cycle(), co.model)
    with launch(co.model, co.data, key_callback=controller.on_key) as viewer:
        next_overlay_s = 0.0
        last_overlay: Any = None

        def step(_: int) -> bool:
            nonlocal next_overlay_s, last_overlay
            if not viewer.is_running():
                return False
            sample = operator()
            with viewer.lock():
                controller.tick(sample)
                if bg_player is not None:
                    import mujoco as _mj  # pyright: ignore[reportMissingTypeStubs]

                    mj_any: Any = _mj
                    bg_player.apply(co.data, co.time_s, frozen=controller.console.powered_off)
                    mj_any.mj_kinematics(co.model, co.data)
                overlay = None
                if co.time_s >= next_overlay_s:
                    next_overlay_s = co.time_s + 1.0 / HUD_OVERLAY_HZ
                    view = controller.hud_view()
                    overlay = (
                        hud_overlays(view, health.snapshot()) if health else hud_overlays(view)
                    )
            if overlay is not None and overlay != last_overlay:
                viewer.set_texts(overlay)  # outside the lock (see docstring)
                last_overlay = overlay
            viewer.sync()
            return True

        stats = FixedRateLoop(CONTROL_HZ).run(step, cfg.steps)
        logger.info("atc: %d steps, %d loop overruns", stats.steps, stats.overruns)
