"""``teleop atc``: 4-bay fleet supervisor with HUD, keyboard stage controls and audio cues.

    teleop atc [--source studio|synthetic|replay] [--unoq-host HOST] [--headless --steps N]
               [--script] [--hotspot | --no-hotspot] [--config configs/atc.yaml]

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

Recovery after a matched stall: ``--recovery-backend dmp`` (default, the learned DMP skills),
``hybrid`` (a gated ACT checkpoint from ``--act-checkpoint``, DMP behind it) or ``act``; ``M``
cycles it live. ``--record-episodes DIR`` saves every recovery as a LeRobot episode and
``--headless --generate N`` produces N scripted rescue episodes. These load their modules only
when used, so the default run needs no torch.

Hooks come from the environment (``.env``): ``SOLANA_RPC_URL``/``SOLANA_KEYPAIR_PATH`` for the
compliance chain and ``VERDA_API_KEY``/``VERDA_BASE_URL``/``NEMOTRON_MODEL`` for the reasoner.
Both fall back to local/offline modes.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import importlib
import logging
from collections.abc import Callable, Generator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import yaml

from teleop.app import make_source
from teleop.atc.coordinator import OperatorInput
from teleop.atc.glove_select import SelectingGlove
from teleop.atc.hud import hud_lines, hud_overlays
from teleop.atc.recovery import RecoveryBackend
from teleop.atc.stage import (
    AudioCues,
    ConsoleSource,
    NullSpeaker,
    ScriptedGlove,
    StageController,
    StageDirector,
    SystemSpeaker,
    UnoQConsolePoller,
)
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
from teleop.sim.hands import REPO_ROOT, load_hand
from teleop.sim.loop import FixedRateLoop
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
    audio: bool = True
    script: bool = False
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
    seed: int = 0


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

    def __init__(self, source: InputSource, wrist: WristSource, retargeter: Retargeter) -> None:
        self.source = source
        self.wrist = wrist
        self.retargeter = retargeter

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
        )

    def latest(self) -> OperatorInput | None:
        frame = self.source.latest().right
        if frame is None:
            return None
        targets = self.retargeter.retarget(frame).positions
        return OperatorInput(self.wrist.wrist(frame).pose, dict(targets))

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
    ap.add_argument("--no-audio", dest="audio", action="store_false", default=None)
    ap.add_argument("--no-background", dest="background_bays", action="store_false", default=None)
    ap.add_argument("--display", choices=DISPLAYS, default=None, help="console (default) or viewer")
    ap.add_argument("--console-port", type=int, default=None)
    ap.add_argument("--console-fps", type=int, default=None)
    ap.add_argument("--no-browser", dest="open_browser", action="store_false", default=None)
    ap.add_argument("--recovery-backend", choices=BACKENDS, default=None)
    ap.add_argument("--act-checkpoint", type=Path, default=None, help="ACT run dir with gate.json")
    ap.add_argument("--act-device", default=None, help="torch device for ACT (default cuda)")
    ap.add_argument("--record-episodes", type=Path, default=None, help="LeRobot dataset dir")
    ap.add_argument("--generate", type=int, default=None, help="headless: N scripted episodes")
    ap.add_argument("--seed", type=int, default=None)
    add_hotspot_args(ap)
    ap.add_argument("--no-hotspot", action="store_true", help="do not keep the booth hotspot up")
    return ap


def _hardware_consoles(cfg: AtcConfig, unoq: SwitchableUnoQ) -> list[ConsoleSource]:
    if cfg.serial_port:
        from teleop.unoq.mcu_bridge import open_console

        transport, kind = open_console(cfg.serial_port)
        bridge = McuConsoleBridge(transport)
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


def _swap_listener(
    cfg: AtcConfig, controller: StageController, suffix: str
) -> Callable[[SwitchableUnoQ], None]:
    """``SwitchableUnoQ.on_swap``: restart the board console poller and refresh the header.

    Runs on the swapping thread. A laptop serial console does not depend on the board link, so
    it stays installed (the panel check's Reconnect reopens it).
    """

    def on_swap(unoq: SwitchableUnoQ) -> None:
        if not cfg.serial_port:
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
    if ns.env_file.is_file():
        from dotenv import load_dotenv

        load_dotenv(ns.env_file)
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


def _run_booth(
    cfg: AtcConfig,
    hooks: Mapping[str, Any],
    hotspot_check: HotspotCheck | None,
    stack: contextlib.ExitStack,
) -> int:
    """Build the stage, the operator and (unless headless) the health checks, then run."""
    from teleop.compliance.solana import SolanaComplianceSink
    from teleop.reasoning.nemotron import VerdaNemotronReasoner

    unoq = SwitchableUnoQ(
        UnoQClient.connect_or_mock(cfg.unoq_host, cfg.unoq_udp_port, cfg.unoq_tcp_port)
    )
    stack.callback(unoq.close)
    sink = SolanaComplianceSink.from_env()
    reasoner = VerdaNemotronReasoner.from_env()
    speak = cfg.audio and not cfg.headless
    hardware = _hardware_consoles(cfg, unoq)
    suffix = f" | compliance {sink.mode} | reasoner {reasoner.mode}"
    controller = StageController.build(
        cast("UnoQClient", unoq),  # same request surface; the coordinator only forwards calls
        sink,
        reasoner,
        audio=AudioCues(SystemSpeaker() if speak else NullSpeaker(), enabled=speak),
        hardware=hardware,
        cycle_dwell_s=cfg.cycle_dwell_s,
        include_background_bays=cfg.background_bays,
        unoq_label=_unoq_label(cfg, unoq) + suffix,
    )
    stack.callback(controller.close)
    logger.info("atc: %s", controller.unoq_label)
    stack.callback(_close_act, controller)
    _setup_recovery(cfg, controller, hooks.get("--act-checkpoint"))
    sink_handle: Any = None
    if "--record-episodes" in hooks:
        scripted = cfg.generate is not None or cfg.source == "synthetic"
        # TODO(T-191): with gloves, SelectingGlove may play the scripted rescue instead
        # (gloves unhealthy at clutch engage), so this startup-wide label can be wrong per
        # episode. Once the episode sink takes a callable label, pass
        # ``lambda: selecting.rescue_input or "glove"`` for a SelectingGlove operator.
        source = "scripted" if scripted else "glove"
        sink_handle = hooks["--record-episodes"](cfg.record_episodes, controller, source)
        controller.coordinator.set_observer(sink_handle)
        stack.callback(_close_episodes, controller, sink_handle)
    if "--generate" in hooks:
        assert cfg.generate is not None
        n = int(hooks["--generate"](controller, sink_handle, cfg.generate, cfg.seed))
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
        panel = PanelCheck(
            unoq,
            cfg.unoq_host,
            cfg.unoq_udp_port,
            cfg.unoq_tcp_port,
            serial_port=cfg.serial_port,
            local_bridge=bridge,
        )
        unoq.on_swap.append(_swap_listener(cfg, controller, suffix))
        stack.callback(unoq.on_swap.clear)
        panel.start()
        stack.callback(panel.stop)
        health = HealthMonitor([hotspot_check, glove_check, panel])
        health.start()
        stack.callback(health.stop)
    return _run(cfg, controller, glove, health)


def _build_glove(
    cfg: AtcConfig, controller: StageController
) -> tuple[GloveOperator | ScriptedGlove | None, StudioSource | None]:
    """The operator input (None under ``--script``) and its Studio listener, if any."""
    if cfg.script:
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
) -> int:
    director = StageDirector(controller) if cfg.script else None
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

            try:
                run_console(
                    controller,
                    operator,
                    port=cfg.console_port,
                    fps=cfg.console_fps,
                    open_browser=cfg.open_browser,
                    steps=cfg.steps,
                    health=health,
                )
            except ConsolePortInUse as exc:
                print(f"teleop atc: console port {exc.port} in use (try --console-port)")
                return 1

    if director is None:
        return 0
    for r in director.results:
        took = f"{r.end_s - r.start_s:5.1f} s" if r.end_s is not None else "   --  "
        print(f"beat {r.name:<30} {'ok' if r.ok else 'FAILED':<6} {took}", flush=True)
    return 0 if director.ok else 1


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
