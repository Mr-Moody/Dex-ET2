"""``teleop atc`` as the one booth command: hotspot, glove, panel health wiring (no hardware)."""

from __future__ import annotations

import contextlib
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

from teleop.commands import atc
from teleop.commands.atc import DEFAULT_CONFIG, AtcConfig, load_config
from teleop.health.model import Level
from teleop.sim.bays import PANDA_XML
from teleop.sim.hands import load_hand


def _models_present() -> bool:
    try:
        right = load_hand("inspire").mjcf["right"]
    except FileNotFoundError:
        return False
    return PANDA_XML.is_file() and right.is_file()


needs_models = pytest.mark.skipif(
    not _models_present(), reason="Panda/Inspire MJCFs missing; run scripts/fetch_models.py"
)

HEALTH_THREADS = ("health-poll", "health-panel", "health-reconnect", "unoq-console")


def _args(tmp_path: Path, *extra: str) -> list[str]:
    """No board (nothing listens on TCP port 1: loopback node) and no .env."""
    return [
        "--unoq-host",
        "127.0.0.1",
        "--unoq-tcp-port",
        "1",
        "--env-file",
        str(tmp_path / "missing.env"),
        "--no-audio",
        "--no-browser",
        *extra,
    ]


@pytest.fixture
def booth(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Stub ``run_console`` (returns at once) and spy on the hotspot builder (real, no password)."""
    import teleop.console.run as console_run

    monkeypatch.chdir(tmp_path)  # compliance audit log lands in tmp_path/outputs
    for var in ("SOLANA_RPC_URL", "SOLANA_KEYPAIR_PATH", "VERDA_API_KEY", "HOTSPOT_PASSWORD"):
        monkeypatch.delenv(var, raising=False)
    seen: dict[str, Any] = {"builds": []}
    real_build = atc.build_hotspot_check

    def spy_build(enabled: bool, env_file: Path, *, strict: bool) -> Any:
        seen["builds"].append((enabled, strict))
        return real_build(enabled, env_file, strict=strict)

    def fake_run_console(controller: Any, operator: Any, **kwargs: Any) -> None:
        health = kwargs["health"]
        health.poll_once()
        seen["operator"] = operator
        seen["health"] = {h.name: h for h in health.snapshot()}
        seen["order"] = [h.name for h in health.snapshot()]

    monkeypatch.setattr(atc, "build_hotspot_check", spy_build)
    monkeypatch.setattr(console_run, "run_console", fake_run_console)
    return seen


def _no_health_threads_left() -> None:
    alive = [t.name for t in threading.enumerate() if t.name.startswith(HEALTH_THREADS)]
    assert alive == []


def test_default_config_is_one_command() -> None:
    cfg = load_config(DEFAULT_CONFIG, {})
    assert cfg.source == "studio"
    assert cfg.hotspot is True
    assert AtcConfig().hotspot is True


def test_hotspot_flags() -> None:
    ns = atc._parser().parse_args(["--no-hotspot"])  # pyright: ignore[reportPrivateUsage]
    assert ns.no_hotspot is True and ns.hotspot is False
    ns = atc._parser().parse_args([])  # pyright: ignore[reportPrivateUsage]
    assert ns.no_hotspot is False and ns.hotspot is False


def test_hotspot_and_no_hotspot_conflict(capsys: pytest.CaptureFixture[str]) -> None:
    assert atc.main(["--hotspot", "--no-hotspot"]) == 2
    assert "--hotspot and --no-hotspot" in capsys.readouterr().out


def test_explicit_hotspot_without_password_exits_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "platform", "win32")  # the check is Windows-only otherwise
    monkeypatch.chdir(tmp_path)
    rc = atc.main(_args(tmp_path, "--source", "synthetic", "--hotspot"))
    assert rc == 1
    assert "HOTSPOT_PASSWORD" in capsys.readouterr().out


@needs_models
def test_headless_skips_hotspot_and_health(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    started: list[str] = []

    def no_hotspot(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("headless must not build the hotspot check")

    class SpyPanel:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def start(self) -> None:
            started.append("panel")

    class SpyMonitor:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def start(self) -> None:
            started.append("monitor")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(atc, "build_hotspot_check", no_hotspot)
    monkeypatch.setattr(atc, "PanelCheck", SpyPanel)
    monkeypatch.setattr(atc, "HealthMonitor", SpyMonitor)
    rc = atc.main(_args(tmp_path, "--source", "synthetic", "--headless", "--script"))
    assert rc == 0
    assert started == []
    assert capsys.readouterr().out.count(" ok ") == 6
    _no_health_threads_left()


@needs_models
def test_default_run_without_password_starts(
    booth: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from teleop.atc.glove_select import SelectingGlove

    selecting: list[SelectingGlove] = []

    class SpySelecting(SelectingGlove):
        def __init__(self, *args: Any) -> None:
            super().__init__(*args)
            selecting.append(self)

    monkeypatch.setattr(atc, "SelectingGlove", SpySelecting)
    rc = atc.main(_args(tmp_path))  # default config: source studio, hotspot on
    assert rc == 0
    assert booth["builds"] == [(True, False)]  # enabled, not strict
    assert booth["order"] == ["hotspot", "gloves", "panel"]
    assert booth["health"]["hotspot"].level is Level.OFF
    assert booth["health"]["panel"].level is Level.FAIL  # loopback node: keyboard only
    assert booth["health"]["gloves"].level is not Level.OFF  # a real Studio listener
    assert len(selecting) == 1  # studio, not scripted: glove or scripted chosen per rescue
    assert selecting[0].glove_ok() is False  # no Studio packets in a test
    _no_health_threads_left()


@needs_models
def test_no_hotspot_flag_disables(booth: dict[str, Any], tmp_path: Path) -> None:
    rc = atc.main(_args(tmp_path, "--source", "synthetic", "--no-hotspot"))
    assert rc == 0
    assert booth["builds"] == [(False, False)]
    hotspot = booth["health"]["hotspot"]
    assert (hotspot.level, hotspot.summary) == (Level.OFF, "disabled")
    assert booth["health"]["gloves"].summary == "synthetic glove"
    _no_health_threads_left()


def test_swap_listener_refreshes_label_and_consoles() -> None:
    from teleop.unoq.server import UnoQServer

    class FakeController:
        unoq_label = "old"

        def __init__(self) -> None:
            self.installed: list[list[Any]] = []

        def replace_hardware(self, consoles: Any) -> None:
            self.installed.append(list(consoles))

    class FakeUnoQ:
        is_mock = True
        host = "127.0.0.1"

    ctl = FakeController()
    cfg = AtcConfig(unoq_host="127.0.0.1")
    swap_listener: Any = atc._swap_listener  # pyright: ignore[reportPrivateUsage]
    listener = swap_listener(cfg, ctl, " | tail")
    fake = FakeUnoQ()
    listener(fake)
    assert ctl.installed == [[]]
    assert ctl.unoq_label == "UNO Q local loopback | tail"

    server = UnoQServer(host="127.0.0.1", udp_port=0, tcp_port=0, name="test-board").start()
    try:
        cfg.unoq_udp_port, cfg.unoq_tcp_port = server.udp_port, server.tcp_port
        fake.is_mock = False
        listener(fake)
        assert ctl.unoq_label == "UNO Q 127.0.0.1 | tail"
        (poller,) = ctl.installed[-1]
        assert type(poller).__name__ == "UnoQConsolePoller"
        poller.close()
    finally:
        server.stop()


def test_viewer_hud_shows_health(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from teleop.health.model import Health

    class FakeViewer:
        def __enter__(self) -> FakeViewer:
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def is_running(self) -> bool:
            return True

        def lock(self) -> contextlib.AbstractContextManager[None]:
            return contextlib.nullcontext()

        def set_texts(self, texts: Any) -> None:
            pass

        def sync(self) -> None:
            pass

    class FakeController:
        def __init__(self) -> None:
            self.coordinator = SimpleNamespace(model=None, data=None, time_s=0.0)

        def on_key(self, _key: int) -> None:
            pass

        def tick(self, _sample: object) -> None:
            self.coordinator.time_s += 1.0 / atc.CONTROL_HZ

        def hud_view(self) -> object:
            return None

    lights = [Health("gloves", Level.FAIL, "waiting for Studio")]

    class Monitor:
        def snapshot(self) -> list[Health]:
            return lights

    seen: list[Any] = []

    def overlays(_view: object, health: Any) -> list[tuple[int, int, str, str]]:
        seen.append(health)
        return [(0, 0, "", "")]

    monkeypatch.setattr(atc, "hud_overlays", overlays)
    run_viewer: Any = atc._run_viewer  # pyright: ignore[reportPrivateUsage]
    viewer = FakeViewer()
    run_viewer(
        SimpleNamespace(steps=6),
        FakeController(),
        lambda: None,
        launch=lambda *_a, **_k: viewer,
        health=Monitor(),
    )
    assert seen and all(h == lights for h in seen)
