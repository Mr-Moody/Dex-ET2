"""``teleop atc`` runs through the fleet ops console by default."""

from __future__ import annotations

import json
import socket
import sys
import threading
from pathlib import Path
from typing import Any

import pytest
from websockets.sync.client import connect

from teleop.commands import atc
from teleop.commands.atc import DEFAULT_CONFIG, _parser, load_config
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


def test_config_defaults_and_flags() -> None:
    cfg = load_config(None, {})
    assert (cfg.display, cfg.console_port, cfg.console_fps, cfg.open_browser) == (
        "console",
        8765,
        30,
        True,
    )
    ns = _parser().parse_args(["--display", "viewer", "--no-browser", "--console-port", "9000"])
    assert ns.display == "viewer" and ns.open_browser is False and ns.console_port == 9000
    ns = _parser().parse_args(["--console-fps", "24"])
    assert ns.console_fps == 24 and ns.display is None and ns.open_browser is None


def test_shipped_yaml_loads() -> None:
    cfg = load_config(DEFAULT_CONFIG, {})
    assert cfg.display == "console" and cfg.console_port == 8765


def test_bad_display_is_a_config_error(tmp_path: Path) -> None:
    bad = tmp_path / "atc.yaml"
    bad.write_text("display: hologram\n", encoding="utf-8")
    with pytest.raises(ValueError, match="display"):
        load_config(bad, {})


# ------------------------------------------------------------------ end to end


class _Sink:
    def emit(self, event: Any) -> None:
        pass

    def close(self) -> None:
        pass


class _Reasoner:
    def submit(self, ctx: Any) -> None:
        pass

    def latest_verdict(self, bay_id: int) -> None:
        return None

    def close(self) -> None:
        pass


@pytest.fixture
def controller() -> Any:
    from teleop.atc.stage import AudioCues, NullSpeaker, StageController
    from teleop.unoq.client import UnoQClient

    unoq = UnoQClient.connect_or_mock("127.0.0.1", 1, 1, connect_timeout_s=0.2)
    ctl = StageController.build(
        unoq,
        _Sink(),
        _Reasoner(),
        audio=AudioCues(NullSpeaker(), enabled=False),
        include_background_bays=True,
    )
    yield ctl
    ctl.close()
    unoq.close()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@needs_models
def test_run_console_ticks_and_streams(offscreen_render_ok: None, controller: Any) -> None:
    from teleop.console.run import run_console

    port = _free_port()
    result: dict[str, Any] = {}

    def run() -> None:
        result["stats"] = run_console(
            controller, lambda: None, port=port, fps=10, open_browser=False, steps=240
        )

    t = threading.Thread(target=run)
    t.start()
    got_telemetry = got_frame = False
    try:
        for _ in range(50):
            try:
                ws = connect(f"ws://127.0.0.1:{port}/ws", max_size=None, open_timeout=1)
                break
            except OSError:
                threading.Event().wait(0.1)
        else:
            pytest.fail("console server never came up")
        with ws:
            while not (got_telemetry and got_frame):
                msg = ws.recv(timeout=10)
                if isinstance(msg, str):
                    body = json.loads(msg)
                    assert body["type"] == "telemetry"
                    assert [b["id"] for b in body["bays"]] == [1, 2, 3, 4]
                    got_telemetry = True
                else:
                    assert msg[2:4] == b"\xff\xd8"
                    got_frame = True
    finally:
        t.join(timeout=60)
    assert got_telemetry and got_frame
    assert result["stats"].steps == 240
    assert controller.coordinator.time_s == pytest.approx(4.0, abs=0.1)


@needs_models
def test_headless_does_not_import_console(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    for name in ("teleop.console.run", "teleop.console.render", "teleop.console.server"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    rc = atc.main(_headless_args(tmp_path) + ["--headless", "--steps", "5"])
    assert rc == 0
    assert "teleop.console.render" not in sys.modules
    assert "teleop.console.server" not in sys.modules


@needs_models
def test_busy_port_exits_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    blocker.bind(("127.0.0.1", 0))
    blocker.listen()
    port = int(blocker.getsockname()[1])
    try:
        rc = atc.main(
            _headless_args(tmp_path)
            + ["--console-port", str(port), "--no-browser", "--steps", "1", "--no-audio"]
        )
    finally:
        blocker.close()
    assert rc == 1
    assert f"console port {port} in use" in capsys.readouterr().out


def _headless_args(tmp_path: Path) -> list[str]:
    return [
        "--source",
        "synthetic",
        "--unoq-host",
        "127.0.0.1",
        "--unoq-tcp-port",
        "1",
        "--env-file",
        str(tmp_path / "missing.env"),
    ]
