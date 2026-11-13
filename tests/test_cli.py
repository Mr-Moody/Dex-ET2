import sys
import types

import pytest

from teleop import cli


def test_unknown_command_returns_2() -> None:
    assert cli.main(["nope"]) == 2


def test_unimplemented_command_reports_cleanly(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "COMMANDS", {"x": "teleop.commands.does_not_exist"})
    assert cli.main(["x"]) == 3
    assert "not implemented yet" in capsys.readouterr().err


def test_dispatch_passes_remaining_args(monkeypatch: pytest.MonkeyPatch) -> None:
    received: list[list[str]] = []

    def fake_main(argv: list[str]) -> int:
        received.append(argv)
        return 0

    module = types.ModuleType("fake_cmd")
    module.main = fake_main  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fake_cmd", module)
    monkeypatch.setattr(cli, "COMMANDS", {"x": "fake_cmd"})
    assert cli.main(["x", "--a", "1"]) == 0
    assert received == [["--a", "1"]]
