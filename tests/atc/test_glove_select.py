from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

from teleop.atc.glove_select import SelectingGlove


class _Ctl:
    def __init__(self) -> None:
        self.bay: int | None = 1
        self.console = SimpleNamespace(clutch=True, selected_bay=1)
        self.notes: list[tuple[str, str]] = []

    def rescue_bay(self) -> int | None:
        return self.bay

    def note(self, text: str, level: str = "warn") -> None:
        self.notes.append((text, level))


class _Src:
    def __init__(self, value: object) -> None:
        self.value = value

    def latest(self) -> object:
        return self.value


def _make(ok: list[bool]) -> tuple[SelectingGlove, _Ctl]:
    ctl = _Ctl()
    sel = SelectingGlove(
        cast("Any", ctl),
        cast("Any", _Src("glove")),
        cast("Any", _Src("scripted")),
        lambda: ok[0],
    )
    return sel, ctl


def test_glove_used_when_ok_at_engage() -> None:
    sel, ctl = _make([True])
    assert sel.latest() == "glove"
    assert sel.rescue_input == "glove"
    assert ctl.notes == []


def test_scripted_used_when_glove_down_at_engage() -> None:
    sel, ctl = _make([False])
    assert sel.latest() == "scripted"
    assert sel.rescue_input == "scripted"
    assert ctl.notes == [("gloves offline: playing scripted rescue", "warn")]
    sel.latest()
    assert len(ctl.notes) == 1  # noted once per rescue


def test_glove_dropout_mid_rescue_holds() -> None:
    ok = [True]
    sel, _ = _make(ok)
    assert sel.latest() == "glove"
    ok[0] = False
    assert sel.latest() == "glove"
    assert sel.rescue_input == "glove"


def test_choice_resets_on_release() -> None:
    ok = [True]
    sel, ctl = _make(ok)
    assert sel.latest() == "glove"
    ctl.console.clutch = False
    assert sel.latest() is None
    ok[0] = False
    ctl.console.clutch = True
    assert sel.latest() == "scripted"
    assert sel.rescue_input == "scripted"


def test_not_engaged_returns_none() -> None:
    sel, ctl = _make([True])
    ctl.console.clutch = False
    assert sel.latest() is None
    assert sel.rescue_input is None
    ctl.console.clutch = True
    ctl.bay = None
    assert sel.latest() is None
    assert sel.rescue_input is None
