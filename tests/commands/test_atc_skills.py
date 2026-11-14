"""T-231: ``teleop atc --skills`` / ``--fresh-skills`` config and startup wiring (no hardware)."""

from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from teleop.atc.types import SkillPrimitive
from teleop.commands import atc
from teleop.commands.atc import DEFAULT_SKILLS, load_config


def _overrides(*argv: str) -> dict[str, Any]:
    ns = atc._parser().parse_args(list(argv))  # pyright: ignore[reportPrivateUsage]
    return {k: v for k, v in vars(ns).items() if k in {"skills", "fresh_skills", "headless"}}


def test_live_run_defaults_to_the_library_file() -> None:
    assert load_config(None, _overrides()).skills == DEFAULT_SKILLS
    assert Path("data/skills/library.json") == DEFAULT_SKILLS


def test_headless_has_no_default_library() -> None:
    assert load_config(None, _overrides("--headless")).skills is None


def test_skills_flag_overrides_both_modes() -> None:
    assert load_config(None, _overrides("--skills", "x.json")).skills == Path("x.json")
    assert load_config(None, _overrides("--headless", "--skills", "x.json")).skills == Path(
        "x.json"
    )


def test_fresh_skills_flag_parses() -> None:
    assert load_config(None, _overrides()).fresh_skills is False
    assert load_config(None, _overrides("--fresh-skills")).fresh_skills is True


@dataclass
class _Cfg:
    part_sku_id: str = "sku"


@dataclass
class _Coordinator:
    library: dict[str, SkillPrimitive] = field(default_factory=dict)
    skill_listeners: list[Any] = field(default_factory=list)
    config: _Cfg = field(default_factory=_Cfg)
    cleared: int = 0

    def reset_library(self) -> bool:
        self.cleared += 1
        self.library = {}
        return True


def _controller() -> Any:
    @dataclass
    class C:
        coordinator: _Coordinator

    return C(_Coordinator())


def _file(path: Path) -> None:
    path.write_text(json.dumps({"version": 1, "skills": []}), encoding="utf-8")


def test_setup_loads_and_subscribes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "library.json"
    _file(path)
    cfg = load_config(None, {"skills": path})
    controller = _controller()
    with contextlib.ExitStack() as stack:
        atc._setup_skills(cfg, controller, stack, None)  # pyright: ignore[reportPrivateUsage]
        assert controller.coordinator.cleared == 1
        assert len(controller.coordinator.skill_listeners) == 1
    assert f"loaded 0 skills from {path} (skipped 0)" in capsys.readouterr().out


def test_fresh_skills_ignores_file(tmp_path: Path) -> None:
    path = tmp_path / "library.json"
    _file(path)
    cfg = load_config(None, {"skills": path, "fresh_skills": True})
    controller = _controller()
    with contextlib.ExitStack() as stack:
        atc._setup_skills(cfg, controller, stack, None)  # pyright: ignore[reportPrivateUsage]
    assert controller.coordinator.cleared == 1  # board and library cleared, file not loaded
    assert not path.exists()
    assert len(list(tmp_path.glob("library-*.json.bak"))) == 1
    assert len(controller.coordinator.skill_listeners) == 1  # the next deploy writes a new file


def test_no_skills_path_touches_nothing(tmp_path: Path) -> None:
    cfg = load_config(None, {"headless": True})
    controller = _controller()
    with contextlib.ExitStack() as stack:
        atc._setup_skills(cfg, controller, stack, None)  # pyright: ignore[reportPrivateUsage]
    assert controller.coordinator.cleared == 0 and controller.coordinator.skill_listeners == []
