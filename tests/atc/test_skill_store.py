"""T-231: the DMP skill library persists to disk and reloads at startup (no hardware, no MuJoCo)."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from teleop.atc.coordinator import AtcCoordinator
from teleop.atc.skill_store import SkillStore
from teleop.atc.types import SkillPrimitive, StallSignature
from teleop.unoq import protocol as P
from teleop.unoq.client import UnoQClient
from teleop.unoq.dmp import HAND_ACTUATORS
from teleop.unoq.server import UnoQServer

SKU = "ev_latch_v1"
JAM_SIG_1 = (0.035, 2.1, 3.4, 1.2)


def synthetic_rescue(duration_s: float = 4.0, hz: float = 60.0) -> dict[str, Any]:
    """A smooth latch-frame wrist + finger demonstration (identity orientation)."""
    t = np.arange(0.0, duration_s + 1e-9, 1.0 / hz)
    u = t / duration_s
    pos = np.stack([0.02 - 0.06 * u, 0.01 * np.sin(2 * np.pi * u), 0.2 - 0.12 * u], axis=1)
    quat = np.tile([0.0, 0.0, 0.0, 1.0], (len(t), 1))
    fingers = np.stack(
        [
            0.8 * np.sin(np.pi * u * (1 + i / len(HAND_ACTUATORS)))
            for i in range(len(HAND_ACTUATORS))
        ],
        axis=1,
    )
    tau = np.where((u > 0.4) & (u < 0.7), 3.0, 0.2)
    return {"t": t, "wrist": np.hstack([pos, quat]), "fingers": fingers, "tau": tau}


@dataclass
class FakeConfig:
    part_sku_id: str = SKU


@dataclass
class FakeCoordinator:
    """The slice of ``AtcCoordinator`` the store touches; ``reset_library`` is the real one."""

    unoq: UnoQClient
    config: FakeConfig = field(default_factory=FakeConfig)
    library: dict[str, SkillPrimitive] = field(default_factory=dict)

    def reset_library(self) -> bool:
        return AtcCoordinator.reset_library(self)  # type: ignore[arg-type]


@pytest.fixture
def server() -> Iterator[UnoQServer]:
    srv = UnoQServer(host="127.0.0.1", udp_port=0, tcp_port=0).start()
    try:
        yield srv
    finally:
        srv.stop()


@pytest.fixture
def client(server: UnoQServer) -> Iterator[UnoQClient]:
    c = UnoQClient("127.0.0.1", server.udp_port, server.tcp_port).connect()
    try:
        yield c
    finally:
        c.close()


@pytest.fixture
def coord(client: UnoQClient) -> FakeCoordinator:
    return FakeCoordinator(client)


@pytest.fixture
def skill(client: UnoQClient) -> SkillPrimitive:
    demo = synthetic_rescue()
    return client.fit_dmp(
        SKU, StallSignature(*JAM_SIG_1), demo["t"], demo["wrist"], demo["fingers"], demo["tau"]
    )


def _saved(path: Path, *skills: SkillPrimitive) -> None:
    entries = [
        {"skill": s.to_dict(), "learned_at": "2026-11-14T10:00:00", "session": None} for s in skills
    ]
    path.write_text(json.dumps({"version": 1, "skills": entries}), encoding="utf-8")


def _ids(path: Path) -> list[str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [e["skill"]["skill_id"] for e in raw["skills"]]


def test_deploy_then_load_restores_library_and_board(
    tmp_path: Path, coord: FakeCoordinator, skill: SkillPrimitive
) -> None:
    path = tmp_path / "library.json"
    store = SkillStore(path, session=lambda: "20261114-100000")
    store.on_deployed(skill)
    store.close()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["version"] == 1 and data["skills"][0]["session"] == "20261114-100000"
    coord.unoq.clear_skills(SKU)
    report = SkillStore(path).load(coord)  # type: ignore[arg-type]
    assert report.loaded == 1 and report.skipped == []
    assert coord.library[skill.skill_id] == skill
    assert coord.unoq.classify_stall(SKU, skill.prototype)[1] == skill.skill_id


def test_load_clears_board_first(
    tmp_path: Path, coord: FakeCoordinator, skill: SkillPrimitive, server: UnoQServer
) -> None:
    coord.unoq.register_skill(skill)  # stale board state, not in the file
    coord.library["stale"] = skill
    _saved(tmp_path / "library.json")
    assert SkillStore(tmp_path / "library.json").load(coord).loaded == 0  # type: ignore[arg-type]
    assert coord.library == {} and server.skills == {}


def test_load_skips_mismatched_actuators_and_bad_json(
    tmp_path: Path, coord: FakeCoordinator, skill: SkillPrimitive
) -> None:
    path = tmp_path / "library.json"
    wrong = replace(
        skill, skill_id="wrong-hand", actuator_names=tuple(reversed(skill.actuator_names))
    ).to_dict()
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "skills": [
                    {"skill": skill.to_dict(), "learned_at": "t", "session": None},
                    {"skill": wrong, "learned_at": "t", "session": None},
                    {"skill": {"skill_id": "broken"}, "learned_at": "t", "session": None},
                ],
            }
        ),
        encoding="utf-8",
    )
    before = path.read_bytes()
    report = SkillStore(path).load(coord)  # type: ignore[arg-type]
    assert report.loaded == 1
    assert len(report.skipped) == 2 and "wrong-hand" in report.skipped[0]
    assert set(coord.library) == {skill.skill_id}
    assert path.read_bytes() == before


def test_skipped_entries_survive_the_next_save(
    tmp_path: Path, coord: FakeCoordinator, skill: SkillPrimitive
) -> None:
    path = tmp_path / "library.json"
    wrong = replace(
        skill, skill_id="wrong-hand", actuator_names=tuple(reversed(skill.actuator_names))
    ).to_dict()
    path.write_text(
        json.dumps(
            {"version": 1, "skills": [{"skill": wrong, "learned_at": "t", "session": None}]}
        ),
        encoding="utf-8",
    )
    store = SkillStore(path)
    store.load(coord)  # type: ignore[arg-type]
    store.on_deployed(skill)
    store.close()
    assert _ids(path) == ["wrong-hand", skill.skill_id]


def test_truncated_file_loads_nothing_and_is_not_overwritten(
    tmp_path: Path, coord: FakeCoordinator
) -> None:
    path = tmp_path / "library.json"
    path.write_text('{"version": 1, "skills": [{"skill"', encoding="utf-8")
    before = path.read_bytes()
    report = SkillStore(path).load(coord)  # type: ignore[arg-type]
    assert report.loaded == 0 and len(report.skipped) == 1
    assert path.read_bytes() == before


def test_deploy_after_truncated_load_keeps_the_broken_file_aside(
    tmp_path: Path, coord: FakeCoordinator, skill: SkillPrimitive
) -> None:
    path = tmp_path / "library.json"
    path.write_text('{"version": 1, "skills": [{"skill"', encoding="utf-8")
    store = SkillStore(path)
    store.load(coord)  # type: ignore[arg-type]
    store.on_deployed(skill)
    store.close()
    assert _ids(path) == [skill.skill_id]
    aside = list(tmp_path.glob("library-*.json.corrupt"))
    assert len(aside) == 1 and aside[0].read_text(encoding="utf-8").startswith('{"version"')


def test_save_is_atomic_and_keeps_every_skill(tmp_path: Path, skill: SkillPrimitive) -> None:
    path = tmp_path / "sub" / "library.json"
    store = SkillStore(path)
    for i in range(5):
        store.on_deployed(replace(skill, skill_id=f"s{i}"))
    store.close()
    assert list(path.parent.glob("*.tmp")) == []
    ids = _ids(path)
    assert ids == [f"s{i}" for i in range(5)]
    assert [e.skill_id for e in store.entries()] == ids


def test_redeploying_same_skill_id_does_not_duplicate(
    tmp_path: Path, skill: SkillPrimitive
) -> None:
    store = SkillStore(tmp_path / "library.json")
    store.on_deployed(skill)
    store.on_deployed(skill)
    store.close()
    assert len(store.entries()) == 1


def test_reset_backs_up_file_and_clears_everything(
    tmp_path: Path, coord: FakeCoordinator, skill: SkillPrimitive, server: UnoQServer
) -> None:
    path = tmp_path / "library.json"
    store = SkillStore(path)
    coord.unoq.register_skill(skill)
    coord.library[skill.skill_id] = skill
    store.on_deployed(skill)
    store.close()
    report = store.reset(coord)  # type: ignore[arg-type]
    assert report.board_cleared is True
    assert report.backup is not None and report.backup.exists()
    assert report.backup.name.startswith("library-") and report.backup.name.endswith(".json.bak")
    assert not path.exists()
    assert coord.library == {} and server.skills == {} and store.entries() == []


def test_reset_with_nothing_saved_has_no_backup(tmp_path: Path, coord: FakeCoordinator) -> None:
    report = SkillStore(tmp_path / "library.json").reset(coord)  # type: ignore[arg-type]
    assert report.backup is None and report.board_cleared is True


def test_reset_with_old_firmware_reports_board_not_cleared(
    tmp_path: Path, coord: FakeCoordinator, skill: SkillPrimitive, server: UnoQServer
) -> None:
    del server._handlers[P.OP_CLEAR_SKILLS]
    coord.library[skill.skill_id] = skill
    report = SkillStore(tmp_path / "library.json").reset(coord)  # type: ignore[arg-type]
    assert report.board_cleared is False
    assert coord.library == {}  # the laptop side is cleared anyway


def test_reset_library_rebinds_so_running_readers_keep_their_dict(
    coord: FakeCoordinator, skill: SkillPrimitive
) -> None:
    coord.library[skill.skill_id] = skill
    held = coord.library
    coord.reset_library()
    assert coord.library is not held and coord.library == {}
    assert held[skill.skill_id] is skill  # a bay mid-skill keeps its own object


def test_reset_discards_a_pending_write(
    tmp_path: Path, coord: FakeCoordinator, skill: SkillPrimitive
) -> None:
    path = tmp_path / "library.json"
    store = SkillStore(path)
    store.on_deployed(skill)
    store.reset(coord)  # type: ignore[arg-type]
    store.close()  # the writer drains; it must not resurrect the old library
    assert not path.exists()


def test_deploy_after_reset_writes_only_the_new_skill(
    tmp_path: Path, coord: FakeCoordinator, skill: SkillPrimitive
) -> None:
    path = tmp_path / "library.json"
    store = SkillStore(path)
    store.on_deployed(skill)
    store.close()
    store.reset(coord)  # type: ignore[arg-type]
    store.on_deployed(replace(skill, skill_id="after"))
    store.close()
    assert _ids(path) == ["after"]


def test_on_deployed_never_writes_on_the_caller_thread(
    tmp_path: Path, skill: SkillPrimitive, monkeypatch: pytest.MonkeyPatch
) -> None:
    import teleop.atc.skill_store as mod

    writers: list[str] = []
    real = mod.os.replace

    def spy(src: object, dst: object) -> None:
        writers.append(threading.current_thread().name)
        real(src, dst)  # type: ignore[arg-type]

    monkeypatch.setattr(mod.os, "replace", spy)
    store = SkillStore(tmp_path / "library.json")
    store.on_deployed(skill)
    store.close()
    assert writers and threading.current_thread().name not in writers


def test_reset_during_autonomous_skill_lets_it_finish(
    tmp_path: Path, coord: FakeCoordinator, skill: SkillPrimitive
) -> None:
    """A bay replaying a skill holds its own object; reset neither touches it nor re-saves."""
    path = tmp_path / "library.json"
    store = SkillStore(path)
    running = skill  # what bay.auto.skill points at
    coord.library[skill.skill_id] = skill
    store.reset(coord)  # type: ignore[arg-type]
    assert running.n_kernels > 0 and coord.library == {}
    store.close()
    assert not path.exists()
