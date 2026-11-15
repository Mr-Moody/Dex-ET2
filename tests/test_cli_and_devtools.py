"""Top-level ``teleop`` CLI dispatch and ``uv run task`` atomic git-ref claim workflow."""

from __future__ import annotations

import importlib
import subprocess
import sys
import types
from pathlib import Path

import pytest

from teleop import cli
from teleop.devtools.task import cmd_claim, load_tasks, main, save_task


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=cwd, check=False, capture_output=True, text=True)


def _write_task(
    tasks: Path, tid: str, status: str = "todo", deps: str = "[]", body: str = "- [ ] x\n"
) -> Path:
    p = tasks / f"{tid}-item.md"
    p.write_text(
        f"---\nid: {tid}\ntitle: Item {tid}\nstatus: {status}\nowner: null\n"
        f"branch: null\ndepends_on: {deps}\nplan: null\nclaimed_at: null\n---\n{body}",
        encoding="utf-8",
        newline="",
    )
    return p


@pytest.fixture
def task_repos(tmp_path: Path) -> tuple[Path, Path, Path]:
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "init", "--bare", "-q", str(origin)], check=True)
    clones: list[Path] = []
    for name in ("a", "b"):
        c = tmp_path / name
        subprocess.run(["git", "init", "-q", str(c)], check=True)
        _git(c, "config", "user.name", "Test")
        _git(c, "config", "user.email", "t@example.com")
        _git(c, "config", "commit.gpgsign", "false")
        _git(c, "remote", "add", "origin", str(origin))
        clones.append(c)
    a, b = clones
    (a / "tasks").mkdir()
    _write_task(a / "tasks", "T-1")
    _git(a, "add", ".")
    assert _git(a, "commit", "-q", "-m", "init").returncode == 0
    assert _git(a, "push", "-q", "origin", "HEAD:refs/heads/main").returncode == 0
    assert _git(b, "pull", "-q", "origin", "main").returncode == 0
    return origin, a, b


def test_cli_commands_importable_and_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    for subcmd, mod_path in cli.COMMANDS.items():
        mod = importlib.import_module(mod_path)
        assert callable(getattr(mod, "main", None)), f"{subcmd} ({mod_path}) missing main()"

    assert cli.main(["unknown-subcommand"]) == 2
    monkeypatch.setattr(cli, "COMMANDS", {"missing": "teleop.commands.does_not_exist"})
    assert cli.main(["missing"]) == 3

    received: list[list[str]] = []
    fake_mod = types.ModuleType("fake_cmd")
    fake_mod.main = lambda argv: received.append(argv) or 0  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fake_cmd", fake_mod)
    monkeypatch.setattr(cli, "COMMANDS", {"x": "fake_cmd"})
    assert cli.main(["x", "--flag", "1"]) == 0
    assert received == [["--flag", "1"]]


def test_task_frontmatter_roundtrip_and_ready_dependencies(tmp_path: Path) -> None:
    body = "\r\n  body with CRLF\n\n- [ ] step\n---\nliteralrule\n"
    _write_task(tmp_path, "T-1", body=body)
    _write_task(tmp_path, "T-2", deps="[T-1]")
    tasks = {t.id: t for t in load_tasks(tmp_path)}
    assert tasks["T-1"].body == body
    assert not tasks["T-2"].is_ready(set())

    tasks["T-1"].status = "done"
    save_task(tasks["T-1"])
    reloaded = {t.id: t for t in load_tasks(tmp_path)}
    assert reloaded["T-1"].body == body
    assert reloaded["T-2"].is_ready({"T-1"})


def test_atomic_git_claim_race_release_and_done(
    task_repos: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    origin, a, b = task_repos
    monkeypatch.chdir(a)
    assert main(["claim", "T-1", "--agent", "agentA"]) == 0
    assert _git(origin, "show-ref", "refs/claims/T-1").returncode == 0
    assert main(["claim", "T-1", "--agent", "agentA2"]) == 1

    # Clone b skips fetch to simulate a simultaneous push race: origin rejects and b rolls back.
    capsys.readouterr()
    assert cmd_claim(b, "T-1", "agentB", fetch=False) == 1
    assert "lost race" in capsys.readouterr().err
    assert _git(b, "show-ref", "refs/claims/T-1").returncode != 0
    assert load_tasks(b / "tasks")[0].status == "todo"

    # Releasing on a deletes local + remote ref so b can claim cleanly, then mark done.
    assert main(["release", "T-1"]) == 0
    assert _git(origin, "show-ref", "refs/claims/T-1").returncode != 0
    monkeypatch.chdir(b)
    assert main(["claim", "T-1", "--agent", "agentB"]) == 0
    assert main(["done", "T-1"]) == 0
    assert _git(origin, "show-ref", "refs/claims/T-1").returncode != 0
    assert load_tasks(b / "tasks")[0].status == "done"
