from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from teleop.devtools.task import TaskFile, load_tasks, main, save_task


def git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=cwd, check=False, capture_output=True, text=True)


def write_task(
    tasks: Path, tid: str, status: str = "todo", deps: str = "[]", body: str = "- [ ] x\n"
) -> Path:
    p = tasks / f"{tid}-thing.md"
    p.write_text(
        f"---\nid: {tid}\ntitle: Thing {tid}\nstatus: {status}\nowner: null\n"
        f"branch: null\ndepends_on: {deps}\nplan: null\nclaimed_at: null\n---\n{body}",
        encoding="utf-8",
        newline="",
    )
    return p


@pytest.fixture
def repos(tmp_path: Path) -> tuple[Path, Path, Path]:
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "init", "--bare", "-q", str(origin)], check=True)
    clones: list[Path] = []
    for name in ("a", "b"):
        c = tmp_path / name
        subprocess.run(["git", "init", "-q", str(c)], check=True)
        git(c, "config", "user.name", "Test")
        git(c, "config", "user.email", "t@example.com")
        git(c, "config", "commit.gpgsign", "false")
        git(c, "remote", "add", "origin", str(origin))
        clones.append(c)
    a, b = clones
    (a / "tasks").mkdir()
    write_task(a / "tasks", "T-1")
    git(a, "add", ".")
    assert git(a, "commit", "-q", "-m", "init").returncode == 0
    assert git(a, "push", "-q", "origin", "HEAD:refs/heads/main").returncode == 0
    assert git(b, "pull", "-q", "origin", "main").returncode == 0
    return origin, a, b


def test_frontmatter_roundtrip_preserves_body(tmp_path: Path) -> None:
    body = "\r\n  weird body\n\n- [ ] a\n\ttabbed  \n---\nnot frontmatter\n"
    write_task(tmp_path, "T-1", deps="[T-0]", body=body)
    (t,) = load_tasks(tmp_path)
    assert t.depends_on == ["T-0"]
    assert t.body == body
    t.status = "claimed"
    save_task(t)
    (t2,) = load_tasks(tmp_path)
    assert t2.status == "claimed"
    assert t2.body == body
    assert t2.path is not None
    assert t2.path.read_bytes().endswith(body.encode("utf-8"))


def test_ready_requires_all_deps_done(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    write_task(tmp_path, "T-1")
    write_task(tmp_path, "T-2", deps="[T-1]")
    tasks = {t.id: t for t in load_tasks(tmp_path)}
    done = {t.id for t in tasks.values() if t.status == "done"}
    assert not tasks["T-2"].is_ready(done)
    t1 = tasks["T-1"]
    t1.status = "done"
    save_task(t1)
    tasks = {t.id: t for t in load_tasks(tmp_path)}
    done = {t.id for t in tasks.values() if t.status == "done"}
    assert tasks["T-2"].is_ready(done)
    assert isinstance(tasks["T-2"], TaskFile)


def test_claim_creates_ref_and_updates_file(repos: tuple[Path, Path, Path], monkeypatch) -> None:
    origin, a, _ = repos
    monkeypatch.chdir(a)
    assert main(["claim", "T-1", "--agent", "agentA"]) == 0
    assert git(a, "show-ref", "refs/claims/T-1").returncode == 0
    assert git(origin, "show-ref", "refs/claims/T-1").returncode == 0
    (t,) = load_tasks(a / "tasks")
    assert t.status == "claimed"
    assert t.owner == "agentA"
    assert t.claimed_at is not None


def test_second_claim_same_clone_fails(
    repos: tuple[Path, Path, Path], monkeypatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _, a, _ = repos
    monkeypatch.chdir(a)
    assert main(["claim", "T-1", "--agent", "agentA"]) == 0
    capsys.readouterr()
    assert main(["claim", "T-1", "--agent", "agentA2"]) == 1
    assert "already claimed" in capsys.readouterr().err


def test_claim_race_two_clones(
    repos: tuple[Path, Path, Path], monkeypatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _, a, b = repos
    monkeypatch.chdir(a)
    assert main(["claim", "T-1", "--agent", "agentA"]) == 0
    monkeypatch.chdir(b)
    capsys.readouterr()
    assert main(["claim", "T-1", "--agent", "agentB", "--no-fetch"]) == 1
    assert "lost race" in capsys.readouterr().err
    assert git(b, "show-ref", "refs/claims/T-1").returncode != 0
    assert load_tasks(b / "tasks")[0].status == "todo"


def test_claim_after_fetch_sees_existing(
    repos: tuple[Path, Path, Path], monkeypatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _, a, b = repos
    monkeypatch.chdir(a)
    assert main(["claim", "T-1", "--agent", "agentA"]) == 0
    monkeypatch.chdir(b)
    capsys.readouterr()
    assert main(["claim", "T-1", "--agent", "agentB"]) == 1
    assert "already claimed by agentA" in capsys.readouterr().err


def test_release_removes_local_and_remote_ref(repos: tuple[Path, Path, Path], monkeypatch) -> None:
    origin, a, _ = repos
    monkeypatch.chdir(a)
    assert main(["claim", "T-1", "--agent", "agentA"]) == 0
    assert main(["release", "T-1"]) == 0
    assert git(a, "show-ref", "refs/claims/T-1").returncode != 0
    assert git(origin, "show-ref", "refs/claims/T-1").returncode != 0
    (t,) = load_tasks(a / "tasks")
    assert t.status == "todo"
    assert t.owner is None


def test_done_sets_status_and_releases(repos: tuple[Path, Path, Path], monkeypatch) -> None:
    origin, a, _ = repos
    monkeypatch.chdir(a)
    assert main(["claim", "T-1", "--agent", "agentA"]) == 0
    assert main(["done", "T-1"]) == 0
    assert git(origin, "show-ref", "refs/claims/T-1").returncode != 0
    (t,) = load_tasks(a / "tasks")
    assert t.status == "done"


def test_list_shows_owner_and_ready(
    repos: tuple[Path, Path, Path], monkeypatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _, a, _ = repos
    monkeypatch.chdir(a)
    assert main(["claim", "T-1", "--agent", "agentA"]) == 0
    capsys.readouterr()
    assert main(["list"]) == 0
    assert "agentA" in capsys.readouterr().out
    assert main(["list", "--ready"]) == 0
    assert "T-1" not in capsys.readouterr().out


def test_reclaim_after_remote_release(repos: tuple[Path, Path, Path], monkeypatch) -> None:
    _, a, b = repos
    monkeypatch.chdir(a)
    assert main(["claim", "T-1", "--agent", "agentA"]) == 0
    monkeypatch.chdir(b)
    assert main(["claim", "T-1", "--agent", "agentB"]) == 1  # fetches A's ref into b
    assert git(b, "show-ref", "refs/claims/T-1").returncode == 0
    monkeypatch.chdir(a)
    assert main(["release", "T-1"]) == 0
    monkeypatch.chdir(b)
    assert main(["claim", "T-1", "--agent", "agentB"]) == 0
    assert load_tasks(b / "tasks")[0].owner == "agentB"
