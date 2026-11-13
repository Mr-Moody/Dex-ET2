"""Task checklist CLI: markdown task files plus race-safe claims via ``refs/claims/<ID>``.

A claim is a unique commit (parent HEAD, message carries agent, time and a nonce) stored
under ``refs/claims/<ID>`` and pushed to origin. Two claimers create different commits, so
the second push is a non-fast-forward and is rejected: exactly one claimer wins.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

STATUSES = ("todo", "claimed", "in_review", "done", "blocked")
_FM_RE = re.compile(r"\A---\r?\n(.*?)\r?\n---(?:\r?\n|\Z)(.*)\Z", re.DOTALL)
_CLAIM_RE = re.compile(r"^claim (\S+) by (.+) at (\S+)$")


@dataclass
class TaskFile:
    id: str
    title: str
    status: str = "todo"
    owner: str | None = None
    branch: str | None = None
    depends_on: list[str] = field(default_factory=lambda: [])
    plan: str | None = None
    claimed_at: str | None = None
    body: str = ""
    path: Path | None = None

    def is_ready(self, done_ids: set[str]) -> bool:
        return self.status == "todo" and all(d in done_ids for d in self.depends_on)

    def frontmatter(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "status": self.status,
            "owner": self.owner,
            "branch": self.branch,
            "depends_on": list(self.depends_on),
            "plan": self.plan,
            "claimed_at": self.claimed_at,
        }


def _opt_str(v: Any) -> str | None:
    return None if v is None else str(v)


def parse_task(text: str, path: Path | None = None) -> TaskFile:
    m = _FM_RE.match(text)
    if m is None:
        raise ValueError(f"{path}: missing YAML frontmatter")
    raw: Any = yaml.safe_load(m.group(1))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: frontmatter is not a mapping")
    fm: dict[str, Any] = raw  # pyright: ignore[reportUnknownVariableType]
    return TaskFile(
        id=str(fm["id"]),
        title=str(fm.get("title", "")),
        status=str(fm.get("status", "todo")),
        owner=_opt_str(fm.get("owner")),
        branch=_opt_str(fm.get("branch")),
        depends_on=[str(d) for d in list[Any](fm.get("depends_on") or [])],
        plan=_opt_str(fm.get("plan")),
        claimed_at=_opt_str(fm.get("claimed_at")),
        body=m.group(2),
        path=path,
    )


def load_tasks(dir: Path) -> list[TaskFile]:
    tasks: list[TaskFile] = []
    for p in sorted(dir.glob("*.md")):
        if p.name.lower() == "readme.md":
            continue
        tasks.append(parse_task(p.read_bytes().decode("utf-8"), p))
    return tasks


def save_task(t: TaskFile) -> None:
    if t.path is None:
        raise ValueError("TaskFile has no path")
    fm = yaml.safe_dump(
        t.frontmatter(), sort_keys=False, allow_unicode=True, default_flow_style=None, width=1000
    )
    t.path.write_bytes(f"---\n{fm}---\n{t.body}".encode())


def _git(
    *args: str, cwd: Path | None = None, check: bool = False
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=check, capture_output=True, text=True, encoding="utf-8"
    )


def _err(msg: str) -> None:
    print(msg, file=sys.stderr)


def repo_root() -> Path:
    r = _git("rev-parse", "--show-toplevel")
    if r.returncode != 0:
        raise SystemExit("not inside a git repository")
    return Path(r.stdout.strip())


def _find(tasks: list[TaskFile], tid: str) -> TaskFile | None:
    return next((t for t in tasks if t.id == tid), None)


def _claim_info(tid: str, cwd: Path) -> tuple[str, int] | None:
    """Return (owner, commit unix time) of the local claim ref, or None."""
    r = _git("log", "-1", "--format=%ct%x09%s", f"refs/claims/{tid}", cwd=cwd)
    if r.returncode != 0 or not r.stdout.strip():
        return None
    ts, _, subject = r.stdout.strip().partition("\t")
    m = _CLAIM_RE.match(subject)
    return (m.group(2) if m else "?", int(ts))


def _age(seconds: float) -> str:
    s = int(max(0, seconds))
    if s < 3600:
        return f"{s // 60}m"
    if s < 86400:
        return f"{s // 3600}h{s % 3600 // 60:02d}m"
    return f"{s // 86400}d"


def cmd_list(root: Path, status: str | None, ready: bool) -> int:
    tasks = load_tasks(root / "tasks")
    done = {t.id for t in tasks if t.status == "done"}
    now = datetime.now(UTC).timestamp()
    for t in tasks:
        if status and t.status != status:
            continue
        if ready and not t.is_ready(done):
            continue
        info = _claim_info(t.id, root)
        owner = f"{info[0]} ({_age(now - info[1])})" if info else (t.owner or "-")
        print(f"{t.id:<8} {t.status:<10} {owner:<24} {t.title}")
    return 0


def cmd_claim(root: Path, tid: str, agent: str, fetch: bool) -> int:
    tasks = load_tasks(root / "tasks")
    task = _find(tasks, tid)
    if task is None:
        _err(f"unknown task {tid}")
        return 1
    if fetch:
        r = _git("fetch", "--prune", "origin", "+refs/claims/*:refs/claims/*", cwd=root)
        if r.returncode != 0:
            _err(f"warning: could not fetch claims from origin (offline?): {r.stderr.strip()}")
    now = datetime.now(UTC).isoformat(timespec="seconds")
    # The nonce keeps the commit unique even for identical tree, parent, agent and second.
    msg = f"claim {tid} by {agent} at {now}\n\nnonce: {uuid.uuid4().hex}\n"
    r = _git("commit-tree", "HEAD^{tree}", "-p", "HEAD", "-m", msg, cwd=root)
    if r.returncode != 0:
        _err(f"git commit-tree failed: {r.stderr.strip()}")
        return 1
    sha = r.stdout.strip()
    ref = f"refs/claims/{tid}"
    r = _git("update-ref", ref, sha, "", cwd=root)
    if r.returncode != 0:
        info = _claim_info(tid, root)
        _err(f"{tid} already claimed by {info[0] if info else 'someone'}")
        return 1
    r = _git("push", "origin", ref, cwd=root)
    if r.returncode != 0:
        _git("update-ref", "-d", ref, cwd=root)
        _err(f"{tid}: lost race (claim rejected by origin): {r.stderr.strip()}")
        return 1
    br = _git("rev-parse", "--abbrev-ref", "HEAD", cwd=root).stdout.strip()
    task.status = "claimed"
    task.owner = agent
    task.branch = br
    task.claimed_at = now
    save_task(task)
    print(f"claimed {tid} for {agent} on branch {br}")
    return 0


def _drop_ref(root: Path, tid: str) -> None:
    ref = f"refs/claims/{tid}"
    _git("update-ref", "-d", ref, cwd=root)
    r = _git("push", "origin", f":{ref}", cwd=root)
    if r.returncode != 0:
        _err(f"warning: could not delete {ref} on origin: {r.stderr.strip()}")


def cmd_release(root: Path, tid: str) -> int:
    task = _find(load_tasks(root / "tasks"), tid)
    if task is None:
        _err(f"unknown task {tid}")
        return 1
    _drop_ref(root, tid)
    task.status = "todo"
    task.owner = None
    task.branch = None
    task.claimed_at = None
    save_task(task)
    print(f"released {tid}")
    return 0


def cmd_status(root: Path, tid: str, status: str) -> int:
    task = _find(load_tasks(root / "tasks"), tid)
    if task is None:
        _err(f"unknown task {tid}")
        return 1
    task.status = status
    save_task(task)
    print(f"{tid} -> {status}")
    return 0


def cmd_done(root: Path, tid: str) -> int:
    rc = cmd_release(root, tid)
    if rc != 0:
        return rc
    return cmd_status(root, tid, "done")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="task", description="Task checklist with atomic claims")
    sub = p.add_subparsers(dest="cmd", required=True)
    ls = sub.add_parser("list")
    ls.add_argument("--status", choices=STATUSES)
    ls.add_argument("--ready", action="store_true")
    cl = sub.add_parser("claim")
    cl.add_argument("id")
    cl.add_argument("--agent", required=True)
    cl.add_argument("--no-fetch", action="store_true", help=argparse.SUPPRESS)  # tests only
    sub.add_parser("release").add_argument("id")
    st = sub.add_parser("status")
    st.add_argument("id")
    st.add_argument("status", choices=STATUSES)
    sub.add_parser("done").add_argument("id")
    a = p.parse_args(argv)
    root = repo_root()
    match a.cmd:
        case "list":
            return cmd_list(root, a.status, a.ready)
        case "claim":
            return cmd_claim(root, a.id, a.agent, not a.no_fetch)
        case "release":
            return cmd_release(root, a.id)
        case "status":
            return cmd_status(root, a.id, a.status)
        case _:
            return cmd_done(root, a.id)


if __name__ == "__main__":
    raise SystemExit(main())
