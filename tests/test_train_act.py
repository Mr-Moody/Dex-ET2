"""Tests for ``scripts/train_act.py`` (T-194). They never import torch or train a model."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from teleop.policy.schema import SCHEMA_VERSION

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_act.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("train_act", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["train_act"] = mod  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def ta() -> ModuleType:
    return _load()


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    root = tmp_path / "gen-smoke-clean"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps({"fps": 30}), encoding="utf-8")
    (root / "meta" / "teleop_episodes.jsonl").write_text('{"episode_index": 0}\n', encoding="utf-8")
    return root


class _Recorder:
    def __init__(self, code: int = 0) -> None:
        self.calls: list[list[str]] = []
        self.code = code

    def __call__(self, argv: list[str]) -> int:
        self.calls.append(argv)
        return self.code


def test_build_train_args(ta: ModuleType, dataset: Path, tmp_path: Path) -> None:
    out = tmp_path / "models" / "act" / "run1"
    args = ta.build_train_args(dataset, out, ta.ActPreset())
    assert "--policy.type=act" in args
    assert f"--dataset.repo_id=local/{dataset.name}" in args
    assert f"--dataset.root={dataset}" in args
    assert f"--output_dir={out}" in args
    assert "--steps=50000" in args
    assert "--batch_size=32" in args
    assert "--policy.chunk_size=30" in args
    assert "--policy.device=cuda" in args


def test_cli_flags_override_preset(ta: ModuleType, dataset: Path, tmp_path: Path) -> None:
    run = _Recorder()
    out = tmp_path / "out"
    argv = ["--dataset", str(dataset), "--out", str(out), "--steps", "200", "--batch-size", "4"]
    assert ta.main([*argv, "--device", "cpu"], run=run) == 0
    (args,) = run.calls
    assert "--steps=200" in args
    assert "--batch_size=4" in args
    assert "--policy.device=cpu" in args


def test_dry_run_does_not_call_run(
    ta: ModuleType, dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = _Recorder()
    out = tmp_path / "out"
    assert ta.main(["--dataset", str(dataset), "--out", str(out), "--dry-run"], run=run) == 0
    assert run.calls == []
    assert "--policy.type=act" in capsys.readouterr().out
    assert not out.exists()


def test_run_json_contents(ta: ModuleType, dataset: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    path = ta.write_run_json(dataset, out, ta.ActPreset(), "abc123")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert path == out / "run.json"
    assert set(data) >= {"schema_version", "dataset_sha256", "git_commit", "preset"}
    assert data["schema_version"] == SCHEMA_VERSION
    assert data["git_commit"] == "abc123"
    assert data["preset"]["steps"] == 50_000

    (dataset / "meta" / "teleop_episodes.jsonl").write_text(
        '{"episode_index": 0}\n{"episode_index": 1}\n', encoding="utf-8"
    )
    again = json.loads(ta.write_run_json(dataset, out, ta.ActPreset(), "abc123").read_text())
    assert again["dataset_sha256"] != data["dataset_sha256"]


def test_main_returns_run_exit_code(ta: ModuleType, dataset: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    assert ta.main(["--dataset", str(dataset), "--out", str(out)], run=_Recorder(3)) == 3
    assert not (out / "run.json").exists()


def test_successful_run_writes_run_json(ta: ModuleType, dataset: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    assert ta.main(["--dataset", str(dataset), "--out", str(out)], run=_Recorder(0)) == 0
    assert json.loads((out / "run.json").read_text())["schema_version"] == SCHEMA_VERSION


def test_existing_out_is_refused(ta: ModuleType, dataset: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    run = _Recorder()
    assert ta.main(["--dataset", str(dataset), "--out", str(out)], run=run) == 2
    assert run.calls == []


def test_missing_dataset_is_an_error(ta: ModuleType, tmp_path: Path) -> None:
    run = _Recorder()
    code = ta.main(["--dataset", str(tmp_path / "nope"), "--out", str(tmp_path / "o")], run=run)
    assert code == 2
    assert run.calls == []
