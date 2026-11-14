"""T-190: ACT release gate manifest (load/save, schema check, fingerprint coverage)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from teleop.atc.types import StallSignature
from teleop.policy.manifest import FaultResult, GateManifest, ManifestError, covers
from teleop.policy.schema import SCHEMA_VERSION

FLAP = [(0.035, 1.2, 2.4, 0.3), (0.036, 1.3, 2.5, 0.31), (0.034, 1.1, 2.3, 0.29)]
PIN = [(0.012, 4.0, 5.5, 1.1), (0.013, 4.1, 5.4, 1.0)]


def _manifest() -> GateManifest:
    return GateManifest(
        schema_version=SCHEMA_VERSION,
        checkpoint_sha256="ab" * 32,
        bar=(9, 10),
        results={
            "flap": FaultResult("flap", 10, 10, 1.8, 2.2, 3),
            "pin": FaultResult("pin", 10, 4, None, 6.1, 0),
        },
        prototypes={"flap": FLAP, "pin": PIN},
        inference_p95_ms=12.5,
    )


def _sig(v: tuple[float, float, float, float]) -> StallSignature:
    return StallSignature(*v)


def test_manifest_roundtrip(tmp_path: Path) -> None:
    m = _manifest()
    path = tmp_path / "gate.json"
    m.save(path)
    assert GateManifest.load(path) == m
    assert m.passed_faults() == ("flap",)


def test_manifest_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="gate.json"):
        GateManifest.load(tmp_path / "gate.json")


def test_manifest_unreadable_raises(tmp_path: Path) -> None:
    path = tmp_path / "gate.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ManifestError):
        GateManifest.load(path)


def test_manifest_wrong_schema_raises(tmp_path: Path) -> None:
    path = tmp_path / "gate.json"
    _manifest().save(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["schema_version"] = SCHEMA_VERSION + 1
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ManifestError, match="schema"):
        GateManifest.load(path)


def test_covers_near_and_far() -> None:
    m = _manifest()
    assert covers(m, _sig(FLAP[0])) == "flap"
    assert covers(m, _sig((0.035, 11.2, 12.4, 5.3))) is None


def test_covers_ignores_failed_fault() -> None:
    assert covers(_manifest(), _sig(PIN[0])) is None
