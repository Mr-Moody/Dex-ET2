"""ACT release gate manifest (``<checkpoint>/gate.json``), written by ``teleop eval``.

The live runtime only runs ACT on a stall whose fingerprint matches a prototype of a fault type
that passed the gate. Prototypes are stored as fingerprint samples, not as UNO Q signature IDs:
the UNO Q numbers signatures per session (``Jam_Sig_1``, ...), so an ID can mean a different
fault in another session.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from teleop.atc.types import StallSignature
from teleop.policy.schema import SCHEMA_VERSION
from teleop.unoq.classifier import MahalanobisStallClassifier

RELEASE_SKU = "release"
DEFAULT_OOD_THRESHOLD = 3.0

Proto = tuple[float, float, float, float]


class ManifestError(ValueError):
    """``gate.json`` is missing, unreadable or was written for another schema."""


@dataclass(frozen=True)
class FaultResult:
    fault: str
    rollouts: int
    passed_rollouts: int
    mean_time_to_clear_s: float | None
    max_tau_hand_inf: float
    dmp_passed_rollouts: int | None
    note: str = ""


@dataclass(frozen=True)
class GateManifest:
    schema_version: int
    checkpoint_sha256: str
    bar: tuple[int, int]  # (passed rollouts needed, rollouts)
    results: dict[str, FaultResult]
    prototypes: dict[str, list[Proto]]
    inference_p95_ms: float | None

    def passed_faults(self) -> tuple[str, ...]:
        need = self.bar[0]
        return tuple(f for f, r in self.results.items() if r.passed_rollouts >= need and not r.note)

    def save(self, path: Path) -> None:
        d: dict[str, Any] = asdict(self)
        d["bar"] = list(self.bar)
        d["prototypes"] = {f: [list(p) for p in ps] for f, ps in self.prototypes.items()}
        Path(path).write_text(json.dumps(d, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> GateManifest:
        path = Path(path)
        try:
            raw: Any = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ManifestError(f"{path} not found: run `teleop eval` on the checkpoint") from exc
        except (OSError, ValueError) as exc:
            raise ManifestError(f"{path} unreadable: {exc}") from exc
        if not isinstance(raw, dict):
            raise ManifestError(f"{path}: expected a JSON object")
        d = cast("dict[str, Any]", raw)
        version: Any = d.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ManifestError(
                f"{path}: schema_version {version!r}, this build expects {SCHEMA_VERSION}"
            )
        try:
            bar: list[Any] = list(d["bar"])
            results: dict[str, Any] = dict(d["results"])
            protos: dict[str, Any] = dict(d["prototypes"])
            p95: Any = d.get("inference_p95_ms")
            return cls(
                schema_version=SCHEMA_VERSION,
                checkpoint_sha256=str(d["checkpoint_sha256"]),
                bar=(int(bar[0]), int(bar[1])),
                results={str(f): FaultResult(**dict(r)) for f, r in results.items()},
                prototypes={str(f): [_proto(p) for p in list(ps)] for f, ps in protos.items()},
                inference_p95_ms=None if p95 is None else float(p95),
            )
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise ManifestError(f"{path}: malformed manifest ({exc})") from exc


def _proto(p: Any) -> Proto:
    a, b, c, d = (float(v) for v in p)
    return (a, b, c, d)


def covers(
    manifest: GateManifest, z: StallSignature, ood_threshold: float = DEFAULT_OOD_THRESHOLD
) -> str | None:
    """Fault label of the nearest passed-fault prototype within ``ood_threshold``, else None."""
    clf = MahalanobisStallClassifier(ood_threshold=ood_threshold)
    for fault in manifest.passed_faults():
        samples = manifest.prototypes.get(fault)
        if samples:
            clf.register_prototype(RELEASE_SKU, fault, samples)
    match = clf.classify(RELEASE_SKU, z.as_vector())
    return None if match.is_ood else match.signature_id
