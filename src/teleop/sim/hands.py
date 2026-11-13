"""Hand model descriptions: MJCF paths and glove-flexion to actuator maps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import mujoco  # pyright: ignore[reportMissingTypeStubs]
import yaml

from teleop.types import Side

REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class FlexMap:
    actuator: str
    joints: list[str]  # canonical OpenXR joint names
    axis: Literal["x", "y", "z"]
    invert: bool = False
    weights: list[float] | None = None


@dataclass
class HandSpec:
    name: str
    mjcf: dict[Side, Path]
    mirror_left: bool
    base_body: str
    flex: list[FlexMap]


def load_hand(name: str, configs_dir: Path = REPO_ROOT / "configs" / "hands") -> HandSpec:
    """Load ``<configs_dir>/<name>.yaml``. Relative MJCF paths resolve from the repo root."""
    raw = yaml.safe_load((configs_dir / f"{name}.yaml").read_text(encoding="utf-8"))

    def resolve(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else REPO_ROOT / path

    return HandSpec(
        name=raw["name"],
        mjcf={"left": resolve(raw["mjcf"]["left"]), "right": resolve(raw["mjcf"]["right"])},
        mirror_left=bool(raw.get("mirror_left", False)),
        base_body=raw["base_body"],
        flex=[FlexMap(**entry) for entry in raw["flex"]],
    )


def _model(spec: HandSpec, side: Side) -> Any:  # mujoco.MjModel (untyped bindings)
    source: Side = "right" if side == "left" and spec.mirror_left else side
    return mujoco.MjModel.from_xml_path(str(spec.mjcf[source]))  # pyright: ignore


def actuator_names(spec: HandSpec, side: Side) -> list[str]:
    """Unprefixed actuator names, in model order."""
    m = _model(spec, side)
    return [str(m.actuator(i).name) for i in range(int(m.nu))]


def joint_names(spec: HandSpec, side: Side) -> list[str]:
    """Unprefixed joint names, in model order."""
    m = _model(spec, side)
    return [str(m.joint(i).name) for i in range(int(m.njnt))]
