"""``teleop record --task "pick up the cube" --episodes N [--max-seconds 60] [SimConfig flags]``.

Each episode resets the sim task and runs until task success or ``--max-seconds``. Every
``control_hz // record_fps`` ticks it renders each camera offscreen and adds a LeRobot frame:
state = hand joint qpos (``state_names`` order) + measured wrist pose, action = actuator ctrl
(``action_names`` order) + commanded wrist pose. ``--task`` is the language instruction stored
in the dataset; the sim task is chosen with ``--sim-task`` (registry name, default from config).
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import time
from pathlib import Path
from typing import Any

import mujoco as _mujoco  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
from numpy.typing import NDArray

from teleop.app import TeleopApp, parse_sim_config
from teleop.commands.sim import add_hotspot_args, maybe_hotspot, setup_logging
from teleop.latency import LatencyStats
from teleop.net.hotspot import HotspotConfigError
from teleop.record.recorder import EpisodeRecorder
from teleop.record.schema import action_names, build_features, state_names
from teleop.sim.hands import joint_names
from teleop.sim.loop import FixedRateLoop
from teleop.types import SIDES, Side

mj: Any = _mujoco
logger = logging.getLogger("teleop.record")
CUBE_BODY = "cube"


class EpisodeIO:
    """Builds state/action vectors and camera images from the running app."""

    def __init__(self, app: TeleopApp) -> None:
        self.app = app
        model = app.model
        qpos = app.sink.state().qpos
        self.joints: dict[Side, list[str]] = {
            s: [n for n in joint_names(app.hand, s) if f"{s}/{n}" in qpos] for s in SIDES
        }
        self.state_names = state_names(self.joints)
        self.action_names = action_names(app.actuators)
        self._act_ids = [
            int(mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, f"{s}/{a}"))
            for s in SIDES
            for a in app.actuators[s]
        ]
        self._base_ids = {
            s: int(mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, f"{s}/{app.hand.base_body}"))
            for s in SIDES
        }
        self._cube_id = int(mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, CUBE_BODY))
        self.renderers: dict[str, Any] = {
            c.name: mj.Renderer(model, c.height, c.width) for c in app.cfg.cameras
        }

    def cube_z(self) -> float | None:
        """Height (m) of the task's ``cube`` body, or None if the task has none."""
        if self._cube_id < 0:
            return None
        return float(self.app.data.xpos[self._cube_id][2])

    def features(self) -> dict[str, dict[str, Any]]:
        return build_features(self.state_names, self.action_names, self.app.cfg.cameras)

    def state(self) -> NDArray[np.float32]:
        data = self.app.data
        st = self.app.sink.state()
        vals = [st.qpos[f"{s}/{n}"] for s in SIDES for n in self.joints[s]]
        for s in SIDES:
            bid = self._base_ids[s]
            w, x, y, z = (float(v) for v in data.xquat[bid])
            vals += [float(v) for v in data.xpos[bid]] + [x, y, z, w]
        return np.asarray(vals, dtype=np.float32)

    def action(self) -> NDArray[np.float32]:
        data = self.app.data
        vals = [float(data.ctrl[i]) for i in self._act_ids]
        wrist = self.app.sink.state().wrist
        for s in SIDES:
            pose = wrist[s]
            pos = pose.position if pose.position is not None else (0.0, 0.0, 0.0)
            vals += [*pos, *pose.orientation]
        return np.asarray(vals, dtype=np.float32)

    def images(self) -> dict[str, NDArray[np.uint8]]:
        out: dict[str, NDArray[np.uint8]] = {}
        for name, r in self.renderers.items():
            r.update_scene(self.app.data, camera=name)
            out[name] = np.asarray(r.render(), dtype=np.uint8)
        return out

    def close(self) -> None:
        for r in self.renderers.values():
            r.close()


def _episode_meta(
    app: TeleopApp,
    start_ns: int,
    start_modes: dict[Side, str],
    score: Any,
    max_cube_z: float | None,
    score_curve: list[float],
) -> dict[str, Any]:
    events: list[tuple[int, str]] = app.hotspot.events if app.hotspot is not None else []
    return {
        "score": float(score.score),
        "success": bool(score.success),
        "elapsed_s": float(score.elapsed_s),
        "calibration": {
            "path": str(app.calibration_path) if app.calibration_path else None,
            "ranges": app.calibration.ranges,
        },
        "wrist_mode": {
            "start": start_modes,
            "changes": [
                {**m, "t_s": (m["t_ns"] - start_ns) / 1e9}
                for m in app.mode_history
                if m["t_ns"] >= start_ns
            ],
        },
        "source_status": {**dataclasses.asdict(app.source.status()), "max_gap_s": app.max_gap_s()},
        "max_cube_z": max_cube_z,
        "score_curve": score_curve,
        "hotspot_events": [
            {"t_s": (t - start_ns) / 1e9, "event": e} for t, e in events if t >= start_ns
        ],
        "latency": app.latency.summary(),
    }


def _run_episode(app: TeleopApp, rec: EpisodeRecorder, io: EpisodeIO, max_ticks: int) -> Any:
    """Tick until task success or ``max_ticks``; returns (last score, max cube z, curve)."""
    every = app.cfg.control_hz // app.cfg.record_fps
    score = app.task.score(app.model, app.data)
    max_cube_z = io.cube_z()
    score_curve: list[float] = []

    def tick(i: int) -> bool:
        nonlocal score, max_cube_z
        if not app.step(i):
            return False
        score = app.task.score(app.model, app.data)
        z = io.cube_z()
        if z is not None:
            max_cube_z = z if max_cube_z is None else max(max_cube_z, z)
        if i % every == 0:
            rec.add_frame(io.state(), io.action(), io.images())
            score_curve.append(float(score.score))
        return not score.success

    FixedRateLoop(app.cfg.control_hz).run(tick, max_ticks)
    return score, max_cube_z, score_curve


def record_episodes(
    app: TeleopApp,
    rec: EpisodeRecorder,
    io: EpisodeIO,
    instruction: str,
    episodes: int,
    max_seconds: float,
    saved: list[int],
) -> None:
    """Record up to ``episodes`` episodes, appending each saved episode index to ``saved``."""
    max_ticks = max(1, round(max_seconds * app.cfg.control_hz))
    for ep in range(episodes):
        app.reset_task()
        app.latency = LatencyStats()
        rec.start_episode(instruction)  # creates the dataset on first use: slow, keep it out
        app.reset_gaps()
        start_ns = time.monotonic_ns()
        start_modes: dict[Side, str] = {s: app.wrist_mode(s) for s in SIDES}
        logger.info("episode %d/%d: %s (cap %.0f s)", ep + 1, episodes, instruction, max_seconds)
        score, max_cube_z, curve = _run_episode(app, rec, io, max_ticks)
        if app.viewer_closed():
            rec.discard_episode()
            logger.info("viewer closed: episode %d discarded", ep + 1)
            return
        idx = rec.end_episode(_episode_meta(app, start_ns, start_modes, score, max_cube_z, curve))
        saved.append(idx)
        logger.info(
            "episode %d saved: success=%s score=%.1f", idx, score.success, float(score.score)
        )


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="teleop record", allow_abbrev=False)
    ap.add_argument("--task", default=None, help="language instruction (default: the task's)")
    ap.add_argument("--sim-task", default=None, help="sim task registry name")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--max-seconds", type=float, default=60.0)
    ap.add_argument("--root", type=Path, default=None, help="default data/datasets/<timestamp>")
    ap.add_argument("--repo-id", default="local/teleop")
    add_hotspot_args(ap)
    ns, rest = ap.parse_known_args(argv)
    setup_logging()
    cfg = parse_sim_config(rest)
    if ns.sim_task is not None:
        cfg.task = ns.sim_task
    if cfg.record_fps <= 0 or cfg.control_hz % cfg.record_fps != 0:
        print(f"teleop record: control_hz {cfg.control_hz} must be a multiple of record_fps")
        return 2
    root: Path = ns.root or Path("data/datasets") / time.strftime("%Y%m%d-%H%M%S")
    try:
        watchdog = maybe_hotspot(ns)
    except (HotspotConfigError, RuntimeError) as exc:
        print(f"teleop record: {exc}")
        return 1
    app = TeleopApp(cfg, hotspot=watchdog)
    instruction: str = ns.task or app.task.instruction
    io = EpisodeIO(app)
    rec = EpisodeRecorder(ns.repo_id, root, cfg.record_fps, io.features(), f"bimanual_{cfg.hand}")
    saved: list[int] = []
    try:
        with app.session(), rec:  # on Ctrl-C/exception: partial episode discarded, finalized
            record_episodes(app, rec, io, instruction, ns.episodes, ns.max_seconds, saved)
    except KeyboardInterrupt:
        logger.info("interrupted; partial episode discarded")
    finally:
        io.close()
        if watchdog is not None:
            watchdog.stop()
    print(f"saved {len(saved)} episode(s) to {root}")
    return 0
