"""Record ATC recoveries as LeRobot v3 episodes (``teleop atc --record-episodes DIR``).

``EpisodeSink`` is the coordinator's ``CoordinatorObserver``: one episode per rescue or
autonomous recovery attempt, from hand-over to bolt cleared or abort (spec section 3).

The 60 Hz loop only pays for a snapshot. On every second ``on_tick`` (30 fps) it stores the
29-value state, the 12-value latch-frame action and the bay's joint and mocap positions. When an
episode ends it goes on a queue to one writer thread, which poses a private copy of the model
from each snapshot, renders the ``wrist`` and ``top`` cameras, and writes the episode with
``EpisodeRecorder`` (sidecar line: ``EpisodeMeta``). The writer owns the GL context and the
dataset, so neither is touched from the control thread.

``close()`` drops episodes still open (an interrupted rescue), writes every finished one and
finalises the dataset, so a session cut short by Ctrl+C still loads.
"""

from __future__ import annotations

import copy
import importlib.util
import logging
import queue
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

import mujoco as _mujoco  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
from numpy.typing import NDArray

from teleop.atc.recovery import EpisodeStartSource
from teleop.atc.types import StallSignature
from teleop.policy.cameras import BayCameraRenderer
from teleop.policy.schema import (
    FPS,
    EpisodeMeta,
    EpisodeSource,
    action_from_command,
    lerobot_features,
    observe,
)
from teleop.types import Pose

if TYPE_CHECKING:
    from teleop.atc.stage import StageController
    from teleop.atc.validator import BayRig

mj: Any = _mujoco
logger = logging.getLogger(__name__)

ROBOT_TYPE = "panda_inspire"
TICKS_PER_FRAME = 2  # 60 Hz loop -> 30 fps dataset
assert 60 // TICKS_PER_FRAME == FPS

OperatorSource = str | Callable[[], str]


class RendererLike(Protocol):
    def render(self, data: Any, bay_id: int) -> Mapping[str, NDArray[np.uint8]]: ...

    def close(self) -> None: ...


class RecorderLike(Protocol):
    def start_episode(self, task: str) -> None: ...

    def add_frame(
        self, state: NDArray[Any], action: NDArray[Any], images: Mapping[str, NDArray[Any]]
    ) -> None: ...

    def end_episode(self, meta: Mapping[str, Any]) -> int: ...

    def discard_episode(self) -> None: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class _Frame:
    state: NDArray[np.float64]
    action: NDArray[np.float64]
    qpos: NDArray[np.float64]
    mocap_pos: NDArray[np.float64]
    mocap_quat: NDArray[np.float64]


@dataclass
class _Episode:
    bay: int
    source: EpisodeSource
    fault: str | None
    signature_id: str | None
    signature: tuple[float, float, float, float] | None
    frames: list[_Frame] = field(default_factory=list[_Frame])
    ticks: int = 0
    body_pos: NDArray[np.float64] | None = None  # model body poses (the latch can move)
    body_quat: NDArray[np.float64] | None = None
    latch_pose: dict[str, float] = field(default_factory=dict[str, float])
    bolt_x: float = 0.0
    max_tau_hand_inf: float = 0.0
    max_arm_ratio: float = 0.0


@dataclass(frozen=True)
class _Finished:
    episode: _Episode
    meta: EpisodeMeta

    @property
    def task(self) -> str:
        return f"recover latch: {self.meta.fault or 'unknown'}"


def session_root(root: Path) -> Path:
    """``root`` if it is new or empty, else a fresh ``root/<YYYYmmdd-HHMMSS>`` session folder."""
    root = Path(root)
    if not root.exists() or not any(root.iterdir()):
        return root
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate = root / stamp
    n = 1
    while candidate.exists():
        candidate = root / f"{stamp}-{n}"
        n += 1
    return candidate


def fault_lookup(controller: StageController) -> Callable[[int], str | None]:
    """``fault_of(bay)``: the kind of the fault active on that bay's current cycle, or None."""

    def fault_of(bay_id: int) -> str | None:
        for bay, kind, phase in controller.hud_view().faults:
            if bay == bay_id and phase == "active":
                return kind
        return None

    return fault_of


class EpisodeSink:
    """``CoordinatorObserver`` that saves every recovery attempt as a LeRobot episode."""

    def __init__(
        self,
        root: Path,
        model: Any,
        data: Any,
        fault_of: Callable[[int], str | None],
        operator_source: OperatorSource = "glove",
        renderer: RendererLike | None = None,
        recorder: RecorderLike | None = None,
        repo_id: str | None = None,
    ) -> None:
        self.root = Path(root)
        self.repo_id = repo_id or f"local/{self.root.name}"
        self.data = data
        self.fault_of = fault_of
        self.operator_source = operator_source
        # The writer poses its own copy: the physics model and data are never touched off-loop.
        self._model: Any = copy.deepcopy(model)
        self._render_data: Any = mj.MjData(self._model)
        self._renderer: RendererLike | None = renderer
        if recorder is None:
            from teleop.record.recorder import EpisodeRecorder

            recorder = EpisodeRecorder(self.repo_id, self.root, FPS, lerobot_features(), ROBOT_TYPE)
        self._recorder: RecorderLike = recorder
        self._open: dict[int, _Episode] = {}
        self._queue: queue.Queue[_Finished | None] = queue.Queue()
        self._lock = threading.Lock()
        self._written = 0
        self._closed = False
        self._thread = threading.Thread(target=self._write_loop, name="episode-writer", daemon=True)
        self._thread.start()

    @property
    def written(self) -> int:
        """Episodes saved to the dataset so far."""
        with self._lock:
            return self._written

    # -- CoordinatorObserver (control thread)

    def on_episode_start(
        self,
        bay_id: int,
        source: EpisodeStartSource,
        signature_id: str | None,
        signature: StallSignature | None,
    ) -> None:
        if self._closed:
            return
        if bay_id in self._open:
            logger.warning("bay %d: episode restarted before it ended; dropping it", bay_id)
        self._open[bay_id] = _Episode(
            bay=bay_id,
            source=self._label(source),
            fault=self.fault_of(bay_id),
            signature_id=signature_id,
            signature=None if signature is None else signature.as_vector(),
        )

    def on_tick(
        self, bay_id: int, rig: BayRig, wrist_world: Pose, fingers: Mapping[str, float]
    ) -> None:
        ep = self._open.get(bay_id)
        if ep is None:
            return
        ep.ticks += 1
        if (ep.ticks - 1) % TICKS_PER_FRAME:
            return
        data = rig.data
        if ep.body_pos is None:
            ep.body_pos = np.array(rig.model.body_pos, dtype=np.float64)
            ep.body_quat = np.array(rig.model.body_quat, dtype=np.float64)
            latch = rig.true_world_latch()  # label only
            x, y, z = latch.position
            ep.latch_pose = {"x": x, "y": y, "z": z, "yaw": latch.yaw_rad}
        ep.frames.append(
            _Frame(
                state=observe(rig),
                action=action_from_command(rig.world_from_latch(), wrist_world, fingers),
                qpos=np.array(data.qpos, dtype=np.float64),
                mocap_pos=np.array(data.mocap_pos, dtype=np.float64),
                mocap_quat=np.array(data.mocap_quat, dtype=np.float64),
            )
        )
        tau = rig.tau_ext()
        ep.max_tau_hand_inf = max(ep.max_tau_hand_inf, rig.tau_hand_inf(tau))
        ep.max_arm_ratio = max(ep.max_arm_ratio, rig.arm_torque_ratio(tau))
        ep.bolt_x = rig.bolt_x()

    def on_episode_end(self, bay_id: int, success: bool, aborted_reason: str | None) -> None:
        ep = self._open.pop(bay_id, None)
        if ep is None or self._closed:
            return
        if not ep.frames:
            logger.info("bay %d: episode ended with no frames; not saved", bay_id)
            return
        meta = EpisodeMeta(
            source=ep.source,
            fault=ep.fault,
            bay=ep.bay,
            signature_id=ep.signature_id,
            signature=ep.signature,
            latch_pose=ep.latch_pose,
            success=success,
            bolt_x_final=ep.bolt_x,
            max_tau_hand_inf=ep.max_tau_hand_inf,
            max_arm_ratio=ep.max_arm_ratio,
            aborted_reason=aborted_reason,
        )
        self._queue.put(_Finished(ep, meta))

    def close(self) -> None:
        """Drop open episodes, write the finished ones and finalise the dataset."""
        if self._closed:
            return
        self._closed = True
        if self._open:
            logger.info("episode sink closing: dropping %d unfinished episode(s)", len(self._open))
        self._open.clear()
        self._queue.put(None)
        self._thread.join()

    # -- writer thread

    def _label(self, source: EpisodeStartSource) -> EpisodeSource:
        if source != "operator":
            return source
        label = self.operator_source() if callable(self.operator_source) else self.operator_source
        return cast("EpisodeSource", label)

    def _write_loop(self) -> None:
        try:
            while (item := self._queue.get()) is not None:
                try:
                    self._write(item)
                except Exception:
                    logger.exception("bay %d: failed to write episode", item.episode.bay)
                    self._recorder.discard_episode()
        finally:
            try:
                self._recorder.close()
            finally:
                if self._renderer is not None:
                    self._renderer.close()

    def _write(self, item: _Finished) -> None:
        ep = item.episode
        if self._renderer is None:
            self._renderer = BayCameraRenderer(self._model)
        m, d = self._model, self._render_data
        if ep.body_pos is not None and ep.body_quat is not None:
            m.body_pos[:] = ep.body_pos
            m.body_quat[:] = ep.body_quat
        self._recorder.start_episode(item.task)
        for f in ep.frames:
            d.qpos[:] = f.qpos
            if f.mocap_pos.size:
                d.mocap_pos[:] = f.mocap_pos
                d.mocap_quat[:] = f.mocap_quat
            mj.mj_kinematics(m, d)
            mj.mj_camlight(m, d)
            images = self._renderer.render(d, ep.bay)
            self._recorder.add_frame(f.state, f.action, images)
        index = self._recorder.end_episode(item.meta.to_json())
        with self._lock:
            self._written += 1
        logger.info(
            "bay %d: saved episode %d (%s, %d frames, success=%s)",
            ep.bay,
            index,
            ep.source,
            len(ep.frames),
            item.meta.success,
        )


def open_episode_sink(
    root: Path, controller: StageController, operator_source: OperatorSource
) -> EpisodeSink:
    """The ``teleop atc --record-episodes`` hook: a sink on a fresh session folder under ``root``.

    ``operator_source`` labels human rescues: ``"glove"``/``"scripted"``, or a callable read at
    each rescue start (the booth's ``SelectingGlove`` picks gloves or the scripted rescue then).
    """
    if importlib.util.find_spec("lerobot") is None:
        raise RuntimeError("--record-episodes needs lerobot: run `uv sync --group train`")
    co = controller.coordinator
    session = session_root(root)
    sink = EpisodeSink(session, co.model, co.data, fault_lookup(controller), operator_source)
    logger.info("recording recovery episodes to %s", session)
    print(f"teleop atc: recording recovery episodes to {session}")
    return sink
