"""LeRobot v3 episode recorder.

Verified against installed lerobot 0.7.0 (``lerobot.datasets.lerobot_dataset``):
  LeRobotDataset.create(repo_id, fps, features, root=None, robot_type=None,
      use_videos=True, ...) -> LeRobotDataset   (write mode)
  ds.add_frame(frame: dict)   # frame must include a "task" key
  ds.save_episode(episode_data=None, parallel_encoding=True)
  ds.clear_episode_buffer(delete_images=True)
  ds.finalize()               # idempotent, mandatory before loading
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Mapping
from pathlib import Path
from types import TracebackType
from typing import Any

import numpy as np
from numpy.typing import NDArray


class RecorderRootExistsError(FileExistsError):
    """The dataset root already exists and is not empty."""


class EpisodeRecorder:
    def __init__(
        self, repo_id: str, root: Path, fps: int, features: dict[str, Any], robot_type: str
    ) -> None:
        self.repo_id = repo_id
        self.root = Path(root)
        if self.root.exists() and any(self.root.iterdir()):
            raise RecorderRootExistsError(
                f"dataset root {self.root} already exists and is not empty; "
                "use a fresh session directory"
            )
        self.fps = fps
        self.features: dict[str, Any] = features
        self.robot_type = robot_type
        self._ds: Any = None
        self._task: str | None = None
        self._open = False
        self._closed = False

    def _dataset(self) -> Any:
        if self._ds is None:
            mod: Any = importlib.import_module("lerobot.datasets.lerobot_dataset")
            self._ds = mod.LeRobotDataset.create(
                self.repo_id,
                self.fps,
                self.features,
                root=self.root,
                robot_type=self.robot_type,
                use_videos=True,
            )
        return self._ds

    def start_episode(self, task: str) -> None:
        if self._closed:
            raise RuntimeError("recorder is closed")
        if self._open:
            raise RuntimeError("an episode is already open")
        self._dataset()
        self._task = task
        self._open = True

    def add_frame(
        self, state: NDArray[Any], action: NDArray[Any], images: Mapping[str, NDArray[Any]]
    ) -> None:
        if not self._open:
            raise RuntimeError("no open episode")
        frame: dict[str, Any] = {
            "observation.state": np.asarray(state, dtype=np.float32),
            "action": np.asarray(action, dtype=np.float32),
            "task": self._task,
        }
        for name, img in images.items():
            frame[f"observation.images.{name}"] = img
        self._dataset().add_frame(frame)

    def end_episode(self, meta: Mapping[str, Any]) -> int:
        if not self._open:
            raise RuntimeError("no open episode")
        ds = self._dataset()
        ds.save_episode()
        self._open = False
        index = int(ds.meta.total_episodes) - 1
        sidecar = self.root / "meta" / "teleop_episodes.jsonl"
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        with sidecar.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"episode_index": index, **meta}, default=str) + "\n")
        return index

    def discard_episode(self) -> None:
        if self._open and self._ds is not None:
            self._ds.clear_episode_buffer()
        self._open = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._ds is not None:
            self._ds.finalize()

    def __enter__(self) -> EpisodeRecorder:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        try:
            self.discard_episode()
        finally:
            self.close()
