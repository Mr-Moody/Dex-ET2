"""Policy cameras: the ``wrist`` and ``top`` images of one bay at the dataset resolution.

``BayCameraRenderer`` reuses one ``mujoco.Renderer`` for every camera. It owns a GL context, so
construct it and call it from one thread only (the episode writer thread, never the 60 Hz loop).
"""

from __future__ import annotations

import contextlib
from collections.abc import Generator
from typing import Any

import mujoco as _mujoco  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
from numpy.typing import NDArray

from teleop.policy.schema import CAMERAS, IMAGE_HW, camera_name

mj: Any = _mujoco

WRIST_ZNEAR_M = 0.005  # the fingertips are ~5-17 cm from the wrist camera
WRIST_ZFAR_M = 10.0


@contextlib.contextmanager
def policy_clip(model: Any, camera: str) -> Generator[None]:
    """Metric clip planes for a body-mounted (wrist) camera while rendering it.

    MuJoCo's ``vis.map.znear``/``zfar`` are fractions of the scene extent. The warehouse is
    ~37 m across, so the default near plane sits ~0.37 m out and clips the whole hand and latch
    from the wrist view. World-fixed cameras keep the model's planes. Restored on exit, so the
    model is unchanged afterwards.
    """
    cam = model.camera(camera)
    if int(model.cam_bodyid[int(cam.id)]) == 0:
        yield
        return
    vis = model.vis.map
    extent = max(float(model.stat.extent), 1e-6)
    znear, zfar = float(vis.znear), float(vis.zfar)
    vis.znear, vis.zfar = WRIST_ZNEAR_M / extent, WRIST_ZFAR_M / extent
    try:
        yield
    finally:
        vis.znear, vis.zfar = znear, zfar


class BayCameraRenderer:
    def __init__(self, model: Any, image_hw: tuple[int, int] = IMAGE_HW) -> None:
        self.model = model
        self.image_hw = image_hw
        self._renderer: Any = None

    def render(self, data: Any, bay_id: int) -> dict[str, NDArray[np.uint8]]:
        """HWC uint8 images keyed by policy camera name (``CAMERAS``) for one bay."""
        if self._renderer is None:
            h, w = self.image_hw
            self._renderer = mj.Renderer(self.model, h, w)
        r = self._renderer
        out: dict[str, NDArray[np.uint8]] = {}
        for cam in CAMERAS:
            name = camera_name(bay_id, cam)
            with policy_clip(self.model, name):
                r.update_scene(data, name)
                out[cam] = np.asarray(r.render(), dtype=np.uint8).copy()
        return out

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
