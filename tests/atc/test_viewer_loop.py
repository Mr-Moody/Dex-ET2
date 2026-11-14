"""``teleop atc`` viewer loop: HUD overlay must never be pushed while holding ``viewer.lock()``.

MuJoCo's ``Handle.set_texts`` spins until the render thread consumes the previous request, and
the render thread needs the viewer lock, so a call under the lock froze the app at 100% CPU.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Generator
from types import SimpleNamespace
from typing import Any, cast

import pytest

from teleop.commands import atc


class FakeViewer:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.texts: list[Any] = []
        self.syncs = 0

    def __enter__(self) -> FakeViewer:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def is_running(self) -> bool:
        return True

    @contextlib.contextmanager
    def lock(self) -> Generator[None]:
        with self._lock:
            yield

    def set_texts(self, texts: Any) -> None:
        assert not self._lock.locked(), "set_texts called while holding viewer.lock()"
        self.texts.append(texts)

    def sync(self) -> None:
        self.syncs += 1


class FakeController:
    def __init__(self) -> None:
        self.coordinator = SimpleNamespace(model=None, data=None, time_s=0.0)
        self.ticks = 0

    def on_key(self, _key: int) -> None:
        return None

    def tick(self, _sample: object) -> None:
        self.ticks += 1
        self.coordinator.time_s = self.ticks / atc.CONTROL_HZ

    def hud_view(self) -> object:
        return None


def test_viewer_loop_sets_texts_outside_lock_and_throttles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Overlay text changes every tick, so throttling (not dedup) bounds the set_texts rate.
    controller = FakeController()
    monkeypatch.setattr(atc, "hud_overlays", lambda _v: [(0, 0, "t", str(controller.ticks))])
    viewer = FakeViewer()
    steps = 60  # 1 s of sim time at CONTROL_HZ
    cfg: Any = SimpleNamespace(steps=steps)

    atc._run_viewer(  # pyright: ignore[reportPrivateUsage]
        cfg, cast(Any, controller), lambda: None, launch=lambda *_a, **_k: viewer
    )

    assert controller.ticks == steps and viewer.syncs == steps
    assert 1 <= len(viewer.texts) <= atc.HUD_OVERLAY_HZ + 1


def test_viewer_loop_skips_unchanged_overlay(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(atc, "hud_overlays", lambda _v: [(0, 0, "static", "")])
    viewer = FakeViewer()
    controller = FakeController()
    cfg: Any = SimpleNamespace(steps=30)

    atc._run_viewer(  # pyright: ignore[reportPrivateUsage]
        cfg, cast(Any, controller), lambda: None, launch=lambda *_a, **_k: viewer
    )

    assert len(viewer.texts) == 1
