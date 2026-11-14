"""Pick the real glove or the scripted rescue once per rescue, at the clutch engage edge."""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Generator
from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    from teleop.atc.coordinator import OperatorInput
    from teleop.atc.stage import ScriptedGlove, StageController

SCRIPTED_NOTE = "gloves offline: playing scripted rescue"


class GloveOperatorLike(Protocol):
    def latest(self) -> OperatorInput | None: ...

    def running(self) -> contextlib.AbstractContextManager[object]: ...


class SelectingGlove:
    """Glove when healthy at engage, else scripted; the choice holds until release.

    ``rescue_input`` is ``None`` while no rescue is engaged.
    """

    def __init__(
        self,
        controller: StageController,
        glove: GloveOperatorLike,
        scripted: ScriptedGlove,
        glove_ok: Callable[[], bool],
    ) -> None:
        self.controller = controller
        self.glove = glove
        self.scripted = scripted
        self.glove_ok = glove_ok
        self.rescue_input: Literal["glove", "scripted"] | None = None

    def latest(self) -> OperatorInput | None:
        ctl = self.controller
        bay = ctl.rescue_bay()
        engaged = bay is not None and ctl.console.clutch and ctl.console.selected_bay == bay
        if not engaged:
            self.rescue_input = None
            self.scripted.latest()  # lets it reset its own engage state
            return None
        if self.rescue_input is None:
            if self.glove_ok():
                self.rescue_input = "glove"
            else:
                self.rescue_input = "scripted"
                ctl.note(SCRIPTED_NOTE)
        if self.rescue_input == "glove":
            return self.glove.latest()
        return self.scripted.latest()

    @contextlib.contextmanager
    def running(self) -> Generator[SelectingGlove]:
        with contextlib.ExitStack() as stack:
            stack.enter_context(self.glove.running())
            stack.enter_context(self.scripted.running())
            yield self
