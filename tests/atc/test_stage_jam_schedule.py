"""Auto-jam schedule and the armed-jam belt preview on the stage controller."""

from __future__ import annotations

from teleop.atc.hud import ConveyorStatus
from teleop.atc.stage import StageController


def test_armed_jam_shows_on_the_incoming_pack_unless_feeding() -> None:
    class _Co:
        @staticmethod
        def conveyors() -> tuple[ConveyorStatus, ...]:
            return (
                ConveyorStatus(bay_id=1, mode="paused"),
                ConveyorStatus(bay_id=2, mode="feeding"),
            )

    ctl = StageController.__new__(StageController)
    ctl.coordinator = _Co()  # type: ignore[assignment]
    ctl.armed = {1: (True, False), 2: (True, False)}
    view = {c.bay_id: c.incoming_jam for c in ctl._conveyors_view()}
    assert view == {1: True, 2: False}
