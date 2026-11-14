"""Canned background cycle for the fleet ops console's Bays 3-4."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mujoco as _mujoco  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
import pytest

from teleop.console.background import (
    BackgroundCycle,
    BackgroundPlayer,
    load_cycle,
    save_cycle,
)
from teleop.sim.bays import PANDA_XML, build_four_bay_scene
from teleop.sim.hands import load_hand

mj: Any = _mujoco


def _models_present() -> bool:
    try:
        right = load_hand("inspire").mjcf["right"]
    except FileNotFoundError:
        return False
    return PANDA_XML.is_file() and right.is_file()


needs_models = pytest.mark.skipif(
    not _models_present(), reason="Panda/Inspire MJCFs missing; run scripts/fetch_models.py"
)


def _two_frame_cycle() -> BackgroundCycle:
    return BackgroundCycle(
        t=np.array([0.0, 2.0]),
        joint_names=("joint1", "bolt_slide"),
        qpos_widths=(1, 1),
        qpos=np.array([[0.0, 0.0], [1.0, 0.06]]),
        confidence=np.array([0.9, 0.95]),
    )


def test_roundtrip(tmp_path: Path) -> None:
    c = _two_frame_cycle()
    save_cycle(c, tmp_path / "c.npz")
    back = load_cycle(tmp_path / "c.npz")
    assert back.joint_names == c.joint_names
    assert back.qpos_widths == c.qpos_widths
    assert np.allclose(back.qpos, c.qpos) and np.allclose(back.t, c.t)
    assert np.allclose(back.confidence, c.confidence)
    assert back.duration_s == pytest.approx(2.0)


def test_committed_cycle_is_plausible() -> None:
    c = load_cycle()
    assert 2.0 < c.duration_s < 30.0
    assert float(c.confidence.min()) > 0.5  # routine: brief contact dips, never near 0.3
    assert "joint1" in c.joint_names and "bolt_slide" in c.joint_names
    assert c.qpos.shape == (len(c.t), sum(c.qpos_widths))
    # The bolt actually slides open during the cycle.
    col = sum(c.qpos_widths[: c.joint_names.index("bolt_slide")])
    assert float(c.qpos[:, col].max()) > 0.05


def test_sample_wraps_continuously_for_arm_joints() -> None:
    player_cycle = _two_frame_cycle()
    q_end, _ = player_cycle.sample(player_cycle.duration_s - 1e-6)
    q_start, _ = player_cycle.sample(0.0)
    assert q_end[0] == pytest.approx(q_start[0], abs=1e-3)  # arm blends back to frame 0
    assert q_end[1] == pytest.approx(0.06, abs=1e-3)  # latch joints snap at the wrap


def test_sample_interpolates_between_frames() -> None:
    q, conf = _two_frame_cycle().sample(0.5)
    assert q[0] == pytest.approx(0.25)
    assert conf == pytest.approx(0.9125)


@needs_models
def test_player_moves_bay3_and_differs_from_bay4() -> None:
    model = build_four_bay_scene()
    data = mj.MjData(model)
    player = BackgroundPlayer(load_cycle(), model)
    conf = player.apply(data, wall_t=1.0)
    assert set(conf) == {3, 4}
    assert all(0.0 <= c <= 1.0 for c in conf.values())
    q3 = float(data.qpos[model.joint("bay3/joint1").qposadr[0]])
    q4 = float(data.qpos[model.joint("bay4/joint1").qposadr[0]])
    assert q3 != pytest.approx(q4)
    q3_later = player.apply(data, wall_t=3.0)
    assert q3_later  # still animating
    assert float(data.qpos[model.joint("bay3/joint1").qposadr[0]]) != pytest.approx(q3)


@needs_models
def test_player_leaves_active_bays_alone() -> None:
    model = build_four_bay_scene()
    data = mj.MjData(model)
    before = np.array(data.qpos[model.joint("bay1/joint1").qposadr[0]])
    BackgroundPlayer(load_cycle(), model).apply(data, wall_t=2.0)
    assert float(data.qpos[model.joint("bay1/joint1").qposadr[0]]) == pytest.approx(before)


@needs_models
def test_player_skips_absent_bays() -> None:
    model = build_four_bay_scene(include_background_bays=False)
    assert BackgroundPlayer(load_cycle(), model).apply(mj.MjData(model), 0.0) == {}


@needs_models
def test_player_freezes_a_powered_off_bay_and_resumes_where_it_stopped() -> None:
    model = build_four_bay_scene()
    data = mj.MjData(model)
    player = BackgroundPlayer(load_cycle(), model)
    adr3 = model.joint("bay3/joint1").qposadr[0]
    adr4 = model.joint("bay4/joint1").qposadr[0]

    player.apply(data, 1.0)
    q3_cut = float(data.qpos[adr3])
    player.apply(data, 3.0, frozen={3})
    assert float(data.qpos[adr3]) == pytest.approx(q3_cut)  # braked
    assert player.conveyor_statuses(3.0, frozen={3})[3].mode == "estop"
    assert player.conveyor_statuses(3.0, frozen={3})[4].mode != "estop"
    q4 = float(data.qpos[adr4])

    player.apply(data, 3.0)  # power back: replay continues from the cut, no jump
    assert float(data.qpos[adr3]) == pytest.approx(q3_cut)
    player.apply(data, 4.0)
    reference = BackgroundPlayer(load_cycle(), model)
    ref_data = mj.MjData(model)
    reference.apply(ref_data, 2.0)  # 2 s of replay since the start, like bay 3 now
    assert float(data.qpos[adr3]) == pytest.approx(float(ref_data.qpos[adr3]))
    assert q4 != pytest.approx(float(data.qpos[adr4]))  # bay 4 never stopped


@needs_models
def test_player_wraps_arm_and_conveyor_pitch_without_snapping() -> None:
    from teleop.console.background import PHASE_OFFSETS
    from teleop.sim.cells import PACK_PITCH_M

    model = build_four_bay_scene()
    data = mj.MjData(model)
    player = BackgroundPlayer(load_cycle(), model)
    _, _, loop_s = player.loop_params()
    t_wrap = (1.0 - PHASE_OFFSETS[3]) * loop_s

    player.apply(data, t_wrap - 1e-4)
    q_before = float(data.qpos[model.joint("bay3/joint1").qposadr[0]])
    st_before = player.conveyor_statuses(t_wrap - 1e-4)[3]

    player.apply(data, t_wrap + 1e-4)
    q_after = float(data.qpos[model.joint("bay3/joint1").qposadr[0]])
    st_after = player.conveyor_statuses(t_wrap + 1e-4)[3]

    assert q_before == pytest.approx(q_after, abs=2e-3)
    assert st_before.pack_dy_m == pytest.approx(PACK_PITCH_M, abs=2e-3)
    assert st_after.pack_dy_m == pytest.approx(0.0, abs=2e-3)
    assert st_before.belt_dist_m == pytest.approx(st_after.belt_dist_m, abs=2e-3)
