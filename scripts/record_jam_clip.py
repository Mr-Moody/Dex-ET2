"""Record a cinematic clip of a Bay 1 flap jam: wide shot, then a zoom into the stall point.

    uv run python scripts/record_jam_clip.py [--out data/clips/bay1_flap_jam.mp4] [--rescue]
        [--size 1920x1080] [--fps 30] [--lead 5] [--slowmo 0.35] [--azimuth 0] [--no-overlay]
        [--captions operator|fleet]

Two passes. The simulation pass plays the same take as ``teleop atc --demo-rescue`` (routine
cycle, the fault rig jams Bay 1's flap, the bolt stalls against it) and keeps a snapshot every
60 Hz tick, so the stall time is known before anything is drawn. The render pass draws those
snapshots offline through the console's ``FeedRenderer`` (same warehouse, lights and conveyor
animation as the CCTV wall) with a free camera: wide on Bay 1, easing into the bolt/flap
contact just before the stall, slow motion through the impact, then a held close-up with a red
alert pulse. ``--rescue`` keeps rolling through the operator rescue with a pull-back.
"""

from __future__ import annotations

import argparse
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import av
import cv2
import mujoco as _mujoco  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
import numpy.typing as npt

from teleop.atc.demo_rescue import DemoRescueDirector
from teleop.atc.stage import StageController
from teleop.atc.types import BOLT_CLEARED_M, BayState
from teleop.commands.atc import DEFAULT_CONFIG, load_config, resolve_seed
from teleop.compliance.protocol import ComplianceEvent
from teleop.console.background import BackgroundPlayer, load_cycle
from teleop.console.render import FeedRenderer
from teleop.console.snapshot import SimSnapshot
from teleop.reasoning.protocol import ReasoningContext, ReasoningVerdict
from teleop.sim.hands import REPO_ROOT
from teleop.unoq.client import UnoQClient
from teleop.unoq.server import UnoQServer

logger = logging.getLogger("record_jam_clip")
mj: Any = _mujoco

BAY = 1
DT = 1.0 / 60.0
DEFAULT_OUT = REPO_ROOT / "data" / "clips" / "bay1_flap_jam.mp4"

# Shots, in the latch frame (metres): look-at point, distance, elevation and azimuth offset (deg)
# from a side-on view along latch +y. The close-up frames the bolt tip against the raised flap.
WIDE = ((0.03, 0.0, 0.10), 1.9, -30.0, 38.0)
MEDIUM = ((0.03, 0.0, 0.04), 0.60, -24.0, 22.0)
CLOSE = ((0.032, 0.0, 0.010), 0.16, -14.0, 6.0)

ZOOM_BEFORE_S = 1.6  # sim seconds before the stall the zoom starts
ZOOM_AFTER_S = 0.25  # ... and lands this long after it
SLOWMO_WINDOW = (-0.8, 1.4)  # sim seconds around the stall played at --slowmo speed
HOLD_S = 2.6  # sim seconds of held close-up after the stall (no --rescue)
ALERT = (235, 52, 52)  # RGB
OPERATOR = (245, 175, 40)
CLEAR = (60, 210, 110)
FLEET = (70, 185, 245)

# Lower-third captions per phase (stall, recovery, cleared): title, subline, colour. The subline
# takes ``{bolt}`` (mm) and ``{state}``. ``fleet`` tells the same take as a skill the fleet learned.
CAPTIONS: dict[str, tuple[tuple[str, str, Any], ...]] = {
    "operator": (
        ("CONTACT STALL  -  FLAP JAM", "bolt stopped at {bolt} mm  |  {state}", ALERT),
        (
            "OPERATOR RESCUE  -  PRESSING FLAP",
            "bolt at {bolt} mm  |  glove teleop, clutch held",
            OPERATOR,
        ),
        ("BOLT CLEARED  -  OPERATOR RESCUE", "bolt at {bolt} mm  |  {state}", CLEAR),
    ),
    "fleet": (
        (
            "CONTACT STALL  -  FLAP JAM",
            "bolt stopped at {bolt} mm  |  matching fleet skill library",
            ALERT,
        ),
        (
            "SELF-RECOVERY  -  FLEET-LEARNED SKILL",
            "bolt at {bolt} mm  |  skill learned from fleet rescues",
            FLEET,
        ),
        (
            "BOLT CLEARED  -  AUTONOMOUS RECOVERY",
            "bolt at {bolt} mm  |  experience shared back to the fleet",
            CLEAR,
        ),
    ),
}
STATE_TEXT = {
    BayState.STALLED: "STALL ALERT",
    BayState.QUEUED: "QUEUED FOR OPERATOR",
    BayState.RESCUE: "AWAITING OPERATOR",
}


# --------------------------------------------------------------------------- simulation pass


class _Sink:
    def emit(self, event: ComplianceEvent) -> None:
        pass

    def close(self) -> None:
        pass


class _Reasoner:
    def submit(self, ctx: ReasoningContext) -> None:
        pass

    def latest_verdict(self, bay_id: int) -> ReasoningVerdict | None:
        return None

    def close(self) -> None:
        pass


@dataclass
class Take:
    snaps: list[SimSnapshot]
    states: list[BayState]
    bolt_mm: list[float]
    t_stall: float
    t_rescue: float | None
    t_clear: float | None
    model: Any
    latch_body: int
    background: bool


def simulate(seed: int | None, rescue: bool, lead_s: float) -> Take:
    server = UnoQServer(host="127.0.0.1", udp_port=0, tcp_port=0, name="clip").start()
    client = UnoQClient("127.0.0.1", server.udp_port, server.tcp_port, local_server=server)
    client.connect()
    try:
        overrides: dict[str, Any] = {"demo_rescue": True, "seed": seed}
        cfg = load_config(DEFAULT_CONFIG, overrides)
        ctl = StageController.build(
            client,
            _Sink(),
            _Reasoner(),
            cycle_dwell_s=cfg.cycle_dwell_s,
            include_background_bays=cfg.background_bays,
            seed=resolve_seed(cfg),
            realism=cfg.realism,
            gate_pass_min=cfg.gate_pass_min,
        )
        director = DemoRescueDirector(ctl, seed=resolve_seed(cfg))
        co = ctl.coordinator
        rig = co.rig(BAY)
        snaps: list[SimSnapshot] = []
        states: list[BayState] = []
        bolt: list[float] = []
        t_stall = t_rescue = t_clear = None
        jammed = False
        end_t = math.inf
        while co.time_s < end_t and not director.done:
            ctl.tick(director.step())
            state = co.state(BAY)
            jammed = jammed or ctl.beat == "demo: flap jam"
            if jammed and t_stall is None and state != BayState.NOMINAL:
                t_stall = co.time_s
                logger.info("stall at %.2f s (bolt %.1f mm)", t_stall, rig.bolt_x() * 1000)
                if not rescue:
                    end_t = t_stall + HOLD_S
            if (
                rescue
                and t_stall is not None
                and t_rescue is None
                and ctl.beat == "demo: operator rescue"
            ):
                t_rescue = co.time_s
            if t_rescue is not None and t_clear is None and rig.bolt_x() >= BOLT_CLEARED_M:
                t_clear = co.time_s
                end_t = t_clear + 2.0
            snaps.append(SimSnapshot.capture(co.data, ctl.hud_view(), co.time_s))
            states.append(state)
            bolt.append(rig.bolt_x() * 1000.0)
            if co.time_s > 240.0:
                raise RuntimeError("no stall within 240 s of sim time")
        if t_stall is None:
            raise RuntimeError("Bay 1 never stalled")
        # Drop the routine lead-in before the clip starts.
        t0 = t_stall - lead_s
        keep = [i for i, s in enumerate(snaps) if s.sim_t >= t0]
        latch_body = int(co.model.body(f"{rig.prefix}latch_base").id)
        return Take(
            [snaps[i] for i in keep],
            [states[i] for i in keep],
            [bolt[i] for i in keep],
            t_stall,
            t_rescue,
            t_clear,
            co.model,
            latch_body,
            bool(ctl.background_bays),
        )
    finally:
        client.close()


# --------------------------------------------------------------------------- camera + timing


def _smooth(a: float) -> float:
    a = min(max(a, 0.0), 1.0)
    return a * a * a * (10.0 + a * (-15.0 + 6.0 * a))


Shot = tuple[tuple[float, float, float], float, float, float]


def _blend(a: Shot, b: Shot, w: float) -> Shot:
    la = np.asarray(a[0]) + w * (np.asarray(b[0]) - np.asarray(a[0]))
    dist = math.exp(math.log(a[1]) + w * (math.log(b[1]) - math.log(a[1])))
    return (
        (float(la[0]), float(la[1]), float(la[2])),
        dist,
        a[2] + w * (b[2] - a[2]),
        a[3] + w * (b[3] - a[3]),
    )


def shot_at(t: float, take: Take) -> Shot:
    """Camera shot at sim time ``t``."""
    ts = take.t_stall
    shot = _blend(WIDE, CLOSE, _smooth((t - (ts - ZOOM_BEFORE_S)) / (ZOOM_BEFORE_S + ZOOM_AFTER_S)))
    if t > ts + ZOOM_AFTER_S:  # slow push-in on the held close-up
        push = _smooth((t - ts - ZOOM_AFTER_S) / 3.0)
        shot = (shot[0], shot[1] * (1.0 - 0.12 * push), shot[2], shot[3])
    if take.t_rescue is not None and t > take.t_rescue - 0.6:  # pull back for the rescue
        shot = _blend(shot, MEDIUM, _smooth((t - take.t_rescue + 0.6) / 1.6))
    return shot


def speed_at(t: float, take: Take, slowmo: float) -> float:
    lo, hi = take.t_stall + SLOWMO_WINDOW[0], take.t_stall + SLOWMO_WINDOW[1]
    ramp = 0.35
    if t < lo - ramp or t > hi + ramp:
        return 1.0
    if t < lo:
        return 1.0 + (slowmo - 1.0) * _smooth((t - lo + ramp) / ramp)
    if t > hi:
        return slowmo + (1.0 - slowmo) * _smooth((t - hi) / ramp)
    return slowmo


def apply_camera(cam: Any, shot: Shot, origin: npt.NDArray[np.float64], r: Any) -> None:
    look, dist, elev, az_off = shot
    side = r @ np.array([0.0, 1.0, 0.0])  # view along latch +y (camera on the -y side)
    base_az = math.degrees(math.atan2(float(side[1]), float(side[0])))
    lookat = origin + r @ np.asarray(look)
    cam.type = mj.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance = dist
    cam.elevation = elev
    cam.azimuth = base_az + az_off


# --------------------------------------------------------------------------- overlay


def _text(img: Any, s: str, org: tuple[int, int], scale: float, color: Any, thick: int) -> None:
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def overlay(
    img: npt.NDArray[np.uint8],
    t: float,
    take: Take,
    bolt_mm: float,
    state: BayState,
    captions: str = "operator",
) -> Any:
    h, w = img.shape[:2]
    k = h / 1080.0
    out = img.copy()
    _text(out, f"BAY {BAY}  CAM 2", (int(40 * k), int(60 * k)), 0.9 * k, (235, 235, 235), 2)
    rel = t - take.t_stall
    _text(out, f"T{rel:+06.2f}s", (int(40 * k), int(100 * k)), 0.7 * k, (190, 190, 190), 1)
    rec_on = int(t * 2) % 2 == 0
    if rec_on:
        cv2.circle(out, (w - int(150 * k), int(52 * k)), int(10 * k), ALERT, -1, cv2.LINE_AA)
    _text(out, "REC", (w - int(128 * k), int(62 * k)), 0.9 * k, (235, 235, 235), 2)
    if rel >= 0.0:
        pulse = math.exp(-rel * 1.6) * (0.6 + 0.4 * math.cos(rel * 9.0))
        if pulse > 0.02:
            border = np.zeros_like(out)
            cv2.rectangle(border, (0, 0), (w - 1, h - 1), ALERT, int(28 * k))
            out = cv2.addWeighted(out, 1.0, border, min(pulse, 1.0), 0.0)
            tint = np.full_like(out, ALERT)
            out = cv2.addWeighted(out, 1.0 - 0.18 * pulse, tint, 0.18 * pulse, 0.0)
        a = _smooth(rel / 0.25)
        y = int(h - 150 * k)
        bar = out.copy()
        cv2.rectangle(bar, (0, y - int(70 * k)), (int(w * 0.62), y + int(60 * k)), (12, 12, 14), -1)
        out = cv2.addWeighted(out, 1.0 - 0.72 * a, bar, 0.72 * a, 0.0)
        label = STATE_TEXT.get(state, state.value.replace("_", " "))
        if take.t_clear is not None and t >= take.t_clear:
            phase = 2
        elif take.t_rescue is not None and t >= take.t_rescue:
            phase = 1
        else:
            phase = 0
        title, sub_fmt, color = CAPTIONS[captions][phase]
        sub = sub_fmt.format(bolt=f"{bolt_mm:.1f}", state=label)
        cv2.rectangle(out, (0, y - int(70 * k)), (int(10 * k), y + int(60 * k)), color, -1)
        if a > 0.3:
            _text(out, title, (int(44 * k), y), 1.35 * k, color, 3)
            _text(out, sub, (int(46 * k), y + int(42 * k)), 0.8 * k, (225, 225, 225), 2)
    return out


# --------------------------------------------------------------------------- render pass


def _snapshot_at(take: Take, t: float, model: Any) -> tuple[SimSnapshot, int]:
    """Snapshot at sim time ``t``, qpos interpolated between the bracketing ticks."""
    snaps = take.snaps
    t0 = snaps[0].sim_t
    f = (t - t0) / DT
    i = min(max(int(math.floor(f)), 0), len(snaps) - 1)
    j = min(i + 1, len(snaps) - 1)
    a = min(max(f - i, 0.0), 1.0)
    s0, s1 = snaps[i], snaps[j]
    qpos = s0.qpos + a * (s1.qpos - s0.qpos)
    mj.mj_normalizeQuat(model, qpos)
    snap = SimSnapshot(qpos, s0.mocap_pos, s0.mocap_quat, s0.view, t)
    return snap, (j if a >= 0.5 else i)


def render(take: Take, out: Path, size: tuple[int, int], fps: int, args: Any) -> int:
    w, h = size
    player = BackgroundPlayer(load_cycle(), take.model) if take.background else None
    fr = FeedRenderer(take.model, player)
    m, d = fr.model, fr.data
    m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), w)
    m.vis.global_.offheight = max(int(m.vis.global_.offheight), h)
    m.vis.quality.shadowsize = max(int(m.vis.quality.shadowsize), 4096)
    renderer = mj.Renderer(m, h, w)
    cam = mj.MjvCamera()
    extent = max(float(m.stat.extent), 1e-6)
    base_near = float(m.vis.map.znear)

    out.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(out), mode="w")
    try:
        stream = container.add_stream("libx264", rate=fps)
        stream.options = {"crf": "17", "preset": "slow"}
    except Exception:  # noqa: BLE001 - no libx264 in this PyAV build
        stream = container.add_stream("mpeg4", rate=fps)
        stream.bit_rate = 16_000_000
    stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"

    t = take.snaps[0].sim_t
    t_end = take.snaps[-1].sim_t
    frames = 0
    try:
        while t <= t_end:
            snap, idx = _snapshot_at(take, t, m)
            d.qpos[:] = snap.qpos
            if snap.mocap_pos.size:
                d.mocap_pos[:] = snap.mocap_pos
                d.mocap_quat[:] = snap.mocap_quat
            if player is not None:
                player.apply(d, t, frozen=snap.view.powered_off)
            fr._animate_conveyors(snap, t)  # pyright: ignore[reportPrivateUsage]
            mj.mj_kinematics(m, d)
            mj.mj_camlight(m, d)
            fr._light_only(BAY)  # pyright: ignore[reportPrivateUsage]
            origin = np.asarray(d.xpos[take.latch_body], dtype=np.float64)
            r = np.asarray(d.xmat[take.latch_body], dtype=np.float64).reshape(3, 3)
            shot = shot_at(t, take)
            shot = (shot[0], shot[1], shot[2], shot[3] + args.azimuth)
            apply_camera(cam, shot, origin, r)
            # Metric near plane that follows the dolly, so the close-up is not clipped.
            m.vis.map.znear = min(base_near, 0.04 * shot[1] / extent)
            renderer.update_scene(d, camera=cam)
            img = renderer.render()
            if not args.no_overlay:
                img = overlay(img, t, take, take.bolt_mm[idx], take.states[idx], args.captions)
            frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(img), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
            frames += 1
            t += speed_at(t, take, args.slowmo) / fps
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()
        renderer.close()
        fr.close()
    return frames


def _size(text: str) -> tuple[int, int]:
    w, _, h = text.lower().partition("x")
    return int(w), int(h)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--size", type=_size, default=(1920, 1080), help="WxH, default 1920x1080")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--lead", type=float, default=5.0, help="sim seconds before the stall")
    ap.add_argument("--slowmo", type=float, default=0.35, help="speed through the impact")
    ap.add_argument("--azimuth", type=float, default=0.0, help="extra camera azimuth, degrees")
    ap.add_argument("--rescue", action="store_true", help="keep rolling through the rescue")
    ap.add_argument("--no-overlay", action="store_true", help="no captions or alert pulse")
    ap.add_argument(
        "--captions",
        choices=tuple(CAPTIONS),
        default="operator",
        help="operator: human rescue story; fleet: the robot recovers with a fleet-learned skill",
    )
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("simulating the jam take...")
    take = simulate(args.seed, args.rescue, args.lead)
    logger.info("rendering %d sim ticks at %dx%d...", len(take.snaps), *args.size)
    frames = render(take, args.out, args.size, args.fps, args)
    logger.info("wrote %s (%d frames, %.1f s)", args.out, frames, frames / args.fps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
