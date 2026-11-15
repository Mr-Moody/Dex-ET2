"""Record the fleet-learning story as one video, from one continuous simulation.

    uv run python scripts/record_fleet_story.py [--out data/clips/fleet_learning_story.mp4]
        [--size 1920x1080] [--fps 30] [--seed 7] [--slowmo 0.35] [--timelapse 4]

Everything shown happens in a single run of the stage, in order:

1. A robot hits a fault: Bay 1's flap is jammed and the bolt stalls against it.
2. A human steps in: an operator rescues Bay 1 by glove (the ``--demo-rescue`` take).
3. The fleet learns: the rescue is fitted as a skill and passes the randomized sim gate
   (time-lapse), then is deployed to the fleet library.
4. Another robot, same fault: Bay 2's flap jams and its bolt stalls.
5. Autonomous recovery: Bay 2 matches the learned skill and clears itself, no operator.

The take is simulated first (a snapshot per 60 Hz tick plus the event times), then each chapter
is rendered through the console's ``FeedRenderer`` with a free camera. Sections fade into each
other with only the lower-third captions on screen; ``--cards`` adds intro, chapter and end title
cards and ``--hud`` the camera/REC corner text. Shot, caption and camera helpers come from
``record_jam_clip.py``.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))

from record_jam_clip import (  # noqa: E402 - sibling script
    ALERT,
    CLEAR,
    CLOSE,
    DT,
    FLEET,
    MEDIUM,
    OPERATOR,
    WIDE,
    Shot,
    _blend,
    _Reasoner,
    _Sink,
    _smooth,
    _text,
    apply_camera,
    mj,
)

from teleop.atc.demo_rescue import DemoRescueDirector  # noqa: E402
from teleop.atc.stage import StageController  # noqa: E402
from teleop.atc.types import BOLT_CLEARED_M, BayState  # noqa: E402
from teleop.commands.atc import DEFAULT_CONFIG, load_config, resolve_seed  # noqa: E402
from teleop.console.background import BackgroundPlayer, load_cycle  # noqa: E402
from teleop.console.render import FeedRenderer  # noqa: E402
from teleop.console.snapshot import SimSnapshot  # noqa: E402
from teleop.sim.bays import OVERVIEW_CAMERA  # noqa: E402
from teleop.sim.hands import REPO_ROOT  # noqa: E402
from teleop.unoq.client import UnoQClient  # noqa: E402
from teleop.unoq.server import UnoQServer  # noqa: E402

logger = logging.getLogger("record_fleet_story")

DEFAULT_OUT = REPO_ROOT / "data" / "clips" / "fleet_learning_story.mp4"
CARD_S = 1.8
END_CARD_S = 3.6
FADE_S = 0.3


# --------------------------------------------------------------------------- simulation


@dataclass
class Story:
    snaps: list[SimSnapshot]
    bolt: dict[int, list[float]]  # bay -> bolt mm per tick
    t_stall1: float
    t_operator: float
    t_clear1: float
    t_gate: float
    t_stall2: float
    t_clear2: float
    gate_passed: int
    gate_total: int
    fit_ms: float
    model: Any
    latch_body: dict[int, int]
    background: bool


def simulate(seed: int | None) -> Story:
    server = UnoQServer(host="127.0.0.1", udp_port=0, tcp_port=0, name="story").start()
    client = UnoQClient("127.0.0.1", server.udp_port, server.tcp_port, local_server=server)
    client.connect()
    try:
        cfg = load_config(DEFAULT_CONFIG, {"demo_rescue": True, "seed": seed})
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
        co = ctl.coordinator
        director = DemoRescueDirector(ctl, seed=resolve_seed(cfg))
        snaps: list[SimSnapshot] = []
        bolt: dict[int, list[float]] = {1: [], 2: []}
        ev: dict[str, float] = {}

        def record() -> None:
            snaps.append(SimSnapshot.capture(co.data, ctl.hud_view(), co.time_s))
            for b in (1, 2):
                bolt[b].append(co.rig(b).bolt_x() * 1000.0)

        jammed = False
        while not director.done:
            ctl.tick(director.step())
            t = co.time_s
            jammed = jammed or ctl.beat == "demo: flap jam"
            if jammed and "stall1" not in ev and co.state(1) != BayState.NOMINAL:
                ev["stall1"] = t
            if "stall1" in ev and "operator" not in ev and ctl.beat == "demo: operator rescue":
                ev["operator"] = t
            if "operator" in ev and "clear1" not in ev and co.rig(1).bolt_x() >= BOLT_CLEARED_M:
                ev["clear1"] = t
            record()
            if t > 240.0:
                raise RuntimeError("Bay 1 story did not finish")
        if not (director.rescued and director.deployed):
            raise RuntimeError("Bay 1 rescue or skill deploy failed; cannot tell the story")
        ev["gate"] = co.time_s
        gate = ctl.gate
        logger.info(
            "bay 1: stall %.1f s, cleared %.1f s, gate %s/%s",
            ev["stall1"],
            ev["clear1"],
            gate.passed,
            gate.total,
        )

        ctl.on_key(ord("G"))  # Bay 2's flap jams on its next cycle
        t_end = math.inf
        autonomous = False
        while co.time_s < t_end:
            ctl.tick(None)
            t = co.time_s
            if "stall2" not in ev and co.state(2) != BayState.NOMINAL:
                ev["stall2"] = t
            if "stall2" in ev and "clear2" not in ev and co.rig(2).bolt_x() >= BOLT_CLEARED_M:
                ev["clear2"] = t
                t_end = t + 3.0
            autonomous = autonomous or co.outcome(2) == "cleared autonomously"
            record()
            if t > ev["gate"] + 90.0:
                raise RuntimeError("Bay 2 never stalled and cleared")
        if not autonomous:
            raise RuntimeError("Bay 2 did not clear autonomously")
        logger.info("bay 2: stall %.1f s, cleared autonomously %.1f s", ev["stall2"], ev["clear2"])
        latch = {b: int(co.model.body(f"{co.rig(b).prefix}latch_base").id) for b in (1, 2)}
        return Story(
            snaps,
            bolt,
            ev["stall1"],
            ev["operator"],
            ev["clear1"],
            ev["gate"],
            ev["stall2"],
            ev["clear2"],
            int(gate.passed or 0),
            int(gate.total or 0),
            float(gate.fit_ms or math.nan),
            co.model,
            latch,
            bool(ctl.background_bays),
        )
    finally:
        client.close()


# --------------------------------------------------------------------------- chapters

Caption = tuple[str, str, Any, float] | None  # title, subline, colour, progress (-1: no bar)


@dataclass
class Segment:
    chapter: str
    title: str
    t0: float
    t1: float
    bay: int | None  # None: fleet overview of Bays 1 and 2
    shot: Callable[[float], Shot]
    speed: Callable[[float], float]
    caption: Callable[[float, float], Caption]  # (sim t, bolt mm of the bay)
    pulse_at: float | None = None
    tag: str | None = None  # e.g. "TIME-LAPSE 4x"


def _slowmo(t_stall: float, slowmo: float) -> Callable[[float], float]:
    lo, hi, ramp = t_stall - 0.8, t_stall + 1.4, 0.35

    def speed(t: float) -> float:
        if t < lo - ramp or t > hi + ramp:
            return 1.0
        if t < lo:
            return 1.0 + (slowmo - 1.0) * _smooth((t - lo + ramp) / ramp)
        if t > hi:
            return slowmo + (1.0 - slowmo) * _smooth((t - hi) / ramp)
        return slowmo

    return speed


def _zoom_in(t_stall: float) -> Callable[[float], Shot]:
    def shot(t: float) -> Shot:
        s = _blend(WIDE, CLOSE, _smooth((t - (t_stall - 1.6)) / 1.85))
        if t > t_stall + 0.25:
            push = _smooth((t - t_stall - 0.25) / 3.0)
            s = (s[0], s[1] * (1.0 - 0.12 * push), s[2], s[3])
        return s

    return shot


def _medium(t0: float) -> Callable[[float], Shot]:
    def shot(t: float) -> Shot:
        push = _smooth((t - t0) / 8.0)
        return (MEDIUM[0], MEDIUM[1] * (1.0 - 0.15 * push), MEDIUM[2], MEDIUM[3])

    return shot


def chapters(s: Story, slowmo: float, timelapse: float) -> list[Segment]:
    gate_txt = f"{s.gate_passed}/{s.gate_total}"
    t_val0 = s.t_clear1 + 1.2

    def cap1(t: float, mm: float) -> Caption:
        if t < s.t_stall1:
            return None
        return ("CONTACT STALL  -  FLAP JAM", f"Bay 1  |  bolt stopped at {mm:.1f} mm", ALERT, -1)

    def cap2(t: float, mm: float) -> Caption:
        if t >= s.t_clear1:
            return (
                "BOLT CLEARED  -  DEMONSTRATION CAPTURED",
                "rescue trajectory recorded for the fleet to learn from",
                CLEAR,
                -1,
            )
        return (
            "HUMAN OPERATOR  -  GLOVE TELEOP",
            f"thumb presses the flap down, index pushes the bolt  |  {mm:.1f} mm",
            OPERATOR,
            -1,
        )

    def cap3(t: float, mm: float) -> Caption:
        if t >= s.t_gate:
            return (
                "SKILL DEPLOYED TO THE FLEET",
                f"{gate_txt} randomized sim rollouts passed",
                CLEAR,
                1.0,
            )
        prog = _smooth((t - t_val0) / max(s.t_gate - t_val0, 1e-3))
        return (
            "LEARNING FROM ONE DEMONSTRATION",
            f"skill fit in {s.fit_ms:.1f} ms  |  validating in randomized sim",
            FLEET,
            prog,
        )

    def cap4(t: float, mm: float) -> Caption:
        if t < s.t_stall2:
            return None
        return (
            "SAME FAULT  -  BAY 2",
            f"bolt stopped at {mm:.1f} mm  |  match found in fleet skill library",
            ALERT,
            -1,
        )

    def cap5(t: float, mm: float) -> Caption:
        if t >= s.t_clear2:
            return (
                "BOLT CLEARED  -  NO HUMAN NEEDED",
                "Bay 2 recovered on its own with the skill Bay 1 taught it",
                CLEAR,
                -1,
            )
        return (
            "AUTONOMOUS RECOVERY  -  FLEET-LEARNED SKILL",
            f"no operator, clutch idle  |  bolt at {mm:.1f} mm",
            FLEET,
            -1,
        )

    one = lambda t: 1.0  # noqa: E731
    split2 = s.t_stall2 + 1.3
    return [
        Segment(
            "1",
            "A ROBOT HITS A FAULT",
            s.t_stall1 - 5.0,
            s.t_operator,
            1,
            _zoom_in(s.t_stall1),
            _slowmo(s.t_stall1, slowmo),
            cap1,
            s.t_stall1,
        ),
        Segment(
            "2",
            "A HUMAN STEPS IN",
            s.t_operator + 0.4,
            s.t_clear1 + 1.2,
            1,
            _medium(s.t_operator),
            one,
            cap2,
        ),
        Segment(
            "3",
            "THE FLEET LEARNS",
            t_val0,
            s.t_gate + 1.5,
            None,
            lambda t: ((0.0, 0.0, 0.0), 1.0, -32.0, 30.0),
            lambda t: timelapse,
            cap3,
            tag=f"TIME-LAPSE {timelapse:g}x",
        ),
        Segment(
            "4",
            "ANOTHER ROBOT, SAME FAULT",
            s.t_stall2 - 4.5,
            split2,
            2,
            _zoom_in(s.t_stall2),
            _slowmo(s.t_stall2, slowmo),
            cap4,
            s.t_stall2,
        ),
        Segment(
            "5", "AUTONOMOUS RECOVERY", split2, s.t_clear2 + 2.5, 2, _medium(split2), one, cap5
        ),
    ]


# --------------------------------------------------------------------------- drawing


def _font(size: int, bold: bool = False) -> Any:
    names = ("segoeuib.ttf", "arialbd.ttf") if bold else ("segoeui.ttf", "arial.ttf")
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def card(size: tuple[int, int], kicker: str, title: str, sub: str | None, alpha: float) -> Any:
    w, h = size
    k = h / 1080.0
    img = Image.new("RGB", (w, h), (10, 11, 14))
    d = ImageDraw.Draw(img)
    a = max(0.0, min(alpha, 1.0))

    def col(c: tuple[int, int, int]) -> tuple[int, int, int]:
        return (int(10 + (c[0] - 10) * a), int(11 + (c[1] - 11) * a), int(14 + (c[2] - 14) * a))

    f_k, f_t, f_s = _font(int(34 * k), True), _font(int(84 * k), True), _font(int(36 * k))
    cy = h // 2
    d.text((w // 2, cy - int(80 * k)), kicker, font=f_k, fill=col((120, 190, 245)), anchor="mm")
    d.text((w // 2, cy), title, font=f_t, fill=col((240, 240, 240)), anchor="mm")
    if sub:
        d.text((w // 2, cy + int(118 * k)), sub, font=f_s, fill=col((170, 170, 175)), anchor="mm")
    lw = int(220 * k * _smooth(a))
    d.rectangle((w // 2 - lw, cy + int(66 * k), w // 2 + lw, cy + int(70 * k)), fill=col(ALERT))
    return np.asarray(img)


def draw_overlay(
    img: npt.NDArray[np.uint8], seg: Segment, t: float, mm: float, hud: bool = False
) -> Any:
    """Lower-third caption and stall pulse; ``hud`` adds the camera/chapter/REC corner text."""
    h, w = img.shape[:2]
    k = h / 1080.0
    out = img.copy()
    if hud:
        cam = f"BAY {seg.bay}  CAM 2" if seg.bay is not None else "FLEET OVERVIEW"
        _text(out, cam, (int(40 * k), int(60 * k)), 0.9 * k, (235, 235, 235), 2)
        chapter = f"{seg.chapter}  {seg.title}"
        _text(out, chapter, (int(40 * k), int(100 * k)), 0.7 * k, (190, 190, 190), 1)
        if seg.tag:
            _text(out, seg.tag, (w - int(330 * k), int(62 * k)), 0.9 * k, (120, 190, 245), 2)
        else:
            if int(t * 2) % 2 == 0:
                center = (w - int(150 * k), int(52 * k))
                cv2.circle(out, center, int(10 * k), ALERT, -1, cv2.LINE_AA)
            _text(out, "REC", (w - int(128 * k), int(62 * k)), 0.9 * k, (235, 235, 235), 2)
    if seg.pulse_at is not None and t >= seg.pulse_at:
        rel = t - seg.pulse_at
        pulse = math.exp(-rel * 1.6) * (0.6 + 0.4 * math.cos(rel * 9.0))
        if pulse > 0.02:
            border = np.zeros_like(out)
            cv2.rectangle(border, (0, 0), (w - 1, h - 1), ALERT, int(28 * k))
            out = cv2.addWeighted(out, 1.0, border, min(pulse, 1.0), 0.0)
            tint = np.full_like(out, ALERT)
            out = cv2.addWeighted(out, 1.0 - 0.18 * pulse, tint, 0.18 * pulse, 0.0)
    cap = seg.caption(t, mm)
    if cap is None:
        return out
    title, sub, color, prog = cap
    y = int(h - 150 * k)
    bar = out.copy()
    cv2.rectangle(bar, (0, y - int(70 * k)), (int(w * 0.66), y + int(60 * k)), (12, 12, 14), -1)
    out = cv2.addWeighted(out, 0.28, bar, 0.72, 0.0)
    cv2.rectangle(out, (0, y - int(70 * k)), (int(10 * k), y + int(60 * k)), color, -1)
    _text(out, title, (int(44 * k), y), 1.35 * k, color, 3)
    _text(out, sub, (int(46 * k), y + int(42 * k)), 0.8 * k, (225, 225, 225), 2)
    if prog >= 0.0:
        x0, x1, yb = int(w * 0.66) + int(30 * k), w - int(60 * k), y - int(10 * k)
        cv2.rectangle(out, (x0, yb), (x1, yb + int(16 * k)), (40, 40, 46), -1)
        cv2.rectangle(out, (x0, yb), (x0 + int((x1 - x0) * prog), yb + int(16 * k)), color, -1)
        n = int(round(prog * 20))
        _text(out, f"rollouts {n}/20", (x0, yb - int(14 * k)), 0.8 * k, (225, 225, 225), 2)
    return out


# --------------------------------------------------------------------------- render


class Writer:
    def __init__(self, out: Path, size: tuple[int, int], fps: int) -> None:
        out.parent.mkdir(parents=True, exist_ok=True)
        self.container: Any = av.open(str(out), mode="w")
        try:
            self.stream: Any = self.container.add_stream("libx264", rate=fps)
            self.stream.options = {"crf": "17", "preset": "slow"}
        except Exception:  # noqa: BLE001 - no libx264 in this PyAV build
            self.stream = self.container.add_stream("mpeg4", rate=fps)
            self.stream.bit_rate = 16_000_000
        self.stream.width, self.stream.height, self.stream.pix_fmt = *size, "yuv420p"
        self.frames = 0

    def write(self, img: Any) -> None:
        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(img), format="rgb24")
        for packet in self.stream.encode(frame):
            self.container.mux(packet)
        self.frames += 1

    def close(self) -> None:
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()


def _times(seg: Segment, fps: int) -> Iterator[float]:
    t = seg.t0
    while t <= seg.t1:
        yield t
        t += seg.speed(t) / fps


def render(
    story: Story,
    out: Path,
    size: tuple[int, int],
    fps: int,
    segs: list[Segment],
    *,
    cards_on: bool = False,
    hud: bool = False,
) -> int:
    w, h = size
    player = BackgroundPlayer(load_cycle(), story.model) if story.background else None
    fr = FeedRenderer(story.model, player)
    m, d = fr.model, fr.data
    m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), w)
    m.vis.global_.offheight = max(int(m.vis.global_.offheight), h)
    m.vis.quality.shadowsize = max(int(m.vis.quality.shadowsize), 4096)
    renderer = mj.Renderer(m, h, w)
    cam = mj.MjvCamera()
    extent = max(float(m.stat.extent), 1e-6)
    base_near = float(m.vis.map.znear)
    overview_id = int(m.camera(OVERVIEW_CAMERA).id)
    snaps = story.snaps
    t_first = snaps[0].sim_t
    writer = Writer(out, size, fps)

    def pose(t: float) -> tuple[SimSnapshot, int]:
        f = (t - t_first) / DT
        i = min(max(int(math.floor(f)), 0), len(snaps) - 1)
        j = min(i + 1, len(snaps) - 1)
        a = min(max(f - i, 0.0), 1.0)
        qpos = snaps[i].qpos + a * (snaps[j].qpos - snaps[i].qpos)
        mj.mj_normalizeQuat(m, qpos)
        s0 = snaps[i]
        return SimSnapshot(qpos, s0.mocap_pos, s0.mocap_quat, s0.view, t), (j if a >= 0.5 else i)

    def cards(kicker: str, title: str, sub: str | None, seconds: float) -> None:
        n = round(seconds * fps)
        for i in range(n):
            u = i / max(n - 1, 1)
            alpha = min(_smooth(u / 0.3), _smooth((1.0 - u) / 0.25))
            writer.write(card(size, kicker, title, sub, alpha))

    try:
        if cards_on:
            cards(
                "FLEET LEARNING",
                "ONE RESCUE, EVERY ROBOT",
                "a warehouse fleet that learns from its operators",
                2.6,
            )
        for seg in segs:
            if cards_on:
                cards(f"CHAPTER {seg.chapter}", seg.title, None, CARD_S)
            times = list(_times(seg, fps))
            fade_n = round(FADE_S * fps)
            for n_i, t in enumerate(times):
                snap, idx = pose(t)
                d.qpos[:] = snap.qpos
                if snap.mocap_pos.size:
                    d.mocap_pos[:] = snap.mocap_pos
                    d.mocap_quat[:] = snap.mocap_quat
                if player is not None:
                    player.apply(d, t, frozen=snap.view.powered_off)
                fr._animate_conveyors(snap, t)  # pyright: ignore[reportPrivateUsage]
                mj.mj_kinematics(m, d)
                mj.mj_camlight(m, d)
                fr._light_only(seg.bay or 1)  # pyright: ignore[reportPrivateUsage]
                shot = seg.shot(t)
                if seg.bay is None:  # the console's fixed warehouse overview camera
                    cam.type = mj.mjtCamera.mjCAMERA_FIXED
                    cam.fixedcamid = overview_id
                    dist = math.inf
                else:
                    body = story.latch_body[seg.bay]
                    origin = np.asarray(d.xpos[body], dtype=np.float64)
                    r = np.asarray(d.xmat[body], dtype=np.float64).reshape(3, 3)
                    apply_camera(cam, shot, origin, r)
                    dist = shot[1]
                m.vis.map.znear = min(base_near, 0.04 * dist / extent)  # inf: base
                renderer.update_scene(d, camera=cam)
                img = renderer.render()
                mm = story.bolt[seg.bay or 1][idx]
                img = draw_overlay(img, seg, t, mm, hud)
                fade = min(1.0, (n_i + 1) / fade_n, (len(times) - n_i) / fade_n)
                if fade < 1.0:
                    img = (img.astype(np.float32) * _smooth(fade)).astype(np.uint8)
                writer.write(img)
        if cards_on:
            rollouts = f"{story.gate_passed}/{story.gate_total}"
            cards(
                "RESULT",
                "1 HUMAN RESCUE  >  WHOLE FLEET",
                f"skill validated {rollouts} in sim, Bay 2 recovered with no operator",
                END_CARD_S,
            )
    finally:
        writer.close()
        renderer.close()
        fr.close()
    return writer.frames


def _size(text: str) -> tuple[int, int]:
    w, _, h = text.lower().partition("x")
    return int(w), int(h)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--size", type=_size, default=(1920, 1080), help="WxH, default 1920x1080")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--slowmo", type=float, default=0.35, help="speed through each impact")
    ap.add_argument("--timelapse", type=float, default=4.0, help="speed of the learning chapter")
    ap.add_argument("--cards", action="store_true", help="add intro, chapter and end title cards")
    ap.add_argument("--hud", action="store_true", help="add camera, chapter and REC corner text")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("simulating the story...")
    story = simulate(args.seed)
    segs = chapters(story, args.slowmo, args.timelapse)
    logger.info("rendering %d chapters at %dx%d...", len(segs), *args.size)
    frames = render(story, args.out, args.size, args.fps, segs, cards_on=args.cards, hud=args.hud)
    logger.info("wrote %s (%d frames, %.1f s)", args.out, frames, frames / args.fps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
