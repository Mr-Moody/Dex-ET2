"""Tier 1 & Tier 2 E2E tests for Features F10, F13, and F14.

Covers:
- F10: `AuditReviewer` (`gemini-4-argon`, `thinkingLevel="LOW"`, 1..6 `<=512px` JPEGs,
       non-blocking fire-and-forget `submit`, `pending` -> `complete` `AuditReport`,
       `dataset_tag` `"approve"`/`"flag"`, `max_reviews_per_hour` rate limit -> `"skipped"`,
       error -> `"error"`, **strictly 0 TTS calls**).
- F13: `ConsoleServer` HTTP `/`, `/audit.js`, `/console.js`, `/console.css`, `/ws`
       `publish_telemetry` + reconnect replay + standalone `{"type": "copilot_audit", ...}`
       WS ingress, `/usr/bin/node` multi-element fake-DOM tests on `audit.js` + `console.js`
       `updateGate`, in-place `builds` stability, zero `https?://` in `WEB_DIR`, button
       `mousedown preventDefault` + `tabIndex == -1`.
- F14: `EpisodeSink.latest_episode_uid` & `EpisodeSink.tag_episode` Race A pre-encode + Race B
       post-encode atomic rewrite keeping `1 line == 1 episode` + concurrent tags + unknown UID,
       fed directly into `run_atc_qc`, `write_clean`, and `teleop.commands.process.main`
       verifying `"copilot flagged"` in `DROP_REASONS`.
"""

from __future__ import annotations

import base64
import importlib
import inspect
import json
import os
import re
import subprocess
import threading
import time
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from websockets.sync.client import connect as ws_connect

import teleop.commands.process as process_cmd
from teleop.atc.episodes import ROBOT_TYPE, EpisodeSink
from teleop.atc.hud import GateStatus, HudView, RescueStatus
from teleop.console.server import WEB_DIR, ConsoleServer
from teleop.console.telemetry import TelemetryAdapter
from teleop.copilot.config import CopilotConfig
from teleop.copilot.gemini import GeminiClient
from teleop.policy.schema import CAMERAS, FPS, IMAGE_HW, lerobot_features
from teleop.process.atc_qc import DEFAULT_SOURCES, DROP_REASONS, AtcQCThresholds, run_atc_qc
from teleop.process.qc import read_sidecar, write_clean
from teleop.record.recorder import EpisodeRecorder
from teleop.types import Pose

from .conftest import (
    E2ECoordinatorRig,
    FakeAudioHarness,
    LoopbackGeminiTLSServer,
    construct_flexible,
    contains_key_recursive,
    jpeg_dimensions,
    make_rescue_keyframes,
    require_attr,
    require_module,
)


def _make_audit_frames(count: int = 6) -> list[Any]:
    gate_mod = importlib.import_module("teleop.copilot.gate") if _has_gate() else None
    raw = make_rescue_keyframes(count)
    if gate_mod is not None and hasattr(gate_mod, "KeyframeJpeg"):
        kf_cls = gate_mod.KeyframeJpeg
        return [kf_cls(label, sim_t, bx, tau, jpg) for label, sim_t, bx, tau, jpg in raw]
    return [jpg for _label, _sim_t, _bx, _tau, jpg in raw]


def _has_gate() -> bool:
    try:
        __import__("teleop.copilot.gate")
    except ModuleNotFoundError:
        return False
    return True


def _build_audit_reviewer(
    *,
    client: Any,
    on_update: Callable[[Any], None] | None = None,
    episode_sink: Any = None,
    max_reviews_per_hour: int = 20,
    speaker_sentinel: Any = None,
) -> Any:
    rev_mod = require_module("teleop.copilot.review")
    rev_cls = require_attr(rev_mod, "AuditReviewer")
    cfg = CopilotConfig(
        gemini_api_key="test-e2e-argon-key",
        max_reviews_per_hour=max_reviews_per_hour,
    )
    return construct_flexible(
        rev_cls,
        client=client,
        gemini_client=client,
        model_client=client,
        cfg=cfg,
        config=cfg,
        model=cfg.review_model,
        review_model=cfg.review_model,
        thinking_level=cfg.review_thinking,
        max_reviews_per_hour=max_reviews_per_hour,
        on_update=on_update,
        callback=on_update,
        sink=on_update,
        episode_sink=episode_sink,
        episodes=episode_sink,
        speaker=speaker_sentinel,
    )


def _submit_audit(
    reviewer: Any,
    *,
    bay: int = 1,
    skill_id: str = "skill_bay1_001",
    episode_uid: str | None = "ep-bay1-001",
    frame_count: int = 6,
) -> str:
    submit_fn = require_attr(reviewer, "submit")
    frames = _make_audit_frames(frame_count)
    jpeg_only = [f.jpeg_bytes if hasattr(f, "jpeg_bytes") else f for f in frames]
    timeline = {
        "bay": bay,
        "skill_id": skill_id,
        "bolt_start_m": 0.014,
        "bolt_final_m": 0.056,
        "peak_tau_hand_nm": 0.42,
        "duration_s": 2.6,
    }
    sig = inspect.signature(submit_fn)
    params = list(sig.parameters.keys())
    # Check if submit takes a single request dataclass
    if len(params) == 1:
        rev_mod = require_module("teleop.copilot.review")
        req_cls = (
            getattr(rev_mod, "AuditRequest", None)
            or getattr(rev_mod, "AuditInput", None)
            or getattr(rev_mod, "ReviewRequest", None)
        )
        if req_cls is not None:
            req = construct_flexible(
                req_cls,
                bay=bay,
                bay_id=bay,
                skill_id=skill_id,
                episode_uid=episode_uid,
                frames=frames,
                keyframes=frames,
                jpegs=jpeg_only,
                telemetry=timeline,
                timeline=timeline,
                summary=timeline,
            )
            return str(submit_fn(req))
    candidates: dict[str, Any] = {
        "bay": bay,
        "bay_id": bay,
        "skill_id": skill_id,
        "episode_uid": episode_uid,
        "frames": frames,
        "keyframes": frames,
        "jpegs": jpeg_only,
        "images": jpeg_only,
        "telemetry": timeline,
        "timeline": timeline,
        "summary": timeline,
    }
    kwargs: dict[str, Any] = {}
    pos: list[Any] = []
    for name, p in sig.parameters.items():
        if p.kind == inspect.Parameter.VAR_KEYWORD:
            continue
        if name in candidates:
            if p.kind == inspect.Parameter.POSITIONAL_ONLY:
                pos.append(candidates[name])
            else:
                kwargs[name] = candidates[name]
    res = submit_fn(*pos, **kwargs)
    return str(res)


class _ZeroRenderer:
    def __init__(self) -> None:
        self.closed = False

    def render(self, data: Any, bay_id: int) -> dict[str, np.ndarray]:
        del data, bay_id
        return {cam: np.zeros((*IMAGE_HW, 3), dtype=np.uint8) for cam in CAMERAS}

    def close(self) -> None:
        self.closed = True


class GatedEpisodeRecorder:
    """Wraps real EpisodeRecorder with threading.Events to deterministically test Race A & B."""

    def __init__(self, inner: EpisodeRecorder) -> None:
        self.inner = inner
        self.entered_end = threading.Event()
        self.allow_end = threading.Event()
        self.allow_end.set()

    def start_episode(self, task: str) -> None:
        self.inner.start_episode(task)

    def add_frame(self, state: Any, action: Any, images: Any) -> None:
        self.inner.add_frame(state, action, images)

    def end_episode(self, meta: Mapping[str, Any]) -> int:
        self.entered_end.set()
        assert self.allow_end.wait(timeout=5.0), "Timed out waiting on allow_end barrier"
        return self.inner.end_episode(meta)

    def discard_episode(self) -> None:
        self.inner.discard_episode()

    def close(self) -> None:
        self.inner.close()


def _drive_synthetic_episode(
    sink: EpisodeSink,
    rig: E2ECoordinatorRig,
    *,
    bay_id: int = 1,
    source: str = "operator",
    ticks: int = 56,
    success: bool = True,
) -> str:
    """Drive >=28 frames (0.93s > 0.5s min_len_s) with smooth wrist motion into EpisodeSink."""
    bay_rig = rig.co.rig(bay_id)
    base_pose = bay_rig.wrist_pose()
    assert base_pose.position is not None
    x0, y0, z0 = base_pose.position
    fingers = {
        k: 0.25
        for k in ("thumb_proximal_yaw", "thumb_proximal_pitch", "index", "middle", "ring", "pinky")
    }

    sink.on_episode_start(bay_id, source, "Jam_Sig_1", None)  # type: ignore[arg-type]
    for step in range(ticks):
        # Smooth +X motion of ~2mm per tick so atc_trim sees active motion (>1e-3 eps)
        wrist = Pose(
            orientation=base_pose.orientation,
            position=(x0 + 0.0012 * step, y0, z0),
        )
        sink.on_tick(bay_id, bay_rig, wrist, fingers)
    sink.on_episode_end(bay_id, success, None if success else "aborted")
    uid_fn = require_attr(sink, "latest_episode_uid")
    uid = uid_fn(bay_id)
    assert isinstance(uid, str) and len(uid) > 0
    return uid


# Multi-element Node.js Fake DOM matching Explorer R1-3 §4.2 Part A3
_NODE_MULTI_DOM_HARNESS = r"""
"use strict";
const fs = require("fs");

const byId = new Map();

class ClassList {
  constructor(el) { this.el = el; this.set = new Set(); }
  add(...cs) { for (const c of cs) if (c) this.set.add(c); }
  remove(...cs) { for (const c of cs) this.set.delete(c); }
  toggle(c, force) {
    const on = force === undefined ? !this.set.has(c) : !!force;
    if (on) this.set.add(c); else this.set.delete(c);
    return on;
  }
  contains(c) { return this.set.has(c); }
}

class El {
  constructor(tag, id = "") {
    this.tag = String(tag).toUpperCase();
    this.id = id;
    this.children = [];
    this.parent = null;
    this.dataset = {};
    this.listeners = {};
    this.hidden = false;
    this.disabled = false;
    this.title = "";
    this._text = "";
    this._html = "";
    this.builds = 0;
    this.mutations = 0;
    this.tabIndex = String(tag).toLowerCase() === "button" ? 0 : -1;
    this.style = {};
    this.attributes = {};
    this.classList = new ClassList(this);
    if (id) byId.set(id, this);
  }
  get className() { return [...this.classList.set].join(" "); }
  set className(v) { this.classList.set = new Set(String(v || "").split(/\s+/).filter(Boolean)); }
  get textContent() {
    if (this.children.length) return this.children.map((c) => c.textContent).join(" ").trim();
    return this._text;
  }
  set textContent(v) {
    this._text = String(v ?? "");
    this._html = "";
    this.children = [];
    this.mutations++;
  }
  get innerHTML() { return this._html || this.textContent; }
  set innerHTML(v) {
    this._html = String(v ?? "");
    this._text = this._html.replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim();
    this.children = [];
    this.builds++;
    if (this._html.includes('class="gate__conds"')) {
      const c = new El("span");
      c.className = "gate__conds";
      this.append(c);
    }
  }
  _adopt(k) {
    if (k === null || k === undefined) { const t = new El("#text"); return t; }
    if (typeof k === "string" || typeof k === "number") {
      const t = new El("#text");
      t._text = String(k);
      return t;
    }
    if (k.parent) k.remove();
    k.parent = this;
    if (k.id) byId.set(k.id, k);
    return k;
  }
  append(...ks) { for (const k of ks) this.children.push(this._adopt(k)); }
  appendChild(k) { const a = this._adopt(k); this.children.push(a); return a; }
  prepend(...ks) { for (const k of [...ks].reverse()) this.children.unshift(this._adopt(k)); }
  remove() {
    if (!this.parent) return;
    const idx = this.parent.children.indexOf(this);
    if (idx >= 0) this.parent.children.splice(idx, 1);
    this.parent = null;
  }
  after(k) {
    if (!this.parent) return;
    const p = this.parent;
    const a = p._adopt(k);
    p.children.splice(p.children.indexOf(this) + 1, 0, a);
  }
  replaceChildren(...ks) {
    this.builds++;
    for (const c of this.children) c.parent = null;
    this.children = [];
    this.append(...ks);
  }
  setAttribute(k, v) {
    this.attributes[k] = String(v);
    if (k === "id") { this.id = String(v); byId.set(this.id, this); }
    if (k === "tabindex") { this.tabIndex = Number(v); }
    if (k === "hidden") { this.hidden = true; }
  }
  removeAttribute(k) {
    delete this.attributes[k];
    if (k === "hidden") this.hidden = false;
  }
  getAttribute(k) {
    return Object.prototype.hasOwnProperty.call(this.attributes, k)
      ? this.attributes[k]
      : null;
  }
  addEventListener(t, f) { (this.listeners[t] = this.listeners[t] || []).push(f); }
  matches(sel) {
    if (!sel) return false;
    if (sel.startsWith("#")) return this.id === sel.slice(1);
    if (sel.startsWith(".")) return this.classList.contains(sel.slice(1));
    return this.tag.toLowerCase() === sel.toLowerCase();
  }
  querySelector(sel) {
    for (const c of this.children) {
      if (c.matches(sel)) return c;
      const found = c.querySelector(sel);
      if (found) return found;
    }
    return null;
  }
  querySelectorAll(sel) {
    const out = [];
    for (const c of this.children) {
      if (c.matches(sel)) out.push(c);
      out.push(...c.querySelectorAll(sel));
    }
    return out;
  }
  find(cls) { return this.querySelector("." + cls); }
  buttons() { return this.querySelectorAll("button"); }
}

const docListeners = {};
global.CustomEvent = class CustomEvent {
  constructor(type, init) { this.type = type; this.detail = init ? init.detail : undefined; }
};
global.WebSocket = { OPEN: 1 };
global.window = global;
global.document = {
  createElement: (tag) => new El(tag),
  createTextNode: (txt) => { const t = new El("#text"); t._text = String(txt); return t; },
  getElementById: (id) => byId.get(id) || null,
  querySelector: (sel) => sel.startsWith("#") ? (byId.get(sel.slice(1)) || null) : null,
  addEventListener: (t, f) => { (docListeners[t] = docListeners[t] || []).push(f); },
  dispatchEvent: (ev) => { for (const f of (docListeners[ev.type] || [])) f(ev); },
};
global.$ = (id) => document.getElementById(id);

const indexHtml = fs.readFileSync(process.env.WEB_DIR + "/index.html", "utf8");
for (const m of indexHtml.matchAll(/<([a-zA-Z0-9-]+)([^>]*\bid="([^"]+)"[^>]*)>/g)) {
  const [, tag, attrs, id] = m;
  const el = new El(tag, id);
  if (/\bhidden\b/.test(attrs)) el.hidden = true;
}
"""


def _run_node_dom_script(js_body: str) -> dict[str, Any]:
    env = dict(os.environ)
    env["WEB_DIR"] = str(WEB_DIR)
    proc = subprocess.run(
        ["/usr/bin/node", "-e", _NODE_MULTI_DOM_HARNESS + "\n" + js_body],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert proc.returncode == 0, (
        f"Node.js fake-DOM script failed (rc={proc.returncode}):\n"
        f"STDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout}"
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])


# ===========================================================================
# FEATURE 10 — Tier 3 `AuditReviewer` Client (`gemini-4-argon`)
# ===========================================================================


class TestFeature10Tier3ArgonAuditReviewerTier1:
    """Tier 1 category-partition tests for `AuditReviewer` (`gemini-4-argon`)."""

    def test_f10_t1_audit_report_dataclass_and_to_dict_match_project_contract(self) -> None:
        rev_mod = require_module("teleop.copilot.review")
        report_cls = require_attr(rev_mod, "AuditReport")
        rep = report_cls(
            audit_id="bay1-001",
            status="complete",
            bay=1,
            skill_id="skill_001",
            episode_uid="ep-001",
            verdict="approve",
            dataset_tag="approve",
            confidence=0.95,
            reasons=("Flap cleanly depressed",),
            operator_note="Ready for ACT dataset.",
            model="gemini-4-argon",
            frame_count=6,
            latency_ms=8120.5,
            wall="20:15:00",
        )
        as_dict = rep.to_dict()
        assert json.loads(json.dumps(as_dict, allow_nan=False))["audit_id"] == "bay1-001"
        for required_key in (
            "audit_id",
            "status",
            "bay",
            "skill_id",
            "episode_uid",
            "verdict",
            "dataset_tag",
            "confidence",
            "reasons",
            "operator_note",
            "model",
            "frame_count",
            "latency_ms",
            "wall",
        ):
            assert required_key in as_dict, f"AuditReport.to_dict() missing {required_key!r}"

    def test_f10_t1_submit_is_non_blocking_emits_pending_then_complete_approve(
        self,
        mock_gemini_server: LoopbackGeminiTLSServer,
    ) -> None:
        hold = threading.Event()
        mock_gemini_server.config.argon_hold_event = hold
        updates: list[Any] = []
        done = threading.Event()

        def on_upd(rep: Any) -> None:
            updates.append(rep)
            if rep.status == "complete":
                done.set()

        gem_client = GeminiClient("test-key", base_url=mock_gemini_server.base_url)
        reviewer = _build_audit_reviewer(client=gem_client, on_update=on_upd)
        try:
            t0 = time.monotonic()
            audit_id = _submit_audit(reviewer, frame_count=6)
            submit_elapsed = time.monotonic() - t0
            assert isinstance(audit_id, str) and audit_id
            assert submit_elapsed < 0.20, f"submit() blocked for {submit_elapsed:.3f}s!"
            assert any(u.status == "pending" for u in updates)

            hold.set()
            assert done.wait(timeout=5.0), "Timed out waiting for complete Argon AuditReport"
            final = updates[-1]
            assert final.status == "complete"
            assert final.verdict == "approve"
            assert final.dataset_tag == "approve"
            assert len(final.reasons) >= 1
            assert final.frame_count == 6
        finally:
            hold.set()
            if hasattr(reviewer, "close"):
                reviewer.close()

    def test_f10_t1_argon_request_uses_low_thinking_and_up_to_6_downscaled_jpegs(
        self,
        mock_gemini_server: LoopbackGeminiTLSServer,
    ) -> None:
        done = threading.Event()
        gem_client = GeminiClient("test-key", base_url=mock_gemini_server.base_url)
        reviewer = _build_audit_reviewer(
            client=gem_client,
            on_update=lambda r: done.set() if r.status == "complete" else None,
        )
        try:
            _ = _submit_audit(reviewer, frame_count=6)
            assert done.wait(timeout=5.0)
            reqs = mock_gemini_server.requests_matching("gemini-4-argon:generateContent")
            assert len(reqs) == 1
            body = reqs[0].json_body
            assert body is not None
            gen_cfg = body["generationConfig"]
            assert gen_cfg["thinkingConfig"]["thinkingLevel"].upper() == "LOW"
            if "responseSchema" in gen_cfg:
                assert not contains_key_recursive(gen_cfg["responseSchema"], "additionalProperties")
            parts = body["contents"][0]["parts"]
            imgs = [p["inlineData"] for p in parts if "inlineData" in p]
            assert 1 <= len(imgs) <= 6
            assert len(imgs) == 6
            for img in imgs:
                w, h = jpeg_dimensions(base64.b64decode(img["data"]))
                assert w <= 512 and h <= 512
        finally:
            if hasattr(reviewer, "close"):
                reviewer.close()

    def test_f10_t1_argon_flag_verdict_produces_flag_dataset_tag_and_reasons(
        self,
        mock_gemini_server: LoopbackGeminiTLSServer,
    ) -> None:
        mock_gemini_server.config.argon_report = {
            "verdict": "flag",
            "dataset_tag": "flag",
            "confidence": 0.89,
            "reasons": ["Thumb brushed within 8mm of -X HV terminal"],
            "operator_note": "Exclude episode from ACT training set.",
        }
        completed: list[Any] = []
        done = threading.Event()

        def _on_flag(r: Any) -> None:
            if r.status == "complete":
                completed.append(r)
                done.set()

        gem_client = GeminiClient("test-key", base_url=mock_gemini_server.base_url)
        reviewer = _build_audit_reviewer(
            client=gem_client,
            on_update=_on_flag,
        )
        try:
            _ = _submit_audit(reviewer, frame_count=4)
            assert done.wait(timeout=5.0)
            rep = completed[-1]
            assert rep.verdict == "flag"
            assert rep.dataset_tag == "flag"
            assert "HV" in rep.reasons[0]
        finally:
            if hasattr(reviewer, "close"):
                reviewer.close()

    def test_f10_t1_strictly_silent_never_invokes_tts_or_audio_subprocesses(
        self,
        mock_gemini_server: LoopbackGeminiTLSServer,
        fake_audio_env: FakeAudioHarness,
    ) -> None:
        class _ExplodingSpeaker:
            def say(self, *_args: Any, **_kwargs: Any) -> int:
                raise AssertionError("Tier 3 Argon MUST NEVER invoke Speaker.say / TTS!")

        done = threading.Event()
        gem_client = GeminiClient("test-key", base_url=mock_gemini_server.base_url)
        reviewer = _build_audit_reviewer(
            client=gem_client,
            on_update=lambda r: done.set() if r.status == "complete" else None,
            speaker_sentinel=_ExplodingSpeaker(),
        )
        try:
            _ = _submit_audit(reviewer, frame_count=6)
            assert done.wait(timeout=5.0)
            tts_reqs = [
                r
                for r in mock_gemini_server.requests
                if "tts" in r.path or "streamGenerateContent" in r.path
            ]
            assert tts_reqs == [], f"Argon triggered unexpected TTS HTTP calls: {tts_reqs}"
            assert fake_audio_env.calls() == [], (
                f"Argon triggered unexpected host audio subprocesses: {fake_audio_env.calls()}"
            )
        finally:
            if hasattr(reviewer, "close"):
                reviewer.close()


class TestFeature10Tier3ArgonAuditReviewerTier2:
    """Tier 2 rate-limit, >6 frame cap, HTTP 500, and malformed JSON tests for F10."""

    def test_f10_t2_exceeding_max_reviews_per_hour_emits_skipped_status_without_api_call(
        self,
        mock_gemini_server: LoopbackGeminiTLSServer,
    ) -> None:
        updates: list[Any] = []
        first_done = threading.Event()

        def _on_rate(r: Any) -> None:
            updates.append(r)
            if r.status == "complete":
                first_done.set()

        gem_client = GeminiClient("test-key", base_url=mock_gemini_server.base_url)
        reviewer = _build_audit_reviewer(
            client=gem_client,
            max_reviews_per_hour=1,
            on_update=_on_rate,
        )
        try:
            _ = _submit_audit(reviewer, skill_id="skill_1", episode_uid="ep-1")
            assert first_done.wait(timeout=5.0)

            _ = _submit_audit(reviewer, skill_id="skill_2", episode_uid="ep-2")
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline and not any(u.status == "skipped" for u in updates):
                time.sleep(0.02)
            assert any(u.status == "skipped" for u in updates), (
                f"Expected `skipped` AuditReport once hourly budget=1 exhausted; got {updates}"
            )
            assert len(mock_gemini_server.requests_matching("gemini-4-argon")) == 1
        finally:
            if hasattr(reviewer, "close"):
                reviewer.close()

    def test_f10_t2_more_than_6_input_frames_subsampled_to_at_most_6_on_wire(
        self,
        mock_gemini_server: LoopbackGeminiTLSServer,
    ) -> None:
        done = threading.Event()
        gem_client = GeminiClient("test-key", base_url=mock_gemini_server.base_url)
        reviewer = _build_audit_reviewer(
            client=gem_client,
            on_update=lambda r: done.set() if r.status == "complete" else None,
        )
        try:
            many_frames = _make_audit_frames(6) + _make_audit_frames(4)  # 10 frames total
            submit_fn = require_attr(reviewer, "submit")
            sig = inspect.signature(submit_fn)
            if "frames" in sig.parameters:
                _ = submit_fn(
                    bay=1, skill_id="s1", episode_uid="e1", frames=many_frames, telemetry={}
                )
            else:
                _ = _submit_audit(reviewer, frame_count=6)
            assert done.wait(timeout=5.0)
            req = mock_gemini_server.requests_matching("gemini-4-argon")[-1]
            imgs = [p for p in req.json_body["contents"][0]["parts"] if "inlineData" in p]  # type: ignore[index]
            assert len(imgs) <= 6
        finally:
            if hasattr(reviewer, "close"):
                reviewer.close()

    def test_f10_t2_argon_http_500_transitions_report_to_error_status_cleanly(
        self,
        mock_gemini_server: LoopbackGeminiTLSServer,
    ) -> None:
        mock_gemini_server.config.argon_status_code = 500
        updates: list[Any] = []
        done = threading.Event()

        def _on_err500(r: Any) -> None:
            updates.append(r)
            if r.status == "error":
                done.set()

        gem_client = GeminiClient("test-key", base_url=mock_gemini_server.base_url)
        reviewer = _build_audit_reviewer(
            client=gem_client,
            on_update=_on_err500,
        )
        try:
            _ = _submit_audit(reviewer)
            assert done.wait(timeout=5.0), f"Expected `error` status on HTTP 500; got {updates}"
            assert updates[-1].status == "error"
            assert updates[-1].dataset_tag is None
        finally:
            if hasattr(reviewer, "close"):
                reviewer.close()

    def test_f10_t2_argon_malformed_json_transitions_report_to_error_status_cleanly(
        self,
        mock_gemini_server: LoopbackGeminiTLSServer,
    ) -> None:
        mock_gemini_server.config.argon_malformed_text = "not valid json {{{"
        updates: list[Any] = []
        done = threading.Event()

        def _on_err500(r: Any) -> None:
            updates.append(r)
            if r.status == "error":
                done.set()

        gem_client = GeminiClient("test-key", base_url=mock_gemini_server.base_url)
        reviewer = _build_audit_reviewer(
            client=gem_client,
            on_update=_on_err500,
        )
        try:
            _ = _submit_audit(reviewer)
            assert done.wait(timeout=5.0)
            assert updates[-1].status == "error"
        finally:
            if hasattr(reviewer, "close"):
                reviewer.close()

    def test_f10_t2_offline_no_client_skips_cleanly_without_raising(self) -> None:
        updates: list[Any] = []
        reviewer = _build_audit_reviewer(client=None, on_update=updates.append)
        try:
            _ = _submit_audit(reviewer)
            time.sleep(0.05)
            assert all(u.status in ("skipped", "pending") for u in updates)
        finally:
            if hasattr(reviewer, "close"):
                reviewer.close()


# ===========================================================================
# FEATURE 13 — Fleet Ops Console UI Audit Card & Gate Status
# ===========================================================================


class TestFeature13FleetOpsConsoleAuditCardTier1:
    """Tier 1 HTTP, WebSocket, TelemetryAdapter, and Node.js DOM tests for F13."""

    def test_f13_t1_web_assets_include_audit_js_and_audit_panel_in_index_html(self) -> None:
        audit_js = WEB_DIR / "audit.js"
        assert audit_js.is_file(), f"Required M4 frontend module missing: {audit_js}"
        index_html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
        assert 'id="audit-panel"' in index_html
        assert 'src="audit.js"' in index_html

    def test_f13_t1_console_server_serves_audit_js_and_core_assets_over_http(self) -> None:
        assert (WEB_DIR / "audit.js").is_file(), (
            f"Required M4 frontend module missing: {WEB_DIR / 'audit.js'}"
        )
        srv = ConsoleServer(port=0)
        srv.start()
        try:
            base = srv.url.rstrip("/")
            for path, expected_ct in (
                ("/", "text/html"),
                ("/audit.js", "javascript"),
                ("/console.js", "javascript"),
                ("/console.css", "text/css"),
            ):
                with urllib.request.urlopen(f"{base}{path}", timeout=5.0) as resp:
                    body = resp.read()
                    assert resp.status == 200
                    assert expected_ct in resp.headers.get("Content-Type", "")
                    assert resp.headers.get("Cache-Control") == "no-store"
                    assert len(body) > 20
        finally:
            srv.stop()

    def test_f13_t1_ws_broadcasts_audit_payload_and_replays_to_late_reconnecting_browser(
        self,
    ) -> None:
        assert (WEB_DIR / "audit.js").is_file(), "M4 console audit support not implemented yet"
        srv = ConsoleServer(port=0)
        srv.start()
        audit_payload = {
            "audit_id": "bay1-1",
            "status": "complete",
            "bay": 1,
            "skill_id": "skill_001",
            "episode_uid": "ep-1",
            "verdict": "approve",
            "dataset_tag": "approve",
            "confidence": 0.94,
            "reasons": ["Flap depressed cleanly"],
            "operator_note": "Approved for ACT.",
            "model": "gemini-4-argon",
            "frame_count": 6,
            "latency_ms": 8100.0,
            "wall": "19:30:00",
        }
        try:
            ws_url = f"ws://127.0.0.1:{srv.port}/ws"
            with ws_connect(ws_url) as ws1:
                srv.publish_telemetry({"type": "telemetry", "layout_id": 1, "audit": audit_payload})
                live_msg = json.loads(ws1.recv(timeout=5.0))
                assert live_msg["audit"] == audit_payload

            # Late browser connects AFTER publish_telemetry -> must immediately receive replay!
            with ws_connect(ws_url) as ws2:
                replayed = json.loads(ws2.recv(timeout=5.0))
                assert replayed["audit"] == audit_payload
        finally:
            srv.stop()

    def test_f13_t1_standalone_copilot_audit_ws_ingress_updates_connected_and_late_browsers(
        self,
    ) -> None:
        srv = ConsoleServer(port=0)
        srv.start()
        audit_payload = {
            "audit_id": "standalone-01",
            "status": "complete",
            "bay": 1,
            "skill_id": "skill_002",
            "episode_uid": "ep-2",
            "verdict": "flag",
            "dataset_tag": "flag",
            "confidence": 0.88,
            "reasons": ["Thumb brushed near -X HV bus"],
            "operator_note": "Flagged for exclusion.",
            "model": "gemini-4-argon",
            "frame_count": 5,
            "latency_ms": 9200.0,
            "wall": "19:35:00",
        }
        try:
            ws_url = f"ws://127.0.0.1:{srv.port}/ws"
            srv.publish_telemetry({"type": "telemetry", "layout_id": 1, "audit": None})
            with ws_connect(ws_url) as ws_browser, ws_connect(ws_url) as ws_copilot:
                _ = ws_browser.recv(timeout=5.0)  # drain initial seed replay
                _ = ws_copilot.recv(timeout=5.0)

                ws_copilot.send(
                    json.dumps({"type": "copilot_audit", "audit": audit_payload, **audit_payload})
                )
                try:
                    raw = ws_browser.recv(timeout=2.0)
                except TimeoutError as exc:
                    raise AssertionError(
                        "ConsoleServer ignored `copilot_audit` WS message instead of broadcasting"
                    ) from exc
                pushed = json.loads(raw)
                extracted = pushed.get("audit") or pushed
                assert extracted.get("verdict") == "flag"
                assert extracted.get("audit_id") == "standalone-01"

            with ws_connect(ws_url) as ws_late:
                late_msg = json.loads(ws_late.recv(timeout=5.0))
                late_audit = late_msg.get("audit") or late_msg
                assert late_audit.get("audit_id") == "standalone-01"
        finally:
            srv.stop()

    def test_f13_t1_nodejs_fake_dom_renders_audit_js_lifecycle_and_gate_vlm_badge(self) -> None:
        assert (WEB_DIR / "audit.js").is_file(), "src/teleop/console/web/audit.js missing"
        script = r"""
const auditSrc = fs.readFileSync(process.env.WEB_DIR + "/audit.js", "utf8");
const consoleSrc = fs.readFileSync(process.env.WEB_DIR + "/console.js", "utf8");
const mGate = consoleSrc.match(/function updateGate\(t\)\s*\{[\s\S]*?\n\}/);
if (!mGate) throw new Error("updateGate(t) not found in console.js");
const updateGate = new Function("t", mGate[0] + "\nreturn updateGate(t);");

eval(auditSrc);
const panel = document.getElementById("audit-panel");
if (!panel) throw new Error("#audit-panel missing in index.html");

// 1. Null audit -> hidden
document.dispatchEvent(new CustomEvent("telemetry", { detail: { audit: null } }));
const hiddenOnNull = panel.hidden;

// 2. Pending state + repeated 10Hz ticks must update in-place without rebuilding DOM
const pending = {
  audit_id: "a1", status: "pending", bay: 1, skill_id: "skill_01", episode_uid: "ep-1",
  verdict: null, dataset_tag: null, confidence: null, reasons: [], operator_note: "",
  model: "gemini-4-argon", frame_count: 6, latency_ms: null, wall: "19:00:01"
};
document.dispatchEvent(new CustomEvent("telemetry", { detail: { audit: pending } }));
const visibleOnPending = !panel.hidden;
const buildsAfterFirstPending = panel.builds;
document.dispatchEvent(new CustomEvent("telemetry", { detail: { audit: pending } }));
document.dispatchEvent(new CustomEvent("telemetry", { detail: { audit: pending } }));
const buildsAfterRepeatPending = panel.builds;

// 3. Complete approve
const approved = {
  ...pending, status: "complete", verdict: "approve", dataset_tag: "approve",
  confidence: 0.95, reasons: ["Flap cleanly depressed"], operator_note: "Approved for ACT.",
  latency_ms: 8200
};
document.dispatchEvent(new CustomEvent("telemetry", { detail: { audit: approved } }));
const approveText = panel.textContent;

// 4. Complete flag
const flagged = {
  ...pending, audit_id: "a2", status: "complete", verdict: "flag", dataset_tag: "flag",
  confidence: 0.90, reasons: ["Near -X HV terminal"], operator_note: "Exclude from ACT.",
  latency_ms: 9100
};
document.dispatchEvent(new CustomEvent("telemetry", { detail: { audit: flagged } }));
const flagText = panel.textContent;

// 5. Update #gate with VLM fields
updateGate({
  gate: {
    fit_ms: 14, passed: 20, total: 20, elapsed_s: 3.4, ok: true,
    conditions: "nominal", worst_failure: null,
    vlm_verdict: "approve", vlm_reason: "Clean bolt hook", vlm_model: "gemini-4-hydrogen"
  }
});
const gateText = document.getElementById("gate").textContent;

console.log(JSON.stringify({
  hiddenOnNull,
  visibleOnPending,
  buildsAfterFirstPending,
  buildsAfterRepeatPending,
  approveText,
  flagText,
  gateText,
}));
"""
        out = _run_node_dom_script(script)
        assert out["hiddenOnNull"] is True
        assert out["visibleOnPending"] is True
        assert out["buildsAfterRepeatPending"] == out["buildsAfterFirstPending"], (
            "audit.js rebuilt DOM on identical 10 Hz pending telemetry!"
        )
        assert re.search(r"approve|cleanly depressed|ACT", out["approveText"], re.IGNORECASE)
        assert re.search(r"flag|HV terminal|Exclude", out["flagText"], re.IGNORECASE)
        assert re.search(r"hydrogen|vlm|approve|Clean bolt", out["gateText"], re.IGNORECASE)


class TestFeature13FleetOpsConsoleAuditCardTier2:
    """Tier 2 zero-external-URL, button focus, NaN scrubbing, and error/skipped UI tests for F13."""

    def test_f13_t2_zero_external_http_or_https_urls_across_all_web_dir_files(self) -> None:
        assert (WEB_DIR / "audit.js").is_file(), "audit.js must exist in WEB_DIR for M4 check"
        text_suffixes = {".html", ".css", ".js", ".svg"}
        checked = 0
        for path in sorted(WEB_DIR.iterdir()):
            if path.is_file() and path.suffix.lower() in text_suffixes:
                checked += 1
                raw = path.read_text(encoding="utf-8")
                assert not re.search(r"https?://", raw, re.IGNORECASE), (
                    f"Forbidden `http://` or `https://` string found in {path.name}"
                )
        assert checked >= 5

    def test_f13_t2_audit_js_buttons_prevent_mousedown_focus_and_set_tabindex_minus_one(
        self,
    ) -> None:
        assert (WEB_DIR / "audit.js").is_file(), "src/teleop/console/web/audit.js missing"
        script = r"""
const auditSrc = fs.readFileSync(process.env.WEB_DIR + "/audit.js", "utf8");
eval(auditSrc);
const panel = document.getElementById("audit-panel");
document.dispatchEvent(new CustomEvent("telemetry", {
  detail: {
    audit: {
      audit_id: "b1", status: "complete", bay: 1, skill_id: "s1", episode_uid: "e1",
      verdict: "flag", dataset_tag: "flag", confidence: 0.85,
      reasons: ["Spike"], operator_note: "Note", model: "gemini-4-argon",
      frame_count: 6, latency_ms: 8000, wall: "19:00:00"
    }
  }
}));
const btns = panel.buttons();
let allOk = true;
for (const b of btns) {
  let prevented = false;
  for (const fn of (b.listeners["mousedown"] || [])) {
    fn({ preventDefault: () => { prevented = true; } });
  }
  if (!prevented && b.tabIndex !== -1) allOk = false;
}
console.log(JSON.stringify({ buttonCount: btns.length, allOk }));
"""
        out = _run_node_dom_script(script)
        assert out["allOk"] is True

    def test_f13_t2_nodejs_fake_dom_handles_error_and_skipped_audit_states_gracefully(
        self,
    ) -> None:
        assert (WEB_DIR / "audit.js").is_file(), "src/teleop/console/web/audit.js missing"
        script = r"""
const auditSrc = fs.readFileSync(process.env.WEB_DIR + "/audit.js", "utf8");
eval(auditSrc);
const panel = document.getElementById("audit-panel");
document.dispatchEvent(new CustomEvent("telemetry", {
  detail: {
    audit: {
      audit_id: "err1", status: "error", bay: 1, skill_id: "s1", episode_uid: null,
      verdict: null, dataset_tag: null, confidence: null, reasons: [],
      operator_note: "Argon request timed out", model: "gemini-4-argon",
      frame_count: 6, latency_ms: null, wall: "19:01:00"
    }
  }
}));
const errText = panel.textContent;
document.dispatchEvent(new CustomEvent("telemetry", {
  detail: {
    audit: {
      audit_id: "sk1", status: "skipped", bay: 1, skill_id: "s2", episode_uid: null,
      verdict: null, dataset_tag: null, confidence: null, reasons: [],
      operator_note: "Hourly review budget reached", model: "gemini-4-argon",
      frame_count: 0, latency_ms: null, wall: "19:02:00"
    }
  }
}));
const skipText = panel.textContent;
console.log(JSON.stringify({ errText, skipText, hidden: panel.hidden }));
"""
        out = _run_node_dom_script(script)
        assert out["hidden"] is False
        assert re.search(r"error|timed out|fail", out["errText"], re.IGNORECASE)
        assert re.search(r"skip|budget", out["skipText"], re.IGNORECASE)

    def test_f13_t2_telemetry_adapter_scrubs_nan_and_serializes_gate_vlm_fields(self) -> None:
        gate_fields = {f.name for f in GateStatus.__dataclass_fields__.values()}
        assert any("vlm" in name or "hydrogen" in name for name in gate_fields), (
            f"GateStatus missing M4 Hydrogen VLM fields; current fields={sorted(gate_fields)}"
        )
        kw: dict[str, Any] = {
            "fit_ms": 12.0,
            "passed": 20,
            "total": 20,
            "elapsed_s": float("nan"),
            "ok": True,
        }
        for k in gate_fields:
            if "verdict" in k:
                kw[k] = "approve"
            elif "latency" in k:
                kw[k] = float("inf")
        gs = GateStatus(**kw)
        view = HudView(
            bays=(),
            queue=(),
            background_bays=(),
            selected_bay=1,
            rescue=RescueStatus(None, False, None),
            gate=gs,
            compliance="ok",
            reasoning="idle",
            beat="",
            unoq="mock",
            actions=(),
            faults=(),
            conveyors=(),
            powered_off=frozenset(),
        )
        adapter = TelemetryAdapter(bays_present=(1, 2, 3, 4), background=(3, 4))
        msg = adapter.build(view, 1.0, (), {})
        # Must serialize strictly with allow_nan=False
        dumped = json.dumps(msg, allow_nan=False)
        assert "approve" in dumped

    def test_f13_t2_malformed_copilot_audit_ws_frames_do_not_crash_server_or_fire_on_key(
        self,
    ) -> None:
        assert (WEB_DIR / "audit.js").is_file(), "M4 console audit support not implemented yet"
        keys_fired: list[int] = []
        srv = ConsoleServer(port=0, on_key=keys_fired.append)
        srv.start()
        try:
            ws_url = f"ws://127.0.0.1:{srv.port}/ws"
            with ws_connect(ws_url) as ws:
                ws.send("not json at all {{{")
                ws.send(json.dumps({"type": "copilot_audit", "audit": "not-a-dict"}))
                ws.send(json.dumps(["array-instead-of-object"]))
                time.sleep(0.05)
            # Server remains healthy and serves "/"
            with urllib.request.urlopen(srv.url, timeout=5.0) as resp:
                assert resp.status == 200
            assert keys_fired == []
        finally:
            srv.stop()


# ===========================================================================
# FEATURE 14 — Async Episode Metadata Tagging & ACT QC Filtering
# ===========================================================================


class TestFeature14EpisodeTaggingAndAtcQcFilteringTier1:
    """Tier 1 tests for `EpisodeSink.tag_episode` (Race A & B) + `run_atc_qc` + `process`."""

    def test_f14_t1_race_a_pre_encode_tag_episode_merges_into_single_jsonl_row(
        self,
        tmp_path: Path,
        coordinator_factory: Callable[..., E2ECoordinatorRig],
    ) -> None:
        rig = coordinator_factory(grasp_gate=None, require_grasp_gate_hook=False)
        ds_root = tmp_path / "ds_race_a"
        rec = EpisodeRecorder("local/race_a", ds_root, FPS, lerobot_features(), ROBOT_TYPE)
        gated = GatedEpisodeRecorder(rec)
        gated.allow_end.clear()

        sink = EpisodeSink(
            ds_root,
            rig.co.model,
            rig.co.data,
            fault_of=lambda _: "flap",
            operator_source="glove",
            renderer=_ZeroRenderer(),
            recorder=gated,
        )
        try:
            uid = _drive_synthetic_episode(sink, rig, bay_id=1)
            assert gated.entered_end.wait(timeout=5.0)
            # Sidecar file has NOT been written yet while gated inside end_episode!
            assert not (ds_root / "meta" / "teleop_episodes.jsonl").exists()

            tag_fn = require_attr(sink, "tag_episode")
            ok = tag_fn(
                uid,
                dataset_tag="flag",
                audit={"audit_id": "race-a-1", "verdict": "flag", "reasons": ["HV proximity"]},
            )
            assert ok is True
            gated.allow_end.set()
        finally:
            gated.allow_end.set()
            sink.close()

        lines = (
            (ds_root / "meta" / "teleop_episodes.jsonl").read_text(encoding="utf-8").splitlines()
        )
        assert len(lines) == 1, f"Expected strictly 1 JSONL row, got {len(lines)}"
        row = json.loads(lines[0])
        assert row["episode_index"] == 0
        assert row["dataset_tag"] == "flag"
        assert row["success"] is True

    def test_f14_t1_race_b_post_encode_tag_episode_atomically_rewrites_sidecar_under_lock(
        self,
        tmp_path: Path,
        coordinator_factory: Callable[..., E2ECoordinatorRig],
    ) -> None:
        rig = coordinator_factory(grasp_gate=None, require_grasp_gate_hook=False)
        ds_root = tmp_path / "ds_race_b"
        sink = EpisodeSink(
            ds_root,
            rig.co.model,
            rig.co.data,
            fault_of=lambda _: "flap",
            operator_source="glove",
            renderer=_ZeroRenderer(),
        )
        try:
            uid0 = _drive_synthetic_episode(sink, rig, bay_id=1)
            uid1 = _drive_synthetic_episode(sink, rig, bay_id=2)
            uid2 = _drive_synthetic_episode(sink, rig, bay_id=1)
            assert len({uid0, uid1, uid2}) == 3

            deadline = time.monotonic() + 5.0
            while sink.written < 3 and time.monotonic() < deadline:
                time.sleep(0.02)
            assert sink.written == 3

            # All 3 rows are already flushed on disk; now tag ep0 & ep1 post-encode!
            tag_fn = require_attr(sink, "tag_episode")
            assert tag_fn(uid0, dataset_tag="flag", audit={"audit_id": "b0", "verdict": "flag"})
            assert tag_fn(
                uid1, dataset_tag="approve", audit={"audit_id": "b1", "verdict": "approve"}
            )
        finally:
            sink.close()

        raw_lines = (
            (ds_root / "meta" / "teleop_episodes.jsonl").read_text(encoding="utf-8").splitlines()
        )
        assert len(raw_lines) == 3, "Atomic rewrite must preserve 1 line == 1 episode"
        sidecar = read_sidecar(ds_root)
        assert sidecar[0]["dataset_tag"] == "flag"
        assert sidecar[1]["dataset_tag"] == "approve"
        assert sidecar[2].get("dataset_tag") is None

    def test_f14_t1_drop_reasons_contains_copilot_flagged_and_run_atc_qc_filters_correctly(
        self,
        tmp_path: Path,
        coordinator_factory: Callable[..., E2ECoordinatorRig],
    ) -> None:
        assert "copilot flagged" in DROP_REASONS, (
            f"'copilot flagged' missing from DROP_REASONS={DROP_REASONS}"
        )
        rig = coordinator_factory(grasp_gate=None, require_grasp_gate_hook=False)
        ds_root = tmp_path / "ds_qc"
        sink = EpisodeSink(
            ds_root,
            rig.co.model,
            rig.co.data,
            fault_of=lambda _: "flap",
            operator_source="glove",
            renderer=_ZeroRenderer(),
        )
        uid_flag = _drive_synthetic_episode(sink, rig, bay_id=1)
        uid_ok = _drive_synthetic_episode(sink, rig, bay_id=2)
        _uid_untagged = _drive_synthetic_episode(sink, rig, bay_id=1)
        sink.close()

        tag_fn = require_attr(sink, "tag_episode")
        assert tag_fn(uid_flag, dataset_tag="flag", audit={"audit_id": "qc-flag"})
        assert tag_fn(uid_ok, dataset_tag="approve", audit={"audit_id": "qc-ok"})

        results = run_atc_qc(ds_root, AtcQCThresholds(), DEFAULT_SOURCES, ranges={})
        assert len(results) == 3
        assert results[0].keep is False
        assert "copilot flagged" in results[0].reasons
        assert results[1].keep is True
        assert results[1].reasons == []
        assert results[2].keep is True
        assert results[2].reasons == []

    def test_f14_t1_write_clean_excludes_flagged_and_preserves_approved_audit_metadata(
        self,
        tmp_path: Path,
        coordinator_factory: Callable[..., E2ECoordinatorRig],
    ) -> None:
        assert "copilot flagged" in DROP_REASONS
        rig = coordinator_factory(grasp_gate=None, require_grasp_gate_hook=False)
        ds_root = tmp_path / "ds_in"
        clean_root = tmp_path / "ds_clean"
        sink = EpisodeSink(
            ds_root,
            rig.co.model,
            rig.co.data,
            fault_of=lambda _: "flap",
            operator_source="glove",
            renderer=_ZeroRenderer(),
        )
        uid0 = _drive_synthetic_episode(sink, rig, bay_id=1)
        uid1 = _drive_synthetic_episode(sink, rig, bay_id=2)
        sink.close()

        tag_fn = require_attr(sink, "tag_episode")
        assert tag_fn(uid0, dataset_tag="flag", audit={"audit_id": "drop-me", "verdict": "flag"})
        assert tag_fn(
            uid1, dataset_tag="approve", audit={"audit_id": "keep-me", "verdict": "approve"}
        )

        results = run_atc_qc(ds_root, AtcQCThresholds(), DEFAULT_SOURCES, ranges={})
        write_clean(ds_root, results, clean_root)

        clean_sidecar = read_sidecar(clean_root)
        assert list(clean_sidecar.keys()) == [0]
        assert clean_sidecar[0]["dataset_tag"] == "approve"
        assert clean_sidecar[0]["source_episode_index"] == 1
        qc_report = json.loads((clean_root / "qc_report.json").read_text(encoding="utf-8"))
        assert qc_report["totals"]["kept"] == 1
        assert qc_report["totals"]["rejected"] == 1
        assert "copilot flagged" in qc_report["episodes"][0]["reasons"]

    def test_f14_t1_teleop_process_cli_main_curates_flagged_vs_approved_end_to_end(
        self,
        tmp_path: Path,
        coordinator_factory: Callable[..., E2ECoordinatorRig],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        assert "copilot flagged" in DROP_REASONS
        rig = coordinator_factory(grasp_gate=None, require_grasp_gate_hook=False)
        ds_root = tmp_path / "cli_in"
        clean_root = tmp_path / "cli_out"
        sink = EpisodeSink(
            ds_root,
            rig.co.model,
            rig.co.data,
            fault_of=lambda _: "flap",
            operator_source="glove",
            renderer=_ZeroRenderer(),
        )
        uid0 = _drive_synthetic_episode(sink, rig, bay_id=1)
        uid1 = _drive_synthetic_episode(sink, rig, bay_id=2)
        _uid2 = _drive_synthetic_episode(sink, rig, bay_id=1)
        sink.close()

        tag_fn = require_attr(sink, "tag_episode")
        assert tag_fn(uid0, dataset_tag="flag", audit={"audit_id": "cli-flag"})
        assert tag_fn(uid1, dataset_tag="approve", audit={"audit_id": "cli-approve"})

        rc = process_cmd.main(["--in", str(ds_root), "--out", str(clean_root)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "kept 2/3 ATC episodes" in out


class TestFeature14EpisodeTaggingAndAtcQcFilteringTier2:
    """Tier 2 concurrency, unknown UID, post-close tag, and multi-reason drop tests for F14."""

    def test_f14_t2_unknown_episode_uid_returns_false_without_corrupting_sidecar(
        self,
        tmp_path: Path,
        coordinator_factory: Callable[..., E2ECoordinatorRig],
    ) -> None:
        rig = coordinator_factory(grasp_gate=None, require_grasp_gate_hook=False)
        ds_root = tmp_path / "ds_unknown"
        sink = EpisodeSink(
            ds_root,
            rig.co.model,
            rig.co.data,
            fault_of=lambda _: "flap",
            operator_source="glove",
            renderer=_ZeroRenderer(),
        )
        _ = _drive_synthetic_episode(sink, rig, bay_id=1)
        sink.close()
        before = (ds_root / "meta" / "teleop_episodes.jsonl").read_text(encoding="utf-8")

        tag_fn = require_attr(sink, "tag_episode")
        ok = tag_fn("ep-nonexistent-99999", dataset_tag="flag", audit={"audit_id": "ghost"})
        assert ok is False
        after = (ds_root / "meta" / "teleop_episodes.jsonl").read_text(encoding="utf-8")
        assert after == before

    def test_f14_t2_concurrent_threads_tagging_distinct_episodes_preserve_all_tags(
        self,
        tmp_path: Path,
        coordinator_factory: Callable[..., E2ECoordinatorRig],
    ) -> None:
        rig = coordinator_factory(grasp_gate=None, require_grasp_gate_hook=False)
        ds_root = tmp_path / "ds_concurrent"
        sink = EpisodeSink(
            ds_root,
            rig.co.model,
            rig.co.data,
            fault_of=lambda _: "flap",
            operator_source="glove",
            renderer=_ZeroRenderer(),
        )
        uids = [_drive_synthetic_episode(sink, rig, bay_id=(i % 2) + 1) for i in range(4)]
        sink.close()

        tag_fn = require_attr(sink, "tag_episode")
        barrier = threading.Barrier(4, timeout=5.0)
        outcomes: list[bool] = []
        lock = threading.Lock()

        def _worker(idx: int) -> None:
            barrier.wait()
            tag = "flag" if idx % 2 == 0 else "approve"
            res = tag_fn(uids[idx], dataset_tag=tag, audit={"audit_id": f"c-{idx}"})
            with lock:
                outcomes.append(bool(res))

        threads = [threading.Thread(target=_worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        assert outcomes == [True, True, True, True]
        lines = (
            (ds_root / "meta" / "teleop_episodes.jsonl").read_text(encoding="utf-8").splitlines()
        )
        assert len(lines) == 4
        sidecar = read_sidecar(ds_root)
        assert [sidecar[i]["dataset_tag"] for i in range(4)] == [
            "flag",
            "approve",
            "flag",
            "approve",
        ]

    def test_f14_t2_latest_episode_uid_returns_none_when_bay_has_no_recorded_episode(
        self,
        tmp_path: Path,
        coordinator_factory: Callable[..., E2ECoordinatorRig],
    ) -> None:
        rig = coordinator_factory(grasp_gate=None, require_grasp_gate_hook=False)
        sink = EpisodeSink(
            tmp_path / "ds_empty_uid",
            rig.co.model,
            rig.co.data,
            fault_of=lambda _: "flap",
            operator_source="glove",
            renderer=_ZeroRenderer(),
        )
        try:
            uid_fn = require_attr(sink, "latest_episode_uid")
            assert uid_fn(4) is None
        finally:
            sink.close()

    def test_f14_t2_all_episodes_flagged_writes_empty_clean_dataset_with_explicit_reason(
        self,
        tmp_path: Path,
        coordinator_factory: Callable[..., E2ECoordinatorRig],
    ) -> None:
        assert "copilot flagged" in DROP_REASONS
        rig = coordinator_factory(grasp_gate=None, require_grasp_gate_hook=False)
        ds_root = tmp_path / "ds_all_flagged"
        clean_root = tmp_path / "ds_all_flagged_clean"
        sink = EpisodeSink(
            ds_root,
            rig.co.model,
            rig.co.data,
            fault_of=lambda _: "flap",
            operator_source="glove",
            renderer=_ZeroRenderer(),
        )
        u0 = _drive_synthetic_episode(sink, rig, bay_id=1)
        u1 = _drive_synthetic_episode(sink, rig, bay_id=2)
        sink.close()

        tag_fn = require_attr(sink, "tag_episode")
        assert tag_fn(u0, dataset_tag="flag", audit={"audit_id": "f0"})
        assert tag_fn(u1, dataset_tag="flag", audit={"audit_id": "f1"})

        results = run_atc_qc(ds_root, AtcQCThresholds(), DEFAULT_SOURCES, ranges={})
        write_clean(ds_root, results, clean_root)
        report = json.loads((clean_root / "qc_report.json").read_text(encoding="utf-8"))
        assert report["totals"]["kept"] == 0
        assert report["totals"]["rejected"] == 2
        assert "rejected by QC" in (report["empty_reason"] or "")

    def test_f14_t2_failed_rescue_also_flagged_accumulates_both_drop_reasons(
        self,
        tmp_path: Path,
        coordinator_factory: Callable[..., E2ECoordinatorRig],
    ) -> None:
        assert "copilot flagged" in DROP_REASONS
        rig = coordinator_factory(grasp_gate=None, require_grasp_gate_hook=False)
        ds_root = tmp_path / "ds_multi_reason"
        sink = EpisodeSink(
            ds_root,
            rig.co.model,
            rig.co.data,
            fault_of=lambda _: "flap",
            operator_source="glove",
            renderer=_ZeroRenderer(),
        )
        uid = _drive_synthetic_episode(sink, rig, bay_id=1, success=False)
        sink.close()

        tag_fn = require_attr(sink, "tag_episode")
        assert tag_fn(uid, dataset_tag="flag", audit={"audit_id": "both-bad"})
        results = run_atc_qc(ds_root, AtcQCThresholds(), DEFAULT_SOURCES, ranges={})
        assert results[0].keep is False
        assert "not successful" in results[0].reasons
        assert "copilot flagged" in results[0].reasons
