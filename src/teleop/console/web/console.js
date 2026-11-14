// Fleet operations console. Display only: the sim pushes telemetry (JSON) and camera frames
// (binary: byte 0 = feed slot, byte 1 = layout_id % 256, then a JPEG) over "/ws". The page
// sends back key presses as GLFW key codes; the glove and the hardware console stay the real
// controls. All state comes from the latest telemetry, so a reload restores the full view.
"use strict";

const WS_PATH = "/ws";
const STALE_MS = 2000;
const BAYS = [1, 2, 3, 4];

const STATE_TEXT = {
  NOMINAL: "Nominal",
  STALLED: "Stalled",
  QUEUED: "Queued",
  RESCUE: "Under rescue",
  VALIDATING: "Validating",
  AUTONOMOUS_SKILL: "Auto skill",
  OOD_ESCALATED: "Refused",
  STANDBY: "Standby",
};
const STATE_COLOR = {
  NOMINAL: "--nominal",
  STALLED: "--stalled",
  QUEUED: "--queued",
  RESCUE: "--rescue",
  VALIDATING: "--validating",
  AUTONOMOUS_SKILL: "--auto",
  OOD_ESCALATED: "--ood",
  STANDBY: "--standby",
};
const VIEW_TEXT = {
  cctv: "Cell camera",
  wrist_cam: "Wrist cam",
  top_ortho: "Top view",
  side_ortho: "Side view",
};

const $ = (id) => document.getElementById(id);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

const ui = {
  layoutKey: null,
  layoutId: null,
  slots: new Map(), // slot -> { el, img, url, last }
  sparks: new Map(), // bay -> { row, canvas, val }
  lastEventKey: null,
  telemetry: null,
  lastActionSeq: null, // null until the first telemetry, so a reload does not replay old toasts
  briefKey: null,
  view: { mode: "auto", pinnedBay: null, cameras: {}, bayOrder: [1, 2, 3, 4], zooms: {} },
  dragBay: null,
  dragRole: null,
  padActive: false,
  slTimer: null,
};

// Keyboard bar. Codes are GLFW key codes (the same ones the MuJoCo viewer receives).
const KEY_BAR = [
  { caps: ["1", "2", "3", "4"], codes: [49, 50, 51, 52], text: "Select bay" },
  { caps: ["Space"], codes: [32], text: "Clutch" },
  { caps: ["M"], codes: [77], text: "Backend" },
  { caps: ["R"], codes: [82], text: "Reset cells" },
];
const CAM_BTNS = [
  ["cctv", "CCTV"],
  ["wrist_cam", "Wrist"],
  ["top_ortho", "Top"],
  ["side_ortho", "Side"],
];
const FAULT_TEXT = {
  armed: { flap: "Flap jam armed", pin: "Bolt pin armed" },
  active: { flap: "Flap jammed", pin: "Bolt pinned" },
};
const faultName = (kind) => (kind === "pin" ? "pinned bolt" : "flap jam");
const TOAST_MS = 4500;
const MAX_TOASTS = 4;

// ------------------------------------------------------------------ view & drag helpers

function currentOrder() {
  return ui.view.bayOrder && ui.view.bayOrder.length === 4 ? ui.view.bayOrder : BAYS;
}

function effectiveFocusBay(t) {
  if (!t || t.mode === "grid") return null;
  const m = t.feeds.find((f) => f.role === "main");
  return m ? m.bay : null;
}

function setLayoutMode(mode, pinnedBay = null) {
  ui.view.mode = mode;
  ui.view.pinnedBay = mode === "pin" ? pinnedBay : null;
  sendClient({ type: "view", mode, pinned_bay: ui.view.pinnedBay, bay_order: currentOrder() });
  syncViewBar();
}

function togglePinBay(bayId) {
  const focused = effectiveFocusBay(ui.telemetry);
  if (ui.view.mode === "pin" && ui.view.pinnedBay === bayId) {
    setLayoutMode("grid", null);
  } else if (focused === bayId && ui.view.mode === "auto") {
    setLayoutMode("grid", null);
  } else {
    setLayoutMode("pin", bayId);
  }
}

function setBayCamera(bayId, camera) {
  ui.view.cameras[bayId] = camera;
  sendClient({ type: "view", bay: bayId, camera });
}

function swapBayOrder(bayA, bayB) {
  if (!bayA || !bayB || bayA === bayB) return;
  const next = [...currentOrder()];
  const ia = next.indexOf(bayA);
  const ib = next.indexOf(bayB);
  if (ia < 0 || ib < 0) return;
  [next[ia], next[ib]] = [next[ib], next[ia]];
  ui.view.bayOrder = next;
  sendClient({ type: "view", bay_order: next });
}

function sendFault(bayId, kind) {
  sendClient({ type: "fault", bay: bayId, kind });
}

function camZoom(camera) {
  const v = (camera && ui.view.zooms[camera]) || [1, 0, 0];
  return { z: v[0], px: v[1], py: v[2] };
}

function setCamZoom(camera, z, px, py) {
  if (!camera) return;
  const cz = Math.max(0.5, Math.min(4.0, z));
  const cpx = Math.max(-1.0, Math.min(1.0, px));
  const cpy = Math.max(-1.0, Math.min(1.0, py));
  ui.view.zooms[camera] = [cz, cpx, cpy];
  sendClient({ type: "view", zoom_cam: camera, zoom: cz, pan_x: cpx, pan_y: cpy });
  refreshZoomBadge(camera);
}

function resetCamZoom(camera) {
  if (!camera) return;
  delete ui.view.zooms[camera];
  sendClient({ type: "view", zoom_cam: camera, reset_zoom: true });
  refreshZoomBadge(camera);
}

function refreshZoomBadge(camera) {
  for (const el of $("wall").querySelectorAll(`.feed[data-camera="${camera}"]`)) {
    const badge = el.querySelector(".feed__zoom");
    if (!badge) continue;
    const { z, px, py } = camZoom(camera);
    const modified = Math.abs(z - 1) > 0.02 || Math.abs(px) > 0.02 || Math.abs(py) > 0.02;
    badge.hidden = !modified;
    if (modified) badge.textContent = `${z.toFixed(1)}× ↺`;
  }
}

function attachZoomPan(el, camera, role) {
  if (!camera || role === "strip") return;
  el.addEventListener(
    "wheel",
    (e) => {
      e.preventDefault();
      e.stopPropagation();
      const cur = camZoom(camera);
      const nextZ = cur.z * Math.exp(-e.deltaY * 0.0018);
      setCamZoom(camera, nextZ, cur.px, cur.py);
    },
    { passive: false }
  );

  let panStart = null;
  let moved = false;
  el.addEventListener("pointerdown", (e) => {
    if (!(e.shiftKey || e.button === 1 || e.button === 2)) return;
    if (e.target && e.target.closest("button")) return;
    e.preventDefault();
    e.stopPropagation();
    const cur = camZoom(camera);
    panStart = { x: e.clientX, y: e.clientY, px: cur.px, py: cur.py, z: cur.z, id: e.pointerId };
    moved = false;
    el.draggable = false;
    el.classList.add("is-panning");
    el.setPointerCapture(e.pointerId);
  });
  el.addEventListener("pointermove", (e) => {
    if (!panStart) return;
    const rect = el.getBoundingClientRect();
    const dx = e.clientX - panStart.x;
    const dy = e.clientY - panStart.y;
    if (Math.hypot(dx, dy) > 3) moved = true;
    const scale = Math.max(0.6, panStart.z * 0.75);
    const nx = panStart.px + dx / (Math.max(rect.width, 1) * scale);
    const ny = panStart.py + dy / (Math.max(rect.height, 1) * scale);
    setCamZoom(camera, panStart.z, nx, ny);
  });
  const endPan = () => {
    if (!panStart) return;
    panStart = null;
    el.draggable = true;
    el.classList.remove("is-panning");
  };
  el.addEventListener("pointerup", endPan);
  el.addEventListener("pointercancel", endPan);
  el.addEventListener("contextmenu", (e) => {
    if (moved) {
      e.preventDefault();
      moved = false;
    }
  });
}

// ------------------------------------------------------------------ feed tiles

function attachDragEvents(el, bayId, role) {
  el.draggable = true;
  el.addEventListener("dragstart", (e) => {
    ui.dragBay = bayId;
    ui.dragRole = role;
    el.classList.add("is-dragging");
    $("wall-wrap").classList.add("is-dragging-pane");
    $("focus-dropzone").querySelector("strong").textContent =
      role === "main"
        ? `Drag Bay ${bayId} onto the strip (or click 4-Up Equal) to unpin`
        : `Drop Bay ${bayId} here to pin into focus`;
    if (e.dataTransfer) {
      e.dataTransfer.effectAllowed = "move";
      e.dataTransfer.setData("text/plain", String(bayId));
    }
  });
  el.addEventListener("dragend", () => {
    ui.dragBay = null;
    ui.dragRole = null;
    el.classList.remove("is-dragging");
    $("wall-wrap").classList.remove("is-dragging-pane");
    $("focus-dropzone").classList.remove("is-over");
    for (const n of $("wall").querySelectorAll(".is-drop-target")) n.classList.remove("is-drop-target");
  });
  el.addEventListener("dragover", (e) => {
    if (!ui.dragBay) return;
    e.preventDefault();
    el.classList.add("is-drop-target");
  });
  el.addEventListener("dragleave", () => el.classList.remove("is-drop-target"));
  el.addEventListener("drop", (e) => {
    e.preventDefault();
    e.stopPropagation();
    el.classList.remove("is-drop-target");
    const srcBay = ui.dragBay;
    const srcRole = ui.dragRole;
    if (!srcBay) return;
    if (role === "main" && srcRole !== "main") setLayoutMode("pin", srcBay);
    else if (role === "strip" && srcRole === "main") setLayoutMode("grid", null);
    else if (srcBay !== bayId) swapBayOrder(srcBay, bayId);
  });
}

function feedTile(bayId, camera, role, bayInfo) {
  const el = document.createElement("div");
  el.className = `feed feed--${role}`;
  el.dataset.bay = String(bayId);
  el.dataset.camera = camera || "";
  el.tabIndex = 0;
  const view = camera ? camera.split("/")[1] : "cctv";
  const isPinned = role === "main";
  if (isPinned) el.classList.add("is-pinned");
  el.innerHTML = `
    <img alt="" data-cam="${camera || ""}">
    <div class="feed__top">
      <span class="feed__name">Bay ${bayId}</span>
      <span class="feed__view">${VIEW_TEXT[view] || ""}</span>
      <button type="button" class="feed__zoom" hidden title="Scroll to zoom, Shift/right-drag to pan. Click to reset.">1.0×</button>
      <span class="feed__state"></span>
      <span class="feed__conf"></span>
    </div>
    <div class="feed__tools"></div>
    <div class="feed__bottom">
      <span class="feed__rec">REC</span>
      <time class="feed__time"></time>
      <span class="feed__armed" hidden></span>
      <span class="feed__replay"></span>
      <span class="feed__bolt" hidden>
        <span class="feed__bolt-label">Bolt</span>
        <span class="feed__bolt-track"><span class="feed__bolt-fill"></span></span>
        <span class="feed__bolt-val"></span>
      </span>
    </div>
    <div class="feed__lost"><strong>Signal lost</strong><span>Waiting for the camera to reconnect</span></div>`;

  const stop = (fn) => (ev) => { ev.stopPropagation(); fn(); };
  el.querySelector(".feed__zoom").addEventListener("click", stop(() => resetCamZoom(camera)));

  const tools = el.querySelector(".feed__tools");
  if (role === "tile") {
    for (const [camKey, label] of CAM_BTNS) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = `ftool${view === camKey ? " is-active" : ""}`;
      btn.textContent = label;
      btn.addEventListener("click", stop(() => setBayCamera(bayId, camKey)));
      tools.append(btn);
    }
  }
  if (role !== "strip" && camera) {
    for (const [label, mult, tip] of [["−", 1 / 1.25, "Zoom out"], ["+", 1.25, "Zoom in"]]) {
      const zb = document.createElement("button");
      zb.type = "button";
      zb.className = "ftool";
      zb.textContent = label;
      zb.title = `${tip} (or scroll wheel)`;
      zb.addEventListener("click", stop(() => {
        const cur = camZoom(camera);
        setCamZoom(camera, cur.z * mult, cur.px, cur.py);
      }));
      tools.append(zb);
    }
  }
  if (role !== "strip" && bayInfo && !bayInfo.background && bayInfo.state !== "STANDBY") {
    const jamBtn = document.createElement("button");
    jamBtn.type = "button";
    jamBtn.className = "ftool ftool--warn";
    jamBtn.textContent = "Jam";
    jamBtn.addEventListener("click", stop(() => sendFault(bayId, "flap")));
    tools.append(jamBtn);
  }
  const pinBtn = document.createElement("button");
  pinBtn.type = "button";
  pinBtn.className = `ftool${isPinned ? " is-active" : ""}`;
  pinBtn.textContent = isPinned ? "Unpin" : "Pin";
  pinBtn.addEventListener("click", stop(() => (isPinned ? setLayoutMode("grid", null) : setLayoutMode("pin", bayId))));
  tools.append(pinBtn);

  el.addEventListener("click", () => selectBay(bayId));
  el.addEventListener("dblclick", (e) => { e.stopPropagation(); togglePinBay(bayId); });
  el.addEventListener("keydown", (e) => { if (e.key === "Enter") selectBay(bayId); });
  attachDragEvents(el, bayId, role);
  attachZoomPan(el, camera, role);
  return el;
}

function standbyTile(bayId) {
  const el = document.createElement("div");
  el.className = "feed feed--tile";
  el.dataset.bay = String(bayId);
  el.dataset.state = "STANDBY";
  el.innerHTML = `
    <div class="feed__top"><span class="feed__name">Bay ${bayId}</span>
      <span class="feed__state">${STATE_TEXT.STANDBY}</span></div>
    <div class="feed__nosignal"><strong>No camera</strong><span>This bay is not running in the simulation</span></div>`;
  return el;
}

function releaseSlot(s) {
  if (s.img) s.img.onload = s.img.onerror = null;
  if (s.url) URL.revokeObjectURL(s.url);
  if (s.pending) URL.revokeObjectURL(s.pending);
  s.url = s.pending = null;
}

function registerSlot(slot, el) {
  const prev = ui.slots.get(slot);
  if (prev) releaseSlot(prev);
  const img = el.querySelector("img");
  ui.slots.set(slot, { el, img, url: null, pending: null, last: performance.now() });
}

function buildWall(t) {
  const wall = $("wall");
  for (const s of ui.slots.values()) releaseSlot(s);
  ui.slots.clear();
  wall.replaceChildren();
  const bySlot = new Map(t.feeds.map((f) => [f.slot, f]));
  const bayMap = new Map(t.bays.map((b) => [b.id, b]));
  const order = (t.view && t.view.bay_order) || currentOrder();
  const mains = t.feeds.filter((x) => x.role === "main");

  wall.classList.remove("is-rescue", "is-pin", "is-single-main");

  if (mains.length > 0) {
    const focusBay = mains[0].bay;
    const isRescueFocus = !!(t.rescue && t.rescue.bay === focusBay);
    wall.classList.add(isRescueFocus ? "is-rescue" : "is-pin");
    if (mains.length === 1) wall.classList.add("is-single-main");

    const head = document.createElement("div");
    head.className = "takeover";
    const focusInfo = bayMap.get(focusBay);
    const refusedBay = focusInfo && focusInfo.state === "OOD_ESCALATED";
    const subtitle = isRescueFocus
      ? refusedBay ? "refused stall" : "operator rescue"
      : "pinned focus";
    head.innerHTML = `<h2>Bay ${focusBay} · ${subtitle}</h2>`;

    const camGroup = document.createElement("div");
    camGroup.className = "takeover__views";
    const activeCam =
      mains.length > 1
        ? (mains[0].camera.endsWith("/wrist_cam") ? "wrist_split" : "split")
        : (mains[0].camera.split("/")[1] || "cctv");
    for (const [camKey, label] of [
      ...CAM_BTNS,
      ["wrist_split", "Wrist + Top"],
      ["split", "Top + Side"],
    ]) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = `vbtn${activeCam === camKey ? " is-active" : ""}`;
      btn.textContent = label;
      btn.addEventListener("click", () => setBayCamera(focusBay, camKey));
      camGroup.append(btn);
    }
    head.append(camGroup);

    const spacer = document.createElement("span");
    spacer.className = "takeover__spacer";
    head.append(spacer);

    if (isRescueFocus) {
      const timer = document.createElement("span");
      timer.className = "takeover__timer";
      timer.id = "rescue-timer";
      const clutch = document.createElement("span");
      clutch.className = "clutch";
      clutch.id = "clutch";
      head.append(timer, clutch);
    }
    const unpin = document.createElement("button");
    unpin.type = "button";
    unpin.className = "vbtn";
    unpin.textContent = "4-Up Equal";
    unpin.addEventListener("click", () => setLayoutMode("grid", null));
    head.append(unpin);
    wall.append(head);

    for (const f of mains) {
      const el = feedTile(f.bay, f.camera, "main", bayMap.get(f.bay));
      wall.append(el);
      registerSlot(f.slot, el);
    }
    const strip = document.createElement("div");
    strip.className = "strip";
    for (const bay of order) {
      const f = t.feeds.find((x) => x.role === "strip" && x.bay === bay);
      if (f) {
        const el = feedTile(bay, f.camera, "strip", bayMap.get(bay));
        strip.append(el);
        registerSlot(f.slot, el);
      } else {
        const el = standbyTile(bay);
        el.classList.add("feed--strip");
        strip.append(el);
      }
    }
    wall.append(strip);
  } else {
    for (const bay of order) {
      const f = [...bySlot.values()].find((x) => x.bay === bay && x.role === "tile");
      if (f) {
        const el = feedTile(bay, f.camera, "tile", bayMap.get(bay));
        wall.append(el);
        registerSlot(f.slot, el);
      } else {
        wall.append(standbyTile(bay));
      }
    }
  }
}

function updateWall(t) {
  const bays = new Map(t.bays.map((b) => [b.id, b]));
  for (const el of $("wall").querySelectorAll(".feed")) {
    const b = bays.get(Number(el.dataset.bay));
    if (!b) continue;
    if (el.dataset.state !== "STANDBY" || b.state !== "STANDBY") el.dataset.state = b.state;
    el.classList.toggle("is-selected", t.selected_bay === b.id && t.mode !== "rescue");
    const state = el.querySelector(".feed__state");
    if (state) state.textContent = STATE_TEXT[b.state] || b.label;
    const conf = el.querySelector(".feed__conf");
    if (conf) conf.textContent = b.state === "STANDBY" ? "" : b.confidence.toFixed(2);
    const time = el.querySelector(".feed__time");
    if (time) time.textContent = t.wall;
    const replay = el.querySelector(".feed__replay");
    if (replay) replay.textContent = b.replayed ? "Replayed routine cycle" : "";
    const armed = el.querySelector(".feed__armed");
    if (armed) {
      const f = b.fault;
      armed.hidden = !f;
      armed.textContent = f ? (FAULT_TEXT[f.phase] || {})[f.kind] || "Fault" : "";
    }
    const bolt = el.querySelector(".feed__bolt");
    if (bolt) {
      const show = b.x_bolt_cm != null && t.bolt_clear_cm;
      bolt.hidden = !show;
      if (show) {
        const frac = Math.max(0, Math.min(1, b.x_bolt_cm / t.bolt_clear_cm));
        bolt.querySelector(".feed__bolt-fill").style.width = `${(frac * 100).toFixed(1)}%`;
        bolt.classList.toggle("is-clear", b.x_bolt_cm >= t.bolt_clear_cm);
        bolt.querySelector(".feed__bolt-val").textContent =
          `${b.x_bolt_cm.toFixed(1)} / ${t.bolt_clear_cm.toFixed(1)} cm`;
      }
    }
  }
  if (t.mode === "rescue" && t.rescue) {
    const timer = $("rescue-timer");
    if (timer) timer.textContent = `${(t.rescue.elapsed_s ?? 0).toFixed(1)} s`;
    const clutch = $("clutch");
    if (clutch) {
      clutch.classList.toggle("is-held", !!t.rescue.clutch);
      clutch.textContent = t.rescue.clutch ? "Clutch held" : "Clutch released";
    }
  }
}

// ------------------------------------------------------------------ rail

function updateQueue(t) {
  const q = $("queue");
  q.replaceChildren();
  if (!t.queue.length) {
    const li = document.createElement("li");
    li.className = "is-empty";
    li.textContent = "No bays waiting";
    q.append(li);
    return;
  }
  t.queue.forEach((bay, i) => {
    const li = document.createElement("li");
    li.innerHTML = `Bay ${bay} <span>${i === 0 ? "next" : `#${i + 1}`}</span>`;
    q.append(li);
  });
}

function ensureSparks() {
  if (ui.sparks.size) return;
  const box = $("sparks");
  for (const bay of BAYS) {
    const row = document.createElement("div");
    row.className = "spark";
    row.innerHTML = `<span class="spark__name">Bay ${bay}</span><canvas></canvas><span class="spark__val"></span>`;
    box.append(row);
    ui.sparks.set(bay, { row, canvas: row.querySelector("canvas"), val: row.querySelector(".spark__val") });
  }
}

function drawSpark(canvas, history, threshold, color) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  if (!w || !h) return;
  if (canvas.width !== Math.round(w * dpr)) canvas.width = Math.round(w * dpr);
  if (canvas.height !== Math.round(h * dpr)) canvas.height = Math.round(h * dpr);
  const g = canvas.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);
  const y = (v) => h - 2 - Math.max(0, Math.min(1, v)) * (h - 4);
  g.strokeStyle = css("--faint");
  g.setLineDash([3, 3]);
  g.beginPath();
  g.moveTo(0, y(threshold));
  g.lineTo(w, y(threshold));
  g.stroke();
  g.setLineDash([]);
  if (history.length < 2) return;
  const n = 150;
  const step = w / (n - 1);
  const x0 = w - (history.length - 1) * step;
  g.strokeStyle = color;
  g.lineWidth = 1.5;
  g.beginPath();
  history.forEach((v, i) => (i ? g.lineTo(x0 + i * step, y(v)) : g.moveTo(x0 + i * step, y(v))));
  g.stroke();
}

function updateSparks(t) {
  ensureSparks();
  $("threshold-note").textContent = `stall below ${t.threshold.toFixed(1)}`;
  for (const b of t.bays) {
    const s = ui.sparks.get(b.id);
    if (!s) continue;
    const standby = b.state === "STANDBY";
    s.row.classList.toggle("is-standby", standby);
    s.row.classList.toggle("is-low", !standby && b.confidence < t.threshold);
    s.val.textContent = standby ? "–" : b.confidence.toFixed(2);
    drawSpark(s.canvas, b.history, t.threshold, css(STATE_COLOR[b.state] || "--nominal"));
  }
}

function updateGate(t) {
  const box = $("gate");
  const g = t.gate;
  if (!g || g.ok === null || g.ok === undefined) {
    box.innerHTML = `<span class="is-empty">${
      g && g.fit_ms != null ? `Skill fitted in ${g.fit_ms.toFixed(0)} ms, checking in simulation` : "No skill learned yet"
    }</span>`;
    return;
  }
  const big = `<span class="gate__big ${g.ok ? "is-ok" : "is-bad"}">${g.passed}/${g.total}</span>`;
  const verdict = g.ok ? "Simulation checks passed, shared with the fleet" : "Simulation checks failed, not shared";
  const timing = [
    g.fit_ms != null ? `fitted in ${g.fit_ms.toFixed(0)} ms` : null,
    g.elapsed_s != null ? `checked in ${g.elapsed_s.toFixed(2)} s` : null,
  ].filter(Boolean).join(", ");
  box.innerHTML = `${big}<span>${verdict}</span><span class="gate__line">${timing}</span>`;
}

function updateLog(t) {
  const last = t.events[t.events.length - 1];
  const key = last ? `${t.events.length}|${last.sim_t}|${last.text}` : "empty";
  if (key === ui.lastEventKey) return;
  ui.lastEventKey = key;
  const countNote = $("activity-count");
  if (countNote) countNote.textContent = t.events.length ? `(${t.events.length})` : "";
  const log = $("log");
  const wasScrolled = log.scrollTop > 12;
  const prevBottom = log.scrollHeight - log.scrollTop;
  log.replaceChildren();
  if (!t.events.length) {
    const li = document.createElement("li");
    li.className = "is-empty";
    li.textContent = "All bays running their routine";
    log.append(li);
    return;
  }
  for (const e of [...t.events].reverse()) {
    const li = document.createElement("li");
    li.dataset.level = e.level;
    const time = document.createElement("time");
    time.textContent = e.wall;
    const text = document.createElement("span");
    text.textContent = e.text;
    li.append(time, text);
    log.append(li);
  }
  if (wasScrolled) {
    log.scrollTop = Math.max(0, log.scrollHeight - prevBottom);
  }
}

// ------------------------------------------------------------------ what is happening now

const kbd = (k) => `<kbd>${k}</kbd>`;

function situation(t) {
  const bays = t.bays.filter((b) => b.state !== "STANDBY");
  const first = (state) => bays.find((b) => b.state === state);
  const clear = t.bolt_clear_cm ? t.bolt_clear_cm.toFixed(1) : "5.2";

  if (t.rescue) {
    const r = t.rescue.bay;
    const b = bays.find((x) => x.id === r);
    const refused = b && b.state === "OOD_ESCALATED";
    const bolt = b && b.x_bolt_cm != null ? `${b.x_bolt_cm.toFixed(1)} cm` : "unknown";
    const tag = refused ? `Refused · Bay ${r}` : `Rescue · Bay ${r}`;
    const tone = refused ? "ood" : "rescue";
    const acts = [];
    if (t.mode === "grid") acts.push({ label: `Focus Bay ${r}`, accent: true, run: () => setLayoutMode("pin", r) });
    if (t.rescue.clutch && !refused && b && b.x_bolt_cm != null && b.x_bolt_cm >= t.bolt_clear_cm) {
      acts.push({ label: "Release Clutch", accent: true, run: () => pressKey(32, "Space") });
      return {
        tone, tag,
        head: `Bolt cleared on Bay ${r}`,
        hint: `Release clutch (${kbd("Space")}) to fit the DMP and run the 3-rollout simulation gate.`,
        keys: [32],
        actions: acts,
      };
    }
    if (t.rescue.clutch) {
      if (!refused && t.dummy) {
        acts.push({ label: "Guided Solve", accent: true, run: () => sendClient({ type: "dummy", action: "solve" }) });
      }
      acts.push({ label: "Release Clutch", accent: false, run: () => pressKey(32, "Space") });
      return {
        tone, tag,
        head: `Teleoperating Bay ${r}`,
        hint: refused
          ? `Pinned bolt fault cannot clear safely. Release (${kbd("Space")}) or reset (${kbd("R")}).`
          : `Slide bolt past ${clear} cm (now ${bolt}) via the teleop pad or gloves, then release ${kbd("Space")}.`,
        keys: [32],
        actions: acts,
      };
    }
    const head = refused
      ? `Bay ${r} refused out-of-distribution fault`
      : `Bay ${r} waiting for operator rescue`;
    if (t.selected_bay !== r) {
      acts.push({ label: `Select Bay ${r}`, accent: true, run: () => selectBay(r) });
      return {
        tone, tag, head,
        hint: `Select Bay ${r} (${kbd(r)}) and engage clutch (${kbd("Space")}) to intervene.`,
        keys: [48 + r],
        actions: acts,
      };
    }
    acts.push({ label: "Engage Clutch", accent: true, run: () => pressKey(32, "Space") });
    if (refused) acts.push({ label: "Reset Cell", accent: false, run: () => pressKey(82, "R") });
    return {
      tone, tag, head,
      hint: refused
        ? `Stall signature outside Mahalanobis gate. Inspect with ${kbd("Space")} or reset with ${kbd("R")}.`
        : `Bay ${r} selected. Engage clutch (${kbd("Space")}) to drive the arm.`,
      keys: [32],
      actions: acts,
    };
  }
  const val = first("VALIDATING");
  if (val) {
    return {
      tone: "validating", tag: "Learning",
      head: `Validating Bay ${val.id} rescue skill`,
      hint: "Fitting 30-kernel RBF-DMP and testing 3 perturbed MuJoCo rollouts before fleet deployment.",
      keys: [],
      actions: [],
    };
  }
  const stalled = first("STALLED");
  if (stalled) {
    return {
      tone: "stalled", tag: `Stall · Bay ${stalled.id}`,
      head: `Bay ${stalled.id} contact stall detected`,
      hint: "Classifying 300 ms contact torque signature against the Fleet Skill Library…",
      keys: [],
      actions: [],
    };
  }
  const auto = first("AUTONOMOUS_SKILL");
  if (auto) {
    return {
      tone: "auto", tag: `Autonomous · Bay ${auto.id}`,
      head: `Bay ${auto.id} executing learned recovery skill`,
      hint: `Warping ${auto.skill || "fleet DMP"} into Bay ${auto.id}'s latch frame inside the torque safety tube.`,
      keys: [],
      actions: [{ label: `Pin Bay ${auto.id}`, accent: false, run: () => setLayoutMode("pin", auto.id) }],
    };
  }
  if (t.queue.length) {
    return {
      tone: "queued", tag: "Queue",
      head: `Bay ${t.queue[0]} queued for operator takeover`,
      hint: "Holding zero-velocity hover until the active rescue slot is free.",
      keys: [],
      actions: [],
    };
  }
  const active = bays.filter((b) => b.fault && b.fault.phase === "active" && b.state === "NOMINAL" && b.x_bolt_cm < t.bolt_clear_cm);
  if (active.length) {
    const b = active[0];
    return {
      tone: "armed", tag: `Fault · Bay ${b.id}`,
      head: `Bay ${b.id} approaching ${faultName(b.fault.kind)}`,
      hint: `Confidence triggers a stall if C(t) stays below ${t.threshold.toFixed(1)} for 300 ms.`,
      keys: [],
      actions: [{ label: `Pin Bay ${b.id}`, accent: false, run: () => setLayoutMode("pin", b.id) }],
    };
  }
  const armed = bays.filter((b) => b.fault && b.fault.phase === "armed");
  if (armed.length) {
    const what = armed.map((b) => `Bay ${b.id} (${faultName(b.fault.kind)})`).join(", ");
    return {
      tone: "armed", tag: "Fault armed",
      head: `Next-cycle fault queued: ${what}`,
      hint: "Applies automatically as soon as the current pallet cycle completes.",
      keys: [],
      actions: armed.map((b) => ({ label: `Cancel Bay ${b.id}`, accent: false, run: () => sendFault(b.id, "clear") })),
    };
  }
  const physicsBays = bays.filter((b) => !b.background);
  const faultBtns = physicsBays.map((b) => ({
    label: `Jam Bay ${b.id}`,
    accent: false,
    run: () => sendFault(b.id, "flap"),
  }));
  if (physicsBays.some((b) => b.id === 2)) {
    faultBtns.push({ label: "Pin Bay 2 (OOD)", accent: false, run: () => sendFault(2, "pin") });
  }
  return {
    tone: "idle",
    tag: t.gate && t.gate.ok ? "Fleet skill active" : "Nominal",
    head: "Autonomous disassembly running across active bays",
    hint: "Drag any bay pane to focus or swap positions, switch camera angles, or inject a fault anytime.",
    keys: [],
    actions: faultBtns,
  };
}

function updateBrief(t) {
  const s = situation(t);
  const actKey = (s.actions || []).map((a) => a.label).join(",");
  const key = `${s.tone}|${s.tag}|${s.head}|${s.hint}|${actKey}`;
  if (key !== ui.briefKey) {
    ui.briefKey = key;
    const brief = $("brief");
    brief.dataset.tone = s.tone;
    $("brief-tag").textContent = s.tag;
    $("brief-head").textContent = s.head;
    $("brief-hint").innerHTML = s.hint;
    const actBox = $("brief-actions");
    actBox.replaceChildren();
    for (const act of s.actions || []) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = `actbtn${act.accent ? " actbtn--accent" : ""}`;
      btn.textContent = act.label;
      btn.addEventListener("click", act.run);
      actBox.append(btn);
    }
    brief.classList.remove("is-new");
    void brief.offsetWidth;
    brief.classList.add("is-new");
  }
  for (const cap of $("keys").querySelectorAll(".key__cap")) {
    cap.classList.toggle("is-suggested", s.keys.includes(Number(cap.dataset.code)));
  }
  const beat = t.beat || "";
  $("beat").hidden = !beat;
  $("beat-name").textContent = beat;
}

// ------------------------------------------------------------------ key bar and toasts

function buildKeyBar() {
  const bar = $("keys");
  for (const group of KEY_BAR) {
    const item = document.createElement("div");
    item.className = "key";
    group.caps.forEach((cap, i) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "key__cap";
      if (cap.length > 1) b.classList.add("key__cap--wide");
      b.dataset.code = String(group.codes[i]);
      b.textContent = cap;
      b.title = `${group.text} (${cap})`;
      b.tabIndex = -1; // never keeps focus, so a later Space press is not also a click
      b.addEventListener("mousedown", (e) => e.preventDefault());
      b.addEventListener("click", () => pressKey(group.codes[i], cap));
      item.append(b);
    });
    const label = document.createElement("span");
    label.className = "key__text";
    label.textContent = group.text;
    item.append(label);
    bar.append(item);
  }
}

function flashCap(code) {
  const cap = $("keys").querySelector(`.key__cap[data-code="${code}"]`);
  if (!cap) return;
  cap.classList.remove("is-pressed");
  void cap.offsetWidth;
  cap.classList.add("is-pressed");
}

function toast(key, text, level) {
  const box = $("toasts");
  const el = document.createElement("div");
  el.className = "toast";
  el.dataset.level = level;
  const k = document.createElement("kbd");
  k.textContent = key;
  const span = document.createElement("span");
  span.textContent = text;
  el.append(k, span);
  box.append(el);
  while (box.children.length > MAX_TOASTS) box.firstElementChild.remove();
  setTimeout(() => {
    el.classList.add("is-leaving");
    setTimeout(() => el.remove(), 300);
  }, TOAST_MS);
}

function updateActions(t) {
  const actions = t.actions || [];
  const newest = actions.length ? actions[actions.length - 1].seq : 0;
  if (ui.lastActionSeq === null) {
    ui.lastActionSeq = newest; // first message after a (re)load: do not replay history
    return;
  }
  for (const a of actions) {
    if (a.seq > ui.lastActionSeq) toast(a.key, a.text, a.level);
  }
  ui.lastActionSeq = Math.max(ui.lastActionSeq, newest);
}

function syncViewBar() {
  const t = ui.telemetry;
  const vmode = (t && t.view && t.view.mode) || ui.view.mode;
  const pinned = t && t.view ? t.view.pinned_bay : ui.view.pinnedBay;
  $("view-grid").classList.toggle("is-active", vmode === "grid");
  $("view-auto").classList.toggle("is-active", vmode === "auto");
  for (const btn of $("pin-bar").querySelectorAll(".vbtn--pin")) {
    btn.classList.toggle("is-active", vmode === "pin" && pinned === Number(btn.dataset.bay));
  }
}

function updateDummy(t) {
  const d = t.dummy;
  $("dummy-badge").hidden = !d;
  $("dummy-panel").hidden = !d;
  if (!d) return;
  $("dummy-meta").textContent =
    d.transport === "studio_udp" ? `Studio UDP${d.pid ? ` · PID ${d.pid}` : ""}` : "Stand-in";
  for (const btn of $("dummy-panel").querySelectorAll("[data-dummy-mode]")) {
    btn.classList.toggle("is-active", btn.dataset.dummyMode === d.mode);
  }
  $("dummy-autofaults").checked = !!d.auto_faults;
  $("dummy-manual").hidden = d.mode === "off";
  $("btn-solve").textContent = d.solving ? "Solving…" : "Guided Solve";
  if (!ui.padActive) {
    const xPct = ((d.dx_cm + 3.0) / 10.0) * 100;
    const zPct = ((4.0 - d.dz_cm) / 7.0) * 100;
    $("wrist-dot").style.left = `${Math.max(4, Math.min(96, xPct))}%`;
    $("wrist-dot").style.top = `${Math.max(6, Math.min(94, zPct))}%`;
    $("sl-thumb").value = String(Math.round((d.thumb || 0) * 100));
    $("sl-index").value = String(Math.round((d.index || 0) * 100));
    $("sl-dy").value = String(Math.round((d.dy_cm || 0) * 10));
  }
}

function render(t) {
  ui.telemetry = t;
  if (t.view) {
    ui.view.mode = t.view.mode || ui.view.mode;
    ui.view.pinnedBay = t.view.pinned_bay ?? null;
    if (Array.isArray(t.view.bay_order) && t.view.bay_order.length === 4) {
      ui.view.bayOrder = t.view.bay_order;
    }
    if (t.view.zooms && typeof t.view.zooms === "object") {
      ui.view.zooms = t.view.zooms;
    }
  }
  const orderStr = ((t.view && t.view.bay_order) || currentOrder()).join("");
  const layoutKey = `${t.layout_id}|${t.mode}|${orderStr}|${t.feeds.map((f) => `${f.slot}:${f.camera}:${f.role}`).join(",")}`;
  if (layoutKey !== ui.layoutKey) {
    ui.layoutKey = layoutKey;
    buildWall(t);
  }
  for (const f of t.feeds) {
    if (f.role !== "strip") refreshZoomBadge(f.camera);
  }
  ui.layoutId = t.layout_id % 256;
  $("clock").textContent = t.wall;
  const active = t.bays.filter((b) => b.state !== "STANDBY").length;
  $("shift").textContent = `${active} bays, 1 operator on shift`;
  syncViewBar();
  updateWall(t);
  updateQueue(t);
  updateSparks(t);
  updateGate(t);
  updateLog(t);
  updateBrief(t);
  updateDummy(t);
  updateActions(t);
  $("compliance").textContent = t.compliance;
  $("reasoning").textContent = t.reasoning;
  $("unoq").textContent = t.unoq;
}

// ------------------------------------------------------------------ frames

// <frame-urls>
// Each slot keeps the URL on screen (s.url) and at most one still loading (s.pending). A frame
// superseded before it loads, a frame that fails to decode and the frame it replaces are all
// revoked, so bursts of frames never leak blob memory.
function showFrame(s, url, revoke) {
  if (s.pending) revoke(s.pending);
  s.pending = url;
  const settle = (ok) => {
    if (s.pending !== url) return; // a stale callback from a superseded frame
    s.pending = null;
    if (ok) {
      if (s.url) revoke(s.url);
      s.url = url;
    } else {
      revoke(url);
    }
  };
  s.img.onload = () => settle(true);
  s.img.onerror = () => settle(false);
  s.img.src = url;
}
// </frame-urls>

function revokeUrl(url) {
  URL.revokeObjectURL(url);
}

function onFrame(buf) {
  const bytes = new Uint8Array(buf);
  if (bytes.length < 3) return;
  const slot = bytes[0];
  if (bytes[1] !== ui.layoutId) return; // rendered for a previous layout
  const s = ui.slots.get(slot);
  if (!s || !s.img) return;
  const url = URL.createObjectURL(new Blob([buf.slice(2)], { type: "image/jpeg" }));
  s.last = performance.now();
  showFrame(s, url, revokeUrl);
  s.el.classList.remove("is-lost");
}

setInterval(() => {
  const now = performance.now();
  for (const s of ui.slots.values()) s.el.classList.toggle("is-lost", now - s.last > STALE_MS);
}, 500);

// ------------------------------------------------------------------ connection

let ws = null;
let backoff = 1000;

function setLink(up) {
  $("live").classList.toggle("is-up", up);
  $("live-text").textContent = up ? "Live" : "Offline";
  $("linkdown").hidden = up;
}

function sendClient(payload) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(payload));
    return true;
  }
  return false;
}

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}${WS_PATH}`);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => {
    backoff = 1000;
    setLink(true);
    sendClient({
      type: "view",
      mode: ui.view.mode,
      pinned_bay: ui.view.pinnedBay,
      bay_order: currentOrder(),
    });
  };
  ws.onmessage = (ev) => {
    if (typeof ev.data === "string") {
      let msg;
      try {
        msg = JSON.parse(ev.data);
      } catch {
        return;
      }
      if (msg.type === "telemetry") render(msg);
    } else {
      onFrame(ev.data);
    }
  };
  ws.onclose = () => {
    setLink(false);
    setTimeout(connect, backoff);
    backoff = Math.min(backoff * 2, 5000);
  };
  ws.onerror = () => ws.close();
}

// ------------------------------------------------------------------ keys (GLFW codes) & controls

function sendKey(code) {
  return sendClient({ type: "key", code });
}

function pressKey(code, name) {
  flashCap(code);
  if (!sendKey(code)) toast(name, "Not connected to the cell controller, key not sent", "warn");
}

function keyName(code) {
  return code === 32 ? "Space" : String.fromCharCode(code);
}

function selectBay(bayId) {
  const b = ui.telemetry && ui.telemetry.bays.find((x) => x.id === bayId);
  if (!b || b.state === "STANDBY") return;
  if (b.background) {
    togglePinBay(bayId);
    return;
  }
  pressKey(48 + bayId, String(bayId));
}

function initViewBar() {
  $("view-grid").addEventListener("click", () => setLayoutMode("grid", null));
  $("view-auto").addEventListener("click", () => setLayoutMode("auto", null));
  const pinBar = $("pin-bar");
  for (const bay of BAYS) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "vbtn vbtn--pin";
    b.dataset.bay = String(bay);
    b.textContent = String(bay);
    b.title = `Pin Bay ${bay} (click again to unpin)`;
    b.addEventListener("click", () => togglePinBay(bay));
    pinBar.append(b);
  }
  const dz = $("focus-dropzone");
  dz.addEventListener("dragover", (e) => {
    if (!ui.dragBay) return;
    e.preventDefault();
    dz.classList.add("is-over");
  });
  dz.addEventListener("dragleave", () => dz.classList.remove("is-over"));
  dz.addEventListener("drop", (e) => {
    e.preventDefault();
    dz.classList.remove("is-over");
    if (!ui.dragBay) return;
    if (ui.dragRole === "main") setLayoutMode("grid", null);
    else setLayoutMode("pin", ui.dragBay);
  });
}

// TODO(rokoko-gloves-swap): Temporary UI controls for DummyOperatorSupervisor
function initDummyControls() {
  for (const btn of $("dummy-panel").querySelectorAll("[data-dummy-mode]")) {
    btn.addEventListener("click", () => sendClient({ type: "dummy", mode: btn.dataset.dummyMode }));
  }
  $("dummy-autofaults").addEventListener("change", (e) => {
    sendClient({ type: "dummy", auto_faults: e.target.checked });
  });
  $("btn-solve").addEventListener("click", () => sendClient({ type: "dummy", action: "solve" }));
  $("btn-zero").addEventListener("click", () => sendClient({ type: "dummy", action: "reset_pose" }));

  const pad = $("wrist-pad");
  const sendPad = (ev) => {
    const rect = pad.getBoundingClientRect();
    const fx = Math.max(0, Math.min(1, (ev.clientX - rect.left) / rect.width));
    const fz = Math.max(0, Math.min(1, (ev.clientY - rect.top) / rect.height));
    $("wrist-dot").style.left = `${fx * 100}%`;
    $("wrist-dot").style.top = `${fz * 100}%`;
    sendClient({ type: "dummy", dx_cm: -3.0 + fx * 10.0, dz_cm: 4.0 - fz * 7.0 });
  };
  pad.addEventListener("pointerdown", (e) => {
    ui.padActive = true;
    pad.setPointerCapture(e.pointerId);
    sendPad(e);
  });
  pad.addEventListener("pointermove", (e) => { if (ui.padActive) sendPad(e); });
  const endPad = () => { ui.padActive = false; };
  pad.addEventListener("pointerup", endPad);
  pad.addEventListener("pointercancel", endPad);
  pad.addEventListener("keydown", (e) => {
    const map = {
      ArrowRight: { dx_cm: 0.4 },
      ArrowLeft: { dx_cm: -0.4 },
      ArrowUp: { dz_cm: 0.3 },
      ArrowDown: { dz_cm: -0.3 },
    };
    if (map[e.key]) {
      e.preventDefault();
      e.stopPropagation();
      sendClient({ type: "dummy", nudge: map[e.key] });
    }
  });
  const bindSlider = (id, fn) => {
    $(id).addEventListener("input", (e) => {
      ui.padActive = true;
      fn(Number(e.target.value));
      clearTimeout(ui.slTimer);
      ui.slTimer = setTimeout(() => { ui.padActive = false; }, 250);
    });
  };
  bindSlider("sl-thumb", (v) => sendClient({ type: "dummy", thumb: v / 100 }));
  bindSlider("sl-index", (v) => sendClient({ type: "dummy", index: v / 100 }));
  bindSlider("sl-dy", (v) => sendClient({ type: "dummy", dy_cm: v / 10 }));
}

window.addEventListener("keydown", (event) => {
  if (event.repeat || event.ctrlKey || event.metaKey || event.altKey) return;
  const tag = event.target && event.target.tagName;
  if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;
  if (event.key === "Escape" && ui.view.mode === "pin") {
    setLayoutMode("grid", null);
    return;
  }
  const k = event.key;
  let code = null;
  if (k === " ") code = 32;
  else if (k.length === 1 && /[0-9]/.test(k)) code = k.charCodeAt(0);
  else if (k.length === 1 && /[a-z]/i.test(k)) code = k.toUpperCase().charCodeAt(0);
  if (code === null) return;
  if (code === 32) event.preventDefault();
  pressKey(code, keyName(code));
});

window.addEventListener("resize", () => ui.telemetry && updateSparks(ui.telemetry));

initViewBar();
initDummyControls();
buildKeyBar();
setLink(false);
$("live-text").textContent = "Connecting";
connect();
