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
const VIEW_TEXT = {
  overview: "Mezzanine camera",
  cctv: "Conveyor station",
  wrist_cam: "Wrist cam",
  top_ortho: "Top view",
  side_ortho: "Side view",
};
const BAY_ZONE = {
  1: "Belt 1, south-west station",
  2: "Belt 2, south-east station",
  3: "Belt 3, north-west station",
  4: "Belt 4, north-east station",
};
const BELT_TEXT = {
  feeding: "Belt feeding",
  paused: "Belt paused",
  estop: "E-STOP · belt halted",
};
const AUDIT_KIND_TEXT = {
  STALL_ALERT: "Stall alert",
  RESCUE_LOGGED: "Rescue logged",
  SKILL_DEPLOYED: "Skill deployed",
  OOD_REFUSAL: "Refusal logged",
};
const AUDIT_STATE_TEXT = { anchored: "anchored", pending: "pending", local: "local only" };

const $ = (id) => document.getElementById(id);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

const ui = {
  layoutKey: null,
  layoutId: null,
  slots: new Map(), // slot -> { el, img, url, last }
  sparks: new Map(), // bay -> { row, canvas, val, pose, hover }
  lastEventKey: null,
  lastAuditKey: null,
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
  const v = (camera && ui.view.zooms[camera]) || [1, 0, 0, 0, 0];
  return { z: v[0], px: v[1], py: v[2], rx: v[3] || 0, ry: v[4] || 0 };
}

function setCamPose(camera, z, px, py, rx, ry, immediate = true) {
  if (!camera) return;
  const cz = Math.max(0.5, Math.min(4.0, z));
  const cpx = Math.max(-2.5, Math.min(2.5, px));
  const cpy = Math.max(-2.5, Math.min(2.5, py));
  const crx = Math.max(-4.0, Math.min(4.0, rx));
  const cry = Math.max(-4.0, Math.min(4.0, ry));
  ui.view.zooms[camera] = [cz, cpx, cpy, crx, cry];
  refreshZoomBadge(camera);
  const send = () =>
    sendClient({
      type: "view",
      zoom_cam: camera,
      zoom: cz,
      pan_x: cpx,
      pan_y: cpy,
      rot_x: crx,
      rot_y: cry,
    });
  if (immediate) {
    send();
    return;
  }
  ui.pendingZoomSend = send;
  if (!ui.zoomRaf) {
    ui.zoomRaf = requestAnimationFrame(() => {
      ui.zoomRaf = 0;
      if (ui.pendingZoomSend) {
        const fn = ui.pendingZoomSend;
        ui.pendingZoomSend = null;
        fn();
      }
    });
  }
}

function setCamZoom(camera, z, px, py, immediate = true) {
  const cur = camZoom(camera);
  setCamPose(camera, z, px, py, cur.rx, cur.ry, immediate);
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
    const { z, px, py, rx, ry } = camZoom(camera);
    const modified =
      Math.abs(z - 1) > 0.02 ||
      Math.abs(px) > 0.02 ||
      Math.abs(py) > 0.02 ||
      Math.abs(rx) > 0.02 ||
      Math.abs(ry) > 0.02;
    badge.hidden = !modified;
    if (modified) badge.textContent = `${z.toFixed(1)}× ↺`;
  }
}

function gestureKind(e) {
  // Shift + Left-click (or Middle-click) -> Pan
  if ((e.button === 0 && e.shiftKey) || e.button === 1) return "pan";
  // Right-click (or Ctrl + Left-click on macOS trackpad) -> Rotate / Orbit
  if (e.button === 2 || (e.button === 0 && e.ctrlKey)) return "rotate";
  return null;
}

function attachZoomPan(el, camera, role) {
  // Suppress native right-click menu on sim camera panes (on macOS, contextmenu fires on press).
  el.addEventListener("contextmenu", (e) => e.preventDefault());
  if (!camera || role === "strip") return;

  el.addEventListener(
    "wheel",
    (e) => {
      e.preventDefault();
      e.stopPropagation();
      const cur = camZoom(camera);
      const nextZ = cur.z * Math.exp(-e.deltaY * 0.0018);
      setCamPose(camera, nextZ, cur.px, cur.py, cur.rx, cur.ry, false);
    },
    { passive: false }
  );

  // Prevent HTML5 dragstart whenever Shift+Left, Right-click, Ctrl+Left, or Middle-click is used;
  // plain Left-click keeps HTML5 draggable active to move/pin/swap panes.
  el.addEventListener("mousedown", (e) => {
    if (gestureKind(e) && !(e.target && e.target.closest("button"))) {
      e.preventDefault();
    }
  });

  let drag = null;
  const onMove = (e) => {
    if (!drag || e.pointerId !== drag.id) return;
    const rect = el.getBoundingClientRect();
    const dx = e.clientX - drag.x;
    const dy = e.clientY - drag.y;
    const w = Math.max(rect.width, 1);
    const h = Math.max(rect.height, 1);
    if (drag.kind === "pan") {
      const zScale = Math.max(0.5, drag.z);
      const nx = drag.px + dx / (w * zScale);
      const ny = drag.py + dy / (h * zScale);
      setCamPose(camera, drag.z, nx, ny, drag.rx, drag.ry, false);
    } else {
      const nrx = drag.rx + (dx / w) * 1.8;
      const nry = drag.ry + (dy / h) * 1.4;
      setCamPose(camera, drag.z, drag.px, drag.py, nrx, nry, false);
    }
  };
  const endDrag = (e) => {
    if (!drag || (e && e.pointerId !== undefined && e.pointerId !== drag.id)) return;
    drag = null;
    ui.panningCam = null;
    el.draggable = true;
    el.classList.remove("is-panning", "is-rotating");
    window.removeEventListener("pointermove", onMove);
    window.removeEventListener("pointerup", endDrag);
    window.removeEventListener("pointercancel", endDrag);
  };

  el.addEventListener("pointerdown", (e) => {
    const kind = gestureKind(e);
    if (!kind) return;
    if (e.target && e.target.closest("button")) return;
    e.preventDefault();
    e.stopPropagation();
    const cur = camZoom(camera);
    drag = {
      kind,
      x: e.clientX,
      y: e.clientY,
      px: cur.px,
      py: cur.py,
      rx: cur.rx,
      ry: cur.ry,
      z: cur.z,
      id: e.pointerId,
    };
    ui.panningCam = camera;
    el.draggable = false;
    el.classList.add(kind === "pan" ? "is-panning" : "is-rotating");
    try {
      el.setPointerCapture(e.pointerId);
    } catch {
      // fallback to window listeners below
    }
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", endDrag);
    window.addEventListener("pointercancel", endDrag);
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
        ? `Drop Bay ${bayId} on the side strip to go back to the grid`
        : `Drop Bay ${bayId} here to give it the stage`;
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

function feedTile(bayId, camera, role, bayInfo, inRescue) {
  const isOverview = bayId === 0 || camera === "overview";
  const el = document.createElement("div");
  el.className = `feed feed--${role}`;
  el.dataset.bay = String(bayId);
  el.dataset.camera = camera || "";
  el.tabIndex = 0;
  const view = isOverview ? "overview" : (camera ? camera.split("/")[1] : "cctv");
  const isPinned = role === "main";
  if (isPinned) el.classList.add("is-pinned");
  const title = isOverview ? "Plant overview" : `Bay ${bayId}`;
  const tips = [BAY_ZONE[bayId], bayInfo && bayInfo.replayed ? "replayed routine cycle" : null].filter(Boolean);
  if (tips.length) el.title = tips.join(" · ");
  el.innerHTML = `
    <div class="feed__top">
      <span class="feed__name">${title}</span>
      <span class="feed__state">${isOverview ? "Live" : ""}</span>
      <button type="button" class="feed__zoom" hidden title="Camera moved (scroll: zoom, Shift+drag: pan, right-drag: orbit). Click to reset.">1.0×</button>
      <span class="feed__conf"><span class="feed__conf-label">C(t)</span><span class="feed__conf-val"></span></span>
    </div>
    <div class="feed__pic">
      <img alt="" data-cam="${camera || ""}">
      <div class="feed__tools"></div>
      <div class="feed__bottom">
        ${isOverview ? "" : '<span class="feed__belt" hidden></span>'}
        <span class="feed__armed" hidden></span>
        <span class="feed__meta">${isPinned ? VIEW_TEXT[view] || "" : ""}</span>
      </div>
      <div class="feed__lost"><strong>Signal lost</strong><span>Waiting for the camera to reconnect</span></div>
    </div>`;

  const stop = (fn) => (ev) => { ev.stopPropagation(); fn(); };
  el.querySelector(".feed__zoom").addEventListener("click", stop(() => resetCamZoom(camera)));

  // Tools: camera angle (grid tiles; the stage has its own switch), zoom, fault, stage.
  const tools = el.querySelector(".feed__tools");
  const tool = (label, tip, run, cls = "") => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = `ftool${cls}`;
    btn.textContent = label;
    btn.title = tip;
    btn.addEventListener("click", stop(run));
    tools.append(btn);
    return btn;
  };
  if (role === "tile" && !isOverview) {
    for (const [camKey, label] of CAM_BTNS) {
      tool(label, `${label} camera`, () => setBayCamera(bayId, camKey), view === camKey ? " is-active" : "");
    }
  }
  if (role !== "strip" && camera) {
    let first = true;
    for (const [label, mult, tip] of [["−", 1 / 1.25, "Zoom out"], ["+", 1.25, "Zoom in"]]) {
      tool(label, `${tip} (or scroll wheel)`, () => {
        const cur = camZoom(camera);
        setCamZoom(camera, cur.z * mult, cur.px, cur.py);
      }, first ? " ftool--gap" : "");
      first = false;
    }
  }
  if (!isOverview && role !== "strip" && !inRescue && bayInfo && !bayInfo.background && bayInfo.state !== "STANDBY") {
    tool("Jam", `Inject a flap jam on Bay ${bayId}`, () => sendFault(bayId, "flap"), " ftool--gap");
  }
  if (!isPinned) {
    tool("Focus", `Give Bay ${bayId} the stage (or double-click)`, () => setLayoutMode("pin", bayId), " ftool--gap");
  }

  if (!isOverview) {
    el.addEventListener("click", () => selectBay(bayId));
    el.addEventListener("dblclick", (e) => { e.stopPropagation(); togglePinBay(bayId); });
    el.addEventListener("keydown", (e) => { if (e.key === "Enter") selectBay(bayId); });
    attachDragEvents(el, bayId, role);
  } else {
    el.addEventListener("dblclick", (e) => { e.stopPropagation(); setLayoutMode("grid", null); });
  }
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
    <div class="feed__pic"><div class="feed__nosignal"><strong>No camera</strong><span>This bay is not running in the simulation</span></div></div>`;
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

  wall.classList.remove("is-rescue", "is-pin", "is-overview");

  if (mains.length > 0) {
    const focusBay = mains[0].bay;
    const isOverviewFocus = focusBay === 0 || mains[0].camera === "overview";
    const isRescueFocus = !isOverviewFocus && !!(t.rescue && t.rescue.bay === focusBay);
    wall.classList.add(isOverviewFocus ? "is-overview" : (isRescueFocus ? "is-rescue" : "is-pin"));

    // Stage header. Rescue: bay, clutch, timer first (the things to watch), then the camera
    // switch and the one way back to the grid.
    const head = document.createElement("div");
    head.className = "takeover";
    const h2 = document.createElement("h2");
    if (isOverviewFocus) {
      h2.textContent = "Plant overview";
    } else {
      const focusInfo = bayMap.get(focusBay);
      const refusedBay = focusInfo && focusInfo.state === "OOD_ESCALATED";
      const sub = document.createElement("span");
      sub.textContent = isRescueFocus ? (refusedBay ? "refused stall" : "operator rescue") : "in focus";
      h2.append(`Bay ${focusBay} `, sub);
      if (BAY_ZONE[focusBay]) h2.title = BAY_ZONE[focusBay];
    }
    head.append(h2);

    if (isRescueFocus) {
      const clutch = document.createElement("span");
      clutch.className = "clutch";
      clutch.id = "clutch";
      const timer = document.createElement("span");
      timer.className = "takeover__timer";
      const value = document.createElement("b");
      value.id = "rescue-timer";
      timer.append("Rescue time", value);
      head.append(clutch, timer);
    }

    const back = document.createElement("button");
    back.type = "button";
    back.className = "vbtn vbtn--solo";
    back.textContent = "Back to grid";
    back.title = "Show all 4 bays (Esc when a bay is pinned)";
    back.addEventListener("click", () => setLayoutMode("grid", null));
    if (!isOverviewFocus) {
      const camGroup = document.createElement("div");
      camGroup.className = "seg takeover__views";
      camGroup.setAttribute("role", "group");
      camGroup.setAttribute("aria-label", `Bay ${focusBay} camera`);
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
    } else {
      back.classList.add("takeover__views");
    }
    head.append(back);
    wall.append(head);

    const stage = document.createElement("div");
    stage.className = `focus-stage ${mains.length > 1 ? "is-split" : "is-single"}`;
    for (const f of mains) {
      const el = feedTile(f.bay, f.camera, "main", bayMap.get(f.bay), isRescueFocus);
      stage.append(el);
      registerSlot(f.slot, el);
    }
    wall.append(stage);

    const strip = document.createElement("div");
    strip.className = "strip";
    strip.addEventListener("dragover", (e) => {
      if (ui.dragRole === "main") {
        e.preventDefault();
        strip.classList.add("is-drop-target");
      }
    });
    strip.addEventListener("dragleave", () => strip.classList.remove("is-drop-target"));
    strip.addEventListener("drop", (e) => {
      if (ui.dragRole === "main") {
        e.preventDefault();
        strip.classList.remove("is-drop-target");
        setLayoutMode("grid", null);
      }
    });
    const otherBays = order.filter((b) => b !== focusBay);
    for (const bay of otherBays) {
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
    const bayNum = Number(el.dataset.bay);
    if (bayNum === 0) {
      const time = el.querySelector(".feed__time");
      if (time) time.textContent = t.wall;
      continue;
    }
    const b = bays.get(bayNum);
    if (!b) continue;
    if (el.dataset.state !== "STANDBY" || b.state !== "STANDBY") el.dataset.state = b.state;
    el.classList.toggle("is-selected", t.selected_bay === b.id && t.mode !== "rescue");
    const state = el.querySelector(".feed__state");
    if (state) state.textContent = STATE_TEXT[b.state] || b.label;
    const conf = el.querySelector(".feed__conf-val");
    if (conf) conf.textContent = b.state === "STANDBY" ? "" : b.confidence.toFixed(2);
    const time = el.querySelector(".feed__time");
    if (time) time.textContent = t.wall;
    const armed = el.querySelector(".feed__armed");
    if (armed) {
      const f = b.fault;
      armed.hidden = !f;
      armed.textContent = f ? (FAULT_TEXT[f.phase] || {})[f.kind] || "Fault" : "";
    }
    const belt = el.querySelector(".feed__belt");
    if (belt) {
      const mode = b.conveyor && b.conveyor.mode;
      belt.hidden = !mode;
      if (mode) {
        belt.dataset.mode = mode;
        belt.textContent = BELT_TEXT[mode] || mode;
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
  // The bay being rescued now, then the bays waiting, first in first out.
  const rows = [];
  if (t.rescue) rows.push({ bay: t.rescue.bay, pos: "", note: "In rescue now", active: true });
  t.queue.forEach((bay, i) => rows.push({ bay, pos: String(i + 1), note: i === 0 ? "Next" : "Waiting" }));
  const key = rows.map((r) => `${r.bay}:${r.pos}:${r.note}`).join("|") || "empty";
  if (key === ui.queueKey) return;
  ui.queueKey = key;
  const q = $("queue");
  q.replaceChildren();
  if (!rows.length) {
    const li = document.createElement("li");
    li.className = "is-empty";
    li.textContent = "No bays waiting";
    q.append(li);
    return;
  }
  for (const r of rows) {
    const li = document.createElement("li");
    if (r.active) li.className = "is-active";
    const span = (cls, text) => {
      const el = document.createElement("span");
      el.className = cls;
      el.textContent = text;
      return el;
    };
    if (r.pos) li.append(span("queue__pos", r.pos));
    li.append(`Bay ${r.bay}`, span("queue__note", r.note));
    q.append(li);
  }
}

function ensureSparks() {
  if (ui.sparks.size) return;
  const box = $("sparks");
  for (const bay of BAYS) {
    const row = document.createElement("div");
    row.className = "spark";
    row.innerHTML = `<span class="spark__name">Bay ${bay}</span><canvas role="img"></canvas><span class="spark__val"></span>`;
    box.append(row);
    const s = {
      row,
      canvas: row.querySelector("canvas"),
      val: row.querySelector(".spark__val"),
      hover: null, // 0..1 across the canvas while the pointer is over it
    };
    s.canvas.addEventListener("pointermove", (e) => {
      const rect = s.canvas.getBoundingClientRect();
      s.hover = Math.max(0, Math.min(1, (e.clientX - rect.left) / Math.max(rect.width, 1)));
      if (ui.telemetry) updateSparks(ui.telemetry);
    });
    s.canvas.addEventListener("pointerleave", () => {
      s.hover = null;
      if (ui.telemetry) updateSparks(ui.telemetry);
    });
    ui.sparks.set(bay, s);
  }
}

const SPARK_SAMPLES = 150;

// C(t) trace: one neutral line over the dashed stall threshold, a dot on the latest sample.
// C(t) dips under the threshold briefly on every routine cycle, so a dip alone is never
// painted as an alarm: `alarm` (the bay is actually stalled) colours the dot.
// Returns the value under the pointer while hovering, else null.
function drawSpark(canvas, history, threshold, hover, alarm) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  if (!w || !h) return null;
  if (canvas.width !== Math.round(w * dpr)) canvas.width = Math.round(w * dpr);
  if (canvas.height !== Math.round(h * dpr)) canvas.height = Math.round(h * dpr);
  const g = canvas.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);
  const pad = 4;
  const y = (v) => h - pad - Math.max(0, Math.min(1, v)) * (h - 2 * pad);
  const yT = y(threshold);
  const bad = css("--stalled");
  g.strokeStyle = css("--faint");
  g.lineWidth = 1;
  g.setLineDash([4, 3]);
  g.beginPath();
  g.moveTo(0, Math.round(yT) + 0.5);
  g.lineTo(w, Math.round(yT) + 0.5);
  g.stroke();
  g.setLineDash([]);
  if (history.length < 2) return null;
  const xEnd = w - pad - 1;
  const step = xEnd / (SPARK_SAMPLES - 1);
  const x = (i) => xEnd - (history.length - 1 - i) * step;
  const trace = (color, top, height) => {
    g.save();
    g.beginPath();
    g.rect(0, top, w, height);
    g.clip();
    g.strokeStyle = color;
    g.lineWidth = 2;
    g.lineJoin = "round";
    g.beginPath();
    history.forEach((v, i) => (i ? g.lineTo(x(i), y(v)) : g.moveTo(x(i), y(v))));
    g.stroke();
    g.restore();
  };
  trace(css("--text"), 0, h);
  const dot = (i) => {
    const v = history[i];
    g.fillStyle = css("--panel");
    g.beginPath();
    g.arc(x(i), y(v), 5, 0, Math.PI * 2);
    g.fill();
    g.fillStyle = alarm ? bad : css("--text");
    g.beginPath();
    g.arc(x(i), y(v), 3.5, 0, Math.PI * 2);
    g.fill();
  };
  if (hover === null) {
    dot(history.length - 1);
    return null;
  }
  const first = x(0);
  const i = Math.max(0, Math.min(history.length - 1, Math.round((hover * w - first) / step)));
  g.strokeStyle = css("--muted");
  g.lineWidth = 1;
  g.beginPath();
  g.moveTo(Math.round(x(i)) + 0.5, 0);
  g.lineTo(Math.round(x(i)) + 0.5, h);
  g.stroke();
  dot(i);
  return history[i];
}

function updateSparks(t) {
  ensureSparks();
  $("threshold-note").textContent = `stall below ${t.threshold.toFixed(1)}`;
  for (const b of t.bays) {
    const s = ui.sparks.get(b.id);
    if (!s) continue;
    const standby = b.state === "STANDBY";
    const alarm = ["STALLED", "QUEUED", "OOD_ESCALATED"].includes(b.state);
    const hovered = standby ? null : drawSpark(s.canvas, b.history, t.threshold, s.hover, alarm);
    const shown = hovered ?? b.confidence;
    s.row.classList.toggle("is-standby", standby);
    s.row.classList.toggle("is-low", alarm && hovered === null);
    s.val.textContent = standby ? "–" : shown.toFixed(2);
    s.canvas.setAttribute(
      "aria-label",
      standby ? `Bay ${b.id} on standby` : `Bay ${b.id} confidence ${b.confidence.toFixed(2)}, ${STATE_TEXT[b.state] || b.label}`
    );
    const parts = [];
    if (b.pose_sigma_mm != null) {
      const yaw = b.pose_sigma_yaw_deg != null ? `/±${b.pose_sigma_yaw_deg.toFixed(1)}°` : "";
      parts.push(`Pose est. ±${Math.round(b.pose_sigma_mm)} mm${yaw}`);
    }
    if (b.touch_correction_mm != null) parts.push(`touch-corrected ${b.touch_correction_mm.toFixed(1)} mm`);
    const pose = parts.join(", "); // detail on demand: the row tooltip
    if (s.row.title !== pose) s.row.title = pose;
  }
}

function updateGate(t) {
  const box = $("gate");
  const g = t.gate;
  const span = (cls, text) => {
    const el = document.createElement("span");
    el.className = cls;
    el.textContent = text;
    return el;
  };
  // Nothing to say until a skill has been fitted: the panel stays out of the rail.
  $("panel-gate").hidden = !g || ((g.ok === null || g.ok === undefined) && g.fit_ms == null);
  if (!g || g.ok === null || g.ok === undefined) {
    delete box.dataset.ok;
    box.replaceChildren(
      span("is-empty", g && g.fit_ms != null
        ? `Skill fitted in ${g.fit_ms.toFixed(0)} ms, checking it in simulation`
        : "No skill learned yet")
    );
    return;
  }
  box.dataset.ok = String(!!g.ok);
  const timing = [
    !g.ok && g.worst_failure ? g.worst_failure : null,
    g.fit_ms != null ? `fitted in ${g.fit_ms.toFixed(0)} ms` : null,
    g.elapsed_s != null ? `checked in ${g.elapsed_s.toFixed(2)} s` : null,
  ].filter(Boolean).join(", ");
  box.replaceChildren(
    span("gate__big", `${g.passed}/${g.total}`),
    span("gate__verdict", g.ok ? "Passed, shared with the fleet" : "Failed, not shared"),
    span("gate__line", timing)
  );
  if (g.conditions) box.append(span("gate__conds", g.conditions));
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
  const refusedBay = first("OOD_ESCALATED");
  if (refusedBay) {
    // Refused but not yet handed to the operator: the banner must not read as "all well".
    const r = refusedBay.id;
    return {
      tone: "ood", tag: `Refused · Bay ${r}`,
      head: `Bay ${r} refused the fleet skill`,
      hint: `This stall is unlike the learned jam, so no skill was replayed. Select Bay ${r} (${kbd(r)}) to take over, or reset the cells (${kbd("R")}).`,
      keys: [48 + r],
      actions: [{ label: `Select Bay ${r}`, accent: true, run: () => selectBay(r) }],
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
  return {
    tone: "idle",
    tag: t.gate && t.gate.ok ? "Fleet skill active" : "Nominal",
    head: "All active bays are running on their own",
    hint: "",
    keys: [],
    actions: [],
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
      // Primary action first, so it sits in the same place whatever else is offered.
      if (act.accent) actBox.prepend(btn);
      else actBox.append(btn);
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
  const ovBtn = $("view-overview");
  if (ovBtn) ovBtn.classList.toggle("is-active", vmode === "overview");
  $("view-grid").classList.toggle("is-active", vmode === "grid");
  $("view-auto").classList.toggle("is-active", vmode === "auto");
}

// Fault injection is a demo control, so it lives in the Demo tools drawer, not in the banner.
function updateDemoFaults(t) {
  const faults = t.bays
    .filter((b) => b.state !== "STANDBY" && !b.background)
    .map((b) => ({ label: `Jam Bay ${b.id}`, bay: b.id, kind: "flap" }));
  if (faults.some((f) => f.bay === 2)) faults.push({ label: "Pin Bay 2 (unseen fault)", bay: 2, kind: "pin" });
  const key = faults.map((f) => f.label).join("|");
  if (key === ui.demoKey) return;
  ui.demoKey = key;
  const box = $("demo-faults");
  box.replaceChildren();
  for (const f of faults) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "actbtn";
    btn.textContent = f.label;
    btn.addEventListener("click", () => sendFault(f.bay, f.kind));
    box.append(btn);
  }
}

function updateDummy(t) {
  const d = t.dummy;
  $("dummy-panel").hidden = !d;
  if (!d) {
    $("dummy-meta").textContent = "";
    return;
  }
  const DUMMY_MODE_TEXT = { manual: "manual", auto: "auto co-pilot", off: "off" };
  $("dummy-meta").textContent = `(dummy glove ${DUMMY_MODE_TEXT[d.mode] || d.mode})`;
  $("dummy-panel").title =
    d.transport === "studio_udp" ? `Studio UDP stand-in${d.pid ? `, PID ${d.pid}` : ""}` : "Stand-in";
  for (const btn of $("dummy-panel").querySelectorAll("[data-dummy-mode]")) {
    btn.classList.toggle("is-active", btn.dataset.dummyMode === d.mode);
  }
  $("dummy-autofaults").checked = !!d.auto_faults;
  $("dummy-manual").hidden = d.mode === "off";
  $("btn-solve").textContent = d.solving ? "Solving…" : "Guided solve";
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
  syncViewBar();
  updateWall(t);
  updateQueue(t);
  updateSparks(t);
  updateGate(t);
  updateLog(t);
  updateBrief(t);
  updateDemoFaults(t);
  updateDummy(t);
  updateActions(t);
  updateCompliance(t);
  // One advisory line, shown only while the advisor has something to say.
  const advice = (t.reasoning || "").trim();
  $("panel-advisor").hidden = !advice || /\bidle\b/i.test(advice);
  if ($("reasoning").textContent !== advice) $("reasoning").textContent = advice;
  $("live").title = t.unoq || "";
}

function buildAuditItem(comp) {
  const kind = comp.kind || "AUDIT_EVENT";
  const item = document.createElement("div");
  item.className = "audit__item";
  item.dataset.kind = kind;

  const kindEl = document.createElement("span");
  kindEl.className = "audit__kind";
  kindEl.textContent = AUDIT_KIND_TEXT[kind] || kind.replace(/_/g, " ").toLowerCase();
  kindEl.title = kind;

  const bayEl = document.createElement("span");
  bayEl.className = "audit__bay";
  bayEl.textContent = comp.bay_id ? `Bay ${comp.bay_id}` : "Fleet";

  const st = comp.state || "local";
  const stateEl = document.createElement("span");
  stateEl.className = "audit__state";
  stateEl.dataset.state = st;
  stateEl.textContent = AUDIT_STATE_TEXT[st] || st;

  const hashEl = document.createElement("a");
  hashEl.className = "audit__hash-link";
  const hashText = document.createElement("span");
  hashText.className = "audit__hash";
  hashEl.target = "_blank";
  hashEl.rel = "noopener";

  const digest = comp.digest || null;
  const txSig = comp.tx_signature || null;
  let url = comp.explorer_url || null;
  if (!url && txSig) {
    url = "http" + "s://explorer.solana.com/tx/" + txSig + "?cluster=devnet";
  }

  const hex = digest || txSig;
  const cleanHex = hex ? (hex.startsWith("0x") ? hex : "0x" + hex) : "0x" + "0".repeat(12);
  const shortHex = cleanHex.length > 18 ? `${cleanHex.slice(0, 8)}…${cleanHex.slice(-6)}` : cleanHex;

  if (url) {
    hashEl.href = url;
    hashText.textContent = shortHex;
    hashEl.append(hashText, " Solana Explorer ↗");
    hashEl.title = `Open the on-chain transaction in Solana Explorer\nTx: ${txSig || url}\nDigest: ${digest || "N/A"}`;
  } else {
    hashText.textContent = shortHex;
    hashEl.append(hashText);
    hashEl.title = `Local SHA-256 hash chain, not anchored on-chain yet\n${digest || ""}\n(Click to copy)`;
    hashEl.addEventListener("click", () => {
      if (digest && navigator.clipboard) {
        navigator.clipboard.writeText(digest).then(() => {
          toast("COPY", "Local SHA-256 hash copied", "info");
        }).catch(() => {});
      }
    });
  }

  item.append(kindEl, bayEl, stateEl, hashEl);
  return item;
}

function updateCompliance(t) {
  const history = Array.isArray(t.compliance_history) && t.compliance_history.length
    ? t.compliance_history
    : (t.compliance_detail ? [t.compliance_detail] : []);

  const badge = $("audit-badge");
  const countEl = $("audit-count");
  const list = $("audit-list");

  const key = history.length
    ? history.map((c) => `${c.digest || ""}:${c.state || ""}:${c.tx_signature || ""}`).join("|")
    : "empty";
  if (key === ui.lastAuditKey) return;
  ui.lastAuditKey = key;

  if (countEl) countEl.textContent = history.length ? `(${history.length})` : "";

  if (!history.length) {
    if (badge) {
      badge.textContent = "";
      badge.dataset.state = "idle";
    }
    if (list) {
      list.replaceChildren();
      const empty = document.createElement("p");
      empty.className = "audit__empty";
      empty.id = "compliance";
      empty.textContent = "No compliance events recorded yet";
      list.append(empty);
    }
    return;
  }

  const latest = history[history.length - 1];
  const latestState = latest.state || "local";
  if (badge) {
    badge.textContent = `latest ${AUDIT_STATE_TEXT[latestState] || latestState}`;
    badge.dataset.state = latestState;
  }

  if (list) {
    const wasScrolled = list.scrollTop > 8;
    list.replaceChildren();
    for (const comp of [...history].reverse()) {
      list.append(buildAuditItem(comp));
    }
    if (!wasScrolled) list.scrollTop = 0;
  }
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
  document.body.classList.toggle("is-down", !up); // everything on screen is now last-known state
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
      if (msg.type === "telemetry") {
        render(msg);
        document.dispatchEvent(new CustomEvent("telemetry", { detail: msg }));
      }
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
  const ovBtn = $("view-overview");
  if (ovBtn) {
    ovBtn.addEventListener("click", () =>
      setLayoutMode(ui.view.mode === "overview" ? "auto" : "overview", null)
    );
  }
  $("view-grid").addEventListener("click", () => setLayoutMode("grid", null));
  $("view-auto").addEventListener("click", () => setLayoutMode("auto", null));
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

// Space, the digits, M and R are control keys, so no button or drawer may hold keyboard focus
// after a mouse click, and a Space release must never activate a focused control.
// Enter still activates whatever has focus.
document.addEventListener("mousedown", (event) => {
  if (event.target instanceof Element && event.target.closest("button, summary")) event.preventDefault();
});
window.addEventListener(
  "keyup",
  (event) => {
    if (event.key !== " ") return;
    if (event.target instanceof Element && event.target.closest("button, summary, a")) event.preventDefault();
  },
  true
);

window.addEventListener("resize", () => ui.telemetry && updateSparks(ui.telemetry));

initViewBar();
initDummyControls();
buildKeyBar();
setLink(false);
$("live-text").textContent = "Connecting";
connect();
