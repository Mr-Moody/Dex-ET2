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
  OOD_ESCALATED: "Needs a human",
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
const VIEW_TEXT = { cctv: "Cell camera", top_ortho: "Top view", side_ortho: "Side view" };

const $ = (id) => document.getElementById(id);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

const ui = {
  layoutKey: null,
  layoutId: null,
  slots: new Map(), // slot -> { el, img, url, last }
  sparks: new Map(), // bay -> { row, canvas, val }
  lastEventKey: null,
  telemetry: null,
};

// ------------------------------------------------------------------ feed tiles

function feedTile(bayId, camera, role) {
  const el = document.createElement("div");
  el.className = `feed feed--${role}`;
  el.dataset.bay = String(bayId);
  el.tabIndex = 0;
  const view = camera ? camera.split("/")[1] : "cctv";
  el.innerHTML = `
    <img alt="">
    <div class="feed__top">
      <span class="feed__name">Bay ${bayId}</span>
      <span class="feed__view">${VIEW_TEXT[view] || ""}</span>
      <span class="feed__state"></span>
      <span class="feed__conf"></span>
    </div>
    <div class="feed__bottom">
      <span class="feed__rec">REC</span>
      <time class="feed__time"></time>
      <span class="feed__replay"></span>
    </div>
    <div class="feed__lost"><strong>Signal lost</strong><span>Waiting for the camera to reconnect</span></div>`;
  el.addEventListener("click", () => selectBay(bayId));
  el.addEventListener("keydown", (e) => {
    if (e.key === "Enter") selectBay(bayId);
  });
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

function registerSlot(slot, el) {
  const prev = ui.slots.get(slot);
  if (prev && prev.url) URL.revokeObjectURL(prev.url);
  ui.slots.set(slot, { el, img: el.querySelector("img"), url: null, last: performance.now() });
}

function buildWall(t) {
  const wall = $("wall");
  for (const s of ui.slots.values()) if (s.url) URL.revokeObjectURL(s.url);
  ui.slots.clear();
  wall.replaceChildren();
  const bySlot = new Map(t.feeds.map((f) => [f.slot, f]));

  if (t.mode === "rescue" && t.rescue) {
    wall.classList.add("is-rescue");
    const head = document.createElement("div");
    head.className = "takeover";
    head.innerHTML = `<h2>You have control of Bay ${t.rescue.bay}</h2>
      <span class="takeover__timer" id="rescue-timer"></span>
      <span class="clutch" id="clutch"></span>`;
    wall.append(head);
    for (const f of t.feeds.filter((x) => x.role === "main")) {
      const el = feedTile(f.bay, f.camera, "main");
      wall.append(el);
      registerSlot(f.slot, el);
    }
    const strip = document.createElement("div");
    strip.className = "strip";
    for (const bay of BAYS) {
      const f = t.feeds.find((x) => x.role === "strip" && x.bay === bay);
      if (f) {
        const el = feedTile(bay, f.camera, "strip");
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
    wall.classList.remove("is-rescue");
    for (const bay of BAYS) {
      const f = [...bySlot.values()].find((x) => x.bay === bay && x.role === "tile");
      if (f) {
        const el = feedTile(bay, f.camera, "tile");
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
  const log = $("log");
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
}

function render(t) {
  ui.telemetry = t;
  const layoutKey = `${t.layout_id}|${t.mode}|${t.feeds.map((f) => `${f.slot}:${f.camera}`).join(",")}`;
  if (layoutKey !== ui.layoutKey) {
    ui.layoutKey = layoutKey;
    buildWall(t);
  }
  ui.layoutId = t.layout_id % 256;
  $("clock").textContent = t.wall;
  const active = t.bays.filter((b) => b.state !== "STANDBY").length;
  $("shift").textContent = `${active} bays, 1 operator on shift`;
  updateWall(t);
  updateQueue(t);
  updateSparks(t);
  updateGate(t);
  updateLog(t);
  $("compliance").textContent = t.compliance;
  $("reasoning").textContent = t.reasoning;
  $("unoq").textContent = t.unoq;
}

// ------------------------------------------------------------------ frames

function onFrame(buf) {
  const bytes = new Uint8Array(buf);
  if (bytes.length < 3) return;
  const slot = bytes[0];
  if (bytes[1] !== ui.layoutId) return; // rendered for a previous layout
  const s = ui.slots.get(slot);
  if (!s || !s.img) return;
  const url = URL.createObjectURL(new Blob([buf.slice(2)], { type: "image/jpeg" }));
  const old = s.url;
  s.url = url;
  s.last = performance.now();
  s.img.onload = () => old && URL.revokeObjectURL(old);
  s.img.src = url;
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

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}${WS_PATH}`);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => {
    backoff = 1000;
    setLink(true);
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

// ------------------------------------------------------------------ keys (GLFW codes)

function sendKey(code) {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "key", code }));
}

function selectBay(bayId) {
  const b = ui.telemetry && ui.telemetry.bays.find((x) => x.id === bayId);
  if (b && b.state !== "STANDBY") sendKey(48 + bayId); // same as pressing 1-4
}

window.addEventListener("keydown", (event) => {
  if (event.repeat || event.ctrlKey || event.metaKey || event.altKey) return;
  const k = event.key;
  let code = null;
  if (k === " ") code = 32;
  else if (k.length === 1 && /[0-9]/.test(k)) code = k.charCodeAt(0);
  else if (k.length === 1 && /[a-z]/i.test(k)) code = k.toUpperCase().charCodeAt(0);
  if (code === null) return;
  if (code === 32) event.preventDefault();
  sendKey(code);
});

window.addEventListener("resize", () => ui.telemetry && updateSparks(ui.telemetry));

setLink(false);
$("live-text").textContent = "Connecting";
connect();
