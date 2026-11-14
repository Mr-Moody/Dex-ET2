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
  lastActionSeq: null, // null until the first telemetry, so a reload does not replay old toasts
  briefKey: null,
};

// Keyboard bar. Codes are GLFW key codes (the same ones the MuJoCo viewer receives).
const KEY_BAR = [
  { caps: ["1", "2", "3", "4"], codes: [49, 50, 51, 52], text: "Select bay" },
  { caps: ["Space"], codes: [32], text: "Clutch on / off" },
  { caps: ["F"], codes: [70], text: "Jam Bay 1 flap" },
  { caps: ["G"], codes: [71], text: "Jam Bay 2 flap" },
  { caps: ["P"], codes: [80], text: "Pin Bay 2 bolt" },
  { caps: ["R"], codes: [82], text: "Reset" },
];
const FAULT_TEXT = {
  armed: { flap: "Flap jam armed", pin: "Bolt pin armed" },
  active: { flap: "Flap jammed", pin: "Bolt pinned" },
};
const faultName = (kind) => (kind === "pin" ? "pinned bolt" : "flap jam");
const TOAST_MS = 4500;
const MAX_TOASTS = 4;

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
      <span class="feed__armed" hidden></span>
      <span class="feed__replay"></span>
      <span class="feed__bolt" hidden>
        <span class="feed__bolt-label">Bolt</span>
        <span class="feed__bolt-track"><span class="feed__bolt-fill"></span></span>
        <span class="feed__bolt-val"></span>
      </span>
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

  if (t.mode === "rescue" && t.rescue) {
    wall.classList.add("is-rescue");
    const head = document.createElement("div");
    head.className = "takeover";
    const refusedBay = t.bays.some((b) => b.id === t.rescue.bay && b.state === "OOD_ESCALATED");
    // The brief above says who is driving; this header only names the takeover.
    head.innerHTML = `<h2>Bay ${t.rescue.bay} · ${refusedBay ? "refused stall" : "operator rescue"}</h2>
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

// ------------------------------------------------------------------ what is happening now

const kbd = (k) => `<kbd>${k}</kbd>`;

// One headline for the whole cell, picked by priority, plus the key(s) worth pressing next.
// Everything comes from telemetry, so the brief is always consistent with the tiles.
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
    if (t.rescue.clutch) {
      return {
        tone, tag,
        head: `You are driving Bay ${r}`,
        hint: refused
          ? `This bolt is pinned and will not clear by hand. Release the clutch with ${kbd("Space")}, then press ${kbd("R")} to reset.`
          : `Slide the bolt past ${clear} cm (now ${bolt}), then release the clutch with ${kbd("Space")}.`,
        keys: [32],
      };
    }
    const head = refused
      ? `Bay ${r} refused to guess and needs a human`
      : `Bay ${r} stalled on a jam it has not learned`;
    if (t.selected_bay !== r) {
      return { tone, tag, head, hint: `Press ${kbd(r)} to select it, then ${kbd("Space")} to take control.`, keys: [48 + r] };
    }
    return {
      tone, tag, head,
      hint: refused
        ? `No validated skill is close enough to this stall. Press ${kbd("Space")} to inspect it by hand, or ${kbd("R")} to reset.`
        : `Press ${kbd("Space")} to engage the clutch and drive Bay ${r} with the gloves.`,
      keys: [32],
    };
  }
  const val = first("VALIDATING");
  if (val) {
    return {
      tone: "validating", tag: "Learning",
      head: `Turning Bay ${val.id}'s rescue into a skill`,
      hint: "The UNO Q fits the motion, then three headless simulations must all clear the bolt before the fleet can use it.",
      keys: [],
    };
  }
  const stalled = first("STALLED");
  if (stalled) {
    return {
      tone: "stalled", tag: `Stall · Bay ${stalled.id}`,
      head: `Bay ${stalled.id} stalled`,
      hint: "Matching the stall against the skills the fleet has learned…",
      keys: [],
    };
  }
  const auto = first("AUTONOMOUS_SKILL");
  if (auto) {
    return {
      tone: "auto", tag: `Autonomous · Bay ${auto.id}`,
      head: `Bay ${auto.id} is clearing the jam on its own`,
      hint: `Replaying learned skill${auto.skill ? ` ${auto.skill}` : ""} inside the torque safety tube. No operator needed.`,
      keys: [],
    };
  }
  if (t.queue.length) {
    return {
      tone: "queued", tag: "Queue",
      head: `Bay ${t.queue[0]} is next in the rescue queue`,
      hint: "It is retracted to a safe hover and will be handed to you as soon as the rescue slot frees up.",
      keys: [],
    };
  }
  // A faulted cycle that has not stalled yet: the bay is still pushing, about to hit it.
  const active = bays.filter((b) => b.fault && b.fault.phase === "active" && b.state === "NOMINAL" && b.x_bolt_cm < t.bolt_clear_cm);
  if (active.length) {
    const b = active[0];
    return {
      tone: "armed", tag: `Fault · Bay ${b.id}`,
      head: `Bay ${b.id} is pushing into a ${faultName(b.fault.kind)}`,
      hint: `Watch its confidence: a stall is declared when it stays below ${t.threshold.toFixed(1)} for 300 ms.`,
      keys: [],
    };
  }
  const armed = bays.filter((b) => b.fault && b.fault.phase === "armed");
  if (armed.length) {
    const what = armed.map((b) => `Bay ${b.id} (${faultName(b.fault.kind)})`).join(" and ");
    return {
      tone: "armed", tag: "Fault armed",
      head: `Fault armed on ${what}`,
      hint: "It takes effect when the bay starts its next cycle.",
      keys: [],
    };
  }
  if (t.gate && t.gate.ok) {
    return {
      tone: "idle", tag: "Routine",
      head: "All bays running their routine",
      hint: `A skill is deployed. Press ${kbd("G")} to jam Bay 2 and watch it clear on its own, or ${kbd("P")} for the refused case.`,
      keys: [71, 80],
    };
  }
  return {
    tone: "idle", tag: "Routine",
    head: "All bays running their routine",
    hint: `Press ${kbd("F")} to jam Bay 1's flap and start a rescue.`,
    keys: [70],
  };
}

function updateBrief(t) {
  const s = situation(t);
  const key = `${s.tone}|${s.tag}|${s.head}|${s.hint}`;
  if (key !== ui.briefKey) {
    ui.briefKey = key;
    const brief = $("brief");
    brief.dataset.tone = s.tone;
    $("brief-tag").textContent = s.tag;
    $("brief-head").textContent = s.head;
    $("brief-hint").innerHTML = s.hint; // built only from our own strings and numbers
    brief.classList.remove("is-new");
    void brief.offsetWidth; // restart the arrival animation
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
  updateBrief(t);
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
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "key", code }));
    return true;
  }
  return false;
}

// Every press flashes its key cap at once; the server's answer arrives as a toast.
function pressKey(code, name) {
  flashCap(code);
  if (!sendKey(code)) toast(name, "Not connected to the cell controller, key not sent", "warn");
}

function keyName(code) {
  return code === 32 ? "Space" : String.fromCharCode(code);
}

function selectBay(bayId) {
  const b = ui.telemetry && ui.telemetry.bays.find((x) => x.id === bayId);
  if (b && b.state !== "STANDBY") pressKey(48 + bayId, String(bayId)); // same as pressing 1-4
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
  pressKey(code, keyName(code));
});

window.addEventListener("resize", () => ui.telemetry && updateSparks(ui.telemetry));

buildKeyBar();
setLink(false);
$("live-text").textContent = "Connecting";
connect();
