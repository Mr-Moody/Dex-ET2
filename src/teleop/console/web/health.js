// Hardware health strip: one pill per check (hotspot, gloves, panel) above the camera wall.
// Listens for the "telemetry" event console.js dispatches; hidden while telemetry has no `health`.
"use strict";

(function () {
  const RECONNECT_TARGETS = ["hotspot", "gloves", "panel"];
  const LABELS = { hotspot: "Hotspot", gloves: "Gloves", panel: "Control panel" };
  const strip = document.getElementById("health");
  const pills = new Map(); // name -> the parts of a pill, built once and updated in place
  let names = null; // check names in render order; the strip is only rebuilt when they change
  const last = {}; // name -> previous level, to toast only on a transition to fail

  function sendReconnect(name) {
    // `ws` is the console.js socket (script-scope binding shared across classic scripts)
    if (typeof ws !== "undefined" && ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "reconnect", target: name }));
    }
  }

  function span(className, text) {
    const el = document.createElement("span");
    el.className = className;
    if (text !== undefined) el.textContent = text;
    return el;
  }

  function build(name) {
    const el = document.createElement("div");
    el.className = "hpill";
    el.dataset.name = name;
    const dot = span("hpill__dot");
    dot.setAttribute("aria-hidden", "true");
    const sum = span("hpill__sum");
    el.append(dot, span("hpill__name", LABELS[name] || name), sum);
    const spin = span("hpill__spin");
    spin.setAttribute("aria-hidden", "true");
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "hpill__btn";
    const label = span("hpill__label");
    btn.append(label);
    // Never keeps focus, so a later Space press (a control key) is not also a click.
    btn.addEventListener("mousedown", (e) => e.preventDefault());
    btn.addEventListener("click", () => sendReconnect(name));
    return { el, sum, detail: span("hpill__detail"), btn, label, spin, shown: {} };
  }

  function setText(node, text) {
    if (node.textContent !== text) node.textContent = text;
  }

  // Attach or detach `p[key]` only on a change, so the same element object stays in the DOM.
  function toggle(p, key, on, attach) {
    if (!!p.shown[key] === on) return;
    p.shown[key] = on;
    if (on) attach();
    else p[key].remove();
  }

  // Telemetry arrives at ~10 Hz and summaries carry live numbers (packet rates, heartbeat ages),
  // so pills are updated in place: rebuilding them would swallow a click on Reconnect.
  function update(p, h) {
    if (p.el.dataset.level !== h.level) p.el.dataset.level = h.level;
    setText(p.sum, h.summary);
    const showDetail = h.level !== "ok" && !!h.detail;
    setText(p.detail, showDetail ? h.detail : "");
    const title = !showDetail && h.detail ? h.detail : "";
    if (p.el.title !== title) p.el.title = title;
    toggle(p, "detail", showDetail, () => p.sum.after(p.detail));
    const busy = !!h.busy;
    const showBtn = !!h.can_reconnect && h.level !== "ok" && RECONNECT_TARGETS.includes(h.name);
    if (p.btn.disabled !== busy) p.btn.disabled = busy;
    setText(p.label, busy ? "Reconnecting" : "Reconnect");
    toggle(p, "spin", busy, () => p.btn.prepend(p.spin));
    toggle(p, "btn", showBtn, () => p.el.append(p.btn));
  }

  function renderHealth(list) {
    if (!Array.isArray(list)) {
      strip.hidden = true;
      return;
    }
    const sig = list.map((h) => h.name).join("|");
    if (sig !== names) {
      names = sig;
      for (const name of [...pills.keys()]) {
        if (!list.some((h) => h.name === name)) pills.delete(name);
      }
      for (const h of list) if (!pills.has(h.name)) pills.set(h.name, build(h.name));
      strip.replaceChildren(...list.map((h) => pills.get(h.name).el));
    }
    strip.hidden = list.length === 0;
    for (const h of list) {
      update(pills.get(h.name), h);
      if (h.level === "fail" && last[h.name] !== undefined && last[h.name] !== "fail") {
        if (typeof toast === "function") toast("!", `${LABELS[h.name] || h.name}: ${h.summary}`, "warn");
      }
      last[h.name] = h.level;
    }
  }

  document.addEventListener("telemetry", (ev) => renderHealth(ev.detail.health));
})();
