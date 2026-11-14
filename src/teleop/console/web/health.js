// Hardware health strip: one pill per check (hotspot, gloves, panel) above the camera wall.
// Listens for the "telemetry" event console.js dispatches; hidden while telemetry has no `health`.
"use strict";

(function () {
  const RECONNECT_TARGETS = ["hotspot", "gloves", "panel"];
  const LABELS = { hotspot: "Hotspot", gloves: "Gloves", panel: "Control panel" };
  const strip = document.getElementById("health");
  const last = {}; // name -> previous level, to toast only on a transition to fail

  function sendReconnect(name) {
    // `ws` is the console.js socket (script-scope binding shared across classic scripts)
    if (typeof ws !== "undefined" && ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "reconnect", target: name }));
    }
  }

  function pill(h) {
    const el = document.createElement("div");
    el.className = "hpill";
    el.dataset.level = h.level;
    el.dataset.name = h.name;
    const dot = document.createElement("span");
    dot.className = "hpill__dot";
    dot.setAttribute("aria-hidden", "true");
    const name = document.createElement("span");
    name.className = "hpill__name";
    name.textContent = LABELS[h.name] || h.name;
    const sum = document.createElement("span");
    sum.className = "hpill__sum";
    sum.textContent = h.summary;
    el.append(dot, name, sum);
    if (h.level !== "ok" && h.detail) {
      const d = document.createElement("span");
      d.className = "hpill__detail";
      d.textContent = h.detail;
      el.append(d);
    } else if (h.detail) {
      el.title = h.detail;
    }
    if (h.can_reconnect && h.level !== "ok" && RECONNECT_TARGETS.includes(h.name)) {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "hpill__btn";
      b.disabled = !!h.busy;
      if (h.busy) {
        const spin = document.createElement("span");
        spin.className = "hpill__spin";
        spin.setAttribute("aria-hidden", "true");
        b.append(spin);
      }
      b.append(h.busy ? "Reconnecting" : "Reconnect");
      b.addEventListener("click", () => sendReconnect(h.name));
      el.append(b);
    }
    return el;
  }

  function renderHealth(list) {
    if (!Array.isArray(list)) {
      strip.hidden = true;
      return;
    }
    strip.replaceChildren(...list.map(pill));
    strip.hidden = list.length === 0;
    for (const h of list) {
      if (h.level === "fail" && last[h.name] !== undefined && last[h.name] !== "fail") {
        if (typeof toast === "function") toast("!", `${LABELS[h.name] || h.name}: ${h.summary}`, "warn");
      }
      last[h.name] = h.level;
    }
  }

  document.addEventListener("telemetry", (ev) => renderHealth(ev.detail.health));
})();
