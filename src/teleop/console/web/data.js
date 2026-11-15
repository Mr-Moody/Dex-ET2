// Data page: start / stop episode recording, list saved recordings and their episodes, delete.
// Opened with the "Data" button in the top bar (URL hash #data); the camera wall stays mounted
// underneath and the WebSocket is shared, so switching pages never drops the live link.
// Listens for the "telemetry" event console.js dispatches; the server sends `t.recordings`.
"use strict";

(function () {
  const el = (id) => document.getElementById(id);
  const FPS = 30;
  const SOURCE_TEXT = { glove: "Glove", scripted: "Scripted", dmp: "DMP", act: "ACT" };
  const STATE_TEXT = { recording: "Recording", saving: "Saving", deleting: "Deleting", saved: "Saved" };
  const openIds = new Set(); // recordings the operator expanded; survives list rebuilds
  let listKey = null;
  let latest = null;

  function send(payload) {
    // sendClient is console.js's helper (classic scripts share the global scope)
    return typeof sendClient === "function" && sendClient(payload);
  }

  // ------------------------------------------------------------------ page switching

  function setPage(name) {
    const page = name === "data" ? "data" : "operate";
    document.body.dataset.page = page;
    el("open-data").setAttribute("aria-pressed", String(page === "data"));
    const want = page === "data" ? "#data" : "";
    if (location.hash !== want) history.replaceState(null, "", want || location.pathname + location.search);
  }

  function fromHash() {
    setPage(location.hash === "#data" ? "data" : "operate"); // unknown hash -> operate
  }

  // ------------------------------------------------------------------ formatting

  function parseIso(s) {
    if (!s) return null;
    const d = new Date(s);
    return Number.isNaN(d.getTime()) ? null : d;
  }

  function clock(d) {
    return d ? d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false }) : "--:--:--";
  }

  function day(d) {
    return d ? d.toLocaleDateString([], { weekday: "short", day: "numeric", month: "short" }) : "";
  }

  function span(seconds) {
    const s = Math.max(0, Math.round(seconds));
    const m = Math.floor(s / 60);
    return m ? `${m} min ${String(s % 60).padStart(2, "0")} s` : `${s} s`;
  }

  function node(tag, className, text) {
    const n = document.createElement(tag);
    if (className) n.className = className;
    if (text !== undefined) n.textContent = text;
    return n;
  }

  // ------------------------------------------------------------------ list

  function episodesTable(eps) {
    if (!eps.length) return node("p", "eps__none", "No episodes yet. A rescue or autonomous recovery adds one.");
    const table = node("table", "eps");
    const head = table.createTHead().insertRow();
    for (const h of ["#", "Time", "Bay", "Source", "Fault", "Length", "Result"]) {
      const th = document.createElement("th");
      th.scope = "col";
      th.textContent = h;
      head.append(th);
    }
    const body = table.createTBody();
    for (const e of eps) {
      const row = body.insertRow();
      const length = typeof e.frames === "number" ? `${(e.frames / FPS).toFixed(1)} s` : "-";
      const cells = [
        e.index ?? "-",
        clock(parseIso(e.at)),
        e.bay ?? "-",
        SOURCE_TEXT[e.source] || e.source || "-",
        e.fault || "-",
        length,
      ];
      for (const v of cells) row.insertCell().textContent = String(v);
      const res = row.insertCell();
      res.className = e.success ? "ok" : "bad";
      res.textContent = e.success ? "Cleared" : `Aborted${e.reason ? ` (${e.reason})` : ""}`;
    }
    return table;
  }

  function recordingCard(r) {
    const card = node("details", "rec");
    card.dataset.state = r.state;
    card.open = openIds.has(r.id);
    card.addEventListener("toggle", () => (card.open ? openIds.add(r.id) : openIds.delete(r.id)));

    const sum = node("summary");
    const start = parseIso(r.started_at);
    const end = parseIso(r.ended_at);
    const when = node("span", "rec__when", clock(start));
    when.append(node("small", "", day(start)));

    const eps = r.episodes || [];
    const ok = eps.filter((e) => e.success).length;
    const info = node("span", "rec__sum");
    const count = node("b", "", `${eps.length} episode${eps.length === 1 ? "" : "s"}`);
    info.append(count);
    if (eps.length) info.append(` · ${ok} cleared · ${eps.length - ok} aborted`);
    if (r.state === "saved" && start && end && end >= start) info.append(` · ${span((end - start) / 1000)}`);
    info.append(node("span", "rec__id", r.id));

    const side = node("span", "rec__side");
    side.append(node("span", "rec__state", STATE_TEXT[r.state] || r.state));
    const del = node("button", "actbtn rec__del", "Delete");
    del.type = "button";
    del.disabled = r.state !== "saved";
    del.title = del.disabled ? "Stop the recording and let it save first" : `Delete ${r.id} from disk`;
    del.addEventListener("click", (ev) => {
      ev.preventDefault(); // inside <summary>: don't toggle the card
      ev.stopPropagation();
      const n = eps.length;
      if (confirm(`Delete recording ${r.id} (${n} episode${n === 1 ? "" : "s"}) from disk? This cannot be undone.`)) {
        send({ type: "record", action: "delete", id: r.id });
      }
    });
    side.append(del);

    sum.append(when, info, side);
    card.append(sum, episodesTable(eps));
    return card;
  }

  function renderList(rec) {
    const key = JSON.stringify(rec.recordings);
    if (key === listKey) return;
    listKey = key;
    const list = el("data-list");
    if (!rec.recordings.length) {
      list.replaceChildren(node("p", "datapage__empty", `No recordings yet in ${rec.root}. Press Start recording, then run a rescue.`));
      return;
    }
    list.replaceChildren(...rec.recordings.map(recordingCard));
  }

  // ------------------------------------------------------------------ telemetry

  function render(rec) {
    latest = rec;
    const btn = el("data-rec");
    const dot = el("data-rec-dot");
    if (!rec) {
      btn.disabled = true;
      btn.textContent = "Start recording";
      dot.hidden = true;
      el("data-timer").textContent = "";
      el("data-note").hidden = false;
      el("data-note").textContent = "Recording is not available in this run (headless or viewer display).";
      return;
    }
    const active = rec.recording;
    btn.disabled = !rec.available;
    btn.classList.toggle("is-recording", !!active);
    btn.textContent = active ? "Stop recording" : "Start recording";
    dot.hidden = !active;
    el("open-data-text").textContent = active ? `Data · REC ${active.episodes}` : "Data";
    const started = active && parseIso(active.started_at);
    el("data-timer").textContent = started ? `REC ${span((Date.now() - started) / 1000)} · ${active.episodes} saved` : "";
    const note = rec.unavailable || rec.note || (rec.saving ? `Saving ${rec.saving} recording(s) to disk...` : "");
    el("data-note").hidden = !note;
    el("data-note").textContent = note;
    el("data-sub").textContent = active
      ? `Recording to ${rec.root}/${active.id}. Every rescue and autonomous recovery is saved as an episode.`
      : `Recordings live in ${rec.root}. Each Start to Stop is one LeRobot dataset.`;
    renderList(rec);
  }

  function init() {
    el("open-data").addEventListener("click", () => setPage(document.body.dataset.page === "data" ? "operate" : "data"));
    el("data-back").addEventListener("click", () => setPage("operate"));
    el("data-rec").addEventListener("click", () => {
      if (!latest || !latest.available) return;
      send({ type: "record", action: latest.recording ? "stop" : "start" });
    });
    window.addEventListener("hashchange", fromHash);
    document.addEventListener("telemetry", (ev) => render(ev.detail.recordings || null));
    fromHash();
  }

  init();
})();
