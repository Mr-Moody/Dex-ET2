// Second-opinion card (#audit-panel): the silent deep audit of the latest rescue episode.
// The audit model reviews keyframes plus the rescue timeline after the episode ends and tags it
// for the ACT dataset (approve keeps it, flag drops it in QC). It never speaks; this card is
// its only voice in the booth.
//
// Listens for the "telemetry" event console.js dispatches; `detail.audit` is
// AuditReport.to_dict() (status pending | complete | skipped | error) or null, which hides the
// card. The card is built once and updated in place, so 10 Hz telemetry never rebuilds it.
"use strict";

(function () {
  const panel = document.getElementById("audit-panel");
  if (!panel) return;

  const STATUS = {
    pending: { chip: "Reviewing", tone: "pending" },
    complete: { chip: "Done", tone: "complete" },
    skipped: { chip: "Skipped", tone: "skipped" },
    error: { chip: "Error", tone: "error" },
  };

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function setText(node, text) {
    if (node.textContent !== text) node.textContent = text;
  }

  function setData(node, key, value) {
    if (node.dataset[key] !== value) node.dataset[key] = value;
  }

  let parts = null; // built on the first audit, then only updated
  let dismissed = null; // audit key the operator closed; the next audit or status shows again
  let lastAudit = null;
  let reasonsKey = null;

  function build() {
    const title = el("h2", "argon__title");
    const name = el("span", "argon__name", "Second opinion");
    const where = el("span", "panel__note argon__where");
    const chip = el("span", "argon__chip");
    const close = el("button", "argon__close", "×");
    close.type = "button";
    close.title = "Hide until the next audit";
    close.setAttribute("aria-label", "Hide the second-opinion card until the next audit");
    // Never keeps focus, so a later Space press (a control key) is not also a click.
    close.tabIndex = -1;
    close.addEventListener("mousedown", (e) => e.preventDefault());
    close.addEventListener("click", () => {
      if (lastAudit) dismissed = auditKey(lastAudit);
      panel.hidden = true;
    });
    title.append(name, where, chip, close);

    const head = el("div", "argon__head");
    const verdict = el("span", "argon__verdict");
    const conf = el("span", "argon__conf");
    head.append(verdict, conf);

    const note = el("p", "argon__note");
    const reasons = el("ul", "argon__reasons");
    const meta = el("p", "argon__meta");
    panel.append(title, head, note, reasons, meta);
    return { where, chip, verdict, conf, note, reasons, meta };
  }

  function auditKey(a) {
    return `${a.audit_id ?? ""}|${a.status ?? ""}`;
  }

  function headline(a) {
    const frames = a.frame_count ? `${a.frame_count} frames` : "the episode";
    switch (a.status) {
      case "pending":
        return { text: `Reviewing ${frames}…`, tone: "pending" };
      case "skipped":
        return { text: "Audit skipped", tone: "skipped" };
      case "error":
        return { text: "Audit failed, episode kept untagged", tone: "error" };
      default:
        if (a.verdict === "flag") return { text: "Flagged, dropped from ACT", tone: "flag" };
        if (a.verdict === "approve") return { text: "Approved for ACT", tone: "approve" };
        return { text: "No verdict returned", tone: "skipped" };
    }
  }

  function metaLine(a) {
    const bits = [];
    if (a.dataset_tag) bits.push(`dataset: ${a.dataset_tag}`);
    if (a.model) bits.push(a.model);
    if (a.frame_count != null) bits.push(`${a.frame_count} frames`);
    if (typeof a.latency_ms === "number") bits.push(`${(a.latency_ms / 1000).toFixed(1)} s`);
    if (a.wall) bits.push(a.wall);
    return bits.join(" · ");
  }

  function updateAudit(a) {
    if (!a || typeof a !== "object") {
      lastAudit = null;
      if (!panel.hidden) panel.hidden = true;
      return;
    }
    lastAudit = a;
    if (dismissed !== null && dismissed === auditKey(a)) return;
    dismissed = null;
    if (!parts) parts = build();
    if (panel.hidden) panel.hidden = false;

    const status = STATUS[a.status] ? a.status : "error";
    const head = headline({ ...a, status });
    setData(panel, "status", status);
    setData(panel, "tone", head.tone);

    const where = [a.bay != null ? `Bay ${a.bay}` : null, a.skill_id || null].filter(Boolean);
    setText(parts.where, where.join(" · "));
    setText(parts.chip, STATUS[status].chip);
    setData(parts.chip, "status", status);
    setText(parts.verdict, head.text);
    setText(parts.conf, typeof a.confidence === "number" ? `${Math.round(a.confidence * 100)}%` : "");

    const noteText = a.operator_note ? String(a.operator_note) : "";
    setText(parts.note, noteText);
    parts.note.hidden = !noteText;

    const reasons = Array.isArray(a.reasons) ? a.reasons.map(String) : [];
    const key = JSON.stringify(reasons);
    if (key !== reasonsKey) {
      reasonsKey = key;
      parts.reasons.replaceChildren(...reasons.map((r) => el("li", "argon__reason", r)));
      parts.reasons.hidden = reasons.length === 0;
    }
    setText(parts.meta, metaLine(a));
  }

  document.addEventListener("telemetry", (ev) => updateAudit(ev.detail ? ev.detail.audit : null));
  window.updateAudit = updateAudit;
})();
