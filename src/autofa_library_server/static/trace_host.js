// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// AutoFA's host for the vendored EvalRDL trace renderer and its local table source.
const main = document.getElementById("main");
const storeBase = `/traces/${main.dataset.traceStore}/api`;
TRACE_SOURCE.url = (path) => storeBase + path;
TRACE_SOURCE.api = (path, opts) => {
  const url = new URL(path, window.location.origin);
  if (url.pathname.endsWith("/events") && url.searchParams.get("include") === "parts") {
    // Bounded viewer hydration is a host endpoint, independent of TraceSource's full events.
    url.pathname += "/preview";
    path = url.pathname + url.search;
  }
  return api(TRACE_SOURCE.url(path), opts);
};

// The bundled renderer exposes partNode in this classic-script host. Keep its image,
// reasoning, JSON and copy controls; enhance the text toggle for our bounded previews.
const previewPartNode = partNode;
partNode = function(p, base) {
  const node = previewPartNode(p, base);
  if (p.part_type !== "text" || !p.text_truncated) return node;
  const body = node.querySelector("[data-raw]");
  let button = node.querySelector(".moretoggle");
  if (!button) {
    button = document.createElement("button");
    button.className = "moretoggle";
    node.appendChild(button);
  }
  const note = document.createElement("span");
  note.className = "part-text-status muted";
  note.setAttribute("role", "status");
  note.textContent = ` · preview of ${p.text_chars} characters`;
  node.appendChild(note);
  let loaded = false, open = false, pending = null;
  body.classList.add("clamped");
  button.textContent = "▸ show more";
  button.title = "Load the complete text of this part";
  const loadFull = async () => {
    if (loaded) return;
    if (pending) return pending;
    pending = (async () => {
    button.disabled = true;
    button.textContent = "loading full text…";
    note.textContent = "";
    note.setAttribute("role", "status");
    try {
      const query = new URLSearchParams({event_idx: p.event_idx, part_idx: p.part_idx,
        lane: p.lane ?? ""});
      const full = await traceApi(`${base}/part-text?${query}`);
      const identity = ["run_id", "task_id", "repeat", "agent_ref", "lane", "event_idx", "part_idx"];
      if (typeof full.text !== "string" || identity.some(key => full[key] !== p[key])) {
        throw new Error("The returned text does not match this part's source identity");
      }
      body.dataset.raw = full.text;
      body.classList.remove("jsonbody", "clamped");
      if (!prettyJSON(body, unescapeText(full.text))) setText(body, full.text);
      loaded = open = true;
      button.textContent = "▾ show less";
      button.title = "Collapse this part";
      note.textContent = ` · complete text · ${full.chars} characters`;
      if (PRETTY.on) applyPretty();
    } catch (error) {
      button.textContent = "retry full text";
      note.setAttribute("role", "alert");
      note.textContent = ` · Full text could not be loaded: ${error.detail || error.message || error}`;
    } finally {
      button.disabled = false;
    }
    })();
    try { await pending; } finally { pending = null; }
  };
  // Native passage links call loadWhole before searching beyond a bounded preview.
  // Manual expansion and automatic highlighting share one read and the same identity check.
  node.dataset.loadable = "";
  node.loadWhole = loadFull;
  node.expand = () => {
    open = true;
    body.classList.remove("clamped");
    if (loaded) button.textContent = "▾ show less";
  };
  button.onclick = async (event) => {
    event.stopPropagation();
    if (loaded) {
      open = !open;
      body.classList.toggle("clamped", !open);
      button.textContent = open ? "▾ show less" : "▸ show more";
      return;
    }
    await loadFull();
  };
  return node;
};

const COLUMN_LABELS = { task_id: "Task", repeat: "Repeat", status: "Status", reward: "Reward",
  n_events: "Events", n_tool_calls: "Tool calls" };
function table(cols, rows, onclick) {
  const t = document.createElement("table");
  t.innerHTML = "<thead><tr>" + cols.map(c => `<th>${esc(COLUMN_LABELS[c] || c)}</th>`).join("")
    + "</tr></thead>";
  const tb = document.createElement("tbody");
  for (const row of rows) {
    const tr = document.createElement("tr");
    tr.innerHTML = cols.map(c => `<td>${row.__html?.[c] ?? esc(fmt(row[c]))}</td>`).join("");
    if (onclick) { tr.className = "link"; tr.onclick = () => onclick(row); }
    tb.appendChild(tr);
  }
  t.appendChild(tb);
  return t;
}

function fillExports(root) {
  // Original raw-file storage is not exposed by the local table store.
  root.querySelectorAll('a[href^="#/raw/"]').forEach(a => a.remove());
  root.querySelectorAll('a[href$="/export"]').forEach(a => {
    a.textContent = "Export rollout tables";
  });
}
const fileHashFor = () => "#/";
const artHashFor = () => null;

// The page header already names the run's model, benchmark and harness.
function runMetaStrip() {
  return document.createDocumentFragment();
}

async function viewRollouts(runId) {
  const runs = await traceApi("/v1/runs");
  const run = runId ? runs.find(r => r.run_id === runId) : runs[0];
  if (!run) throw new Error("Run not found in this local store");
  const rows = await traceApi(`/v1/runs/${encodeURIComponent(run.run_id)}/rollouts`);
  main.replaceChildren(runMetaStrip(run),
    table(["task_id", "repeat", "status", "reward", "n_events", "n_tool_calls"], rows, r => {
      location.hash = `#/rollout/${encodeURIComponent(run.run_id)}/${encTask(r.task_id)}/${r.repeat}`;
    }));
}

function fullTextLink(seg, params) {
  const link = document.getElementById("full-trace-text");
  const match = (seg[4] || "").match(/^e(\d+)(?:-p(\d+))?(?:~pretty)?$/);
  link.hidden = seg[0] !== "rollout" || !match;
  if (link.hidden) { link.removeAttribute("href"); return; }
  const query = new URLSearchParams({event_idx: match[1]});
  if (match[2] != null) query.set("part_idx", match[2]);
  if (params.has("lane")) query.set("lane", params.get("lane"));
  link.href = TRACE_SOURCE.url(`/v1/rollouts/${encodeURIComponent(seg[1])}`
    + `/${encTask(seg[2])}/${seg[3]}/text?${query}`);
}

async function route() {
  try {
    const full = location.hash.replace(/^#\/?/, "");
    const qi = full.indexOf("?");
    const path = qi < 0 ? full : full.slice(0, qi);
    const params = new URLSearchParams(qi < 0 ? "" : full.slice(qi + 1));
    const seg = path.split("/").map(decodeURIComponent);
    fullTextLink(seg, params);
    if (seg[0] === "rollout" && seg.length >= 4) {
      const key = `#/rollout/${encodeURIComponent(seg[1])}/${encTask(seg[2])}/${seg[3]}`;
      const match = (seg[4] || "").match(/^e(\d+)(?:-p(\d+))?(~pretty)?$/);
      // Passage links must expose their text even when the current rollout or a
      // previously opened one chose the graph. Preserve an explicitly named layout.
      if (match && !params.has("mode") && (
        (typeof TRACE_MODE !== "undefined" && TRACE_MODE === "graph")
        || (typeof TRACE_CHOICE !== "undefined" && TRACE_CHOICE === "graph")
      )) {
        params.set("mode", "linear");
      }
      if (match && params.has("lane")) {
        // A link that names its lane is a full source address: verify it before the jump.
        // One that names none is resolved by the renderer, as the native viewer does.
        const query = new URLSearchParams({ lane: params.get("lane") });
        const target = await traceApi(`/v1/rollouts/${encodeURIComponent(seg[1])}`
          + `/${encTask(seg[2])}/${seg[3]}/events/${match[1]}/parts?${query}`);
        if (match[2] != null && !target.parts.some(p => p.part_idx === +match[2])) {
          throw new Error("Part not found in this event");
        }
      }
      if (key === curRollout && match) {
        if (match[3] || params.get("pretty") === "1") setPretty(true);
        if (params.has("mode") && curTraceSetMode) curTraceSetMode(params.get("mode"));
        if (curTraceSetLanes) curTraceSetLanes(params);
        return jumpToEvent(+match[1], match[2] != null ? +match[2] : null,
          params.get("lane") || "");
      }
      await viewRollout(seg[1], seg[2], seg[3], params);
      // The renderer may make an implicit lane explicit while opening an event.
      fullTextLink(seg, new URLSearchParams(location.hash.split("?")[1] || ""));
      return;
    }
    curRollout = null;
    await viewRollouts(seg[0] === "run" ? seg[1] : null);
  } catch (error) {
    main.replaceChildren($(`<div class="empty">${esc(error.message || error)}</div>`));
  }
}
window.addEventListener("hashchange", route);
route();
