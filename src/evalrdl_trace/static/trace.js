// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The rollout trace view: lanes, events, parts, tool calls, reasoning, images, the pretty and
// tree layouts, the sidebar tree and the provenance rail. Loaded before the page's own script.
//
// Data: every read goes through TRACE_SOURCE.api(path), which answers the serve's rollout routes
// (/v1/rollouts/{run}/{task}/{repeat}, .../events, .../lanes, .../trace_map, .../provenance,
// .../edges, /v1/runs/{run}). A page that has that JSON another way replaces TRACE_SOURCE.api.
// Every URL the page puts in an href or src (an image's view and download, an artifact's links,
// the rollout's export) goes through TRACE_SOURCE.url(path), which returns the URL to use, or
// null to draw no link. A page served under another root or another host replaces it.
//
// Host: a classic script sharing the page's global scope, loaded after kit.js (the helpers every
// host shares) and before the page's own script. The page provides main, table, runMetaStrip,
// fillExports, fileHashFor, artHashFor and route. The service's index.html, the local debug
// reader (`evalrdl-server trace`, which serves index.html) and viewer.html (`evalrdl-trace serve`) host
// it.

// a trace has no filters, but it has settings — which of the two layouts it is read in
// and whether escapes are decoded — and a reader who says "look at event 12" means the one they
// were looking at. `~pretty` on the event anchor is the older spelling of `pretty`; it is still
// read (every link copied before this carries it) and no longer written.
const ROLLOUT_FIELDS = [
  urlField("mode", "text", "mode"),
  urlField("pretty", "flag", "pretty"),
  // `#…/e12` names an event_idx, and event_idx is per-lane — in a three-lane rollout three
  // different events answer to 12, so the anchor alone can resolve to the wrong one. `lane`
  // is the missing half of the address. It is written only when the rollout has more than one
  // lane, so every link ever copied off a single-lane rollout still means exactly what it did.
  urlField("lane", "text", "lane"),
  // …and the two things a reader DOES to a multi-lane trace, so that "look at this" carries the
  // view they were looking at: `solo` is the one lane shown, `fold` the lanes collapsed to a row.
  urlField("solo", "text", "solo"),
  urlField("fold", "set", "fold"),
];

const TRACE_SOURCE = { api: (path, opts) => api(path, opts), url: (path) => path };
const traceApi = (path, opts) => TRACE_SOURCE.api(path, opts);
const traceUrl = (path) => path == null ? null : TRACE_SOURCE.url(path);

// What a provenance record's words (input, reader, landmark, question) mean: the reader library's
// README, at the URL the host sets as `window.READER_DSL_DOC`. With none, no link is drawn.
const READER_DSL_DOC = window.READER_DSL_DOC || "";
const readerDocTip = (s) => READER_DSL_DOC ? `${s}\nhow a reader works: ${READER_DSL_DOC}` : s;
function readerDocLink(text) {
  if (!READER_DSL_DOC) return document.createTextNode("");
  const a = document.createElement("a"); a.className = "readerdoc";
  a.href = READER_DSL_DOC; a.target = "_blank"; a.rel = "noopener";
  a.textContent = text || "how a reader works"; a.title = READER_DSL_DOC;
  return a;
}

const PART_CAP = 8000;   // cap per-part text in the DOM; huge blobs OOM the tab (Safari)
const TCALL_CAP = 300;   // fold tool-call args past this (code_exec dumps whole scripts)
function partNode(p, base) {
  if (p.part_type === "image" && p.artifact_idx != null && base
      && traceUrl(`${base}/artifacts/${p.artifact_idx}/view`)) {
    // the image RENDERS from /view — pod-served, magic-number-checked, nosniff+CSP —
    // and /download is a separate link right under it. They are two routes because they answer
    // two questions, and because /download is `attachment` for everything including this.
    // loading=lazy + the event virtualization means the fetch happens on scroll-into-view.
    const view = traceUrl(`${base}/artifacts/${p.artifact_idx}/view`);
    const dl = traceUrl(`${base}/artifacts/${p.artifact_idx}/download`);
    const d = document.createElement("div");
    d.className = "part image";
    const img = document.createElement("img");
    img.loading = "lazy";
    img.src = view;
    img.alt = "image" + (p.media_type ? " " + p.media_type : "");
    img.style.cssText = "max-height:260px;max-width:100%;border-radius:4px;cursor:zoom-in";
    img.title = "click to open full size";
    img.onclick = () => window.open(view, "_blank");
    const foot = $(`<div class="mono" style="margin-top:4px;font-size:11px">
      <a href="${view}" target="_blank" rel="noopener">open full size</a> ·
      <a href="${dl}">download</a></div>`);
    img.onerror = () => {
      img.replaceWith($(`<span class="mono" style="opacity:.6">[image${p.media_type ? " " + esc(p.media_type) : ""} — not renderable inline]</span>`));
    };
    d.append(img, foot);
    return d;
  }
  let body = p.part_type === "text" ? (p.text ?? "")
    : `[${p.part_type}${p.media_type ? " " + p.media_type : ""}${p.artifact_idx != null ? " -> artifact " + p.artifact_idx : ""}]`;
  const cut = body.length > PART_CAP;
  if (cut) body = body.slice(0, PART_CAP) + `… [${body.length} chars]`;
  const cls = p.channel === "reasoning" ? "reasoning" : (p.channel === "tool_result" ? "tool_result" : "");
  const header = p.channel === "reasoning"
    ? `thinking · ${p.origin}` + (p.chars != null ? ` · ${p.chars} chars` : "") : null;
  const cell = textCell(cls, body, header);
  const capped = p.part_type === "text" && (cut || /… \[\d+ chars\]$/.test(p.text || ""));
  if (capped && base && p.event_idx != null && p.part_idx != null) wholeText(cell, p, base);
  return cell;
}
// A part whose text was cut, for the DOM here or by the server's point read, reads the rest from
// `.../events/{e}/parts/{p}/text` one window at a time, each appended to what is shown.
function wholeText(cell, p, base) {
  const body = cell.querySelector(".clampbody");
  const b = document.createElement("button"); b.className = "moretoggle wholetext";
  b.textContent = "▸ load the whole text";
  let next = 0, got = "", eof = false;
  const more = () => {
    b.disabled = true;
    const q = new URLSearchParams({ start: String(next) });
    if (p.lane != null) q.set("lane", p.lane);
    return traceApi(`${base}/events/${p.event_idx}/parts/${p.part_idx}/text?${q}`, { timeout: 120000 }).then((r) => {
      got += r.text || ""; next = r.end; eof = !!r.eof;
      body.dataset.raw = got; setText(body, got); body.classList.remove("clamped");
      b.disabled = false;
      if (r.eof) b.remove();
      else b.textContent = `▸ load more (${next.toLocaleString()} of ${r.chars.toLocaleString()} chars)`;
    }, () => { b.disabled = false; b.textContent = "failed — retry"; eof = true; });
  };
  b.onclick = (e) => { e.stopPropagation(); more(); };
  // the rest of the text, every window; what a `?text=` search reads when the shown text misses it
  cell.dataset.loadable = "";
  cell.loadWhole = async () => { while (!eof) await more(); };
  cell.appendChild(b);
}
// preview-then-expand cell (text / tool_result / reasoning): OPEN showing the first ~4 lines with a
// single caret to expand to full. Short cells (≤4 lines, not long) render with no toggle. The text
// lives in one [data-raw] node so prettify + copy-content work in both preview and expanded states.
// Cheap: nothing extra is fetched — the clamp is pure CSS on already-hydrated text.
// A body that PARSES as JSON is coloured by key/value; anything else is left exactly as the
// producer wrote it. The test is a real `JSON.parse`, not a regex on the first character —
// a harness bookkeeping record is JSON and a shell command that merely starts with `{` is not.
// setText stays the fallback, so a parse failure can only lose colour, never text.
function prettyJSON(node, body) {
  let v; try { v = JSON.parse(body); } catch (e) { return false; }
  if (v === null || typeof v !== "object") return false;
  const esc2 = (x) => String(x).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
  const html = JSON.stringify(v, null, 2).replace(
    /("(\\u[\da-fA-F]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(\.\d*)?([eE][+\-]?\d+)?)/g,
    (m) => {
      let c = "jn";
      if (/^"/.test(m)) c = /:$/.test(m) ? "jk" : "js";
      else if (/true|false|null/.test(m)) c = "jb";
      return `<span class="${c}">${esc2(m)}</span>`;
    });
  node.classList.add("jsonbody"); node.innerHTML = html; return true;
}
function textCell(cls, body, header) {
  const d = document.createElement("div");
  d.className = "part txtcell" + (cls ? " " + cls : "");
  if (header) { const h = document.createElement("div"); h.className = "cellhead"; h.textContent = header; d.appendChild(h); }
  const t = document.createElement("div"); t.className = "clampbody";
  t.dataset.raw = body;                      // copy-content and prettify read the raw text
  if (!prettyJSON(t, body)) setText(t, body);
  d.appendChild(t);
  if (body.split("\n").length > 4 || body.length > 280) {
    t.classList.add("clamped");
    const btn = document.createElement("button"); btn.className = "moretoggle";
    let open = false;
    const sync = () => { t.classList.toggle("clamped", !open); btn.textContent = open ? "▾ show less" : "▸ show more"; };
    btn.onclick = (e) => { e.stopPropagation(); open = !open; sync(); };
    d.expand = () => { open = true; sync(); };
    sync();
    d.appendChild(btn);
  }
  return d;
}

function copyText(text, btn) {
  const revert = btn && (() => { const t = btn.dataset.label; btn.textContent = "copied"; setTimeout(() => { btn.textContent = t; }, 1000); });
  if (btn) btn.dataset.label = btn.textContent;
  if (navigator.clipboard?.writeText) navigator.clipboard.writeText(text).then(revert, revert);
  else { const ta = document.createElement("textarea"); ta.value = text; document.body.appendChild(ta);
    ta.select(); try { document.execCommand("copy"); } catch (e) {} ta.remove(); revert && revert(); }
}
// per-part header strip (matches the event-level bar): copy link (full URL incl. #fragment) + copy
// content. A "prettified" chip (CSS-gated on body.pretty-on) sits OUTSIDE the dimmed .pbar so the
// "this cell was altered" signal stays legible; it flags that the anchor carries a ~pretty marker.
function partBar(frag, withCopy) {
  // Always carries the "prettified" chip; the copy link/content buttons are added only when the
  // event has MULTIPLE parts — for a single-part event the event-level .ecopy bar already covers
  // copy, so adding a per-part copy bar there would just duplicate it.
  const head = document.createElement("div"); head.className = "phead";
  const chip = document.createElement("span"); chip.className = "pchip"; chip.textContent = "prettified";
  chip.title = "prettify is on — the copied anchor restores it (view differs from raw)";
  head.append(chip);
  if (withCopy) {
    const bar = document.createElement("div"); bar.className = "pbar";
    const mk = (label, val) => { const b = document.createElement("button"); b.className = "pbtn";
      b.textContent = label; b.onclick = (e) => { e.stopPropagation(); copyText(val(b), b); }; return b; };
    bar.append(
      mk("copy link", (b) => location.origin + location.pathname + withSelection(frag(), b.closest(".partwrap"))),
      mk("copy content", (b) => nodeRaw(b.closest(".partwrap"))));
    head.append(bar);
  }
  return head;
}
// A task id may contain `/` (compute_eval `cublas_applications/154`, switchyard
// `scaleai/swe-bench-pro__…`). One layer of escaping is undone by the ASGI server before the
// route is matched, so the slash reappears as a path separator and the request 404s. Escape
// just that character a second time; every other byte is encoded exactly as before, so URLs
// for slash-free ids are unchanged.
const encTask = (s) => encodeURIComponent(s).replace(/%2F/gi, "%252F");
async function viewRollout(runId, taskId, repeat, params) {
  const base = `/v1/rollouts/${encodeURIComponent(runId)}/${encTask(taskId)}/${repeat}`;
  // The address as the router decoded it, re-encoded once: compared against location.hash.
  const rolloutKey = `#/rollout/${encodeURIComponent(runId)}/${encodeURIComponent(taskId)}/${repeat}`;
  // hydration continues as long as we're on this rollout — an appended /e<idx> event anchor
  // (deep-link) keeps the same key, so it never stops the lazy loading.
  const sameRollout = () => hashPath() === rolloutKey || hashPath().startsWith(rolloutKey + "/");
  curRollout = rolloutKey;
  clearText();
  curTraceSetMode = null;   // until this view has built its toggle; see the fast path in route()
  curEvLookup = null; curLaneReveal = null; curTraceSetLanes = null; curAllBoxes = null;
  // The two module-level switches, given the view's address to live in. Read first so the trace
  // renders in the state the link was copied in rather than flipping into it afterwards.
  // An address that names a setting overrules the session; one that names none keeps the
  // session's, so walking between rollouts does not undo the layout the reader chose — and a
  // link copied out of here names both, so the second reader is not shown the first one's.
  const params_ = params || new URLSearchParams();
  const tstate = readFields(ROLLOUT_FIELDS, params_, {});
  if (!params_.has("mode")) tstate.mode = TRACE_CHOICE;
  // The legacy `~pretty` anchor is read HERE, not at the deep-link parser below: `traceAddress()`
  // rewrites the address into the `?pretty=1` spelling within a few lines, so by the time the
  // parser runs the only signal a cold `…/e12~pretty` link carried is already gone.
  if (!params_.has("pretty")) tstate.pretty = /~pretty$/.test(hashPath()) || PRETTY.on;
  if (!["linear", "tree", "graph"].includes(tstate.mode)) tstate.mode = "";
  TRACE_CHOICE = tstate.mode;
  if (tstate.pretty !== PRETTY.on) setPretty(tstate.pretty);
  const traceQ = (over) => {
    const q = writeFields(ROLLOUT_FIELDS, over ? { ...tstate, ...over } : tstate);
    return q ? "?" + q : "";
  };
  // The event anchor is part of the path, so the address is rebuilt around whatever anchor the
  // reader is on rather than around the rollout — toggling a layout must not scroll them away.
  const traceAddress = () => syncAddress(
    (hashPath().replace(/^#/, "").replace(/~pretty$/, "") || rolloutKey.slice(1)), ROLLOUT_FIELDS, tstate,
    hashTexts().map(t => "text=" + t));
  // A link to an event carries the event's OWN lane, not the view's — `lane` is half the
  // identifier, so it is not a setting the link inherits. `multiLane` is decided below, once the
  // events are in; until then the third argument is simply never passed.
  const evHash = (idx, part, lane) =>
    rolloutKey + "/e" + idx + (part != null ? "-p" + part : "")
    + traceQ(lane !== undefined ? { lane: multiLane ? (lane || "") : "" } : null);
  let multiLane = false;
  traceAddress();   // the session's settings, said out loud, before anything is copied off here
  main.replaceChildren(loadingBox("loading trace…"));
  // Light tiers only — NARROW heads + tool calls + artifacts render immediately; parts stream in
  // below. Tool-call args_full is a texts-tier S3 read (cold + slow) so it is NOT fetched eagerly:
  // the trace renders from the narrow args_head and full args load lazily on expand (loadFullArgs).
  // `provenance` is the eleventh table. Its route is live; when it is not, the absence is
  // REPORTED rather than swallowed — see `provSection`.
  const [ro, events, calls, artifacts, run, tmap, prov, edgeBody, psum] = await Promise.all([
    traceApi(base), traceApi(base + "/events"), traceApi(base + "/tool_calls"),
    traceApi(base + "/artifacts"), traceApi("/v1/runs/" + encodeURIComponent(runId)).catch(() => null),
    traceApi(base + "/trace_map").catch(() => null),
    // The route shipped returning `{rows:[…]}`; an earlier draft returned a bare array, and this
    // page wrapped that draft's shape a second time — so a LIVE route still read as "no route",
    // because `prov.rows` was an object and the array test failed. Accept both spellings.
    traceApi(base + "/provenance").then(r => ({ rows: Array.isArray(r) ? r : (r && r.rows) || [] }))
      .catch(e => ({ err: e && e.message || "unavailable" })),
    // The twelfth table. A stated relation between two events -- the producer's own, not ours.
    traceApi(base + "/edges").catch(() => null),
    // Each model call's request, where the store keeps one. A host without the route has none.
    traceApi(base + "/prompts").catch(() => null)]);
  // What each edge kind means and which end it implies came first, from the store's vocabulary.
  const edgeKinds = (edgeBody && edgeBody.kinds) || {};
  const edges = (edgeBody && edgeBody.edges) || [];
  const promptByKey = {};
  for (const c of (psum && psum.calls) || []) if (c.has_request !== false) promptByKey[evKey(c.lane, c.event_idx)] = c;
  // Everything keyed on an event is keyed on (lane, event_idx). `event_idx` alone collides the
  // moment a rollout holds two lanes — 3 events answer to #12 in the three-lane rollout — and a
  // collision here is silent: the wrong tool call under the wrong event, with no symptom.
  const callsByEvent = {};
  for (const c of calls) (callsByEvent[evKey(c.lane, c.event_idx)] ??= []).push(c);
  // reverse map: a tool_result event -> { fromEvent: emitting event_idx, tools: [names] }. Built
  // from every call's own result_event_idx, so multi-call events link correctly AND a result shared by
  // several calls records them all. Drives the "← from event #N" backlink, the tree-mode result tag,
  // and the tree ordering (resultSet). A call and its result are the same sequence, so the result
  // carries the call's lane.
  const resultMeta = {};
  for (const c of calls) if (c.result_event_idx != null) {
    const m = (resultMeta[evKey(c.lane, c.result_event_idx)] ??= { fromEvent: c.event_idx, tools: [] });
    m.tools.push(c.tool_name || "tool");
  }
  const resultSet = new Set(Object.keys(resultMeta));
  // ── a request's replayed messages ─────────────────────────────────────────────────────
  // A `model_request` replays earlier messages into the model. The producer names them by a
  // MESSAGE id (`m0000_8d08d250cbf3`), and MEASURED over its 18 files, 0 of 535 such ids is an
  // event id -- so that pointer reaches nothing this store holds and the page cannot follow it.
  // The same record carries the bridge that CAN be followed: `consumed_event_ids`, landed as
  // the `consumed` relation, 546 rows and 546 resolved, whose ends are `source_event_id`.
  // So the count unfolds into the EVENTS the messages were built from, not into the messages.
  const evIdxBySid = {};
  for (const ev of events) if (ev.source_event_id != null) evIdxBySid[ev.source_event_id] = ev;
  const consumedBy = {};
  for (const e of (edges || [])) {
    if (e.space !== "consumed" || !e.resolved) continue;
    const target = evIdxBySid[e.to_key];
    if (target) (consumedBy[e.frm_key] ??= []).push(target);
  }
  for (const list of Object.values(consumedBy)) list.sort((a, b) => a.event_idx - b.event_idx);
  // Every other stated edge, at each end this page can find: (lane, input, key) where the key is
  // an event's `source_event_id`. An end keyed by a reader's own input key is named, not linked.
  const evByName = {};
  for (const ev of events) if (ev.source_event_id != null)
    evByName[[ev.lane || "", ev.src_input || "", ev.source_event_id].join("\0")] = ev;
  const relsAt = {};
  for (const e of (edges || [])) {
    if (e.space === "consumed") continue;
    const f = e.frm_task_id === ro.task_id ? evByName[[e.frm_lane || "", e.frm_input || "", e.frm_key].join("\0")] : null;
    const t = e.to_task_id === ro.task_id ? evByName[[e.to_lane || "", e.to_input || "", e.to_key].join("\0")] : null;
    if (f) (relsAt[evKey(f.lane, f.event_idx)] ??= []).push({ dir: "\u2192", e, other: t, name: `${e.to_input || ""}/${e.to_key}` });
    if (t) (relsAt[evKey(t.lane, t.event_idx)] ??= []).push({ dir: "\u2190", e, other: f, name: `${e.frm_input || ""}/${e.frm_key}` });
  }
  // ── the producer's OWN tree ───────────────────────────────────────────────────────────
  // Some producers state a parent pointer per event and the store may carry it. The
  // frontier-agentic-v1 schema writes `parent` on every record: MEASURED over its 18 files,
  // 1,225 of 1,225 non-root pointers resolve, 0 dangle, 50 nodes have more than one child and
  // one has five. That is a real tree and it is the thing a person opens this page to follow.
  // Read from whichever column the store lands it in; absent is the normal case and costs
  // nothing. NEVER a bare event_idx — a reparse renumbers those — so the pointer is matched
  // on `source_event_id`, which is the producer's own id and non-null on every row.
  const PARENT_COLS = ["parent_source_event_id", "parent_event_id", "parent"];
  const keyBySrc = {};
  for (const ev of events) if (ev.source_event_id != null)
    keyBySrc[String(ev.source_event_id)] = evKey(ev.lane, ev.event_idx);
  const parentOf = {}, childrenOf = {}, dangling = [];
  let nParentEdges = 0;
  for (const ev of events) {
    let raw = null;
    for (const c of PARENT_COLS) if (ev[c] != null && ev[c] !== "") { raw = String(ev[c]); break; }
    if (raw == null) continue;
    const k = evKey(ev.lane, ev.event_idx);
    const pk = keyBySrc[raw];
    if (pk == null) { dangling.push({ k, raw }); continue; }   // stated, unresolvable: SAID, not hidden
    if (pk === k) continue;                                    // self-parent is not a tree
    parentOf[k] = pk; (childrenOf[pk] ??= []).push(k); nParentEdges++;
  }
  // A node with more than one child is where the agent went two ways, and it is the single most
  // debugging-relevant thing on the page.
  const branchAt = new Set(Object.keys(childrenOf).filter((k) => childrenOf[k].length > 1));
  // What distinguishes the children of a branch is a question about the DATA, so it is answered
  // from the rows and never from what a fork usually means. If the children differ in a field the
  // page can see, name that field; if they are identical in every field the page holds, say the
  // difference is unmeasured rather than guessing at a compaction, a retry or a sub-agent.
  const evByKey = {};
  for (const ev of events) evByKey[evKey(ev.lane, ev.event_idx)] = ev;
  const BRANCH_FIELDS = ["lane_kind", "kind_raw", "kind", "role", "actor", "model_name", "finish_reason"];
  function branchWhy(k) {
    const kids = (childrenOf[k] || []).map((c) => evByKey[c]).filter(Boolean);
    if (kids.length < 2) return "";
    const differing = BRANCH_FIELDS.filter((f) => new Set(kids.map((e) => String(e[f] ?? ""))).size > 1);
    if (!differing.length) return "the children differ in no field this page holds — unmeasured";
    return differing.map((f) => f + " " + kids.map((e) => String(e[f] ?? "\u2014")).join(" vs ")).join("  ·  ");
  }
  const hasTree = nParentEdges > 0;
  // ── the same call, seen by two lanes ─────────────────────────────────────────────────────
  // `call_id_source` is the PRODUCER's id for a tool call, and two lanes describing the same
  // minutes carry the same one: MEASURED corpus-wide, 39,663 ids appear in exactly 2 lanes and
  // 77 in 3. That is the only cross-lane relationship the store actually holds, and nothing
  // showed it. It is resolved HERE by (id, lane) and never by index: the pairing columns carry
  // an index with no target lane, and trace and session agree on `event_idx` for 0 of the 19
  // ids they share, so an index-resolved cross-lane link would point at the wrong row every
  // time and say nothing about it.
  const lanesByCallId = {};
  for (const c of calls) if (c.call_id_source)
    (lanesByCallId[c.call_id_source] ??= new Set()).add(c.lane ?? "");
  const alsoIn = (c, lane) => {
    const set = c && c.call_id_source ? lanesByCallId[c.call_id_source] : null;
    if (!set || set.size < 2) return null;
    return [...set].filter((l) => l !== (lane ?? "")).map((l) => l || "(unnamed)");
  };
  // "bash, bash" -> "bash ×2"; distinct names -> "bash, python"
  const toolLabel = (tools) => { const n = {};
    for (const t of tools) n[t] = (n[t] || 0) + 1;
    return Object.entries(n).map(([t, c]) => c > 1 ? `${t} ×${c}` : t).join(", "); };
  // memoized full-args map (lane, event_idx, call_idx -> args_full); the slow texts read runs once, only
  // if the user actually expands a folded tool call. Each lane numbers its events from 0.
  let fullArgsProm = null;
  const argsKey = (c) => [c.lane || "", c.event_idx, c.call_idx].join("\0");
  const loadFullArgs = () => (fullArgsProm ??= traceApi(base + "/tool_calls?include=args", { timeout: 120000 })
    .then(rows => { const m = {}; for (const c of rows) m[argsKey(c)] = c.args_full; return m; }));

  // ── the lane model ────────────────────────────────────────────────────────────────────────
  // One pass over the events in the order the API returned them. That order IS the layout: the
  // events endpoint orders by lane then event_idx, so a lane is a contiguous block and the rail
  // needs no topological sort — the list is already the total order the git idiom requires.
  // `first`/`last` are LIST POSITIONS, which is also how hydration is keyed, so the rail's caps
  // and the loader agree without either knowing about the other.
  const laneList = [], laneByKey = {};
  events.forEach((ev, pos) => {
    const k = ev.lane || "";
    let L = laneByKey[k];
    if (!L) {
      L = laneByKey[k] = { key: k, kind: laneKind(k), inst: laneInst(k), slug: laneSlug(k),
                           n: 0, first: pos, last: pos, minIdx: ev.event_idx, maxIdx: ev.event_idx,
                           col: laneList.length, boxes: [], band: null, prov: null };
      L.color = laneColor(L.col);
      laneList.push(L);
    }
    L.n++; L.last = pos;
    if (ev.event_idx < L.minIdx) L.minIdx = ev.event_idx;
    if (ev.event_idx > L.maxIdx) L.maxIdx = ev.event_idx;
  });
  multiLane = laneList.length > 1;
  // No layout chosen: graph where there are lanes to draw, the timeline where there is one or the
  // address names an event.
  const defMode = multiLane && !/\/e\d+(-p\d+)?(~pretty)?$/.test(hashPath()) ? "graph" : "linear";
  TRACE_MODE = tstate.mode || defMode;
  const RAILW = laneRailWidth(laneList.length);
  // A rail column per lane, capped at eight; everything past the cap shares the last column and
  // is marked `~`, rather than the gutter growing without bound (git's own `graph_max_lanes`).
  const laneOverflow = laneList.length > LANE_CAP;

  // ── provenance ────────────────────────────────────────────────────────────────────────────
  // Every stored fact can name which input, in which reader, reading which file, produced it.
  // The rows that land in `events` are one per LANE — measured on this rollout, the three rows
  // with lands='events' carry n_values 63 / 274 / 35, which are exactly the three lane counts,
  // and their `input` values (harbor_session / harbor_subagent / harbor_trajectory) name the
  // lane kinds. So a lane's provenance is the event-landing row that matches it, and an event's
  // provenance is its lane's. Matching is by n_values first (exact, and unique here) and by the
  // kind appearing in the input name second; an ambiguous match is left unmatched rather than
  // guessed, because a wrong source file is worse than none.
  const provRows = Array.isArray(prov && prov.rows) ? prov.rows : null;
  const provErr = prov && prov.err;
  // `taskId` arrives from the router still carrying one layer of escaping — the route spells a
  // literal `/` in a task id as `%252F` and `route()` decodes once, so 566 ids in the corpus reach
  // here as `…%2F…` while the stored `task_id` has the bare slash. Compare against both spellings
  // or every provenance row for those tasks is silently dropped.
  const taskIdRaw = (() => { try { return decodeURIComponent(taskId); } catch (e) { return taskId; } })();
  const provMine = provRows
    ? provRows.filter(r => !r.task_id || r.task_id === taskId || r.task_id === taskIdRaw)
    : null;
  if (provMine) {
    const evRows = provMine.filter(r => r.lands === "events");
    for (const L of laneList) {
      const byCount = evRows.filter(r => r.n_values === L.n);
      const byKind = evRows.filter(r => String(r.input || "").toLowerCase().includes(L.kind));
      L.prov = byCount.length === 1 ? byCount[0] : (byKind.length === 1 ? byKind[0] : null);
    }
  }
  // The sentence an event says when asked where it came from. Kept short enough to live in a
  // `title`, which is how a SINGLE-lane rollout answers the question without gaining any chrome.
  const provLine = (L) => {
    if (!L || !L.prov) return null;
    const p = L.prov;
    return `read by ${p.reader || "?"}${p.producer && p.producer !== p.reader ? " ← " + p.producer : ""}`
         + ` as input ${p.input || "?"} from ${String(p.reads || "?").split("|").join(" + ")}`;
  };

  // ── lane view state ───────────────────────────────────────────────────────────────────────
  // Collapsed lanes and the soloed lane are settings, so they live in the address next to `mode`
  // and `pretty` — "look at this" then carries the view, not just the place.
  const foldSet = () => new Set(tstate.fold || []);
  function setFold(key, on) {
    const s = foldSet(); if (on) s.add(laneId(key)); else s.delete(laneId(key));
    tstate.fold = laneList.filter(L => s.has(laneId(L.key))).map(L => laneId(L.key));   // lane order, not click order
    traceAddress(); applyLaneView();
  }
  function setSolo(id) { tstate.solo = id || ""; traceAddress(); applyLaneView(); }
  function applyLaneView() {
    if (!multiLane) return;
    const s = foldSet(), solo = tstate.solo;
    for (const L of laneList) {
      const hidden = !!solo && laneId(L.key) !== solo;
      const fold = s.has(laneId(L.key));
      if (L.band) {
        L.band.classList.toggle("lane-hidden", hidden);
        L.band.classList.toggle("folded", fold);
        L.band.__sync(fold, laneId(L.key) === solo);
      }
      for (const box of L.boxes) box.classList.toggle("lane-hidden", hidden || fold);
      const t = curTreeNodes && curTreeNodes.__lanes && curTreeNodes.__lanes[L.key];
      if (t) { t.hdr.hidden = hidden; t.grp.hidden = hidden; t.setOpen(!fold && !hidden); }
    }
    for (const g of laneChips) g.__sync();
  }
  // Jumping into a lane that is folded or hidden un-hides it first: a link must never land on a
  // row the current view has collapsed away.
  function revealLane(key) {
    key = key || "";
    if (!multiLane || !laneByKey[key]) return;
    let changed = false;
    if (tstate.solo && tstate.solo !== laneId(key)) { tstate.solo = ""; changed = true; }
    const s = foldSet();
    if (s.has(laneId(key))) { s.delete(laneId(key)); tstate.fold = laneList.filter(L => s.has(laneId(L.key))).map(L => laneId(L.key)); changed = true; }
    if (changed) { traceAddress(); applyLaneView(); }
  }
  curLaneReveal = revealLane;
  function gotoLane(L) { revealLane(L.key); location.hash = evHash(events[L.first].event_idx, null, L.key); }

  const laneChips = [];
  function laneBar() {
    const bar = document.createElement("div"); bar.className = "lanebar";
    bar.dataset.tour = "trace-lanes";
    const lbl = document.createElement("span"); lbl.className = "lbl";
    lbl.textContent = `${laneList.length} lanes`;
    lbl.title = "a lane is one ordered sequence inside this rollout; each numbers its events from 0";
    bar.appendChild(lbl);
    for (const L of laneList) {
      const g = document.createElement("span"); g.className = "lgrp";
      g.style.setProperty("--c", L.color);
      g.dataset.lane = L.key;
      const chip = document.createElement("button"); chip.className = "lchip"; chip.dataset.act = "jump";
      chip.title = provLine(L) ? readerDocTip(`jump to the first event of ${L.key}\n${provLine(L)}`)
                               : `jump to the first event of ${L.key}`;
      const sw = document.createElement("i"); sw.className = "sw";
      const nm = document.createElement("span"); nm.textContent = laneName(L.key);
      const n = document.createElement("span"); n.className = "n"; n.textContent = L.n;
      chip.append(sw, nm, n);
      chip.onclick = () => gotoLane(L);
      const bf = document.createElement("button"); bf.className = "lbtn"; bf.dataset.act = "fold"; bf.textContent = "–";
      bf.onclick = () => setFold(L.key, !foldSet().has(laneId(L.key)));
      const bs = document.createElement("button"); bs.className = "lbtn"; bs.dataset.act = "solo"; bs.textContent = "only";
      bs.onclick = () => setSolo(tstate.solo === laneId(L.key) ? "" : laneId(L.key));
      g.append(chip, bf, bs);
      // hovering a chip lights its rows and dims the rest — the cheapest way to answer "which of
      // these 372 rows are the sub-agent's" without moving anything.
      const hot = (on) => {
        // Nothing to point at if the lane is folded away or another lane is soloed — dimming the
        // page to highlight nothing is worse than not reacting.
        const live = on && !foldSet().has(laneId(L.key)) && !(tstate.solo && tstate.solo !== laneId(L.key));
        traceMain.classList.toggle("dimming", live);
        for (const box of L.boxes) box.classList.toggle("hot", live);
        if (L.band) L.band.classList.toggle("hot", live);
      };
      g.onmouseenter = () => hot(true); g.onmouseleave = () => hot(false);
      g.__sync = () => {
        const folded = foldSet().has(laneId(L.key)), solo = tstate.solo === laneId(L.key);
        g.classList.toggle("folded", folded);
        g.classList.toggle("solo", solo);
        bf.classList.toggle("on", folded);
        bf.title = folded ? `expand ${L.key}` : `collapse ${L.key} to one row`;
        bs.classList.toggle("on", solo);
        bs.title = solo ? "show all lanes again" : `show only ${L.key}`;
      };
      g.__sync();
      laneChips.push(g); bar.appendChild(g);
    }
    const all = document.createElement("button"); all.className = "lall"; all.textContent = "reset";
    all.title = "show every lane, expanded";
    all.onclick = () => { tstate.solo = ""; tstate.fold = []; traceAddress(); applyLaneView(); };
    bar.appendChild(all);
    return bar;
  }


  main.replaceChildren($(`<div class="crumb"><a href="#/">runs</a> /
      <a href="#/run/${encodeURIComponent(runId)}">${esc(runId)}</a> / ${esc(unesc(taskId))} · repeat ${esc(repeat)}</div>
    <div class="bar">
      ${statusPills(ro)}
      <span class="mono">${bits([
        ro.reward == null ? "" : `reward=${fmt(ro.reward)}`,
        esc(ro.record_schema || ""),
        ro.n_events == null ? "" : `${fmt(ro.n_events)} events${multiLane ? ` in ${laneList.length} lanes` : ""}`,
        ro.n_tool_calls == null ? "" : `${fmt(ro.n_tool_calls)} tool calls`,
      ])}</span>
      <span class="qbox tsearch"></span>
      <label class="toggle"><input type="checkbox" class="js-pretty"> prettify <span style="opacity:.6">(decode \\n \\t \\uXXXX)</span></label>
      <span class="rawlinks">
        ${traceUrl(`${base}/export`) ? `<a class="btnlink" href="${esc(traceUrl(`${base}/export`))}"
           title="this rollout's rows, one parquet file per table">Export rows (.zip)</a>` : ""}
        <span class="rawlinks" data-exports="rollout" data-base="${base}"></span>
      </span>
      <span class="rawlinks">
        <a class="btnlink primary" href="#/raw/${encodeURIComponent(runId)}">Browse raw files</a>
        ${traceUrl(`/v1/runs/${encodeURIComponent(runId)}/raw/download`) ? `<a class="btnlink"
           href="${esc(traceUrl(`/v1/runs/${encodeURIComponent(runId)}/raw/download`))}"
           title="the whole run as one archive, byte for byte">Download full run</a>` : ""}
      </span>
    </div>`));
  fillExports(main);
  if (ro.n_events > 0) mountTraceSearch(main, base, evHash);
  else main.querySelector(".tsearch")?.remove();
  if (run) main.appendChild(runMetaStrip(run));
  if (!(ro.n_events > 0)) main.appendChild(emptyBox(ro));
  main.appendChild(provSection());

  // ── provenance, on the page ───────────────────────────────────────────────────────────────
  // One collapsed line, on every rollout including the single-lane ones: the eleventh table says
  // which input, in which reader, reading which file, produced each row stored here. When the
  // route is missing the line says THAT, with the contract it needs — an absent panel would read
  // as "this rollout has no provenance", which is a different and false statement.
  // ── a rollout that holds nothing says why, or says that it does not know ──────────────
  // `status` is OUR ingest verdict and it is `ok` on every one of the 5,043 rollouts that
  // hold zero events — it means "we stored what we were given", never "the agent ran".
  // Rendering it alone over an empty body told a person the rollout was fine. The
  // producer's own verdict is a different column: MEASURED, 4,954 of those 5,043 say
  // `unobserved` and 45 say `unhealthy`, which the page was asserting as `ok`. The
  // remaining 44 are genuinely unexplained and must SAY so; an empty panel reads as "the
  // agent did nothing", which is the false statement this box exists to stop.
  function statusPills(ro) {
    const hv = ro.health_verdict;
    const empty = !(ro.n_events > 0);
    const bad = hv === "unhealthy" || (empty && !hv);
    const parts = [`<span class="pill ${ro.status === "ok" && !bad ? "ok" : "bad"}"
        title="our ingest verdict: whether we stored what we were given">${esc(ro.status)}</span>`];
    if (hv) parts.push(`<span class="pill ${hv === "healthy" ? "ok" : "bad"}"
        title="the PRODUCER's verdict on its own observation of this rollout">${esc(hv)}</span>`);
    else if (empty) parts.push(`<span class="pill bad"
        title="no producer verdict was stored for this rollout">unexplained</span>`);
    return parts.join("");
  }

  function emptyBox(ro) {
    const hv = ro.health_verdict;
    const d = document.createElement("div"); d.className = "emptybox";
    const say = hv === "unobserved"
      ? "The producer ran this rollout and reported that it could not observe it. "
        + "The transcript was never kept, so there is nothing to show — this is not an "
        + "agent that did nothing."
      : hv === "unhealthy"
      ? "The producer marked this rollout unhealthy. No events were stored for it."
      : hv
      ? `The producer's verdict on this rollout is "${esc(hv)}", and no events were stored.`
      : "Nothing in the store says why this rollout holds no events. That is a gap in our "
        + "reading, not a statement about the agent: the producer may have written a "
        + "verdict we do not read, or a transcript under a filename nothing matches.";
    d.innerHTML = `<b>This rollout holds no events.</b> ${say}`;
    return d;
  }

  function provSection() {
    const d = document.createElement("details"); d.className = "prov";
    const sm = document.createElement("summary");
    sm.textContent = provMine
      ? `provenance · ${provMine.length} record${provMine.length === 1 ? "" : "s"} — which input, in which reader, reading which file, produced these rows`
      : "provenance · not served by the API — what is missing";
    sm.append(" · ", readerDocLink());
    d.appendChild(sm);
    if (provMine && provMine.length) {
      // `reads` is pipe-separated when a question was answered from several files; " + " says
      // that they were read together, where a bare space would read as one impossible path.
      const rows = provMine.map(r => ({ ...r, reads: String(r.reads ?? "").split("|").join("  +  ") }));
      d.appendChild(table(["question", "input", "reader", "producer", "reads", "site", "lands", "n_values"], rows));
    } else if (provMine) {
      d.appendChild($(`<div class="empty">the provenance table holds no record for this rollout.</div>`));
    } else {
      d.appendChild($(`<div class="provmiss">
        <b>There is no API route serving provenance.</b>
        The request failed: <code>${esc(String(provErr || "no response"))}</code>.
        <br>The table exists in the store — <code>&lt;run_id&gt;/provenance.parquet</code>, columns
        <code>run_id, question, input, site, reader, producer, reads, lands, task_id, n_values</code>
        — and this page already reads it: the rows with <code>lands = 'events'</code> are exactly one
        per lane (their <code>n_values</code> are the lane event counts), so every lane header would
        name the file it was read from and every event could answer where it came from.
        <br>What it needs: <code>GET /v1/rollouts/{run_id}/{task_id}/{repeat}/provenance</code>
        returning this run's rows where <code>task_id = {task_id} OR task_id IS NULL</code>, as JSON
        with the ten columns unchanged. Until then the lane headers say "source unknown" rather
        than guessing.</div>`));
    }
    return d;
  }

  if (artifacts.length) {
    main.appendChild($("<h2>artifacts</h2>"));
    // Table half: an image artifact shows a thumbnail AND keeps its own download
    // link. `view_url` is set by the serve (the UI cannot see the recorded mime, and many
    // image rows carry neither a name nor a mime).
    for (const a of artifacts) { a.view_url = traceUrl(a.view_url); a.download_url = traceUrl(a.download_url); }
    for (const a of artifacts) a.__html = {
      preview: a.view_url
        ? `<a href="${a.view_url}" target="_blank" rel="noopener"><img class="artthumb" loading="lazy" src="${a.view_url}" alt="artifact ${a.artifact_idx}" onerror="this.replaceWith(document.createTextNode('—'))"></a>`
        : "",
      // the NAME opens the artifact in the shared viewer — the same interface, the
      // same prettify and line links, as a raw file. `download` is still its own column, so
      // nothing is lost: the click target changes meaning, the capability does not.
      name: artHashFor(runId, taskId, repeat, a)
        ? `<a href="${artHashFor(runId, taskId, repeat, a)}" title="open in the file viewer">${esc(a.name || "(unnamed)")}</a>`
        : (a.download_url
           ? `<a href="${a.download_url}">${esc(a.name || "(unnamed)")}</a>` : esc(a.name || "(unnamed)")),
      download: a.download_url ? `<a href="${a.download_url}">download</a>` : "",
    };
    main.appendChild(table(
      ["artifact_idx","preview","name","kind","role","origin","storage","size_bytes","mime","download"],
      artifacts));
  }
  if (ro.attempts?.length) {
    main.appendChild($("<h2>attempts</h2>"));
    main.appendChild(table(["attempt_idx","failure_class","is_terminal","error_message"], ro.attempts));
  }

  // A call's prompt is opened where its event is: jump there, then open the panel.
  const openPrompt = (lane, idx) => {
    location.hash = evHash(idx, null, lane);
    const box = curEvLookup && curEvLookup(lane || "", idx);
    const d = box && box.querySelector("details.prompt");
    if (d) d.open = true;
  };
  if (psum && psum.n_model_calls > 0)
    main.appendChild(promptContext(psum, (c) => openPrompt(c.lane, c.event_idx)));

  const total = events.length;
  // `372 events` was the sum across lanes presented as if it were one conversation — the session
  // lane in that rollout is 63. The count is still the total, because that is what it is, but it
  // now says across how many lanes, and the breakdown is one line below in the lane bar.
  const head = $(`<h2>trace (${total} events${multiLane ? ` across ${laneList.length} lanes` : ""}) <span id="load-note" class="mono" style="opacity:.6">· scroll to load</span><span id="pretty-count" class="mono" style="opacity:.7;color:var(--reason-edge)"></span></h2>`);
  main.appendChild(head);
  if (multiLane) main.appendChild(laneBar());
  // Two columns: a sticky schema-rooted tree sidebar + the streamed events. The tree is built
  // from the texts-free /trace_map aggregate; deep-links keep the events column in sync.
  const wrap = document.createElement("div"); wrap.className = "trace-wrap";
  const side = document.createElement("div"); side.className = "trace-side";
  const traceMain = document.createElement("div"); traceMain.className = "trace-main";
  if (multiLane) {
    traceMain.classList.add("has-lanes");
    traceMain.style.setProperty("--rw", RAILW + "px");
    side.style.top = "calc(var(--sticky-top, 0px) + 50px)";   // clear the sticky lane bar rather than slide under it
  }
  traceMain.dataset.tour = "trace-linear";   // guided tour step 3
  side.dataset.tour = "trace-tree";          // guided tour step 4 mentions it; the toggle is the target
  wrap.append(side, traceMain);
  main.appendChild(wrap);
  curTreeNodes = traceTree(side, tmap, evHash, laneList, laneByKey);
  curTreeSide = side;
  // wire every prettify toggle (top bar + sticky sidebar copy) through the shared setter
  document.querySelectorAll(".js-pretty").forEach(c => {
    c.checked = PRETTY.on;
    c.onchange = () => { tstate.pretty = c.checked; setPretty(c.checked); traceAddress(); }; });
  // Linear / Tree / Graph mode toggle (module-persisted; default `defMode`). Tree groups each
  // tool_result under the call that produced it; timeline order is NOT preserved (per the tooltip + note).
  const barEl = main.querySelector(".bar");
  const seg = document.createElement("span"); seg.className = "segmented";
  seg.dataset.tour = "trace-mode";   // guided tour step 4
  const bLin = document.createElement("button"); bLin.textContent = "linear";
  bLin.dataset.tour = "trace-mode-linear";
  const bTree = document.createElement("button"); bTree.textContent = "tree";
  bTree.dataset.tour = "trace-mode-tree";
  bTree.title = "Tree view groups tool results under the call that produced them — timeline (history) order is NOT preserved.";
  const bGraph = document.createElement("button"); bGraph.textContent = "graph";
  seg.append(bLin, bTree, bGraph); if (barEl) barEl.appendChild(seg);
  const treeNote = document.createElement("div"); treeNote.className = "tree-note";
  treeNote.textContent = "Tree view: tool results are grouped under the call that produced them — timeline (history) order is NOT preserved.";
  main.insertBefore(treeNote, wrap);
  const graphBox = document.createElement("div"); graphBox.className = "trace-graph";
  main.appendChild(graphBox);
  const syncMode = () => {
    bLin.classList.toggle("on", TRACE_MODE === "linear");
    bTree.classList.toggle("on", TRACE_MODE === "tree");
    bGraph.classList.toggle("on", TRACE_MODE === "graph");
    bGraph.classList.toggle("inert", !multiLane);
    bGraph.title = multiLane
      ? "Graph view: each lane a column, events placed by the order the lanes and their links state."
      : "This rollout has one lane — the graph would be the timeline, unchanged.";
    wrap.hidden = TRACE_MODE === "graph";
    if (TRACE_MODE !== "graph" && !drawer.hidden) closeDrawer();
    graphBox.hidden = TRACE_MODE !== "graph";
    if (TRACE_MODE === "graph" && !graphBox.firstChild) drawGraph();
    const claim = treeClaim();
    treeNote.textContent = claim.text;
    treeNote.classList.toggle("empty", !claim.ok);
    treeNote.hidden = TRACE_MODE !== "tree";
    bTree.title = claim.ok ? claim.text
      : "This rollout has no structure to nest — tree mode would show the timeline unchanged.";
    bTree.classList.toggle("inert", !claim.ok);
  };

  // Heads + tool calls render up front (light). Parts hydrate on demand as each event scrolls into
  // view (virtualized) — building every event's parts at once OOMs the tab. Hydration is keyed by LIST
  // POSITION (slotByPos/boxByPos): the backend keys grains on (agent_ref, event_idx) and orders events
  // by (agent_ref, event_idx), so event_idx is NOT globally unique once a rollout has >1 agent_ref lane
  // and list-position ≠ event_idx. Position matches the offset window 1:1 for both single- and multi-
  // lane. Every other map below is keyed on (lane, event_idx) — `boxByKey` — because that pair is the
  // event's actual identity; the old event_idx-keyed map silently merged three different events in
  // the three-lane rollout.
  // Tree-mode reorder just MOVES the existing box node (position rides on the box), so hydration is
  // unaffected by display order.
  const slotByPos = [], boxByPos = [], boxByKey = {};
  const laneByPos = events.map(ev => laneByKey[ev.lane || ""]);
  // A link from one event to another stays inside the lane — a tool result is the same sequence
  // as the call that produced it — so the lane is carried across, not re-derived.
  const evLinkA = (label, idx, lane) => {
    const h = evHash(idx, null, lane);
    const a = document.createElement("a"); a.className = "rlink"; a.href = h;
    a.textContent = label;
    a.onclick = (e) => { e.preventDefault(); e.stopPropagation(); location.hash = h; };
    return a;
  };

  // ── the rail ──────────────────────────────────────────────────────────────────────────────
  // Every row draws every lane's column: the lane the row is in as a solid spine with a node,
  // the others as a 1 px guide. The constant full-width grid is what makes "there are three
  // sequences and this row is in the middle one" readable from any row without scrolling to a
  // header — and it is O(lanes) per row with no dependence on the neighbours, which is what lets
  // it coexist with the list's lazy hydration.
  function railFor(L, opts) {
    const r = document.createElement("div"); r.className = "lanerail";
    for (const M of laneList) {
      const t = document.createElement("i");
      t.style.left = LANE_X(M.col) + "px";
      t.style.setProperty("--c", M.color);
      if (M === L) t.className = "self";
      r.appendChild(t);
    }
    const dot = document.createElement("b");
    dot.style.left = LANE_X(L.col) + "px";
    dot.style.setProperty("--c", L.color);
    r.appendChild(dot);
    if (opts && opts.open) {   // the lane opens here: a stub out to its column, git's branch point
      const u = document.createElement("u");
      u.style.left = "2px"; u.style.top = "15px";
      u.style.width = Math.max(0, LANE_X(L.col) - 2) + "px";
      u.style.background = L.color;
      r.appendChild(u);
    }
    if (L.col >= LANE_CAP) {   // past the eight-column cap: marked, never silently dropped
      const s = document.createElement("s");
      s.textContent = "~"; s.style.left = (LANE_X(L.col) + 5) + "px"; s.style.top = "6px";
      s.style.setProperty("--c", L.color);
      r.appendChild(s);
    }
    return r;
  }

  // The event's answer to "where did this come from", opened in place under its header: the
  // provenance record for its lane — which input, in which reader, reading which file.
  function toggleProvPop(box, L) {
    const open = box.querySelector(":scope > .provpop");
    if (open) { open.remove(); return; }
    const p = document.createElement("div"); p.className = "provpop";
    const dl = document.createElement("dl");
    const add = (k, v) => { const dt = document.createElement("dt"); dt.textContent = k;
      const dd = document.createElement("dd"); dd.textContent = v; dl.append(dt, dd); };
    add("lane", L.key);
    add("events", `${L.n}, numbered #${L.minIdx}–#${L.maxIdx} within this lane`);
    if (L.prov) {
      const q = L.prov;
      add("question", q.question ?? "—");
      add("input", q.input ?? "—");
      add("reader", q.reader ?? "—");
      add("producer", q.producer ?? "—");
      add("reads", String(q.reads ?? "—").split("|").join("\n"));
      add("site", q.site ?? "—");
      add("lands", `${q.lands ?? "—"} · ${q.n_values ?? "?"} values`);
    } else {
      add("source", provRows ? "no provenance row matched this lane"
                             : `unavailable — GET ${base}/provenance is not served`);
    }
    const dt = document.createElement("dt"); dt.textContent = "reader DSL";
    const dd = document.createElement("dd"); dd.appendChild(readerDocLink());
    dl.append(dt, dd);
    p.appendChild(dl);
    box.insertBefore(p, box.querySelector(".parts-slot"));
  }

  function makeBox(ev, pos) {
    const L = laneByPos[pos];
    const key = evKey(ev.lane, ev.event_idx);
    const box = document.createElement("div");
    // The kind drives a 3px left edge (see `.event.k-*`). `evKind` is the displayed word;
    // the class uses the VOCABULARY kind so a producer's own spelling cannot invent a colour.
    box.className = "event k-" + String(ev.kind || "other").replace(/[^a-z_]/gi, "");
    // `ev-<idx>` stopped being unique the moment a rollout held two lanes, so the DOM id carries
    // both halves of the identifier. A single-lane rollout with a named lane gets the same
    // treatment; `curEvLookup` below is what keeps the older `#…/e12` links resolving.
    box.id = "ev-" + key;
    box.dataset.idx = ev.event_idx;
    box.dataset.pos = pos;
    box.dataset.key = key;
    // Fold signature for runs of the same record stated over and over. It is SIZE, not bytes:
    // the parts are not hydrated at render time, so claiming "identical" would be a claim the
    // page cannot check. Same lane, same producer word, same role, same part count and same
    // character count is enough to say "these are the same shape" and let a person expand.
    box.dataset.sig = [ev.lane || "", ev.kind_raw || ev.kind || "", ev.role || "",
                       ev.n_parts ?? "", ev.text_chars ?? "", ev.n_tool_calls ?? ""].join("|");
    box.dataset.kindraw = ev.kind_raw || ev.kind || "event";
    box.dataset.chars = ev.text_chars ?? "";
    if (ev.lane) box.dataset.lane = ev.lane;
    if (multiLane) {
      box.style.setProperty("--c", L.color);
      box.appendChild(railFor(L));
      L.boxes.push(box);
    }
    const ehead = document.createElement("div"); ehead.className = "ehead";
    const info = document.createElement("span"); info.className = "einfo";
    // The provenance sentence rides on the `#N` link's own tooltip. That is how a SINGLE-lane
    // rollout — 4 492 of 4 498 — answers "where did this come from" without gaining one pixel.
    const src = provLine(L);
    info.innerHTML =
      `<a class="evlink" href="${esc(evHash(ev.event_idx, null, ev.lane))}" title="${esc(src ? readerDocTip("link to this event · " + src) : "link to this event")}">#${ev.event_idx}</a> · <span class="ekind">${esc(evKind(ev))}</span>${ev.role ? " · " + esc(ev.role) : ""}`
      + (ev.model_name ? ` · ${esc(ev.model_name)}` : "")
      + (ev.completion_tokens != null ? ` · ${ev.completion_tokens} tok` : "");
    // The row's own provenance (store-v12 `src_*`): the file and line it was read out of.
    if (ev.src_path) {
      const a = document.createElement("a"); a.className = "rlink esrc";
      a.href = fileHashFor(encodeURIComponent(runId), ev.src_path) + (ev.src_line != null ? "?L=" + ev.src_line : "");
      a.textContent = "source";
      a.title = readerDocTip(`${ev.src_path}${ev.src_line != null ? ":" + ev.src_line : ""}`
        + (ev.src_input ? ` — read by input ${ev.src_input}` : ""));
      info.append(document.createTextNode(" · "), a);
    }
    if (multiLane) {
      const chip = document.createElement("button"); chip.className = "lanechip";
      chip.textContent = laneShort(L.key);
      chip.title = readerDocTip(`lane: ${L.key}\n${src || "provenance unavailable"}\nclick for the full record`);
      chip.onclick = (e) => { e.stopPropagation(); toggleProvPop(box, L); };
      info.prepend(chip);
    }
    const rm = resultMeta[key];
    if (rm) {
      const n = rm.tools.length;
      info.append(document.createTextNode(" "),
        evLinkA(`← from event #${rm.fromEvent}` + (n > 1 ? ` (${n} calls)` : ""), rm.fromEvent, ev.lane));
      // tree-mode only (CSS-gated on .event.nested): says WHICH call produced this nested result
      const tag = document.createElement("span"); tag.className = "rtag";
      tag.textContent = `↳ result of ${toolLabel(rm.tools)} (from #${rm.fromEvent})`;
      info.append(document.createTextNode(" "), tag);
    }
    // The events this request was built from. Same link family as the call/result pairing above,
    // because it is the same kind of fact: an edge the PRODUCER stated, followed by source id and
    // never by position. The system prompt is deliberately named as absent rather than left to
    // read as a shortfall -- MEASURED, every file's one `developer` message is a message this
    // producer never emits as an event, so a count of N unfolds into N-1 links by construction.
    const built = consumedBy[ev.source_event_id];
    if (built && built.length) {
      info.append(document.createTextNode(" "));
      const wrap = document.createElement("span"); wrap.className = "rtag-built";
      wrap.appendChild(document.createTextNode("↩ replayed from "));
      built.forEach((t, i) => {
        if (i) wrap.appendChild(document.createTextNode(", "));
        wrap.appendChild(evLinkA(`#${t.event_idx}`, t.event_idx, t.lane));
      });
      const note = document.createElement("span"); note.className = "muted";
      note.textContent = " \u00b7 the system prompt is not an event in this producer";
      note.title = "This producer replays messages by message id, and 0 of 535 message ids are "
        + "an event id. The links follow `consumed_event_ids`, which names the events the "
        + "messages were built from. Its one `developer` message per file is never emitted as "
        + "an event, so it has no link.";
      wrap.appendChild(note);
      info.append(wrap);
    }
    for (const x of relsAt[key] || []) {
      const sp = document.createElement("span"); sp.className = "rtag-built erel";
      const inf = (edgeKinds[x.e.space] || {}).inferred_from;
      sp.appendChild(document.createTextNode(`${x.dir} ${inf ? "≈ " : ""}${x.e.space} `));
      if (inf) sp.title = `matched by text (${inf}), not by an id` + (x.e.reason ? `\n${x.e.reason}` : "");
      if (x.other) sp.appendChild(evLinkA(`#${x.other.event_idx}${x.other.lane ? ` (${x.other.lane})` : ""}`,
                                          x.other.event_idx, x.other.lane));
      else {
        const n = document.createElement("span"); n.className = "mono"; n.textContent = x.name;
        n.title = x.e.resolved === false ? `dangling: ${x.e.reason || "no reason recorded"}`
          : "resolved by the reader on its own key, which no event column carries";
        sp.appendChild(n);
      }
      info.append(document.createTextNode(" "), sp);
    }
    ehead.appendChild(info);
    // persistent per-event copy toolbar in the header row (never covers the first text line)
    const copy = document.createElement("span"); copy.className = "ecopy";
    const cl = document.createElement("button"); cl.className = "pbtn"; cl.textContent = "copy link";
    cl.dataset.tour = "copy-link";   // guided tour step 5 POINTS at this; it never presses it
    cl.onclick = () => copyText(location.origin + location.pathname + withSelection(evHash(ev.event_idx, null, ev.lane), box), cl);
    const cc = document.createElement("button"); cc.className = "pbtn"; cc.textContent = "copy content";
    cc.onclick = async () => {   // hydrate this box's page first if its parts aren't in the DOM yet
      const slot = slotByPos[pos];
      if (slot && !slot.dataset.done) await loadPage(Math.floor(pos / PAGE));
      if (!sameRollout()) return;   // navigated away mid-await → don't copy an empty string
      copyText(nodeRaw(slot), cc);
    };
    copy.append(cl, cc); ehead.appendChild(copy);
    box.appendChild(ehead);
    const parts = document.createElement("div"); parts.className = "parts-slot";
    box.appendChild(parts);
    slotByPos[pos] = parts; boxByPos[pos] = box; boxByKey[key] ??= box;
    for (const c of callsByEvent[key] || []) {
      const head = String(c.args_head ?? "");
      const chars = c.args_chars != null ? c.args_chars : head.length;
      const suffix = (c.status != null && c.status !== "unknown") ? ` [${c.status}]` : "";
      // per-call → result link (uses this call's own result_event_idx, correct for multi-call events)
      const rlink = c.result_event_idx != null ? evLinkA(`→ result #${c.result_event_idx}`, c.result_event_idx, ev.lane) : null;
      // The same producer id in another lane: a second account of this call, not a second call.
      const others = alsoIn(c, ev.lane);
      const xlane = others ? (() => {
        const sp = document.createElement("span"); sp.className = "xlane";
        setText(sp, ` · also in ${others.join(", ")}`);
        sp.title = `the producer's own id ${c.call_id_source} appears in `
          + `${others.length + 1} lanes — the same call, described twice`;
        return sp; })() : null;
      // long OR head-truncated args -> fold behind tool_name(...); fetch args_full lazily on expand
      if (chars > TCALL_CAP || chars > head.length) {
        const d = document.createElement("details"); d.className = "tcall";
        const s = document.createElement("summary");
        const tx = document.createElement("span");
        const nm = document.createElement("span"); nm.className = "tname"; setText(nm, `→ ${c.tool_name}`);
        const ar = document.createElement("span"); ar.className = "targs"; setText(ar, `(…)${suffix}`);
        tx.append(nm, ar);
        s.appendChild(tx); if (rlink) s.append(document.createTextNode(" "), rlink);
        if (xlane) s.appendChild(xlane);
        const body = document.createElement("div");
        const k = argsKey(c);
        let pulling = null;
        const pull = () => pulling ??= (body.replaceChildren(slotSpinner()),
          loadFullArgs().then(m => { body.replaceChildren(); setText(body, m[k] ?? head); },
            () => { body.replaceChildren(); setText(body, head + " … [args failed to load]"); }));
        d.ontoggle = () => { if (d.open) pull(); };
        d.dataset.loadable = ""; d.loadWhole = pull;
        d.append(s, body); box.appendChild(d);
      } else {
        const d = document.createElement("div"); d.className = "tcall";
        const tx = document.createElement("span");
        const nm = document.createElement("span"); nm.className = "tname"; setText(nm, `→ ${c.tool_name}`);
        const ar = document.createElement("span"); ar.className = "targs"; setText(ar, `(${head})${suffix}`);
        tx.append(nm, ar);
        d.appendChild(tx); if (rlink) d.append(document.createTextNode(" "), rlink);
        if (xlane) d.appendChild(xlane);
        box.appendChild(d);
      }
    }
    // ── an event that carries nothing says so ───────────────────────────────────────────
    // MEASURED: 5,824 model_call events hold neither a part nor a tool call, and only 1,174
    // of 42,230 carry any text at all. Rendered as a blank box they read as a turn that was
    // lost. Most are not: a capture record has no field for what the model SAID, so the
    // producer never wrote one. The box names the producer's own word for the event, so a
    // person can tell an empty record from a dropped one.
    // ── a model call shows the wire, because that is the fact it owns ──────────────
    // Its prose was removed from the reader: all 310 of the lane's texts on example 29
    // were BYTE-IDENTICAL to the conversation's, so a `call` was rendering as a thought.
    const isCall = ev.kind === "model_call" || ev.kind_raw === "model_call" || ev.lane_kind === "model_call";
    if (isCall) {
      const bits = [];
      const add = (k, v, suf) => { if (v !== null && v !== undefined && v !== "")
        bits.push([k, suf ? `${v}${suf}` : String(v)]); };
      add("in", ev.prompt_tokens); add("out", ev.completion_tokens);
      add("cached", ev.cached_tokens); add("reasoning", ev.reasoning_tokens);
      add("latency", ev.latency_s === null || ev.latency_s === undefined
            ? null : `${Number(ev.latency_s).toFixed(2)}`, "s");
      add("finish", ev.finish_reason); add("id", ev.response_id);
      if (bits.length) {
        const w = document.createElement("div"); w.className = "wire";
        for (const [k, v] of bits) {
          const sp = document.createElement("span");
          const kk = document.createElement("span"); kk.className = "wk";
          kk.textContent = k + " ";
          const vv = document.createElement("b"); vv.textContent = v;
          sp.appendChild(kk); sp.appendChild(vv); w.appendChild(sp);
        }
        box.appendChild(w);
      }
    }
    const pc = promptByKey[key];
    if (pc) box.appendChild(promptPanel(pc, base, openPrompt));
    if (!(ev.n_parts > 0) && !(callsByEvent[key] || []).length && !pc) {
      const n = document.createElement("div"); n.className = "nocontent";
      setText(n, `no text — ${ev.kind_raw || ev.kind || "event"}`
        + `${ev.lane_kind ? ` from the ${ev.lane_kind} lane` : ""}`
        + `${ev.model_name ? ` · ${ev.model_name}` : ""}`);
      n.title = "this event was stored with no parts and no tool calls; "
        + "the producer wrote no text for it";
      box.appendChild(n);
    }
    return box;
  }
  events.forEach(makeBox);
  // Older links carry no lane. Resolve them to the first event with that index rather than to
  // nothing, so nothing copied before lanes existed breaks; the lane is then written back into
  // the address, which is how an ambiguous link repairs itself on arrival.
  curEvLookup = (lane, idx) => boxByKey[evKey(lane, idx)]
    || boxByPos.find(b => +b.dataset.idx === idx) || null;

  // ── lane header rows ──────────────────────────────────────────────────────────────────────
  // One per lane, a SIBLING of the event boxes rather than a child of the first one, so tree
  // mode can re-order the boxes underneath it without dragging the header somewhere strange.
  // It is the one place that says the numbering restarts, and why.
  function laneBand(L) {
    const b = document.createElement("div"); b.className = "laneband";
    b.style.setProperty("--c", L.color);
    b.dataset.lane = L.key;
    b.appendChild(railFor(L, { open: true }));
    const kind = document.createElement("span"); kind.className = "lk"; kind.textContent = L.kind;
    const gloss = document.createElement("span"); gloss.className = "lg";
    gloss.textContent = LANE_GLOSS[L.kind] ? `· ${LANE_GLOSS[L.kind]}` : "";
    const inst = document.createElement("span"); inst.className = "li"; inst.textContent = L.inst;
    const num = document.createElement("span"); num.className = "lnum";
    const src = document.createElement("div"); src.className = "lsrc";
    if (L.prov) {
      src.innerHTML = `from <b>${esc(String(L.prov.reads || "").split("|").join(" + "))}</b>`
        + ` · input <b>${esc(L.prov.input || "?")}</b> · reader <b>${esc(L.prov.reader || "?")}</b>`
        + (L.prov.producer && L.prov.producer !== L.prov.reader ? ` · producer <b>${esc(L.prov.producer)}</b>` : "")
        + ` · ${fmt(L.prov.n_values)} values · `;
      src.appendChild(readerDocLink());
    } else {
      src.className = "lsrc none";
      src.textContent = provRows ? "no provenance row matched this lane"
        : "source unknown — the API serves no provenance route (see “provenance” at the top)";
    }
    const acts = document.createElement("span"); acts.className = "lacts";
    const mk = (label, title, fn) => { const x = document.createElement("button");
      x.className = "lact"; x.textContent = label; x.title = title; x.onclick = fn; return x; };
    const bFold = mk("", "", () => setFold(L.key, !foldSet().has(laneId(L.key)))); bFold.dataset.act = "fold";
    const bSolo = mk("", "", () => setSolo(tstate.solo === laneId(L.key) ? "" : laneId(L.key))); bSolo.dataset.act = "solo";
    const bLink = mk("copy lane link", "a link that opens this rollout on this lane alone", () =>
      copyText(location.origin + location.pathname + rolloutKey
               + traceQ({ solo: laneId(L.key), lane: "", fold: [] }), bLink));
    const next = laneList[L.col + 1];
    const bNext = next ? mk("next lane ↓", `jump to ${next.key}`, () => gotoLane(next)) : null;
    acts.append(bFold, bSolo, bLink); if (bNext) acts.appendChild(bNext);
    // Only the labels that SAY something. `kind`, `gloss` and `inst` are all empty for the
    // unnamed lane, and an empty flex child still takes its share of `gap:3px 10px` -- so the
    // header's first visible text sat 30px right of every event under it (measured: 417 against
    // 388) and read as an indent the lane does not have. A lane is not inside anything.
    b.append(...[kind, gloss, inst].filter(el => el.textContent), num, acts, src);
    b.__sync = (folded, solo) => {
      num.textContent = folded
        ? `${L.n} events collapsed — #${L.minIdx}–#${L.maxIdx}`
        : `${L.n} events · #${L.minIdx}–#${L.maxIdx}, numbered from ${L.minIdx} in this lane`;
      bFold.textContent = folded ? "expand" : "collapse";
      bFold.title = folded ? `show this lane's ${L.n} events again`
                           : `collapse this lane's ${L.n} events to this row`;
      bSolo.textContent = solo ? "show all lanes" : "only this lane";
      bSolo.title = solo ? "bring the other lanes back" : "hide every other lane";
    };
    b.__sync(false, false);
    return b;
  }
  if (multiLane) for (const L of laneList) L.band = laneBand(L);

  // Display order (returns box nodes, not indices). Linear = list order (boxByPos), correct for any
  // number of lanes. Tree = event spine at top level (result events skipped there); each call drops
  // its result box nested right after, recursing so a result that itself emits calls nests its own
  // children (chained results). Tree keying is (lane, event_idx) — it used to group by `agent_ref`,
  // which is '' on every row in the corpus, so lanes were invisible to it and two events sharing an
  // event_idx collapsed onto one another. A call never reaches across lanes, and the event list is
  // already lane-contiguous, so lane blocks survive the reorder without being special-cased.
  function orderFor(mode) {
    if (mode !== "tree") return boxByPos.map((box, pos) => ({ box, nested: false, lane: laneByPos[pos] }));
    const out = [], placed = new Set();
    const laneOfKey = {};
    for (const ev of events) laneOfKey[evKey(ev.lane, ev.event_idx)] = ev.lane;
    // A producer that states its own tree wins: a parent pointer is the structure the agent
    // actually had, and a call/result nesting is a weaker guess at the same thing. Depth is the
    // real depth so the indent means something, and children keep the order the events arrived
    // in — a producer that numbered its events already sorted them.
    if (hasTree) {
      // Indent is BRANCH depth, not distance from the root. A parent chain is mostly linear — the
      // producer's 68-event rollout is one chain with a single fork — and indenting by path length
      // marched every event one step further right until the content column vanished. Measured on
      // that rollout: path depth reached 9 by event #11 and 60+ by the end; branch depth is 0 for
      // the whole mainline and 1 inside the one fork. Git draws a mainline straight and gives a rail
      // only to a branch, for the same reason: the indent has to MEAN something.
      const walk = (k, depth) => {
        const box = boxByKey[k]; if (!box || placed.has(box)) return;
        const kids = childrenOf[k] || [];
        // A tool result is marked nested even at depth 0. When the producer states its own
        // tree, a result is its call's ONLY child, so it continues the mainline and `depth`
        // stays 0 -- which left `.event.nested` unset, and the `↳ result of <tool> (from #N)`
        // tag is CSS-gated on exactly that class. MEASURED on the reported rollout: 11 of 11
        // tool_calls carry `result_event_idx` and 11 of 11 results carry a parent, and tree
        // mode drew neither. The call-to-result edge is the one a person most wants to see,
        // so it earns the indent and the rung whatever its depth.
        out.push({ box, key: k, nested: depth > 0 || resultSet.has(k),
                   result: resultSet.has(k), depth,
                   lane: laneByKey[laneOfKey[k] || ""],
                   branch: kids.length > 1 ? kids.length : 0 });
        placed.add(box);
        // one child continues the line; several open a level, and the FIRST keeps the mainline so a
        // linear read stays a linear read.
        if (kids.length === 1) walk(kids[0], depth);
        else kids.forEach((c, i) => walk(c, i === 0 ? depth : depth + 1));
      };
      for (const ev of events) { const k = evKey(ev.lane, ev.event_idx); if (!parentOf[k]) walk(k, 0); }
      for (const ev of events) walk(evKey(ev.lane, ev.event_idx), 0);   // any cycle-orphaned node
      return out;
    }
    const push = (lane, idx, nested) => {
      const k = evKey(lane, idx);
      const box = boxByKey[k]; if (!box || placed.has(box)) return;
      out.push({ box, nested, depth: nested ? 1 : 0, lane: laneByKey[lane || ""], branch: 0 }); placed.add(box);
      for (const c of callsByEvent[k] || [])
        if (c.result_event_idx != null) push(lane, c.result_event_idx, true);   // chained result
    };
    for (const ev of events) if (!resultSet.has(evKey(ev.lane, ev.event_idx))) push(ev.lane, ev.event_idx, false);
    for (const ev of events) push(ev.lane, ev.event_idx, false);   // any orphan result
    return out;
  }
  // What tree mode can actually do here, in one sentence, decided by the data rather than by the
  // button. A note that promises a regrouping the page did not perform is worse than no note.
  function treeClaim() {
    if (hasTree) return { ok: true, text: "Tree view: the producer's own parent pointer \u2014 "
      + nParentEdges + " edges, " + branchAt.size + " branch point"
      + (branchAt.size === 1 ? "" : "s")
      + (dangling.length ? ", " + dangling.length + " pointer"
         + (dangling.length === 1 ? "" : "s") + " naming an event this rollout does not hold" : "")
      + ". Indent is branch depth \u2014 a straight run stays straight; timeline order is NOT preserved." };
    const nests = Object.keys(resultMeta).length;
    if (nests) return { ok: true, text: "Tree view: tool results are grouped under the call that "
      + "produced them (" + nests + " nested) — timeline (history) order is NOT preserved." };
    return { ok: false, text: "Tree view has nothing to nest here: this rollout states no parent "
      + "pointer and no tool result names the call that produced it, so the order below is the "
      + "timeline, unchanged." };
  }
  // A producer that states one fact N times in a row fills a screen with a single sentence.
  // MEASURED on path-tracing-reverse: 5 of 7 bookkeeping events are `last-prompt` records of
  // 300 chars each, and they were the whole first screen. Runs of FOUR or more identical
  // signatures collapse to one row; three or fewer are left alone, because a short repeat is
  // usually real. Tree mode is skipped — it reorders, so adjacency there means nothing.
  const FOLD_RUN = 4;
  function foldRuns(frag) {
    if (TRACE_MODE === "tree") return;
    const kids = [...frag.children];
    let i = 0;
    while (i < kids.length) {
      const sig = kids[i].dataset ? kids[i].dataset.sig : null;
      if (!sig || !kids[i].classList.contains("event")) { i++; continue; }
      let j = i;
      while (j + 1 < kids.length && kids[j + 1].dataset && kids[j + 1].dataset.sig === sig
             && kids[j + 1].classList.contains("event")) j++;
      const n = j - i + 1;
      if (n >= FOLD_RUN) {
        const row = document.createElement("div");
        row.className = "dupfold";
        const word = kids[i].dataset.kindraw, ch = kids[i].dataset.chars;
        row.innerHTML = `▸ <b>${n} consecutive ${esc(word)} events</b> of the same shape`
          + (ch ? ` — ${esc(ch)} characters each` : "") + ` · click to expand`;
        row.title = "the producer wrote this record " + n + " times in a row; the page folds the "
          + "run so the rest of the lane is readable. Sizes match — the page does not claim the "
          + "bytes are identical, only that the shape is.";
        let open = false;
        row.onclick = () => {
          open = !open;
          for (let k = i; k <= j; k++) kids[k].style.display = open ? "" : "none";
          row.innerHTML = open
            ? `▾ <b>${n} ${esc(word)} events</b> · click to fold`
            : `▸ <b>${n} consecutive ${esc(word)} events</b> of the same shape`
              + (ch ? ` — ${esc(ch)} characters each` : "") + ` · click to expand`;
        };
        for (let k = i; k <= j; k++) kids[k].style.display = "none";
        frag.insertBefore(row, kids[i]);
      }
      i = j + 1;
    }
  }
  function renderBoxes() {
    const frag = document.createDocumentFragment();
    const order = orderFor(TRACE_MODE);
    // The rail's end cap belongs to the last row a lane has IN THIS ORDER, which tree mode can
    // change, so it is recomputed per render rather than baked into the box.
    for (const L of laneList) L.__last = null;
    for (const o of order) if (o.lane) o.lane.__last = o.box;
    const seen = new Set();
    let lastDepth = 0;
    for (const o of order) {
      o.box.classList.toggle("nested", o.nested);
      // Depth is the producer's own depth, capped for layout only: past 8 levels the indent would
      // eat the content column, and the depth is still printed on the rung.
      const d = Math.min(o.depth || 0, 8);
      // A tool result sits one step INSIDE its call. The depth indent alone put them at the
      // same margin -- measured, call and result both 22px -- because a result is its call's
      // only child and so shares its branch depth. The inline style is what the depth needs
      // and it overrode the `.event.nested` class indent, so the extra step is added here.
      const indent = d * 22 + (o.result ? 26 : 0);
      o.box.style.marginLeft = TRACE_MODE === "tree" && indent ? indent + "px" : "";
      o.box.classList.toggle("branch", TRACE_MODE === "tree" && !!o.branch);
      let rung = o.box.querySelector(":scope > .rung");
      // A label on every row of a branch is noise; the reader needs to know where a level STARTS
      // and where the agent forked. So the rung is drawn only when the depth changes from the row
      // above, or when this node is itself a fork.
      const changed = d !== lastDepth; lastDepth = d;
      if (TRACE_MODE === "tree" && (o.branch || (d && changed))) {
        if (!rung) { rung = document.createElement("div"); rung.className = "rung";
                     o.box.insertBefore(rung, o.box.firstChild); }
        rung.textContent = (d ? "\u2937 branch " + (o.depth) : "mainline")
          + (o.branch ? "  \u00b7  branches into " + o.branch + " \u2014 " + branchWhy(o.key) : "");
        rung.hidden = false;
      } else if (rung) rung.hidden = true;
      const L = o.lane;
      if (L && L.band && !seen.has(L)) {
        seen.add(L);
        L.band.classList.toggle("first", seen.size === 1);
        frag.appendChild(L.band);
      }
      o.box.classList.toggle("lane-last", !!L && L.__last === o.box);
      frag.appendChild(o.box);   // moves the existing node — hydration + observers preserved, no refetch
    }
    foldRuns(frag);
    traceMain.appendChild(frag);
    applyLaneView();
  }
  // ── the graph ─────────────────────────────────────────────────────────────────────────────
  // The links between events, each named by (lane, event_idx): what readers stated (`/edges`)
  // and the producer's own parent pointer. graph.js decides what each says about order.
  function graphLinks() {
    const links = [];
    for (const e of (edges || [])) {
      const f = e.space === "consumed" ? evIdxBySid[e.frm_key]
        : e.frm_task_id === ro.task_id ? evByName[[e.frm_lane || "", e.frm_input || "", e.frm_key].join("\0")] : null;
      const t = e.space === "consumed" ? evIdxBySid[e.to_key]
        : e.to_task_id === ro.task_id ? evByName[[e.to_lane || "", e.to_input || "", e.to_key].join("\0")] : null;
      if (f && t) links.push({ frm: evKey(f.lane, f.event_idx), to: evKey(t.lane, t.event_idx),
                               space: e.space, first: (edgeKinds[e.space] || {}).earlier || null,
                               inferred: !!(edgeKinds[e.space] || {}).inferred_from, note: e.reason || "" });
    }
    // The producer's own parent pointer: the parent came first.
    for (const [k, pk] of Object.entries(parentOf)) links.push({ frm: pk, to: k, space: "parent", first: "frm" });
    // The same tool call in several lanes, by the producer's call id: one link per pair of
    // neighbouring lanes, so three accounts of a call draw two links, not three; none where a
    // stated link already joins the two events.
    const joined = new Set(links.flatMap((l) => [l.frm + "\0" + l.to, l.to + "\0" + l.frm]));
    const byCallId = {};
    for (const c of calls) if (c.call_id_source) (byCallId[c.call_id_source] ??= []).push(c);
    for (const cs of Object.values(byCallId)) {
      const ends = [...new Map(cs.map((c) => [c.lane || "", evKey(c.lane, c.event_idx)])).values()];
      for (let i = 0; i + 1 < ends.length; i++)
        if (!joined.has(ends[i] + "\0" + ends[i + 1]))
          links.push({ frm: ends[i], to: ends[i + 1], space: "same_call", first: null });
    }
    return links;
  }
  // The start of each event's first text part, filled in as the parts pages arrive.
  const graphPreview = {};
  function graphLabel(ev) {
    const k = evKey(ev.lane, ev.event_idx), cs = callsByEvent[k] || [];
    const model = ev.kind === "model_call" || ev.lane_kind === "model_call";
    const who = model ? "model_call" : ev.kind === "message" && ev.role ? ev.role : evKind(ev);
    const what = model
      ? [ev.prompt_tokens != null ? `${kfmt(ev.prompt_tokens)} in` : "",
         ev.completion_tokens != null ? `${kfmt(ev.completion_tokens)} out` : "",
         ev.latency_s != null ? `${ev.latency_s.toFixed(1)} s` : ""].filter(Boolean).join(" · ")
      : cs.length ? cs.map((c) => `${c.tool_name || "tool"}(${c.args_head || ""})`).join(", ")
      : graphPreview[k] ?? (ev.text_chars ? `${ev.text_chars} chars` : "");
    return `#${ev.event_idx} ${who}${what ? " · " + what : ""}`;
  }
  function graphTitle(ev) {
    return `#${ev.event_idx} ${evKind(ev)}${ev.role ? " · " + ev.role : ""}`
      + (ev.text_chars ? ` · ${ev.text_chars} chars` : "")
      + (ev.started_at ? `\n${ev.started_at}` : "") + "\nclick to open beside the graph";
  }
  // ── the event panel ───────────────────────────────────────────────────────────────────────
  // Graph mode opens an event beside the graph rather than leaving it: the event's own contents
  // from the point read, its tool calls with their whole arguments, its request when it is a
  // model call, and its links, each of which opens its other end here.
  const drawer = document.createElement("aside"); drawer.className = "evdrawer"; drawer.hidden = true;
  main.appendChild(drawer);
  let graphCtl = null, graphLinkList = [], drawerKey = null;
  const closeDrawer = () => { drawer.hidden = true; drawerKey = null; if (graphCtl) graphCtl.select(null); };
  const laneSeq = {};
  for (const ev of events) (laneSeq[ev.lane || ""] ??= []).push(evKey(ev.lane, ev.event_idx));
  const onKey = (e) => {
    if (!drawer.isConnected) { document.removeEventListener("keydown", onKey); return; }
    if (drawer.hidden || (e.target instanceof Element && e.target.closest("input, textarea"))) return;
    if (e.key === "Escape") closeDrawer();
    else if (e.key === "ArrowDown" || e.key === "ArrowUp") { e.preventDefault(); stepDrawer(e.key === "ArrowDown" ? 1 : -1); }
  };
  document.addEventListener("keydown", onKey);
  function stepDrawer(d) {
    const ev = evByKey[drawerKey]; if (!ev) return;
    const seq = laneSeq[ev.lane || ""], i = seq.indexOf(drawerKey);
    if (seq[i + d]) openEvent(seq[i + d]);
  }
  // A producer's clock is an ISO string or epoch seconds; both are shown as UTC ISO.
  function drawerTime(v) {
    const n = Number(v);
    return Number.isFinite(n) && String(v).trim() !== "" ? new Date(n * 1000).toISOString() : String(v);
  }
  function drawerButton(label, title, act) {
    const b = document.createElement("button"); b.className = "dbtn"; b.textContent = label; b.title = title;
    b.onclick = (e) => { e.stopPropagation(); act(); };
    return b;
  }
  async function openEvent(key) {
    const ev = evByKey[key]; if (!ev) return;
    drawerKey = key; drawer.hidden = false;
    if (graphCtl) graphCtl.select(key);
    const L = laneByKey[ev.lane || ""];
    const head = document.createElement("div"); head.className = "dhead";
    head.style.setProperty("--c", L ? L.color : "#8a8f9a");
    const title = document.createElement("div"); title.className = "dtitle";
    title.innerHTML = `<span class="dlane">${esc(laneName(ev.lane))}</span> `
      + `<b>#${ev.event_idx}</b> · <span class="ekind">${esc(evKind(ev))}</span>${ev.role ? " · " + esc(ev.role) : ""}`
      + (ev.model_name ? ` · ${esc(ev.model_name)}` : "")
      + (ev.prompt_tokens != null ? ` · ${kfmt(ev.prompt_tokens)} in` : "")
      + (ev.completion_tokens != null ? ` · ${kfmt(ev.completion_tokens)} out` : "")
      + (ev.started_at ? ` · <span class="mono">${esc(drawerTime(ev.started_at))}</span>` : "");
    const tools = document.createElement("div"); tools.className = "dtools";
    tools.append(drawerButton("↑", "previous event in this lane (↑)", () => stepDrawer(-1)),
                 drawerButton("↓", "next event in this lane (↓)", () => stepDrawer(1)),
                 drawerButton("linear view", "open this event in the linear view", () => {
                   closeDrawer(); setMode("linear"); location.hash = evHash(ev.event_idx, null, ev.lane); }),
                 drawerButton("×", "close (Esc)", closeDrawer));
    head.append(title, tools);
    const body = document.createElement("div"); body.className = "dbody";
    body.appendChild(slotSpinner());
    drawer.replaceChildren(head, body);
    const other = (l) => evByKey[l.frm === key ? l.to : l.frm];
    const links = graphLinkList.filter((l) => (l.frm === key || l.to === key) && other(l))
      .sort((a, b) => (other(a).lane || "").localeCompare(other(b).lane || "") || other(a).event_idx - other(b).event_idx);
    let got;
    try {
      got = await traceApi(`${base}/events/${ev.event_idx}/parts?lane=${encodeURIComponent(ev.lane || "")}`,
                           { timeout: 120000 });
    } catch (e) { got = { error: e && e.message || String(e) }; }
    if (drawerKey !== key) return;   // another event was opened while this one loaded
    body.replaceChildren();
    const row = got && got.events && got.events[0];
    if (!row) { body.textContent = "this event failed to load: " + ((got && got.error) || "no row"); return; }
    for (const p of row.parts || []) body.appendChild(partNode({ ...p, lane: ev.lane }, base));
    // A model call reads as what it was sent, then what it answered.
    const pc = promptByKey[key];
    if (pc) {
      const pp = promptPanel(pc, base, (lane, idx) => openEvent(evKey(lane, idx)), true);
      body.appendChild(pp); pp.open = true;
    }
    for (const c of row.tool_calls || []) {
      const args = String(c.args_full ?? c.args_head ?? "");
      const cell = textCell("tcallargs", args, `→ ${c.tool_name || "tool"}`);
      if (c.result_event_idx != null) {
        const rk = evKey(ev.lane, c.result_event_idx);
        cell.querySelector(".cellhead").append(" ", drawerButton(`result #${c.result_event_idx}`,
          "open the event holding this call's result", () => openEvent(rk)));
      }
      body.appendChild(cell);
    }
    if (!(row.parts || []).length && !(row.tool_calls || []).length) {
      const n = document.createElement("div"); n.className = "part muted"; n.textContent = "· no parts"; body.appendChild(n);
    }
    // The panel is for reading one event, so its texts open unfolded.
    for (const b of body.querySelectorAll(".moretoggle:not(.wholetext)")) b.click();
    if (links.length) {
      const box = document.createElement("div"); box.className = "dlinks";
      const h = document.createElement("div"); h.className = "cellhead"; h.textContent = "links"; box.appendChild(h);
      for (const l of links) {
        const oe = other(l), ok = evKey(oe.lane, oe.event_idx);
        const inf = (edgeKinds[l.space] || {}).inferred_from;
        box.appendChild(drawerButton(`${l.frm === key ? "→" : "←"} ${inf ? "≈ " : ""}${l.space} · ${laneName(oe.lane)} #${oe.event_idx} ${evKind(oe)}`,
          ((edgeKinds[l.space] || {}).means || l.space) + (inf ? `\nmatched by text (${inf}), not by an id` : "")
            + (l.note ? "\n" + l.note : "") + "\nclick to open the other end", () => openEvent(ok)));
      }
      body.appendChild(box);
    }
    if (PRETTY.on) applyPretty();
  }
  // A producer's clock is an ISO string or epoch seconds; ms, or null where it is neither.
  function graphClock(v) {
    if (v == null || v === "") return null;
    const n = Number(v);
    const ms = Number.isFinite(n) ? n * 1000 : Date.parse(v);
    return Number.isNaN(ms) ? null : ms;
  }
  function drawGraph() {
    const nodes = events.map((ev) => {
      const t = graphClock(ev.started_at);
      return { key: evKey(ev.lane, ev.event_idx), lane: ev.lane || "", t: Number.isNaN(t) ? null : t,
               kind: ev.kind, label: graphLabel(ev), title: graphTitle(ev) };
    });
    const links = graphLinkList = graphLinks();
    const cols = graphColumns(laneList.map((L) => L.key), nodes, links);
    const lanes = cols.map((k) => ({ key: k, name: laneName(k), color: laneByKey[k].color }));
    const host = document.createElement("div");
    const note = document.createElement("div"); note.className = "graph-note";
    graphBox.replaceChildren(note, host);
    graphCtl = graphRender(host, lanes, nodes, links, openEvent, graphBox.clientWidth || 1200);
    const placed = graphCtl.placed;
    const by = {};
    for (const l of placed) if (l.drawn) by[l.space] = (by[l.space] || 0) + 1;
    const dropped = placed.filter((l) => l.dropped).length;
    const timed = nodes.filter((v) => v.t != null).length;
    note.textContent = `Each lane is a column. An event is below another only where its lane or a drawn link `
      + `says so${timed ? `, or both carry a clock (${timed} of ${nodes.length} do)` : ""}; level events in two `
      + `lanes are not claimed simultaneous — hover one to fade those it is not ordered against. `
      + (Object.keys(by).length ? "Links: " + Object.entries(by).map(([k, v]) => `${v} ${k}`).join(", ") + "."
         : "No link between lanes is stated for this rollout.")
      + (Object.keys(by).some((k) => (edgeKinds[k] || {}).inferred_from)
         ? " Dotted amber links are matched by equal text, not by an id, and order nothing." : "")
      + (dropped ? ` ${dropped} link${dropped === 1 ? "" : "s"} contradict an order already kept and are drawn dashed.` : "");
  }
  const setMode = (m) => { if (TRACE_MODE === m) return;
    TRACE_MODE = m; tstate.mode = TRACE_CHOICE = m === defMode ? "" : m;
    traceAddress(); syncMode(); renderBoxes(); };
  curTraceSetMode = setMode;   // so an address for THIS rollout can change the layout without a re-render
  // …and the same for the lane settings, so `?solo=…&fold=…` arriving at the rollout already on
  // screen changes the view instead of being silently dropped by the router's fast path.
  curTraceSetLanes = (hq) => {
    if (!multiLane) return;
    const want = hq.has("solo") ? (hq.get("solo") || "") : tstate.solo;
    const wantFold = hq.has("fold") ? hq.getAll("fold").filter(Boolean) : tstate.fold;
    if (want === tstate.solo && String(wantFold) === String(tstate.fold)) return;
    tstate.solo = want; tstate.fold = wantFold; traceAddress(); applyLaneView();
  };
  bLin.onclick = () => setMode("linear");
  bTree.onclick = () => setMode("tree");
  bGraph.onclick = () => setMode("graph");
  syncMode(); renderBoxes();

  const PAGE = 50;
  const loaded = new Set();   // page indices whose parts are hydrated
  const pageProm = {};        // memoized per-page promise: `await loadPage(p)` waits for real hydration
  let io = null;
  const loadPage = (page) => (pageProm[page] ??= hydratePage(page));
  async function hydratePage(page) {
    if (!sameRollout()) { delete pageProm[page]; return; }
    loaded.add(page);
    const off = page * PAGE;
    // spinner in each not-yet-hydrated slot of this page while the S3 texts read is in flight
    // (a cold read of a big part_texts sidecar can take tens of seconds — see the trace head note).
    // Hydration is by LIST POSITION: the offset window [off, off+PAGE) IS the i-th..(i+PAGE)-th events
    // the backend returns (same ORDER BY agent_ref, event_idx), so slotByPos[off+i] ↔ win[i] exactly —
    // correct even when event_idx repeats across agent_ref lanes (position-independent of event_idx).
    const pagePos = [];
    for (let pos = off; pos < Math.min(off + PAGE, total); pos++) {
      const slot = slotByPos[pos];
      if (slot && !slot.dataset.done) { slot.replaceChildren(slotSpinner()); pagePos.push(pos); }
    }
    { const hs = head.querySelector("#load-note"); if (hs) hs.textContent = "· loading parts…"; }
    let win;
    // parts read can be a cold DuckDB-httpfs pull of a big part_texts sidecar — allow it longer
    try { win = await traceApi(`${base}/events?include=parts&offset=${off}&limit=${PAGE}`, { timeout: 120000 }); }
    catch { loaded.delete(page); delete pageProm[page];   // drop memo so re-scroll retries
      pagePos.forEach(pos => { const s = slotByPos[pos]; if (s && !s.dataset.done) s.replaceChildren(); });
      const n = head.querySelector("#load-note"); if (n) n.textContent = "· parts failed to load — scroll to retry"; return; }
    // Navigated away mid-await. This used to `return` bare, which left the page in
    // `loaded`, left `pageProm[page]` memoized as a RESOLVED promise, and left a spinner
    // in every slot of the window -- so coming back re-entered the memo, did nothing, and
    // the slots said "loading parts" forever with no retry. Undo exactly what the catch
    // branch undoes.
    if (!sameRollout()) {
      loaded.delete(page); delete pageProm[page];
      pagePos.forEach(pos => { const s = slotByPos[pos]; if (s && !s.dataset.done) s.replaceChildren(); });
      return;
    }
    win.forEach((ev, i) => {
      const pos = off + i;
      const first = (ev.parts || []).find((p) => p.part_type === "text" && String(p.text ?? "").trim());
      if (first) {
        const k = evKey(ev.lane, ev.event_idx);
        graphPreview[k] = String(first.text).replace(/\s+/g, " ").trim().slice(0, 160);
        if (graphCtl && evByKey[k]) graphCtl.relabel(k, graphLabel(evByKey[k]));
      }
      const slot = slotByPos[pos];
      if (!slot || slot.dataset.done) return;
      slot.dataset.done = "1";
      slot.replaceChildren();   // clear the loading spinner
      const multiPart = (ev.parts || []).length > 1;
      (ev.parts || []).forEach((p, pos) => {
        // The anchor names the part's own index: a producer's part indices need not run 0..n-1.
        const pi = p.part_idx ?? pos;
        // each part gets a stable anchor + a header strip. copy buttons only when the event has
        // multiple parts (single-part copy is served by the event-level bar — no duplication).
        const w = document.createElement("div");
        // the part anchor carries the lane too, for the same reason the event anchor does
        w.className = "partwrap"; w.id = `p-${evKey(ev.lane, ev.event_idx)}-${pi}`;
        const frag = () => evHash(ev.event_idx, pi, ev.lane);
        w.append(partBar(frag, multiPart), partNode(p, base));
        slot.appendChild(w);
      });
      // An event whose slot renders to nothing must say which nothing it is. A turn that
      // carried no prose, and a turn whose prose is whitespace the producer wrote, are
      // different facts and look identical once drawn.
      const parts = ev.parts || [];
      const blank = parts.length
        && parts.every(p => p.part_type === "text" && !String(p.text ?? "").trim());
      if (!parts.length || blank) {
        const hasCall = (callsByEvent[ev.event_idx] || []).length > 0;
        const chars = parts.reduce((n, p) => n + (p.chars ?? String(p.text ?? "").length), 0);
        const note = document.createElement("div");
        note.className = "part muted";
        note.textContent = blank
          ? `· whitespace only — ${parts.length} part(s), ${chars} character(s), no visible text`
          : (hasCall ? "· no text — tool call only" : "· no parts");
        slot.appendChild(note);
      }
      if (io) io.unobserve(boxByPos[pos]);
    });
    // A window shorter than the slot range it spun up leaves the tail spinning with
    // nothing coming to fill it. Clear those rather than let them spin forever.
    pagePos.forEach(pos => { const s = slotByPos[pos]; if (s && !s.dataset.done) s.replaceChildren(); });
    if (PRETTY.on) applyPretty();   // newly-hydrated cells honor an active prettify toggle + refresh the chip/count
    const note = head.querySelector("#load-note");
    if (note) {
      // COUNT THE SLOTS, not the pages. `loaded.size * PAGE` counted pages that had only
      // STARTED and assumed every one was full, so with 68 events in pages of 50 it read
      // `50/68` and could never reach 68: the last page holds 18, and 2*50 only passes
      // `total` by accident. In tree mode display order is not list order, so pages
      // arrive out of order and the count was wrong in both directions.
      const seen = Object.values(slotByPos).filter(s => s && s.dataset.done).length;
      // Say what advances it. A bare `50/68 loaded` that never moves reads as a stalled
      // spinner -- it was reported as "loading parts indefinitely" -- when the remaining
      // events are simply below the fold and hydrate on scroll.
      if (seen >= total) note.remove();
      else note.textContent = `· ${seen}/${total} loaded — scroll for more`;
    }
  }
  // Hydrate a page when any of its events nears the viewport; preload ~2 screens ahead. The box's LIST
  // POSITION (dataset.pos) picks the page — independent of DISPLAY order, so tree mode doesn't change
  // what gets fetched, and lane-2 events map to the right offset page.
  io = new IntersectionObserver((entries) => {
    for (const e of entries) if (e.isIntersecting) loadPage(Math.floor(+e.target.dataset.pos / PAGE));
  }, { rootMargin: "1200px 0px" });
  const allBoxes = boxByPos;
  allBoxes.forEach(b => io.observe(b));
  curAllBoxes = async () => {
    for (let p = 0; p < Math.ceil(allBoxes.length / PAGE); p++) await loadPage(p);
    return allBoxes;
  };
  // Finish on our own. Scroll-driven hydration alone left a 68-event rollout sitting at
  // `50/68 loaded` until somebody scrolled the RIGHT pane -- and on a page whose content
  // already fits, or whose reader scrolls the sidebar instead, that scroll never comes.
  // It was reported twice as "loading parts indefinitely". The observer still does the
  // work near the viewport; this only guarantees the tail arrives without being asked.
  // Sequential and yielding, so a large rollout stays responsive: the biggest this corpus
  // holds is 1,762 events, which is 36 windows.
  (async () => {
    const pages = Math.ceil(allBoxes.length / PAGE);
    for (let p = 0; p < pages; p++) {
      if (!sameRollout()) return;          // navigated away: stop, and leave the memo clean
      if (!loaded.has(p)) await loadPage(p);
      await new Promise(r => setTimeout(r, 0));
    }
  })();
  // scroll-spy: highlight the tree node for the top-most event in a thin band near the viewport top,
  // keeping the sidebar synced to the reading position (doesn't touch fold state or the page scroll).
  // Keyed on the BOX, so the lane comes with it; the "top-most" visible box is chosen by DOM
  // offsetTop so it works in either linear or tree order.
  const visible = new Set();
  const spy = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (e.isIntersecting) visible.add(e.target); else visible.delete(e.target);
    }
    if (visible.size && sameRollout()) {
      let top = null, topOff = Infinity;
      for (const b of visible) if (b.offsetTop < topOff) { topOff = b.offsetTop; top = b; }
      if (top) markActive(top.dataset.lane || "", +top.dataset.idx, false);
    }
  }, { rootMargin: "-6% 0px -82% 0px" });
  allBoxes.forEach(b => spy.observe(b));
  loadPage(0);   // eager first page so content shows without waiting on the observer
  // deep-link: a trailing /e<idx>[-p<part>][~pretty] anchor jumps there (loads its page first).
  // Matched on the PATH, because the address now carries this view's settings behind a `?`. The
  // `~pretty` group still has to be MATCHED or an old anchor would not parse as an anchor at
  // all; it is acted on at the top of this function, before the address is rewritten.
  // The lane half of the anchor rides in `?lane=` (see ROLLOUT_FIELDS) — the path keeps the shape
  // every link ever copied off this page already has. Which PAGE to hydrate is a LIST POSITION, not
  // an event_idx: in the three-lane rollout the sub-agent's #0 is position 63, and dividing the
  // index by the page size would have loaded the wrong window.
  const jm = hashPath().match(/\/e(\d+)(?:-p(\d+))?(~pretty)?$/);
  if (jm) { const idx = +jm[1], part = jm[2] != null ? +jm[2] : null;
    const box = curEvLookup(tstate.lane, idx);
    // A link that named no lane resolved to one anyway; say which, so the ambiguous address
    // repairs itself on arrival and the next copy of it is unambiguous.
    if (box && multiLane && !tstate.lane) { tstate.lane = box.dataset.lane || ""; traceAddress(); }
    await loadPage(Math.floor((box ? +box.dataset.pos : idx) / PAGE));
    jumpToEvent(idx, part, tstate.lane); }
}

let curRollout = null;   // the rollout hash currently rendered (so an event deep-link just scrolls)
let TRACE_MODE = "linear";   // trace column layout: "linear" (timeline) | "tree" (results under calls) | "graph" (lanes as columns)
let TRACE_CHOICE = "";       // the layout the reader chose, session-persisted; "" = the rollout's default
let curTreeNodes = null; // event_idx -> sidebar node, for active-highlight sync
let curTraceSetMode = null;   // the live trace's layout setter, for same-rollout navigation
// The live trace's lane machinery, so the router's same-rollout fast path can act on an address
// that names a lane without re-rendering: resolve (lane, event_idx) to a box, un-hide a lane a
// link points into, and apply `?solo=`/`?fold=`.
let curEvLookup = null;
let curAllBoxes = null;   // every event box of the live trace, its parts hydrated
let curLaneReveal = null;
let curTraceSetLanes = null;
let curTreeSide = null;  // the sidebar scroll container, so spy/jump can keep the active node in view
// Sidebar as a real hierarchical tree rooted in the schema relationships: the top level is the
// event spine (events that aren't themselves a tool result); a tool call folds under its emitting
// event, the paired *result* event folds under that call, and every artifact folds under whatever
// it binds to (event / call / result-event via event_idx+call_idx+part_idx). Foldable per-node
// (▸/▾) with fold-all / unfold-all. Built from the texts-free /trace_map aggregate; every row
// deep-links via the same #/rollout/…/eN machinery the inline event anchors use.
// call vs result dots are deliberately distinct hues (amber vs teal), not the old near-identical golds
// The label for an event. A kind outside the registry's vocabulary is stored as `other` with the
// producer's literal in `kind_raw` — show the literal, because `other` is the one thing
// every escaped event has in common and so tells a reader nothing. A function declaration, not a
// const: both call sites sit above this line.
// A separator between two facts is only a separator when both are there. Rendering the
// list and dropping the blanks gives "reward=1 · 3 events" rather than "reward=1 · · events".
function bits(parts) {
  return parts.filter((x) => x !== "" && x != null).join(" · ");
}
// `task_id` travels through the address double-escaped, because the server percent-decodes
// once before routing. The reader sees the id, not our transport.
function unesc(v) {
  try { return decodeURIComponent(String(v ?? "")); } catch { return String(v ?? ""); }
}
function evKind(ev) {
  return (ev.kind === "other" && ev.kind_raw) ? ev.kind_raw : (ev.kind || "other");
}
const KIND_COLOR = { system: "#9aa3b5", message: "#3a6fd9", reasoning: "#7a3fb8",
                     tool_call: "#d9822b", tool_result: "#1f9e8f" };
// ── lanes ────────────────────────────────────────────────────────────────────────────────────
// A lane is ONE ordered sequence inside a rollout, and a rollout may hold several: the harness's
// own trace of the trial, the agent's session log, each sub-agent it spawned, a monitor. The
// value is `kind` or `kind/instance`. `event_idx` numbers PER LANE, so two lanes both start at 0
// — which is the whole reason any of this exists: without a lane the list reads #44, #45, #46,
// #0, #1 and nothing on the page says why.
//
// Colour is ORDINAL — the lane's column in the rail — not semantic. `subagent/…` is the second
// lane in this rollout and the third in the next one, and a palette keyed on the kind would have
// to invent an entry for every kind nobody has seen yet. Eight columns, then one shared overflow
// column marked `~`, which is what git itself does (`graph_max_lanes`, `graph_needs_truncation`).
const LANE_PAL = ["#3a6fd9", "#c2681f", "#0f8f82", "#8a3fb8", "#b3325c", "#2e7d32",
                  "#7a6a1f", "#4a6b8a"];
const LANE_CAP = LANE_PAL.length;
const LANE_OVERFLOW_COLOR = "#8a8f9a";
// The kind is a token the producer chose; the gloss is what it means to a reader. An unknown kind
// keeps its own token rather than being glossed into something invented.
const LANE_GLOSS = { trace: "harness trace", session: "agent session", subagent: "sub-agent",
                     monitor: "monitor", verifier: "verifier" };
const laneKind = (l) => String(l || "").split("/")[0];
const laneInst = (l) => String(l || "").split("/").slice(1).join("/");
// One slug per lane name: every character outside [A-Za-z0-9-] becomes `_<hex>_`, so
// `worker/a` and `worker_a` stay apart.
const laneSlug = (l) => String(l || "").replace(/[^A-Za-z0-9-]/gu, (c) => `_${c.codePointAt(0).toString(16)}_`);
// `session/ac0c9db6-f4e1-43b0-bdfa-eb99840a2acb` is 44 characters and would dominate every row.
// The kind is what a reader needs inline; the instance is on the chip's own row in the header.
const laneShort = (l) => { const i = laneInst(l);
  return i ? `${laneKind(l)}·${i.replace(/^agent-/, "").slice(0, 6)}` : laneKind(l) || "main"; };
const laneName = (l) => l || "main";
// A lane in `solo`/`fold`: the unnamed lane is "/", since "" already means no lane.
const laneId = (l) => l || "/";
const laneColor = (col) => col >= LANE_CAP ? LANE_OVERFLOW_COLOR : LANE_PAL[col];
// Rail geometry: a 12 px pitch per column plus a 9 px left inset, which is the whole cost of the
// gutter — 3 lanes is 48 px, the 8-column cap is 117 px.
const LANE_X = (col) => 9 + Math.min(col, LANE_CAP) * 12;
const laneRailWidth = (n) => LANE_X(Math.min(n, LANE_CAP + 1) - 1) + 12;
// The key an event is addressed by. `event_idx` alone stopped being unique the moment a rollout
// held two lanes; every map, DOM id and deep link below carries both.
const evKey = (lane, idx) => (lane ? laneSlug(lane) + "-" : "") + idx;
function traceTree(side, tmap, evHash, laneList, laneByKey) {
  side.replaceChildren();
  if (!tmap || !tmap.events) { side.hidden = true; return null; }
  side.hidden = false;
  // The sidebar groups by LANE. It used to group by `agent_ref`, which is the empty string on
  // every row in the corpus — so there was exactly one group, and the list read #44, #45, #46,
  // #0, #1, #2 with nothing to say that a second sequence had started. Everything keyed on an
  // event is keyed on (lane, event_idx) for the same reason the trace column is.
  const evByKey = {};
  for (const e of tmap.events) evByKey[evKey(e.lane, e.event_idx)] = e;
  const callsByEvent = {}, artByEvent = {}, looseArt = [], nestedResult = new Set();
  for (const c of tmap.tool_calls || []) {
    (callsByEvent[evKey(c.lane, c.event_idx)] ??= []).push(c);
    if (c.result_event_idx != null) nestedResult.add(evKey(c.lane, c.result_event_idx));
  }
  // `artifacts` carries no lane (the table has none), so a bound artifact can only be placed when
  // exactly one lane holds that event_idx. When several do, the binding is genuinely ambiguous and
  // the artifact goes to the loose bucket rather than under a guess.
  const lanesForIdx = {};
  for (const ev of tmap.events) (lanesForIdx[ev.event_idx] ??= new Set()).add(ev.lane || "");
  for (const a of tmap.artifacts || []) {
    const owners = a.event_idx == null ? null : lanesForIdx[a.event_idx];
    if (!owners || owners.size !== 1) { a.__ambiguous = !!owners; looseArt.push(a); }
    else (artByEvent[evKey([...owners][0], a.event_idx)] ??= []).push(a);
  }
  const lanes = {};
  for (const ev of tmap.events) (lanes[ev.lane || ""] ??= []).push(ev);
  const laneNames = (laneList && laneList.length ? laneList.map(L => L.key) : Object.keys(lanes))
    .filter(k => lanes[k]);
  const nodeByKey = { __byIdx: {}, __lanes: {} };
  const toggles = [];   // {set(bool)} for every foldable node, for fold-all / unfold-all
  const goto = (lane, idx) => (e) => { e.stopPropagation(); location.hash = evHash(idx, null, lane); };

  const bar = document.createElement("div"); bar.className = "tree-bar";
  const h = document.createElement("span"); h.className = "tree-h";
  const laneTail = laneNames.length > 1 ? ` · ${laneNames.length} lanes` : "";
  h.textContent = (tmap.total_events > tmap.event_cap
    ? `trace · showing first ${tmap.event_cap} of ${tmap.total_events} events`
    : `trace · ${tmap.total_events} events`) + laneTail;
  const bU = document.createElement("button"); bU.className = "tree-btn"; bU.textContent = "unfold all";
  const bF = document.createElement("button"); bF.className = "tree-btn"; bF.textContent = "fold all";
  bU.onclick = () => toggles.forEach(t => t.set(true));
  bF.onclick = () => toggles.forEach(t => t.set(false));
  const pl = document.createElement("label"); pl.className = "tree-pretty";
  const pc = document.createElement("input"); pc.type = "checkbox"; pc.className = "js-pretty";
  pl.append(pc, document.createTextNode(" prettify (decode \\n \\t \\uXXXX)"));
  bar.append(h, bU, bF, pl);   // sticky, so prettify stays reachable while scrolling a long trace
  side.appendChild(bar);

  // a node = wrapper[ row(caret,dot,idx,label,meta) + kids ]; finalize() decides caret vs spacer
  function node(depth, spec) {
    const wrap = document.createElement("div");
    const row = document.createElement("div"); row.className = "trow";
    row.style.paddingLeft = (4 + depth * 13) + "px";
    const caret = document.createElement("span"); caret.className = "caret";
    const dot = document.createElement("span"); dot.className = "dot";
    if (spec.dot) dot.style.background = spec.dot; else dot.style.visibility = "hidden";
    const idx = document.createElement("span"); idx.className = "tidx"; idx.textContent = spec.idx || "";
    const lbl = document.createElement("span"); lbl.className = "tlbl" + (spec.lblClass ? " " + spec.lblClass : "");
    lbl.textContent = spec.label;
    const meta = document.createElement("span"); meta.className = "tmeta"; meta.textContent = spec.meta || "";
    row.append(caret, dot, idx, lbl, meta);
    if (spec.title) row.title = spec.title;
    if (spec.onClick) { row.classList.add("nav"); row.onclick = spec.onClick; }
    const kids = document.createElement("div"); kids.className = "tkids";
    wrap.append(row, kids);
    wrap.__row = row; wrap.__kids = kids; wrap.__caret = caret;
    return wrap;
  }
  function finalize(wrap, open) {
    const { __kids: kids, __caret: caret } = wrap;
    if (!kids.childNodes.length) { caret.classList.add("spacer"); return; }
    let o = !!open;
    const apply = () => { caret.textContent = o ? "▾" : "▸"; kids.style.display = o ? "" : "none"; };
    caret.onclick = (e) => { e.stopPropagation(); o = !o; apply(); };
    wrap.__expand = () => { if (!o) { o = true; apply(); } };
    toggles.push({ set: (v) => { o = v; apply(); } });
    apply();
  }
  const artMeta = (a) => `${a.storage}${a.size_bytes ? " · " + a.size_bytes + "b" : ""}`;
  function addArtifact(a, depth, host, lane) {
    const w = node(depth, {
      dot: "#2f9145", lblClass: "tart",
      label: (a.kind || "data") + " " + (a.name || "#" + a.artifact_idx),
      meta: a.part_idx != null ? "p" + a.part_idx : (a.call_idx != null ? "call" + a.call_idx : ""),
      title: `${a.kind || ""} ${a.name || ""} · ${artMeta(a)}`
        + (a.__ambiguous ? `\nbinds to event #${a.event_idx}, which exists in more than one lane —`
                         + ` the artifacts table carries no lane, so which one is unknown` : ""),
      onClick: a.event_idx != null && lane !== undefined ? goto(lane, a.event_idx) : null,
    });
    host.appendChild(w); finalize(w, false);
  }
  // build an event subtree: event-level artifacts, then each tool call (with its result event nested)
  const callTail = (c) => c.status && !["unknown", "ok"].includes(c.status) ? " [" + c.status + "]" : "";
  function addEvent(ev, depth, host, open) {
    const lane = ev.lane || "";
    const key = evKey(lane, ev.event_idx);
    const kind = evKind(ev);
    const bits = [];
    if (ev.n_parts) bits.push(ev.n_parts + "p");
    if (ev.completion_tokens) bits.push(ev.completion_tokens + "t");
    const evCalls = callsByEvent[key] || [];
    // exactly one call → fold the call onto the event row (#N tool_call → tool(...)), no child row
    const solo = evCalls.length === 1 ? evCalls[0] : null;
    const soloLbl = solo ? " → " + (solo.tool_name || "tool") + callTail(solo) : "";
    const w = node(depth, {
      dot: KIND_COLOR[kind] || KIND_COLOR[ev.kind] || "#9aa3b5", idx: "#" + ev.event_idx,
      label: kind + (ev.role ? "·" + ev.role : "") + soloLbl, meta: bits.join(" "),
      title: (solo && solo.args_head ? `#${ev.event_idx} ${kind} → ${solo.tool_name}(${String(solo.args_head).slice(0, 90)})`
        : `#${ev.event_idx} ${kind}${ev.role ? " · " + ev.role : ""}${soloLbl}`)
        + (lane ? `\nlane ${lane}` : ""),
      onClick: goto(lane, ev.event_idx),
    });
    nodeByKey[key] = w;
    nodeByKey.__byIdx[ev.event_idx] ??= w;
    host.appendChild(w);
    const arts = artByEvent[key] || [];
    for (const a of arts) if (a.call_idx == null) addArtifact(a, depth + 1, w.__kids, lane);
    // inline reasoning lives as reasoning-channel parts inside the event — surface it as a child
    if (ev.n_reasoning_parts) {
      const rn = node(depth + 1, { dot: KIND_COLOR.reasoning, label: "thinking",
        meta: ev.n_reasoning_parts > 1 ? ev.n_reasoning_parts + "×" : "",
        title: "inline reasoning", onClick: goto(lane, ev.event_idx) });
      rn.__row.classList.add("tthink");
      w.__kids.appendChild(rn); finalize(rn, false);
    }
    for (const c of evCalls) {
      // merged: the call's own row is the event row; its children (call artifacts + result) nest under
      // the event. Non-merged (multi-call): one call row per call, children under it (unchanged).
      let ch, cdepth;
      if (solo) { ch = w; cdepth = depth + 1; }
      else {
        ch = node(depth + 1, {
          dot: KIND_COLOR.tool_call, lblClass: "tnode-call",
          label: "→ " + (c.tool_name || "tool") + callTail(c),
          title: c.args_head ? `${c.tool_name}(${String(c.args_head).slice(0, 100)})` : (c.tool_name || "tool"),
          onClick: goto(lane, c.result_event_idx != null ? c.result_event_idx : ev.event_idx),
        });
        w.__kids.appendChild(ch); cdepth = depth + 2;
      }
      for (const a of arts) if (a.call_idx === c.call_idx) addArtifact(a, cdepth, ch.__kids, lane);
      if (c.result_event_idx != null) {
        const rev = evByKey[evKey(lane, c.result_event_idx)];
        if (rev) addEvent(rev, cdepth, ch.__kids, false);
        else {
          const rl = node(cdepth, { dot: KIND_COLOR.tool_result, idx: "#" + c.result_event_idx,
            label: "result", onClick: goto(lane, c.result_event_idx) });
          ch.__kids.appendChild(rl); finalize(rl, false);
        }
      }
      if (!solo) finalize(ch, false);
    }
    finalize(w, open);
  }

  // One group per lane, with its own header carrying the lane's colour, its count, and the index
  // range — so the restart from #62 to #0 is never unexplained. The header folds the whole lane,
  // which is what makes a 274-event sub-agent one row instead of a scroll.
  for (const lane of laneNames) {
    let host = side;
    if (laneNames.length > 1) {
      const L = (laneByKey && laneByKey[lane]) || null;
      const rows = lanes[lane];
      const hdr = document.createElement("div"); hdr.className = "tlane";
      hdr.style.setProperty("--c", L ? L.color : "#8a8f9a");
      const car = document.createElement("span"); car.className = "tcar"; car.textContent = "▾";
      const sw = document.createElement("i"); sw.className = "sw";
      const nm = document.createElement("span"); nm.className = "tnm"; nm.textContent = laneShort(lane);
      const n = document.createElement("span"); n.className = "tn";
      n.textContent = `${rows.length} · #${rows[0].event_idx}–${rows[rows.length - 1].event_idx}`;
      hdr.append(car, sw, nm, n);
      hdr.title = `${lane}\n${rows.length} events, numbered from ${rows[0].event_idx} in this lane`;
      const grp = document.createElement("div"); grp.className = "tlgrp";
      let open = true;
      const setOpen = (v) => { open = v; car.textContent = v ? "▾" : "▸"; grp.classList.toggle("hid", !v); };
      hdr.onclick = () => setOpen(!open);
      // deliberately NOT in `toggles`: "fold all" folds each event's children so the spine stays
      // readable, and folding the lanes away too would leave three rows and no spine.
      // the trace column's own fold/solo drives this group too, so the two halves of the page
      // never disagree about which lanes are on screen
      nodeByKey.__lanes[lane] = { hdr, grp, setOpen };
      side.append(hdr, grp);
      host = grp;
    }
    for (const ev of lanes[lane])
      if (!nestedResult.has(evKey(ev.lane, ev.event_idx))) addEvent(ev, 0, host, false);
  }
  if (looseArt.length) {
    const amb = looseArt.filter(a => a.__ambiguous).length;
    const g = node(0, { label: `rollout artifacts (${looseArt.length})`, lblClass: "tart",
      title: amb ? `${amb} of these bind to an event_idx that exists in more than one lane; the`
                 + ` artifacts table carries no lane, so they cannot be placed` : "" });
    side.appendChild(g);
    for (const a of looseArt) addArtifact(a, 1, g.__kids);
    finalize(g, false);
  }
  return nodeByKey;
}
function scrollSideInto(row) {   // scroll ONLY the sidebar (never the page) to reveal a row
  const side = curTreeSide;
  if (!side || !row) return;
  const sr = side.getBoundingClientRect(), rr = row.getBoundingClientRect();
  if (rr.top < sr.top + 6) side.scrollTop -= (sr.top + 6 - rr.top);
  else if (rr.bottom > sr.bottom - 6) side.scrollTop += (rr.bottom - (sr.bottom - 6));
}
// Highlight the tree node for `idx`. expand=true (click/deep-link) opens collapsed ancestors;
// expand=false (scroll-spy) leaves fold state alone and highlights the nearest visible ancestor.
function markActive(lane, idx, expand) {
  if (!curTreeNodes) return;
  // (lane, event_idx) is the key. A link that carries no lane — everything copied before lanes
  // existed — falls back to the first node with that index rather than highlighting nothing.
  const wrap = curTreeNodes[evKey(lane, idx)] || curTreeNodes.__byIdx[idx];
  if (!wrap) return;
  if (expand) {
    for (let el = wrap.parentElement; el; el = el.parentElement)
      if (el.classList.contains("tkids") && el.parentElement.__expand) el.parentElement.__expand();
  }
  let w = wrap;
  while (w.__row && w.__row.offsetParent === null) {   // hidden inside a collapsed branch
    const kids = w.parentElement, pw = kids && kids.parentElement;
    if (!pw || !pw.__row) break;
    w = pw;
  }
  const row = w.__row;
  if (curTreeNodes.__active && curTreeNodes.__active !== row) curTreeNodes.__active.classList.remove("active");
  row.classList.add("active"); curTreeNodes.__active = row;
  scrollSideInto(row);
}
function flash(el) { el.classList.remove("flash"); void el.offsetWidth; el.classList.add("flash"); }
function jumpToEvent(idx, part, lane) {
  const el = (curEvLookup && curEvLookup(lane || "", idx)) || document.getElementById("ev-" + idx);
  if (!el) { markActive(lane || "", idx, true); return; }
  // The box knows its own lane, so a link that named none still lands in the right sequence and
  // the sidebar highlights the right node. A folded or soloed-away lane is opened first — a link
  // must never land on a row the current view has collapsed.
  const lane2 = el.dataset.lane || "";
  if (curLaneReveal) curLaneReveal(lane2);
  markActive(lane2, idx, true);
  el.scrollIntoView({ behavior: "smooth", block: "start" });
  flash(el);
  if (part != null) {
    const pe = document.getElementById(`p-${el.dataset.key || idx}-${part}`);
    if (pe) { pe.scrollIntoView({ behavior: "smooth", block: "center" }); flash(pe); }
  }
  markText(el, part);
}

// ── text highlights: `#…/e12?text=into%20Japanese` ───────────────────────────────────────────
// A link to an event may name text in it: `text=` in the spelling of a URL text fragment,
// `[prefix-,]start[,end][,-suffix]`, each piece percent-encoded, repeatable. Every match in the
// event (in the part, for `-p<part>`) is highlighted and the first is scrolled to; matching
// ignores case and treats any run of whitespace as one. Text a part shows only part of is read
// whole before it is reported missing.
// The browser's own spelling, `#…/e12:~:text=…` ("copy link to highlight"), is removed from
// location.hash before any script reads it; the navigation entry and hashchange's newURL still
// carry it, and it is moved into `?text=` before the page's router reads the address.
const hashTexts = (h) => String(h == null ? location.hash : h).split("?").slice(1).join("?")
  .split("&").filter(s => s.startsWith("text=") && s.length > 5).map(s => s.slice(5));
function foldTextDirective(url) {
  const at = String(url || "").indexOf("#");
  const frag = at < 0 ? "" : url.slice(at + 1), i = frag.indexOf(":~:");
  if (i < 0) return;
  const texts = frag.slice(i + 3).split("&").filter(s => s.startsWith("text=") && s.length > 5);
  const head = frag.slice(0, i);
  if (!texts.length || hashPath("#" + head) !== hashPath(location.hash.split(":~:")[0])) return;
  history.replaceState(history.state, "", "#" + head + (head.includes("?") ? "&" : "?") + texts.join("&"));
}
foldTextDirective(location.hash);
foldTextDirective(performance.getEntriesByType?.("navigation")[0]?.name);
window.addEventListener("hashchange", (e) => foldTextDirective(e.newURL));

// Search over the rollout's messages and tool call arguments: every word must occur in one text;
// `*` and `?` glob. Hits come from the store's own text index, so nothing here waits for parts to
// stream in. Enter and Shift+Enter walk the hits; choosing one opens the event and marks the words
// with the same `?text=` link a copied highlight uses.
function mountTraceSearch(main, base, evHash) {
  const host = main.querySelector(".tsearch");
  if (!host) return;
  const input = document.createElement("input");
  input.type = "search"; input.className = "qinput"; input.placeholder = "search messages…  ( / )";
  input.title = "every word must occur in one message or tool call argument; * and ? glob";
  input.spellcheck = false; input.autocomplete = "off";
  const count = document.createElement("span"); count.className = "tcount";
  const panel = document.createElement("div"); panel.className = "thits"; panel.hidden = true;
  host.append(input, count);
  main.querySelector(".bar").after(panel);
  let hits = [], cur = -1, seq = 0, ctl = null, timer = null, words = [];
  const wordsOf = (q) => [...q.toLowerCase().matchAll(/"([^"]+)"|(\S+)/g)].map((m) => m[1] || m[2]);
  // the literal run of a glob word, for the highlight link
  const literal = (w) => w.split(/[*?]+/).sort((a, b) => b.length - a.length)[0];
  const mark = (snip) => {
    let out = esc(snip);
    for (const w of words.map(literal).filter(Boolean)) {
      const re = new RegExp("(" + esc(w).replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + ")", "gi");
      out = out.replace(re, "<mark>$1</mark>");
    }
    return out;
  };
  const go = (i) => {
    if (!hits.length) return;
    cur = (i + hits.length) % hits.length;
    const h = hits[cur];
    const at = evHash(h.event_idx, h.kind === "part" ? h.idx : null, h.lane);
    const texts = words.map(literal).filter(Boolean).map((w) => "text=" + encodeURIComponent(w));
    location.hash = at + (texts.length ? (at.includes("?") ? "&" : "?") + texts.join("&") : "");
    count.textContent = `${cur + 1}/${hits.length}${hits.truncated ? "+" : ""}`;
    panel.querySelectorAll(".thit").forEach((n, k) => n.classList.toggle("on", k === cur));
    panel.querySelector(".thit.on")?.scrollIntoView({ block: "nearest" });
  };
  const paint = () => {
    panel.replaceChildren();
    for (const [i, h] of hits.entries()) {
      const row = document.createElement("div"); row.className = "thit";
      row.innerHTML = `<span class="mono">#${h.event_idx}${h.lane ? " · " + esc(h.lane) : ""} · ${
        h.kind === "part" ? "part " : "call "}${h.idx}</span><span class="tsnip">${mark(h.snippet)}</span>`;
      row.onmousedown = (e) => { e.preventDefault(); go(i); };
      panel.appendChild(row);
    }
    panel.hidden = !hits.length && !input.value.trim();
    if (!hits.length && input.value.trim()) panel.innerHTML = '<div class="thit none">no message holds every word</div>';
  };
  const run = async () => {
    const q = input.value.trim(), my = ++seq;
    if (ctl) ctl.abort();
    if (!q) { hits = []; cur = -1; count.textContent = ""; panel.hidden = true; return; }
    ctl = new AbortController();
    try {
      const r = await api(`${base}/search?q=${encodeURIComponent(q)}`, { signal: ctl.signal });
      if (my !== seq) return;
      words = wordsOf(q); hits = r.hits; hits.truncated = r.truncated; cur = -1;
      count.textContent = hits.length ? `${hits.length}${r.truncated ? "+" : ""}` : "0";
      paint();
    } catch (e) {
      if (e.name === "AbortError" || my !== seq) return;
      count.textContent = ""; panel.hidden = false;
      panel.innerHTML = `<div class="thit none">${esc(e.detail || "search failed")}</div>`;
    }
  };
  input.oninput = () => { clearTimeout(timer); timer = setTimeout(run, 180); };
  input.onkeydown = (e) => {
    if (e.key === "Enter") { e.preventDefault(); clearTimeout(timer); if (!hits.length) run(); else go(cur + (e.shiftKey ? -1 : 1)); }
    else if (e.key === "Escape") { input.blur(); panel.hidden = true; }
  };
  input.onfocus = () => { if (hits.length || input.value.trim()) panel.hidden = false; };
}

function textTarget(raw) {
  const dec = (s) => { try { return decodeURIComponent(s); } catch (_) { return s; } };
  const c = raw.split(",");
  let prefix = "", suffix = "";
  if (c.length > 1 && c[0].endsWith("-")) prefix = dec(c.shift().slice(0, -1));
  if (c.length > 1 && c[c.length - 1].startsWith("-")) suffix = dec(c.pop().slice(1));
  if (c.length < 1 || c.length > 2 || !c[0]) return null;
  const pat = (s) => s.trim().split(/\s+/).map(w => w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("\\s+");
  const start = dec(c[0]), end = c[1] ? dec(c[1]) : "";
  if (!start.trim() || (c[1] && !end.trim())) return null;
  const re = (pre, suf) => new RegExp((pre.trim() ? `(${pat(pre)}\\s*)` : "()")
    + `(${pat(start)}${end ? `[\\s\\S]*?${pat(end)}` : ""})`
    + (suf.trim() ? `(?=\\s*${pat(suf)})` : ""), "giu");
  // A browser writes its context from the page as drawn, buttons included; when that context is
  // not in the trace's text, the text alone is matched.
  return { strict: re(prefix, suffix), loose: prefix.trim() || suffix.trim() ? re("", "") : null };
}
// Ranges of every match in each [data-raw] node: the text a part, a tool call or a note shows.
function textRanges(nodes, res) {
  const out = [];
  for (const n of nodes) {
    const tn = [], w = document.createTreeWalker(n, NodeFilter.SHOW_TEXT);
    let s = "";
    for (let t; (t = w.nextNode());) { tn.push([t, s.length]); s += t.data; }
    const at = (o, endSide) => {
      let k = tn.length - 1;
      while (k > 0 && (tn[k][1] > o || (endSide && tn[k][1] === o))) k--;
      return [tn[k][0], o - tn[k][1]];
    };
    for (const re of res) {
      re.lastIndex = 0;
      for (let m; (m = re.exec(s));) {
        if (!m[2]) { re.lastIndex++; continue; }
        const r = document.createRange(), a = m.index + m[1].length;
        r.setStart(...at(a, false)); r.setEnd(...at(a + m[2].length, true));
        out.push(r);
      }
    }
  }
  return out;
}
// One mark at a time; it follows its text when a host, prettify or a whole-text read rewrites it.
let textMark = { gen: 0, watch: null };
const clearText = () => {
  textMark.gen++; textMark.watch?.disconnect(); textMark.watch = null;
  window.CSS?.highlights?.delete("trace-text");
};
function paintText(ranges) {
  if (window.CSS?.highlights && window.Highlight) CSS.highlights.set("trace-text", new Highlight(...ranges));
  else { const sel = getSelection(); sel.removeAllRanges(); if (ranges.length) sel.addRange(ranges[0]); }
}
async function markText(el, part) {
  clearText();
  const gen = textMark.gen;
  document.querySelectorAll(".textmiss").forEach(n => n.remove());
  const raws = hashTexts();
  const res = raws.map(textTarget);
  if (!res.some(Boolean)) return;
  const slot = el.querySelector(".parts-slot");
  if (slot && !slot.dataset.done) await new Promise((done) => {
    const mo = new MutationObserver(() => { if (slot.dataset.done) { mo.disconnect(); done(); } });
    mo.observe(slot, { attributes: true, attributeFilter: ["data-done"] });
    setTimeout(() => { mo.disconnect(); done(); }, 120000);
  });
  const scope = (part != null && document.getElementById(`p-${el.dataset.key}-${part}`)) || el;
  const search = (root, t) => {
    const nodes = [...root.querySelectorAll("[data-raw]")];
    const r = textRanges(nodes, [t.strict]);
    return r.length || !t.loose ? r : textRanges(nodes, [t.loose]);
  };
  // Where each named text is searched: the linked event (or part) first.
  const plan = res.map(t => t && [scope, t]);
  const missing = () => plan.some(x => x && !search(...x).length);
  // Text a part shows only the start of, a prompt not yet opened, a tool call's folded arguments:
  // read whole before any named text is looked for elsewhere.
  if (missing()) await Promise.all([...scope.querySelectorAll("[data-loadable]")].map(n => n.loadWhole()));
  // Then the rest of the rollout, in order, as a browser's find would: the first event holding it.
  const elsewhere = [];
  if (missing() && curAllBoxes) {
    const boxes = (await curAllBoxes()).filter(b => b && b !== el);
    plan.forEach((x, i) => {
      if (!x || search(...x).length) return;
      const box = boxes.find(b => search(b, x[1]).length);
      if (box) { plan[i] = [box, x[1]]; elsewhere.push(box); }
    });
  }
  if (gen !== textMark.gen) return;
  const say = (text) => {
    const n = document.createElement("div"); n.className = "part muted textmiss"; n.textContent = text;
    (slot || el).prepend(n);
  };
  const dec = (t) => { try { return decodeURIComponent(t); } catch (_) { return t; } };
  const absent = raws.filter((_, i) => !plan[i] || !search(...plan[i]).length);
  if (absent.length) say("· the linked text is not in this trace: " + absent.map(dec).join(" · "));
  if (elsewhere.length) say("· the linked text is in "
    + [...new Set(elsewhere)].map(b => `#${b.dataset.idx}` + (b.dataset.lane ? ` (${b.dataset.lane})` : "")).join(", ")
    + ", not in this " + (scope === el ? "event" : "part"));
  const ranges = () => plan.flatMap(x => x ? search(...x) : []);
  const first = ranges();
  if (!first.length) return;
  for (const box of elsewhere) if (curLaneReveal) curLaneReveal(box.dataset.lane || "");
  for (const r of first) {
    for (let n = r.startContainer.parentElement; n && n !== document.body; n = n.parentElement) {
      if (n.tagName === "DETAILS") n.open = true;
      if (n.classList.contains("pmsg-t")) n.classList.add("whole");
      n.expand?.();
      if (n.classList.contains("event")) break;
    }
  }
  paintText(first);
  let queued = false;
  textMark.watch = new MutationObserver(() => {
    if (queued) return; queued = true;
    requestAnimationFrame(() => { queued = false; if (gen === textMark.gen) paintText(ranges()); });
  });
  textMark.watch.observe(elsewhere.length ? (el.closest(".trace-main") || document.body) : scope, { childList: true, subtree: true, characterData: true });
  const target = first.find(r => scope.contains(r.startContainer)) || first[0];
  const box = target.startContainer.parentElement?.closest(".event");
  if (box && box !== el) { markActive(box.dataset.lane || "", +box.dataset.idx, true); flash(box); }
  const at = target.getBoundingClientRect();
  window.scrollTo({ top: window.scrollY + at.top - window.innerHeight / 2, behavior: "smooth" });
}
// A link copied while text inside `scope` is selected names that text.
function withSelection(hash, scope) {
  const sel = getSelection(), t = sel && !sel.isCollapsed ? sel.toString().trim() : "";
  if (!t || !scope || !scope.contains(sel.anchorNode) || !scope.contains(sel.focusNode)) return hash;
  const enc = encodeURIComponent(t.replace(/\s+/g, " ").slice(0, 300)).replace(/[-,&]/g, c => "%" + c.charCodeAt(0).toString(16).toUpperCase());
  return hash + (hash.includes("?") ? "&" : "?") + "text=" + enc;
}

// ── a model call's prompt ─────────────────────────────────────────────────────────────────
// A call's request lists its messages; most calls re-send an earlier call's messages and add a
// few. `/prompts` names, per call, the earlier call it continues (its base), how many of the
// base's messages it kept, dropped and added, and what that makes it. The chart draws the prompt's
// size per call and marks the calls where the agent rewrote its history; each model-call event
// carries its own prompt, opened on demand, with the kept run folded to one line.
const PROMPT_CLASS = {
  compaction:  { label: "compaction",  color: "#eb6834", shape: "tri",
                 says: "drops part of its base's history, and its prompt is at most half its base's or holds fewer messages" },
  new_context: { label: "new context", color: "#2a78d6", shape: "diamond",
                 says: "shares no leading message with any earlier call (a summariser, a fresh sub-context)" },
  rewrite:     { label: "rewrite",     color: "#1baf7a", shape: "dot",
                 says: "drops or edits part of its base's history without shrinking by half" },
};
const PROMPT_QUIET = { first: "first call", append: "continues the previous call",
                       repeat: "sends the previous call's request again, unchanged",
                       resume: "continues an earlier call, not the previous one" };
const kfmt = (n) => n == null ? "—" : n >= 1e6 ? (n / 1e6).toFixed(n >= 1e7 ? 0 : 1) + "M"
  : n >= 1e3 ? (n / 1e3).toFixed(n >= 1e4 ? 0 : 1) + "k" : String(n);
const promptSize = (c, byTok) => byTok ? c.prompt_tokens : c.chars;
function promptCallLine(c) {
  const cls = PROMPT_CLASS[c.class];
  const what = cls ? cls.label : (PROMPT_QUIET[c.class] || c.class);
  if (c.has_request === false)
    return `#${c.event_idx} · ${what} · ${kfmt(c.prompt_tokens)} in`
      + (c.cached_tokens != null ? ` (${kfmt(c.cached_tokens)} cached)` : "")
      + (c.completion_tokens != null ? ` · ${kfmt(c.completion_tokens)} out` : "");
  const base = c.base_event_idx == null ? "" : ` · base #${c.base_event_idx}: kept ${c.shared}, dropped ${c.dropped}, +${c.added}`;
  return `#${c.event_idx} · ${what} · ${kfmt(c.prompt_tokens)} tok · ${c.n_messages} messages`
    + ` · ${kfmt(c.chars)} chars${base}`;
}
function promptShape(cls, x, y, r) {
  const spec = PROMPT_CLASS[cls];
  const ns = "http://www.w3.org/2000/svg";
  let el;
  if (spec.shape === "tri") {
    el = document.createElementNS(ns, "path");
    el.setAttribute("d", `M${x - r},${y - r * 0.8} L${x + r},${y - r * 0.8} L${x},${y + r} Z`);
  } else if (spec.shape === "diamond") {
    el = document.createElementNS(ns, "path");
    el.setAttribute("d", `M${x},${y - r} L${x + r},${y} L${x},${y + r} L${x - r},${y} Z`);
  } else {
    el = document.createElementNS(ns, "circle");
    el.setAttribute("cx", x); el.setAttribute("cy", y); el.setAttribute("r", r * 0.8);
  }
  el.setAttribute("fill", spec.color);
  el.style.stroke = "var(--panel, #fcfcfb)"; el.setAttribute("stroke-width", "2");
  return el;
}
// The rollout's context panel: one line of counts, the chart, and the table of every call that
// did more than append. `jump(c)` opens that call's event and its prompt.
function promptContext(sum, jump) {
  const box = document.createElement("div"); box.className = "pctx";
  const calls = sum.calls || [];
  const usage = sum.source === "usage";
  const n = sum.n_model_calls, k = usage ? calls.length : sum.n_with_request;
  const marked = calls.filter(c => PROMPT_CLASS[c.class]);
  const head = $(`<h2>model calls · prompts</h2>`).firstChild;
  box.appendChild(head);
  const line = document.createElement("div"); line.className = "pctx-note";
  line.textContent = usage
    ? `${n} model responses, sized by the input tokens each reported (the producer stored no request). `
    : k === n
    ? `${n} model calls, each with the request it sent. `
    : `requests stored for ${k} of ${n} model calls — the producer kept no request for the rest. `;
  if (k) line.textContent += marked.length
    ? `${marked.length} ${usage ? "shrank the context" : "rewrote their history"}: ${Object.entries(PROMPT_CLASS)
        .map(([c, s]) => `${sum.classes[c] || 0} ${s.label}`).join(", ")}.`
    : usage ? "The context only grew." : "None rewrote its history: every call continues an earlier one.";
  box.appendChild(line);
  if (k < 2) return box;
  const byTok = calls.every(c => c.prompt_tokens != null);
  // Drawn at the width it is shown at, so its text is the page's size rather than scaled.
  const W = Math.max(480, Math.round((main && main.clientWidth) || 1000) - 2), H = 170, L = 48, R = 10, T = 10, B = 22;
  const ys = calls.map(c => promptSize(c, byTok) || 0);
  const max = Math.max(1, ...ys);
  const X = (i) => L + (calls.length === 1 ? 0 : i * (W - L - R) / (calls.length - 1));
  const Y = (v) => T + (H - T - B) * (1 - v / max);
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.setAttribute("class", "pctx-chart");
  svg.setAttribute("width", W); svg.setAttribute("height", H);
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", `prompt size per model call, ${byTok ? "tokens" : "characters"}`);
  for (const f of [0, 0.5, 1]) {
    const y = Y(max * f);
    const g = document.createElementNS(ns, "line");
    g.setAttribute("x1", L); g.setAttribute("x2", W - R); g.setAttribute("y1", y); g.setAttribute("y2", y);
    g.setAttribute("class", "pctx-grid"); svg.appendChild(g);
    const t = document.createElementNS(ns, "text");
    t.setAttribute("x", L - 6); t.setAttribute("y", y + 4); t.setAttribute("text-anchor", "end");
    t.setAttribute("class", "pctx-axis"); t.textContent = kfmt(Math.round(max * f)); svg.appendChild(t);
  }
  const xl = document.createElementNS(ns, "text");
  xl.setAttribute("x", W - R); xl.setAttribute("y", H - 4); xl.setAttribute("text-anchor", "end");
  xl.setAttribute("class", "pctx-axis");
  xl.textContent = `model call → (${calls.length})  ·  y: prompt ${byTok ? "tokens" : "characters"}`;
  svg.appendChild(xl);
  // The line follows the agent's own context; a call sharing nothing with any earlier one is a
  // side context (a summariser, a fresh helper) and stands apart as its marker alone.
  const path = document.createElementNS(ns, "path");
  path.setAttribute("d", calls.map((c, i) => [c, i]).filter(([c]) => c.class !== "new_context")
    .map(([c, i], j, a) => `${j && a[j - 1][0].lane === c.lane ? "L" : "M"}${X(i).toFixed(1)},${Y(ys[i]).toFixed(1)}`).join(" "));
  path.setAttribute("class", "pctx-line"); svg.appendChild(path);
  const hover = document.createElementNS(ns, "line");
  hover.setAttribute("class", "pctx-hover"); hover.setAttribute("y1", T); hover.setAttribute("y2", H - B);
  hover.style.display = "none"; svg.appendChild(hover);
  calls.forEach((c, i) => { if (PROMPT_CLASS[c.class]) svg.appendChild(promptShape(c.class, X(i), Y(ys[i]), 6)); });
  // Compactions are what a reader came for: label each, while there are few enough to read.
  const comp = calls.map((c, i) => [c, i]).filter(([c]) => c.class === "compaction");
  if (comp.length <= 6) for (const [c, i] of comp) {
    const t = document.createElementNS(ns, "text");
    const right = X(i) > W - 160;
    t.setAttribute("x", X(i) + (right ? -9 : 9)); t.setAttribute("y", Math.max(T + 10, Y(ys[i]) - 8));
    t.setAttribute("text-anchor", right ? "end" : "start"); t.setAttribute("class", "pctx-label");
    const b = c.base_event_idx == null ? null : calls.find(x => x.lane === c.lane && x.event_idx === c.base_event_idx);
    t.textContent = `#${c.event_idx} compaction${b ? ` ${kfmt(promptSize(b, byTok))} → ${kfmt(ys[i])}` : ""}`;
    svg.appendChild(t);
  }
  // One hit band per call, wider than its mark: hover names the call, click opens it.
  const step = calls.length > 1 ? (W - L - R) / (calls.length - 1) : 10;
  const tip = document.createElement("div"); tip.className = "pctx-tip"; tip.hidden = true;
  calls.forEach((c, i) => {
    const r = document.createElementNS(ns, "rect");
    r.setAttribute("x", X(i) - step / 2); r.setAttribute("width", Math.max(step, 2));
    r.setAttribute("y", T); r.setAttribute("height", H - T - B); r.setAttribute("class", "pctx-hit");
    r.onmouseenter = () => {
      hover.setAttribute("x1", X(i)); hover.setAttribute("x2", X(i)); hover.style.display = "";
      tip.textContent = promptCallLine(c); tip.hidden = false;
    };
    r.onmouseleave = () => { hover.style.display = "none"; tip.hidden = true; };
    r.onclick = () => jump(c);
    svg.appendChild(r);
  });
  const fig = document.createElement("div"); fig.className = "pctx-fig";
  fig.append(svg, tip);
  const legend = document.createElement("div"); legend.className = "pctx-legend";
  for (const [cls, spec] of Object.entries(PROMPT_CLASS)) {
    const item = document.createElement("span"); item.title = spec.says;
    const s = document.createElementNS(ns, "svg"); s.setAttribute("viewBox", "0 0 16 16");
    s.setAttribute("width", "14"); s.setAttribute("height", "14");
    s.appendChild(promptShape(cls, 8, 8, 6));
    item.append(s, document.createTextNode(` ${spec.label} (${sum.classes[cls] || 0})`));
    legend.appendChild(item);
  }
  box.append(fig, legend);
  if (marked.length) {
    const t = document.createElement("table"); t.className = "pctx-table";
    t.innerHTML = `<thead><tr><th>call</th><th>what</th><th>base</th><th>kept</th><th>dropped</th>`
      + `<th>added</th><th>prompt tok (base → call)</th></tr></thead>`;
    const tb = document.createElement("tbody");
    const byIdx = {}; for (const c of calls) byIdx[c.lane + "\0" + c.event_idx] = c;
    for (const c of marked.slice(0, 200)) {
      const b = c.base_event_idx == null ? null : byIdx[c.lane + "\0" + c.base_event_idx];
      const tr = document.createElement("tr");
      const a = document.createElement("a"); a.href = "#"; a.textContent = "#" + c.event_idx;
      a.onclick = (e) => { e.preventDefault(); jump(c); };
      const td0 = document.createElement("td"); td0.appendChild(a); tr.appendChild(td0);
      const spec = PROMPT_CLASS[c.class];
      const tdc = document.createElement("td");
      const s = document.createElementNS(ns, "svg"); s.setAttribute("viewBox", "0 0 16 16");
      s.setAttribute("width", "12"); s.setAttribute("height", "12"); s.appendChild(promptShape(c.class, 8, 8, 6));
      tdc.append(s, document.createTextNode(" " + spec.label)); tdc.title = spec.says; tr.appendChild(tdc);
      for (const v of [c.base_event_idx == null ? "—" : "#" + c.base_event_idx, c.shared, c.dropped, c.added,
                       `${kfmt(b ? b.prompt_tokens : null)} → ${kfmt(c.prompt_tokens)}`]) {
        const td = document.createElement("td"); td.textContent = v == null ? "—" : v; tr.appendChild(td);
      }
      tb.appendChild(tr);
    }
    t.appendChild(tb);
    // Many marked calls (a per-call reminder rewrites every one) fold behind their count.
    const fold = document.createElement("details"); fold.className = "pctx-rows";
    fold.open = marked.length <= 12;
    const fs = document.createElement("summary");
    fs.textContent = `the ${marked.length} call${marked.length > 1 ? "s" : ""} that ${usage ? "shrank the context" : "rewrote their history"}`
      + (marked.length > 200 ? " (first 200)" : "");
    fold.append(fs, t); box.appendChild(fold);
  }
  return box;
}
// One call's prompt, inside its event: the summary line closed; opened, the parameters and tools,
// the kept run folded to one line (expanded in pages), what the base had that this call dropped,
// and every message after the kept run with a preview and its whole text on demand.
const PROMPT_PAGE = 50;
function promptPanel(c, base, gotoEvent, unfold = false) {
  const d = document.createElement("details"); d.className = "prompt";
  d.dataset.cls = c.class;
  const s = document.createElement("summary");
  const spec = PROMPT_CLASS[c.class];
  const tag = document.createElement("span"); tag.className = "ptag" + (spec ? " marked" : "");
  if (spec) { tag.style.setProperty("--c", spec.color); tag.textContent = spec.label; tag.title = spec.says; }
  else { tag.textContent = PROMPT_QUIET[c.class] || c.class; }
  s.append(document.createTextNode(`prompt · ${c.n_messages} messages · ${kfmt(c.chars)} chars`
    + (c.n_tools ? ` · ${c.n_tools} tools` : "") + ` · ${c.dialect} `), tag);
  if (c.base_event_idx != null) {
    const a = document.createElement("a"); a.href = "#"; a.className = "rlink";
    a.textContent = `base #${c.base_event_idx}`;
    a.onclick = (e) => { e.preventDefault(); e.stopPropagation(); gotoEvent(c.lane, c.base_event_idx); };
    s.append(document.createTextNode(" · "), a,
      document.createTextNode(`: kept ${c.shared}, dropped ${c.dropped}, +${c.added}`));
  }
  d.appendChild(s);
  const body = document.createElement("div"); body.className = "pbody";
  d.appendChild(body);
  const url = `${base}/prompts/${c.event_idx}?lane=${encodeURIComponent(c.lane || "")}`;
  const msgsUrl = (hs) => `${base}/prompt_messages?` + hs.map(h => "hash=" + encodeURIComponent(h)).join("&");
  let loading = null;
  const load = () => loading ??= (body.replaceChildren(slotSpinner()),
    traceApi(url, { timeout: 120000 }).then((p) => { body.replaceChildren(); renderPrompt(p); },
      (e) => { body.replaceChildren(); body.textContent = "prompt failed to load: " + (e && e.message || e); }));
  d.ontoggle = () => { if (d.open) load(); };
  // every message whole, folded runs included: what a `?text=` search reads on a miss
  d.dataset.loadable = "";
  d.loadWhole = async () => {
    await load();
    await Promise.all([...body.querySelectorAll("[data-loadable]")].map(n => n.loadWhole()));
  };
  function msgRow(m, text) {
    const row = document.createElement("div"); row.className = "pmsg st-" + m.state;
    const h = document.createElement("div"); h.className = "pmsg-h";
    h.textContent = `${m.pos} · ${m.role || m.item_type || "item"}${m.item_type && m.role ? " · " + m.item_type : ""}`
      + ` · ${kfmt(m.chars)} chars` + (m.state === "kept" ? "" : ` · ${m.state}`);
    const t = document.createElement("div"); t.className = "pmsg-t";
    setText(t, text ?? m.preview ?? "");
    row.append(h, t);
    const full = m.preview_chars == null ? true : m.preview_chars > (m.preview || "").length;
    if (full || text == null) {
      const b = document.createElement("button"); b.className = "pbtn"; b.textContent = "whole message";
      let pending = null;
      const whole = () => pending ??= (b.disabled = true,
        traceApi(msgsUrl([m.msg_hash]), { timeout: 120000 }).then((r) => {
          const x = (r.messages || [])[0];
          if (!x) { b.textContent = "not found"; return; }
          let body = x.text || "";
          if (body.length > PART_CAP) body = body.slice(0, PART_CAP) + `… [${body.length} chars — export for full text]`;
          setText(t, body); t.classList.add("whole");
          const j = document.createElement("details"); j.className = "pjson";
          const js = document.createElement("summary"); js.textContent = "as sent (JSON)";
          const pre = document.createElement("pre");
          let raw = x.msg_json || "";
          if (raw.length > PART_CAP) raw = raw.slice(0, PART_CAP) + `… [${x.chars} chars${x.truncated ? ", capped" : ""}]`;
          pre.textContent = raw; j.append(js, pre); row.appendChild(j); b.remove();
        }, () => { pending = null; b.disabled = false; b.textContent = "failed — retry"; }));
      b.onclick = whole;
      row.dataset.loadable = ""; row.loadWhole = whole;
      h.appendChild(b);
    }
    return row;
  }
  // A folded run of messages: `n` of them, previews fetched a page at a time on expand.
  function foldRun(label, items, hashOf, open = false) {
    const f = document.createElement("details"); f.className = "pfold";
    const fs = document.createElement("summary"); fs.textContent = label; f.appendChild(fs);
    if (open) queueMicrotask(() => { f.open = true; });
    const list = document.createElement("div"); f.appendChild(list);
    let shown = 0;
    const more = document.createElement("button"); more.className = "pbtn";
    const page = () => {
      const slice = items.slice(shown, shown + PROMPT_PAGE); shown += slice.length;
      more.disabled = true;
      return traceApi(msgsUrl(slice.map(hashOf)), { timeout: 120000 }).then((r) => {
        const byH = {}; for (const x of r.messages || []) byH[x.msg_hash] = x;
        for (const it of slice) {
          const x = byH[hashOf(it)] || {};
          const m = typeof it === "string" ? { pos: "−", msg_hash: it, role: x.role, item_type: x.item_type,
                                               chars: x.chars, state: "dropped" } : it;
          const text = (x.text || "").slice(0, 240);
          list.appendChild(msgRow({ ...m, preview: text, preview_chars: (x.text || "").length }, null));
        }
        more.disabled = false;
        more.textContent = `${items.length - shown} more`; more.hidden = shown >= items.length;
        list.appendChild(more);
      });
    };
    more.onclick = page;
    let opened = false;
    f.ontoggle = () => { if (f.open && !opened) { opened = true; page(); } };
    f.dataset.loadable = "";
    f.loadWhole = async () => {
      opened = true;
      while (shown < items.length) await page();
      await Promise.all([...list.querySelectorAll("[data-loadable]")].map(n => n.loadWhole()));
    };
    return f;
  }
  function renderPrompt(p) {
    const meta = document.createElement("div"); meta.className = "pmeta";
    if (p.params) {
      const j = document.createElement("details"); j.className = "pjson";
      const js = document.createElement("summary"); js.textContent = "parameters";
      const pre = document.createElement("pre"); pre.textContent = JSON.stringify(p.params, null, 2);
      j.append(js, pre); meta.appendChild(j);
    }
    if (p.tools && p.tools.length) {
      const tl = document.createElement("div"); tl.className = "ptools";
      tl.textContent = `tools (${p.tools.length}): ` + p.tools.map(t => t.name || t.sha256.slice(0, 12)).join(", ");
      meta.appendChild(tl);
    }
    body.appendChild(meta);
    const kept = p.messages.filter(m => m.state === "kept");
    if (kept.length) body.appendChild(foldRun(
      `${kept.length} message${kept.length > 1 ? "s" : ""} kept from base #${p.call.base_event_idx} (0–${kept.length - 1})`,
      kept, (m) => m.msg_hash, unfold));
    for (const m of p.messages) if (m.state !== "kept") body.appendChild(msgRow(m, null));
    if (p.n_dropped) body.appendChild(foldRun(
      `${p.n_dropped} message${p.n_dropped > 1 ? "s" : ""} of base #${p.call.base_event_idx} not re-sent`
        + (p.dropped.length < p.n_dropped ? ` (first ${p.dropped.length} listed)` : ""),
      p.dropped, (h) => h));
  }
  return d;
}
