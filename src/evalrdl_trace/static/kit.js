// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The page's own helpers that the trace view uses too: building and escaping markup, the API
// call, the prettify toggle, loading spinners, and the address a view's settings are kept in.
// A classic script: its names are globals of the page that loads it, before trace.js.

const $ = (h) => { const d = document.createElement("div"); d.innerHTML = h; return d; };
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const fmt = (v) => v == null ? "" : (typeof v === "number" && !Number.isInteger(v) ? v.toFixed(2) : v);
// A name's colour: the same name is the same hue on every page, picked by hash from a fixed set
// spread over the hue wheel. The chip is a soft tint of it with the hue darkened for the text,
// as the tag and annotation chips are, so no name shouts.
const CHIP_COLORS = ["#2563eb", "#0891b2", "#059669", "#65a30d", "#ca8a04", "#ea580c", "#dc2626",
  "#db2777", "#9333ea", "#4f46e5", "#0d9488", "#92400e", "#be185d", "#7c3aed", "#15803d", "#0369a1",
  "#737373", "#c2410c", "#334155", "#a21caf", "#84cc16", "#0ea5e9", "#e11d48", "#8b5cf6"];
function nameColor(name) {
  let h = 2166136261;
  for (const c of String(name)) h = Math.imul(h ^ c.codePointAt(0), 16777619) >>> 0;
  return CHIP_COLORS[h % CHIP_COLORS.length];
}
// A reader stack ("a / b / c") as one tinted chip per layer. `label` may mark up a layer's text.
function stackChips(stack, label, link) {
  const names = String(stack || "").split(" / ").map((s) => s.trim()).filter(Boolean);
  if (!names.length) return "";
  return `<span class="rchips">` + names.map((n) =>
    `<span class="rchip${link ? " flink" : ""}" style="--c:${nameColor(n)}"${link
      ? ` data-filter="reader=${encodeURIComponent(n)}" title="runs read by ${esc(n)}"` : ` title="reader: ${esc(n)}"`}>${label ? label(n) : esc(n)}</span>`
  ).join("") + `</span>`;
}
// api() with a hard timeout — a stalled fetch (network blip, request caught mid-deploy) must fail
// with a visible error instead of leaving the trace spinner hanging forever.
// `signal` cancels the call for a caller that has since asked for something else; that rejects
// with the AbortError itself, not the timeout's message.
const api = async (path, { timeout = 45000, signal } = {}) => {
  const ctl = new AbortController();
  const t = setTimeout(() => ctl.abort(), timeout);
  if (signal) signal.aborted ? ctl.abort() : signal.addEventListener("abort", () => ctl.abort(), { once: true });
  try {
    const r = await fetch(path, { signal: ctl.signal });
    if (!r.ok) {
      const err = new Error(path + " -> " + r.status);
      try { err.detail = (await r.json()).error; } catch (_) { /* not a JSON refusal */ }
      throw err;
    }
    return await r.json();
  } catch (e) {
    if (e.name === "AbortError" && !(signal && signal.aborted)) throw new Error(path + " -> timeout after " + (timeout / 1000) + "s");
    throw e;
  } finally { clearTimeout(t); }
};

// ── prettify (gdpval-style): decode literal escape sequences so double-escaped tool output
//    (real newlines shown as \n, plus \uXXXX, \" …) renders as readable text. ──
const PRETTY = { on: false };
function unescapeText(s) {
  if (!PRETTY.on || s == null) return s;
  return String(s)
    .replace(/\\u([0-9a-fA-F]{4})/g, (_, h) => String.fromCharCode(parseInt(h, 16)))
    .replace(/\\r\\n/g, "\n").replace(/\\n/g, "\n").replace(/\\r/g, "\n")
    .replace(/\\t/g, "\t").replace(/\\"/g, '"').replace(/\\\\/g, "\\");
}
// every rendered text node stashes its raw text so the toggle re-applies without a refetch
function setText(node, raw) { node.dataset.raw = raw ?? ""; node.textContent = unescapeText(raw ?? ""); return node; }
// prettify rewrites every [data-raw] node; flag ONLY the cells whose text actually changes so the
// per-part "prettified" chip is signal (not noise) — pure DOM class toggle, no perf cost.
function applyPretty() {
  document.querySelectorAll("[data-raw]").forEach(n => {
    const raw = n.dataset.raw, out = unescapeText(raw);
    // A JSON-coloured cell is re-coloured rather than flattened: this rewrite runs on every
    // prettify toggle, and plain `textContent =` would silently drop the markup.
    if (n.classList.contains("jsonbody") && prettyJSON(n, out)) { /* recoloured */ }
    else n.textContent = out;
    const w = n.closest ? n.closest(".partwrap") : null;
    if (w) w.classList.toggle("pretty-changed", PRETTY.on && out !== raw);
  });
  const pc = document.getElementById("pretty-count");
  if (pc) { const n = document.querySelectorAll(".partwrap.pretty-changed").length;
    pc.textContent = PRETTY.on && n ? ` · ${n} cells decoded` : ""; }
}
// rendered text of a hydrated part/event node group, honoring the prettify toggle
function nodeRaw(el) {
  if (!el) return "";
  const raws = []; el.querySelectorAll("[data-raw]").forEach(n => raws.push(n.dataset.raw));
  const joined = raws.join("\n\n");
  return PRETTY.on ? unescapeText(joined) : joined;
}
// single source of truth for the prettify toggle so the top bar and the sticky sidebar copy stay in sync
function setPretty(on) {
  PRETTY.on = on;
  document.body.classList.toggle("pretty-on", on);   // gates the per-part "prettified" chip
  document.querySelectorAll(".js-pretty").forEach(c => { c.checked = on; });
  applyPretty();
}
function loadingBox(label) {
  const d = document.createElement("div"); d.className = "loading";
  const s = document.createElement("span"); s.className = "spinner";
  d.append(s, document.createTextNode(" " + (label || "loading…"))); return d;
}
function slotSpinner() {   // small inline spinner for a parts slot while its S3 texts read is in flight
  const d = document.createElement("div"); d.className = "loading"; d.style.padding = "6px 2px";
  const s = document.createElement("span"); s.className = "spinner";
  s.style.cssText = "width:14px;height:14px;border-width:2px";
  d.append(s, document.createTextNode(" loading parts…")); return d;
}

// ── what a view's settings are, declared once ──────────────────────────────────────────────
//
// A view's settings are a LIST. Each field says how it is spelled in a query string (`kind`),
// where it lives on the view's state object (`at`, dotted), and what it is for: `narrow` decides
// which rows there are, `order` decides which come first, and a field that is neither (the page,
// the page size, a display mode) is the reader's alone.
//
// The list is the only place any of that is written down. The state object is BUILT from it, the
// address is written from it, and the query the server is asked is written from it — so a filter
// cannot reach the server without also reaching the address, and adding a setting is adding a
// line here. That is what the round-trip test enumerates rather than repeats.
const URL_KINDS = {
  text: { empty: () => "",    read: (p, n) => p.get(n) || "",
          write: (p, n, v) => { if (v) p.set(n, String(v)); } },
  // Repeated, not comma-joined: a model name may hold anything, and `URLSearchParams` already
  // escapes each value on its own. `a=x&a=y` needs no agreement about a separator.
  set:  { empty: () => [],    read: (p, n) => p.getAll(n).filter(Boolean),
          write: (p, n, v) => { for (const x of v || []) p.append(n, x); } },
  flag: { empty: () => false, read: (p, n) => p.get(n) === "1",
          write: (p, n, v) => { if (v) p.set(n, "1"); } },
  num:  { empty: () => 0,     read: (p, n) => Number(p.get(n)) || 0,
          write: (p, n, v) => { if (v) p.set(n, String(v)); } },
  // A sort direction is ±1 in the state and a word in the address, because `dir=asc` is what the
  // server has always been asked for and what a reader can read.
  dir:  { empty: () => 0,     read: (p, n) => (p.get(n) === "asc" ? 1 : p.get(n) === "desc" ? -1 : 0),
          write: (p, n, v) => { if (v) p.set(n, v > 0 ? "asc" : "desc"); } },
};
const urlField = (name, kind, at, opts) => ({ name, kind, at: at || name, ...(opts || {}) });
const atGet = (o, at) => at.split(".").reduce((x, k) => (x == null ? x : x[k]), o);
const atSet = (o, at, v) => { const ks = at.split("."), last = ks.pop();
  let x = o; for (const k of ks) x = (x[k] ??= {}); x[last] = v; };
const sameSetting = (a, b) => JSON.stringify(a ?? null) === JSON.stringify(b ?? null);

// Read a query string onto a state object. An absent field takes the view's default, or its
// kind's empty value — so this is also how a view's state is CREATED (`blankState`), and a state
// object can hold no setting the list does not declare.
function readFields(fields, params, state) {
  const st = state || {};
  for (const f of fields) {
    atSet(st, f.at, params.has(f.name) ? URL_KINDS[f.kind].read(params, f.name)
                                       : ("def" in f ? f.def : URL_KINDS[f.kind].empty()));
  }
  return st;
}
const blankState = (fields) => readFields(fields, new URLSearchParams(), {});
// The FILLED fields, spelled. A setting left at its default is not a setting the reader chose,
// so it is not in the address and not in the request — which is what keeps an untouched listing
// at a bare `#/` and keeps the server's own defaults authoritative.
function writeFields(fields, state, only) {
  const p = new URLSearchParams();
  for (const f of fields) {
    if (only && !only(f)) continue;
    const v = atGet(state, f.at);
    if (sameSetting(v, "def" in f ? f.def : URL_KINDS[f.kind].empty())) continue;
    URL_KINDS[f.kind].write(p, f.name, v);
  }
  return p.toString();
}
const NARROWING = (f) => !!f.narrow;                 // which rows there are
const NARROW_OR_ORDER = (f) => !!(f.narrow || f.order);   // …and which come first
// `#/` and an empty hash are the same address; writing one over the other would be a change the
// reader did not make, and the guided tour compares hashes for equality.
const hashPath = (h) => String(h == null ? location.hash : h).split("?")[0] || "#/";
// the address is rewritten IN PLACE. An assignment would re-enter the router on the view
// that just changed a filter, and would push one history entry per keystroke.
// `extra`: already-spelled `name=value` pairs kept verbatim after the fields.
function syncAddress(path, fields, state, extra) {
  const q = [writeFields(fields, state), ...(extra || [])].filter(Boolean).join("&");
  const want = "#" + path + (q ? "?" + q : "");
  const now = location.hash === "" || location.hash === "#" ? "#/" : location.hash;
  if (now !== want) history.replaceState(null, "", want);
}
