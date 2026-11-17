// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The trace's graph mode: each lane a column, read left to right, and each event placed top to
// bottom by the order the data states -- a lane's own order, the links a reader stated between
// events, and wall clocks where both ends have one. Height means "after" only along a lane or a
// drawn link; two events level with each other in different lanes are not claimed simultaneous.
"use strict";

// A link carries `first`: which end happened first, as the reader that stated it declares
// ("frm", "to"), or nothing, in which case it is drawn but orders nothing.
// How strongly a link pulls its two ends together when heights are chosen; a lane may stretch.
const GRAPH_WEIGHT = { lane: 1, stated: 8, clock: 0.5 };
const GRAPH_NODE_H = 22, GRAPH_GAP = 4, GRAPH_LINK_GAP = 4, GRAPH_SWEEPS = 80;
const GRAPH_CLOCK_CAP = 4000;   // past this many events, clocks add no constraints (cycle checks)

// nodes: [{key, lane, t}] lane-contiguous, each lane in its own order; t is ms or null.
// links: [{frm, to, space}] by node key. Returns {y, lanes, links} where each link carries
// `dropped` when keeping it would contradict an order already kept.
function graphLayout(nodes, links) {
  const n = nodes.length, at = new Map(nodes.map((v, i) => [v.key, i]));
  const out = Array.from({ length: n }, () => []), inn = Array.from({ length: n }, () => []);
  const add = (a, b, gap, w) => { out[a].push([b, gap, w]); inn[b].push([a, gap, w]); };
  const reaches = (src, dst) => {   // is dst already after src?
    const seen = new Uint8Array(n), stack = [src];
    while (stack.length) {
      const v = stack.pop(); if (v === dst) return true;
      for (const [w] of out[v]) if (!seen[w]) { seen[w] = 1; stack.push(w); }
    }
    return false;
  };
  for (let i = 0; i + 1 < n; i++)
    if (nodes[i].lane === nodes[i + 1].lane) add(i, i + 1, GRAPH_NODE_H + GRAPH_GAP, GRAPH_WEIGHT.lane);
  const kept = links.map((l) => {
    const a = at.get(l.frm), b = at.get(l.to), first = l.first;
    if (a == null || b == null || a === b) return { ...l, drawn: false };
    const [u, v] = first === "to" ? [b, a] : [a, b];
    // A link to the next event of the same lane is what the lane already says; it adds nothing.
    if (nodes[u].lane === nodes[v].lane && v === u + 1) return { ...l, drawn: false };
    if (!first) return { ...l, drawn: true, u, v };
    if (reaches(v, u)) return { ...l, drawn: true, u, v, dropped: true };
    add(u, v, GRAPH_LINK_GAP, GRAPH_WEIGHT.stated);
    return { ...l, drawn: true, u, v };
  });
  if (n <= GRAPH_CLOCK_CAP) {
    const timed = nodes.map((v, i) => [v.t, i]).filter(([t]) => t != null).sort((p, q) => p[0] - q[0]);
    for (let k = 0; k + 1 < timed.length; k++) {
      const [ta, a] = timed[k], [tb, b] = timed[k + 1];
      if (ta < tb && nodes[a].lane !== nodes[b].lane && !reaches(b, a)) add(a, b, 0, GRAPH_WEIGHT.clock);
    }
  }
  const order = graphTopo(out, inn);
  const y = new Float64Array(n);
  for (const v of order) for (const [w, gap] of out[v]) y[w] = Math.max(y[w], y[v] + gap);
  // Each sweep moves every event to the weighted median of where its links would put it, within
  // what its neighbours allow, until nothing moves: a call settles beside what it produced, and
  // one that produced nothing settles between its neighbours instead of blocking the next.
  for (let s = 0; s < GRAPH_SWEEPS; s++) {
    let moved = 0;
    for (const v of s % 2 ? order : [...order].reverse()) {
      let lo = 0, hi = Infinity; const want = [];
      for (const [p, gap, w] of inn[v]) { lo = Math.max(lo, y[p] + gap); want.push([y[p] + gap, w]); }
      for (const [q, gap, w] of out[v]) { hi = Math.min(hi, y[q] - gap); want.push([y[q] - gap, w]); }
      if (!want.length) continue;
      const next = Math.min(hi, Math.max(lo, graphMedian(want)));
      moved = Math.max(moved, Math.abs(next - y[v])); y[v] = next;
    }
    if (moved < 0.5) break;
  }
  return { y, links: kept };
}

function graphTopo(out, inn) {
  const deg = inn.map((l) => l.length), queue = [], order = [];
  deg.forEach((d, i) => { if (!d) queue.push(i); });
  while (queue.length) {
    const v = queue.shift(); order.push(v);
    for (const [w] of out[v]) if (!--deg[w]) queue.push(w);
  }
  return order;
}

// The weighted median; where the weight splits exactly between two values, their midpoint.
function graphMedian(pairs) {
  pairs.sort((a, b) => a[0] - b[0]);
  const half = pairs.reduce((s, [, w]) => s + w, 0) / 2;
  let acc = 0;
  for (let i = 0; i < pairs.length; i++) {
    acc += pairs[i][1];
    if (acc > half) return pairs[i][0];
    if (acc === half) return i + 1 < pairs.length ? (pairs[i][0] + pairs[i + 1][0]) / 2 : pairs[i][0];
  }
  return pairs[pairs.length - 1][0];
}

// Lane columns: the first lane stays leftmost; the rest are ordered so linked lanes sit close.
function graphColumns(laneKeys, nodes, links) {
  if (laneKeys.length <= 2 || laneKeys.length > 7) return laneKeys;
  const laneOf = new Map(nodes.map((v) => [v.key, v.lane]));
  const pairs = links.map((l) => [laneOf.get(l.frm), laneOf.get(l.to)]).filter(([a, b]) => a != null && b != null && a !== b);
  let best = laneKeys, bestCost = Infinity;
  const perm = (rest, acc) => {
    if (!rest.length) {
      const col = new Map(acc.map((k, i) => [k, i]));
      const cost = pairs.reduce((s, [a, b]) => s + Math.abs(col.get(a) - col.get(b)), 0);
      if (cost < bestCost) { bestCost = cost; best = acc; }
      return;
    }
    rest.forEach((k, i) => perm([...rest.slice(0, i), ...rest.slice(i + 1)], [...acc, k]));
  };
  perm(laneKeys.slice(1), [laneKeys[0]]);
  return best;
}

// Draws the graph into `host`. `lanes`: [{key, name, color}] in column order; `nodes` as for
// graphLayout plus {kind, label, title}; `pick(key)` opens an event in the linear view.
function graphRender(host, lanes, nodes, links, pick, width) {
  const PAD = 16, COL_GAP = 72;
  const COL_W = Math.max(190, Math.min(300, Math.floor((width - 2 * PAD - (lanes.length - 1) * COL_GAP) / lanes.length)));
  // A lane name is shown whole: on two lines where one does not hold it.
  const HEAD = lanes.some((L) => L.name.length * 7.3 > COL_W) ? 48 : 30;
  const { y, links: placed } = graphLayout(nodes, links);
  const col = new Map(lanes.map((L, i) => [L.key, i]));
  const x = (i) => PAD + col.get(nodes[i].lane) * (COL_W + COL_GAP);
  const H = HEAD + Math.max(0, ...y) + GRAPH_NODE_H + PAD, W = PAD * 2 + lanes.length * COL_W + (lanes.length - 1) * COL_GAP;
  const NS = "http://www.w3.org/2000/svg";
  const el = (tag, attrs, parent) => {
    const e = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    if (parent) parent.appendChild(e);
    return e;
  };
  host.replaceChildren();
  const head = document.createElement("div"); head.className = "ghead";
  head.style.width = W + "px"; head.style.height = HEAD + "px";
  for (const L of lanes) {
    const h = document.createElement("div"); h.className = "glane";
    h.style.left = PAD + col.get(L.key) * (COL_W + COL_GAP) + "px"; h.style.width = COL_W + "px";
    h.style.setProperty("--c", L.color); h.textContent = L.name; h.title = L.key || "(main lane)";
    head.appendChild(h);
  }
  const svg = el("svg", { width: W, height: H, class: "gsvg", role: "img",
                          "aria-label": `${nodes.length} events in ${lanes.length} lanes` });
  svg.style.marginTop = -HEAD + "px";
  el("path", { d: "M0,0 L8,4 L0,8 z" },
     el("marker", { id: "garrow", viewBox: "0 0 8 8", refX: 8, refY: 4, markerWidth: 7, markerHeight: 7,
                  markerUnits: "userSpaceOnUse", orient: "auto" },
        el("defs", {}, svg)));
  for (const L of lanes) {
    const cx = PAD + col.get(L.key) * (COL_W + COL_GAP);
    el("rect", { x: cx - 4, y: 0, width: COL_W + 8, height: H, class: "gcol", style: `--c:${L.color}` }, svg);
  }
  const edgeG = el("g", { class: "gedges" }, svg), nodeG = el("g", { class: "gnodes" }, svg);
  const mid = (i) => HEAD + y[i] + GRAPH_NODE_H / 2;
  const touching = Array.from({ length: nodes.length }, () => []);
  for (const l of placed) {
    if (!l.drawn) continue;
    const { u, v } = l;
    let d;
    if (nodes[u].lane === nodes[v].lane) {
      const bx = x(u) - 6, bend = Math.min(60, 10 + Math.abs(mid(v) - mid(u)) / 8);
      d = `M${bx},${mid(u)} C${bx - bend},${mid(u)} ${bx - bend},${mid(v)} ${bx},${mid(v)}`;
    } else {
      const right = x(u) < x(v);
      const sx = right ? x(u) + COL_W : x(u), tx = right ? x(v) : x(v) + COL_W, c = (tx - sx) / 2;
      d = `M${sx},${mid(u)} C${sx + c},${mid(u)} ${tx - c},${mid(v)} ${tx},${mid(v)}`;
    }
    const p = el("path", { d, class: `gedge s-${l.space}${l.inferred ? " inferred" : ""}${l.dropped ? " dropped" : ""}`,
                           ...(l.first ? { "marker-end": "url(#garrow)" } : {}) }, edgeG);
    el("title", {}, p).textContent = l.space + (l.inferred ? " — matched by text, not by an id" : "") + (l.dropped
      ? " — not used for placement: it contradicts an order already kept" : "");
    touching[u].push(p); touching[v].push(p);
  }
  const boxes = nodes.map((nd, i) => {
    const g = el("g", { class: `gnode k-${String(nd.kind || "other").replace(/[^a-z_]/gi, "")}`,
                        transform: `translate(${x(i)},${HEAD + y[i]})`, tabindex: 0 }, nodeG);
    el("rect", { width: COL_W, height: GRAPH_NODE_H, rx: 4, class: "gbox" }, g);
    el("rect", { width: 3, height: GRAPH_NODE_H, class: "gstripe" }, g);
    el("text", { x: 9, y: GRAPH_NODE_H / 2 + 4 }, g).textContent = graphClip(nd.label, (COL_W - 14) / 6.7);
    el("title", {}, g).textContent = nd.title;
    g.onclick = () => pick(nd.key);
    g.onkeydown = (e) => { if (e.key === "Enter") pick(nd.key); };
    return g;
  });
  // Hovering an event fades every event it is not ordered against, so a level neighbour in
  // another lane is visibly not "at the same time".
  const succ = Array.from({ length: nodes.length }, () => []), pred = Array.from({ length: nodes.length }, () => []);
  for (let i = 0; i + 1 < nodes.length; i++) if (nodes[i].lane === nodes[i + 1].lane) { succ[i].push(i + 1); pred[i + 1].push(i); }
  for (const l of placed) if (l.drawn && !l.dropped && l.first) { succ[l.u].push(l.v); pred[l.v].push(l.u); }
  const reach = (start, adj, mark) => { const st = [start];
    while (st.length) { const v = st.pop(); for (const w of adj[v]) if (!mark[w]) { mark[w] = 1; st.push(w); } } };
  boxes.forEach((g, i) => {
    g.onmouseenter = () => {
      const mark = new Uint8Array(nodes.length); mark[i] = 1;
      reach(i, succ, mark); reach(i, pred, mark);
      svg.classList.add("hovering");
      boxes.forEach((b, j) => b.classList.toggle("unordered", !mark[j]));
      g.classList.add("hot"); for (const p of touching[i]) p.classList.add("hot");
    };
    g.onmouseleave = () => {
      svg.classList.remove("hovering");
      for (const b of boxes) b.classList.remove("unordered");
      g.classList.remove("hot"); for (const p of touching[i]) p.classList.remove("hot");
    };
  });
  const scroller = document.createElement("div"); scroller.className = "gscroll";
  scroller.append(head, svg);
  host.appendChild(scroller);
  const at = new Map(nodes.map((v, i) => [v.key, i]));
  // Marks one event as the one open beside the graph, and scrolls it into view if it is not.
  const select = (key) => {
    boxes.forEach((b) => b.classList.remove("picked"));
    const i = at.get(key); if (i == null) return;
    boxes[i].classList.add("picked");
    const top = HEAD + y[i], left = x(i);
    if (top < scroller.scrollTop + HEAD || top + GRAPH_NODE_H > scroller.scrollTop + scroller.clientHeight)
      scroller.scrollTop = top - scroller.clientHeight / 3;
    if (left < scroller.scrollLeft || left + COL_W > scroller.scrollLeft + scroller.clientWidth)
      scroller.scrollLeft = left - PAD;
  };
  const relabel = (key, label) => {
    const i = at.get(key); if (i == null) return;
    boxes[i].querySelector("text").textContent = graphClip(label, (COL_W - 14) / 6.7);
  };
  return { placed, select, relabel, height: H, width: W };
}

function graphClip(s, max) {
  s = String(s ?? "").replace(/\s+/g, " ");
  return s.length > max ? s.slice(0, Math.max(1, Math.floor(max) - 1)) + "…" : s;
}
