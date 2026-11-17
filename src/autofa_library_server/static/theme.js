// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The colour scheme: the failure-card report's own preference key, mirrored into a cookie so the
// server draws the next page, and the report it embeds, in it from the first byte.
(() => {
  const KEY = "autofa-fc-theme-preference-v2";
  const root = document.documentElement;
  const set = (scheme) => {
    root.dataset.theme = scheme;
    try { localStorage.setItem(KEY, scheme); } catch (e) {}
    document.cookie = `autofa_theme=${scheme}; path=/; max-age=31536000; SameSite=Lax`;
    document.querySelectorAll("[data-theme-toggle]").forEach((b) => {
      b.textContent = scheme === "dark" ? "Light" : "Dark";
    });
  };
  let stored = null;
  try { stored = localStorage.getItem(KEY); } catch (e) {}
  if ((stored === "dark" || stored === "light") && stored !== root.dataset.theme) set(stored);
  document.addEventListener("click", (event) => {
    if (!event.target.closest("[data-theme-toggle]")) return;
    set(root.dataset.theme === "dark" ? "light" : "dark");
    // A report reads its scheme from its address.
    document.querySelectorAll("iframe.report-frame").forEach((frame) => {
      const url = new URL(frame.src, location.href);
      url.searchParams.set("theme", root.dataset.theme);
      frame.src = url.pathname + url.search;
    });
  });
})();
