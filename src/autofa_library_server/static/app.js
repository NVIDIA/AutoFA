// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

(() => {
  "use strict";

  let cards = Array.from(document.querySelectorAll(".run-card"));
  const search = document.querySelector("#search-filter");
  const model = document.querySelector("#model-filter");
  const benchmark = document.querySelector("#benchmark-filter");
  const owner = document.querySelector("#owner-filter");
  const harness = document.querySelector("#harness-filter");
  const dateFrom = document.querySelector("#date-from-filter");
  const dateTo = document.querySelector("#date-to-filter");
  const count = document.querySelector("#visible-count");
  const empty = document.querySelector("#empty-state");

  function applyFilters() {
    if (!cards.length && !count) return;
    const query = (search?.value || "").trim().toLowerCase();
    const selectedModel = model?.value || "";
    const selectedBenchmark = benchmark?.value || "";
    const selectedOwner = owner?.value || "";
    const selectedHarness = harness?.value || "";
    const selectedDateFrom = dateFrom?.value || "";
    const selectedDateTo = dateTo?.value || "";
    let visible = 0;
    cards.forEach((card) => {
      const matches =
        (!query || card.dataset.search.toLowerCase().includes(query)) &&
        (!selectedModel || card.dataset.model === selectedModel) &&
        (!selectedBenchmark ||
          card.dataset.benchmarks.split("|").includes(selectedBenchmark)) &&
        (!selectedOwner || card.dataset.owner === selectedOwner) &&
        (!selectedHarness || card.dataset.harness === selectedHarness) &&
        (!selectedDateFrom || card.dataset.date >= selectedDateFrom) &&
        (!selectedDateTo || card.dataset.date <= selectedDateTo);
      card.hidden = !matches;
      if (matches) visible += 1;
    });
    if (count) count.textContent = String(visible);
    if (empty) empty.hidden = visible !== 0;
  }

  [search, model, benchmark, owner, harness, dateFrom, dateTo].forEach((control) => {
    control?.addEventListener("input", applyFilters);
    control?.addEventListener("change", applyFilters);
  });

  document.querySelectorAll("select[data-auto-submit]").forEach((select) => {
    select.addEventListener("change", () => {
      select.form?.requestSubmit();
    });
  });

  document.querySelectorAll("form[data-confirm]").forEach((form) => {
    form.addEventListener("submit", (event) => {
      if (!window.confirm(form.dataset.confirm)) event.preventDefault();
    });
  });

  document.querySelectorAll("[data-copy-value]").forEach((button) => {
    button.addEventListener("click", async () => {
      const value = button.dataset.copyValue || "";
      if (!value) return;
      const previous = button.textContent;
      try {
        await navigator.clipboard.writeText(value);
        button.textContent = "Link copied";
      } catch {
        window.prompt("Copy this public link:", value);
      } finally {
        window.setTimeout(() => {
          button.textContent = previous;
        }, 1600);
      }
    });
  });

  const failureFrames = Array.from(
    document.querySelectorAll("iframe[data-failure-bridge]")
  );
  failureFrames.forEach((frame) => {
    frame.addEventListener("load", () => {
      const functionName = frame.dataset.selectedFailure || "";
      if (!functionName) return;
      frame.contentWindow?.postMessage({
        namespace: "autofa.failure-cards",
        type: "failure-focus",
        function_name: functionName,
      }, "*");
    });
  });

  window.addEventListener("message", async (event) => {
    const frame = failureFrames.find(
      (candidate) => candidate.contentWindow === event.source
    );
    const message = event.data;
    if (
      frame &&
      message &&
      typeof message === "object" &&
      message.namespace === "autofa.source-data" &&
      message.type === "open" &&
      typeof message.url === "string" &&
      Object.keys(message).sort().join("|") === "namespace|type|url"
    ) {
      try {
        // The report's origin is opaque under sandboxing. Resolve only on the
        // trusted host, so saved relative trace links survive a server port change.
        const target = new URL(message.url, window.location.origin);
        const localTrace = frame.dataset.localTraces === "true" &&
          target.origin === window.location.origin &&
          /^\/traces\/[0-9a-f]{64}\/$/.test(target.pathname) &&
          /^#\/rollout\/[^/?#]+\/[^/?#]+\/[0-9]+(?:\/e[0-9]+(?:-p[0-9]+)?)?(?:\?lane=[^&#]*(?:&text=[^&#]+)*)?$/.test(target.hash);
        let cloudTrace = false;
        if (frame.dataset.dataApiBase) {
          const allowed = new URL(frame.dataset.dataApiBase);
          cloudTrace = target.origin === allowed.origin &&
            target.pathname.replace(/\/+$/, "") === allowed.pathname.replace(/\/+$/, "") &&
            target.hash.startsWith("#/rollout/");
        }
        if (
          ["http:", "https:"].includes(target.protocol) &&
          (localTrace || cloudTrace) &&
          !target.username && !target.password && !target.search &&
          target.hash.startsWith("#/rollout/")
        ) {
          const link = document.createElement("a");
          link.href = target.href;
          link.target = "_blank";
          link.rel = "noopener noreferrer";
          link.click();
        }
      } catch {
        // Ignore malformed or untrusted report navigation requests.
      }
      return;
    }
    if (
      !frame ||
      !message ||
      typeof message !== "object" ||
      message.namespace !== "autofa.failure-cards" ||
      message.type !== "failure-link-request" ||
      typeof message.function_name !== "string" ||
      !message.function_name ||
      message.function_name.length > 500 ||
      Object.keys(message).sort().join("|") !==
        "function_name|namespace|type"
    ) return;

    const functionName = message.function_name;
    const pageUrl = new URL(window.location.href);
    pageUrl.searchParams.set("failure", functionName);
    window.history.replaceState(null, "", pageUrl);
    frame.dataset.selectedFailure = functionName;

    const copyButton = document.querySelector("[data-copy-value]");
    const canonicalUrl = new URL(
      copyButton?.dataset.copyValue || pageUrl.href,
      window.location.href
    );
    canonicalUrl.searchParams.set("failure", functionName);
    if (copyButton) copyButton.dataset.copyValue = canonicalUrl.href;
    try {
      await navigator.clipboard.writeText(canonicalUrl.href);
      if (copyButton) {
        const previous = copyButton.textContent;
        copyButton.textContent = "Failure link copied";
        window.setTimeout(() => {
          copyButton.textContent = previous;
        }, 1600);
      }
    } catch {
      window.prompt("Copy this failure-mode link:", canonicalUrl.href);
    }
  });

  applyFilters();

  async function refreshSavedReports() {
    const progress = document.querySelector("#catalog-progress");
    if (!progress) return;
    try {
      // Poll the current snapshot without queuing another scan on every refresh.
      const response = await fetch("/", {
        cache: "no-store",
        headers: { "X-AutoFA-Catalog-Poll": "1" },
      });
      if (!response.ok) throw new Error("Library refresh failed");
      const page = new DOMParser().parseFromString(await response.text(), "text/html");
      const grid = document.querySelector("#run-grid");
      const updatedGrid = page.querySelector("#run-grid");
      if (grid && updatedGrid) grid.replaceChildren(...updatedGrid.childNodes);
      cards = Array.from(document.querySelectorAll(".run-card"));
      // Retain the user's search and selections as saved report versions appear.
      [model, benchmark, owner, harness].forEach((control) => {
        if (!control) return;
        const selected = control.value;
        const updated = page.querySelector(`#${control.id}`);
        if (updated) control.replaceChildren(...updated.childNodes);
        control.value = selected;
      });
      const empty = document.querySelector("#empty-state");
      const updatedEmpty = page.querySelector("#empty-state");
      if (empty && updatedEmpty) empty.replaceChildren(...updatedEmpty.childNodes);
      applyFilters();
      if (!page.querySelector("#catalog-progress")) {
        progress.textContent = "Saved libraries and reports are ready.";
        progress.removeAttribute("id");
        return;
      }
    } catch {
      progress.textContent = "Waiting for saved libraries and reports. You can refresh this page to retry.";
    }
    window.setTimeout(refreshSavedReports, 5000);
  }
  if (document.querySelector("#catalog-progress")) {
    window.setTimeout(refreshSavedReports, 5000);
  }
})();
