# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cheating check: screen rollouts, review the flagged ones, summarise the verdicts."""

from __future__ import annotations

import hashlib
import json
import re
import sys
import tempfile
from collections import Counter
from collections.abc import Callable, Collection, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from contextvars import copy_context
from pathlib import Path
from typing import Any

from autofa.core.fields import DEFAULT_CONTENT_FIELDS
from autofa.preflight.cheating.detector import (
    ANYWHERE,
    COMMANDS,
    EDITING,
    EDITS,
    RUBRIC,
    confirm,
    screen,
)

VERSION = 4  # bump when screening logic changes outside the hashed patterns
# Everything the judge can read (system prompt included), so a cached verdict is reused only
# for the same evidence.
CONTENT = tuple(sorted(DEFAULT_CONTENT_FIELDS))


def _client(cfg: Any) -> Any:
    """Keep the cheating review on Chat Completions with endpoint-default reasoning."""
    from autofa.llm.endpoints import endpoint_for_config
    from autofa.llm.hub import make_hub_client

    endpoint = endpoint_for_config(cfg, "judge", api="chat_completions")
    return make_hub_client(base_url=endpoint.base_url, api_key_env=endpoint.api_key_env)


def check_version(cfg: Any) -> str:
    # Discovery transport settings do not change this workflow's requests. Preserve
    # existing verdict caches and the capped review cohort when those settings change.
    patterns = [p.pattern for p in (ANYWHERE, COMMANDS, EDITS, EDITING)]
    raw = json.dumps([VERSION, RUBRIC, *patterns, cfg.cheating_check_model])
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _key(item: dict[str, Any], version: str) -> str:
    raw = json.dumps([version, item.get("_item_id"), *(item.get(f) for f in CONTENT)], default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


class _Cache:
    """Judged results by rollout content and check version; unjudged ones are retried later."""

    def __init__(self, version: str):
        from autofa.config import autofa_home

        self.path = autofa_home() / "preflight" / "cheating" / f"{version}.jsonl"
        self.rows: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                self.rows[row["key"]] = row["result"]

    def tokens_per_review(self) -> int | None:
        """Median tokens of this version's earlier reviews: a measurement, not a price guess."""
        spent = sorted(int(r["tokens"]) for r in self.rows.values() if r.get("tokens"))
        return spent[len(spent) // 2] if spent else None

    def put(self, key: str, result: dict[str, Any]) -> None:
        if result["status"] in ("unjudged", "not_reviewed"):
            return
        self.rows[key] = result
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"key": key, "result": result}, default=str) + "\n")


def _clean(result: dict[str, Any]) -> dict[str, Any]:
    return {**result, "status": "clean", "cheated": False, "pass_contaminated": False}


def judge_items(items: Iterable[dict[str, Any]], cfg: Any, workdir: Path, *,
                max_reviews: int | None = None,
                on_plan: Callable[[dict[str, Any]], None] | None = None,
                on_review: Callable[[dict[str, Any]], None] | None = None,
                ) -> dict[str, dict[str, Any]]:
    """Screen every item and review escalated ones.

    `max_reviews` caps the LLM reviews; escalated rollouts beyond it stay `not_reviewed`. The
    reviewed ones are a pseudo-random sample (by cache key), so the counts are not biased towards
    the first tasks of a run and the same rollouts are picked on every run. `on_plan` hears how
    many reviews are about to start, before the first one does; `on_review` hears each result as
    its review finishes.
    """
    version = check_version(cfg)
    cache = _Cache(version)
    results: dict[str, dict[str, Any]] = {}
    # Flagged rollouts with their screen, or their cached verdict when one exists.
    flagged: list[tuple[str, str, dict[str, Any], dict[str, Any] | None,
                        dict[str, Any] | None]] = []
    for item in items:
        iid, key = str(item["_item_id"]), _key(item, version)
        cached = cache.rows.get(key)
        if cached is not None and not cached.get("escalated"):
            results[iid] = {**cached, "cached": True}
            continue
        if cached is not None:
            flagged.append((iid, key, item, None, cached))
            continue
        s = screen(item)
        if s["escalate"]:
            flagged.append((iid, key, item, s, None))
        else:
            results[iid] = _clean(s)
            cache.put(key, results[iid])

    if max_reviews is not None:
        # Sample before looking at the cache, so a rerun picks the same rollouts and reuses
        # their verdicts instead of paying for new ones.
        flagged.sort(key=lambda f: hashlib.sha256(f[1].encode()).hexdigest())
        for iid, _, item, s, _ in flagged[max_reviews:]:
            results[iid] = {**(s or screen(item)), "status": "not_reviewed", "escalated": True,
                            "error": f"review cap of {max_reviews} reached"}
        flagged = flagged[:max_reviews]
    escalated: list[tuple[str, str, dict[str, Any], dict[str, Any]]] = []
    for iid, key, item, s, cached in flagged:
        if cached is not None:
            results[iid] = {**cached, "cached": True}
        else:
            escalated.append((iid, key, item, s))
    if on_plan is not None:
        on_plan({"flagged": len(escalated) + sum(r["status"] == "not_reviewed"
                                                 for r in results.values()),
                 "reviewing": len(escalated),
                 "reused": sum(bool(r.get("cached") and r.get("escalated"))
                               for r in results.values()),
                 "tokens_per_review": cache.tokens_per_review()})
    if not escalated:
        return results
    # Without an endpoint every review would come back unjudged: fail instead of reporting that.
    client = _client(cfg)

    reviews = workdir / "reviews"
    reviews.mkdir(parents=True, exist_ok=True)

    def review(iid: str, item: dict[str, Any], s: dict[str, Any]) -> dict[str, Any]:
        directory = Path(tempfile.mkdtemp(prefix=re.sub(r"[^\w.-]+", "_", iid)[:60] + "-",
                                          dir=reviews)) / "review"
        try:
            result = confirm(item, iid, model=cfg.cheating_check_model, client=client,
                             directory=directory)
        except Exception as exc:  # noqa: BLE001 - a failed review is unjudged, never clean
            result = {"status": "unjudged", "error": f"review failed: {exc}"}
        return {**s, **result, "escalated": True, "review_dir": str(directory)}

    with ThreadPoolExecutor(max(1, cfg.judge_workers)) as pool:
        futures = {pool.submit(copy_context().run, review, iid, item, s): (iid, key)
                   for iid, key, item, s in escalated}
        for future in as_completed(futures):
            iid, key = futures[future]
            results[iid] = future.result()
            cache.put(key, results[iid])
            if on_review is not None:
                try:
                    on_review(results[iid])
                except Exception as exc:  # noqa: BLE001 - a display failure must not cost the reviews
                    on_review = None  # stop updating the display first
                    # The display often fails because the output is gone: a broken pipe, a closed
                    # stream or a terminal that cannot encode the text.
                    with suppress(OSError, ValueError):
                        print(f"warning: review progress stopped: {exc}", file=sys.stderr)
    return results


def summarize(results: dict[str, dict[str, Any]], cfg: Any,
              passed: Collection[str] = ()) -> dict[str, Any]:
    """Counts over every checked rollout; `passed` holds the IDs of the ones that succeeded.

    `contaminated` is cheating that could have produced a success, whatever the outcome;
    `contaminated_passes` is the part of it that did pass."""
    statuses = Counter(r["status"] for r in results.values())
    passed = set(passed)
    return {
        "version": check_version(cfg), "model": cfg.cheating_check_model,
        "rollouts": len(results), "passed": sum(iid in passed for iid in results),
        "escalated": sum(bool(r.get("escalated")) for r in results.values()),
        "cheated": sum(bool(r.get("cheated")) for r in results.values()),
        "contaminated": sum(bool(r.get("pass_contaminated")) for r in results.values()),
        "contaminated_passes": sum(bool(r.get("pass_contaminated")) and iid in passed
                                   for iid, r in results.items()),
        "unjudged": statuses["unjudged"], "not_reviewed": statuses["not_reviewed"],
        "cached": sum(bool(r.get("cached")) for r in results.values()),
        # Tokens spent by this invocation; cached verdicts cost nothing again.
        "tokens": sum(int(r.get("tokens") or 0) for r in results.values() if not r.get("cached")),
        "statuses": dict(statuses),
    }
