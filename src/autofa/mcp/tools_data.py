# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dataset MCP tools."""

from __future__ import annotations

from typing import Any

from autofa.data import parquet_store as ps
from autofa.data.trajectory import (
    get_step_detail,
    parse_trajectory,
)
from autofa.defaults import setting_value
from autofa.mcp.runtime import err, get_runtime, ok

# Agent traces can be multi-MB JSON; truncate only after parse.
_TRAJECTORY_PARSE_MAX_CHARS = 5_000_000
_TRAJECTORY_COVERAGE_SAMPLE = 40
# read_field_chunk's default window leaves room for JSON escaping and metadata below the
# MCP result ceiling (server.MCP_HARD_CEILING_CHARS), so mostly-ASCII text arrives inline.
# Structured readers keep their max_chars as a whole-page budget instead.
DEFAULT_READ_CHARS = 30_000


def get_trace_api(
    name: str | None = None, offset: int = 0, limit: int = 12_000,
) -> dict[str, Any]:
    """Read paged autofa.trace contracts; omit name for all, or select 'steps' / 'Step'.

    Includes signatures, defaults, return fields, optional keys, and evidence limits.
    Follow next_offset with the same name until null; join reference strings in order.
    complete means this single page holds the entire selected reference. The full
    reference is also saved as trace_api.md in each session workspace. This is static
    Python API documentation, not recorded trace content; it needs no dataset access.
    """
    import hashlib
    import json

    from autofa.trace_reference import trace_api_reference

    if type(offset) is not int or offset < 0 or type(limit) is not int or limit <= 0:
        return err("offset must be nonnegative and limit must be a positive integer")
    try:
        result = trace_api_reference(name)
    except ValueError as exc:
        return err(str(exc))
    reference = result.pop("reference")
    if offset > len(reference):
        return err("offset exceeds reference length", total_chars=len(reference))
    result.update(
        offset=offset, offset_unit="characters", total_chars=len(reference),
        reference_sha256=hashlib.sha256(reference.encode()).hexdigest(),
    )

    def page(end: int) -> dict[str, Any]:
        return ok(
            **result, reference=reference[offset:end],
            next_offset=end if end < len(reference) else None,
            complete=offset == 0 and end == len(reference),
        )

    end = min(len(reference), offset + min(limit, 12_000))
    # Bound the serialized response too: escaped Unicode can be much larger than
    # the reference text. The soft MCP limit is 30k; leave space for its envelope.
    lower, upper = offset, end
    while lower < upper:
        middle = (lower + upper + 1) // 2
        if len(json.dumps(page(middle))) <= 24_000:
            lower = middle
        else:
            upper = middle - 1
    if len(json.dumps(page(lower))) > 24_000 or (lower == offset and offset < len(reference)):
        return err("API reference metadata exceeds the tool budget; read trace_api.md")
    return page(lower)


def get_trace_record(
    item_id: str, section: str = "coverage", pointer: str = "", *,
    lane: str | None = None, event_idx: int | None = None,
    offset: int = 0, limit: int = 20, max_chars: int = 50_000,
    fetch_missing: bool = False,
) -> dict[str, Any]:
    """Read execution metadata, events, attempts, artifacts, definitions and source relations.

    Start with section='coverage'. Use section='events', lane='', event_idx=259 for a
    source event (not a parsed step number); pointer='/parts/0/text' reads its text.
    Sections use autofa.trace.record keys; section='record' opens the complete structure.
    JSON pointers and next_offset page all recorded values without discarding long fields.
    Reads are offline by default. For missing provenance, relations or prompt_summaries,
    explicitly set fetch_missing=True with that section to retrieve only this item's
    resource from its configured source API. Results are cached separately; the dataset
    and detector inputs stay unchanged. Repeat fetch_missing=True when paging the result.
    """
    from autofa import trace

    if fetch_missing and section not in {"provenance", "relations", "prompt_summaries"}:
        return err("fetch_missing requires section=provenance|relations|prompt_summaries")
    if fetch_missing and (lane is not None or event_idx is not None):
        return err("lane/event_idx selectors require section='events'")
    item = _load_item_raw(item_id)
    if item is None:
        return err(f"item not found: {item_id}")
    if pointer and not pointer.startswith("/"):
        return err("pointer must be empty or start with /")
    if section == "coverage":
        availability = trace.coverage(item)
        value = availability
    else:
        record = trace.record(item)
        if fetch_missing and record["availability"].get(section, {}).get("status") != "available":
            try:
                from autofa.config import autofa_home
                from autofa.data.source_resources import fetch_execution_resource
            except ImportError:
                return err("On-demand source metadata retrieval is unavailable in this build")
            try:
                fetch_execution_resource(
                    record, section,
                    base_url=setting_value(get_runtime().config, "data_api_base_url") or "",
                    root=autofa_home() / "cache" / "source-resources",
                )
            except (RuntimeError, OSError, ValueError) as exc:
                return err(str(exc), item_id=item_id, section=section)
        availability = record["availability"]
        if section == "record":
            value = record
        elif section in record:
            value = record[section]
        else:
            return err("unknown execution section", section=section,
                       available_sections=["coverage", "record", *record])
    if lane is not None or event_idx is not None:
        if section != "events":
            return err("lane/event_idx selectors require section='events'")
        value = [event for event in value
                 if (lane is None or event.get("lane") == lane)
                 and (event_idx is None or event.get("event_idx") == event_idx)]
        if event_idx is not None:
            if len(value) != 1:
                return err("source event missing or ambiguous; specify its lane and event_idx",
                           matches=len(value), availability=availability.get("events"))
            value = value[0]
    try:
        for token in pointer.split("/")[1:]:
            token = token.replace("~1", "/").replace("~0", "~")
            if isinstance(value, list):
                index = int(token)
                if index < 0:
                    raise IndexError
                value = value[index]
            elif isinstance(value, dict):
                value = value[token]
            else:
                raise KeyError(token)
    except (ValueError, IndexError, KeyError, TypeError) as exc:
        return err(f"invalid execution pointer: {exc}", item_id=item_id, pointer=pointer)
    return {
        **_parsed_page(value, pointer=pointer, offset=offset, limit=limit, max_chars=max_chars),
        "item_id": item_id, "section": section, "lane": lane, "event_idx": event_idx,
        "availability": availability.get(section) if section not in {"record", "coverage"}
        else availability,
        **({"resource_metadata": record.get("resource_metadata", {}).get(section, {})}
           if fetch_missing else {}),
        "hint": "Follow next_offset until is_final; use child_pointers for large structures. "
                "An unavailable resource is not evidence that no such event occurred."
                + (" Repeat fetch_missing=True for cached supplemental pages. These resources "
                   "are not added to the dataset or detector inputs." if fetch_missing else ""),
    }


def _response_text(item: dict[str, Any]) -> Any:
    from autofa.data.input_view import response_value

    return response_value(item)


def _load_item_raw(item_id: str) -> dict[str, Any] | None:
    """Fetch canonical values; outward-facing tools bound their own responses."""
    rt = get_runtime()
    result = ps.get_item(
        rt.dataset,
        item_id,
        max_chars=None,
        include_trace=False,
    )
    if not result.get("ok"):
        return None
    return result["item"]


def dataset_info() -> dict[str, Any]:
    """Corpus summary plus chunk estimates for full-parquet processing."""
    rt = get_runtime()
    info = rt.dataset.info(scan_limit=rt.run.eval_limit or 50_000)
    est = rt.dataset.estimate_matching_rows()
    chunk = int(setting_value(rt.config, "eval_chunk_size") or 512)
    info["chunk_size_default"] = chunk
    info["estimated_corpus_rows"] = est
    if est:
        info["estimated_chunks"] = (int(est) + chunk - 1) // chunk
    info["default_evaluation_mode"] = "capped"
    info["default_llm_eval_sample"] = max(
        0, min(20_000, int(setting_value(rt.config, "llm_eval_sample") or 0)),
    )
    info["hint"] = (
        "Use sample_items / peek_chunk to inspect data in pieces. "
        "Call describe_item_fields for per-column byte costs before large reads. "
        "Use read_field_chunk for lossless paging of any text field; "
        "view='detector' exposes the exact derived detector fields. "
        "Use get_input for parsed input messages/tools and get_model_calls for captured "
        "requests/responses; read_parsed_field navigates nested values with JSON pointers. "
        "For agent runs start with get_trajectory; search_trajectory finds full-trace "
        "matches and get_trajectory_context opens neighboring events. Expand with "
        "get_trajectory_step; detectors can use autofa.trace.steps. "
        "get_trace_record starts with resource coverage and exposes execution metadata, "
        "all retained lanes, attempts, artifacts, tool definitions and source relations. "
        "submit_function and evaluate_functions use capped mode with a fixed LLM sample; "
        "evaluate_functions(mode='full') streams the entire corpus in chunks."
    )
    # Parse coverage + capabilities (records diagnostics).
    from itertools import chain, islice

    from autofa.data.corpus_health import (
        capabilities_from_coverage,
        measure_parse_coverage,
    )

    # Parse the sample as a stream too: forty full trajectories can already be
    # hundreds of megabytes.
    batches = rt.dataset.iter_batches(
        batch_size=_TRAJECTORY_COVERAGE_SAMPLE, max_items=_TRAJECTORY_COVERAGE_SAMPLE,
    )
    try:
        coverage = measure_parse_coverage(islice(
            chain.from_iterable(batches), _TRAJECTORY_COVERAGE_SAMPLE,
        ))
    finally:
        if callable(close := getattr(batches, "close", None)):
            close()
    info["trajectory_coverage"] = coverage
    if coverage.get("warning"):
        info["trajectory_warning"] = coverage["warning"]
    caps = capabilities_from_coverage(coverage)
    readiness = getattr(rt, "llm_readiness", None)
    if readiness is None:
        # Constructing a client does not establish that its key/model works.
        try:
            hub = rt.ensure_hub_llm()
            if hub is None or getattr(hub, "available", True) is False:
                raise RuntimeError("LLM backend unavailable")
            readiness = {"status": "unverified", "configured": True, "available": None}
        except Exception:  # noqa: BLE001
            from autofa.llm.endpoints import endpoint_setup_help

            readiness = {
                "status": "not_configured", "configured": False, "available": False,
                "message": "LLM detectors cannot be evaluated. "
                f"{endpoint_setup_help(rt.config)}",
            }
    info["llm_detectors"] = dict(readiness)
    info["llm_detectors"].setdefault("model", rt.config.feature_model)
    api = info["llm_detectors"].setdefault(
        "api", setting_value(rt.config, "feature_api"),
    )
    info["llm_detectors"]["methods"] = ["classify_options", "classify", "localise"]
    if (info["llm_detectors"].get("label_api") or api) == "decisions":
        info["llm_detectors"]["methods"].append("decide")
    caps["llm_backend"] = (
        bool(readiness.get("configured")) and readiness.get("available") is not False
    )
    if readiness.get("available") is not True:
        info["llm_detector_warning"] = readiness.get("message") or (
            "LLM detector access is unverified. Do not interpret unavailable "
            "LLM evaluations as evidence that a behavior is absent."
        )
    info["capabilities"] = caps
    return ok(**info)


def describe_item_fields() -> dict[str, Any]:
    """Schema + read-cost report: columns, population, exact bytes, length hints."""
    from autofa.data.trace_fields import TRACE_FIELDS

    rt = get_runtime()
    cols = sorted(rt.dataset.available_columns())
    byte_stats: dict[str, Any] = {}
    try:
        byte_stats = ps.column_byte_stats(rt.dataset.parquet_paths)
    except Exception:  # noqa: BLE001
        byte_stats = {}

    # Length distribution from a small sample.
    lengths: dict[str, list[int]] = {f: [] for f in ps.TEXT_FIELDS}
    try:
        for batch in rt.dataset.iter_batches(batch_size=32, max_items=64):
            for it in batch:
                for f in ps.TEXT_FIELDS:
                    val = it.get(f)
                    if isinstance(val, str) and val:
                        lengths[f].append(len(val))
    except Exception:  # noqa: BLE001
        pass

    length_stats: dict[str, Any] = {}
    read_hints: dict[str, str] = {}
    for f, xs in lengths.items():
        if not xs:
            continue
        xs_sorted = sorted(xs)
        n = len(xs_sorted)
        median = xs_sorted[n // 2]
        p95 = xs_sorted[min(n - 1, int(n * 0.95))]
        mx = xs_sorted[-1]
        length_stats[f] = {"n": n, "median": median, "p95": p95, "max": mx}
        if median >= 50_000:
            read_hints[f] = (
                f"median {median:,} chars, p95 {p95:,}, max {mx:,} — "
                f"page with read_field_chunk(item_id=..., field='{f}', offset=0)"
            )
        elif median >= 5_000:
            read_hints[f] = (
                f"median {median:,} chars — prefer get_trajectory / read_field_chunk "
                f"over sample_items truncation"
            )

    return ok(
        columns=cols,
        text_fields=list(ps.TEXT_FIELDS),
        identity_fields=list(ps.IDENTITY_FIELDS),
        derived_trace_fields=list(TRACE_FIELDS),
        derived_input_fields=["input_text", "input_messages", "input_tools", "input_parse"],
        column_bytes=byte_stats,
        length_stats=length_stats,
        read_hints=read_hints,
        hint=(
            "derived_trace_fields are filled at detector eval time from response_text "
            "when it is an agent trajectory. Use read_field_chunk for lossless reads."
        ),
    )


def dataset_chunks(chunk_size: int | None = None) -> dict[str, Any]:
    """Describe how to walk the bound corpus in chunks (for full-parquet processing)."""
    rt = get_runtime()
    size = int(chunk_size or setting_value(rt.config, "eval_chunk_size") or 512)
    size = max(1, min(size, 4096))
    est = rt.dataset.estimate_matching_rows()
    n_chunks = (int(est) + size - 1) // size if est else None
    return ok(
        chunk_size=size,
        estimated_corpus_rows=est,
        estimated_chunks=n_chunks,
        how_to=(
            "Call peek_chunk(chunk_index=0..N-1) to inspect truncated rows, "
            "or evaluate_functions(mode='full', chunk_size=...) to score detectors "
            "over the full corpus without loading it all into RAM."
        ),
    )


def peek_chunk(
    chunk_index: int = 0,
    chunk_size: int | None = None,
    max_chars: int | None = None,
) -> dict[str, Any]:
    """Load one chunk of the corpus for inspection (truncated text fields)."""
    rt = get_runtime()
    size = int(chunk_size or setting_value(rt.config, "eval_chunk_size") or 512)
    size = max(1, min(size, 4096))
    out = rt.dataset.peek_chunk(
        int(chunk_index),
        chunk_size=size,
        max_chars=max_chars or min(1500, rt.config.text_truncate_chars),
    )
    return ok(**out) if out.get("ok", True) else err(out.get("error", "peek failed"))


def _library_coverage_snapshot(
    *, include_item_ids: bool = False, minimum_grade: str = "OK",
) -> dict[str, Any]:
    from autofa.behavior_policy import strictness_for_config
    from autofa.metrics.library_coverage import library_failure_coverage
    from autofa.metrics.library_quality_coverage import library_quality_coverage
    from autofa.objectives import discovery_strategy
    from autofa.session.family_coverage import cached_task_inventory

    rt = get_runtime()
    inventory = cached_task_inventory(rt)
    snapshot = library_failure_coverage(
        rt.library, inventory["task_by_failed_item_id"], run_id=rt.run.run_id,
        objective_strategy=discovery_strategy(rt.config), include_item_ids=include_item_ids,
        behavior_strictness=strictness_for_config(rt.config),
        task_by_item_id=inventory["task_by_item_id"],
    )
    return {"n_items": inventory["n_items"], "n_passed": inventory["n_passed_attempts"],
            "n_unknown_outcomes": inventory["n_unknown_outcomes"], **snapshot,
            "library_quality_coverage": library_quality_coverage(
                rt.library, inventory, run_id=rt.run.run_id,
                behavior_strictness=strictness_for_config(rt.config),
                minimum_grade=minimum_grade,
            )}


def library_coverage(minimum_grade: str = "OK") -> dict[str, Any]:
    """Read union failure coverage, distinct datapoints, repeats and marginal cases.

    Observed matches and confirmations under the selected review policy are separate.
    library_quality_coverage reports the overlap-deduplicated union of OK-or-better
    detector matches by default; minimum_grade="GOOD" selects Good-or-better matches.
    Its eligible count includes only detectors with usable current match evidence;
    n_good_detectors still counts actual Good-or-better detectors. The remaining
    match fields include all current detectors. This cutoff does not change grades
    or discovery targets.
    Parent/child overlaps count once. No detectors or judges are executed.
    Repeated attempts on one task are single-task diagnostics, not evidence of
    cross-datapoint generalization. Task IDs come from source metadata.
    """
    try:
        from autofa.session.task_coverage import task_coverage_snapshot

        return ok(**_library_coverage_snapshot(minimum_grade=minimum_grade),
                  task_coverage=task_coverage_snapshot(get_runtime()))
    except (ValueError, OSError) as exc:
        return err(f"Cannot establish failure coverage: {exc}")


def _selected_benchmarks(
    run: Any, benchmark: str | None, benchmarks: list[str] | None,
) -> list[str] | None:
    """Resolve explicit selectors before the run fallback, within the selected corpus."""
    allowed = run.resolved_benchmarks()
    selected = benchmarks or ([benchmark] if benchmark else allowed)
    if allowed and (outside := sorted(set(selected) - set(allowed))):
        raise ValueError(
            f"Benchmark filter is outside the selected run corpus: {outside}. "
            f"Allowed benchmarks: {allowed}"
        )
    return selected or None


def sample_items(
    n_fail: int | None = None,
    n_pass: int | None = None,
    benchmark: str | None = None,
    benchmarks: list[str] | None = None,
    prefer_informative: bool | None = None,
    by_benchmark: bool | None = None,
    min_response_chars: int = 0,
    require_fields: list[str] | None = None,
    seed: int | None = None,
    coverage: str = "all",
    distinct_tasks: bool | None = None,
) -> dict[str, Any]:
    """Sample traces, optionally prioritizing failures without current evidence.

    Omitted counts use configured defaults: 12 failed and 2 successful attempts.
    Omit seed to use the current round's derived seed. An explicit seed overrides
    only this call; it does not change the main seed or the round sequence.
    coverage='unmatched' excludes saved current detector positives;
    'unconfirmed' excludes individually confirmed causal cases. These are
    discovery filters, not judge sampling or claims that other cases are solved.
    'unconfirmed_tasks' targets new task coverage: exclude every failed rollout
    of tasks already confirmed by detectors meeting the discovery quality floors.
    Pass controls retain their normal sampling pool in every mode.
    For coverage-filtered mixed-benchmark runs, omitted by_benchmark balances
    sampling across selected benchmarks; explicit False preserves pooled sampling.
    This balances benchmark groups, not distinct source runs sharing a benchmark.
    distinct_tasks defaults to True for all discovery sampling:
    choose at most one attempt per known task in each outcome group. Unknown
    task identities remain explicitly unknown, never distinct-task evidence.
    A failed and passing rollout of the same task may both appear as a contrast.
    Use distinct_tasks=False to inspect more rollouts; they do not add task breadth.
    Explicit benchmark filters narrow the selected run corpus. A nonempty
    benchmarks list takes precedence over benchmark; omit both for the run scope.
    """
    if coverage not in {"all", "unmatched", "unconfirmed", "unconfirmed_tasks"}:
        return err("coverage must be all, unmatched, unconfirmed, or unconfirmed_tasks")
    rt = get_runtime()
    cfg = rt.config
    n_fail = n_fail if n_fail is not None else cfg.sample_fail_default
    n_pass = n_pass if n_pass is not None else cfg.sample_pass_default
    prefer = cfg.prefer_informative if prefer_informative is None else prefer_informative
    n_fail = min(int(n_fail), cfg.max_sample_per_call)
    n_pass = min(int(n_pass), cfg.max_sample_per_call)
    try:
        selected_benchmarks = _selected_benchmarks(rt.run, benchmark, benchmarks)
    except ValueError as exc:
        return err(str(exc))
    if seed is None:
        # SSoT controls the hypothesis prompt, not round-to-round sample diversity.
        seed = int(getattr(rt.run, "exploration_seed", 0) or 0)
    snapshot = None
    task_snapshot = None
    excluded = None
    diverse = True if distinct_tasks is None else distinct_tasks
    from autofa.session.family_coverage import cached_task_inventory

    # Keep source identities visible even when explicitly comparing repeat
    # rollouts, so their number cannot be mistaken for cross-task support.
    try:
        inventory = cached_task_inventory(rt)
    except (ValueError, OSError) as exc:
        return err(f"Cannot establish task identities: {exc}")
    if coverage == "unconfirmed_tasks":
        from autofa.session.task_coverage import task_coverage_snapshot

        task_snapshot = task_coverage_snapshot(rt, include_task_ids=True)
        if not task_snapshot["available"]:
            return err("Cannot establish quality-confirmed task coverage", **task_snapshot)
        covered_tasks = set(task_snapshot.pop("confirmed_failed_task_ids"))
        excluded = {item_id for item_id, task in inventory["task_by_failed_item_id"].items()
                    if task is not None and task in covered_tasks}
        excluded.update(inventory["unknown_outcome_item_ids"])
    elif coverage != "all":
        try:
            snapshot = _library_coverage_snapshot(include_item_ids=True)
        except (ValueError, OSError) as exc:
            return err(f"Cannot establish failure coverage: {exc}")
        ids_key = ("known_matched_failed_item_ids" if coverage == "unmatched"
                   else "review_confirmed_failed_item_ids")
        excluded = set(snapshot[ids_key])
        # Unknown benchmark outcomes are not known failed traces. Preserve the
        # legacy mixed pool in coverage='all', but never put them in a missed-
        # failure audit or its fixed failure denominator.
        excluded.update(inventory["unknown_outcome_item_ids"])
        snapshot.pop("known_matched_failed_item_ids")
        snapshot.pop("review_confirmed_failed_item_ids")
        for row in snapshot["detectors"]:
            row.pop("known_matched_failed_item_ids")
            row.pop("review_confirmed_failed_item_ids")
    # An empty run selector means the whole dataset, not a single benchmark.
    corpus_benchmarks = selected_benchmarks or {
        row.get("benchmark") for row in inventory["identity_by_item_id"].values()
        if row.get("benchmark")
    }
    balance_benchmarks = (coverage != "all" and len(corpus_benchmarks) > 1
                          if by_benchmark is None else by_benchmark)
    result = ps.sample_items(
        rt.dataset,
        n_fail=n_fail,
        n_pass=n_pass,
        benchmark=benchmark,
        benchmarks=selected_benchmarks,
        prefer_informative=prefer,
        by_benchmark=balance_benchmarks,
        min_response_chars=min_response_chars,
        require_fields=require_fields,
        seed=int(seed),
        max_chars=cfg.text_truncate_chars,
        max_total=cfg.max_sample_per_call,
        exclude_fail_item_ids=excluded,
        task_by_item_id=inventory["task_by_item_id"],
        distinct_tasks=diverse,
    )
    result["seed_used"] = int(seed)
    result["distinct_tasks"] = diverse
    result["by_benchmark"] = balance_benchmarks
    task_map = inventory["task_by_item_id"]
    for item in result["items"]:
        item["source_task_identity"] = task_map.get(item["_item_id"])
    failed_ids = {item["_item_id"] for item in result["items"]
                  if item["_item_id"] in inventory["task_by_failed_item_id"]}
    result["n_failed_tasks"] = len({task_map[item_id] for item_id in failed_ids
                                   if task_map.get(item_id) is not None})
    result["n_failed_attempts_without_task_identity"] = sum(
        task_map.get(item_id) is None for item_id in failed_ids)
    result["n_repeated_failed_attempts"] = (
        len(failed_ids) - result["n_failed_attempts_without_task_identity"]
        - result["n_failed_tasks"])
    if excluded is not None:
        result["coverage_filter"] = coverage
        result["n_excluded_known_failures"] = len(excluded) - len(
            inventory["unknown_outcome_item_ids"])
        if snapshot is not None:
            result["library_coverage"] = snapshot
        if task_snapshot is not None:
            result["task_coverage"] = task_snapshot
    return ok(**result)


def search_items(
    is_fail: bool | None = None,
    benchmark: str | None = None,
    benchmarks: list[str] | None = None,
    query: str | None = None,
    strata_difficulty: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> dict[str, Any]:
    """Search the selected corpus for trace evidence.

    Explicit benchmark filters narrow the selected run corpus. A nonempty
    benchmarks list takes precedence over benchmark; omit both for the run scope.
    """
    rt = get_runtime()
    try:
        selected_benchmarks = _selected_benchmarks(rt.run, benchmark, benchmarks)
    except ValueError as exc:
        return err(str(exc))
    kwargs: dict[str, Any] = dict(
        is_fail=is_fail,
        benchmark=benchmark,
        benchmarks=selected_benchmarks,
        query=query,
        strata_difficulty=strata_difficulty,
        limit=min(int(limit), rt.config.max_sample_per_call),
        offset=max(0, int(offset)),
        max_chars=rt.config.text_truncate_chars,
    )
    result = ps.search_items(rt.dataset, **kwargs)
    return ok(**result)


def get_item(
    item_id: str, include_trace: bool = False, max_chars: int | None = None,
) -> dict[str, Any]:
    rt = get_runtime()
    result = ps.get_item(
        rt.dataset,
        item_id,
        max_chars=max_chars or rt.config.text_truncate_chars * 3,
        include_trace=include_trace,
    )
    if not result.get("ok"):
        return err(result.get("error", "not found"))
    item = result["item"]
    # Completeness metadata for text fields.
    meta: dict[str, Any] = {}
    for key in ps.TEXT_FIELDS:
        val = item.get(key)
        if isinstance(val, str) and "…[truncated" in val:
            meta[key] = {
                "truncated": True,
                "full_read": f"read_field_chunk(item_id={item_id!r}, field={key!r}, offset=0)",
            }
    return ok(item=item, field_meta=meta or None)


def _parsed_page(value: Any, *, pointer: str, offset: int, limit: int, max_chars: int) -> dict:
    """Page structures without cutting serialized JSON or hiding omitted values."""
    import json

    offset, limit = max(0, int(offset)), max(1, min(int(limit), 200))
    max_chars = max(256, min(int(max_chars), 100_000))
    if isinstance(value, dict):
        keys = list(value)
        page = {k: value[k] for k in keys[offset:offset + limit]}
        total, count = len(keys), len(page)
    elif isinstance(value, list):
        page = value[offset:offset + limit]
        total, count = len(value), len(page)
    elif isinstance(value, str):
        page = value[offset:offset + max_chars]
        total, count = len(value), len(page)
    else:
        page, total, count = value, 1, 1
    if not isinstance(page, str) and len(json.dumps(page, ensure_ascii=False)) > max_chars:
        children = (list(page) if isinstance(page, dict) else
                    list(range(offset, offset + count)) if isinstance(page, list) else [])
        paths = [pointer + "/" + str(k).replace("~", "~0").replace("/", "~1")
                 for k in children]
        return err("structured page exceeds max_chars; choose a child pointer or smaller limit",
                   pointer=pointer, child_pointers=paths, total=total)
    end = offset + count
    return ok(value=page, value_type=type(value).__name__, pointer=pointer,
              offset=offset, total=total, next_offset=end if end < total else None,
              is_final=end >= total)


def get_input(
    item_id: str, section: str = "messages", offset: int = 0, limit: int = 20,
    max_chars: int = 50_000,
) -> dict[str, Any]:
    """Read parsed initial input: section='messages'|'tools'|'text', with provenance and paging."""
    from autofa.data.input_view import parsed_input

    if section not in {"messages", "tools", "text"}:
        return err("section must be messages|tools|text")
    item = _load_item_raw(item_id)
    if item is None:
        return err(f"item not found: {item_id}")
    view = parsed_input(item)
    result = _parsed_page(view[section], pointer="", offset=offset,
                          limit=limit, max_chars=max_chars)
    return {**result, "item_id": item_id, "origin": view["origin"],
            "limitations": view["limitations"], "section": section,
            "field": {"messages": "input_messages", "tools": "input_tools",
                      "text": "input_text"}[section],
            "hint": "Use read_parsed_field with this field and a child pointer for large values."}


def get_model_calls(
    item_id: str, offset: int = 0, limit: int = 1, max_chars: int = 50_000,
) -> dict[str, Any]:
    """Read captured calls as parsed request/response messages and tools; page with offset/limit."""
    from autofa import trace
    from autofa.data.input_view import response_value
    from autofa.data.structured import decode_container

    item = _load_item_raw(item_id)
    if item is None:
        return err(f"item not found: {item_id}")
    blob = decode_container(response_value(item))
    if isinstance(blob, list) and len(blob) == 1:
        blob = blob[0]
    evidence = blob.get("model_call_evidence", {}) if isinstance(blob, dict) else {}
    total = len(evidence.get("calls", []))
    offset, limit = max(0, int(offset)), max(1, min(int(limit), 5))
    calls = trace.model_calls(item, offset=offset, limit=limit)
    result = _parsed_page(calls, pointer="", offset=0, limit=limit, max_chars=max_chars)
    if not result["ok"]:
        result["child_pointers"] = [f"/{i}" for i in range(offset, offset + len(calls))]
    return {**result, "item_id": item_id, "field": "model_calls", "offset": offset,
            "total": total,
            "next_offset": offset + len(calls) if offset + len(calls) < total else None,
            "is_final": offset + len(calls) >= total,
            "hint": "read_parsed_field(field='model_calls', pointer='/0/request/0/content') "
                    "reads a nested value; absent calls mean no captured model requests/responses."}


def read_parsed_field(
    item_id: str, field: str, pointer: str = "", offset: int = 0, limit: int = 20,
    max_chars: int = 50_000,
) -> dict[str, Any]:
    """Read input, trajectory_steps, model_calls or execution with RFC 6901 pointers.

    Page objects, arrays or text without cutting serialized structures.
    """
    from autofa import trace
    from autofa.data.input_view import derive_input_fields

    if field == "execution":
        return get_trace_record(item_id, section="record", pointer=pointer,
                                offset=offset, limit=limit, max_chars=max_chars)
    item = _load_item_raw(item_id)
    if item is None:
        return err(f"item not found: {item_id}")
    if pointer and not pointer.startswith("/"):
        return err("pointer must be empty or start with /")
    tokens = [s.replace("~1", "/").replace("~0", "~") for s in pointer.split("/")[1:]]
    try:
        if field == "model_calls":
            if not tokens:
                return get_model_calls(item_id, offset=offset, limit=limit, max_chars=max_chars)
            index = int(tokens.pop(0))
            if index < 0:
                raise IndexError
            value = trace.model_calls(item, offset=index, limit=1)[0]
        elif field == "trajectory_steps":
            parsed = parse_trajectory(_response_text(item))
            if not parsed.ok:
                return err(parsed.warning or "trajectory unavailable", item_id=item_id)
            value = parsed._steps
        elif field in {"input_text", "input_messages", "input_tools", "input_parse"}:
            value = derive_input_fields(item)[field]
        else:
            return err("field must be input_text|input_messages|input_tools|input_parse|"
                       "trajectory_steps|model_calls|execution")
        for token in tokens:
            if isinstance(value, list):
                index = int(token)
                if index < 0:
                    raise IndexError
                value = value[index]
            elif isinstance(value, dict):
                value = value[token]
            else:
                raise KeyError(token)
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        return err(f"invalid parsed pointer or structure: {exc}", item_id=item_id, pointer=pointer)
    return {**_parsed_page(value, pointer=pointer, offset=offset, limit=limit, max_chars=max_chars),
            "item_id": item_id, "field": field}


def _field_view(item: dict[str, Any], field: str, view: str) -> tuple[Any, dict[str, Any]]:
    from autofa.data.trace_fields import derive_trace_fields

    selected = derive_trace_fields(item) if view == "detector" else item
    val = selected.get(field)
    stored = item.get(field)
    return val, {
        "view": view,
        "provenance": "trajectory" if val != stored else "stored",
        "stored_chars": len(stored) if isinstance(stored, str) else None,
        "detector_view": "Use view='detector' to inspect the fields passed to detectors.",
    }


def _field_text(val: Any) -> str:
    import json

    return val if isinstance(val, str) else json.dumps(val, ensure_ascii=False, default=str)


def get_field_meta(item_id: str, field: str, view: str = "stored") -> dict[str, Any]:
    """Field length/provenance; view='detector' applies the same derivation as execution."""
    if view not in {"stored", "detector"}:
        return err("view must be 'stored' or 'detector'")
    item = _load_item_raw(item_id)
    if item is None:
        return err(f"item not found: {item_id}")
    val, meta = _field_view(item, field, view)
    if val is None:
        return err(f"field not present: {field}", item_id=item_id, field=field, **meta)
    text = _field_text(val)
    return ok(
        **meta,
        item_id=item_id,
        field=field,
        total_chars=len(text),
        is_string=isinstance(val, str),
        hint=(f"read_field_chunk(item_id={item_id!r}, field={field!r}, "
              f"view={view!r}, offset=0, limit={DEFAULT_READ_CHARS})"),
    )


def read_field_chunk(
    item_id: str,
    field: str,
    offset: int = 0,
    limit: int = DEFAULT_READ_CHARS,
    view: str = "stored",
) -> dict[str, Any]:
    """Lossless char-window into any text field. Prefer this over get_item for large traces.
    Use view='detector' for the derived fields passed to detector execution.

    Returns ``{text, offset, returned_chars, total_chars, next_offset, is_final}``.
    Larger windows exceed the MCP result ceiling and come back as a saved result to
    page with read_tool_result.
    """
    if view not in {"stored", "detector"}:
        return err("view must be 'stored' or 'detector'")
    item = _load_item_raw(item_id)
    if item is None:
        return err(f"item not found: {item_id}")
    val, meta = _field_view(item, field, view)
    if val is None:
        return err(f"field not present: {field}", item_id=item_id, field=field, **meta)
    text = _field_text(val)
    total = len(text)
    offset = max(0, int(offset))
    limit = max(1, min(int(limit), 200_000))
    if offset >= total:
        return ok(
            **meta,
            item_id=item_id,
            field=field,
            text="",
            offset=offset,
            returned_chars=0,
            total_chars=total,
            next_offset=None,
            is_final=True,
            truncated=False,
        )
    chunk = text[offset : offset + limit]
    next_off = offset + len(chunk)
    is_final = next_off >= total
    return ok(
        **meta,
        item_id=item_id,
        field=field,
        text=chunk,
        offset=offset,
        returned_chars=len(chunk),
        total_chars=total,
        next_offset=None if is_final else next_off,
        is_final=is_final,
        truncated=not is_final,
        full_read=None
        if is_final
        else (f"read_field_chunk(item_id={item_id!r}, field={field!r}, "
              f"view={view!r}, offset={next_off})"),
    )


def corpus_trace_stats(n: int = 200, seed: int | None = None) -> dict[str, Any]:
    """Fail-vs-pass contrastive stats over tools, steps, surfaces, error signatures.

    Computed on a bounded uniform-ish sample. Lift suggests WHERE to look —
    never pipe high-lift tokens straight into a regex detector.
    """
    from autofa.data.corpus_stats import contrastive_corpus_stats
    rt = get_runtime()
    n = max(20, min(int(n), 2000))
    if seed is None:
        seed = int(getattr(rt.run, "exploration_seed", 0) or 0)
    # Mixed fail/pass draw over the selected corpus.
    sample = ps.sample_items(
        rt.dataset,
        n_fail=n // 2,
        n_pass=n - n // 2,
        prefer_informative=True,
        seed=int(seed),
        max_chars=_TRAJECTORY_PARSE_MAX_CHARS,
        max_total=n,
    )
    # sample_items truncates — reload raw for stats when possible.
    items = []
    for it in sample.get("items") or []:
        iid = it.get("_item_id")
        if iid:
            raw = _load_item_raw(str(iid))
            items.append(raw if raw is not None else it)
        else:
            items.append(it)
    stats = contrastive_corpus_stats(items)
    # Aggregate error signatures can guide later detector edits even though the
    # reply contains no individual IDs for the generic MCP exposure hook.
    if callable(getattr(rt.run, "run_dir", None)):
        from autofa.session.family_coverage import verified_identity_projection
        from autofa.session.review_evidence import record_discovery_exposure

        item_ids = [str(item["_item_id"]) for item in items if item.get("_item_id")]
        try:
            exposure = [verified_identity_projection(rt, item_id) for item_id in item_ids]
        except (AttributeError, ValueError, OSError):
            exposure = [{"_item_id": item_id} for item_id in item_ids]
        if exposure:
            record_discovery_exposure(rt.run.run_dir(), exposure)
    stats["seed_used"] = int(seed)
    return ok(**stats)


def get_trajectory(
    item_id: str,
    include_observations: bool = False,
    max_step_chars: int = 2_000,
    offset: int = 0,
    limit: int = 50,
) -> dict[str, Any]:
    """Parse an item's response_text/generation into a structured agent trajectory.

    Prefer this over raw get_item when exploring OpenHands/ATIF or steps-envelope JSON.
    Observations are summarized by length unless include_observations=True.
    Paginate with offset/limit (0-based offset into steps_summary).
    """
    item = _load_item_raw(item_id)
    if item is None:
        return err(f"item not found: {item_id}")
    text = _response_text(item)
    parsed = parse_trajectory(text)
    offset = max(0, int(offset))
    limit = max(1, min(int(limit), 200))
    max_step_chars = max(200, min(int(max_step_chars), 20_000))

    summary = parsed.to_dict(include_steps=False)
    steps_all = summary.pop("steps_summary", None) or []
    # Cap bulky fields in the summary envelope (full page is under `steps`).
    excerpts = summary.get("reasoning_excerpts") or []
    if len(excerpts) > 8:
        summary["reasoning_excerpts"] = excerpts[:8]
        summary["reasoning_excerpts_truncated"] = True
    if summary.get("workspace_diff_preview"):
        preview = str(summary["workspace_diff_preview"])
        if len(preview) > max_step_chars:
            summary["workspace_diff_preview"] = preview[:max_step_chars] + "…"

    total = len(steps_all)
    page = steps_all[offset : offset + limit]
    from autofa.data.trace_navigation import trace_events

    events = trace_events(item, include_content=False)
    for summary_step in page:
        event = events[summary_step["index"] - 1]
        for key in ("source_event", "paired_step_index", "paired_call_idx", "result_step_indices"):
            if key in event:
                summary_step[key] = event[key]

    if include_observations and parsed._steps:
        enriched = []
        for s in page:
            idx = int(s.get("index") or 0)
            detail = get_step_detail(parsed._steps, idx, max_chars=max_step_chars)
            row = dict(s)
            if detail:
                row["observation_excerpt"] = detail.get("observation") or ""
                row["reasoning_excerpt"] = (detail.get("reasoning") or "")[:max_step_chars]
            enriched.append(row)
        page = enriched

    out: dict[str, Any] = {
        "item_id": item_id,
        "response_chars": len(_field_text(text)),
        "parsed": summary,
        "steps_offset": offset,
        "steps_limit": limit,
        "steps_total": total,
        "next_offset": offset + len(page) if offset + len(page) < total else None,
        "steps": page,
        "hint": (
            "Explore ordered events with search_trajectory and get_trajectory_context. "
            "Read full step contents with get_trajectory_step or read_parsed_field. "
            "Detectors can use autofa.trace.steps for roles and tool/result relationships. "
            "Combined text fields are optional search views, not a substitute for sequence review."
        ),
    }
    if not parsed.ok:
        out["ok_parse"] = False
        if parsed.warning:
            out["warning"] = parsed.warning
        return ok(**out)
    out["ok_parse"] = True
    return ok(**out)


def list_trajectory_tools(item_id: str) -> dict[str, Any]:
    """Tool-name histogram and example args from a parsed agent trajectory."""
    item = _load_item_raw(item_id)
    if item is None:
        return err(f"item not found: {item_id}")
    parsed = parse_trajectory(_response_text(item))
    if not parsed.ok:
        return ok(
            item_id=item_id,
            ok_parse=False,
            tools=[],
            n_tool_calls=0,
            warning=parsed.warning,
            tool_definitions=parsed.tool_definitions,
        )
    return ok(
        item_id=item_id,
        ok_parse=True,
        dialect=parsed.dialect,
        n_tool_calls=parsed.n_tool_calls,
        tools=[
            {
                "name": t.name,
                "count": t.count,
                "example_args_excerpt": t.example_args_excerpt,
            }
            for t in parsed.tools
        ],
        tool_definitions=parsed.tool_definitions,
    )


def get_trajectory_step(
    item_id: str,
    step_index: int,
    max_chars: int = 50_000,
    field: str | None = None,
    offset: int = 0,
) -> dict[str, Any]:
    """Return one trajectory step (reasoning, tool_calls, observation).

    step_index is 1-based (matches steps_summary.index from get_trajectory).
    Optional ``field`` + ``offset`` page a single sub-field (reasoning /
    message / observation) when it exceeds max_chars.
    """
    item = _load_item_raw(item_id)
    if item is None:
        return err(f"item not found: {item_id}")
    parsed = parse_trajectory(_response_text(item))
    if not parsed.ok or not parsed._steps:
        return err(
            parsed.warning
            or "response is not a parseable steps trajectory",
            item_id=item_id,
            ok_parse=False,
        )
    max_chars = max(500, min(int(max_chars), 100_000))
    detail = get_step_detail(parsed._steps, int(step_index), max_chars=10**9)
    if detail is None:
        return err(
            f"step_index out of range: {step_index} (n_steps={parsed.n_steps})",
            item_id=item_id,
            n_steps=parsed.n_steps,
        )
    from autofa.data.trace_navigation import trace_events

    event = trace_events(item, include_content=False)[int(step_index) - 1]
    navigation = {k: event[k] for k in (
        "step_id", "source_event", "paired_step_index", "paired_call_idx", "result_step_indices",
    ) if k in event}
    detail.update(navigation)

    # Sub-field paging.
    if field:
        key = str(field)
        if key not in {"reasoning", "message", "observation"}:
            return err(f"field must be reasoning|message|observation, got {key!r}")
        full = str(detail.get(key) or "")
        total = len(full)
        offset = max(0, int(offset))
        chunk = full[offset : offset + max_chars]
        next_off = offset + len(chunk)
        is_final = next_off >= total
        return ok(
            **navigation,
            item_id=item_id,
            step_index=int(step_index),
            field=key,
            text=chunk,
            offset=offset,
            returned_chars=len(chunk),
            total_chars=total,
            next_offset=None if is_final else next_off,
            is_final=is_final,
            truncated=not is_final,
        )

    # Default: truncate each sub-field with completeness metadata.
    meta: dict[str, Any] = {}
    for key in ("reasoning", "message", "observation"):
        full = str(detail.get(key) or "")
        total = len(full)
        if total > max_chars:
            detail[key] = full[:max_chars] + f"…[+{total - max_chars} chars]"
            meta[key] = {
                "truncated": True,
                "returned_chars": max_chars,
                "total_chars": total,
                "full_read": (
                    f"get_trajectory_step(item_id={item_id!r}, step_index={step_index}, "
                    f"field={key!r}, offset=0, max_chars={max_chars})"
                ),
            }
        else:
            meta[key] = {
                "truncated": False,
                "returned_chars": total,
                "total_chars": total,
            }
    return ok(
        item_id=item_id,
        dialect=parsed.dialect,
        n_steps=parsed.n_steps,
        step=detail,
        field_meta=meta,
    )


def _event_preview(event: dict[str, Any], max_chars: int) -> dict[str, Any]:
    """Bound every section; full structured data is available through existing readers."""
    return {k: v for k, v in event.items() if k not in {"sections", "tool_calls"}} | {
        "sections": [{"key": p["key"], "label": p["label"], "text": p["text"][:max_chars],
                      "total_chars": len(p["text"]), "truncated": len(p["text"]) > max_chars}
                     for p in event["sections"]],
    }


def get_trajectory_context(
    item_id: str, step_index: int, before: int = 2, after: int = 2, max_chars: int = 2000,
) -> dict[str, Any]:
    """Read ordered events around a step, including reasoning, calls and results.

    Previews explicitly mark omissions. Use get_trajectory_step for full text;
    read_parsed_field(field='trajectory_steps') pages nested tool arguments.
    """
    from autofa.data.trace_navigation import trace_events

    item = _load_item_raw(item_id)
    if item is None:
        return err(f"item not found: {item_id}")
    events = trace_events(item)
    if not 1 <= step_index <= len(events):
        return err("step_index out of range or unparseable trajectory", steps_total=len(events))
    start = max(0, step_index - 1 - max(0, min(int(before), 10)))
    end = min(len(events), step_index + max(0, min(int(after), 10)))
    return ok(item_id=item_id, steps_total=len(events), steps_offset=start,
              next_offset=end if end < len(events) else None,
              events=[_event_preview(e, max(100, min(int(max_chars), 5000)))
                      for e in events[start:end]])


def search_trajectory(
    item_id: str, query: str, field: str = "all", offset: int = 0, limit: int = 20,
    case_sensitive: bool = False,
) -> dict[str, Any]:
    """Search literal text across the full ordered trace, returning event-local offsets.

    field: all|reasoning|message|observation|tool. offset/limit page occurrences,
    not a prefix of the trace. Open a hit with get_trajectory_context, then inspect
    the full step and later recovery before drawing a behavioral conclusion.
    """
    import re

    from autofa.data.trace_navigation import trace_events

    if not query or len(query) > 2000:
        return err("query must contain between 1 and 2000 characters")
    if field not in {"all", "reasoning", "message", "observation", "tool"}:
        return err("field must be all|reasoning|message|observation|tool")
    item = _load_item_raw(item_id)
    if item is None:
        return err(f"item not found: {item_id}")
    events = trace_events(item)
    if not events:
        return err("response is not a parseable steps trajectory")
    offset, limit = max(0, int(offset)), max(1, min(int(limit), 100))
    pattern = re.compile(re.escape(query), 0 if case_sensitive else re.IGNORECASE)
    matches, total = [], 0
    for event in events:
        for part in event["sections"]:
            if field != "all" and part["key"].split(":")[0] != field:
                continue
            for match in pattern.finditer(part["text"]):
                if offset <= total < offset + limit:
                    start, end = match.span()
                    lo, hi = max(0, start - 160), min(len(part["text"]), end + 160)
                    hit = {"step_index": event["index"], "step_id": event["step_id"],
                           "source": event["source"], "section": part["key"],
                           "start": start, "end": end, "quote": match.group(),
                           "context": part["text"][lo:hi], "context_start": lo,
                           "context_end": hi, "total_chars": len(part["text"])}
                    if "source_event" in event:
                        hit["source_event"] = event["source_event"]
                    matches.append(hit)
                total += 1
    return ok(item_id=item_id, matches=matches, total_matches=total, offset=offset,
              next_offset=offset + len(matches) if offset + len(matches) < total else None,
              steps_total=len(events), hint="Read get_trajectory_context around each hit; "
              "check subsequent actions and results for correction or recovery.")
