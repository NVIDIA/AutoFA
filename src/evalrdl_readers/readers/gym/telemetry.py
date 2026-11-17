# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OTLP sidecars: observations, with ownership proved independently of trace grouping.

The collector's wire objects remain alongside decoded attributes. In particular, JSON's
int64 strings and the full metric point are retained: converting timestamps to datetime
or histograms to averages is deliberately a consumer decision.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from datetime import UTC, datetime

from evalrdl_readers.core import (
    ROOT,
    Answer,
    At,
    Card,
    Contribute,
    Grain,
    Input,
    Join,
    Key,
    Lane,
    Link,
    Merge,
    Question,
    Unreadable,
)

from ..common import primary_stream, records, text_parts
from .record import _gym_key

# The reader result retains observations, even when the input is streamed. Bound the
# optional sidecars before parsing: multi-GB metric exports otherwise exhaust the
# decoder's memory and prevent the evaluation itself from being ingested.
TELEMETRY_MAX_BYTES = 256 * 1024 * 1024


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def identity(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def any_value(value):
    """Decode OTLP AnyValue without erasing the original typed wire representation."""
    if "intValue" in value:
        return int(value["intValue"])
    if "arrayValue" in value:
        return [any_value(v) for v in value["arrayValue"].get("values", ())]
    if "kvlistValue" in value:
        return attributes(value["kvlistValue"].get("values", ()))
    if "doubleValue" in value:
        number = value["doubleValue"]
        # Proto JSON may spell a double as `1`; retain the declared floating type.
        # Its special non-finite spellings remain strings, as the wire JSON represents them.
        return float(number) if isinstance(number, (int, float)) else number
    for name in ("stringValue", "boolValue", "bytesValue"):
        if name in value:
            return value[name]
    return None


def attributes(values):
    return {a["key"]: any_value(a.get("value", {})) for a in values}


def flatten(envelope, signal):
    """Yield one span or metric point, carrying its resource/scope/schema context."""
    plural = "Spans" if signal == "span" else "Metrics"
    for resource in envelope.get(f"resource{plural}", ()):
        res = resource.get("resource", {})
        for scope in resource.get(f"scope{plural}", ()):
            context = {
                "resource": res,
                "resource_attributes": attributes(res.get("attributes", ())),
                "resource_schema_url": resource.get("schemaUrl"),
                "scope": scope.get("scope", {}),
                "scope_schema_url": scope.get("schemaUrl"),
            }
            if signal == "span":
                for span in scope.get("spans", ()):
                    yield {
                        **context,
                        "signal": signal,
                        "raw": span,
                        "attributes": attributes(span.get("attributes", ())),
                    }
            else:
                for metric in scope.get("metrics", ()):
                    for kind in ("gauge", "sum", "histogram", "exponentialHistogram", "summary"):
                        if kind in metric:
                            instrument = metric[kind]
                            header = {k: v for k, v in metric.items() if k != kind}
                            for point in instrument.get("dataPoints", ()):
                                yield {
                                    **context,
                                    "signal": signal,
                                    "instrument_kind": kind,
                                    "metric": header,
                                    "aggregation_temporality": instrument.get(
                                        "aggregationTemporality"
                                    ),
                                    "is_monotonic": instrument.get("isMonotonic"),
                                    "attributes": attributes(point.get("attributes", ())),
                                    "raw": point,
                                }


def capture_key(record):
    """Use the saved physical attempt's capture identity, preserving opaque IDs."""
    explicit = record.get("_ng_rollout_id")
    if explicit is not None:
        key = str(explicit)
    elif record.get("_ng_task_index") is not None and record.get("_ng_rollout_index") is not None:
        key = f"{record['_ng_task_index']}-{record['_ng_rollout_index']}"
    else:
        return None
    attempt = record.get("_ng_attempt_index")
    if attempt is not None and int(attempt) > 0:
        key += f"-a{int(attempt)}"
    return key


# vLLM's completion prefix and optional numeric parallel-sample suffix are explicit.
# Arbitrary backend suffixes are not stripped: an unknown format cannot prove ownership.
_REQUEST = re.compile(r"^(?:chatcmpl-)?(.+)-[0-9a-fA-F]{8}(?:-[0-9]+)?$")


def request_capture(request_id):
    match = _REQUEST.fullmatch(request_id) if isinstance(request_id, str) else None
    return match.group(1) if match else None


def span_id(observation):
    raw = observation["raw"]
    return f"{raw.get('traceId', '')}/{raw.get('spanId', '')}"


def parent_id(observation):
    raw = observation["raw"]
    parent = raw.get("parentSpanId")
    return (
        f"{raw.get('traceId', '')}/{parent}"
        if isinstance(parent, str) and parent and set(parent) != {"0"}
        else None
    )


def valid_span(observation):
    """Malformed identities or intervals remain wire observations, never lane events."""
    raw = observation["raw"]
    for field, width in (("traceId", 32), ("spanId", 16)):
        value = raw.get(field)
        if (
            not isinstance(value, str)
            or not re.fullmatch(f"[0-9a-fA-F]{{{width}}}", value)
            or set(value) == {"0"}
        ):
            return False
    stamps = [raw.get(field) for field in ("startTimeUnixNano", "endTimeUnixNano")]
    try:
        return all(not isinstance(v, bool) and 0 <= int(v) < 2**64 for v in stamps) and int(
            stamps[1]
        ) >= int(stamps[0])
    except (TypeError, ValueError):
        return False


def correlate(spans, captures, contested=()):
    """Resolve verified parent ancestry and request IDs; links never convey ownership.

    `captures` maps exact physical capture IDs to sets of canonical reader keys. Conflicting
    evidence poisons the branch, including evidence naming an unknown physical attempt.
    Neither a shared trace ID nor a time interval contributes evidence.
    """
    result = {}
    seeds = {}
    for sid, observation in spans.items():
        attrs = observation["attributes"]
        stated = []
        if attrs.get("nemo.gym.rollout.id") is not None:
            stated.append(str(attrs["nemo.gym.rollout.id"]))
        if attrs.get("gen_ai.request.id") is not None:
            request = request_capture(attrs["gen_ai.request.id"])
            stated.append(
                request
                if request is not None
                else f"unrecognized-request:{attrs['gen_ai.request.id']}"
            )
        seeds[sid] = set(stated)

    # Iterative walking bounds Python's stack even for a deep HTTP/service trace.
    for start in sorted(spans):
        path, seen = [], set()
        current = start
        while current in spans and current not in result and current not in seen:
            seen.add(current)
            path.append(current)
            current = parent_id(spans[current])
        cyclic = current in seen
        inherited = result.get(current, (frozenset(), False))
        if cyclic:
            inherited = (frozenset(), True)
        for sid in reversed(path):
            own = seeds[sid]
            candidates = set(inherited[0]) | own
            conflict = (
                inherited[1]
                or len(candidates) > 1
                or sid in contested
                or not valid_span(spans[sid])
            )
            parent = spans.get(parent_id(spans[sid]))
            if parent is not None:
                child_run = spans[sid]["resource_attributes"].get("run_id")
                parent_run = parent["resource_attributes"].get("run_id")
                conflict |= bool(child_run and parent_run and child_run != parent_run)
            if candidates:
                mapped = [captures.get(k, set()) for k in candidates]
                conflict |= any(len(keys) != 1 for keys in mapped)
            result[sid] = (frozenset(candidates), conflict)
            inherited = result[sid]
    return result


def index_observations(observations):
    """Exact repeated exports collapse; differing versions of a span remain contested."""
    unique, variants = {}, defaultdict(set)
    for observation in observations:
        digest = identity(observation)
        unique.setdefault(digest, observation)
        if observation["signal"] == "span":
            variants[span_id(observation)].add(digest)
    spans = {sid: unique[min(hashes)] for sid, hashes in variants.items()}
    conflicts = {sid for sid, hashes in variants.items() if len(hashes) > 1}
    return unique, spans, conflicts


def _clock(nanos):
    if nanos is None:
        return None
    seconds, fraction = divmod(int(nanos), 1_000_000_000)
    return datetime.fromtimestamp(seconds, UTC).replace(microsecond=fraction // 1000).isoformat()


def _lane(observation):
    return "vllm" if observation["raw"].get("name") == "llm_request" else "otel"


def _observations(src):
    # Rigid patterns overlap at supported depths (e.g. artifacts/otel and */otel).
    # One physical file is read once, independently of how many patterns name it.
    docs = {d.relpath: d for i in range(_TELEMETRY_READ_COUNT) for d in src.docs(i)}
    remaining = TELEMETRY_MAX_BYTES
    for signal, filename in (("span", "traces.jsonl"), ("metric", "metrics.jsonl")):
        for doc in (docs[p] for p in sorted(docs) if docs[p].name == filename):
            if doc.size > remaining:
                yield Unreadable(
                    "run.telemetry",
                    doc.relpath,
                    f"telemetry byte budget exceeded: {doc.size} bytes, {remaining} remaining "
                    f"of {TELEMETRY_MAX_BYTES}; sidecar not decoded, raw file retained",
                )
                continue
            remaining -= doc.size
            for item in records(doc, "run.telemetry"):
                if isinstance(item, Unreadable):
                    yield item
                else:
                    line, envelope = item
                    wrapper = "resourceSpans" if signal == "span" else "resourceMetrics"
                    if not isinstance(envelope, dict) or wrapper not in envelope:
                        yield Unreadable(
                            "run.telemetry",
                            f"{doc.relpath}:{line}",
                            f"OTLP envelope has no {wrapper}",
                        )
                    else:
                        for observation in flatten(envelope, signal):
                            yield observation, {"path": doc.relpath, "line": line}


def gym_telemetry(src):
    """Read optional collector files and bind only the selected saved physical attempts.

    Two passes keep the runtime's source cursor on the actual OTLP line when yielding
    events, after the ownership graph is complete. Metrics stay streamed on the second
    pass; only their digests and provenance are indexed.
    """
    if not any(src.has(i) for i in range(_TELEMETRY_READ_COUNT)):
        return
    captures = defaultdict(set)
    stream = primary_stream(
        [d for i in range(_TELEMETRY_READ_COUNT, _TELEMETRY_READ_COUNT + 4) for d in src.docs(i)]
    )
    if stream is not None:
        for item in records(stream, "rollout.events"):
            if isinstance(item, Unreadable):
                yield item
            else:
                line, record = item
                capture = capture_key(record)
                if capture is not None:
                    captures[capture].add(_gym_key(record, line))

    locations, spans, versions, run_ids = defaultdict(list), {}, defaultdict(set), set()
    for item in _observations(src):
        if isinstance(item, Unreadable):
            # Report on the emitting pass, at its own source cursor.
            continue
        observation, provenance = item
        digest = identity(observation)
        locations[digest].append(provenance)
        producer_run = observation["resource_attributes"].get("run_id")
        if producer_run is not None:
            run_ids.add(str(producer_run))
        if observation["signal"] == "span":
            sid = span_id(observation)
            versions[sid].add(digest)
            spans.setdefault(sid, observation)
    contested = {sid for sid, hashes in versions.items() if len(hashes) > 1}
    associations = correlate(spans, captures, contested)
    if len(run_ids) == 1:
        yield Answer("run.source_run_id", next(iter(run_ids)))

    owners = {}
    for sid, (candidates, conflict) in associations.items():
        if len(candidates) == 1 and not conflict and len(run_ids) <= 1:
            capture = next(iter(candidates))
            owners[sid] = (next(iter(captures[capture])), capture)

    emitted = set()
    for item in _observations(src):
        if isinstance(item, Unreadable):
            yield item
            continue
        observation, provenance = item
        digest = identity(observation)
        if digest in emitted:
            continue
        emitted.add(digest)
        sid = span_id(observation) if observation["signal"] == "span" else None
        ownership = owners.get(sid)
        candidates, conflict = associations.get(sid, (frozenset(), False))
        payload = {
            **observation,
            "key": digest,
            "provenance": locations[digest],
            "producer_run_id": observation["resource_attributes"].get("run_id"),
            "correlation": {
                "status": "assigned"
                if ownership
                else ("conflicting" if conflict or len(run_ids) > 1 else "unassigned"),
                "capture_ids": sorted(candidates),
                "selected_capture_id": ownership[1] if ownership else None,
                "rollout_key": list(ownership[0]) if ownership else None,
            },
        }
        # Every observation is available at run grain for the schema/UI consumers,
        # including job/startup roots, shared metrics, and unresolved associations.
        yield Answer("run.telemetry", payload, input_key=digest)
        if ownership is not None and sid is not None and sid not in contested:
            raw = observation["raw"]
            start, end = raw.get("startTimeUnixNano"), raw.get("endTimeUnixNano")
            parent = parent_id(observation)
            same_owner = parent in owners and owners[parent] == ownership
            same_lane = same_owner and _lane(spans[parent]) == _lane(observation)
            refs = {}
            if same_owner and not same_lane:
                refs["telemetry_parent"] = [parent]
            request = observation["attributes"].get("gen_ai.request.id")
            if isinstance(request, str) and request:
                refs["served_request"] = [request]
            yield Answer(
                "rollout.events",
                {
                    "source_event_id": sid,
                    "read_seq": f"{int(start):020d}" if start is not None else sid,
                    "started_at": _clock(start),
                    "ended_at": _clock(end),
                    "latency_s": (int(end) - int(start)) / 1e9 if start and end else None,
                    "kind": "other",
                    "kind_raw": raw.get("name"),
                    "role": "observation",
                    "parent_event_id": parent if same_lane else None,
                    "response_id": request,
                    "parts": text_parts(canonical(payload)),
                },
                key=ownership[0],
                input_key=sid,
                lane=(_lane(observation), ""),
                keys={"telemetry_span": sid},
                refs=refs,
            )


RUN_TELEMETRY = Question(
    name="run.telemetry",
    grain=Grain.RUN,
    merge=Merge.UNION,
    lands="runs.telemetry",
    means=(
        "collector observations with exact OTLP values, dimensions, provenance and ownership "
        "evidence"
    ),
)

# Collector sidecars are siblings of artifacts in job layouts and inside the output
# directory in embedded ones. Enumerated over producer captures; no recursive guesses.
_DIRECTORIES = ("otel", "artifacts/otel", "results/otel", "*/otel", "*/artifacts/otel")
_TELEMETRY_READ_COUNT = len(_DIRECTORIES) * 2
GYM_TELEMETRY = Input(
    name="gym_telemetry",
    via=gym_telemetry,
    reads=tuple(
        [
            At(ROOT, f"{directory}/{signal}.jsonl")
            for directory in _DIRECTORIES
            for signal in ("traces", "metrics")
        ]
        + [
            At(ROOT, f"{directory}*.jsonl")
            for directory in ("", "artifacts/", "results/", "*/artifacts/")
        ]
    ),
    answers={
        "run.telemetry": Card.ANY(because="collector sidecars are opt-in and may be empty"),
        "run.source_run_id": Card.ANY(
            because="only a unique producer resource run_id identifies this run"
        ),
        "rollout.events": Card.ANY(
            because="only observations with verified selected-attempt ownership are rollout events"
        ),
    },
    mints=(
        Key(
            "telemetry_span",
            optional=True,
            because="traceId/spanId identifies an assigned span; metrics mint no event identity",
        ),
    ),
    names=(
        Key(
            "telemetry_parent",
            optional=True,
            because="a verified parent in another telemetry lane is a stated cross-lane edge",
        ),
        Key(
            "served_request",
            optional=True,
            because="an engine request ID may identify a captured model response",
        ),
    ),
)

TELEMETRY_LANES = tuple(
    Lane(
        kind=kind,
        order_by="read_seq",
        identity="source_event_id",
        because=(
            "collector timing is an observation about a rollout, independent of its "
            "conversation order"
        ),
    )
    for kind in ("otel", "vllm")
)
TELEMETRY_ATTACH = (
    Contribute(
        to="run.telemetry",
        input="gym_telemetry",
        order_by="key",
        identity="key",
        because=(
            "the complete typed observation identifies an export; repeat exports add "
            "provenance, not points"
        ),
    ),
    Contribute(
        to="rollout.events",
        input="gym_telemetry",
        order_by="read_seq",
        identity="source_event_id",
        because=(
            "a trace/span identity deduplicates exports without merging separate requests or "
            "attempts"
        ),
    ),
    Link(
        word="telemetry_parent",
        to="rollout.events",
        frm_input="gym_telemetry",
        to_input="gym_telemetry",
        on=Join("telemetry_span"),
        cardinality=Card.ONE,
        to_card=None,
        because=(
            "only same-trace parent pointers with agreeing selected-attempt ownership cross lanes"
        ),
    ),
    Link(
        word="served_request",
        to="rollout.events",
        frm_input="gym_telemetry",
        to_input="gym_inline_calls",
        on=Join("response_id"),
        cardinality=Card.ONE,
        to_card=None,
        because=(
            "where the adapter preserves upstream IDs, gen_ai.request.id names the captured "
            "response; rewritten IDs leave an explicit unresolved edge"
        ),
    ),
)
