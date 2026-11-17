# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Complete, outcome-blind evidence and audited reads for one judge session."""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from autofa.core.fields import DEFAULT_CONTENT_FIELDS, DEFAULT_METADATA_FIELDS, LABEL_BANNED
from autofa.data.trace_fields import derive_trace_fields

# Bump when projection/parser semantics change (including shared input/trajectory
# helpers or bounded-receipt completeness). Field allowlist and redaction changes
# are also fingerprinted below. This identity invalidates aggregate judgments as
# well as run-local checkpoints; revalidating a receipt alone cannot invalidate
# an already saved aggregate.
EVIDENCE_VERSION = "judge-evidence-v12-binary-review-projection"
MAX_READ_CHARS = 16_000
_OUTCOMES = (
    LABEL_BANNED
    | DEFAULT_METADATA_FIELDS
    | {
        "reward",
        "resolved",
        "is_resolved",
        "tb_is_resolved",
        "correct",
        "final_metrics",
        "metrics",
        "grading",
        "verification",
        "verifier_artifacts",
    }
)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _text(value: Any) -> str:
    return value if isinstance(value, str) else _json(value)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def evidence_projection_hash() -> str:
    """Identify the evidence policy without loading source files or run data.

    Shared detector field policy can expand independently of this module. Such
    changes must invalidate saved reviews even when their claims are unchanged.
    Parsing and role/tool-result interpretation remain versioned explicitly.
    """
    from autofa.data.evidence_refs import EVIDENCE_REFS_VERSION
    from autofa.metrics.judge_projection import PROJECTION_VERSION

    return _hash(_json({
        "version": EVIDENCE_VERSION,
        "evidence_refs_version": EVIDENCE_REFS_VERSION,
        "binary_projection_version": PROJECTION_VERSION,
        "content_fields": sorted(DEFAULT_CONTENT_FIELDS),
        "outcome_fields": sorted(_OUTCOMES),
        "artifact_transform_version": _artifact_transform_version(),
    }))


def _artifact_transform_version() -> str:
    from autofa.data.artifact_content import ARTIFACT_TRANSFORM_VERSION

    return ARTIFACT_TRANSFORM_VERSION


def _artifact_manifest(item: dict[str, Any]) -> dict[str, Any]:
    from autofa import trace
    from autofa.data.artifact_content import artifact_inspection_policy
    from autofa.data.execution_policy import is_evaluation_record, sanitize_metadata
    from autofa.data.inline_images import image_inventory

    catalog = [sanitize_metadata(row) for row in trace.artifacts(item)
               if not is_evaluation_record(row)]
    for row in catalog:
        policy = artifact_inspection_policy(row)
        if policy is not None:
            # Reader upgrades invalidate only reviews containing affected formats.
            row["inspection_policy"] = policy
    result = {"catalog": catalog, "transform_version": _artifact_transform_version()}
    _, limitations = image_inventory(item)
    if limitations:
        result["input_image_inventory_complete"] = False
        result["limitations"] = limitations
    return result


def _content_value(value: Any, redactions: list[str], field: str) -> Any:
    """Filter known envelope metadata, never words inside messages/tool arguments."""
    encoded = isinstance(value, str)
    if encoded:
        from autofa.data.structured import decode_container

        obj = decode_container(value)
    else:
        obj = value
    envelopes = obj if isinstance(obj, list) else [obj]
    changed = False
    cleaned = []
    for envelope in envelopes:
        if not isinstance(envelope, dict) or not (
            isinstance(envelope.get("steps"), list) or isinstance(envelope.get("execution"), dict)
        ):
            cleaned.append(envelope)
            continue
        # Redact fully reconstructed records before choosing new references. A
        # reference may otherwise point at content that filtering has changed.
        from autofa.data.evidence_refs import expand_trace

        try:
            expanded = expand_trace(envelope)
        except ValueError:
            # Unsupported references remain original evidence, never absent data.
            expanded = envelope
        changed = changed or expanded != envelope
        # Only envelope metadata is filtered. A tool argument named 'reward',
        # a model's success claim, or a user discussing grading remains evidence.
        safe = dict(expanded)
        for key in _OUTCOMES & safe.keys():
            safe.pop(key)
            redactions.append(f"{field}.{key}")
            changed = True
        metadata = safe.get("metadata")
        if isinstance(metadata, dict):
            safe["metadata"] = {k: v for k, v in metadata.items() if k not in _OUTCOMES}
            removed = _OUTCOMES & metadata.keys()
            redactions.extend(f"{field}.metadata.{k}" for k in sorted(removed))
            changed = changed or bool(removed)
        execution = safe.get("execution")
        if isinstance(execution, dict):
            from autofa.data.execution_policy import sanitize_metadata

            safe["execution"] = sanitize_metadata(execution, rollout=True)
            if safe["execution"] != execution:
                redactions.append(f"{field}.execution")
                changed = True
        cleaned.append(safe)
    if not changed:
        return value
    result = cleaned if isinstance(obj, list) else cleaned[0]
    return _json(result) if encoded else result


def _compact_content(value: Any) -> Any:
    """Remove verified source mirrors while retaining every unique stored value."""
    from autofa.data.evidence_refs import compact_trace
    from autofa.data.structured import decode_container

    encoded = isinstance(value, str)
    decoded = decode_container(value) if encoded else value
    wrapped = isinstance(decoded, list) and len(decoded) == 1
    envelope = decoded[0] if wrapped else decoded
    if not isinstance(envelope, dict):
        return value
    try:
        compact = compact_trace(envelope, include_rendered=True)
    except ValueError:
        return value  # Keep malformed/unknown references available for inspection.
    if compact == envelope:
        return value
    result = [compact] if wrapped else compact
    return _json(result) if encoded else result


def _source_position(navigation: Any, candidate: dict) -> tuple | None:
    """Recompute one source occurrence, including offsets within source-only parts."""
    locations = navigation.locate(candidate)
    if len(locations) != 1:
        return None
    location = locations[0]
    source = location.get("source_event")
    if not isinstance(source, dict):
        return None
    start, end = location.get("start"), location.get("end")
    source_only = "step_index" not in location
    if source_only:
        # Navigation exposes an external link for auxiliary lanes. Recover exact
        # decoded offsets from its verified raw span, without guessing by quote.
        field = candidate["field"]
        spans = [span for span in navigation.regions.get(field, [])
                 if "step_index" not in span and span.get("source_event") == source
                 and span.get("section") == location.get("section")
                 and span["start"] <= candidate["start"] < candidate["end"] <= span["end"]]
        if len(spans) != 1 or not isinstance(spans[0].get("text"), str):
            return None
        span = spans[0]
        text = navigation.fields[field]
        try:
            prefix = json.loads('"' + text[span["start"]:candidate["start"]] + '"')
            through = json.loads('"' + text[span["start"]:candidate["end"]] + '"')
            prefix.encode("utf-8")
            through.encode("utf-8")
        except ValueError:
            return None
        if not span["text"].startswith(through) or not through.startswith(prefix):
            return None
        start, end = len(prefix), len(through)
    if type(start) is not int or type(end) is not int or not 0 <= start < end:
        return None
    return (tuple(sorted(source.items())), location.get("section"), start, end, source_only)


def _source_candidates(navigation: Any, field: str, quote: str, digest: str, target: tuple):
    """Invert verified source offsets, without scanning every repeated quote."""
    source, section, start, end, source_only = target
    source, text = dict(source), navigation.fields[field]
    seen = set()
    for span in navigation.regions.get(field, []):
        if span.get("section") != section or ("step_index" not in span) != source_only:
            continue
        event = None if source_only else navigation.events[span["step_index"] - 1]
        identity = span.get("source_event") or (event or {}).get("source_event")
        if (not identity or any(identity.get(key) != source.get(key)
                                for key in ("lane", "event_idx"))
                or "part_idx" in identity and identity["part_idx"] != source.get("part_idx")):
            continue
        base = 0 if source_only else span.get("target_base")
        value = span.get("text")
        if base is None and isinstance(value, str) and event is not None:
            part = next(part for part in event["sections"] if part["key"] == section)
            if value == part["text"]:
                base = 0
            elif value.strip() == part["text"]:
                base = -(len(value) - len(value.lstrip()))
        begin = finish = None
        if (type(base) is int and isinstance(value, str) and span.get("json_value")
                and 0 <= start - base < end - base <= len(value)):
            # Compact snapshots use this JSON serializer. Verify its exact raw
            # prefixes as well, so future/other encodings cannot shift a hint.
            prefix = json.dumps(value[:start - base], ensure_ascii=False)[1:-1]
            through = json.dumps(value[:end - base], ensure_ascii=False)[1:-1]
            begin, finish = span["start"] + len(prefix), span["start"] + len(through)
            if (text[span["start"]:begin] != prefix or text[span["start"]:finish] != through
                    or text[begin:finish] != quote):
                begin = finish = None
        if begin is None:
            # Structured tool arguments may render with different whitespace.
            # Only a unique quote within this verified source span is a lead;
            # repeated, non-invertible content must not cause quadratic work.
            begin = text.find(quote, span["start"], span["end"])
            if begin < 0 or text.find(quote, begin + 1, span["end"]) >= 0:
                continue
            finish = begin + len(quote)
        if (begin, finish) in seen:
            continue
        proposed = {"field": field, "start": begin, "end": finish,
                    "quote": quote, "field_sha256": digest}
        if _source_position(navigation, proposed) == target:
            seen.add((begin, finish))
            yield proposed


def _rebase_compact_localisation(
    record: dict, item: dict, fields: dict[str, str], compacted_fields: set[str],
) -> dict:
    """Preserve the selected occurrence when source sharing changes raw offsets.

    The generic redaction fallback searches by quote. On duplicated source traces
    that can replace a late selected event with earlier, unrelated occurrences.
    Validate against the original field and match source identity plus offsets
    before the ordinary localisation limit is applied. Stored navigation is never
    trusted, and unverifiable/ambiguous hints are withheld rather than broadened.
    """
    from autofa.core.localisation import MAX_QUOTE_CHARS, passage
    from autofa.data.trace_navigation import TraceNavigation

    rebased, navigations, seen = [], {}, set()
    candidates = record.get("passages")
    for candidate in candidates if isinstance(candidates, list) else []:
        if not isinstance(candidate, dict):
            continue
        field = candidate.get("field")
        if not isinstance(field, str) or field not in compacted_fields:
            rebased.append(candidate)
            continue
        original, text = _text(item[field]), fields[field]
        if original == text:
            rebased.append(candidate)
            continue
        start, end, quote = (candidate.get(key) for key in ("start", "end", "quote"))
        if (type(start) is not int or type(end) is not int
                or not isinstance(quote, str) or not 0 < len(quote) <= MAX_QUOTE_CHARS
                or not 0 <= start < end <= len(original) or original[start:end] != quote
                or candidate.get("field_sha256") != _hash(original)):
            continue
        if field not in navigations:
            navigations[field] = (TraceNavigation({field: original}),
                                  TraceNavigation({field: text}), _hash(text))
        previous, current, digest = navigations[field]
        target = _source_position(previous, candidate)
        if target is None:
            continue
        matches = list(_source_candidates(current, field, quote, digest, target))
        # Original auxiliary tool/artifact locators can lack call/artifact IDs.
        # Multiple equal source spans therefore cannot identify the selected one.
        if not matches or (target[-1] and len(matches) != 1):
            continue
        chosen = matches[0]
        identity = (field, chosen["start"], chosen["end"])
        if identity not in seen:
            rebased.append(passage(field, text, chosen["start"], chosen["end"],
                                   method=candidate.get("method", "quote"), field_sha256=digest))
            seen.add(identity)
    return {**record, "passages": rebased}


def make_snapshot(
    item: dict[str, Any], *, item_id: str, source: str,
    failure_localisation: dict[str, Any] | None = None,
    claim_contract: dict[str, Any] | None = None,
    _include_structure: bool = True,
) -> dict[str, Any]:
    redactions: list[str] = []
    stored = {
        key: _content_value(value, redactions, key)
        for key, value in item.items()
        if key in DEFAULT_CONTENT_FIELDS and value is not None
    }
    # This is a lossless stored-evidence representation, not a derived summary.
    # Assign it before enrichment so whole-trace reviews still have to read it.
    compacted_fields = set()
    for field in ("response_text", "generation", "trace"):
        if field in stored:
            compact = _compact_content(stored[field])
            if compact != stored[field]:
                compacted_fields.add(field)
            stored[field] = compact
    # Derive from the filtered content, so no second route exposes envelope labels.
    from autofa.data.structured import decode_container

    derivation_input = dict(stored)
    if "input_parse" in stored:
        derivation_input["input_parse"] = decode_container(stored["input_parse"])
    enriched = derive_trace_fields(derivation_input)
    fields = {
        key: _text(value)
        for key, value in enriched.items()
        if key in DEFAULT_CONTENT_FIELDS and value is not None
    }
    # Derivation may not recognize a provider's original structured messages.
    # Keep that stored evidence rather than replacing it with an empty projection.
    for key in ("input_messages", "input_parse"):
        if key in stored:
            fields[key] = _text(stored[key])
    # A judge needs the customer's messages and the caller of each tool, too.
    # The detector-oriented action surface intentionally omits some dialogue.
    from autofa.data.input_view import response_value
    from autofa.metrics.judge_trajectory import judge_trajectory_text

    trajectory_text = judge_trajectory_text(response_value(stored))
    if trajectory_text:
        fields["trajectory_text"] = trajectory_text
    # Freeze artifact locators against retained source bytes before creating the
    # shorter text view. Both backends inspect these same original fields.
    from autofa.metrics.judge_projection import PROJECTION_VERSION, project_fields

    artifact_source_fields = dict(fields)
    artifact_manifest = _artifact_manifest(artifact_source_fields)
    fields, projection_maps = project_fields(fields, artifact_manifest["catalog"])
    from autofa.metrics.judge_structure import trace_structure

    structure = trace_structure(fields, set(stored)) if _include_structure else None
    if structure is not None:
        fields["trace_structure"] = _json(structure)
    fields["detector_source"] = source
    manifest = {
        key: {
            "total_chars": len(value),
            "sha256": _hash(value),
            "origin": "stored" if key in stored and artifact_source_fields.get(key, value)
            == _text(stored[key]) else "derived",
        }
        for key, value in fields.items()
    }
    manifest["detector_source"]["origin"] = "detector"
    limitations = []
    for key in ("response_text", "generation", "trace"):
        try:
            blob = json.loads(fields.get(key, ""))
        except ValueError:
            continue
        for envelope in blob if isinstance(blob, list) else [blob]:
            if isinstance(envelope, dict) and envelope.get("evidence_scope"):
                limitations.append(envelope["evidence_scope"])
            if isinstance(envelope, dict) and "_autofa_evidence_refs" in envelope:
                from autofa.data.evidence_refs import expand_trace

                try:
                    expand_trace(envelope)
                except ValueError:
                    limitations.append({
                        "field": key,
                        "limitation": "Invalid compact source references; original evidence "
                                      "is retained, but its projection cannot be trusted.",
                    })
    identity = {
        "version": EVIDENCE_VERSION, "projection_hash": evidence_projection_hash(),
        # The checked structure is an optional, reproducible index of existing
        # immutable source fields, not new evidence. Keep old snapshot/checkpoint
        # identities stable when adding it; its own version and source hash are
        # visible in the view. Saved whole-trace contracts still require all
        # original content, and old snapshots without the view remain readable.
        "item_id": item_id, "fields": {
            key: value for key, value in fields.items() if key != "trace_structure"
        },
    }
    if projection_maps:
        identity["review_projection"] = {"version": PROJECTION_VERSION,
                                         "fields": projection_maps}
    # Mixed-lane inventories were previously unavailable. Rechecking them adds
    # a usable evidence view, so an old "unavailable" judgment must not bypass
    # review through the completed-aggregate cache. Unaffected v1 inventories
    # retain their existing snapshot/checkpoint identities.
    if structure and structure.get("available") and structure.get("version", 1) >= 2:
        identity["structure_sha256"] = _hash(fields["trace_structure"])
    # Artifact source locators must refer to the same frozen representation used
    # by inspect_artifact/view_artifact, including derived input messages.
    if artifact_manifest["catalog"] or artifact_manifest.get("limitations"):
        identity["artifact_manifest"] = artifact_manifest
    if claim_contract is not None:
        identity["claim_contract"] = claim_contract
    optional_navigation = {}
    if (failure_localisation is not None
            and failure_localisation.get("item_id") in (None, "", item_id)):
        from autofa.core.localisation import for_judge

        record = failure_localisation
        if compacted_fields and record.get("source_sha256") == _hash(source):
            record = _rebase_compact_localisation(
                record, item, artifact_source_fields, compacted_fields,
            )
        if projection_maps:
            record = _rebase_projected_localisation(record, fields, projection_maps)
        localised = for_judge(record, fields, source)
        if localised is not None:
            # Optional quote retrieval must never change the evidence being
            # judged, or invalidate a completed review when highlights arrive.
            if ((record.get("deferred_localisation") or {}).get("status") == "pending"
                    and not localised["passages"]):
                localised["status"] = "deferred"
            optional_navigation = {
                "optional_failure_localisation": True,
                "failure_localisation": localised,
            }
    return {
        **identity,
        "snapshot_id": _hash(_json(identity)),
        **optional_navigation,
        "fields": fields,
        **({"artifact_source_fields": {key: artifact_source_fields[key]
                                     for key in projection_maps}} if projection_maps else {}),
        "manifest": manifest,
        "source_limitations": limitations,
        "redacted_metadata": sorted(redactions),
        "excluded_columns": sorted(set(item) - DEFAULT_CONTENT_FIELDS),
    }


def _rebase_projected_localisation(record: dict, fields: dict, maps: dict) -> dict:
    """Preserve exact verified text selections across binary payload replacement."""
    from autofa.metrics.judge_projection import projected_span

    passages = []
    candidates = record.get("passages")
    for candidate in candidates if isinstance(candidates, list) else []:
        if not isinstance(candidate, dict):
            continue
        field = candidate.get("field")
        mapping = maps.get(field)
        if not mapping:
            passages.append(candidate)
            continue
        start, end = candidate.get("start"), candidate.get("end")
        if (candidate.get("field_sha256") == _hash(fields[field])
                and type(start) is int and type(end) is int
                and 0 <= start < end <= len(fields[field])
                and fields[field][start:end] == candidate.get("quote")):
            passages.append(candidate)
            continue
        if (candidate.get("field_sha256") != mapping["source_sha256"]
                or type(start) is not int or type(end) is not int
                or not 0 <= start < end <= mapping["source_chars"]):
            continue
        span = projected_span(mapping, start, end)
        if span is not None and fields[field][span[0]:span[1]] == candidate.get("quote"):
            passages.append({**candidate, "start": span[0], "end": span[1],
                             "field_sha256": _hash(fields[field])})
    return {**record, "passages": passages}


class EvidenceReview:
    """A single rollout's immutable snapshot, read coverage, notes and audit log."""

    def __init__(self, snapshot: dict[str, Any], directory: Path):
        self.snapshot = snapshot
        self.fields = snapshot["fields"]
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "evidence.json").write_text(_json(snapshot), encoding="utf-8")
        self.epoch = 0
        self.reads: list[dict[str, Any]] = []
        self.citations: dict[str, dict[str, Any]] = {}
        self.notes = ""
        self.session: dict[str, Any] = {}
        self.artifact_pins: dict[int | tuple, dict[str, Any]] = {}
        self.artifact_reads: list[dict[str, Any]] = []
        self.artifact_inspections: dict[int | tuple, dict[str, Any]] = {}
        self.artifact_transcripts: dict[tuple, dict[str, Any]] = {}
        self.artifact_accesses: list[dict[str, Any]] = []
        self.intervals: dict[str, dict[str, Any]] = {}
        self._interval_navigation: dict[str, Any] | None = None
        self._supporting_passage_resolver = None
        self._supporting_passage_result = None

    @classmethod
    def restore(cls, snapshot: dict[str, Any], directory: Path) -> EvidenceReview:
        """Reconstruct a completed review for validation, without modifying its archive."""
        def substantive(value):
            value = dict(value)
            if value.pop("optional_failure_localisation", False):
                value.pop("failure_localisation", None)
            return value

        saved_snapshot = json.loads((directory / "evidence.json").read_text())
        if substantive(saved_snapshot) != substantive(snapshot):
            raise ValueError("saved evidence does not match the requested snapshot")
        review = cls.__new__(cls)
        review.snapshot, review.fields, review.directory = snapshot, snapshot["fields"], directory
        review.epoch, review.reads, review.citations = 0, [], {}
        review.notes, review.session = "", {}
        review.artifact_pins, review.artifact_reads, review.artifact_inspections = {}, [], {}
        review.artifact_transcripts = {}
        review.artifact_accesses = []
        review.intervals, review._interval_navigation = {}, None
        review._supporting_passage_resolver = None
        review._supporting_passage_result = None
        notes = directory / "review-notes.txt"
        if notes.is_file():
            review.notes = notes.read_text(encoding="utf-8")
        for line in (directory / "audit.jsonl").read_text().splitlines():
            row = json.loads(line)
            review.epoch = max(review.epoch, row["epoch"])
            if row["event"] == "read":
                review.reads.append({key: row[key] for key in ("field", "start", "end", "epoch")})
            elif row["event"] == "citation":
                review.citations[row["citation_id"]] = {
                    key: value for key, value in row.items() if key not in {"event", "epoch"}
                }
            elif row["event"] == "session_result":
                review.session = {key: value for key, value in row.items()
                                  if key not in {"event", "epoch", "attempt"}}
            elif row["event"] == "artifact_pin":
                key = review._artifact_key(row["artifact_idx"], row["pin"].get("member_path"))
                review.artifact_pins[key] = row["pin"]
            elif row["event"] == "artifact_access":
                receipt = row["receipt"]
                previous = next((old for old in review.artifact_accesses
                                 if review._artifact_access_key(old)
                                 == review._artifact_access_key(receipt)), None)
                if previous is not None:
                    review.artifact_accesses.remove(previous)
                review.artifact_accesses.append(receipt)
            elif row["event"] == "artifact_read":
                review.artifact_reads.append({key: value for key, value in row.items()
                                              if key != "event"})
            elif row["event"] == "review_interval":
                review.intervals[row["receipt"]["interval_id"]] = row["receipt"]
            elif row["event"] == "supporting_passages":
                review._supporting_passage_result = row["result"]
            elif row["event"] == "replacement_context":
                review.reads.clear()
                review.artifact_reads.clear()
        review._revalidate_artifacts()
        return review

    def record(self, event: str, **data: Any) -> None:
        with (self.directory / "audit.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(_json({"event": event, "epoch": self.epoch, **data}) + "\n")

    def compacted(self, metadata: dict[str, Any]) -> None:
        self.epoch += 1
        self.record("compact_boundary", metadata=metadata)

    def start_replacement_context(self, metadata: dict[str, Any]) -> None:
        """Start a fresh context without granting it a previous session's reads."""
        self.epoch += 1
        self.reads.clear()
        self.artifact_reads.clear()
        self.record("replacement_context", metadata=metadata)

    @property
    def artifact_source_fields(self) -> dict[str, str]:
        """Retained bytes for artifact readers; never a paged text-review surface."""
        return {**self.fields, **self.snapshot.get("artifact_source_fields", {})}

    def manifest(self) -> dict[str, Any]:
        manifest = {
            key: self.snapshot[key]
            for key in (
                "version",
                "projection_hash",
                "item_id",
                "snapshot_id",
                "manifest",
                "source_limitations",
                "redacted_metadata",
                "excluded_columns",
            )
        }
        if "failure_localisation" in self.snapshot:
            manifest["failure_localisation"] = self.snapshot["failure_localisation"]
        if "claim_contract" in self.snapshot:
            manifest["claim_contract"] = self.snapshot["claim_contract"]
        if "artifact_manifest" in self.snapshot:
            manifest["artifact_manifest"] = self.snapshot["artifact_manifest"]
        if "review_projection" in self.snapshot:
            manifest["review_projection"] = self.snapshot["review_projection"]
        if any(row.get("required_review_scope") == "bounded_interval"
               for row in self.snapshot.get("claim_contract", {}).get("requirements", [])):
            manifest["interval_navigation"] = self.interval_events()
        return manifest

    def supporting_passages(self) -> dict[str, Any]:
        """Retrieve optional navigation for this frozen item, never a new label."""
        from autofa.core.localisation import for_judge
        from autofa.util.cancellation import cancellation_guard, check_cancelled

        check_cancelled()
        if self._supporting_passage_result is None:
            if self._supporting_passage_resolver is not None:
                raw = self._supporting_passage_resolver()
                check_cancelled()
                if raw.get("item_id") not in (None, "", self.snapshot["item_id"]):
                    raise ValueError("Supporting passages belong to a different review item")
                raw = _rebase_projected_localisation(
                    raw, self.fields, self.snapshot.get("review_projection", {}).get("fields", {}),
                )
                result = for_judge(raw, self.fields, self.fields["detector_source"])
                if result is None:
                    raise ValueError("Supporting passages belong to a different detector source")
            else:
                result = self.snapshot.get("failure_localisation") or {
                    "status": "unavailable", "passages": [],
                    "reason": "No saved supporting passages are available. Read the "
                              "original evidence; do not rerun the detector to invent them.",
                }
            result = deepcopy(result)
            result["instructions"] = (
                "Optional detector navigation, not a verdict or an evidence read. Read the "
                "original fields and surrounding context with read_evidence; check "
                "counterevidence and use cite_evidence. Extraction does not reconstruct the "
                "classifier's reasoning. No passage does not mean no failure."
            )
            with cancellation_guard():
                self.record("supporting_passages", result=result)
                self._supporting_passage_result = result
        return deepcopy(self._supporting_passage_result)

    def interval_navigation(self) -> dict[str, Any]:
        from autofa.metrics.judge_intervals import interval_navigation

        if self._interval_navigation is None:
            self._interval_navigation = interval_navigation(self.snapshot)
        return deepcopy(self._interval_navigation)

    def interval_events(self, start_event: int = 1, limit: int = 20) -> dict[str, Any]:
        """Page original event locators without reading their contents."""
        navigation = self.interval_navigation()
        if not navigation["available"]:
            return navigation
        if (type(start_event) is not int or not 1 <= start_event <= navigation["event_count"]
                or type(limit) is not int or limit < 1):
            raise ValueError("start_event must be a valid 1-based index and limit positive")
        end = min(navigation["event_count"], start_event - 1 + min(limit, 40))
        return {**navigation, "events": navigation["events"][start_event - 1:end],
                "next_event": end + 1 if end < navigation["event_count"] else None,
                "note": "Locators do not mark evidence read. Page with interval_events; "
                        "read event ranges using read_evidence and cite the original field."}

    def _interval_receipt(
        self, requirement_id: str, start_event: int, end_event: int,
        start_citation_id: str = "", end_citation_id: str = "",
    ) -> dict[str, Any]:
        from autofa.metrics.judge_intervals import merge_ranges

        requirements = self.snapshot.get("claim_contract", {}).get("requirements", [])
        requirement = next((row for row in requirements if row["id"] == requirement_id), {})
        if requirement.get("required_review_scope") != "bounded_interval":
            raise ValueError("requirement_id must name a frozen bounded_interval clause")
        navigation = self.interval_navigation()
        if not navigation["available"]:
            raise ValueError(navigation["reason"])
        if (type(start_event) is not int or type(end_event) is not int
                or not 1 <= start_event <= end_event <= navigation["event_count"]):
            raise ValueError("interval events must be ordered 1-based interval_navigation indexes")
        selected = navigation["events"][start_event - 1:end_event]
        if any(not event["complete"] for event in selected):
            raise ValueError("interval contains incomplete or truncated source events")
        for side, index, citation_id in (
            ("start", start_event, start_citation_id), ("end", end_event, end_citation_id),
        ):
            boundary = requirement["review_interval"][side]
            if boundary["kind"] in {"run_start", "run_end"}:
                expected = 1 if boundary["kind"] == "run_start" else navigation["event_count"]
                if index != expected or citation_id:
                    raise ValueError(
                        f"{side} must use frozen {boundary['kind']} at event {expected}")
                continue
            citation = self.citations.get(citation_id) if isinstance(citation_id, str) else None
            event = navigation["events"][index - 1]
            if not citation or not any(
                citation.get("field") == span["field"]
                and type(citation.get("start")) is int and type(citation.get("end")) is int
                and span["start"] <= citation["start"] < citation["end"] <= span["end"]
                for span in event["ranges"]
            ):
                raise ValueError(
                    f"{side} citation must identify original evidence at event {index}")
            if not self.covers(citation["field"], citation["start"], citation["end"], current=True):
                raise ValueError(f"{side} boundary evidence was not read after latest compaction")
        payload = {
            "snapshot_id": self.snapshot["snapshot_id"],
            "claim_contract_id": self.snapshot["claim_contract"]["contract_id"],
            "requirement_id": requirement_id, "start_event": start_event, "end_event": end_event,
            "start_citation_id": start_citation_id, "end_citation_id": end_citation_id,
            "review_interval": deepcopy(requirement["review_interval"]),
            "required_ranges": merge_ranges([
                span for event in selected for span in event["ranges"]]),
        }
        return {**payload, "interval_id": "interval-" + _hash(_json(payload))}

    def review_interval(
        self, requirement_id: str, start_event: int, end_event: int,
        start_citation_id: str = "", end_citation_id: str = "",
    ) -> dict[str, Any]:
        """Bind frozen boundaries to original events; this does not mark any text read."""
        receipt = self._interval_receipt(
            requirement_id, start_event, end_event, start_citation_id, end_citation_id,
        )
        if receipt["interval_id"] not in self.intervals:
            self.intervals[receipt["interval_id"]] = receipt
            self.record("review_interval", receipt=receipt)
        return {**deepcopy(receipt), "complete": self._interval_complete(receipt),
                "note": "Read every required_range using read_evidence. Entire boundary events "
                        "are included: assess before/after within each batch in recorded order. "
                        "Binding verifies source location, not the semantic boundary predicate. "
                        "Recovery and failure contribution still require separate evidence."}

    def _interval_complete(self, receipt: dict[str, Any]) -> bool:
        return all(self.covers(span["field"], span["start"], span["end"])
                   for span in receipt["required_ranges"])

    def _validate_intervals(self, verdict: dict[str, Any]) -> str | None:
        requirements = {row["id"]: row for row in
                        self.snapshot.get("claim_contract", {}).get("requirements", [])}
        for assessment in verdict.get("claim_assessments", []):
            requirement_id = assessment.get("requirement_id")
            requirement = requirements.get(requirement_id, {})
            if (requirement.get("required_review_scope") != "bounded_interval"
                    or assessment.get("status") != "supported"):
                continue
            interval_id = assessment.get("review_interval_id")
            receipt = self.intervals.get(interval_id) if isinstance(interval_id, str) else None
            if not receipt or receipt["requirement_id"] != requirement_id:
                return (f"supported bounded clause {requirement_id} "
                        "needs an issued review_interval_id")
            try:
                expected = self._interval_receipt(**{key: receipt[key] for key in (
                    "requirement_id", "start_event", "end_event",
                    "start_citation_id", "end_citation_id",
                )})
            except ValueError as exc:
                return str(exc)
            if receipt != expected:
                return "bounded interval receipt differs from the frozen evidence or claim"
            refs = assessment.get("citations", [])
            cited = {ref.get("citation_id") for ref in refs if isinstance(ref, dict)}
            if any(receipt[key] and receipt[key] not in cited
                   for key in ("start_citation_id", "end_citation_id")):
                return f"bounded clause {requirement_id} must cite its selected boundary events"
            for span in receipt["required_ranges"]:
                if not self.covers(span["field"], span["start"], span["end"]):
                    return (f"bounded clause {requirement_id} has unread interval content in "
                            f"{span['field']} [{span['start']}, {span['end']}); "
                            "read every required range or mark the clause unknown")
        return None

    def _artifact_catalog(self) -> list[dict[str, Any]]:
        return self.snapshot.get("artifact_manifest", {}).get("catalog", [])

    @staticmethod
    def _artifact_key(artifact_idx: int, member_path: str | None = None):
        return artifact_idx if member_path is None else (artifact_idx, member_path)

    @staticmethod
    def _member_options(member_path: str | None) -> dict:
        return {"member_path": member_path} if member_path is not None else {}

    def prepare_artifact_support(
        self, capability: str, install: bool = False, model: str = "base",
    ) -> dict:
        """Prepare a supported local reader; setup itself does not inspect evidence."""
        from autofa.data.artifact_dependencies import prepare_artifact_support

        result = prepare_artifact_support(capability, install=install, model=model)
        self.record("artifact_support", capability=capability, install=install,
                    status=result.get("status"))
        return result

    def list_artifacts(self) -> dict[str, Any]:
        """List frozen catalog descriptors; listing does not inspect file contents."""
        return {"snapshot_id": self.snapshot["snapshot_id"],
                "artifacts": deepcopy(self._artifact_catalog()),
                "pins": deepcopy(list(self.artifact_pins.values())),
                "note": "Unretrieved, unsupported and unread artifact content remains unknown."}

    def _artifact_allowed(self, artifact_idx: int) -> None:
        if type(artifact_idx) is not int or sum(
            row.get("artifact_idx") == artifact_idx for row in self._artifact_catalog()
        ) != 1:
            raise ValueError("artifact_idx must identify exactly one artifact in this snapshot")

    def _pin_artifact(self, artifact_idx: int, evidence: dict[str, Any], *, image=False) -> None:
        if evidence.get("verified") is not True or not isinstance(
            evidence.get("source_identity"), dict,
        ):
            raise ValueError("artifact evidence has no verified source identity")
        digest = (evidence.get("artifact_sha256", evidence.get("original_sha256"))
                  if image else evidence.get("sha256"))
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("artifact evidence has no verified content hash")
        version = evidence.get("transform_version")
        if version != self.snapshot["artifact_manifest"]["transform_version"]:
            raise ValueError("artifact transform version changed; start a new evidence snapshot")
        row = next(row for row in self._artifact_catalog()
                   if row.get("artifact_idx") == artifact_idx)
        declared = row.get("sha256")
        member_path = evidence.get("member_path")
        catalog_digest = evidence.get("archive_sha256") if member_path is not None else digest
        if isinstance(declared, str) and declared and declared.lower() != catalog_digest:
            raise ValueError("retrieved artifact hash differs from the frozen source catalog")
        if member_path is not None:
            # Pin the archive as well as the selected member. Different members
            # must come from the same original archive throughout this review.
            parent = evidence["source_identity"].get("archive_source_identity")
            self._pin_artifact(artifact_idx, {
                "verified": True, "sha256": catalog_digest,
                "source_identity": parent, "transform_version": version,
            })
        pin = {"artifact_idx": artifact_idx, "sha256": digest,
               "source_identity": evidence.get("source_identity"), "transform_version": version,
               **self._member_options(member_path)}
        key = self._artifact_key(artifact_idx, member_path)
        prior = self.artifact_pins.get(key)
        if prior is not None and pin != prior:
            raise ValueError("artifact bytes or source identity changed during this review")
        if prior is None:
            self.artifact_pins[key] = pin
            self.record("artifact_pin", artifact_idx=artifact_idx, pin=pin)

    @staticmethod
    def _artifact_fingerprint(evidence: dict[str, Any]) -> str:
        # Cache paths are machine-specific; the actual evidence and hashes are not.
        return _hash(_json({key: value for key, value in evidence.items()
                            if key not in {"path", "download_path", "data_url"}}))

    @staticmethod
    def _artifact_access_key(receipt: dict) -> tuple:
        return (receipt["artifact_idx"], receipt.get("member_path"), receipt.get("page"),
                _json(receipt.get("transcription")))

    def _track_artifact_access(
        self, artifact_idx: int, evidence: dict[str, Any], *, page: int | None = None,
        transcription: dict | None = None,
    ) -> None:
        receipt = {"artifact_idx": artifact_idx, "page": page,
                   **self._member_options(evidence.get("member_path")),
                   **({"transcription": transcription} if transcription is not None else {}),
                   "available": evidence.get("status") == "available",
                   "fingerprint": self._artifact_fingerprint(evidence)}
        previous = next((old for old in self.artifact_accesses
                         if all(old.get(key) == receipt.get(key) for key in
                                ("artifact_idx", "page", "member_path", "transcription"))), None)
        if previous is not None and previous["fingerprint"] == receipt["fingerprint"]:
            return
        if previous is not None and previous != receipt:
            # A missing reader can be installed on demand before content is read.
            # Successful evidence remains frozen; setup cannot replace cited evidence.
            matching = [read for read in self.artifact_reads
                        if read["artifact_idx"] == artifact_idx
                        and read.get("member_path") == receipt.get("member_path")]
            if page is not None:
                matching = [read for read in matching
                            if read["kind"] == "image" and read.get("page") == page]
            elif transcription is not None:
                matching = [read for read in matching
                            if read.get("transcription") == transcription]
            else:
                matching = [read for read in matching if read["kind"] == "text"
                            and not read["locator"].startswith("speech:")]
            unread = not matching
            if previous.get("available") is False and receipt["available"] and unread:
                self.artifact_accesses.remove(previous)
                previous = None
            else:
                raise ValueError("artifact evidence or availability changed during this review")
        if previous is None:
            self.artifact_accesses.append(receipt)
            self.record("artifact_access", receipt=receipt)

    def _inspect_artifact(
        self, artifact_idx: int, member_path: str | None = None,
    ) -> dict[str, Any]:
        from autofa import trace

        self._artifact_allowed(artifact_idx)
        result = trace.artifact(self.artifact_source_fields, artifact_idx,
                                **self._member_options(member_path))
        key = self._artifact_key(artifact_idx, member_path)
        self._track_artifact_access(artifact_idx, result)
        if result.get("sha256") and result.get("verified") is True:
            self._pin_artifact(artifact_idx, result)
            self.artifact_inspections[key] = result
        if result.get("status") != "available":
            return result
        self._pin_artifact(artifact_idx, result)
        self.artifact_inspections[key] = result
        return result

    def inspect_artifact(
        self, artifact_idx: int, member_path: str | None = None,
    ) -> dict[str, Any]:
        """Return segment locators/lengths without counting their unshown text as read."""
        result = self._inspect_artifact(artifact_idx, member_path)
        summary = {key: deepcopy(value) for key, value in result.items() if key != "segments"}
        summary["segments"] = [{key: deepcopy(value) for key, value in segment.items()
                                if key != "text"} | {"total_chars": len(segment.get("text", "")),
                                "text_sha256": _hash(segment.get("text", ""))}
                               for segment in result.get("segments", [])]
        return summary

    def _artifact_segment(
        self, artifact_idx: int, locator: str, member_path: str | None = None,
    ) -> tuple[dict, str]:
        if locator.startswith("speech:"):
            matches = [(result, row["text"]) for key, result in self.artifact_transcripts.items()
                       if key[:2] == (artifact_idx, member_path)
                       for row in result.get("segments", []) if row.get("locator") == locator]
            if len(matches) != 1:
                raise ValueError("transcript locator unavailable; transcribe_artifact first")
            return matches[0]
        result = self._inspect_artifact(artifact_idx, member_path)
        if result.get("verified") is not True or not result.get("sha256"):
            raise ValueError("verified artifact text is unavailable; inspect_artifact for limits")
        matches = [segment for segment in result.get("segments", [])
                   if segment.get("locator") == locator]
        if len(matches) != 1 or not isinstance(matches[0].get("text"), str):
            raise ValueError("artifact locator is unavailable or ambiguous; inspect_artifact first")
        return result, matches[0]["text"]

    def read_artifact(
        self, artifact_idx: int, locator: str, offset: int = 0, limit: int = MAX_READ_CHARS,
        member_path: str | None = None,
    ) -> dict[str, Any]:
        """Read an exact text segment while pinning its artifact and transform hashes."""
        if type(offset) is not int or offset < 0 or type(limit) is not int or limit < 1:
            raise ValueError("offset must be nonnegative and limit a positive integer")
        result, text = self._artifact_segment(artifact_idx, locator, member_path)
        if offset > len(text):
            raise ValueError("offset exceeds artifact segment length")
        end = min(len(text), offset + min(limit, MAX_READ_CHARS))
        receipt = {"kind": "text", "artifact_idx": artifact_idx, "locator": locator,
                   "artifact_sha256": result["sha256"], "sha256": _hash(text),
                   "transform_version": result["transform_version"],
                   "source_identity": result.get("source_identity"),
                   **self._member_options(member_path),
                   "start": offset, "end": end, "epoch": self.epoch}
        self.artifact_reads.append(receipt)
        self.record("artifact_read", **{k: v for k, v in receipt.items() if k != "epoch"})
        return {"snapshot_id": self.snapshot["snapshot_id"], **receipt, "text": text[offset:end],
                "total_chars": len(text), "next_offset": end if end < len(text) else None,
                "is_final": end == len(text), "limitations": result.get("limitations", []),
                "status": result.get("status"), "complete": result.get("complete", False)}

    def view_artifact(
        self, artifact_idx: int, page: int = 1, member_path: str | None = None,
    ) -> dict[str, Any]:
        """Return the actual rendered image; the MCP adapter delivers its pixels to the judge."""
        from autofa import trace

        self._artifact_allowed(artifact_idx)
        result = trace.artifact_image(self.artifact_source_fields, artifact_idx, page=page,
                                      **self._member_options(member_path))
        self._track_artifact_access(artifact_idx, result, page=page)
        if result.get("artifact_sha256") and result.get("verified") is True:
            self._pin_artifact(artifact_idx, result, image=True)
        if result.get("status") != "available":
            return result
        self._pin_artifact(artifact_idx, result, image=True)
        import base64

        url = result.get("data_url", "")
        if not isinstance(url, str) or not url.startswith("data:image/") or ";base64," not in url:
            raise ValueError("artifact renderer returned no image content")
        encoded = url.split(";base64,", 1)[1]
        try:
            image_bytes = base64.b64decode(encoded, validate=True)
        except ValueError as exc:
            raise ValueError("invalid rendered image encoding") from exc
        if hashlib.sha256(image_bytes).hexdigest() != result.get("sha256"):
            raise ValueError("rendered artifact image does not match its content hash")
        locator = result.get("locator")
        if not isinstance(locator, str) or not locator:
            raise ValueError("rendered artifact has no source locator")
        # Viewing a retained image is sufficient visual inspection without a
        # separate metadata-only tool call. Inventory the source so whole-trace
        # checks know its required pages/text. This grants no text-read credit:
        # document segments and any other visual pages still require reads.
        if self._artifact_key(artifact_idx, member_path) not in self.artifact_inspections:
            self._inspect_artifact(artifact_idx, member_path)
        receipt = {"kind": "image", "artifact_idx": artifact_idx, "locator": locator,
                   "page": page, "artifact_sha256": result["artifact_sha256"],
                   **self._member_options(member_path),
                   "sha256": result["sha256"], "transform_version": result["transform_version"],
                   "source_identity": result.get("source_identity"), "epoch": self.epoch,
                   "page_count": result.get("page_count"),
                   "complete": result.get("complete", False),
                   "limitations": deepcopy(result.get("limitations", []))}
        self.artifact_reads.append(receipt)
        self.record("artifact_read", **{k: v for k, v in receipt.items() if k != "epoch"})
        return {**result, "receipt": receipt}

    def transcribe_artifact(
        self, artifact_idx: int, start_seconds: float = 0, end_seconds: float | None = None,
        model: str = "base", member_path: str | None = None,
    ) -> dict[str, Any]:
        """Read a bounded local speech estimate with source/model/timestamp provenance."""
        from autofa import trace
        from autofa.data.artifact_audio import _window

        self._artifact_allowed(artifact_idx)
        start_seconds, end_seconds = _window(start_seconds, end_seconds)
        request = {"start_seconds": start_seconds, "end_seconds": end_seconds, "model": model}
        result = trace.artifact_transcript(self.artifact_source_fields, artifact_idx, **request,
                                            **self._member_options(member_path))
        self._track_artifact_access(artifact_idx, result, transcription=request)
        if result.get("status") != "available":
            return result
        self._pin_artifact(artifact_idx, result)
        key = (artifact_idx, member_path, _json(request))
        self.artifact_transcripts[key] = result
        for segment in result.get("segments", []):
            receipt = {"kind": "text", "artifact_idx": artifact_idx,
                       "locator": segment["locator"], "artifact_sha256": result["sha256"],
                       "sha256": _hash(segment["text"]), "start": 0, "end": len(segment["text"]),
                       "source_identity": result["source_identity"], "epoch": self.epoch,
                       "transform_version": result["transform_version"],
                       "transcription": request, **self._member_options(member_path)}
            self.artifact_reads.append(receipt)
            self.record("artifact_read", **{k: v for k, v in receipt.items() if k != "epoch"})
        return result

    def cite_artifact(
        self, artifact_idx: int, locator: str, quote: str = "", offset: int = 0,
        member_path: str | None = None,
    ) -> dict[str, Any]:
        """Issue a citation for exact read text or a viewed image locator."""
        self._artifact_allowed(artifact_idx)
        if not isinstance(locator, str) or not locator or not isinstance(quote, str):
            raise ValueError("locator and quote must be strings, with a nonempty locator")
        reads = [read for read in self.artifact_reads if read["artifact_idx"] == artifact_idx
                 and read.get("member_path") == member_path
                 and read["locator"] == locator and read["epoch"] == self.epoch]
        if quote:
            if len(quote) > 2000 or type(offset) is not int or offset < 0:
                raise ValueError("invalid artifact quote or offset")
            _, text = self._artifact_segment(artifact_idx, locator, member_path)
            start = text.find(quote, offset)
            if start < 0 or not self._artifact_covers(reads, start, start + len(quote)):
                raise ValueError("exact artifact quote was not read after the latest compaction")
            read = next(read for read in reads if read["kind"] == "text")
            citation = {**read, "start": start, "end": start + len(quote), "quote": quote}
        else:
            images = [read for read in reads if read["kind"] == "image"]
            if not images:
                raise ValueError("artifact image was not viewed after the latest compaction")
            citation = dict(images[-1])
        citation.pop("epoch", None)
        source_link = self._artifact_source_link(citation)
        if source_link:
            citation["source_link"] = source_link
        citation["citation_id"] = f"cite-{len(self.citations) + 1}"
        self.citations[citation["citation_id"]] = citation
        self.record("citation", **citation)
        return deepcopy(citation)

    @staticmethod
    def _artifact_source_link(citation: dict[str, Any]) -> str | None:
        """Use the supported download route; page/sheet locators remain explicit metadata."""
        from urllib.parse import quote, urlsplit

        identity = citation.get("source_identity") or {}
        if identity.get("kind") == "archive_member":
            identity = identity.get("archive_source_identity") or {}
        base = identity.get("api_base_url")
        if not isinstance(base, str):
            return None
        parsed = urlsplit(base)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username
                or parsed.password or parsed.query or parsed.fragment):
            return None
        keys = ("run_id", "task_id", "repeat", "artifact_idx")
        if any(identity.get(key) is None for key in keys):
            return None
        run_id, task_id, repeat, index = (quote(str(identity[key]), safe="") for key in keys)
        rollout = f"{base.rstrip('/')}/v1/rollouts/{run_id}/{task_id}/{repeat}"
        return f"{rollout}/artifacts/{index}/download"

    @staticmethod
    def _artifact_covers(reads: list[dict], start: int, end: int) -> bool:
        current = start
        for row in sorted((r for r in reads if r["kind"] == "text"), key=lambda r: r["start"]):
            if row["start"] <= current:
                current = max(current, row["end"])
        return current >= end

    def artifact_evidence_identity(self) -> str:
        """Bind actual retrieved bytes/transforms separately from the immutable catalog snapshot."""
        return _hash(_json({"snapshot_id": self.snapshot["snapshot_id"],
                            "pins": sorted(self.artifact_pins.values(),
                                           key=lambda p: (p["artifact_idx"],
                                                          p.get("member_path", ""))),
                            "accesses": self.artifact_accesses,
                            "reads": [{k: v for k, v in r.items() if k != "epoch"}
                                      for r in self.artifact_reads]}))

    def _revalidate_artifacts(self) -> None:
        """Re-resolve every used source and transform before trusting a saved verdict."""
        from autofa import trace

        for access in self.artifact_accesses:
            options = self._member_options(access.get("member_path"))
            key = self._artifact_key(access["artifact_idx"], access.get("member_path"))
            if access.get("transcription") is not None:
                current = trace.artifact_transcript(
                    self.artifact_source_fields, access["artifact_idx"],
                    **access["transcription"], **options,
                )
                transcript_key = (access["artifact_idx"], access.get("member_path"),
                                  _json(access["transcription"]))
                self.artifact_transcripts[transcript_key] = current
            elif access["page"] is None:
                current = trace.artifact(self.artifact_source_fields, access["artifact_idx"],
                                         **options)
                self.artifact_inspections[key] = current
            else:
                current = trace.artifact_image(
                    self.artifact_source_fields, access["artifact_idx"],
                    page=access["page"], **options,
                )
            if self._artifact_fingerprint(current) != access["fingerprint"]:
                raise ValueError("saved artifact evidence or availability is stale")
        for key, pin in list(self.artifact_pins.items()):
            artifact_idx = pin["artifact_idx"]
            result = trace.artifact(self.artifact_source_fields, artifact_idx,
                                    **self._member_options(pin.get("member_path")))
            self._pin_artifact(artifact_idx, result)
            if result.get("sha256") != pin["sha256"]:
                raise ValueError("saved artifact bytes are stale")
            self.artifact_inspections[key] = result
        for read in self.artifact_reads:
            if read["kind"] == "image":
                current = trace.artifact_image(self.artifact_source_fields, read["artifact_idx"],
                                               page=read["page"],
                                               **self._member_options(read.get("member_path")))
                self._pin_artifact(read["artifact_idx"], current, image=True)
                digest = current.get("sha256")
            else:
                _, text = self._artifact_segment(read["artifact_idx"], read["locator"],
                                                  read.get("member_path"))
                digest = _hash(text)
            if digest != read["sha256"]:
                raise ValueError("saved artifact transformation is stale")

    def _validate_artifact_citation(self, citation: dict[str, Any]) -> str | None:
        identifier = citation.get("citation_id")
        issued = self.citations.get(identifier) if isinstance(identifier, str) else None
        if issued != citation:
            return "artifact citation was not issued by cite_artifact"
        reads = [read for read in self.artifact_reads
                 if all(read.get(key) == citation.get(key) for key in (
                     "kind", "artifact_idx", "locator", "artifact_sha256", "sha256",
                     "transform_version", "source_identity",
                     "member_path",
                 )) and read["epoch"] == self.epoch]
        if not reads:
            return "cited artifact evidence was not read after the latest compaction"
        if citation["kind"] == "image":
            return None
        start, end = citation.get("start"), citation.get("end")
        if type(start) is not int or type(end) is not int or not 0 <= start < end:
            return "invalid artifact citation offsets"
        if not self._artifact_covers(reads, start, end):
            return "cited artifact text was not read after the latest compaction"
        try:
            _, text = self._artifact_segment(citation["artifact_idx"], citation["locator"],
                                              citation.get("member_path"))
        except (TypeError, ValueError) as exc:
            return str(exc)
        if citation.get("quote") != text[start:end] or _hash(text) != citation["sha256"]:
            return "artifact citation quote or transform hash does not match source"
        return None

    def _whole_artifact_error(self) -> str | None:
        manifest = self.snapshot.get("artifact_manifest", {})
        if manifest.get("input_image_inventory_complete") is False:
            return "whole-trace verdict has an incomplete retained-image inventory"
        for artifact in self._artifact_catalog():
            index = artifact.get("artifact_idx")
            inspection = self.artifact_inspections.get(index)
            if not inspection or inspection.get("complete") is not True:
                return f"whole-trace verdict has unread or unavailable artifact {index}"
            for segment in inspection.get("segments", []):
                reads = [read for read in self.artifact_reads if read["artifact_idx"] == index
                         and read["locator"] == segment.get("locator")]
                if not self._artifact_covers(reads, 0, len(segment.get("text", ""))):
                    locator = segment.get("locator")
                    return f"whole-trace verdict has unread artifact {index}:{locator}"
            if inspection.get("requires_visual_review"):
                views = [read for read in self.artifact_reads
                         if read["artifact_idx"] == index and read["kind"] == "image"]
                count = inspection.get("visual_pages")
                if type(count) is not int and views:
                    count = views[-1].get("page_count")
                if type(count) is not int or count < 1 or not set(range(1, count + 1)) <= {
                    view["page"] for view in views
                }:
                    return f"whole-trace verdict has unread or unavailable artifact visuals {index}"
                if any(view.get("complete") is not True or view.get("limitations")
                       for view in views):
                    return f"whole-trace verdict has incomplete artifact visual coverage {index}"
        return None

    def read(self, field: str, offset: int = 0, limit: int = MAX_READ_CHARS) -> dict[str, Any]:
        if field not in self.fields:
            raise ValueError(f"unknown field: {field}")
        if type(offset) is not int or offset < 0 or type(limit) is not int or limit < 1:
            raise ValueError("offset must be nonnegative and limit must be positive integers")
        text = self.fields[field]
        if offset > len(text):
            raise ValueError(f"offset exceeds field length {len(text)}")
        end = min(len(text), offset + min(limit, MAX_READ_CHARS))
        receipt = {"field": field, "start": offset, "end": end, "epoch": self.epoch}
        self.reads.append(receipt)
        self.record("read", **{k: v for k, v in receipt.items() if k != "epoch"})
        return {
            "snapshot_id": self.snapshot["snapshot_id"],
            **receipt,
            "text": text[offset:end],
            "total_chars": len(text),
            "next_offset": end if end < len(text) else None,
            "is_final": end == len(text),
        }

    def search(self, field: str, query: str, offset: int = 0, limit: int = 8) -> dict[str, Any]:
        if field not in self.fields or not query or len(query) > 1000:
            raise ValueError(
                "search needs a known field and a nonempty query of at most 1000 chars"
            )
        if type(offset) is not int or offset < 0 or type(limit) is not int or limit < 1:
            raise ValueError("invalid search offset/limit")
        # Literal, Unicode-aware matching keeps offsets in the original string.
        matches = []
        next_offset = None
        for match in re.finditer(re.escape(query), self.fields[field][offset:], re.IGNORECASE):
            start, end = offset + match.start(), offset + match.end()
            if len(matches) >= min(limit, 8):
                next_offset = start
                break
            context_start = max(0, start - 200)
            context = self.read(field, context_start, end + 200 - context_start)
            match_result = {"match_start": start, "match_end": end, "context": context}
            if field != "detector_source":
                try:
                    match_result["citation"] = self._register_citation(field, start, end)
                except ValueError as exc:
                    match_result["citation_unavailable"] = str(exc)
            matches.append(match_result)
        self.record("search", field=field, query=query, offset=offset, next_offset=next_offset)
        return {
            "matches": matches,
            "next_offset": next_offset,
            "note": "Search covers literal matches only; it is not a full semantic review.",
        }

    def _original_citation_span(self, field: str, start: int, end: int) -> dict | None:
        mapping = self.snapshot.get("review_projection", {}).get("fields", {}).get(field)
        if not mapping:
            return None
        from autofa.metrics.judge_projection import source_span

        span = source_span(mapping, start, end)
        if span is None:
            raise ValueError("artifact references are navigation, not source quotes; "
                             "view_artifact and cite_artifact for image evidence")
        return {"field": field, "start": span[0], "end": span[1],
                "field_sha256": mapping["source_sha256"]}

    def _register_citation(self, field: str, start: int, end: int) -> dict[str, Any]:
        citation_id = f"cite-{len(self.citations) + 1}"
        citation = {
            "citation_id": citation_id,
            "field": field,
            "start": start,
            "end": end,
            "quote": self.fields[field][start:end],
        }
        original = self._original_citation_span(field, start, end)
        if original is not None:
            citation["source_span"] = original
        self.citations[citation_id] = citation
        self.record("citation", **citation)
        return dict(citation)

    def cite(
        self, field: str, quote: str, offset: int = 0, limit: int = 8,
    ) -> dict[str, Any]:
        """Locate an exact quote only within evidence read in the current epoch.

        Repeated quotes return distinct references for the judge to disambiguate.
        This tool does not read new evidence or guess which occurrence was intended.
        """
        if not isinstance(field, str) or field not in self.fields or field == "detector_source":
            raise ValueError("cite needs an evidence field, not detector_source")
        if not isinstance(quote, str) or not quote or len(quote) > 2000:
            raise ValueError("cite needs an exact nonempty quote of at most 2000 characters")
        if type(offset) is not int or offset < 0 or type(limit) is not int or limit < 1:
            raise ValueError("invalid citation offset/limit")
        citations = []
        next_offset = None
        for start, end in self._ranges(field, current=True):
            position = self.fields[field].find(quote, max(start, offset), end)
            while position != -1:
                if len(citations) >= min(limit, 8):
                    next_offset = position
                    break
                citations.append(self._register_citation(field, position, position + len(quote)))
                position = self.fields[field].find(quote, position + 1, end)
            if next_offset is not None:
                break
        if not citations:
            raise ValueError(
                "exact quote not found in evidence read after the latest compaction; "
                "read the relevant passage, then copy its exact text"
            )
        return {
            "citations": citations,
            "next_offset": next_offset,
            "note": "Use the citation_id for the intended occurrence. If several match, "
            "inspect their context or cite a longer quote to disambiguate.",
        }

    def resolve_citations(self, verdict: dict[str, Any]) -> dict[str, Any]:
        """Expand issued IDs for validation and archival; retain explicit legacy spans."""
        citations = verdict.get("citations")
        if not isinstance(citations, list):
            return verdict
        resolved = []
        for citation in citations:
            if isinstance(citation, dict) and "citation_id" in citation:
                citation_id = citation["citation_id"]
                if not isinstance(citation_id, str) or citation_id not in self.citations:
                    raise ValueError("unknown citation ID; use cite_evidence or search_evidence")
                issued = self.citations[citation_id]
                if any(key in citation and citation[key] != value for key, value in issued.items()):
                    raise ValueError("citation ID conflicts with supplied source span")
                citation = dict(issued)
            resolved.append(citation)
        return {**verdict, "citations": resolved}

    def save_notes(self, notes: str) -> dict[str, Any]:
        if len(notes) > 12_000:
            raise ValueError("review notes exceed 12000 characters; preserve references concisely")
        self.notes = notes
        (self.directory / "review-notes.txt").write_text(notes, encoding="utf-8")
        self.record("save_notes", characters=len(notes))
        return {"saved": True, "characters": len(notes)}

    def status(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot["snapshot_id"],
            "claim_contract": self.snapshot.get("claim_contract"),
            "epoch": self.epoch,
            "notes": self.notes,
            "artifact_pins": deepcopy(list(self.artifact_pins.values())),
            "artifact_reads": deepcopy(self.artifact_reads),
            "artifact_accesses": deepcopy(self.artifact_accesses),
            "artifact_evidence_id": self.artifact_evidence_identity(),
            "review_intervals": [
                {**deepcopy(receipt), "complete": self._interval_complete(receipt),
                 "unread_ranges": self.unread_ranges(receipt["required_ranges"])}
                for receipt in self.intervals.values()
            ],
            "coverage": {
                key: {
                    "total_chars": len(value),
                    "read_ranges": self._ranges(key),
                    "current_ranges": self._ranges(key, current=True),
                    "unread_ranges": self.unread_ranges([
                        {"field": key, "start": 0, "end": len(value)}]),
                }
                for key, value in self.fields.items()
            },
        }

    def unread_ranges(self, spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Navigate unreviewed content without marking it read or shortening its scope."""
        missing = []
        for span in spans:
            cursor, end = span["start"], span["end"]
            for start_read, end_read in self._ranges(span["field"]):
                if end_read <= cursor:
                    continue
                if start_read >= end:
                    break
                if start_read > cursor:
                    missing.append({"field": span["field"], "start": cursor,
                                    "end": min(start_read, end)})
                cursor = max(cursor, end_read)
            if cursor < end:
                missing.append({"field": span["field"], "start": cursor, "end": end})
        return missing

    def completion_plan(self, assessments: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Reconcile unknown claims with audited reads of their frozen evidence scope.

        A model's ``evidence_unavailable`` or ``ambiguous_evidence`` label cannot
        waive an unread required interval. Local claims do not imply exhaustive
        reads: their explicit inspection acknowledgement identifies relevant ranges,
        or explains missing/ambiguous evidence. The original review budget still applies.
        """
        requirements = {row["id"]: row for row in
                        self.snapshot.get("claim_contract", {}).get("requirements", [])}
        pending = []
        for assessment in assessments:
            parts = [assessment, *assessment.get("components", [])]
            unknown = [part for part in parts if part.get("status") == "unknown"]
            if not unknown:
                continue
            explicitly_unfinished = any(
                part.get("unknown_reason") == "review_incomplete" for part in unknown)
            requirement_id = assessment["requirement_id"]
            scope = requirements.get(requirement_id, {}).get("required_review_scope", "local")
            task = {"requirement_id": requirement_id, "required_review_scope": scope}
            if scope == "whole_trace":
                task["unread_ranges"] = self.unread_ranges(self._whole_trace_ranges())
                task["artifact_gap"] = self._unfinished_artifact_review()
                if not task["unread_ranges"] and not task["artifact_gap"]:
                    continue
            elif scope == "bounded_interval":
                interval_id = assessment.get("review_interval_id")
                receipt = self.intervals.get(interval_id) if isinstance(interval_id, str) else None
                if receipt and receipt["requirement_id"] == requirement_id:
                    task["review_interval_id"] = receipt["interval_id"]
                    task["unread_ranges"] = self.unread_ranges(receipt["required_ranges"])
                    if not task["unread_ranges"]:
                        continue
                else:
                    navigation = self.interval_navigation()
                    if not navigation["available"]:
                        # An unsupported/truncated event inventory cannot be fixed
                        # by requesting the same impossible interval binding again.
                        continue
                    unread_events = [
                        event["event_index"] for event in navigation["events"]
                        if self.unread_ranges(event["ranges"])
                    ]
                    if not unread_events and not explicitly_unfinished:
                        continue
                    task["next_step"] = (
                        "Bind the frozen interval with review_interval first. Use "
                        "interval_events to locate its boundary events and inspect "
                        "the relevant unread event ranges; do not assume unavailable "
                        "evidence from an unbound interval or read unrelated events. "
                        "If the boundary remains ambiguous after inspection, retain "
                        "that ambiguity explicitly.")
                    task["unread_event_count"] = len(unread_events)
            else:
                inspections = [part.get("evidence_inspection") for part in unknown]
                missing_inspection = any(not isinstance(value, dict) for value in inspections)
                ranges = [span for value in inspections if isinstance(value, dict)
                          for span in value.get("ranges", [])]
                unread = self.unread_ranges(ranges)
                pending_inspection = any(
                    isinstance(value, dict) and value.get("status") == "pending"
                    for value in inspections)
                if not (explicitly_unfinished or missing_inspection
                        or unread or pending_inspection):
                    continue
                task.update(unread_ranges=unread,
                            inspection_acknowledgement_missing=missing_inspection)
                task["next_step"] = (
                    "Inspect the relevant retained evidence, preserving the local scope. "
                    "Return evidence_inspection with status, exact ranges and reason for "
                    "each unknown local assessment. Complete means inspected but ambiguous; "
                    "pending identifies available unread evidence; unavailable identifies "
                    "a missing source fact. Use review_status for read receipts. Do not "
                    "read unrelated fields or assume the answer must confirm the detector.")
            pending.append(task)
        return pending

    def _inspection_error(self, verdict: dict[str, Any]) -> str | None:
        """Validate declared local evidence ranges without inferring relevance from prose."""
        for assessment in verdict.get("claim_assessments", []):
            for part in [assessment, *assessment.get("components", [])]:
                inspection = part.get("evidence_inspection")
                if inspection is None:
                    continue  # Missing acknowledgement gets bounded completion, not rejection.
                if (not isinstance(inspection, dict)
                        or inspection.get("status") not in {"complete", "pending", "unavailable"}
                        or not isinstance(inspection.get("ranges"), list)
                        or not isinstance(inspection.get("reason"), str)
                        or not inspection["reason"].strip()):
                    return "evidence_inspection needs status, ranges and a specific reason"
                if part.get("status") != "unknown" and inspection["status"] != "complete":
                    return ("pending or unavailable evidence_inspection requires "
                            "an unknown assessment")
                for span in inspection["ranges"]:
                    if not isinstance(span, dict):
                        return "invalid evidence_inspection range"
                    field, start, end = (span.get(key) for key in ("field", "start", "end"))
                    if (not isinstance(field, str) or field not in self.fields
                            or field == "detector_source" or type(start) is not int
                            or type(end) is not int
                            or not 0 <= start < end <= len(self.fields[field])):
                        return "invalid evidence_inspection field or offsets"
        return None

    def reconcile_completion(self, assessments: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Keep unfinished scoped reads actionable in feedback, without changing truth."""
        tasks = {task["requirement_id"]: task for task in self.completion_plan(assessments)}
        if not tasks:
            return assessments
        result = deepcopy(assessments)
        for assessment in result:
            task = tasks.get(assessment["requirement_id"])
            if task is None:
                continue
            assessment["completion_evidence"] = task
            for part in [assessment, *assessment.get("components", [])]:
                if part.get("status") != "unknown":
                    continue
                if part.get("unknown_reason") != "review_incomplete":
                    part["original_unknown_reason"] = part.get("unknown_reason")
                    part["unknown_reason"] = "review_incomplete"
        return result

    def _unfinished_artifact_review(self) -> str | None:
        """Return retrievable/uninspected artifact work, not known missing source."""
        for artifact in self._artifact_catalog():
            index = artifact.get("artifact_idx")
            inspection = self.artifact_inspections.get(index)
            if inspection is None:
                return f"whole-trace verdict has unread or unavailable artifact {index}"
            if inspection.get("status") != "available":
                continue
            for segment in inspection.get("segments", []):
                reads = [read for read in self.artifact_reads if read["artifact_idx"] == index
                         and read["locator"] == segment.get("locator")]
                if not self._artifact_covers(reads, 0, len(segment.get("text", ""))):
                    locator = segment.get("locator")
                    return f"whole-trace verdict has unread artifact {index}:{locator}"
            if inspection.get("requires_visual_review"):
                views = [read for read in self.artifact_reads
                         if read["artifact_idx"] == index and read["kind"] == "image"]
                count = inspection.get("visual_pages")
                if type(count) is not int and views:
                    count = views[-1].get("page_count")
                if type(count) is int and count > 0 and not set(range(1, count + 1)) <= {
                    view["page"] for view in views
                }:
                    return f"whole-trace verdict has unread artifact visuals {index}"
        return None

    def _whole_trace_ranges(self) -> list[dict[str, Any]]:
        """Original stored evidence, without requiring duplicate derived views."""
        return [
            {"field": field, "start": 0, "end": len(value)}
            for field, value in self.fields.items()
            if self.snapshot["manifest"][field]["origin"] == "stored"
        ]

    def _ranges(self, field: str, *, current: bool = False) -> list[list[int]]:
        ranges: list[list[int]] = []
        for read in sorted(self.reads, key=lambda r: r["start"]):
            if read["field"] != field or current and read["epoch"] != self.epoch:
                continue
            if ranges and read["start"] <= ranges[-1][1]:
                ranges[-1][1] = max(ranges[-1][1], read["end"])
            else:
                ranges.append([read["start"], read["end"]])
        return ranges

    def covers(self, field: str, start: int, end: int, *, current: bool = False) -> bool:
        return start == end or any(
            a <= start and b >= end for a, b in self._ranges(field, current=current)
        )

    def validate(
        self, verdict: dict[str, Any], *, allow_no_citations: bool = False,
        allow_incomplete_coverage: bool = False,
    ) -> str | None:
        """Reject unsupported verdicts; validated citations do not prove semantic correctness."""
        if verdict.get("item_id") != self.snapshot["item_id"]:
            return "missing or mismatched item ID"
        if verdict.get("snapshot_id") != self.snapshot["snapshot_id"]:
            return "missing or mismatched evidence snapshot"
        if not self.covers("detector_source", 0, len(self.fields["detector_source"])):
            return "detector source was not fully read"
        scope = verdict.get("scope")
        if scope not in ("local", "bounded_interval", "whole_trace"):
            return "scope must be local, bounded_interval or whole_trace"
        contract = self.snapshot.get("claim_contract") or {}
        exhaustive_ids = {
            requirement["id"] for requirement in contract.get("requirements", [])
            if requirement.get("required_review_scope") == "whole_trace"
        }
        supported_exhaustive = [
            assessment["requirement_id"] for assessment in verdict.get("claim_assessments", [])
            if (isinstance(assessment, dict)
                and assessment.get("requirement_id") in exhaustive_ids
                and assessment.get("status") == "supported")
        ]
        if supported_exhaustive and scope != "whole_trace":
            return (
                "supported exhaustive clauses require scope=whole_trace: "
                + ", ".join(supported_exhaustive)
                + "; read all stored evidence, or mark unestablished clauses unknown"
            )
        interval_error = self._validate_intervals(verdict)
        if interval_error:
            return interval_error
        inspection_error = self._inspection_error(verdict)
        if inspection_error:
            return inspection_error
        citations = verdict.get("citations")
        if not isinstance(citations, list) or (not citations and not allow_no_citations):
            return "verdict has no source citations"
        for citation in citations:
            if not isinstance(citation, dict):
                return "invalid citation"
            if "artifact_idx" in citation:
                error = self._validate_artifact_citation(citation)
                if error:
                    return error
                continue
            field, start, end = (citation.get(k) for k in ("field", "start", "end"))
            if (
                not isinstance(field, str)
                or field not in self.fields
                or field == "detector_source"
                or type(start) is not int
                or type(end) is not int
                or not 0 <= start < end <= len(self.fields[field])
            ):
                return "invalid citation field or offsets"
            if citation.get("quote") != self.fields[field][start:end]:
                return "citation quote does not match source"
            try:
                original = self._original_citation_span(field, start, end)
            except ValueError as exc:
                return str(exc)
            if "source_span" in citation and citation["source_span"] != original:
                return "citation source span does not match retained source"
            if not self.covers(field, start, end, current=True):
                return "cited evidence was not read after the latest compaction"
        # A cited contradiction can refute an exhaustive claim locally even when
        # the reviewer selected whole_trace. Derive it from the frozen contract,
        # never the model's top-level verdict. Every asserted exhaustive support
        # still needs complete review, regardless of a contradiction elsewhere.
        contradicted = False
        if contract:
            from autofa.metrics.claim_contract import derive_claim_verdict

            try:
                contradicted = derive_claim_verdict(contract, verdict) is False
            except (ValueError, TypeError, KeyError):
                pass  # Invalid assessments never waive evidence coverage.
        if scope == "whole_trace" and (
            supported_exhaustive or not (allow_incomplete_coverage or contradicted)
        ):
            for span in self._whole_trace_ranges():
                if not self.covers(span["field"], span["start"], span["end"]):
                    return f"whole-trace verdict has unread content in {span['field']}"
            error = self._whole_artifact_error()
            if error:
                return error
        return None

    def initial_context(self, max_chars: int) -> str:
        """Inline complete small fields, keeping large evidence available by indexed reads."""
        manifest = self.manifest()
        inline_fields = {}
        if len(_json(self.fields)) <= max_chars and all(
            len(value) <= MAX_READ_CHARS for value in self.fields.values()
        ):
            inline_fields = self.fields
        else:
            # Every verdict requires the complete detector source. A large trace
            # should not force an extra retrieval (or verdict repair) for small
            # code that fits the same inline evidence budget. Never inline a
            # truncated source or record unread trace content as delivered.
            source = {"detector_source": self.fields["detector_source"]}
            if (len(_json(source)) <= max_chars
                    and len(source["detector_source"]) <= MAX_READ_CHARS):
                inline_fields = source
        if inline_fields:
            manifest["complete_fields"] = {key: self.read(key) for key in inline_fields}
        manifest["instructions"] = (
            "Fields in complete_fields are delivered in full and recorded as read. "
            "Use read_evidence for detector_source if not inlined and for relevant evidence. "
            "If locating the detector's supporting passages would help, call "
            "retrieve_supporting_passages for this example. This optional request may run "
            "a quote extractor; do not request it when the evidence is already clear. "
            "Returned passages are navigation only, not reads, citations or a new label. "
            "Use cite_evidence "
            "with an exact quote from a read passage to obtain citation_id references; "
            "do not calculate character offsets yourself. Preserve citations and "
            "counterevidence using save_review_notes. After compaction use review_status "
            "and reread cited passages. No evidence is absent merely because it is unread."
            " Compact source references identify duplicate representations; their original "
            "event/call records remain in the same stored field. Read those records and "
            "all unique content for whole-trace claims."
            " Whole-trace text coverage requires fields marked origin=stored in the manifest; "
            "derived views of those same fields do not require a second exhaustive read."
            " For bounded clauses use interval_navigation and review_interval to bind frozen "
            "boundaries to original event citations; read every required range, including "
            "whole command batches. A search result or interval binding is not a complete read."
            " Artifact catalogs describe files: use list_artifacts and inspect_artifact, "
            "then read_artifact/view_artifact for page/sheet/slide evidence and cite_artifact."
            " When present, trace_structure verifies original event identities, part counts "
            "and separate assistant-message counts without reading their text. Read and cite "
            "it for local structural facts only when available=true. It does not prove "
            "semantic absence, recovery, task termination or causation, and does not waive "
            "any frozen whole_trace or bounded_interval requirement."
        )
        if any(row.get("source_kind") == "retained_input_image"
               for row in self._artifact_catalog()):
            manifest["instructions"] += (
                " This example retains input images in the artifact catalog. For a claim that "
                "depends on a chart or other visual input, use view_artifact to inspect the actual "
                "image and cite_artifact for its verified citation. 'Retained but not inspected' "
                "means the image has not been viewed, not that it is missing. Reading encoded "
                "image data as text does not inspect its pixels. If image viewing is unavailable, "
                "report that limitation; do not infer visual facts from the model's description."
                " Binary payloads in review text use stable artifact:// references; the "
                "original bytes remain available to artifact readers. Text offsets and required "
                "coverage refer to this projected view; no encoded payload paging is required."
            )
        if self._artifact_catalog():
            manifest["instructions"] += (
                " Missing supported readers can be installed with prepare_artifact_support "
                "(capability='office'|'audio'|'speech', install=True). Speech setup also downloads "
                "the selected local model; evidence reads never download models automatically. "
                "For ZIP files, inspect the member inventory, then use member_path with the "
                "read/view/transcribe/cite tools. An archive listing is not member inspection. "
                "Audio views are 30-second waveform/spectrogram measurements, not listening. "
                "transcribe_artifact supplies timestamped speech estimates that can be wrong; "
                "cite their returned speech locators. Transcripts and signal plots do not "
                "establish music content, tone, or perceptual audio quality."
            )
        return _json(manifest)
