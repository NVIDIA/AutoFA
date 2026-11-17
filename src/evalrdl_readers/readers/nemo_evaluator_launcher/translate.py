# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The evaluator's record streams: its manifest, and each record's envelope.

    <bench>/eval-*.json            one or two run manifests
    <bench>/results.jsonl          one record per attempted problem and repeat
    <bench>/trajectories.jsonl     the same records with their trajectories

The envelope (task, repeat, reward, status, tokens, the scorer's logs) is the evaluator's. A
record's `trajectory` is ATIF documents: the evaluator's own where it wrapped an agent that writes
none, read here; the agent harness's otherwise, read by harbor's reader and joined to the record
by `record_ref`.
"""

from __future__ import annotations

import json

from evalrdl_readers.core import Answer, Decoder, Sources, Unreadable

from ..common import (
    as_float,
    dig,
    event_kind,
    field_artifact,
    has_value,
    record_ref,
    records,
    text_parts,
    without_reference_fields,
)
from ..dialects.atif import atif_documents, atif_rows, atif_steps, atif_totals, written_by_wrapper

#: The evaluator's single-turn record (`prompt`, `model_response`, no trajectory); declared so
#: such records are not `Unreadable`.
SINGLE_TURN = Decoder(
    "single-turn",
    decides=lambda rec: (
        not isinstance(rec.get("trajectory"), list)
        and bool(rec.get("prompt") or rec.get("model_response"))
    ),
    because="a record with no trajectory: the turns are `prompt` and `model_response`, one "
    "user message and one assistant message",
)


def _leaves(node, prefix=""):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _leaves(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _leaves(v, f"{prefix}[{i}]")
    else:
        yield prefix, node


def _differing_leaves(docs) -> list[str]:
    """Every leaf path on which two run manifests disagree."""
    maps = [dict(_leaves(d.json() or {})) for d in docs]
    keys = set().union(*maps) if maps else set()
    return sorted(k for k in keys if len({repr(m.get(k, "<absent>")) for m in maps}) > 1)


def nel_manifest(src: Sources):
    docs = sorted(src.docs(0), key=lambda d: d.name)
    if not docs:
        return
    chosen = docs[0]
    body = chosen.json() or {}
    note = None
    if len(docs) > 1:
        differing = _differing_leaves(docs)
        note = (
            f"{len(docs)} run manifests; kept {chosen.name} by lexical filename order. "
            f"They differ on {len(differing)} leaf fields: {differing}"
        )
    yield Answer("run.source_run_id", body.get("run_id"))
    # Verbatim, `@1.0` and all: the registry owns every spelling.
    yield Answer("run.benchmark", dig(body, "config", "benchmark"), note=note)
    yield Answer("run.model", dig(body, "config", "model"), note=note)
    yield Answer("run.started_at", body.get("timestamp"), note=note)
    yield Answer("run.n_expected", body.get("n_results"), note=note)
    yield Answer("run.n_manifests", len(docs))
    scores = dig(body, "benchmark", "scores", default={}) or {}
    pass1 = scores.get("pass@1")
    yield Answer(
        "run.score",
        as_float(pass1.get("value")) if isinstance(pass1, dict) else None,
        note=note,
    )


def _row_key(rec: dict, line: int) -> tuple[str]:
    idx = rec.get("problem_idx")
    rep = rec.get("repeat")
    if idx is None:
        return (f"line:{line}",)
    return (f"{idx}/{rep}" if rep is not None else str(idx),)


#: The producer's end status -> the store's `ok`/`failed`; the raw word is `rollout.stop_reason`.
EVAL_STATUS = {"passed": "ok", "failed": "failed", "runner_error": "failed"}


def nel_results(src: Sources):
    for item in records(src.doc(0), "rollout.score"):
        if isinstance(item, Unreadable):
            yield item
            continue
        line, rec = item
        key = _row_key(rec, line)
        ref = record_ref(src.doc(0), line)
        yield Answer("rollout.results_record", ref, key=key, keys={"results_record": ref})
        yield Answer("rollout.score", as_float(rec.get("reward")), key=key)
        # The task name, not `problem_idx`; the producer writes one of the two spellings.
        task = dig(rec, "metadata", "task_id") or dig(rec, "seed_metadata", "task_id")
        if task is not None:
            # Omit rather than answer None: a None outranks hc_traj_meta's value in a settle.
            yield Answer("rollout.task", task, key=key)
        yield Answer("rollout.repeat", rec.get("repeat"), key=key)
        if rec.get("tokens") is not None:
            yield Answer("rollout.n_tokens", rec.get("tokens"), key=key)
        raw = rec.get("status")
        if raw is not None:
            # An unknown status is unreadable, not guessed.
            if raw not in EVAL_STATUS:
                yield Unreadable(
                    "rollout.status",
                    where=f"line {line}",
                    detail=f"status={raw!r} is not one of "
                    f"{sorted(EVAL_STATUS)}; the store's column "
                    f"holds ok or failed and guessing which would "
                    f"colour this rollout",
                )
            else:
                yield Answer("rollout.status", EVAL_STATUS[raw], key=key)
                yield Answer("rollout.stop_reason", raw, key=key)


def nel_traj_meta(src: Sources):
    """The per-rollout facts of a trajectory record.

    `model` is a model-response document with the name inside; `tokens` a mapping or scalar.
    """
    for item in records(src.doc(0), "rollout.score"):
        if isinstance(item, Unreadable):
            yield item
            continue
        line, rec = item
        key = _row_key(rec, line)
        ref = record_ref(src.doc(0), line)
        yield Answer("rollout.trajectory_record", ref, key=key, keys={"trajectory_record": ref})
        model = rec.get("model") if isinstance(rec.get("model"), dict) else {}
        yield Answer("rollout.score", as_float(rec.get("reward")), key=key)
        yield Answer("rollout.task", dig(rec, "seed_metadata", "task_id"), key=key)
        yield Answer("rollout.agent_model", model.get("model"), key=key)
        yield Answer("rollout.error", rec.get("failure_category"), key=key)
        yield Answer("rollout.started_at", rec.get("timestamp"), key=key)
        tokens = model.get("tokens")
        if isinstance(tokens, dict):
            yield Answer("rollout.n_tokens", tokens.get("total"), key=key)
            yield Answer("rollout.n_prompt_tokens", tokens.get("prompt"), key=key)
            yield Answer("rollout.n_completion_tokens", tokens.get("completion"), key=key)
        elif isinstance(tokens, int) and not isinstance(tokens, bool):
            yield Answer("rollout.n_tokens", tokens, key=key)


def nel_own_events(src: Sources):
    """What the evaluator itself wrote of a rollout's turns: a single-turn record's prompt and
    response, and an ATIF document it wrote around an agent that writes none. The second
    stream only fills rollouts the first did not cover."""
    seen: set[tuple] = set()
    for i in (0, 1):
        for doc in src.docs(i):
            yield from _own_events(src, doc, seen)


def _own_events(src: Sources, doc, seen: set[tuple]):
    for item in records(doc, "rollout.events"):
        if isinstance(item, Unreadable):
            yield item
            continue
        line, rec = item
        key = _row_key(rec, line)
        if key in seen:
            continue
        docs = atif_documents(rec)
        if docs is None:
            said = [("user", rec.get("prompt")), ("assistant", rec.get("model_response"))]
            if not any(v for _role, v in said):
                continue
            seen.add(key)
            for n, (role, body) in enumerate(said):
                if not body:
                    continue
                yield Answer(
                    "rollout.events",
                    {
                        "source_event_id": f"{line}:{n}",
                        "read_seq": f"{line:09d}:{n:03d}",
                        "role": role,
                        **event_kind("message"),
                        "parts": text_parts(body),
                    },
                    key=key,
                    input_key=f"{line}:{n}",
                    via=SINGLE_TURN.name,
                )
            continue
        seen.add(key)
        for part in docs:
            if not written_by_wrapper(part):
                continue
            dec, steps = atif_steps(src, part, "rollout.events")
            if isinstance(dec, Unreadable):
                yield dec
                continue
            totals = atif_totals(part)
            if "prompt" in totals:
                yield Answer("rollout.n_prompt_tokens", totals["prompt"], key=key, via=dec.name)
            if "completion" in totals:
                yield Answer(
                    "rollout.n_completion_tokens", totals["completion"], key=key, via=dec.name
                )
            for n, step in enumerate(steps):
                for row in atif_rows(step, n):
                    yield Answer(
                        "rollout.events",
                        row,
                        key=key,
                        input_key=f"{line}:{row['source_event_id']}",
                        via=dec.name,
                    )
            diff = dig(part, "final_metrics", "workspace_diff_preview")
            if diff:
                yield Answer(
                    "rollout.artifacts",
                    field_artifact(str(diff), kind="patch", role="output", name=None),
                    key=key,
                    input_key=f"{line}:patch",
                    via=dec.name,
                )


def _first_outcome_field(rec: dict, name: str) -> tuple[bool, object]:
    """Read one allowlisted outcome field, preferring scoring_details over the top level; a
    blank string is no value."""
    details = rec.get("scoring_details")
    for owner in (details, rec):
        if isinstance(owner, dict) and has_value(owner.get(name)):
            return True, owner[name]
    return False, None


def _outcome_text(value: object) -> str:
    return (
        value
        if isinstance(value, str)
        else json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def _outcome_artifacts(rec: dict):
    """Portable verifier and candidate-output evidence from an exact field allowlist."""
    for field, name in (
        ("test_output", "test-output.log"),
        ("test_stdout", "test_stdout"),
        ("test_stderr", "test_stderr"),
    ):
        present, value = _first_outcome_field(rec, field)
        if present:
            yield (
                name,
                field_artifact(
                    _outcome_text(value),
                    kind="log",
                    role="verifier",
                    name=name,
                ),
            )

    evaluation = {}
    for field in ("test_results", "evaluation_completed"):
        present, value = _first_outcome_field(rec, field)
        if present:
            evaluation[field] = (
                without_reference_fields(value) if field == "test_results" else value
            )
    if evaluation:
        name = "evaluation-result.json"
        yield (
            name,
            field_artifact(
                json.dumps(
                    evaluation,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                kind="data",
                role="verifier",
                name=name,
            ),
        )

    present, patch = _first_outcome_field(rec, "model_patch")
    if present:
        name = "model.patch"
        yield (
            name,
            field_artifact(
                _outcome_text(patch),
                kind="patch",
                role="output",
                name=name,
            ),
        )


def _substantive_outcome(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (dict, list, tuple, set)):
        return bool(value)
    return True


def _merge_evaluation_artifacts(first: dict, later: dict) -> dict:
    """Fill empty first-stream evaluation fields without overriding real values."""
    merged = json.loads(str(first["text"]))
    candidate = json.loads(str(later["text"]))
    for field, value in candidate.items():
        if field not in merged or (
            not _substantive_outcome(merged[field]) and _substantive_outcome(value)
        ):
            merged[field] = value
    name = "evaluation-result.json"
    return field_artifact(
        json.dumps(merged, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        kind="data",
        role="verifier",
        name=name,
    )


def nel_scorer_logs(src: Sources):
    """Retain allowlisted outcomes, filling empty results fields from trajectories."""
    captured: dict[tuple, dict[str, tuple[dict, int]]] = {}
    order: list[tuple[tuple, str]] = []
    for i in (0, 1):
        for doc in src.docs(i):
            for item in records(doc, "rollout.artifacts"):
                if isinstance(item, Unreadable):
                    yield item
                    continue
                line, rec = item
                key = _row_key(rec, line)
                artifacts = list(_outcome_artifacts(rec))
                if not artifacts:
                    continue
                by_name = captured.setdefault(key, {})
                for name, artifact in artifacts:
                    existing = by_name.get(name)
                    if existing is None:
                        by_name[name] = (artifact, line)
                        order.append((key, name))
                    elif name == "evaluation-result.json":
                        by_name[name] = (
                            _merge_evaluation_artifacts(existing[0], artifact),
                            existing[1],
                        )
    for key, name in order:
        artifact, line = captured[key][name]
        yield Answer(
            "rollout.artifacts",
            artifact,
            key=key,
            input_key=f"{line}:{name}",
        )
