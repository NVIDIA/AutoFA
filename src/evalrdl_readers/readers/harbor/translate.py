# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

from evalrdl_readers.core import Answer, Sources, Unreadable

from ..common import (
    ARTIFACT_INDEX,
    artifact_row,
    as_float,
    dig,
    event_kind,
    field_artifact,
    records,
    site_key,
    text_parts,
    without_reference_fields,
)
from ..dialects.anthropic import anthropic_message
from ..dialects.atif import atif_rows, atif_steps, atif_totals
from ..dialects.sessions import session_shape
from .decoders import CTRF, RCA_REPORT, REWARD_V46, REWARD_V47, SCALEAI_OUT
from .openinference import openinference_rows, otlp_spans

# ---------------------------------------------------------------------- translators

_PORTABLE_VERIFIER_FIELDS = ("rewards", "diagnostics", "error", "errors")


def trial_result(src: Sources):
    if not src.has(0):
        return  # a trial holding only lock.json
    doc = src.doc(0).json() or {}
    key = (site_key(src.site),)
    yield Answer("rollout.task", doc.get("task_name"), key=key)
    yield Answer("rollout.agent_name", dig(doc, "agent_info", "name"), key=key)
    yield Answer("rollout.started_at", doc.get("started_at"), key=key)
    yield Answer("rollout.finished_at", doc.get("finished_at"), key=key)
    yield Answer("rollout.cost", as_float(dig(doc, "agent_result", "cost_usd")), key=key)
    yield Answer(
        "rollout.agent_model", dig(doc, "agent_info", "model_info", "name"), key=key
    )
    exc = doc.get("exception_info")
    yield Answer(
        "rollout.error",
        None if not isinstance(exc, dict) else exc.get("exception_type"),
        key=key,
    )


def trial_reward(src: Sources):
    if not src.has(0):
        return
    rewards = dig(src.doc(0).json() or {}, "verifier_result", "rewards")
    if not isinstance(rewards, dict):
        return
    key = (site_key(src.site),)
    if "composite_score" in rewards:
        yield Answer("rollout.score", as_float(rewards["composite_score"]), key=key,
                     via=REWARD_V46.name)
    elif "reward" in rewards:
        yield Answer("rollout.score", as_float(rewards["reward"]), key=key,
                     via=REWARD_V47.name)


def trial_verifier_result(src: Sources):
    """Retain Harbor's complete verifier result as one explicit outcome artifact."""
    if not src.has(0):
        return
    body = src.doc(0).json() or {}
    verifier_result = body.get("verifier_result")
    if not isinstance(verifier_result, dict):
        return
    portable = {
        field: without_reference_fields(verifier_result[field])
        for field in _PORTABLE_VERIFIER_FIELDS
        if field in verifier_result
    }
    if not portable:
        return
    yield Answer(
        "rollout.artifacts",
        field_artifact(
            json.dumps(
                portable,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            kind="data",
            role="verifier",
            name="verifier-result.json",
        ),
        key=(site_key(src.site),),
        input_key=src.doc(0).relpath,
    )


def trial_reward_txt(src: Sources):
    if not src.has(0):
        return
    yield Answer("rollout.score", as_float(src.doc(0).text()), key=(site_key(src.site),))


def trial_reward_json(src: Sources):
    if not src.has(0):
        return
    doc = src.doc(0).json()
    if not isinstance(doc, dict):
        return
    for name in ("composite_score", "reward"):
        if name in doc:
            yield Answer("rollout.score", as_float(doc[name]), key=(site_key(src.site),))
            return


def trial_criteria(src: Sources):
    """Every per-criterion score, flattened into one mapping; each file also kept whole."""
    key = (site_key(src.site),)
    out: dict[str, float | None] = {}
    for doc in src.docs(0):
        body = doc.json()
        name = doc.relpath.split("/")[-2]
        if isinstance(body, dict) and "reward" in body:
            out[name] = as_float(body["reward"])
        if body:
            body = without_reference_fields(body)
            yield Answer("rollout.artifacts",
                         field_artifact(json.dumps(body, sort_keys=True), kind="data",
                                        role="verifier", name=f"criteria/{name}.json"),
                         key=key, input_key=doc.relpath)
    if out:
        yield Answer("rollout.criteria", out, key=key)


def trial_exception(src: Sources):
    if not src.has(0):
        return
    text = src.doc(0).text().strip()
    head = text.splitlines()[0] if text else ""
    yield Answer("rollout.error", head[:200] or None, key=(site_key(src.site),))


def trial_decision(src: Sources):
    if not src.has(0):
        return
    doc = src.doc(0).json() or {}
    yield Answer("rollout.stop_reason", doc.get("decision"), key=(site_key(src.site),))


def _report_text(doc, body, kept) -> str:
    """The report's own bytes, unless a reference field had to be removed from it."""
    if kept == body:
        return doc.text()
    return json.dumps(kept, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def trial_verifier(src: Sources):
    for i, dec in ((0, CTRF), (1, SCALEAI_OUT), (2, RCA_REPORT)):
        if not src.has(i):
            continue
        body = src.doc(i).json()
        if body is None:
            continue
        kept = without_reference_fields(body)
        yield Answer("rollout.verifier", kept, key=(site_key(src.site),), via=dec.name)
        if dec is CTRF:
            yield Answer(
                "rollout.artifacts",
                field_artifact(
                    _report_text(src.doc(i), body, kept),
                    kind="data",
                    role="verifier",
                    name="ctrf.json",
                ),
                key=(site_key(src.site),),
                input_key=src.doc(i).relpath,
                via=dec.name,
            )


def trial_verifier_reports(src: Sources):
    """The verifier reports besides CTRF, kept whole as outcome evidence."""
    for i in (0, 1):
        if not src.has(i):
            continue
        doc = src.doc(i)
        body = doc.json()
        if body is None:
            continue
        yield Answer(
            "rollout.artifacts",
            field_artifact(
                _report_text(doc, body, without_reference_fields(body)),
                kind="data",
                role="verifier",
                name=doc.name,
            ),
            key=(site_key(src.site),),
            input_key=doc.relpath,
        )


def trial_verifier_logs(src: Sources):
    """Retain explicit raw verifier logs whose reports can omit decisive diagnostics."""
    key = (site_key(src.site),)
    for input_idx in (0, 1):
        for doc in src.docs(input_idx):
            yield Answer(
                "rollout.artifacts",
                artifact_row(doc, role="verifier"),
                key=key,
                input_key=doc.relpath,
            )


def trial_trajectory(src: Sources):
    if not src.has(0):
        return
    traj = src.doc(0).json()
    if traj is not None and not isinstance(traj, dict):
        # An agent that writes its own event list here (NOOA) leaves harbor no ATIF document.
        yield Unreadable(
            "rollout.events",
            where=src.doc(0).relpath,
            detail=f"a JSON {type(traj).__name__}, not an ATIF trajectory object; no turn of it is read",
        )
        return
    if not isinstance(traj, dict):
        return
    key = (site_key(src.site),)
    dec, steps = atif_steps(src, traj, "rollout.events")
    if isinstance(dec, Unreadable):
        yield dec
        return
    totals = atif_totals(traj)
    if "prompt" in totals:
        yield Answer("rollout.n_prompt_tokens", totals["prompt"], key=key, via=dec.name)
    if "completion" in totals:
        yield Answer("rollout.n_completion_tokens", totals["completion"], key=key,
                     via=dec.name)
    for i, step in enumerate(steps):
      for row in atif_rows(step, i):
        yield Answer(
            "rollout.events", row, key=key, input_key=row["source_event_id"],
            keys={"atif_step": f"{key[0]}/{row['source_event_id']}"}, via=dec.name,
            # The harness's record: a separate lane from the agent's session log.
            lane=("trace", ""),
        )


def trial_session(src: Sources):
    """The agent's session log (claude-code or codex), decoded per record.

    Sub-agent files, in a directory named after this file's stem, are `harbor_subagent`'s."""
    key = (site_key(src.site),)
    for i in (0, 1):
        for doc in src.docs(i):
            session = doc.name[: -len(".jsonl")]
            for item in records(doc, "rollout.events"):
                if isinstance(item, Unreadable):
                    yield item
                    continue
                line, rec = item
                dec = src.decode(rec, "rollout.events")
                if isinstance(dec, Unreadable):
                    yield dec
                    continue
                got = session_shape(dec, rec)
                yield Answer(
                    "rollout.events",
                    {
                        "source_event_id": got["id"] or f"{session}:{line:06d}",
                        # Same id space as `source_event_id`; may point at a record not
                        # emitted as an event (compact_boundary).
                        "parent_event_id": rec.get("parentUuid"),
                        # The only field linking trunk and sub-agent files (spawn edge).
                        "prompt_id": rec.get("promptId"),
                        "started_at": rec.get("timestamp"),
                        "role": got["role"],
                        **event_kind(got["kind"]),
                        "parts": got["parts"],
                        "tool_calls": got["tool_calls"],
                        "model_name": got["model_name"],
                        "session": session,
                        **({"answers_call_ids": got["answers"]} if got["answers"] else {}),
                    },
                    key=key,
                    input_key=f"{session}:{line}",
                    via=dec.name,
                    lane=(got["lane"], session),
                )


def trial_subagent(src: Sources):
    key = (site_key(src.site),)
    for doc in src.docs(0):
        agent = doc.name[: -len(".jsonl")]
        for item in records(doc, "rollout.events"):
            if isinstance(item, Unreadable):
                yield item
                continue
            line, rec = item
            msg = rec.get("message")
            got = (anthropic_message(msg) if isinstance(msg, dict)
                   else {"parts": text_parts(msg), "tool_calls": [], "kind": None,
                         "answers": []})
            yield Answer(
                "rollout.events",
                {
                    "source_event_id": rec.get("uuid") or f"{agent}:{line}",
                    "parent_event_id": rec.get("parentUuid"),
                    # Shared with the spawning trunk.
                    "prompt_id": rec.get("promptId"),
                    "started_at": rec.get("timestamp"),
                    "role": (msg or {}).get("role") if isinstance(msg, dict) else (rec.get("userType") or rec.get("type")),
                    **event_kind(got["kind"] or rec.get("type")),
                    "parts": got["parts"],
                    "tool_calls": got["tool_calls"],
                    "model_name": (msg or {}).get("model") if isinstance(msg, dict) else None,
                    # Per row: the lane's instance is the agent.
                    "actor": agent,
                    "session": rec.get("sessionId"),
                    # Call ids share the trunk's id space, so the bridge resolves within a lane.
                    **({"answers_call_ids": got["answers"]} if got.get("answers") else {}),
                },
                key=key,
                input_key=f"{agent}:{line}",
                keys={"subagent": f"{key[0]}/{agent}"},
                lane=("subagent", agent),
            )


def trial_subagent_meta(src: Sources):
    key = (site_key(src.site),)
    for doc in src.docs(0):
        body = doc.json() or {}
        agent = doc.name[: -len(".meta.json")]
        yield Answer(
            "rollout.subagents",
            {
                "agent": agent,
                "kind": body.get("agentType"),
                "description": body.get("description"),
            },
            key=key,
            input_key=agent,
            keys={"subagent": f"{key[0]}/{agent}"},
        )


def trial_tool_results(src: Sources):
    """Tool-result side-files, with a preview event and complete pageable artifact."""
    key = (site_key(src.site),)
    for doc in src.docs(0):
        yield Answer(
            "rollout.events",
            {
                "source_event_id": doc.name,
                "started_at": None,
                "role": "tool",
                **event_kind("tool_result_sidefile"),
                "parts": text_parts(doc.text(limit=4096)),
            },
            key=key,
            input_key=doc.name,
        )
        yield Answer(
            "rollout.artifacts",
            artifact_row(doc, role="trace"),
            key=key,
            input_key=doc.relpath,
        )


def trial_traces(src: Sources):
    """OTLP spans: the OpenInference conversation when the trace has LLM spans, else one
    answer per line (a trace without one is huge and says only span names)."""
    key = (site_key(src.site),)
    spans, batches = [], []
    for i in (0, 1):
        for doc in src.docs(i):
            for item in records(doc, "rollout.events"):
                if isinstance(item, Unreadable):
                    yield item
                    continue
                line, rec = item
                got = otlp_spans(rec)
                spans.extend(got)
                if got:
                    batches.append((doc, line, got))
    convo = openinference_rows(spans)
    if convo is not None:
        for row in convo["rows"]:
            yield Answer("rollout.events", row, key=key, input_key=row["read_seq"])
        if convo["prompt_tokens"] is not None:
            yield Answer("rollout.n_prompt_tokens", convo["prompt_tokens"], key=key)
        if convo["completion_tokens"] is not None:
            yield Answer("rollout.n_completion_tokens", convo["completion_tokens"], key=key)
        return
    for n, (doc, line, got) in enumerate(batches):
        t0 = got[0].get("startTimeUnixNano")
        yield Answer(
            "rollout.events",
            {
                "source_event_id": got[0].get("spanId") or f"{doc.name}:{line}",
                "read_seq": f"{int(t0):020d}" if str(t0 or "").isdigit() else f"~{n:09d}",
                "started_at": t0,
                "role": "otel",
                **event_kind("span_batch"),
                "parts": text_parts([s.get("name") for s in got]),
            },
            key=key,
            input_key=f"{doc.name}:{line}",
        )


def trial_agent_result(src: Sources):
    """`agent/result.json`: only the LangGraph shape (`{"messages": [...]}`) is read.

    nooa-bench writes a self-report under the same name; a read pattern cannot tell them apart."""
    if not src.has(0):
        return
    body = src.doc(0).json()
    if not isinstance(body, dict):
        return
    messages = body.get("messages")
    if not isinstance(messages, list):
        return
    key = (site_key(src.site),)
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        calls = msg.get("tool_calls") or []
        yield Answer(
            "rollout.events",
            {
                "source_event_id": msg.get("id") or str(i),
                # Position, zero-padded; the id does not sort in order.
                "read_seq": f"{i:09d}",
                "started_at": None,
                "role": msg.get("type"),
                **event_kind("tool_call" if calls else (msg.get("type") or "message")),
                "parts": text_parts(msg.get("content")),
            },
            key=key,
            input_key=str(i),
        )


def trial_artifacts(src: Sources):
    """Every payload in the trial's artifacts/, skipping the `manifest.json` index.

    Skipped here, not by `Ignore`: the file is matched by this input, so an Ignore would be CONTESTED.
    """
    key = (site_key(src.site),)
    for doc in src.docs(0):
        if doc.name == ARTIFACT_INDEX:
            continue
        yield Answer(
            "rollout.artifacts", artifact_row(doc), key=key, input_key=doc.relpath,
        )


def trial_agent_store(src: Sources):
    """The SQLite agent store as separate files; `.db` without `-wal` misses recent writes."""
    key = (site_key(src.site),)
    for doc in src.docs(0):
        yield Answer(
            "rollout.artifacts", artifact_row(doc), key=key, input_key=doc.relpath,
            note="part of a live SQLite store; the -wal holds writes the .db does not",
        )


def eval_score(doc: dict):
    """The run-level score harbor publishes, or None.

    Accepts `metrics: [{"mean": ...}]` and `[{"composite_score": ...}]`; skips evals with
    `n_trials: 0`, whose `mean: 0.0` is not a score.
    """
    evals = dig(doc, "stats", "evals", default={}) or {}
    values = []
    for block in evals.values():
        if not isinstance(block, dict) or not block.get("n_trials"):
            continue
        for m in block.get("metrics") or []:
            if not isinstance(m, dict):
                continue
            if m.get("mean") is not None:
                values.append(as_float(m["mean"]))
            elif m.get("composite_score") is not None:
                values.append(as_float(m["composite_score"]))
    return values[0] if len(values) == 1 else None


def job_result(src: Sources):
    """harbor's job-level rollup."""
    doc = src.doc(0).json() or {}
    yield Answer("run.source_run_id", doc.get("id"))
    yield Answer("run.n_trials", doc.get("n_total_trials"))
    yield Answer("run.started_at", doc.get("started_at"))
    cost = as_float(dig(doc, "stats", "cost_usd"))
    if cost is not None:
        yield Answer("run.cost", cost)
    yield Answer("run.score", eval_score(doc))


def job_config(src: Sources):
    """The job's first agent block: which agent ran and the model reference it was given."""
    doc = src.doc(0).json() or {}
    agents = doc.get("agents") or []
    first = agents[0] if isinstance(agents, list) and agents else {}
    if not isinstance(first, dict):
        first = {}
    if first.get("name"):
        yield Answer("run.agent_harness", first["name"])
    if first.get("model_name"):
        yield Answer("run.agent_model_ref", first["model_name"])
