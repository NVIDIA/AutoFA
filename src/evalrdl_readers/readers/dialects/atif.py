# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ATIF, harbor's agent trajectory format: steps, their calls and their token counts, and the
decoders that pick a document's version by the `schema_version` it states. Harbor writes it into
a trial's `agent/trajectory.json`; an evaluator embeds it in its own records."""

from __future__ import annotations

from evalrdl_readers.core import Decoder

from ..common import args_text, dig, event_kind, text_parts, tool_call


def atif_parts(step: dict) -> list[dict]:
    """Every part of one ATIF step: reasoning, message, and observation results as text."""
    out = text_parts(step.get("reasoning_content"), channel="reasoning")
    out += text_parts(step.get("message"))
    for res in dig(step, "observation", "results") or ():
        body = res.get("content") if isinstance(res, dict) else res
        out += text_parts(body, channel="tool_result", origin="observation")
    return out


def atif_totals(traj: dict) -> dict:
    """What one ATIF trajectory says the whole rollout cost, in the store's column names."""
    final = traj.get("final_metrics")
    if not isinstance(final, dict):
        return {}
    out = {}
    for column, key in (
        ("prompt", "total_prompt_tokens"),
        ("completion", "total_completion_tokens"),
    ):
        value = final.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            out[column] = value
    return out


#: The two spellings ATIF writes a step's token counts under, store column -> keys; a step
#: carries one. Per-step cost and timing keys are not read (no store column).
ATIF_STEP_TOKENS: dict[str, tuple[str, str]] = {
    "prompt_tokens": ("prompt_tokens", "input_tokens"),
    "completion_tokens": ("completion_tokens", "output_tokens"),
    "cached_tokens": ("cached_tokens", "cache_read_tokens"),
}


def atif_step_tokens(step: dict) -> dict:
    """What one ATIF step says it cost, in the store's own event column names."""
    metrics = step.get("metrics")
    if not isinstance(metrics, dict):
        return {}
    out = {}
    for column, keys in ATIF_STEP_TOKENS.items():
        for key in keys:
            value = metrics.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                out[column] = value
                break
    return out


def atif_calls(step: dict) -> list[dict]:
    """The tool calls one ATIF step made."""
    out = []
    for call in step.get("tool_calls") or ():
        if not isinstance(call, dict):
            continue
        out.append(
            tool_call(
                # ATIF spells these `function_name`/`tool_call_id`.
                call.get("function_name") or call.get("name"),
                call.get("tool_call_id") or call.get("id"),
                args_text(call.get("arguments")),
                status=call.get("status"),
            )
        )
    return out


# ------------------------------------------------- a record's document, by its schema version


def _says(version: str):
    """The decider for one ATIF version: the record names its own schema."""
    return lambda rec: isinstance(rec, dict) and rec.get("schema_version") == version


ATIF_12 = Decoder(
    "atif-v1.2",
    decides=_says("ATIF-v1.2"),
    because="the ATIF document says schema_version ATIF-v1.2",
)
ATIF_15 = Decoder(
    "atif-v1.5",
    decides=_says("ATIF-v1.5"),
    because="the ATIF document says schema_version ATIF-v1.5",
)
ATIF_16 = Decoder(
    "atif-v1.6",
    decides=_says("ATIF-v1.6"),
    because="the ATIF document says schema_version ATIF-v1.6",
)
ATIF_17 = Decoder(
    "atif-v1.7",
    decides=_says("ATIF-v1.7"),
    because="the ATIF document says schema_version ATIF-v1.7",
)

ATIF_CANDIDATES = (ATIF_12, ATIF_15, ATIF_16, ATIF_17)

#: Why an `Order` ranks these decoders, wherever one must.
TRAJECTORY_BECAUSE = (
    "an ATIF document names exactly one `schema_version`, so two of these decoders can "
    "never answer one rollout; the order exists only because the library requires a total "
    "order over an input's decoders, and newest-first is the order a producer would be "
    "believed in if it ever wrote two"
)


def atif_steps(src, traj: dict, question: str):
    """(decoder-or-Unreadable, steps). The library picks; this only reads the steps."""
    dec = src.decode(traj, question)
    steps = traj.get("steps")
    return dec, steps if isinstance(steps, list) else []


def atif_rows(step: dict, idx: int) -> list[dict]:
    """One ATIF step as events: a turn, its calls, and their results.

    Split only when every call has a result. Tokens and `model_name` stay off result rows
    so a step's cost is counted once."""
    calls = atif_calls(step)
    results = [r for r in (dig(step, "observation", "results") or ()) if isinstance(r, dict)]
    # A call may have several results (a multi-result observation): each is kept.
    by_call: dict = {}
    for r in results:
        if r.get("source_call_id"):
            by_call.setdefault(r["source_call_id"], []).append(r)
    ids = [c.get("call_id_source") for c in calls]
    if not calls or len(by_call) != len(calls) or not all(i in by_call for i in ids):
        row = atif_row(step, idx)
        # Unsplit, the step's results are parts of this same event: it answers its own calls.
        if calls and results and any(ids):
            row["answers_call_ids"] = [i for i in ids if i]
        return [row]

    base = atif_row(step, idx)
    sid = base["source_event_id"]
    head = dict(base)
    head["parts"] = text_parts(step.get("reasoning_content"), channel="reasoning") + text_parts(
        step.get("message")
    )
    head["tool_calls"] = []
    head.update(event_kind("message"))
    # No empty turn row for a step that is only tool use.
    out = [head] if head["parts"] else []
    for n, (call, cid) in enumerate(zip(calls, ids, strict=True), start=1):
        row = dict(base)
        row["source_event_id"] = f"{sid}.{n:03d}"
        row["parts"] = []
        row["tool_calls"] = [call]
        row.update(event_kind("tool_call"))
        out.append(row)
        for k, result in enumerate(by_call[cid]):
            res = dict(base)
            # `.001a`, not `.r001`: natural-sort tiebreak must put each result right after its
            # call; a second result of the same call is `.001b`.
            res["source_event_id"] = f"{sid}.{n:03d}{chr(ord('a') + k) if k < 26 else f'z{k}'}"
            res["parts"] = text_parts(
                result.get("content"), channel="tool_result", origin="observation"
            )
            res["tool_calls"] = []
            res["answers_call_ids"] = [cid]
            res["model_name"] = None
            for col in ATIF_STEP_TOKENS:
                res.pop(col, None)
            res.update(event_kind("tool_result"))
            out.append(res)
    return out


def atif_row(step: dict, idx: int) -> dict:
    """One ATIF step, normalised; field names are the same in v1.2 through v1.7."""
    return {
        "source_event_id": str(step.get("step_id", idx)),
        "started_at": step.get("timestamp"),
        "role": step.get("source"),
        **event_kind("tool_call" if step.get("tool_calls") else "message"),
        "parts": atif_parts(step),
        "tool_calls": atif_calls(step),
        "model_name": step.get("model_name"),
        **atif_step_tokens(step),
    }


#: `agent.name` of the ATIF documents an evaluator writes itself, around an agent that writes
#: none (pi's stdout as one step); every other document is the agent harness's.
WRAPPER_AUTHORS = frozenset({"nemo-evaluator"})


def atif_documents(rec: dict) -> list[dict] | None:
    """The ATIF documents a record carries in `trajectory`, or None for a record with none."""
    traj = rec.get("trajectory")
    return [d for d in traj if isinstance(d, dict)] if isinstance(traj, list) else None


def written_by_wrapper(doc: dict) -> bool:
    return dig(doc, "agent", "name") in WRAPPER_AUTHORS
