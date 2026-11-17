# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Harbor's ATIF documents inside another producer's record streams.

An evaluator that runs harbor writes one record per trial into `results.jsonl` and
`trajectories.jsonl`, and puts the trajectory harbor's agent wrote, ATIF, into the record. The
record is the evaluator's; the document is harbor's. This reader reads only the documents, each
keyed by the record that holds it (`record_ref`), and the host joins that record to its trial.
A document the evaluator wrote itself (`written_by_wrapper`) is the evaluator's to read.
"""

from __future__ import annotations

import hashlib

from evalrdl_readers.core import (
    Answer,
    AnyOf,
    At,
    Card,
    Contribute,
    Decoders,
    Grain,
    Input,
    Landmark,
    Merge,
    Order,
    Population,
    Question,
    Reader,
    Rows,
    Sources,
    Unreadable,
)

from .. import questions as Q
from ..common import dig, dumps, field_artifact, record_ref, records
from ..dialects.atif import (
    ATIF_CANDIDATES,
    TRAJECTORY_BECAUSE,
    atif_documents,
    atif_rows,
    atif_steps,
    atif_totals,
    written_by_wrapper,
)
from ..store_schema import READER_SCHEMA

STREAMS = Landmark(
    name="harbor_streams",
    marker=AnyOf("trajectories.jsonl", "results.jsonl"),
    where=("",),
    expect=Card.ONE(because="the record streams an evaluator writes into one directory"),
)

ATIF_TRAJECTORY_RECORD = Question(
    name="rollout.atif_trajectory_record",
    grain=Grain.ROLLOUT,
    merge=Merge.ONE,
    lands="rollouts.atif_trajectory_record",
    means="the `trajectories.jsonl` record a rollout's ATIF document was read from",
)
ATIF_RESULTS_RECORD = Question(
    name="rollout.atif_results_record",
    grain=Grain.ROLLOUT,
    merge=Merge.ONE,
    lands="rollouts.atif_results_record",
    means="the `results.jsonl` record a rollout's ATIF document was read from",
)


def _harbor_documents(rec: dict) -> list[dict]:
    return [d for d in atif_documents(rec) or () if not written_by_wrapper(d)]


def _digest(doc: dict) -> str:
    return hashlib.sha256(dumps(doc).encode()).hexdigest()


def _document_answers(src: Sources, doc, line: int, rec: dict, question: str, space: str):
    ref = record_ref(doc, line)
    key = (ref,)
    named = False
    for n, part in enumerate(_harbor_documents(rec)):
        dec, steps = atif_steps(src, part, "rollout.events")
        if isinstance(dec, Unreadable):
            yield dec
            continue
        if not named:
            # The record is named once, stamped by the first document read from it.
            named = True
            yield Answer(question, ref, key=key, keys={space: ref}, via=dec.name)
        totals = atif_totals(part)
        if "prompt" in totals:
            yield Answer("rollout.n_prompt_tokens", totals["prompt"], key=key, via=dec.name)
        if "completion" in totals:
            yield Answer("rollout.n_completion_tokens", totals["completion"], key=key, via=dec.name)
        for i, step in enumerate(steps):
            # A step splits into call and result rows.
            for row in atif_rows(step, i):
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
                input_key=f"{line}:patch:{n}",
                via=dec.name,
            )


def atif_in_trajectories(src: Sources):
    """Every harbor ATIF document in `trajectories.jsonl`."""
    if not src.has(0):
        return
    doc = src.doc(0)
    for item in records(doc, "rollout.events"):
        if isinstance(item, Unreadable):
            yield item
            continue
        line, rec = item
        if _harbor_documents(rec):
            yield from _document_answers(
                src, doc, line, rec, "rollout.atif_trajectory_record", "trajectory_record"
            )


def atif_in_results(src: Sources):
    """Harbor ATIF documents in `results.jsonl` that `trajectories.jsonl` does not hold: the
    two streams carry the same documents where both have one, and `results.jsonl` is sometimes
    the only copy."""
    if not src.has(0):
        return
    held: set[str] = set()
    if src.has(1):
        for item in records(src.doc(1), "rollout.events"):
            if isinstance(item, Unreadable):
                yield item
                continue
            held.update(_digest(d) for d in _harbor_documents(item[1]))
    doc = src.doc(0)
    for item in records(doc, "rollout.events"):
        if isinstance(item, Unreadable):
            yield item
            continue
        line, rec = item
        mine = _harbor_documents(rec)
        if mine and not all(_digest(d) in held for d in mine):
            yield from _document_answers(
                src, doc, line, rec, "rollout.atif_results_record", "results_record"
            )


ATIF_DECODERS = Decoders(
    candidates=ATIF_CANDIDATES,
    because=(
        "ATIF version is a property of the document, and one stream can interleave versions, "
        "so sniffing the first record would misread the rest"
    ),
)
ATIF_ANSWERS = {
    "rollout.events": Card.ANY(
        because="a document with no steps is the obvious variant of a failure stub"
    ),
    "rollout.n_prompt_tokens": Card.ANY(
        because="`final_metrics.total_prompt_tokens` is missing from some documents"
    ),
    "rollout.n_completion_tokens": Card.ANY(
        because="the same block, absent from the same documents"
    ),
    "rollout.artifacts": Card.ANY(
        because="`final_metrics.workspace_diff_preview` is written only where the agent changed files"
    ),
}

ATIF_IN_TRAJECTORIES = Input(
    name="harbor_atif_trajectories",
    via=atif_in_trajectories,
    reads=[At(STREAMS, "trajectories.jsonl")],
    decoders=ATIF_DECODERS,
    answers={
        **ATIF_ANSWERS,
        "rollout.atif_trajectory_record": Card.ANY(
            because="a record whose trajectory harbor did not write has no harbor document"
        ),
    },
)
ATIF_IN_RESULTS = Input(
    name="harbor_atif_results",
    via=atif_in_results,
    reads=[At(STREAMS, "results.jsonl"), At(STREAMS, "trajectories.jsonl")],
    decoders=ATIF_DECODERS,
    answers={
        **ATIF_ANSWERS,
        "rollout.atif_results_record": Card.ANY(
            because="only a document `trajectories.jsonl` does not hold is read from here"
        ),
    },
)

_TOKEN_SOURCES = (
    tuple(f"harbor_atif_trajectories/{d.name}" for d in ATIF_CANDIDATES)[::-1]
    + tuple(f"harbor_atif_results/{d.name}" for d in ATIF_CANDIDATES)[::-1]
)

RECORDS = Reader(
    schema=READER_SCHEMA,
    counts=Population.ATTEMPTED,
    name="harbor-records",
    landmarks=(STREAMS,),
    inputs=(("harbor", (ATIF_IN_TRAJECTORIES, ATIF_IN_RESULTS)),),
    questions=(
        Q.ROLLOUT_EVENTS,
        Q.ROLLOUT_N_PROMPT_TOKENS,
        Q.ROLLOUT_N_COMPLETION_TOKENS,
        Q.ROLLOUT_ARTIFACTS,
        ATIF_TRAJECTORY_RECORD,
        ATIF_RESULTS_RECORD,
    ),
    rows=(
        Rows(
            question=Grain.ROLLOUT,
            may_create=("harbor_atif_trajectories", "harbor_atif_results"),
            because=(
                "read alone, each document is named by the record that holds it; a host that "
                "enumerates the trials declares its own"
            ),
        ),
    ),
    orders=(
        Order(
            question="rollout.atif_trajectory_record",
            order=tuple(f"harbor_atif_trajectories/{d.name}" for d in ATIF_CANDIDATES)[::-1],
            keep_loser=True,
            because=TRAJECTORY_BECAUSE,
        ),
        Order(
            question="rollout.atif_results_record",
            order=tuple(f"harbor_atif_results/{d.name}" for d in ATIF_CANDIDATES)[::-1],
            keep_loser=True,
            because=TRAJECTORY_BECAUSE,
        ),
        Order(
            question="rollout.n_prompt_tokens",
            order=_TOKEN_SOURCES,
            keep_loser=True,
            because=TRAJECTORY_BECAUSE + "; the `trajectories.jsonl` copy ranks first",
        ),
        Order(
            question="rollout.n_completion_tokens",
            order=_TOKEN_SOURCES,
            keep_loser=True,
            because=TRAJECTORY_BECAUSE + "; the `trajectories.jsonl` copy ranks first",
        ),
    ),
    attach=(
        Contribute(
            to="rollout.events",
            input="harbor_atif_trajectories",
            order_by="started_at",
            identity="source_event_id",
            because="ATIF numbers its own steps within a document and stamps its own timestamps",
        ),
        Contribute(
            to="rollout.events",
            input="harbor_atif_results",
            order_by="started_at",
            identity="source_event_id",
            because="the same documents, from the other stream",
        ),
        Contribute(
            to="rollout.artifacts",
            input="harbor_atif_trajectories",
            order_by="name",
            identity="sha256",
            because="a patch has no name; its digest identifies it, as it does every other artifact",
        ),
        Contribute(
            to="rollout.artifacts",
            input="harbor_atif_results",
            order_by="name",
            identity="sha256",
            because="the same patches, from the other stream",
        ),
    ),
)
