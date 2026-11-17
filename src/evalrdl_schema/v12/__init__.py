# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""store-v12: v11 plus the reader layer's columns and tables.

`lane` joins the five trace tables' keys: a rollout has several sequences, each numbered
from 0. v11 rows read as lane `""` (not NULL: key columns must join); nothing is rewritten.
"""

from __future__ import annotations

from ..dsl import Add, Column, Dtype, Grain, Schema, Table, evolve
from ..v11 import V11

#: The lane a row written before lanes existed belongs to.
LANE_DEFAULT = ""

V12: Schema = evolve(
    V11,
    "store-v12",
    note="lanes, per-row provenance, stated relations, and the copy receipt",
    add=(
        Add(
            "runs",
            columns=(
                Column("benchmark_run_id", Dtype.STR,
                       "the group of runs this run belongs to, when the producer states one (an evaluation of a "
                       "larger benchmark run); NULL where it does not. A person or an agent may assert one"),
                Column("copy_complete", Dtype.BOOL,
                       "whether the stored copy of the run is the whole run folder"),
                Column("copy_missing_files", Dtype.INT,
                       "how many files our own transport did not bring, from the copy receipt beside the rundir"),
                Column("copy_incomplete_reason", Dtype.STR,
                       "why the copy is short -- a size cap, a denylist, an interrupted fetch. NULL when `copy_complete`"),
            ),
        ),
        Add(
            "rollouts",
            columns=(
                Column("health_verdict", Dtype.STR,
                       "what the producer said about its own observation of this rollout; `status` is "
                       "whether the rollout was ingested, a different question"),
                Column("error", Dtype.STR,
                       "the producer's own failure class for a rollout that did not run"),
            ),
        ),
        Add(
            "events", key=('run_id', 'task_id', 'repeat', 'agent_ref', 'lane', 'event_idx'),
            columns=(
                Column("lane", Dtype.STR,
                       "`<kind>/<instance>`: WHICH ordered sequence of the rollout this event belongs to. A rollout holds several -- the harness trace, the agent session, each sub-agent -- and each numbers its events from 0", backfill=LANE_DEFAULT),
                Column("lane_kind", Dtype.STR,
                       "the `<kind>` half of `lane`, so rows can be selected by lane kind without parsing `lane`"),
                Column("parent_event_id", Dtype.STR,
                       "the producer's own parent pointer, as the parent's `source_event_id`, never a position: "
                       "`event_idx` is assigned when a lane is ordered. A parent is in the same lane; a "
                       "cross-lane link is a `relations` row"),
                Column("prompt_id", Dtype.STR,
                       "the producer's own grouping of a turn with everything it caused; across lane kinds it "
                       "links the turn that called a sub-agent to the sub-agent's sequence"),
                Column("response_id", Dtype.STR,
                       "the model server's id of the call that produced this event, which joins it to the "
                       "serving log"),
                Column("latency_ttft_ms", Dtype.FLOAT,
                       "time to first token of the call that produced this event, where the producer timed it"),
                Column("src_input", Dtype.STR,
                       "the reader input that read this row; with `src_path` and `src_line`, where this one row came from"),
                Column("src_path", Dtype.STR,
                       "the rundir-relative file this row was read out of"),
                Column("src_line", Dtype.INT,
                       "the line within `src_path`, for a record stream; NULL for a whole-document source"),
            ),
        ),
        Add(
            "parts", key=('run_id', 'task_id', 'repeat', 'agent_ref', 'lane', 'event_idx', 'part_idx'),
            columns=(
                Column("lane", Dtype.STR,
                       "the lane of the event this part belongs to; see `events.lane`", backfill=LANE_DEFAULT),
            ),
        ),
        Add(
            "part_texts", key=('run_id', 'task_id', 'repeat', 'agent_ref', 'lane', 'event_idx', 'part_idx'),
            columns=(
                Column("lane", Dtype.STR,
                       "the lane of the event this text belongs to; see `events.lane`", backfill=LANE_DEFAULT),
            ),
        ),
        Add(
            "tool_calls", key=('run_id', 'task_id', 'repeat', 'agent_ref', 'lane', 'event_idx', 'call_idx'),
            columns=(
                Column("lane", Dtype.STR,
                       "the lane of the event that emitted this call; see `events.lane`", backfill=LANE_DEFAULT),
            ),
        ),
        Add(
            "tool_call_texts", key=('run_id', 'task_id', 'repeat', 'agent_ref', 'lane', 'event_idx', 'call_idx'),
            columns=(
                Column("lane", Dtype.STR,
                       "the lane of the event that emitted this call; see `events.lane`", backfill=LANE_DEFAULT),
            ),
        ),
    ),
    tables=(
        Table(
            "provenance", Grain.RUN,
            key=('run_id', 'question', 'input', 'site'),
            columns=(
                Column("run_id", Dtype.STR,
                       "the run this record is about"),
                Column("question", Dtype.STR,
                       "the reader's own question name -- `rollout.score`"),
                Column("input", Dtype.STR,
                       "the named unit inside the reader that answered it"),
                Column("site", Dtype.STR,
                       "`landmark@path`: WHICH DIRECTORY the bytes came from. In the key "
                       "because one (question, input) can read several, and without it no "
                       "record names the file"),
                Column("path", Dtype.STR,
                       "WHICH FILE, relative to the run root. NULL where a row was minted "
                       "rather than read out of bytes"),
                Column("first_line", Dtype.INT,
                       "the first line this record's values came from, where the file has "
                       "lines. NULL for a whole-file json/yaml parse"),
                Column("last_line", Dtype.INT,
                       "the last line of that span"),
                Column("reader", Dtype.STR,
                       "the reader that claimed the folder"),
                Column("producer", Dtype.STR,
                       "the declared producer group the input belongs to -- who wrote the bytes"),
                Column("reads", Dtype.STR,
                       "the input's declared read patterns, `landmark:pattern`"),
                Column("lands", Dtype.STR,
                       "the store column the question declares"),
                Column("task_id", Dtype.STR,
                       "the one rollout this record covers, else NULL"),
                Column("n_values", Dtype.INT,
                       "settled values this record accounts for"),
            ),
        ),
        Table(
            "relations", Grain.RUN,
            key=('run_id', 'space', 'frm_question', 'frm_key', 'to_question', 'to_key', 'kind'),
            columns=(
                Column("run_id", Dtype.STR,
                       "the run whose reader stated this edge"),
                Column("space", Dtype.STR,
                       "the declared key space this edge was matched on"),
                Column("kind", Dtype.STR,
                       "`relate` for a stated relation; an asserter picks its own"),
                Column("frm_question", Dtype.STR,
                       "the question owning the row the edge leaves"),
                Column("frm_task_id", Dtype.STR,
                       "the rollout that end sits in, where it has one"),
                Column("frm_lane", Dtype.STR,
                       "the lane of the row the edge leaves; an id belongs to its lane's author"),
                Column("frm_input", Dtype.STR,
                       "the input that read the row the edge leaves"),
                Column("frm_key", Dtype.STR,
                       "`input_key`: the producer's own id for that row. An endpoint is a NAME, "
                       "never a position -- a reparse renumbers a position and every stored "
                       "edge silently repoints"),
                Column("to_question", Dtype.STR,
                       "the question owning the row the edge points at"),
                Column("to_task_id", Dtype.STR,
                       "the rollout that end sits in, where it has one"),
                Column("to_lane", Dtype.STR,
                       "the lane of the row the edge points at"),
                Column("to_input", Dtype.STR,
                       "the input that read the row the edge points at"),
                Column("to_key", Dtype.STR,
                       "the producer's own id the edge points at"),
                Column("resolved", Dtype.BOOL,
                       "whether the far end was found. False records an edge naming nothing, "
                       "which is a row rather than a silence -- never repointed at the "
                       "nearest plausible target"),
                Column("reason", Dtype.STR,
                       "why it did not resolve"),
                Column("asserted_by", Dtype.STR,
                       "what claimed this edge -- a reader on a read, a person or agent on a "
                       "POST. The two are one kind of thing told apart by a column, not data "
                       "versus patch"),
                Column("asserted_at", Dtype.STR,
                       "when that claim was made"),
            ),
        ),
    ),
)
