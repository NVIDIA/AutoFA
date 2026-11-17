# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`ReadResult` -> `{table: [row_dict, ...]}` in the tables `evalrdl-schema` declares.

``lands="table.column"`` / ``lands="table"`` is the only mapping; a value with no matching
column is dropped and counted, never guessed. A ``Mapping`` value is spread across the landing
table's columns; ``Row.position`` supplies the ``*_idx`` key column.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from evalrdl_schema import CURRENT, parquet_row_group_rows, polars_schemas, widest_value

from .core import Grain, Kind
from .provenance import (
    lane_kind as _lane_kind,
)
from .provenance import (
    lane_name as _lane_name,
)
from .provenance import (
    provenance_rows as _provenance,
)
from .provenance import (
    question_index as _question_index,
)
from .provenance import (
    relation_rows as _relations,
)
from .provenance import (
    rollout_ident,
)
from .readers.common import split_inline_think
from .readers.store_schema import NO_HOME
from .readers.store_schema import TEXT_SIBLING as _READER_TEXT_SIBLING

_DECLARED = polars_schemas(CURRENT)
SCHEMAS = _DECLARED
SCHEMA_VERSION = CURRENT.version

# --------------------------------------------------------------------------- schema

#: Never extend at import: only the writer process would see it. Declare in the schema.
APP_COLUMNS: dict[str, tuple[str, ...]] = {t: tuple(s) for t, s in SCHEMAS.items()}

STORE_COLUMNS: dict[str, tuple[str, ...]] = APP_COLUMNS


class DuplicateRollout(ValueError):
    """Two rollouts of one run share a task and a repeat."""


class SchemaDrift(RuntimeError):
    """The columns the readers land in are not the store's."""


def check_schema_agreement() -> None:
    """Fail if the bridge's columns diverge from the declared schema."""
    if STORE_COLUMNS != {t: tuple(c) for t, c in polars_schemas(CURRENT).items()}:
        raise SchemaDrift(
            "the bridge is writing columns that are not in the declared schema; "
            "declare them in a new version rather than beside it"
        )


#: A table, not `runs` columns: ADD COLUMN locks a populated table.
PROVENANCE_TABLE = "provenance"
PROVENANCE_SCHEMA: dict[str, Any] = _DECLARED["provenance"]

RELATIONS_TABLE = "relations"

#: Endpoints are names, never positions: a reparse renumbers positions.
#: `asserted_by` is the reader (on a read) or the person/agent (on a POST).
RELATIONS_SCHEMA: dict[str, Any] = _DECLARED["relations"]

_ROLLOUT_PK = ("run_id", "task_id", "repeat", "agent_ref")
#: `events` answer keys holding child rows; written at the parent's `event_idx`, which is
#: assigned last.
NESTED: dict[str, tuple[str, str]] = {
    # events key -> (child table, that table's second index column)
    "parts": ("parts", "part_idx"),
    "tool_calls": ("tool_calls", "call_idx"),
}
#: Call-to-result edge by producer id; resolved to positions after the lane is ordered.
ANSWERED_CALLS_KEY = "answers_call_ids"

#: `agent_ref` of a rollout whose agent no source names, as runs already committed spell it.
DEFAULT_AGENT_REF = "main"

#: Per-rollout rollups (`n_events`, `n_tool_calls`, turns, tokens, reasoning) count the
#: rollout's own turns. A lane kind is either one of its accounts, counted, or a view about
#: it or a copy of it, not counted. The unnamed lane is always counted. Every declared kind
#: must be in one of the two (`test_rollups_count_the_rollout_not_its_copies`).
ROLLUP_LANES_COUNTED = {
    "trace": "the harness's ATIF record of a harbor trial; for most hosts the only account",
    "session": "the agent's own session log",
    "subagent": "a sub-agent's own turns, which the parent's turns do not repeat",
    "history": "gdpval's multi-turn transcript; the stream record holds only the final exchange",
}
ROLLUP_LANES_NOT_COUNTED = {
    "otel": "runtime observations about a rollout, not conversation turns",
    "vllm": "engine observations of requests already counted by the conversation",
    "model_call": "the serving layer's capture of the calls the transcript already holds; "
    "counted, each tool call would count twice",
    "verdict": "the quality pass's account of the rollout, written after it",
    "judge": "a second model grading the answer, not a turn of the rollout",
    "bookkeeping": "what the harness did around the conversation, not in it",
    "context": "what the harness rendered into the prompt beside the conversation, not a turn",
    "codex-ui": "codex's UI rendering of turns its session already records",
    "attempt": "an earlier try of the same rollout that failed; the rollout is the try that "
    "finished",
}


#: A rollout's `record_schema` in production's spelling, where the reader names the input (or
#: decoder) that read its record; production filters and stored rows use these.
RECORD_SCHEMA_SPELLING = {
    "gym_transcript": "responses-api",
    "gym_failures": "responses-api",
    # the scoring input: a scored record with no conversation (532 of prod swebench-pro 6828bdb0)
    "gym_rollouts": "responses-api",
    "gym_trial_agent": "responses-api",
    "gym_tau2": "tau2-result",
    "gdpval_history": "stirrup",
    "in_logs": "inspect-sample",
    "ns_outputs": "nemo-skills-output",
    "ag_events": "frontier-agentic-v1",
    "harbor-single-turn": "atif",
    "single-turn": "atif",
}
#: Inputs that name a rollout's record shape over the others answering its main lane: a tau2
#: record is a responses-api record with tau2's own trajectory beside it.
RECORD_SCHEMA_FIRST = ("gym_tau2",)


def _record_refs(root: str, record_of: dict, line_starts: Mapping[str, Sequence[int]] = {}) -> dict:
    """`<path>:<byte offset>` of each rollout's primary record, as committed runs store it.

    A line number becomes the offset of that line's start; a whole-file record is offset 0.
    Offsets the read already took are used as they are; any other file is scanned once.
    """
    wanted: dict[str, set[int]] = {}
    for _, rel, line in record_of.values():
        wanted.setdefault(rel, set()).add(line)
    starts: dict[tuple[str, int], int] = {}
    for rel, lines in wanted.items():
        need = sorted(n for n in lines if n > 0)
        starts[(rel, 0)] = 0
        if not need:
            continue
        known = line_starts.get(rel)
        if known is not None:
            for n in need:
                if n <= len(known):
                    starts[(rel, n)] = known[n - 1]
            continue
        try:
            starts.update(((rel, n), off) for n, off in _line_offsets(Path(root) / rel, need))
        except OSError:
            continue
    return {
        k: f"{rel}:{starts[(rel, line)]}"
        for k, (_, rel, line) in record_of.items()
        if (rel, line) in starts
    }


_SCAN_BLOCK = 8 * 1024 * 1024


def _line_offsets(path: Path, need: list[int]):
    """(line, offset of its start) for each of `need`, sorted ascending, in one block scan."""
    i, n, base = 0, 1, 0
    if need[0] == 1:
        yield 1, 0
        i = 1
    with open(path, "rb", buffering=0) as f:
        while i < len(need):
            block = f.read(_SCAN_BLOCK)
            if not block:
                return
            at = block.find(b"\n")
            while at != -1 and i < len(need):
                n += 1
                if n == need[i]:
                    yield n, base + at + 1
                    i += 1
                at = block.find(b"\n", at + 1)
            base += len(block)


#: A second account of the rollout's own turns: a count is the larger of it and the rest, never
#: their sum. Prod gdpval 42a332b5: the stream repeats the history for most rollouts (4,914 of
#: 4,946 calls) and holds none of it for others, and history-only rollouts have no stream.
#: An agent's session log is its own account of the turns the trace or trajectory holds.
ROLLUP_LANES_ALTERNATE = {"history", "session"}

#: Turns no account of the parent repeats: added to the larger account, never compared with it.
#: A NOOA session tree mounted in a harbor trial with no account of its own lands its root as
#: `session` and its children as `subagent`; compared, the larger of the two was one session.
ROLLUP_LANES_ADDED = {"subagent"}


def _counted(pkey: tuple) -> bool:
    """Whether a row keyed rollout(4) + lane + ... counts toward its rollout's rollups."""
    lane = ((pkey[4] if len(pkey) > 4 else "") or "").split("/", 1)[0]
    return lane not in ROLLUP_LANES_NOT_COUNTED


def _account(pkey: tuple) -> tuple:
    """The rollout and which of its accounts this row belongs to."""
    lane = ((pkey[4] if len(pkey) > 4 else "") or "").split("/", 1)[0]
    return pkey[:4] + (lane if lane in ROLLUP_LANES_ALTERNATE | ROLLUP_LANES_ADDED else "",)


def _larger_account(by_account: Counter) -> Counter:
    out: Counter = Counter()
    for k, v in by_account.items():
        if k[4] not in ROLLUP_LANES_ADDED:
            out[k[:4]] = max(out[k[:4]], v)
    for k, v in by_account.items():
        if k[4] in ROLLUP_LANES_ADDED:
            out[k[:4]] += v
    return out


def _event_pkey(pk: dict) -> tuple:
    """The `out["events"]` key of this row, as `cell()` builds it."""
    return tuple(pk[c] for c in _ROLLOUT_PK[:4]) + (pk.get("lane"),) + (pk["event_idx"],)


#: Long strings live in a sibling table so queries not asking for them skip them. One map,
#: the readers' own, so a reader and the write step cannot disagree on where a column goes.
TEXT_SIBLING = _READER_TEXT_SIBLING

INDEX_COLUMNS: dict[str, tuple[str, ...]] = {
    "runs": (),
    "rollouts": (),
    "attempts": ("attempt_idx",),
    "events": ("event_idx",),
    "event_texts": ("event_idx",),
    "parts": ("event_idx", "part_idx"),
    "part_texts": ("event_idx", "part_idx"),
    "tool_calls": ("event_idx", "call_idx"),
    "tool_call_texts": ("event_idx", "call_idx"),
    "artifacts": ("artifact_idx",),
    "artifact_texts": ("artifact_idx",),
    "tool_defs": ("def_idx",),
    "tool_def_texts": ("def_idx",),
    "prompt_messages": ("msg_idx",),
}


def _expected_source(result) -> str:
    """Which input supplied a non-None `run.n_expected`, or `none`."""
    for row in result.rows.get(("run.n_expected", ()), ()):
        if row.value is not None:
            return row.input
    return "none"


# --------------------------------------------------------------------------- loss


#: The closed list of reasons a value read by the reader does not reach a store column.
NO_HOME_REASON = "no_home"
NO_COLUMN = "no_column"
UNKEYABLE_GRAIN = "unkeyable_grain"
UNKEYABLE_INDEX = "unkeyable_index"
NO_POSITION = "no_position"
WRONG_TYPE = "wrong_type"
SCALAR_IN_TABLE = "scalar_into_whole_table"
COLLISION = "collision"

REASON_TEXT = {
    NO_HOME_REASON: "the store has no column for this fact (evalrdl.intake.readers NO_HOME)",
    NO_COLUMN: "the mapping key names no column of the landing table",
    UNKEYABLE_GRAIN: "a question landing in runs that is not at run grain",
    UNKEYABLE_INDEX: "the key's components past the rollout do not match the table's index columns",
    NO_POSITION: "the row has no position and the table's index column needs one",
    WRONG_TYPE: "the value cannot be held by the column's type",
    SCALAR_IN_TABLE: "a whole-table landing whose value is not a mapping, so it names no column",
    COLLISION: "two values for one column of one row; the first was kept",
}


@dataclass
class Loss:
    """What a conversion produced and what it could not carry. Addable across runs."""

    rows_in: int = 0
    rows_landed: int = 0
    rows_landed_all_null: int = 0
    rows_dropped: Counter = field(default_factory=Counter)

    cells_in: int = 0
    cells_landed: int = 0
    cells_landed_null: int = 0
    cells_dropped: Counter = field(default_factory=Counter)

    rows_out: Counter = field(default_factory=Counter)
    #: (question, reason) -> n cells, so a drop can be traced to the declaration.
    by_question: Counter = field(default_factory=Counter)
    #: (table, key, reason) -> n cells, the "what had nowhere to go" list.
    by_slot: Counter = field(default_factory=Counter)

    @property
    def rows_dropped_total(self) -> int:
        return sum(self.rows_dropped.values())

    def add(self, other: Loss) -> None:
        self.rows_in += other.rows_in
        self.rows_landed += other.rows_landed
        self.rows_landed_all_null += other.rows_landed_all_null
        self.cells_in += other.cells_in
        self.cells_landed += other.cells_landed
        self.cells_landed_null += other.cells_landed_null
        for a, b in (
            (self.rows_dropped, other.rows_dropped),
            (self.cells_dropped, other.cells_dropped),
            (self.rows_out, other.rows_out),
            (self.by_question, other.by_question),
            (self.by_slot, other.by_slot),
        ):
            a.update(b)


@dataclass
class Conversion:
    tables: dict[str, list[dict]]
    loss: Loss
    run_id: str
    run_uuid: str
    provenance: list[dict] = field(default_factory=list)
    relations: list[dict] = field(default_factory=list)
    #: False: zero rollouts; write no table rows, only `record` (like `EmptyDecodeError`).
    admitted: bool = True
    #: Failure record shaped like `parse_consumer`'s `quarantine.json`.
    record: dict = field(default_factory=dict)
    #: Unresolved call ids stay null and are counted, never matched to a nearby call.
    calls_paired: int = 0
    calls_unpaired: int = 0


# --------------------------------------------------------------------------- coercion


_KINDS: dict[Any, str | None] = {}


def _kind(dtype: Any) -> str | None:
    """The coercion a column type takes; resolved once per type, not per value."""
    try:
        return _KINDS[dtype]
    except KeyError:
        pass
    kind = next(
        (
            k
            for k, t in (
                ("str", pl.Utf8),
                ("bool", pl.Boolean),
                ("int", pl.Int64),
                ("float", pl.Float64),
            )
            if dtype == t
        ),
        None,
    )
    _KINDS[dtype] = kind
    return kind


def _coerce(value: Any, dtype: Any) -> tuple[bool, Any]:
    """(ok, value) if the column can hold `value` without lossy conversion."""
    return _coerce_kind(value, _kind(dtype))


def _coerce_kind(value: Any, kind: str | None) -> tuple[bool, Any]:
    if value is None:
        return True, None
    if kind == "str":
        if isinstance(value, str):
            return True, value
        if isinstance(value, bool):
            return False, None
        if isinstance(value, (int, float)):
            return True, repr(value)
        return False, None
    if kind == "bool":
        return (True, value) if isinstance(value, bool) else (False, None)
    if kind == "int":
        if isinstance(value, bool):
            return False, None
        if isinstance(value, int):
            return True, value
        if isinstance(value, float) and float(value).is_integer():
            return True, int(value)
        return False, None
    if kind == "float":
        if isinstance(value, bool):
            return False, None
        if isinstance(value, (int, float)):
            return True, float(value)
        return False, None
    return False, None


# --------------------------------------------------------------------------- ids


FINGERPRINT_HEAD = FINGERPRINT_TAIL = 4096


def fingerprint_file(path: Path) -> str:
    """`<size>:<sha256 of the first and last few KiB>`; reads kilobytes, whatever the size."""
    size = path.stat().st_size
    h = hashlib.sha256()
    with path.open("rb") as f:
        h.update(f.read(FINGERPRINT_HEAD))
        if size > FINGERPRINT_HEAD:
            f.seek(max(FINGERPRINT_HEAD, size - FINGERPRINT_TAIL))
            h.update(f.read(FINGERPRINT_TAIL))
    return f"{size}:{h.hexdigest()}"


def run_identity(result, tree_root: str) -> tuple[str, str]:
    """(run_id, run_uuid), content-derived over the files the read opened."""
    h = hashlib.sha256()
    for rel in sorted(result.read_paths):
        h.update(rel.encode())
        h.update(b"\0")
        h.update(fingerprint_file(Path(tree_root, rel)).encode())
    digest = h.hexdigest()
    return digest[:16], digest[:32]


# --------------------------------------------------------------------------- convert


#: (a rollout-creating producer produced rows, the reader minted rollout keys) -> why empty.
EMPTY_KIND: dict[tuple[bool, bool], tuple[str, str]] = {
    (False, False): (
        "no_rollout_producer",
        "no producer this reader declares can create a rollout produced any row: the "
        "outer layers spoke and the one that makes rollouts did not",
    ),
    (True, False): (
        "rollout_source_empty",
        "the producer that owns the rollout source produced rows and no rollout: the "
        "source was there and yielded none -- the findings below say why",
    ),
    (True, True): (
        "bridge_dropped_rollouts",
        "the reader enumerated rollouts and no column of the store held one. This is a "
        "loss in this bridge, not in the folder, and `n_rollouts=0` would be a false "
        "statement about the read",
    ),
    (False, True): (
        "bridge_dropped_rollouts",
        "the reader enumerated rollouts and no column of the store held one. This is a "
        "loss in this bridge, not in the folder, and `n_rollouts=0` would be a false "
        "statement about the read",
    ),
}

#: (read was ok, the run was admitted) -> the verdict written beside the parquet.
VERDICT: dict[tuple[bool, bool], str] = {
    (True, True): "ok",
    (False, True): "lossy",
    (True, False): "empty",
    (False, False): "empty",
}


def _rollout_creators(reader) -> set[str]:
    """The producers that own an input this reader declares may create a rollout."""
    named: set[str] = set()
    for decl in (*reader.rows_by_grain.values(), *reader.rows_decl.values()):
        named.update(decl.may_create)
    return {reader.producer_of.get(n) for n in named} - {None}


def _mean(xs: list[int]) -> float | None:
    return (sum(xs) / len(xs)) if xs else None


def _p95(xs: list[int]) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    return float(s[min(len(s) - 1, max(0, math.ceil(0.95 * len(s)) - 1))])


def _rollout_findings(result) -> dict[str, list[str]]:
    """LOSS findings per rollout, by substring match of the key; under-reports."""
    from evalrdl_readers.core import LOSS

    # Findings can be as numerous as rollouts. Index key strings by length so a
    # finding scans its own text, rather than every rollout in the run. Keep all
    # overlapping substring matches and duplicate string representations.
    counts = Counter(str(k) for k in {k[0] for (q, k) in result.rows if len(k) == 1})
    by_length: dict[int, set[str]] = defaultdict(set)
    for key in counts:
        if key:
            by_length[len(key)].add(key)
    out: dict[str, list[str]] = defaultdict(list)
    for f in result.findings:
        if f.kind not in LOSS:
            continue
        blob = f"{f.where} {f.detail}"
        matched = set()
        for size, keys in by_length.items():
            for start in range(len(blob) - size + 1):
                key = blob[start : start + size]
                if key in keys:
                    matched.add(key)
        if matched:
            line = f.line()
            for key in sorted(matched):
                out[key].extend([line] * counts[key])
    return out


#: The columns of a quarantined rollout row, as the adapter path writes them.
_QUARANTINED = {
    "record_schema": "other",
    "reward": None,
    "is_scored": False,
    "status": "failed",
    "ingest_status": "quarantined",
    "n_events": None,
    "n_assistant_turns": None,
    "n_reasoning_segments": None,
    "n_tool_calls": None,
    "n_artifacts": None,
    "n_truncated_events": None,
    "reasoning_char_share": None,
    "has_encrypted_reasoning": None,
    "prompt_tokens": None,
    "completion_tokens": None,
    "total_tokens": None,
    "started_at": None,
    "finished_at": None,
}


def _byte_offset(root: str, rel: str, line: int | None) -> int | None:
    """Byte offset of 1-based `line` in `root/rel`, the adapter's raw_record_ref unit."""
    if line is None:
        return None
    try:
        with open(Path(root) / rel, "rb") as f:
            off = 0
            for i, raw in enumerate(f, 1):
                if i == line:
                    return off
                off += len(raw)
    except OSError:
        pass
    return None


def _unparseable(result, qindex) -> list[tuple[str, int | None, str]]:
    """(file, line, detail) of each rollout record no reader could parse, once per record."""
    seen: dict[tuple[str, int | None], str] = {}
    for f in result.findings:
        if f.kind is not Kind.UNREADABLE or not f.where or f.question not in qindex:
            continue
        if qindex[f.question].grain is not Grain.ROLLOUT:
            continue
        detail = (f.detail or "").split(" — ")[0]
        if not detail.startswith(("unparseable record", "record is a ")):
            continue
        rel, _, line = f.where.rpartition(":")
        if not rel or not line.isdigit():
            continue
        seen.setdefault((rel, int(line)), detail)
    return [
        (rel, line, d)
        for (rel, line), d in sorted(seen.items(), key=lambda x: (x[0][0], x[0][1] or 0))
    ]


def convert(  # noqa: C901
    result,
    *,
    path: str,
    label: str,
    readers,
    now: str | None = None,
    run_id: str | None = None,
    run_uuid: str | None = None,
    consume: bool = False,
    harness_of: Mapping[str, str] | None = None,
    stamps: Mapping[str, Any] | None = None,
    receipt: Callable[[str], dict] | None = None,
) -> Conversion:
    """One `ReadResult` -> the store's tables plus a loss count.

    `consume=True` empties `result.rows` as they are written, so a large read's answers and
    its table rows are not both held at the peak; the result's rows are gone afterwards.

    `run_id` is the content address the raw archive is keyed under; derived here if not passed.
    `harness_of` names the harness of each reader; `stamps` gives the runs row's
    `vocab_version` and `rollup_rev`; `receipt(path)` gives its `copy_*` columns.
    """
    check_schema_agreement()
    if not result.claimed:
        raise ValueError(f"{path}: no reader claimed this folder; there is nothing to convert")

    qindex = _question_index(readers).get(result.reader)
    if qindex is None:
        raise ValueError(f"{path}: read by unknown reader {result.reader!r}")
    reader = {r.name: r for r in readers}[result.reader]

    if not (run_id and run_uuid):
        derived_id, derived_uuid = run_identity(result, path)
        run_id, run_uuid = run_id or derived_id, run_uuid or derived_uuid
    loss = Loss()

    ident = rollout_ident(result, qindex)

    # The key's `agent_ref` is the agent that ran the rollout, else the harness the run names,
    # else "main": the rule prod's keys were minted under, so a reparse keeps them.
    agent_of: dict[Hashable, str] = {}
    run_agent = DEFAULT_AGENT_REF
    for (qname, key), rows in result.rows.items():
        q = qindex.get(qname)
        if q is None or q.lands not in ("rollouts.agent_name", "runs.agent_harness"):
            continue
        for row in rows:
            if row.value is None or not str(row.value).strip():
                continue
            if q.lands == "runs.agent_harness":
                run_agent = str(row.value)
            elif len(key) == 1:
                agent_of[key[0]] = str(row.value)

    # `(task_id, repeat)` is the rollout; `agent_ref` labels it. Two agents on one are refused.
    agent_at: dict[tuple, str] = {}

    def rollout_key(key: tuple) -> dict[str, Any]:
        task_id, repeat = ident(key[0])
        agent = agent_of.get(key[0], run_agent)
        seen = agent_at.setdefault((task_id, repeat), agent)
        if seen != agent:
            raise DuplicateRollout(
                f"{path}: task {task_id!r} repeat {repeat} is read as two rollouts, by agents "
                f"{seen!r} and {agent!r}; a rollout is one task and one repeat"
            )
        return {"run_id": run_id, "task_id": task_id, "repeat": repeat, "agent_ref": agent}

    # Pass 2: every row into a table cell.
    out: dict[str, dict[tuple, dict]] = {t: {} for t in STORE_COLUMNS}

    artifact_seq: dict[tuple, int] = {}

    #: events pkey -> producer call ids this event answers; resolved by `_pair_calls`.
    answered_calls: dict[tuple, list[str]] = {}

    def _write_nested(k: str, value: Any, pk: dict, out, loss, qname: str) -> bool:
        """An events key naming child rows. True if it was one, so the caller drops it."""
        spec = NESTED.get(k)
        if spec is None or not isinstance(value, (list, tuple)):
            return False
        child, index_col = spec
        for i, entry in enumerate(value):
            if not isinstance(entry, Mapping):
                loss.cells_in += 1
                loss.cells_dropped[WRONG_TYPE] += 1
                continue
            ckey = dict(pk)
            ckey[index_col] = i
            entry = dict(entry)
            # Inlined bytes become an artifact back-linked to this event and part.
            art = entry.pop("artifact", None)
            if isinstance(art, Mapping):
                rollout = tuple(pk[c] for c in _ROLLOUT_PK[:4])
                idx = artifact_seq.setdefault(
                    rollout, len([k for k in out["artifacts"] if k[:4] == rollout])
                )
                artifact_seq[rollout] = idx + 1
                akey = {**{c: pk[c] for c in _ROLLOUT_PK[:4]}, "artifact_idx": idx}
                for col, val in art.items():
                    loss.cells_in += 1
                    if cell("artifacts", akey, str(col), val, qname) is None:
                        loss.cells_landed += 1
                for col, val in (
                    ("event_idx", pk.get("event_idx")),
                    (index_col.replace("part_idx", "part_idx"), i),
                ):
                    if val is not None:
                        cell("artifacts", akey, col, val, qname)
                entry["artifact_idx"] = idx
            for col, val in entry.items():
                loss.cells_in += 1
                if cell(child, ckey, str(col), val, qname) is None:
                    loss.cells_landed += 1
        return True

    def _note_answered(k: str, value: Any, pk: dict) -> bool:
        """Record the call ids this event answers. Always False: the key stays a named loss."""
        if k == ANSWERED_CALLS_KEY and isinstance(value, (list, tuple)):
            ids = [str(v) for v in value if v not in (None, "")]
            if ids:
                answered_calls[_event_pkey(pk)] = ids
        return False

    def _pair_calls(out) -> tuple[int, int]:
        """Resolve call-to-result edges into positions; `(resolved, unresolved)`.

        Within a lane only: lanes of one rollout share call ids. Unresolved ids stay null.
        """
        by_lane: dict[tuple, dict[str, tuple[int, int]]] = {}
        for ckey, crow in out["tool_calls"].items():
            src = crow.get("call_id_source")
            if src is None:
                continue
            # ckey is rollout(4) + lane + (event_idx, call_idx)
            by_lane.setdefault(ckey[:5], {}).setdefault(str(src), ckey[5:])
        # Edges implied by landed columns: a tool_result's own id (opensandbox) or its
        # `parent` (agentic-v1) naming the call.
        derived: dict[tuple, list[str]] = {}
        # The first event of each lane carrying an id, and the calls each event made, in order.
        first_with_id: dict[tuple, tuple] = {}
        for pkey, pev in out["events"].items():
            sid = pev.get("source_event_id")
            if sid is not None:
                first_with_id.setdefault((pkey[:5], sid), pkey)
        calls_of_event: dict[tuple, list[str]] = {}
        for lane, calls in by_lane.items():
            for i, w in calls.items():
                calls_of_event.setdefault((lane, w[0]), []).append(i)
        for ekey, ev in out["events"].items():
            if ekey in answered_calls or (ev.get("kind") or "") != "tool_result":
                continue
            calls = by_lane.get(ekey[:5]) or {}
            own = ev.get("source_event_id")
            if own is not None and str(own) in calls:
                derived[ekey] = [str(own)]
                continue
            par = ev.get("parent_event_id")
            if par is None:
                continue
            pkey = first_with_id.get((ekey[:5], par))
            if pkey is not None:
                got = calls_of_event.get((ekey[:5], pkey[5]))
                if got:
                    derived[ekey] = list(got)
        pairs = {**derived, **answered_calls}

        resolved = unresolved = 0
        for ekey, ids in pairs.items():
            calls = by_lane.get(ekey[:5]) or {}
            hit = next(((i, calls[i]) for i in ids if i in calls), None)
            if hit is None:
                unresolved += len(ids)
                continue
            _, (call_ev, call_idx) = hit
            ev = out["events"].get(ekey)
            # An event answering its own calls (an unsplit ATIF step) pairs with nothing else.
            if ev is not None and call_ev != ekey[5]:
                ev["paired_event_idx"] = call_ev
                ev["paired_call_idx"] = call_idx
            for i in ids:
                where = calls.get(i)
                if where is None:
                    unresolved += 1
                    continue
                call_row = out["tool_calls"].get(ekey[:5] + where)
                if call_row is not None:
                    call_row["result_event_idx"] = ekey[5]
                    resolved += 1
        return resolved, unresolved

    def _count_children(out, loss) -> None:
        """Per-event counts over the child parts and tool calls."""
        n_parts: Counter = Counter()
        n_reasoning: Counter = Counter()
        n_image: Counter = Counter()
        text_chars: Counter = Counter()
        reasoning_chars: Counter = Counter()
        encrypted: set = set()
        for pkey, row in out["parts"].items():
            ev = pkey[:-1]
            n_parts[ev] += 1
            chars = row.get("chars") or 0
            text_chars[ev] += chars
            if row.get("channel") == "reasoning":
                n_reasoning[ev] += 1
                reasoning_chars[ev] += chars
            if row.get("part_type") == "image":
                n_image[ev] += 1
            if row.get("part_type") == "encrypted":
                encrypted.add(ev)
        n_calls: Counter = Counter()
        for pkey in out["tool_calls"]:
            n_calls[pkey[:-1]] += 1
        for pkey, row in out["events"].items():
            if out["parts"]:
                row["n_parts"] = n_parts.get(pkey, 0)
                row["n_reasoning_parts"] = n_reasoning.get(pkey, 0)
                row["n_image_parts"] = n_image.get(pkey, 0)
                row["text_chars"] = text_chars.get(pkey, 0)
                row["reasoning_chars"] = reasoning_chars.get(pkey, 0)
                row["has_encrypted_reasoning"] = pkey in encrypted
            if out["tool_calls"]:
                row["n_tool_calls"] = n_calls.get(pkey, 0)

    key_getters: dict[str, Any] = {}

    def row_key(table: str, pk: dict[str, Any]) -> tuple:
        """The table's row identity: rollout, then lane where the table has one (or two lanes'
        index 0 would collide), then its index columns."""
        get = key_getters.get(table)
        if get is None:
            head = _ROLLOUT_PK[: 1 if table == "runs" else 4]
            tail = INDEX_COLUMNS[table]
            has_lane = "lane" in STORE_COLUMNS[table]

            def get(pk, _h=head, _t=tail, _lane=has_lane):
                return (
                    tuple([pk[c] for c in _h])
                    + ((pk.get("lane"),) if _lane else ())
                    + tuple([pk[c] for c in _t])
                )

            key_getters[table] = get
        return get(pk)

    def cell(table: str, pk: dict[str, Any], column: str, value: Any, qname: str) -> str | None:
        """Write one value; None on success, else the drop reason.

        A column found only in the text sibling is written there under the same key.
        """
        sibling = TEXT_SIBLING.get(table)
        if column not in STORE_COLUMNS[table] and sibling and column in STORE_COLUMNS[sibling]:
            table = sibling
        dtype = SCHEMAS[table].get(column)
        if dtype is None:
            return NO_HOME_REASON if f"{table}.{column}" in NO_HOME else NO_COLUMN
        ok, coerced = _coerce_kind(value, _kind(dtype))
        if not ok:
            return WRONG_TYPE
        pkey = row_key(table, pk)
        row = out[table].setdefault(pkey, dict(pk))
        if column in row and row[column] is not None and row[column] != coerced:
            return COLLISION
        row[column] = coerced
        return None

    schema_of: dict[tuple, str] = {}
    #: Fallback record schema when no decoder is declared: the input that read the events.
    input_of: dict[tuple, list[str]] = {}
    lane_input_of: dict[tuple, str] = {}
    scored_by: dict[tuple, str] = {}
    #: The rollout's primary record, as (rank, path, line): main-lane events, then any lane,
    #: then any rollout answer; the earliest line wins within a rank.
    record_of: dict[tuple, tuple[int, str, int]] = {}
    for (qname, key), rows in result.rows.items():
        for row in rows:
            _, _, decoder = row.source.partition("/")
            if len(key) < 1:
                continue
            this = ident(key[0])
            if row.at_path is not None:
                rank = (0 if not _lane_name(row.lane) else 1) if qname == "rollout.events" else 2
                seen = (rank, row.at_path, row.at_line or 0)
                if this not in record_of or seen < record_of[this]:
                    record_of[this] = seen
            if decoder:
                schema_of.setdefault(this, decoder)
            elif qname == "rollout.events" and row.input and not _lane_name(row.lane):
                inputs = input_of.setdefault(this, [])
                if row.input not in inputs:
                    inputs.append(row.input)
            elif (
                qname == "rollout.events"
                and row.input
                and _lane_kind(row.lane) in ROLLUP_LANES_COUNTED
            ):
                lane_input_of.setdefault(this, row.input)
            elif qname == "rollout.score" and row.input:
                # A rollout with no conversation is still a record of the scoring input.
                scored_by.setdefault(this, row.input)

    def record_schema(this) -> str | None:
        inputs = input_of.get(this) or []
        preferred = [i for i in inputs if i in RECORD_SCHEMA_FIRST]
        got = (
            schema_of.get(this)
            or (preferred or inputs or [None])[0]
            or lane_input_of.get(this)
            or scored_by.get(this)
        )
        return RECORD_SCHEMA_SPELLING.get(got, got)

    record_ref = _record_refs(path, record_of, getattr(result, "line_starts", {}))

    rollout_notes = {ident(k0): v for k0, v in _rollout_findings(result).items()}
    provenance = _provenance(result, reader, qindex, run_id, ident)
    keys_read = len({k for (q, k) in result.rows if qindex[q].grain is Grain.ROLLOUT and k})

    for qname, key in sorted(result.rows, key=lambda k: (k[0], str(k[1]))):
        rows = result.rows.pop((qname, key)) if consume else result.rows[(qname, key)]
        q = qindex.get(qname)
        if q is None:
            continue
        table, _, column = q.lands.partition(".")
        for row in rows:
            loss.rows_in += 1
            landed = 0
            landed_null = 0
            reasons: list[str] = []

            # The reader's key past the rollout is the table's index; a rollout-grain
            # stream landing in a lower table uses `position` instead.
            idx = INDEX_COLUMNS[table]
            # Key column: the unnamed lane is "", never NULL (store-v12).
            lane = _lane_name(row.lane) or ""
            lane_kind = _lane_kind(row.lane)
            if table == "runs":
                pk = {"run_id": run_id} if q.grain is Grain.RUN else None
                keyfail = None if pk else UNKEYABLE_GRAIN
            elif len(key) - 1 == len(idx):
                pk = rollout_key(key)
                pk.update(dict(zip(idx, (int(k) for k in key[1:]), strict=True)))
                pk["lane"] = lane
                keyfail = None
            elif len(key) == 1 and len(idx) <= 1:
                if idx and row.position is None:
                    pk, keyfail = None, NO_POSITION
                else:
                    pk = rollout_key(key)
                    pk.update({c: int(row.position) for c in idx})
                    pk["lane"] = lane
                    keyfail = None
            else:
                pk, keyfail = None, UNKEYABLE_INDEX

            values: list[tuple[str, Any]]
            # A mapping spreads across columns only for a whole-table landing.
            if isinstance(row.value, Mapping) and "." not in q.lands:
                values = [(str(k), v) for k, v in row.value.items()]
                if (
                    table == "events"
                    and row.value.get("role") == "assistant"
                    and isinstance(row.value.get("parts"), list)
                ):
                    values = [(k, split_inline_think(v) if k == "parts" else v) for k, v in values]
                if table == "events" and pk is not None:
                    values = [
                        (k, v)
                        for k, v in values
                        if not _write_nested(k, v, pk, out, loss, qname)
                        and not _note_answered(k, v, pk)
                    ]
            elif isinstance(row.value, Mapping) and any(
                f"{column}_{k}" in STORE_COLUMNS[table] for k in row.value
            ):
                # Named column: spread only into declared `<column>_<key>` columns; the
                # rest stays one named loss under `column`.
                values = [
                    (f"{column}_{k}", v)
                    for k, v in row.value.items()
                    if f"{column}_{k}" in STORE_COLUMNS[table]
                ]
                rest = {
                    k: v
                    for k, v in row.value.items()
                    if f"{column}_{k}" not in STORE_COLUMNS[table]
                }
                if rest:
                    values.append((column, rest))
            elif "." not in q.lands:
                values = []
                loss.cells_in += 1
                loss.cells_dropped[SCALAR_IN_TABLE] += 1
                loss.by_question[(qname, SCALAR_IN_TABLE)] += 1
                loss.by_slot[(table, "<the whole row>", SCALAR_IN_TABLE)] += 1
                reasons.append(SCALAR_IN_TABLE)
            else:
                values = [(column, row.value)]

            if table == "events" and pk is not None and lane_kind is not None:
                cell(table, pk, "lane_kind", lane_kind, qname)

            # Per-event source input/file/line; `provenance` cannot answer this per row.
            if table == "events" and pk is not None:
                cell(table, pk, "src_input", row.input, qname)
                if row.at_path is not None:
                    cell(table, pk, "src_path", row.at_path, qname)
                if row.at_line is not None:
                    cell(table, pk, "src_line", row.at_line, qname)

            # Rebase `file://` from the read site to the run root via `Row.at_path`.
            if table == "artifacts" and row.at_path is not None and isinstance(row.value, Mapping):
                if str(row.value.get("uri") or "").startswith("file://"):
                    values = [(c, v) for c, v in values if c != "uri"]
                    values.append(("uri", f"file://{row.at_path}"))

            for col, val in values:
                loss.cells_in += 1
                why = keyfail or cell(table, pk, col, val, qname)
                if why is None:
                    landed += 1
                    if val is None:
                        landed_null += 1
                else:
                    loss.cells_dropped[why] += 1
                    loss.by_question[(qname, why)] += 1
                    loss.by_slot[(table, col, why)] += 1
                    reasons.append(why)

            loss.cells_landed += landed
            loss.cells_landed_null += landed_null
            if landed:
                loss.rows_landed += 1
                if landed == landed_null:
                    loss.rows_landed_all_null += 1
            else:
                loss.rows_dropped[reasons[0] if reasons else NO_COLUMN] += 1

    # Pass 3: the run row and counts over what was written.
    events_per_rollout: Counter = Counter()
    artifacts_per_rollout: Counter = Counter()
    for pkey in out["events"]:
        if _counted(pkey):
            events_per_rollout[_account(pkey)] += 1
    for pkey in out["artifacts"]:
        artifacts_per_rollout[pkey[:4]] += 1

    # 0 = the reader enumerates these and found none; null = it never enumerates them.
    any_events = "rollout.events" in reader.questions
    any_artifacts = "rollout.artifacts" in reader.questions

    # Same event-grain counts as `driver.py`; no `n_reasoning_segments`: it counts parts.
    ast_per_rollout: Counter = Counter()
    calls_per_rollout: Counter = Counter()
    n_events_timed = 0
    # Reasoning share is left null when a rollout has an encrypted part (unknown chars).
    reasoning_chars: Counter = Counter()
    text_chars: Counter = Counter()
    reasoning_parts: Counter = Counter()
    encrypted: set = set()
    for pkey, row in out["parts"].items():
        if not _counted(pkey):
            continue
        who = _account(pkey)
        text_chars[who] += row.get("chars") or 0
        if row.get("channel") == "reasoning":
            reasoning_chars[who] += row.get("chars") or 0
            reasoning_parts[who] += 1
        if row.get("part_type") == "encrypted":
            encrypted.add(who[:4])
    _count_children(out, loss)
    n_paired, n_unpaired = _pair_calls(out)
    # A call whose result was read returned; the producer's own status wins where it wrote one.
    own_calls = [r for pk, r in out["tool_calls"].items() if _counted(pk)]
    for r in own_calls:
        if r.get("result_event_idx") is not None and r.get("status") is None:
            r["status"] = "ok"
    pct_calls_paired = (
        (sum(r.get("result_event_idx") is not None for r in own_calls) / len(own_calls))
        if own_calls
        else None
    )

    ev_prompt: Counter = Counter()
    ev_completion: Counter = Counter()
    truncated: Counter = Counter()
    #: One model invocation's usage is stamped on every event it produced; count it once.
    usage_seen: set = set()
    #: Tokens by account too (prod gdpval 42a332b5: the stream 909,572, its history 110,402,244).
    lane_prompt: Counter = Counter()
    lane_completion: Counter = Counter()
    for pkey, row in out["events"].items():
        if not _counted(pkey):
            continue
        who = pkey[:4]
        req = row.get("request_idx")
        once = (
            (pkey[:5], req)
            if req is not None
            and (row.get("prompt_tokens") is not None or row.get("completion_tokens") is not None)
            else None
        )
        if once is None or once not in usage_seen:
            usage_seen.add(once)
            lane_prompt[_account(pkey)] += row.get("prompt_tokens") or 0
            lane_completion[_account(pkey)] += row.get("completion_tokens") or 0
        if row.get("kind") == "message" and row.get("role") in ("assistant", "agent"):
            ast_per_rollout[_account(pkey)] += 1
        # `n_tool_calls` counts `tool_calls` rows, not tool_call events: one event can hold many.
        if row.get("started_at") is not None:
            n_events_timed += 1
        if row.get("length_truncated"):
            truncated[who] += 1
    ev_prompt, ev_completion = _larger_account(lane_prompt), _larger_account(lane_completion)

    # Record schema = the decoder in `Row.source` (`input/decoder`).
    any_rollout_ts = False
    for pkey in out["tool_calls"]:
        if _counted(pkey):
            calls_per_rollout[_account(pkey)] += 1
    events_per_rollout, ast_per_rollout, calls_per_rollout = (
        _larger_account(events_per_rollout),
        _larger_account(ast_per_rollout),
        _larger_account(calls_per_rollout),
    )
    text_chars, reasoning_chars, reasoning_parts = (
        _larger_account(text_chars),
        _larger_account(reasoning_chars),
        _larger_account(reasoning_parts),
    )

    for pkey, row in out["rollouts"].items():
        row["n_events"] = events_per_rollout.get(pkey, 0) if any_events else None
        row["n_artifacts"] = artifacts_per_rollout.get(pkey, 0) if any_artifacts else None
        row["n_assistant_turns"] = ast_per_rollout.get(pkey, 0) if any_events else None
        row["n_tool_calls"] = calls_per_rollout.get(pkey, 0) if any_events else None
        row["is_scored"] = row.get("reward") is not None
        this = (row["task_id"], row["repeat"])
        if row.get("record_schema") is None:
            row["record_schema"] = record_schema(this)
        if row.get("raw_record_ref") is None:
            row["raw_record_ref"] = record_ref.get(this)
        if any_events:
            row["n_truncated_events"] = truncated.get(pkey, 0)
            # Sum event tokens only where the producer states no rollout total.
            for col, src in (("prompt_tokens", ev_prompt), ("completion_tokens", ev_completion)):
                if row.get(col) is None and src.get(pkey):
                    row[col] = src[pkey]
        row["ingest_status"] = "ok"
        # A reader-supplied status wins; otherwise `failed` iff an error was read.
        if row.get("status") is None:
            row["status"] = "failed" if row.get("error") is not None else "ok"
        if row.get("started_at") is not None or row.get("finished_at") is not None:
            any_rollout_ts = True
        if out["parts"]:
            row["n_reasoning_segments"] = reasoning_parts.get(pkey, 0)
            row["has_encrypted_reasoning"] = pkey in encrypted
            row["reasoning_char_share"] = (
                None
                if pkey in encrypted or not text_chars.get(pkey)
                else reasoning_chars[pkey] / text_chars[pkey]
            )
        notes = rollout_notes.get((row["task_id"], row["repeat"])) or []
        if notes:
            row["ingest_violation"] = " | ".join(notes)[:4000]

    # A rollout seen only via events or artifacts still needs its row.
    for src in ("events", "artifacts"):
        for pkey in out[src]:
            if pkey[:4] not in out["rollouts"]:
                out["rollouts"][pkey[:4]] = {
                    "run_id": pkey[0],
                    "task_id": pkey[1],
                    "repeat": pkey[2],
                    "agent_ref": pkey[3],
                    "ingest_status": "ok",
                    "is_scored": False,
                    "n_events": events_per_rollout.get(pkey[:4], 0) if any_events else None,
                    "n_artifacts": artifacts_per_rollout.get(pkey[:4], 0)
                    if any_artifacts
                    else None,
                    "n_assistant_turns": ast_per_rollout.get(pkey[:4], 0) if any_events else None,
                    "n_tool_calls": calls_per_rollout.get(pkey[:4], 0) if any_events else None,
                    "status": "ok",
                    "record_schema": record_schema((pkey[1], pkey[2])),
                    "raw_record_ref": record_ref.get((pkey[1], pkey[2])),
                }

    # Quarantine: a record the readers could not decode lands as a row.
    decoded = {pk[:4] for t in ("events", "artifacts") for pk in out[t]}
    n_quarantined = 0
    if any_events:
        for pkey, row in out["rollouts"].items():
            if pkey in decoded or row.get("reward") is not None or row.get("error") is not None:
                continue
            ref = record_ref.get((row["task_id"], row["repeat"]))
            if ref is None:
                continue
            row.update(_QUARANTINED, ingest_violation=f"codec miss: {ref}", raw_record_ref=ref)
            n_quarantined += 1
    for rel, line, detail in _unparseable(result, qindex):
        ref = f"{rel}:{_byte_offset(path, rel, line)}"
        qkey = (run_id, f"__quarantined__:{rel}:{line}", 0, DEFAULT_AGENT_REF)
        out["rollouts"][qkey] = {
            "run_id": run_id,
            "task_id": qkey[1],
            "repeat": 0,
            "agent_ref": qkey[3],
            **_QUARANTINED,
            "ingest_violation": detail,
            "raw_record_ref": ref,
        }
        n_quarantined += 1
    decoded_keys = [
        k for k, r in out["rollouts"].items() if r.get("ingest_status") != "quarantined"
    ]

    counts = [events_per_rollout.get(k, 0) for k in decoded_keys]
    ast = [ast_per_rollout.get(k, 0) for k in decoded_keys]
    calls = [calls_per_rollout.get(k, 0) for k in decoded_keys]
    n_events_total = sum(counts)
    shares = [
        r["reasoning_char_share"]
        for r in out["rollouts"].values()
        if r.get("reasoning_char_share") is not None
    ]
    run_row = out["runs"].setdefault((run_id,), {"run_id": run_id})
    run_row.update(
        {
            "run_uuid": run_uuid,
            "label": label,
            "source_path": path,
            **(receipt(path) if receipt else {}),
            "source_format": result.reader,
            "harness": (harness_of or {}).get(result.reader),
            "reader_stack": " / ".join(
                g for g in result.declared_stack if g in {r["producer"] for r in provenance}
            )
            or None,
            "schema_version": SCHEMA_VERSION,
            "vocab_version": (stamps or {}).get("vocab_version"),
            "rollup_rev": (stamps or {}).get("rollup_rev"),
            "ingested_at": now or datetime.now(UTC).isoformat(),
            "n_rollouts": len(decoded_keys),
            "n_quarantined_rollouts": n_quarantined,
            "n_artifacts": len(out["artifacts"]),
            # Null, not 0.0, when the reader never enumerates events.
            "mean_events": (sum(counts) / len(counts)) if (counts and any_events) else None,
            "p95_events": _p95(counts) if any_events else None,
            "mean_assistant_turns": _mean(ast) if (ast and any_events) else None,
            "mean_tool_calls": _mean(calls) if (calls and any_events) else None,
            "pct_events_timed": (n_events_timed / n_events_total) if n_events_total else None,
            "pct_calls_paired": pct_calls_paired,
            "timing_grain": (
                "event"
                if n_events_timed
                else "rollout"
                if any_rollout_ts
                else "run"
                if run_row.get("run_ts")
                else "none"
            ),
            "total_prompt_tokens": sum(
                r.get("prompt_tokens") or 0 for r in out["rollouts"].values()
            ),
            "total_completion_tokens": sum(
                r.get("completion_tokens") or 0 for r in out["rollouts"].values()
            ),
            # Readers do not canonicalise, so `benchmark` is already the producer's string.
            "benchmark_raw": run_row.get("benchmark"),
            "expected_source": _expected_source(result),
            "mean_reasoning_char_share": _mean(shares) if shares else None,
            "n_rollouts_reasoning_share_uncomputable": (
                sum(1 for r in out["rollouts"].values() if r.get("reasoning_char_share") is None)
                if out["parts"]
                else None
            ),
        }
    )

    tables = {
        t: [{c: r.get(c) for c in STORE_COLUMNS[t]} for r in rs.values()] for t, rs in out.items()
    }
    for t, rs in tables.items():
        loss.rows_out[t] = len(rs)

    # Pass 4: provenance, relations, and whether the run is admitted.
    loss.rows_out[PROVENANCE_TABLE] = len(provenance)
    stamped = now or datetime.now(UTC).isoformat()
    relations = _relations(result, run_id, f"reader:{result.reader}", stamped, ident)
    loss.rows_out[RELATIONS_TABLE] = len(relations)

    admitted = bool(out["rollouts"])
    spoke = bool(_rollout_creators(reader) & {r["producer"] for r in provenance})
    kind, why = EMPTY_KIND[(spoke, bool(keys_read))]
    observed = [g for g in result.declared_stack if g in {r["producer"] for r in provenance}]
    record = {
        "run_id": run_id,
        "run_uuid": run_uuid,
        "label": label,
        "source_path": path,
        "reader": result.reader,
        "verdict": VERDICT[(result.ok, admitted)],
        "reason": ("" if admitted else kind),
        "detail": ("" if admitted else why),
        "declared_stack": list(result.declared_stack),
        "observed_producers": observed,
        "silent_producers": [g for g in result.declared_stack if g not in observed],
        "n_rollout_keys_read": keys_read,
        "n_rollouts_landed": len(out["rollouts"]),
        "findings": dict(sorted(Counter(f.kind.value for f in result.findings).items())),
        "finding_lines": [f.line() for f in result.findings],
        # Only refused runs carry provenance here; admitted runs have the table.
        "provenance": ([] if admitted else provenance),
        "ts": now or datetime.now(UTC).isoformat(),
    }
    return Conversion(
        tables=tables,
        loss=loss,
        run_id=run_id,
        run_uuid=run_uuid,
        provenance=provenance,
        relations=relations,
        admitted=admitted,
        record=record,
        calls_paired=n_paired,
        calls_unpaired=n_unpaired,
    )


def _write_parquet(df: pl.DataFrame, path: Path) -> None:
    """`df` as Parquet, its row groups bounded by bytes so no page passes 2 GiB."""
    df.write_parquet(path, row_group_size=parquet_row_group_rows(widest_value(df), 512 * 512))


def write_relations(staged_dir, rows: list[dict]) -> None:
    """Write the relations table into a staged run partition."""
    _write_parquet(
        pl.DataFrame(rows, schema=RELATIONS_SCHEMA), Path(staged_dir) / f"{RELATIONS_TABLE}.parquet"
    )


def write_tables(conv: Conversion, directory) -> None:
    """Write a conversion's tables, provenance and relations as one run's `<table>.parquet` files."""
    for table, rows in conv.tables.items():
        if rows:
            cols = _DECLARED[table]
            _write_parquet(
                pl.DataFrame(
                    [{c: r.get(c) for c in cols} for r in rows], schema=cols, orient="row"
                ),
                Path(directory) / f"{table}.parquet",
            )
    write_provenance(directory, conv.provenance)
    write_relations(directory, conv.relations)


def write_provenance(staged_dir, rows: list[dict]) -> None:
    """Write the provenance table into a staged run partition (committed atomically)."""
    _write_parquet(
        pl.DataFrame(rows, schema=PROVENANCE_SCHEMA),
        Path(staged_dir) / f"{PROVENANCE_TABLE}.parquet",
    )


# --------------------------------------------------------------------------- report


def render_loss(loss: Loss, *, detail: bool = True) -> str:
    lines = [
        f"rows the reader produced        {loss.rows_in:>8}",
        f"rows that reached a table       {loss.rows_landed:>8}"
        f"   (of which every value null: {loss.rows_landed_all_null})",
        f"rows dropped                    {loss.rows_dropped_total:>8}",
    ]
    for reason, n in loss.rows_dropped.most_common():
        tag = "NO_HOME" if reason == NO_HOME_REASON else reason
        lines.append(f"    {n:>8}  {tag}: {REASON_TEXT[reason]}")
    lines += [
        "",
        f"values (cells) the reader produced {loss.cells_in:>8}",
        f"values written to a column         {loss.cells_landed:>8}"
        f"   (of which null: {loss.cells_landed_null})",
        f"values dropped                     {loss.cells_in - loss.cells_landed:>8}",
    ]
    for reason, n in loss.cells_dropped.most_common():
        tag = "NO_HOME" if reason == NO_HOME_REASON else reason
        lines.append(f"    {n:>8}  {tag}: {REASON_TEXT[reason]}")
    lines.append("")
    lines.append("rows written, by table:")
    for t in (*STORE_COLUMNS, PROVENANCE_TABLE):
        n = loss.rows_out.get(t, 0)
        lines.append(f"    {t:<18} {n:>8}" + ("" if n else "   (empty)"))
    if detail and loss.by_slot:
        lines.append("")
        lines.append("what had nowhere to go, by slot (top 30):")
        for (table, col, reason), n in loss.by_slot.most_common(30):
            tag = "NO_HOME" if reason == NO_HOME_REASON else reason
            lines.append(f"    {n:>8}  {table}.{col:<24} {tag}")
    return "\n".join(lines)
