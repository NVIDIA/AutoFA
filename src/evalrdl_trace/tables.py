# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`TablesSource`: a `TraceSource` over one run's tables on disk, read through the schema.

The directory holds one `<table>.parquet` per declared table, as a run's store partition or an
unpacked `trace` export does; a table it lacks reads as empty. Rows come back as the store's
catalog answers them: every declared column, ordered by key, texts joined where asked.
"""

from __future__ import annotations

import base64
import binascii
from pathlib import Path

import polars as pl

from evalrdl_schema import CURRENT
from evalrdl_schema.tables import RunTables

from .source import ArtifactBytes, ArtifactResolver, RolloutKey

#: Events a trace map or timeline lists at most.
EVENT_CAP = 5000
#: Characters of one part's text or one call's arguments a point read returns.
TEXT_CAP = 4000

EVENT_ORDER = ["agent_ref", "lane", "event_idx"]


def _key_sorted(rows: list[dict], keys: list[str]) -> list[dict]:
    rows.sort(key=lambda r: tuple((r[k] is None, "" if r[k] is None else r[k]) for k in keys))
    return rows


def _capped(rows: list[dict], col: str, cap: int) -> list[dict]:
    for r in rows:
        if isinstance(r.get(col), str) and len(r[col]) > cap:
            r[col] = r[col][:cap] + f"… [{len(r[col])} chars]"
    return rows


def folder_resolver(root: Path) -> ArtifactResolver:
    """Reads `file://<path>` artifacts relative to `root`, the run folder the tables were read
    from; any other `uri`, and any path that leaves `root`, reads as None."""
    root = Path(root).resolve()

    def resolve(row: dict) -> bytes | None:
        uri = row.get("uri") or ""
        if not uri.startswith("file://"):
            return None
        path = (root / uri[len("file://") :].lstrip("/")).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            return None
        return path.read_bytes()

    return resolve


def _inline(text: str) -> bytes:
    if text.startswith("data:") and ";base64," in text:
        try:
            return base64.b64decode(text.partition(",")[2], validate=True)
        except (ValueError, binascii.Error):
            pass
    return text.encode("utf-8", "replace")


class TablesSource:
    def __init__(
        self,
        directory: Path,
        *,
        event_cap: int = EVENT_CAP,
        text_cap: int = TEXT_CAP,
        resolver: ArtifactResolver | None = None,
    ):
        self.tables = RunTables(
            Path(directory), {t.name: tuple(t.column_names) for t in CURRENT.tables}
        )
        self.event_cap = event_cap
        self.text_cap = text_cap
        self.resolver = resolver

    # ── reading ──
    def _of(self, table: str, key: RolloutKey, lane: str | None = None) -> pl.LazyFrame:
        lf = self.tables.scan(table).filter(
            (pl.col("run_id") == key.run_id)
            & (pl.col("task_id") == key.task_id)
            & (pl.col("repeat") == key.repeat)
        )
        return lf if lane is None else lf.filter(pl.col("lane") == lane)

    @staticmethod
    def _rows(lf: pl.LazyFrame, order: list[str] | None = None) -> list[dict]:
        if order:
            lf = lf.sort(order, nulls_last=True, maintain_order=True)
        return lf.collect().to_dicts()

    def _texted(self, table: str, texts: str, col: str, on: list[str], key, lane) -> pl.LazyFrame:
        text = self._of(texts, key, lane).select(*on, col)
        return self._of(table, key, lane).join(text, on=on, how="left", nulls_equal=True,
                                                maintain_order="left")  # fmt: skip

    # ── TraceSource ──
    def rollout(self, key: RolloutKey) -> dict | None:
        rows = self._rows(self._of("rollouts", key))
        if not rows:
            return None
        row = rows[0]
        row["attempts"] = self.attempts(key)
        return row

    def attempts(self, key: RolloutKey) -> list[dict]:
        return self._rows(self._of("attempts", key), ["agent_ref", "attempt_idx"])

    def artifacts(self, key: RolloutKey) -> list[dict]:
        return self._rows(self._of("artifacts", key), ["agent_ref", "artifact_idx"])

    def artifact_bytes(self, key: RolloutKey, artifact_idx: int) -> ArtifactBytes | None:
        rows = self._rows(self._of("artifacts", key).filter(pl.col("artifact_idx") == artifact_idx))
        if not rows:
            return None
        row = rows[0]
        if row.get("storage") == "inline":
            texts = self._rows(
                self._of("artifact_texts", key).filter(pl.col("artifact_idx") == artifact_idx)
            )
            text = texts[0].get("text") if texts else None
            data = _inline(text) if isinstance(text, str) else None
        else:
            data = self.resolver(row) if self.resolver and row.get("uri") else None
        return None if data is None else ArtifactBytes(data, row.get("mime"))

    def lanes(self, key: RolloutKey) -> list[dict]:
        rows = (
            self._of("events", key)
            .group_by("agent_ref", "lane")
            .agg(
                pl.len().cast(pl.Int64).alias("n_events"),
                pl.col("event_idx").min().alias("first_event_idx"),
                pl.col("event_idx").max().alias("last_event_idx"),
                pl.col("n_tool_calls").sum().fill_null(0).cast(pl.Int64).alias("n_tool_calls"),
            )
            .collect()
            .to_dicts()
        )
        return _key_sorted(rows, ["agent_ref", "lane"])

    def events(self, key, *, parts, offset=None, limit=None, lane=None) -> list[dict]:
        lf = self._of("events", key, lane).sort(EVENT_ORDER, nulls_last=True)
        if offset or limit is not None:
            lf = lf.slice(int(offset or 0), None if limit is None else int(limit))
        rows = lf.collect().to_dicts()
        if parts:
            prows = self._rows(
                self._texted("parts", "part_texts", "text",
                             ["agent_ref", "lane", "event_idx", "part_idx"], key, lane),
                [*EVENT_ORDER, "part_idx"],
            )  # fmt: skip
            grouped: dict[tuple, list[dict]] = {}
            for p in prows:
                grouped.setdefault((p["agent_ref"], p.get("lane"), p["event_idx"]), []).append(p)
            for ev in rows:
                ev["parts"] = grouped.get((ev["agent_ref"], ev.get("lane"), ev["event_idx"]), [])
        return rows

    def tool_calls(self, key, *, args, lane=None) -> list[dict]:
        lf = (
            self._texted("tool_calls", "tool_call_texts", "args_full",
                         ["agent_ref", "lane", "event_idx", "call_idx"], key, lane)
            if args
            else self._of("tool_calls", key, lane)
        )  # fmt: skip
        return self._rows(lf, [*EVENT_ORDER, "call_idx"])

    def event_parts(self, key: RolloutKey, event_idx: int, lane: str | None = None) -> dict | None:
        at = pl.col("event_idx") == event_idx
        evs = self._rows(self._of("events", key, lane).filter(at), ["agent_ref", "lane"])
        if not evs:
            return None
        parts = self._rows(
            self._texted("parts", "part_texts", "text",
                         ["agent_ref", "lane", "event_idx", "part_idx"], key, lane).filter(at),
            ["agent_ref", "lane", "part_idx"],
        )  # fmt: skip
        calls = self._rows(
            self._texted("tool_calls", "tool_call_texts", "args_full",
                         ["agent_ref", "lane", "event_idx", "call_idx"], key, lane).filter(at),
            ["agent_ref", "lane", "call_idx"],
        )  # fmt: skip
        _capped(parts, "text", self.text_cap)
        _capped(calls, "args_full", self.text_cap)

        def by_lane(rows):
            out: dict[tuple, list[dict]] = {}
            for r in rows:
                out.setdefault((r.get("agent_ref"), r.get("lane")), []).append(r)
            return out

        pl_, cl_ = by_lane(parts), by_lane(calls)
        for ev in evs:
            slot = (ev.get("agent_ref"), ev.get("lane"))
            ev["parts"] = pl_.get(slot, [])
            ev["tool_calls"] = cl_.get(slot, [])
        return {"events": evs, "parts": parts, "tool_calls": calls, "text_cap_chars": self.text_cap}

    def part_text(self, key: RolloutKey, event_idx: int, part_idx: int, lane=None) -> list[dict]:
        at = (pl.col("event_idx") == event_idx) & (pl.col("part_idx") == part_idx)
        rows = self._rows(
            self._texted("parts", "part_texts", "text",
                         ["agent_ref", "lane", "event_idx", "part_idx"], key, lane).filter(at),
            ["agent_ref", "lane"],
        )  # fmt: skip
        return [{"lane": r.get("lane"), "text": r.get("text")} for r in rows]

    def provenance(self, key: RolloutKey) -> list[dict]:
        t, r = pl.col("task_id"), pl.col("repeat")
        lf = self.tables.scan("provenance").filter(
            (pl.col("run_id") == key.run_id)
            & (t.is_null() | ((t == key.task_id) & (r.is_null() | (r == key.repeat))))
        )
        return _key_sorted(lf.collect().to_dicts(), ["task_id", *CURRENT.table("provenance").key])

    def edges(self, key: RolloutKey) -> list[dict]:
        def end(side: str) -> pl.Expr:
            rep = pl.col(f"{side}_repeat")
            return (pl.col(f"{side}_task_id") == key.task_id) & (
                rep.is_null() | (rep == key.repeat)
            )

        lf = self.tables.scan("relations").filter(
            (pl.col("run_id") == key.run_id) & (end("frm") | end("to"))
        )
        return _key_sorted(lf.collect().to_dicts(), list(CURRENT.table("relations").key))

    def trace_map(self, key: RolloutKey) -> dict | None:
        if self.rollout(key) is None:
            return None
        events = self._of("events", key)

        def dist(table: str, col: str) -> dict:
            counts = (
                self._of(table, key).group_by(col).agg(pl.len().alias("n")).collect().to_dicts()
            )
            counts.sort(key=lambda r: (-r["n"], r[col] is None, r[col] or ""))
            return {r[col] or "∅": r["n"] for r in counts}

        return {
            "total_events": events.select(pl.len()).collect().item(),
            "event_cap": self.event_cap,
            "events": self._rows(
                events.select(
                    "agent_ref",
                    "lane",
                    "event_idx",
                    "kind",
                    "kind_raw",
                    "role",
                    "n_parts",
                    "n_reasoning_parts",
                    "n_tool_calls",
                    "completion_tokens",
                    "paired_event_idx",
                    "paired_call_idx",
                )
                .sort(EVENT_ORDER, nulls_last=True)
                .head(self.event_cap)
            ),  # fmt: skip
            "tool_calls": self._rows(
                self._of("tool_calls", key).select(
                    "agent_ref",
                    "lane",
                    "event_idx",
                    "call_idx",
                    "tool_name",
                    "status",
                    "args_head",
                    "result_event_idx",
                ),
                [*EVENT_ORDER, "call_idx"],
            ),  # fmt: skip
            "artifacts": self._rows(
                self._of("artifacts", key).select(
                    "agent_ref",
                    "artifact_idx",
                    "name",
                    "kind",
                    "storage",
                    "size_bytes",
                    "event_idx",
                    "call_idx",
                    "part_idx",
                ),
                ["agent_ref", "artifact_idx"],
            ),  # fmt: skip
            "kind_counts": dist("events", "kind"),
            "part_channels": dist("parts", "channel"),
            "part_origins": dist("parts", "origin"),
        }

    def requests(self, key: RolloutKey) -> list[dict]:
        return self._rows(
            self._of("event_texts", key).filter(pl.col("request_dialect").is_not_null()),
            ["agent_ref", "lane", "event_idx"],
        )

    def prompt_messages(self, key, hashes=None, *, text=False) -> list[dict]:
        lf = self._of("prompt_messages", key)
        if hashes is not None:
            lf = lf.filter(pl.col("msg_hash").is_in(list(hashes)))
        cols = [
            pl.col("msg_hash"),
            pl.col("msg_json").str.len_chars().cast(pl.Int64).alias("chars"),
            pl.col("msg_json").str.json_path_match("$.role").alias("role"),
            pl.col("msg_json").str.json_path_match("$.type").alias("item_type"),
        ]
        if text:
            cols.append(pl.col("msg_json"))
        return self._rows(lf.select(cols))

    def tool_defs(self, key: RolloutKey) -> list[dict]:
        return self._rows(self._of("tool_defs", key).select("sha256", "name"))

    def timeline_rows(self, key: RolloutKey) -> dict | None:
        ros = self._rows(
            self._of("rollouts", key).select("record_schema", "status", "started_at", "finished_at")
        )
        if not ros:
            return None
        events = self._of("events", key)
        return {
            "rollout": ros[0],
            "total_events": events.select(pl.len()).collect().item(),
            "events": self._rows(
                events.select(
                    "agent_ref",
                    "lane",
                    "event_idx",
                    "kind",
                    "role",
                    "started_at",
                    "ended_at",
                    "completion_tokens",
                    "n_tool_calls",
                )
                .sort(EVENT_ORDER, nulls_last=True)
                .head(self.event_cap)
            ),  # fmt: skip
            "tool_calls": self._rows(
                self._of("tool_calls", key).select(
                    "agent_ref",
                    "lane",
                    "event_idx",
                    "call_idx",
                    "tool_name",
                    "status",
                    "started_at",
                    "ended_at",
                    "latency_s",
                    "result_event_idx",
                ),
                [*EVENT_ORDER, "call_idx"],
            ),  # fmt: skip
        }
