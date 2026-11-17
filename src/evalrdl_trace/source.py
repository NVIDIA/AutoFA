# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`TraceSource`: what a store answers about one rollout.

Every method takes the rollout's key and returns rows as plain dicts, ready to serialize. A
method for a rollout the store does not have returns `None` where it returns one object, and an
empty list where it returns rows.
"""

from __future__ import annotations

from typing import NamedTuple, Protocol


class RolloutKey(NamedTuple):
    run_id: str
    task_id: str
    repeat: int


class TraceSource(Protocol):
    #: How many events `trace_map` and `timeline_rows` return at most.
    event_cap: int

    def rollout(self, key: RolloutKey) -> dict | None: ...

    def lanes(self, key: RolloutKey) -> list[dict]:
        """One row per (agent_ref, lane) of the rollout's events, with its event range."""

    def events(
        self,
        key: RolloutKey,
        *,
        parts: bool,
        offset: int | None = None,
        limit: int | None = None,
        lane: str | None = None,
    ) -> list[dict]: ...

    def event_parts(self, key: RolloutKey, event_idx: int, lane: str | None = None) -> dict | None:
        """One event's rows, parts and tool calls; `lane=None` matches every lane numbering an
        event `event_idx`."""

    def part_text(
        self, key: RolloutKey, event_idx: int, part_idx: int, lane: str | None = None
    ) -> list[dict]:
        """`lane` and the whole, uncapped `text` of each part numbered `part_idx` in an event
        numbered `event_idx`: one row per lane that has it, every lane when `lane=None`."""

    def tool_calls(self, key: RolloutKey, *, args: bool, lane: str | None = None) -> list[dict]: ...

    # Optional: `search(key, q, limit) -> list[dict]`, the parts and tool call arguments holding
    # every word of `q`, in trace order, each `kind` ("part" | "call"), `lane`, `event_idx`,
    # `idx` (part or call index) and a `snippet` around the match. A store without it 404s.

    def attempts(self, key: RolloutKey) -> list[dict]: ...

    def provenance(self, key: RolloutKey) -> list[dict]:
        """The run's provenance rows narrowed to this rollout, and the run-grain rows."""

    def edges(self, key: RolloutKey) -> list[dict]:
        """Every stated edge with either end in this rollout, resolved or not."""

    def trace_map(self, key: RolloutKey) -> dict | None: ...

    def timeline_rows(self, key: RolloutKey) -> dict | None:
        """`rollout`, `total_events`, `events` and `tool_calls`, as `assemble.timeline` reads
        them."""

    def requests(self, key: RolloutKey) -> list[dict]:
        """The `event_texts` rows of the rollout's events that carry a model call's request:
        `agent_ref`, `lane`, `event_idx`, `request_dialect`, `request_messages`, `request_tools`,
        `request_params`. Empty where the store keeps no requests."""

    def prompt_messages(
        self, key: RolloutKey, hashes: list[str] | None = None, *, text: bool = False
    ) -> list[dict]:
        """The rollout's `prompt_messages`, or those named in `hashes`: `msg_hash`, `chars` (of
        `msg_json`), `role`, `item_type`, and `msg_json` itself when `text`."""

    def tool_defs(self, key: RolloutKey) -> list[dict]:
        """`sha256` and `name` of each tool definition the rollout's model was offered."""


class ArtifactBytes(NamedTuple):
    data: bytes
    #: The row's `mime`, else None; the view route decides image-ness from the bytes.
    media_type: str | None


class ArtifactResolver(Protocol):
    def __call__(self, row: dict) -> bytes | None:
        """The bytes of one `artifacts` row that is not stored inline, read from where its `uri`
        points; None where this host cannot reach them."""


class ArtifactSource(Protocol):
    """What the artifact routes read: a rollout's `artifacts` rows and their bytes."""

    def artifacts(self, key: RolloutKey) -> list[dict]: ...

    def artifact_bytes(self, key: RolloutKey, artifact_idx: int) -> ArtifactBytes | None:
        """None where the rollout has no such artifact or its bytes cannot be read."""
