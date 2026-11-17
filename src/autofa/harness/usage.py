# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Normalize native harness counters without treating unavailable billing as free.

Claude model totals include subagents; its top-level ``usage`` does not. Codex
token notifications and account dollar totals are independent counters. Keep
those streams separate so neither subagents nor resumed history is added twice.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from uuid import uuid4

from autofa.session.usage import (
    CODEX_COMPACTION_USAGE_GAP,
    capture_context,
    record_usage,
    source_cost_kind,
)
from autofa.util.usage_values import field as _get
from autofa.util.usage_values import nonnegative_number as _number
from autofa.util.usage_values import normalize_tokens


class _NativeUsage:
    def __init__(self, *, run_dir=None, role="discovery", model=None, effort=None):
        self.context = capture_context(run_dir, role=role)
        self.role = role
        self.model = model
        self.effort = effort
        self.call_id = uuid4().hex
        self.sequence = 0

    def current_context(self):
        if self.context is None or self.role != "discovery":
            return self.context
        context = capture_context(self.context["run_dir"], role=self.role,
                                  refresh_round=True) or self.context
        return {**context, **{key: self.context[key] for key in (
            "admission_id", "dispatch_identity", "dispatch_started_at") if key in self.context}}

    def record(self, *, spanning=False, **kwargs):
        context = self.current_context()
        if context is None:
            return
        if spanning and self.context["round_index"] != context["round_index"]:
            kwargs["round_span"] = [self.context["round_index"], context["round_index"]]
            context = {**context, "round_index": None}
        self.sequence += 1
        record_usage(context, event_id=f"{self.call_id}:{self.sequence}",
                     effort=self.effort, **kwargs)


class ClaudeUsage(_NativeUsage):
    """One query/print invocation, with durable cumulative session baselines."""

    def __init__(self, *, resume=None, **kwargs):
        super().__init__(**kwargs)
        if self.context is not None:
            self.context["dispatch_identity"] = {"kind": "harness", "provider": "claude",
                                                 "model": self.model, "role": self.role}
        self.session_id = resume or self.call_id
        self.resumed = bool(resume)
        self.saw_result = False
        self.reset = False

    def observe(self, message):
        subtype = _get(message, "subtype")
        if subtype == "init":
            self.session_id = _get(_get(message, "data", {}), "session_id", self.session_id)
        if subtype == "session_reset" or _get(message, "new_conversation_id") is not None:
            self.reset = True
            self.resumed = False
        # Result-only: assistant output counters can be placeholders. Counting
        # assistant + result or parent usage + model_usage duplicates spend.
        if _get(message, "num_turns") is None:
            return
        self.saw_result = True
        self.session_id = _get(message, "session_id") or self.session_id
        status = "failed" if _get(message, "is_error") else "complete"
        if _get(message, "terminal_reason") in {"aborted_streaming", "aborted_tools"}:
            status = "interrupted"
        models = _get(message, "model_usage") or _get(message, "modelUsage")
        cost = _number(_get(message, "total_cost_usd"))
        empty_failure = status != "complete" and cost == 0 and not models
        if isinstance(models, dict) and models:
            for model, usage in models.items():
                self.record(
                    spanning=True, provider="claude", model=model,
                    usage=normalize_tokens(usage, anthropic=True),
                    source_id=f"claude:{self.session_id}:tokens:{model}", cumulative=True,
                    baseline_unknown=self.resumed, reset=self.reset, status=status,
                    measurement="usage",
                    note="Claude cumulative model usage includes subagents.",
                )
        else:
            self.record(
                spanning=True, provider="claude", model=self.model,
                usage=({} if empty_failure else
                       normalize_tokens(_get(message, "usage"), anthropic=True)), status=status,
                measurement="usage",
                note="Claude per-query usage; subagent token breakdown unavailable.",
            )
        self.record(
            spanning=True, provider="claude", model=None,
            cost_usd=None if empty_failure else cost, cost_kind="estimated",
            source_id=f"claude:{self.session_id}:cost", cumulative=True,
            baseline_unknown=self.resumed, reset=self.reset, status=status,
            measurement="cost",
            note="Claude Code client-side cost estimate including subagents; not an invoice. "
                 "Session dollars are not attributed to individual models.",
        )
        self.reset = False

    def finish(self, status="incomplete"):
        if not self.saw_result:
            self.record(
                spanning=True, provider="claude", model=self.model, status=status,
                note="Stream ended without finalized usage; partial assistant output counts "
                     "are placeholders and have not been treated as complete usage.",
            )


async def track_claude_messages(messages: AsyncIterator, **kwargs):
    from autofa.session.admission import paid_dispatch

    tracker = ClaudeUsage(**kwargs)
    status = "incomplete"
    with paid_dispatch(tracker.context):
        try:
            async for message in messages:
                tracker.observe(message)
                yield message
        except (asyncio.CancelledError, KeyboardInterrupt):
            status = "interrupted"
            raise
        except BaseException:
            status = "failed"
            raise
        finally:
            # Preserve the durable reservation and its identity until a missing
            # result is recorded. A kill cannot erase exposure between these steps.
            tracker.finish(status)


class CodexUsage(_NativeUsage):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.context is not None:
            self.context["dispatch_identity"] = {"kind": "harness", "provider": "codex",
                                                 "model": self.model, "role": self.role}
        self.thread_id = ""
        self.token_baseline = False
        self.saw_tokens = False
        self.cost_baseline: dict[str, float] = {}
        self.resumed = False
        self.cost_usd: float | None = None
        self.cost_kind = "unknown"
        self.selected_cost_kind: str | None = None
        self._saw_compaction = False
        # Cost checkpoints advance self.context after a round transition. A
        # completed turn's historical items still belong to the original span.
        self._turn_start_round = self.context["round_index"] if self.context else None

    def observe(self, method: str, params: dict) -> None:
        """Observe counters and disclose compaction work absent from those counters."""
        if method == "thread/tokenUsage/updated":
            self.tokens(params)
        elif method == "thread/compacted" or (
            method in {"item/started", "item/completed"}
            and _get(params.get("item"), "type") == "contextCompaction"
        ):
            self.compaction_gap(params)
        elif method == "turn/completed":
            # Some servers expose only the completed turn's item list. Its
            # compaction time is unknown if the turn advanced discovery rounds.
            for item in _get(params.get("turn"), "items", []) or []:
                if _get(item, "type") == "contextCompaction":
                    self.compaction_gap({**params, "item": item}, turn_summary=True)

    def compaction_gap(self, params: dict, *, turn_summary: bool = False) -> None:
        """Persist an immutable gap identified by a compaction item or legacy turn.

        Codex's public compaction notifications contain identity, not usage.
        Its ordinary token counter can omit compaction tokens. A later token
        notification therefore cannot establish that this work was counted.
        Keep measured counters intact instead of adding guessed token or cost
        deltas. Coalescing also covers modern and legacy duplicate notifications.
        """
        try:
            context = self.current_context()
            if context is None:
                return
            thread = params.get("threadId") or self.thread_id
            turn = params.get("turnId") or _get(params.get("turn"), "id") or self.call_id
            item_id = _get(params.get("item"), "id")
            span = None
            if not item_id and self.role == "discovery":
                # Legacy events cannot distinguish duplicate notifications from
                # multiple compactions within a turn, or locate them on Resume.
                context = {**context, "round_index": None}
            elif turn_summary and self._turn_start_round != context["round_index"]:
                if type(self._turn_start_round) is int and type(context["round_index"]) is int:
                    span = [self._turn_start_round, context["round_index"]]
                context = {**context, "round_index": None}
            source = f"codex:{thread}:{turn}:compaction"
            event_id = f"{source}:item:{item_id}" if item_id else f"{source}:legacy"
            record_usage(
                context, event_id=event_id, provider="codex", model=self.model,
                source_id=source, measurement="usage",
                status=CODEX_COMPACTION_USAGE_GAP, effort=self.effort, round_span=span,
                note="Codex context compaction has no separately attributable token usage "
                     "in the public notifications. Known tokens may omit that work. "
                     "Available cumulative dollars and credits are retained independently; "
                     "no additional cost or tokens have been inferred.",
            )
            # This flag only avoids a redundant generic finish notice. Every
            # notification still attempts a durable write, including retries.
            self._saw_compaction = True
        except Exception:
            # Accounting must not turn completed compaction into a failed turn.
            # record_usage also preserves ordinary ledger-write failures as gaps.
            pass

    def tokens(self, params):
        usage = params.get("tokenUsage") or {}
        total = normalize_tokens(usage.get("total"))
        if not total:
            return
        source = f"codex:{params.get('threadId') or self.thread_id}:tokens"
        if not self.token_baseline:
            last = normalize_tokens(usage.get("last"))
            # Resumed threads may contain historical spend; `last` identifies
            # their new completion. Fresh threads have a known zero baseline,
            # including earlier new requests whose notifications were not seen.
            self.record(provider="codex", model=self.model,
                        usage={key: max(0, value - last.get(key, 0)) if self.resumed else 0
                               for key, value in total.items()},
                        source_id=source, cumulative=True, baseline=True, measurement="usage")
            self.token_baseline = True
        self.saw_tokens = True
        self.record(provider="codex", model=self.model, usage=total,
                    source_id=source, cumulative=True, status="observed", measurement="usage")

    async def account(self, client, *, baseline=False, status="complete") -> bool:
        """Optional read-only accounting RPC; unsupported routes never fail a turn."""
        if self.context is None or not self.thread_id:
            return False
        data = None
        try:
            async with asyncio.timeout(1.0):
                response = await client.request("account/usage/read", {"threadId": self.thread_id})
            data = response.get("threadUsage")
        except Exception:
            pass
        if not isinstance(data, dict) or data.get("threadId") != self.thread_id:
            if not baseline:
                self.record(spanning=True, provider="codex", model=self.model, status=status,
                            measurement="cost",
                            note="Codex account/usage/read did not expose thread dollar usage.")
            return False
        native = _number(data.get("nativeUsageUsdMicros"))
        estimated = _number(data.get("estimatedUsageUsdMicros"))
        if self.selected_cost_kind is None:
            try:
                self.selected_cost_kind = source_cost_kind(
                    self.context, f"codex:{self.thread_id}:account",
                )
            except Exception:
                self.record(spanning=True, provider="codex", model=self.model, status=status,
                            measurement="cost", note="Saved Codex dollar counter could not be "
                            "read; switching billing counters would risk double counting.")
                return False
            if self.selected_cost_kind is None:
                if native is not None:
                    self.selected_cost_kind = "reported"
                elif estimated is not None:
                    self.selected_cost_kind = "estimated"
        kind = self.selected_cost_kind or "unknown"
        # Native dollars and credit-derived estimates are incomparable counters.
        # Keep the first selected route across turns/resumes. Temporarily missing
        # native dollars must not switch to estimates, then charge both later.
        dollars = native if kind == "reported" else estimated if kind == "estimated" else None
        dollars = dollars / 1_000_000 if dollars is not None else None
        credits = _number(data.get("estimatedUsageCreditsMicros"))
        credits = credits / 1_000_000 if credits is not None else None
        if baseline and dollars is not None:
            self.cost_baseline[kind] = dollars
        elif dollars is not None and (not self.resumed or kind in self.cost_baseline):
            self.cost_usd = max(0.0, dollars - self.cost_baseline.get(kind, 0.0))
            self.cost_kind = kind
        self.record(
            spanning=not baseline, provider="codex", model=None,
            cost_usd=dollars, cost_kind=kind if dollars is not None else "unknown",
            credits=credits, credit_source_id=f"codex:{self.thread_id}:account",
            source_id=f"codex:{self.thread_id}:account:{kind}", measurement="cost",
            cumulative=True, baseline=baseline,
            baseline_unknown=self.resumed and kind not in self.cost_baseline, status=status,
            note="Codex thread account usage; dollar availability depends on billing route.",
        )
        return dollars is not None

    def finish(self, status):
        if not self.saw_tokens and not self._saw_compaction:
            self.record(provider="codex", model=self.model, status=status,
                        measurement="usage",
                        note="Codex did not emit token usage for this attempt.")
