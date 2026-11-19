# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared evidence-work accounting for independent judge reviews.

The limits bound local tool work, delivered text and elapsed time. They are not
provider billing-token limits: neither harness can enforce those mid-request.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any

from autofa.metrics.judge_evidence import MAX_READ_CHARS, EvidenceReview

MAX_TOOL_CALLS = 512
MAX_TOTAL_TOOL_CALLS = 2 * MAX_TOOL_CALLS
MAX_TEXT_CHARS = 8_000_000
MAX_FINAL_CHARS = 128_000
MAX_SECONDS = 600
MAX_TRANSPORT_RETRIES = 2
MAX_REPLACEMENTS = 1


class ReviewBudgetError(RuntimeError):
    """Required frozen evidence work cannot fit the bounded review allowance."""

    def __init__(self, message: str, *, capacity: bool = False):
        super().__init__(message)
        self.capacity = capacity


def review_budget_failure(reason: Any) -> str | None:
    """Use typed current errors and one compatibility parser for saved text receipts."""
    if isinstance(reason, ReviewBudgetError):
        return "capacity" if reason.capacity else "attempt"
    text = str(reason).lower()
    if "review workload exceeds bounded capacity before model dispatch" in text:
        return "capacity"
    if any(term in text for term in (
        "review tool-call budget exhausted", "review delivered-text budget exhausted",
        "review elapsed-time budget exhausted", "review final-output limit exceeded",
        "review isolated replacement limit exhausted",
    )):
        return "attempt"
    return None


@dataclass(frozen=True)
class ReviewWorkload:
    scope: str
    evidence_chars: int
    required_read_calls: int
    reread_calls: int
    citation_calls: int
    artifact_calls: int
    navigation_calls: int
    finalization_calls: int
    tool_call_limit: int
    text_char_limit: int = MAX_TEXT_CHARS
    final_char_limit: int = MAX_FINAL_CHARS
    timeout_seconds: int = MAX_SECONDS


def plan_review_budget(review: EvidenceReview, max_turns: int = 64) -> ReviewWorkload:
    """Plan from projected evidence, before dispatch; derived duplicates are excluded.

    ``max_turns`` remains the baseline allowance. Additional calls are justified
    by the frozen workload, up to a fixed shared ceiling. Local predicates do not
    acquire an exhaustive-read requirement merely because a trace is large.
    A bounded interval is located only when review_interval binds it during the
    review, so its full-trace reservation is an upper bound: it is capped at the
    shared ceiling and is never by itself a reason to refuse dispatch.
    """
    if max_turns < 1 or max_turns > MAX_TOOL_CALLS:
        raise ReviewBudgetError(f"review baseline must be between 1 and {MAX_TOOL_CALLS}",
                                capacity=True)
    requirements = review.snapshot.get("claim_contract", {}).get("requirements", [])
    scopes = {row.get("required_review_scope", "local") for row in requirements}
    scope = (
        "whole_trace"
        if "whole_trace" in scopes
        else ("bounded_interval" if "bounded_interval" in scopes else "local")
    )
    # A bounded clause is supported only through an interval receipt, which needs a
    # certified event order. Without one, reading the whole trace cannot support the
    # clause, so its evidence reads are planned like a local clause.
    reserve_trace = scope == "whole_trace" or (
        scope == "bounded_interval" and review.interval_navigation()["available"]
    )
    source = review.fields["detector_source"]
    fields = [
        value
        for field, value in review.fields.items()
        if field != "detector_source" and review.snapshot["manifest"][field]["origin"] == "stored"
    ]
    source_reads = math.ceil(len(source) / MAX_READ_CHARS)
    full_reads = sum(math.ceil(len(value) / MAX_READ_CHARS) for value in fields)
    # Until semantic interval boundaries are bound, navigation may need the full
    # source. This reserves capacity; it does not broaden the frozen predicate.
    reads = (full_reads if reserve_trace else min(full_reads, 8)) + source_reads
    chars = len(source) + (
        sum(len(value) for value in fields)
        if reserve_trace
        else min(sum(len(value) for value in fields), 8 * MAX_READ_CHARS)
    )
    clauses = sum(row.get("kind") == "defining_clause" for row in requirements)
    citations = max(2, 2 * clauses)
    artifact_calls = 0
    for artifact in review._artifact_catalog():
        inspection = review.artifact_inspections.get(artifact.get("artifact_idx"), {})
        text_pages = sum(
            math.ceil(len(row.get("text", "")) / MAX_READ_CHARS)
            for row in inspection.get("segments", [])
        )
        visual_pages = inspection.get("visual_pages")
        # Unknown artifacts have a bounded discovery reserve, not an assertion
        # that four calls guarantee complete inspection of an arbitrary archive.
        artifact_calls += 2 + max(
            2, text_pages + (visual_pages if type(visual_pages) is int else 0)
        )
    if scope == "local":
        artifact_calls = min(artifact_calls, 8)
    bounded = sum(row.get("required_review_scope") == "bounded_interval" for row in requirements)
    navigation = 2 + 6 * bounded
    # Pagination estimates describe a perfect single pass. Partial pages,
    # boundary context and refreshed decisive citations require additional reads.
    rereads = math.ceil(reads / 2) if reserve_trace else 0
    finalization = 8
    required = reads + rereads + citations + artifact_calls + navigation + finalization
    limit = max(max_turns, required)
    over = limit > MAX_TOOL_CALLS or chars + required * 512 > MAX_TEXT_CHARS
    fixed = source_reads + citations + artifact_calls + navigation + finalization
    if over and scope == "bounded_interval" and reserve_trace and fixed <= MAX_TOOL_CALLS:
        # Only the bound interval's required ranges must be read. SDK turn and tool
        # limits are fixed when an attempt starts, so fund the shared ceiling now.
        # Runtime charges still stop at it, and validation still rejects support for
        # any unread required range. Work outside the interval must still fit.
        limit, over = MAX_TOOL_CALLS, False
    if over:
        raise ReviewBudgetError(
            f"review workload exceeds bounded capacity before model dispatch: scope={scope}, "
            f"projected_chars={chars}, required_read_calls={reads}, "
            f"planned_tool_calls={limit}, tool_cap={MAX_TOOL_CALLS}, "
            f"text_cap={MAX_TEXT_CHARS}. Preserve the frozen claim/cohort and completed "
            "reviews; inspect the evidence projection or explicitly configure a supported "
            "review capacity before retrying. Unread evidence is review_incomplete.",
            capacity=True,
        )
    return ReviewWorkload(
        scope, chars, reads, rereads, citations, artifact_calls, navigation, finalization, limit
    )


class ReviewBudget:
    """One allowance across repairs/reconnects, with at most one explicit replacement."""

    def __init__(self, review: EvidenceReview, max_turns: int = 64):
        self.review = review
        self.plan = plan_review_budget(review, max_turns)
        self.limit = self.plan.tool_call_limit
        self.used = 0
        self.text_chars = 0
        self.replacements = 0
        self.transport_retries = 0
        self.started = time.monotonic()
        self.attempts: list[dict[str, Any]] = []
        self.pending_start: int | None = None
        self.verdict_repairs = 0
        self.completion_repairs = 0
        self.last_completion_progress = None
        review.record("review_budget", **asdict(self.plan))

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)

    @property
    def remaining_seconds(self) -> float:
        return max(0, self.plan.timeout_seconds - (time.monotonic() - self.started))

    def check(self) -> None:
        if self.remaining_seconds <= 0:
            raise ReviewBudgetError(
                "review elapsed-time budget exhausted; preserve frozen review progress"
            )
        if self.text_chars >= self.plan.text_char_limit:
            raise ReviewBudgetError(
                "review delivered-text budget exhausted; remaining evidence is review_incomplete"
            )

    def charge_tool(self, name: str) -> None:
        self.check()
        if self.remaining <= 0:
            raise ReviewBudgetError(
                "review tool-call budget exhausted; remaining evidence is review_incomplete"
            )
        self.used += 1
        self.review.record("review_budget_tool", tool=name, **self.metadata())

    def charge_text(self, result: dict[str, Any]) -> None:
        self.text_chars += sum(len(item.get("text", "")) for item in result.get("content", []))
        self.check()

    def begin_attempt(self) -> int:
        self.check()
        self.pending_start = self.used
        return self.used

    def finish_attempt(self, start_used: int, session: dict[str, Any]) -> dict[str, Any]:
        self.pending_start = None
        actual_tool_calls = self.used - start_used
        # A no-tool model invocation still consumes one work unit, preventing
        # malformed-output or connection retries from becoming free dispatches.
        if self.used == start_used:
            self.used += 1
        session = {
            **session,
            "tool_calls": actual_tool_calls,
            "budget_used": self.used - start_used,
            **self.metadata(),
        }
        self.attempts.append(session)
        self.review.session = {**session, "attempts": list(self.attempts)}
        self.review.record("session_result", attempt=len(self.attempts) - 1, **session)
        return session

    def final_text(self, text: str) -> None:
        if len(text) > self.plan.final_char_limit:
            raise ReviewBudgetError(
                "review final-output limit exceeded; preserve the frozen claim and saved evidence"
            )
        self.text_chars += len(text)
        self.check()
        self.review.session.update(self.metadata())
        self.review.record("review_budget_output", characters=len(text), **self.metadata())

    def replacement(self) -> None:
        if self.replacements >= MAX_REPLACEMENTS:
            raise ReviewBudgetError("review isolated replacement limit exhausted")
        self.check()
        self.replacements += 1
        # Explicitly reserve one new context's required reads; time/text limits
        # are shared and never replenished. No ordinary reconnect grants calls.
        self.limit = min(MAX_TOTAL_TOOL_CALLS, self.used + self.plan.tool_call_limit)
        self.review.record(
            "isolated_replacement_budget",
            replacement_calls=self.plan.tool_call_limit,
            **self.metadata(),
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "budget_total_used": self.used,
            "budget_limit": self.limit,
            "budget_remaining": self.remaining,
            "budget_text_chars": self.text_chars,
            "budget_text_limit": self.plan.text_char_limit,
            "budget_replacements": self.replacements,
            "budget_transport_retries": self.transport_retries,
            "budget_hard_tool_limit": MAX_TOTAL_TOOL_CALLS,
        }


def recovery_prompt(review: EvidenceReview, original_prompt: str, *, replacement: bool) -> str:
    context = "replacement isolated session" if replacement else "resumed session"
    # Notes and receipts are navigation. Never turn previously inlined evidence
    # into a new read receipt by replaying an old prompt across a new epoch.
    return (
        f"Continue the same frozen evidence review in a {context}. "
        "The original frozen claim, cohort and completed judgments are unchanged. "
        "Use review_status for saved notes and read receipts. Reread decisive evidence "
        "before citing it. "
        + (
            "This fresh context must read every original required scope again; historical "
            "receipts and notes are navigation, not evidence delivered here. "
            if replacement
            else ""
        )
        + "No transport failure is a detector counterexample.\n\n"
        + original_prompt
    )


def missing_resume_session(error: Exception) -> bool:
    """Only explicit missing-session failures permit dropping a resume handle."""
    message = str(error).lower()
    return any(
        value in message
        for value in (
            "session not found",
            "thread not found",
            "no conversation found",
            "session does not exist",
            "thread does not exist",
            "threadnotfound",
            "sessionnotfound",
        )
    )
