# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Restricted Codex sessions for claim compilation and audited evidence review."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from autofa.metrics.claim_contract import contract_behavior_strictness
from autofa.metrics.judge_backend import judge_reasoning_metadata
from autofa.metrics.judge_budget import ReviewBudget, ReviewBudgetError, recovery_prompt
from autofa.metrics.judge_evidence import EvidenceReview


async def _run_turn(**kwargs):
    from autofa.harness.codex_app_server import run_codex_turn

    return await run_codex_turn(**kwargs)


def _completed_text(result, *, stage: str) -> str:
    if result.status != "completed" or result.error:
        message = f"incomplete {stage}: {result.error or result.status}"
        if result.retryable:
            raise ConnectionError(message)
        raise RuntimeError(message)
    if not result.text.strip():
        raise RuntimeError(f"{stage} returned no result")
    return result.text


def _session_metadata(result, *, model: str, effort: str | None) -> dict[str, Any]:
    return {
        "judge_model": model, **judge_reasoning_metadata(harness="codex", effort=effort),
        "session_id": result.thread_id, "turn_id": result.turn_id,
        "subtype": "success" if result.status == "completed" and not result.error else "error",
        "is_error": result.status != "completed" or bool(result.error),
        "stop_reason": result.error_code, "num_turns": 1,
        "usage": result.usage, "cost_usd": getattr(result, "cost_usd", None),
        "cost_kind": getattr(result, "cost_kind", "unknown"),
        "cost_scope": "turn",
    }


async def run_codex_compiler(
    *, prompt: str, model: str, cli_path: str | None, directory: Path, attempt: int,
    system_prompt: str, effort: str | None = None,
) -> str:
    async def deny_tool(name, args):
        return {"success": False,
                "contentItems": [{"type": "inputText", "text": "No tools are available."}]}

    result = await _run_turn(
        prompt=prompt, system_prompt=system_prompt, model=model, cwd=str(directory),
        cli_path=cli_path, effort=None if effort in (None, "default") else effort,
        tools=[], call_tool=deny_tool, restricted=True, timeout_s=180, max_tool_calls=0,
        usage_role="compiler",
    )
    (directory / f"session-{attempt}.json").write_text(
        json.dumps(_session_metadata(result, model=model, effort=effort), indent=2),
        encoding="utf-8",
    )
    return _completed_text(result, stage="claim compilation")


def _codex_tool_result(result: dict[str, Any]) -> dict[str, Any]:
    content = []
    for item in result.get("content", []):
        if item.get("type") == "text":
            content.append({"type": "inputText", "text": item["text"]})
        elif item.get("type") == "image":
            content.append({"type": "inputImage",
                            "imageUrl": f"data:{item['mimeType']};base64,{item['data']}"})
    return {"success": not result.get("isError", False), "contentItems": content}


def _image_tool_description(name: str, description: str) -> str:
    if name != "view_artifact":
        return description
    return description + (
        " IMAGE DELIVERY IN functions.exec: the returned value may be a string containing "
        "metadata followed by a data:image/... URL on its own line, not an object with .content. "
        "Forward the actual image, using: const r = await tools.view_artifact({artifact_idx:0}); "
        "if (typeof r === 'string') { for (const line of r.split('\\n')) { "
        "if (line.startsWith('data:image/')) image(line, 'original'); "
        "else if (line.trim()) text(line); } } "
        "else { for (const p of (Array.isArray(r) ? r : (r.contentItems ?? r.content ?? []))) { "
        "if (p.type === 'inputImage') image(p.imageUrl, 'original'); "
        "else if (p.type === 'image') image(p, 'original'); "
        "else if (p.text) text(p.text); } } "
        "Use the requested artifact_idx/page/member_path in that call. Do not text() or "
        "stringify the whole result, and never enumerate a string with Object.keys/entries: "
        "image data can exceed the text/IPC limit. "
        "A metadata receipt alone is not visual inspection."
    )


async def run_codex_judge_session(
    *, prompt: str, cwd: Path, model: str, cli_path: str | None, review: EvidenceReview,
    max_turns: int = 64, system_prompt: str | None = None, resume: str | None = None,
    effort: str | None = None, budget: ReviewBudget | None = None,
) -> str:
    from autofa.metrics.claude_judge import (
        MAX_COMPLETION_REPAIRS,
        MAX_VERDICT_REPAIRS,
        _async_evidence_tool_result,
        _completion_repair_plan,
        _continue_review,
        _evidence_tool_specs,
        _verdict_repair_prompt,
        judge_system_prompt,
        parse_session_result,
    )

    specs = _evidence_tool_specs(review)
    callbacks = {name: fn for name, _description, _schema, fn in specs}
    tools = [{"name": name, "description": _image_tool_description(name, description),
              "inputSchema": schema}
             for name, description, schema, _fn in specs]
    budget = budget or ReviewBudget(review, max_turns)
    tool_calls = 0
    original_prompt = prompt
    review_lock = asyncio.Lock()

    async def call_tool(name, args):
        nonlocal tool_calls
        budget.charge_tool(name)
        tool_calls += 1
        if name not in callbacks:
            return {"success": False, "contentItems": [
                {"type": "inputText", "text": "Only registered evidence tools are available."},
            ]}
        result = await _async_evidence_tool_result(
            name, callbacks[name], args, lock=review_lock,
        )
        budget.charge_text(result)
        return _codex_tool_result(result)

    async def on_thread(thread_id):
        async with review_lock:
            review.session["session_id"] = thread_id
            review.record("session_init", session_id=thread_id, judge_harness="codex")

    async def on_event(method, params):
        item = params.get("item") or {}
        if method == "thread/compacted" or (
            method == "item/completed" and item.get("type") == "contextCompaction"
        ):
            async with review_lock:
                review.compacted({"method": method, **params})

    from autofa.harness.codex_app_server import CodexTurnConnection

    async with CodexTurnConnection() as connection:
        for attempt in range(MAX_VERDICT_REPAIRS + MAX_COMPLETION_REPAIRS + 1):
            budget.check()
            tool_calls = 0
            start_used = budget.begin_attempt()
            result = await _run_turn(
                prompt=prompt, system_prompt=(judge_system_prompt(contract_behavior_strictness(
                    review.snapshot.get("claim_contract") or {},
                )) if system_prompt is None else system_prompt),
                model=model, cwd=str(cwd), cli_path=cli_path,
                effort=None if effort in (None, "default") else effort,
                tools=tools, call_tool=call_tool, restricted=True, resume=resume,
                on_thread=on_thread, on_event=on_event, max_tool_calls=budget.remaining,
                connection=connection,
                timeout_s=budget.remaining_seconds,
                usage_role="judge",
            )
            if (resume and result.error_code == "restrictedToolsUnsupported"
                    and budget.replacements == 0 and not result.turn_id and tool_calls == 0):
                review.record("isolated_resume_failed", attempt=attempt,
                              **_session_metadata(result, model=model, effort=effort))
                budget.replacement()
                review.start_replacement_context({"reason": "isolated_session_replacement",
                                                  "previous_session_id": result.thread_id})
                continuation = "" if prompt == original_prompt else "\n\n" + prompt
                prompt = recovery_prompt(review, original_prompt + continuation, replacement=True)
                resume = None
                result = await _run_turn(
                    prompt=prompt,
                    system_prompt=(judge_system_prompt(contract_behavior_strictness(
                        review.snapshot.get("claim_contract") or {},
                    )) if system_prompt is None else system_prompt),
                    model=model, cwd=str(cwd), cli_path=cli_path,
                    effort=None if effort in (None, "default") else effort,
                    tools=tools, call_tool=call_tool, restricted=True, resume=None,
                    on_thread=on_thread, on_event=on_event, max_tool_calls=budget.remaining,
                    connection=connection,
                    timeout_s=budget.remaining_seconds,
                    usage_role="judge",
                )
            budget.finish_attempt(start_used, _session_metadata(result, model=model, effort=effort))
            if result.error_code == "toolCallLimitExceeded":
                raise ReviewBudgetError(
                    "review tool-call budget exhausted; remaining evidence is review_incomplete")
            raw = _completed_text(result, stage="judge session")
            budget.final_text(raw)
            (review.directory / f"reply-attempt-{len(budget.attempts) - 1}.txt").write_text(
                raw, encoding="utf-8")
            item_id = review.snapshot["item_id"]
            parsed = parse_session_result(raw, [{"item_id": item_id}],
                                          source=review.fields["detector_source"],
                                          reviews={item_id: review}, harness="codex")
            errors = [row["validation_error"] for row in parsed["verdicts"]
                      if row.get("validation_error")]
            completion = _completion_repair_plan(review, parsed)
            if (budget.remaining <= 0 or not result.thread_id
                    or not _continue_review(review, budget, errors, completion)):
                return raw
            review.record("verdict_validation_failed" if errors else "review_completion_pending",
                          attempt=attempt, errors=errors, completion=completion)
            resume = result.thread_id
            prompt = _verdict_repair_prompt(review, errors, completion=completion, budget=budget)
            (review.directory / f"repair-{attempt + 1}.txt").write_text(prompt, encoding="utf-8")
        raise AssertionError("unreachable Codex judge repair state")
