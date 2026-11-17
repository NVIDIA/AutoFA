# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evidence reviews, judge replies, prefilter workflows and scripted LLM clients."""

import json
from types import SimpleNamespace as NS

import pytest

from autofa.config import Config, RunConfig
from autofa.core.library import FunctionLibrary
from autofa.data.source_completeness import GENERATION_CUTOFF_NOTICE
from autofa.mcp import tools_eval, tools_library
from autofa.mcp.runtime import Runtime, set_runtime
from autofa.metrics import claude_judge
from autofa.metrics.claim_contract import JUDGE_CONTRACT_VERSION, build_claim_contract
from autofa.metrics.judge_evidence import EvidenceReview, make_snapshot

from support.repair_review import ItemsDataset, case

# Whole-trace judge review and completion.
SOURCE = "def detect(item): return True"
DESCRIPTION = "No backup is made anywhere in the run."
QUOTE = "The original input was overwritten."
TEXT = QUOTE + " " + "Remaining events. " * 1000


def _review(tmp_path, *, chars=None, scope="whole_trace", name="review"):
    """Review TEXT, or a ``chars``-long trace opening with QUOTE, under one frozen claim."""
    contract = build_claim_contract(
        description=DESCRIPTION, mechanism="", source=SOURCE,
        defining_clauses=[{
            "id": "no_backup", "source_field": "description", "quote": DESCRIPTION,
            "predicate": DESCRIPTION, "required_review_scope": "whole_trace",
        }],
    )
    if scope != "whole_trace":
        for requirement in contract["requirements"]:
            requirement["required_review_scope"] = scope
    text = TEXT if chars is None else (QUOTE + " Observed event." * (chars // 16 + 1))[:chars]
    return EvidenceReview(make_snapshot(
        {"response_text": text}, item_id="case", source=SOURCE, claim_contract=contract,
    ), tmp_path / name)


def read_ranges(review, ranges):
    """Read every character of ``ranges`` through ``review``, one bounded read at a time."""
    for span in ranges:
        offset = span["start"]
        while offset < span["end"]:
            offset = review.read(span["field"], offset, span["end"] - offset)["end"]


def _reply(review, *, unknown_reason=None, unknown=True, citation_id=None):
    if citation_id is None:
        citation_id = review.cite("response_text", QUOTE)["citations"][0]["citation_id"]
    assessments = []
    for requirement in review.snapshot["claim_contract"]["requirements"]:
        unresolved = unknown and requirement["id"] == "no_backup"
        assessment = {
            "requirement_id": requirement["id"],
            "status": "unknown" if unresolved else "supported",
            "rationale": ("The remaining interval is unread." if unresolved
                          else "Original evidence."),
            "citations": [{"citation_id": citation_id}], "counterevidence": [],
        }
        if unresolved and unknown_reason:
            assessment["unknown_reason"] = unknown_reason
        assessments.append(assessment)
    return json.dumps({"verdicts": [{
        "item_id": "case", "snapshot_id": review.snapshot["snapshot_id"],
        "claim_contract_id": review.snapshot["claim_contract"]["contract_id"],
        "scope": "whole_trace", "claim_assessments": assessments,
    }], "behavior_specificity": 0.8})


# Judge local inspection.
LOCAL_DESCRIPTION = "The final delivered file omits the required content."
FINAL = "<html><section>Cover</section><section>Index</section></html>"


def local_review(tmp_path):
    contract = build_claim_contract(
        description=LOCAL_DESCRIPTION, mechanism="", source=SOURCE,
        defining_clauses=[{
            "id": "final_artifact", "source_field": "description", "quote": LOCAL_DESCRIPTION,
            "predicate": LOCAL_DESCRIPTION, "required_review_scope": "local",
        }],
    )
    text = "An earlier draft was truncated.\n" + "Unrelated earlier work. " * 2000 + FINAL
    return EvidenceReview(make_snapshot(
        {"response_text": text}, item_id="case", source=SOURCE, claim_contract=contract,
    ), tmp_path / "review")


def local_reply(review, *, inspection=None, status="unknown"):
    citation = review.cite("response_text", "An earlier draft was truncated.")["citations"][0]
    assessments = []
    for requirement in review.snapshot["claim_contract"]["requirements"]:
        target = requirement["id"] == "final_artifact"
        assessment = {
            "requirement_id": requirement["id"], "status": status if target else "supported",
            "rationale": "Inspect the final artifact independently of the earlier draft.",
            "citations": [{"citation_id": citation["citation_id"]}], "counterevidence": [],
        }
        if target and status == "unknown":
            assessment["unknown_reason"] = "ambiguous_evidence"
            if inspection is not None:
                assessment["evidence_inspection"] = inspection
        assessments.append(assessment)
    return json.dumps({"verdicts": [{
        "item_id": "case", "snapshot_id": review.snapshot["snapshot_id"],
        "claim_contract_id": review.snapshot["claim_contract"]["contract_id"],
        "scope": "local", "claim_assessments": assessments,
    }], "behavior_specificity": 0.8})


# Judge prefilter workflow.
BAD_SOURCE = "def guess(item, llm_client=None):\n    return 'guess' in item['response_text']\n"
GOOD_SOURCE = "def guess(item, llm_client=None):\n    return 'derive' in item['response_text']\n"


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    monkeypatch.setattr("autofa.config.runs_root", lambda: tmp_path / "runs")
    monkeypatch.setattr(tools_eval, "_llm_adapter", lambda: None)
    monkeypatch.setattr("autofa.metrics.improvements.classify_improvement",
                        lambda **kwargs: {"targets": [], "status": "insufficient_evidence"})
    monkeypatch.setattr(claude_judge, "judge_behavior_with_claude", lambda **kwargs: pytest.fail(
        "Workflow tests must explicitly stub any scheduled judge calls"))
    rt = Runtime(
        dataset=ItemsDataset([case(i, correct=i >= 10, text="I guess." if i < 2 or i >= 12
                                   else "I derive the value.") for i in range(20)]),
        library=FunctionLibrary(),
        run=RunConfig(run_id="prefilter", model_id="m", benchmarks=["b"], parquet_paths=[]),
        config=Config(harness="claude", behavior_judge_model="claude-sonnet-5-5",
                      behavior_strictness="causal"),
    )
    set_runtime(rt)
    return rt


def submit(source=BAD_SOURCE, *, kind="regex"):
    return tools_library.submit_function(
        name="guess", source=source, description="The assistant guesses a value.",
        mechanism="The assistant supplies a value without showing its derivation.",
        kind=kind,
    )


def judge_result(fn, items):
    from autofa.metrics.association import selected_bucket

    bucket, mode = selected_bucket(fn.scores)
    contract = build_claim_contract(source=fn.source, description=fn.description,
                                    mechanism=fn.mechanism)
    snapshots = {
        item["_item_id"]: make_snapshot(
            item, item_id=item["_item_id"], source=fn.source,
            failure_localisation=fn.failure_localisations.get(item["_item_id"]),
            claim_contract=contract,
        )["snapshot_id"] for item in items
    }
    return {
        "backend": "claude_agent_sdk", "judge_contract_version": JUDGE_CONTRACT_VERSION,
        "judge_policy_hash": claude_judge.judge_policy_hash(),
        "claim_hash": contract["contract_id"], "claim_contract": contract,
        "n_judged": len(items), "n_yes": len(items), "n_no": 0,
        "n_uncertain": 0, "n_errors": 0, "n_completed": len(items),
        "behavior_purity": 1.0, "behavior_specificity": .8,
        "verdicts": [{
            "item_id": item["_item_id"], "verdict": "yes", "completion_status": "complete",
            "assigned_label": bucket if mode == "exact" and bucket != "True" else True,
            "verification_status": "confirmed", "claim_contract_id": contract["contract_id"],
            "snapshot_id": snapshots[item["_item_id"]],
            "initial_review": {
                "item_id": item["_item_id"], "verdict": "yes", "completion_status": "complete",
                "claim_contract_id": contract["contract_id"],
                "snapshot_id": snapshots[item["_item_id"]],
            },
        } for item in items],
    }


# Reasoning trace that ends at a provider generation cutoff, as imports normalize it.
def _trace(*, answer=False):
    def event(index, kind, role, text, channel):
        return {
            "event_idx": index, "lane": "", "kind": kind, "role": role, "n_parts": 1,
            "parts": [{"event_idx": index, "lane": "", "part_idx": 0, "part_type": "text",
                       "channel": channel, "text": text, "chars": len(text)}],
        }

    events = [
        event(0, "message", "user", "Return a final answer.", "content"),
        event(1, "reasoning", "output", "I will output A. Reconsider. " * 40_000, "reasoning"),
    ]
    events[1].update(length_truncated=True, finish_reason="incomplete", completion_tokens=100000)
    if answer:
        events.append(event(2, "message", "assistant", "B", "content"))
    return {
        "steps": [{"source": "user", "message": "Return a final answer."},
                  {"source": "assistant", "reasoning_content": events[1]["parts"][0]["text"]}],
        "evidence_scope": {"source": "original API event parts and tool arguments",
                           "event_lane": "", "limitations": {},
                           "notices": {GENERATION_CUTOFF_NOTICE: 1}},
        "execution": {
            "rollout": {"n_events": len(events), "n_assistant_turns": int(answer),
                        "n_reasoning_segments": 1, "n_tool_calls": 0, "completion_tokens": 100000},
            "events": events, "tool_calls": [],
            "availability": {"events": {
                "status": "available",
                "reason": "Exported table retrieved completely; evaluation records are excluded.",
            }},
        },
    }


# Cheating check.
def cheating_review(tmp_path, text="ran git log --all --grep=1234 then git show abc"):
    item = {"prompt_text": "Fix issue 1234", "response_text": text}
    return EvidenceReview(make_snapshot(item, item_id="m|b|p|0", source=""), tmp_path / "review")


# Scripted chat-completions clients for HubLLM tests.
def response(content, finish="stop", **message):
    return NS(choices=[NS(finish_reason=finish, message=NS(content=content, **message))])


def client_for(*responses):
    """Return a client replying in order; an exception is raised, a callable is called."""
    calls = []
    replies = iter(responses)

    def complete(**kwargs):
        calls.append(kwargs)
        reply = next(replies)
        if isinstance(reply, BaseException):
            raise reply
        return reply() if callable(reply) else reply

    return NS(chat=NS(completions=NS(create=complete))), calls
