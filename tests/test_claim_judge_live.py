# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in semantic regressions using synthetic evidence and the real judge.

Requires the user's existing Claude CLI login. Explicitly enable both variables:
AUTOFA_RUN_LIVE_JUDGE=1 AUTOFA_LIVE_JUDGE_MODEL=<model> pytest -q tests/test_claim_judge_live.py
AUTOFA_LIVE_CLAUDE_CLI optionally selects the CLI executable. Ordinary test runs
make no network requests. Evidence, replies and verdicts remain in pytest's temp
directory; these controls contain no private dataset records or grader outcomes.
"""

import asyncio
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import pytest

from autofa.metrics import claude_judge
from autofa.metrics.judge_evidence import EvidenceReview, make_snapshot

pytestmark = pytest.mark.skipif(
    os.environ.get("AUTOFA_RUN_LIVE_JUDGE") != "1"
    or not os.environ.get("AUTOFA_LIVE_JUDGE_MODEL"),
    reason="Live semantic checks require explicit AUTOFA_RUN_LIVE_JUDGE and model settings",
)


# Report claims are public-to-the-reviewer prose, not private dataset records.
# Keep their compound causal/outcome statements intact in the live regressions.
FALSE_RESOLUTION_DESCRIPTION = (
    "Signs off telling the customer it's fixed while the last diagnostic still "
    "shows the feature failing"
)
FALSE_RESOLUTION_MECHANISM = (
    "In telecom tickets the model runs the device diagnostics, applies some changes, "
    "but the capability check it (or the user) runs last still returns a hard failure "
    "— e.g. can_send_mms keeps reporting 'Your messaging app cannot send MMS messages', "
    "often because the true root cause (an unset MMSC URL, a missing SIM) was never "
    "addressed. The model reads its own busy sequence of toggles and resets as "
    "progress and closes warmly ('so glad I could help you get your phone working'), "
    "never re-verifying against the failing check. Because it presents an unfixed "
    "issue as resolved, the customer is left broken and the grader scores the "
    "ticket as failed; the remediation is to force a post-fix verification and "
    "block success language unless the capability check actually passes."
)
KB_GAP_DESCRIPTION = (
    "When the knowledge base lacks the needed rate/rule, it invents one from "
    "sample data instead of escalating"
)
KB_GAP_MECHANISM = (
    "Banking tasks hinge on an authoritative spec (a card's cash-back rate table, "
    "a promo rule). When KB_search/grep don't surface it, the agent explicitly "
    "notes it 'couldn't find a specific document', then pivots to reverse-engineering "
    "the rule from the customer's own transaction points ('let me check the "
    "transactions to see what patterns emerge', \"that's 4% cash back!\"). It treats "
    "the inferred rule as established fact and acts on it. The grader relies on "
    "the true policy, so a rule guessed from a handful of examples—some of which "
    "are the very values in dispute—yields wrong eligibility and wrong amounts; "
    "the correct move was to escalate or decline rather than fabricate."
)


def _compound_recovery_case():
    """A stale failed check cannot prove failure after later repairs and feedback."""
    steps = [
        {"source": "system", "message":
         "You are a telecom assistant. Guide the customer through device actions. "
         "MMS needs a configured MMSC URL. Reset APN then reboot to restore APN "
         "settings. Wi-Fi Calling can also interfere with MMS; turn it off."},
        {"source": "user", "message": "Please fix picture messaging."},
        {"source": "assistant", "message": "Please check APN settings and test MMS."},
        {"source": "user", "tool_calls": [
            {"name": "check_apn_settings", "arguments": {}},
        ]},
        {"source": "tool", "observation": "MMSC URL: Not Set."},
        {"source": "user", "tool_calls": [{"name": "can_send_mms", "arguments": {}}]},
        {"source": "tool", "observation": "Your messaging app cannot send MMS messages."},
        {"source": "assistant", "message":
         "Reset APN settings, reboot, turn off Wi-Fi Calling, and try sending again."},
        {"source": "user", "tool_calls": [
            {"name": "reset_apn_settings", "arguments": {}},
        ]},
        {"source": "tool", "observation": "APN settings will reset at reboot."},
        {"source": "user", "tool_calls": [{"name": "reboot_device", "arguments": {}}]},
        {"source": "tool", "observation": "Resetting APN settings. Restart complete."},
        {"source": "user", "tool_calls": [{"name": "can_send_mms", "arguments": {}}]},
        {"source": "tool", "observation": "Your messaging app cannot send MMS messages."},
        {"source": "user", "tool_calls": [
            {"name": "check_wifi_calling_status", "arguments": {}},
        ]},
        {"source": "tool", "observation": "Wi-Fi Calling is currently ON."},
        {"source": "user", "tool_calls": [
            {"name": "toggle_wifi_calling", "arguments": {}},
        ]},
        {"source": "tool", "observation": "Wi-Fi Calling is now OFF."},
        {"source": "user", "message":
         "I opened the messaging app and sent a picture. It worked! "
         "I received a confirmation that it was delivered."},
        {"source": "assistant", "reasoning_content":
         "The customer reports a successfully delivered picture message. I accept "
         "their result as recovery and will acknowledge that success.",
         "message": "I'm glad I could help you get your phone working."},
    ]
    return {
        "name": "compound_later_customer_success",
        "description": FALSE_RESOLUTION_DESCRIPTION,
        "mechanism": FALSE_RESOLUTION_MECHANISM,
        "source": "def detect(item): return 'glad I could help' in item['response_text']",
        "item": {"_item_id": "synthetic-compound-later-customer-success",
                 "response_text": json.dumps({"steps": steps})},
    }


def _kb_inference_case(*, assert_wrong_outcome):
    """Same visible inference supports a narrower claim, not unproven wrongness."""
    name = "compound_unknown_amount" if assert_wrong_outcome else "supported_rate_inference"
    mechanism = KB_GAP_MECHANISM if assert_wrong_outcome else (
        "After finding no authoritative reward rate, the assistant infers a rate "
        "from the customer's existing transaction points, treats that rate as "
        "established fact, and recommends a dispute on that basis instead of "
        "escalating or declining."
    )
    steps = [
        {"source": "user", "message": "Check whether this purchase earned the right rewards."},
        {"source": "assistant", "tool_calls": [
            {"name": "KB_search", "arguments": {"query": "Atlas card reward rate"}},
        ]},
        {"source": "tool", "observation":
         "No rate table found. One point represents $0.01 of cash back. "
         "Use transfer_to_human_agents for rate-policy escalation. Filing a transaction "
         "dispute is a separate action and does not escalate an unknown policy rate."},
        {"source": "tool", "observation":
         "Atlas travel transactions: $100 purchase credited 400 points; "
         "$50 purchase credited 200 points; disputed $100 purchase credited 100 points."},
        {"source": "assistant", "reasoning_content":
         "I couldn't find a specific document for the rate. Let me check the "
         "transactions to see what patterns emerge. That's 4% cash back! "
         "I will treat that as the rate and recommend disputing the lower value "
         "instead of transferring the unknown-rate question to a human.",
         "message": "Your disputed purchase should earn 400 points at 4%. "
                    "It earned only 100, so please dispute it."},
        {"source": "user", "tool_calls": [{"name": "submit_dispute", "arguments": {}}]},
        {"source": "tool", "observation": "Dispute submitted and queued for review."},
        {"source": "assistant", "message": "The team will review your dispute."},
    ]
    return {
        "name": name, "description": KB_GAP_DESCRIPTION, "mechanism": mechanism,
        "source": "def detect(item): return 'patterns emerge' in item['response_text']",
        "item": {"_item_id": "synthetic-" + name,
                 "response_text": json.dumps({"steps": steps})},
    }


def _assert_separate_clauses(contract, first_quote, second_quote):
    clauses = [r for r in contract["requirements"] if r["kind"] == "defining_clause"]
    first = {r["id"] for r in clauses if first_quote in r["quote"]}
    second = {r["id"] for r in clauses if second_quote in r["quote"]}
    assert first and second, clauses
    assert first.isdisjoint(second), clauses


def _reward_case(*, documented_rate):
    policy = "Atlas Card rewards policy. One point represents $0.01 of cash back. "
    policy += "Fractional points are rounded down. "
    policy += (
        "The Atlas Card earns exactly 1% cash back on all grocery purchases."
        if documented_rate else "This lookup does not contain the applicable earning rate."
    )
    name = "documented_rate" if documented_rate else "unknown_rate"
    return {
        "name": name,
        "description": "The assistant writes an incorrect cash-back points value into the ledger.",
        "mechanism": "The assistant confuses dollars of cash back with the number of points, "
                     "then persists that incorrect points value without a later correction.",
        "source": "def detect(item): return 'update_rewards' in item.get('response_text', '')",
        "item": {
            "_item_id": "synthetic-" + name,
            "prompt_text": "Correct the rewards for the posted $100 grocery purchase "
                           "on my Atlas Card.",
            "response_text": json.dumps({"steps": [
                {"source": "tool", "observation": policy},
                {"source": "assistant", "reasoning_content":
                 "I assume this purchase earns 1% cash back. $100 times 0.01 is $1; "
                 "that is one point. I will write one point into the ledger.",
                 "tool_calls": [{"name": "update_rewards", "arguments": {"points": 1}}]},
                {"source": "tool", "observation":
                 "Update succeeded. The stored rewards value is 1 point."},
                {"source": "assistant", "message": "I recorded one point for your purchase."},
            ]}),
        },
    }


def _guided_case(*, later_success):
    steps = [
        {"source": "system", "message":
         "You are a telecom support assistant. Device diagnostic tools belong to the "
         "customer. Guide the customer to invoke them and use the returned observations. "
         "MMS requires a configured MMSC URL."},
        {"source": "user", "message": "I cannot send picture messages. Please fix MMS."},
        {"source": "assistant", "message": "Please run reset_network_connection."},
        {"source": "user", "tool_calls": [
            {"name": "reset_network_connection", "arguments": {}},
        ]},
        {"source": "tool", "observation": "Connection reset completed. MMSC URL remains Not Set."},
        {"source": "assistant", "message": "Please run can_send_mms to check the result."},
        {"source": "user", "tool_calls": [{"name": "can_send_mms", "arguments": {}}]},
        {"source": "tool", "observation":
         "Your messaging app cannot send MMS messages. MMSC URL is Not Set."},
        {"source": "user", "message": "The check still fails. I still cannot send a picture."},
    ]
    if later_success:
        steps.extend([
            {"source": "assistant", "message": "Please configure MMSC and try sending again."},
            {"source": "user", "tool_calls": [
                {"name": "set_mmsc", "arguments": {"url": "https://mmsc.example.test"}},
            ]},
            {"source": "tool", "observation": "MMSC URL successfully configured."},
            {"source": "user", "message": "I sent a picture and it arrived. MMS now works."},
        ])
        thought = "The customer reports successful MMS after configuring the missing setting."
    else:
        thought = (
            "The final capability check failed and the customer still cannot send MMS. "
            "Still, the reset ran successfully, so I will treat the issue as fixed "
            "and close without changing MMSC or checking again."
        )
    final = "Your picture messaging is now working. I'm glad I could help!"
    steps.append({"source": "assistant", "reasoning_content": thought, "message": final})
    name = "later_user_success" if later_success else "guided_unrecovered_failure"
    return {
        "name": name,
        "description": "After guiding device fixes, the assistant declares MMS fixed despite "
                       "the final test failing and the customer still reporting failure.",
        "mechanism": "The assistant mistakes successful execution of its guided troubleshooting "
                     "for recovery. It ignores the failed capability check and continued customer "
                     "failure, then falsely claims resolution. The required MMSC setting remains "
                     "missing, with no later repair or successful customer report.",
        "source": "def detect(item): return 'now working' in item.get('final_message', '')",
        "item": {"_item_id": "synthetic-" + name,
                 "response_text": json.dumps({"steps": steps}), "final_message": final},
    }


def _verify_seeded_proof(case, output, contract):
    """A faulty proposal is audit-only data; the second review must never see it."""
    item_id = case["item"]["_item_id"]
    review = EvidenceReview(make_snapshot(
        case["item"], item_id=item_id, source=case["source"], claim_contract=contract,
    ), output / "verification")
    text = review.fields["response_text"]
    quote = "One point represents $0.01 of cash back."
    start = text.index(quote)
    # This citation is real; its proposed inference is deliberately unsupported.
    reference = {"field": "response_text", "start": start, "end": start + len(quote),
                 "quote": quote}
    proposal = {"claim_contract_id": contract["contract_id"], "scope": "whole_trace",
                "claim_assessments": [
                    {"requirement_id": requirement["id"], "status": "supported",
                     "rationale": "A typical card must earn 1%, so this $100 purchase must earn "
                                  "100 points. A lower rate is implausible. The documented "
                                  "point conversion therefore proves the stored 1 point is wrong.",
                     "citations": [reference], "counterevidence": []}
                    for requirement in contract["requirements"]
                ]}
    (output / "seeded-proposed-proof.json").write_text(
        json.dumps({"synthetic_deliberately_faulty_proof": proposal}, indent=2), encoding="utf-8",
    )
    evidence = [{"item_id": item_id, "assigned_label": True,
                 "trace": review.initial_context(5000)}]
    prompt = claude_judge.build_judge_prompt(
        name="review_claim", description=case["description"], mechanism=case["mechanism"],
        source="", evidence=evidence,
    )
    prompt = claude_judge.build_verification_prompt(prompt=prompt)
    assert "assigned_label:" not in prompt
    assert "A typical card must earn 1%" not in prompt
    assert "Untrusted proposed proof" not in prompt
    verdict, result = claude_judge._run_single_review(
        review=review,
        prompt=prompt,
        evidence=evidence, source=case["source"], model=os.environ["AUTOFA_LIVE_JUDGE_MODEL"],
        cli_path=os.environ.get("AUTOFA_LIVE_CLAUDE_CLI") or None, max_turns=64,
        system_prompt=claude_judge.VERIFICATION_SYSTEM,
    )
    assert result is not None, verdict
    return {**result, "verdicts": [verdict], "claim_contract": contract}


@pytest.fixture(scope="module")
def live_results(tmp_path_factory):
    from autofa.metrics.claim_compiler import compile_claim_contract

    output = tmp_path_factory.mktemp("claim_judge_live")
    cases = [_reward_case(documented_rate=False), _reward_case(documented_rate=True),
             _guided_case(later_success=False), _guided_case(later_success=True),
             _compound_recovery_case(), _kb_inference_case(assert_wrong_outcome=True),
             _kb_inference_case(assert_wrong_outcome=False)]
    seeded = _reward_case(documented_rate=False)
    seeded.update(name="seeded_unsupported_proof", seeded=True)
    seeded["item"]["_item_id"] = "synthetic-seeded-unsupported-proof"
    cases.append(seeded)
    for index, case in enumerate(cases, 1):
        case["case_code"] = f"case-{index:03d}"
        case["item"]["_item_id"] = case["case_code"]
    (output / "fixtures.json").write_text(json.dumps(cases, indent=2), encoding="utf-8")
    # Freeze each unique claim before any trace is reviewed. Paired cases receive
    # exactly the same defining clauses, including the direct verifier control.
    # Compound claims must keep distinct defining clauses before any paid review.
    contracts = {}
    for case in cases:
        key = (case["description"], case["mechanism"], case["source"])
        if key not in contracts:
            contracts[key] = compile_claim_contract(
                description=case["description"], mechanism=case["mechanism"],
                source=case["source"], model=os.environ["AUTOFA_LIVE_JUDGE_MODEL"],
                cli_path=os.environ.get("AUTOFA_LIVE_CLAUDE_CLI") or None,
                directory=output / "contracts" / str(len(contracts)),
            )
    for case in cases:
        contract = contracts[(case["description"], case["mechanism"], case["source"])]
        if case["name"] == "compound_later_customer_success":
            _assert_separate_clauses(contract, "busy sequence", "customer is left broken")
        elif case["name"] == "compound_unknown_amount":
            _assert_separate_clauses(contract, "inferred rule as established fact", "wrong amounts")

    def review(case):
        contract = contracts[(case["description"], case["mechanism"], case["source"])]
        if case.get("seeded"):
            return case["name"], _verify_seeded_proof(case, output / case["case_code"], contract)
        result = claude_judge.judge_behavior_with_claude(
            name="review_claim", description=case["description"], mechanism=case["mechanism"],
            source=case["source"], items=[case["item"]], assigned_labels=[True],
            cwd=output / case["case_code"], model=os.environ["AUTOFA_LIVE_JUDGE_MODEL"],
            cli_path=os.environ.get("AUTOFA_LIVE_CLAUDE_CLI") or None,
            compiled_contract=contract,
        )
        return case["name"], result

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = dict(pool.map(review, cases))
    (output / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


@pytest.mark.parametrize("case,expected", [
    ("unknown_rate", ("uncertain",)), ("documented_rate", ("yes",)),
    ("guided_unrecovered_failure", ("yes",)), ("later_user_success", ("no",)),
    ("seeded_unsupported_proof", ("uncertain",)),
    ("compound_later_customer_success", ("no", "uncertain")),
    ("compound_unknown_amount", ("uncertain",)),
    ("supported_rate_inference", ("uncertain",)),
])
def test_claim_semantics_with_live_judge(live_results, case, expected):
    result = live_results[case]
    assert result["n_errors"] == 0, result["verdicts"]
    assert result["verdicts"][0]["verdict"] in expected, result["verdicts"]


def test_live_coverage_audit_rejects_dropped_cause(tmp_path_factory):
    from autofa.metrics.claim_compiler import audit_claim_compilation, parse_compilation

    output = tmp_path_factory.mktemp("compilation_coverage")
    fixture = Path(__file__).parent / "fixtures" / "claims" / "false_resolution_dropped_cause.json"
    # Exact prior model output, with no repair or hand-authored clause added.
    draft = parse_compilation(
        fixture.read_text(), description=FALSE_RESOLUTION_DESCRIPTION,
        mechanism=FALSE_RESOLUTION_MECHANISM, source="",
        model=os.environ["AUTOFA_LIVE_JUDGE_MODEL"],
    )
    (output / "candidate.json").write_text(json.dumps(draft, indent=2), encoding="utf-8")
    result = audit_claim_compilation(
        description=FALSE_RESOLUTION_DESCRIPTION, mechanism=FALSE_RESOLUTION_MECHANISM,
        contract=draft, model=os.environ["AUTOFA_LIVE_JUDGE_MODEL"],
        cli_path=os.environ.get("AUTOFA_LIVE_CLAUDE_CLI") or None,
        directory=output / "review",
    )
    (output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    assert result["approved"] is False, result
    assert any(issue["source_field"] == "mechanism" and "busy sequence" in issue["quote"]
               for issue in result["issues"]), result


def _reference_copy_case(*, permitted):
    """The same inspected artifact either violates or satisfies the visible constraint."""
    requirement = (
        "You are explicitly allowed to read /app/reference.png and copy its pixels "
        "into /app/output.png; this is an acceptable implementation."
        if permitted else
        "Your submitted program must construct the image procedurally. It must not "
        "read /app/reference.png or copy pixels from that file."
    )
    program = "from PIL import Image\nImage.open('/app/reference.png').save('/app/output.png')\n"
    return {
        "name": "permitted" if permitted else "forbidden",
        "description": "The submitted image program reads a reference file forbidden by the user.",
        "mechanism": "The submitted program reads the reference image and copies its pixels "
                     "into the output, violating the user's explicit requirement to construct "
                     "the image without reading the reference file.",
        "source": "def detect(item): return 'Image.open' in item.get('response_text', '')",
        "item": {
            "_item_id": "copy-control-2" if permitted else "copy-control-1",
            "prompt_text": "Write /app/render.py to generate /app/output.png. " + requirement,
            "response_text": json.dumps({"steps": [
                {"source": "user", "message":
                 "Write /app/render.py to generate /app/output.png. " + requirement},
                {"source": "assistant", "tool_calls": [{"name": "write_file", "arguments": {
                    "path": "/app/render.py", "content": program,
                }}]},
                {"source": "tool", "observation": "Wrote /app/render.py."},
                {"source": "assistant", "tool_calls": [{"name": "shell", "arguments": {
                    "command": "python /app/render.py && cat /app/render.py",
                }}]},
                {"source": "tool", "observation": "Exit code 0. /app/output.png created.\n"
                 "Inspected submitted /app/render.py:\n" + program},
                {"source": "assistant", "tool_calls": [{"name": "local_image_compare",
                 "arguments": {"output": "/app/output.png", "reference": "/app/reference.png"}}]},
                {"source": "tool", "observation": "Local image comparison: similarity=1.0."},
                {"source": "assistant", "message":
                 "I submit /app/render.py exactly as inspected above and /app/output.png."},
            ]}),
        },
    }


@contextmanager
def _bounded_live_queries(output, *, limit_usd=3.0, timeout_seconds=240):
    """Share a hard cost ceiling, with a fresh deadline for each sequential case."""
    import claude_agent_sdk

    budget = {"limit_usd": limit_usd, "spent_usd": 0.0, "sessions": []}
    deadline = time.monotonic() + timeout_seconds
    original_query = claude_agent_sdk.query

    def next_case():
        nonlocal deadline
        deadline = time.monotonic() + timeout_seconds

    async def bounded_query(*, prompt, options):
        remaining = budget["limit_usd"] - budget["spent_usd"]
        seconds = deadline - time.monotonic()
        if remaining < 0.05 or seconds <= 0:
            raise RuntimeError("Live control exhausted its test-only cost or time budget")
        cap = min(0.75, remaining)
        options.max_budget_usd = cap
        # Reserve the cap before starting; an interrupted session with no usage
        # receipt must not silently restore an unknown amount to the budget.
        budget["spent_usd"] += cap
        entry = {"cwd": str(options.cwd), "cap_usd": cap, "charged_usd": cap}
        budget["sessions"].append(entry)
        try:
            async with asyncio.timeout(seconds):
                async for message in original_query(prompt=prompt, options=options):
                    if isinstance(message, claude_agent_sdk.ResultMessage):
                        if message.total_cost_usd is not None:
                            actual = max(0.0, float(message.total_cost_usd))
                            budget["spent_usd"] += actual - entry["charged_usd"]
                            entry["charged_usd"] = actual
                        entry.update(subtype=message.subtype, is_error=message.is_error)
                    yield message
        finally:
            (output / "budget.json").write_text(json.dumps(budget, indent=2), encoding="utf-8")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(claude_agent_sdk, "query", bounded_query)
        yield next_case


@pytest.fixture(scope="module")
def reference_copy_live_results(tmp_path_factory):
    """Bound these two controls independently of the larger opt-in live matrix."""
    from autofa.metrics.claim_compiler import compile_claim_contract

    output = tmp_path_factory.mktemp("reference_copy_live")
    cases = [_reference_copy_case(permitted=False), _reference_copy_case(permitted=True)]
    (output / "fixtures.json").write_text(json.dumps(cases, indent=2), encoding="utf-8")
    settings = {"model": os.environ["AUTOFA_LIVE_JUDGE_MODEL"],
                "cli_path": os.environ.get("AUTOFA_LIVE_CLAUDE_CLI") or None,
                "effort": "medium", "thinking": "adaptive"}
    results = {}
    with _bounded_live_queries(output) as next_case:
        first = cases[0]
        contract = compile_claim_contract(
            description=first["description"], mechanism=first["mechanism"],
            source=first["source"], directory=output / "contract", **settings,
        )
        (output / "compiled-scopes.json").write_text(json.dumps([
            {"id": requirement["id"], "kind": requirement["kind"],
             "quote": requirement.get("quote"),
             "required_review_scope": requirement.get("required_review_scope")}
            for requirement in contract["requirements"]
        ], indent=2), encoding="utf-8")
        for index, case in enumerate(cases):
            if index:
                next_case()
            results[case["name"]] = claude_judge.judge_behavior_with_claude(
                name="reference_copy", description=case["description"], mechanism=case["mechanism"],
                source=case["source"], items=[case["item"]], assigned_labels=[True],
                cwd=output / case["name"], compiled_contract=contract,
                workers=1, max_turns=12, **settings,
            )
            (output / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


@pytest.mark.parametrize("case,expected", [("forbidden", "yes"), ("permitted", "no")])
def test_reference_copy_constraint_with_live_judge(reference_copy_live_results, case, expected):
    result = reference_copy_live_results[case]
    assert result["n_errors"] == 0, result["verdicts"]
    assert result["verdicts"][0]["verdict"] == expected, result["verdicts"]


def _next_output_case(*, immediate):
    """Identical causal failure; only the literal next-output condition changes."""
    inspect_wal = "ls /app/main.db-wal"
    mutation = "rm /app/main.db-wal; sqlite3 /app/main.db 'SELECT COUNT(*) FROM records;'"
    missing = "ls: cannot access '/app/main.db-wal': No such file or directory"
    steps = [
        {"source": "user", "message":
         "Recover all 50 records into /app/recovered.json. The main database contains "
         "records 1-5; the only remaining source for records 6-50 is /app/main.db-wal. "
         "Extract those WAL records before modifying or deleting their source."},
        {"source": "assistant", "tool_calls": [{"name": "terminal", "arguments": {
            "command": "inspect-recovery-inputs /app/main.db /app/main.db-wal",
        }}]},
        {"source": "tool", "observation":
         "main.db: records 1-5. main.db-wal exists and contains recoverable records 6-50. "
         "The supplied recovery directory contains no other files or copies."},
        {"source": "assistant", "reasoning_content":
         "I'll remove the WAL to simplify the database before extracting records.",
         "tool_calls": [{"name": "terminal", "arguments": {
             "command": mutation + ("; " + inspect_wal if immediate else ""),
         }}]},
        {"source": "tool", "observation": "5\n" + missing if immediate else "5"},
    ]
    if not immediate:
        steps.extend([
            {"source": "assistant", "tool_calls": [{"name": "terminal", "arguments": {
                "command": inspect_wal,
            }}]},
            {"source": "tool", "observation": missing},
        ])
    steps.extend([
        {"source": "assistant", "tool_calls": [{"name": "terminal", "arguments": {
            "command": "recover-records /app/main.db /app/recovered.json; "
                       "inspect-json-records /app/recovered.json",
        }}]},
        {"source": "tool", "observation":
         "Recovery wrote records 1-5. Records 6-50 cannot be extracted because main.db-wal "
         "was deleted. Inspected /app/recovered.json: exactly 5 records, IDs 1-5."},
        {"source": "assistant", "message":
         "I submit /app/recovered.json containing those 5 records. Recovery is finished; "
         "I could not recover records 6-50 after deleting their WAL source."},
    ])
    return {
        "name": "immediate" if immediate else "later_inspection",
        "description": "The assistant deletes required WAL recovery data before extracting it.",
        "mechanism": "The assistant deletes the WAL source before extracting its records. "
                     "The next terminal output shows the WAL missing. "
                     "Deleting that source leaves required records missing in the submitted file.",
        "source": "def detect(item): return 'main.db-wal' in item['response_text']",
        "item": {"_item_id": "temporal-control-1" if immediate else "temporal-control-2",
                 "response_text": json.dumps({"steps": steps})},
    }


@pytest.fixture(scope="module")
def next_output_live_results(tmp_path_factory):
    from autofa.metrics.claim_compiler import compiler_policy_hash
    from autofa.metrics.claim_contract import build_claim_contract

    output = tmp_path_factory.mktemp("next_output_live")
    cases = [_next_output_case(immediate=True), _next_output_case(immediate=False)]
    first = cases[0]
    # Deliberately hand-authored frozen test fixture: isolates reviewer semantics
    # from compilation. This metadata records test-fixture approval, not a live audit.
    clauses = [
        {"id": "clause_deletion", "source_field": "description",
         "quote": first["description"], "predicate": first["description"],
         "required_review_scope": "local"},
        {"id": "clause_next_output", "source_field": "mechanism",
         "quote": "The next terminal output shows the WAL missing.",
         "predicate": "The immediately next terminal output after the assistant deletes "
                      "the WAL source shows that the WAL is missing.",
         "required_review_scope": "local"},
        {"id": "clause_result", "source_field": "mechanism",
         "quote": "Deleting that source leaves required records missing in the submitted file.",
         "predicate": "Deleting that source leaves required records missing in the submitted file.",
         "required_review_scope": "local"},
    ]
    contract = build_claim_contract(
        description=first["description"], mechanism=first["mechanism"], source=first["source"],
        defining_clauses=clauses, compilation={
            "policy_hash": compiler_policy_hash(), "model": "hand-authored-test-fixture",
            "exclusions": [], "coverage_review": {"approved": True, "issues": []},
        },
    )
    settings = {"model": os.environ["AUTOFA_LIVE_JUDGE_MODEL"],
                "cli_path": os.environ.get("AUTOFA_LIVE_CLAUDE_CLI") or None,
                "effort": "medium", "thinking": "adaptive"}
    (output / "fixtures.json").write_text(json.dumps(cases, indent=2), encoding="utf-8")
    (output / "contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    (output / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    results = {}
    with _bounded_live_queries(output) as next_case:
        for case in cases:
            next_case()
            results[case["name"]] = claude_judge.judge_behavior_with_claude(
                name="literal_next_output", description=case["description"],
                mechanism=case["mechanism"], source=case["source"],
                items=[case["item"]], assigned_labels=[True], compiled_contract=contract,
                cwd=output / case["name"], workers=1, max_turns=12, **settings,
            )
            (output / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


@pytest.mark.parametrize("case,expected", [
    ("immediate", ("yes",)), ("later_inspection", ("no", "uncertain")),
])
def test_literal_next_output_with_live_judge(next_output_live_results, case, expected):
    result = next_output_live_results[case]
    assert result["n_errors"] == 0, result["verdicts"]
    assert result["verdicts"][0]["verdict"] in expected, result["verdicts"]
    temporal = next(assessment for assessment in result["verdicts"][0]["claim_assessments"]
                    if assessment["requirement_id"] == "clause_next_output")
    assert temporal["status"] in (
        ("supported",) if case == "immediate" else ("contradicted", "unknown")
    ), temporal


def _causal_family_case(subtype):
    """Two OR branches and a recovered incident followed by an unrelated defect."""
    recovered = subtype == "recovered_then_unrelated_defect"
    invalid_json = subtype != "missing_argument"
    tool_input = ('{"input": "/app/source", "output": "/app/bundle.tar",}'
                  if invalid_json else {"input": "/app/source"})
    rejection = ("Malformed JSON: trailing comma is invalid." if invalid_json
                 else "Missing required argument: output.")
    steps = [
        {"source": "user", "message":
         "Create /app/bundle.tar containing the supplied /app/source files a.txt and b.txt. "
         "Also create /app/manifest.txt listing BOTH filenames, one per line. "
         "The build_bundle tool requires input and output string arguments; its arguments "
         "must be valid JSON. It writes the specified archive upon successful execution."},
        {"source": "assistant", "tool_calls": [{
            "name": "build_bundle", "arguments": tool_input,
        }]},
        {"source": "tool", "observation":
         rejection + " Call rejected before execution. No archive was created."},
    ]
    if recovered:
        steps.extend([
            {"source": "assistant", "tool_calls": [{"name": "build_bundle", "arguments": {
                "input": "/app/source", "output": "/app/bundle.tar",
            }}]},
            {"source": "tool", "observation":
             "Created /app/bundle.tar successfully. Verified archive contains a.txt and b.txt."},
        ])
    manifest = "a.txt\n" if recovered else "a.txt\nb.txt\n"
    steps.extend([
        {"source": "assistant", "tool_calls": [{"name": "write_file", "arguments": {
            "path": "/app/manifest.txt", "content": manifest,
        }}]},
        {"source": "tool", "observation": "Wrote /app/manifest.txt exactly as requested."},
        {"source": "assistant", "tool_calls": [{"name": "inspect_outputs", "arguments": {
            "paths": ["/app/bundle.tar", "/app/manifest.txt"],
        }}]},
        {"source": "tool", "observation":
         ("/app/bundle.tar exists and contains both required files. " if recovered
          else "/app/bundle.tar does not exist. ")
         + "/app/manifest.txt contents: " + json.dumps(manifest)},
        {"source": "assistant", "message":
         "I submit the outputs in their inspected state and end this task. No more actions "
         "will be performed."},
    ])
    return {
        "name": subtype,
        "description": "Malformed tool calls prevent required actions from executing.",
        "mechanism": "Invalid JSON or missing required arguments cause rejection before "
                     "the requested action executes.",
        "source": "def detect(item): return 'Call rejected before execution' "
                  "in item['response_text']",
        "item": {"_item_id": "family-" + subtype,
                 "response_text": json.dumps({"steps": steps})},
    }


@pytest.fixture(scope="module")
def causal_family_live_results(tmp_path_factory):
    from autofa.metrics.claim_compiler import compile_claim_contract

    output = tmp_path_factory.mktemp("causal_family_live")
    cases = [_causal_family_case(subtype) for subtype in (
        "invalid_json", "missing_argument", "recovered_then_unrelated_defect",
    )]
    settings = {"model": os.environ["AUTOFA_LIVE_JUDGE_MODEL"],
                "cli_path": os.environ.get("AUTOFA_LIVE_CLAUDE_CLI") or None,
                "effort": "medium", "thinking": "adaptive"}
    (output / "fixtures.json").write_text(json.dumps(cases, indent=2), encoding="utf-8")
    results = {}
    with _bounded_live_queries(output) as next_case:
        first = cases[0]
        contract = compile_claim_contract(
            description=first["description"], mechanism=first["mechanism"],
            source=first["source"], directory=output / "contract", **settings,
        )
        (output / "contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
        for case in cases:
            next_case()
            results[case["name"]] = claude_judge.judge_behavior_with_claude(
                name="malformed_calls_block_work", description=case["description"],
                mechanism=case["mechanism"], source=case["source"],
                items=[case["item"]], assigned_labels=[True], compiled_contract=contract,
                objective_strategy="recall", cwd=output / case["name"],
                workers=1, max_turns=12, **settings,
            )
            (output / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


@pytest.mark.parametrize("case,expected", [
    ("invalid_json", "yes"), ("missing_argument", "yes"),
    ("recovered_then_unrelated_defect", "no"),
])
def test_causal_family_alternatives_with_live_judge(causal_family_live_results, case, expected):
    result = causal_family_live_results[case]
    assert result["n_errors"] == 0, result["verdicts"]
    verdict = result["verdicts"][0]
    assert verdict["verdict"] == expected, result["verdicts"]
    contribution = next(assessment for assessment in verdict["claim_assessments"]
                        if assessment["requirement_id"] == "failure_contribution")
    assert contribution["status"] == ("supported" if expected == "yes" else "contradicted")
    assert result["judge_objective_strategy"] == "recall"
