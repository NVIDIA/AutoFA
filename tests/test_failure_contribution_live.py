# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in real-judge controls for root causes versus recovered parsing errors.

Run with AUTOFA_RUN_LIVE_JUDGE=1 AUTOFA_LIVE_JUDGE_MODEL=<model>.
Uses synthetic tasks, identical detector prose and a trace-blind compiled contract.
No private benchmark records or outcome labels are provided as evidence.
"""

import json
import os

import pytest

from autofa.metrics.claude_judge import judge_behavior_with_claude

pytestmark = pytest.mark.skipif(
    os.environ.get("AUTOFA_RUN_LIVE_JUDGE") != "1"
    or not os.environ.get("AUTOFA_LIVE_JUDGE_MODEL"),
    reason="Requires explicit live-judge settings and an existing Claude login",
)

DESCRIPTION = "Sends replies the harness cannot parse on two or more turns"
MECHANISM = (
    "The harness requires a JSON object containing commands with a keystrokes field. "
    "At least two replies are rejected because a command lacks that field. "
    "Rejected replies execute no commands, so those turns make no progress."
)
SOURCE = (
    "def detect(item):\n"
    "    from autofa.trace import steps\n"
    "    return sum(s.source == 'user' and 'Previous response had parsing errors' "
    "in s.message for s in steps(item)) >= 2\n"
)


def _steps(ending):
    write_report = "printf 'id,value\\n1,7\\n' > /app/report.csv\n"
    steps = [
        {"source": "system", "message":
         "Reply with a JSON object containing commands; each command needs keystrokes. "
         "Rejected replies execute nothing. The required report does not exist yet."},
        {"source": "user", "message":
         "Create /app/report.csv with exactly two lines: id,value and 1,7."},
    ]
    for _ in range(2):
        steps.extend([
            {"source": "assistant", "message": json.dumps({
                "commands": [{"text": write_report}],
            })},
            {"source": "user", "message":
             "Previous response had parsing errors: missing keystrokes. No command executed."},
        ])
        if ending in {"wrong_artifact", "success"}:
            steps.extend([
                {"source": "assistant", "message": json.dumps({
                    "commands": [{"keystrokes": write_report}],
                })},
                {"source": "user", "message":
                 "Command accepted and executed. /app/report.csv contains:\nid,value\n1,7\n"},
            ])
    if ending == "budget_exhausted":
        steps.append({"source": "user", "message":
                      "Turn limit reached after the two rejected replies. Session terminated. "
                      "No commands were executed; /app/report.csv does not exist."})
    elif ending in {"wrong_artifact", "success"}:
        contents = "name,score\\n1,7\\n" if ending == "wrong_artifact" else "id,value\\n1,7\\n"
        steps.extend([
            {"source": "user", "message": "40 turns remain."},
            {"source": "assistant", "reasoning_content":
             ("I will use name,score as the column names." if ending == "wrong_artifact"
              else "I will use the requested id,value column names."),
             "message": json.dumps({"commands": [{
                 "keystrokes": "printf '" + contents + "' > /app/report.csv\n"
             }]})},
            {"source": "user", "message":
             "Command accepted and executed. /app/report.csv now contains:\n"
             + contents.replace("\\n", "\n")},
            {"source": "assistant", "message": "The report is ready. I am done."},
        ])
    return steps


@pytest.fixture(scope="module")
def contribution_results(tmp_path_factory):
    output = tmp_path_factory.mktemp("failure_contribution_live")
    # Opaque IDs avoid giving the reviewer the expected attribution or outcome.
    cases = ["wrong_artifact", "success", "budget_exhausted", "missing_ending"]
    items = [{"_item_id": f"case-{index}", "response_text": json.dumps({"steps": _steps(case)})}
             for index, case in enumerate(cases)]
    result = judge_behavior_with_claude(
        name="review_claim", description=DESCRIPTION, mechanism=MECHANISM, source=SOURCE,
        items=items, assigned_labels=[True] * len(items), cwd=output,
        model=os.environ["AUTOFA_LIVE_JUDGE_MODEL"],
        cli_path=os.environ.get("AUTOFA_LIVE_CLAUDE_CLI") or None,
        workers=4, effort="medium", thinking="adaptive",
    )
    (output / "results.json").write_text(json.dumps(result, indent=2))
    return dict(zip(cases, result["verdicts"], strict=True))


@pytest.mark.parametrize("case,expected,status", [
    ("wrong_artifact", "no", "contradicted"),
    ("success", "no", "contradicted"),
    ("budget_exhausted", "yes", "supported"),
    ("missing_ending", "uncertain", "unknown"),
])
def test_root_cause_attribution_with_live_judge(contribution_results, case, expected, status):
    row = contribution_results[case]
    assert row["verdict"] == expected, row
    assessment = next(a for a in row["claim_assessments"]
                      if a["requirement_id"] == "failure_contribution")
    assert assessment["status"] == status, assessment
    # The negative controls must reject causality even while confirming the incident.
    literal = [a for a in row["claim_assessments"]
               if a["requirement_id"] in {"description", "mechanism"}]
    assert literal and all(a["status"] == "supported" for a in literal), row


def _doubt_steps(case):
    timeout = 60 if case == "known_defect" else 30
    retries = 9 if case == "unrelated_defect" else 2
    concern = (
        "I see timeout=60 even though the task requires 30. That mismatch may fail. "
        "The checker will probably accept 60 too, so I won't change it."
        if case == "known_defect" else
        "I am not sure timeout=30 is right; maybe it should be 60. It is probably fine."
    )
    return [
        {"source": "user", "message":
         "Write settings.json with exactly timeout=30 and retries=2, both JSON integers. "
         "Use a JSON reply with commands and task_complete; commands may be empty."},
        {"source": "assistant", "message": json.dumps({"commands": [{
            "keystrokes": 'printf \'{"timeout":' + str(timeout) + ',"retries":'
            + str(retries) + "}\' > settings.json\ncat settings.json\n"
        }], "task_complete": False})},
        {"source": "user", "message": "Command executed. Stored settings.json contents: "
         + json.dumps({"timeout": timeout, "retries": retries})},
        {"source": "assistant", "message": '{"commands":[],"task_complete":true}'},
        {"source": "user", "message":
         "Are you sure you want to mark the task as complete? "
         "You can still edit or check the file."},
        {"source": "assistant", "reasoning_content": concern + " I will submit now.",
         "message": '{"commands":[],"task_complete":true}'},
    ]


@pytest.fixture(scope="module")
def doubt_results(tmp_path_factory):
    output = tmp_path_factory.mktemp("doubt_contribution_live")
    cases = ["correct_artifact", "unrelated_defect", "known_defect"]
    result = judge_behavior_with_claude(
        name="review_claim",
        description="Names a concrete doubt about its deliverable at the confirmation gate, "
                    "waves it off, and submits",
        mechanism="The harness asks the agent to re-confirm before submission. The agent names "
                  "a concern about a specific part of the deliverable, dismisses it with "
                  "speculation, and confirms without a command to settle the concern.",
        source="def detect(item): return 'probably' in item['response_text']",
        items=[{"_item_id": f"case-{i}", "response_text": json.dumps({"steps": _doubt_steps(case)})}
               for i, case in enumerate(cases)],
        assigned_labels=[True] * len(cases), cwd=output,
        model=os.environ["AUTOFA_LIVE_JUDGE_MODEL"],
        cli_path=os.environ.get("AUTOFA_LIVE_CLAUDE_CLI") or None,
        workers=3, effort="medium", thinking="adaptive",
    )
    (output / "results.json").write_text(json.dumps(result, indent=2))
    return dict(zip(cases, result["verdicts"], strict=True))


@pytest.mark.parametrize("case,expected,status", [
    ("correct_artifact", "no", "contradicted"),
    ("unrelated_defect", "no", "contradicted"),
    ("known_defect", "yes", "supported"),
])
def test_doubt_requires_a_specific_causal_defect(doubt_results, case, expected, status):
    row = doubt_results[case]
    assert row["verdict"] == expected, row
    assessment = next(a for a in row["claim_assessments"]
                      if a["requirement_id"] == "failure_contribution")
    assert assessment["status"] == status, assessment
