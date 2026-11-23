# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The taxonomy report aggregates judge classifications into views, agreement and provenance."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from autofa.taxonomy import report
from autofa.taxonomy.schema import TAXONOMY_V0_PATH, TAXONOMY_V1_PATH, load_taxonomy

NO_SUP = report.NO_SUPPORTED_CODE


def _case(
    rollout_id: str,
    judge: str,
    code: str,
    *,
    category: str = "User interactions",
    evidence_support: str = "complete",
) -> dict[str, object]:
    return {
        "status": "ok", "rollout_id": rollout_id, "judge_configuration": judge,
        "selected_code": code, "display_mode": code, "main_category": category,
        "evidence_support": evidence_support, "benchmark": "bench", "task_id": rollout_id,
        "repeat": 0,
    }


def _judgment(judge: str, code: str, *, support: str = "complete", **fields) -> dict[str, object]:
    return {
        **_case("r1", f"provider/{judge}", code, evidence_support=support),
        "judge_model": judge, "run_id": "run", "agent_ref": "test-agent", "fault_side": "model",
        **fields,
    }


def _result_record(
    rollout_id: str,
    *,
    status: str = "ok",
    taxonomy_sha256: str = "taxonomy-a",
    model: str | None = "judge-a",
    **fields,
) -> dict[str, object]:
    row: dict[str, object] = {
        "kind": "taxonomy_classification", "status": status,
        "taxonomy_sha256": taxonomy_sha256, "protocol_version": "rubric-a",
        "source": {"rollout_id": rollout_id, "run_id": "run-a", "task_id": rollout_id,
                   "repeat": 0, "benchmark": "bench"},
        "judge": {"provider": "chat", "model": model, "api": "chat_completions"},
    }
    if status == "ok":
        row |= {"selected_code": "OWN-RSN", "selected_label": "Reasoning failure",
                "selected_path": ["root", "Model interactions", "Reasoning failure"]}
    else:
        row["error"] = "judge timed out"
    return row | fields


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text("".join(f"{json.dumps(row)}\n" for row in rows), encoding="utf-8")
    return path


def _write_manifest(results: Path, *, judge=None, second_pass: str = "off", **fields) -> None:
    results.with_name(f"{results.stem}.manifest.json").write_text(json.dumps({
        "exporter": "autofa-taxonomy", "exporter_version": 0, "schema": "store-v17",
        "runs": [], "reference_data": [],
        "taxonomy": {"sha256": "taxonomy-a", "second_pass": second_pass},
        "judge": judge or {"provider": "chat", "model": "judge-a"},
        "results": {"file": results.name,
                    "sha256": hashlib.sha256(results.read_bytes()).hexdigest()},
        **fields,
    }), encoding="utf-8")


def _write_custom_taxonomy(path: Path) -> str:
    path.write_text(json.dumps({
        "id": "custom", "name": "Custom taxonomy", "version": "1",
        "root": {"id": "root", "label": "Custom root", "description": "Custom root", "children": [
            {"id": "reasoning", "code": "OWN-RSN", "label": "Custom reasoning",
             "description": "Custom definition", "fault_side": "custom-side"},
        ]},
    }), encoding="utf-8")
    return report._taxonomy_sha256(path)


# --- votes and agreement ---


@pytest.mark.parametrize(
    ("judgments", "expected"),
    [
        ([_judgment("luna", "A")],
         {"selected_code": "A", "decision": "accept", "judge_count": 1, "representative": 0}),
        # A strict majority wins and reports the weakest supporting evidence.
        ([_judgment("luna", "A"), _judgment("sol", "A", support="partial"),
          _judgment("flash", "B")],
         {"selected_code": "A", "decision": "accept", "agreeing_judges": 2,
          "evidence_support": "partial"}),
        ([_judgment("luna", "A"), _judgment("sol", "B")],
         {"selected_code": report.JUDGES_DISAGREE_CODE, "decision": "review",
          "decision_reason": report.JUDGES_DISAGREE_CODE, "representative": None}),
        ([_judgment("luna", NO_SUP), _judgment("sol", NO_SUP)],
         {"decision": "review", "decision_reason": NO_SUP}),
        ([_judgment("luna", "A", support="insufficient", decision="review"),
          _judgment("sol", "A")],
         {"decision": "review", "decision_reason": "insufficient_evidence"}),
    ],
    ids=["single-judge", "strict-majority", "no-majority", "agreed-no-supported",
         "insufficient-evidence"],
)
def test_rollout_votes_accept_only_a_supported_strict_majority(judgments, expected) -> None:
    combined = report._aggregate_rollout(judgments)

    assert {key: combined[key] for key in expected} == expected


@pytest.mark.parametrize(
    ("codes", "level"),
    [
        (("A", "A"), "same_leaf"),
        ((NO_SUP, NO_SUP), "all_no_supported"),
        ((NO_SUP, "A"), "some_no_supported"),
        (("A", "B"), "same_category"),
        (("A", "C"), "different_category"),
    ],
)
def test_agreement_levels_separate_abstention_leaf_and_category_disagreement(codes, level):
    judgments = [_judgment("luna", codes[0]), _judgment("sol", codes[1])]
    if codes[1] == "C":
        judgments[1]["main_category"] = "Harness interactions"

    assert report._agreement_level(judgments) == level


def test_agreement_deduplicates_retries_and_needs_overlapping_judges() -> None:
    agreement = report._compute_agreement([
        _case("r1", "provider/judge-a", "A"),
        _case("r1", "provider/judge-a", "B"),  # A retry by the same judge on the same rollout.
        _case("r2", "provider/judge-b", "A"),
    ])

    assert agreement["available"] is False
    assert "no rollout has multiple valid judge results" in agreement["unavailable_reason"]


def test_two_judge_overlap_reports_agreement_and_disagreement() -> None:
    agreement = report._compute_agreement([
        _case("r1", "provider/judge-a", "A"),
        _case("r1", "provider/judge-b", "A"),
        _case("r2", "provider/judge-a", "A"),
        _case("r2", "provider/judge-b", "B"),
        _case("r3", "provider/judge-a", "C", category="Harness interactions"),
        _case("r3", "provider/judge-b", "D"),
        _case("r4", "provider/judge-a", "A", evidence_support="partial"),
        _case("r4", "provider/judge-b", "A"),
    ])

    assert agreement["available"] is True
    assert [(item["rollout_id"], item["agreement_level"]) for item in agreement["disagreements"]
            ] == [("r2", "same_category"), ("r3", "different_category")]


def test_latest_unusable_attempt_supersedes_stale_success() -> None:
    succeeded = _result_record("rollout-a")
    failed = _result_record("rollout-a", status="error")

    assert report.latest_classification_rows([succeeded, failed]) == [failed]
    assert report.aggregate_classification_rows([succeeded, failed]) == []


def test_incompatible_results_do_not_vote_together(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "mixed.jsonl", [
        _result_record("same-rollout", taxonomy_sha256="taxonomy-a"),
        _result_record("same-rollout", taxonomy_sha256="taxonomy-b", selected_code="OWN-IF"),
    ])

    data = report.build_report([tmp_path])

    assert len(data["rollouts"]) == 2
    assert all(case["judge_count"] == 1 for case in data["rollouts"])
    assert len(data["meta"]["compatibility_groups"]) == 2


# --- judge identity ---


def test_judge_configuration_covers_effective_settings_but_not_timeout() -> None:
    row = {
        "_second_pass_method": "off",
        "judge": {"provider": "chat", "model": "us/azure/openai/gpt-5.6-luna",
                  "api": "chat_completions", "reasoning_effort": "endpoint_default",
                  "max_tokens": 4096, "base_url": "https://judge.example/v1/",
                  "timeout_seconds": 300},
    }
    configuration = report._judge_configuration(row)

    for fragment in (";api=chat_completions;", ";reasoning=endpoint_default;",
                     ";max_tokens=4096;", ";second_pass=off;"):
        assert fragment in configuration
    assert configuration.endswith(";endpoint=https://judge.example/v1")
    for changed in ({"api": "responses"}, {"reasoning_effort": "high"}, {"max_tokens": 8192},
                    {"base_url": "https://other.example/v1"}):
        assert report._judge_configuration({**row, "judge": {**row["judge"], **changed}}
                                           ) != configuration
    assert report._judge_configuration({**row, "_second_pass_method": "llm"}) != configuration
    timeout_only = {**row, "judge": {**row["judge"], "timeout_seconds": 60}}
    assert report._judge_configuration(timeout_only) == configuration


@pytest.mark.parametrize("row,fragments", [
    ({
        "_second_pass_method": "agentic",
        "judge": {"provider": "chat", "model": "first-pass-model"},
        "second_pass": {"method": "agentic", "receipts": {
            "harness": "codex", "model": "review-model", "max_turns": 32,
        }},
    }, ["second_pass=agentic", "second_pass_harness=codex", "second_pass_model=review-model",
        "second_pass_max_turns=32"]),
    ({
        "classification_method": "agentic_taxonomy_classification",
        "first_pass": {"method": "agentic"},
        "second_pass": {"method": "off"},
        "judge": {"provider": "agentic", "model": "gpt-6.1-sol", "harness": "codex",
                  "configured_max_turns": 32},
    }, ["first_pass=agentic", "first_pass_harness=codex", "first_pass_max_turns=32",
        "second_pass=off"]),
], ids=["agentic_review", "agentic_first_pass"])
def test_judge_configuration_identifies_agentic_routes(row, fragments) -> None:
    configuration = report._judge_configuration(row)

    for fragment in fragments:
        assert f";{fragment};" in configuration


def test_agentic_review_limit_distinguishes_legacy_and_investigation_budgets() -> None:
    assert report._agentic_review_limit({"max_turns": 32}) == "32 turns"
    assert report._agentic_review_limit({
        "max_turns": 128,
        "tool_budget_semantics": "investigation_calls_v1",
    }) == "128 tool calls"
    assert report._agentic_review_limit({
        "max_turns": None,
        "tool_budget_semantics": "investigation_calls_v1",
    }) == "unlimited"


def test_judge_configuration_keeps_composed_judges_separate() -> None:
    row = {
        "first_pass": {"method": "llm", "model": "first-a", "configuration_sha256": "a" * 64},
        "second_pass": {"method": "agentic", "receipts": {"harness": "codex", "model": "reviewer"}},
        "judge": {"model": "reviewer", "second_pass_configuration_sha256": "b" * 64},
    }
    configuration = report._judge_configuration(row)

    for changed in (
        {"first_pass": {**row["first_pass"], "model": "first-b", "configuration_sha256": "c" * 64}},
        {"judge": {**row["judge"], "second_pass_configuration_sha256": "d" * 64}},
        {"parent_classification": {"judge_configuration": "parent/reviewer-b"}},
    ):
        assert report._judge_configuration({**row, **changed}) != configuration


def test_review_retry_keeps_one_composed_judge_identity() -> None:
    judge = {
        "provider": "chat", "requested_model": "first-model", "configuration_sha256": "a" * 64,
        "second_pass_requested_model": "review-model",
        "second_pass_configuration_sha256": "b" * 64, "harness": "codex",
        "configured_max_turns": 32,
    }
    shared = {
        "rollout_id": "rollout", "selected_code": "OWN-IF",
        "first_pass": {"method": "llm", "status": "completed", "requested_model": "first-model",
                       "configuration_sha256": "a" * 64},
    }
    failed = {**shared, "status": "review_failed", "judge": {**judge, "model": "first-model"},
              "second_pass": {"method": "agentic", "status": "failed"}}
    succeeded = {
        **shared, "status": "ok", "judge": {**judge, "model": "review-model"},
        "second_pass": {"method": "agentic", "status": "completed", "receipts": {
            "harness": "codex", "model": "review-model", "max_turns": 32,
        }},
    }
    failed["judge_configuration"] = report._judge_configuration(failed)
    succeeded["judge_configuration"] = report._judge_configuration(succeeded)

    assert failed["judge_configuration"] == succeeded["judge_configuration"]
    # Whichever attempt is latest decides; a failed review never contributes a verdict.
    assert report._latest_judgments([failed, succeeded]) == {
        report._report_key(succeeded): [succeeded]
    }
    assert report._latest_judgments([succeeded, failed]) == {}
    for rows in ([failed], [succeeded, failed]):
        assert report._build_data(rows, taxonomy_terms=[], population={})["rollouts"] == []


def test_sibling_manifest_settings_keep_same_model_judges_distinct(tmp_path: Path) -> None:
    row = _result_record("same-rollout",
                         judge={"provider": "chat", "model": "eccn-claude-5-5-sonnet"})
    first = _write_jsonl(tmp_path / "first.jsonl", [row])
    second = _write_jsonl(tmp_path / "second.jsonl", [row])
    _write_manifest(first, judge={
        "provider": "chat", "model": "eccn-claude-5-5-sonnet",
        "base_url": "https://first.example/v1", "api": "chat_completions",
        "reasoning_effort": "high", "max_tokens": 4096,
    })
    _write_manifest(second, second_pass="llm", judge={
        "provider": "chat", "model": "eccn-claude-5-5-sonnet",
        "base_url": "https://second.example/v1", "api": "responses",
        "reasoning_effort": "medium", "max_tokens": 8192,
    })

    [case] = report.build_report([first, second])["rollouts"]

    assert case["judge_count"] == 2
    assert len({item["judge_configuration"] for item in case["judgments"]}) == 2
    assert {item["second_pass_method"] for item in case["judgments"]} == {"off", "llm"}
    assert {item["judge_base_url"] for item in case["judgments"]} == {
        "https://first.example/v1", "https://second.example/v1",
    }


# --- records ---


def test_rollout_identity_supports_source_and_incident_fallback() -> None:
    assert report._row_rollout_id({"source": {"rollout_id": "source-id"}}) == "source-id"
    assert report._row_rollout_id({"incident_id": "incident-id"}) == "incident-id"


def test_legacy_decisions_and_main_category_follow_current_rules() -> None:
    legacy = report._compact_record({
        "status": "ok", "selected_code": "OWN-IF", "decision": "review",
        "decision_reasons": ["edge_confidence_below_threshold"],
        "selected_path": ["root", "User interactions", "Owner — model", "leaf"],
    })
    routed = report._compact_record({
        "status": "ok", "selected_code": NO_SUP, "decision": "review", "selected_path": [],
    })
    insufficient = report._compact_record({
        "status": "ok", "selected_code": "OWN-RSN",
        "diagnosis": {"evidence_support": "insufficient"},
    })

    assert legacy["decision"] == "accept"
    assert "decision_reasons" not in legacy
    assert routed["decision"] == "review"
    assert insufficient["decision"] == "review"
    # The main category is the top-level family, or the routing code itself.
    assert legacy["main_category"] == "User interactions"
    assert routed["main_category"] == NO_SUP


def test_diagnosis_localizes_and_orders_causal_chain() -> None:
    def compact(diagnosis):
        return report._compact_record({
            "status": "ok", "selected_code": "OWN-RSN", "diagnosis": diagnosis,
            "source": {"rollout_id": "run:task:0", "run_id": "run", "task_id": "task"},
        })

    localized = compact({
        "evidence_support": "partial", "critical_event": {"event_ref": 5},
        "causal_chain": [
            {"source_type": "other", "event_ref": None, "claim": "outcome"},
            {"source_type": "trace_event", "event_ref": 8, "claim": "symptom"},
            {"source_type": "trace_event", "event_ref": 5, "claim": "cause"},
            "malformed trailing prose",  # Older records may carry unparsed chain entries.
        ],
    })
    unlocalized = compact({"causal_chain": [{"event_ref": 9, "claim": "later in trace"},
                                            {"event_ref": 2, "claim": "earlier in trace"}]})

    assert localized["localization_status"] == "localized"
    assert [item["event_ref"] for item in localized["causal_chain_display"]] == [5, 8, None]
    assert unlocalized["localization_status"] == "mode_supported_localization_unknown"
    assert [item["claim"] for item in unlocalized["causal_chain_display"]] == [
        "earlier in trace", "later in trace",
    ]
    mixed_primary_references = compact({
        "causal_chain": [
            {"event_ref": 9, "claim": "later primary event"},
            {
                "event_ref": {"lane": "conversation", "event_idx": 2},
                "claim": "earlier primary event",
            },
        ],
    })
    assert [
        item["claim"] for item in mixed_primary_references["causal_chain_display"]
    ] == ["earlier primary event", "later primary event"]
    independent_lanes = compact({
        "causal_chain": [
            {
                "source_type": "trace_event",
                "event_ref": {"lane": "worker", "event_idx": 11},
                "claim": "worker cause",
            },
            {
                "source_type": "trace_event",
                "event_ref": {"lane": "conversation", "event_idx": 9},
                "claim": "primary symptom",
            },
        ],
    })
    assert [
        item["claim"] for item in independent_lanes["causal_chain_display"]
    ] == ["worker cause", "primary symptom"]


def test_report_inventories_invalid_files_and_failed_judges(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "ok.jsonl", [_result_record("rollout-a")])
    _write_jsonl(tmp_path / "failed.jsonl", [_result_record(
        "rollout-a", status="error",
        judge={"provider": "chat", "requested_model": "judge-b", "attempted": True},
    )])
    (tmp_path / "empty.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "corrupt.jsonl").write_text("{broken\n", encoding="utf-8")
    (tmp_path / "unicode.jsonl").write_bytes(b"\xff")
    (tmp_path / "other.jsonl").write_text('{"kind":"other"}\n', encoding="utf-8")

    data = report.build_report([tmp_path])

    assert len(data["rollouts"]) == 1
    assert {item["state"] for item in data["input_inventory"]} == {"included", "skipped"}
    assert sum(item["failed"] for item in data["judge_attempts"]) == 1
    assert any("unreadable" in warning for warning in data["warnings"])


def test_report_rejects_a_folder_with_no_classification_results(tmp_path: Path) -> None:
    (tmp_path / "empty.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "other.jsonl").write_text('{"kind":"other"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="no usable taxonomy classification JSONL found"):
        report.build_report([tmp_path])


def test_report_separates_judge_failures_from_pre_judge_issues(tmp_path: Path) -> None:
    judge = {"provider": "chat", "model": "judge-a", "api": "chat_completions"}
    _write_jsonl(tmp_path / "results.jsonl", [
        _result_record("successful"),
        _result_record("judge-failure", status="error", judge={**judge, "attempted": True}),
        _result_record("preparation-failure", status="error",
                       judge={**judge, "attempted": False}),
        # Older errors never recorded whether the judge ran, so they are not judge failures.
        _result_record("legacy-error", status="error"),
    ])

    data = report.build_report([tmp_path])

    [attempt] = data["judge_attempts"]
    assert (attempt["attempted"], attempt["succeeded"], attempt["failed"]) == (2, 1, 1)
    issues = {item["rollout_id"]: item["stage"] for item in data["processing_issues"]}
    assert set(issues) == {"preparation-failure", "legacy-error"}
    assert issues["legacy-error"] == "judge attempt not recorded"


def test_report_error_redacts_secrets_and_home_paths() -> None:
    message = (
        f"failed at {Path.home()}/private?api_key=secret-value and /tmp/private/file "
        "via https://judge.example.test/v1 with sk-abcdefghijk"
    )

    safe = report._report_error(message + " " + "x" * 600)

    assert str(Path.home()) not in safe
    assert "/tmp/private/file" not in safe
    assert "https://judge.example.test/v1" in safe
    assert "secret-value" not in safe
    assert "sk-abcdefghijk" not in safe
    assert len(safe) == 500


# --- taxonomy definitions ---


def test_taxonomy_terms_read_fault_side(tmp_path: Path) -> None:
    taxonomy = tmp_path / "taxonomy.json"
    taxonomy.write_text(json.dumps({"root": {"label": "root", "children": [
        {"code": "TOOL", "label": "Tool", "description": "tool", "fault_side": "tool"},
        {"code": "UNSET", "label": "Unset", "description": "unset"},
    ]}}), encoding="utf-8")

    ordered = report._taxonomy_terms(taxonomy)
    terms = {item["code"]: item for item in ordered}

    assert terms["TOOL"]["fault_side"] == "tool"
    assert terms["UNSET"]["fault_side"] == "unspecified"
    assert [item["code"] for item in ordered[:3]] == ["TOOL", "UNSET", NO_SUP]


def test_taxonomy_terms_require_a_canonical_hash_match(tmp_path: Path) -> None:
    taxonomy = tmp_path / "custom.json"
    taxonomy_hash = _write_custom_taxonomy(taxonomy)
    results = _write_jsonl(tmp_path / "results.jsonl",
                           [_result_record("rollout-a", taxonomy_sha256=taxonomy_hash)])

    mismatched = report.build_report([results])
    matched = report.build_report([results], taxonomy=taxonomy)

    assert not [term for term in mismatched["taxonomy_terms"] if term["kind"] == "taxonomy"]
    assert mismatched["rollouts"][0]["judgments"][0]["fault_side"] == "unspecified"
    custom = next(term for term in matched["taxonomy_terms"] if term["code"] == "OWN-RSN")
    assert custom["definition"] == "Custom definition"
    assert matched["rollouts"][0]["judgments"][0]["fault_side"] == "custom-side"


@pytest.mark.parametrize("path", [TAXONOMY_V0_PATH, TAXONOMY_V1_PATH], ids=["v0", "v1"])
def test_source_license_fields_keep_recorded_taxonomy_hashes(tmp_path: Path, path: Path) -> None:
    # Results classified before the shipped source blocks stated the paper's license
    # recorded the plain canonical hash of the document without those fields.
    document = json.loads(path.read_text(encoding="utf-8"))
    document["source"] = {key: value for key, value in document["source"].items()
                          if key not in {"license", "license_url"}}
    recorded = hashlib.sha256(json.dumps(
        document, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
    ).encode()).hexdigest()
    results = _write_jsonl(tmp_path / "results.jsonl",
                           [_result_record("rollout-a", taxonomy_sha256=recorded)])

    data = report.build_report([results], taxonomy=path)

    assert load_taxonomy(path).content_sha256 == recorded
    assert [term for term in data["taxonomy_terms"] if term["kind"] == "taxonomy"]
    assert data["taxonomy_source"]["license"] == "CC-BY-4.0"


def test_taxonomy_attribution_marks_the_adaptation_and_links_the_license() -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required to render the report attribution")
    lines = report.HTML.splitlines()
    harness = "\n".join([
        "const nodes = {}, document = {getElementById: id => nodes[id] ??= {}};",
        "let D;",
        *(next(line for line in lines if line.startswith(prefix))
          for prefix in ("const esc=", "function renderTerminology(")),
        "console.log(JSON.stringify(JSON.parse(process.argv[1]).map(source => {",
        "  D = {taxonomy_source: source, taxonomy_terms: [], constants: {judge_rules: []}};",
        "  delete nodes['taxonomy-source'];",
        "  renderTerminology();",
        "  return nodes['taxonomy-source'].innerHTML;",
        "})));",
    ])
    sources = [report._taxonomy_source(TAXONOMY_V1_PATH),
               {"title": "Custom taxonomy", "url": "https://example.test/taxonomy"}]

    shipped, custom = json.loads(subprocess.check_output(
        [node, "-e", harness, json.dumps(sources)], text=True))

    assert shipped.startswith("Failure modes adapted from ")
    assert ('(Harsh Raj et al., Scale AI, arXiv:2607.28802, licensed under <a class="source" '
            'href="https://creativecommons.org/licenses/by/4.0/" target="_blank" '
            'rel="noopener">CC-BY-4.0</a>).') in shipped
    assert custom.startswith("Failure modes from ") and "licensed" not in custom


def test_mixed_taxonomies_omit_definitions_even_when_one_matches(tmp_path: Path) -> None:
    taxonomy = tmp_path / "custom.json"
    taxonomy_hash = _write_custom_taxonomy(taxonomy)
    results = _write_jsonl(tmp_path / "results.jsonl", [
        _result_record("rollout-a", taxonomy_sha256=taxonomy_hash),
        _result_record("rollout-b", taxonomy_sha256="other-taxonomy"),
    ])

    data = report.build_report([results], taxonomy=taxonomy)

    assert not [term for term in data["taxonomy_terms"] if term["kind"] == "taxonomy"]
    assert all(judgment["fault_side"] == "unspecified"
               for case in data["rollouts"] for judgment in case["judgments"])


# --- manifests, links and population ---


def test_report_links_to_manifest_source_api_with_optional_override(tmp_path: Path) -> None:
    results = _write_jsonl(tmp_path / "results.jsonl", [_result_record("task/a")])
    _write_manifest(results, reference_data=[
        {"run_id": "run-a", "export_url": "https://eval.example/v1/v1/runs/run-a/export/autofa"},
    ])

    data = report.build_report([results])
    [case] = data["rollouts"]
    [source] = data["source_cells"]
    assert case["viewer_url"].startswith("https://eval.example/v1/#/rollout/run-a/task%252Fa/0")
    assert source["run_viewer_url"] == "https://eval.example/v1/#/run/run-a"

    overridden = report.build_report([results], viewer_url="https://viewer.example")
    assert overridden["rollouts"][0]["viewer_url"].startswith("https://viewer.example/#/rollout/")
    with pytest.raises(ValueError, match="HTTP\\(S\\)"):
        report.build_report([results], viewer_url="javascript:alert(1)")


def test_incomplete_classification_manifest_is_skipped(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "results.jsonl", [_result_record("rollout-a")])
    (tmp_path / "results.manifest.json").write_text(json.dumps({
        "exporter": "autofa-taxonomy", "exporter_version": 0, "schema": "store-v17",
    }), encoding="utf-8")

    data = report.build_report([tmp_path])

    manifest = next(
        item for item in data["input_inventory"] if item["format"] == "classification_manifest"
    )
    assert manifest["state"] == "skipped"
    assert data["source_cells"][0]["population_failures"] is None


def test_source_cells_report_full_run_accuracy_for_a_classified_subset(tmp_path: Path) -> None:
    results = _write_jsonl(tmp_path / "results.jsonl",
                           [_result_record("rollout-a"), _result_record("rollout-b")])
    _write_manifest(results, second_pass="llm", runs=[{"run_id": "run-a", "population": {
        "rows": 2_200, "scored": 2_158, "zero_reward": 1_371, "positive_reward": 787,
        "unscored": 42,
    }}])

    [cell] = report.build_report([results])["source_cells"]

    assert cell["diagnosed_rollouts"] == 2
    assert cell["population_scored"] == 2_158
    assert cell["population_failures"] == 1_371
    assert cell["population_positive"] == 787


# --- HTML and publishing ---


def test_top_ten_modes_and_display_only_other_bucket_never_fold_routing_codes() -> None:
    counts = {f"mode-{index:02d}": 20 - index for index in range(12)}
    counts[NO_SUP] = 1
    counts[report.JUDGES_DISAGREE_CODE] = 1

    shown = report._top_modes(counts)

    assert [code for code, _ in shown[:10]] == [f"mode-{index:02d}" for index in range(10)]
    assert (NO_SUP, 1) in shown
    assert (report.JUDGES_DISAGREE_CODE, 1) in shown
    assert shown[-1] == (report.OTHER_MODES_CODE, counts["mode-10"] + counts["mode-11"])
    assert len(shown) == 13


def test_embedded_report_json_escapes_html_openers(tmp_path: Path) -> None:
    results = _write_jsonl(tmp_path / "results.jsonl", [
        _result_record("rollout-a", selected_label="<!--<script>alert(1)</script>"),
    ])
    output = tmp_path / "report.html"

    report.write_report([results], output)
    html = output.read_text(encoding="utf-8")

    assert "<!--<script>alert(1)</script>" not in html
    assert "\\u003c!--\\u003cscript>alert(1)\\u003c/script>" in html


def test_annotation_csv_neutralizes_spreadsheet_formulas() -> None:
    assert r"^[\s\x00-\x1f\x7f-\x9f]*[=+\-@]" in report.HTML
    assert "text=`'${text}`" in report.HTML


def test_publish_report_uploads_the_generated_report(tmp_path, monkeypatch) -> None:
    data = {"source_cells": [
        {"benchmark": "hle", "evaluated_model": f"model-{name}", "run_id": f"run-{name}",
         "run_export_url": f"https://runs.test/v1/runs/run-{name}/export/autofa"}
        for name in ("a", "b")
    ]}
    path = tmp_path / "combined.html"
    path.write_text(report.HTML.replace("__DATA__", json.dumps(data).replace("<", "\\u003c")),
                    encoding="utf-8")
    uploads = []

    class Client:
        def upload_library(self, **kwargs):
            uploads.append(kwargs)
            return SimpleNamespace(dashboard_url="https://lib.test/libraries/1",
                                   failure_card_url="")

    monkeypatch.setattr("autofa.config.load_config",
                        lambda: SimpleNamespace(remote_library_base_url="https://lib.test"))

    url = report.publish_report(path, name="vpr-classify", client=Client())

    assert url == "https://lib.test/libraries/1"
    [upload] = uploads
    assert upload["metadata"]["branch_key"] == "vpr-classify"
    assert upload["metadata"]["model_id"] == "2 evaluated models"
    assert upload["metadata"]["benchmarks"] == ["hle"]
    assert upload["metadata"]["data_identity"] == {"version": 1, "sources": [
        {"kind": "evalrdl", "origin": "https://runs.test", "run_id": f"run-{name}"}
        for name in ("a", "b")
    ]}
    assert upload["library_json"]["functions"] == []
    assert upload["failure_cards_html"] == path
