# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Terminal views share verdicts, report order and metric scope without inventing scores."""

from copy import deepcopy
from io import StringIO
from types import SimpleNamespace

import pytest
from rich.console import Console
from rich.text import Text

from autofa.config import Config
from autofa.core.library import FeatureFunction, FunctionLibrary
from autofa.harness.commands import _show_library
from autofa.mcp import runtime
from autofa.metrics.association import StreamingBucketAccumulator, contingency
from autofa.metrics.verdicts import (
    current_behavior_scores,
    extract_metrics,
    extract_report_metrics,
    format_flagged,
    verdict_for_scores,
)
from autofa.objectives import apply_preset
from autofa.reports.failure_cards import build_data
from autofa.ui.activity import AgentActivity
from autofa.ui.explorer import _build_tree
from autofa.ui.theme import verdict_color

from support.task_support import _console, _function, _ranked_function


def _rendered(console):
    return Text.from_ansi(console.export_text(styles=True))


def _check_badge(console, verdict):
    rendered = _rendered(console)
    offset = rendered.plain.index(verdict["label"])
    color = rendered.get_style_at_offset(console, offset).color
    assert color is not None
    assert color.get_truecolor().hex.lower() == verdict_color(verdict["code"]).lower()
    return rendered.plain


def _causal_scores(fn):
    return current_behavior_scores(fn.scores, source=fn.review_source,
                                   description=fn.description, mechanism=fn.mechanism,
                                   behavior_strictness="causal")


@pytest.mark.parametrize("state, code", [
    ("good_coverage", "GOOD"), ("unknown_tasks", "OK"),
    ("broad_small_review", "GOOD"),  # Matched breadth, not the confirmed sample, sets Good.
    ("historical_review", "OK"), ("sampled_support", "GOOD"),
    ("review", "UNSCORED"), ("pending_review", "UNSCORED"), ("inconclusive_review", "UNSCORED"),
    ("sampled_review_skipped", "UNSCORED"), ("missing_strata", "UNSCORED"),
    ("diagnostic_only", "UNSCORED"), ("diagnostic_empty", "EMPTY"),
    ("pending_precision", "UNSCORED"), ("weak", "WEAK"), ("empty", "EMPTY"),
    ("passing_matches", "WEAK"), ("partial_errors", "GOOD"), ("incomplete", "UNSCORED"),
    ("stale_evaluation", "UNSCORED"),
])
def test_terminal_verdict_codes_follow_shared_classification(state, code):
    fn = _function(state)
    before = deepcopy(fn.scores)
    assert verdict_for_scores(_causal_scores(fn))["code"] == code
    assert fn.scores == before


@pytest.mark.parametrize("state", [
    "good_coverage", "pending_review", "incomplete", "stale_evaluation", "missing_strata",
])
def test_verdict_label_and_color_match_across_terminal_views(tmp_path, monkeypatch, state):
    fn = _function(state)
    library = FunctionLibrary(functions=[fn])
    config = Config(behavior_strictness="causal")
    rt = SimpleNamespace(library=library, scores_history=[], config=config,
                         run=SimpleNamespace(run_dir=lambda: tmp_path))
    monkeypatch.setattr(runtime, "_RUNTIME", rt)
    before = deepcopy(library.to_dict())
    verdict = verdict_for_scores(_causal_scores(fn))

    activity_console = _console()
    snapshot = AgentActivity(activity_console)._library_snapshot(force=True)
    activity_console.print(snapshot["lines"][0])
    activity_text = _check_badge(activity_console, verdict)
    library_console = _console()
    _show_library(library_console)
    library_text = _check_badge(library_console, verdict)
    node = _build_tree(library, objective_config=config)[0]
    assert node.data["_verdict"] == verdict
    assert verdict["label"] in node.label

    views = (activity_text, library_text, node.detail)
    if state in {"incomplete", "missing_strata"}:
        # Unresolved or unrepresented scores are never shown as measured zeros.
        for text in views:
            assert "precision=0.00" not in text and "prec=0.00" not in text
            assert "100.0%" not in text
    if state == "stale_evaluation":
        assert "Coverage=40.0%" not in activity_text
        assert "precision=0.90" not in node.detail
        assert "prec=0.90" not in library_text
    if state == "pending_review":
        # A pending review does not hide completed detector measurements.
        assert "Coverage=40.0%" in activity_text
        assert "precision=0.90" in node.detail
        assert "prec=0.90" in library_text
    assert library.to_dict() == before


@pytest.mark.parametrize("scores,expected", [
    pytest.param({"sampled": True, "sample_n": 3000, "n_scored": 3000, "n_rows_scored": 1,
                  "n_flagged": 1, "pct_caught": .0001},
                 "<0.1% (estimate; 1/3000 sampled attempts)", id="nonzero-rate-not-rounded-to-0"),
    pytest.param({"sampled": True, "sample_n": 200, "n_scored": 200, "n_flagged": 20.0,
                  "pct_caught": .02, "bucket": "positive", "bucket_mode": "exact",
                  "buckets": [{"bucket": "other_category", "n_rows_scored": 9},
                              {"bucket": "positive", "n_rows_scored": 2}]},
                 "2.0% (estimate; 2/200 sampled attempts)", id="observed-not-weighted-count"),
])
def test_sampled_flagged_counts_never_substitute_weighted_population_count(scores, expected):
    assert format_flagged(scores) == expected


def test_diagnostic_processing_errors_do_not_appear_as_completed_reviews():
    fn = _function("diagnostic_only")
    fn.scores["diagnostic_review"].update(
        n_completed=1, n_yes=1, n_errors=1, n_review_incomplete=1, n_uncertain=1,
    )
    detail = extract_metrics(fn.scores)["diagnostic_review_detail"]
    assert "Development controls reviewed: 1; adherence unavailable." in detail
    assert "2 controls still need review recovery" in detail
    assert fn.scores["behavior_n_judged"] == 0


def test_quality_then_coverage_order_matches_report_in_every_terminal_view(tmp_path, monkeypatch):
    # Insertion order and a precision-first optimization both disagree with the
    # visible ranking. The top-five cap must be applied after quality and coverage.
    specifications = [
        ("weak_widest", "good", .95, .3),
        ("good_precise", "good", .2, 1),
        ("pending_widest", "pending_review", .99, 1),
        ("ok_wide", "good", .9, .6),
        ("no_matches", "empty", 0, 0),
        ("ok_narrow", "good", .0075, .8),
        ("good_wide", "good", .7, .8),
        ("good_small", "good", .015, .9),
    ]
    functions = []
    for name, state, coverage, precision in specifications:
        fn = _function(state)
        fn.name = name
        fn.scores.update(recall=coverage, ppv_at_pi0=precision)
        functions.append(fn)
    library = FunctionLibrary(functions=functions)
    config = apply_preset(Config(behavior_strictness="causal"), "precision")
    rt = SimpleNamespace(library=library, scores_history=[], config=config,
                         run=SimpleNamespace(run_dir=lambda: tmp_path))
    monkeypatch.setattr(runtime, "_RUNTIME", rt)
    before = deepcopy(library.to_dict())
    expected = ["good_wide", "good_precise", "good_small", "ok_wide", "ok_narrow",
                "weak_widest", "no_matches", "pending_widest"]
    report = build_data(library)
    assert [fn["name"] for fn in report["functions"]] == expected

    activity_console = _console()
    activity = AgentActivity(activity_console)
    snapshot = activity._library_snapshot(force=True)
    assert [next(fn.name for fn in functions if f"{fn.name}  " in line)
            for line in snapshot["lines"]] == expected
    activity_console.print(activity._render())
    activity_text = _rendered(activity_console).plain
    assert all(name in activity_text for name in expected[:5])
    assert not any(name in activity_text for name in expected[5:])

    nodes = _build_tree(library, top_x=5, objective_config=config)
    assert [node.data["name"] for node in nodes] == expected[:5]

    library_console = _console()
    _show_library(library_console)
    library_text = _rendered(library_console).plain
    assert sorted(expected, key=library_text.index) == expected
    assert library.to_dict() == before


def test_codex_dispatch_can_fill_missing_arguments_without_showing_prompt_content():
    activity = AgentActivity(Console(force_terminal=False, no_color=True))
    activity.on_codex_event("item/started", {"item": {
        "id": "a", "type": "dynamicToolCall", "tool": "plan_function_repair",
    }})
    activity.update_tool_arguments("plan_function_repair", {
        "name": "wrong_order", "action": "tighten_detector",
        "rationale": "Improve precision using supported boundaries. SECRET_EVIDENCE",
        "source": "SECRET_SOURCE",
    })
    line = activity._spinner_text(done=False)
    assert "wrong_order" in line
    assert "SECRET" not in line
    assert "SECRET" not in repr(activity._active_tool_arguments)


@pytest.fixture
def display(tmp_path, monkeypatch):
    fn = FeatureFunction(name="example", source="def detect(item): return True", description="test")
    rt = SimpleNamespace(library=FunctionLibrary(functions=[fn]), scores_history=[],
                         config=Config(behavior_strictness="causal"),
                         run=SimpleNamespace(run_dir=lambda: tmp_path))
    monkeypatch.setattr(runtime, "_RUNTIME", rt)
    activity = AgentActivity(Console(force_terminal=False, no_color=True))

    def render(scores):
        fn.scores = scores
        snapshot = activity._library_snapshot(force=True)
        return Text.from_markup(snapshot["lines"][0]).plain

    return render


def test_live_sample_metrics_are_population_weighted_and_separate_from_adherence(display):
    accumulator = StreamingBucketAccumulator()
    # 200 scored attempts represent 1,500 attempts across three strata. Raw
    # sample recall is 50/100; the weighted estimate is 520/1,200 instead.
    for group, n, matches, weight, correct in (
        ("f1", 50, 20, 20, False), ("f2", 50, 30, 4, False), ("p", 100, 20, 3, True),
    ):
        for index in range(n):
            accumulator.update(index < matches, is_correct=correct,
                               item_id=f"{group}-{index}", weight=weight)
    scores = next(row for row in accumulator.finalize(sampled=True) if row["a"] == 520)
    scores.update(evaluated=True, sample_n=200, sample_frame_n=1500,
                  evaluation_scope_complete=True, n_scored=200,
                  behavior_purity=0.75, behavior_n_judged=4)
    before = deepcopy(scores)
    association, review = display(scores).split(" | ")
    assert "Coverage=43.3%" in association
    assert "Coverage=50.0%" not in association
    assert "adherence=0.75" in review and "estimate" not in review
    assert extract_report_metrics(scores)["recall"] == pytest.approx(520 / 1200)
    assert scores == before  # Rendering does not change scores or their scope.


PARTIAL = {**contingency(10, 5, 30, 55), "n_scored": 100}


@pytest.mark.parametrize("scores, shown, hidden", [
    pytest.param({**PARTIAL, "eval_truncated": True, "evaluation_scope_complete": False},
                 ["partial scan", "Coverage=25.0%"], [], id="truncated-scan"),
    pytest.param({**contingency(10, 5, 90, 95), "explains": "parent",
                  "within_recall": 0.5, "within_ppv_at_pi0": 0.8},
                 ["Coverage within parent=50.0%"], [], id="parent-scope-named"),
    pytest.param({**contingency(999, 1, 99001, 999), "n_scored": 101000},
                 ["NOT EVALUATED", "Coverage=<1%"], ["Coverage=1.0%"],
                 id="coverage-below-one-percent-not-rounded-up"),
    pytest.param({**contingency(0, 0, 0, 0), "evaluated": True, "n_scored": 0,
                  "evaluation_scope_complete": True, "evaluation_complete": False,
                  "metrics_scope": "resolved_items", "n_insufficient": 200},
                 ["NOT EVALUATED", "metrics unavailable"], ["NO MATCHES", "Coverage=0.0%"],
                 id="no-resolved-results-are-unavailable-not-zero"),
])
def test_row_discloses_scope_without_changing_selected_coverage(display, scores, shown, hidden):
    rendered = display(scores)
    assert all(text in rendered for text in shown), rendered
    assert not any(text in rendered for text in hidden), rendered


def test_cli_matches_report_for_stale_reviews_pending_scores_and_parent_scope(
    tmp_path, monkeypatch,
):
    stale = _ranked_function("stale", .99, 1, 1)
    stale.source = "def detect(item): return False"
    parent = _ranked_function("conditional", .1, .8, 1)
    parent.scores.update(within_parent="parent", within_recall=.9, within_ppv_at_pi0=.8)
    ready = _ranked_function("ready", .7, .8, 1)
    pending = FeatureFunction(name="pending", source="def detect(item): return True",
                              description="")
    library = FunctionLibrary(functions=[stale, pending, ready, parent])
    rt = SimpleNamespace(library=library, scores_history=[],
                         config=apply_preset(Config(behavior_strictness="causal"), "precision"),
                         run=SimpleNamespace(run_dir=lambda: tmp_path))
    monkeypatch.setattr(runtime, "_RUNTIME", rt)
    activity = AgentActivity(Console(file=StringIO(), width=240, force_terminal=False,
                                     no_color=True))
    before = deepcopy(library.to_dict())
    snapshot = activity._library_snapshot(force=True)
    shown = [next(fn.name for fn in library.functions if f"{fn.name}  " in line)
             for line in snapshot["lines"]]
    expected = [row["name"] for row in build_data(library)["functions"]]
    assert shown == expected == ["conditional", "ready", "stale", "pending"]
    assert "Coverage within parent=90.0%" in Text.from_markup(snapshot["lines"][0]).plain
    assert library.to_dict() == before
