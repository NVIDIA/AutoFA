# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generated prompts and skills carry the runtime's guidance, settings and executable recipes."""

import hashlib
import json
import re
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path
from textwrap import indent
from types import SimpleNamespace

import pytest

from autofa.config import Config
from autofa.core.dependencies import build_bundle
from autofa.core.detector_lint import lint_detector_source
from autofa.core.helpers import HelperFunction
from autofa.core.library import FeatureFunction, FunctionLibrary
from autofa.core.sandbox import DetectorAbstained, run_detector
from autofa.data.artifact_store import ArtifactStore
from autofa.mcp import runtime
from autofa.mcp.tool_results import PAGE_CHARS, read_tool_result, save_tool_result
from autofa.metrics import claim_contract
from autofa.metrics.discovery_grades import agent_quality_guidance as quality_target_guidance
from autofa.objectives import apply_preset, objective_guidance, snapshot_objectives
from autofa.session.behavior_guidance import (
    OUTCOME_EVIDENCE_GUIDANCE,
    REPORT_EXPLANATION_GUIDANCE,
    SAVED_RESULT_PAGING_GUIDANCE,
    behavior_policy_prompt,
    candidate_check,
    mechanism_evidence_guidance,
    render_skill,
)
from autofa.session.discovery_budget import DISCOVERY_ROUND_GUIDANCE, discovery_budget_feedback
from autofa.session.guidance import guidance_prompt, snapshot_guidance
from autofa.session.prompts import (
    BIDIRECTIONAL_DISCOVERY_WORKFLOW,
    DISCOVERY_REVIEW_SCHEDULING_GUIDANCE,
    POST_VALIDATION_POLISHING_GUIDANCE,
    RECALL_DISCOVERY_WORKFLOW,
    ROUND_EXPLORATION_GUIDANCE,
    VERSION_RECOVERY_GUIDANCE,
    build_system_prompt,
    format_continue_kickoff,
    format_discover_kickoff,
    write_session_workspace,
)

from support.config import save_config

SKILL_SOURCES = Path(__file__).parents[1] / "src/autofa/skills"
CAUSAL_GATE = "Mandatory judge acceptance: failure_contribution"
# Retired guidance and tool arguments that must not return to any generated surface.
RETIRED = ("Good discovery target", "minimum Good", "2–4 sentence", "retry_relationship",
           'mode="development"', "record_missed_failure_audit",
           "first round with deterministic Python")


def kickoffs(run_dir, **kwargs):
    return [kickoff(top_x=3, max_rounds=2, run_dir=run_dir, **kwargs)
            for kickoff in (format_discover_kickoff, format_continue_kickoff)]


# The harness picks the skill directory and the mode picks the judge acceptance gate.
@pytest.mark.parametrize("harness,skill_directory,mode", [
    ("claude", ".claude", "causal"), ("codex", ".agents", "behavioral"),
])
def test_regenerated_workspace_and_kickoffs_carry_shared_guidance_once(
    tmp_path, harness, skill_directory, mode,
):
    config = Config(harness=harness, behavior_strictness=mode)
    skills = tmp_path / skill_directory / "skills"
    protocol_path = tmp_path / "autofa_protocol.md"
    # Continue/resume regenerate the workspace and must replace earlier instructions.
    stale = "An unjudged candidate needs judge_function; keep tightening the first family"
    for path in (protocol_path, skills / "discover/SKILL.md"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(stale)
    write_session_workspace(tmp_path, objective_config=config, exploration_ssot=False,
                            min_failure_modes=2)
    protocol = protocol_path.read_text()
    discover, validate = ((skills / name / "SKILL.md").read_text()
                          for name in ("discover", "validate"))
    discover_kickoff, continue_kickoff = kickoffs(
        tmp_path, objective_config=config, exploration_ssot=False, min_failure_modes=2,
    )
    for prompt in (protocol, discover_kickoff, continue_kickoff):
        for guidance in (RECALL_DISCOVERY_WORKFLOW, quality_target_guidance(),
                         DISCOVERY_ROUND_GUIDANCE, BIDIRECTIONAL_DISCOVERY_WORKFLOW,
                         DISCOVERY_REVIEW_SCHEDULING_GUIDANCE, VERSION_RECOVERY_GUIDANCE,
                         ROUND_EXPLORATION_GUIDANCE, POST_VALIDATION_POLISHING_GUIDANCE):
            assert prompt.count(guidance) == 1
        assert objective_guidance(config) in prompt
    for kickoff in (discover_kickoff, continue_kickoff):
        assert "Aim for 2 failure detectors at this round's active grade" in kickoff
    for prompt in (protocol, discover):
        for guidance in (SAVED_RESULT_PAGING_GUIDANCE, REPORT_EXPLANATION_GUIDANCE,
                         POST_VALIDATION_POLISHING_GUIDANCE, mechanism_evidence_guidance(config)):
            assert guidance in prompt
    for prompt in (protocol, discover, discover_kickoff, continue_kickoff):
        assert candidate_check(config) in prompt
    for prompt in (protocol, discover, validate, discover_kickoff, continue_kickoff):
        assert behavior_policy_prompt(config) in prompt
        assert f"behavior_strictness={mode}" in prompt
        assert OUTCOME_EVIDENCE_GUIDANCE in prompt
        assert (CAUSAL_GATE in prompt) == (mode == "causal")
        assert "<!-- AUTOFA_" not in prompt
        assert stale not in prompt
        for phrase in RETIRED:
            assert phrase not in prompt
    # Installed skills are rendered from the shipped sources, not separately maintained copies.
    for name in ("discover/SKILL.md", "validate/SKILL.md", "discover/implementation_recipes.md"):
        assert (skills / name).read_text() == render_skill(
            (SKILL_SOURCES / name).read_text(), config,
        )
    assert f"{skill_directory}/skills/discover/implementation_recipes.md" in protocol


def test_regenerated_workspace_and_kickoffs_follow_the_saved_run_policy(tmp_path):
    # The run snapshot is authoritative even if a user's default changes later.
    mode = "contextual"
    (tmp_path / "config.json").write_text(json.dumps({"round": 1, "behavior_strictness": mode}))
    write_session_workspace(tmp_path, exploration_ssot=True, exploration_seed=42)
    surfaces = [
        (tmp_path / "autofa_protocol.md").read_text(),
        build_system_prompt(run_dir=tmp_path),
        *kickoffs(tmp_path, exploration_ssot=True, exploration_seed=43),
    ]
    expected = mechanism_evidence_guidance(Config(behavior_strictness=mode))
    for surface in surfaces:
        assert f"behavior_strictness={mode}" in surface
        assert expected in surface


@pytest.mark.parametrize("settings,markers", [
    ({"discovery_wait_for_plateau": True, "discovery_terminate_if_stalled": True,
      "discovery_stall_rounds": 4},
     ["Continue until coverage plateaus: On", "Terminate if stalled: On",
      "after 4 consecutive completed, validated"]),
    ({"discovery_wait_for_plateau": False, "discovery_terminate_if_stalled": False},
     ["Continue until coverage plateaus: Off", "Terminate if stalled: Off"]),
])
def test_saved_stopping_settings_reach_workspace_and_kickoffs(tmp_path, settings, markers):
    config = Config(**settings)
    (tmp_path / "config.json").write_text(json.dumps({
        "objective_settings": snapshot_objectives(config), **settings,
    }))
    write_session_workspace(tmp_path, objective_config=config, exploration_ssot=False)
    for prompt in [(tmp_path / "autofa_protocol.md").read_text(),
                   *kickoffs(tmp_path, exploration_ssot=False)]:
        for marker in markers:
            assert marker in prompt


@pytest.mark.parametrize("kickoff", [format_discover_kickoff, format_continue_kickoff])
def test_kickoff_uses_active_preference_and_respects_saved_run(tmp_path, kickoff):
    preset = "balanced"
    config = apply_preset(Config(), preset)
    (tmp_path / "config.json").write_text(json.dumps({
        "objective_settings": snapshot_objectives(config),
    }))
    kwargs = dict(top_x=3, max_rounds=2, exploration_ssot=False, run_dir=tmp_path)
    saved = kickoff(**kwargs)
    assert f"Discovery strategy: {preset}" in saved
    assert objective_guidance(config) in saved
    override = "recall"
    active = kickoff(**kwargs, objective_config=apply_preset(Config(), override))
    assert f"Discovery strategy: {override}" in active
    assert f"Discovery strategy: {preset}" not in active


# Each mode with a judge acceptance gate reads its own requirement from the claim contract.
@pytest.mark.parametrize("mode,requirement_name", [
    ("contextual", "FAILURE_RELEVANCE"), ("causal", "FAILURE_CONTRIBUTION"),
])
def test_authoring_uses_the_judges_current_acceptance_text(
    tmp_path, monkeypatch, mode, requirement_name,
):
    # A contract change must reach discovery without editing a second prompt copy.
    requirement = getattr(claim_contract, requirement_name)
    sentinel = requirement["text"] + " Contract revision sentinel."
    monkeypatch.setitem(requirement, "text", sentinel)
    config = Config(harness="claude", behavior_strictness=mode)
    prompt = build_system_prompt(objective_config=config, exploration_ssot=False)
    assert prompt.index(sentinel) < prompt.index("## Protocol for /discover")
    write_session_workspace(tmp_path, objective_config=config, exploration_ssot=False)
    assert sentinel in (tmp_path / "autofa_protocol.md").read_text()
    assert sentinel in (tmp_path / ".claude/skills/discover/SKILL.md").read_text()


def test_discovery_label_guidance_does_not_change_frozen_judge_scope():
    run = SimpleNamespace(analysis_focus="both", behavior_strictness="causal")
    discovery = guidance_prompt(run, audience="discovery")
    judge = guidance_prompt(run, audience="judge")
    assert "behavior_strictness=causal" in discovery
    assert OUTCOME_EVIDENCE_GUIDANCE in discovery
    assert OUTCOME_EVIDENCE_GUIDANCE not in judge
    assert discovery.replace(OUTCOME_EVIDENCE_GUIDANCE, "") == judge


def test_both_harnesses_receive_coverage_tool_and_sampling_filter():
    from autofa.mcp.server import TOOL_CALLABLES, codex_tool_specs

    assert "library_coverage" in TOOL_CALLABLES
    specs = {tool["name"]: tool for tool in codex_tool_specs()}
    assert "library_coverage" in specs
    assert "coverage" in specs["sample_items"]["inputSchema"]["properties"]


@pytest.mark.parametrize("current,maximum,phase,remaining", [
    (8, 10, "explore", 2),  # consolidation starts with one round remaining
    (4, 5, "consolidate", 1),
    (5, 5, "boundary", 0),
    (6, 5, "overrun", 0),
])
def test_round_budget_advice_tracks_progress_without_a_stopping_rule(
    current, maximum, phase, remaining,
):
    result = discovery_budget_feedback(current, maximum)
    assert (result["round"], result["max_rounds"]) == (current, maximum)
    assert (result["phase"], result["rounds_remaining"]) == (phase, remaining)
    assert f"{remaining} round advances before the planned boundary" in result["message"]


def test_analysis_focus_selects_scope_and_explicit_model_keeps_original_prompt():
    assert guidance_prompt(SimpleNamespace(analysis_focus="model")) == ""
    assert "analysis_focus=harness" in guidance_prompt(SimpleNamespace(analysis_focus="harness"))


def test_analysis_focus_defaults_to_both_and_rejects_unknown_values():
    assert snapshot_guidance(SimpleNamespace())["analysis_focus"] == "both"
    with pytest.raises(ValueError, match="analysis_focus"):
        snapshot_guidance(SimpleNamespace(analysis_focus="other"))


def test_run_resume_uses_saved_focus_not_changed_defaults(tmp_path, monkeypatch):
    from autofa.config import RunConfig, load_config

    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path / "home"))
    save_config(Config(analysis_focus="harness"))
    run = RunConfig.create([], "test", [], load_config())
    expected = guidance_prompt(run)
    save_config(Config(analysis_focus="both"))
    restored = RunConfig.load(run.run_id)
    assert guidance_prompt(restored) == expected
    assert restored.analysis_focus == "harness"
    assert load_config().analysis_focus == "both"


def test_legacy_run_has_unchanged_model_scope(tmp_path, monkeypatch):
    from autofa.config import RunConfig

    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    directory = tmp_path / "runs" / "legacy"
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(json.dumps({
        "run_id": "legacy", "parquet_paths": [], "model_id": "test",
    }))
    assert guidance_prompt(RunConfig.load("legacy")) == ""


@pytest.mark.parametrize("harness_name", ["claude", "codex"])
def test_saved_scope_reaches_real_session_prompt_and_workspace(
    tmp_path, monkeypatch, harness_name,
):
    from autofa.config import RunConfig
    from autofa.data.parquet_store import Dataset
    from autofa.harness.claude import TurnResult
    from autofa.llm.hub import HubProbeResult
    from autofa.mcp.runtime import Runtime
    from autofa.session import runner

    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path / "home"))
    root = tmp_path / "skills"
    root.mkdir()
    skill = root / "harness.md"
    skill.write_text("Harness capability: explicit validation error codes")
    reference_folder = root / "reference"
    reference_folder.mkdir()
    categories = reference_folder / "categories.md"
    categories.write_text("Skill taxonomy category: ambiguous errors")
    run = RunConfig.create([], "test", [], Config(
        analysis_focus="harness", skills_dir=str(root), harness="claude",
    ))
    skill.unlink()
    categories.unlink()
    (run.run_dir() / "session.json").write_text(json.dumps({"session_id": "legacy-claude"}))
    cfg = Config(analysis_focus="model", library_root=str(tmp_path / "catalog"),
                 harness=harness_name)
    live = Runtime(dataset=Dataset(parquet_paths=[], model_id="test"),
                   library=FunctionLibrary(model_id="test"), run=run, config=cfg)
    monkeypatch.setattr("autofa.mcp.runtime._RUNTIME", live)
    monkeypatch.setattr(live, "ensure_hub_llm", lambda: SimpleNamespace(
        probe=lambda: HubProbeResult(ok=True, model="stub"),
    ))
    captured = []

    class Harness:
        async def run_session(self, **kwargs):
            captured.append(kwargs)
            return None

        async def run_turn(self, *args, **kwargs):
            captured.append(kwargs)
            return TurnResult(session_id=None)

    def build_server(*args):
        assert harness_name == "claude", "Codex must not initialize the Claude MCP server"
        return object()

    monkeypatch.setattr(runner, "load_runtime_from_run", lambda *args: live)
    monkeypatch.setattr(runner, "get_harness", lambda *args, **kwargs: Harness())
    monkeypatch.setattr(runner, "build_mcp_server", build_server)
    monkeypatch.setattr("autofa.data.remote_library_publish.maybe_publish_run",
                        lambda *args, **kwargs: None)
    monkeypatch.setattr("autofa.util.memory.apply_memory_tuning",
                        lambda *args, **kwargs: SimpleNamespace(summary=lambda: "test"))
    monkeypatch.setattr("autofa.diagnostics._DIAGNOSTICS", SimpleNamespace(
        configure=lambda *args: None, preflight_lines=lambda: [],
    ))
    monkeypatch.setattr("autofa.mcp.tools_eval.final_evaluate_functions",
                        lambda **kwargs: {"ok": True, "n": 0})
    monkeypatch.setattr("autofa.metrics.final_review.final_review_functions",
                        lambda **kwargs: {"ok": True, "unresolved": [], "n": 0})
    runner.run_session(run, cfg=cfg)
    assert len(captured) == 1
    assert captured[0]["resume"] == ("legacy-claude" if harness_name == "claude" else None)
    assert run.harness == harness_name
    assert json.loads((run.run_dir() / "config.json").read_text())["harness"] == harness_name
    if harness_name == "codex":
        assert captured[0]["mcp_servers"] == {"autofa": True}
    system = captured[0]["system_prompt"]
    workspace = (run.run_dir() / "autofa_protocol.md").read_text()
    for text in (system, workspace):
        assert "analysis_focus=harness" in text
        assert "Harness capability: explicit validation error codes" in text
        assert "Skill taxonomy category: ambiguous errors" in text
    autoloaded = (run.run_dir() / ("AGENTS.md" if harness_name == "codex"
                                  else "CLAUDE.md")).read_text()
    assert "Harness capability: explicit validation error codes" not in autoloaded
    assert "autofa_protocol.md" in autoloaded
    assert len(autoloaded) < 2000


def test_judge_gets_analysis_scope():
    from autofa.metrics.claude_judge import build_judge_prompt

    run = SimpleNamespace(analysis_focus="harness")
    prompt = build_judge_prompt(name="tool_failure", mechanism="Contract mismatch",
                                description="Missing feedback", source="", evidence=[],
                                analysis_guidance=guidance_prompt(run))
    assert "analysis_focus=harness" in prompt


def test_cli_focus_override_is_per_run(tmp_path, monkeypatch):
    from autofa.cli import _discovery_config
    from autofa.config import load_config

    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path / "home"))
    save_config(Config(analysis_focus="model"))
    cfg = _discovery_config(SimpleNamespace(analysis_focus="both"))
    assert cfg.analysis_focus == "both"
    assert load_config().analysis_focus == "model"


def test_resume_rejects_ignored_guidance_overrides(tmp_path, monkeypatch, capsys):
    from autofa.cli import main

    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path / "home"))
    monkeypatch.setattr("autofa.cli.load_dotenv", lambda: None)
    with pytest.raises(SystemExit, match="2"):
        main(["resume", "--run", "old", "--analysis-focus", "harness"])
    assert "Resume uses the saved scope" in capsys.readouterr().err


@pytest.fixture(scope="module")
def recipe_sources():
    prompt = build_system_prompt(exploration_ssot=False)
    section = prompt.split("### Implementation recipes", 1)[1].split(
        "Composed detectors likewise", 1,
    )[0]
    fragments = re.findall(r"```python\n(.*?)\n```", section, re.S)
    assert len(fragments) == 3
    assert "transforms_protected_literal" not in prompt
    # Test-only bindings exercise the API fragments, not detector templates in the prompt.
    bindings = {
        "Python": ("", "linked_actions"),
        "Hybrid": (
            'evidence_pattern = "value"\nevidence_budget = 12000\n'
            'exact_facts = {"requirement": item.get("prompt_text")}\n'
            'distinction_rubric = "Interpret the supplied evidence and its scope."\n',
            "label",
        ),
        "LLM": (
            'claim_evidence = {"reasoning": item.get("reasoning_text"), '
            '"execution": item.get("response_text")}\n'
            'claim_rubric = "Interpret the supplied evidence; missing facts are insufficient."\n',
            "label",
        ),
    }
    sources = {}
    for (kind, (setup, result)), fragment in zip(bindings.items(), fragments, strict=True):
        source = "def detect(item, llm_client=None):\n" + indent(
            setup + fragment + "\nreturn " + result, "    ",
        )
        accepted, reason = lint_detector_source(source)
        assert accepted, reason
        sources[kind] = source
    return sources


def recipe_item():
    return {
        "prompt_text": "Preserve the supplied value in the delivered record.",
        "reasoning_text": "Compare the recorded values and subsequent corrections.",
        "response_text": json.dumps({"steps": [
            {"step_id": 15, "source": "user", "message": "Keep value 7."},
            {"step_id": 20, "source": "assistant", "tool_calls": [
                {"id": "c1", "function": {"name": "write", "arguments": '{"value": 8}'}},
                {"id": "c2", "function": {"name": "read", "arguments": '{"key": "other"}'}},
            ]},
            {"step_id": 21, "source": "user", "message": "An unrelated observation."},
            {"step_id": 22, "source": "tool", "call_id": "c2", "paired_event_idx": 20,
             "observation": '{"value": 99}'},
            {"step_id": 23, "source": "tool", "call_id": "c1", "paired_event_idx": 20,
             "observation": '{"value": 8}'},
            {"step_id": 30, "source": "assistant", "tool_calls": [
                {"id": "c3", "function": {"name": "write", "arguments": '{"value": 7}'}},
            ]},
            {"step_id": 31, "source": "tool", "call_id": "c3", "paired_event_idx": 30,
             "observation": '{"value": 7}'},
            {"step_id": 40, "source": "assistant", "message": "Delivered corrected value 7."},
        ]}),
        "is_correct": True,
        "judge_text": "SECRET_GRADER_MUST_NOT_ENTER_CLASSIFIER",
    }


class Classifier:
    available = True

    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    def classify_options(self, text, rubric, **kwargs):
        self.calls.append((text, rubric, kwargs))
        return self.answer


def test_python_recipe_preserves_parallel_call_groups_and_later_correction(recipe_sources):
    actions = run_detector(recipe_sources["Python"], recipe_item())
    assert len(actions) == 2
    first, corrected = actions
    assert first["step"] == 2
    assert [call["arguments"] for call in first["calls"]] == [{"value": 8}, {"key": "other"}]
    # Results follow recorded linkage, not adjacency or position in the parallel call list.
    assert first["results"] == [
        {"step": 4, "value": {"value": 99}}, {"step": 5, "value": {"value": 8}},
    ]
    assert corrected["calls"][0]["call_id"] == "c3"
    assert corrected["results"] == [{"step": 7, "value": {"value": 7}}]


def test_python_recipe_preserves_missing_results_and_unparseable_arguments(recipe_sources):
    item = {"response_text": json.dumps({"steps": [
        {"source": "assistant", "tool_calls": [
            {"name": "write", "arguments": "{unparseable"},
        ]},
    ]})}
    actions = run_detector(recipe_sources["Python"], item)
    assert actions[0]["calls"][0]["arguments"] == "{unparseable"
    assert actions[0]["results"] == []
    assert run_detector(recipe_sources["Python"], {}) == []


@pytest.mark.parametrize("kind,label", [("Hybrid", "contradicted"), ("LLM", "supported")])
def test_semantic_recipes_preserve_explicit_outcomes_and_recovery(recipe_sources, kind, label):
    client = Classifier(label)
    assert run_detector(recipe_sources[kind], recipe_item(), client) == label
    text, _, options = client.calls[0]
    evidence = json.loads(text)
    if kind == "Hybrid":
        assert "Preserve the supplied value" in evidence["facts"]["requirement"]
        assert "Delivered corrected value 7" in evidence["selection"]["text"]
        assert evidence["selection"]["selection_complete"]
    else:
        assert "Preserve the supplied value" in evidence["task"]
        assert "subsequent corrections" in evidence["evidence"]["reasoning"]
        assert "Delivered corrected value 7" in evidence["evidence"]["execution"]
    assert "SECRET_GRADER" not in text
    assert "is_correct" not in text
    assert options["options"] == ["supported", "contradicted", "insufficient"]
    assert options["focus"] == "full"
    assert options["max_chars"] == len(text)


@pytest.mark.parametrize("kind", ["Hybrid", "LLM"])
def test_semantic_recipe_insufficient_abstains(recipe_sources, kind):
    with pytest.raises(DetectorAbstained, match="Required evidence"):
        run_detector(recipe_sources[kind], recipe_item(), Classifier("insufficient"))


def test_hybrid_recipe_missing_trace_abstains_before_classification(recipe_sources):
    client = Classifier("supported")
    with pytest.raises(DetectorAbstained, match="execution evidence unavailable"):
        run_detector(recipe_sources["Hybrid"], {}, client)
    assert client.calls == []


def test_hybrid_recipe_search_miss_preserves_scope_instead_of_becoming_negative(recipe_sources):
    item = {"response_text": json.dumps({"steps": [
        {"source": "assistant", "message": "No matching terms in this observation."},
    ]})}
    client = Classifier("insufficient")
    with pytest.raises(DetectorAbstained):
        run_detector(recipe_sources["Hybrid"], item, client)
    selection = json.loads(client.calls[0][0])["selection"]
    assert selection["matched_step_indices"] == []
    assert selection["full_trace"] is True


def test_hybrid_recipe_discloses_omissions_and_keeps_later_recovery(recipe_sources):
    from autofa.trace import steps

    item = recipe_item()
    original = json.loads(item["response_text"])
    original["steps"][5:5] = [
        {"source": "tool", "observation": "Unrelated output. " * 500} for _ in range(30)
    ]
    item["response_text"] = json.dumps(original)
    client = Classifier("insufficient")
    with pytest.raises(DetectorAbstained):
        run_detector(recipe_sources["Hybrid"], item, client)
    selection = json.loads(client.calls[0][0])["selection"]
    assert selection["full_trace"] is False
    assert selection["omitted_step_indices"]
    assert "Delivered corrected value 7" in selection["text"]
    # Selecting evidence leaves the complete source available for further inspection.
    assert len(steps(item)) == len(original["steps"])


def test_llm_recipe_preserves_supplied_evidence_beyond_default_window(recipe_sources):
    item = recipe_item()
    item["response_text"] = "Earlier unrelated evidence. " * 1000 + item["response_text"]
    client = Classifier("contradicted")
    assert run_detector(recipe_sources["LLM"], item, client) == "contradicted"
    text, _, options = client.calls[0]
    assert len(text) > 16000
    assert options["max_chars"] == len(text)
    assert json.loads(text)["evidence"]["execution"] == item["response_text"]


@pytest.mark.parametrize("kind", ["Hybrid", "LLM"])
def test_semantic_recipe_does_not_convert_provider_errors_to_negatives(recipe_sources, kind):
    class FailingClassifier(Classifier):
        def classify_options(self, *args, **kwargs):
            raise RuntimeError("invalid classification response")

    with pytest.raises(RuntimeError, match="invalid classification response"):
        run_detector(recipe_sources[kind], recipe_item(), FailingClassifier(None))


@pytest.fixture
def saved_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "_RUNTIME", SimpleNamespace(
        run=SimpleNamespace(run_dir=lambda: tmp_path),
    ))
    original = {
        "payload": {
            "text": 'Unicode 雪 🧪; quote " and slash \\ and newline\n' * 1400,
            "later_recovery": {"present": True, "quote": "The correction passed."},
            "missing": None,
        },
        "unselected": {"evidence": "Keep the original result intact."},
    }
    saved = save_tool_result(original)
    return saved["result_id"], original


def _pages(result_id, pointer=""):
    pages, offset = [], 0
    while offset is not None:
        page = read_tool_result(result_id, pointer=pointer, offset=offset)
        assert page["ok"]
        pages.append(page)
        offset = page["next_offset"]
    return pages


def _run_paging_recipe(pages, result_id, *, pointer="", transport="string", max_pages=100):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is needed to execute the discovery paging recipe")
    recipes = re.findall(r"```(?:javascript|js)\n(.*?)```", SAVED_RESULT_PAGING_GUIDANCE, re.S)
    assert len(recipes) == 1, "Exercise the actual shared recipe, not a copied implementation"
    fixture = {
        "pages": pages, "resultId": result_id, "pointer": pointer,
        "transport": transport, "maxPages": max_pages,
    }
    script = "const fixture = " + json.dumps(fixture) + ";\n" + """
const calls = [];
const tools = {
  async read_tool_result(args) {
    calls.push(args);
    const page = fixture.pages[calls.length - 1];
    if (!page) throw Error("Unexpected extra read: retry or restart");
    const serialized = fixture.transport === "string" ||
      (fixture.transport === "mixed" && calls.length % 2 === 1);
    return serialized ? JSON.stringify(page) : page;
  }
};
""" + recipes[0] + """
(async () => {
  try {
    const value = await readSavedJson(fixture.resultId, fixture.pointer, fixture.maxPages);
    console.log(JSON.stringify({value, calls}));
  } catch (error) {
    console.log(JSON.stringify({
      error: String(error.message || error), calls,
      continuation: {result_id: error.result_id, pointer: error.pointer,
                     offset: error.offset, text: error.text}
    }));
  }
})();
"""
    output = subprocess.run(
        [node], input=script, text=True, capture_output=True, check=True, timeout=10,
    )
    return json.loads(output.stdout)


def test_paging_recipe_reconstructs_full_evidence_and_stops_on_final_incomplete_page(
    saved_evidence,
):
    result_id, original = saved_evidence
    pointer = "/payload"
    pages = _pages(result_id, pointer)
    assert len(pages) > 2
    # This is the precise server response that trapped the original agent loop.
    assert pages[-1]["next_offset"] is None
    assert pages[-1]["complete"] is False
    assert pages[-1]["full_result_complete"] is False

    # Alternate serialized and object pages so both tool-result transports are parsed.
    result = _run_paging_recipe(pages, result_id, pointer=pointer, transport="mixed")

    assert "error" not in result
    assert result["value"] == original["payload"]
    assert [call["offset"] for call in result["calls"]] == [p["offset"] for p in pages]
    assert all(call["result_id"] == result_id for call in result["calls"])
    assert all(call["pointer"] == pointer for call in result["calls"])
    assert all(call.get("describe", False) is False for call in result["calls"])
    assert all(call.get("limit", PAGE_CHARS) == PAGE_CHARS for call in result["calls"])


MISSING = object()


# Values a loosened check would accept: a truthy non-true "ok", a non-string list "text",
# an equal-looking string "offset", and a non-advancing or missing (not null) "next_offset".
@pytest.mark.parametrize("field,value", [
    ("ok", "true"), ("text", []), ("offset", "12000"),
    ("next_offset", PAGE_CHARS), ("next_offset", MISSING),
])
def test_paging_recipe_rejects_errors_and_malformed_pages_without_retry_or_reset(
    saved_evidence, field, value,
):
    result_id, _ = saved_evidence
    pages = deepcopy(_pages(result_id)[:2])
    if value is MISSING:
        pages[1].pop(field)
    else:
        pages[1][field] = value
    if field == "ok" and value is not MISSING:
        pages[1]["error"] = "Source evidence unavailable"

    result = _run_paging_recipe(pages, result_id)

    assert "value" not in result
    assert result["error"]
    assert "Unexpected extra read" not in result["error"]
    assert [call["offset"] for call in result["calls"]] == [0, PAGE_CHARS]


def test_paging_recipe_limit_exposes_continuation_without_claiming_partial_evidence_complete(
    saved_evidence,
):
    result_id, _ = saved_evidence
    pages = _pages(result_id)
    result = _run_paging_recipe(pages, result_id, max_pages=2)

    assert "value" not in result
    assert "incomplete" in result["error"]
    assert result["continuation"] == {
        "result_id": result_id, "pointer": "", "offset": 2 * PAGE_CHARS,
        "text": pages[0]["text"] + pages[1]["text"],
    }
    assert [call["offset"] for call in result["calls"]] == [0, PAGE_CHARS]
    # The limit belongs to this convenience recipe, not the evidence API.
    continuation = read_tool_result(result_id, offset=2 * PAGE_CHARS)
    assert continuation["ok"]
    assert continuation["text"] == pages[2]["text"]


def test_paging_recipe_single_page_value_retains_null_and_does_not_fetch_sibling_evidence(
    saved_evidence,
):
    result_id, _ = saved_evidence
    pointer = "/payload/missing"
    result = _run_paging_recipe(_pages(result_id, pointer), result_id, pointer=pointer)

    assert "error" not in result
    assert result["value"] is None
    assert [call["pointer"] for call in result["calls"]] == [pointer]


# The discover skill's implementation_recipes.md fragments, bound to synthetic inputs.
@pytest.fixture(scope="module")
def recipes():
    reference = Path(__file__).parents[1] / "src/autofa/skills/discover/implementation_recipes.md"
    fragments = re.findall(r"```python\n(.*?)\n```", reference.read_text(), re.S)
    bindings = {
        "requirements": (
            'selected_tool = "set_value"\nrequired_fields = {"entity", "value"}\n',
            '{"requirements": requirements, "comparisons": comparisons}',
        ),
        "state": ('entity_field = "entity"\nstate_field = "value"\n', "transitions"),
        "artifact": ('artifact_idx = 7\n',
                     '{"catalog": catalog, "document": document, "text": located_text}'),
        "reuse": ('import json\n',
                  'json.dumps({"extracted": extracted, "parent": parent_result})'),
    }
    result = {}
    for (name, (setup, expression)), fragment in zip(bindings.items(), fragments, strict=True):
        source = "def detect(item, llm_client=None):\n" + indent(
            setup + fragment + "\nreturn " + expression, "    ",
        )
        accepted, reason = lint_detector_source(source)
        assert accepted, reason
        result[name] = source
    return result


def structured_item():
    return {
        "input_json": {
            "messages": [{"role": "user", "content": "Preserve the requested value."}],
            "tools": [{"type": "function", "function": {"name": "set_value",
                       "parameters": {"type": "object", "required": ["entity", "value"]}}}],
        },
        "response_text": json.dumps({
            "execution": {"tool_definitions": [{"def_idx": 6, "name": "set_value",
                "def_json": '{"name":"set_value","description":"Recorded contract"}'}]},
            "steps": [
                {"step_id": 10, "source": "assistant", "tool_calls": [
                    {"id": "c1", "function": {"name": "set_value",
                                             "arguments": '{"entity":"A","value":8}'}},
                    {"id": "c2", "function": {"name": "set_value",
                                             "arguments": '{"entity":"B"}'}},
                ]},
                {"step_id": 13, "source": "tool", "call_id": "c2", "paired_event_idx": 10,
                 "observation": {"entity": "B", "value": None}},
                {"step_id": 12, "source": "tool", "call_id": "c1", "paired_event_idx": 10,
                 "observation": {"entity": "A", "value": 8}},
                {"step_id": 20, "source": "assistant", "tool_calls": [
                    {"id": "c3", "function": {"name": "set_value",
                                             "arguments": '{"entity":"A","value":7}'}}
                ]},
                {"step_id": 21, "source": "tool", "call_id": "c3", "paired_event_idx": 20,
                 "observation": {"entity": "A", "value": 7}},
                {"step_id": 25, "source": "tool", "observation": {"entity": "B", "value": 0}},
            ],
        }),
    }


def test_requirements_keep_initial_and_recorded_definitions_with_actual_arguments(recipes):
    result = run_detector(recipes["requirements"], structured_item())
    requirements = result["requirements"]
    assert "Preserve the requested value" in requirements["task"]
    assert requirements["initial_tools"][0]["function"]["parameters"]["required"] == [
        "entity", "value",
    ]
    assert "Recorded contract" in requirements["recorded_tools"][0]["def_json"]
    assert requirements["availability"]["tool_definitions"]["status"] == "available"
    comparisons = result["comparisons"]
    assert [row["missing_required_fields"] for row in comparisons] == [[], ["value"], []]
    assert comparisons[0]["arguments"] == {"entity": "A", "value": 8}
    assert [row["call_id"] for row in comparisons] == ["c1", "c2", "c3"]
    # Linking is to the whole calling step, never an invented per-call assignment.
    assert comparisons[0]["results"] == comparisons[1]["results"] == [
        {"step": 2, "value": {"entity": "B", "value": None}},
        {"step": 3, "value": {"entity": "A", "value": 8}},
    ]


def test_malformed_arguments_and_missing_definitions_remain_explicit_unknowns(recipes):
    item = {"response_text": json.dumps({"steps": [
        {"source": "assistant", "tool_calls": [
            {"name": "set_value", "arguments": "{broken"},
        ]},
    ]})}
    result = run_detector(recipes["requirements"], item)
    assert result["requirements"]["initial_tools"] == []
    assert result["requirements"]["availability"]["tool_definitions"]["status"] == "unavailable"
    comparison = result["comparisons"][0]
    assert comparison["missing_required_fields"] is None
    assert comparison["arguments"] == "{broken" and comparison["results"] == []


def test_state_tracks_observations_and_later_correction_instead_of_attempted_writes(recipes):
    result = run_detector(recipes["state"], structured_item())
    assert [row["entity"] for row in result] == ["B", "A", "A", "B"]
    assert [row["observed"] for row in result] == [None, 8, 7, 0]
    assert result[2] == {"step": 5, "entity": "A", "previous_known": True,
                         "previous": 8, "observed": 7}
    # Explicit null is retained separately from no earlier observation.
    assert result[0]["previous_known"] is False
    assert result[3]["previous_known"] is True and result[3]["previous"] is None
    assert run_detector(recipes["state"], {}) == []
    item = structured_item()
    data = json.loads(item["response_text"])
    data["steps"] = data["steps"][:4]  # A correction is attempted but its result is absent.
    item["response_text"] = json.dumps(data)
    assert [row["observed"] for row in run_detector(recipes["state"], item)] == [None, 8]


@pytest.fixture
def retained_artifact(tmp_path, monkeypatch):
    body = b"Requested value: 7\nDelivered value: 7\n"
    calls = []

    class LocalProvider:
        base_url = "https://synthetic.example"

        def download_artifact(self, run_id, task_id, repeat, artifact_idx, dest):
            calls.append((run_id, task_id, repeat, artifact_idx))
            dest.write_bytes(body)
            return dest

    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    monkeypatch.setattr(ArtifactStore, "_provider", lambda self: LocalProvider())
    item = {"response_text": json.dumps({"steps": [], "execution": {
        "rollout": {"run_id": "run", "task_id": "task", "repeat": 0},
        "source": {"kind": "trace_api", "base_url": "https://synthetic.example",
                   "trace_export_sha256": "a" * 64, "export_generation": "generation"},
        "artifacts": [{"artifact_idx": 7, "name": "delivered.txt", "mime": "text/plain",
                       "size_bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}],
    }})}
    return item, body, calls


def test_artifact_recipe_uses_source_identity_and_retains_exact_text_locations(
    recipes, retained_artifact,
):
    item, body, calls = retained_artifact
    evidence = run_detector(recipes["artifact"], item)
    assert calls == [("run", "task", 0, 7)]  # Not catalog ordinal zero.
    assert evidence["catalog"][0]["artifact_idx"] == 7
    assert evidence["document"]["sha256"] == hashlib.sha256(body).hexdigest()
    assert [row["locator"] for row in evidence["text"]] == ["line:1", "line:2"]
    assert "".join(row["text"] for row in evidence["text"]) == body.decode()


def test_artifact_recipe_abstains_when_source_bytes_are_unavailable(recipes):
    with pytest.raises(DetectorAbstained, match="artifact evidence is incomplete or unavailable"):
        run_detector(recipes["artifact"], {})


def reusable_library(parent_source="def parent(item): return 'not_supported'"):
    library = FunctionLibrary(functions=[FeatureFunction(
        name="parent", source=parent_source, description="Synthetic existing detector",
    )])
    library.helpers.upsert(HelperFunction(
        name="source_text", entrypoint="extract",
        source="def extract(item): return {'text': item.get('response_text')}",
        output_schema={"type": "object", "required": ["text"]},
    ))
    return library


def test_reuse_recipe_preserves_structured_results_and_exact_labels_with_pinned_versions(recipes):
    library = reusable_library()
    bundle = build_bundle(library, depends_on=["parent"], uses_helpers={"extract": "source_text"})
    library.get("parent").source = "def parent(item): return 'supported'"
    library.helpers.upsert(HelperFunction(name="source_text", entrypoint="extract",
                                         source="def extract(item): return 'changed'"))
    result = json.loads(run_detector(recipes["reuse"], {"response_text": "Source"}, bundle=bundle))
    assert result == {"extracted": {"text": "Source"}, "parent": "not_supported"}


@pytest.mark.parametrize("parent_source,error,match", [
    ("def parent(item): library.abstain('Missing evidence')",
     DetectorAbstained, "Missing evidence"),
    ("def parent(item): return json.loads(item['response_text'])['value']", KeyError, "value"),
])
def test_reuse_recipe_propagates_abstentions_and_execution_errors(
    recipes, parent_source, error, match,
):
    library = reusable_library(parent_source)
    bundle = build_bundle(library, depends_on=["parent"], uses_helpers={"extract": "source_text"})
    with pytest.raises(error, match=match):
        run_detector(recipes["reuse"], {"response_text": "{}"}, bundle=bundle)
