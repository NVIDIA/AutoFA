# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Objective presets and weights drive ranking; retired rank modes fail when loaded."""

from __future__ import annotations

import json
import tomllib
from dataclasses import replace
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console

from autofa import cli, config
from autofa.config import (
    Config,
    RunConfig,
    ensure_default_config,
    load_config,
    resolve_objective_axes,
)
from autofa.metrics.pareto import (
    _scalar_score,
    frontier_value,
    library_frontier_metrics,
    library_frontier_points,
    pareto_front,
    point_from_scores,
    rank_functions,
)
from autofa.objectives import (
    OBJECTIVE_FIELDS,
    OBJECTIVE_LABELS,
    apply_preset,
    config_for_new_discovery,
    config_for_run,
    config_from_run_dir,
    discovery_strategy,
    judge_objective_guidance,
    preset_name,
    snapshot_objectives,
    weights_by_axis,
)
from autofa.ui import settings

from support.config import save_config

pytestmark = pytest.mark.usefixtures("installed_codex")

LEGACY_STANDARD_AXES = ("ppv_at_pi0", "recall", "behavior_purity")
# Retired four-axis set: still loadable, but new discovery selects current Recall.
LEGACY = {
    "objective_axes": ["ppv_at_pi0", "recall", "behavior_purity", "behavior_specificity"],
    "objective_rank": "geometric_mean",
    "objective_weights": [],
}


def write_settings(path, values):
    path.write_text("# My settings\n" + "\n".join(
        f"{key} = {config._toml_value(value)}  # retain {key}"
        for key, value in values.items()
    ) + '\n[private]\nlabel = "retained"\n')


def candidates():
    return [{"name": name, "scores": dict(zip(OBJECTIVE_LABELS, values, strict=True))}
            for name, values in [
                ("precise", [.95, .35, .8]),
                ("broad", [.35, .95, .8]),
                ("balanced", [.65, .65, .8]),
            ]]


@pytest.mark.parametrize("preset,winner", [
    ("balanced", "balanced"), ("precision", "precise"), ("recall", "broad"),
])
def test_presets_control_actual_ranking(preset, winner):
    cfg = apply_preset(Config(), preset)
    assert rank_functions(candidates(), config=cfg)[0]["name"] == winner
    # Weights belong to named axes, so reordering both lists keeps the same ranking.
    reordered = replace(cfg, objective_axes=list(reversed(cfg.objective_axes)),
                        objective_weights=list(reversed(cfg.objective_weights)))
    assert weights_by_axis(reordered) == weights_by_axis(cfg)
    assert rank_functions(candidates(), config=reordered) == rank_functions(
        candidates(), config=cfg,
    )


def test_positive_preferences_do_not_change_pareto_membership():
    functions = [{**row, "scores": {**row["scores"], "evaluated": True}}
                 for row in candidates()]
    fronts = [pareto_front(library_frontier_metrics(
        functions, config=apply_preset(Config(), preset),
    )["points"]) for preset in ("balanced", "precision", "recall")]
    assert fronts[0] == fronts[1] == fronts[2]
    assert len(fronts[0]) == 3


def test_zero_weight_disables_axis_without_changing_weight_alignment(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    cfg = replace(Config(), objective_weights=[0, 50, 50])
    settings.save_analysis_settings(cfg)
    loaded = load_config()
    assert loaded.objective_axes == list(OBJECTIVE_LABELS)
    assert loaded.objective_weights == [0, 50, 50]
    assert resolve_objective_axes(loaded) == list(OBJECTIVE_LABELS)[1:]
    assert rank_functions(candidates(), config=loaded)[0]["name"] == "broad"


@pytest.mark.parametrize("weights", [
    [0, 0, 0], [-1, 1, 1], [float("nan"), 1, 1], [True, 1, 1], [1, 1],
], ids=["all-zero", "negative", "nan", "bool", "wrong-length"])
def test_invalid_weights_fail_visibly(weights):
    with pytest.raises(ValueError):
        weights_by_axis(replace(Config(), objective_weights=weights))


def test_default_is_recall_and_proportional_weights_have_same_ranking(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    ensure_default_config()
    cfg = load_config()
    assert preset_name(cfg) == "Recall"
    assert cfg.objective_rank == "geometric_mean"
    assert cfg.objective_axes == ["ppv_at_pi0", "task_recall", "behavior_purity"]
    other = replace(cfg, objective_weights=[40, 100, 60])
    assert preset_name(other) == "Recall"
    assert rank_functions(candidates(), config=cfg) == rank_functions(candidates(), config=other)


def test_new_discovery_preserves_current_geometric_preferences():
    preset = apply_preset(Config(), "precision")
    assert config_for_new_discovery(preset) is preset
    # Reordered custom weights, including an inactive axis, are current preferences too.
    cfg = Config(objective_axes=["behavior_purity", "ppv_at_pi0", "task_recall"],
                 objective_weights=[65, 35, 0])
    assert config_for_new_discovery(cfg) is cfg
    assert weights_by_axis(cfg) == {"behavior_purity": .65, "ppv_at_pi0": .35,
                                    "task_recall": 0}
    # Invalid current weights fail visibly instead of being replaced by a preset.
    with pytest.raises(ValueError, match="one weight"):
        config_for_new_discovery(Config(objective_weights=[20, 80]))


def test_standard_presets_reward_detector_task_breadth_over_repeated_attempt_matches():
    functions = [
        {"name": "repeated_task", "scores": {
            "ppv_at_pi0": .95, "recall": .8, "task_recall": .1, "behavior_purity": 1,
        }},
        {"name": "cross_task_parent", "scores": {
            "ppv_at_pi0": .95, "recall": .2, "task_recall": .6, "behavior_purity": 1,
        }},
    ]
    cfg = apply_preset(Config(), "precision")
    assert rank_functions(functions, config=cfg)[0]["name"] == "cross_task_parent"
    old = replace(cfg, objective_axes=list(LEGACY_STANDARD_AXES))
    assert rank_functions(functions, config=old)[0]["name"] == "repeated_task"


@pytest.mark.parametrize("missing", [None, float("nan"), 1.1])
def test_task_recall_unavailable_never_uses_attempt_recall_or_fabricates_zero(missing):
    cfg = apply_preset(Config(), "recall")
    fn = {"name": "unavailable", "scores": {
        "evaluated": True, "ppv_at_pi0": 1, "recall": 1, "sensitivity": 1,
        "task_recall": missing, "behavior_purity": 1,
    }}
    assert point_from_scores(fn["scores"], cfg.objective_axes) is None
    candidate = rank_functions([fn], config=cfg)[0]
    assert not candidate["ranking"]["eligible"]
    assert candidate["ranking"]["status"] == "task_coverage_unavailable"
    assert library_frontier_metrics([fn], config=cfg)["points"] == []
    del fn["scores"]["task_recall"]
    assert point_from_scores(fn["scores"], cfg.objective_axes) is None


def test_task_recall_zero_is_measured_and_inactive_task_axis_does_not_block_ranking():
    cfg = Config()
    fn = {"name": "zero", "scores": {
        "evaluated": True, "ppv_at_pi0": 1, "task_recall": 0, "behavior_purity": 1,
    }}
    assert point_from_scores(fn["scores"], cfg.objective_axes) == (1, 0, 1)
    assert rank_functions([fn], config=cfg)[0]["ranking"]["eligible"]
    del fn["scores"]["task_recall"]
    assert rank_functions(
        [fn], config=replace(cfg, objective_weights=[50, 0, 50]),
    )[0]["ranking"]["eligible"]


@pytest.mark.parametrize("axis", ["recall", "task_recall"])
def test_both_recall_axes_steer_discovery_without_reinterpreting_the_metric(axis):
    cfg = Config(objective_axes=["ppv_at_pi0", axis, "behavior_purity"],
                 objective_weights=[20, 50, 30])
    assert discovery_strategy(cfg) == "recall"


def test_real_detector_task_ranking_checks_current_claim_and_preserves_saved_metric():
    from copy import deepcopy

    from autofa.metrics.claim_contract import claim_hash

    claim = {"source": "def detect(item): return True", "description": "Original claim",
             "mechanism": "Same causal mechanism"}
    fingerprint = claim_hash(**claim)
    fn = {"name": "parent", **claim, "scores": {
        "evaluated": True, "task_recall": .6, "task_recall_scope": "full_selected_corpus",
        "evaluation_claim_hash": fingerprint, "task_recall_claim_hash": fingerprint,
    }}
    cfg = Config(objective_axes=["task_recall"], objective_weights=[1])
    original = deepcopy(fn["scores"])
    assert rank_functions([fn], config=cfg)[0]["ranking"]["eligible"]
    fn["description"] = "Changed claim"
    assert not rank_functions([fn], config=cfg)[0]["ranking"]["eligible"]
    assert library_frontier_metrics([fn], config=cfg)["points"] == []
    assert fn["scores"] == original


def test_run_snapshot_does_not_follow_later_global_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    cfg = apply_preset(Config(), "recall")
    run = RunConfig.create([], "model", ["bench"], cfg)
    cfg.objective_weights[0] = 999
    saved = RunConfig.load(run.run_id)
    effective = config_for_run(apply_preset(Config(), "precision"), saved)
    assert preset_name(effective) == "Recall"
    assert preset_name(config_from_run_dir(run.run_dir())) == "Recall"
    assert snapshot_objectives(effective) == saved.objective_settings
    from autofa.mcp.runtime import load_runtime_from_run

    rt = load_runtime_from_run(saved, apply_preset(Config(), "precision"))
    assert rank_functions(candidates(), config=rt.config)[0]["name"] == "broad"


def test_slider_keyboard_autosaves_and_preserves_other_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    path = ensure_default_config()
    path.write_text(path.read_text() + '\n# Extension\ncustom_key = "preserve"\n')
    cfg = load_config()
    calls = []
    save = settings.save_analysis_settings

    def save_and_check(candidate):
        save(candidate)
        calls.append(load_config().objective_weights)

    monkeypatch.setattr(settings, "save_analysis_settings", save_and_check)
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        # Precision slider: 20 -> 25. Back retains the autosaved change.
        pipe.send_text("\x1b[B\x1b[B\x1b[Cq")
        result = settings.configure_objectives(cfg, persistent=True)
    assert calls == [[25, 50, 30]]
    assert preset_name(result) == "Custom"
    assert load_config().objective_weights == result.objective_weights
    assert '# Extension\ncustom_key = "preserve"' in path.read_text()


def test_slider_normalizes_retired_objective_before_edit_save_and_launch(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    axes = [*LEGACY_STANDARD_AXES, "behavior_specificity"]
    cfg = Config(objective_axes=axes, objective_weights=[1] * 4)
    save_config(cfg)
    before = snapshot_objectives(cfg)
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        # The first slider edits precision while retaining nonzero task recall.
        pipe.send_text("\x1b[B\x1b[B\x1b[Cq")
        result = settings.configure_objectives(load_config(), persistent=True)
    assert result.objective_axes == list(OBJECTIVE_LABELS)
    assert result.objective_weights == [25, 50, 30]
    assert result.objective_rank == "geometric_mean"
    assert snapshot_objectives(cfg) == before
    assert snapshot_objectives(load_config()) == snapshot_objectives(result)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False)
    launched = cli._discovery_config(SimpleNamespace())
    assert snapshot_objectives(launched) == snapshot_objectives(result)
    run = RunConfig.create([], "model", ["bench"], launched)
    assert RunConfig.load(run.run_id).objective_settings == snapshot_objectives(result)


def test_slider_preserves_current_custom_axis_weights_when_reordering_for_display():
    cfg = Config(objective_axes=["behavior_purity", "task_recall", "ppv_at_pi0"],
                 objective_weights=[70, 0, 30])
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        pipe.send_text("\x1b[B\x1b[B\x1b[Cq")
        result = settings.configure_objectives(cfg)
    assert result.objective_axes == list(OBJECTIVE_LABELS)
    assert result.objective_weights == [35, 0, 70]
    assert cfg.objective_weights == [70, 0, 30]


def test_slider_failed_save_keeps_previous_values(monkeypatch):
    def fail(_):
        raise OSError("Disk full")

    monkeypatch.setattr(settings, "save_analysis_settings", fail)
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        pipe.send_text("\x1b[B\x1b[Cq")
        result = settings.configure_objectives(Config(), persistent=True)
    assert preset_name(result) == "Recall"


def test_discovery_preset_is_per_run_and_does_not_replace_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    before = ensure_default_config().read_text()
    selections = iter(["objective", "precision", "done"])
    def choose(title, options, **kwargs):
        value = next(selections)
        return next(option for option in options if option.value == value)

    monkeypatch.setattr(settings, "select_option", choose)
    result = settings.configure_analysis(Config())
    assert preset_name(result) == "Precision"
    assert ensure_default_config().read_text() == before


def test_guidance_uses_weighted_preferences_in_native_session():
    from autofa.session.prompts import build_system_prompt

    cfg = apply_preset(Config(), "recall")
    text = build_system_prompt(objective_config=cfg, exploration_ssot=False)
    assert "recall=50.0%" in text


def test_balanced_geometric_does_not_hide_negligible_recall():
    functions = [
        {"name": "tiny", "scores": {"ppv_at_pi0": 1, "task_recall": .001,
                                    "behavior_purity": 1}},
        {"name": "useful", "scores": {"ppv_at_pi0": .75, "task_recall": .4,
                                      "behavior_purity": .85}},
    ]
    cfg = apply_preset(Config(), "balanced")
    assert rank_functions(functions, config=cfg)[0]["name"] == "useful"


def test_geometric_scalar_has_real_zero_and_handles_tiny_values_and_inactive_axes():
    def score(values, weights):
        return _scalar_score(values, weights=weights)

    assert score([1, 0, 1], [1, 1, 1]) == 0
    assert score([0, .5, .8], [0, 1, 1]) == pytest.approx((.5 * .8) ** .5)
    assert score([1e-300, 1e-300, 1], [1, 1, 1]) == pytest.approx(1e-200, rel=1e-12, abs=0)


def test_geometric_frontier_matches_ranking_and_named_axis_order():
    cfg = apply_preset(Config(), "recall")
    values = [.3, .8, .9]
    fn = {"name": "family", "scores": {
        **dict(zip(OBJECTIVE_LABELS, values, strict=True)), "evaluated": True,
    }}
    metrics = library_frontier_metrics([fn], config=cfg)
    expected = .3 ** .2 * .8 ** .5 * .9 ** .3
    assert metrics["frontier_value"] == pytest.approx(expected)
    reversed_cfg = replace(cfg, objective_axes=list(reversed(cfg.objective_axes)),
                           objective_weights=list(reversed(cfg.objective_weights)))
    assert library_frontier_metrics([fn], config=reversed_cfg)["frontier_value"] == (
        pytest.approx(expected)
    )
    assert frontier_value([values], config=cfg) == pytest.approx(expected)


def test_four_axis_run_keeps_snapshot_and_ranking(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    historical = Config(objective_axes=[*LEGACY_STANDARD_AXES, "behavior_specificity"],
                        objective_weights=[25] * 4)
    run = RunConfig.create([], "model", ["bench"], historical)
    restored = config_for_run(Config(), RunConfig.load(run.run_id))
    assert snapshot_objectives(restored) == snapshot_objectives(historical)
    assert preset_name(restored) != "Balanced"
    functions = [
        {"name": "specific", "scores": {"ppv_at_pi0": .9, "recall": .4,
                                         "behavior_purity": .8,
                                         "behavior_specificity": 1}},
        {"name": "broad", "scores": {"ppv_at_pi0": .8, "recall": .5,
                                      "behavior_purity": .8,
                                      "behavior_specificity": 0}},
    ]
    assert rank_functions(functions, config=restored)[0]["name"] == "specific"
    assert snapshot_objectives(config_from_run_dir(run.run_dir())) == snapshot_objectives(
        historical,
    )


@pytest.mark.parametrize("rank", ["weighted_sum", "product", "lexicographic"])
def test_retired_rank_in_saved_run_fails_instead_of_changing_its_scoring(
    rank, tmp_path, monkeypatch,
):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    run = RunConfig.create([], "model", ["bench"], Config())
    path = run.run_dir() / "config.json"
    data = json.loads(path.read_text())
    data["objective_settings"]["objective_rank"] = rank
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="objective_rank must be 'geometric_mean'"):
        config_for_run(Config(), RunConfig.load(run.run_id))
    with pytest.raises(ValueError, match="objective_rank must be 'geometric_mean'"):
        config_from_run_dir(run.run_dir())


def test_saved_run_ignores_snapshot_fields_this_version_does_not_define(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    cfg = apply_preset(Config(), "precision")
    run = RunConfig.create([], "model", ["bench"], cfg)
    path = run.run_dir() / "config.json"
    data = json.loads(path.read_text())
    data["objective_settings"]["objective_primary"] = "recall"
    path.write_text(json.dumps(data))
    restored = config_for_run(Config(), RunConfig.load(run.run_id))
    assert snapshot_objectives(restored) == snapshot_objectives(cfg)
    assert snapshot_objectives(config_from_run_dir(run.run_dir())) == snapshot_objectives(cfg)


def test_search_strategy_does_not_change_judge_guidance():
    cfg = apply_preset(Config(), "recall")
    assert discovery_strategy(cfg) == "recall"
    assert judge_objective_guidance(cfg) == judge_objective_guidance("balanced")


@pytest.mark.parametrize("weights,expected", [
    ([40, 35, 25], "precision"), ([35, 40, 25], "recall"), ([20, 20, 60], "balanced"),
])
def test_custom_weights_select_reproducible_discovery_direction(weights, expected):
    assert discovery_strategy(replace(Config(), objective_weights=weights)) == expected


def test_specificity_remains_opt_in_advanced_axis():
    cfg = Config(objective_axes=["behavior_specificity"], objective_weights=[1])
    functions = [
        {"name": "broad", "scores": {"behavior_specificity": .2}},
        {"name": "specific", "scores": {"behavior_specificity": .9}},
    ]
    assert rank_functions(functions, config=cfg)[0]["name"] == "specific"
    assert preset_name(cfg) == "Custom"
    assert discovery_strategy(cfg) == "balanced"


def test_frontier_points_uses_explicit_config_axes():
    cfg = Config(objective_axes=["recall"], objective_weights=[1])
    functions = [{"name": "family", "scores": {"recall": .6, "evaluated": True}}]
    assert library_frontier_points(functions, config=cfg) == [(.6,)]


def test_active_objective_strategy_reuses_judgment_without_rewriting_history(
    tmp_path, monkeypatch,
):
    from autofa.metrics.claim_contract import JUDGE_CONTRACT_VERSION, claim_hash
    from autofa.metrics.claude_judge import judge_policy_hash

    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    claim = {"source": "def detect(item): return True", "description": "Claim",
             "mechanism": "Claim"}
    fn = {"name": "family", **claim, "scores": {
        "evaluated": True, "ppv_at_pi0": .9, "recall": .6, "task_recall": .5,
        "behavior_purity": 1,
        "behavior_n_judged": 1, "behavior_n_yes": 1, "behavior_n_no": 0,
        "behavior_n_uncertain": 0, "behavior_n_errors": 0,
        "judge_contract_version": JUDGE_CONTRACT_VERSION,
        "judge_objective_strategy": "recall", "judge_policy_hash": judge_policy_hash(),
        "claim_hash": claim_hash(**claim),
        "evaluation_claim_hash": claim_hash(**claim),
        "task_recall_claim_hash": claim_hash(**claim),
        "task_recall_scope": "full_selected_corpus",
    }}
    # Search preferences change ranking, not the claim's validation standard.
    causal = Config(behavior_strictness="causal")
    assert rank_functions([fn], config=apply_preset(causal, "recall"))[0]["ranking"]["eligible"]
    assert rank_functions(
        [fn], config=apply_preset(causal, "precision"),
    )[0]["ranking"]["eligible"]
    assert library_frontier_metrics(
        [fn], config=apply_preset(causal, "precision"),
    )["points"]
    # Read-only report helpers with no active strategy use recorded provenance.
    assert rank_functions([fn])[0]["ranking"]["eligible"]
    assert library_frontier_metrics([fn])["points"]
    assert fn["scores"]["judge_objective_strategy"] == "recall"
    assert "behavior_judgment_stale" not in fn["scores"]


# --- Saved objective settings and run snapshots load without rewriting choices ---


def test_new_configuration_and_run_use_task_recall_and_preset_can_opt_old_settings_in(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    fresh = config.load_config()
    assert fresh.objective_axes == ["ppv_at_pi0", "task_recall", "behavior_purity"]
    run = config.RunConfig.create([], "model", ["bench"], fresh)
    assert config.RunConfig.load(run.run_id).objective_settings == snapshot_objectives(fresh)
    write_settings(tmp_path / "config.toml", {"top_x": 7})
    # Omitted objective fields take the current defaults.
    assert snapshot_objectives(config.load_config()) == snapshot_objectives(fresh)
    write_settings(tmp_path / "config.toml", LEGACY)
    historical = config.load_config()
    assert historical.objective_axes == LEGACY["objective_axes"]
    opted_in = apply_preset(historical, "recall")
    save_config(opted_in)
    assert preset_name(config.load_config()) == "Recall"
    new_run = config.RunConfig.create([], "model", ["bench"], opted_in)
    assert config.RunConfig.load(new_run.run_id).objective_settings == snapshot_objectives(opted_in)


def test_explicit_custom_objective_survives_loading(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    path = tmp_path / "config.toml"
    write_settings(path, LEGACY)
    cfg = config.load_config()
    assert snapshot_objectives(cfg) == LEGACY
    assert preset_name(cfg) == "Custom"
    assert discovery_strategy(cfg) == "balanced"
    content = path.read_text()
    assert {key: tomllib.loads(content)[key] for key in OBJECTIVE_FIELDS} == LEGACY
    assert "# My settings" in content
    assert "# retain objective_axes" in content
    assert tomllib.loads(content)["private"] == {"label": "retained"}
    before = path.stat().st_mtime_ns
    assert config.migrate_config_toml(path) is False
    config.load_config()
    assert path.stat().st_mtime_ns == before
    run = config.RunConfig.create([], "model", ["bench"], cfg)
    assert snapshot_objectives(config_for_run(
        config.Config(), config.RunConfig.load(run.run_id),
    )) == LEGACY


@pytest.mark.parametrize("custom", [
    {"objective_weights": [40, 30, 20, 10]},
])
def test_custom_objective_is_not_replaced(tmp_path, monkeypatch, custom):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    settings = LEGACY | custom
    path = tmp_path / "config.toml"
    write_settings(path, settings)
    cfg = config.load_config()
    stored = tomllib.loads(path.read_text())
    assert {key: stored[key] for key in OBJECTIVE_FIELDS} == settings
    assert (cfg.objective_rank, cfg.objective_weights) == (
        settings["objective_rank"], settings["objective_weights"],
    )


@pytest.mark.parametrize("rank", ["weighted_sum", "product", "lexicographic"])
def test_retired_objective_rank_fails_at_load_instead_of_falling_back(
    tmp_path, monkeypatch, rank,
):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    write_settings(tmp_path / "config.toml", LEGACY | {"objective_rank": rank})
    with pytest.raises(ValueError, match="objective_rank must be 'geometric_mean'"):
        config.load_config()
    with pytest.raises(ValueError, match="objective_rank must be 'geometric_mean'"):
        config.load_config(read_only=True)
    with pytest.raises(ValueError, match="objective_rank must be 'geometric_mean'"):
        Config(objective_rank=rank)


def test_pre_change_template_settings_with_objective_primary_load_and_save(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    current = {"objective_axes": ["ppv_at_pi0", "task_recall", "behavior_purity"],
               "objective_rank": "geometric_mean", "objective_weights": [20, 50, 30]}
    # config.toml files generated from the earlier template still name objective_primary.
    write_settings(tmp_path / "config.toml", current | {"objective_primary": "ppv_at_pi0"})
    assert snapshot_objectives(config.load_config(read_only=True)) == current
    loaded = config.load_config()
    assert snapshot_objectives(loaded) == current
    assert preset_name(loaded) == "Recall"
    settings.save_analysis_settings(loaded)
    assert snapshot_objectives(config.load_config()) == current


@pytest.mark.parametrize("weights", [[], [20, 50, 30]])
def test_explicit_attempt_recall_settings_and_snapshots_keep_their_axes(
    tmp_path, monkeypatch, weights,
):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    saved = {"objective_axes": list(LEGACY_STANDARD_AXES),
             "objective_rank": "geometric_mean", "objective_weights": weights}
    path = tmp_path / "config.toml"
    write_settings(path, saved)
    loaded = config.load_config()
    assert snapshot_objectives(loaded) == saved
    assert {key: tomllib.loads(path.read_text())[key] for key in OBJECTIVE_FIELDS} == saved
    run = config.RunConfig.create([], "model", ["bench"], loaded)
    restored = config_for_run(config.Config(), config.RunConfig.load(run.run_id))
    assert snapshot_objectives(restored) == saved


def test_failed_atomic_migration_preserves_original_settings(tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    write_settings(path, LEGACY)
    before = path.read_bytes()

    def fail_replace(self, target):
        assert self.parent == path.parent
        assert target == path
        raise OSError("replacement failed")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="replacement failed"):
        config.migrate_config_toml(path)
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_resume_keeps_saved_objective_snapshot_after_settings_migration(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    path = tmp_path / "config.toml"
    write_settings(path, LEGACY)
    old = config.Config(**LEGACY, analysis_outputs=["agent"])
    run = config.RunConfig.create([], "model", ["bench"], old)
    before = (run.run_dir() / "config.json").read_bytes()
    monkeypatch.setattr(cli, "console", Console(file=StringIO()))
    monkeypatch.setattr("autofa.util.memory.apply_memory_tuning",
                        lambda *a, **kw: SimpleNamespace(summary=lambda: "memory tuning"))
    seen = []
    monkeypatch.setattr("autofa.session.runner.run_session",
                        lambda selected, *, cfg, **kw: seen.append(config_for_run(cfg, selected)))
    monkeypatch.setattr("autofa.session.research_status.read_research_status", lambda _: {})
    cli._resume(SimpleNamespace(run_id=run.run_id))
    assert snapshot_objectives(config.load_config()) == LEGACY
    assert snapshot_objectives(seen[0]) == LEGACY
    assert (run.run_dir() / "config.json").read_bytes() == before


def test_cli_optimization_override_applies_to_the_new_run_only(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    write_settings(tmp_path / "config.toml", LEGACY)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False)
    cfg = cli._discovery_config(SimpleNamespace(optimization="recall"))
    run = config.RunConfig.create([], "model", ["bench"], cfg)
    assert preset_name(config_for_run(Config(), config.RunConfig.load(run.run_id))) == "Recall"
    assert snapshot_objectives(config.load_config()) == LEGACY


@pytest.mark.parametrize("retired", [
    {}, {"objective_axes": list(LEGACY_STANDARD_AXES)},
], ids=["specificity-axes", "attempt-recall-axes"])
def test_fresh_discovery_configuration_retires_legacy_without_rewriting_settings(
    tmp_path, monkeypatch, retired,
):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    saved = LEGACY | retired
    write_settings(tmp_path / "config.toml", saved)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False)
    cfg = cli._discovery_config(SimpleNamespace())
    assert preset_name(cfg) == "Recall" and cfg.objective_axes == list(OBJECTIVE_LABELS)
    assert snapshot_objectives(config.load_config()) == saved


@pytest.mark.parametrize("argv", [
    ["new", "--rounds", "0"], ["new", "--rounds", "1000000000"],
    ["publish", "--rounds", "2"], ["new", "--rounds", "2", "--keep-going"],
], ids=["zero", "too-many", "ignored-by-publish", "conflicts-keep-going"])
def test_cli_rejects_invalid_or_ignored_rounds_before_setup(monkeypatch, argv):
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    monkeypatch.setattr(cli, "ensure_default_config",
                        lambda: pytest.fail("Must reject before setup"))
    with pytest.raises(SystemExit) as error:
        cli.main(argv)
    assert error.value.code == 2


@pytest.mark.parametrize("command,objective", [("new", {}), ("continue", LEGACY)],
                         ids=["new-default", "continue-legacy"])
def test_explicit_rounds_skip_depth_picker_and_preserve_pilot_limits(
    tmp_path, monkeypatch, command, objective,
):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    cfg = replace(config.Config(), min_failure_modes=0, analysis_outputs=["agent"])
    cfg = replace(cfg, **objective)
    monkeypatch.setattr(cli, "load_config", lambda: cfg)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr("autofa.util.memory.apply_memory_tuning", lambda *a, **kw: None)
    monkeypatch.setattr("autofa.wizard.depth.select_discovery_depth",
                        lambda **kw: pytest.fail("Explicit --rounds must bypass the picker"))
    from autofa.data.preparation import PreparedData

    selection = PreparedData(parquet_paths=[], model_id="model", benchmarks=["bench"],
                             source_paths=[], source_kind="parquet",
                             source_harness="unknown", reference_data=[])
    monkeypatch.setattr(cli, "run_dataset_wizard", lambda **kw: selection)
    (tmp_path / "seed.json").write_text(json.dumps({"version": 1, "functions": []}))
    monkeypatch.setattr(cli, "_select_continue_seed",
                        lambda *a: cli.ContinueSeed(str(tmp_path / "seed.json")))
    monkeypatch.setattr(cli, "load_runtime_from_run",
                        lambda *a: SimpleNamespace(persist_library=lambda: None))
    monkeypatch.setattr("autofa.run_spec.preflight_autofa_config", lambda *a, **kw: None)
    launched = []
    monkeypatch.setattr("autofa.session.runner.run_session",
                        lambda run, **kw: launched.append(run))
    args = SimpleNamespace(rounds=2, folder=None, keep_going=False, eval_limit=None,
                           no_auto=False, api_run_id="remote")
    getattr(cli, f"_{command}")(args)
    assert len(launched) == 1
    run = config.RunConfig.load(launched[0].run_id)
    assert run.max_rounds == 2
    assert run.min_failure_modes == 0
    assert preset_name(config_for_run(cfg, run)) == "Recall"
