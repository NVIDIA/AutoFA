# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exploration seeds rotate per round and survive Resume, Continue and portable exports."""

import asyncio
import json
import threading
from copy import deepcopy
from functools import partial

import pytest

from autofa.config import Config, RunConfig
from autofa.core.library import FeatureFunction, FunctionLibrary
from autofa.data.parquet_store import Dataset
from autofa.mcp import tools_library
from autofa.mcp.runtime import Runtime, load_runtime_from_run
from autofa.session.exploration import (
    kickoff_ssot_lines,
    novelty_warning,
    resolve_exploration_seed,
    round_exploration_seed,
    ssot_prompt_block,
    ssot_seed_string,
)
from autofa.session.family_discovery import STATE_FILE, initialize_family_discovery
from autofa.session.prompts import (
    build_system_prompt,
    format_continue_kickoff,
    format_discover_kickoff,
)
from autofa.util.cancellation import Cancellation, cancellation_scope


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path / "autofa"))


def test_exploration_seeds_are_stable_rotate_per_round_and_follow_fixed_main_seed():
    def sequence(run_id, base=None):
        return [resolve_exploration_seed(run_id=run_id, round_idx=i, override=base)
                for i in range(12)]

    derived = sequence("run-a")
    assert derived == sequence("run-a")
    assert derived[0] != derived[1]
    assert derived[0] != sequence("run-b")[0]
    fixed = sequence("run-a", 42)
    assert fixed[0] == 42
    assert len(set(fixed)) == 12
    assert fixed == sequence("run-b", 42)
    assert fixed != sequence("run-a", 43)
    assert ssot_seed_string(42) == ssot_seed_string(42) != ssot_seed_string(43)


@pytest.mark.parametrize("builder", [
    build_system_prompt,
    partial(format_discover_kickoff, top_x=5, max_rounds=3),
    partial(format_continue_kickoff, top_x=5, max_rounds=3),
], ids=["system", "discover", "continue"])
def test_ssot_prompts_share_supplied_string_only_when_enabled(builder):
    token = ssot_seed_string(42)
    assert token in builder(exploration_ssot=True, exploration_seed=42)
    assert token not in builder(exploration_ssot=False, exploration_seed=42)


def test_ssot_templates_are_opt_in_and_scoped_by_domain():
    """SWE-agent templates must not leak onto non-agentic corpora."""
    assert "never_ran_tests" not in ssot_prompt_block(exploration_seed=9)
    assert "never_ran_tests" in ssot_prompt_block(
        exploration_seed=0, exploration_templates=True, domain="agentic_coding")
    assert "never_ran_tests" not in ssot_prompt_block(exploration_seed=0,
                                                      exploration_templates=True)


@pytest.mark.parametrize("fixed,templates,strictness", [
    (False, False, "causal"), (True, True, "behavioral"),
])
def test_advance_round_reuses_seeded_kickoff(monkeypatch, fixed, templates, strictness):
    config = Config(exploration_templates=templates, behavior_strictness=strictness)
    if not fixed:
        config.exploration_seed = None
    run = RunConfig.create([], "m", [], config)
    run.corpus_domain = "generic"
    rt = Runtime(run=run, config=config, library=FunctionLibrary(),
                 dataset=Dataset(parquet_paths=[], model_id="m"))
    monkeypatch.setattr(tools_library, "get_runtime", lambda: rt)
    monkeypatch.setattr("autofa.mcp.runtime.get_runtime", lambda: rt)
    monkeypatch.setattr(tools_library, "_detector_exploration", lambda: {"message": "Explore"})
    monkeypatch.setattr(tools_library, "emit_runtime_status", lambda *a, **kw: None)
    seeds = [run.exploration_seed]
    for round_idx in (1, 2):
        result = tools_library.advance_round()
        expected_seed = resolve_exploration_seed(
            run_id=run.run_id, round_idx=round_idx, override=42 if fixed else None,
        )
        assert result["ok"]
        assert result["exploration_seed"] == expected_seed
        expected_kickoff = kickoff_ssot_lines(
            exploration_seed=expected_seed, exploration_templates=templates, domain="generic",
            objective_config=config,
        )
        assert expected_kickoff in result["message"]
        assert ssot_seed_string(expected_seed) in result["message"]
        rt.run = RunConfig.load(run.run_id)
        assert rt.run.round == round_idx
        assert rt.run.exploration_seed == expected_seed
        assert rt.run.exploration_seed_fixed is fixed
        seeds.append(rt.run.exploration_seed)
    assert len(set(seeds)) == 3


@pytest.mark.parametrize("cancel_before_transition", [False, True])
def test_advance_round_cancellation_keeps_resume_and_continue_cursors_consistent(
    monkeypatch, cancel_before_transition,
):
    config = Config()
    run = RunConfig.create([], "m", [], config)
    rt = load_runtime_from_run(run, config)
    library_path = rt.persist_library()
    initialize_family_discovery(rt)
    config_path = run.run_dir() / "config.json"
    family_path = run.run_dir() / STATE_FILE
    before = (config_path.read_bytes(), library_path.read_bytes(), family_path.read_bytes(),
              deepcopy(rt.library))
    monkeypatch.setattr(tools_library, "_detector_exploration", lambda: {"message": "Explore"})
    monkeypatch.setattr(tools_library, "emit_runtime_status", lambda *a, **kw: None)
    monkeypatch.setattr(
        "autofa.session.discovery_target.failure_mode_target_for_runtime", lambda rt: {},
    )
    token = Cancellation()
    if cancel_before_transition:
        token.cancel()
        with cancellation_scope(token), pytest.raises(asyncio.CancelledError):
            tools_library.advance_round()
        assert run.round == run.exploration_sequence_index == 0
        assert (config_path.read_bytes(), library_path.read_bytes(), family_path.read_bytes(),
                rt.library) == before
        return

    config_saved = threading.Event()
    cancellation_requested = threading.Event()
    save = run.save

    def save_then_allow_cancellation():
        path = save()
        config_saved.set()
        assert cancellation_requested.wait(timeout=5)
        return path

    def cancel_after_config_saved():
        if config_saved.wait(timeout=5):
            # Make the race deterministic: without the transition guard this
            # cancellation finishes before the library save can begin.
            if token.lock.acquire(blocking=False):
                try:
                    token.cancel()
                finally:
                    token.lock.release()
                cancellation_requested.set()
            else:
                cancellation_requested.set()
                token.cancel()

    monkeypatch.setattr(run, "save", save_then_allow_cancellation)
    cancelling_thread = threading.Thread(target=cancel_after_config_saved, daemon=True)
    cancelling_thread.start()
    try:
        with cancellation_scope(token):
            assert tools_library.advance_round()["ok"]
    finally:
        cancelling_thread.join(timeout=5)
    assert not cancelling_thread.is_alive() and token.event.is_set()
    resumed = RunConfig.load(run.run_id)
    assert resumed.round == resumed.exploration_sequence_index == 1
    assert FunctionLibrary.load(library_path).exploration_state["sequence_index"] == 1
    family = json.loads(family_path.read_text())
    assert family["round"] == resumed.round
    assert [previous["round"] for previous in family["history"]] == [0]
    continued = RunConfig.create([], "m", [], config, seed_library_path=str(library_path))
    assert continued.round == 0 and continued.exploration_sequence_index == 2
    kickoff = format_continue_kickoff(
        top_x=3, max_rounds=3, exploration_ssot=True,
        exploration_seed=continued.exploration_seed, objective_config=config,
    )
    assert ssot_seed_string(continued.exploration_seed) in kickoff
    assert ssot_seed_string(resumed.exploration_seed) not in kickoff


def test_novelty_warning_names_a_near_duplicate_but_not_a_distinct_mechanism():
    existing = [{"name": "retry_thrash",
                 "description": "Retries the same failing command until the budget dies",
                 "mechanism": "The model hits a non-zero exit then re-runs nearly the same "
                              "command instead of diagnosing the root cause."}]
    duplicate = novelty_warning(description=existing[0]["description"],
                                mechanism=existing[0]["mechanism"], existing=existing)
    assert "retry_thrash" in duplicate
    assert novelty_warning(
        description="Writes its own test, sees it pass, ships on it",
        mechanism="The model writes a small script with its own asserts, sees them go green, "
                  "and declares fixed while the hidden grader still fails.",
        existing=existing,
    ) is None


# Discovery sequence checkpoints survive Resume, Continue, and portable exports.


@pytest.mark.parametrize("main_seed", [None, 0])
def test_portable_continue_advances_once_and_resume_preserves_cursor(tmp_path, main_seed):
    config = Config(exploration_seed=main_seed)
    source = RunConfig.create([], "m", [], config)
    assert source.exploration_sequence_index == source.round == 0
    base = source.exploration_base_seed
    source.round = source.exploration_sequence_index = 3
    source.exploration_seed = round_exploration_seed(base, 3)
    source.save()
    runtime = load_runtime_from_run(source, config)
    export = tmp_path / "portable.json"
    result = tools_library.library_save(to_catalog=True, path=str(export))
    expected = {"version": 1, "base_seed": base, "sequence_index": 3}
    assert result["ok"]
    for saved in result["saved"]:
        assert FunctionLibrary.load(saved).exploration_state == expected

    # The portable checkpoint remains authoritative if its original run later advances.
    source.round = source.exploration_sequence_index = 5
    source.exploration_seed = round_exploration_seed(base, 5)
    source.save()
    continued = RunConfig.create(
        [], "m", [], Config(exploration_seed=999), seed_library_path=str(export),
        seed_source_run_id=source.run_id,
    )
    assert continued.exploration_base_seed == base
    assert continued.exploration_sequence_index == 4
    assert continued.exploration_seed == round_exploration_seed(base, 4)
    assert continued.round == 0  # Continuation still gets its full discovery-round budget.
    resumed = RunConfig.load(continued.run_id)
    resumed_runtime = load_runtime_from_run(resumed, config)
    assert resumed_runtime.run.exploration_sequence_index == 4
    assert resumed_runtime.run.exploration_seed == continued.exploration_seed
    next_seed = resumed_runtime.persist_library()
    twice = RunConfig.create([], "m", [], config, seed_library_path=str(next_seed))
    assert twice.exploration_base_seed == base
    assert twice.exploration_sequence_index == 5
    assert twice.exploration_seed == round_exploration_seed(base, 5)
    assert runtime.run is source


def test_unset_exploration_cursor_loads_the_automatic_main_seed_without_rewriting():
    run = RunConfig(run_id="unset-cursor", parquet_paths=[], model_id="m")
    config_path = run.save()
    saved = json.loads(config_path.read_text())
    saved.pop("exploration_base_seed")
    saved.pop("exploration_sequence_index")
    config_path.write_text(json.dumps(saved))
    before = config_path.read_bytes()

    loaded = RunConfig.load("unset-cursor")
    base = resolve_exploration_seed(run_id="unset-cursor")
    assert loaded.exploration_base_seed == base
    assert loaded.exploration_sequence_index == 0
    assert loaded.exploration_seed == round_exploration_seed(base, 0)
    assert config_path.read_bytes() == before


@pytest.mark.parametrize("saved_round, next_index", [(None, 1), (4, 5)])
def test_legacy_export_without_origin_advances_past_available_round(
    tmp_path, saved_round, next_index,
):
    fn = FeatureFunction("old", "Existing detector", "def old(item): return False",
                         meta={} if saved_round is None else {"round": saved_round})
    seed = tmp_path / "legacy-export.json"
    FunctionLibrary(functions=[fn]).save(seed)
    continued = RunConfig.create([], "m", [], Config(), seed_library_path=str(seed))
    assert continued.exploration_base_seed == 42
    assert continued.exploration_sequence_index == next_index
    assert continued.exploration_seed == round_exploration_seed(42, next_index)


@pytest.mark.parametrize("state", [
    {}, {"version": 2, "base_seed": 42, "sequence_index": 0},
    {"version": 1, "base_seed": 42, "sequence_index": -1},
], ids=["missing", "unknown-version", "negative-index"])
def test_invalid_portable_cursor_does_not_silently_restart(state):
    with pytest.raises(ValueError, match="exploration state"):
        FunctionLibrary.from_dict({"functions": [], "exploration_state": state})


def test_detector_import_preserves_destination_sequence(tmp_path):
    run = RunConfig.create([], "m", [], Config())
    runtime = load_runtime_from_run(run, Config())
    runtime.persist_library()
    incoming = tmp_path / "import.json"
    FunctionLibrary(exploration_state={"version": 1, "base_seed": 99, "sequence_index": 7}).save(
        incoming,
    )
    assert tools_library.library_load(str(incoming))["ok"]
    assert runtime.library.exploration_state == {
        "version": 1, "base_seed": 42, "sequence_index": 0,
    }
