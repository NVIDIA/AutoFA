# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Assistance retains origins/exposure without transplanting destination grades."""

from copy import deepcopy

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from autofa.config import Config, RunConfig
from autofa.core.library import FeatureFunction, FunctionLibrary
from autofa.data.parquet_store import Dataset
from autofa.mcp import runtime, tools_contributions, tools_library
from autofa.session.contributions import definition_identity, record_revision, unresolved_audits
from autofa.session.review_evidence import review_context


@pytest.fixture
def rt(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path / "home"))
    path = tmp_path / "data.parquet"
    pq.write_table(pa.Table.from_pylist([
        {"model_id": "m", "benchmark": "b", "problem_id": "task-a", "run_idx": 0,
         "is_correct": False, "response_text": "an unsupported step"},
        {"model_id": "m", "benchmark": "b", "problem_id": "task-b", "run_idx": 0,
         "is_correct": True, "response_text": "the supported step"},
    ]), path)
    instance = runtime.Runtime(
        dataset=Dataset([path]), library=FunctionLibrary(model_id="m", benchmark="b"),
        run=RunConfig(run_id="destination", model_id="m", parquet_paths=[str(path)]),
        config=Config(),
    )
    monkeypatch.setattr(runtime, "_RUNTIME", instance)
    instance.persist_library()
    return instance


def detector(name="candidate"):
    return FeatureFunction(
        name=name, description="Unsupported step", mechanism="A step contradicts its premise",
        source=f"def {name}(item):\n    return 'unsupported' in item.get('response_text', '')",
        kind="python", scores={"behavior_verdicts": [{"item_id": "old", "verdict": "yes"}],
                               "recall": .9, "evaluated": True},
    )


def test_preview_then_unscored_import_retains_source_and_exposure(rt, tmp_path):
    source = FunctionLibrary(model_id="source", functions=[detector()])
    path = tmp_path / "seed.json"
    source.save(path)
    before = path.read_bytes()
    preview = tools_contributions.library_import_candidates(str(path))
    assert preview["ok"] and not preview["applied"]
    assert not rt.library.functions
    result = tools_contributions.library_import_candidates(str(path), apply=True)
    assert result["ok"] and result["committed"] == ["candidate"]
    candidate = rt.library.get("candidate")
    assert candidate.scores == {}
    assert candidate.failure_localisations == {}
    assert candidate.meta["contribution_provenance"]["origin"]["actor"] == "unknown"
    assert candidate.meta["contribution_provenance"]["revisions"][-1]["action"] == "import"
    assert "old" in review_context(rt.run.run_dir(), "candidate")["reviewed_item_ids"]
    assert path.read_bytes() == before
    assert tools_contributions.library_import_candidates(str(path), apply=True)["skipped"] == [
        "candidate"]
    assert FunctionLibrary.load(rt.working_library_path()).contributions == rt.library.contributions


def test_same_name_and_renamed_claim_cannot_overwrite_or_bypass_repair(rt, tmp_path):
    existing = detector()
    rt.library.upsert(existing)
    rt.persist_library()
    changed = detector()
    changed.source += "\n# different implementation"
    path = tmp_path / "seed.json"
    FunctionLibrary(functions=[changed]).save(path)
    result = tools_contributions.library_import_candidates(str(path), apply=True)
    assert result["conflicts"] and not result["applied"]
    assert rt.library.get("candidate").scores == existing.scores
    FunctionLibrary(functions=[detector("renamed")]).save(path)
    result = tools_contributions.library_import_candidates(str(path), apply=True)
    assert result["conflicts"][0]["owner"] == "candidate"
    assert rt.library.get("renamed") is None


def test_import_preserves_pinned_helper_and_detector_dependencies(rt, tmp_path):
    from autofa.core.dependencies import build_bundle, execution_bundle
    from autofa.core.helpers import HelperFunction
    from autofa.core.sandbox import run_detector

    parent = detector("parent")
    seed = FunctionLibrary(functions=[parent])
    seed.helpers.upsert(HelperFunction(name="normalize", entrypoint="normalize",
                                      source="def normalize(text): return text.lower()"))
    bundle = build_bundle(seed, depends_on=["parent"], uses_helpers={"lower": "normalize"})
    child = FeatureFunction(
        name="child", kind="python", description="A refined cause", mechanism="A specific cause",
        source="def child(item):\n    return library.fire('parent', item) "
        "and 'assertionerror' in helpers.call('lower', item.get('response_text', ''))",
        depends_on=["parent"], uses_helpers=bundle["helpers"],
        meta={"dependency_bundle": bundle},
    )
    seed.upsert(child)
    path = tmp_path / "seed.json"
    seed.save(path)
    result = tools_contributions.library_import_candidates(str(path), names=["child"], apply=True)
    assert result["ok"] and result["committed"] == ["parent", "child"], result
    imported = rt.library.get("child")
    assert execution_bundle(imported) == bundle
    assert run_detector(imported.source, {"response_text": "unsupported AssertionError"},
                        bundle=execution_bundle(imported)) is True
    assert rt.library.get("parent").scores == imported.scores == {}


def test_failed_import_admission_keeps_existing_library_and_exposure(rt, tmp_path, monkeypatch):
    source = FunctionLibrary(functions=[detector()])
    path = tmp_path / "seed.json"
    source.save(path)
    monkeypatch.setattr("autofa.session.repair_workflow.record_submission",
                        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("repair blocked")))
    result = tools_contributions.library_import_candidates(str(path), apply=True)
    assert not result["ok"] and result["committed"] == []
    assert rt.library.get("candidate") is None
    assert "old" in review_context(rt.run.run_dir(), "candidate")["reviewed_item_ids"]


def test_native_edit_keeps_external_origin(rt):
    original = detector()
    original.scores = {}
    record_revision(original, actor="driver")
    rt.library.upsert(original)
    rt.persist_library()
    result = tools_library.submit_function(
        original.name, original.source + "\n# native refinement", description=original.description,
        mechanism=original.mechanism, kind="python", auto_evaluate=False,
    )
    assert result["ok"], result
    provenance = rt.library.get(original.name).meta["contribution_provenance"]
    assert provenance["origin"]["actor"] == "driver"
    assert provenance["revisions"][-1]["actor"] == "native"
    assert provenance["revisions"][-1]["action"] == "edit"


def test_external_inspection_reserves_verified_task_identity_and_is_portable(rt):
    item_id = rt.dataset.identity_index().rows[0].item_id
    result = tools_contributions.record_external_inspection([item_id], "Inspect original premise")
    assert result["ok"], result
    context = review_context(rt.run.run_dir(), "future")
    assert item_id in context["reviewed_item_ids"]
    assert '["b","task-a"]' in context["reviewed_task_groups"]
    reloaded = FunctionLibrary.load(rt.working_library_path())
    assert reloaded.contributions[0]["kind"] == "inspection"
    repeat = tools_contributions.record_external_inspection([item_id], "Inspect original premise")
    assert repeat["receipt"] == result["receipt"]
    assert len(rt.library.contributions) == 1
    rejected = tools_contributions.record_external_inspection(["ambiguous-short-id"], "inspect")
    assert not rejected["ok"]
    assert len(rt.library.contributions) == 1


def test_source_audit_preserves_native_scores_and_survives_export(rt, monkeypatch):
    from autofa.reports.failure_cards import build_data, refresh_report_data

    fn = detector()
    rt.library.upsert(fn)
    rt.persist_library()
    scores = deepcopy(fn.scores)
    item_id = rt.dataset.identity_index().rows[0].item_id
    audit = tools_contributions.record_source_audit(
        fn.name, item_id, "Original source contradicts the saved reading", "artifact:image:0",
        "a" * 64,
    )
    assert audit["ok"], audit
    reloaded = FunctionLibrary.load(rt.working_library_path())
    assert reloaded.get(fn.name).scores == scores
    assert unresolved_audits(reloaded)[0]["definition_sha256"] == definition_identity(fn)
    report = build_data(reloaded, dataset=rt.dataset)
    assert "contributions" not in report["meta"]
    assert "contributions" not in refresh_report_data(report, reloaded)["meta"]
    resolved = tools_contributions.record_source_audit(
        fn.name, item_id, "Reconciled against the original source", "artifact:image:0", "a" * 64,
        resolves=audit["receipt"]["id"],
    )
    assert resolved["ok"] and not unresolved_audits(rt.library)
    assert rt.library.get(fn.name).scores == scores


def test_hypothetical_adherence_crosses_threshold_without_changing_native_judgments(rt):
    from autofa.session.contributions import qualification_sensitivity

    item_id = rt.dataset.identity_index().rows[0].item_id
    fn = detector()
    fn.scores.update(behavior_n_completed=57, behavior_n_yes=52, behavior_purity=52 / 57,
                     behavior_verdicts=[{"item_id": item_id, "verdict": "yes"}] + [
                         {"item_id": f"saved-{i}", "verdict": "yes" if i < 51 else "no"}
                         for i in range(56)])
    rt.library.upsert(fn)
    rt.persist_library()
    before = deepcopy(fn.scores)
    audit = tools_contributions.record_source_audit(
        fn.name, item_id, "Original source contradicts this affirmative", "artifact:original",
        "a" * 64, proposed_verdict="no",
    )
    assert audit["ok"]
    result = qualification_sensitivity(rt.library)[0]
    assert result["available"]
    assert result["hypothetical_adherence"] == 51 / 57
    assert result["native_adherence"] == 52 / 57
    assert not result["meets_good_adherence_floor"]
    assert rt.library.get(fn.name).scores == before
    # Another review cohort cannot silently resolve the disputed original item.
    rt.library.get(fn.name).scores["behavior_verdicts"][-1]["reason"] = "new cohort"
    stale = qualification_sensitivity(rt.library)[0]
    assert not stale["available"] and "cohort changed" in stale["reason"]
    assert len(unresolved_audits(rt.library)) == 1


def test_nonfinite_saved_verdict_keeps_audit_visible_without_sensitivity(rt):
    from autofa.session.contributions import current_disputed_names, qualification_sensitivity

    item_id = rt.dataset.identity_index().rows[0].item_id
    fn = detector()
    fn.scores.update(behavior_n_completed=1, behavior_n_yes=1, behavior_purity=1,
                     behavior_verdicts=[{"item_id": item_id, "verdict": "yes"}])
    rt.library.upsert(fn)
    audit = tools_contributions.record_source_audit(
        fn.name, item_id, "Source conflicts with the verdict", "artifact:original",
        "a" * 64, proposed_verdict="no",
    )
    assert audit["ok"]
    rt.library.get(fn.name).scores["behavior_verdicts"][0]["confidence"] = float("nan")
    rt.persist_library()
    reloaded = FunctionLibrary.load(rt.working_library_path())
    audits = unresolved_audits(reloaded)
    assert current_disputed_names(reloaded, audits) == [fn.name]
    assert not qualification_sensitivity(reloaded)[0]["available"]
    assert len(audits) == 1


def test_malformed_portable_audit_is_rejected_before_report_or_upload():
    from autofa.data.remote_library_api import RemoteLibraryError, prepare_library_payload
    from autofa_library_server.schemas import validate_library_payload

    payload = {"version": 1, "model_id": "m", "functions": [], "contributions": [
        {"version": 1, "id": "invalid", "kind": "source_audit", "actor": "driver"},
    ]}
    with pytest.raises(ValueError, match="source audit"):
        FunctionLibrary.from_dict(payload)
    with pytest.raises(RemoteLibraryError, match="source audit"):
        prepare_library_payload(payload)
    with pytest.raises(ValueError, match="source audit"):
        validate_library_payload(payload)


@pytest.mark.parametrize('caller', ['native', 'driver'])
def test_contribution_actor_comes_from_invocation_path(rt, caller):
    import asyncio
    import json

    from autofa.mcp.server import invoke_tool

    item_id = rt.dataset.identity_index().rows[0].item_id
    result = asyncio.run(invoke_tool('record_external_inspection', {
        'item_ids': [item_id], 'reason': 'Invocation provenance regression',
    }, caller=caller))
    result = json.loads(result['content'][0]['text'])
    assert result['ok'], result
    assert result['receipt']['actor'] == caller


def test_actor_is_not_advertised_as_a_caller_control():
    from autofa.mcp.server import codex_tool_specs

    names = {'record_external_inspection', 'record_source_audit',
             'library_import_candidates', 'submit_function'}
    found = {spec['name']: spec for spec in codex_tool_specs() if spec['name'] in names}
    assert found.keys() == names
    for spec in found.values():
        assert 'actor' not in spec['inputSchema']['properties']
