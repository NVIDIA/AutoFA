# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Report connections use frozen recorded provenance and never alter grades or invent families."""

import json
import re
from copy import deepcopy

from autofa.core.dependencies import review_source
from autofa.core.library import FeatureFunction, FunctionLibrary
from autofa.metrics.claim_contract import claim_hash
from autofa.reports.failure_cards import (
    _extract_autofa_data,
    build_data,
    refresh_report_data,
    render_html,
)
from autofa.reports.recorded_relationships import read_state, recorded_relationships
from autofa.reports.related_failures import MAX_RELATIONSHIPS, project_related_failures

from support.task_support import _function as reviewed_function
from support.task_support import run_node


def report_function(name, *, title=None, recall=.1, precision=.9, adherence=1):
    return {"name": name, "title": title or name, "card_id": f"card-{name}",
            "metrics": {"recall": recall, "precision": precision,
                        "ppv_at_pi0": precision, "behavior_purity": adherence,
                        "n_matched_failed_tasks": 10}, "buckets": []}


def edge(source, target, kind="synthesis", rationale="Recorded connection"):
    return {"source": source, "target": target, "relation": kind, "rationale": rationale}


def detector(name):
    return FeatureFunction(name=name, description=f"Failure {name}", mechanism=f"Mechanism {name}",
                           source=f"def {name}(item): return True")


def fingerprint(fn):
    return claim_hash(source=review_source(fn), description=fn.description, mechanism=fn.mechanism)


def record(fn, *, seeds=()):
    return {
        "survey": {"family_name": f"family_{fn.name}", "detector_name": fn.name,
                   "description": fn.description, "mechanism": fn.mechanism,
                   "approach": "bottom_up" if seeds else "top_down",
                   "seed_functions": [{"name": seed.name, "claim_hash": fingerprint(seed)}
                                      for seed in seeds],
                   "evidence": [{"item_id": fn.name, "role": "support", "quotes": [
                       {"field": "response_text", "quote": f"Observed {fn.name}"}]}]},
        "assessment": {"outcome": "assessed", "detector_name": fn.name,
                       "claim_hash": fingerprint(fn)},
    }


def _render(functions, edges, *, projection_override=None):
    """Execute the real embedded report renderer in Node, without a browser."""
    html = render_html({"functions": functions, "meta": {"recorded_relationships": edges}})
    data = _extract_autofa_data(html)
    if projection_override is not None:
        data["meta"]["related_failures"] = projection_override
    helpers = html[html.index("const VERDICT_PRESENTATIONS"):html.index("const RAW =")]
    renderer = html[html.index("function escapeHtml("):html.index("function boot(")]
    script = ("const STATE = {selection: null}; const document = {addEventListener() {}};\n"
              + helpers + renderer + "\nconst RAW = " + json.dumps(data) + ";\n"
              + """
const before = JSON.stringify(RAW);
const functions = visibleReportFunctions(RAW.functions);
const connections = reportConnections(functions, RAW.meta.related_failures);
const cards = functions.map((fn, i) => renderCard(fn, i + 1, functions.length, connections));
if (JSON.stringify(RAW) !== before) throw Error('Report renderer mutated saved data');
console.log(JSON.stringify({
  html: connections.html, labels: Object.fromEntries(connections.labels),
  notes: Object.fromEntries(connections.notes), cards,
  order: functions.map(fn => fn.name), count: RAW.meta.n_functions,
}));
""")
    return json.loads(run_node(script))


# Projection: explicit recorded edges only, never rescoring or inferring hierarchy.


def test_groups_only_explicit_visible_connections_without_reordering_or_mutating():
    functions = [report_function("broad"), report_function("unrelated"),
                 report_function("child"), report_function("ok", precision=.5),
                 report_function("weak", precision=.1),
                 report_function("unreviewed", adherence=None)]
    relationships = [edge("child", "broad"), edge("ok", "child", "explains"),
                     edge("weak", "broad"), edge("unreviewed", "broad")]
    before = deepcopy((functions, relationships))
    result = project_related_failures(functions, relationships)
    assert len(result["groups"]) == 1
    assert result["groups"][0]["names"] == ["broad", "child", "ok"]
    assert len(result["relationships"]) == 2
    assert (functions, relationships) == before
    assert set(result) == {"version", "relationships", "groups"}


def test_names_dependencies_and_overlap_do_not_infer_a_relationship():
    functions = [report_function("missing_answer"), report_function("missing_final_answer")]
    functions[1].update(depends_on=["missing_answer"], explains="missing_answer")
    for row in functions:
        row["example_fail_ids"] = ["same-item"]
    assert project_related_failures(functions, [])["groups"] == []
    first, second = detector("a"), detector("a_subtype")
    second.depends_on = ["a"]
    for fn in (first, second):
        fn.example_fail_ids = ["shared"]
    assert recorded_relationships(FunctionLibrary(functions=[first, second])) == []


def test_directed_relations_keep_their_recorded_direction():
    # Recorded edges are never canonicalized; b -> a must not become a -> b.
    result = project_related_failures([report_function("a"), report_function("b")],
                                      [edge("b", "a", "synthesis")])
    assert result["relationships"][0]["relation"] == "synthesis"
    assert result["relationships"][0]["source"] == "b"


def test_group_ids_stay_stable_when_recall_changes_or_edges_are_duplicated():
    functions = [report_function("a"), report_function("b")]
    relationships = [edge("a", "b"), edge("a", "b", rationale="Repeated record")]
    first = project_related_failures(functions, relationships)
    functions.reverse()
    functions[0]["metrics"]["recall"] = .5
    second = project_related_failures(functions, relationships[::-1] + relationships)
    assert first["groups"][0]["id"] == second["groups"][0]["id"]
    assert len(second["relationships"]) == 1
    assert second["groups"][0]["names"] == ["b", "a"]


def test_invalid_edges_are_ignored_without_touching_the_library():
    functions = [report_function("a"), report_function("b")]
    result = project_related_failures(functions, [None, {}, edge("a", "a"),
                                                 edge("a", "b", "distinct"),
                                                 edge([], "b"), edge("a", "gone")])
    assert result["relationships"] == result["groups"] == []


def test_edge_cap_is_deterministic():
    functions = [report_function(str(i)) for i in range(MAX_RELATIONSHIPS + 2)]
    edges = [edge(str(i), str(i + 1)) for i in range(MAX_RELATIONSHIPS + 1)]
    first = project_related_failures(functions, edges)
    second = project_related_failures(functions, list(reversed(edges)))
    assert first == second
    assert len(first["relationships"]) == MAX_RELATIONSHIPS


# Recorded provenance: only current assessed synthesis records and explains links become links.


def test_report_writer_reads_sibling_links_without_changing_library_or_discovery(
    tmp_path, monkeypatch,
):
    from autofa.config import Config
    from autofa.reports import failure_cards

    parent, first, second = (detector(name) for name in ("parent", "first", "second"))
    library = FunctionLibrary(functions=[parent, first, second])
    library_path = tmp_path / "library.json"
    state_path = tmp_path / "family_discovery.json"
    library_path.write_text(json.dumps(library.to_dict()))
    state_path.write_text(json.dumps({"comparisons": [record(parent, seeds=[first, second])]}))
    before = {path: path.read_bytes() for path in (library_path, state_path)}
    monkeypatch.setattr(failure_cards, "try_dataset_from_dir", lambda _: None)
    monkeypatch.setattr(failure_cards, "try_dataset_for_library", lambda _: None)
    monkeypatch.setattr("autofa.config.load_config", lambda: Config())
    output = failure_cards.write_failure_cards(library_path, out=tmp_path / "report.html")
    data = failure_cards._extract_autofa_data(output.read_text())
    assert [(row["source"], row["target"], row["relation"])
            for row in data["meta"]["recorded_relationships"]] == [
        ("parent", "first", "synthesis"), ("parent", "second", "synthesis"),
    ]
    # These fixtures are unscored: their saved links cannot expose them as findings.
    assert data["meta"]["related_failures"]["groups"] == []
    assert all(path.read_bytes() == content for path, content in before.items())


def test_explains_is_an_explanatory_link_and_does_not_mutate_library():
    first, second = detector("parent"), detector("detail")
    second.explains = first.name
    lib = FunctionLibrary(functions=[first, second])
    before = deepcopy(lib.to_dict())
    edges = recorded_relationships(lib)
    assert [(row["source"], row["target"], row["relation"]) for row in edges] == [
        ("detail", "parent", "explains")]
    assert edges[0]["source_claim_hash"] == fingerprint(second)
    assert edges[0]["target_claim_hash"] == fingerprint(first)
    assert lib.to_dict() == before
    second.explains = "missing"
    assert recorded_relationships(lib) == []
    second.explains = second.name
    assert recorded_relationships(lib) == []


def test_assessed_synthesis_preserves_only_current_seed_links_across_history():
    parent, first, second = detector("parent"), detector("first"), detector("second")
    lib = FunctionLibrary(functions=[parent, first, second])
    row = record(parent, seeds=(first, second))
    state = {"history": [{"comparisons": [row]}]}
    before = deepcopy(state)
    assert [(r["source"], r["target"], r["relation"]) for r in
            recorded_relationships(lib, state)] == [
        ("parent", "first", "synthesis"), ("parent", "second", "synthesis")]
    assert state == before
    unfinished = deepcopy(row)
    unfinished["assessment"]["outcome"] = "deferred"
    assert recorded_relationships(lib, unfinished) == []
    second.source += "\n# changed revision"
    assert [r["target"] for r in recorded_relationships(lib, state)] == ["first"]
    parent.source += "\n# changed parent"
    assert recorded_relationships(lib, state) == []


def test_exported_edges_survive_without_sidecar_but_not_revisions_or_removed_links():
    parent, child = detector("parent"), detector("child")
    lib = FunctionLibrary(functions=[parent, child])
    saved = recorded_relationships(lib, {"comparisons": [record(parent, seeds=[child])]})
    assert len(saved) == 1
    assert recorded_relationships(lib, previous=saved) == saved
    child.source += "\n# changed"
    assert recorded_relationships(lib, previous=saved) == []
    child.explains = parent.name
    saved = recorded_relationships(lib)
    child.explains = None
    assert recorded_relationships(lib, previous=saved) == []
    saved[0]["relation"] = "synthesis"  # An execution link cannot claim synthesis provenance.
    assert recorded_relationships(lib, previous=saved) == []


def test_reader_ignores_bad_state_and_reads_only_sibling_ledger(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "family_discovery.json").write_text("not json")
    state = {"comparisons": []}
    (second / "family_discovery.json").write_text(json.dumps(state))
    assert read_state([None, first, second / "library.json"]) == state
    assert read_state([first]) is None


def test_malformed_and_cyclic_records_do_not_crash_or_mutate():
    parent = detector("parent")
    state = {"survey": None, "history": [None, "bad"]}
    state["history"].append(state)
    assert recorded_relationships(FunctionLibrary(functions=[parent]), state) == []


# Rendering: the embedded report renderer shows connections without reordering cards.


def test_connected_failures_get_group_labels_without_reordering_cards():
    functions = [report_function("parent", title="Parent", recall=.04),
                 report_function("independent", recall=.2),
                 report_function("child", title="Child", recall=.3)]
    edges = [edge("parent", "child")]
    before = deepcopy((functions, edges))
    rendered = _render(functions, edges)
    assert rendered["labels"] == {"child": "A.1", "parent": "A.2"}
    assert re.search(r"A\.1</span>.*?>Child</a>.*?A\.2</span>.*?>Parent</a>",
                     rendered["html"], re.DOTALL)
    assert rendered["order"] == ["child", "independent", "parent"]
    assert rendered["count"] == 3
    assert rendered["order"] == _render(functions, [])["order"]
    assert (functions, edges) == before


def test_hidden_and_retired_nodes_never_become_visible_through_relationships():
    functions = [report_function("a"), report_function("b"),
                 report_function("hidden", precision=.1)]
    edges = [edge("a", "b"), edge("hidden", "a"), edge("retired", "a")]
    # A host can retain old group metadata while replacing the detector list.
    projection = {"groups": [{"id": "old-group", "names": ["a", "b", "hidden", "retired"],
                              "relationships": edges}]}
    rendered = _render(functions, edges, projection_override=projection)
    assert rendered["order"] == ["a", "b"]
    assert set(rendered["labels"]) == {"a", "b"}
    assert rendered["count"] == 2
    output = rendered["html"] + "".join(rendered["notes"].values())
    assert "hidden" not in output and "retired" not in output


def test_titles_rationales_and_names_are_escaped_in_connection_html():
    title = '<img src=x onerror="alert(1)">'
    rationale = '<script>alert("x")</script>'
    name = 'child" onclick="alert(2)'
    rendered = _render([report_function("parent", title=title), report_function(name)],
                       [edge(name, "parent", rationale=rationale)])
    output = rendered["html"] + "".join(rendered["notes"].values())
    assert '<img' not in output and '<script>' not in output
    assert '&lt;img' in output and '&lt;script&gt;' in output
    assert 'data-failure-jump="child&quot; onclick=&quot;alert(2)"' in output
    assert '#failure=child%22%20onclick%3D%22alert(2)' in output


def test_completed_confirmations_cannot_hide_processing_error_during_report_compaction():
    fn = reviewed_function("good")
    fn.scores.update(behavior_n_judged=6, behavior_n_completed=5, behavior_n_errors=1)
    fn.scores["behavior_verdicts"].append({"item_id": "pending", "verdict": "error"})
    library = FunctionLibrary(functions=[fn])
    data = build_data(library)
    assert data["functions"][0]["verdict"]["code"] == "UNSCORED"
    assert data["functions"][0]["metrics"]["behavior_processing_pending"] is True
    assert data["functions"][0]["metrics"]["behavior_purity"] == 1
    assert _render(data["functions"], [])["order"] == []
    # Upgrade a saved compact payload that omitted the new processing flag.
    data["functions"][0]["metrics"].pop("behavior_processing_pending")
    refreshed = refresh_report_data(data, library)
    assert refreshed["functions"][0]["metrics"]["behavior_processing_pending"] is True
    assert _render(refreshed["functions"], [])["order"] == []


def test_build_and_refresh_keep_recorded_relationships_portable_and_invalidate_changed_claim():
    parent, child = reviewed_function("good"), reviewed_function("good")
    parent.name, child.name = "parent", "child"
    library = FunctionLibrary(functions=[parent, child])
    state = {"comparisons": [record(parent, seeds=[child])]}
    before = deepcopy(library.to_dict())
    data = build_data(library, relationship_state=state)
    edges = data["meta"]["recorded_relationships"]
    assert len(edges) == 1 and edges[0]["relation"] == "synthesis"
    assert _render(data["functions"], edges)["labels"] == {"child": "A.1", "parent": "A.2"}
    # The embedded provenance is sufficient when viewing outside the run directory.
    refreshed = refresh_report_data(data, library)
    assert refreshed["meta"]["recorded_relationships"] == edges
    assert library.to_dict() == before
    child.source = "def detect(item): return False"
    updated = refresh_report_data(data, library)
    assert updated["meta"]["recorded_relationships"] == []
    flat = _render(updated["functions"], updated["meta"]["recorded_relationships"])
    assert flat["html"] == "" and flat["labels"] == flat["notes"] == {}
    assert all("related-card-note" not in card for card in flat["cards"])
