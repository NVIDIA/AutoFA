# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The efficiency report: summary rules, pricing, per-card comparisons, layouts and publishing."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import autofa.config
from autofa import cli
from autofa.config import Config, RunConfig
from autofa.core.library import FeatureFunction, FunctionLibrary
from autofa.data import remote_library_publish
from autofa.data.evalrdl_efficiency import EfficiencyReadiness
from autofa.data.parquet_store import apply_column_aliases, make_item_id
from autofa.data.remote_library_api import RemotePublication
from autofa.data.remote_library_publish import publish_run
from autofa.metrics import task_support
from autofa.reports import efficiency_card, failure_cards
from autofa.reports.efficiency_card import (
    _OVERVIEW_JS,
    _card_sections,
    _inject_overview,
    _item_attempts,
    _report_detectors,
    _summary,
    resolve_pricing,
)
from autofa.wizard.dataset import DatasetSelection

from support.task_support import run_node

_REPORT_SHELL = """<style></style>
<div class="doc-eyebrow">Model Failure Cards</div>
<h1 id="doc-h1">Failure modes</h1>
<main class="main-shell"><div id="cards"></div></main>
<script id="autofa-data" type="application/json">{meta}</script>
<script>
boot();
</script>
"""


def _interactive(monkeypatch) -> None:
    monkeypatch.setattr(cli.sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(cli.sys, "stdout", SimpleNamespace(isatty=lambda: True))


def _summary_metric(name, *, median, total=(0, 0), n=(50, 30), statistic=0.5,
                    significant=True, coverage=1.0):
    return {
        "name": name,
        "coverage": coverage,
        "pass": {"n": n[0], "median": median[0], "total": total[0]},
        "fail": {"n": n[1], "median": median[1], "total": total[1]},
        "permutation_test": {"statistic": statistic, "significant": significant},
    }


def test_summary_classifies_metrics_with_fixed_rules() -> None:
    metrics = [
        _summary_metric("simulated_cost_usd", median=(1, 3), total=(40, 60), statistic=0.3),
        _summary_metric("input_tokens", median=(10, 40), statistic=0.6),
        _summary_metric("n_turns", median=(8, 4), statistic=0.4),
        _summary_metric("total_tokens", median=(10, 20), statistic=0.15),
        _summary_metric("prompt_tokens", median=(1000, 1010), statistic=0.3),
        _summary_metric("answer_tokens", median=(5, 9), statistic=0.05),
        _summary_metric("n_tool_calls", median=(3, 6), significant=False),
        _summary_metric("cached_tokens", median=(1, 2), coverage=0.5),
        _summary_metric("tool_arg_chars", median=(1, 2), n=(50, 10)),
        _summary_metric("reasoning_tokens", median=(1, 5), statistic=0.45),
        _summary_metric("output_tokens", median=(2, 6), statistic=0.5),
        _summary_metric("tool_output_chars", median=(1, 5), statistic=0.3),
        _summary_metric("final_context_tokens", median=(5, 1), statistic=0.3),
    ]
    names = [metric["name"] for metric in metrics]

    def section(*keys):
        return {"metrics": list(keys), "charts": {key: {"label": key.title()} for key in keys}}

    sections = {
        "all": section(*names[:9]),
        "tokens": section("reasoning_tokens", "output_tokens"),
        "activity": section("tool_output_chars", "final_context_tokens"),
    }
    summary = _summary({"metrics": metrics}, sections)
    assert [
        (item["label"], item["direction"], item["size"]) for item in summary["differ"]
    ] == [
        ("Input_Tokens", "up", "large"),
        ("Cached_Tokens", "up", "large"),
        ("Token usage", "up", "large"),
        ("N_Turns", "down", "large"),
        ("Tool_Output_Chars", "up", "moderate"),
        ("Final_Context_Tokens", "down", "moderate"),
        ("Total_Tokens", "up", "small"),
    ]
    assert summary["cost"] == {"attempt_share": 30 / 80, "cost_share": 0.6}
    assert summary["negligible"] == ["Prompt_Tokens", "Answer_Tokens"]
    assert summary["not_significant"] == ["N_Tool_Calls"]
    assert summary["insufficient"] == ["Tool_Arg_Chars"]
    assert summary["partial"] == {"Cached_Tokens": 0.5}


def test_pricing_prefers_model_config_and_has_disclosed_fallback() -> None:
    configured = resolve_pricing(SimpleNamespace(efficiency_model_prices={
        "provider/model": {"input": 1.0, "cached": 0.2, "output": 4.0},
    }), "provider/model")
    assert configured["source"] == "model_config"
    assert configured["output_usd_per_million"] == 4.0
    assert resolve_pricing(SimpleNamespace(), "provider/other") == {
        "source": "simulated_fallback",
        "model_id": "provider/other",
        "input_usd_per_million": 0.15,
        "cached_usd_per_million": 0.03,
        "output_usd_per_million": 0.50,
    }


# --- Choosing and running the efficiency analysis ---

@pytest.mark.parametrize("saved,explicit,expected", [
    (None, None, {"agent"}),
    (["agent"], ["efficiency"], {"efficiency"}),
], ids=["default", "flag_overrides_saved"])
def test_analysis_defaults_and_cli_overrides(monkeypatch, saved, explicit, expected) -> None:
    cfg = Config() if saved is None else Config(analysis_outputs=saved)
    monkeypatch.setattr(cli, "load_config", lambda: cfg)
    monkeypatch.setattr(cli.sys, "stdin", SimpleNamespace(isatty=lambda: explicit is not None))
    monkeypatch.setattr(cli.sys, "stdout", SimpleNamespace(isatty=lambda: explicit is not None))
    monkeypatch.setattr(
        "autofa.ui.settings.configure_analysis",
        lambda *args, **kwargs: pytest.fail("Explicit flags and noninteractive runs need no menu"),
    )
    configured = cli._discovery_config(SimpleNamespace(analysis_outputs=explicit))
    assert set(configured.analysis_outputs) == expected


@pytest.mark.parametrize("toggles,expected", [
    ([], {"agent"}),
    (["efficiency"], {"efficiency", "agent"}),
    (["efficiency", "agent"], {"efficiency"}),
], ids=["failure_analysis", "both", "efficiency_only"])
def test_new_uses_initial_analysis_settings_before_selecting_data(
    tmp_path, monkeypatch, toggles, expected,
) -> None:
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    _interactive(monkeypatch)
    monkeypatch.setattr(cli, "load_config", Config)
    choices = iter(["analyses", *toggles, "done", "done"] if toggles else ["done"])
    events = []

    def choose(title, options, **kwargs):
        assert "data" not in events
        value = next(choices)
        events.append(value)
        return next(option for option in options if option.value == value)

    def dataset(**kwargs):
        assert events[-1] == "done"
        events.append("data")
        return DatasetSelection(parquet_paths=[], model_id="model", benchmarks=["bench"])

    monkeypatch.setattr("autofa.ui.settings.select_option", choose)
    monkeypatch.setattr(cli, "run_dataset_wizard", dataset)
    validated, preflighted, started = [], [], []
    monkeypatch.setattr(cli, "_show_efficiency_note", lambda message, warning: None)
    monkeypatch.setattr(
        "autofa.data.efficiency_support.verify_efficiency_data",
        lambda selection: validated.append(selection) or EfficiencyReadiness(40, 20, 20, 40),
    )
    monkeypatch.setattr("autofa.run_spec.preflight_autofa_config",
                        lambda config: preflighted.append(config))
    run = SimpleNamespace(run_id="test-run", save=lambda: None)
    monkeypatch.setattr(cli, "_create_analysis_run", lambda *args, **kwargs: run)
    monkeypatch.setattr("autofa.estimates.prepare_estimator", lambda *args: None)
    monkeypatch.setattr(
        "autofa.wizard.depth.select_discovery_depth",
        lambda **kwargs: SimpleNamespace(max_rounds=3, top_x=5, min_failure_modes=4),
    )
    monkeypatch.setattr("autofa.session.runner.run_session",
                        lambda *args, **kwargs: started.append("agent"))
    monkeypatch.setattr(cli, "_generate_efficiency_metrics",
                        lambda *args: started.append("efficiency"))
    monkeypatch.setattr(cli, "_offer_efficiency_publication", lambda *args: None)
    monkeypatch.setattr(cli, "select_option",
                        lambda *args, **kwargs: pytest.fail("No later analysis picker"))

    cli._new(SimpleNamespace(folder=None, keep_going=False, eval_limit=None, no_auto=True))

    assert set(run.analysis_outputs) == expected
    assert [set(config.settings.analysis_outputs) for config in preflighted] == [expected]
    # Prepared efficiency data is checked exactly once, and only when efficiency is selected.
    assert len(validated) == int("efficiency" in expected)
    assert started == (["agent"] if "agent" in expected else ["efficiency"])


def test_cli_generates_selected_efficiency_metrics_without_agent(tmp_path, monkeypatch) -> None:
    calls = []
    run = SimpleNamespace(run_dir=lambda: tmp_path / "run", model_id="model")

    def write(selected, output_dir, **kwargs):
        calls.append((selected, output_dir, kwargs))
        output_dir.mkdir()
        (output_dir / "efficiency.html").write_text("card")
        (output_dir / "efficiency.json").write_text("{}")
        return output_dir / "efficiency.html", output_dir / "efficiency.json", {
            "n_items": 3, "metrics": [{}, {}],
        }

    monkeypatch.setattr(efficiency_card, "write_efficiency_card", write)
    monkeypatch.setattr(cli.webbrowser, "open", lambda url: True)
    assert cli._generate_efficiency_metrics(run, SimpleNamespace())  # noqa: SLF001
    [(selected, output_dir, options)] = calls
    assert (selected, output_dir) == (run, tmp_path / "run")
    assert options["reference_data"] == []
    # The run's model selects which recorded requests count, not only the price.
    assert (options["pricing"]["source"], options["pricing"]["model_id"]) == (
        "simulated_fallback", "model",
    )


@pytest.mark.parametrize("remote", [True, False], ids=["remote_library", "local_only"])
def test_efficiency_only_ui_offers_publication_only_with_a_library(monkeypatch, remote) -> None:
    run = SimpleNamespace(run_id="run-1")
    cfg = SimpleNamespace(remote_library_base_url="https://library.example.test",
                          remote_library_mode="cloud") if remote else Config()
    calls, menus = [], []
    _interactive(monkeypatch)
    monkeypatch.setattr(cli.console, "print", lambda *args, **kwargs: None)

    def select_publish(title, options, **kwargs):
        menus.append(title)
        return next(option for option in options if option.value == "publish")

    monkeypatch.setattr(cli, "select_option", select_publish)
    monkeypatch.setattr(remote_library_publish, "maybe_publish_run", lambda *args, **kwargs: (
        calls.append((args, kwargs)) or SimpleNamespace(library_id="library-1")
    ))
    cli._offer_efficiency_publication(run, cfg)  # noqa: SLF001
    assert len(menus) == len(calls) == (1 if remote else 0)
    if remote:
        assert calls == [((run, cfg), {"trigger": "manual_cli", "console": cli.console})]


def test_efficiency_only_publication_uploads_standalone_report(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    dataset_path = tmp_path / "dataset.parquet"
    dataset_path.write_bytes(b"canonical fixture")
    (tmp_path / "manifest.json").write_text(json.dumps({
        "trace_exports": [{"sha256": "efficiency-fixture", "rollouts": None}],
    }))
    bulk = tmp_path.parent / "source-bulk-efficiency-fixture"
    bulk.mkdir(exist_ok=True)
    (bulk / "download.json").write_text(json.dumps({
        "run_id": "source-run", "archive_sha256": "efficiency-fixture",
    }))
    with sqlite3.connect(bulk / "trace.sqlite") as db:
        db.execute("CREATE TABLE records (tbl TEXT, payload TEXT)")
    public_reference = {
        "run_id": "source-run",
        "export_url": "https://runs.example.test/v1/runs/source-run/export/autofa",
        "source_harness": "example",
    }
    run = RunConfig(
        run_id="efficiency-run", parquet_paths=[str(dataset_path)], model_id="model-a",
        benchmarks=["bench-a"], analysis_outputs=["efficiency"],
        reference_data=[{**public_reference, "parquet_path": str(dataset_path)}],
    )
    marker = "PRIVATE-TRAJECTORY-MARKER"
    library = FunctionLibrary(model_id="model-a", benchmark="bench-a", functions=[
        FeatureFunction(name="old-detector", description=marker,
                        source="def old_detector(item): return True",
                        failure_localisations={"private-item": {"passages": [{"quote": marker}]}}),
    ])

    class Runtime:
        dataset = object()

        def __init__(self):
            self.library = library

        def persist_library(self):
            path = run.run_dir() / "library.json"
            self.library.save(path)
            return path

    uploaded = []

    class Client:
        def upload_library(self, **kwargs):
            branch_key = kwargs["metadata"]["branch_key"]
            uploaded.append(kwargs)
            return RemotePublication(
                library_id="library-1", branch_id="branch-1", version_id="version-1",
                branch_key=branch_key, version_number=1, source_run_id=run.run_id,
                dashboard_url="https://library.example.test/library-1",
            )

    def write_report(selected_run, output_dir, **kwargs):
        assert selected_run is run and "library" not in kwargs
        assert kwargs["reference_data"] == [public_reference]
        path = Path(output_dir) / "efficiency.html"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("<html>efficiency only</html>", encoding="utf-8")
        path.with_suffix(".json").write_text("{}", encoding="utf-8")
        return path, path.with_suffix(".json"), {}

    monkeypatch.setattr(efficiency_card, "write_efficiency_card", write_report)
    monkeypatch.setattr(failure_cards, "build_report_assets_zip",
                        lambda report_dir, *, html_name: b"assets")
    publish_run(run, Config(remote_library_base_url="https://library.example.test"),
                trigger="manual_cli", runtime=Runtime(), client=Client(), show_progress=False)

    [upload] = uploaded
    report = Path(upload["failure_cards_html"])
    assert report.name == "efficiency.html"
    assert report.read_text() == "<html>efficiency only</html>"
    assert marker.encode() not in upload["report_assets_zip"]
    empty = FunctionLibrary(model_id="model-a", benchmark="bench-a")
    assert upload["library_json"] == empty.to_dict()
    # The local library keeps its detectors on disk and in memory; only the upload is emptied.
    assert marker in (run.run_dir() / "library.json").read_text()
    assert library.functions[0].description == marker


# --- Report layouts ---

def test_standalone_efficiency_report_uses_empty_report_library(tmp_path, monkeypatch) -> None:
    run = SimpleNamespace(library_model_id=lambda: "model-a", benchmark_label=lambda: "bench-a")
    captured = {}

    def write_shell(library, *, out, **kwargs):
        captured["library"] = library
        Path(out).write_text(_REPORT_SHELL.format(meta='{"meta":{}}'), encoding="utf-8")

    monkeypatch.setattr(efficiency_card, "_evalrdl_sections", lambda selected_run, pricing=None: (
        {"n_items": 0, "n_fail": 0, "metrics": [], "pricing": {}}, {},
    ))
    monkeypatch.setattr(failure_cards, "write_failure_cards", write_shell)
    html_path, _, _ = efficiency_card.write_efficiency_card(run, tmp_path)
    library = captured["library"]
    assert (library.model_id, library.benchmark, library.functions) == ("model-a", "bench-a", [])
    assert '"efficiency_layout": "efficiency-tab"' in html_path.read_text(encoding="utf-8")


def test_tab_layout_keeps_failure_modes_separate() -> None:
    html = _REPORT_SHELL.format(meta='{"meta":{"n_items":10,"n_failures":2,"base_fail_rate":0.2}}')
    rendered = _inject_overview(html, {"n_items": 10, "n_fail": 8}, sections={}, layout="tabs")
    payload = json.loads(rendered.split('type="application/json">', 1)[1].split("</script>", 1)[0])
    meta = payload["meta"]
    assert meta["efficiency_layout"] == "tabs"
    assert (meta["n_items"], meta["n_failures"], meta["base_fail_rate"]) == (10, 2, 0.2)
    assert meta["efficiency_metrics"]["n_fail"] == 8
    assert 'id="report-tab-failures"' in rendered and 'id="report-tab-efficiency"' in rendered
    # Re-injecting replaces the previous efficiency block instead of stacking another one.
    rerendered = _inject_overview(rendered, {"n_items": 10, "n_fail": 8}, sections={},
                                  layout="tabs")
    assert rerendered.count("/* —— AutoFA efficiency metrics —— */") == 2


_FAKE_PAGE = """
const byId = {};
function element(id) {
  const node = {
    id, children: [], hidden: false, parentNode: null, innerHTML: '',
    classList: {add() {}}, setAttribute() {}, addEventListener() {},
    closest() { return main; },
    querySelectorAll() { return [element(''), element('')]; },
    appendChild(child) {
      child.parentNode?.children.splice(child.parentNode.children.indexOf(child), 1);
      child.parentNode = node;
      node.children.push(child);
    },
    insertBefore(child) { node.appendChild(child); },
    remove() { node.parentNode = null; delete byId[node.id]; },
  };
  return node;
}
const main = element('main');
for (const id of ['library-coverage', 'kpi-band', 'execution-status', 'cards']) {
  byId[id] = element(id);
  main.appendChild(byId[id]);
}
const document = {getElementById: id => byId[id] || null, createElement: () => element('')};
const location = {hash: ''};
const escapeHtml = value => String(value ?? '');
var boot = function () {};
"""


def _boot_overview(report, *, layout, inspect, setup=""):
    raw = {"meta": {"efficiency_layout": layout, "efficiency_metrics": report}}
    script = (
        _FAKE_PAGE + f"const RAW = {json.dumps(raw)};\nglobalThis.queueMicrotask = () => {{}};\n"
        + setup + _OVERVIEW_JS + f"\nboot();\nconsole.log(JSON.stringify({inspect}));"
    )
    return json.loads(run_node(script))


@pytest.mark.parametrize("layout,expected", [
    ("efficiency-tab", None),
    ("tabs", "report-panel-failures"),
])
def test_library_coverage_stays_out_of_efficiency_views(layout, expected) -> None:
    report = {"n_items": 0, "n_fail": 0, "metrics": [], "pricing": {}, "sections": {}}
    box = "document.getElementById('library-coverage')"
    assert _boot_overview(report, layout=layout, inspect=f"{box} && {box}.parentNode.id") == (
        expected
    )


def test_partly_recorded_metrics_are_flagged_with_their_coverage() -> None:
    def metric(name, label, n_pass, n_fail):
        return {"name": name, "label": label, "coverage": (n_pass + n_fail) / 1152,
                "overall": {"n": n_pass + n_fail}, "pass": {"n": n_pass}, "fail": {"n": n_fail}}

    report = {
        "n_items": 1152, "n_pass": 504, "n_fail": 648, "pricing": {}, "sections": {},
        "metrics": [
            metric("input_tokens", "Input tokens", 504, 315),
            metric("reasoning_tokens", "Reasoning tokens", 504, 315),
            metric("n_turns", "Assistant turns", 504, 648),
        ],
        "summary": {
            "differ": [], "cost": None, "negligible": [], "not_significant": [],
            "insufficient": [],
            "partial": {"Input tokens": 819 / 1152, "Reasoning tokens": 819 / 1152},
        },
    }
    html = " ".join(_boot_overview(
        report, layout="replace", inspect="document.getElementById('cards').innerHTML",
    ).split())
    notice = html[html.index("Incomplete data."):html.index("</div>")]
    assert ("Input tokens, Reasoning tokens: 819 of 1,152 rollouts "
            "(504 of 504 correct, 315 of 648 incorrect)") in notice
    assert "Assistant turns" not in notice


# --- Efficiency of the attempts each failure card matched ---

def _card_run():
    """30 problems with two attempts each: 0-14 correct, 15-29 incorrect."""
    tasks = [f"t{index}" for index in range(30) for _ in range(2)]
    repeats = [0, 1] * 30
    groups = ["pass" if index < 15 else "fail" for index in range(30) for _ in range(2)]
    tokens = [100.0 + row if group == "pass" else 500.0 + row
              for row, group in enumerate(groups)]
    attempts = {"task_id": tasks, "repeat": repeats, "groups": groups,
                "values": {"total_tokens": tokens}}
    payload = {"metrics": [{"name": "total_tokens", "label": "Total tokens", "unit": "tokens",
                            "sources": {"evalrdl": 60}}]}
    item_attempts = {f"m|b|{task}|{repeat}": (task, repeat)
                     for task, repeat in zip(tasks, repeats, strict=True)}
    return payload, attempts, item_attempts


def _matches(*problems):
    return {f"m|b|t{index}|{repeat}": f"t{index}" for index in problems for repeat in (0, 1)}


def test_failure_cards_compare_their_matched_attempts_from_twenty_attempts() -> None:
    payload, attempts, item_attempts = _card_run()
    cards = _card_sections(payload, attempts, {
        "frequent": {"matches": _matches(*range(15, 28)), "sampled": True},
        "rare": {"matches": _matches(15, 16), "sampled": False},
        "unrecorded": {"matches": None},
        "unlinked": {"matches": {"elsewhere|x|y|0": "y"}},
    }, item_attempts)
    frequent = cards["frequent"]
    chart = frequent["charts"]["total_tokens"]
    assert (frequent["n"], frequent["sampled"], frequent["metrics"]) == (
        26, True, ["total_tokens"],
    )
    assert [chart[group]["n"] for group in ("correct", "card", "other")] == [30, 26, 4]
    assert chart["significant"] is True
    assert cards["rare"] == {"n": 4, "min_attempts": 20, "sampled": False}
    assert cards["unrecorded"] == cards["unlinked"] == {"unavailable": True}


@pytest.mark.parametrize("row,attempt", [
    ({"model_id": "m", "benchmark": "b", "problem_id": "p1", "run_idx": 1}, ("p1", 1)),
    ({"model_id": "m", "benchmark": "b", "task_id": "p1", "repeat": 2}, ("p1", 2)),
], ids=["report_columns", "identity_aliases"])
def test_card_attempts_are_linked_by_the_report_item_ids(tmp_path, row, attempt) -> None:
    path = tmp_path / "dataset.parquet"
    pq.write_table(pa.Table.from_pylist([row]), path)
    assert _item_attempts(SimpleNamespace(parquet_paths=[str(tmp_path)])) == {
        make_item_id(apply_column_aliases(dict(row)), path): attempt,
    }


def test_unreadable_datasets_leave_card_attempts_unlinked(tmp_path) -> None:
    broken = tmp_path / "dataset.parquet"
    broken.write_bytes(b"not parquet")
    missing = tmp_path / "missing.parquet"
    assert _item_attempts(SimpleNamespace(parquet_paths=[str(broken), str(missing)])) == {}


def test_only_detectors_with_report_cards_are_compared(monkeypatch) -> None:
    monkeypatch.setattr(
        task_support, "_matched_failed_task_map", lambda scores: scores.get("matches"),
    )
    library = SimpleNamespace(functions=[
        SimpleNamespace(name="shown", scores={"matches": {"i": "t"}, "sampled": True}),
        SimpleNamespace(name="hidden", scores={"matches": {"j": "t"}}),
    ])
    html = ('<script id="autofa-data" type="application/json">'
            '{"functions": [{"name": "shown"}]}</script>')
    assert _report_detectors(html, library) == {
        "shown": {"matches": {"i": "t"}, "sampled": True},
    }
    assert _report_detectors(html, None) is None


_FAKE_CARDS = """
const inserted = {};
byId.cards.querySelectorAll = () => ['frequent', 'rare', 'unrecorded', 'plain'].map(name => ({
  getAttribute: () => name,
  querySelector: () => ({insertAdjacentHTML: (where, html) => { inserted[name] = html; }}),
}));
"""


@pytest.mark.parametrize("layout", ["tabs", "efficiency-tab"])
def test_card_efficiency_renders_only_in_combined_reports(layout) -> None:
    payload, attempts, item_attempts = _card_run()
    cards = _card_sections(payload, attempts, {
        "frequent": {"matches": _matches(*range(15, 28))},
        "rare": {"matches": _matches(15, 16)},
        "unrecorded": {"matches": None},
    }, item_attempts)
    report = {"n_items": 60, "n_fail": 30, "metrics": [], "pricing": {}, "sections": {},
              "cards": cards}
    inserted = _boot_overview(report, layout=layout, inspect="inserted", setup=_FAKE_CARDS)
    if layout == "efficiency-tab":
        assert inserted == {}
        return
    assert set(inserted) == {"frequent", "rare", "unrecorded"}
    assert 'data-card="frequent"' in inserted["frequent"]
    assert "at least 20 are needed" in " ".join(inserted["rare"].split())


def test_card_charts_stay_in_the_report_and_out_of_efficiency_json(tmp_path, monkeypatch) -> None:
    html_path = tmp_path / "failure_cards.html"
    html_path.write_text(
        '<style></style><main><div id="cards"></div></main>'
        '<script id="autofa-data" type="application/json">{"meta":{},"functions":[]}</script>'
        "<script>\nboot();\n</script>",
        encoding="utf-8",
    )
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    run = SimpleNamespace(analysis_outputs=["agent", "efficiency"], model_id="m")
    cards = {"detector": {"n": 4, "min_attempts": 20, "sampled": False}}
    monkeypatch.setattr(autofa.config.RunConfig, "load", lambda run_id: run)
    monkeypatch.setattr(autofa.config, "load_config", lambda: None)
    monkeypatch.setattr(efficiency_card, "_evalrdl_sections", lambda *args, **kwargs: (
        {"n_items": 1, "n_fail": 1, "metrics": [], "pricing": {}, "cards": cards}, {},
    ))
    assert efficiency_card.attach_saved_efficiency(html_path, SimpleNamespace(functions=[]))
    saved = json.loads((tmp_path / efficiency_card.DATA_NAME).read_text(encoding="utf-8"))
    embedded = json.loads(efficiency_card._DATA_RE.search(
        html_path.read_text(encoding="utf-8")).group(2))
    assert "cards" not in saved
    assert embedded["meta"]["efficiency_metrics"]["cards"] == cards
