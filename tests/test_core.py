# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline tests: metrics, buckets, sandbox, detector lint and composition, library, verdicts."""

from __future__ import annotations

import pytest

from autofa.core.dependencies import build_bundle, execution_bundle
from autofa.core.detector_lint import lint_detector_source
from autofa.core.fields import policy_from_settings, scan_banned_fields
from autofa.core.library import FeatureFunction, FunctionLibrary
from autofa.core.sandbox import SAFE_REGEX, RegexTimeoutError, compile_detector, run_detector
from autofa.mcp.tools_eval import (
    judge_evidence_sample_size,
    judge_reservoir_cap,
    sample_fire_bucket_evidence,
)
from autofa.metrics.association import (
    StreamingBucketAccumulator,
    contingency,
    matches_selected_bucket,
    pct_caught_from_scores,
    pick_primary_bucket,
    selected_bucket,
)
from autofa.metrics.pareto import pareto_front, rank_functions
from autofa.metrics.verdicts import (
    extract_metrics,
    resolved_behavior_purity,
    resolved_behavior_specificity,
    verdict_for_metrics,
    verdict_for_scores,
)

LONG_GRIND = '''
def long_grind_toolcalls(item, llm_client=None):
    rt = item.get("response_text") or item.get("generation") or ""
    if not rt:
        return False
    n = rt.count("source_call_id")
    return n >= 110
'''

HEAVY_EDIT = '''
def heavy_edit_churn(item, llm_client=None):
    import re
    rt = item.get("response_text") or item.get("generation") or ""
    n_edits = len(re.findall(r'"command"\\s*:\\s*"(?:str_replace|insert|append)"', rt))
    return n_edits >= 15
'''

MARATHON = '''
def verification_illusion_marathon(item, llm_client=None):
    import re
    cot = item.get("chain_of_thought") or ""
    rt = item.get("response_text") or ""
    hay = cot[-8000:] + "\\n" + rt[-12000:]
    has_pass = bool(re.search(r"=+\\s*\\d+\\s+passed", hay, re.I))
    narr = bool(re.search(r"all tests? (?:pass(?:ed)?|are passing)|issue is (?:now )?fixed",
                          hay, re.I))
    return bool(has_pass and narr and len(rt) > 80000)
'''

CONTENT_ONLY = '''
def retries_same_error(item, llm_client=None):
    import re
    text = (item.get("response_text") or "") + (item.get("chain_of_thought") or "")
    fails = re.findall(r"AssertionError|Traceback \\(most recent call last\\)", text)
    return len(fails) >= 3
'''

PASSTHROUGH = '''
def passthrough_parent(item, llm_client=None):
    return library.fire("long_grind_toolcalls", item)
'''


class LabelClient:
    def __init__(self, label, other=None):
        self.label, self.other = label, other

    def classify(self, prompt, text):
        return self.label


def test_contingency_reports_wald_or_ci_only_when_every_cell_is_positive():
    """Pure fire on fails (b=0) must not claim a Wald OR CI that straddles 1."""
    s = contingency(23, 0, 50, 16)
    assert s["or_ci_low"] is None and s["or_ci_high"] is None
    assert s["odds_ratio"] > 1.0 and s["fpr"] == 0.0
    assert s["fpr_ci_low"] <= 0.0 <= s["fpr_ci_high"]
    assert s["recall"] == pytest.approx(23 / 73)
    assert s["ppv_at_pi0"] == pytest.approx(1.0)
    # f1_at_pi0 orders bucket rows; with ppv=1 it is 2r/(1+r).
    assert s["f1_at_pi0"] == pytest.approx(23 / 48)
    assert s["pct_caught"] == pytest.approx(23 / 89)
    assert s["recall_ci_low"] < s["recall"] < s["recall_ci_high"]
    assert s["fisher_p"] < 0.05

    s = contingency(20, 10, 30, 40)
    assert s["or_ci_low"] < s["odds_ratio"] < s["or_ci_high"]


def test_pct_caught_from_cells_when_rate_withheld():
    # Unrepresented sampling strata null the stored rate but keep the cells.
    withheld = {"a": 634.0, "b": 0.0, "c": 251.0, "d": 2115.0, "n_items": 3000,
                "pct_caught": None, "metrics_scope": "unrepresented_strata"}
    assert pct_caught_from_scores(withheld) == pytest.approx(634 / 3000)
    # The judge path's feedback rate wins over the cells when the top-level rate is null.
    judged = {"pct_caught": None, "feedback": {"pct_caught": 0.0}, "a": 634.0, "b": 0.0,
              "n_items": 3000, "metrics_scope": "unrepresented_strata"}
    assert pct_caught_from_scores(judged) == 0.0
    assert pct_caught_from_scores({"pct_caught": 0.25}) == pytest.approx(0.25)
    assert pct_caught_from_scores({}) == 0.0


def test_claim_calibration_synthesize_and_format():
    from autofa.metrics.claim_calibration import (
        format_calibration_hint,
        parse_claim_calibration,
        synthesize_calibration_from_verdicts,
    )

    cal = parse_claim_calibration({"claim_calibration": {
        "status": "too_strong", "claim_note": "Overstates certainty.", "rewrite_hint": "Soften.",
    }})
    assert cal["status"] == "overclaim"
    assert "too strong" in format_calibration_hint(cal).lower()
    syn = synthesize_calibration_from_verdicts(
        [{"verdict": "yes", "rationale": "ok"}] * 3, purity=0.9,
    )
    assert syn["status"] == "good"
    assert format_calibration_hint(syn) is None


def test_quality_badge_scored_vs_empty():
    scored = verdict_for_metrics({"pct_caught": 0.2, "recall": 0.5, "precision": 1.0,
                                  "ppv_at_pi0": 1.0, "behavior_purity": 0.8})
    assert scored["code"] == "OK"
    empty = verdict_for_metrics(extract_metrics(
        {"recall": 0, "a": 0, "b": 0, "n_items": 10, "evaluated": True},
    ))
    assert empty["code"] == "EMPTY"
    assert verdict_for_scores({"recall": 0})["code"] == "UNSCORED"

    # Placeholder 0.0 from evaluate without judge metadata is unjudged, not a zero score.
    stamped = {"recall": 0.9, "ppv_at_pi0": 0.3, "behavior_purity": 0.0,
               "n_items": 10, "evaluated": True}
    assert resolved_behavior_purity(stamped) is None
    assert verdict_for_scores(stamped)["code"] == "UNSCORED"
    unreviewed = {k: v for k, v in stamped.items() if k != "behavior_purity"}
    assert verdict_for_scores(unreviewed)["label"] == "NOT EVALUATED"
    assert resolved_behavior_specificity(unreviewed) is None
    judged_zero = {**stamped, "judge_backend": "claude_agent_sdk", "behavior_n_judged": 8}
    assert resolved_behavior_purity(judged_zero) == 0.0


@pytest.mark.parametrize("scores,label,tone", [
    pytest.param({"evaluated": True, "n_items": 10, "n_flagged": 4, "recall": 0.4,
                  "ppv_at_pi0": 0.9, "behavior_purity": 1.0}, "OK", "warn",
                 id="high-rates-without-task-breadth"),
    pytest.param({"evaluated": True, "n_items": 140, "n_flagged": 22, "recall": 0.24,
                  "ppv_at_pi0": 0.56, "behavior_purity": 0.5, "behavior_specificity": 0.2},
                 "OK", "warn", id="low-specificity-keeps-ok"),
    pytest.param({"evaluated": True, "n_items": 10, "n_flagged": 1, "recall": 0.0,
                  "ppv_at_pi0": 0.0, "behavior_purity": 1.0}, "WEAK", "bad",
                 id="matches-only-passes"),
    pytest.param({"evaluated": True, "n_items": 10, "n_flagged": 0, "recall": 0.0},
                 "NO MATCHES", "neutral", id="no-matches"),
    pytest.param({}, "NOT EVALUATED", "neutral", id="empty"),
    pytest.param({"evaluated": True, "n_items": 10, "n_scored": 0, "recall": 0.0},
                 "NOT EVALUATED", "neutral", id="nothing-scored"),
])
def test_quality_badge_states(scores, label, tone):
    badge = verdict_for_scores(scores)
    assert (badge["label"], badge["tone"]) == (label, tone)
    if "behavior_specificity" in scores:
        assert resolved_behavior_specificity(scores) == 0.2


def test_quality_and_frontier_badges_require_source_backed_task_breadth_for_good():
    from support.task_support import attach_task_support

    fn = FeatureFunction(name="supported", source="def detect(item): return True",
                         description="Observable mistake",
                         mechanism="The mistake prevents completion",
                         scores={"evaluated": True, "n_items": 100, "n_flagged": 5,
                                 "recall": .05, "ppv_at_pi0": .9, "behavior_purity": 1.0,
                                 "behavior_n_judged": 2, "behavior_n_completed": 2,
                                 "behavior_n_yes": 2})
    assert verdict_for_scores(fn.scores)["label"] == "OK"
    attach_task_support(fn, task_n=5)
    fn.scores["behavior_verdicts"] = fn.scores["behavior_verdicts"][:2]
    assert verdict_for_scores(fn.scores)["label"] == "GOOD"


def _bucket_rows(preds, labels):
    accumulator = StreamingBucketAccumulator()
    for pred, label in zip(preds, labels, strict=True):
        accumulator.update(pred, is_correct=label)
    return accumulator.finalize()


def test_pick_primary_bucket_prefers_fire_side_on_high_fail_corpus():
    """Regression: majority False must not become the headline / judge bucket.

    Sparse precise fires (True) on a high-fail corpus: False has higher recall/f1
    but is the non-fire side, so primary metrics and behavior_purity must use True.
    """
    preds = [True] * 14 + [False] * (82 - 14 + 18)
    labels = [False] * 82 + [True] * 18  # False=fail, True=pass
    buckets = _bucket_rows(preds, labels)
    by = {r["bucket"]: r for r in buckets}
    assert (by["True"]["precision"], by["True"]["n"]) == (pytest.approx(1.0), 14)
    assert by["False"]["recall"] > by["True"]["recall"]
    primary = pick_primary_bucket(buckets)
    assert (primary["bucket"], primary["precision"]) == ("True", pytest.approx(1.0))


def test_pick_primary_bucket_categorical_skips_ok_pass():
    preds = ["abandoned_plan"] * 5 + ["ok"] * 20
    labels = [False] * 15 + [True] * 10  # 5 fire-fails, 10 ok-fails, 10 ok-passes
    primary = pick_primary_bucket(_bucket_rows(preds, labels))
    assert (primary["bucket"], primary["precision"]) == ("abandoned_plan", pytest.approx(1.0))


@pytest.mark.parametrize("negative", [False, "ok", None])
def test_pick_primary_bucket_does_not_promote_only_negative_predictions(negative):
    buckets = _bucket_rows([negative] * 4, [False, False, True, True])
    assert pick_primary_bucket(buckets) is None


def test_truthy_bucket_selection_matches_boolean_fires():
    selection = selected_bucket({"bucket": "positive", "bucket_mode": "truthy"})
    assert matches_selected_bucket(True, selection)
    assert matches_selected_bucket("True", selection)
    assert not matches_selected_bucket(False, selection)


def test_exact_positive_bucket_selection_matches_only_that_value():
    selection = selected_bucket({"bucket": "positive", "bucket_mode": "exact"})
    assert matches_selected_bucket("positive", selection)
    assert not matches_selected_bucket("other", selection)
    assert not matches_selected_bucket(True, selection)


def test_unknown_bucket_mode_never_counts_matches():
    selection = selected_bucket({"bucket": "positive", "bucket_mode": "unsupported"})
    assert not matches_selected_bucket("positive", selection)
    assert not matches_selected_bucket(True, selection)


def test_judge_evidence_sample_size_and_random():
    assert [judge_evidence_sample_size(n) for n in (0, 2, 30, 31, 200)] == [0, 2, 3, 4, 20]
    assert judge_evidence_sample_size(14, evidence_n=12) == 12  # raises the floor
    assert judge_evidence_sample_size(100, evidence_n=3) == 10  # cannot go below auto

    pool = list(range(20))
    a = sample_fire_bucket_evidence(pool, n=10, seed=42)
    assert a == sample_fire_bucket_evidence(pool, n=10, seed=42)
    assert len(a) == 10 and set(a) <= set(pool)
    assert a != sample_fire_bucket_evidence(pool, n=10, seed=99)
    assert sample_fire_bucket_evidence(pool, n=100, seed=1) == pool


def test_judge_reservoir_never_clips_the_sample():
    for scan_limit in (1, 40, 15_000):
        for evidence_n in (None, 5, 400):
            cap = judge_reservoir_cap(scan_limit, evidence_n=evidence_n)
            for n_fires in (1, min(10, scan_limit), scan_limit // 2, scan_limit):
                n_sample = judge_evidence_sample_size(n_fires, evidence_n=evidence_n)
                assert n_sample <= min(n_fires, cap) or n_fires == 0


def test_banned_fields_scan():
    src = 'def detect(item, llm=None):\n    return item["is_correct"]\n'
    assert "is_correct" in scan_banned_fields(src)
    with pytest.raises(ValueError, match="disallowed|banned"):
        compile_detector(src)


def test_content_mode_blocks_judge_metadata():
    pol = policy_from_settings(detector_fields="content")
    bad = 'def detect(item, llm_client=None):\n    return bool(item.get("judge_text"))\n'
    assert "judge_text" in scan_banned_fields(bad, pol)
    with pytest.raises(ValueError, match="judge_text"):
        compile_detector(bad, policy=pol)
    compile_detector(bad, policy=policy_from_settings(detector_fields="metadata"))

    good = '''
import re
def verification_illusion_marathon(item, llm_client=None):
    cot = item.get("chain_of_thought") or ""
    rt = item.get("response_text") or ""
    return bool(re.search(r"passed", cot + rt, re.I))
'''
    fn = compile_detector(good, policy=pol)
    assert fn({"chain_of_thought": "all tests passed", "response_text": ""}, None) is True
    item = {"chain_of_thought": "x", "response_text": "y", "judge_text": "cheat"}
    assert run_detector(good, item, policy=pol) is False


@pytest.mark.parametrize("compiled", [False, True], ids=["module", "compiled"])
@pytest.mark.parametrize("name", [
    "findall", "finditer", "fullmatch", "match", "search", "split", "sub", "subn",
])
def test_sandbox_regex_defaults_timeout_and_keeps_positional_concurrency(
    monkeypatch, compiled, name,
):
    """Each method's positional `concurrent` index decides whether a default is injected."""
    import regex

    # Actual regex API argument layouts, stopping just before concurrent.
    if name in {"match", "search", "fullmatch"}:
        args = ("a", None, None) if compiled else ("(a)", "a", 0, None, None, False)
    elif name in {"findall", "finditer"}:
        args = ("a", None, None, False) if compiled else ("(a)", "a", 0, None, None, False)
        if name == "finditer" and not compiled:
            args += (False,)
    elif name == "split":
        args = ("a", 0) if compiled else ("(a)", "a", 0, 0)
    else:
        args = ("X", "a", 0, None, None) if compiled else ("(a)", "X", "a", 0, 0, None, None)

    native_pattern = regex.compile("(a)")
    native_call = getattr(native_pattern if compiled else regex, name)
    calls = []

    def observed(*call_args, **call_kwargs):
        calls.append((call_args, call_kwargs))
        result = native_call(*call_args, **call_kwargs)
        return list(result) if name == "finditer" else result

    if compiled:
        class ObservedPattern:
            def __getattr__(self, method):
                return observed if method == name else getattr(native_pattern, method)

        wrapped = SAFE_REGEX.compile("(a)")
        wrapped._pattern = ObservedPattern()
        call = getattr(wrapped, name)
    else:
        monkeypatch.setattr(regex, name, observed)
        call = getattr(SAFE_REGEX, name)

    for extra in [(), (True,)]:
        result = call(*args, *extra)
        if name == "finditer":
            result = list(result)
        assert result
    assert [call_args for call_args, _ in calls] == [args, (*args, True)]
    implicit, positional = (call_kwargs for _, call_kwargs in calls)
    assert implicit == {"timeout": 0.1, "concurrent": False}
    assert positional == {"timeout": 0.1}


@pytest.mark.parametrize("search", [
    're.search(r"(a+)+$", text)', 're.compile(r"(a+)+$").search(text)',
])
def test_sandbox_regex_timeout(search):
    src = f"""
import re
def detect(item, llm=None):
    text = item.get("response_text") or ""
    return bool({search})
"""
    with pytest.raises(RegexTimeoutError):
        run_detector(src, {"response_text": "a" * 50_000 + "X"})


@pytest.mark.parametrize("compiled", [False, True], ids=["module", "compiled"])
def test_sandbox_regex_iterator_timeout_stays_a_local_execution_error(compiled):
    text = "a" * 50_000 + "X"
    iterator = (SAFE_REGEX.compile(r"(a+)+$").finditer(text) if compiled
                else SAFE_REGEX.finditer(r"(a+)+$", text))
    with pytest.raises(RegexTimeoutError):
        next(iterator)


def test_library_roundtrip_keeps_composition_metadata(tmp_path):
    lib = FunctionLibrary(model_id="m", benchmark="b")
    lib.upsert(FeatureFunction(
        name="child", description="d", source=CONTENT_ONLY, depends_on=["parent"],
        explains="parent", meta={"deps": {"parent": LONG_GRIND}},
    ))
    path = tmp_path / "library.json"
    lib.save(path)
    loaded = FunctionLibrary.load(path)
    assert (loaded.model_id, loaded.benchmark) == ("m", "b")
    child = loaded.get("child")
    assert child.depends_on == ["parent"] and child.explains == "parent"
    assert child.meta == {"deps": {"parent": LONG_GRIND}}


def test_pareto_front_keeps_non_dominated_points():
    front = pareto_front([(0.9, 0.1, 0.5), (0.5, 0.9, 0.5), (0.4, 0.05, 0.2)])
    assert front == [(0.9, 0.1, 0.5), (0.5, 0.9, 0.5)]


def test_rank_functions():
    from autofa.config import Config

    config = Config(objective_axes=["ppv_at_pi0"], objective_weights=[1])
    fns = [{"name": "a", "scores": {"ppv_at_pi0": 0.5}},
           {"name": "b", "scores": {"ppv_at_pi0": 0.8}}]
    for candidates in (fns, list(reversed(fns))):
        top = rank_functions(candidates, top_x=1, config=config)
        assert top[0]["name"] == "b"
        assert top[0]["ranking"]["eligible"]


@pytest.mark.parametrize("src", [LONG_GRIND, HEAVY_EDIT, PASSTHROUGH],
                         ids=["count", "regex-count", "passthrough"])
def test_lint_rejects_structural_proxies(src):
    ok, reason = lint_detector_source(src)
    assert not ok and reason


def test_lint_accepts_marathon_and_content():
    assert lint_detector_source(MARATHON)[0]
    assert lint_detector_source(CONTENT_ONLY)[0]


@pytest.mark.parametrize("field,reader", [
    ("prompt_tokens", "metadata(item)"), ("latency_s", "steps(item)[0].metadata"),
])
def test_execution_metadata_only_threshold_is_still_a_proxy(field, reader):
    source = f'''
from autofa.trace import metadata, events, steps
def detect(item):
    return {reader}.get({field!r}, 0) > 1000
'''
    ok, reason = lint_detector_source(source)
    assert not ok and "scaffold" in reason


def test_execution_metadata_can_stratify_a_behavioral_mechanism():
    assert lint_detector_source('''
from autofa.trace import steps
def detect(item):
    return any(s.metadata.get('prompt_tokens', 0) > 1000
               and 'AssertionError' in s.observation for s in steps(item))
''')[0]


def _pinned(*peers: tuple[str, str]) -> dict:
    """Pin ``peers`` as submission does; each later peer composes the previous one."""
    library, previous = FunctionLibrary(), None
    for name, source in peers:
        bundle = build_bundle(library, depends_on=[previous] if previous else None)
        library.upsert(FeatureFunction(name=name, description=name, source=source,
                                       meta={"dependency_bundle": bundle} if bundle else {}))
        previous = name
    return build_bundle(library, depends_on=[previous])


def test_compose_library_fire():
    parent = '''
def parent_fn(item, llm_client=None):
    return "long" in (item.get("response_text") or "")
'''
    child = '''
def child_fn(item, llm_client=None):
    import re
    if not library.fire("parent_fn", item):
        return False
    return bool(re.search(r"AssertionError", item.get("response_text") or ""))
'''
    assert lint_detector_source(child)[0]
    bundle = _pinned(("parent_fn", parent))
    fn = compile_detector(child, bundle=bundle)
    assert fn({"response_text": "long run\\nAssertionError: x"}) is True
    assert fn({"response_text": "short"}) is False
    item = {"response_text": "long\\nAssertionError"}
    assert run_detector(child, item, bundle=bundle) is True


def test_composition_requires_pinned_dependencies():
    a = 'def a(item, llm_client=None):\n    return library.fire("b", item)\n'
    # A detector saved before dependency bundles kept only meta.deps; it fails clearly.
    pre_bundle = FeatureFunction(name="a", description="a", source=a, depends_on=["b"],
                                 meta={"deps": {"b": "def b(item):\n    return True\n"}})
    remedy = "no pinned dependency bundle for 'b'; recreate it with depends_on= in the current"
    with pytest.raises(KeyError, match=remedy):
        run_detector(a, {"response_text": "x"}, bundle=execution_bundle(pre_bundle))
    # A composed peer saved without its own pinned bundle cannot be pinned again.
    unpinned = FunctionLibrary(functions=[FeatureFunction(name="a", description="a", source=a)])
    with pytest.raises(ValueError, match="'a' calls library.fire without a pinned dependency"):
        build_bundle(unpinned, depends_on=["a"])


@pytest.mark.parametrize("compiled,label", [
    (False, "stops_unverified"), (True, "stops_unverified"), (True, "verified_done"),
])
def test_composed_llm_inherits_client_and_respects_parent(compiled, label):
    parent = '''
def parent(item, llm_client=None):
    if llm_client is None:
        return "unknown"
    return llm_client.classify("verified?", item.get("response_text", "")) == "stops_unverified"
'''
    child = '''
def child(item, llm_client=None):
    if not library.fire("parent", item):
        return False
    return "not found" in item.get("response_text", "")
'''
    item = {"response_text": "npm: not found; installed npm, tests passed"}
    client = LabelClient(label)
    bundle = _pinned(("parent", parent))
    if compiled:
        result = compile_detector(child, bundle=bundle)(item, client)
    else:
        result = run_detector(child, item, llm=client, bundle=bundle)
    assert result is (label == "stops_unverified")


PARENT_LLM = "def parent(item, llm_client=None):\n    return llm_client.classify('p', 't')\n"
MIDDLE = "def middle(item, llm_client=None):\n    return library.fire('parent', item)\n"


def test_nested_llm_dependency_clients_are_isolated_across_workers():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    child = "def child(item, llm_client=None):\n    return library.fire('middle', item)\n"
    fn = compile_detector(child, bundle=_pinned(("parent", PARENT_LLM), ("middle", MIDDLE)))
    barrier = Barrier(2, timeout=3)

    class Client(LabelClient):
        def classify(self, prompt, text):
            barrier.wait()
            return self.label

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(fn, {}, Client(label)) for label in ("yes", "no")]
        assert [future.result() for future in futures] == ["yes", "no"]


def test_dependency_explicit_client_override_restores_outer_context():
    child = '''
def child(item, llm_client=None):
    first = library.fire("middle", item, llm_client=llm_client.other)
    second = library.fire("middle", item)
    return (first, second)
'''
    fn = compile_detector(child, bundle=_pinned(("parent", PARENT_LLM), ("middle", MIDDLE)))
    assert fn({}, LabelClient("outer", LabelClient("override"))) == ("override", "outer")
    with pytest.raises(AttributeError):
        compile_detector(MIDDLE, bundle=_pinned(("parent", PARENT_LLM)))({})
