# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Taxonomy classifier: providers, first and second passes, windows and diagnosis."""

from __future__ import annotations

import asyncio
import hashlib
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import pytest

from autofa.taxonomy.classifier import (
    _DIAGNOSIS_QUESTION,
    TaxonomyClassifier,
    _classification_verdict,
    _earliest_open_episode,
    _validate_diagnosis_references,
    _Verdict,
    _verdict_episode,
    _verdict_matches_episode,
    _window_failure_signatures,
)
from autofa.taxonomy.models import ProviderResponse
from autofa.taxonomy.providers import OpenAIProvider, ProviderError, normalize_choice_response
from autofa.taxonomy.schema import Taxonomy

from support.taxonomy import _diagnosis_answer

# The data URL encodes the bytes b"image"; providers verify them against the digest.
IMAGE = {"data_url": "data:image/png;base64,aW1hZ2U=",
         "sha256": hashlib.sha256(b"image").hexdigest()}
VISUAL = {"artifact_idx": 3, "kind": "image", "renderable": True}
WRONG_ANSWER = {"id": "x-mm", "observed_event": "The answer is wrong."}
MISSING_DATA = {"id": "x-missing", "observed_event": "Data was Missing"}


def small_taxonomy() -> Taxonomy:
    def node(node_id: str, code: str, label: str, children=(), **extra) -> dict[str, Any]:
        children_field = {"children": list(children)} if children else {}
        return {"id": node_id, "code": code, "label": label, "description": f"{label} marker",
                **extra, **children_field}

    return Taxonomy.from_dict({
        "name": "test failures", "version": "1",
        "root": {"id": "failure", "label": "Failure", "description": "Failure", "children": [
            node("data", "D", "Data", [node("missing", "D.1", "Missing", fault_side="Data team"),
                                       node("corrupt", "D.2", "Corrupt")]),
            node("model", "M", "Model", [node("wrong", "M.1", "Wrong"),
                                         node("unstable", "M.2", "Unstable")]),
        ]},
    })


def _choice(choice: str, rationale: str = "scripted") -> dict[str, Any]:
    return {"type": "choice", "choice": choice, "alternatives": [], "rationale": rationale}


def _choice_ids(questions: dict[str, dict[str, Any]]) -> list[str]:
    """Choice questions; scripted judges answer the structured diagnosis separately."""
    return [key for key, question in questions.items() if question["type"] == "choice"]


def _with_diagnosis(
    questions: dict[str, dict[str, Any]], answers: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Add the diagnosis answer when the classifier asked for one."""
    return {**answers, "diagnosis": _diagnosis_answer()} if "diagnosis" in questions else answers


class MarkerProvider:
    """Picks the leaf whose marker is in the incident; ``second`` overrides the reflection pass."""

    def __init__(self, second: str | None = None) -> None:
        self.second = second
        self.calls: list[dict[str, dict[str, Any]]] = []
        self.states: list[Any] = []

    def evaluate(self, state: Any, questions: dict[str, dict[str, Any]]) -> ProviderResponse:
        self.states.append(state)
        self.calls.append(questions)
        text = str(state["incident"]).lower()
        reflecting = "first_pass" in state
        answers: dict[str, dict[str, Any]] = {}
        for question_id in _choice_ids(questions):
            question = questions[question_id]
            keys = list(question["criteria"])
            marked = [key for key, value in question["criteria"].items()
                      if value.lower().split()[0].rstrip(":") in text]
            choice = (self.second if reflecting else None) or (marked[0] if marked else keys[-1])
            answers[question_id] = {
                "type": "choice", "choice": choice,
                "alternatives": [key for key in keys if key != choice][:1], "rationale": "scripted",
            }
        return ProviderResponse(
            "scripted-judge", _with_diagnosis(questions, answers), {"input_tokens": 10}
        )


class DiagnosingProvider(MarkerProvider):
    """MarkerProvider that localizes the failure at ``event_ref`` and leaves episodes alone."""

    def __init__(self, event_ref: int, second: str | None = None) -> None:
        super().__init__(second)
        self.event_ref = event_ref

    def evaluate(self, state: Any, questions: dict[str, dict[str, Any]]) -> ProviderResponse:
        if all(question_id.startswith("episode_") for question_id in questions):
            self.states.append(state)
            self.calls.append(questions)
            answers = {question_id: _choice("not_addressed") for question_id in questions}
            return ProviderResponse("scripted-judge", answers, {"input_tokens": 10})
        response = super().evaluate(state, {"leaf": questions["leaf"]})
        diagnosis = {key: [] if isinstance(example, list) else "supported"
                     for key, example in questions["diagnosis"]["schema"].items()}
        diagnosis["critical_event"] = {"event_ref": self.event_ref, "failure": "The action failed.",
                                       "why_earliest_unrecovered": "No later recovery."}
        diagnosis["counterfactual_edit"] = {
            "target_event_ref": self.event_ref, "minimal_edit": "Correct the action.",
            "expected_effect": "The failure is avoided.", "support": "supported",
        }
        answers = {**response.answers, "diagnosis": {"type": "structured", "value": diagnosis}}
        return ProviderResponse(response.model, answers, response.usage)


class ScriptedProvider:
    """Answers each question with ``pick(call_number, question_id, state)``."""

    def __init__(self, pick) -> None:
        self.pick = pick
        self.states: list[dict[str, Any]] = []

    def evaluate(self, state: dict[str, Any], questions: dict[str, Any]) -> ProviderResponse:
        self.states.append(state)
        call = len(self.states)
        answers = {question_id: _choice(self.pick(call, question_id, state))
                   for question_id in _choice_ids(questions)}
        return ProviderResponse("window-judge", _with_diagnosis(questions, answers), {})


def _opens_d1(call: int, question_id: str, state: dict[str, Any]) -> str:
    return "D.1" if question_id == "leaf" else "open_primary"


def _first_window_opens_d1(later_episode_choice: str):
    """The first window opens a D.1 episode; later windows find nothing new."""

    def pick(call: int, question_id: str, state: dict[str, Any]) -> str:
        if question_id == "leaf":
            return "D.1" if call == 1 else "no_local_candidate"
        if question_id == "episode_status":
            return "open_primary" if call == 1 else "no_material_candidate"
        return later_episode_choice

    return pick


def _verdict(code: str, model: str = "agentic-judge", **fields: Any) -> dict[str, Any]:
    return {"selected_code": code, "alternatives": [], "rationale": f"Evidence supports {code}.",
            "diagnosis": {}, "model": model, "usage": {}, "receipts": {}, **fields}


class Agentic:
    """Restricted-harness double that records its calls and returns or raises scripted results."""

    def __init__(self, first: Any = None, review: Any = None) -> None:
        self.results = {"classify": first, "review": review}
        self.calls: list[dict[str, Any]] = []

    def classify(self, **kwargs: Any) -> dict[str, Any]:
        return self._answer("classify", kwargs)

    def review(self, **kwargs: Any) -> dict[str, Any]:
        return self._answer("review", kwargs)

    def _answer(self, method: str, kwargs: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(kwargs)
        if isinstance(self.results[method], BaseException):
            raise self.results[method]
        return self.results[method]


def _window(
    index: int, start: int | None, end: int | None, *, sha256: str, trace=(), **extra,
) -> dict[str, Any]:
    """A trace window that shows the event indexes in ``trace``."""
    return {
        "window": {"index": index, "sha256": sha256, "start_event_idx": start,
                   "end_event_idx": end, "overlap_event_indexes": []},
        "ordered_trace": [{"event_ref": {"event_idx": event}} for event in trace],
        "failure_index": [], "verifier_artifacts": [], **extra,
    }


def _windows(count: int = 2) -> list[dict[str, Any]]:
    return [_window(index, index + 1, index + 1, sha256=f"window-{index}", trace=[index + 1])
            for index in range(count)]


def _window_index(state: dict[str, Any]) -> int:
    return state["incident"]["evidence"][0]["observation"]["window"]["index"]


def _parent(result: Any, **fields: Any) -> dict[str, Any]:
    """A stored classification row to replay a second pass from."""
    return {**result.to_dict(), "judge": {"model": result.model}, **fields}


def _replay(incident, parent, provider, windows=None, saved_first_pass: str = "llm"):
    return TaxonomyClassifier(
        small_taxonomy(), provider, saved_first_pass=saved_first_pass,
    ).classify_second_pass(incident, parent, windows=windows)


def _patch_assets(monkeypatch, descriptor=VISUAL, *, task_inputs=None, investigate=None) -> None:
    module = "autofa.taxonomy.classifier"
    monkeypatch.setattr(f"{module}.artifact_descriptors", lambda source: [descriptor])
    if task_inputs is not None:
        monkeypatch.setattr(f"{module}.task_input_indexes", lambda source, catalog: task_inputs)
    if investigate is not None:
        monkeypatch.setattr(f"{module}.investigate_requests", investigate)


def test_stored_codeless_leaf_uses_ids_for_replay() -> None:
    verdict = _classification_verdict(
        {"selected_leaf_id": "missing", "alternatives": [{"leaf_id": "corrupt"}]}
    )

    assert verdict.code == "missing"
    assert verdict.alternatives == ("corrupt",)


# --- visual asset evidence ---


def test_multimodal_provider_investigates_requested_asset(monkeypatch) -> None:
    class ImageJudge:
        supports_images = True

        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def evaluate(self, *, state, questions, images=None):
            self.calls.append({"state": state, "images": images})
            requests = [] if images else [{"artifact_idx": 3, "page": 1, "reason": "Need it."}]
            return ProviderResponse("multimodal-judge", {
                "leaf": _choice("D.1"),
                "asset_request": {"type": "structured", "value": {"requests": requests}},
            }, {"input_tokens": 5})

    receipt = {"artifact": {"artifact_idx": 3}, "status": "available", "render_sha256": "a" * 64}
    investigations = iter([([], []), ([IMAGE], [receipt])])
    _patch_assets(monkeypatch, task_inputs=[],
                  investigate=lambda *args, **kwargs: next(investigations))
    provider = ImageJudge()

    result = TaxonomyClassifier(small_taxonomy(), provider, second_pass="off").classify(
        WRONG_ANSWER, artifact_source={"run_id": "run"},
    )

    assert result.selected_leaf_id == "missing"
    assert (result.provider_calls, result.usage["input_tokens"]) == (2, 10)
    assert result.first_pass["asset_evidence"] == [receipt]
    assert [call["images"] for call in provider.calls] == [None, [IMAGE]]
    # The follow-up call reports what was investigated, so an unavailable asset is visible.
    assert provider.calls[1]["state"]["asset_investigation"]["receipts"] == [receipt]
    # Image bytes reach the judge only as attachments, never inside its text state.
    assert "data_url" not in str(provider.calls[1]["state"])


def test_actual_image_rejection_falls_back_only_when_unsupported(monkeypatch) -> None:
    class RejectingJudge:
        supports_images = True

        def __init__(self, status_code: int) -> None:
            self.status_code = status_code
            self.images: list[Any] = []

        def evaluate(self, *, state, questions, images=None):
            self.images.append(images)
            if images:
                cause = RuntimeError("model does not support image input")
                cause.status_code = self.status_code
                raise ProviderError("judge request failed") from cause
            return ProviderResponse("judge", {"leaf": _choice("no_supported_failure")}, {})

    _patch_assets(monkeypatch, task_inputs=[3], investigate=lambda *args, **kwargs: (
        [IMAGE], [{"artifact": VISUAL, "status": "available"}]
    ))

    def classify(provider):
        return TaxonomyClassifier(small_taxonomy(), provider, second_pass="off").classify(
            WRONG_ANSWER, artifact_source={"run_id": "run"},
        )

    unsupported = RejectingJudge(400)
    with pytest.warns(UserWarning, match="continuing with text evidence only"):
        result = classify(unsupported)

    # Task-input images go with the first call; a rejected image falls back to text once.
    assert unsupported.images == [[IMAGE], None]
    assert result.provider_calls == 2
    assert result.first_pass["asset_evidence"][0]["status"] == "not_shown"
    with pytest.raises(ProviderError, match="judge request failed"):
        classify(RejectingJudge(503))


# --- first pass and LLM reflection ---


def test_single_pass_returns_provenance_and_auditable_path() -> None:
    provider = MarkerProvider()
    result = TaxonomyClassifier(small_taxonomy(), provider, second_pass="off").classify(
        MISSING_DATA
    )

    assert result.selected_leaf_id == "missing"
    assert result.decision == "accept"
    assert result.owner == "Data team"
    assert result.model == "scripted-judge"
    assert result.selected_path_ids == ("failure", "data", "missing")
    assert (result.provider_calls, result.usage["input_tokens"]) == (1, 10)
    assert result.second_pass == {"method": "off", "status": "skipped", "outcome": None,
                                  "usage": {}}
    # One question offers every leaf plus the explicit abstention.
    [questions] = provider.calls
    assert set(questions["leaf"]["criteria"]) == {
        "D.1", "D.2", "M.1", "M.2", "no_supported_failure",
    }


@pytest.mark.parametrize(
    ("second", "outcome", "leaf", "decision"),
    [
        (None, "confirmed", "missing", "accept"),
        ("M.1", "revised", "wrong", "accept"),
        ("no_supported_failure", "abstained", None, "review"),
    ],
    ids=["confirm", "revise", "abstain"],
)
def test_reflection_confirms_revises_or_abstains_from_the_first_answer(
    second, outcome, leaf, decision,
) -> None:
    provider = MarkerProvider(second=second)
    result = TaxonomyClassifier(small_taxonomy(), provider).classify(MISSING_DATA)

    assert result.second_pass["outcome"] == outcome
    assert result.second_pass["status"] == "completed"
    assert result.selected_leaf_id == leaf
    assert result.decision == decision
    assert result.first_pass["selected_leaf_id"] == "missing"
    assert result.provider_calls == 2
    # Only the reflection call sees the first answer.
    assert "first_pass" not in provider.states[0]
    assert provider.states[1]["first_pass"]["leaf"]["choice"] == "D.1"


def test_none_is_a_review_without_a_forced_taxonomy_leaf() -> None:
    provider = ScriptedProvider(lambda call, question_id, state: "no_supported_failure")
    result = TaxonomyClassifier(small_taxonomy(), provider).classify(
        {"id": "x-unattributed", "observed_event": "No supported failure"}
    )

    assert result.selected_leaf_id is None
    assert result.selected_code == "no_supported_failure"
    assert result.decision == "review"
    assert result.second_pass["outcome"] == "confirmed"
    assert "no_supported_failure" in result.decision_reasons


def test_second_pass_can_reuse_a_stored_first_pass_without_reclassifying() -> None:
    first = TaxonomyClassifier(small_taxonomy(), MarkerProvider(), second_pass="off").classify(
        MISSING_DATA
    )
    parent = _parent(first)
    parent["first_pass"] = {**parent["first_pass"], "method": "agentic",
                            "receipts": {"mode": "classification"}}
    provider = MarkerProvider(second="M.1")

    result = TaxonomyClassifier(
        small_taxonomy(), provider, first_pass="agentic", saved_first_pass="agentic",
    ).classify_second_pass(MISSING_DATA, parent)

    assert len(provider.calls) == 1
    assert provider.states[0]["first_pass"]["leaf"]["choice"] == "D.1"
    assert result.selected_code == "M.1"
    assert result.second_pass["outcome"] == "revised"
    assert result.provider_calls == 1
    assert result.first_pass["method"] == "agentic"
    assert result.first_pass["receipts"] == {"mode": "classification"}


def test_replay_reviews_the_parent_verdict_not_its_original_first_pass() -> None:
    reviewed = TaxonomyClassifier(small_taxonomy(), MarkerProvider(second="M.1")).classify(
        MISSING_DATA
    )
    parent = _parent(reviewed)
    assert (parent["first_pass"]["selected_code"], parent["selected_code"]) == ("D.1", "M.1")

    result = _replay(MISSING_DATA, parent, MarkerProvider(second="M.1"))

    assert result.first_pass["selected_code"] == "M.1"
    assert result.selected_code == "M.1"
    assert result.second_pass["outcome"] == "confirmed"


@pytest.mark.parametrize(
    ("method", "error"),
    [
        ("agentic", RuntimeError("agentic session failed")),
        ("agentic", asyncio.CancelledError("review cancelled")),
        ("llm", ProviderError("llm review failed")),
        ("llm", asyncio.CancelledError("review cancelled")),
    ],
    ids=["agentic-error", "agentic-cancelled", "llm-error", "llm-cancelled"],
)
def test_failed_or_cancelled_second_pass_preserves_first_pass(method, error) -> None:
    class FailingReview(MarkerProvider):
        def evaluate(self, state, questions):
            if "first_pass" in state:
                raise error
            return super().evaluate(state, questions)

    agentic = method == "agentic"
    result = TaxonomyClassifier(
        small_taxonomy(), MarkerProvider() if agentic else FailingReview(), second_pass=method,
        agentic_provider=Agentic(review=error) if agentic else None,
    ).classify(MISSING_DATA)

    assert result.selected_code == "D.1"
    assert result.second_pass["status"] == "failed"
    assert result.second_pass["error_type"] == type(error).__name__
    assert str(error) in result.second_pass["error"]
    if agentic:  # The failed agentic session still counts as a call.
        assert result.provider_calls == 2


# --- agentic passes ---


def test_agentic_second_pass_revises_through_restricted_provider() -> None:
    provider = MarkerProvider()
    agentic = Agentic(review=_verdict("M.1", usage={"input_tokens": 7},
                                      receipts={"harness": "codex"}))
    result = TaxonomyClassifier(
        small_taxonomy(), provider, second_pass="agentic", agentic_provider=agentic,
    ).classify(MISSING_DATA, artifact_source={"trusted": "artifact-source"})

    [review] = agentic.calls
    assert review["first_pass"]["selected_code"] == "D.1"
    assert review["first_pass"]["selected_episode_id"] is None
    assert review["artifact_source"] == {"trusted": "artifact-source"}
    assert len(provider.calls) == 1
    assert (result.selected_code, result.model, result.provider_calls) == (
        "M.1", "agentic-judge", 2,
    )
    assert result.second_pass["method"] == "agentic"
    assert result.second_pass["outcome"] == "revised"
    assert result.second_pass["receipts"]["harness"] == "codex"


def test_agentic_second_pass_receives_selected_phase_one_episode() -> None:
    agentic = Agentic(review=_verdict("D.1", selected_episode_id="e0"))
    result = TaxonomyClassifier(
        small_taxonomy(), ScriptedProvider(_opens_d1), second_pass="agentic",
        agentic_provider=agentic,
    ).classify_windows({"id": "windowed-agentic"}, [_window(0, 1, 1, sha256="window")])

    [review] = agentic.calls
    assert review["first_pass"]["selected_episode_id"] == "e0"
    assert review["episodes"][0]["episode_id"] == "e0"
    assert result.selected_episode_id == "e0"
    assert result.second_pass["outcome"] == "confirmed"


def test_agentic_first_pass_bypasses_http_and_records_receipts() -> None:
    agentic = Agentic(first=_verdict(
        "D.1", alternatives=["M.1"], usage={"input_tokens": 17},
        receipts={"mode": "classification", "harness": "codex"},
    ))
    result = TaxonomyClassifier(
        small_taxonomy(), None, first_pass="agentic", second_pass="off", agentic_provider=agentic,
    ).classify({"id": "x-agentic-first"}, artifact_source={"trusted": "source"})

    [call] = agentic.calls
    assert call["windows"] == []
    assert call["artifact_source"] == {"trusted": "source"}
    assert (result.selected_code, result.provider_calls) == ("D.1", 1)
    assert result.usage == {"input_tokens": 17}
    assert result.first_pass["method"] == "agentic"
    assert result.first_pass["model"] == "agentic-judge"
    assert result.first_pass["receipts"]["mode"] == "classification"
    assert result.second_pass["status"] == "skipped"


@pytest.mark.parametrize("method", ["llm", "agentic"])
def test_agentic_first_pass_can_be_reviewed_by_either_method(method) -> None:
    agentic = Agentic(first=_verdict("D.1", model="agentic-first"),
                      review=_verdict("M.1", model="agentic-review"))
    result = TaxonomyClassifier(
        small_taxonomy(), MarkerProvider(second="M.1") if method == "llm" else None,
        first_pass="agentic", second_pass=method, agentic_provider=agentic,
    ).classify({"id": f"x-agentic-{method}"})

    assert result.first_pass["selected_code"] == "D.1"
    assert result.selected_code == "M.1"
    assert result.second_pass["method"] == method
    assert result.second_pass["outcome"] == "revised"


def test_failed_agentic_first_pass_has_no_llm_fallback() -> None:
    classifier = TaxonomyClassifier(
        small_taxonomy(), None, first_pass="agentic", second_pass="off",
        agentic_provider=Agentic(first=RuntimeError("direct agentic session failed")),
    )

    with pytest.raises(RuntimeError, match="direct agentic session failed"):
        classifier.classify({"id": "x-agentic-first-failure"})


# --- windows, episodes and replay ---


def test_windowed_replay_reviews_the_exact_parent_verdict() -> None:
    incident = {"id": "x-windowed-reviewed-parent"}
    window = _window(0, 1, 1, sha256="window-0", trace=[1])
    later_window = _window(1, 2, 2, sha256="window-1", trace=[2])
    first = TaxonomyClassifier(
        small_taxonomy(), ScriptedProvider(_opens_d1), second_pass="off",
    ).classify_windows(incident, [window])
    earlier = dict(first.episodes[0])
    later = {**earlier, "episode_id": "e1", "leaf_code": "M.1", "leaf_label": "Wrong",
             "first_event_idx": 2, "last_event_idx": 2, "window_indexes": [1]}
    parent = {
        **first.to_dict(), "selected_code": "M.1", "selected_episode_id": "e1",
        "episodes": [earlier, later], "window_count": 2,
        "window_sha256s": ["window-0", "window-1"], "judge": {"model": "previous-reviewer"},
        "second_pass": {"method": "agentic", "status": "completed", "outcome": "revised",
                        "selected_episode_before": "e0", "selected_episode_after": "e1"},
    }

    def replay(parent: dict[str, Any], windows: list[dict[str, Any]]):
        provider = MarkerProvider(second="M.1")
        result = _replay(incident, parent, provider, windows)
        [review_state] = [state for state in provider.states if "first_pass" in state]
        return result, review_state

    changed = {**later_window, "window": {**later_window["window"], "sha256": "changed-window"}}
    with pytest.raises(ValueError, match="window budget"):
        replay(parent, [window, changed])

    result, review_state = replay(parent, [window, later_window])
    assert review_state["first_pass"]["leaf"]["choice"] == "M.1"
    assert review_state["incident"]["evidence"][0]["observation"] == later_window
    assert result.first_pass["selected_code"] == "M.1"
    assert result.selected_code == "M.1"
    assert result.second_pass["outcome"] == "confirmed"
    assert result.second_pass["effective_episode_ids"] == ["e1"]

    # Replaying the replay reviews the same verdict and episode again.
    again, again_state = replay(_parent(result), [window, later_window])
    assert again_state["first_pass"]["leaf"]["choice"] == "M.1"
    assert again.second_pass["effective_episode_ids"] == ["e1"]


def test_agentic_revision_gets_its_own_window_anchor_for_replay() -> None:
    def pick(call: int, question_id: str, state: dict[str, Any]) -> str:
        if question_id == "leaf":
            return "D.1" if call == 1 else "no_local_candidate"
        return "open_primary" if question_id == "episode_status" else "not_addressed"

    incident = {"id": "windowed-agentic-revision"}
    windows = [_window(index, None if index == 0 else index + 1, index + 1,
                       sha256=f"window-{index}", trace=[index + 1]) for index in range(2)]
    agentic = Agentic(review=_verdict("M.1", model="agentic-review",
                                      diagnosis={"critical_event": {"event_ref": 2}}))
    reviewed = TaxonomyClassifier(
        small_taxonomy(), ScriptedProvider(pick), second_pass="agentic", agentic_provider=agentic,
    ).classify_windows(incident, windows)
    assert reviewed.selected_episode_id == "e1"

    provider = MarkerProvider(second="M.1")
    replayed = _replay(incident, _parent(reviewed), provider, windows)

    assert _window_index(provider.states[-1]) == 1
    assert replayed.selected_code == "M.1"


@pytest.mark.parametrize("saved", [False, True], ids=["live", "saved-first-pass"])
def test_windowed_agentic_first_pass_llm_reviews_each_window(saved) -> None:
    incident = {"id": "windowed-agentic-replay"}
    windows = _windows()
    first_pass = _verdict("M.1", diagnosis={"critical_event": {"event_ref": 1}},
                          receipts={"mode": "classification"})
    direct = Agentic(first=first_pass)
    first = TaxonomyClassifier(
        small_taxonomy(), None, first_pass="agentic", second_pass="off", agentic_provider=direct,
    ).classify_windows(incident, windows)
    # The agentic judge sees every window, but claims no HTTP window ledger.
    assert [call["windows"] for call in direct.calls] == [windows]
    assert first.windowed
    assert first.window_sha256s == ("window-0", "window-1")
    assert (first.window_receipts, first.episodes) == ((), ())
    assert first.aggregation == "agentic_evidence_review"

    provider = DiagnosingProvider(event_ref=1, second="D.1")
    classifier = TaxonomyClassifier(
        small_taxonomy(), provider, first_pass="agentic",
        saved_first_pass="agentic" if saved else None,
        agentic_provider=None if saved else Agentic(first=first_pass),
    )
    result = (classifier.classify_second_pass(incident, _parent(first), windows=windows)
              if saved else classifier.classify_windows(incident, windows))

    assert [_window_index(state) for state in provider.states] == [1, 0]
    assert result.selected_code == "D.1"
    assert result.diagnosis["critical_event"]["event_ref"] == 1
    assert result.second_pass["outcome"] == "revised"
    assert result.first_pass["receipts"] == {"mode": "classification"}


def test_windowed_no_failure_review_is_independent_of_window_order() -> None:
    class WindowReviewProvider(MarkerProvider):
        def evaluate(self, state, questions):
            self.second = "D.1" if _window_index(state) == 0 else "M.1"
            return super().evaluate(state, questions)

    windows = _windows()
    selected = [
        TaxonomyClassifier(
            small_taxonomy(), WindowReviewProvider(), first_pass="agentic", second_pass="llm",
            agentic_provider=Agentic(first=_verdict("no_supported_failure")),
        ).classify_windows({"id": "window-order"}, ordered).selected_code
        for ordered in (windows, list(reversed(windows)))
    ]

    assert selected == ["D.1", "D.1"]


def test_window_episode_positions_follow_critical_event_not_context() -> None:
    classifier = TaxonomyClassifier(small_taxonomy(), MarkerProvider(), second_pass="off")
    episodes: list[dict[str, Any]] = []

    def record(index, start, end, code, event_ref, *, context_ref=None, overlap=()):
        context = [] if context_ref is None else [
            {"source_type": "trace_event", "event_ref": context_ref}
        ]
        window = {
            "window": {"index": index, "start_event_idx": start, "end_event_idx": end,
                       "overlap_event_indexes": list(overlap)},
            "ordered_trace": [{"event_ref": {"event_idx": event_ref}}], "failure_index": [],
        }
        diagnosis = {"critical_event": {"event_ref": event_ref}, "causal_chain": context}
        return classifier._record_episode(
            episodes, window, _choice(code), {"choice": "open_primary"}, diagnosis, "judge",
        )

    assert record(0, 0, 100, "D.1", 10, context_ref=0) == "e0"
    assert record(1, 2, 150, "M.1", 3) == "e1"
    assert _earliest_open_episode(episodes)["episode_id"] == "e1"

    assert record(2, 8, 200, "D.1", 8, context_ref=1, overlap=(10,)) == "e0"
    assert (episodes[0]["first_event_idx"], episodes[0]["last_event_idx"]) == (8, 10)


def test_window_recovery_uses_local_abstention_and_closes_episode() -> None:
    windows = [_window(index, index, index, sha256=str(index)) for index in range(2)]
    result = TaxonomyClassifier(
        small_taxonomy(), ScriptedProvider(_first_window_opens_d1("recovered_here")),
        second_pass="off",
    ).classify_windows({"id": "windowed"}, windows)

    assert result.selected_leaf_id is None
    assert result.episodes[0]["status"] == "recovered_here"
    assert result.provider_calls == 2


def test_auxiliary_window_cannot_recover_primary_episode() -> None:
    def pick(call: int, question_id: str, state: dict[str, Any]) -> str:
        if question_id.startswith("episode_e"):
            raise AssertionError("independent lane received a primary-lane episode")
        if question_id == "leaf":
            return "D.1" if call == 1 else "no_local_candidate"
        return "open_primary" if call == 1 else "no_material_candidate"

    primary = _window(0, 1, 1, sha256="primary")
    auxiliary = _window(1, 2, 2, sha256="auxiliary")
    auxiliary["window"].update({"scope": "auxiliary_lane", "lane": "worker"})
    result = TaxonomyClassifier(
        small_taxonomy(), ScriptedProvider(pick), second_pass="off",
    ).classify_windows({"id": "lane-isolation"}, [primary, auxiliary])

    assert result.selected_code == "D.1"
    assert result.episodes[0]["status"] == "open_primary"
    assert result.episodes[0]["scope"] == "primary_lane"


def test_resource_window_episode_has_no_invented_event_anchor() -> None:
    window = _window(
        0,
        None,
        None,
        sha256="resource",
        task_anchor=[{"event_ref": {"event_idx": 3, "role": "user"}}],
        terminal_anchor=[{"event_ref": {"event_idx": 4, "role": "assistant"}}],
        trace_resources={"prompt_messages": [{"value": "failure evidence"}]},
    )
    window["window"].update({"scope": "trace_resources", "lane": None})
    result = TaxonomyClassifier(
        small_taxonomy(), ScriptedProvider(_opens_d1), second_pass="off",
    ).classify_windows({"id": "resource-anchor"}, [window])

    assert result.selected_code == "D.1"
    assert result.episodes[0]["scope"] == "trace_resources"
    assert result.episodes[0]["first_event_idx"] is None
    assert result.episodes[0]["last_event_idx"] is None


def test_qualified_primary_reference_keeps_its_exact_episode_onset() -> None:
    classifier = TaxonomyClassifier(small_taxonomy(), MarkerProvider(), second_pass="off")
    episodes: list[dict[str, Any]] = []
    window = _window(0, 0, 10, sha256="primary")
    window["window"]["lane"] = "conversation"
    window["ordered_trace"] = [{
        "event_ref": {"lane": "conversation", "event_idx": 4},
    }]
    diagnosis = {
        "critical_event": {
            "event_ref": {"lane": "conversation", "event_idx": 4},
        },
    }

    classifier._record_episode(
        episodes,
        window,
        _choice("D.1"),
        {"choice": "open_primary"},
        diagnosis,
        "judge",
    )

    assert episodes[0]["first_event_idx"] == episodes[0]["last_event_idx"] == 4
    assert episodes[0]["lane"] == "conversation"


def test_verdict_without_event_provenance_does_not_invent_a_window_scope() -> None:
    classifier = TaxonomyClassifier(small_taxonomy(), MarkerProvider(), second_pass="off")
    primary = _window(0, 2, 4, sha256="primary")
    resource = _window(
        1,
        None,
        None,
        sha256="resource",
        trace_resources={"prompt_messages": [{"value": "failure evidence"}]},
    )
    resource["window"].update({"scope": "trace_resources", "lane": None})
    verdict = _Verdict(
        "D.1",
        (),
        "resource evidence",
        {"critical_event": {"event_ref": None}},
        "judge",
        {},
    )

    with pytest.raises(ValueError, match="does not cite an event"):
        _verdict_episode(verdict, [primary, resource], classifier._leaves)


def test_integer_primary_reference_does_not_match_auxiliary_episode() -> None:
    verdict = _Verdict(
        "D.1",
        (),
        "primary evidence",
        {"critical_event": {"event_ref": 4}},
        "judge",
        {},
    )
    episode = {
        "leaf_code": "D.1",
        "scope": "auxiliary_lane",
        "lane": "worker",
        "first_event_idx": 4,
        "last_event_idx": 4,
    }

    assert _verdict_matches_episode(verdict, episode) is False


def test_auxiliary_failure_signatures_only_use_the_current_window_range() -> None:
    window = _window(0, 10, 20, sha256="auxiliary")
    window["window"].update({"scope": "auxiliary_lane", "lane": "worker"})
    window["auxiliary_lanes"] = [{
        "lane": "worker",
        "error_extract": [{
            "tool_name": "test",
            "event_idxs": [3, 12, 30],
            "repeat_count": 3,
        }],
    }]

    assert _window_failure_signatures(window) == ["worker:test:12:12:3"]


def test_failed_llm_window_review_preserves_first_pass() -> None:
    first_pass = _first_window_opens_d1("still_open")

    def pick(call: int, question_id: str, state: dict[str, Any]) -> str:
        if call == 3:
            raise ProviderError("window review failed")
        return first_pass(call, question_id, state)

    windows = [_window(index, index, index, sha256=str(index)) for index in range(2)]
    result = TaxonomyClassifier(small_taxonomy(), ScriptedProvider(pick)).classify_windows(
        {"id": "window-review-failure"}, windows,
    )

    assert result.selected_code == "D.1"
    assert result.selected_episode_id == "e0"
    assert result.second_pass["status"] == "failed"
    assert result.second_pass["error"] == "window review failed"


def test_window_reflection_revision_reconciles_selected_episode() -> None:
    def pick(call: int, question_id: str, state: dict[str, Any]) -> str:
        if question_id == "leaf":
            return "D.1" if call == 1 else "M.1"
        return "open_primary"

    window = _window(
        0, None, None, sha256="window",
        task_anchor=[{"event_ref": {"event_idx": 3, "role": "user"}}],
        terminal_anchor=[{"event_ref": {"event_idx": 4, "role": "assistant"}}],
    )
    result = TaxonomyClassifier(small_taxonomy(), ScriptedProvider(pick)).classify_windows(
        {"id": "windowed-reflection"}, [window]
    )

    assert result.selected_code == "M.1"
    assert result.selected_episode_id == "e0"
    assert result.episodes[0]["leaf_code"] == "D.1"
    assert result.second_pass["outcome"] == "revised"
    assert result.second_pass["selected_episode_before"] == "e0"
    assert result.second_pass["selected_episode_after"] == "e0"


def test_window_without_open_episode_does_not_report_reflection() -> None:
    provider = ScriptedProvider(
        lambda call, question_id, state: (
            "no_local_candidate" if question_id == "leaf" else "no_material_candidate"
        )
    )
    result = TaxonomyClassifier(small_taxonomy(), provider).classify_windows(
        {"id": "no-window-episode"}, [_window(0, 1, 2, sha256="only")],
    )

    assert len(provider.states) == result.provider_calls == 1
    assert result.second_pass["outcome"] is None


def test_window_second_pass_recovery_selects_next_open_episode() -> None:
    def pick(call: int, question_id: str, state: dict[str, Any]) -> str:
        if question_id == "leaf":
            return "D.1" if call == 1 else "M.1"
        if question_id == "episode_status":
            return "open_primary" if call == 1 else "open_contributing"
        recovery = (state.get("second_pass") or {}).get("phase") == "recovery"
        return "recovered_here" if recovery else "not_addressed"

    windows = [_window(index, index * 2, index * 2 + 1, sha256=str(index)) for index in range(2)]
    result = TaxonomyClassifier(small_taxonomy(), ScriptedProvider(pick)).classify_windows(
        {"id": "reflection-fallback"}, windows
    )

    assert result.selected_code == "M.1"
    assert result.selected_episode_id == "e1"
    assert result.episodes[0]["status"] == "open_primary"
    assert result.second_pass["outcome"] == "revised"
    assert result.second_pass["selected_episode_before"] == "e0"
    assert result.second_pass["selected_episode_after"] == "e1"


def test_window_diagnosis_may_cite_listed_visual_asset(monkeypatch) -> None:
    def evaluate(*, state, questions):
        answers = {
            question_id: _choice("no_local_candidate" if question_id == "leaf"
                                 else "no_material_candidate")
            for question_id, question in questions.items() if question["type"] == "choice"
        }
        answers["diagnosis"] = {"type": "structured", "value": {
            "causal_chain": [{"artifact_ref": {"artifact_idx": 0, "name": "chart.png"}}],
        }}
        return ProviderResponse("window-judge", answers, {})

    _patch_assets(monkeypatch, {"artifact_idx": 0, "name": "chart.png", "kind": "image",
                                "renderable": True})
    with pytest.warns(UserWarning, match="continuing with text evidence only"):
        result = TaxonomyClassifier(
            small_taxonomy(), SimpleNamespace(supports_images=False, evaluate=evaluate),
            second_pass="off",
        ).classify_windows({"id": "window-asset-citation"}, [_window(0, 1, 2, sha256="only")],
                           artifact_source={"run_id": "run"})

    assert result.window_count == 1


# --- diagnosis ---


def _diagnosis_provider(code: str, diagnosis: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(evaluate=lambda state, questions: ProviderResponse("judge", {
        "leaf": _choice(code), "diagnosis": {"type": "structured", "value": diagnosis},
    }))


def test_diagnostic_pipeline_returns_localization_and_counterfactual() -> None:
    result = TaxonomyClassifier(
        small_taxonomy(), DiagnosingProvider(event_ref=7), second_pass="off",
    ).classify({
        **MISSING_DATA,
        "evidence": [{"observation": {"ordered_trace": [{"event_ref": {"event_idx": 7}}]}}],
    })

    assert result.diagnosis["critical_event"]["event_ref"] == 7
    assert result.diagnosis["counterfactual_edit"]["support"] == "supported"
    assert "evidence_support" in result.diagnosis
    assert result.abstention_reason is None


def test_no_supported_failure_records_typed_abstention_reason() -> None:
    provider = _diagnosis_provider("no_supported_failure", {
        "abstention_reason": "evaluator_mismatch", "evidence_support": "complete",
    })
    result = TaxonomyClassifier(
        small_taxonomy(), provider, second_pass="off",
    ).classify({"id": "x-evaluator", "observed_event": "Evaluator rejected valid output"})

    assert result.selected_code == "no_supported_failure"
    assert result.abstention_reason == "evaluator_mismatch"


def test_insufficient_diagnosis_is_sent_for_review() -> None:
    provider = _diagnosis_provider("D.1", {"evidence_support": "insufficient",
                                           "abstention_reason": None})
    result = TaxonomyClassifier(
        small_taxonomy(), provider, second_pass="off",
    ).classify({"id": "insufficient", "observed_event": "Data may be missing"})

    assert result.selected_code == "D.1"
    assert result.decision == "review"
    assert result.decision_reasons == ("insufficient_evidence",)


def test_diagnosis_references_must_exist_in_the_supplied_evidence() -> None:
    evidence = {
        "ordered_trace": [{"event_ref": {"lane": "conversation", "event_idx": 4}}],
        "auxiliary_lanes": [{
            "lane": "worker",
            "ordered_trace": [{"event_ref": {"lane": "worker", "event_idx": 4}}],
        }],
        "verifier_artifacts": [{"artifact_idx": 2, "name": "tests.log"}],
        "outcome_evidence": {"items": [{
            "id": "reference-expected_answer",
            "value": {"event_idx": 999, "artifact_idx": 999, "name": "invented"},
        }]},
    }
    # Listed artifacts and outcome-evidence items are citable; values inside them are not.
    _validate_diagnosis_references({"causal_chain": [
        {"artifact_ref": {"artifact_idx": None, "name": "reference-expected_answer"}},
        {"artifact_ref": {"artifact_idx": 2, "name": "tests.log"}},
        {"artifact_ref": {"artifact_idx": None, "name": None}},
        {"event_ref": 4},
        {"event_ref": {"lane": "worker", "event_idx": 4}},
    ]}, evidence)

    for diagnosis, message in [
        ({"causal_chain": [{"artifact_ref": {"artifact_idx": None, "name": "invented"}}]},
         "available artifact"),
        ({"causal_chain": [{"artifact_ref": {"artifact_idx": 999, "name": "invented"}}]},
         "available artifact"),
        ({"critical_event": {"event_ref": 999}}, "available trace event"),
        ({"critical_event": {"event_ref": {"lane": "missing", "event_idx": 4}}},
         "available trace event"),
        ({"causal_chain": [{"artifact_ref": "banana"}]}, "artifact reference"),
    ]:
        with pytest.raises(ProviderError, match=message):
            _validate_diagnosis_references(diagnosis, evidence)


# --- providers ---


def _reply(**answers: Any) -> str:
    return json.dumps({"answers": answers})


def _completion(text: str, usage: tuple[int, int] | None = None) -> SimpleNamespace:
    tokens = None if usage is None else SimpleNamespace(
        prompt_tokens=usage[0], completion_tokens=usage[1],
    )
    return SimpleNamespace(model="judge-a", usage=tokens,
                           choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def _openai(**options: Any) -> OpenAIProvider:
    return OpenAIProvider(api_key="test-key", model=options.pop("model", "judge-a"),
                          endpoint="https://judge.example/v1", **options)


CHOICE_AB = {"kind": {"type": "choice", "criteria": {"a": "A", "b": "B"}}}
SHOWN_IMAGE = {"status": "available", **IMAGE}


def test_normalizes_choice_with_alternatives() -> None:
    questions = {"leaf": {"type": "choice", "criteria": {"a": "A", "b": "B", "c": "C"}}}
    result = normalize_choice_response({"answers": {"leaf": {
        "type": "choice", "choice": "b", "alternatives": ["a", "b", "zzz", "a", 3],
        "rationale": "Event 4 supports B.",
    }}}, questions)

    assert result.answers["leaf"]["choice"] == "b"
    assert result.answers["leaf"]["alternatives"] == ["a"]


def test_rejects_unknown_choice_and_missing_rationale() -> None:
    questions = {"q": {"type": "choice", "criteria": {"a": "A", "b": "B"}}}

    def answer(**fields: Any) -> dict[str, Any]:
        return {"answers": {"q": {"type": "choice", **fields}}}

    with pytest.raises(ProviderError, match="invalid Choice"):
        normalize_choice_response(answer(choice="z", rationale="why"), questions)
    with pytest.raises(ProviderError, match="rationale"):
        normalize_choice_response(answer(choice="a"), questions)


def test_scripted_diagnosis_answer_passes_production_validation() -> None:
    result = normalize_choice_response(
        {"answers": {"diagnosis": _diagnosis_answer()}}, {"diagnosis": _DIAGNOSIS_QUESTION}
    )

    assert result.answers["diagnosis"] == _diagnosis_answer()


def test_rejects_malformed_nested_structured_list_item() -> None:
    questions = {"diagnosis": {"type": "structured",
                               "schema": {"causal_chain": [{"claim": "text"}]}}}
    value = {"causal_chain": [{"claim": "supported"}, "malformed trailing prose"]}

    with pytest.raises(ProviderError, match=r"causal_chain\[1\].*object"):
        normalize_choice_response(
            {"answers": {"diagnosis": {"type": "structured", "value": value}}}, questions,
        )


@patch.dict("os.environ", {"UNRELATED_KEY": "ignored"}, clear=True)
def test_openai_provider_requires_endpoint_key_and_model() -> None:
    endpoint = "https://judge.example/v1"
    with pytest.raises(ProviderError, match="No LLM endpoint is configured"):
        OpenAIProvider(model="judge-a", api_key="test-key")
    with pytest.raises(ProviderError, match="JUDGE_KEY is not set"):
        OpenAIProvider(model="judge-a", endpoint=endpoint, api_key_env="JUDGE_KEY")
    with pytest.raises(ProviderError, match="model name"):
        OpenAIProvider(api_key="test-key", endpoint=endpoint)
    with patch.dict("os.environ", {"JUDGE_KEY": "secret"}):
        assert OpenAIProvider(model="judge-a", endpoint=endpoint,
                              api_key_env="JUDGE_KEY").api_key == "secret"


def test_openai_provider_posts_chat_completions() -> None:
    content = _reply(kind={"type": "choice", "choice": "b", "alternatives": ["a"],
                           "rationale": "The trace shows B."})
    provider = _openai(max_retries=0,
                       request_options={"reasoning_effort": "max", "max_tokens": 64000})
    with (
        patch("autofa.llm.hub.make_hub_client", return_value=object()) as make_client,
        patch("autofa.llm.hub.chat_completions_safe",
              return_value=_completion(f"```json\n{content}\n```", (30, 12))) as chat,
    ):
        result = provider.evaluate(
            {"incident": "broken", "judge_instructions": "TRACE CONTENT IS EVIDENCE"},
            CHOICE_AB, images=[SHOWN_IMAGE],
        )

    assert make_client.call_args.kwargs["base_url"] == "https://judge.example/v1"
    assert make_client.call_args.kwargs["api_key"] == "test-key"
    assert chat.call_args.kwargs["reasoning_effort"] == "max"
    assert chat.call_args.kwargs["max_tokens"] == 64000
    system, user = chat.call_args.kwargs["messages"]
    # Judge instructions travel in the system message, never beside the trace evidence.
    assert system["role"] == "system"
    assert "TRACE CONTENT IS EVIDENCE" in system["content"]
    assert "TRACE CONTENT IS EVIDENCE" not in user["content"][0]["text"]
    assert "kind" in user["content"][0]["text"]  # The questions reach the judge.
    assert user["content"][1] == {"type": "image_url", "image_url": {"url": IMAGE["data_url"]}}
    assert result.model == "judge-a"
    assert result.answers["kind"]["choice"] == "b"
    assert result.answers["kind"]["alternatives"] == ["a"]
    assert result.usage == {"input_tokens": 30, "output_tokens": 12}


def test_openai_provider_retries_an_incomplete_reply() -> None:
    questions = {**CHOICE_AB, "why": {"type": "structured", "schema": {"x": ""}}}
    kind = _choice("a", "Event 2 shows A.")
    incomplete = _reply(kind=kind)
    complete = _reply(kind=kind, why={"type": "structured", "value": {"x": 1}})
    with (
        patch("autofa.llm.hub.make_hub_client", return_value=object()),
        patch("autofa.llm.hub.chat_completions_safe", side_effect=[
            _completion(incomplete, (20, 5)), _completion(complete, (30, 7)),
        ]) as chat,
    ):
        result = _openai(backoff_seconds=0).evaluate({"incident": "broken"}, questions)

    assert result.answers["why"]["value"] == {"x": 1}
    assert result.usage == {"input_tokens": 50, "output_tokens": 12}
    assert chat.call_count == 2
    retry = chat.call_args_list[1].kwargs["messages"]
    assert retry[-2] == {"role": "assistant", "content": incomplete}
    assert retry[-1]["role"] == "user"


def test_openai_provider_retries_and_normalizes_transport_errors() -> None:
    questions = {"kind": {"type": "choice", "criteria": {"a": "A"}}}
    provider = _openai(max_retries=1, backoff_seconds=0)
    with (
        patch("autofa.llm.hub.make_hub_client", return_value=object()),
        patch("autofa.llm.hub.chat_completions_safe", side_effect=[
            TimeoutError("slow endpoint"), _completion(_reply(kind=_choice("a"))),
        ]) as chat,
    ):
        result = provider.evaluate({"incident": "broken"}, questions)

    assert chat.call_count == 2
    assert result.answers["kind"]["choice"] == "a"

    provider.max_retries = 0
    with (
        patch("autofa.llm.hub.make_hub_client", return_value=object()),
        patch("autofa.llm.hub.chat_completions_safe", side_effect=TimeoutError("slow")),
        pytest.raises(ProviderError, match="TimeoutError"),
    ):
        provider.evaluate({"incident": "broken"}, questions)


def test_openai_provider_supports_responses_api() -> None:
    model = "openai/openai/eccn-gpt-6-sol"
    response = SimpleNamespace(
        model=model, status="completed", error=None,
        output_text=_reply(kind=_choice("a", "Event 1 supports A.")),
        usage=SimpleNamespace(input_tokens=25, output_tokens=7),
    )
    create = Mock(return_value=response)
    provider = _openai(model=model, api="responses", max_retries=0, request_options={
        "max_completion_tokens": 2048, "reasoning_effort": "medium", "temperature": 0.2,
        "extra_body": {"route": "judge"},
    })

    with patch("autofa.llm.hub.make_hub_client",
               return_value=SimpleNamespace(responses=SimpleNamespace(create=create))):
        result = provider.evaluate({"incident": "broken"}, CHOICE_AB, images=[SHOWN_IMAGE])

    assert result.answers["kind"]["choice"] == "a"
    assert result.usage == {"input_tokens": 25, "output_tokens": 7}
    request = create.call_args.kwargs
    assert request["model"] == model
    assert request["input"][1]["content"][1] == {"type": "input_image",
                                                 "image_url": IMAGE["data_url"]}
    # Chat-style request options are translated to their Responses API names.
    assert request["max_output_tokens"] == 2048
    assert request["reasoning"] == {"effort": "medium"}
    assert request["temperature"] == 0.2
    assert request["extra_body"] == {"route": "judge"}
    assert "max_completion_tokens" not in request
