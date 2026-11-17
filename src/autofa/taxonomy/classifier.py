# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Taxonomy classification of one incident or ordered trace windows."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import warnings
from collections import Counter
from dataclasses import dataclass
from typing import Any

from .assets import (
    ASSET_INVESTIGATION_PROTOCOL,
    MAX_ASSET_ROUNDS,
    AssetEvidenceError,
    artifact_descriptors,
    asset_request_question,
    investigate_requests,
    task_input_indexes,
)
from .judge_guidance import judge_instructions
from .models import AnalysisResult, FirstPassMethod, ProviderResponse, SecondPassMethod
from .providers import DecisionProvider, ProviderError, image_input_is_unsupported
from .schema import Taxonomy, TaxonomyNode
from .trace_digest import serialized_characters

PROTOCOL_VERSION = "autofa-interaction-evidence-v2"
NO_FAILURE_CODE = "no_supported_failure"
ABSTENTION_REASONS = (
    "insufficient_evidence",
    "missing_observability",
    "evaluator_mismatch",
    "non_model_failure",
    "taxonomy_gap",
    "unknown",
)
# A window may add no candidate even when another window supports the global failure.
NO_LOCAL_CANDIDATE = "no_local_candidate"
# Ordered by selection priority.
_OPEN_STATUSES = ("open_primary", "open_contributing")
_LEDGER_FIELDS = (
    "episode_id",
    "leaf_code",
    "leaf_label",
    "first_event_idx",
    "last_event_idx",
    "scope",
    "lane",
    "status",
    "failure_signatures",
)

_LEAF_INSTRUCTIONS = (
    "Which category is the best-supported primary causal failure? Select exactly "
    "one category. If no supplied category has affirmative support, select "
    f"'{NO_FAILURE_CODE}'. Inspect event indexes, original argument types, "
    "tool errors, and verifier lines before deciding."
)
_REFLECTION_INSTRUCTIONS = (
    "This is a second look. Your first-pass answer is in the state. Review the trace again and "
    "either confirm that category or revise it if the evidence supports a different one. If no "
    f"supplied category has affirmative support, select '{NO_FAILURE_CODE}'."
)
_NO_FAILURE_CRITERION = (
    "The observed outcome is unsuccessful or disfavored, but the available evidence does "
    "not sufficiently support any supplied failure category. Use when the outcome alone "
    "is the only negative signal or evidence is insufficient to attribute a cause."
)
_DIAGNOSIS_INSTRUCTIONS = (
    "Starting from the observed system-level failure, trace the causal chain backward. "
    "Identify the earliest supported event or omission from which execution did not recover. "
    "Test causality with a minimal-edit counterfactual: under the same preceding context, state "
    "the smallest feasible edit to that event that would plausibly have avoided or materially "
    "changed the failed outcome. Keep trace event indexes separate from verifier or artifact "
    "references: event_ref is only for the ordered trace; artifact_ref carries the artifact "
    "index and name. Use source_type 'verifier_artifact' for grader artifacts and "
    "'rollout_artifact' for task inputs or outputs listed in available_assets. For outcome "
    "summaries or any other non-artifact evidence, use source_type 'other' "
    "and set artifact_ref fields to null. Never encode a tool name, outcome id, or trace event "
    "as an artifact. Cite the critical event and each causal link using the correct reference "
    "type. Use unknown when evidence or the counterfactual link is insufficient. "
    "When the selected leaf is no_supported_failure, set abstention_reason to exactly one supplied "
    "reason. Use taxonomy_gap only when affirmative evidence supports a model-caused failure but "
    "none of the supplied leaves fit. Otherwise set abstention_reason to null. "
    "Do not invent omitted events."
)
_DIAGNOSIS_SCHEMA = {
    "observed_system_failure": "Concrete terminal failure supported by outcome evidence.",
    "causal_chain": [
        {
            "source_type": "trace_event|verifier_artifact|rollout_artifact|other",
            "event_ref": (
                "Primary event index, or {lane,event_idx} for an auxiliary lane; otherwise null."
            ),
            "artifact_ref": {
                "artifact_idx": "Exact listed artifact index, otherwise null.",
                "name": "Exact listed artifact name, otherwise null.",
            },
            "claim": "Observable event and its causal role.",
            "evidence": "Quoted or tightly paraphrased trace evidence.",
        }
    ],
    "critical_event": {
        "event_ref": "Primary event index, auxiliary {lane,event_idx}, or null.",
        "failure": "Specific action, decision, observation failure, or omission.",
        "why_earliest_unrecovered": "Why earlier candidates recovered or were not causal.",
    },
    "counterfactual_edit": {
        "target_event_ref": "Primary event index, auxiliary {lane,event_idx}, or null.",
        "minimal_edit": "Smallest feasible local change under the same prior context.",
        "expected_effect": "How that edit would plausibly change the terminal outcome.",
        "support": "supported|weak|unknown",
    },
    "evidence_support": "complete|partial|insufficient",
    "abstention_reason": (
        "insufficient_evidence|missing_observability|evaluator_mismatch|non_model_failure|"
        "taxonomy_gap|unknown, only for no_supported_failure; otherwise null"
    ),
}
_DIAGNOSIS_QUESTION = {
    "type": "structured",
    "instructions": _DIAGNOSIS_INSTRUCTIONS,
    "schema": _DIAGNOSIS_SCHEMA,
}
_WINDOW_LEAF_INSTRUCTIONS = (
    "Which new primary failure candidate is best supported in this window? "
    "Select no_local_candidate when this window adds none."
)
_NO_LOCAL_CRITERION = (
    "This window adds no new material failure candidate. This is local to the window and "
    "does not imply that the full trajectory has no failure."
)
_EPISODE_STATUS_CRITERIA = {
    "open_primary": "A new material primary cause remains unrecovered.",
    "open_contributing": "A new unrecovered contributor is supported but is not primary.",
    "recovered_in_window": "The new candidate is explicitly corrected in this window.",
    "downstream_symptom": "The candidate is a consequence of an earlier failure.",
    "no_material_candidate": "This window adds no material failure episode.",
}
_RECOVERY_CRITERIA = {
    "recovered_here": "A visible successful action explicitly corrects this episode.",
    "superseded": (
        "The earlier episode was recovered/downstream or a more specific "
        "independent cause replaces it."
    ),
    "still_open": "This window shows the episode remains unresolved.",
    "not_addressed": "This window does not establish recovery or persistence.",
}


@dataclass(frozen=True)
class _Verdict:
    """One judge answer to the leaf question."""

    code: str
    alternatives: tuple[str, ...]
    rationale: str
    diagnosis: dict[str, Any]
    model: str
    usage: dict[str, int]
    provider_calls: int = 1
    asset_evidence: tuple[dict[str, Any], ...] = ()


class TaxonomyClassifier:
    """Classify one incident or a sequence of trace windows."""

    def __init__(
        self,
        taxonomy: Taxonomy,
        provider: DecisionProvider | None,
        *,
        first_pass: FirstPassMethod = "llm",
        second_pass: SecondPassMethod = "llm",
        agentic_provider: Any = None,
        saved_first_pass: FirstPassMethod | None = None,
    ) -> None:
        if first_pass not in {"llm", "agentic"}:
            raise ValueError("first_pass must be llm or agentic")
        if saved_first_pass not in {None, "llm", "agentic"}:
            raise ValueError("saved_first_pass must be llm or agentic")
        if saved_first_pass is not None and saved_first_pass != first_pass:
            raise ValueError("saved_first_pass must match first_pass provenance")
        if second_pass not in {"off", "llm", "agentic"}:
            raise ValueError("second_pass must be off, llm, or agentic")
        required_methods = {second_pass}
        if saved_first_pass is None:
            required_methods.add(first_pass)
        if "llm" in required_methods and provider is None:
            raise ValueError("llm taxonomy passes require an HTTP decision provider")
        if "agentic" in required_methods and agentic_provider is None:
            raise ValueError("agentic taxonomy passes require the taxonomy agentic provider")
        self.taxonomy = taxonomy
        self.provider = provider
        self.first_pass = first_pass
        self.second_pass = second_pass
        self.agentic_provider = agentic_provider
        self.saved_first_pass = saved_first_pass
        leaves = list(taxonomy.leaves())
        self._leaves = {node.code or node.id: node for node in leaves}
        reserved = {NO_FAILURE_CODE, NO_LOCAL_CANDIDATE} & self._leaves.keys()
        if len(self._leaves) != len(leaves) or reserved:
            raise ValueError("leaf codes must be unique and must not reuse reserved choices")
        self._leaf_criteria = {
            key: f"{node.label}: {node.criterion()}" for key, node in self._leaves.items()
        }

    def classify(
        self,
        incident: dict[str, Any],
        *,
        artifact_source: dict[str, Any] | None = None,
    ) -> AnalysisResult:
        """Classify one incident with a single question over all leaves."""
        if self.saved_first_pass is not None:
            raise ValueError("saved-result classifiers can only run classify_second_pass")
        incident_id, incident_sha256 = _incident_identity(incident)
        if self.first_pass == "agentic":
            return self._classify_agentically(
                incident_id,
                incident_sha256,
                incident,
                windows=[],
                artifact_source=artifact_source,
            )
        first = self._ask(
            incident,
            _LEAF_INSTRUCTIONS,
            artifact_source=artifact_source,
        )
        return self._conclude(
            incident_id,
            incident_sha256,
            incident,
            first,
            provider_calls=first.provider_calls,
            usage=Counter(first.usage),
            artifact_source=artifact_source,
        )

    def classify_second_pass(
        self,
        incident: dict[str, Any],
        parent: dict[str, Any],
        *,
        windows: list[dict[str, Any]] | None = None,
        artifact_source: dict[str, Any] | None = None,
    ) -> AnalysisResult:
        """Apply only the configured second pass to a stored classification."""
        if self.second_pass == "off":
            raise ValueError("classify_second_pass requires llm or agentic")
        if parent.get("taxonomy_sha256") != self.taxonomy.content_sha256:
            raise ValueError("parent classification uses a different taxonomy")
        incident_id, incident_sha256 = _incident_identity(incident)
        if parent.get("incident_sha256") != incident_sha256:
            raise ValueError("parent classification evidence differs from the current source")
        first = _classification_verdict(parent)
        first_pass_details = _classification_first_pass(parent)
        if not parent.get("windowed"):
            return self._conclude(
                incident_id,
                incident_sha256,
                incident,
                first,
                provider_calls=0,
                usage=Counter(),
                artifact_source=artifact_source,
                first_pass_details=first_pass_details,
            )
        windows = windows or []
        episodes = list(copy.deepcopy(parent.get("episodes") or ()))
        selected_episode_id = parent.get("selected_episode_id")
        effective_episode_ids = _classification_effective_episode_ids(parent, episodes)
        if not windows:
            raise ValueError("parent classification requires prepared trace windows")
        expected_window_sha256s = tuple(
            str(value) for value in parent.get("window_sha256s") or ()
        )
        current_window_sha256s = tuple(
            str(window["window"]["sha256"]) for window in windows
        )
        if expected_window_sha256s and current_window_sha256s != expected_window_sha256s:
            raise ValueError(
                "prepared trace windows differ from the parent classification; "
                "reuse its evidence cache and window budget"
            )
        if first_pass_details["method"] == "agentic":
            return self._conclude(
                incident_id=incident_id,
                incident_sha256=incident_sha256,
                evidence={"incident": incident, "windows": windows},
                first=first,
                provider_calls=0,
                usage=Counter(),
                artifact_source=artifact_source,
                selected_episode_id=selected_episode_id,
                review_windows=windows,
                agentic_incident=incident,
                first_pass_details=first_pass_details,
                windowed=True,
                window_count=len(windows),
                window_sha256s=tuple(window["window"]["sha256"] for window in windows),
                window_receipts=tuple(
                    copy.deepcopy(parent.get("window_receipts") or ())
                ),
                aggregation=str(parent.get("aggregation") or "agentic_evidence_review"),
                episodes=tuple(episodes),
            )
        return self._conclude_windows(
            incident_id,
            incident_sha256,
            incident,
            windows,
            first,
            episodes=episodes,
            receipts=list(copy.deepcopy(parent.get("window_receipts") or ())),
            provider_calls=0,
            usage=Counter(),
            selected_episode_id=selected_episode_id,
            artifact_source=artifact_source,
            first_pass_details=first_pass_details,
            effective_episode_ids=effective_episode_ids,
        )

    def classify_windows(
        self,
        incident: dict[str, Any],
        windows: list[dict[str, Any]],
        *,
        artifact_source: dict[str, Any] | None = None,
    ) -> AnalysisResult:
        """Classify ordered windows with a rolling ledger of failure episodes."""
        if self.saved_first_pass is not None:
            raise ValueError("saved-result classifiers can only run classify_second_pass")
        if not windows:
            raise ValueError("windows must not be empty")
        incident_id, incident_sha256 = _incident_identity(incident)
        if self.first_pass == "agentic":
            return self._classify_agentically(
                incident_id,
                incident_sha256,
                incident,
                windows=windows,
                artifact_source=artifact_source,
            )
        episodes: list[dict[str, Any]] = []
        receipts = [
            self._classify_window(
                incident,
                window,
                episodes,
                artifact_source=artifact_source,
            )
            for window in windows
        ]
        usage: Counter[str] = Counter()
        for receipt in receipts:
            usage.update(receipt["usage"])
        selected = _earliest_open_episode(episodes)
        if selected is None:
            first = _Verdict(
                NO_FAILURE_CODE,
                (),
                "",
                receipts[-1].get("diagnosis") or {},
                receipts[-1]["model"],
                {},
            )
        else:
            first = _episode_verdict(selected)
        return self._conclude_windows(
            incident_id,
            incident_sha256,
            incident,
            windows,
            first,
            episodes=episodes,
            receipts=receipts,
            provider_calls=sum(receipt.get("provider_calls", 1) for receipt in receipts),
            usage=usage,
            selected_episode_id=selected["episode_id"] if selected else None,
            artifact_source=artifact_source,
        )

    def _questions(self, instructions: str) -> dict[str, dict[str, Any]]:
        return {
            "leaf": {
                "type": "choice",
                "instructions": instructions,
                "criteria": {**self._leaf_criteria, NO_FAILURE_CODE: _NO_FAILURE_CRITERION},
            },
            "diagnosis": _DIAGNOSIS_QUESTION,
        }

    def _evaluate(
        self,
        *,
        state: dict[str, Any],
        questions: dict[str, dict[str, Any]],
        artifact_source: dict[str, Any] | None,
    ) -> tuple[ProviderResponse, tuple[dict[str, Any], ...], int]:
        """Let a multimodal HTTP judge request verified artifact images."""
        if self.provider is None:
            raise ProviderError("HTTP taxonomy operation requires a decision provider")
        visual_catalog = [
            item for item in artifact_descriptors(artifact_source) if item.get("renderable")
        ]
        if not visual_catalog or artifact_source is None:
            return self.provider.evaluate(state=state, questions=questions), (), 1
        state["available_assets"] = visual_catalog
        if not getattr(self.provider, "supports_images", False):
            return self._evaluate_without_images(state, questions, visual_catalog)

        multimodal_questions = {**questions, "asset_request": asset_request_question()}
        requested: set[tuple[int, int]] = set()
        usage: Counter[str] = Counter()
        state["asset_investigation_policy"] = (
            "Images attached to the task prompt are supplied up front. Request other visual "
            "artifacts only when they are needed for causal classification. Missing or "
            "unavailable visual evidence requires insufficient support or "
            "missing_observability, not a text-only guess."
        )
        try:
            images, initial = investigate_requests(
                artifact_source,
                visual_catalog,
                [
                    {"artifact_idx": index, "page": 1, "reason": "Attached to the task prompt."}
                    for index in task_input_indexes(artifact_source, visual_catalog)
                ],
                requested=requested,
                trigger="task_input",
            )
        except AssetEvidenceError as exc:
            raise ProviderError(f"task input images are unusable: {exc}") from exc
        receipts: list[dict[str, Any]] = list(initial)
        base_state = state
        for round_index in range(MAX_ASSET_ROUNDS):
            call_state = {
                **base_state,
                "asset_investigation": {
                    "round": round_index,
                    "receipts": copy.deepcopy(receipts),
                },
            }
            try:
                response = self.provider.evaluate(
                    state=call_state,
                    questions=multimodal_questions,
                    images=images or None,
                )
            except ProviderError as exc:
                if not images or not image_input_is_unsupported(exc):
                    raise
                response, fallback_receipts, fallback_calls = (
                    self._evaluate_without_images(base_state, questions, visual_catalog)
                )
                return (
                    ProviderResponse(
                        model=response.model,
                        answers=response.answers,
                        usage=dict(usage + Counter(response.usage)),
                    ),
                    fallback_receipts,
                    round_index + 1 + fallback_calls,
                )
            usage.update(response.usage)
            requests = _structured(response.answers, "asset_request").get("requests")
            if not requests:
                return (
                    ProviderResponse(
                        model=response.model,
                        answers=response.answers,
                        usage=dict(usage),
                    ),
                    tuple(receipts),
                    round_index + 1,
                )
            if round_index + 1 == MAX_ASSET_ROUNDS:
                raise ProviderError(
                    f"judge continued requesting assets after {MAX_ASSET_ROUNDS} rounds"
                )
            try:
                new_images, new_receipts = investigate_requests(
                    artifact_source,
                    visual_catalog,
                    requests,
                    requested=requested,
                    existing_images=images,
                )
            except AssetEvidenceError as exc:
                raise ProviderError(f"invalid judge asset request: {exc}") from exc
            if not new_receipts:
                raise ProviderError("judge requested only assets already investigated")
            images.extend(new_images)
            receipts.extend(new_receipts)
        raise AssertionError("asset investigation loop exhausted unexpectedly")

    def _evaluate_without_images(
        self,
        state: dict[str, Any],
        questions: dict[str, dict[str, Any]],
        visual_catalog: list[dict[str, Any]],
    ) -> tuple[ProviderResponse, tuple[dict[str, Any], ...], int]:
        """Run a text-only request while recording unavailable visuals."""
        if self.provider is None:
            raise ProviderError("HTTP taxonomy operation requires a decision provider")
        warnings.warn(
            f"judge {getattr(self.provider, 'model', 'unknown')} does not accept images; "
            "continuing with text evidence only",
            UserWarning,
            stacklevel=2,
        )
        state["asset_investigation_policy"] = (
            "These visual artifacts exist but were not shown to this judge. Do not infer "
            "their contents from the model's own description of them. When the failure "
            "depends on what they show, report insufficient support or "
            "missing_observability rather than a text-only guess."
        )
        receipts = tuple(
            {
                "protocol": ASSET_INVESTIGATION_PROTOCOL,
                "artifact": copy.deepcopy(item),
                "trigger": "judge_without_image_input",
                "status": "not_shown",
            }
            for item in visual_catalog
        )
        return self.provider.evaluate(state=state, questions=questions), receipts, 1

    def _ask(
        self,
        incident: dict[str, Any],
        instructions: str,
        first: _Verdict | None = None,
        *,
        window: bool = False,
        open_failure_ledger: list[dict[str, Any]] | None = None,
        artifact_source: dict[str, Any] | None = None,
    ) -> _Verdict:
        if self.provider is None:
            raise ProviderError("HTTP taxonomy operation requires a decision provider")
        state: dict[str, Any] = {
            "incident": incident,
            "judge_instructions": judge_instructions(
                window=window,
                reflection=window and first is not None,
            ),
        }
        if open_failure_ledger is not None:
            state["open_failure_ledger"] = open_failure_ledger
        if first is not None:
            state["first_pass"] = {
                "leaf": {
                    "choice": first.code,
                    "alternatives": list(first.alternatives),
                    "rationale": first.rationale,
                },
                "diagnosis": first.diagnosis,
            }
        response, asset_evidence, provider_calls = self._evaluate(
            state=state,
            questions=self._questions(instructions),
            artifact_source=artifact_source,
        )
        leaf = _choice(response.answers, "leaf")
        diagnosis = _structured(response.answers, "diagnosis")
        _validate_diagnosis_references(diagnosis, state)
        return _Verdict(
            leaf["choice"],
            tuple(leaf["alternatives"]),
            leaf["rationale"],
            diagnosis,
            response.model,
            response.usage,
            provider_calls,
            asset_evidence,
        )

    def _conclude(
        self,
        incident_id: str,
        incident_sha256: str,
        evidence: dict[str, Any],
        first: _Verdict,
        *,
        provider_calls: int,
        usage: Counter[str],
        artifact_source: dict[str, Any] | None = None,
        selected_episode_id: str | None = None,
        review_windows: list[dict[str, Any]] | None = None,
        agentic_incident: dict[str, Any] | None = None,
        **window_fields: Any,
    ) -> AnalysisResult:
        """Apply the configured second pass and return the final result."""
        final = first
        reflection_outcome = None
        second_pass_usage: dict[str, int] = {}
        second_pass_details: dict[str, Any] | None = None
        if (
            self.second_pass == "llm"
            and review_windows
            and first.code != NO_FAILURE_CODE
            and _verdict_window(first, review_windows) is not None
        ):
            episode = _verdict_episode(first, review_windows, self._leaves)
            return self._conclude_windows(
                incident_id,
                incident_sha256,
                agentic_incident or evidence,
                review_windows,
                first,
                episodes=[episode],
                receipts=[],
                provider_calls=provider_calls,
                usage=usage,
                selected_episode_id=str(episode["episode_id"]),
                artifact_source=artifact_source,
                first_pass_details=window_fields.pop("first_pass_details", None),
                aggregation=str(
                    window_fields.get("aggregation") or "agentic_evidence_review"
                ),
            )
        if self.second_pass != "off":
            if self.second_pass == "agentic":
                provider_calls += 1
                try:
                    final, agentic = self._agentic_second_pass(
                        incident=agentic_incident or evidence,
                        first=first,
                        windows=review_windows or [],
                        episodes=[],
                        selected_episode_id=None,
                        artifact_source=artifact_source,
                    )
                except asyncio.CancelledError as exc:
                    second_pass_details = _second_pass_failure(exc)
                except Exception as exc:  # noqa: BLE001 - preserve the completed first pass
                    second_pass_details = _second_pass_failure(exc)
                else:
                    second_pass_usage = dict(agentic.get("usage") or {})
                    usage.update(second_pass_usage)
                    reflection_outcome = _reflection(first, final)
                    second_pass_details = {
                        "status": "completed",
                        "receipts": agentic.get("receipts") or {},
                    }
            else:
                reviewed: list[tuple[int, int, _Verdict]] = []
                review_usage: Counter[str] = Counter()
                review_provider_calls = 0
                try:
                    review_evidence = []
                    for fallback_index, window in enumerate(review_windows or []):
                        metadata = window.get("window") or {}
                        window_index = metadata.get("index")
                        if not isinstance(window_index, int):
                            window_index = fallback_index
                        start_event_idx = metadata.get("start_event_idx")
                        if not isinstance(start_event_idx, int):
                            start_event_idx = window_index
                        review_evidence.append(
                            (
                                window_index,
                                start_event_idx,
                                _window_incident(agentic_incident or evidence, window),
                            )
                        )
                    if not review_evidence:
                        review_evidence = [(0, 0, evidence)]
                    for window_index, start_event_idx, llm_evidence in review_evidence:
                        verdict = self._ask(
                            llm_evidence,
                            _REFLECTION_INSTRUCTIONS,
                            first,
                            artifact_source=artifact_source,
                        )
                        reviewed.append((window_index, start_event_idx, verdict))
                        review_provider_calls += verdict.provider_calls
                        review_usage.update(verdict.usage)
                except asyncio.CancelledError as exc:
                    second_pass_details = _second_pass_failure(exc)
                except Exception as exc:  # noqa: BLE001 - preserve the completed first pass
                    second_pass_details = _second_pass_failure(exc)
                else:
                    supported = [
                        item for item in reviewed if item[2].code != NO_FAILURE_CODE
                    ]
                    candidates = supported or reviewed
                    _, _, final = min(
                        candidates,
                        key=lambda item: (
                            (
                                1
                                if (
                                    (reference := _diagnosis_event_reference(
                                        item[2].diagnosis
                                    ))
                                    is not None
                                    and reference[0] is not None
                                )
                                else 0
                            ),
                            (
                                item[0]
                                if reference is not None and reference[0] is not None
                                else (
                                    reference[1]
                                    if reference is not None
                                    else item[1]
                                )
                            ),
                            item[0],
                            item[2].code,
                        ),
                    )
                    provider_calls += review_provider_calls
                    usage = usage + review_usage
                    second_pass_usage = dict(review_usage)
                    reflection_outcome = _reflection(first, final)
                    second_pass_details = {
                        "asset_evidence": list(final.asset_evidence),
                    }
        return self._result(
            incident_id=incident_id,
            incident_sha256=incident_sha256,
            verdict=final,
            first=first,
            provider_calls=provider_calls,
            usage=dict(usage),
            selected_episode_id=selected_episode_id,
            reflection_outcome=reflection_outcome,
            second_pass_usage=second_pass_usage,
            second_pass_details=second_pass_details,
            **window_fields,
        )

    def _conclude_windows(
        self,
        incident_id: str,
        incident_sha256: str,
        incident: dict[str, Any],
        windows: list[dict[str, Any]],
        first: _Verdict,
        *,
        episodes: list[dict[str, Any]],
        receipts: list[dict[str, Any]],
        provider_calls: int,
        usage: Counter[str],
        selected_episode_id: str | None,
        artifact_source: dict[str, Any] | None = None,
        first_pass_details: dict[str, Any] | None = None,
        effective_episode_ids: set[str] | None = None,
        aggregation: str = "rolling_failure_ledger",
    ) -> AnalysisResult:
        """Run an immutable recovery sweep, then review one surviving primary."""
        original_selected_episode_id = selected_episode_id
        final = first
        second_pass_usage: Counter[str] = Counter()
        recovery_receipts: list[dict[str, Any]] = []
        rejected_episode_ids: list[str] = []
        effective_open = (
            set(effective_episode_ids)
            if effective_episode_ids is not None
            else {
                str(episode["episode_id"])
                for episode in episodes
                if episode.get("status") in _OPEN_STATUSES
            }
        )

        def failed_llm_review(exc: BaseException) -> AnalysisResult:
            return self._result(
                incident_id=incident_id,
                incident_sha256=incident_sha256,
                verdict=first,
                first=first,
                provider_calls=provider_calls,
                usage=dict(usage),
                selected_episode_id=original_selected_episode_id,
                reflection_outcome=None,
                second_pass_usage=dict(second_pass_usage),
                second_pass_details=_second_pass_failure(exc),
                first_pass_details=first_pass_details,
                windowed=True,
                window_count=len(windows),
                window_sha256s=tuple(window["window"]["sha256"] for window in windows),
                window_receipts=tuple(receipts),
                aggregation=aggregation,
                episodes=tuple(episodes),
            )

        if self.second_pass == "agentic":
            provider_calls += 1
            try:
                final, agentic = self._agentic_second_pass(
                    incident=incident,
                    first=first,
                    windows=windows,
                    episodes=episodes,
                    selected_episode_id=original_selected_episode_id,
                    artifact_source=artifact_source,
                )
            except asyncio.CancelledError as exc:
                second_pass_details = _second_pass_failure(exc)
                reflection_outcome = None
            except Exception as exc:  # noqa: BLE001 - preserve the completed first pass
                second_pass_details = _second_pass_failure(exc)
                reflection_outcome = None
            else:
                second_pass_usage.update(agentic.get("usage") or {})
                usage.update(second_pass_usage)
                selected_episode_id = agentic.get("selected_episode_id")
                if (
                    selected_episode_id is None
                    and final.code != NO_FAILURE_CODE
                    and _verdict_window(final, windows) is not None
                    and not _verdict_matches_episode(
                        final,
                        next(
                            (
                                episode
                                for episode in episodes
                                if episode.get("episode_id")
                                == original_selected_episode_id
                            ),
                            None,
                        ),
                    )
                ):
                    episode = _verdict_episode(
                        final,
                        windows,
                        self._leaves,
                        episode_id=f"e{len(episodes)}",
                    )
                    episodes.append(episode)
                    effective_open.add(str(episode["episode_id"]))
                    selected_episode_id = str(episode["episode_id"])
                elif selected_episode_id is None:
                    selected_episode_id = original_selected_episode_id
                if final.code == NO_FAILURE_CODE:
                    selected_episode_id = None
                reflection_outcome = _reflection(
                    first,
                    final,
                    selected_episode_before=original_selected_episode_id,
                    selected_episode_after=selected_episode_id,
                )
                second_pass_details = {
                    "status": "completed",
                    "selected_episode_before": original_selected_episode_id,
                    "selected_episode_after": selected_episode_id,
                    "effective_episode_ids": (
                        sorted(effective_open)
                        if selected_episode_id is not None
                        else []
                    ),
                    "receipts": agentic.get("receipts") or {},
                }
            return self._result(
                incident_id=incident_id,
                incident_sha256=incident_sha256,
                verdict=final,
                first=first,
                provider_calls=provider_calls,
                usage=dict(usage),
                selected_episode_id=selected_episode_id,
                reflection_outcome=reflection_outcome,
                second_pass_usage=dict(second_pass_usage),
                second_pass_details=second_pass_details,
                first_pass_details=first_pass_details,
                windowed=True,
                window_count=len(windows),
                window_sha256s=tuple(window["window"]["sha256"] for window in windows),
                window_receipts=tuple(receipts),
                aggregation=aggregation,
                episodes=tuple(episodes),
            )
        if self.second_pass == "llm" and selected_episode_id is not None:
            for window in windows:
                index = window["window"]["index"]
                eligible = [
                    episode
                    for episode in episodes
                    if episode.get("episode_id") in effective_open
                    and episode.get("window_indexes")
                    and episode["window_indexes"][0] < index
                    and _episode_matches_window(episode, window)
                ]
                if not eligible:
                    continue
                try:
                    receipt = self._review_recovery_window(
                        incident,
                        window,
                        eligible,
                        artifact_source=artifact_source,
                    )
                except asyncio.CancelledError as exc:
                    return failed_llm_review(exc)
                except Exception as exc:  # noqa: BLE001 - preserve the completed first pass
                    return failed_llm_review(exc)
                recovery_receipts.append(receipt)
                provider_calls += receipt.get("provider_calls", 1)
                usage.update(receipt["usage"])
                second_pass_usage.update(receipt["usage"])
                for episode_id, answer in receipt["episode_recovery"].items():
                    if answer["choice"] in {"recovered_here", "superseded"}:
                        effective_open.discard(episode_id)

            selected = next(
                (
                    episode
                    for episode in episodes
                    if episode.get("episode_id") == original_selected_episode_id
                    and str(episode.get("episode_id")) in effective_open
                ),
                None,
            ) or _earliest_episode(episodes, effective_open)
            selected_episode_id = selected["episode_id"] if selected else None
            while selected is not None:
                candidate = (
                    first
                    if first_pass_details is not None
                    and selected.get("episode_id") == original_selected_episode_id
                    else _episode_verdict(selected)
                )
                opened = selected["window_indexes"][0]
                shown = next(
                    (window for window in windows if window["window"]["index"] == opened),
                    None,
                )
                if shown is None:
                    raise ProviderError(
                        f"selected episode refers to missing window {opened}"
                    )
                try:
                    final = self._ask(
                        _window_incident(incident, shown),
                        _REFLECTION_INSTRUCTIONS,
                        candidate,
                        window=True,
                        open_failure_ledger=[
                            {key: episode.get(key) for key in _LEDGER_FIELDS}
                            for episode in episodes
                            if episode.get("episode_id") in effective_open
                        ],
                        artifact_source=artifact_source,
                    )
                except asyncio.CancelledError as exc:
                    return failed_llm_review(exc)
                except Exception as exc:  # noqa: BLE001 - preserve the completed first pass
                    return failed_llm_review(exc)
                provider_calls += final.provider_calls
                usage.update(final.usage)
                second_pass_usage.update(final.usage)
                if final.code != NO_FAILURE_CODE:
                    break
                rejected_episode_ids.append(str(selected["episode_id"]))
                effective_open.discard(str(selected["episode_id"]))
                selected = _earliest_episode(episodes, effective_open)
                selected_episode_id = selected["episode_id"] if selected else None
            if selected is None:
                final = _Verdict(
                    NO_FAILURE_CODE,
                    (),
                    final.rationale if rejected_episode_ids else "",
                    final.diagnosis if rejected_episode_ids else {},
                    final.model if rejected_episode_ids else first.model,
                    {},
                )

        reflection_outcome = (
            _reflection(
                first,
                final,
                selected_episode_before=original_selected_episode_id,
                selected_episode_after=selected_episode_id,
            )
            if self.second_pass == "llm" and original_selected_episode_id is not None
            else None
        )
        return self._result(
            incident_id=incident_id,
            incident_sha256=incident_sha256,
            verdict=final,
            first=first,
            provider_calls=provider_calls,
            usage=dict(usage),
            selected_episode_id=selected_episode_id,
            reflection_outcome=reflection_outcome,
            second_pass_usage=dict(second_pass_usage),
            second_pass_details={
                "selected_episode_before": original_selected_episode_id,
                "selected_episode_after": selected_episode_id,
                "rejected_episode_ids": rejected_episode_ids,
                "recovery_receipts": recovery_receipts,
                "effective_episode_ids": sorted(effective_open),
                "asset_evidence": list(final.asset_evidence),
            },
            first_pass_details=first_pass_details,
            windowed=True,
            window_count=len(windows),
            window_sha256s=tuple(window["window"]["sha256"] for window in windows),
            window_receipts=tuple(receipts),
            aggregation=aggregation,
            episodes=tuple(episodes),
        )

    def _classify_agentically(
        self,
        incident_id: str,
        incident_sha256: str,
        incident: dict[str, Any],
        *,
        windows: list[dict[str, Any]],
        artifact_source: dict[str, Any] | None,
    ) -> AnalysisResult:
        if self.agentic_provider is None:
            raise ProviderError("agentic first pass requires the taxonomy agentic provider")
        result = self.agentic_provider.classify(
            taxonomy=self.taxonomy,
            incident=incident,
            windows=windows,
            artifact_source=artifact_source,
        )
        verdict = self._agentic_verdict(result, incident=incident, windows=windows)
        window_fields: dict[str, Any] = {}
        if windows:
            window_fields = {
                "windowed": True,
                "window_count": len(windows),
                "window_sha256s": tuple(
                    window["window"]["sha256"] for window in windows
                ),
                "window_receipts": (),
                "aggregation": "agentic_evidence_review",
                "episodes": (),
            }
        review_evidence = {"incident": incident, "windows": windows} if windows else incident
        return self._conclude(
            incident_id=incident_id,
            incident_sha256=incident_sha256,
            evidence=review_evidence,
            first=verdict,
            provider_calls=1,
            usage=Counter(result.get("usage") or {}),
            artifact_source=artifact_source,
            selected_episode_id=None,
            first_pass_details={
                "status": "completed",
                "model": verdict.model,
                "usage": dict(result.get("usage") or {}),
                "receipts": result.get("receipts") or {},
            },
            review_windows=windows,
            agentic_incident=incident,
            **window_fields,
        )

    def _agentic_second_pass(
        self,
        *,
        incident: dict[str, Any],
        first: _Verdict,
        windows: list[dict[str, Any]],
        episodes: list[dict[str, Any]],
        selected_episode_id: str | None,
        artifact_source: dict[str, Any] | None,
    ) -> tuple[_Verdict, dict[str, Any]]:
        if self.agentic_provider is None:
            raise ProviderError("agentic second pass requires the taxonomy agentic provider")
        result = self.agentic_provider.review(
            taxonomy=self.taxonomy,
            incident=incident,
            first_pass={
                "selected_code": first.code,
                "alternatives": list(first.alternatives),
                "rationale": first.rationale,
                "diagnosis": first.diagnosis,
                "selected_episode_id": selected_episode_id,
            },
            windows=windows,
            episodes=episodes,
            artifact_source=artifact_source,
        )
        return (
            self._agentic_verdict(result, incident=incident, windows=windows),
            result,
        )

    def _agentic_verdict(
        self,
        result: dict[str, Any],
        *,
        incident: dict[str, Any],
        windows: list[dict[str, Any]],
    ) -> _Verdict:
        code = str(result.get("selected_code") or NO_FAILURE_CODE)
        if code != NO_FAILURE_CODE and code not in self._leaves:
            raise ProviderError(f"agentic classifier returned unknown taxonomy code {code!r}")
        diagnosis = dict(result.get("diagnosis") or {})
        if code == NO_FAILURE_CODE:
            diagnosis.setdefault("abstention_reason", result.get("abstention_reason"))
        evidence = {
            "incident": incident,
            "windows": windows,
            "verifier_artifacts": result.get("artifact_catalog") or [],
        }
        _validate_diagnosis_references(diagnosis, evidence)
        alternatives = tuple(
            str(value)
            for value in result.get("alternatives") or []
            if str(value) in self._leaves and str(value) != code
        )
        return _Verdict(
            code=code,
            alternatives=alternatives,
            rationale=str(result.get("rationale") or ""),
            diagnosis=diagnosis,
            model=str(result.get("model") or "unknown"),
            usage=dict(result.get("usage") or {}),
        )

    def _review_recovery_window(
        self,
        incident: dict[str, Any],
        window: dict[str, Any],
        episodes: list[dict[str, Any]],
        *,
        artifact_source: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.provider is None:
            raise ProviderError("LLM recovery review requires a decision provider")
        ledger = [{key: episode.get(key) for key in _LEDGER_FIELDS} for episode in episodes]
        questions = {
            f"episode_{episode['episode_id']}": {
                "type": "choice",
                "instructions": (
                    "On a second pass, does this later window establish what happened "
                    "to the named earlier failure episode?"
                ),
                "criteria": dict(_RECOVERY_CRITERIA),
            }
            for episode in episodes
        }
        response, asset_evidence, provider_calls = self._evaluate(
            state={
                "incident": _window_incident(incident, window),
                "open_failure_ledger": ledger,
                "judge_instructions": judge_instructions(window=True, reflection=True),
                "second_pass": {"method": "llm", "phase": "recovery"},
            },
            questions=questions,
            artifact_source=artifact_source,
        )
        return {
            "window": window["window"],
            "episode_recovery": {
                episode["episode_id"]: _choice(
                    response.answers,
                    f"episode_{episode['episode_id']}",
                )
                for episode in episodes
            },
            "model": response.model,
            "usage": response.usage,
            "provider_calls": provider_calls,
            "asset_evidence": list(asset_evidence),
        }

    def _classify_window(
        self,
        incident: dict[str, Any],
        window: dict[str, Any],
        episodes: list[dict[str, Any]],
        *,
        artifact_source: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.provider is None:
            raise ProviderError("LLM window classification requires a decision provider")
        span = window["window"]
        open_before = [
            episode
            for episode in episodes
            if episode["status"] in _OPEN_STATUSES
            and _episode_matches_window(episode, window)
        ]
        window_incident = _window_incident(incident, window)
        ledger = [{key: episode.get(key) for key in _LEDGER_FIELDS} for episode in open_before]
        state = {
            "incident": window_incident,
            "open_failure_ledger": ledger,
            "judge_instructions": judge_instructions(window=True),
        }
        response, asset_evidence, provider_calls = self._evaluate(
            state=state,
            questions=self._window_questions(open_before),
            artifact_source=artifact_source,
        )
        recovery: dict[str, dict[str, Any]] = {}
        updates: list[dict[str, Any]] = []
        for episode in open_before:
            answer = _choice(response.answers, f"episode_{episode['episode_id']}")
            recovery[episode["episode_id"]] = answer
            if answer["choice"] in {"recovered_here", "superseded"}:
                episode["status"] = answer["choice"]
                episode["recovered_at_event_idx"] = span.get("end_event_idx")
                updates.append({"episode_id": episode["episode_id"], "status": answer["choice"]})
        leaf = _choice(response.answers, "leaf")
        disposition = _choice(response.answers, "episode_status")
        diagnosis = _structured(response.answers, "diagnosis")
        _validate_diagnosis_references(diagnosis, state)
        return {
            "window": span,
            "input_characters": serialized_characters(window),
            "leaf": leaf,
            "episode_status": disposition,
            "prior_episode_recovery": recovery,
            "episode_updates": updates,
            "new_episode_id": self._record_episode(
                episodes, window, leaf, disposition, diagnosis, response.model
            ),
            "open_episode_ids_after": [
                e["episode_id"] for e in episodes if e["status"] in _OPEN_STATUSES
            ],
            "model": response.model,
            "usage": response.usage,
            "diagnosis": diagnosis,
            "provider_calls": provider_calls,
            "asset_evidence": list(asset_evidence),
        }

    def _record_episode(
        self,
        episodes: list[dict[str, Any]],
        window: dict[str, Any],
        leaf: dict[str, Any],
        disposition: dict[str, Any],
        diagnosis: dict[str, Any],
        model: str,
    ) -> str | None:
        """Open or extend a matching failure episode."""
        span = window["window"]
        event_reference = _diagnosis_event_reference(diagnosis)
        episode_scope = str(span.get("scope") or "primary_lane")
        window_lane = str(span.get("lane") or "")
        episode_lane = window_lane if episode_scope != "trace_resources" else ""
        onset = None
        if event_reference is not None and (
            (
                episode_scope == "primary_lane"
                and event_reference[0] in {None, window_lane}
            )
            or (
                episode_scope == "auxiliary_lane"
                and event_reference[0] == episode_lane
            )
        ):
            onset = event_reference[1]
        start = onset if onset is not None else span.get("start_event_idx")
        end = onset if onset is not None else span.get("end_event_idx")
        if (
            episode_scope != "trace_resources"
            and (not isinstance(start, int) or not isinstance(end, int))
        ):
            indexes = [
                index
                for field in ("task_anchor", "ordered_trace", "terminal_anchor")
                for event in window.get(field) or []
                if isinstance(event, dict)
                for index in [(event.get("event_ref") or {}).get("event_idx")]
                if isinstance(index, int)
            ]
            if indexes:
                start, end = min(indexes), max(indexes)
        code = leaf["choice"]
        if code == NO_LOCAL_CANDIDATE or disposition["choice"] not in _OPEN_STATUSES:
            return None
        if episode_scope != "trace_resources" and (
            not isinstance(start, int) or not isinstance(end, int)
        ):
            return None
        signatures = _window_failure_signatures(window)
        overlap = span.get("overlap_event_indexes") or []
        if episode_scope != "trace_resources":
            for episode in episodes:
                if episode["status"] in _OPEN_STATUSES and _same_episode(
                    episode, code, signatures, overlap, episode_scope, episode_lane
                ):
                    _extend_episode(episode, span, disposition, signatures, start, end)
                    return episode["episode_id"]
        episode_id = f"e{len(episodes)}"
        episodes.append(
            {
                "episode_id": episode_id,
                "leaf_code": code,
                "leaf_label": self._leaves[code].label,
                "first_event_idx": start,
                "last_event_idx": end,
                "scope": episode_scope,
                "lane": episode_lane,
                "status": disposition["choice"],
                "failure_signatures": signatures,
                "window_indexes": [span["index"]],
                "alternatives": leaf["alternatives"],
                "rationale": leaf["rationale"],
                "diagnosis": diagnosis,
                "model": model,
            }
        )
        return episode_id

    def _window_questions(self, open_episodes: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        questions: dict[str, dict[str, Any]] = {
            "leaf": {
                "type": "choice",
                "instructions": _WINDOW_LEAF_INSTRUCTIONS,
                "criteria": {**self._leaf_criteria, NO_LOCAL_CANDIDATE: _NO_LOCAL_CRITERION},
            },
            "episode_status": {
                "type": "choice",
                "instructions": "What is the disposition of this window's new candidate?",
                "criteria": dict(_EPISODE_STATUS_CRITERIA),
            },
        }
        for episode in open_episodes:
            questions[f"episode_{episode['episode_id']}"] = {
                "type": "choice",
                "instructions": (
                    "Does this window explicitly recover the named earlier failure episode?"
                ),
                "criteria": dict(_RECOVERY_CRITERIA),
            }
        questions["diagnosis"] = _DIAGNOSIS_QUESTION
        return questions

    def _path(self, node: TaxonomyNode) -> tuple[tuple[str, ...], tuple[str | None, ...]]:
        path_ids = self.taxonomy.path_ids(node.id)
        return path_ids, tuple(self.taxonomy.node(node_id).code for node_id in path_ids)

    def _alternative(self, node: TaxonomyNode) -> dict[str, Any]:
        path_ids, path_codes = self._path(node)
        return {
            "leaf_id": node.id,
            "code": node.code,
            "label": node.label,
            "path": self.taxonomy.labels_for_path(path_ids),
            "path_ids": path_ids,
            "path_codes": path_codes,
        }

    def _result(
        self,
        *,
        incident_id: str,
        incident_sha256: str,
        verdict: _Verdict,
        first: _Verdict,
        provider_calls: int,
        usage: dict[str, int],
        selected_episode_id: str | None,
        reflection_outcome: str | None,
        second_pass_usage: dict[str, int],
        second_pass_details: dict[str, Any] | None = None,
        first_pass_details: dict[str, Any] | None = None,
        **window_fields: Any,
    ) -> AnalysisResult:
        node = self._leaves.get(verdict.code)
        first_node = self._leaves.get(first.code)
        path_ids, path_codes = self._path(node) if node is not None else ((), ())
        insufficient = verdict.diagnosis.get("evidence_support") == "insufficient"
        return AnalysisResult(
            incident_id=incident_id,
            incident_sha256=incident_sha256,
            taxonomy_id=self.taxonomy.taxonomy_id,
            taxonomy_name=self.taxonomy.name,
            taxonomy_version=self.taxonomy.version,
            taxonomy_sha256=self.taxonomy.content_sha256,
            protocol_version=PROTOCOL_VERSION,
            model=verdict.model,
            decision="accept" if node and not insufficient else "review",
            decision_reasons=(
                ()
                if node and not insufficient
                else ("insufficient_evidence",)
                if insufficient
                else (NO_FAILURE_CODE,)
            ),
            selected_leaf_id=node.id if node else None,
            selected_code=node.code if node else NO_FAILURE_CODE,
            selected_label=node.label if node else "No supported failure category",
            selected_path=self.taxonomy.labels_for_path(path_ids),
            selected_path_ids=path_ids,
            selected_path_codes=path_codes,
            alternatives=tuple(
                self._alternative(self._leaves[key])
                for key in verdict.alternatives
                if key in self._leaves
            ),
            owner=node.fault_side if node else None,
            remediation=node.remediation if node else (),
            related=tuple(
                {
                    "id": related_id,
                    "code": self.taxonomy.node(related_id).code,
                    "label": self.taxonomy.node(related_id).label,
                }
                for related_id in (node.related if node else ())
            ),
            trace=(
                {
                    "question": "window_ledger" if window_fields.get("windowed") else "all_leaves",
                    "selected_leaf_id": node.id if node else None,
                    "rationale": verdict.rationale or None,
                },
            ),
            diagnosis=verdict.diagnosis,
            abstention_reason=_abstention_reason(verdict),
            provider_calls=provider_calls,
            usage=usage,
            first_pass={
                "method": self.first_pass,
                "status": "completed",
                "selected_leaf_id": first_node.id if first_node else None,
                "selected_code": first_node.code if first_node else NO_FAILURE_CODE,
                "alternatives": list(first.alternatives),
                "rationale": first.rationale or None,
                "abstention_reason": _abstention_reason(first),
                "model": first.model,
                "usage": dict(first.usage),
                "provider_calls": first.provider_calls,
                "asset_evidence": list(first.asset_evidence),
                **(first_pass_details or {}),
            },
            second_pass={
                "method": self.second_pass,
                "status": "completed" if reflection_outcome is not None else "skipped",
                "outcome": reflection_outcome,
                "usage": second_pass_usage,
                **(second_pass_details or {}),
            },
            selected_episode_id=selected_episode_id,
            **window_fields,
        )


def _reflection(
    first: _Verdict,
    final: _Verdict,
    *,
    selected_episode_before: str | None = None,
    selected_episode_after: str | None = None,
) -> str:
    if final.code == NO_FAILURE_CODE and first.code != NO_FAILURE_CODE:
        return "abstained"
    if (
        final.code == first.code
        and selected_episode_after == selected_episode_before
    ):
        return "confirmed"
    return "revised"


def _second_pass_failure(exc: BaseException) -> dict[str, Any]:
    return {
        "status": "failed",
        "error_type": type(exc).__name__,
        "error": str(exc),
    }


def _classification_verdict(record: dict[str, Any]) -> _Verdict:
    code = str(
        record.get("selected_code")
        or record.get("selected_leaf_id")
        or NO_FAILURE_CODE
    )
    alternatives = tuple(
        str(item.get("code") or item.get("leaf_id"))
        for item in record.get("alternatives") or []
        if isinstance(item, dict) and (item.get("code") or item.get("leaf_id"))
    )
    trace = record.get("trace") or []
    rationale = (
        str(trace[0].get("rationale") or "")
        if trace and isinstance(trace[0], dict)
        else ""
    )
    judge = record.get("judge") or {}
    return _Verdict(
        code=code,
        alternatives=alternatives,
        rationale=rationale,
        diagnosis=dict(record.get("diagnosis") or {}),
        model=str(judge.get("model") or judge.get("requested_model") or "unknown"),
        usage={},
    )


def _classification_first_pass(record: dict[str, Any]) -> dict[str, Any]:
    details = copy.deepcopy(record.get("first_pass") or {})
    if not isinstance(details, dict):
        details = {}
    method = details.get("method")
    if method not in {"llm", "agentic"}:
        method = (
            "agentic"
            if record.get("classification_method") == "agentic_taxonomy_classification"
            else "llm"
        )
    if record.get("status") != "review_failed":
        for key in (
            "selected_leaf_id",
            "selected_code",
            "alternatives",
            "rationale",
            "abstention_reason",
            "model",
            "usage",
            "provider_calls",
            "asset_evidence",
        ):
            details.pop(key, None)
    details["method"] = method
    return details


def _classification_effective_episode_ids(
    record: dict[str, Any],
    episodes: list[dict[str, Any]],
) -> set[str]:
    effective = {
        str(episode["episode_id"])
        for episode in episodes
        if episode.get("status") in _OPEN_STATUSES
    }
    second_pass = record.get("second_pass")
    if not isinstance(second_pass, dict):
        return effective
    persisted = second_pass.get("effective_episode_ids")
    if isinstance(persisted, list):
        return {
            str(episode_id)
            for episode_id in persisted
            if str(episode_id) in effective
        }
    effective.difference_update(
        str(episode_id) for episode_id in second_pass.get("rejected_episode_ids") or []
    )
    for receipt in second_pass.get("recovery_receipts") or []:
        if not isinstance(receipt, dict):
            continue
        recovery = receipt.get("episode_recovery")
        if not isinstance(recovery, dict):
            continue
        for episode_id, answer in recovery.items():
            if isinstance(answer, dict) and answer.get("choice") in {
                "recovered_here",
                "superseded",
            }:
                effective.discard(str(episode_id))
    return effective


def _abstention_reason(verdict: _Verdict) -> str | None:
    if verdict.code != NO_FAILURE_CODE:
        return None
    reason = verdict.diagnosis.get("abstention_reason")
    return reason if reason in ABSTENTION_REASONS else "unknown"


def _incident_identity(incident: dict[str, Any]) -> tuple[str, str]:
    if not isinstance(incident, dict) or not incident:
        raise ValueError("incident must be a non-empty JSON object")
    incident_id = _incident_id(incident)
    canonical = json.dumps(incident, sort_keys=True, ensure_ascii=False, default=str)
    return incident_id, hashlib.sha256(canonical.encode()).hexdigest()


def _choice(answers: dict[str, dict[str, Any]], question_id: str) -> dict[str, Any]:
    answer = answers.get(question_id)
    if not isinstance(answer, dict) or "choice" not in answer:
        raise ProviderError(f"missing Choice answer for {question_id!r}")
    return answer


def _structured(answers: dict[str, dict[str, Any]], question_id: str) -> dict[str, Any]:
    answer = answers.get(question_id)
    if answer is None:
        return {}
    if not isinstance(answer, dict) or not isinstance(answer.get("value"), dict):
        raise ProviderError(f"missing Structured answer for {question_id!r}")
    return dict(answer["value"])


def _validate_diagnosis_references(diagnosis: dict[str, Any], evidence: Any) -> None:
    if not diagnosis:
        return
    event_indexes, event_refs, artifacts, outcome_ids = _available_references(evidence)

    def visit(value: Any, path: str) -> None:
        if isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")
            return
        if not isinstance(value, dict):
            return
        for key, item in value.items():
            item_path = f"{path}.{key}" if path else key
            if key in {"event_ref", "target_event_ref"}:
                valid = (
                    item is None
                    or (type(item) is int and item in event_indexes)
                    or (
                        isinstance(item, dict)
                        and isinstance(item.get("lane"), str)
                        and type(item.get("event_idx")) is int
                        and (item["lane"], item["event_idx"]) in event_refs
                    )
                )
                if not valid:
                    raise ProviderError(
                        f"diagnosis {item_path} does not refer to an available trace event"
                    )
            elif key == "artifact_ref":
                _validate_artifact_ref(item, item_path, artifacts, outcome_ids)
            visit(item, item_path)

    visit(diagnosis, "")


def _validate_artifact_ref(
    ref: Any, path: str, artifacts: dict[int, set[str]], outcome_ids: set[str]
) -> None:
    """Accept a supplied rollout artifact, or an outcome item id with a null index."""
    if ref is None:
        return
    if not isinstance(ref, dict):
        raise ProviderError(f"diagnosis {path} must be an artifact reference")
    artifact_idx, name = ref.get("artifact_idx"), ref.get("name")
    if artifact_idx is None and (name is None or name in outcome_ids):
        return
    if type(artifact_idx) is not int or artifact_idx not in artifacts:
        raise ProviderError(f"diagnosis {path} does not refer to an available artifact")
    names = artifacts[artifact_idx]
    if name is not None and (not isinstance(name, str) or (names and name not in names)):
        raise ProviderError(f"diagnosis {path}.name does not match the available artifact")


def _diagnosis_event_onset(diagnosis: dict[str, Any]) -> int | None:
    """Return the validated critical event, excluding contextual citations."""
    reference = _diagnosis_event_reference(diagnosis)
    return reference[1] if reference is not None else None


def _diagnosis_event_reference(
    diagnosis: dict[str, Any],
) -> tuple[str | None, int] | None:
    """Return a legacy-primary or lane-qualified critical event identity."""
    critical = diagnosis.get("critical_event")
    if not isinstance(critical, dict):
        return None
    reference = critical.get("event_ref")
    if type(reference) is int:
        return None, reference
    if (
        isinstance(reference, dict)
        and isinstance(reference.get("lane"), str)
        and type(reference.get("event_idx")) is int
    ):
        return reference["lane"], reference["event_idx"]
    return None


def _available_references(  # noqa: C901
    value: Any,
) -> tuple[set[int], set[tuple[str, int]], dict[int, set[str]], set[str]]:
    event_indexes: set[int] = set()
    event_refs: set[tuple[str, int]] = set()
    artifacts: dict[int, set[str]] = {}
    outcome_ids: set[str] = set()

    def trace_references(items: Any, *, primary: bool) -> None:
        if not isinstance(items, list):
            return
        for item in items:
            if not isinstance(item, dict):
                continue
            event_ref = item.get("event_ref")
            if isinstance(event_ref, dict) and type(event_ref.get("event_idx")) is int:
                lane = str(event_ref.get("lane") or "")
                event_refs.add((lane, event_ref["event_idx"]))
                if primary:
                    event_indexes.add(event_ref["event_idx"])

    def verifier_references(items: Any) -> None:
        if not isinstance(items, list):
            return
        for item in items:
            if not isinstance(item, dict) or type(item.get("artifact_idx")) is not int:
                continue
            names = artifacts.setdefault(item["artifact_idx"], set())
            if isinstance(item.get("name"), str):
                names.add(item["name"])

    def ledger_references(items: Any) -> None:
        if not isinstance(items, list):
            return
        for item in items:
            if not isinstance(item, dict):
                continue
            for key in ("first_event_idx", "last_event_idx", "recovered_at_event_idx"):
                if type(item.get(key)) is int:
                    lane = item.get("lane")
                    if isinstance(lane, str) and lane:
                        event_refs.add((lane, item[key]))
                    else:
                        event_indexes.add(item[key])

    def outcome_references(item: Any) -> None:
        if not isinstance(item, dict) or not isinstance(item.get("items"), list):
            return
        outcome_ids.update(
            entry["id"]
            for entry in item["items"]
            if isinstance(entry, dict) and isinstance(entry.get("id"), str)
        )

    def visit(item: Any, *, auxiliary: bool = False) -> None:
        if isinstance(item, list):
            for child in item:
                visit(child, auxiliary=auxiliary)
            return
        if not isinstance(item, dict):
            return
        for key, child in item.items():
            if key in {"ordered_trace", "task_anchor", "terminal_anchor"}:
                trace_references(child, primary=not auxiliary)
            elif key == "auxiliary_lanes":
                visit(child, auxiliary=True)
            elif key in {"verifier_artifacts", "available_assets"}:
                verifier_references(child)
            elif key == "open_failure_ledger":
                ledger_references(child)
            elif key == "outcome_evidence":
                outcome_references(child)
            else:
                visit(child, auxiliary=auxiliary)

    visit(value)
    return event_indexes, event_refs, artifacts, outcome_ids


def _earliest_open_episode(episodes: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Earliest open primary episode, else the earliest open contributing one."""
    return _earliest_episode(
        episodes,
        {
            str(episode["episode_id"])
            for episode in episodes
            if episode.get("status") in _OPEN_STATUSES
        },
    )


def _episode_matches_window(
    episode: dict[str, Any],
    window: dict[str, Any],
) -> bool:
    """Only one ordered lane may recover an episode from that same lane."""
    span = window.get("window") or {}
    scope = str(span.get("scope") or "primary_lane")
    if scope == "trace_resources":
        return False
    episode_scope = str(episode.get("scope") or "primary_lane")
    if episode_scope != scope:
        return False
    if scope == "auxiliary_lane":
        return str(episode.get("lane") or "") == str(span.get("lane") or "")
    return True


def _earliest_episode(
    episodes: list[dict[str, Any]],
    episode_ids: set[str],
) -> dict[str, Any] | None:
    for status in _OPEN_STATUSES:
        candidates = [
            episode
            for episode in episodes
            if episode.get("status") == status
            and str(episode.get("episode_id")) in episode_ids
        ]
        if candidates:
            return min(
                candidates,
                key=lambda episode: (
                    {
                        "primary_lane": 0,
                        "auxiliary_lane": 1,
                        "trace_resources": 2,
                    }.get(str(episode.get("scope") or "primary_lane"), 3),
                    (
                        episode["first_event_idx"]
                        if isinstance(episode.get("first_event_idx"), int)
                        else min(episode.get("window_indexes") or [0])
                    ),
                ),
            )
    return None


def _episode_verdict(episode: dict[str, Any]) -> _Verdict:
    return _Verdict(
        episode["leaf_code"],
        tuple(episode["alternatives"]),
        episode["rationale"],
        episode.get("diagnosis") or {},
        episode["model"],
        {},
    )


def _verdict_episode(
    verdict: _Verdict,
    windows: list[dict[str, Any]],
    leaves: dict[str, TaxonomyNode],
    *,
    episode_id: str = "agentic-proposal",
) -> dict[str, Any]:
    shown = _verdict_window(verdict, windows)
    if shown is None:
        raise ValueError("verdict does not cite an event present in the review windows")
    event_reference = _diagnosis_event_reference(verdict.diagnosis)
    assert event_reference is not None
    onset = event_reference[1]
    span = shown["window"]
    scope = str(span.get("scope") or "primary_lane")
    return {
        "episode_id": episode_id,
        "leaf_code": verdict.code,
        "leaf_label": leaves[verdict.code].label,
        "first_event_idx": onset,
        "last_event_idx": onset,
        "scope": scope,
        "lane": str(span.get("lane") or "") if scope != "trace_resources" else "",
        "status": "open_primary",
        "failure_signatures": _window_failure_signatures(shown),
        "window_indexes": [span["index"]],
        "alternatives": list(verdict.alternatives),
        "rationale": verdict.rationale,
        "diagnosis": verdict.diagnosis,
        "model": verdict.model,
    }


def _verdict_window(
    verdict: _Verdict,
    windows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Resolve a review window only from an event reference the verdict actually cites."""
    event_reference = _diagnosis_event_reference(verdict.diagnosis)
    if event_reference is None:
        return None
    lane, event_idx = event_reference
    return next(
        (
            window
            for window in windows
            if (
                (lane is None and event_idx in _available_references(window)[0])
                or (
                    lane is not None
                    and (lane, event_idx) in _available_references(window)[1]
                )
            )
        ),
        None,
    )


def _verdict_matches_episode(
    verdict: _Verdict,
    episode: dict[str, Any] | None,
) -> bool:
    if not episode or episode.get("leaf_code") != verdict.code:
        return False
    event_reference = _diagnosis_event_reference(verdict.diagnosis)
    onset = event_reference[1] if event_reference is not None else None
    scope = str(episode.get("scope") or "primary_lane")
    if event_reference is not None:
        if event_reference[0] is None and scope != "primary_lane":
            return False
        if (
            event_reference[0] is not None
            and str(episode.get("lane") or "") != event_reference[0]
        ):
            return False
    if onset is None:
        return True
    first = episode.get("first_event_idx")
    last = episode.get("last_event_idx")
    return isinstance(first, int) and isinstance(last, int) and first <= onset <= last


def _extend_episode(
    episode: dict[str, Any],
    span: dict[str, Any],
    disposition: dict[str, Any],
    signatures: list[str],
    first_event_idx: int,
    last_event_idx: int,
) -> None:
    episode["first_event_idx"] = min(episode["first_event_idx"], first_event_idx)
    episode["last_event_idx"] = max(episode["last_event_idx"], last_event_idx)
    if disposition["choice"] == "open_primary":
        episode["status"] = "open_primary"
    episode["window_indexes"].append(span["index"])
    episode["failure_signatures"] = sorted({*episode["failure_signatures"], *signatures})


def _window_incident(incident: dict[str, Any], window: dict[str, Any]) -> dict[str, Any]:
    window_index = window["window"]["index"]
    return {
        **incident,
        "id": f"{_incident_id(incident)}:window:{window_index}",
        "evidence": [
            {
                "evidence_id": f"ordered-trace-window-{window_index}",
                "kind": "bounded_ordered_trace_window",
                "observation": window,
            }
        ],
    }


def _incident_id(incident: dict[str, Any]) -> str:
    return str(incident.get("id") or incident.get("finding_id") or "unknown")


def _window_failure_signatures(window: dict[str, Any]) -> list[str]:
    if window["window"].get("scope") == "auxiliary_lane":
        start = window["window"].get("start_event_idx")
        end = window["window"].get("end_event_idx")
        if not isinstance(start, int) or not isinstance(end, int):
            return []
        signatures = []
        window_lane = str(window["window"].get("lane") or "")
        for lane in window.get("auxiliary_lanes") or []:
            if (
                not isinstance(lane, dict)
                or str(lane.get("lane") or "") != window_lane
            ):
                continue
            for item in lane.get("error_extract") or []:
                if not isinstance(item, dict):
                    continue
                indexes = [
                    index
                    for index in item.get("event_idxs") or [item.get("event_idx")]
                    if isinstance(index, int) and start <= index <= end
                ]
                if indexes:
                    signatures.append(
                        f"{window_lane}:{item.get('tool_name')}:"
                        f"{min(indexes)}:{max(indexes)}:{item.get('repeat_count', 1)}"
                    )
        return signatures
    start = window["window"].get("start_event_idx")
    end = window["window"].get("end_event_idx")
    if not isinstance(start, int) or not isinstance(end, int):
        return []
    signatures: list[str] = []
    for item in window.get("failure_index") or []:
        indexes = [index for index in item.get("event_idxs") or [] if isinstance(index, int)]
        if any(start <= index <= end for index in indexes):
            signatures.append(
                f"{item.get('tool_name')}:{min(indexes)}:{max(indexes)}:"
                f"{item.get('repeat_count', 1)}"
            )
    return signatures


def _same_episode(
    episode: dict[str, Any],
    code: str,
    signatures: list[str],
    overlap: list[int],
    scope: str,
    lane: str,
) -> bool:
    episode_scope = str(episode.get("scope") or "primary_lane")
    if (
        episode["leaf_code"] != code
        or episode_scope != scope
        or str(episode.get("lane") or "") != lane
    ):
        return False
    if set(episode["failure_signatures"]) & set(signatures):
        return True
    return any(
        episode["first_event_idx"] <= event_idx <= episode["last_event_idx"]
        for event_idx in overlap
    )
