# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapter for OpenAI's typed POST /decisions contract.

Reference: https://developers.openai.com/api/reference/resources/decisions
Use the SDK's public HTTP method so this also works before the installed SDK
exposes a generated ``decisions`` resource. Authentication and retries stay on
the configured client. Free-form text and extracted strings use generation APIs.
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any


def _same(left: Any, right: Any) -> bool:
    # The API explicitly distinguishes boolean true from the string "true".
    return type(left) is type(right) and left == right


def _questions(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(questions, list) or not 1 <= len(questions) <= 400:
        raise ValueError("Decisions requires between 1 and 400 questions")
    for question in questions:
        if not isinstance(question, dict):
            raise ValueError("Each decision question must be an object")
        kind = question.get("type")
        if kind not in {"predicate", "choice", "score"}:
            raise ValueError("Decision question type must be predicate, choice or score")
        if (not isinstance(question.get("instructions"), str)
                or not question["instructions"].strip()):
            raise ValueError("Decision questions require nonblank instructions")
        if question.get("name") is not None and not isinstance(question["name"], str):
            raise ValueError("Decision question names must be strings")
        allowed = {"type", "name", "instructions"}
        if kind in {"choice", "score"}:
            key, field = ("choices", "value") if kind == "choice" else ("levels", "label")
            allowed.add(key)
            values = question.get(key)
            if not isinstance(values, list) or not values:
                raise ValueError(f"Decision {kind} requires nonempty {key}")
            seen = []
            for value in values:
                if (not isinstance(value, dict) or field not in value
                        or type(value[field]) not in ((str, bool) if kind == "choice" else (str,))
                        or set(value) - {field, "description"}
                        or (value.get("description") is not None
                            and not isinstance(value["description"], str))):
                    raise ValueError(f"Invalid decision {key} entry")
                if any(_same(value[field], old) for old in seen):
                    raise ValueError(f"Decision {key} must be unique")
                seen.append(value[field])
        if set(question) - allowed:
            raise ValueError("Unsupported decision question fields")
    return deepcopy(questions)


def _number(value: Any, *, probability: bool = True) -> None:
    if (type(value) not in (int, float) or not math.isfinite(value)
            or (probability and not 0 <= value <= 1)):
        raise RuntimeError("Decision API returned an invalid probability or score")


def _answers(result: Any, questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    answers = result.get("answers") if isinstance(result, dict) else None
    if not isinstance(answers, list) or len(answers) != len(questions):
        raise RuntimeError("Decision API returned an incomplete answer list")
    for answer, question in zip(answers, questions, strict=True):
        if not isinstance(answer, dict) or answer.get("name") != question.get("name"):
            raise RuntimeError("Decision API returned a mismatched question name")
        kind = answer.get("type")
        if kind == "refusal":
            continue  # Preserve per-question refusals for callers of decide().
        if kind != question["type"]:
            raise RuntimeError("Decision API returned the wrong answer type")
        if kind == "predicate":
            _number(answer.get("probability"))
            continue
        _number(answer.get("confidence"))
        key, field = ("choices", "value") if kind == "choice" else ("levels", "label")
        allowed = [entry[field] for entry in question[key]]
        if kind == "choice":
            if not any(_same(answer.get("choice"), value) for value in allowed):
                raise RuntimeError("Decision API returned a choice outside the requested options")
        else:
            _number(answer.get("score"), probability=False)
        probabilities = answer.get("probabilities")
        if not isinstance(probabilities, list) or len(probabilities) != len(allowed):
            raise RuntimeError("Decision API returned an incomplete probability distribution")
        remaining = list(allowed)
        for entry in probabilities:
            if not isinstance(entry, dict):
                raise RuntimeError("Decision API returned an invalid probability distribution")
            _number(entry.get("probability"))
            match = next((i for i, value in enumerate(remaining)
                          if _same(entry.get(field), value)), None)
            if match is None:
                raise RuntimeError("Decision API returned an unexpected or duplicate option")
            remaining.pop(match)
            if kind == "score" and type(entry.get("value")) is not int:
                raise RuntimeError("Decision API returned an invalid level value")
    return answers


def create_decision(client: Any, *, model: str, input: str | list[dict[str, Any]],
                    questions: list[dict[str, Any]], timeout: float | None = None,
                    usage_role: str = "detector") -> list[dict]:
    from autofa.llm.capabilities import APIUnavailable, invoke_api
    from autofa.util.cancellation import check_cancelled

    check_cancelled()
    questions = _questions(questions)
    options = {"timeout": timeout} if timeout is not None else {}
    def request() -> list[dict]:
        from autofa.util.model_budget import reserve_model_call

        if not callable(getattr(client, "post", None)):
            raise APIUnavailable("decisions")
        reserve_model_call()
        from autofa.llm.usage import observed_call

        result = observed_call(
            client.post, "/decisions", client=client, usage_model=model, api="decisions",
            role=usage_role, cast_to=dict[str, Any],
            body={"model": model, "input": input, "questions": questions}, options=options,
        )
        check_cancelled()
        return _answers(result, questions)

    return invoke_api(client, model, "decisions", request)
