# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenAI-compatible judge adapter."""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Protocol

from .models import ProviderResponse

ALTERNATIVE_LIMIT = 4


class ProviderError(RuntimeError):
    """A judge request failed or returned an invalid answer."""


def image_input_is_unsupported(error: ProviderError) -> bool:
    """Return whether an endpoint explicitly rejects image input."""
    cause = error.__cause__
    status = getattr(cause, "status_code", None)
    if status is not None and status not in {400, 415, 422}:
        return False
    message = f"{error} {cause}".lower()
    image_terms = ("image", "vision", "multimodal", "input_image")
    unsupported_terms = (
        "not support",
        "not a multimodal model",
        "unsupported",
        "does not accept",
        "not allowed",
        "invalid content type",
    )
    return any(term in message for term in image_terms) and any(
        term in message for term in unsupported_terms
    )


class DecisionProvider(Protocol):
    def evaluate(
        self,
        state: str | dict[str, Any] | list[Any],
        questions: dict[str, dict[str, Any]],
        *,
        images: list[dict[str, Any]] | None = None,
    ) -> ProviderResponse: ...


def effective_reasoning_effort(provider: Any) -> str:
    """Return the reasoning setting used by a judge provider."""
    options = getattr(provider, "request_options", None) or {}
    return str(options.get("reasoning_effort") or "endpoint_default")


def provider_configuration_sha256(provider: Any) -> str:
    """Hash judge settings that can change a classification."""
    from .assets import (
        ASSET_INVESTIGATION_PROTOCOL,
        MAX_ASSET_REQUESTS,
        MAX_ASSET_ROUNDS,
        MAX_TOTAL_IMAGE_BYTES,
    )

    payload = {
        "model": getattr(provider, "model", None),
        "endpoint": getattr(provider, "endpoint", None),
        "api": getattr(provider, "api", None),
        "request_options": getattr(provider, "request_options", None) or {},
        "reasoning_effort": effective_reasoning_effort(provider),
        # Providers take output limits from request_options; keep the key for stable hashes.
        "max_tokens": None,
        "asset_investigation": {
            "protocol": ASSET_INVESTIGATION_PROTOCOL,
            "max_requests": MAX_ASSET_REQUESTS,
            "max_rounds": MAX_ASSET_ROUNDS,
            "max_total_image_bytes": MAX_TOTAL_IMAGE_BYTES,
        },
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


_ANSWER_FORMAT = (
    "Answer every question about the state. Reply with one json object and no markdown. "
    'Choice shape: {"answers": {"<question_id>": {"type": "choice", '
    '"choice": "<criterion key>", "alternatives": ["<criterion key>"], '
    '"rationale": "<brief evidence summary>"}}}. '
    'Structured-answer shape: {"type": "structured", "value": '
    "{<fields required by the question schema>}}. "
    "Use exactly that question's criterion keys. choice is the single best-supported key. "
    f"alternatives lists up to {ALTERNATIVE_LIMIT} other serious keys, strongest first, or []. "
    "Keep rationale under 120 words, cite concrete event indexes, tool arguments/errors, or "
    "verifier lines, and explain why the strongest alternative is weaker. "
    "Do not provide hidden chain-of-thought; provide only the concise evidence summary."
)


def normalize_choice_response(
    data: dict[str, Any], questions: dict[str, dict[str, Any]]
) -> ProviderResponse:
    """Validate answers and return them in the internal shape."""
    answers = data.get("answers")
    if not isinstance(answers, dict):
        raise ProviderError("response is missing an answers object")
    missing = set(questions) - set(answers)
    if missing:
        raise ProviderError(f"response omitted answers: {sorted(missing)}")
    normalized: dict[str, dict[str, Any]] = {}
    for question_id, question in questions.items():
        answer = answers[question_id]
        if not isinstance(answer, dict) or answer.get("type") != question.get("type"):
            raise ProviderError(f"invalid answer type for {question_id!r}")
        normalized[question_id] = (
            _normalize_choice(question_id, answer, question.get("criteria"))
            if question.get("type") == "choice"
            else _normalize_structured(question_id, answer, question.get("schema"))
        )
    return ProviderResponse(model=str(data.get("model", "unknown")), answers=normalized)


def _normalize_choice(question_id: str, answer: dict[str, Any], criteria: Any) -> dict[str, Any]:
    choice = answer.get("choice")
    if not isinstance(criteria, dict) or choice not in criteria:
        raise ProviderError(f"invalid Choice answer for {question_id!r}")
    rationale = answer.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ProviderError(f"Choice rationale for {question_id!r} must be text")
    listed = answer.get("alternatives")
    listed = listed if isinstance(listed, list) else []
    alternatives = dict.fromkeys(
        key for key in listed if isinstance(key, str) and key in criteria and key != choice
    )
    return {
        "type": "choice",
        "choice": choice,
        "alternatives": list(alternatives)[:ALTERNATIVE_LIMIT],
        "rationale": rationale.strip(),
    }


def _normalize_structured(question_id: str, answer: dict[str, Any], schema: Any) -> dict[str, Any]:
    if answer.get("type") != "structured" or not isinstance(answer.get("value"), dict):
        raise ProviderError(f"invalid Structured answer for {question_id!r}")
    if not isinstance(schema, dict) or not schema:
        raise ProviderError(f"structured question {question_id!r} has no schema")
    missing = set(schema) - set(answer["value"])
    if missing:
        raise ProviderError(
            f"Structured answer for {question_id!r} omitted fields: {sorted(missing)}"
        )
    _validate_schema_shape(answer["value"], schema, path=question_id)
    return {"type": "structured", "value": answer["value"]}


def _validate_schema_shape(value: Any, schema: Any, *, path: str) -> None:
    """Validate required object and list fields."""
    if isinstance(schema, dict):
        if not isinstance(value, dict):
            raise ProviderError(f"Structured answer field {path!r} must be an object")
        missing = set(schema) - set(value)
        if missing:
            raise ProviderError(
                f"Structured answer field {path!r} omitted fields: {sorted(missing)}"
            )
        for key, child_schema in schema.items():
            _validate_schema_shape(value[key], child_schema, path=f"{path}.{key}")
    elif isinstance(schema, list):
        if not isinstance(value, list):
            raise ProviderError(f"Structured answer field {path!r} must be a list")
        if schema:
            for index, item in enumerate(value):
                _validate_schema_shape(item, schema[0], path=f"{path}[{index}]")


def _json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ProviderError(f"model response was not JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ProviderError("model response JSON must be an object")
    return data


@dataclass
class OpenAIProvider:
    """Judge behind any OpenAI-compatible endpoint."""

    api_key: str | None = None
    model: str | None = None
    endpoint: str | None = None
    api_key_env: str | None = None
    api: str = "chat_completions"
    timeout_seconds: float = 300.0
    max_retries: int = 3
    backoff_seconds: float = 0.5
    request_options: dict[str, Any] = field(default_factory=dict)
    supports_images: bool = True
    defer_api_key: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        from autofa.defaults import default_value
        from autofa.llm.endpoints import completion_options

        self.api_key_env = self.api_key_env or default_value("hub_api_key_env")
        self.endpoint = self.endpoint or default_value("hub_base_url")
        completion_options({}, self.request_options)
        if not self.model:
            raise ProviderError("OpenAI-compatible provider requires a model name")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.api not in {"chat_completions", "responses"}:
            raise ValueError("api must be chat_completions or responses")
        if not self.defer_api_key:
            self.ensure_api_key()

    def ensure_api_key(self) -> None:
        """Resolve credentials only when this provider has pending work."""
        from autofa.config import global_env_path
        from autofa.llm.endpoints import NO_AUTH, NO_ENDPOINT_HELP
        from autofa.llm.hub import API_KEY_ENV_FALLBACKS, get_api_key

        if not str(self.endpoint or "").strip():
            raise ProviderError(NO_ENDPOINT_HELP)
        self.api_key = self.api_key or get_api_key(str(self.api_key_env))
        if not self.api_key and self.api_key_env != NO_AUTH:
            aliases = (
                API_KEY_ENV_FALLBACKS
                if self.api_key_env == "INFERENCE_HUB_API_KEY"
                else ()
            )
            raise ProviderError(
                f"{' / '.join((str(self.api_key_env), *aliases))} is not set. "
                f"Save {self.api_key_env}=<your-key> in {global_env_path()} or export it."
            )

    def evaluate(
        self,
        state: str | dict[str, Any] | list[Any],
        questions: dict[str, dict[str, Any]],
        *,
        images: list[dict[str, Any]] | None = None,
    ) -> ProviderResponse:
        if not questions:
            raise ProviderError("at least one question is required")
        from autofa.llm.endpoints import completion_options
        from autofa.llm.hub import _image_blocks, chat_completions_safe, make_hub_client

        self.ensure_api_key()
        client = make_hub_client(
            api_key=self.api_key, base_url=str(self.endpoint), api_key_env=self.api_key_env,
        )
        if hasattr(client, "with_options"):
            client = client.with_options(max_retries=0)
        user_state = state
        policy = None
        if isinstance(state, dict) and isinstance(state.get("judge_instructions"), str):
            user_state = {key: value for key, value in state.items() if key != "judge_instructions"}
            policy = state["judge_instructions"]
        user_text = "Return json.\n" + json.dumps(
            {"state": user_state, "questions": questions}, ensure_ascii=False
        )
        image_blocks = _image_blocks(images)
        if image_blocks:
            user_content: Any = (
                [
                    {"type": "input_text", "text": user_text},
                    *[
                        {
                            "type": "input_image",
                            "image_url": block["image_url"]["url"],
                        }
                        for block in image_blocks
                    ],
                ]
                if self.api == "responses"
                else [{"type": "text", "text": user_text}, *image_blocks]
            )
        else:
            user_content = user_text
        messages = [
            {
                "role": "system",
                "content": _ANSWER_FORMAT
                + (f"\n\nJudging policy:\n{policy}" if policy else ""),
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]
        total_usage: Counter[str] = Counter()
        for attempt in range(self.max_retries + 1):
            content: str | None = None
            try:
                if self.api == "responses":
                    options = completion_options({}, self.request_options)
                    for key in ("max_tokens", "max_completion_tokens"):
                        if key in options:
                            options["max_output_tokens"] = options.pop(key)
                    configured_effort = options.pop("reasoning_effort", None)
                    if configured_effort is not None:
                        options["reasoning"] = {"effort": configured_effort}
                    from autofa.util.cancellation import check_cancelled
                    from autofa.util.model_budget import reserve_model_call

                    # Keep cancellation and dispatch accounting without changing
                    # the taxonomy request's API, options, or response contract.
                    check_cancelled()
                    reserve_model_call()
                    try:
                        completion = client.responses.create(
                            model=str(self.model),
                            input=messages,
                            timeout=self.timeout_seconds,
                            **options,
                        )
                    finally:
                        check_cancelled()
                else:
                    options = completion_options({}, self.request_options)
                    completion = chat_completions_safe(
                        client,
                        model=str(self.model),
                        messages=messages,
                        timeout=self.timeout_seconds,
                        response_format={"type": "json_object"},
                        **options,
                    )
                total_usage.update(_usage(getattr(completion, "usage", None)))
                content = (
                    _responses_content(completion)
                    if self.api == "responses"
                    else _completion_content(completion)
                )
                parsed = _json_object(content)
                response = normalize_choice_response(
                    {
                        "model": getattr(completion, "model", None) or self.model,
                        "answers": parsed.get("answers"),
                    },
                    questions,
                )
                return ProviderResponse(
                    model=response.model,
                    answers=response.answers,
                    usage=dict(total_usage),
                )
            except Exception as exc:  # noqa: BLE001 - normalize endpoint and response failures
                status_code = getattr(exc, "status_code", None)
                if attempt == self.max_retries or (
                    not isinstance(exc, ProviderError)
                    and status_code in {400, 401, 403, 404, 422}
                ):
                    if isinstance(exc, ProviderError):
                        raise
                    raise ProviderError(
                        f"judge request failed ({type(exc).__name__}): {exc}"
                    ) from exc
                if isinstance(exc, ProviderError):
                    messages = [
                        *messages[:2],
                        *(
                            [{"role": "assistant", "content": content}]
                            if isinstance(content, str) and content.strip()
                            else []
                        ),
                        {
                            "role": "user",
                            "content": (
                                f"Your previous JSON failed validation: {exc}. "
                                "Return a complete replacement JSON object, not a patch. "
                                f"It must contain answers for every question id: "
                                f"{sorted(questions)}. Preserve each question's declared type. "
                                "A structured answer must use "
                                '{"type":"structured","value":{...}} and include every schema '
                                "field."
                            ),
                        },
                    ]
                time.sleep(self.backoff_seconds * (2**attempt))
        raise AssertionError("chat judge retry loop exhausted unexpectedly")


def _usage(usage: Any) -> dict[str, int]:
    counts = {
        "input_tokens": (
            getattr(usage, "prompt_tokens", None)
            if getattr(usage, "prompt_tokens", None) is not None
            else getattr(usage, "input_tokens", None)
        ),
        "output_tokens": (
            getattr(usage, "completion_tokens", None)
            if getattr(usage, "completion_tokens", None) is not None
            else getattr(usage, "output_tokens", None)
        ),
    }
    return {key: value for key, value in counts.items() if isinstance(value, int)}


def _completion_content(completion: Any) -> str:
    choices = getattr(completion, "choices", None)
    if not choices:
        raise ProviderError("chat judge response is missing choices")
    content = getattr(getattr(choices[0], "message", None), "content", None)
    if not isinstance(content, str) or not content.strip():
        raise ProviderError("chat judge response is missing message content")
    return content


def _responses_content(response: Any) -> str:
    if getattr(response, "error", None):
        raise ProviderError(f"responses judge returned an error: {response.error}")
    if getattr(response, "status", None) in {"failed", "cancelled", "incomplete"}:
        detail = getattr(response, "incomplete_details", None)
        raise ProviderError(f"responses judge is incomplete: {detail or response.status}")
    text = getattr(response, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()
    parts = []
    for item in getattr(response, "output", None) or []:
        for content in getattr(item, "content", None) or []:
            kind = (
                content.get("type")
                if isinstance(content, dict)
                else getattr(content, "type", None)
            )
            value = (
                content.get("text")
                if isinstance(content, dict)
                else getattr(content, "text", None)
            )
            if kind == "output_text" and isinstance(value, str):
                parts.append(value)
    result = "".join(parts).strip()
    if not result:
        raise ProviderError("responses judge returned no output text")
    return result
