# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Separate execution evidence from external targets and transport credentials.

This is a metadata policy, not a text scrubber. Model-visible content, tool
arguments/results and schemas are evidence even when their keys happen to be
``reward`` or ``status``. Outside those content boundaries, external evaluation
fields are excluded. Execution status, timing, usage and provider extensions
remain available; rollout-level status/success are external target summaries.
"""

from __future__ import annotations

import json
import re
import urllib.parse
from copy import deepcopy
from typing import Any

from autofa.core.fields import LABEL_BANNED

_TARGET_FIELDS = LABEL_BANNED | {
    "reward", "rewards", "score", "scores", "grade", "grades", "graded",
    "correct", "correctness", "is_resolved", "tb_is_resolved",
    "expected_answer", "reference_answer", "ground_truth",
    "target", "targets", "outcome", "outcomes", "verdict", "verdicts",
    "grader", "grader_text", "grader_result", "grader_results", "grading",
    "judge", "judge_text", "judge_result", "judge_results", "scorer", "scoring",
    "scoring_details", "scoring_details_json", "hb_scoring_details_json",
    "evaluation", "evaluation_result", "evaluation_results", "eval_result", "eval_results",
    "tests_status", "fail_to_pass", "pass_to_pass", "pass_to_fail",
    "is_scored", "is_graded", "is_evaluated",
}
_ROLLOUT_TARGET_FIELDS = {"status", "success", "successful", "failed", "failure", "resolved"}
_CREDENTIAL_FIELDS = {
    "authorization", "proxy_authorization", "api_key", "apikey", "x_api_key",
    "access_token", "refresh_token", "id_token", "auth_token", "bearer_token",
    "password", "passwd", "secret", "client_secret", "credentials", "credential",
    "private_key", "cookie", "cookies", "set_cookie", "signed_url", "presigned_url",
    "x_amz_signature", "x_amz_security_token", "x_amz_credential", "x_goog_signature",
    "x_goog_credential",
}
_OPAQUE_FIELDS = {
    "content", "content_raw", "text", "message", "reasoning", "reasoning_content",
    "thinking", "thought", "summary", "arguments", "arguments_raw", "args", "args_full",
    "args_head", "input", "output", "output_raw", "instructions", "system", "prompt",
    "parameters", "parameters_raw", "input_schema", "input_schema_raw", "json_schema",
    "observation_raw",
    "def_json",
}
_EXECUTION_CONTAINERS = {
    "steps", "events", "tool_calls", "calls", "messages", "request", "response",
    "request_metadata", "response_metadata", "attempts", "artifacts",
}
_EVALUATION_ROLES = {"grader", "judge", "scorer", "verdict", "evaluation", "evaluator"}
# Model-server references of graders (NeMo Gym model_ref names such as an arena's
# judge_model, AA-Omniscience's genrm_model or AA-LCR's aalcr_v1_1_judge_model).
# A grader names its role as a whole word; a reference naming the policy never counts.
_EVALUATION_MODEL_WORDS = {"judge", "grader", "scorer", "evaluator", "genrm"}
_EVALUATION_MODEL_REFS = {"reward_model"}
# Such a reference as serialized JSON: a name or {"name": ...}, possibly JSON-encoded
# again (escaped quotes), in any case. A cheap superset of is_evaluation_model_ref for
# names of up to 100 characters around the role word, bounded so a long unquoted run
# cannot backtrack quadratically; the parsed records decide.
_MODEL_REF_TEXT = (
    r'model_ref\\*"\s*:\s*\\*"?(?:\{[^}]{0,200}?)?\\*"(?i:[^"]{0,100}?(?:'
    + "|".join(sorted(_EVALUATION_MODEL_WORDS))
    + r')[^"]{0,100}|reward[_-]model)\\*"'
)
_MODEL_REF_PATTERNS = (re.compile(_MODEL_REF_TEXT), re.compile(_MODEL_REF_TEXT.encode()))
_ENCODED_METADATA_FIELDS = {
    "metadata", "metrics", "attributes", "metadata_json", "metrics_json", "attributes_json",
    "request_params", "request_messages", "request_tools", "request_metadata", "response_metadata",
    "request", "response", "function", "function_call", "tool_calls", "tools", "agent",
    "msg_json",
}


def _key(value: Any) -> str:
    return str(value).lower().replace("-", "_")


def is_evaluation_table(name: str) -> bool:
    """Exclude explicitly named target tables, not generic execution results/metrics."""
    normalized = _key(name).removesuffix(".parquet")
    return normalized in {
        "score", "scores", "grade", "grades", "reward", "rewards", "label", "labels",
        "target", "targets", "outcome", "outcomes", "verdict", "verdicts",
        "grader", "graders", "judge", "judges", "scorer", "scorers", "scoring",
        "evaluation", "evaluations", "eval", "evals",
    } or normalized.startswith(("grader_", "judge_", "scorer_", "evaluation_", "eval_"))


def _object(value: Any) -> Any:
    """Decode a JSON-encoded object; captured provenance can be stored as a string."""
    if isinstance(value, str) and value.lstrip().startswith("{"):
        try:
            return json.loads(value)
        except (ValueError, RecursionError):
            return None
    return value


def _model_ref(value: Any) -> str:
    """Name of a model-server reference, given as a name or as ``{"name": ...}``."""
    value = _object(value)
    if isinstance(value, dict):
        value = value.get("name")
    return _key(value) if isinstance(value, str) else ""


def is_evaluation_model_ref(name: str) -> bool:
    """Whether a normalized model reference names a grader rather than the policy."""
    words = re.split(r"[\W_]+", name)
    return "policy" not in words and (name in _EVALUATION_MODEL_REFS or any(
        word in _EVALUATION_MODEL_WORDS for word in words
    ))


def mentions_evaluation_model(text: str | bytes) -> bool:
    """Whether serialized JSON can hold a call made through a grader's model reference.

    A cheap check before parsing: False rules such calls out, True still needs
    ``is_evaluation_record`` on the parsed records.
    """
    return _MODEL_REF_PATTERNS[isinstance(text, bytes)].search(text) is not None


def is_evaluation_record(row: Any) -> bool:
    """Identify explicitly labelled evaluation records, never generic failures.

    Only provenance fields are inspected. A model's text mentioning a judge or
    a tool returning ``status='failed'`` does not turn execution into a target.
    A model call made through a grader's ``model_ref`` (top level or in
    ``response_metadata``) is one too, such as an arena judge's comparison.
    """
    if not isinstance(row, dict):
        return False
    metadata = _object(row.get("response_metadata"))
    references = [row.get("model_ref")]
    if isinstance(metadata, dict):
        references.append(metadata.get("model_ref"))
    if any(is_evaluation_model_ref(_model_ref(reference)) for reference in references):
        return True
    for field in (
        "actor", "kind", "kind_raw", "role", "source", "lane", "lane_kind",
        "frm_lane", "to_lane", "asserted_by", "offered_by", "origin",
    ):
        value = row.get(field)
        if field == "source" and (
            not isinstance(value, str) or value.lower() not in _EVALUATION_ROLES
        ):
            continue  # An arbitrary source path is not a role/actor declaration.
        if isinstance(value, str) and any(
            part in _EVALUATION_ROLES for part in re.split(r"[\s/.:_-]+", value.lower())
        ):
            return True
    return False


def _observation(value: Any) -> Any:
    # ATIF wrappers carry execution metadata next to opaque tool content.
    decoded = value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except (ValueError, RecursionError):
            return value
    if isinstance(decoded, dict) and isinstance(decoded.get("results"), list) and all(
        isinstance(result, dict) and "source_call_id" in result
        for result in decoded["results"]
    ):
        return sanitize_metadata(decoded)
    return deepcopy(value)


def _message_container(value: Any) -> bool:
    """Distinguish provider message envelopes from arbitrary structured task data."""
    entries = value if isinstance(value, list) else [value]
    return bool(entries) and all(
        isinstance(entry, dict) and (
            isinstance(entry.get("role"), str)
            or entry.get("type") in {
                "message", "reasoning", "function_call", "function_call_output",
            }
        ) for entry in entries
    )


def _metadata_uri(value: Any) -> Any:
    """Strip bearer query parameters/userinfo only from transport metadata URLs."""
    if not isinstance(value, str):
        return deepcopy(value)
    try:
        parsed = urllib.parse.urlsplit(value)
        if parsed.scheme not in {"http", "https", "s3", "gs"}:
            return value
        query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
        clean = [(key, child) for key, child in query
                 if _key(key) not in _CREDENTIAL_FIELDS | {"token", "signature", "sig"}]
        netloc = parsed.netloc.rsplit("@", 1)[-1]
        if clean == query and netloc == parsed.netloc:
            return value
        return urllib.parse.urlunsplit(parsed._replace(
            netloc=netloc, query=urllib.parse.urlencode(clean),
        ))
    except ValueError:
        return value


def sanitize_metadata(value: Any, *, rollout: bool = False) -> Any:
    """Deep-copy metadata with targets/credentials excluded at known boundaries.

    ``rollout=True`` also excludes aggregate status/success/failure (including
    nested rollout metrics). Event, call, attempt and message containers switch
    back to execution context, where status and timing are useful evidence.
    Encoded metadata containers are decoded only at named metadata boundaries;
    arbitrary strings and model-visible payloads are never interpreted.
    """
    if isinstance(value, list):
        return [sanitize_metadata(entry, rollout=rollout) for entry in value
                if not is_evaluation_record(entry)]
    if not isinstance(value, dict):
        return deepcopy(value)
    result = {}
    for key, child in value.items():
        name = _key(key)
        if name in _TARGET_FIELDS or name in _CREDENTIAL_FIELDS:
            continue
        if rollout and name in _ROLLOUT_TARGET_FIELDS:
            continue
        if name in {"uri", "url", "endpoint", "base_url", "download_url"}:
            result[key] = _metadata_uri(child)
            continue
        if name == "message" and not {"source", "role"}.intersection(value) and (
            _message_container(child)
        ):
            # Chat choices wrap message objects. Their envelope metadata is
            # filterable; a trajectory step's message payload remains opaque.
            result[key] = sanitize_metadata(child)
            continue
        if name in _OPAQUE_FIELDS:
            result[key] = deepcopy(child)
            continue
        if name == "observation":
            result[key] = _observation(child)
            continue
        if is_evaluation_record(child):
            continue
        if name in _ENCODED_METADATA_FIELDS and isinstance(child, str):
            try:
                decoded = json.loads(child)
            except (ValueError, RecursionError):
                pass
            else:
                if isinstance(decoded, (dict, list)):
                    result[key] = json.dumps(
                        sanitize_metadata(decoded, rollout=rollout), ensure_ascii=False,
                    )
                    continue
        result[key] = sanitize_metadata(
            child, rollout=rollout and name not in _EXECUTION_CONTAINERS,
        )
    return result
