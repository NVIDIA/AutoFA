# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Responses transport with the Chat completion contract used by existing callers."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


class ResponsesUnsupportedRequest(ValueError):
    """A valid Chat request whose contract cannot be represented by this adapter."""


def _input(messages: list[dict]) -> list[dict]:
    result = []
    for message in messages:
        role, content = message.get("role"), message.get("content")
        if message.get("name") is not None or role == "function":
            raise ResponsesUnsupportedRequest("Named/legacy function messages require Chat "
                                              "Completions")
        if role == "tool":
            result.append({"type": "function_call_output", "call_id": message["tool_call_id"],
                           "output": content})
            continue
        if role not in {"user", "assistant", "system", "developer"}:
            raise ValueError("Responses transport requires supported message roles")
        if content is not None and not (role == "assistant" and content == ""
                                        and message.get("tool_calls")):
            if isinstance(content, list):
                blocks = []
                for block in content:
                    if block.get("type") in {"text", "input_text"}:
                        blocks.append({"type": "input_text", "text": block["text"]})
                    elif block.get("type") == "image_url":
                        image = block["image_url"]
                        detail = {"detail": image["detail"]} if "detail" in image else {}
                        blocks.append({"type": "input_image", "image_url": image["url"], **detail})
                    else:
                        raise ResponsesUnsupportedRequest(
                            "Responses transport cannot preserve this content block; "
                            "select Chat Completions for this request")
                content = blocks
            elif not isinstance(content, str):
                raise ValueError("Message content must be text or supported content blocks")
            result.append({"role": role, "content": content})
        for call in message.get("tool_calls") or []:
            if call.get("type") != "function":
                raise ResponsesUnsupportedRequest(
                    "Responses transport supports function tool calls only")
            result.append({"type": "function_call", "call_id": call["id"],
                           "name": call["function"]["name"],
                           "arguments": call["function"]["arguments"]})
    return result


def _options(options: dict) -> dict:
    result = deepcopy(options)
    if result.pop("stream", False) or result.pop("n", 1) != 1:
        raise ResponsesUnsupportedRequest(
            "Responses adapter supports one non-streaming completion; "
            "select Chat Completions for streaming or multiple choices")
    unsupported = {"stop", "seed", "logprobs", "top_logprobs", "frequency_penalty",
                   "presence_penalty", "logit_bias", "functions", "function_call",
                   "stream_options", "audio", "modalities", "prediction"}.intersection(result)
    if unsupported:
        raise ResponsesUnsupportedRequest("Responses transport cannot preserve Chat option(s) "
                                          + ", ".join(sorted(unsupported))
                                          + "; select Chat Completions")
    extra = result.pop("extra_body", None)
    extra = deepcopy(extra) if isinstance(extra, dict) else {}
    # OpenAI-compatible providers often put reasoning controls in extra_body.
    for key in ("reasoning_effort", "reasoning", "response_format"):
        if key in extra:
            result[key] = extra.pop(key)
    budgets = [key for key in ("max_tokens", "max_completion_tokens", "max_output_tokens")
               if key in result]
    if len(budgets) > 1:
        raise ValueError("Specify only one output token limit")
    if budgets:
        result["max_output_tokens"] = result.pop(budgets[0])
    if "reasoning_effort" in result:
        effort = result.pop("reasoning_effort")
        reasoning = dict(result.get("reasoning") or {})
        reasoning["effort"] = effort
        result["reasoning"] = reasoning
    if isinstance(result.get("reasoning"), dict) and "enabled" in result["reasoning"]:
        reasoning = dict(result["reasoning"])
        enabled = reasoning.pop("enabled")
        if enabled is False:
            reasoning["effort"] = "none"
        elif enabled is True and "effort" not in reasoning:
            reasoning["effort"] = "medium"
        result["reasoning"] = reasoning
    if "response_format" in result:
        format_ = result.pop("response_format")
        if not isinstance(format_, dict):
            raise ValueError("Responses adapter requires a JSON response_format object")
        if format_.get("type") == "json_schema":
            format_ = {"type": "json_schema", **format_["json_schema"]}
        text = dict(result.get("text") or {})
        text["format"] = format_
        result["text"] = text
    if "tools" in result:
        tools = []
        for tool in result["tools"]:
            if tool.get("type") != "function":
                raise ResponsesUnsupportedRequest("Responses adapter supports function tools only")
            tools.append({"type": "function", **tool["function"]})
        result["tools"] = tools
    choice = result.get("tool_choice")
    if isinstance(choice, dict):
        if choice.get("type") != "function":
            raise ResponsesUnsupportedRequest(
                "Responses adapter supports function tool choices only")
        result["tool_choice"] = {"type": "function", "name": choice["function"]["name"]}
    result.setdefault("store", False)
    if extra:
        result["extra_body"] = extra
    return result


def _completion(response: Any, model: str) -> Any:
    from openai.types.chat import ChatCompletion

    data = response if isinstance(response, dict) else response.model_dump()
    status = data.get("status")
    if status not in {"completed", "incomplete"}:
        raise RuntimeError("Responses API did not complete the generation request")
    output = data.get("output")
    if not isinstance(output, list):
        raise RuntimeError("Responses API returned no output list")
    text, refusals, calls = [], [], []
    for item in output:
        if item.get("type") == "function_call":
            calls.append({"id": item["call_id"], "type": "function",
                          "function": {"name": item["name"], "arguments": item["arguments"]}})
        elif item.get("type") == "message":
            if item.get("status") not in {None, "completed"} and status == "completed":
                raise RuntimeError("Responses API returned an unfinished output message")
            for block in item.get("content") or []:
                if block.get("type") == "output_text":
                    text.append(block["text"])
                elif block.get("type") == "refusal":
                    refusals.append(block.get("refusal") or "Request refused")
                else:
                    raise RuntimeError("Responses API returned an unsupported answer block")
        elif item.get("type") != "reasoning":
            raise RuntimeError("Responses API returned an unsupported output type")
    reason = "tool_calls" if calls else "stop"
    if status == "incomplete":
        details = data.get("incomplete_details") or {}
        reason = "content_filter" if details.get("reason") == "content_filter" else "length"
    usage = data.get("usage")
    converted_usage = None
    if usage:
        converted_usage = {"prompt_tokens": usage.get("input_tokens", 0),
                           "completion_tokens": usage.get("output_tokens", 0),
                           "total_tokens": usage.get("total_tokens", 0)}
        if usage.get("input_tokens_details"):
            converted_usage["prompt_tokens_details"] = usage["input_tokens_details"]
        if usage.get("output_tokens_details"):
            converted_usage["completion_tokens_details"] = usage["output_tokens_details"]
    return ChatCompletion(
        id=data.get("id") or "response", object="chat.completion",
        created=int(data.get("created_at") or 0), model=data.get("model") or model,
        choices=[{"index": 0, "finish_reason": reason, "message": {
            "role": "assistant", "content": "".join(text) if text else None,
            "refusal": "\n".join(refusals) if refusals else None,
            "tool_calls": calls or None,
        }}], usage=converted_usage,
    )


def create_response_completion(client: Any, *, model: str, messages: list[dict],
                               usage_role: str = "detector", **options: Any) -> Any:
    from autofa.llm.capabilities import APIUnavailable, invoke_api
    from autofa.llm.hub import _compatible_completion

    wire = _options(options)
    input_ = _input(messages)

    def request() -> Any:
        resource = getattr(client, "responses", None)
        if not callable(getattr(resource, "create", None)):
            raise APIUnavailable("responses")
        response = _compatible_completion(resource.create, model=model, input=input_,
                                          _usage_client=client, _usage_role=usage_role,
                                          _usage_api="responses", **wire)
        return _completion(response, model)

    return invoke_api(client, model, "responses", request)
