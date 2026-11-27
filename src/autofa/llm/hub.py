# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenAI-compatible LLM client; historical hub names remain import-compatible."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from contextlib import ExitStack, contextmanager
from contextvars import copy_context
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from autofa.defaults import default_value
from autofa.llm.endpoints import (
    NO_AUTH,
    NO_ENDPOINT_HELP,
    completion_options,
    validate_base_url,
)
from autofa.util.model_errors import model_error_diagnostic

log = logging.getLogger(__name__)

# Tried when the configured variable is unset: NVIDIA agent sandboxes and CI export
# the same hub key as INFERENCE_API_KEY.
API_KEY_ENV_FALLBACKS = ("INFERENCE_API_KEY",)

# Startup probe: tiny completion to catch bad keys / missing models / slow routes.
DEFAULT_PROBE_TIMEOUT_S = 20.0
DEFAULT_PROBE_SLOW_S = 8.0
_PROBE_PROMPT = 'Reply with exactly the word OK. No punctuation.'

_RESTRICTED_KWARGS = ("temperature", "top_p")


def resolve_api_key_env(env_name: str | None = None) -> str | None:
    """Resolve the configured key; only the legacy default has an alias."""
    if env_name is None:
        env_name = default_value("hub_api_key_env")
    if env_name == NO_AUTH:
        return None
    aliases = API_KEY_ENV_FALLBACKS if env_name == "INFERENCE_HUB_API_KEY" else ()
    for name in (env_name, *aliases):
        if os.environ.get(name, "").strip():
            return name
    return None


def get_api_key(env_name: str | None = None) -> str | None:
    name = resolve_api_key_env(env_name)
    return os.environ[name].strip() if name else None


def make_hub_client(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    api_key_env: str | None = None,
) -> Any:
    """Return a client for any OpenAI-compatible Chat Completions API root."""
    base_url = default_value("hub_base_url") if base_url is None else base_url
    if isinstance(base_url, str) and not base_url.strip():
        raise RuntimeError(NO_ENDPOINT_HELP)
    base_url = validate_base_url(base_url)
    if api_key_env is None:
        api_key_env = default_value("hub_api_key_env")
    from autofa.config import global_env_path, load_dotenv

    if api_key is None and api_key_env != NO_AUTH:
        # Programmatic sessions and resumed runs need the same shared credentials
        # as the CLI, including when launched from a fresh worktree.
        load_dotenv()
    key = api_key or get_api_key(api_key_env)
    if not key and api_key_env != NO_AUTH:
        aliases = API_KEY_ENV_FALLBACKS if api_key_env == "INFERENCE_HUB_API_KEY" else ()
        tried = " / ".join((api_key_env, *aliases))
        raise RuntimeError(
            f"{tried} not set — required for LLM-based detectors. "
            f"Save {api_key_env}=<your-key> in {global_env_path()} "
            "to configure every worktree, or export the variable in your shell."
        )
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError("openai package required for OpenAI-compatible LLM calls") from e
    if api_key_env == NO_AUTH:
        # OpenAI-compatible servers with authentication disabled ignore this
        # placeholder. Never use a real environment key for this explicit mode.
        return OpenAI(api_key="unused", base_url=base_url)
    return OpenAI(api_key=key, base_url=base_url)


def chat_completions_safe(
    client: Any,
    *,
    model: str,
    messages: list[dict],
    usage_role: str | None = None,
    **kwargs: Any,
) -> Any:
    """Adapt explicit parameter rejections without a provider/model registry.

    Retry only an unsupported sampling parameter or a requested token-field
    rename. Authentication, rate limits and other failures are not reinterpreted.
    Explicit request options can avoid these compatibility retries altogether.
    Raw callers opt into discovery accounting with ``usage_role``; unrelated
    workflows must not inherit a saved runtime's or enclosing scope's ledger.
    """
    completions = client.chat.completions
    if getattr(completions, "_autofa_adapts_completion_options", False) is True:
        # The detector wrapper applies endpoint settings before adaptation. Its
        # marker also passes through evidence-capture proxies, whose create()
        # must still observe the final response. Do not nest retry loops.
        return completions.create(model=model, messages=messages, **kwargs)
    return _compatible_completion(
        completions.create, model=model, messages=messages,
        _usage_client=client, _usage_role=usage_role, **kwargs,
    )


def _compatible_completion(create: Callable[..., Any], *args: Any,
                           _usage_client: Any = None, _usage_role: str | None = None,
                           _usage_api: str = "chat_completions", **kwargs: Any) -> Any:
    """Adapt the already merged wire options; never reapply rejected defaults."""
    from autofa.util.cancellation import check_cancelled

    if "max_tokens" in kwargs and "max_completion_tokens" in kwargs:
        raise ValueError("Specify only one token limit parameter")
    adjusted: set[str] = set()
    while True:
        check_cancelled()
        try:
            from autofa.util.model_budget import reserve_model_call

            reserve_model_call()
            if _usage_role is None:
                result = create(*args, **kwargs)
            else:
                from autofa.llm.usage import observed_call

                result = observed_call(create, *args, client=_usage_client,
                                       usage_model=kwargs.get("model", "unknown"),
                                       api=_usage_api, role=_usage_role, **kwargs)
            _check_response_cancelled(result)
            return result
        except Exception as exc:
            # A routed request can bypass the raw Chat wrapper's finally block.
            # Drain the request, then honor cancellation before interpreting its
            # error or attempting a compatibility retry.
            check_cancelled()
            if getattr(exc, "status_code", None) not in {400, 422}:
                raise
            body = getattr(exc, "body", None)
            error = body.get("error", body) if isinstance(body, dict) else {}
            error = error if isinstance(error, dict) else {}
            param = error.get("param") or getattr(exc, "param", None)
            code = error.get("code") or getattr(exc, "code", None)
            message = str(error.get("message") or exc).lower()
            unsupported = code in {"unsupported_parameter", "unsupported_value"} or any(
                word in message for word in ("not supported", "unsupported", "not support")
            )
            if not unsupported or not isinstance(param, str) or param in adjusted:
                raise
            if param in _RESTRICTED_KWARGS and param in kwargs:
                if code == "unsupported_value" and not any(
                    word in message for word in ("default", "not supported", "not support")
                ):
                    raise
                kwargs.pop(param)
            elif param == "max_tokens" and param in kwargs and "max_completion_tokens" in message:
                kwargs["max_completion_tokens"] = kwargs.pop(param)
            elif (param == "max_completion_tokens" and param in kwargs
                  and "max_tokens" in message):
                kwargs["max_tokens"] = kwargs.pop(param)
            else:
                raise
            adjusted.add(param)


def _check_response_cancelled(response: Any) -> None:
    """Release a response whose ownership cancellation prevents us from returning."""
    from asyncio import CancelledError

    from autofa.util.cancellation import check_cancelled

    try:
        check_cancelled()
    except CancelledError:
        # Stream/APIResponse exposes close(); LegacyAPIResponse exposes only
        # http_response. Neither requires consuming or parsing a streaming body.
        close = getattr(response, "close", None)
        if not callable(close):
            close = getattr(getattr(response, "http_response", None), "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                # Cleanup must not turn cancellation into a detector error.
                log.debug("Failed to close cancelled LLM response", exc_info=True)
        raise


def _provider_refusal() -> None:
    """A provider refusal is unknown evidence, never a detector negative."""
    from autofa.core.sandbox import DetectorAbstained

    raise DetectorAbstained(
        "Classifier provider refused or filtered the request; no label available")


def _refused_choice(choice: Any) -> bool:
    message = choice.message
    if (getattr(choice, "finish_reason", None) == "content_filter"
            or getattr(message, "refusal", None)):
        return True
    content = getattr(message, "content", None)
    return isinstance(content, list) and any(
        (block.get("type") if isinstance(block, dict) else getattr(block, "type", None))
        in {"refusal", "content_filter"} for block in content)


def completion_text(response: Any) -> str:
    """Missing, refused or truncated output must never turn into a cached label."""
    choices = getattr(response, "choices", None)
    if not choices:
        raise RuntimeError("LLM endpoint returned no completion choices")
    choice = choices[0]
    message = choice.message
    if _refused_choice(choice):
        _provider_refusal()
    if getattr(choice, "finish_reason", None) in {"length", "tool_calls"}:
        raise RuntimeError("LLM completion is incomplete; check the model output token budget")
    content = getattr(message, "content", None)
    # Native reasoning/reasoning_content fields are intentionally not answers.
    # Some compatible servers instead return typed content blocks.
    if isinstance(content, list):
        parts = []
        for block in content:
            kind = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
            if kind in {"refusal", "content_filter"}:
                _provider_refusal()
            if kind in {"reasoning", "thinking"}:
                continue
            value = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
            if kind not in {"text", "output_text"} or not isinstance(value, str):
                raise RuntimeError("LLM endpoint returned an unsupported content block")
            parts.append(value)
        content = "".join(parts)
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("LLM endpoint returned no text; check model and output token budget")
    content = content.strip()
    # Local reasoning models may place a marked reasoning prefix in content.
    # Strip only leading, delimited blocks; never search prose for a label or
    # remove tags embedded inside an evidence quote / JSON string.
    while match := re.match(r"<(think|thinking|analysis)>", content, re.IGNORECASE):
        end = re.search(rf"</{match[1]}>", content, re.IGNORECASE)
        if end is None:
            raise RuntimeError("LLM endpoint returned incomplete reasoning without a final answer")
        content = content[end.end():].strip()
    final = re.fullmatch(r"<final>\s*(.*?)\s*</final>", content, re.IGNORECASE | re.DOTALL)
    if final:
        content = final[1].strip()
    elif content.lower().startswith("<final>"):
        raise RuntimeError("LLM endpoint returned an incomplete final answer")
    if not content:
        raise RuntimeError("LLM endpoint returned reasoning without a final answer")
    return content


_INSUFFICIENT_EVIDENCE = "__autofa_insufficient_evidence__"
_WINDOW_INSTRUCTIONS = (
    "AutoFA supplied exact source fragments and separate analyzer window metadata below. "
    "Omitted ranges and fragment boundaries were created by the analyzer, not the author. "
    "Do not treat omissions as broken code, an incomplete original reply, or proof of absence. "
    "Decide from the retained evidence only when it establishes the claim; if missing "
    "content is required to decide, answer " + _INSUFFICIENT_EVIDENCE + "."
)


def _insufficient_evidence() -> None:
    from autofa.core.sandbox import DetectorAbstained

    raise DetectorAbstained("Required evidence is unavailable in the selected text window")


@dataclass(frozen=True)
class _ClassifyEvidence:
    text: str
    incomplete: bool


def _classify_evidence(
    text: str,
    *,
    focus: str = "auto",
    max_chars: int = 16_000,
    require_complete: bool = False,
) -> _ClassifyEvidence:
    """Pick a classify window under an explicit focus policy.

    Prefer passing the full surface into ``classify`` and setting ``focus=`` —
    do not hand-slice ``text[-4000:]`` in detector source.
    """
    from autofa.llm.windowing import text_window_evidence

    # Raw trajectory JSON is denser; keep a slightly tighter default budget.
    if focus in (None, "", "auto") and text and text.lstrip().startswith(("[", "{")):
        max_chars = min(max_chars, 8_000)
    evidence = text_window_evidence(text, focus=focus or "auto", max_chars=max_chars)
    incomplete = not evidence["complete"]
    if incomplete and require_complete:
        _insufficient_evidence()
    if not incomplete:
        return _ClassifyEvidence(text, False)
    return _ClassifyEvidence(
        "--- analyzer window metadata ---\n"
        + json.dumps(evidence["analyzer_window_metadata"], ensure_ascii=False)
        + "\n--- exact source fragments ---\n"
        + json.dumps(evidence["source_fragments"], ensure_ascii=False),
        True,
    )


def _cache_key(model: str, prompt: str, text: str) -> str:
    import hashlib

    h = hashlib.sha1()
    h.update(model.encode())
    h.update(b"\0")
    h.update(prompt.encode())
    h.update(b"\0")
    h.update(text.encode())
    return h.hexdigest()


def _image_blocks(images: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Accept verified local image receipts; never fetch caller-provided remote URLs."""
    if not images:
        return []
    if not isinstance(images, list) or len(images) > 16:
        raise ValueError("images must be a list of at most 16 artifact image receipts")
    result = []
    for value in images:
        if not isinstance(value, dict) or value.get("status") != "available":
            raise ValueError("All artifact images must be available; abstain on missing evidence")
        url = value.get("data_url")
        if not isinstance(url, str) or not url.startswith((
            "data:image/png;base64,", "data:image/jpeg;base64,",
            "data:image/webp;base64,", "data:image/gif;base64,",
        )):
            raise ValueError("images must contain local encoded image bytes")
        if len(url) > 17 * 1024 * 1024:
            raise ValueError("Artifact image exceeds delivery limit")
        try:
            data = base64.b64decode(url.split(",", 1)[1], validate=True)
        except ValueError as exc:
            raise ValueError("Invalid artifact image encoding") from exc
        if not data or hashlib.sha256(data).hexdigest() != value.get("sha256"):
            raise ValueError("Artifact image checksum does not match receipt")
        result.append({"type": "image_url", "image_url": {"url": url}})
    return result


@dataclass(frozen=True)
class HubProbeResult:
    """Outcome of a one-shot hub connectivity / model probe."""

    ok: bool
    model: str
    latency_s: float | None = None
    error: str | None = None
    reply: str | None = None
    slow: bool = False
    label_api: str | None = None
    text_api: str | None = None
    diagnostic: dict[str, str | int] | None = None


def humanize_hub_error(exc: BaseException) -> str:
    """Turn OpenAI/httpx exceptions into a short operator-facing message."""
    from autofa.llm.capabilities import APIUnavailable

    if isinstance(exc, APIUnavailable):
        return str(exc)
    status = getattr(exc, "status_code", None)
    if status is None:
        body = getattr(exc, "body", None)
        if isinstance(body, dict):
            status = body.get("status") or body.get("code")
    code = getattr(exc, "code", None)
    msg = str(exc).strip() or type(exc).__name__
    # Prefer the first line; hub errors are often multi-line JSON blobs.
    msg = msg.split("\n", 1)[0].strip()
    if len(msg) > 160:
        msg = msg[:157] + "…"
    if status in {401, 403} or code in {"invalid_api_key", "unauthorized"}:
        return f"auth failed (HTTP {status or '?'}) — check API key"
    if status == 404 or (isinstance(msg, str) and "model" in msg.lower() and "not" in msg.lower()):
        return f"model unavailable (HTTP {status or 404}): {msg}"
    if status == 429:
        return f"rate limited (HTTP 429): {msg}"
    if status is not None:
        return f"HTTP {status}: {msg}"
    if isinstance(exc, (TimeoutError, FuturesTimeout)):
        return msg if msg and msg != type(exc).__name__ else "timed out"
    return msg


class _CancellableCompletions:
    """Keep raw calls under endpoint configuration and tool cancellation."""

    _autofa_adapts_completion_options = True

    def __init__(self, completions: Any, request_options: Callable[[], dict[str, Any]],
                 *, deferred: bool = False, route: Callable[..., Any] | None = None,
                 special: bool = False,
                 compatibility: Callable[[str], None] | None = None,
                 usage_client: Any = None, usage_role: str = "detector") -> None:
        self._completions = completions
        self._request_options = request_options
        self._deferred = deferred
        self._route = route
        self._special = special
        self._compatibility = compatibility
        self._usage_client = usage_client
        self._usage_role = usage_role

    def __getattr__(self, name: str) -> Any:
        if name in {"with_raw_response", "with_streaming_response"}:
            return _CancellableCompletions(
                getattr(self._completions, name), self._request_options,
                deferred=name == "with_streaming_response", route=self._route, special=True,
                compatibility=self._compatibility,
                usage_client=self._usage_client, usage_role=self._usage_role,
            )
        return getattr(self._completions, name)

    def create(self, *args: Any, **kwargs: Any) -> Any:
        return self._request("create", *args, **kwargs)

    def parse(self, *args: Any, **kwargs: Any) -> Any:
        return self._request("parse", *args, **kwargs)

    def stream(self, *args: Any, **kwargs: Any) -> Any:
        return self._request("stream", *args, **kwargs)

    def _request(self, method: str, *args: Any, **kwargs: Any) -> Any:
        from autofa.util.cancellation import check_cancelled

        check_cancelled()
        request = getattr(self._completions, method)
        options = completion_options(kwargs, self._request_options())
        if self._route is not None:
            if method != "create" or self._deferred or self._special:
                if self._compatibility is None:
                    raise ValueError("Responses routing supports ordinary non-streaming create(); "
                                     "select Chat Completions for parse or streaming SDK surfaces")
                self._compatibility("The requested parse/stream/raw SDK surface requires Chat "
                                    "Completions compatibility.")
            else:
                return self._route(*args, **options)
        from autofa.llm.routing import chat_options

        options = chat_options(options)
        if self._deferred or method == "stream":
            return _deferred_completion(request, *args, _usage_client=self._usage_client,
                                        _usage_role=self._usage_role, **options)
        result = None
        try:
            result = _compatible_completion(request, *args, _usage_client=self._usage_client,
                                            _usage_role=self._usage_role, **options)
            return result
        finally:
            # An already-sent synchronous request must drain, but neither its
            # late result nor its error may let a cancelled detector continue.
            _check_response_cancelled(result)


@contextmanager
def _deferred_completion(request: Callable[..., Any], *args: Any,
                         **kwargs: Any) -> Iterator[Any]:
    """Adapt lazy SDK requests when entered, retaining their response lifetime."""
    from autofa.util.cancellation import check_cancelled

    with ExitStack() as stack:
        try:
            # These SDK methods send the request in __enter__, not when the
            # context manager is constructed. Retry that boundary exactly once
            # per rejected option, and close any opened response on cancellation.
            yield _compatible_completion(
                lambda *a, **kw: stack.enter_context(request(*a, **kw)),
                *args, **kwargs,
            )
        finally:
            check_cancelled()


class _CancellableChat:
    def __init__(self, chat: Any, request_options: Callable[[], dict[str, Any]],
                 *, deferred: bool = False, route: Callable[..., Any] | None = None,
                 special: bool = False,
                 compatibility: Callable[[str], None] | None = None,
                 usage_client: Any = None, usage_role: str = "detector") -> None:
        self._chat = chat
        self._request_options = request_options
        self._deferred = deferred
        self._route = route
        self._special = special
        self._compatibility = compatibility
        self._usage_client = usage_client
        self._usage_role = usage_role

    def __getattr__(self, name: str) -> Any:
        if name == "completions":
            return _CancellableCompletions(
                self._chat.completions, self._request_options, deferred=self._deferred,
                route=self._route, special=self._special,
                compatibility=self._compatibility,
                usage_client=self._usage_client, usage_role=self._usage_role,
            )
        if name in {"with_raw_response", "with_streaming_response"}:
            return _CancellableChat(
                getattr(self._chat, name), self._request_options,
                deferred=name == "with_streaming_response", route=self._route, special=True,
                compatibility=self._compatibility,
                usage_client=self._usage_client, usage_role=self._usage_role,
            )
        return getattr(self._chat, name)


class HubLLM:
    """Passed into detectors as ``llm`` / ``llm_client``.

    Supports:
      - ``llm.classify(prompt, text, focus="auto") -> str``
      - ``llm.chat.completions.create(...)`` (OpenAI-compatible)
      - ``chat_completions_safe(llm, model=..., messages=...)``
    """

    available = True
    # Cache keyed by client namespace, model, options, prompt and text. Shared across
    # the LLM eval worker pool, so mutation is locked — the eviction path below is a
    # read-then-pop that races without it.
    _CLASSIFY_CACHE: dict[str, str] = {}
    _CLASSIFY_CACHE_MAX = 10_000
    _CLASSIFY_LOCK = threading.Lock()

    def __init__(
        self,
        client: Any,
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        request_options: dict[str, Any] | None = None,
        api: str | None = None,
        usage_role: str = "detector",
    ) -> None:
        if model is None:
            model = default_value("feature_model")
        if api is None:
            api = default_value("feature_api")
        if request_options is None:
            request_options = default_value("feature_request_options")
        if api not in {"auto", "responses", "chat_completions", "decisions"}:
            raise ValueError("Detector API must be auto, responses, chat_completions or decisions")
        self._client = client
        self.usage_role = usage_role
        self.api = api
        self.model = model
        self.temperature = temperature
        self.max_tokens = 64 if max_tokens is None else max_tokens
        self._auto_budget = max_tokens is None
        self.request_options = dict(request_options or {})
        completion_options({}, self.request_options)
        self._cache_namespace = uuid4().hex
        # Shared with quote extractors for this endpoint/model adapter. Other
        # clients and models must discover their own default output budgets.
        self._budget_state = {"floor": 0}
        self._compatibility_state = {"used": False}
        self._budget_lock = threading.Lock()
        # Expose OpenAI surface so generated code can use llm_client.chat...
        self.chat = _CancellableChat(
            client.chat, lambda: self.request_options,
            usage_client=client, usage_role=usage_role,
            route=self._raw_completion if api != "chat_completions" else None,
            compatibility=self._compatibility_route if api in {"auto", "decisions"} else None,
        )

    def _compatibility_route(self, reason: str) -> None:
        with self._budget_lock:
            if self._compatibility_state["used"]:
                return
            self._compatibility_state["used"] = True
        log.warning("Using Chat Completions compatibility for a request: %s", reason)

    def _raw_completion(self, *args: Any, **kwargs: Any) -> Any:
        from autofa.llm.routing import routed_completion

        if args:
            raise ValueError("Routed completion requests require keyword arguments")
        return routed_completion(self._client, api=self.api, usage_role=self.usage_role,
                                 on_compatibility=self._compatibility_route, **kwargs)

    def routing_identity(self) -> dict:
        from autofa.llm.capabilities import routing_identity

        return {**routing_identity(self._client, self.model, self.api),
                "chat_compatibility_used": self._compatibility_state["used"]}

    @property
    def text_api(self) -> str | None:
        from autofa.llm.capabilities import cached_capability

        if self.api == "chat_completions":
            return "chat_completions"
        available = cached_capability(self._client, self.model, "responses")
        if available is True:
            return "responses"
        if available is False and self.api != "responses":
            return "chat_completions"
        if available is None and self._compatibility_state["used"]:
            return "chat_completions"
        return None

    @property
    def label_api(self) -> str | None:
        from autofa.llm.capabilities import cached_capability

        if self.api == "decisions":
            return "decisions" if cached_capability(self._client, self.model, "decisions") else None
        if self.api == "auto":
            available = cached_capability(self._client, self.model, "decisions")
            if available is True:
                return "decisions"
            if available is None:
                return None
        return self.text_api

    def _complete_text(self, messages: list[dict], options: dict[str, Any], *,
                       auto_budget: bool = True, usage_role: str | None = None,
                       **kwargs: Any) -> str:
        """Learn a 4096-token default and allow one final bounded 8192-token rescue.

        Reasoning tokens can exhaust a tiny label/probe budget before any answer
        is produced. User-specified limits remain hard limits. Never cache or
        parse a truncated answer, even if it happens to contain a valid label.
        """
        from autofa.llm.routing import routed_completion

        explicit_budget = any(k in getattr(self, "request_options", {}) for k in
                              ("max_tokens", "max_completion_tokens", "max_output_tokens"))
        budget_key = next((k for k in ("max_tokens", "max_completion_tokens", "max_output_tokens")
                           if k in options), None)
        adaptive = (auto_budget and getattr(self, "_auto_budget", False)
                    and not explicit_budget and budget_key)
        if adaptive:
            with self._budget_lock:
                floor = self._budget_state["floor"]
            options = {**options, budget_key: max(options[budget_key], floor)}
        def complete(request_options):
            return routed_completion(
                self._client, api=self.api, model=self.model, messages=messages,
                usage_role=usage_role or self.usage_role,
                on_compatibility=self._compatibility_route, **request_options, **kwargs,
            )

        response = complete(options)
        for limit in (4096, 8192):
            choices = getattr(response, "choices", None)
            if not (adaptive and options[budget_key] < limit and choices
                    and getattr(choices[0], "finish_reason", None) == "length"
                    and not _refused_choice(choices[0])):
                continue
            log.info("Retrying truncated LLM answer model=%s with %s output tokens",
                     self.model, limit)
            response = complete({**options, budget_key: limit})
            # Learn only the accepted default. The final rescue remains local
            # to this request; explicit limits and refusals never enter here.
            if limit == 4096:
                with self._budget_lock:
                    self._budget_state["floor"] = 4096
        return completion_text(response)

    def probe(
        self,
        *,
        timeout_s: float = DEFAULT_PROBE_TIMEOUT_S,
        slow_s: float = DEFAULT_PROBE_SLOW_S,
    ) -> HubProbeResult:
        """One tiny completion to verify the configured model answers.

        Catches bad keys, unknown model IDs, and very slow routes before
        discovery spends minutes on LLM detectors. Does not mutate the
        classify cache.
        """
        timeout_s = max(0.1, float(timeout_s))
        slow_s = max(0.01, float(slow_s))

        def remaining_timeout() -> float:
            remaining = timeout_s - (time.monotonic() - t0)
            if remaining <= 0:
                raise TimeoutError(f"timed out after {timeout_s:.0f}s")
            return remaining

        def _call() -> str:
            if self.api in {"auto", "decisions"}:
                from autofa.llm.capabilities import APIUnavailable
                from autofa.llm.decisions import create_decision

                try:
                    answers = create_decision(
                        self._client, model=self.model, input="The connection works.",
                        questions=[{"type": "choice", "instructions": "Choose OK.",
                                    "choices": [{"value": "OK"}, {"value": "FAILED"}]}],
                        timeout=remaining_timeout(), usage_role="classifier_probe",
                    )
                    if answers[0].get("choice") != "OK":
                        raise RuntimeError("Decision API probe did not return OK")
                except APIUnavailable:
                    if self.api == "decisions":
                        raise
            return self._complete_text(
                [{"role": "user", "content": _PROBE_PROMPT}],
                completion_options(
                    # Responses requires at least 16 output tokens. Keep the
                    # smaller Chat probe and respect explicit caller budgets.
                    {"temperature": 0.0,
                     "max_tokens": min(8 if self.api == "chat_completions" else 16,
                                       self.max_tokens)},
                    self.request_options,
                ),
                timeout=remaining_timeout(), usage_role="classifier_probe",
            )

        t0 = time.monotonic()
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            fut = pool.submit(copy_context().run, _call)
            reply = fut.result(timeout=timeout_s)
        except FuturesTimeout:
            latency = time.monotonic() - t0
            return HubProbeResult(
                ok=False,
                model=self.model,
                latency_s=latency,
                error=f"timed out after {timeout_s:.0f}s",
                label_api=self.label_api, text_api=self.text_api,
            )
        except Exception as e:  # noqa: BLE001
            latency = time.monotonic() - t0
            log.warning("LLM endpoint probe failed model=%s: %s", self.model, e)
            return HubProbeResult(
                ok=False,
                model=self.model,
                latency_s=latency,
                error=humanize_hub_error(e),
                diagnostic=model_error_diagnostic(e),
                label_api=self.label_api, text_api=self.text_api,
            )
        finally:
            # Don't block startup on a hung HTTP call after we already timed out.
            pool.shutdown(wait=False, cancel_futures=True)

        latency = time.monotonic() - t0
        return HubProbeResult(
            ok=True,
            model=self.model,
            latency_s=latency,
            reply=reply,
            slow=latency >= slow_s,
            label_api=self.label_api, text_api=self.text_api,
        )

    def classify(
        self,
        prompt: str,
        text: str,
        *,
        focus: str = "auto",
        max_chars: int | None = None,
        images: list[dict[str, Any]] | None = None,
        require_complete: bool = False,
    ) -> str:
        """Classify ``text`` under a deliberate evidence window.

        Pass the full surface; set ``focus`` from the claim
        (``head`` / ``tail`` / ``ends`` / ``full`` / ``auto``). Do not pre-slice
        with ``text[-4000:]`` — that recreates envelope blindness on derived fields.
        Optional images are available artifact_image receipts with verified encoded bytes.
        Vision support is model-dependent; failed image requests never retry as text-only.
        ``require_complete`` abstains before dispatch when windowing omitted input.
        Otherwise exact fragments may support a decision, with an explicit
        insufficient-evidence escape instead of forced negative classification.
        """
        from autofa.util.cancellation import check_cancelled

        check_cancelled()
        kwargs: dict[str, Any] = {"focus": focus, "require_complete": require_complete}
        if max_chars is not None:
            kwargs["max_chars"] = int(max_chars)
        evidence = _classify_evidence(text, **kwargs)
        window = evidence.text
        if evidence.incomplete:
            prompt = f"{prompt}\n\n{_WINDOW_INSTRUCTIONS}"
        options = completion_options(
            {"temperature": self.temperature, "max_tokens": self.max_tokens},
            getattr(self, "request_options", {}),
        )
        # Equal model IDs at different endpoints/accounts need separate results.
        image_blocks = _image_blocks(images)
        def cache_key() -> str:
            identity = self.routing_identity()
            namespace = json.dumps([str(getattr(self._client, "base_url", "")),
                                    identity["transport_contract_version"],
                                    identity["endpoint_account_namespace"], self.api, self.text_api,
                                    self._cache_namespace, options],
                                   sort_keys=True)
            return _cache_key(namespace + self.model, prompt, window + (
                "\0images\0" + json.dumps(image_blocks, sort_keys=True) if image_blocks else ""
            ))
        key = cache_key()
        with self._CLASSIFY_LOCK:
            cached = self._CLASSIFY_CACHE.get(key)
        if cached is not None:
            return cached
        messages = [
            {
                "role": "user",
                "content": ([{"type": "text", "text": f"{prompt}\n\n--- text ---\n{window}"},
                             *image_blocks] if image_blocks else
                            f"{prompt}\n\n--- text ---\n{window}"),
            }
        ]
        out = self._complete_text(messages, options)
        if (evidence.incomplete
                and out.strip().strip(".!").strip().strip("\"'`").casefold()
                == _INSUFFICIENT_EVIDENCE):
            _insufficient_evidence()
        with self._CLASSIFY_LOCK:
            if len(self._CLASSIFY_CACHE) >= self._CLASSIFY_CACHE_MAX:
                # Drop the oldest entry (dict preserves insertion order).
                self._CLASSIFY_CACHE.pop(next(iter(self._CLASSIFY_CACHE)), None)
            self._CLASSIFY_CACHE[cache_key()] = out
        return out

    def localise(self, prompt: str, text: str, *, timeout: float | None = None,
                 **kwargs: Any) -> str:
        """Extract quotes with room for JSON; independent of the detector label call.

        Uses the detector model, never the Claude behavior-purity judge. The
        distinct extraction prompt also separates cache entries from labels.
        """
        budget = 1200 if self._auto_budget else self.max_tokens
        client = self._client
        if timeout is not None:
            if timeout <= 0:
                raise TimeoutError("Optional highlighting request deadline expired")
            # Bound optional work independently of the main detector's transport.
            # SDK retries could otherwise multiply the report's time budget.
            client = client.with_options(timeout=timeout, max_retries=0)
        extractor = HubLLM(client, model=self.model, temperature=0.0, max_tokens=budget,
                           request_options=self.request_options, api=self.api,
                           usage_role="highlighting")
        extractor._auto_budget = self._auto_budget
        extractor._cache_namespace = self._cache_namespace
        extractor._budget_state = self._budget_state
        extractor._budget_lock = self._budget_lock
        extractor._compatibility_state = self._compatibility_state
        return extractor.classify(prompt, text, **kwargs)

    def classify_options(
        self,
        text: str,
        question: str,
        options: list[str] | None = None,
        max_tokens: int | None = None,
        *,
        focus: str = "auto",
        max_chars: int | None = None,
        images: list[dict[str, Any]] | None = None,
        require_complete: bool = False,
    ) -> str:
        opts = options or ["yes", "no"]
        kwargs: dict[str, Any] = {"focus": focus, "require_complete": require_complete}
        if max_chars is not None:
            kwargs["max_chars"] = int(max_chars)
        evidence = _classify_evidence(text, **kwargs)
        if evidence.incomplete:
            if _INSUFFICIENT_EVIDENCE not in opts:
                opts = [*opts, _INSUFFICIENT_EVIDENCE]
            question = f"{question}\n\n{_WINDOW_INSTRUCTIONS}"
        if self.api in {"auto", "decisions"}:
            from autofa.llm.capabilities import APIUnavailable

            try:
                answer = self.decide([{
                    "type": "choice", "instructions": question,
                    "choices": [{"value": option} for option in opts],
                }], evidence.text, focus="full", max_chars=len(evidence.text), images=images)[0]
            except APIUnavailable:
                if self.api == "decisions":
                    raise
            else:
                if answer["type"] == "refusal":
                    _provider_refusal()
                if answer["choice"] == _INSUFFICIENT_EVIDENCE:
                    _insufficient_evidence()
                return answer["choice"]
        opt_str = " | ".join(f"'{o}'" for o in opts)
        prompt = (
            f"{question}\n\nReturn EXACTLY one of: {opt_str}. No explanation.\n\n"
            f"--- text ---\n{evidence.text}"
        )
        image_blocks = _image_blocks(images)
        content = ([{"type": "text", "text": prompt}, *image_blocks]
                   if image_blocks else prompt)
        messages = [{"role": "user", "content": content}]
        for format_attempt in range(2):
            raw = self._complete_text(
                messages,
                completion_options(
                    {"temperature": 0.0, "max_tokens": max_tokens or self.max_tokens},
                    self.request_options,
                ),
                auto_budget=max_tokens is None,
            )
            # Accept exact options only, allowing ordinary surrounding punctuation.
            normalized = raw.strip().strip(".!").strip().strip("\"'`").casefold()
            for candidate in (raw.casefold(), normalized):
                matches = [o for o in opts if o.casefold() == candidate]
                if len(matches) == 1:
                    if matches[0] == _INSUFFICIENT_EVIDENCE:
                        _insufficient_evidence()
                    return matches[0]
                if matches:
                    raise RuntimeError(
                        "LLM endpoint did not return exactly one of the requested options")
            # One formatting retry for a completed but malformed short label.
            # Never guess its meaning, retry a valid option, or parse explanations.
            # Refusals/truncation remain governed by _complete_text above.
            if (format_attempt or len(normalized) > 64 or not normalized.isascii()
                    or not normalized.replace("_", "").isalpha()):
                break
            log.info("Retrying malformed choice label model=%s", self.model)
            messages = [*messages, {"role": "assistant", "content": raw}, {
                "role": "user",
                "content": (
                    "The previous reply did not exactly match an allowed label. "
                    "Reassess the same supplied evidence and return EXACTLY one of: "
                    f"{opt_str}. No explanation."
                ),
            }]
        raise RuntimeError("LLM endpoint did not return exactly one of the requested options")

    def decide(self, questions: list[dict[str, Any]], text: str, *, focus: str = "auto",
               max_chars: int | None = None,
               images: list[dict[str, Any]] | None = None,
               require_complete: bool = False) -> list[dict]:
        """Native Decisions: ordered predicate, choice or score answers on shared evidence.

        Returns validated answer objects, including per-question refusals. Uses
        this model and endpoint regardless of the default classify_options API.
        Images must be verified artifact receipts, as for classify().
        """
        from autofa.llm.decisions import create_decision

        if self.api == "responses":
            raise ValueError("decide() requires native Decisions probabilities. "
                             "Use classify_options "
                             "for fixed labels with Responses, or select Auto / Decisions.")
        kwargs: dict[str, Any] = {"focus": focus, "require_complete": require_complete}
        if max_chars is not None:
            kwargs["max_chars"] = int(max_chars)
        evidence = _classify_evidence(text, **kwargs)
        window = evidence.text
        if evidence.incomplete:
            questions = [
                {**q, "instructions": q.get("instructions", "") + "\n\n"
                 + _WINDOW_INSTRUCTIONS.replace(
                     "answer " + _INSUFFICIENT_EVIDENCE,
                     "return a refusal for that question",
                 )}
                for q in questions
            ]
        blocks = _image_blocks(images)
        evidence = ([{"role": "user", "content": [
            {"type": "input_text", "text": window},
            *[{"type": "input_image", "image_url": block["image_url"]["url"]} for block in blocks],
        ]}] if blocks else window)
        return create_decision(self._client, model=self.model, input=evidence, questions=questions,
                               usage_role=self.usage_role)
