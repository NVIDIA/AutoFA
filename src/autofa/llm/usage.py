# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Observe API accounting without consuming bodies or retaining model inputs."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack
from typing import Any
from uuid import uuid4

from autofa.util.usage_values import field as _field
from autofa.util.usage_values import nonnegative_number as _money
from autofa.util.usage_values import normalize_tokens, usage_endpoint

log = logging.getLogger(__name__)


def normalized_usage(response: Any) -> dict[str, int]:
    return normalize_tokens(_field(response, "usage"))


def _endpoint(client: Any) -> str | None:
    return usage_endpoint(getattr(client, "base_url", None))


class _Observation:
    def __init__(self, context: dict, *, client: Any, model: str, api: str,
                 options: dict[str, Any]) -> None:
        self.context = context
        self.event_id = "api:" + uuid4().hex
        self.endpoint = _endpoint(client)
        self.model = model
        self.api = api
        self.health_settings = {key: options[key] for key in (
            "reasoning_effort", "reasoning", "temperature", "max_tokens", "max_output_tokens",
        ) if key in options}
        self.source_id = None
        self.usage: dict[str, int] = {}
        self.cost = None
        self._last_record = None
        self._admission: ExitStack | None = None
        effort = options.get("reasoning_effort") or _field(options.get("reasoning"), "effort")
        self.effort = effort if effort in {
            "none", "minimal", "low", "medium", "high", "xhigh", "max",
        } else None
        self.note = f"{api} API dispatch; absent counters or dollar amounts are unknown."
        retries = getattr(client, "max_retries", 0)
        if type(retries) is int and retries > 0:
            self.note += " SDK-internal retries may not expose usage for failed attempts."

    def health(self, error: BaseException | None = None) -> None:
        try:
            from autofa.session.detector_service import observe_dispatch

            observe_dispatch({**self.context, "usage_event_id": self.event_id},
                             model=self.model, endpoint=self.endpoint,
                             api=self.api, settings=self.health_settings, error=error)
        except Exception:
            log.debug("Could not record detector service health", exc_info=False)

    def finish_dispatch(self) -> None:
        """Release shared admission once the lazy response is consumed or closed."""
        admission, self._admission = self._admission, None
        if admission is not None:
            admission.close()

    def __del__(self):
        # Abandoned responses cannot retain a process-local reservation forever.
        # Do not consume or close provider bodies from an observer destructor.
        try:
            self.finish_dispatch()
        except Exception:
            pass

    def observe(self, response: Any = None, *, status: str = "complete") -> None:
        """Upsert this physical dispatch; subsequent stream snapshots are not new calls."""
        try:
            from autofa.session.usage import record_usage

            self.usage.update(normalized_usage(response))
            # Only explicit USD fields are sufficiently unambiguous. Generic
            # `cost`/credits fields have provider-specific units and stay unknown.
            for value in (_field(response, "cost_usd"),
                          _field(_field(response, "usage"), "cost_usd")):
                cost = _money(value)
                if cost is not None:
                    self.cost = cost
            source_id = _field(response, "id")
            if isinstance(source_id, str) and len(source_id) <= 200:
                self.source_id = source_id
            fingerprint = (tuple(sorted(self.usage.items())), self.cost, self.source_id, status)
            if fingerprint == self._last_record:
                return
            record_usage(
                self.context, event_id=self.event_id, provider="openai_compatible",
                model=self.model, usage=self.usage or None, cost_usd=self.cost,
                cost_kind="reported" if self.cost is not None else "unknown",
                source_id=self.source_id, endpoint=self.endpoint, effort=self.effort,
                status=status, note=self.note,
            )
            self._last_record = fingerprint
        except Exception:
            # Accounting must not change detector behavior or print private bodies.
            log.debug("Could not record API usage", exc_info=False)

    def wrap(self, value: Any) -> Any:
        self.observe(value)
        # Only lazy SDK surfaces need a proxy. Ordinary model return values keep
        # identity, and raw/stream bodies remain untouched until the caller reads.
        if callable(getattr(value, "parse", None)) or callable(
            getattr(value, "get_final_completion", None)
        ) or (not isinstance(value, (Mapping, str, bytes))
              and callable(getattr(value, "__next__", None))):
            return _ObservedResponse(value, self)
        return value

    def observe_event(self, value: Any) -> None:
        self.observe(value)
        # SDK .stream() yields events, whose chunk/snapshot carries usage.
        for name in ("chunk", "snapshot", "completion"):
            nested = _field(value, name)
            if _field(nested, "usage") is not None:
                self.observe(nested)


class _ObservedResponse:
    """Delegate lazy SDK operations, observing only values the caller requested."""

    def __init__(self, response: Any, observation: _Observation) -> None:
        self._response = response
        self._observation = observation

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._response, name)
        if name in {"parse", "get_final_completion", "until_done"} and callable(value):
            def observed(*args: Any, **kwargs: Any) -> Any:
                try:
                    result = value(*args, **kwargs)
                except BaseException as exc:
                    if isinstance(exc, Exception):
                        self._observation.health(exc)
                    self._observation.observe(_field(exc, "completion"), status="failed")
                    self._observation.finish_dispatch()
                    raise
                if name == "until_done":
                    self._observation.observe(_field(self._response, "current_completion_snapshot"))
                    self._observation.finish_dispatch()
                    return result
                wrapped = self._observation.wrap(result)
                if not isinstance(wrapped, _ObservedResponse):
                    self._observation.finish_dispatch()
                return wrapped
            return observed
        return value

    def __iter__(self) -> Iterator[Any]:
        try:
            for value in self._response:
                self._observation.observe_event(value)
                yield value
        except BaseException as exc:
            if isinstance(exc, Exception):
                self._observation.health(exc)
            self._observation.observe(status="failed")
            raise
        finally:
            self._observation.finish_dispatch()

    def __next__(self) -> Any:
        try:
            value = next(self._response)
        except StopIteration:
            self._observation.finish_dispatch()
            raise
        except BaseException as exc:
            if isinstance(exc, Exception):
                self._observation.health(exc)
            self._observation.observe(status="failed")
            self._observation.finish_dispatch()
            raise
        self._observation.observe_event(value)
        return value

    def __enter__(self) -> Any:
        try:
            entered = self._response.__enter__()
        except BaseException:
            self._observation.finish_dispatch()
            raise
        return self if entered is self._response else self._observation.wrap(entered)

    def __exit__(self, *args: Any) -> Any:
        try:
            return self._response.__exit__(*args)
        finally:
            self._observation.finish_dispatch()

    def close(self) -> Any:
        try:
            close = getattr(self._response, "close", None)
            if not callable(close):
                close = getattr(getattr(self._response, "http_response", None), "close", None)
            return close() if callable(close) else None
        finally:
            self._observation.finish_dispatch()


def observed_call(create: Callable[..., Any], *args: Any, client: Any, usage_model: str,
                  api: str, role: str = "detector", **kwargs: Any) -> Any:
    """Wrap an already-budgeted actual dispatch, before result interpretation."""
    context = None
    try:
        from autofa.session.usage import capture_context

        context = capture_context(role=role)
        if context is not None:
            context["dispatch_identity"] = {"kind": "api", "role": role,
                                            "provider": "openai_compatible", "model": usage_model,
                                            "endpoint": _endpoint(client)}
        observation = (_Observation(context, client=client, model=usage_model,
                                    api=api, options=kwargs)
                       if context is not None else None)
    except Exception:
        observation = None
    from autofa.session.admission import paid_dispatch

    with ExitStack() as admission:
        admission.enter_context(paid_dispatch(context))
        if observation is None:
            return create(*args, **kwargs)
        try:
            result = create(*args, **kwargs)
        except BaseException as exc:
            if isinstance(exc, Exception):
                observation.health(exc)
            response = _field(exc, "completion") or _field(exc, "body")
            observation.observe(response, status="failed")
            raise
        observation.health()
        wrapped = observation.wrap(result)
        if isinstance(wrapped, _ObservedResponse):
            observation._admission = admission.pop_all()
        return wrapped
