# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conservative discovery fallback choices from the installed Codex catalog.

This is an explicit application preference order, not a universal quality ranking.
It favors the known frontier tier, then successive workhorse generations, then
the smaller tier. Catalog ordering, display labels and arbitrary model-name
numbers are not evidence that another model is a suitable downgrade.
"""

from __future__ import annotations

import re

from autofa.harness.model_catalog import HarnessModel, ModelCatalog
from autofa.util.cancellation import check_cancelled

_PREFERENCE = (
    "gpt-6-astra",
    "gpt-6.1-sol",
    "gpt-6-sol",
    "gpt-5.6-sol",
    "gpt-6-luna",
)
_POSITION = {value: index for index, value in enumerate(_PREFERENCE)}
_NAMESPACE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


def _known_identity(value: str) -> tuple[str, str] | None:
    namespace, separator, model = value.rpartition("/")
    if not separator:
        namespace, model = "", value
    elif not _NAMESPACE.fullmatch(namespace):
        return None
    return (namespace, model) if model in _POSITION else None


def _identity(model: HarnessModel) -> tuple[str, str] | None:
    identity = _known_identity(model.resolved_model or model.value)
    if identity is None:
        return None
    # A gateway alias may resolve to a bare concrete name. Keep its advertised
    # selector's route instead of accidentally switching to another provider.
    namespace, separator, _alias = model.value.rpartition("/")
    if separator:
        if not _NAMESPACE.fullmatch(namespace) or identity[0] not in ("", namespace):
            return None
        return namespace, identity[1]
    return identity


def lower_codex_models(catalog: ModelCatalog, current_model: str) -> tuple[HarnessModel, ...]:
    """Return advertised lower preferences, preserving the selected gateway namespace.

Only exact known IDs (or catalog-resolved aliases) participate. Unknown models,
specialist models, failed catalogs and a primary absent from the catalog return
no choices. This helper never edits the configured primary or loads a judge model.
The caller owns capacity-error detection and the bounded retry/switch policy.
"""
    check_cancelled()
    if catalog.error or (current := catalog.find(current_model)) is None:
        return ()
    identity = _identity(current)
    if identity is None:
        return ()
    namespace, name = identity
    # Deduplicate aliases with deterministic preference for a concrete selector.
    selected: dict[str, HarnessModel] = {}
    for candidate in catalog.models:
        candidate_id = _identity(candidate)
        if (
            candidate_id is None
            or candidate_id[0] != namespace
            or _POSITION[candidate_id[1]] <= _POSITION[name]
        ):
            continue
        candidate_name = candidate_id[1]
        concrete = f"{namespace}/{candidate_name}" if namespace else candidate_name
        previous = selected.get(candidate_name)
        if previous is None or (candidate.value != concrete, candidate.value) < (
            previous.value != concrete, previous.value,
        ):
            selected[candidate_name] = candidate
    return tuple(selected[key] for key in _PREFERENCE if key in selected)


def effort_for_model(model: HarnessModel, requested: str | None) -> str | None:
    """Preserve an advertised effort, otherwise use the model's advertised default."""
    check_cancelled()
    if model.supports_effort is False:
        return None
    if requested not in (None, "default") and requested in model.effort_levels:
        return requested
    return model.default_effort if model.default_effort in model.effort_levels else None
