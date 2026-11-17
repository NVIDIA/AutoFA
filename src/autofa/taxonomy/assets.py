# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source-addressed visual evidence for taxonomy judges."""

from __future__ import annotations

import base64
from copy import deepcopy
from pathlib import Path
from typing import Any

ASSET_INVESTIGATION_PROTOCOL = "taxonomy-asset-investigation-v1"
MAX_ASSET_REQUESTS = 4
MAX_ASSET_ROUNDS = 3
MAX_TOTAL_IMAGE_BYTES = 24 * 1024 * 1024

_DESCRIPTOR_FIELDS = (
    "artifact_idx",
    "name",
    "kind",
    "role",
    "origin",
    "mime",
    "mime_type",
    "size_bytes",
    "sha256",
    "completeness",
    "lane",
    "event_idx",
    "part_idx",
    "call_idx",
)
_VISUAL_KINDS = {"image", "pdf", "document", "presentation", "spreadsheet"}
_VISUAL_SUFFIXES = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".pdf",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


class AssetEvidenceError(ValueError):
    """A judge requested invalid or over-budget visual evidence."""


def artifact_descriptors(artifact_source: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return bounded artifact metadata with no bytes, URLs, or local paths."""
    if not artifact_source:
        return []
    from autofa import trace
    from autofa.data.artifact_evidence import is_reference_artifact
    from autofa.data.execution_policy import is_evaluation_record, sanitize_metadata

    descriptors = []
    for raw in trace.artifacts(artifact_source):
        if (
            type(raw.get("artifact_idx")) is not int
            or is_evaluation_record(raw)
            or is_reference_artifact(raw)
        ):
            continue
        clean = sanitize_metadata(raw)
        descriptor = {
            key: deepcopy(clean.get(key))
            for key in _DESCRIPTOR_FIELDS
            if clean.get(key) is not None
        }
        descriptor["renderable"] = _is_visual(descriptor)
        descriptors.append(descriptor)
    return sorted(descriptors, key=lambda item: int(item["artifact_idx"]))


def task_input_indexes(
    artifact_source: dict[str, Any], catalog: list[dict[str, Any]]
) -> list[int]:
    """Visual artifacts attached to the task prompt before the model's first reply."""
    from autofa import trace

    roles: dict[tuple[str, int], str] = {}
    first_reply: dict[str, int] = {}
    for event in trace.events(artifact_source):
        event_idx = event.get("event_idx")
        if type(event_idx) is not int:
            continue
        lane = event.get("lane") or ""
        role = str(event.get("role") or "").lower()
        roles[(lane, event_idx)] = role
        if role in {"assistant", "agent"} and event_idx < first_reply.get(lane, event_idx + 1):
            first_reply[lane] = event_idx
    selected = []
    for item in catalog:
        lane, event_idx = item.get("lane") or "", item.get("event_idx")
        if (
            item.get("renderable")
            and type(event_idx) is int
            and roles.get((lane, event_idx)) in {"user", "system"}
            and event_idx < first_reply.get(lane, event_idx + 1)
        ):
            selected.append(item["artifact_idx"])
    return selected[:MAX_ASSET_REQUESTS]


def asset_request_question() -> dict[str, Any]:
    """Structured request used by a text pass before any visual bytes are fetched."""
    return {
        "type": "structured",
        "instructions": (
            "Request only visual artifacts whose contents are necessary to classify the "
            "failure causally. Use the exact artifact_idx from available_assets and a 1-based "
            "page. Do not request an artifact already listed in asset_investigation. Return an "
            "empty requests list when the visible evidence is sufficient or after all needed "
            "assets have been supplied. Briefly state why each requested asset is necessary."
        ),
        "schema": {
            "requests": [
                {
                    "artifact_idx": "Integer artifact identity.",
                    "page": "Positive 1-based page; use 1 for an image.",
                    "reason": "Brief evidence need.",
                }
            ]
        },
    }


def investigate_requests(
    artifact_source: dict[str, Any],
    catalog: list[dict[str, Any]],
    requests: Any,
    *,
    requested: set[tuple[int, int]],
    existing_images: list[dict[str, Any]] | None = None,
    trigger: str = "judge_request",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Resolve requested artifacts through the verified artifact store."""
    if not isinstance(requests, list):
        raise AssetEvidenceError("asset requests must be a list")
    available = {item["artifact_idx"]: item for item in catalog}
    pending: list[tuple[int, int, str]] = []
    for index, request in enumerate(requests):
        if not isinstance(request, dict):
            raise AssetEvidenceError(f"asset request {index} must be an object")
        artifact_idx, page = request.get("artifact_idx"), request.get("page")
        reason = request.get("reason")
        if type(artifact_idx) is not int or artifact_idx not in available:
            raise AssetEvidenceError(f"asset request {index} has an unknown artifact_idx")
        if type(page) is not int or page < 1:
            raise AssetEvidenceError(f"asset request {index} page must be a positive integer")
        if not isinstance(reason, str) or not reason.strip():
            raise AssetEvidenceError(f"asset request {index} reason must be nonempty text")
        if not available[artifact_idx].get("renderable"):
            raise AssetEvidenceError(f"artifact {artifact_idx} is not renderable visual evidence")
        identity = (artifact_idx, page)
        if identity not in requested and all(item[:2] != identity for item in pending):
            pending.append((artifact_idx, page, reason.strip()))
    if len(requested) + len(pending) > MAX_ASSET_REQUESTS:
        raise AssetEvidenceError(
            f"a classification may inspect at most {MAX_ASSET_REQUESTS} artifact images"
        )

    from autofa import trace

    images: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    total_bytes = sum(
        _decoded_image_size(image.get("data_url"), context="previously rendered artifact")
        for image in existing_images or []
    )
    for artifact_idx, page, reason in pending:
        result = trace.artifact_image(artifact_source, artifact_idx, page=page)
        requested.add((artifact_idx, page))
        receipt = {
            "protocol": ASSET_INVESTIGATION_PROTOCOL,
            "artifact": deepcopy(available[artifact_idx]),
            "page": page,
            "trigger": trigger,
            "reason": reason,
            "status": result.get("status"),
            "source_identity": deepcopy(result.get("source_identity")),
            "artifact_sha256": result.get("artifact_sha256"),
            "render_sha256": result.get("sha256"),
            "transform_version": result.get("transform_version"),
            "locator": result.get("locator"),
            "limitations": deepcopy(result.get("limitations") or []),
            **(
                {"unavailable_reason": result.get("reason")}
                if result.get("status") != "available"
                else {}
            ),
        }
        receipts.append(receipt)
        if result.get("status") != "available":
            continue
        data_url = result.get("data_url")
        if not isinstance(data_url, str) or ";base64," not in data_url:
            raise AssetEvidenceError(
                f"artifact {artifact_idx} has no verified rendered image bytes"
            )
        total_bytes += _decoded_image_size(data_url, context=f"artifact {artifact_idx}")
        if total_bytes > MAX_TOTAL_IMAGE_BYTES:
            raise AssetEvidenceError(
                "requested artifact images exceed the "
                f"{MAX_TOTAL_IMAGE_BYTES}-byte aggregate limit"
            )
        images.append(result)
    return images, receipts


def _decoded_image_size(data_url: Any, *, context: str) -> int:
    if not isinstance(data_url, str) or ";base64," not in data_url:
        raise AssetEvidenceError(f"{context} has no verified rendered image bytes")
    try:
        return len(base64.b64decode(data_url.split(";base64,", 1)[1], validate=True))
    except ValueError as exc:
        raise AssetEvidenceError(f"{context} has invalid rendered image bytes") from exc


def _is_visual(descriptor: dict[str, Any]) -> bool:
    kind = str(descriptor.get("kind") or "").lower()
    mime = str(descriptor.get("mime") or descriptor.get("mime_type") or "").lower()
    suffix = Path(str(descriptor.get("name") or "")).suffix.lower()
    return kind in _VISUAL_KINDS or mime.startswith("image/") or suffix in _VISUAL_SUFFIXES
