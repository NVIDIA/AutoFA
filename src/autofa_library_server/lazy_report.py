# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local dashboard snapshots whose full trajectories are read only on demand.

Descriptors are minted by the local catalog, never accepted as upload instructions.
They refer only to checksum-validated managed stores, not arbitrary filesystem paths.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from autofa.core.library import FunctionLibrary
from autofa.data.local_trace_store import resolve_store
from autofa.data.parquet_store import Dataset, make_item_id
from autofa.data.reference_data import (
    make_local_reference_data,
    normalize_reference_data,
    rollout_url,
)
from autofa.data.row_index import _row_is_correct
from autofa.reports.failure_cards import (
    _collect_example_ids,
    _compact_item,
    _item_filename,
    build_data,
    render_html,
)

from .library_identity import same_library_content

KEY_PREFIX = "local-report:"
ITEMS_BASE = "failure_cards_data/items"


class LazyReportAssets(list[tuple[str, bytes]]):
    """Trusted internal descriptors, distinguished from uploaded ordinary JSON."""


def renderer_files() -> tuple[Path, ...]:
    """Implementation inputs that can change a lazy report or resolved asset."""
    from autofa import behavior_policy
    from autofa.data import trace_navigation
    from autofa.metrics import verdicts
    from autofa.reports import failure_cards, ordering, recorded_relationships, related_failures
    from autofa.ui import theme

    return (Path(__file__), Path(failure_cards.__file__), Path(ordering.__file__),
            Path(recorded_relationships.__file__), Path(related_failures.__file__),
            Path(trace_navigation.__file__),
            Path(verdicts.__file__), Path(theme.__file__), Path(behavior_policy.__file__))


def renderer_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in renderer_files():
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def report_for_saved_run(settings, source_run_id: str, library_payload: dict):
    """Enrich a local upload only from a matching run already owned by this server.

    An uploaded report, descriptor, path, or reference is never authority to read a
    local dataset. The saved run's library must match, including packed transport.
    """
    if (not source_run_id or source_run_id in {".", ".."}
            or Path(source_run_id).name != source_run_id or "\\" in source_run_id):
        return None
    root = Path(settings.runs_dir).resolve()
    run = root / source_run_id
    if run.is_symlink() or run.resolve().parent != root:
        return None
    saved = {}
    for name, maximum in (("library.json", settings.max_local_library_bytes),
                          ("config.json", settings.max_metadata_bytes)):
        path = run / name
        try:
            if path.is_symlink():
                return None
            with path.open("rb") as stream:
                content = stream.read(maximum + 1)
            if len(content) > maximum:
                return None
            saved[name] = json.loads(content)
            if not isinstance(saved[name], dict):
                return None
        except (OSError, ValueError):
            return None
    if not same_library_content(saved["library.json"], library_payload):
        return None
    from autofa.reports.recorded_relationships import read_state
    from autofa.session.status_log import latest_turn_status

    return build_lazy_report(
        FunctionLibrary.from_dict(library_payload),
        saved["config.json"].get("reference_data", []),
        source_harness=saved["config.json"].get("source_harness") or "unknown",
        relationship_state=read_state([run]),
        agent_execution=latest_turn_status(run),
    )


def build_lazy_report(library: FunctionLibrary, references: list[dict], *, source_harness: str,
                      relationship_state: dict | None = None,
                      agent_execution: dict | None = None):
    """Return a report and tiny descriptors, or None for an unsupported/missing source.

    Only identity columns are decoded. Standalone reports retain their existing full
    evidence behavior; this representation is private to the local dashboard.
    """
    references = normalize_reference_data(references)
    if not references or any(ref.get("kind") != "local_evalrdl" for ref in references):
        return None
    resolved = []
    for reference in references:
        try:
            store = resolve_store(reference["store_id"])
            manifest = json.loads((store / "manifest.json").read_text())
            if manifest["run_id"] != reference["run_id"]:
                return None
        except (OSError, ValueError):
            return None
        resolved.append((reference, store / "dataset.parquet"))

    data = build_data(library, source_harness=source_harness, reference_data=references,
                      relationship_state=relationship_state)
    renderer = renderer_fingerprint()
    needed = _collect_example_ids(data["functions"])
    qualified_bases = {iid.rsplit("|", 1)[0] for iid in needed}
    items, descriptors = {}, {}
    n_items, n_failures, labels_known = 0, 0, True
    for reference, path in resolved:
        dataset = Dataset([path])
        for batch in dataset.iter_batches(columns=[
            "model_id", "benchmark", "problem_id", "run_idx", "hb_problem_idx",
            "is_correct", "accuracy",
        ]):
            for row in batch:
                n_items += 1
                correct = _row_is_correct(row)
                n_failures += correct is False
                labels_known = labels_known and correct is not None
                canonical = row["_item_id"]
                legacy = make_item_id(row, path, include_source=False)
                candidates = {canonical, legacy}
                if legacy in qualified_bases:
                    candidates.add(make_item_id(row, path, include_source=True))
                for iid in candidates & needed:
                    if iid in descriptors:
                        # An old ID lacking source identity must never pick an arbitrary run.
                        raise ValueError("Ambiguous local report example identity")
                    url = rollout_url(reference, problem_id=row.get("problem_id"),
                                      run_idx=row.get("run_idx"))
                    items[iid] = {"item_id": iid, "source_data_url": url}
                    descriptors[iid] = {
                        "version": 1, "store_id": reference["store_id"],
                        "renderer": renderer,
                        "run_id": reference["run_id"], "item_id": canonical,
                        "report_item_id": iid, "source_harness": source_harness,
                    }
    data = build_data(library, items=items, source_harness=source_harness,
                      reference_data=references, relationship_state=relationship_state)
    for function in data["functions"]:
        for iid, record in function.get("failure_localisations", {}).items():
            if iid in descriptors:
                descriptors[iid].setdefault("localisations", {})[function["name"]] = record
    assets = LazyReportAssets()
    for iid, descriptor in sorted(descriptors.items()):
        assets.append((f"{ITEMS_BASE}/{_item_filename(iid)}",
                       json.dumps(descriptor, ensure_ascii=False, allow_nan=False).encode()))
    data["items"] = {}
    if agent_execution:
        data["meta"]["agent_execution"] = agent_execution
    data["item_index"] = {iid: _item_filename(iid) for iid in descriptors}
    data["meta"].update(items_base=ITEMS_BASE, n_embedded_items=len(descriptors),
                        local_lazy_items=True, local_report_renderer=renderer)
    if data["meta"]["n_items"] is None:
        data["meta"].update(n_items=n_items, n_failures=n_failures if labels_known else None,
                            base_fail_rate=n_failures / n_items
                            if n_items and labels_known else None)
    return render_html(data).encode(), assets


def resolve_lazy_asset(payload: bytes) -> bytes:
    """Resolve one internal descriptor without a full-corpus scan or detector calls."""
    from autofa.data.reference_data import event_url

    descriptor = json.loads(payload)
    if descriptor.get("version") != 1:
        raise ValueError("Unsupported local report descriptor")
    store = resolve_store(descriptor["store_id"])
    manifest = json.loads((store / "manifest.json").read_text())
    if manifest["run_id"] != descriptor["run_id"]:
        raise ValueError("Local report source identity changed")
    dataset = Dataset([store / "dataset.parquet"])
    location = dataset.identity_index().by_id.get(descriptor["item_id"])
    if location is None:
        raise FileNotFoundError("Local report example is unavailable")
    rows = dataset.read_locations([location])
    if len(rows) != 1 or rows[0]["_item_id"] != descriptor["item_id"]:
        raise FileNotFoundError("Local report example is unavailable")
    reference = make_local_reference_data(
        store_id=descriptor["store_id"], run_id=descriptor["run_id"],
        source_harness=descriptor["source_harness"], parquet_path=dataset.parquet_paths[0],
    )
    item = _compact_item(rows[0], reference_data=[reference],
                         localisations=descriptor.get("localisations"))
    item["item_id"] = descriptor["report_item_id"]
    for record in item["localisations"].values():
        for passage in record.get("passages", []):
            for location in passage.get("trace_locations", []):
                location["source_url"] = event_url(
                    item["source_data_url"], location.get("source_event"),
                    location.get("quote", passage.get("quote")),
                    context=location.get("source_context"),
                )
    return json.dumps(item, ensure_ascii=False, allow_nan=False).encode()
