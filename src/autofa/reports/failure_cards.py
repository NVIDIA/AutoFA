# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lean HTML Model Failure Cards report (offline, single file).

Inspired by failures-scale dashboard_v7 Model Failure Card tab — same visual
language (dark NVIDIA green, progressive disclosure), but driven only by
``library.json`` (no taxonomy sidecars / Explore tabs).

When a bound dataset (or run ``config.json`` with parquet paths) is available,
example fail trajectories are embedded so bucket / example clicks can open an
inspector drawer — same spot-check workflow as the old Explore inspector.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from autofa.core.dependencies import review_source
from autofa.core.library import FunctionLibrary
from autofa.data.reference_data import (
    data_api_location,
    event_url,
    inspect_url,
    local_viewer_base,
    normalize_reference_data,
    reference_for_parquet,
    rollout_url,
)
from autofa.metrics.pareto import function_ranking_eligibility
from autofa.metrics.verdicts import (
    has_been_evaluated,
    verdict_for_metrics,
    verdict_js_classify_fn,
)
from autofa.reports.detector_summary import summarize_function_mechanism
from autofa.reports.ordering import (
    REPORT_AXES,
    has_report_scores,
    order_report_functions,
    report_assessment,
    report_ordering_js,
    visible_report_functions,
)

if TYPE_CHECKING:
    from autofa.data.parquet_store import Dataset

REPORT_NAME = "failure_cards.html"
BUNDLE_ZIP_PATH = "/autofa-bundle.zip"
ProgressFn = Callable[[str], None]


def _progress(cb: ProgressFn | None, message: str) -> None:
    if cb is not None:
        try:
            cb(message)
        except Exception:  # noqa: BLE001
            pass


def bundle_zip_filename(report_dir: Path) -> str:
    """Suggested download name from ``…/{model}/{benchmark}/`` or run dir."""
    parts = [p for p in (report_dir.parent.name, report_dir.name) if p and p not in {".", "/"}]
    # Drop homedir noise like ".autofa" / "library" when present.
    skip = {".autofa", "library", "runs", "home"}
    parts = [p for p in parts if p.lower() not in skip]
    if len(parts) >= 2:
        return f"{parts[-2]}_{parts[-1]}_autofa.zip"
    if parts:
        return f"{parts[-1]}_autofa.zip"
    return "failure_cards_bundle.zip"


def build_report_bundle_zip(
    report_dir: Path,
    *,
    html_name: str = REPORT_NAME,
) -> tuple[bytes, str]:
    """Zip a self-contained HTML report + ``library.json`` for offline sharing.

    Sidecar trajectories are re-embedded into the HTML so opening the file in a
    browser works without a localhost server (``file://`` cannot fetch
    ``*_data/items/*.json``). Returns ``(zip_bytes, suggested_filename)``.
    """
    import io
    import zipfile

    report_dir = Path(report_dir).resolve()
    html_path = report_dir / html_name
    if not html_path.is_file():
        raise FileNotFoundError(f"Missing report HTML: {html_path}")

    offline_html = render_offline_html(report_dir, html_name=html_name)
    filename = bundle_zip_filename(report_dir)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(html_name, offline_html)
        lib = report_dir / "library.json"
        if lib.is_file():
            zf.write(lib, arcname=lib.name)
        zf.writestr(
            "README.txt",
            (
                "AutoFA failure-cards export\n"
                "==========================\n\n"
                f"Open {html_name} in any browser (double-click is fine).\n"
                "Traces are embedded in the HTML — no local server needed.\n"
                "library.json holds the run's detectors for Continue / research.\n"
            ),
        )
    return buf.getvalue(), filename


def _extract_autofa_data(html: str) -> dict[str, Any]:
    """Parse the ``#autofa-data`` JSON blob from a rendered report page."""
    import re

    m = re.search(
        r'<script id="autofa-data" type="application/json">(.*?)</script>',
        html,
        flags=re.DOTALL,
    )
    if not m:
        raise ValueError("HTML is missing embedded autofa-data JSON")
    return json.loads(m.group(1))


def build_report_assets_zip(
    report_dir: Path,
    *,
    html_name: str = REPORT_NAME,
) -> bytes | None:
    """Archive lazy-loaded trajectory sidecars for cloud publication."""
    import io
    import zipfile

    report_dir = Path(report_dir).resolve()
    html_path = report_dir / html_name
    data = _extract_autofa_data(html_path.read_text(encoding="utf-8"))
    index = data.get("item_index") or {}
    items_base = (data.get("meta") or {}).get("items_base")
    if not index or not items_base:
        return None

    base = PurePosixPath(str(items_base))
    if base.is_absolute() or any(part in {"", ".", ".."} for part in base.parts):
        raise ValueError("Report contains an invalid trajectory sidecar path")

    archive = io.BytesIO()
    written: set[str] = set()
    with zipfile.ZipFile(
        archive,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=5,
    ) as bundle:
        for filename in index.values():
            relative = base / str(filename)
            if (
                relative.is_absolute()
                or any(part in {"", ".", ".."} for part in relative.parts)
                or relative.suffix != ".json"
            ):
                raise ValueError("Report contains an invalid trajectory sidecar path")
            archive_name = relative.as_posix()
            if archive_name in written:
                continue
            source = report_dir.joinpath(*relative.parts)
            if not source.is_file():
                raise FileNotFoundError(f"Missing report trajectory sidecar: {source}")
            bundle.writestr(archive_name, source.read_bytes())
            written.add(archive_name)
    return archive.getvalue()


def load_report_data_with_sidecars(
    report_dir: Path,
    *,
    html_name: str = REPORT_NAME,
) -> dict[str, Any]:
    """Load page data and merge on-disk sidecar trajectories into ``items``."""
    report_dir = Path(report_dir).resolve()
    html_path = report_dir / html_name
    data = _extract_autofa_data(html_path.read_text(encoding="utf-8"))
    items = dict(data.get("items") or {})
    index = data.get("item_index") or {}
    base = (data.get("meta") or {}).get("items_base")
    if base and index:
        items_dir = report_dir / Path(str(base))
        for iid, fname in index.items():
            key = str(iid)
            if key in items:
                continue
            side = items_dir / str(fname)
            if not side.is_file():
                continue
            try:
                items[key] = json.loads(side.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
    data = dict(data)
    data["items"] = items
    data["item_index"] = {}
    meta = dict(data.get("meta") or {})
    meta["items_base"] = None
    meta["n_embedded_items"] = len(items)
    meta["offline_export"] = True
    data["meta"] = meta
    return data


def render_offline_html(
    report_dir: Path,
    *,
    html_name: str = REPORT_NAME,
) -> str:
    """Self-contained HTML with trajectories inlined (works via ``file://``)."""
    return render_html(load_report_data_with_sidecars(report_dir, html_name=html_name))


def _library_coverage_fingerprint(library: FunctionLibrary) -> str:
    """Bind portable coverage to the exact saved detector evidence and grades."""
    from autofa.session.review_evidence import function_review_history, merge_histories

    payload = {
        "coverage_policy": "ok_or_better_v2",
        "functions": sorted((fn.to_dict() for fn in library.functions),
                            key=lambda fn: fn.get("name", "")),
        "review_history": merge_histories(library.review_history,
                                          function_review_history(library.functions)),
        "model_id": library.model_id,
        "benchmark": library.benchmark,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def _report_library_coverage(library: FunctionLibrary, dataset: Dataset | None) -> dict:
    """Use full source identities, never the report's selected example rows."""
    from autofa.metrics.library_quality_coverage import library_quality_coverage
    from autofa.session.family_coverage import failed_task_inventory

    inventory = None
    if dataset is not None:
        try:
            inventory = failed_task_inventory(dataset)
        except Exception:  # noqa: BLE001
            # Optional source access must not prevent viewing saved detector
            # results, and saved per-detector estimates cannot replace a corpus.
            pass
    return {
        **library_quality_coverage(library, inventory, saved_assessment=True),
        "library_fingerprint": _library_coverage_fingerprint(library),
    }


def build_data(
    library: FunctionLibrary,
    *,
    dataset: Dataset | None = None,
    items: dict[str, dict[str, Any]] | None = None,
    source_harness: str = "unknown",
    reference_data: list[dict[str, str]] | None = None,
    prefer_source_links: bool = False,
    source_links_only: bool = False,
    progress: ProgressFn | None = None,
    relationship_state: dict | None = None,
) -> dict[str, Any]:
    """Compact envelope embedded into the HTML page.

    ``items`` maps ``item_id → inspector payload``. Saved example IDs select
    trajectories without re-running detectors. Cloud reports can retain only
    source links for examples with exact Data API provenance. With
    ``source_links_only``, unlinked examples retain their saved evidence and
    identity without reading or embedding trajectories.
    """
    from autofa.data.trace_navigation import NAVIGATION_VERSION
    from autofa.reports.recorded_relationships import recorded_relationships

    _progress(progress, "Compacting functions…")
    library_coverage = _report_library_coverage(library, dataset)
    raw = [f.to_dict() for f in library.functions]
    # Display eligibility follows each saved review's strategy. Search preferences
    # must not relabel historical reviews or change the report's display order.
    ranked = [{**fn, "ranking": function_ranking_eligibility(fn, REPORT_AXES)} for fn in raw]

    n_items = 0
    n_failures = 0
    for fn in ranked:
        s = fn.get("scores") or {}
        if s.get("metrics_scope") == "resolved_items":
            # Resolved detector calls do not describe the complete dataset cohort.
            # Use a complete detector census or the identity-index fallback below.
            continue
        # Prefer the sampling frame size when present (true corpus N); else round
        # weighted floats so LLM estimates don't truncate to N-1 via int().
        frame = s.get("sample_frame_n")
        ni = s.get("n_items")
        if frame is not None:
            n_items = max(n_items, int(round(float(frame))))
        elif ni is not None:
            n_items = max(n_items, int(round(float(ni))))
        # Prefer contingency: a + c = failures caught + missed = n_fail
        a = s.get("a")
        c = s.get("c")
        if a is not None and c is not None:
            n_failures = max(n_failures, int(round(float(a) + float(c))))
        elif s.get("n_fail_caught") is not None and s.get("recall"):
            rec = float(s["recall"])
            if rec > 0:
                n_failures = max(
                    n_failures, int(round(float(s["n_fail_caught"]) / rec))
                )

    counts_known = bool(n_items)
    if not counts_known and dataset is not None:
        # An empty library still has a cohort. Count identity/label columns only.
        allowed = dataset.allowed_benchmarks()
        rows = [row for row in dataset.identity_index().rows
                if (not dataset.model_id or dataset.model_id == "*"
                    or not row.model_id or row.model_id == dataset.model_id)
                and (not allowed or row.benchmark in allowed)]
        n_items = len(rows)
        n_failures = (sum(row.is_correct is False for row in rows)
                      if all(row.is_correct is not None for row in rows) else None)
        counts_known = True
    base = (n_failures / n_items) if n_items and n_failures is not None else None
    functions = order_report_functions([
        _compact_function(fn, base, review_history=library.review_history) for fn in ranked
    ])
    localisations: dict[str, dict[str, dict]] = {}
    for fn in functions:
        for iid, record in fn.get("failure_localisations", {}).items():
            localisations.setdefault(iid, {})[fn["name"]] = record

    # Cards never re-score detectors — metrics come from library.json only
    # (census happens during discovery). Dataset is only used to
    # load trajectories for already-saved example ids.
    if source_links_only:
        reference_data = [ref for ref in normalize_reference_data(reference_data)
                          if data_api_location(ref) is not None]
    embedded = ({iid: dict(item) for iid, item in items.items()}
                if items and not source_links_only else {})
    needed = _collect_example_ids(functions)
    if dataset is not None and not embedded:
        if needed:
            subject = "example identities" if source_links_only else "saved trajectories"
            _progress(progress, f"Loading {len(needed)} {subject}…")
            embedded = _load_example_items(
                dataset,
                needed,
                reference_data=reference_data,
                prefer_source_links=prefer_source_links,
                source_links_only=source_links_only,
                progress=progress,
                localisations=localisations,
            )
    if source_links_only:
        for iid in needed - embedded.keys():
            embedded[iid] = {"item_id": iid, "source_only": True, "source_data_url": ""}

    # Saved localisations carry source IDs even in reports that publish only
    # rollout links.
    for fn in functions:
        records = {}
        for iid, saved in fn.get("failure_localisations", {}).items():
            it = embedded.get(iid, {})
            record = it.get("localisations", {}).get(fn["name"], saved)
            # Only current navigation is verified. Other coordinates keep the
            # saved quote and whole-rollout link instead.
            verified_locations = record.get("navigation_version") == NAVIGATION_VERSION
            records[iid] = {**record, "passages": [
                {**p, "trace_locations": [
                    {**loc, "source_url": event_url(it.get("source_data_url", ""),
                                                   loc.get("source_event"),
                                                   loc.get("quote", p.get("quote")),
                                                   context=loc.get("source_context"))}
                    for loc in (p.get("trace_locations", []) if verified_locations else [])
                ]} for p in record.get("passages", [])
            ]}
        fn["failure_localisations"] = records
    for it in embedded.values():
        it.pop("localisations", None)

    public_references = []
    for reference in normalize_reference_data(reference_data, include_local_paths=False):
        item = dict(reference)
        item["inspect_url"] = inspect_url(reference)
        public_references.append(item)

    return {
        "meta": {
            "model_id": library.model_id or "",
            "benchmark": library.benchmark or "all",
            "n_functions": len(functions),
            "n_items": n_items if counts_known else None,
            "n_failures": n_failures if counts_known else None,
            "base_fail_rate": base,
            "n_embedded_items": len(embedded),
            "source_harness": str(source_harness or "unknown"),
            "reference_data": public_references,
            "library_quality_coverage": library_coverage,
            "recorded_relationships": recorded_relationships(library, relationship_state),
        },
        "functions": functions,
        "items": embedded,
    }


def _compact_function(
    fn: dict[str, Any], base_fail_rate: float | None, *, review_history=None,
) -> dict[str, Any]:
    from autofa.core.localisation import detector_hash
    from autofa.metrics.claim_contract import claim_hash
    from autofa.metrics.improvements import resolve_improvement
    assessment = report_assessment(fn, review_history=review_history)
    scores, metrics = assessment["scores"], assessment["metrics"]
    verdict = verdict_for_metrics(metrics, scores=scores)
    buckets_raw = scores.get("buckets") or []
    if "buckets" not in scores and scores.get("bucket") is not None:
        buckets_raw = [scores]
    buckets = [_compact_bucket(b) for b in buckets_raw]
    best = None
    if buckets:
        best = max(buckets, key=lambda b: float(b.get("purity") or 0))

    title = _display_title(fn)
    source = (fn.get("source") or "").strip()
    authored_how = (fn.get("function_mechanism") or "").strip()
    fingerprint = detector_hash(fn.get("source") or "", (fn.get("meta") or {}).get("deps"),
                                bundle=(fn.get("meta") or {}).get("dependency_bundle"))
    localisations = {key: record for key, record in (fn.get("failure_localisations") or {}).items()
                     if record.get("detector_hash") == fingerprint}
    how = authored_how or summarize_function_mechanism(
        source, kind=str(fn.get("kind") or "regex")
    )
    final_review = scores.get("final_review")
    if not isinstance(final_review, dict) or final_review.get("claim_hash") != claim_hash(
        source=review_source(fn), description=fn.get("description") or "",
        mechanism=fn.get("mechanism") or fn.get("description") or fn.get("name") or "",
    ):
        final_review = None
    return {
        "name": fn.get("name") or "unnamed",
        "card_id": failure_mode_card_id(str(fn.get("name") or "unnamed")),
        "title": title,
        "description": (fn.get("description") or "").strip(),
        "mechanism": (fn.get("mechanism") or "").strip(),
        "function_mechanism": how,
        "function_mechanism_authored": bool(authored_how),
        "kind": fn.get("kind") or "regex",
        "source": source,
        "uses_helpers": fn.get("uses_helpers") or {},
        "dependency_sources": review_source(fn) if (fn.get("meta") or {}).get("dependency_bundle") else "",
        "example_fail_ids": list(fn.get("example_fail_ids") or []),
        "failure_localisations": localisations,
        "verdict": verdict,
        "ranking": fn.get("ranking"),
        "behavior_validation": assessment["behavior_validation"],
        "final_review": final_review,
        "confirmation_availability": scores.get("confirmation_availability"),
        "metrics": {
            "evaluation_status": metrics["evaluation_status"],
            "unavailable_reason": metrics["unavailable_reason"],
            "coverage_available": metrics["coverage_available"],
            "normalized_precision_available": metrics["normalized_precision_available"],
            "badge_n_flagged": metrics["badge_n_flagged"],
            "pct_caught": metrics["pct_caught"],
            "flagged_detail": metrics["flagged_detail"],
            "scoring_detail": metrics["scoring_detail"],
            "diagnostic_review_detail": metrics["diagnostic_review_detail"],
            "final_validation_skip_detail": scores.get("final_validation_skip_detail", ""),
            "recall": metrics["recall"],
            "task_recall": _num(scores.get("task_recall")),
            "task_recall_scope": scores.get("task_recall_scope", "unavailable"),
            "task_recall_reason": scores.get("task_recall_reason"),
            # task_recall retains its evaluated-cohort numerator; breadth can
            # include later source-verified matches outside that scoring panel.
            "n_evaluated_matched_failed_tasks": _int(scores.get("n_matched_failed_tasks")),
            "n_evaluated_matched_failed_attempts": _int(scores.get("n_matched_failed_attempts")),
            "n_matched_failed_tasks": metrics.get("n_matched_failed_tasks"),
            "matched_task_support_status": metrics.get("matched_task_support_status"),
            "n_matched_failed_attempts": metrics.get("n_matched_failed_attempts"),
            "n_confirmed_failed_tasks": metrics.get("n_confirmed_failed_tasks"),
            "confirmed_task_support_status": metrics.get("confirmed_task_support_status"),
            **{key: metrics.get(key) for key in (
                "estimated_n_confirmed_failed_tasks", "estimated_n_matched_failed_tasks",
                "task_support_estimate_status", "task_support_estimate_reason",
                "task_support_comparison", "task_support_comparison_basis",
                "n_sampled_failed_tasks", "n_sampled_confirmed_failed_tasks",
                "n_sampled_matched_failed_tasks",
            )},
            "n_failed_tasks": _int(scores.get("n_failed_tasks")),
            "precision": metrics["precision"],
            "ppv_at_pi0": metrics["ppv_at_pi0"],
            "behavior_purity": metrics["behavior_purity"],
            "behavior_specificity": metrics["behavior_specificity"],
            "behavior_validation_status": metrics["behavior_validation_status"],
            "behavior_processing_pending": metrics.get("behavior_processing_pending", False),
            "f1_at_pi0": _num(scores.get("f1_at_pi0")),
            "odds_ratio": _num(scores.get("odds_ratio")),
            "or_ci_low": _num(scores.get("or_ci_low")),
            "or_ci_high": _num(scores.get("or_ci_high")),
            "fpr": _num(scores.get("fpr")),
            "fpr_ci_low": _num(scores.get("fpr_ci_low")),
            "fpr_ci_high": _num(scores.get("fpr_ci_high")),
            "fisher_p": _num(scores.get("fisher_p")),
            "n_flagged": _num(scores.get("n_flagged")),
            "n_fail_caught": _num(scores.get("n_fail_caught")),
            "base_fail_rate": base_fail_rate,
            "recall_ci_low": _num(scores.get("recall_ci_low")),
            "recall_ci_high": _num(scores.get("recall_ci_high")),
            "raw_ppv_ci_low": _num(scores.get("raw_ppv_ci_low")),
            "raw_ppv_ci_high": _num(scores.get("raw_ppv_ci_high")),
        },
        "sampled": bool(scores.get("sampled")),
        "sample_n": _int(scores.get("sample_n")),
        "sample_frame_n": _int(scores.get("sample_frame_n")),
        "n_scored": _int(scores.get("n_scored", scores.get("n_items"))),
        "evaluation": {
            "evaluated": scores.get("evaluated", has_been_evaluated(scores)),
            **{
                key: scores[key] for key in (
                    "metrics_scope", "assessment_coverage", "evaluation_complete", "n_scored",
                    "evaluation_stale", "n_items",
                    "sampling_limitation", "sample_unrepresented_strata",
                    "errors", "n_insufficient", "n_unresolved", "parent_errors",
                    "abstention_bounds", "unresolved_bounds", "within_errors",
                    "within_n_insufficient", "within_n_unresolved", "within_assessment_coverage",
                    "within_abstention_bounds", "within_unresolved_bounds",
                ) if key in scores
            },
        },
        "best_bucket": best,
        "buckets": buckets,
        "judge_rationale": (scores.get("judge_rationale") or "").strip() or None,
        "improvement": resolve_improvement(
            scores.get("improvement"), source=review_source(fn),
            description=fn.get("description") or "", mechanism=fn.get("mechanism") or "",
        ),
        "badge": verdict["label"],
        "feedback": scores.get("feedback") if isinstance(scores.get("feedback"), dict) else None,
    }


def failure_mode_card_id(function_name: str) -> str:
    """Return a stable, HTML-safe card id derived only from a function name."""
    name = str(function_name or "unnamed")
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "function"
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:12]
    return f"failure-mode-{slug[:48]}-{digest}"


def _compact_bucket(b: dict[str, Any]) -> dict[str, Any]:
    purity = _num(b.get("purity") if b.get("purity") is not None else b.get("precision"))
    return {
        "bucket": str(b.get("bucket", "")),
        "n": _int(b.get("n")),
        "n_fail": _int(b.get("n_fail")),
        "n_pass": _int(b.get("n_pass")),
        "purity": purity,
        "recall": _num(b.get("recall") if b.get("recall") is not None else b.get("sensitivity")),
        "ppv_at_pi0": _num(b.get("ppv_at_pi0")),
        "f1_at_pi0": _num(b.get("f1_at_pi0")),
        "odds_ratio": _num(b.get("odds_ratio")),
        "or_ci_low": _num(b.get("or_ci_low")),
        "or_ci_high": _num(b.get("or_ci_high")),
        "fpr": _num(b.get("fpr")),
        "fpr_ci_low": _num(b.get("fpr_ci_low")),
        "fpr_ci_high": _num(b.get("fpr_ci_high")),
        "fisher_p": _num(b.get("fisher_p")),
        "recall_ci_low": _num(b.get("recall_ci_low")),
        "recall_ci_high": _num(b.get("recall_ci_high")),
        "is_high_purity": bool(b.get("is_high_purity")) if b.get("is_high_purity") is not None else (
            purity is not None and purity >= 0.70
        ),
        "example_fail_ids": list(b.get("example_fail_ids") or []),
        "example_pass_ids": list(b.get("example_pass_ids") or []),
    }


def _is_detector_centric_title(text: str) -> bool:
    t = text.strip().lower()
    return t.startswith(
        (
            "fires when",
            "detects when",
            "flags when",
            "matches when",
            "returns true when",
            "labels when",
            "tags when",
        )
    )


def _display_title(fn: dict[str, Any]) -> str:
    """Use a complete authored sentence, never a character-truncated fragment."""
    name = str(fn.get("name") or "unnamed")
    for field in ("description", "mechanism"):
        text = " ".join(str(fn.get(field) or "").split())
        if not text or _is_detector_centric_title(text):
            continue
        # Split at sentence boundaries, not dots within paths or decimal numbers.
        first = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text, maxsplit=1)[0]
        if not first.endswith(("...", "…")):
            return first.rstrip(".")
    return name.replace("_", " ")


def _num(v: Any) -> float | None:
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x


def _int(v: Any) -> int | None:
    x = _num(v)
    return int(x) if x is not None else None


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _collect_example_ids(functions: list[dict[str, Any]]) -> set[str]:
    """Union of function- and bucket-level example fail/pass ids."""
    out: set[str] = set()
    for fn in functions:
        for eid in (fn.get("improvement") or {}).get("evidence_ids", []):
            if eid:
                out.add(str(eid))
        for eid in fn.get("example_fail_ids") or []:
            if eid:
                out.add(str(eid))
        for b in fn.get("buckets") or []:
            for key in ("example_fail_ids", "example_pass_ids"):
                for eid in b.get(key) or []:
                    if eid:
                        out.add(str(eid))
    return out


def _compact_item(
    item: dict[str, Any],
    *,
    reference_data: list[dict[str, str]] | None = None,
    localisations: dict[str, dict] | None = None,
) -> dict[str, Any]:
    """Project one parquet row into the inspector payload (full text, no truncation)."""
    from autofa.core.fields import DEFAULT_CONTENT_FIELDS
    from autofa.data.parquet_store import _is_correct
    from autofa.data.trace_navigation import TraceNavigation
    from autofa.data.trajectory import format_actions, format_reasoning, parse_trajectory

    iid = str(item.get("_item_id") or item.get("item_id") or "")

    prompt_key = next(
        (k for k in ("prompt_text", "question") if item.get(k)),
        "prompt_text",
    )
    response_key = next(
        (k for k in ("response_text", "generation") if item.get(k)),
        "response_text",
    )
    reasoning_key = next(
        (
            k
            for k in ("reasoning_text", "reasoning_content", "chain_of_thought")
            if item.get(k)
        ),
        None,
    )
    judge_key = "judge_text" if item.get("judge_text") else None

    from autofa.data.input_view import parsed_input

    input_data = parsed_input(item)
    prompt = input_data["text"] or _as_text(item.get(prompt_key) or "")
    response_value = item.get(response_key) or ""
    response = (response_value if isinstance(response_value, str)
                else json.dumps(response_value, ensure_ascii=False))
    reasoning = _as_text(item.get(reasoning_key) or "") if reasoning_key else ""
    judge = _as_text(item.get(judge_key) or "") if judge_key else ""
    correct = _is_correct(item)
    reference = reference_for_parquet(
        reference_data,
        item.get("_source_parquet"),
    )
    source_data_url = (
        rollout_url(
            reference,
            problem_id=item.get("problem_id"),
            run_idx=item.get("run_idx"),
        )
        if reference is not None
        else ""
    )

    parsed = parse_trajectory(response)
    steps = parsed._steps if parsed.ok else []
    response_kind = "text"
    response_display = response
    reasoning_from_steps = False
    if steps:
        response_kind = "agent_transcript"
        if not reasoning:
            reasoning = format_reasoning(steps)
            reasoning_from_steps = bool(reasoning)
            reasoning_key = reasoning_key or "steps[].reasoning_content"
        # Human-readable rendering of the same raw response_text / generation blob.
        response_display = format_actions(steps) or response

    navigation = TraceNavigation(item)
    events = navigation.events
    for event in events:
        event["source_url"] = event_url(source_data_url, event.get("source_event"))

    return {
        "item_id": iid,
        "label": 1 if correct else 0,  # 0 = fail (matches old dashboard convention)
        "is_correct": correct,
        "benchmark": item.get("benchmark"),
        "difficulty": item.get("strata_difficulty") or item.get("difficulty"),
        "subject": item.get("strata_subject") or item.get("domain"),
        "finish_reason": item.get("finish_reason"),
        "response_kind": response_kind,
        "n_agent_steps": len([s for s in steps if str(s.get("source") or "") in {"agent", "assistant"}])
        if steps
        else None,
        "trajectory_dialect": parsed.dialect if parsed.ok else None,
        "field_map": {
            "prompt": prompt_key,
            "reasoning": reasoning_key,
            "response": response_key,
            "judge": judge_key,
            "reasoning_from_steps": reasoning_from_steps,
            "response_view": (
                "parsed_steps"
                if response_kind == "agent_transcript"
                else "raw"
            ),
        },
        "prompt_head": prompt,
        "input_parse": {"origin": input_data["origin"], "limitations": input_data["limitations"]},
        "response_head": response_display,
        # Raw string detectors see via item.get("response_text") / generation.
        "response_raw": response if response_kind == "agent_transcript" else None,
        "reasoning_head": reasoning,
        "judge_head": judge,
        "prompt_chars": len(prompt),
        "response_chars": len(response),
        "reasoning_chars": len(reasoning),
        "source_data_url": source_data_url,
        "trace_events": events,
        "field_hashes": {key: hashlib.sha256(value.encode("utf-8")).hexdigest()
                         for key, value in navigation.fields.items()
                         if key in DEFAULT_CONTENT_FIELDS and isinstance(value, str)},
        "localisations": {name: navigation.attach(record)
                          for name, record in (localisations or {}).items()},
    }


def _load_example_items(
    dataset: Dataset,
    item_ids: set[str],
    *,
    reference_data: list[dict[str, str]] | None = None,
    prefer_source_links: bool = False,
    source_links_only: bool = False,
    progress: ProgressFn | None = None,
    localisations: dict[str, dict[str, dict]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Single streaming pass over the dataset; stop once every id is found."""
    from autofa.data.parquet_store import _is_correct, make_item_id

    needed = {str(i) for i in item_ids if i}
    if not needed:
        return {}
    references = normalize_reference_data(reference_data)
    if len(dataset.parquet_paths) > 1:
        # A path-free reference can name one source, but cannot tell which file
        # supplied a row in a mixture. Never apply it to an unrelated rollout.
        references = [r for r in references if r.get("parquet_path")]
    # Local reports keep their embedded evidence and exact passage highlights even
    # when the source store is unavailable or the report is opened elsewhere.
    if source_links_only:
        references = [ref for ref in references if data_api_location(ref) is not None]
    elif prefer_source_links and (not references or any(local_viewer_base(r) for r in references)):
        prefer_source_links = False
    by_path = {r["parquet_path"]: r for r in references if r.get("parquet_path")}
    if len(references) == 1 and len(dataset.parquet_paths) == 1 and not by_path:
        by_path[str(dataset.parquet_paths[0].expanduser().resolve())] = references[0]
    # Identity columns are enough to build exact links; do not read multi-GB
    # response/reasoning columns just to discard them before publication.
    batches = dataset.iter_batches(columns=[
        "model_id", "benchmark", "problem_id", "run_idx", "hb_problem_idx",
        "is_correct", "accuracy",
    ]) if prefer_source_links or source_links_only else dataset.iter_batches()
    found: dict[str, dict[str, Any]] = {}
    n_scan = 0
    for batch in batches:
        for it in batch:
            n_scan += 1
            iid = str(it.get("_item_id") or "")
            candidates = [iid]
            source_path = it.get("_source_parquet")
            if source_path:
                legacy_id = make_item_id(
                    it,
                    Path(str(source_path)),
                    include_source=False,
                )
                if legacy_id != iid:
                    candidates.append(legacy_id)
            matched_id = next(
                (
                    candidate
                    for candidate in candidates
                    if candidate in needed and candidate not in found
                ),
                None,
            )
            if matched_id is not None:
                if prefer_source_links or source_links_only:
                    reference = by_path.get(str(Path(str(source_path)).expanduser().resolve()))
                    url = rollout_url(
                        reference or {}, problem_id=it.get("problem_id"), run_idx=it.get("run_idx"),
                    )
                    if not url and not source_links_only:
                        continue
                    correct = _is_correct(it)
                    compact = {
                        "source_only": True, "source_data_url": url,
                        "benchmark": it.get("benchmark"), "is_correct": correct,
                        "label": None if correct is None else int(correct),
                    }
                else:
                    compact = _compact_item(it, reference_data=references,
                                            localisations=(localisations or {}).get(matched_id))
                compact["item_id"] = matched_id
                found[matched_id] = compact
                if progress and (len(found) % 25 == 0 or len(found) == len(needed)):
                    _progress(
                        progress,
                        f"Loading trajectories · {len(found)}/{len(needed)} "
                        f"(scanned {n_scan})",
                    )
            elif progress and n_scan % 500 == 0:
                _progress(
                    progress,
                    f"Loading trajectories · {len(found)}/{len(needed)} "
                    f"(scanned {n_scan})",
                )
        if len(found) >= len(needed):
            break
    if prefer_source_links and not source_links_only and needed - found.keys():
        # Local/mixed sources without a verifiable rollout link retain their
        # full payload. Never assign another source's run to an unbound row.
        found.update(_load_example_items(
            dataset, needed - found.keys(),
            reference_data=(references if len(dataset.parquet_paths) == 1 else
                            [r for r in references if r.get("parquet_path")]),
            progress=progress,
            localisations=localisations,
        ))
    return found


def try_dataset_from_dir(directory: Path | None) -> Dataset | None:
    """Build a Dataset from a run ``config.json`` next to the library, if present."""
    if directory is None:
        return None
    cfg_path = Path(directory) / "config.json"
    if not cfg_path.is_file():
        return None
    try:
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return _dataset_from_config_dict(data)


def try_report_provenance_from_dir(
    directory: Path | None,
) -> tuple[str, list[dict[str, str]]]:
    """Load report-safe source provenance from a sibling run config."""
    if directory is None:
        return "unknown", []
    cfg_path = Path(directory) / "config.json"
    try:
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "unknown", []
    if not isinstance(data, Mapping):
        return "unknown", []
    return (
        str(data.get("source_harness") or "unknown"),
        list(data.get("reference_data") or []),
    )


def _dataset_from_config_dict(data: dict[str, Any]) -> Dataset | None:
    raw_paths = data.get("parquet_paths") or []
    paths = [Path(p).expanduser() for p in raw_paths if p]
    paths = [p for p in paths if p.exists()]
    if not paths:
        return None
    from autofa.data.parquet_store import Dataset

    benchmarks = list(data.get("benchmarks") or [])
    model_id = data.get("model_id")
    return Dataset(
        parquet_paths=paths,
        model_id=None if model_id in {"*", "ALL", ""} else model_id,
        benchmarks=benchmarks,
    )


def try_dataset_for_library(library: FunctionLibrary) -> Dataset | None:
    """Find parquet for a durable catalog library via the newest matching run.

    Catalog dirs (``~/.autofa/library/{model}/{bench}/``) have no ``config.json``.
    Spot-check trajectories still need the bound parquet, so we reuse the most
    recent run config with the same model or mixture ID and benchmark label.
    """
    model = (library.model_id or "").strip()
    if not model:
        return None
    try:
        from autofa.config import RunConfig, runs_root
    except Exception:  # noqa: BLE001
        return None

    root = runs_root()
    if not root.is_dir():
        return None

    bench = (library.benchmark or "").strip().lower()
    want_specific = bool(bench) and bench not in {"all", "all-benchmarks", "all_benchmarks"}

    configs = sorted(
        root.glob("*/config.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for cfg_path in configs:
        try:
            data = json.loads(cfg_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if model.startswith("mixture-"):
            try:
                run = RunConfig.load(cfg_path.parent.name)
                # Keep catalogs saved under the old single-model mixture name
                # usable after new publications switch to the actual model.
                if run.dataset_id != model and run.library_model_id() != model:
                    continue
            except (OSError, ValueError, TypeError):
                continue
            # The mixture ID already includes the complete benchmark filter.
            ds = _dataset_from_config_dict(data)
            if ds is not None:
                return ds
            continue
        elif str(data.get("model_id") or "").strip() != model:
            continue
        if want_specific:
            run_benches = {
                str(b).strip().lower()
                for b in (data.get("benchmarks") or [])
                if b
            }
            # A model library can span multiple benchmarks. Match its aggregate
            # label as well as a single benchmark.
            if run_benches and bench not in run_benches:
                try:
                    run = RunConfig.load(cfg_path.parent.name)
                except (OSError, ValueError, TypeError):
                    continue
                if run.benchmark_label().lower() != bench:
                    continue
        ds = _dataset_from_config_dict(data)
        if ds is not None:
            return ds
    return None


def render_html(data: dict[str, Any]) -> str:
    from autofa.behavior_policy import STRICTNESS_LABELS
    from autofa.metrics.claim_contract import JUDGE_CONTRACT_VERSION
    from autofa.reports.related_failures import project_related_failures

    functions = data.get("functions") or []
    scored = [fn for fn in functions if has_report_scores(fn)]
    meta = dict(data.get("meta") or {})
    meta["related_failures"] = project_related_failures(
        visible_report_functions(functions), meta.get("recorded_relationships") or [])
    visible = visible_report_functions(functions)
    meta.update(n_functions=len(visible),
                n_hidden_functions=len(functions) - len(visible),
                n_pending_scores=max(meta.get("n_pending_scores", 0),
                                     len(functions) - len(scored)))
    # Keep the full payload for inspection/export; cards and index use the same
    # visibility projection, including when a host replaces the embedded rows.
    data = {**data, "meta": meta, "functions": order_report_functions(functions)}
    payload = json.dumps(data, allow_nan=False, ensure_ascii=False)
    payload = payload.replace("</", "<\\/")
    meta = data.get("meta") or {}
    title = f"AutoFA · {meta.get('model_id') or '?'} / {meta.get('benchmark') or 'all'}"
    rendered = (
        _HTML_TEMPLATE.replace("{{TITLE}}", _escape_html(title))
        .replace("{{DATA_JSON}}", payload)
        .replace("{{VERDICT_JS}}", verdict_js_classify_fn() + report_ordering_js())
        .replace("{{JUDGE_CONTRACT_VERSION}}", JUDGE_CONTRACT_VERSION)
        .replace("{{JUDGE_POLICY_LABELS}}", json.dumps(STRICTNESS_LABELS))
    )
    if isinstance(meta.get("efficiency_metrics"), dict):
        from autofa.reports.efficiency_card import _render_overview

        rendered = _render_overview(rendered, layout=meta.get("efficiency_layout") or "replace")
    return rendered


def refresh_report_data(
    data: dict[str, Any], library: FunctionLibrary,
    *, relationship_state: dict | None = None,
) -> dict[str, Any]:
    """Refresh saved card presentation while retaining its cohort and trajectories.

    Read only the matching version's saved library; never evaluate detectors or
    load a dataset. Keep verified passage mappings and apply the shared display order.
    """
    saved_functions = {
        fn.name: {**fn.to_dict(), "ranking": function_ranking_eligibility(fn, REPORT_AXES)}
        for fn in library.functions
    }
    meta = dict(data.get("meta") or {})
    from autofa.reports.recorded_relationships import recorded_relationships

    meta["recorded_relationships"] = recorded_relationships(
        library, relationship_state, previous=meta.get("recorded_relationships"))
    functions = []
    for previous in data.get("functions", []):
        name = previous.get("name") or "unnamed"
        function = dict(previous)
        if name in saved_functions:
            current = _compact_function(
                saved_functions[name], (data.get("meta") or {}).get("base_fail_rate"),
                review_history=library.review_history,
            )
            for item_id, record in (previous.get("failure_localisations") or {}).items():
                fresh = current["failure_localisations"].get(item_id)
                if fresh and fresh.get("detector_hash") == record.get("detector_hash"):
                    current["failure_localisations"][item_id] = record
            # This is a view of the published assessment. A newer local judge
            # policy must not silently change its scores, eligibility or order.
            # Still invalidate evidence if the detector or its claim has changed.
            # Older reports omit empty optional dependency sources.
            same_claim = all(previous.get(key) == current.get(key) for key in (
                "source", "description", "mechanism",
            )) and (previous.get("dependency_sources") or "") == (
                current.get("dependency_sources") or ""
            )
            if (same_claim
                    and current["behavior_validation"]["status"] not in {"historical", "stale"}
                    and not current["evaluation"].get("evaluation_stale")):
                for key in ("metrics", "ranking", "behavior_validation", "verdict", "badge"):
                    if previous.get(key) is not None:
                        if key == "metrics":
                            # Preserve the published measurements while deriving task support
                            # from that version's saved evidence, without another judge call.
                            historical_metrics = dict(previous[key])
                            historical_metrics.pop("final_validation_skip_detail", None)
                            if (previous[key].get("final_validation_skip_detail")
                                    or current[key].get("final_validation_skip_detail")):
                                # Scheduling receipts cannot survive a changed assessment
                                # merely because the detector text stayed the same.
                                historical_metrics.pop("scoring_detail", None)
                                historical_metrics.pop("unavailable_reason", None)
                            current[key] = {**{
                                field: current[key].get(field) for field in (
                                    "unavailable_reason", "flagged_detail", "scoring_detail",
                                    "diagnostic_review_detail", "final_validation_skip_detail",
                                    "behavior_processing_pending",
                                    "n_confirmed_failed_tasks", "confirmed_task_support_status",
                                    "n_matched_failed_attempts", "n_matched_failed_tasks",
                                    "matched_task_support_status",
                                    "n_evaluated_matched_failed_tasks",
                                    "n_evaluated_matched_failed_attempts",
                                    "estimated_n_confirmed_failed_tasks",
                                    "estimated_n_matched_failed_tasks",
                                    "task_support_estimate_status", "task_support_estimate_reason",
                                    "task_support_comparison", "task_support_comparison_basis",
                                    "n_sampled_failed_tasks", "n_sampled_confirmed_failed_tasks",
                                    "n_sampled_matched_failed_tasks",
                                )
                            }, **historical_metrics}
                        elif key not in {"verdict", "badge"}:
                            current[key] = previous[key]
                current["verdict"] = verdict_for_metrics(current["metrics"])
                current["badge"] = current["verdict"]["label"]
            function.update(current)
        function["card_id"] = failure_mode_card_id(name)
        function["title"] = _display_title(function)
        functions.append(function)
    meta.update(n_functions=len(functions))
    saved_coverage = meta.get("library_quality_coverage")
    if (not isinstance(saved_coverage, dict)
            or saved_coverage.get("library_fingerprint") != _library_coverage_fingerprint(library)):
        # Re-rendering cannot re-establish the cohort from a handful of embedded
        # examples. Retain a matching published union; invalidate changed inputs.
        meta["library_quality_coverage"] = _report_library_coverage(library, None)
    return {**data, "meta": meta, "functions": order_report_functions(functions)}


def _item_filename(item_id: str) -> str:
    """Stable filesystem-safe name for an item payload."""
    return hashlib.sha1(str(item_id).encode("utf-8")).hexdigest() + ".json"


def _spill_items_to_disk(
    data: dict[str, Any],
    *,
    out_path: Path,
    progress: ProgressFn | None = None,
) -> dict[str, Any]:
    """Write full trajectories next to the HTML; keep only an index in the page.

    Full untruncated text makes an inline JSON payload hundreds of MB. Sidecar
    files load on demand when the report is served over HTTP.
    """
    data, files = spill_items(data, f"{out_path.stem}_data/items")
    if not files:
        return data
    items_dir = out_path.parent / f"{out_path.stem}_data" / "items"
    if items_dir.exists():
        for old in items_dir.glob("*.json"):
            old.unlink()
    items_dir.mkdir(parents=True, exist_ok=True)
    total = len(files)
    for i, (fname, content) in enumerate(files.items(), 1):
        (items_dir / fname).write_bytes(content)
        if progress and (i % 50 == 0 or i == total):
            _progress(progress, f"Writing sidecars · {i}/{total}")
    return data


def spill_items(data: dict[str, Any], base: str) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Move full trajectories out of the page data into sidecar files under ``base``.

    Returns the page data, which keeps source-only links and an index, and the sidecar files by
    name.
    """
    all_items = data.get("items") or {}
    links = {iid: item for iid, item in all_items.items() if item.get("source_only")}
    items = {iid: item for iid, item in all_items.items() if iid not in links}
    if not items:
        data.setdefault("item_index", {})
        meta = data.setdefault("meta", {})
        meta["items_base"] = None
        meta["n_embedded_items"] = len(links)
        return data, {}
    files: dict[str, bytes] = {}
    item_index: dict[str, str] = {}
    for iid, payload in items.items():
        fname = _item_filename(iid)
        files[fname] = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        item_index[str(iid)] = fname
    data = dict(data)
    data["items"] = links
    data["item_index"] = item_index
    meta = dict(data.get("meta") or {})
    meta["items_base"] = base
    meta["n_embedded_items"] = len(item_index) + len(links)
    data["meta"] = meta
    return data, files


def write_failure_cards(
    library: FunctionLibrary | Path,
    *,
    out: Path | None = None,
    dataset: Dataset | None = None,
    source_harness: str | None = None,
    reference_data: list[dict[str, str]] | None = None,
    prefer_source_links: bool = False,
    source_links_only: bool = False,
    progress: ProgressFn | None = None,
) -> Path:
    """Write ``failure_cards.html`` (+ full-text item sidecars) next to the library.

    Uses **saved** library scores only — detectors and judge reviews are not
    re-run here. Full-corpus evaluation belongs to discovery, before saving.

    When ``dataset`` is omitted, tries to resolve parquet paths from a sibling
    ``config.json`` (run directory) so example trajectories can be written.
    Open the report via ``serve_failure_cards`` / ``autofa dashboard`` so the
    browser can fetch sidecars over localhost (not ``file://``).
    ``source_links_only`` produces a lightweight Cloud report with saved evidence
    and verified source links, without reading or embedding full trajectories.
    """
    lib_dir: Path | None = None
    if isinstance(library, Path):
        lib_path = Path(library)
        lib = FunctionLibrary.load(lib_path)
        lib_dir = lib_path.parent
        default_out = lib_dir / REPORT_NAME
    else:
        lib = library
        default_out = out or Path(REPORT_NAME)
    out_path = Path(out) if out is not None else default_out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    _progress(progress, "Resolving dataset…")
    ds = dataset
    if ds is None:
        for candidate in (lib_dir, out_path.parent):
            ds = try_dataset_from_dir(candidate)
            if ds is not None:
                break
    if ds is None:
        ds = try_dataset_for_library(lib)
    if ds is None:
        _progress(progress, "No parquet bound — writing cards without trajectories")
    else:
        _progress(progress, f"Dataset ready · {len(ds.parquet_paths)} parquet file(s)")

    resolved_harness = source_harness
    resolved_references = reference_data
    if resolved_harness is None or resolved_references is None:
        for candidate in (lib_dir, out_path.parent):
            candidate_harness, candidate_references = try_report_provenance_from_dir(
                candidate
            )
            if resolved_harness is None and candidate_harness != "unknown":
                resolved_harness = candidate_harness
            if resolved_references is None and candidate_references:
                resolved_references = candidate_references
            if resolved_harness is not None and resolved_references is not None:
                break

    from autofa.objectives import config_from_run_dir, snapshot_objectives

    objectives = config_from_run_dir(lib_dir) or config_from_run_dir(out_path.parent)
    from autofa.reports.recorded_relationships import read_state

    data = _spill_items_to_disk(
        build_data(
            lib,
            dataset=ds,
            relationship_state=read_state([lib_dir, out_path.parent]),
            source_harness=resolved_harness or "unknown",
            reference_data=resolved_references,
            prefer_source_links=prefer_source_links,
            source_links_only=source_links_only,
            progress=progress,
        ),
        out_path=out_path,
        progress=progress,
    )
    from autofa.session.status_log import latest_turn_status

    data["meta"]["agent_execution"] = latest_turn_status(lib_dir or out_path.parent)
    if objectives is not None:
        data["meta"]["objective_settings"] = snapshot_objectives(objectives)
    _progress(progress, "Rendering HTML…")
    out_path.write_text(render_html(data), encoding="utf-8")
    n = (data.get("meta") or {}).get("n_embedded_items") or 0
    _progress(progress, f"Done · {n} trajectories indexed")
    return out_path.resolve()


def write_failure_cards_with_status(
    library: FunctionLibrary | Path,
    *,
    out: Path | None = None,
    dataset: Dataset | None = None,
    source_harness: str | None = None,
    reference_data: list[dict[str, str]] | None = None,
    prefer_source_links: bool = False,
    source_links_only: bool = False,
    console: Any | None = None,
) -> Path:
    """Like ``write_failure_cards`` but with a Rich spinner for CLI feedback."""
    from rich.console import Console
    from rich.status import Status

    from autofa.ui.theme import GREEN

    con = console or Console()
    with Status(
        f"[{GREEN}]Building failure cards…[/]",
        console=con,
        spinner="dots",
    ) as status:

        def progress(message: str) -> None:
            status.update(f"[{GREEN}]Building failure cards[/] · {message}")

        return write_failure_cards(
            library,
            out=out,
            dataset=dataset,
            source_harness=source_harness,
            reference_data=reference_data,
            prefer_source_links=prefer_source_links,
            source_links_only=source_links_only,
            progress=progress,
        )

def file_url(path: Path) -> str:
    """file:// URL for terminal click-to-open (sidecars will not load)."""
    return Path(path).resolve().as_uri()


def serve_failure_cards(html_path: Path, *, port: int = 0) -> str:
    """Serve the report directory on localhost; return the page URL.

    Full trajectories live in ``{stem}_data/items/`` and are fetched on click,
    so HTTP is required (``file://`` cannot load them). Also exposes
    ``GET /autofa-bundle.zip`` — a self-contained HTML (trajectories embedded)
    plus ``library.json`` for offline sharing.
    """
    import threading
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
    from urllib.parse import unquote, urlparse

    html_path = Path(html_path).resolve()
    directory = html_path.parent
    html_name = html_path.name

    class _QuietHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            super().__init__(*args, directory=str(directory), **kwargs)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A003
            return

        def do_GET(self) -> None:  # noqa: N802
            path = unquote(urlparse(self.path).path)
            if path.rstrip("/") == BUNDLE_ZIP_PATH.rstrip("/"):
                self._serve_bundle_zip()
                return
            super().do_GET()

        def _serve_bundle_zip(self) -> None:
            try:
                payload, filename = build_report_bundle_zip(
                    directory, html_name=html_name
                )
            except Exception as e:  # noqa: BLE001
                body = f"bundle failed: {e}\n".encode()
                self.send_response(500)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/zip")
            self.send_header(
                "Content-Disposition",
                f'attachment; filename="{filename}"',
            )
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", port), _QuietHandler)
    chosen = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return f"http://127.0.0.1:{chosen}/{html_path.name}"


def _escape_html(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


# ---------------------------------------------------------------------------
# Self-contained page (CSS + JS). No external assets.
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{{TITLE}}</title>
<style>
:root {
  --gf-green: #76B900;
  --gf-green-bright: #8FE000;
  --gf-bad: #FF6B6B;
  --gf-warn: #E0A800;
  --gf-info: #6EA8FF;
  --bg-page: #000;
  --bg-surface: #0D0D0D;
  --bg-card: #1A1A1A;
  --bg-card-2: #1f1f1f;
  --bg-card-3: #262626;
  --bg-card-hover: #232323;
  --line-1: #262626;
  --line-2: #404040;
  --text-1: #fff;
  --text-2: #CCCCCC;
  --text-3: #A6A6A6;
  --text-4: #737373;
  --font: system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: "JetBrains Mono", "SF Mono", Menlo, monospace;
  --r2: 8px;
  --shadow-1: 0 1px 2px rgba(0,0,0,0.4);
}
body.theme-light {
  --gf-green: #4E7A00;
  --gf-info: #2660CC;
  --gf-bad: #B82E35;
  --gf-warn: #896600;
  --bg-page: #FAFAFA;
  --bg-surface: #FFFFFF;
  --bg-card: #FFFFFF;
  --bg-card-2: #F2F2F2;
  --bg-card-3: #E5E5E5;
  --bg-card-hover: #F2F2F2;
  --line-1: #E5E5E5;
  --line-2: #CCCCCC;
  --text-1: #0D0D0D;
  --text-2: #404040;
  --text-3: #606060;
  --text-4: #686868;
  --shadow-1: 0 1px 2px rgba(0,0,0,0.06);
}
* { box-sizing: border-box; }
html, body {
  margin: 0; padding: 0;
  background: var(--bg-page); color: var(--text-1);
  font-family: var(--font); font-size: 14px; line-height: 1.5;
  -webkit-font-smoothing: antialiased;
}
code, .mono { font-family: var(--mono); font-size: 12px; }
body { min-height: 100vh; }
button { font-family: inherit; }
:focus-visible { outline: 2px solid var(--gf-info); outline-offset: 4px; }
[hidden] { display: none !important; }

.chrome-bar {
  position: sticky; top: 0; z-index: 30;
  background: color-mix(in srgb, var(--bg-surface) 92%, transparent);
  backdrop-filter: blur(20px);
  border-bottom: 1px solid var(--line-1);
}
.chrome-inner { width: 100%; margin: 0 auto; padding: 14px 24px; }
.chrome-row { display: flex; flex-wrap: wrap; align-items: center; gap: 16px; min-height: 44px; }
.brand-mark { display: inline-flex; align-items: center; gap: 10px; padding-right: 16px; border-right: 1px solid var(--line-1); }
.brand-mark .nv-eye {
  width: 18px; height: 18px; background: var(--gf-green);
  clip-path: polygon(50% 0%, 100% 50%, 50% 100%, 0% 50%);
}
.brand-mark .nv-text { font-weight: 700; font-size: 13px; letter-spacing: 0.04em; text-transform: uppercase; }
.nv-product { color: var(--gf-green); }
.doc-title { display: flex; flex: 1 1 280px; flex-direction: column; gap: 2px; min-width: 0; overflow-wrap: anywhere; }
.doc-eyebrow { font-size: 10px; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase; color: var(--text-3); }
.doc-title h1 { margin: 0; font-size: 16px; font-weight: 700; line-height: 1.2; }
.chrome-spacer { flex: 1; }
.chrome-actions { display: inline-flex; align-items: center; gap: 8px; flex-shrink: 0; }
@media (max-width: 720px) {
  .doc-title { flex-basis: 100%; }
  .chrome-spacer { display: none; }
  .chrome-actions { flex-wrap: wrap; max-width: 100%; }
}
.chrome-btn {
  height: 32px; padding: 0 12px;
  background: var(--bg-card-3); color: var(--text-1);
  border: 1px solid var(--line-1); border-radius: 4px;
  font-size: 12px; font-weight: 600; cursor: pointer;
}
.chrome-btn:hover { background: var(--bg-card-hover); border-color: var(--line-2); }
.chrome-btn:disabled { opacity: 0.55; cursor: wait; }

.main-shell { width: 100%; margin: 0 auto; padding: 24px; }
.library-coverage { margin-bottom: 22px; }
.library-coverage h2 { margin: 0 0 6px; font-size: 20px; }
.library-coverage-intro, .library-coverage-note {
  color: var(--text-2); font-size: 13px; line-height: 1.5; margin: 0 0 12px;
}
.library-coverage-grid {
  display: grid; grid-template-columns: minmax(0, 1.4fr) minmax(0, 1fr); gap: 12px;
}
.library-coverage-card {
  padding: 20px 22px; border: 1px solid var(--line-2);
  border-radius: var(--r2); background: var(--bg-card);
}
.library-coverage-card.primary {
  border-color: var(--gf-green); background: rgba(118,185,0,0.055);
}
.library-coverage-card h3 {
  margin: 0; color: var(--text-2); font-size: 12px; font-weight: 700;
  letter-spacing: 0.06em; text-transform: uppercase;
}
.library-coverage-value { font-size: 38px; font-weight: 700; margin: 7px 0; line-height: 1.1; }
.library-coverage-card.primary .library-coverage-value { font-size: 48px; }
.library-coverage-value.measured { color: var(--gf-green); }
.library-coverage-detail { margin: 0; color: var(--text-2); font-size: 13px; line-height: 1.5; }
.library-coverage-scope { display: block; margin-top: 6px; font-weight: 600; }
.library-coverage-note { margin: 10px 0 0; font-size: 12px; }
@media (max-width: 720px) {
  .library-coverage-grid { grid-template-columns: 1fr; }
  .library-coverage-card { padding: 16px; }
}
.kpi-band { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 20px; }
.execution-status {
  border-left: 4px solid #c77b15; padding: 12px 16px;
  margin-bottom: 20px; background: var(--surface);
}
.execution-status p { margin: 6px 0 0; }
@media (max-width: 720px) { .kpi-band { grid-template-columns: 1fr 1fr; } }
.kpi-card {
  background: var(--bg-card); border: 1px solid var(--line-1);
  border-radius: var(--r2); padding: 14px 16px;
}
.kpi-eyebrow { font-size: 10px; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase; color: var(--text-3); }
.kpi-value { font-size: 24px; font-weight: 700; letter-spacing: -0.02em; margin-top: 4px; }
.kpi-value.green { color: var(--gf-green); }
.failure-index { margin-bottom: 24px; }
.failure-index h2 { font-size: 14px; margin: 0 0 8px; }
.failure-index ol { margin: 0; padding-left: 24px; color: var(--text-3); }
.failure-index li { padding: 4px 0 4px 4px; }
.failure-index a {
  color: var(--gf-info); font-weight: 700; overflow-wrap: anywhere;
  text-decoration: underline; text-decoration-thickness: 1px; text-underline-offset: 3px;
}
.failure-index a:hover, .failure-index a:focus-visible {
  background: var(--bg-card-hover); text-decoration-thickness: 2px;
}
.failure-index-toggle {
  margin: 8px 0 0 28px; padding: 6px 12px; border-radius: 6px;
  border: 1px solid var(--line-2); background: var(--bg-card);
  color: var(--gf-info); font-size: 13px; font-weight: 700; cursor: pointer;
}
.failure-index-toggle:hover { background: var(--bg-card-hover); }
.related-failures { margin: 0 0 24px; }
.related-failures h2 { font-size: 16px; margin: 0 0 8px; }
.related-failures p { color: var(--text-3); font-size: 13px; }
.related-group { border: 1px solid var(--line-1); border-radius: 8px;
  padding: 16px; margin: 12px 0; background: var(--bg-card); }
.related-group h3 { font-size: 14px; margin: 0 0 8px; }
.related-group ul { list-style: none; padding-left: 0; margin: 4px 0; }
.related-group li { padding: 4px 0; }
.related-group a, .related-card-note a { color: var(--text-2); text-decoration: underline; }
.related-label { color: var(--text-3); font-weight: 600; margin-right: 6px; }
.related-card-note { margin: 8px 0; font-size: 13px; color: var(--text-3); }
.related-card-note li { margin: 4px 0; }

.tax-grid { display: grid; grid-template-columns: 1fr; gap: 14px; }
.tax-card {
  background: var(--bg-card); border: 1px solid var(--line-1);
  border-left: 4px solid var(--line-2);
  border-radius: var(--r2); padding: 20px 24px;
  box-shadow: var(--shadow-1);
  min-width: 0; max-width: 100%; position: relative;
  scroll-margin-top: 180px;
}
.tax-card.good { border-left-color: var(--gf-green); }
.tax-card.warn { border-left-color: var(--gf-warn); }
.tax-card.bad { border-left-color: var(--gf-bad); }
.tax-card.neutral { border-left-color: var(--line-2); }
.tax-card.failure-selected {
  border-color: var(--gf-info);
  outline: 2px solid color-mix(in srgb, var(--gf-info) 70%, transparent);
  box-shadow: 0 0 0 5px color-mix(in srgb, var(--gf-info) 18%, transparent);
}
.tax-eyebrow {
  font-size: 10px; font-weight: 700; letter-spacing: 0.14em;
  text-transform: uppercase; color: var(--gf-green); margin-bottom: 4px;
}
.tax-card-head {
  display: flex; align-items: flex-start; gap: 12px;
  list-style: none; cursor: pointer; padding-right: 68px;
}
.tax-card-head::-webkit-details-marker { display: none; }
.card-chevron { color: var(--text-3); font-size: 18px; transition: transform .15s; }
.tax-card-content[open] > summary .card-chevron { transform: rotate(90deg); }
.tax-card-content:not([open]) .tax-subtitle {
  margin-bottom: 0;
}
.tax-card-body { padding-top: 12px; }
.tax-card-head:hover .tax-title {
  color: var(--gf-info);
}
.tax-card-heading { flex: 1; min-width: 0; }
.tax-card-link {
  position: absolute; top: 20px; right: 24px;
  color: var(--gf-info); font-size: 12px; font-weight: 600;
  text-decoration: none; border: 1px solid rgba(110,168,255,0.40);
  border-radius: 4px; padding: 4px 8px;
}
.tax-card-link:hover { background: rgba(110,168,255,0.12); color: var(--text-1); }
.tax-title {
  overflow-wrap: anywhere; font-size: 22px; font-weight: 700;
  line-height: 1.25; margin: 0 0 12px 0;
}
.tax-subtitle {
  overflow-wrap: anywhere; font-size: 12px; color: var(--text-3);
  font-family: var(--mono); margin: -8px 0 12px 0;
}
.tax-field { margin: 8px 0 12px 0; }
.tax-field-label {
  font-size: 10px; font-weight: 700; letter-spacing: 0.12em;
  text-transform: uppercase; color: var(--text-3); margin-bottom: 4px;
}
.tax-field-value {
  max-width: 100ch; color: var(--text-1); font-size: 14px;
  line-height: 1.6; white-space: pre-wrap;
}

.tax-statcards {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 8px; margin: 4px 0 14px;
}
.tax-statcard {
  border: 1px solid var(--line-1); border-radius: 8px;
  padding: 8px 12px; min-width: 0; background: var(--bg-card-2);
  display: flex; flex-direction: column; gap: 2px; position: relative;
}
.tax-statcard-label {
  display: flex; align-items: center; justify-content: space-between; gap: 8px;
  min-height: 30px; font-size: 10px; font-weight: 700; letter-spacing: 0.05em;
  text-transform: uppercase; color: var(--text-3);
}
.metric-help { display: inline-flex; flex-shrink: 0; }
.metric-help button {
  display: inline-flex; align-items: center; justify-content: center;
  width: 17px; height: 17px; padding: 0; border: 1px solid currentColor;
  border-radius: 50%; background: transparent; color: var(--text-3);
  font-size: 11px; font-weight: 600; cursor: help;
}
.metric-tooltip {
  display: none; position: absolute; top: calc(100% + 6px); left: 0; z-index: 40;
  width: min(260px, calc(100vw - 96px)); padding: 10px 12px;
  border: 1px solid var(--line-2); border-radius: 6px; background: var(--bg-surface);
  color: var(--text-1); box-shadow: 0 4px 16px rgba(0,0,0,.15);
  font-size: 12px; font-weight: 400; line-height: 1.5; letter-spacing: 0; text-transform: none;
}
.tax-statcard:nth-last-child(-n+2) .metric-tooltip { left: auto; right: 0; }
.metric-help:hover .metric-tooltip, .metric-help:focus-within .metric-tooltip { display: block; }
.tax-statcard-value {
  overflow-wrap: anywhere; font-size: 16px; font-weight: 700; color: var(--text-1);
}
.tax-statcard-detail { font-size: 11px; color: var(--text-3); }
.tax-statcard.good, .tax-statcard.green { --status-color: var(--gf-green); }
.tax-statcard.bad { --status-color: var(--gf-bad); }
.tax-statcard.warn { --status-color: var(--gf-warn); }
.tax-statcard.info { --status-color: var(--gf-info); }
.tax-statcard.neutral { --status-color: var(--text-3); }
.tax-statcard:is(.good, .green, .bad, .warn, .info, .neutral) {
  background: color-mix(in srgb, var(--status-color) 10%, var(--bg-card));
  border-color: color-mix(in srgb, var(--status-color) 40%, var(--line-1));
}
.tax-statcard:is(.good, .green, .bad, .warn, .info, .neutral) .tax-statcard-value {
  color: var(--status-color);
}
.tax-sampled {
  margin: -6px 0 14px 0; padding: 8px 12px; font-size: 11px; line-height: 1.5;
  color: var(--gf-warn); background: color-mix(in srgb, var(--gf-warn) 8%, var(--bg-card));
  border: 1px solid color-mix(in srgb, var(--gf-warn) 30%, var(--line-1)); border-radius: 8px;
}

.tax-detectors-label {
  font-size: 10px; font-weight: 700; letter-spacing: 0.12em;
  text-transform: uppercase; color: var(--text-3); margin: 8px 0 6px 0;
}
.tax-detector {
  border: 1px solid var(--line-1); border-radius: 8px;
  margin: 0 0 6px 0; background: var(--bg-card-2); overflow: hidden;
}
.tax-detector-head {
  display: flex; align-items: center; gap: 10px; cursor: pointer;
  padding: 10px 14px; user-select: none; width: 100%; text-align: left;
  border: 0; background: transparent; color: var(--text-1); flex-wrap: wrap;
}
.detector-hint { color: var(--text-3); font-size: 12px; }
.tax-detector-head:hover { background: var(--bg-card-hover); }
.tax-detector-signature {
  font-family: var(--mono); font-size: 12.5px; color: var(--text-1);
  flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.hl-k { color: #76B900; font-weight: 600; }
.hl-fn { color: #6EA8FF; font-weight: 600; }
.hl-c { color: var(--text-4); font-style: italic; }
.hl-s { color: #C8FF6A; }
.hl-deco { color: #E0A800; }
.hl-b { color: #8FE000; }
.hl-self { color: #E0A800; font-style: italic; }
.hl-n { color: #E0A800; }
body.theme-light .hl-k { color: #4E7A00; }
body.theme-light .hl-fn { color: #2660CC; }
body.theme-light .hl-c { color: #737373; }
body.theme-light .hl-s { color: #5A8F00; }
body.theme-light .hl-deco { color: #B07A00; }
body.theme-light .hl-b { color: #5A8F00; }
body.theme-light .hl-self { color: #B07A00; }
body.theme-light .hl-n { color: #B07A00; }
.chips { display: flex; gap: 6px; flex-wrap: wrap; flex-shrink: 0; }
.chip {
  font-size: 10px; padding: 2px 6px; background: var(--bg-card-3);
  border-radius: 4px; color: var(--text-2); border: 1px solid transparent;
}
.chip.good { color: var(--gf-green); border-color: rgba(118,185,0,0.35); }
.chip.bad { color: var(--gf-bad); border-color: rgba(255,107,107,0.35); }
.chip.warn { color: var(--gf-warn); border-color: rgba(224,168,0,0.35); }
.tax-detector-body {
  display: none; padding: 4px 14px 14px; border-top: 1px solid var(--line-1);
  background: var(--bg-card); font-size: 12px; color: var(--text-2);
  min-width: 0; overflow-x: hidden;
}
.tax-detector.open .tax-detector-body { display: block; }
.tax-detector-body h4 {
  margin: 14px 0 6px; font-size: 11px; font-weight: 700;
  letter-spacing: 0.12em; text-transform: uppercase; color: var(--text-3);
}
.bucket-table-wrap { max-width: 100%; overflow-x: auto; }
.bucket-open {
  border: 0; background: transparent; padding: 6px 0; color: var(--gf-info); font: inherit;
  cursor: pointer; white-space: nowrap;
}
.bucket-open:hover { text-decoration: underline; }
.bucket-table { width: 100%; border-collapse: collapse; font-size: 11px; }
.bucket-table th, .bucket-table td {
  text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--line-1);
}
.bucket-table th { color: var(--text-3); font-weight: 600; font-size: 10px; text-transform: uppercase; letter-spacing: 0.08em; }
.bucket-table tr.hp td { background: rgba(118,185,0,0.08); }
.bucket-table tr.clickable { cursor: pointer; }
.bucket-table tr.clickable:hover td { background: rgba(118,185,0,0.14); }
.bucket-table tr.selected td {
  background: rgba(110,168,255,0.14);
  outline: 1px solid rgba(110,168,255,0.45);
}
.bucket-hint { font-size: 11px; color: var(--text-4); margin: 6px 0 0; }
.tag.clickable { cursor: pointer; }
.tag.clickable:hover { border-color: var(--gf-green); color: var(--text-1); }
.tag.selected {
  border-color: var(--gf-info); color: var(--gf-info);
  background: rgba(110,168,255,0.12);
}

/* ── Inspector drawer (item spot-check) ───────────────────── */
.inspector {
  display: none;
  position: fixed; left: 0; right: 0; bottom: 0; z-index: 50;
  background: var(--bg-card);
  border-top: 1px solid var(--line-2);
  box-shadow: 0 -12px 40px rgba(0,0,0,0.45);
  max-height: min(72vh, 720px);
  flex-direction: column;
}
.inspector.open { display: flex; }
body.has-inspector { padding-bottom: min(72vh, 720px); }
.inspector-head {
  display: flex; align-items: center; gap: 12px;
  padding: 12px 18px; border-bottom: 1px solid var(--line-1);
  flex-shrink: 0;
}
.ins-eyebrow {
  font-size: 10px; font-weight: 700; letter-spacing: 0.14em;
  text-transform: uppercase; color: var(--text-3);
}
.ins-title {
  font-size: 15px; font-weight: 700; color: var(--text-1);
  font-family: var(--mono); word-break: break-word; flex: 1; min-width: 0;
}
.inspector-body {
  display: grid;
  grid-template-columns: 1.5fr 1fr;
  grid-template-rows: minmax(0, 1fr);
  gap: 0; min-height: 0; flex: 1; overflow: hidden;
  height: min(60vh, 620px);
}
.inspector.single-item .inspector-body { grid-template-columns: minmax(0, 1fr); }
.inspector.single-item .inspector-side { display: none; }
.inspector.single-item .inspector-main { border-right: 0; }
@media (max-width: 780px) {
  .inspector-body { grid-template-columns: 1fr; grid-template-rows: 1fr auto; }
  .inspector-side { border-left: 0; border-top: 1px solid var(--line-1); max-height: 180px; }
}
.inspector-main, .inspector-side {
  padding: 14px 18px; overflow-y: auto; min-height: 0; min-width: 0;
}
.inspector-main { border-right: 1px solid var(--line-1); }
.inspector-main h4, .inspector-side h4 {
  margin: 12px 0 6px; font-size: 11px; font-weight: 700;
  letter-spacing: 0.12em; text-transform: uppercase; color: var(--text-3);
}
.inspector-main h4:first-child, .inspector-side h4:first-child { margin-top: 0; }
.ins-section {
  margin: 0 0 10px; border: 1px solid var(--line-1); border-radius: 6px;
  background: var(--bg-card-2); overflow: hidden;
}
.ins-section > summary {
  cursor: pointer; user-select: none; list-style: none;
  padding: 8px 12px; font-size: 11px; font-weight: 700;
  letter-spacing: 0.12em; text-transform: uppercase; color: var(--text-3);
  display: flex; align-items: center; gap: 8px;
}
.ins-section > summary::-webkit-details-marker { display: none; }
.ins-section > summary::before {
  content: '▾'; font-size: 10px; color: var(--text-4); width: 10px;
}
.ins-section:not([open]) > summary::before { content: '▸'; }
.ins-section > summary:hover { color: var(--text-1); background: var(--bg-card-hover); }
.ins-section > summary .ins-sec-hint {
  margin-left: auto; font-weight: 500; letter-spacing: 0; text-transform: none;
  font-size: 10px; color: var(--text-4);
}
.failure-localisation, .item-validation { margin-bottom: 20px; }
.failure-localisation p, .item-validation p {
  font-size: 12px; color: var(--text-2); line-height: 1.5;
}
.failure-localisation .ins-section > summary {
  text-transform: none; letter-spacing: 0; font-family: var(--mono);
}
.passage-heading { min-width: 0; overflow-wrap: anywhere; }
.passage-destination, .passage-source-destination, .passage-position, .passage-coordinates {
  display: block; line-height: 1.6;
}
.passage-destination { color: var(--text-1); font-size: 12px; }
.passage-source-destination { color: var(--gf-info); }
.passage-position { color: var(--text-2); font-weight: 400; }
.passage-coordinates { color: var(--text-3); font-weight: 400; }
.failure-localisation mark, mark.trace-hit {
  color: var(--text-1); background: rgba(255, 190, 60, 0.3);
  border-bottom: 2px solid #d59419; border-radius: 2px;
}
.passage-jump { cursor: pointer; }
.passage-jump:hover, .passage-jump:focus-visible { outline: 2px solid var(--gf-info); }
.trace-links { display: flex; flex-wrap: wrap; align-items: center; gap: 10px; padding: 8px 12px; }
.trace-links button, [data-trace-step] {
  color: var(--text-1); background: var(--bg-card-2); border: 1px solid var(--line-1);
  padding: 5px 9px; border-radius: 4px; cursor: pointer;
}
.trace-event > summary { text-transform: none; letter-spacing: 0; }
.trace-labels, .trace-preview {
  display: block; font-size: 11px; line-height: 1.6; font-weight: 400;
}
.trace-labels { color: var(--gf-info); }
.trace-preview { color: var(--text-3); }
.trace-event.trace-selected { border-color: var(--gf-info); }
.trace-part h4 { padding: 0 12px; margin: 10px 0 4px; }
.trace-combined { margin: 20px 0; }
.trace-combined > summary { cursor: pointer; padding: 10px 0; color: var(--text-3); }
.trace-search { width: 100%; padding: 9px; margin-bottom: 8px; box-sizing: border-box;
  color: var(--text-1); background: var(--bg-card-2);
  border: 1px solid var(--line-1); border-radius: 5px; }
.trace-search-results { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 10px; }
.ins-field-chip {
  font-family: var(--mono); font-size: 10px; font-weight: 500;
  letter-spacing: 0; text-transform: none;
  color: var(--gf-info); border: 1px solid rgba(110,168,255,0.35);
  background: rgba(110,168,255,0.08);
  border-radius: 3px; padding: 1px 6px; margin-left: 8px;
}
.ins-field-note {
  margin: 0 0 10px; padding: 8px 12px;
  background: var(--bg-card-2); border: 1px solid var(--line-1);
  border-left: 3px solid var(--gf-info); border-radius: 0 6px 6px 0;
  font-size: 12px; color: var(--text-2); line-height: 1.45;
}
.ins-field-note code {
  font-family: var(--mono); font-size: 11px; color: var(--gf-info);
}
.ins-section pre {
  margin: 0; padding: 10px 12px; border-top: 1px solid var(--line-1);
  max-height: none; overflow: visible;
  white-space: pre-wrap; word-break: break-word;
  font-family: var(--mono); font-size: 11px; line-height: 1.45; color: var(--text-1);
  background: transparent; border: 0; border-radius: 0;
}
.inspector-main pre {
  margin: 0 0 10px; padding: 10px 12px;
  background: var(--bg-card-2); border: 1px solid var(--line-1);
  border-radius: 6px; max-height: none; overflow: visible;
  white-space: pre-wrap; word-break: break-word;
  font-family: var(--mono); font-size: 11px; line-height: 1.45; color: var(--text-1);
}
.ins-stat-strip { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 10px; }
.ins-stat {
  border: 1px solid var(--line-1); border-radius: 6px;
  padding: 6px 10px; background: var(--bg-card-2); min-width: 72px;
}
.ins-stat-label {
  font-size: 9px; font-weight: 700; letter-spacing: 0.1em;
  text-transform: uppercase; color: var(--text-3);
}
.ins-stat-value { font-size: 14px; font-weight: 700; margin-top: 2px; }
.ins-stat-value.bad { color: var(--gf-bad); }
.ins-stat-value.green { color: var(--gf-green); }
.ins-source-row { margin: 0 0 10px; }
.ins-source-link {
  display: inline-flex; align-items: center; gap: 5px;
  color: var(--gf-green); font-size: 12px; font-weight: 700;
  text-decoration: none;
}
.ins-source-link:hover { text-decoration: underline; }
.ins-empty { color: var(--text-3); font-size: 13px; padding: 24px 8px; text-align: center; }
.python-source {
  display: block;
  box-sizing: border-box;
  width: 100%;
  max-width: 100%;
  max-height: 280px;
  margin: 0;
  overflow: auto;
  font-size: 11px;
  background: var(--bg-card-2); border: 1px solid var(--line-1);
  border-radius: 6px; padding: 10px 12px;
  /* Keep indent, but wrap long regex / string lines inside the card width */
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  word-break: break-word;
  color: var(--text-1); font-family: var(--mono); line-height: 1.45;
  tab-size: 4;
}
.python-source .hl-k { font-weight: 600; }
.taglist { display: flex; flex-wrap: wrap; gap: 6px; }
.tag {
  font-family: var(--mono); font-size: 11px; padding: 2px 8px;
  border-radius: 4px; background: var(--bg-card-3); color: var(--text-2);
  border: 1px solid var(--line-1); max-width: 100%; overflow: hidden; text-overflow: ellipsis;
}
.tag.bad { color: var(--gf-bad); border-color: rgba(255,107,107,0.35); }
.tag.green { color: var(--gf-green); border-color: rgba(118,185,0,0.35); }
.taglist-scroll {
  max-height: 220px; overflow-y: auto; padding-right: 4px;
}
.review-summary {
  max-width: 110ch; margin: 0 0 16px; padding: 10px 12px; border-left: 3px solid var(--line-2);
  color: var(--text-2); font-size: 13px;
}
.review-summary strong { color: var(--text-1); }
.review-details { margin-bottom: 16px; }
.review-details > .review-summary { margin-bottom: 0; }
@media (max-width: 820px) {
  .tax-statcards { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .tax-statcard:nth-child(odd) .metric-tooltip { left: 0; right: auto; }
  .tax-statcard:nth-child(even) .metric-tooltip { left: auto; right: 0; }
  .tax-statcard:last-child .metric-tooltip { left: auto; right: 0; }
}
@media (max-width: 540px) {
  .chrome-bar { position: static; }
  .main-shell, .chrome-inner { padding: 16px; }
  .tax-card { padding: 16px; }
  .tax-card-link { top: 16px; right: 16px; }
  .tax-title { font-size: 19px; }
}
details.tax-details { margin-top: 10px; border-top: 1px solid var(--line-1); padding-top: 10px; }
details.tax-details > summary {
  cursor: pointer; user-select: none; list-style: none;
  font-size: 12px; font-weight: 600; color: var(--text-3); padding: 6px 0;
}
details.tax-details > summary::-webkit-details-marker { display: none; }
details.tax-details > summary::before { content: '▸'; margin-right: 6px; font-size: 10px; }
details.tax-details[open] > summary::before { content: '▾'; }
details.tax-details > summary:hover { color: var(--text-1); }
.empty { color: var(--text-3); padding: 40px 0; text-align: center; }
[data-tip] { cursor: help; }
</style>
</head>
<body class="theme-light">
<header class="chrome-bar">
  <div class="chrome-inner">
    <div class="chrome-row">
      <div class="brand-mark">
        <span class="nv-eye" aria-hidden="true"></span>
        <span class="nv-text">NVIDIA <span class="nv-product">AutoFA</span></span>
      </div>
      <div class="doc-title">
        <div class="doc-eyebrow">Model Failure Cards</div>
        <h1 id="doc-h1">Failure modes</h1>
      </div>
      <div class="chrome-spacer"></div>
      <div class="chrome-actions">
        <button type="button" class="chrome-btn" id="download-html" title="Download this report as an .html file">Download HTML</button>
        <button type="button" class="chrome-btn" id="download-zip" title="Download a self-contained ZIP (HTML with trajectories embedded + library.json). Works offline — no server.">Download ZIP</button>
        <button type="button" class="chrome-btn" id="theme-toggle">Dark</button>
      </div>
    </div>
  </div>
</header>
<main class="main-shell">
  <section id="library-coverage" class="library-coverage"
    aria-labelledby="library-coverage-title"></section>
  <section class="execution-status" id="execution-status" role="status" hidden></section>
  <div class="kpi-band" id="kpi-band"></div>
  <nav id="failure-index" class="failure-index" aria-label="Failure modes at a glance" hidden></nav>
  <section id="related-failures" class="related-failures"
    aria-labelledby="related-failures-title" hidden></section>
  <div id="cards"></div>
</main>
<aside id="inspector" class="inspector" aria-live="polite">
  <div class="inspector-head">
    <div style="min-width:0; flex:1;">
      <div class="ins-eyebrow" id="ins-eyebrow">Example</div>
      <div class="ins-title" id="ins-title"></div>
    </div>
    <button type="button" class="chrome-btn" id="ins-close">Close</button>
  </div>
  <div class="inspector-body">
    <div class="inspector-main" id="ins-main"></div>
    <div class="inspector-side" id="ins-side"></div>
  </div>
</aside>
<script id="autofa-data" type="application/json">{{DATA_JSON}}</script>
<script>
{{VERDICT_JS}}
const RAW = JSON.parse(document.getElementById('autofa-data').textContent);
const ITEMS = Object.assign({}, RAW.items || {});
const ITEM_INDEX = RAW.item_index || {};
const ITEMS_BASE = (RAW.meta && RAW.meta.items_base) || null;
const STATE = { selection: null }; // { type:'bucket'|'item', fn, bucket?, itemId? }
const FAILURE_LINK_MESSAGE = Object.freeze({
  namespace: 'autofa.failure-cards',
  type: 'failure-link-request',
});

async function ensureItem(id) {
  if (!id) return null;
  if (ITEMS[id]) return ITEMS[id];
  const file = ITEM_INDEX[id];
  if (!file || !ITEMS_BASE) return null;
  try {
    const resp = await fetch(`${ITEMS_BASE}/${file}`);
    if (!resp.ok) throw new Error(String(resp.status));
    const payload = await resp.json();
    // Local dashboard assets resolve saved passages against the immutable trace
    // on demand. Install those verified mappings before rendering the inspector.
    for (const fn of RAW.functions || []) {
      const record = (payload.localisations || {})[fn.name];
      const saved = (fn.failure_localisations || {})[id];
      if (record && saved && record.detector_hash === saved.detector_hash) {
        fn.failure_localisations[id] = record;
      }
    }
    ITEMS[id] = payload;
    return payload;
  } catch (e) {
    return { __load_error: true, item_id: id };
  }
}

function escapeHtml(s) {
  return String(s == null ? '' : s)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
function pct(x, digits=0) {
  if (x == null || Number.isNaN(x)) return '—';
  return (100 * Number(x)).toFixed(digits) + '%';
}
function num(x, digits=2) {
  if (x == null || Number.isNaN(x)) return '—';
  return Number(x).toFixed(digits);
}
// Lightweight Python highlighter (same approach as failures-scale dashboard).
const PY_KEYWORDS = ['if','elif','else','for','while','try','except','finally','return','yield','import','from','as','with','pass','break','continue','not','and','or','in','is','True','False','None','lambda','raise','global','nonlocal','async','await','assert','del'];
const PY_BUILTINS = ['len','range','print','int','float','str','list','dict','set','tuple','bool','bytes','enumerate','zip','map','filter','sorted','reversed','isinstance','issubclass','type','hasattr','getattr','setattr','min','max','sum','abs','round','any','all','iter','next','open','repr','id','dir','vars','staticmethod','classmethod','property','super','frozenset','chr','ord','hex','oct','bin','format','divmod','pow','callable'];
const _TQD = '"' + '""';
const _TQS = "'" + "''";
const _DQ = '"';
const _SQ = "'";
const PY_TOKEN_RE = new RegExp([
  '(#[^\\n]*)',
  '(' + _TQD + '[\\s\\S]*?' + _TQD + '|' + _TQS + '[\\s\\S]*?' + _TQS
    + '|' + _DQ + '(?:\\\\.|[^' + _DQ + '\\\\\\n])*' + _DQ
    + '|' + _SQ + '(?:\\\\.|[^' + _SQ + '\\\\\\n])*' + _SQ + ')',
  '\\b(def|class)\\b(\\s+)([A-Za-z_]\\w*)',
  '(@\\w+(?:\\.\\w+)*)',
  '\\b(' + PY_KEYWORDS.join('|') + ')\\b',
  '\\b(self|cls)\\b',
  '\\b(' + PY_BUILTINS.join('|') + ')\\b',
  '\\b(0x[0-9a-fA-F]+|0o[0-7]+|0b[01]+|\\d+(?:\\.\\d+)?(?:[eE][+-]?\\d+)?)\\b'
].join('|'), 'g');

function highlightPython(src) {
  if (!src) return '';
  const escaped = String(src)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  return escaped.replace(PY_TOKEN_RE, (m, comment, str, defkw, ws, defname, deco, kw, slf, bi, num) => {
    if (comment) return `<span class="hl-c">${comment}</span>`;
    if (str)     return `<span class="hl-s">${str}</span>`;
    if (defkw)   return `<span class="hl-k">${defkw}</span>${ws}<span class="hl-fn">${defname}</span>`;
    if (deco)    return `<span class="hl-deco">${deco}</span>`;
    if (kw)      return `<span class="hl-k">${kw}</span>`;
    if (slf)     return `<span class="hl-self">${slf}</span>`;
    if (bi)      return `<span class="hl-b">${bi}</span>`;
    if (num)     return `<span class="hl-n">${num}</span>`;
    return m;
  });
}

function highlightSig(name, source) {
  let sig = `def ${name}(item, llm_client=None):`;
  if (source) {
    const m = source.match(/^def\s+\w+\s*\([^)]*\)\s*(?:->\s*[^:]+)?\s*:/m);
    if (m) sig = m[0];
  }
  return highlightPython(sig);
}

let metricTipId = 0;
function sc(label, val, cls, tip, detail = '') {
  const id = `metric-tip-${++metricTipId}`;
  return `<div class="tax-statcard ${cls||''}">
    <div class="tax-statcard-label">${escapeHtml(label)}${tip ? `
      <span class="metric-help"><button type="button" aria-label="About ${escapeHtml(label)}"
        aria-describedby="${id}">i</button><span class="metric-tooltip" role="tooltip"
        id="${id}">${escapeHtml(tip)}</span></span>` : ''}</div>
    <div class="tax-statcard-value">${escapeHtml(val)}</div>
    ${detail ? `<div class="tax-statcard-detail">${escapeHtml(detail)}</div>` : ''}
  </div>`;
}

function setDetectorOpen(detector, open) {
  detector.classList.toggle('open', open);
  detector.querySelector('.tax-detector-head').setAttribute('aria-expanded', String(open));
}

function findFn(name) {
  return (RAW.functions || []).find(f => f.name === name) || null;
}

function requestedFailureName() {
  const query = new URLSearchParams(window.location.search).get('failure');
  if (query) return query;
  const rawHash = window.location.hash.replace(/^#/, '');
  if (!rawHash) return null;
  const hashSelection = new URLSearchParams(rawHash).get('failure');
  if (hashSelection) return hashSelection;
  try {
    const card = document.getElementById(decodeURIComponent(rawHash));
    return card && card.getAttribute('data-function-name');
  } catch (e) {
    return null;
  }
}

function focusFailureMode(functionName) {
  document.querySelectorAll('.tax-card.failure-selected').forEach(card => {
    card.classList.remove('failure-selected');
  });
  if (!functionName) return false;
  const fn = findFn(functionName);
  if (!fn) return false;
  const card = document.getElementById(fn.card_id);
  if (!card) return false;
  card.classList.add('failure-selected');
  const detector = card.querySelector('.tax-detector');
  card.querySelector('.tax-card-content').open = true;
  if (detector) setDetectorOpen(detector, true);
  card.scrollIntoView({ behavior: 'smooth', block: 'start' });
  card.focus({ preventScroll: true });
  return true;
}

function postFailureLink(functionName) {
  // A sandboxed iframe has an opaque origin, so targetOrigin cannot name the
  // wrapper. Constrain the bridge to window.parent and this tiny namespaced shape.
  window.parent.postMessage({
    ...FAILURE_LINK_MESSAGE,
    function_name: functionName,
  }, '*');
}

function findBucket(fn, bucket) {
  if (!fn) return null;
  return (fn.buckets || []).find(b => String(b.bucket) === String(bucket)) || null;
}

function shortId(id) {
  const s = String(id || '');
  if (s.length <= 48) return s;
  const parts = s.split('|');
  if (parts.length >= 3) return parts.slice(2).join('|');
  return s.slice(0, 20) + '…' + s.slice(-20);
}

function renderBuckets(buckets, fnName, metricsUnavailable=false, unavailableReason='') {
  if (metricsUnavailable) return '<p class="bucket-hint">'
    + escapeHtml(unavailableReason || 'No resolved detector results') + '; '
    + 'bucket metrics are unavailable.</p>';
  if (!buckets || !buckets.length) return '<p style="color:var(--text-4);">No buckets</p>';
  const rows = buckets.map(b => {
    const hp = b.is_high_purity ? 'hp' : '';
    const failIds = b.example_fail_ids || [];
    const passIds = b.example_pass_ids || [];
    const nEx = failIds.length + passIds.length;
    const clickable = nEx ? 'clickable' : '';
    const sel = STATE.selection
      && STATE.selection.type === 'bucket'
      && STATE.selection.fn === fnName
      && String(STATE.selection.bucket) === String(b.bucket)
      ? 'selected' : '';
    const recCi = (b.recall_ci_low != null && b.recall_ci_high != null)
      ? ` [${pct(b.recall_ci_low,0)}–${pct(b.recall_ci_high,0)}]` : '';
    const fprTxt = (b.fpr != null)
      ? `${pct(b.fpr,1)}${(b.fpr_ci_low != null && b.fpr_ci_high != null)
          ? ` [${pct(b.fpr_ci_low,0)}–${pct(b.fpr_ci_high,0)}]` : ''}`
      : '—';
    const tip = nEx
      ? `Click to inspect ${failIds.length} fail + ${passIds.length} pass examples`
      : '';
    return `<tr class="${hp} ${clickable} ${sel}" data-fn="${escapeHtml(fnName)}" data-bucket="${escapeHtml(b.bucket)}" ${tip ? `title="${escapeHtml(tip)}"` : ''}>
      <td class="mono">${nEx ? `<button type="button" class="bucket-open"
        aria-label="View examples in bucket ${escapeHtml(b.bucket)}">${escapeHtml(b.bucket)}
        <span aria-hidden="true">↗</span></button>` : escapeHtml(b.bucket)}</td>
      <td>${b.n ?? '—'}</td>
      <td>${b.n_fail ?? '—'}</td>
      <td>${pct(b.purity, 0)}</td>
      <td>${pct(b.recall, 1)}${recCi}</td>
      <td>${num(b.ppv_at_pi0, 2)}</td>
      <td>${fprTxt}</td>
    </tr>`;
  }).join('');
  const anyEx = buckets.some(b =>
    (b.example_fail_ids || []).length + (b.example_pass_ids || []).length);
  return `<div class="bucket-table-wrap"><table class="bucket-table">
    <thead><tr>
      <th>Bucket</th><th>Items</th><th>Failures</th><th>Failure rate</th>
      <th>Coverage [CI]</th><th>Normalized precision</th><th>False positive rate [CI]</th>
    </tr></thead>
    <tbody>${rows}</tbody>
  </table></div>
  <p class="bucket-hint">${anyEx
    ? 'Select a bucket ↗ to view its failed and passed examples. ' : ''}
    CI shows the estimated range (95% confidence interval).</p>`;
}

function renderItemTags(ids, selectedId, cls) {
  return (ids || []).map(id => {
    const has = !!(ITEMS[id] || ITEM_INDEX[id]);
    const sel = selectedId === id ? 'selected' : '';
    return `<span class="tag ${cls || 'bad'} clickable ${sel}" data-jump-item="${escapeHtml(id)}" title="${escapeHtml(id)}${has ? '' : ' (text not available)'}">${escapeHtml(shortId(id))}</span>`;
  }).join('');
}

function insSection(title, body, { open = true, field = null } = {}) {
  const text = body == null || body === '' ? '(empty)' : String(body);
  const chip = field
    ? `<span class="ins-field-chip" title="Detector field: item.get('${field}')">item["${escapeHtml(field)}"]</span>`
    : '';
  const fieldAttr = field ? ` data-field="${escapeHtml(field)}"` : '';
  return `<details class="ins-section"${fieldAttr}${open ? ' open' : ''}>
    <summary><span>${escapeHtml(title)}${chip}</span><span class="ins-sec-hint">click to collapse</span></summary>
    <pre>${escapeHtml(text)}</pre>
  </details>`;
}

function currentItem() {
  return STATE.selection && ITEMS[STATE.selection.itemId];
}

function renderTrace(it) {
  const events = it.trace_events || [];
  if (!events.length) return '';
  return `<section class="trace-timeline"><h3>Trace · ${events.length} events</h3>
    <p>Execution order. Expand an event to read its full contents. Linked calls and results
      keep the surrounding context accessible.</p>
    <input class="trace-search" data-trace-search type="search" aria-label="Search full trace"
      placeholder="Search all events, including collapsed contents…">
    <div class="trace-search-results" aria-live="polite"></div>
    ${events.map(event => {
      const identity = event.source_event ? ` · source event ${event.source_event.event_idx}` : '';
      const labels = event.sections.map(s => s.label).join(' · ');
      const preview = Array.from((event.sections[0] || {}).text || '');
      const excerpt = preview.slice(0, 130).join('') + (preview.length > 130 ? '…' : '');
      return `<details class="trace-event ins-section" data-step-index="${event.index}">
        <summary><span>Step ${event.index} · ${escapeHtml(event.source)}${escapeHtml(identity)}
          <span class="trace-labels">${escapeHtml(labels)}</span>
          <span class="trace-preview">${escapeHtml(excerpt)}</span>
        </span></summary><div class="trace-event-body"></div></details>`;
    }).join('')}</section>`;
}

function fillTraceEvent(details) {
  const it = currentItem();
  const index = Number(details.dataset.stepIndex);
  const event = it && (it.trace_events || []).find(e => e.index === index);
  if (!event) return;
  const body = details.querySelector('.trace-event-body');
  if (body.dataset.loaded) return;
  body.dataset.loaded = 'true';
  const related = [];
  if (event.paired_step_index) related.push([event.paired_step_index, 'Calling step']);
  (event.result_step_indices || []).forEach(i => related.push([i, 'Result step']));
  body.innerHTML = `<div class="trace-links">${related.map(([index, label]) =>
      `<button type="button" data-trace-step="${index}">${label} ${index}</button>`).join(' ')}
      ${event.source_url ? `<a class="ins-source-link" href="${escapeHtml(event.source_url)}"
        target="_blank" rel="noopener noreferrer">Open source event ↗</a>` : ''}</div>`
    + event.sections.map(part => `<section class="trace-part"
        data-section="${escapeHtml(part.key)}">
        <h4>${escapeHtml(part.label)}</h4><pre>${escapeHtml(part.text)}</pre></section>`).join('');
}

function showTraceLocation(location, quote) {
  const selector = `.trace-event[data-step-index="${Number(location.step_index)}"]`;
  const details = document.querySelector(selector);
  if (!details) return false;
  details.open = true;
  fillTraceEvent(details);
  let target = details;
  const part = Array.from(details.querySelectorAll('.trace-part'))
    .find(p => p.dataset.section === location.section);
  if (part && quote) {
    const pre = part.querySelector('pre');
    const text = Array.from(pre.textContent);
    const start = location.start, end = location.end;
    if (Number.isInteger(start) && Number.isInteger(end) && start >= 0 && end > start
        && text.slice(start, end).join('') === quote) {
      pre.innerHTML = escapeHtml(text.slice(0, start).join('')) + '<mark class="trace-hit">'
        + escapeHtml(quote) + '</mark>' + escapeHtml(text.slice(end).join(''));
      target = pre.querySelector('mark');
    } else target = part;
  }
  target.scrollIntoView({block: 'center'});
  details.classList.add('trace-selected');
  return true;
}

function openSource(url) {
  if (!url || !/^https?:\/\//.test(url)) return;
  if (window.parent !== window) window.parent.postMessage({
    namespace: 'autofa.source-data', type: 'open', url,
  }, '*');
  else window.open(url, '_blank', 'noopener,noreferrer');
}

function jumpPassage(index) {
  const selection = STATE.selection || {};
  const fn = findFn(selection.fn);
  const record = fn && (fn.failure_localisations || {})[selection.itemId];
  const p = record && (record.passages || [])[index];
  if (!p) return;
  document.querySelectorAll('.trace-selected').forEach(el => el.classList.remove('trace-selected'));
  for (const loc of p.trace_locations || []) {
    if (showTraceLocation(loc, loc.quote || p.quote)) return;
  }
  const external = (p.trace_locations || []).find(loc => loc.source_url);
  if (external) { openSource(external.source_url); return; }
  // Plain-text and older reports still navigate to the exact field occurrence.
  const it = currentItem();
  if (!it || (it.field_hashes || {})[p.field] !== p.field_sha256) return;
  const fieldMap = (it && it.field_map) || {};
  const field = p.field === 'reasoning_text' ? (fieldMap.reasoning || p.field) : p.field;
  const section = Array.from(document.querySelectorAll('details[data-field]')).find(el => {
    if (el.dataset.field !== field) return false;
    const text = Array.from(el.querySelector('pre').textContent);
    return text.length === p.field_chars && text.slice(p.start, p.end).join('') === p.quote;
  });
  if (!section) return;
  const pre = section.querySelector('pre');
  const text = Array.from(pre.textContent);
  if (text.length !== p.field_chars || text.slice(p.start, p.end).join('') !== p.quote) return;
  for (let parent = section; parent; parent = parent.parentElement) {
    if (parent.tagName === 'DETAILS') parent.open = true;
  }
  pre.innerHTML = escapeHtml(text.slice(0, p.start).join('')) + '<mark class="trace-hit">'
    + escapeHtml(p.quote) + '</mark>' + escapeHtml(text.slice(p.end).join(''));
  pre.querySelector('mark').scrollIntoView({block: 'center'});
}

document.addEventListener('toggle', ev => {
  if (ev.target.matches && ev.target.matches('.trace-event') && ev.target.open) {
    fillTraceEvent(ev.target);
  }
}, true);

document.addEventListener('input', ev => {
  if (!ev.target.matches('[data-trace-search]')) return;
  const query = ev.target.value.toLowerCase();
  const events = (currentItem() || {}).trace_events || [];
  const matches = query
    ? events.filter(e => e.sections.some(s => s.text.toLowerCase().includes(query))) : [];
  const limitNote = matches.length > 50 ? ' · first 50 shown' : '';
  document.querySelector('.trace-search-results').innerHTML = query
    ? `<span>${matches.length} matching events${limitNote}</span>`
      + matches.slice(0, 50).map(e => `<button type="button" data-trace-step="${e.index}">
          Step ${e.index} · ${escapeHtml(e.source)}</button>`).join('')
    : '';
});

function renderFailureLocalisation(fn, itemId) {
  const record = fn && (fn.failure_localisations || {})[itemId];
  const heading = '<h3>Failure localisation</h3>';
  if (!record) {
    return `<section class="failure-localisation">${heading}
      <p>No passage evidence saved for this detector and example.
      Re-evaluate older detectors to collect localisation for their positive labels.</p></section>`;
  }
  const passages = record.passages || [];
  const status = passages.length
    ? `<p>Highlighted candidate evidence with surrounding context when available.
       Coordinates refer to the highlighted passage in the named field:
       Unicode characters, starting at 0, end exclusive; lines and columns start at 1.</p>`
    : '<p>No verifiable passage was located. Inspect the full trace for this detector’s claim.</p>';
  const body = passages.map((p, index) => {
    const coordinates = `Detector field: ${p.field} · characters [${p.start}, ${p.end}) · `
      + `line ${p.start_line}:${p.start_column} to ${p.end_line}:${p.end_column}`;
    // Python offsets count Unicode characters; JS string offsets count UTF-16
    // units. Slice code points so emoji before a hit cannot move the highlight.
    const context = Array.from(p.context || '');
    const start = p.start - p.context_start;
    const end = p.end - p.context_start;
    const hasContext = Number.isInteger(start) && Number.isInteger(end)
      && start >= 0 && end > start && end <= context.length
      && context.slice(start, end).join('') === p.quote;
    const highlighted = hasContext
      ? `${p.context_start > 0 ? '…' : ''}${escapeHtml(context.slice(0, start).join(''))}`
        + `<mark>${escapeHtml(context.slice(start, end).join(''))}</mark>`
        + `${escapeHtml(context.slice(end).join(''))}${p.context_end < p.field_chars ? '…' : ''}`
      : `<mark>${escapeHtml(p.quote || '')}</mark>`;
    const it = ITEMS[itemId];
    const locations = p.trace_locations || [];
    const external = locations.find(loc => loc.source_url);
    const events = (it && it.trace_events) || [];
    const local = locations.find(loc => events.some(event => event.index === loc.step_index));
    const event = local && events.find(event => event.index === local.step_index);
    const destination = local || external || locations[0];
    const part = event && event.sections.find(part => part.key === local.section);
    const sectionLabel = part ? part.label : destination
      ? ({reasoning: 'Reasoning', message: 'Message', observation: 'Tool result'}[
          destination.section]
        || (String(destination.section || '').startsWith('tool:') ? 'Tool call' : 'Passage')) : '';
    const destinationLabel = event
      ? `Step ${event.index} · ${event.source} · ${sectionLabel}` : sectionLabel;
    const sourceEvent = external && external.source_event;
    const sourceLabel = sourceEvent && Number.isInteger(sourceEvent.event_idx)
      ? `event e${sourceEvent.event_idx}`
        + (sourceEvent.part_idx != null ? ` · part p${sourceEvent.part_idx}` : '') : 'event';
    const sourceDestination = external
      ? `Trace viewer: ${sourceLabel} · `
        + (sourceEvent && sourceEvent.lane ? `lane ${sourceEvent.lane}` : 'conversation')
      : it && it.source_data_url ? 'Trace viewer: whole rollout · event mapping unavailable' : '';
    let position = '';
    if (part && Number.isInteger(local.start) && Number.isInteger(local.end)) {
      const chars = Array.from(part.text);
      if (local.start >= 0 && local.end > local.start && local.end <= chars.length
          && chars.slice(local.start, local.end).join('') === (local.quote || p.quote)) {
        const lineAt = offset => {
          const prefix = chars.slice(0, offset);
          const line = prefix.filter(ch => ch === '\n').length + 1;
          return `${line}:${offset - prefix.lastIndexOf('\n')}`;
        };
        position = `In this step: line ${lineAt(local.start)} to ${lineAt(local.end)}`
          + ` · characters [${local.start}, ${local.end})`;
      }
    }
    const canJump = external || (it && !it.source_only &&
      (locations.length || (it.field_hashes || {})[p.field] === p.field_sha256));
    const sourceLink = external
      ? `<a class="ins-source-link" href="${escapeHtml(external.source_url)}"
          title="Opens the source event. Go to passage highlights the exact text in this report."
          target="_blank" rel="noopener noreferrer">Open viewer ${escapeHtml(sourceLabel)} ↗</a>`
      : it && it.source_data_url
        ? `<a class="ins-source-link" href="${escapeHtml(it.source_data_url)}" target="_blank"
            rel="noopener noreferrer">Open whole rollout ↗</a>` : '';
    const jumpAttrs = canJump ? ` class="passage-jump" role="button" tabindex="0"
      data-passage-index="${index}" aria-label="Go to highlighted passage"` : '';
    const jumpButton = canJump ? `<button type="button" data-passage-index="${index}">
      Go to passage</button>` : '';
    return `<details class="ins-section" open>
      <summary><span class="passage-heading">
        ${destinationLabel ? `<span class="passage-destination">${
          escapeHtml(destinationLabel)}</span>` : ''}
        ${sourceDestination ? `<span class="passage-source-destination">${
          escapeHtml(sourceDestination)}</span>` : ''}
        ${position ? `<span class="passage-position">${escapeHtml(position)}</span>` : ''}
        <span class="passage-coordinates">${escapeHtml(coordinates)}</span></span>
        <span class="ins-sec-hint">${escapeHtml(p.method || '')}</span></summary>
      <div${jumpAttrs}>
        <pre>${highlighted}</pre></div>
      <div class="trace-links">${jumpButton} ${sourceLink}</div>
    </details>`;
  }).join('');
  const notes = (record.notes || []).map(n => `<p>${escapeHtml(n)}</p>`).join('');
  return `<section class="failure-localisation">${heading}${status}${body}${notes}
    ${record.truncated ? '<p>Passage evidence was limited; consult the full fields.</p>' : ''}
    </section>`;
}

function renderItemMain(it) {
  if (!it) {
    const nIdx = Object.keys(ITEM_INDEX).length;
    const base = ITEMS_BASE || '(none)';
    return `<div class="ins-empty">No trajectory payload for this example.<br/><br/>
      items_base=<code>${escapeHtml(base)}</code> · indexed=${nIdx}.<br/>
      Regenerate from a run (or Browse library after a run exists for this model) so AutoFA can
      bind the parquet and write <code>*_data/items/</code> sidecars. Then open the
      <code>http://127.0.0.1/…</code> link (not <code>file://</code>).</div>`;
  }
  if (it.__load_error) {
    return `<div class="ins-empty">Could not load full trajectory for<br/><code>${escapeHtml(it.item_id || '')}</code>.<br/><br/>
      This live report loads trajectories from sidecar files over
      <code>http://127.0.0.1/…</code>. For a portable offline copy, use
      <strong>Download ZIP</strong> (embeds trajectories in the HTML so
      <code>file://</code> works).</div>`;
  }
  const fail = it.label === 0 || it.is_correct === false;
  if (it.source_only) {
    const label = it.is_correct == null ? 'UNKNOWN' : fail ? 'FAIL' : 'PASS';
    return `<div class="ins-stat-strip">
      <div class="ins-stat"><div class="ins-stat-label">Label</div>
        <div class="ins-stat-value">${label}</div></div>
      <div class="ins-stat"><div class="ins-stat-label">Benchmark</div>
        <div class="ins-stat-value">${escapeHtml(it.benchmark || '—')}</div></div>
    </div>
    ${it.source_data_url ? `<p>The full trajectory is available in the source run inspector.</p>
    <p class="ins-source-row"><a class="ins-source-link"
      href="${escapeHtml(it.source_data_url)}" target="_blank"
      rel="noopener noreferrer">Open original rollout ↗</a></p>`
      : `<p>Full trace is not included in this report. No verified EvalRDL link is available
        for this example. Open the local report to inspect its full trajectory.</p>`}`;
  }
  const fm = it.field_map || {};
  const kind = it.response_kind === 'agent_transcript'
    ? `agent transcript${it.n_agent_steps != null ? ` · ${it.n_agent_steps} assistant events` : ''}`
    : 'text';
  const sizes = [
    it.prompt_chars != null ? `prompt ${it.prompt_chars}` : null,
    it.reasoning_chars != null ? `cot ${it.reasoning_chars}` : null,
    it.response_chars != null ? `response ${it.response_chars}` : null,
  ].filter(Boolean).join(' · ');

  const respField = fm.response || 'response_text';
  const promptField = fm.prompt || 'prompt_text';
  const reasonField = fm.reasoning || 'reasoning_text';
  const isParsed = it.response_kind === 'agent_transcript' && fm.response_view === 'parsed_steps';

  const fieldNote = isParsed
    ? `<div class="ins-field-note">
        <strong>Detector view ≠ this pretty print.</strong>
        Functions receive the raw string <code>item["${escapeHtml(respField)}"]</code>
        (alias <code>generation</code>) — often a large agent-trace JSON (or plain text),
        depending on the dataset.
        The “agent actions” section below is a <em>dashboard rendering</em> of that same field
        for humans. Counts like <code>text.count('"command":')</code> run on the
        <strong>raw</strong> string, not on this structured layout.
      </div>`
    : `<div class="ins-field-note">
        Section chips show the <code>item[…]</code> keys your detector reads
        (e.g. <code>item.get("response_text") or item.get("generation")</code>).
      </div>`;

  const outTitle = isParsed
    ? 'Output · agent actions (parsed view)'
    : 'Output · response';
  const reasonTitle = fm.reasoning_from_steps
    ? 'Chain-of-thought (from steps[].reasoning_content)'
    : 'Chain-of-thought / reasoning';
  const sourceLink = it.source_data_url
    ? `<p class="ins-source-row"><a class="ins-source-link"
         href="${escapeHtml(it.source_data_url)}" target="_blank"
         rel="noopener noreferrer">View in Data API ↗</a></p>`
    : '';
  const combinedOpen = !(it.trace_events || []).length;
  const combinedStart = combinedOpen ? '' : '<details class="trace-combined">'
    + '<summary>Combined text fields · compatibility and search</summary>';

  return `
    <div class="ins-stat-strip">
      <div class="ins-stat"><div class="ins-stat-label">Label</div><div class="ins-stat-value ${fail ? 'bad' : 'green'}">${fail ? 'FAIL' : 'PASS'}</div></div>
      <div class="ins-stat"><div class="ins-stat-label">Benchmark</div><div class="ins-stat-value" style="font-size:12px;">${escapeHtml(it.benchmark || '—')}</div></div>
      <div class="ins-stat"><div class="ins-stat-label">Difficulty</div><div class="ins-stat-value" style="font-size:12px;">${escapeHtml(String(it.difficulty || '—'))}</div></div>
      <div class="ins-stat"><div class="ins-stat-label">Trace</div><div class="ins-stat-value" style="font-size:12px;">${escapeHtml(kind)}</div></div>
    </div>
    ${sourceLink}
    ${sizes ? `<p class="bucket-hint">Source sizes: ${escapeHtml(sizes)}. Click a section title (or the text box) to collapse it.</p>` : ''}
    ${insSection('Input · prompt / question', it.prompt_head || '(no prompt text)',
      { field: promptField, open: combinedOpen })}
    ${renderTrace(it)}
    ${combinedStart}
    ${fieldNote}
    ${insSection(reasonTitle, it.reasoning_head || '(no CoT / reasoning recorded)',
      { field: reasonField, open: combinedOpen })}
    ${insSection(outTitle, it.response_head || '(no response text)',
      { field: isParsed ? 'actions_text' : respField, open: combinedOpen })}
    ${isParsed && it.response_raw
      ? insSection(
          'Raw response string (what detectors see)',
          it.response_raw,
          { open: false, field: respField }
        )
      : ''}
    ${combinedOpen ? '' : '</details>'}
    ${it.judge_head ? insSection('Judge / grading', it.judge_head, { field: fm.judge || 'judge_text' }) : ''}
  `;
}

function bucketExampleIds(b) {
  if (!b) return { fails: [], passes: [] };
  return {
    fails: b.example_fail_ids || [],
    passes: b.example_pass_ids || [],
  };
}

async function renderInspector() {
  const drawer = document.getElementById('inspector');
  const sel = STATE.selection;
  if (!sel) {
    drawer.classList.remove('open');
    document.body.classList.remove('has-inspector');
    return;
  }
  drawer.classList.add('open');
  drawer.classList.toggle('single-item', sel.type === 'item');
  document.body.classList.add('has-inspector');

  const eyebrow = document.getElementById('ins-eyebrow');
  const title = document.getElementById('ins-title');
  const main = document.getElementById('ins-main');
  const side = document.getElementById('ins-side');
  const fn = findFn(sel.fn);

  if (sel.type === 'bucket') {
    const b = findBucket(fn, sel.bucket);
    const { fails, passes } = bucketExampleIds(b);
    const ids = fails.concat(passes);
    const itemId = sel.itemId || fails[0] || passes[0] || null;
    if (itemId && sel.itemId !== itemId) sel.itemId = itemId;
    eyebrow.textContent = `${sel.fn} · bucket ${sel.bucket}`;
    title.textContent = itemId || '(no examples)';
    main.innerHTML = itemId
      ? renderFailureLocalisation(fn, itemId)
        + `<div class="ins-empty">Loading full trajectory…</div>`
      : `<div class="ins-empty">No example ids stored for this bucket.</div>`;
    const nKnown = ids.filter(id => ITEMS[id] || ITEM_INDEX[id]).length;
    side.innerHTML = `
      <h4>Fails in bucket (${fails.length})</h4>
      <div class="taglist taglist-scroll">${renderItemTags(fails, itemId, 'bad') || '<em style="color:var(--text-4);">none</em>'}</div>
      <h4 style="margin-top:14px;">Passes in bucket (${passes.length})</h4>
      <div class="taglist taglist-scroll">${renderItemTags(passes, itemId, 'green') || '<em style="color:var(--text-4);">none</em>'}</div>
      <p class="bucket-hint" style="margin-top:10px;">${nKnown} of ${ids.length} trajectories available.
        Spot-check whether the trajectory matches the claimed mechanism.</p>
      ${b ? `<h4 style="margin-top:14px;">Bucket stats</h4>
        <div class="ins-stat-strip">
          <div class="ins-stat"><div class="ins-stat-label">n</div><div class="ins-stat-value">${b.n ?? '—'}</div></div>
          <div class="ins-stat"><div class="ins-stat-label">fail</div><div class="ins-stat-value">${b.n_fail ?? '—'}</div></div>
          <div class="ins-stat"><div class="ins-stat-label">pass</div><div class="ins-stat-value">${b.n_pass ?? '—'}</div></div>
          <div class="ins-stat"><div class="ins-stat-label">fires→%fail</div><div class="ins-stat-value">${pct(b.purity,0)}</div></div>
        </div>` : ''}
    `;
    if (itemId) {
      const it = await ensureItem(itemId);
      if (STATE.selection === sel && STATE.selection.itemId === itemId) {
        main.innerHTML = renderItemValidation(fn, itemId)
          + renderFailureLocalisation(fn, itemId) + renderItemMain(it);
      }
    }
  } else if (sel.type === 'item') {
    const itemId = sel.itemId;
    eyebrow.textContent = sel.fn ? `${sel.fn} · example` : 'Example';
    title.textContent = itemId || '';
    main.innerHTML = renderItemValidation(fn, itemId) + renderFailureLocalisation(fn, itemId)
      + `<div class="ins-empty">Loading full trajectory…</div>`;
    side.innerHTML = '';
    const it = await ensureItem(itemId);
    if (STATE.selection === sel && STATE.selection.itemId === itemId) {
      main.innerHTML = renderItemValidation(fn, itemId)
        + renderFailureLocalisation(fn, itemId) + renderItemMain(it);
    }
  }
}

function select(sel) {
  // Re-clicking the exact same selection closes the drawer.
  if (STATE.selection && sel
      && STATE.selection.type === sel.type
      && STATE.selection.fn === sel.fn
      && String(STATE.selection.bucket || '') === String(sel.bucket || '')
      && String(STATE.selection.itemId || '') === String(sel.itemId || '')) {
    STATE.selection = null;
  } else {
    STATE.selection = sel;
  }
  // Re-render open detector bodies so selected row highlight updates.
  document.querySelectorAll('.tax-detector.open').forEach(det => {
    const name = det.getAttribute('data-name');
    const fn = findFn(name);
    if (!fn) return;
    const body = det.querySelector('.tax-detector-body');
    if (body) body.innerHTML = renderDetectorBody(fn);
  });
  renderInspector();
}

function hasNoResolvedResults(fn) {
  const n = (fn.evaluation || {}).n_scored ?? fn.n_scored;
  return n != null && Number(n) === 0;
}

function unavailableEvaluationReason(fn) {
  const evaluation = fn.evaluation || {};
  if (evaluation.metrics_scope === 'unrepresented_strata') {
    return evaluation.sampling_limitation
      || 'Population rates unavailable: scoring did not represent every stratum.';
  }
  if (evaluation.evaluation_stale) return 'Saved detector evaluation is outdated';
  if (evaluation.evaluated === false) return 'Detector has no current evaluation';
  return hasNoResolvedResults(fn) ? 'No resolved detector results' : '';
}

function renderDetectorBody(fn) {
  const m = fn.metrics || {};
  const best = fn.best_bucket || {};
  const unavailableReason = unavailableEvaluationReason(fn);
  const associationUnavailable = Boolean(unavailableReason);
  const precision = associationUnavailable
    ? null : (best.purity != null ? best.purity : m.precision);
  const failed = associationUnavailable
    ? null : (best.n_fail != null ? best.n_fail : m.n_fail_caught);
  const flagged = associationUnavailable ? null : (best.n != null ? best.n : m.n_flagged);
  const counts = failed != null && flagged != null
    ? ` · ${num(failed, 0)} failed / ${num(flagged, 0)} flagged` : '';
  const noMatches = !associationUnavailable && m.n_flagged === 0 && !(fn.buckets || []).length;
  return `
    <p class="bucket-hint failure-rate-detail"><strong>Failure rate when flagged:</strong>
      ${pct(precision, 0)}${counts}
      ${fn.sampled && !associationUnavailable ? '(estimated)' : ''}
      · Overall ${pct(associationUnavailable ? null : m.base_fail_rate, 0)}.
      ${associationUnavailable
        ? `${escapeHtml(unavailableReason)}; association metrics are unavailable.`
        : (fn.evaluation || {}).metrics_scope === 'resolved_items'
          ? 'Raw precision on resolved rollouts only.' : 'Raw precision on this dataset.'}</p>
    <h4>Buckets · select a group to explore examples</h4>
    ${noMatches
      ? '<p class="bucket-hint">No matches among the resolved evaluated examples; '
        + 'there are no detected examples to inspect. Unresolved examples remain unknown.</p>'
      : renderBuckets(fn.buckets, fn.name, associationUnavailable, unavailableReason)}
    <h4>Source code</h4>
    <pre class="python-source">${highlightPython(fn.source || '(empty)')}</pre>
    ${fn.dependency_sources ? `<details class="tax-details"><summary>Pinned dependencies</summary>
      <pre class="python-source">${highlightPython(fn.dependency_sources)}</pre></details>` : ''}
  `;
}

function reviewValidationStatus(validation) {
  const v = validation || {};
  if (v.judge_contract_version === '{{JUDGE_CONTRACT_VERSION}}') {
    if (v.judge_behavior_strictness !== undefined
        && !['behavioral', 'contextual', 'causal'].includes(v.judge_behavior_strictness)) {
      return 'stale';
    }
    return v.status;
  }
  if (['unjudged', 'stale', 'skipped_pre_filter', 'no_fresh_evidence', 'error']
      .includes(v.status)) {
    return v.status;
  }
  return v.status || v.sampled || (v.verdicts || []).length ? 'legacy' : 'unjudged';
}

function savedReviewPolicy(validation) {
  const v = validation || {};
  const labels = {{JUDGE_POLICY_LABELS}};
  const policies = {
    behavioral: {
      label: labels.behavioral,
      requirement: 'the full stated description; causal evidence is required only when claimed',
      confirmed: 'matched the stated description',
      yes: 'The judge confirmed the described behavior. No additional causal requirement '
        + 'was imposed beyond the stated claim.',
      no: 'The judge rejected this example as evidence for the stated behavior.',
      uncertain: 'The judge could not establish the stated behavior.',
      requirementId: null,
    },
    contextual: {
      label: labels.contextual,
      requirement: 'the full description and supported failure relevance, without requiring '
        + 'exact defect attribution',
      confirmed: 'matched the description with supported failure relevance',
      yes: 'The judge confirmed the described behavior and supported failure relevance, '
        + 'without requiring exact defect attribution.',
      no: 'The judge rejected this example as evidence for the behavior or its failure relevance.',
      uncertain: 'The judge could not establish the behavior and its failure relevance.',
      requirementId: 'failure_relevance',
    },
    causal: {
      label: labels.causal,
      requirement: 'the full description and a material contribution to the task failure',
      confirmed: 'matched the description and contributed to the task failure',
      yes: 'The judge confirmed the described behavior and its contribution to the task failure.',
      no: 'The judge rejected this example as evidence for the claimed cause of task failure.',
      uncertain: 'The judge could not establish the claimed cause of task failure.',
      requirementId: 'failure_contribution',
    },
  };
  // Interpret each saved review, not the policy selected for another run or Continue.
  const name = v.judge_behavior_strictness === undefined ? 'causal' : v.judge_behavior_strictness;
  return (typeof name === 'string' && Object.prototype.hasOwnProperty.call(policies, name))
    ? policies[name] : {
    label: 'Unknown policy', requirement: 'its saved review requirements',
    confirmed: 'were confirmed under the saved review',
    yes: 'The saved review marked this example confirmed, but its policy is unavailable.',
    no: 'The saved review rejected this example; its policy is unavailable.',
    uncertain: 'The saved review was inconclusive; its policy is unavailable.',
    requirementId: null,
  };
}

function renderItemValidation(fn, itemId) {
  if (!fn || !itemId) return '';
  const v = fn.behavior_validation || {};
  const review = (v.verdicts || []).find(row => row.item_id === itemId);
  const policy = savedReviewPolicy(v);
  let summary;
  if (!review) {
    summary = 'Detector match only: this example was not reviewed by the judge. '
      + 'The sample score does not validate this individual example.';
  } else if (reviewValidationStatus(v) === 'historical') {
    const outcomes = {
      yes: policy.confirmed,
      no: 'was rejected',
      uncertain: 'was inconclusive',
      error: 'could not be reviewed',
    };
    summary = `In the saved review, this example ${outcomes[review.verdict] || 'has no verdict'}. `
      + `Judge policy: ${policy.label}. Current evidence checks have changed since that review.`;
    if (review.rationale) summary += ' ' + review.rationale;
  } else if (v.judge_contract_version !== '{{JUDGE_CONTRACT_VERSION}}'
      || ['legacy', 'stale', 'skipped_pre_filter'].includes(reviewValidationStatus(v))) {
    summary = 'This example has a historical review. Root-cause attribution '
      + 'requires a review under the current criteria.';
  } else {
    const labels = {
      yes: policy.yes,
      no: policy.no,
      uncertain: policy.uncertain,
      error: 'The judge could not complete the review of this example.',
    };
    summary = labels[review.verdict] || 'No completed review is available for this example.';
    const assessments = review.claim_assessments || [];
    const reason = (review.verdict === 'no'
      ? assessments.find(a => a.status === 'contradicted')
      : assessments.find(a => a.requirement_id === policy.requirementId));
    const explanation = reason ? reason.rationale : review.rationale;
    if (explanation) summary += ' ' + explanation;
    summary = `Judge policy: ${policy.label}. ` + summary;
  }
  return `<div class="item-validation"><h4>Example review</h4>`
    + `<p>${escapeHtml(summary)}</p></div>`;
}

function renderBehaviorSummary(validation, rationale, confirmationAvailability) {
  const v = validation || {};
  const status = reviewValidationStatus(v);
  const policy = savedReviewPolicy(v);
  const unavailable = {
    stale: 'The detector or review criteria have changed; a new AI review is needed.',
    legacy: 'This score comes from an older AI review. Root-cause attribution '
      + 'has not been validated under the current criteria.',
    unjudged: 'This detector has not been reviewed by AI yet.',
    no_fresh_evidence: 'No fresh validation examples are available in the scanned data.',
    skipped_pre_filter: 'AI review was skipped because the detector '
      + 'did not meet the initial score requirements.',
    error: 'The AI review could not be completed, so detector adherence is unknown.',
  };
  let summary = unavailable[status];
  if (v.review_schedule && v.review_schedule.status === 'deferred') {
    summary = 'Automatic adherence review deferred: ' + v.review_schedule.reason
      + ' This is a scheduling decision, not a behavioral verdict.';
  }
  const development = (v.sample || {}).phase === 'development';
  if (status === 'no_fresh_evidence' && development && !v.sampled
      && (v.sample || {}).target_n === 0) {
    summary = 'The detector produced no matches to review in its evaluated sample.';
  }
  if (!summary) {
    if (v.sampled) {
      const parts = [
        `${v.confirmed || 0} ${policy.confirmed}`];
      if (v.rejected) parts.push(`${v.rejected} ${v.rejected === 1 ? 'was' : 'were'} rejected`);
      if (v.uncertain) {
        parts.push(`${v.uncertain} ${v.uncertain === 1 ? 'was' : 'were'} inconclusive`);
      }
      if (v.processing_errors) parts.push(`${v.processing_errors} could not be reviewed`);
      summary = `Of ${v.sampled} examples checked by AI, ${parts.join(', ')}. `
        + 'Other detector matches have not been validated by this sample review.';
    } else {
      summary = 'There is not enough review evidence to confirm this detector yet.';
    }
  }
  if (status === 'historical') {
    summary = 'Historical review. ' + summary
      + ' This report preserves the scores from that review; using them for new discovery '
      + 'requires validation under the current evidence checks.';
  }
  if (status === 'limited_evidence' || (status === 'historical' && (v.sample || {}).limited)) {
    summary += ` Limited evidence: ${v.sampled} reviewed against a target of `
      + `${(v.sample || {}).target_n}. This review does not establish independent confirmation.`;
  }
  if (status === 'reused_evidence' || (status === 'historical' && (v.sample || {}).reused_n)) {
    summary += ` Reused evidence: ${(v.sample || {}).reused_n} previously reviewed examples `
      + 'were re-judged for this detector. These scores do not establish fresh validation.';
  }
  if (development && !['stale', 'legacy', 'skipped_pre_filter'].includes(status)) {
    summary += ' This is a development review; it does not establish independent confirmation.';
  }
  if (confirmationAvailability && confirmationAvailability.status === 'confirmation_unavailable'
      && !['stale', 'legacy', 'skipped_pre_filter'].includes(status)) {
    summary += ' Independent confirmation was attempted, but the scan found no tasks without '
      + 'prior discovery or review exposure. Earlier review results are retained.';
  }
  if (!unavailable[status] || status === 'legacy') {
    // Older libraries can retain useful judge prose without the newer review contract.
    const reviews = v.verdicts || [];
    const example = ['no', 'uncertain', 'yes'].map(verdict =>
      reviews.find(review => review.verdict === verdict && review.rationale)
    ).find(Boolean);
    const reviewText = example ? example.rationale : rationale;
    const first = String(reviewText || '').trim().split(/(?<=[.!?])\s+/)[0];
    if (first && first.length <= 300 && !first.endsWith('…') && !first.endsWith('...')) {
      if (!/^\d+\/\d+ (?:claims|labels) confirmed;/.test(first)) {
        const prefix = status === 'legacy' ? ' Earlier review: '
          : (example ? ' One example: ' : ' ');
        summary += prefix + first + (/[.!?]$/.test(first) ? '' : '.');
      }
    }
  }
  const policyNote = v.judge_contract_version === '{{JUDGE_CONTRACT_VERSION}}'
    ? `Judge policy: ${policy.label}. ` : '';
  return `<details class="tax-details review-details"><summary>AI review</summary>`
    + `<p class="review-summary">${escapeHtml(policyNote + summary)}</p></details>`;
}

function finalReviewDetail(fn) {
  const review = fn.final_review || {};
  if (!['unavailable', 'interrupted', 'running'].includes(review.status)) return '';
  const label = review.status === 'unavailable' ? 'Final review unavailable'
    : review.status === 'interrupted' ? 'Final review interrupted' : 'Final review pending';
  return label + (review.reason ? ': ' + String(review.reason) : '.');
}

function unavailableAdherenceReason(fn) {
  const skipped = (fn.metrics || {}).final_validation_skip_detail;
  if (skipped) {
    const adherence = (fn.metrics || {}).behavior_purity;
    return typeof adherence === 'number' && Number.isFinite(adherence)
      && adherence >= 0 && adherence <= 1 ? '' : skipped;
  }
  const finalDetail = finalReviewDetail(fn);
  const diagnosticDetail = (fn.metrics || {}).diagnostic_review_detail;
  if (finalDetail) return [finalDetail, diagnosticDetail].filter(Boolean).join(' ');
  const score = (fn.metrics || {}).behavior_purity;
  if (typeof score === 'number' && Number.isFinite(score) && score >= 0 && score <= 1) {
    return '';
  }
  const validation = fn.behavior_validation || {};
  if (validation.review_schedule && validation.review_schedule.status === 'deferred') {
    const deferred = 'Automatic review deferred: ' + validation.review_schedule.reason;
    return [deferred, diagnosticDetail].filter(Boolean).join(' ');
  }
  if (diagnosticDetail) return diagnosticDetail;
  const status = reviewValidationStatus(validation);
  // A missing review score does not invalidate completed detector predictions.
  // Explain the missing component beside its value, without changing the badge.
  if (status === 'stale') return 'Review unavailable: the saved review needs refresh.';
  if (status === 'skipped_pre_filter') {
    return 'Review skipped: ' + (validation.reason
      || 'the detector did not meet the score requirements.');
  }
  if ((fn.metrics || {}).badge_n_flagged === 0) return 'No matches to review.';
  const errors = Number(validation.processing_errors || 0);
  const uncertain = Number(validation.uncertain || 0);
  const reasons = [];
  if (errors) reasons.push(`${errors} processing ${errors === 1 ? 'error' : 'errors'}`);
  if (uncertain) {
    reasons.push(`${uncertain} inconclusive ${uncertain === 1 ? 'review' : 'reviews'}`);
  }
  if (reasons.length) {
    return 'Review unavailable: ' + reasons.join(', ') + '; no confirmed or rejected verdicts.';
  }
  if (status === 'no_fresh_evidence') return 'Review unavailable: no eligible evidence.';
  return 'Review pending: no usable judge score.';
}

function renderCard(fn, idx, total, connections = null) {
  // Always classify from metrics (same rules as autofa.metrics.verdicts) — do not
  // trust a frozen verdict blob from an older generator.
  const m = fn.metrics || {};
  const validation = fn.behavior_validation || {};
  const validationStatus = reviewValidationStatus(validation);
  const evaluation = fn.evaluation || {};
  const errors = Number(evaluation.errors || 0);
  const insufficient = Number(evaluation.n_insufficient || 0);
  const unresolved = Number(evaluation.n_unresolved ?? (errors + insufficient));
  const nScored = evaluation.n_scored == null ? null : Number(evaluation.n_scored);
  const parentErrors = Number(evaluation.parent_errors || 0);
  const resolvedOnly = evaluation.metrics_scope === 'resolved_items';
  const unavailableReason = unavailableEvaluationReason(fn);
  const associationUnavailable = Boolean(unavailableReason);
  const v = classifyFromMetrics(
    {...m, behavior_validation_status: validationStatus}, evaluation);
  // Quality uses the weakest metric tier; evaluation limitations remain visible
  // independently of the verdict and the available metric values.
  const incomplete = associationUnavailable || resolvedOnly || unresolved > 0 || parentErrors > 0
    || evaluation.evaluation_complete === false || evaluation.evaluation_stale
    || (evaluation.assessment_coverage != null && evaluation.assessment_coverage < 1);
  const vCls = verdictClass(v.code);
  const border = cardClass(v.code);

  const mech = fn.mechanism
    ? `<div class="tax-field"><div class="tax-field-label">Why this happens</div>
      <div class="tax-field-value">${escapeHtml(fn.mechanism)}</div></div>`
    : '';
  const fnMech = fn.function_mechanism
    ? `<div class="tax-field"><div class="tax-field-label">How the detector works</div>
      <div class="tax-field-value">${escapeHtml(fn.function_mechanism)}</div></div>`
    : '';
  const titleNorm = (fn.title || '').trim().replace(/\.+$/, '');
  const descNorm = (fn.description || '').trim().replace(/\.+$/, '');
  const practice = (fn.description
      && fn.description !== fn.mechanism
      && descNorm !== titleNorm)
    ? `<div class="tax-field"><div class="tax-field-label">What this means in practice</div><div class="tax-field-value">${escapeHtml(fn.description)}</div></div>`
    : '';

  const cardVerdict = sc('Verdict', `${v.symbol || ''} ${v.label}`.trim(),
    vCls,
    'Overall quality combines coverage, normalized precision, detector adherence, '
      + 'and support across distinct failed tasks. '
      + (v.reason || 'It has not been scored yet.'),
    v.code === 'UNSCORED' ? v.reason : '');
  const coverageTip = resolvedOnly
    ? 'Of failed rollouts with a resolved detector result, the percentage this detector catches. '
      + 'Repeated rollouts count separately. Unresolved rollouts are excluded; '
      + 'this is not full-dataset coverage.'
    : incomplete ? 'Evaluation is incomplete. Re-evaluate before interpreting coverage.'
      : 'Of all failed rollouts, the percentage this detector catches. '
        + 'Repeated rollouts count separately.';
  const scopeDetail = resolvedOnly ? 'Resolved rollouts only'
    : (incomplete ? 'Incomplete evaluation' : '');
  const supportCount = value => typeof value === 'number' && Number.isSafeInteger(value)
    && value >= 0 ? value : null;
  const failedAttempts = supportCount(m.n_matched_failed_attempts);
  const matchedTasks = supportCount(m.n_matched_failed_tasks);
  const confirmedTasks = supportCount(m.n_confirmed_failed_tasks);
  const estimatedTasks = matchedSupportEstimate(m);
  const coverageSupport = [
    failedAttempts === null ? '' : `${failedAttempts} failed rollouts matched`,
    matchedTasks === null ? '' : `${matchedTasks} distinct failed tasks matched`,
    confirmedTasks === null ? 'Task confirmations unavailable'
      : `${confirmedTasks} confirmed failed tasks`,
    estimatedTasks === null ? ''
      : `~${estimatedTasks.toFixed(1)} matched failed tasks estimated in full dataset`,
  ].filter(Boolean).join(' · ');
  const cardCov = sc('Coverage', coveragePercent(associationUnavailable ? null : m.recall),
    metricClass('recall', associationUnavailable ? null : m.recall),
    coverageTip + ' Observed task counts are distinct from full-dataset estimates. '
      + 'Estimated support can qualify for provisional Good during sampling; final '
      + 'qualification requires matches on at least five observed distinct failed tasks. '
      + 'The reviewed subset measures adherence separately; five reviewed confirmations are not '
      + 'required by the task floor. Confirmations are not necessarily independent validation.',
    associationUnavailable ? 'Unavailable'
      : [scopeDetail, coverageSupport].filter(Boolean).join(' · '));
  const cardPpv = sc('Normalized precision', num(associationUnavailable ? null : m.ppv_at_pi0, 2),
    metricClass('ppv_at_pi0', associationUnavailable ? null : m.ppv_at_pi0),
    'Precision adjusted to a dataset where 30% of examples fail, '
      + 'so detectors from different datasets can be compared. A score closer to 1 is better.',
    associationUnavailable ? 'Unavailable' : scopeDetail);
  const bp = m.behavior_purity;
  const bpVal = (bp != null && bp >= 0)
    ? num(bp, 2) + (validationStatus === 'historical' ? ' (saved)'
      : validationStatus === 'legacy' ? ' (legacy)' : '') : '—';
  const bpCls = metricClass('behavior_purity', bp);
  const reviewPolicy = savedReviewPolicy(validation);
  const reviewPolicyCurrent = validation.judge_contract_version === '{{JUDGE_CONTRACT_VERSION}}'
    && validationStatus !== 'historical';
  const adherenceTip = validationStatus === 'historical'
    ? `Saved adherence under judge policy: ${reviewPolicy.label}. The quality badge uses this `
      + 'recorded score. Current evidence checks have changed since the review. '
    : reviewPolicyCurrent
    ? `Judge policy: ${reviewPolicy.label}. Share of sampled matches where AI review confirms `
      + `${reviewPolicy.requirement}. Unsampled matches are not validated. `
    : ['legacy', 'stale'].includes(validationStatus)
      ? 'This score reflects the saved historical review criteria. It does not establish '
        + 'validation under the current criteria or validated root-cause attribution. '
      : 'No current review score is available. A detector match alone does not establish '
        + 'adherence to its stated claim. ';
  const reviewSample = validation.sample || {};
  const reviewDetail = [
    validationStatus === 'stale' ? 'Review needs refresh' : '',
    validation.status === 'limited_evidence'
      || (validationStatus === 'historical' && reviewSample.limited)
      ? `${validation.sampled}/${reviewSample.target_n} target reviews` : '',
    validation.status === 'reused_evidence'
      || (validationStatus === 'historical' && reviewSample.reused_n) ? 'Reused evidence' : '',
  ].filter(Boolean).join(' · ');
  const cardBeh = sc('Detector adherence', bpVal, bpCls, adherenceTip
      + 'Adherence describes reviewed matches; it is not a population-weighted estimate. '
      + 'A score closer to 1 is better; inconclusive reviews count in the total. '
      + 'Execution failures remain unfinished and do not count as rejections. '
      + 'Legacy means a review under older criteria.',
    [reviewPolicyCurrent ? reviewPolicy.label
      : ['legacy', 'historical'].includes(validationStatus) ? 'Historical review' : '',
      reviewDetail, unavailableAdherenceReason(fn)].filter(Boolean).join(' · '));

  const sampledNote = fn.sampled
    ? `<div class="tax-sampled">Coverage and precision are estimates from a stratified
        sample of
        ${fn.sample_n || '?'} of ${fn.sample_frame_n || '?'} rows (LLM detector).
        Ranges are 95% confidence intervals. They are not directly comparable to the
        exact metrics of regex detectors.</div>`
    : '';
  const evaluationNote = incomplete
    ? `<details class="tax-details review-details evaluation-summary">
      <summary>Evaluation incomplete</summary><p class="review-summary">
      ${associationUnavailable
        ? `${escapeHtml(unavailableReason)}; association metrics are unavailable.`
        : resolvedOnly ? 'Association metrics use resolved rollouts only.'
        : 'Re-evaluate before interpreting these association metrics.'}
      ${nScored != null && resolvedOnly
        ? `Resolved: ${nScored}/${nScored + unresolved} attempts.` : ''}
      ${unresolved} unresolved (${errors} execution errors; ${insufficient} insufficient evidence).
      ${parentErrors ? `${parentErrors} parent-detector errors.` : ''}</p></details>`
    : '';

  const cardId = fn.card_id || '';
  const failureHash = `#failure=${encodeURIComponent(fn.name)}`;
  return `<article
    id="${escapeHtml(cardId)}"
    class="tax-card ${border}"
    data-function-name="${escapeHtml(fn.name)}"
    tabindex="-1">
    <a class="tax-card-link" href="${escapeHtml(failureHash)}"
      data-failure-link="${escapeHtml(fn.name)}"
      aria-label="Link to failure mode ${escapeHtml(fn.name)}">Link</a>
    <details class="tax-card-content" open>
    <summary class="tax-card-head" title="Expand or collapse this failure card">
      <span class="card-chevron" aria-hidden="true">›</span>
      <div class="tax-card-heading">
        <div class="tax-eyebrow">Failure mode ${idx} of ${total}${connections?.labels.has(fn.name)
          ? ' · ' + escapeHtml(connections.labels.get(fn.name)) : ''}</div>
        <h2 class="tax-title">${escapeHtml(fn.title || fn.name)}</h2>
        <div class="tax-subtitle">${escapeHtml(fn.name)}</div>
      </div>
    </summary>
    <div class="tax-card-body">
    ${connections?.notes.get(fn.name) || ''}
    <div class="tax-statcards">${cardVerdict}${cardCov}${cardPpv}${cardBeh}</div>
    ${m.scoring_detail ? `<p class="bucket-hint scoring-summary">${escapeHtml(m.scoring_detail)}
      ${(associationUnavailable && evaluation.metrics_scope !== 'unrepresented_strata')
        || !m.flagged_detail ? ''
        : ` · Flagged: ${escapeHtml(m.flagged_detail)}`}</p>` : ''}
    ${evaluationNote}
    ${sampledNote}
    ${renderBehaviorSummary(fn.behavior_validation, fn.judge_rationale,
                           fn.confirmation_availability)}
    ${mech}
    ${fnMech}
    ${practice}
    <div class="tax-detectors-label">Detector</div>
    <div class="tax-detector" data-name="${escapeHtml(fn.name)}">
      <button type="button" class="tax-detector-head" aria-expanded="false"
        aria-controls="${escapeHtml(cardId)}-detector">
        <span class="tax-detector-signature">${highlightSig(fn.name, fn.source)}</span>
        <span class="detector-hint">Code and examples</span>
        <span class="chip">${escapeHtml(fn.kind || 'regex')}</span>
      </button>
      <div class="tax-detector-body" id="${escapeHtml(cardId)}-detector">
        ${renderDetectorBody(fn)}</div>
    </div>
    </div>
    </details>
  </article>`;
}

function reportConnections(functions, projection) {
  const byName = new Map(functions.map(fn => [fn.name, fn]));
  const labels = new Map(), notes = new Map();
  const link = name => `<a href="#failure=${encodeURIComponent(name)}"
    data-failure-jump="${escapeHtml(name)}">${escapeHtml(byName.get(name)?.title || name)}</a>`;
  const letter = index => {
    let value = '';
    while (index) {
      index -= 1;
      value = String.fromCharCode(65 + index % 26) + value;
      index = Math.floor(index / 26);
    }
    return value;
  };
  const sections = [];
  for (const group of projection?.groups || []) {
    const members = functions.map(fn => fn.name).filter(name => (group.names || []).includes(name));
    const edges = (group.relationships || []).filter(row => members.includes(row.source)
      && members.includes(row.target) && row.source !== row.target);
    if (members.length < 2 || !edges.length) continue;
    const groupLabel = letter(sections.length + 1);
    members.forEach((name, index) => labels.set(name, `${groupLabel}.${index + 1}`));
    const renderNode = name =>
      `<li><span class="related-label">${escapeHtml(labels.get(name))}</span>${link(name)}</li>`;
    const relationText = {
      synthesis: ['Synthesized from', 'Used to synthesize'],
      explains: ['Explains matches of', 'Matches explained by'],
    };
    for (const name of members) {
      const descriptions = edges.filter(row => row.source === name || row.target === name)
        .map(row => {
          const outgoing = row.source === name;
          const other = outgoing ? row.target : row.source;
          const text = relationText[row.relation] || ['Related to', 'Related to'];
          const action = text[outgoing ? 0 : 1];
          return `<li>${action} ${link(other)}${row.rationale
            ? ` — ${escapeHtml(row.rationale)}` : ''}</li>`;
        });
      notes.set(name, `<ul class="related-card-note">${descriptions.join('')}</ul>`);
    }
    sections.push(`<section class="related-group" id="${escapeHtml(group.id)}">
      <h3>Connected failures ${groupLabel}</h3>
      <ul>${members.map(renderNode).join('')}</ul></section>`);
  }
  return {labels, notes, html: sections.length
    ? '<h2 id="related-failures-title">Connected failures</h2>'
      + '<p>Recorded connections between findings. '
      + 'Failure cards below remain ordered by quality and coverage.</p>'
      + sections.join('') : ''};
}

function renderLibraryCoverage(summary) {
  const coverage = summary || {};
  const available = coverage.available === true;
  const exact = coverage.status === 'exact';
  const rate = available ? coverageRate(coverage.coverage) : null;
  const taskRate = available ? coverageRate(coverage.task_coverage) : null;
  const value = v => {
    if (v == null) return '—';
    // Aggregate coverage has no detector-grade boundaries. Preserve tiny
    // positive values without composing a lower bound with a "<" or ">" label.
    const percent = v > 0 && v < .0005
      ? String(Number((100 * v).toPrecision(3))) : (100 * v).toFixed(1);
    return `${exact ? '' : '≥'}${percent}%`;
  };
  const count = n => Number.isInteger(n) && n >= 0 ? n.toLocaleString() : '—';
  const gradeLabel = coverage.minimum_grade === 'OK' ? 'OK-or-better' : 'Good-or-better';
  const nQualifying = coverage.n_qualifying_detectors;
  const nEligible = coverage.n_eligible_detectors;
  const scope = !available ? 'Unavailable'
    : nQualifying === 0 ? `No ${gradeLabel} detectors yet`
    : exact ? 'Full-dataset coverage' : 'Observed lower bound';
  const eligible = available && Number.isInteger(nEligible)
    ? `Using ${count(nEligible)} ${gradeLabel} detector${nEligible === 1 ? '' : 's'}. `
    : '';
  const attemptDetail = rate == null
    ? (coverage.reason || 'The full failure inventory and current match records are needed.')
    : `${count(coverage.n_matched_failed_attempts)} / `
      + `${count(coverage.n_failed_attempts)} failed rollouts matched.`;
  const taskDetail = taskRate == null
    ? (coverage.task_reason || (!available && coverage.reason)
      || 'Distinct-task coverage needs verified task identities for every failed rollout.')
    : `${count(coverage.n_matched_failed_tasks)} / `
      + `${count(coverage.n_failed_tasks)} distinct failed tasks matched.`;
  const limitation = available && nQualifying === 0
    ? `No detectors currently meet ${gradeLabel} requirements, so qualifying coverage is zero.`
    : available && !exact
    ? 'Sampled or incomplete scans can leave other matches unseen. '
      + 'This is an observed minimum, not an extrapolated estimate.'
    : available ? 'Every eligible detector has a complete scan of this corpus.'
    : 'Saved detector percentages and displayed examples cannot establish combined coverage.';
  const nExcluded = coverage.n_excluded_detectors;
  const exclusions = nExcluded > 0
    ? ` ${count(nExcluded)} ${gradeLabel} detector(s) `
      + 'lack usable current match evidence.'
    : '';
  return `<h2 id="library-coverage-title">Library coverage</h2>
    <p class="library-coverage-intro">${escapeHtml(eligible)}
      Failures matched by at least one ${gradeLabel} detector.
      Overlapping matches count once.</p>
    <div class="library-coverage-grid">
      <div class="library-coverage-card primary">
        <h3>Combined Coverage</h3>
        <div class="library-coverage-value${rate != null && rate > 0 ? ' measured' : ''}"
          >${value(rate)}</div>
        <p class="library-coverage-detail">${escapeHtml(attemptDetail)}
          <span class="library-coverage-scope">${escapeHtml(scope)}</span></p>
      </div>
      <div class="library-coverage-card">
        <h3>Distinct-task coverage</h3>
        <div class="library-coverage-value${taskRate != null && taskRate > 0 ? ' measured' : ''}"
          >${value(taskRate)}</div>
        <p class="library-coverage-detail">${escapeHtml(taskDetail)}
          <span class="library-coverage-scope">
            Each task counts once across repeated attempts.</span></p>
      </div>
    </div>
    <p class="library-coverage-note">${escapeHtml(limitation + exclusions)}
      Coverage uses all saved matches from qualifying detectors,
      not only judge-reviewed examples.</p>`;
}

function renderExecutionStatus(execution) {
  const status = execution?.status;
  if (!['interrupted', 'failed', 'incomplete', 'started'].includes(status)) return '';
  const resumable = execution.resumable === true;
  const title = resumable
    ? 'Discovery interrupted · resumable' : 'Discovery completion not recorded';
  const detail = resumable
    ? 'These are partial results. Resume the saved run to continue from its checkpoints.'
    : 'These are saved results from an unfinished agent turn; '
      + 'they do not establish completed discovery.';
  return `<strong>${title}</strong><p>${detail}</p>`;
}

function boot() {
  const meta = RAW.meta || {};
  const execution = document.getElementById('execution-status');
  execution.innerHTML = renderExecutionStatus(meta.agent_execution);
  execution.hidden = !execution.innerHTML;
  const fns = visibleReportFunctions(RAW.functions || []);
  const root = document.getElementById('cards');
  const connections = reportConnections(fns, meta.related_failures);
  const related = document.getElementById('related-failures');
  related.innerHTML = connections.html;
  related.hidden = !connections.html;
  document.getElementById('doc-h1').textContent =
    `${meta.model_id || '?'} / ${meta.benchmark || 'all'}`;
  document.getElementById('library-coverage').innerHTML =
    renderLibraryCoverage(meta.library_quality_coverage);
  document.getElementById('kpi-band').innerHTML = `
    <div class="kpi-card"><div class="kpi-eyebrow">Failure modes</div>
      <div class="kpi-value green">${fns.length}</div></div>
    <div class="kpi-card"><div class="kpi-eyebrow">Items</div><div class="kpi-value">${meta.n_items ?? '—'}</div></div>
    <div class="kpi-card"><div class="kpi-eyebrow">Failures</div><div class="kpi-value">${meta.n_failures ?? '—'}</div></div>
    <div class="kpi-card"><div class="kpi-eyebrow">Overall failure rate</div>
      <div class="kpi-value">${pct(meta.base_fail_rate,0)}</div></div>
  `;

  const index = document.getElementById('failure-index');
  index.hidden = !fns.length;
  const moreFailuresLabel = `Show ${fns.length - 3} more failure${fns.length === 4 ? '' : 's'}`;
  index.innerHTML = `<h2>Failure modes at a glance</h2><ol id="failure-index-list">${fns.map((fn, i) =>
    `<li${i >= 3 ? ' hidden data-extra-failure' : ''}><a href="#failure=${encodeURIComponent(fn.name)}"
      data-failure-jump="${escapeHtml(fn.name)}">${escapeHtml(fn.title || fn.name)}</a></li>`
  ).join('')}</ol>${fns.length > 3 ? `<button type="button" class="failure-index-toggle"
    aria-expanded="false" aria-controls="failure-index-list">${moreFailuresLabel}</button>` : ''}`;
  const indexToggle = index.querySelector('.failure-index-toggle');
  if (indexToggle) {
    indexToggle.addEventListener('click', () => {
      const expanded = indexToggle.getAttribute('aria-expanded') !== 'true';
      index.querySelectorAll('[data-extra-failure]').forEach(item => { item.hidden = !expanded; });
      indexToggle.setAttribute('aria-expanded', String(expanded));
      indexToggle.textContent = expanded ? 'Show fewer failures' : moreFailuresLabel;
    });
  }
  document.querySelectorAll('[data-failure-jump]').forEach(link => {
    link.addEventListener('click', () => focusFailureMode(link.dataset.failureJump));
  });
  root.innerHTML = fns.length
    ? `<div class="tax-grid">${fns.map((fn,i) => renderCard(fn, i+1, fns.length,
      connections)).join('')}</div>`
    : '<div class="empty">No OK-or-better detectors yet. Candidates remain in the saved library.'
      + '</div>';
  root.querySelectorAll('[data-failure-jump]').forEach(link => {
    link.addEventListener('click', () => focusFailureMode(link.dataset.failureJump));
  });

  root.querySelectorAll('.tax-detector-head').forEach(head => {
    head.addEventListener('click', () => {
      setDetectorOpen(head.parentElement, !head.parentElement.classList.contains('open'));
    });
  });
  root.querySelectorAll('[data-failure-link]').forEach(link => {
    link.addEventListener('click', () => {
      focusFailureMode(link.getAttribute('data-failure-link'));
      postFailureLink(link.getAttribute('data-failure-link'));
    });
  });
  // Global delegate: bucket rows + example tags + section collapse.
  document.addEventListener('click', (ev) => {
    const passageJump = ev.target.closest('[data-passage-index]');
    if (passageJump) {
      const selection = window.getSelection && window.getSelection();
      if (!(selection && String(selection).length)) {
        ev.preventDefault();
        jumpPassage(Number(passageJump.dataset.passageIndex));
      }
      return;
    }
    const traceJump = ev.target.closest('[data-trace-step]');
    if (traceJump) {
      ev.preventDefault();
      showTraceLocation({step_index: Number(traceJump.dataset.traceStep)});
      return;
    }
    const sourceLink = ev.target.closest('.ins-source-link');
    if (sourceLink && window.parent !== window) {
      ev.preventDefault();
      window.parent.postMessage({
        namespace: 'autofa.source-data',
        type: 'open',
        // Keep local links relative: the parent owns the current Library origin,
        // including when this report is a sandboxed or spilled asset.
        url: sourceLink.getAttribute('href'),
      }, '*');
      return;
    }
    // Click body of an open inspector section to collapse (unless selecting text).
    const secPre = ev.target.closest('.ins-section > pre');
    if (secPre) {
      const sel = window.getSelection && window.getSelection();
      if (!(sel && String(sel).length > 0)) {
        const details = secPre.parentElement;
        if (details && details.tagName === 'DETAILS') details.open = false;
      }
      return;
    }
    const tag = ev.target.closest('[data-jump-item]');
    if (tag) {
      ev.preventDefault();
      ev.stopPropagation();
      const itemId = tag.getAttribute('data-jump-item');
      const row = tag.closest('tr[data-bucket]');
      const det = tag.closest('.tax-detector');
      const fnName = (row && row.getAttribute('data-fn'))
        || (det && det.getAttribute('data-name'))
        || (STATE.selection && STATE.selection.fn)
        || '';
      select({ type: 'item', fn: fnName, itemId });
      return;
    }
    const row = ev.target.closest('tr.clickable[data-bucket]');
    if (row) {
      ev.preventDefault();
      const fnName = row.getAttribute('data-fn');
      const bucket = row.getAttribute('data-bucket');
      const fn = findFn(fnName);
      const b = findBucket(fn, bucket);
      const fails = (b && b.example_fail_ids) || [];
      const passes = (b && b.example_pass_ids) || [];
      const first = fails[0] || passes[0] || null;
      // Ensure detector stays open.
      const det = row.closest('.tax-detector');
      if (det) setDetectorOpen(det, true);
      select({ type: 'bucket', fn: fnName, bucket, itemId: first });
    }
  });

  document.getElementById('ins-close').addEventListener('click', () => {
    STATE.selection = null;
    document.querySelectorAll('.tax-detector.open').forEach(det => {
      const name = det.getAttribute('data-name');
      const fn = findFn(name);
      const body = det.querySelector('.tax-detector-body');
      if (fn && body) body.innerHTML = renderDetectorBody(fn);
    });
    renderInspector();
  });
  document.addEventListener('keydown', (ev) => {
    if ((ev.key === 'Enter' || ev.key === ' ') && ev.target.matches('[data-passage-index]')) {
      ev.preventDefault();
      jumpPassage(Number(ev.target.dataset.passageIndex));
    }
    if (ev.key === 'Escape' && STATE.selection) {
      document.getElementById('ins-close').click();
    }
  });
  window.addEventListener('hashchange', () => {
    focusFailureMode(requestedFailureName());
  });

  const themeBtn = document.getElementById('theme-toggle');
  // Older reports saved their dark default as if the user had chosen it.
  // Start light once, then remember only an explicit choice in this UI.
  const themePreferenceKey = 'autofa-fc-theme-preference-v2';
  const applyTheme = (light) => {
    document.body.classList.toggle('theme-light', light);
    themeBtn.textContent = light ? 'Dark' : 'Light';
  };
  let light = true;
  try { light = localStorage.getItem(themePreferenceKey) !== 'dark'; } catch (e) {}
  // A page embedding the report in a sandbox, where storage is unavailable, passes its scheme.
  const askedTheme = new URLSearchParams(location.search).get('theme');
  if (askedTheme === 'dark' || askedTheme === 'light') light = askedTheme === 'light';
  applyTheme(light);
  themeBtn.addEventListener('click', () => {
    const light = !document.body.classList.contains('theme-light');
    applyTheme(light);
    try { localStorage.setItem(themePreferenceKey, light ? 'light' : 'dark'); } catch (e) {}
  });

  const dlBtn = document.getElementById('download-html');
  if (dlBtn) {
    dlBtn.addEventListener('click', async () => {
      const fallbackName = 'failure_cards.html';
      const name = (location.pathname.split('/').filter(Boolean).pop()) || fallbackName;
      const prev = dlBtn.textContent;
      dlBtn.disabled = true;
      dlBtn.textContent = 'Downloading…';
      try {
        let blob;
        try {
          const resp = await fetch(location.href, { cache: 'no-store' });
          if (!resp.ok) throw new Error(String(resp.status));
          blob = await resp.blob();
        } catch (e) {
          // file:// or fetch blocked — serialize the live document (includes embedded data).
          const html = '<!DOCTYPE html>\n' + document.documentElement.outerHTML;
          blob = new Blob([html], { type: 'text/html;charset=utf-8' });
        }
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = name.endsWith('.html') ? name : fallbackName;
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
      } finally {
        dlBtn.disabled = false;
        dlBtn.textContent = prev;
      }
    });
  }

  const dlZipBtn = document.getElementById('download-zip');
  if (dlZipBtn) {
    dlZipBtn.addEventListener('click', async () => {
      if (location.protocol === 'file:') {
        window.alert(
          'Download ZIP only works when the report is opened via AutoFA (http://127.0.0.1/…). The ZIP itself is self-contained — after download you can open the HTML offline.'
        );
        return;
      }
      const prev = dlZipBtn.textContent;
      dlZipBtn.disabled = true;
      dlZipBtn.textContent = 'Zipping…';
      try {
        const resp = await fetch('/autofa-bundle.zip', { cache: 'no-store' });
        if (!resp.ok) {
          const detail = await resp.text().catch(() => '');
          throw new Error(detail || ('HTTP ' + resp.status));
        }
        const blob = await resp.blob();
        let name = 'failure_cards_bundle.zip';
        const cd = resp.headers.get('Content-Disposition') || '';
        const m = /filename=\"([^\"]+)\"/i.exec(cd);
        if (m && m[1]) name = m[1];
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = name;
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
      } catch (e) {
        window.alert('ZIP download failed: ' + (e && e.message ? e.message : e));
      } finally {
        dlZipBtn.disabled = false;
        dlZipBtn.textContent = prev;
      }
    });
  }

  // Defer until the freshly rendered cards have layout, including inside an iframe.
  window.requestAnimationFrame(() => {
    focusFailureMode(requestedFailureName());
  });
}

boot();
</script>
</body>
</html>
"""
