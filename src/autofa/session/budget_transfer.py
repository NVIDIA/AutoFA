# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One-use transfer of a known local campaign's remaining allowance on Continue."""

from __future__ import annotations

import copy
import logging
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from autofa.session.recovery import read_json, run_lock
from autofa.util.atomic import atomic_write_json


def _source(seed_path: str | None, source_run_id: str | None) -> Path | None:
    if not seed_path:
        return None
    from autofa.config import runs_root
    from autofa.data.library_provenance import read_catalog_provenance

    path = Path(seed_path).expanduser()
    provenance = read_catalog_provenance(path)
    source_run_id = source_run_id or (provenance.source_run_id if provenance else None)
    candidates = [path.parent]
    if source_run_id and Path(source_run_id).name == source_run_id:
        candidates.append(runs_root() / source_run_id)
    for candidate in candidates:
        if read_json(candidate / "config.json").get("run_id"):
            return candidate.resolve()
    return None


@contextmanager
def inherit_continuation_budget(run, *, seed_path: str | None, source_run_id: str | None):
    """Handoff a local source's remaining cap without changing recorded source spend.

    Continue is a new execution, not a new resource grant. The inactive source
    gives up further paid work only after the child has initialized successfully.
    Unknown/remote origins require the caller's explicit campaign allocation;
    their unseen historical costs cannot be reconstructed from detector scores.
    """
    from autofa.session.usage import admission_usage

    directory = _source(seed_path, source_run_id)
    source = read_json(directory / "config.json") if directory else {}
    policy = source.get("campaign_policy") or {}
    if directory is None or policy.get("spend_limit_usd") is None:
        yield
        return
    with run_lock(directory):
        source = read_json(directory / "config.json")
        policy = source.get("campaign_policy") or {}
        state = source.get("campaign_state") or {}
        previous = state.get("budget_transferred")
        if previous:
            raise ValueError("This campaign allowance was already transferred to run "
                             + str(previous.get("run_id")) + "; continue from that run instead")
        money = admission_usage(directory)["monetary_view"]
        allowed = (money.get("cost_complete")
                   or policy.get("unknown_cost_policy") == "allow_estimate"
                   and money.get("estimated_cost_complete"))
        spent = money.get("admission_accounted_usd")
        if not allowed or spent is None:
            raise ValueError("Cannot Continue a capped campaign with unresolved spending; "
                             "reconcile its usage first")
        # Continue inherits the original finite grant, not changed global defaults.
        # A new grant requires an explicit source policy update before this handoff.
        run.campaign_policy = copy.deepcopy(policy)
        run.campaign_state = {
            "prior_spend_usd": spent,
            "budget_origin_run_id": state.get("budget_origin_run_id", source["run_id"]),
            "overtime": copy.deepcopy(state.get("overtime") or {}),
        }
        transfer_id = uuid4().hex
        transfer = {"id": transfer_id, "run_id": run.run_id,
                    "source_run_id": source["run_id"], "source_run_dir": str(directory),
                    "prior_spend_usd": spent}
        run.campaign_state["budget_received"] = transfer
        yield
        # Both directories now exist, but the child cannot dispatch until create
        # returns. This journal makes a crash between the two saves explicit.
        journal = directory / "supervision/budget-transfers" / (transfer_id + ".json")
        atomic_write_json(journal, {**transfer, "state": "prepared"})
        source["campaign_state"] = {**state, "budget_transferred": transfer}
        try:
            atomic_write_json(directory / "config.json", source)
        finally:
            saved = read_json(directory / "config.json").get("campaign_state") or {}
            if (saved.get("budget_transferred") or {}).get("id") == transfer_id:
                run._budget_handoff_committed = True
        try:
            atomic_write_json(journal, {**transfer, "state": "committed"})
        except OSError:
            # Source and child configuration are the authoritative commit. A
            # missing final audit update must not discard the initialized child.
            logging.getLogger(__name__).warning("Budget handoff committed; audit refresh pending")
