# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from .data_cards import move_branch, resolve_library_alias
from .data_identity import data_identity_key, normalize_data_identity
from .db import Database
from .schemas import Identity, LibraryRecord

_UNRESTRICTED_LISTING = object()


class SelectorNotFoundError(LookupError):
    pass


class BranchIdentityConflictError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class PublicationPlan:
    library_id: str
    branch_id: str
    library_exists: bool
    branch_exists: bool
    upgrade_data_identity: bool = False
    previous_library_id: str | None = None


def _row_to_dict(row: Any) -> dict[str, Any]:
    item = dict(row)
    if "benchmarks_json" in item:
        item["benchmarks"] = json.loads(item.pop("benchmarks_json"))
    if "reference_data_json" in item:
        item["reference_data"] = json.loads(item.pop("reference_data_json"))
    if "data_identity_json" in item:
        item["data_identity"] = json.loads(item.pop("data_identity_json") or "null")
    if "data_identities_json" in item:
        identities = [json.loads(value) for value in json.loads(item.pop("data_identities_json"))]
        item["data_identities"] = [identity for identity in identities if identity is not None]
        item["data_group_count"] = len(identities)
        item["is_merged"] = len(identities) > 1
        item["data_identity"] = identities[0] if len(identities) == 1 else None
        statuses = json.loads(item.pop("data_statuses_json"))
        item["data_identity_status"] = (
            "legacy_mixed" if "legacy_mixed" in statuses else
            "unknown" if "unknown" in statuses or not statuses else "known"
        )
    return item


def to_record(item: dict[str, Any]) -> LibraryRecord:
    return LibraryRecord.model_validate(
        {
            key: value
            for key, value in item.items()
            if key not in {"library_key", "report_key", "sequence"}
        }
    )


def _activity_time(alias: str) -> str:
    # Imported snapshots retain their immutable import timestamps. Catalog order
    # uses the source's publication/save time, including older databases on refresh.
    return f"""CASE WHEN {alias}.publish_trigger = 'local_catalog' THEN COALESCE(
        (SELECT COALESCE(activity.published_at, activity.saved_at)
         FROM local_source_activity activity
         WHERE activity.source_run_id = {alias}.source_run_id),
        {alias}.created_at) ELSE {alias}.created_at END"""


LATEST_LIBRARY_SELECT = f"""
    SELECT
        logical.id,
        (SELECT json_group_array(data_identity_json) FROM library_data_identities
            WHERE library_id=logical.id) AS data_identities_json,
        (SELECT json_group_array(status) FROM library_data_identities
            WHERE library_id=logical.id) AS data_statuses_json,
        logical.owner_id,
        owner.username AS owner_username,
        owner.name AS owner_name,
        owner.avatar_url AS owner_avatar,
        owner.is_public AS owner_is_public,
        version.source_run_id,
        version.model_id,
        version.benchmarks_json,
        version.function_count,
        logical.created_at,
        {_activity_time('version')} AS updated_at,
        version.version_number AS revision,
        version.library_sha256,
        version.report_sha256,
        version.library_key,
        version.report_key,
        version.harness,
        version.source_harness,
        version.reference_data_json,
        version.data_identity_json,
        version.publish_trigger,
        logical.is_public,
        branch.id AS latest_branch_id,
        version.id AS latest_version_id,
        (
            SELECT COUNT(*)
            FROM library_branches counted_branch
            WHERE counted_branch.library_id = logical.id
        ) AS branch_count,
        (
            SELECT COUNT(*)
            FROM library_versions counted_version
            WHERE counted_version.library_id = logical.id
        ) AS version_count,
        version.sequence
    FROM logical_libraries logical
    JOIN owners owner ON owner.id = logical.owner_id
    JOIN library_versions version ON version.sequence = (
        SELECT latest.sequence
        FROM library_versions latest
        WHERE latest.library_id = logical.id
        ORDER BY julianday({_activity_time('latest')}) DESC, latest.sequence DESC
        LIMIT 1
    )
    JOIN library_branches branch ON branch.id = version.branch_id
"""

LATEST_LIBRARY_ORDER = (
    f" ORDER BY julianday({_activity_time('version')}) DESC, version.sequence DESC, logical.id ASC"
)

VERSION_SELECT = """
    SELECT
        version.id,
        version.library_id,
        version.branch_id,
        branch.branch_key,
        version.version_number,
        version.source_run_id,
        logical.owner_id,
        owner.username AS owner_username,
        owner.name AS owner_name,
        owner.avatar_url AS owner_avatar,
        owner.is_public AS owner_is_public,
        version.model_id,
        version.benchmarks_json,
        version.function_count,
        version.created_at,
        version.library_sha256,
        version.report_sha256,
        version.library_key,
        version.report_key,
        version.harness,
        version.source_harness,
        version.reference_data_json,
        version.data_identity_json,
        version.publish_trigger,
        logical.is_public,
        version.sequence
    FROM library_versions version
    JOIN library_branches branch ON branch.id = version.branch_id
    JOIN logical_libraries logical ON logical.id = version.library_id
    JOIN owners owner ON owner.id = logical.owner_id
"""


class LibraryRepository:
    def __init__(self, database: Database):
        self.database = database

    def resolve_library_id(self, library_id: str, branch_id: str | None = None) -> str:
        """Resolve historical card/branch links; callers must authorize the resulting card."""
        with self.database.read() as connection:
            return resolve_library_alias(connection, library_id, branch_id)

    def merge_libraries(
        self, *, owner_id: int, source_library_id: str, target_library_id: str,
    ) -> dict[str, Any]:
        """Join owned data cards without changing immutable reports or their IDs."""
        with self.database.transaction() as connection:
            source_id = resolve_library_alias(connection, source_library_id)
            target_id = resolve_library_alias(connection, target_library_id)
            self._merge_card_membership(
                connection, owner_id=owner_id, source_id=source_id, target_id=target_id,
            )
            row = connection.execute(
                LATEST_LIBRARY_SELECT + " WHERE logical.id=?", (target_id,),
            ).fetchone()
        if row is None:
            raise RuntimeError("merged data card has no saved version")
        return _row_to_dict(row)

    def _merge_card_membership(
        self, connection: Any, *, owner_id: int, source_id: str, target_id: str,
    ) -> None:
        """Move an entire owned card within the caller's transaction.

        Source discovery can identify an unknown branch already explicitly merged
        with other datasets. Preserve that owner's grouping when joining the known
        source card, instead of detaching just the newly identified branch.
        """
        rows = {row["id"]: row for row in connection.execute(
            "SELECT * FROM logical_libraries WHERE owner_id=? AND id IN (?,?)",
            (owner_id, source_id, target_id),
        )}
        if source_id not in rows or target_id not in rows:
            raise SelectorNotFoundError("library selector not found")
        if source_id == target_id:
            raise ValueError("select two different data cards")
        connection.execute(
            "UPDATE logical_libraries SET is_public=? WHERE id=?",
            (min(rows[source_id]["is_public"], rows[target_id]["is_public"]), target_id),
        )
        branches = connection.execute(
            "SELECT id FROM library_branches WHERE library_id=?", (source_id,),
        ).fetchall()
        for branch in branches:
            move_branch(connection, branch["id"], source_id, target_id)
        connection.execute(
            "UPDATE library_data_identities SET library_id=? WHERE library_id=?",
            (target_id, source_id),
        )
        connection.execute(
            "UPDATE library_aliases SET library_id=? WHERE library_id=?",
            (target_id, source_id),
        )
        connection.execute(
            "UPDATE library_branch_aliases SET library_id=? WHERE library_id=?",
            (target_id, source_id),
        )
        connection.execute(
            "INSERT INTO library_aliases(alias_id,library_id) VALUES(?,?)",
            (source_id, target_id),
        )
        connection.execute("DELETE FROM logical_libraries WHERE id=?", (source_id,))

    def upsert_owner(self, identity: Identity, *, updated_at: str) -> None:
        with self.database.transaction() as connection:
            connection.execute(
                """
                INSERT INTO owners (id, username, name, avatar_url, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    username = excluded.username,
                    name = excluded.name,
                    avatar_url = excluded.avatar_url,
                    updated_at = excluded.updated_at
                """,
                (
                    identity.id,
                    identity.username,
                    identity.name,
                    identity.avatar_url,
                    updated_at,
                ),
            )

    def get_owned(self, owner_id: int, library_id: str) -> dict[str, Any] | None:
        library_id = self.resolve_library_id(library_id)
        with self.database.read() as connection:
            row = connection.execute(
                LATEST_LIBRARY_SELECT + " WHERE logical.id = ? AND logical.owner_id = ?",
                (library_id, owner_id),
            ).fetchone()
        return _row_to_dict(row) if row else None

    def list_owned(
        self,
        owner_id: int,
        *,
        limit: int,
        offset: int,
        model_id: str | None = None,
        benchmark: str | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        sql = LATEST_LIBRARY_SELECT + " WHERE logical.owner_id = ?"
        params: list[Any] = [owner_id]
        if model_id:
            sql += " AND version.model_id = ?"
            params.append(model_id)
        sql += LATEST_LIBRARY_ORDER
        with self.database.read() as connection:
            rows = connection.execute(sql, params).fetchall()
        items = [_row_to_dict(row) for row in rows]
        if benchmark:
            items = [item for item in items if benchmark in item["benchmarks"]]
        total = len(items)
        return items[offset : offset + limit], total

    def get_visible(
        self,
        viewer_id: int | None,
        library_id: str,
    ) -> dict[str, Any] | None:
        library_id = self.resolve_library_id(library_id)
        sql = (
            LATEST_LIBRARY_SELECT
            + """
            WHERE logical.id = ?
                AND owner.is_public = 1
                AND logical.is_public = 1
            """
        )
        params: list[Any] = [library_id]
        if viewer_id is not None:
            sql = (
                LATEST_LIBRARY_SELECT
                + """
                WHERE logical.id = ?
                    AND (
                        (owner.is_public = 1 AND logical.is_public = 1)
                        OR logical.owner_id = ?
                    )
                """
            )
            params.append(viewer_id)
        with self.database.read() as connection:
            row = connection.execute(sql, params).fetchone()
        return _row_to_dict(row) if row else None

    def list_visible(
        self,
        viewer_id: int,
        *,
        limit: int,
        offset: int,
    ) -> tuple[list[dict[str, Any]], int]:
        with self.database.read() as connection:
            rows = connection.execute(
                LATEST_LIBRARY_SELECT
                + """
                WHERE (
                    owner.is_public = 1 AND logical.is_public = 1
                ) OR logical.owner_id = ?
                """
                + LATEST_LIBRARY_ORDER,
                (viewer_id,),
            ).fetchall()
        items = [_row_to_dict(row) for row in rows]
        return items[offset : offset + limit], len(items)

    @staticmethod
    def _benchmarks_json(benchmarks: list[str]) -> str:
        return json.dumps(benchmarks, ensure_ascii=False, separators=(",", ":"))

    def _resolve_plan(
        self, connection: Any, *, owner_id: int, model_id: str, benchmarks_json: str,
        branch_key: str, known_library_id: str | None, known_branch_id: str | None,
        new_library_id: str, new_branch_id: str, data_identity: dict | None = None,
    ) -> PublicationPlan:
        identity = normalize_data_identity(data_identity)
        key = data_identity_key(identity, branch_key)
        logical = connection.execute(
            "SELECT library_id AS id FROM library_data_identities "
            "WHERE owner_id = ? AND data_key = ?",
            (owner_id, key),
        ).fetchone()
        branch = connection.execute(
            "SELECT id, library_id FROM library_branches WHERE owner_id = ? AND branch_key = ?",
            (owner_id, branch_key),
        ).fetchone()
        selected_id = None
        if known_library_id is not None:
            selected_id = resolve_library_alias(
                connection, known_library_id,
                known_branch_id or (branch["id"] if branch else None),
            )
            selected = connection.execute(
                "SELECT id FROM logical_libraries WHERE id = ? AND owner_id = ?",
                (selected_id, owner_id),
            ).fetchone()
            if selected is None:
                raise SelectorNotFoundError("library selector not found")
        upgrade = False
        if branch is not None:
            known_bindings = connection.execute(
                "SELECT DISTINCT data_identity_json FROM library_versions WHERE branch_id=? "
                "AND data_identity_json IS NOT NULL AND data_identity_json != 'null'",
                (branch["id"],),
            ).fetchall()
            if any(data_identity_key(json.loads(row["data_identity_json"]), branch_key) != key
                   for row in known_bindings):
                raise BranchIdentityConflictError(
                    "branch is bound to different source data; start a new branch"
                )
            unknown = connection.execute(
                "SELECT 1 FROM library_data_identities WHERE owner_id=? AND data_key=? "
                "AND library_id=? AND status='unknown'",
                (owner_id, data_identity_key(None, branch_key), branch["library_id"]),
            ).fetchone()
            # Old local uploads had no content identity. Bind that same owned
            # branch once complete provenance arrives; known bindings never drift.
            upgrade = identity is not None and unknown is not None and not connection.execute(
                "SELECT 1 FROM library_versions WHERE branch_id=? "
                "AND data_identity_json IS NOT NULL AND data_identity_json != 'null' LIMIT 1",
                (branch["id"],),
            ).fetchone()
            if upgrade:
                library_id = logical["id"] if logical is not None else branch["library_id"]
            elif logical is None or logical["id"] != branch["library_id"]:
                raise BranchIdentityConflictError(
                    "branch is bound to different or unavailable source data; start a new branch"
                )
            else:
                library_id = branch["library_id"]
            library_exists = True
        elif logical is not None:
            library_id, library_exists = logical["id"], True
        else:
            library_id, library_exists = new_library_id, False
        if (selected_id is not None and selected_id != library_id
                and not (upgrade and selected_id == branch["library_id"])):
            raise BranchIdentityConflictError("library selector belongs to different source data")
        if known_branch_id is not None and (branch is None or known_branch_id != branch["id"]):
            raise SelectorNotFoundError("branch selector not found")
        return PublicationPlan(
            library_id=library_id, branch_id=branch["id"] if branch is not None else new_branch_id,
            library_exists=library_exists, branch_exists=branch is not None,
            upgrade_data_identity=bool(upgrade),
            previous_library_id=branch["library_id"] if upgrade else None,
        )

    def prepare_publication(
        self,
        *,
        owner_id: int,
        model_id: str,
        benchmarks: list[str],
        branch_key: str,
        known_library_id: str | None,
        known_branch_id: str | None,
        new_library_id: str,
        new_branch_id: str,
        data_identity: dict | None = None,
    ) -> PublicationPlan:
        """Plan source grouping without writes; append rechecks it inside its transaction."""
        with self.database.read() as connection:
            return self._resolve_plan(
                connection,
                owner_id=owner_id,
                model_id=model_id,
                benchmarks_json=self._benchmarks_json(benchmarks),
                branch_key=branch_key,
                known_library_id=known_library_id,
                known_branch_id=known_branch_id,
                new_library_id=new_library_id,
                new_branch_id=new_branch_id,
                data_identity=data_identity,
            )

    def append_publication(
        self,
        *,
        plan: PublicationPlan,
        identity: Identity,
        branch_key: str,
        source_run_id: str,
        model_id: str,
        benchmarks: list[str],
        known_library_id: str | None,
        known_branch_id: str | None,
        version_id: str,
        created_at: str,
        function_count: int,
        library_sha256: str,
        report_sha256: str,
        library_key: str,
        report_key: str,
        harness: str,
        source_harness: str,
        reference_data: list[dict[str, Any]],
        publish_trigger: str,
        report_assets: list[dict[str, Any]] | None = None,
        data_identity: dict | None = None,
    ) -> dict[str, Any]:
        """Atomically append a version and preserve merged membership during source upgrades."""
        benchmarks_json = self._benchmarks_json(benchmarks)
        input_identity = normalize_data_identity(data_identity)
        input_key = data_identity_key(input_identity, branch_key)
        with self.database.transaction() as connection:
            resolved = self._resolve_plan(
                connection,
                owner_id=identity.id,
                model_id=model_id,
                benchmarks_json=benchmarks_json,
                branch_key=branch_key,
                known_library_id=known_library_id,
                known_branch_id=known_branch_id,
                new_library_id=plan.library_id,
                new_branch_id=plan.branch_id,
                data_identity=input_identity,
            )
            if resolved.library_id != plan.library_id or resolved.branch_id != plan.branch_id:
                raise RuntimeError("publication identity changed while artifacts were written")

            connection.execute(
                """
                INSERT INTO owners (id, username, name, avatar_url, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    username = excluded.username,
                    name = excluded.name,
                    avatar_url = excluded.avatar_url,
                    updated_at = excluded.updated_at
                """,
                (
                    identity.id,
                    identity.username,
                    identity.name,
                    identity.avatar_url,
                    created_at,
                ),
            )
            if not resolved.library_exists:
                connection.execute(
                    """
                    INSERT INTO logical_libraries (
                        id, owner_id, model_id, benchmarks_json, created_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        plan.library_id,
                        identity.id,
                        model_id,
                        benchmarks_json,
                        created_at,
                    ),
                )
            if resolved.upgrade_data_identity:
                previous_id = resolved.previous_library_id
                connection.execute(
                    "DELETE FROM library_data_identities WHERE owner_id=? AND data_key=?",
                    (identity.id, data_identity_key(None, branch_key)),
                )
                if previous_id != plan.library_id:
                    self._merge_card_membership(
                        connection, owner_id=identity.id, source_id=previous_id,
                        target_id=plan.library_id,
                    )
            connection.execute(
                "INSERT INTO library_data_identities("
                "owner_id,data_key,library_id,data_identity_json,status) "
                "VALUES(?,?,?,?,?) ON CONFLICT(owner_id,data_key) DO NOTHING",
                (identity.id, input_key, plan.library_id,
                 json.dumps(input_identity, sort_keys=True, separators=(",", ":")),
                 "known" if input_identity is not None else "unknown"),
            )
            if not resolved.branch_exists:
                connection.execute(
                    """
                    INSERT INTO library_branches (
                        id, library_id, owner_id, branch_key, created_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        plan.branch_id,
                        plan.library_id,
                        identity.id,
                        branch_key,
                        created_at,
                    ),
                )
            next_number = connection.execute(
                """
                SELECT COALESCE(MAX(version_number), 0) + 1
                FROM library_versions
                WHERE branch_id = ?
                """,
                (plan.branch_id,),
            ).fetchone()[0]
            connection.execute(
                """
                INSERT INTO library_versions (
                    id, library_id, branch_id, version_number, source_run_id,
                    created_at, function_count, library_sha256, report_sha256,
                    library_key, report_key, harness, source_harness,
                    reference_data_json, publish_trigger, model_id, benchmarks_json,
                    data_identity_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    version_id,
                    plan.library_id,
                    plan.branch_id,
                    next_number,
                    source_run_id,
                    created_at,
                    function_count,
                    library_sha256,
                    report_sha256,
                    library_key,
                    report_key,
                    harness,
                    source_harness,
                    json.dumps(
                        reference_data,
                        ensure_ascii=False,
                        allow_nan=False,
                        separators=(",", ":"),
                    ),
                    publish_trigger, model_id, benchmarks_json,
                    json.dumps(input_identity, sort_keys=True, separators=(",", ":")),
                ),
            )
            for asset in report_assets or []:
                connection.execute(
                    """
                    INSERT INTO report_assets (
                        version_id, path, artifact_key, size_bytes, sha256
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        version_id,
                        asset["path"],
                        asset["artifact_key"],
                        asset["size_bytes"],
                        asset["sha256"],
                    ),
                )
            row = connection.execute(
                VERSION_SELECT + " WHERE version.id = ?",
                (version_id,),
            ).fetchone()
        if row is None:
            raise RuntimeError("publication append did not return a version")
        return _row_to_dict(row)

    def get_branch(
        self,
        library_id: str,
        branch_id: str,
    ) -> dict[str, Any] | None:
        library_id = self.resolve_library_id(library_id, branch_id)
        with self.database.read() as connection:
            row = connection.execute(
                """
                SELECT
                    branch.id,
                    branch.library_id,
                    branch.branch_key,
                    branch.created_at,
                    latest.created_at AS updated_at,
                    latest.id AS latest_version_id,
                    latest.version_number AS latest_version_number,
                    latest.source_run_id AS latest_source_run_id,
                    (
                        SELECT COUNT(*)
                        FROM library_versions counted
                        WHERE counted.branch_id = branch.id
                    ) AS version_count
                FROM library_branches branch
                JOIN library_versions latest ON latest.sequence = (
                    SELECT MAX(candidate.sequence)
                    FROM library_versions candidate
                    WHERE candidate.branch_id = branch.id
                )
                WHERE branch.library_id = ? AND branch.id = ?
                """,
                (library_id, branch_id),
            ).fetchone()
        return dict(row) if row else None

    def list_branches(
        self,
        library_id: str,
        *,
        limit: int,
        offset: int,
        viewer_id: int | None | object = _UNRESTRICTED_LISTING,
    ) -> tuple[list[dict[str, Any]], int]:
        """List metadata with query-time visibility when a viewer is supplied.

        Public callers must pass the viewer (None for anonymous). Authorization in
        this SELECT also protects against card merges after endpoint preflight.
        Omitted viewer is reserved for trusted internal callers.
        """
        library_id = self.resolve_library_id(library_id)
        visibility = ""
        params: list[Any] = [library_id]
        if viewer_id is not _UNRESTRICTED_LISTING:
            visibility = (
                " AND ((owner.is_public=1 AND logical.is_public=1) OR logical.owner_id=?)"
            )
            params.append(viewer_id)
        with self.database.read() as connection:
            rows = connection.execute(
                f"""
                SELECT
                    branch.id,
                    branch.library_id,
                    branch.branch_key,
                    branch.created_at,
                    latest.created_at AS updated_at,
                    latest.id AS latest_version_id,
                    latest.version_number AS latest_version_number,
                    latest.source_run_id AS latest_source_run_id,
                    (
                        SELECT COUNT(*)
                        FROM library_versions counted
                        WHERE counted.branch_id = branch.id
                    ) AS version_count,
                    latest.sequence
                FROM library_branches branch
                JOIN logical_libraries logical ON logical.id=branch.library_id
                JOIN owners owner ON owner.id=logical.owner_id
                JOIN library_versions latest ON latest.sequence = (
                    SELECT MAX(candidate.sequence)
                    FROM library_versions candidate
                    WHERE candidate.branch_id = branch.id
                )
                WHERE branch.library_id = ? {visibility}
                ORDER BY latest.sequence DESC, branch.id ASC
                """,
                params,
            ).fetchall()
        items = [dict(row) for row in rows]
        return items[offset : offset + limit], len(items)

    def list_versions(
        self,
        library_id: str,
        branch_id: str,
        *,
        limit: int,
        offset: int,
        viewer_id: int | None | object = _UNRESTRICTED_LISTING,
    ) -> tuple[list[dict[str, Any]], int]:
        """List metadata with query-time visibility when a viewer is supplied.

        Public callers must pass the viewer (None for anonymous). Authorization in
        this SELECT also protects against card merges after endpoint preflight.
        Omitted viewer is reserved for trusted internal callers.
        """
        library_id = self.resolve_library_id(library_id, branch_id)
        visibility = ""
        params: list[Any] = [library_id, branch_id]
        if viewer_id is not _UNRESTRICTED_LISTING:
            visibility = (
                " AND ((owner.is_public=1 AND logical.is_public=1) OR logical.owner_id=?)"
            )
            params.append(viewer_id)
        with self.database.read() as connection:
            rows = connection.execute(
                VERSION_SELECT
                + f"""
                WHERE version.library_id = ? AND version.branch_id = ? {visibility}
                ORDER BY version.version_number DESC
                """,
                params,
            ).fetchall()
        items = [_row_to_dict(row) for row in rows]
        return items[offset : offset + limit], len(items)

    def get_version_visible(
        self,
        viewer_id: int | None,
        library_id: str,
        branch_id: str,
        version_id: str,
    ) -> dict[str, Any] | None:
        library_id = self.resolve_library_id(library_id, branch_id)
        visibility = "owner.is_public = 1 AND logical.is_public = 1"
        params: list[Any] = [library_id, branch_id, version_id]
        if viewer_id is not None:
            visibility = f"(({visibility}) OR logical.owner_id = ?)"
            params.append(viewer_id)
        with self.database.read() as connection:
            row = connection.execute(
                VERSION_SELECT
                + f"""
                WHERE version.library_id = ?
                    AND version.branch_id = ?
                    AND version.id = ?
                    AND {visibility}
                """,
                params,
            ).fetchone()
        return _row_to_dict(row) if row else None

    def get_report_asset(
        self,
        version_id: str,
        path: str,
    ) -> dict[str, Any] | None:
        with self.database.read() as connection:
            row = connection.execute(
                """
                SELECT asset.path, asset.artifact_key, asset.size_bytes, asset.sha256
                FROM report_assets asset
                WHERE asset.version_id = ? AND asset.path = ?
                """,
                (version_id, path),
            ).fetchone()
        return dict(row) if row else None

    def iter_artifact_keys(self, owner_id: int, library_id: str) -> Iterator[str]:
        """Enumerate deletion artifacts only for the exact owned canonical card ID."""
        with self.database.read() as connection:
            cursor = connection.execute(
                """
                SELECT version.library_key AS artifact_key
                FROM library_versions version
                JOIN logical_libraries logical ON logical.id = version.library_id
                WHERE logical.owner_id = ? AND logical.id = ?

                UNION

                SELECT version.report_key AS artifact_key
                FROM library_versions version
                JOIN logical_libraries logical ON logical.id = version.library_id
                WHERE logical.owner_id = ? AND logical.id = ?

                UNION

                SELECT asset.artifact_key
                FROM report_assets asset
                JOIN library_versions version ON version.id = asset.version_id
                JOIN logical_libraries logical ON logical.id = version.library_id
                WHERE logical.owner_id = ? AND logical.id = ?

                ORDER BY artifact_key
                """,
                (
                    owner_id,
                    library_id,
                    owner_id,
                    library_id,
                    owner_id,
                    library_id,
                ),
            )
            while rows := cursor.fetchmany(100):
                for row in rows:
                    yield row["artifact_key"]

    def delete_owned(self, owner_id: int, library_id: str) -> bool:
        """Delete the exact owned card; historical navigation aliases cannot delete it."""
        with self.database.transaction() as connection:
            cursor = connection.execute(
                "DELETE FROM logical_libraries WHERE id = ? AND owner_id = ?",
                (library_id, owner_id),
            )
        return cursor.rowcount == 1
