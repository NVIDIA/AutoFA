# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Card aliases, branch moves and version immutability for source-bound library cards.

Artifact contents, version IDs and branch IDs are never rewritten. Only card
membership changes, with explicit aliases for historical links.
"""

from __future__ import annotations


def resolve_library_alias(connection, library_id: str, branch_id: str | None = None) -> str:
    """Resolve historical links, preferring a branch's actual current residence."""
    seen = set()
    while library_id not in seen:
        seen.add(library_id)
        row = None
        if branch_id is not None:
            if connection.execute(
                "SELECT 1 FROM library_branches WHERE id=? AND library_id=?",
                (branch_id, library_id),
            ).fetchone():
                # A branch can return to an earlier card after a merge or source
                # upgrade. Its current residence wins over stale historical links.
                return library_id
            row = connection.execute(
                "SELECT library_id FROM library_branch_aliases "
                "WHERE alias_id = ? AND branch_id = ?",
                (library_id, branch_id),
            ).fetchone()
        if row is None:
            row = connection.execute(
                "SELECT library_id FROM library_aliases WHERE alias_id = ?",
                (library_id,),
            ).fetchone()
        if row is None or row["library_id"] == library_id:
            return library_id
        library_id = row["library_id"]
    raise ValueError("library alias cycle")


def move_branch(connection, branch_id: str, source_id: str, target_id: str) -> None:
    """Move branch metadata and point all its historical card links at the destination."""
    if source_id == target_id:
        return
    connection.execute(
        "UPDATE library_branches SET library_id = ? WHERE id = ? AND library_id = ?",
        (target_id, branch_id, source_id),
    )
    # The composite FK cascades this mapping; update it explicitly as well, so the
    # move also holds on a connection without foreign-key enforcement.
    connection.execute(
        "UPDATE library_versions SET library_id = ? WHERE branch_id = ? AND library_id = ?",
        (target_id, branch_id, source_id),
    )
    connection.execute(
        "UPDATE library_branch_aliases SET library_id=? WHERE branch_id=?",
        (target_id, branch_id),
    )
    connection.execute(
        "DELETE FROM library_branch_aliases WHERE alias_id=? AND branch_id=?",
        (target_id, branch_id),
    )
    connection.execute(
        "INSERT OR REPLACE INTO library_branch_aliases(alias_id, branch_id, library_id) "
        "VALUES (?, ?, ?)",
        (source_id, branch_id, target_id),
    )


def install_immutability_trigger(connection) -> None:
    columns = [row["name"] for row in connection.execute("PRAGMA table_info(library_versions)")]
    checks = " OR ".join(
        f"NEW.{name} IS NOT OLD.{name}" for name in columns if name != "library_id"
    )
    connection.execute("DROP TRIGGER IF EXISTS prevent_library_version_updates")
    connection.execute(f"""
        CREATE TRIGGER prevent_library_version_updates
        BEFORE UPDATE ON library_versions
        WHEN ({checks}) OR NOT EXISTS (
            SELECT 1 FROM logical_libraries old_card JOIN logical_libraries new_card
                ON new_card.owner_id = old_card.owner_id
            WHERE old_card.id = OLD.library_id AND new_card.id = NEW.library_id
        )
        BEGIN
            SELECT RAISE(ABORT, 'library versions are immutable');
        END
    """)
