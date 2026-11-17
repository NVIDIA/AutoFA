# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sqlite3
import time
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path

from .data_cards import install_immutability_trigger

SCHEMA_VERSION = 7
_SETUP_ATTEMPTS = 5


def _lock_conflict(error: sqlite3.OperationalError) -> bool:
    """True for SQLITE_BUSY or SQLITE_LOCKED, including their extended codes."""
    code = getattr(error, "sqlite_errorcode", None)
    return code is not None and code & 0xFF in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}


class LegacySchemaError(RuntimeError):
    """Raised when an existing database is at an earlier, unknown or missing schema version."""


class NewerSchemaError(RuntimeError):
    """Raised when an existing database is at a schema version newer than this release."""


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)

    def connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(
            self.path,
            timeout=5.0,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=5000")
        return connection

    def initialize(self) -> None:
        """Create the current schema in a new database, or open a current one unchanged.

        A database at any other schema version is rejected before anything is written.
        """
        # Servers starting together on a new file race to switch it to WAL, and SQLite
        # reports the loser's lock upgrade as busy at once, without waiting for
        # busy_timeout. An attempt writes only the schema, in one transaction that a
        # failed attempt rolls back, so it is safe to repeat. Library data is written
        # only after this returns and is never retried here.
        for attempt in range(_SETUP_ATTEMPTS):
            try:
                self._initialize_once()
                return
            except sqlite3.OperationalError as error:
                if attempt == _SETUP_ATTEMPTS - 1 or not _lock_conflict(error):
                    raise
                time.sleep(0.025 * 2**attempt)

    def _initialize_once(self) -> None:
        """Run one setup attempt: probe the schema, switch to WAL, create the schema if new."""
        current = None
        if self.path.exists() and Path(f"{self.path}-wal").exists():
            # Probe read-only: a read-write connection closing last would checkpoint the
            # pending WAL frames into the main file and delete the -wal.
            uri = f"{self.path.resolve().as_uri()}?mode=ro"
            with closing(sqlite3.connect(uri, uri=True, timeout=5.0)) as probe:
                probe.row_factory = sqlite3.Row
                current = self._has_current_schema(probe)
        with closing(self.connect()) as connection:
            if current is None:
                # No -wal: nothing to checkpoint. Unlike a read-only connection, this one
                # also removes the -wal/-shm it creates when it is the last to close.
                current = self._has_current_schema(connection)
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
            if current:
                return
            # One transaction, committed below (closing without it rolls back): other
            # starters see no tables or all of them, and a starter that waited for the
            # lock repeats only idempotent statements. The text matches what the previous
            # release stored for newly created databases (including the quoted
            # logical_libraries name), so new and existing fresh-created databases have
            # identical schemas.
            connection.executescript(
                """
                BEGIN IMMEDIATE;

                CREATE TABLE IF NOT EXISTS schema_info (
                    singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                    version INTEGER NOT NULL
                );

                INSERT OR IGNORE INTO schema_info (singleton, version)
                VALUES (1, 7);

                CREATE TABLE IF NOT EXISTS owners (
                    id INTEGER PRIMARY KEY,
                    username TEXT NOT NULL,
                    name TEXT NOT NULL,
                    avatar_url TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL,
                    is_public INTEGER NOT NULL DEFAULT 1
                        CHECK(is_public IN (0, 1))
                );

                CREATE TABLE IF NOT EXISTS "logical_libraries" (
                    id TEXT PRIMARY KEY,
                    owner_id INTEGER NOT NULL REFERENCES owners(id)
                        ON UPDATE CASCADE ON DELETE CASCADE,
                    model_id TEXT NOT NULL, benchmarks_json TEXT NOT NULL, created_at TEXT NOT NULL,
                    is_public INTEGER NOT NULL DEFAULT 1 CHECK(is_public IN (0,1)),
                    UNIQUE(id, owner_id)
                );

                CREATE TABLE IF NOT EXISTS library_branches (
                    id TEXT PRIMARY KEY,
                    library_id TEXT NOT NULL,
                    owner_id INTEGER NOT NULL,
                    branch_key TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(library_id, owner_id)
                        REFERENCES logical_libraries(id, owner_id)
                        ON UPDATE CASCADE
                        ON DELETE CASCADE,
                    UNIQUE(owner_id, branch_key),
                    UNIQUE(id, library_id)
                );

                CREATE TABLE IF NOT EXISTS library_versions (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    id TEXT NOT NULL UNIQUE,
                    library_id TEXT NOT NULL,
                    branch_id TEXT NOT NULL,
                    version_number INTEGER NOT NULL CHECK(version_number >= 1),
                    source_run_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    function_count INTEGER NOT NULL CHECK(function_count >= 0),
                    library_sha256 TEXT NOT NULL,
                    report_sha256 TEXT NOT NULL,
                    library_key TEXT NOT NULL UNIQUE,
                    report_key TEXT NOT NULL UNIQUE,
                    harness TEXT NOT NULL DEFAULT 'unknown',
                    source_harness TEXT NOT NULL DEFAULT 'unknown',
                    reference_data_json TEXT NOT NULL DEFAULT '[]',
                    publish_trigger TEXT NOT NULL DEFAULT 'unknown',
                    model_id TEXT,
                    benchmarks_json TEXT,
                    data_identity_json TEXT,
                    FOREIGN KEY(branch_id, library_id)
                        REFERENCES library_branches(id, library_id)
                        ON UPDATE CASCADE
                        ON DELETE CASCADE,
                    UNIQUE(branch_id, version_number)
                );

                CREATE INDEX IF NOT EXISTS idx_logical_libraries_owner_model
                    ON logical_libraries(owner_id, model_id);
                CREATE INDEX IF NOT EXISTS idx_library_branches_library
                    ON library_branches(library_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_library_versions_library_latest
                    ON library_versions(library_id, sequence DESC);
                CREATE INDEX IF NOT EXISTS idx_library_versions_branch_latest
                    ON library_versions(branch_id, version_number DESC);

                CREATE TABLE IF NOT EXISTS local_file_snapshots (
                    source_path TEXT PRIMARY KEY,
                    fingerprint TEXT NOT NULL,
                    signature_json TEXT NOT NULL,
                    report_pending INTEGER NOT NULL DEFAULT 0,
                    source_kind TEXT NOT NULL,
                    version_id TEXT NOT NULL
                        REFERENCES library_versions(id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS local_source_activity (
                    source_run_id TEXT PRIMARY KEY,
                    published_at TEXT,
                    saved_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS report_assets (
                    version_id TEXT NOT NULL
                        REFERENCES library_versions(id)
                        ON DELETE CASCADE,
                    path TEXT NOT NULL,
                    artifact_key TEXT NOT NULL UNIQUE,
                    size_bytes INTEGER NOT NULL CHECK(size_bytes >= 0),
                    sha256 TEXT NOT NULL,
                    PRIMARY KEY(version_id, path)
                );

                CREATE INDEX IF NOT EXISTS idx_report_assets_version
                    ON report_assets(version_id);

                -- 'legacy_mixed' stays valid: databases upgraded to schema 7 hold such cards.
                CREATE TABLE IF NOT EXISTS library_data_identities (
                    owner_id INTEGER NOT NULL, data_key TEXT NOT NULL,
                    library_id TEXT NOT NULL, data_identity_json TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('known','unknown','legacy_mixed')),
                    PRIMARY KEY(owner_id, data_key),
                    FOREIGN KEY(library_id, owner_id) REFERENCES logical_libraries(id, owner_id)
                        ON UPDATE CASCADE ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_library_data_identities_card
                    ON library_data_identities(library_id);

                CREATE TABLE IF NOT EXISTS library_aliases (
                    alias_id TEXT PRIMARY KEY,
                    library_id TEXT NOT NULL REFERENCES logical_libraries(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS library_branch_aliases (
                    alias_id TEXT NOT NULL, branch_id TEXT NOT NULL
                        REFERENCES library_branches(id) ON DELETE CASCADE,
                    library_id TEXT NOT NULL REFERENCES logical_libraries(id) ON DELETE CASCADE,
                    PRIMARY KEY(alias_id, branch_id)
                );
                """
            )
            install_immutability_trigger(connection)
            connection.commit()

    def _has_current_schema(self, connection: sqlite3.Connection) -> bool:
        """Return False for an empty database; reject any schema but the current one."""
        tables = {
            row["name"]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
            if not row["name"].startswith("sqlite_")
        }
        if not tables:
            return False
        version = "none"
        if "schema_info" in tables:
            row = connection.execute(
                "SELECT version FROM schema_info WHERE singleton = 1"
            ).fetchone()
            version = "unknown" if row is None else row["version"]
        if isinstance(version, int) and version > SCHEMA_VERSION:
            raise NewerSchemaError(
                f"AutoFA library database {self.path} uses schema version {version}, which is "
                f"newer than this release supports (schema version {SCHEMA_VERSION}); "
                "upgrade AutoFA to open it."
            )
        if version != SCHEMA_VERSION:
            raise LegacySchemaError(
                f"AutoFA library database {self.path} uses schema version {version}, which is "
                f"no longer supported (this release opens schema version {SCHEMA_VERSION} only). "
                "Set the file aside, with any -wal/-shm files beside it, to start a new database."
            )
        return True

    @contextmanager
    def read(self) -> Iterator[sqlite3.Connection]:
        connection = self.connect()
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
