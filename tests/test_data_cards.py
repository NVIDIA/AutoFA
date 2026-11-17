# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data cards: source grouping, merges, the schema-7 database, hub routes and views."""

from __future__ import annotations

import json
import sqlite3
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from copy import deepcopy
from html.parser import HTMLParser
from pathlib import Path
from threading import Barrier, BrokenBarrierError, local
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from jinja2 import Environment, FileSystemLoader, select_autoescape

import autofa_library_server
from autofa_library_server.auth import AuthContext
from autofa_library_server.data_cards import move_branch
from autofa_library_server.data_identity import normalize_data_identity
from autofa_library_server.db import Database, LegacySchemaError, NewerSchemaError
from autofa_library_server.repository import (
    BranchIdentityConflictError,
    LibraryRepository,
    SelectorNotFoundError,
    to_record,
)
from autofa_library_server.schemas import Identity

from support.library_server import LIBRARY
from support.library_server import local_app as local_app

CARD_TABLES = (
    "logical_libraries", "library_branches", "library_versions",
    "library_data_identities", "library_aliases", "library_branch_aliases",
)


def identity(*sources):
    return {"version": 1, "sources": list(sources)}


def source(run_id="source-a"):
    """EvalRDL source-data identity."""
    return identity({"kind": "evalrdl", "origin": "https://data.example", "run_id": run_id})


def local_source(letter):
    """Local content identity whose digest repeats ``letter``."""
    return identity({"kind": "local", "sha256": letter * 64})


def table_rows(database, tables=CARD_TABLES):
    """Capture the complete card history and alias membership."""
    with database.read() as connection:
        return {table: [dict(row) for row in connection.execute(
            f"SELECT * FROM {table} ORDER BY 1")] for table in tables}


def make_private(repository, library_id):
    """Store a private card directly; no current route changes card visibility."""
    with repository.database.transaction() as connection:
        updated = connection.execute(
            "UPDATE logical_libraries SET is_public=0 WHERE id=?", (library_id,),
        ).rowcount
    assert updated == 1


def saved_versions(repository):
    """Saved version rows by id, ignoring which card currently owns them."""
    with repository.database.read() as connection:
        rows = [dict(row) for row in connection.execute("SELECT * FROM library_versions")]
    return {row["id"]: {k: v for k, v in row.items() if k != "library_id"} for row in rows}


class Page(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.tags = []
        self.text = []
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))

    def handle_data(self, data):
        self.text.append(data)


# Repository


@pytest.fixture
def repository(tmp_path):
    database = Database(tmp_path / "cards.db")
    database.initialize()
    return LibraryRepository(database)


def append(repository, branch="branch-a", *, data=None, owner=1,
           model="model", benchmarks=None, selected=None, ids=None):
    """Publish one version through the repository's prepare/append protocol.

    ``ids`` fixes the (new card, new branch, version) IDs instead of random ones.
    """
    common = dict(model_id=model, benchmarks=benchmarks or ["benchmark"], branch_key=branch,
                  known_library_id=selected, known_branch_id=None, data_identity=data)
    library_id, branch_id, version_id = ids or [str(uuid.uuid4()) for _ in range(3)]
    plan = repository.prepare_publication(owner_id=owner, new_library_id=library_id,
                                          new_branch_id=branch_id, **common)
    return repository.append_publication(
        plan=plan, identity=Identity(id=owner, username=f"user-{owner}", name=f"User {owner}"),
        source_run_id="analysis-" + version_id, version_id=version_id,
        created_at="2026-10-29T12:00:00Z", function_count=1, library_sha256="library-digest",
        report_sha256="report-digest", library_key=f"{version_id}/library",
        report_key=f"{version_id}/report", harness="claude", source_harness="test",
        reference_data=[], publish_trigger="manual_cli", **common,
    )


def test_source_identity_not_model_or_benchmark_controls_grouping(repository):
    first = append(repository, data=source(), model="before", benchmarks=["one"])
    second = append(repository, "branch-b", data=source(), model="after", benchmarks=["two"])
    assert first["library_id"] == second["library_id"]
    card = repository.get_owned(1, first["library_id"])
    assert card["branch_count"] == 2
    assert card["model_id"] == "after" and card["benchmarks"] == ["two"]
    historical = repository.get_version_visible(
        1, first["library_id"], first["branch_id"], first["id"],
    )
    assert historical["model_id"] == "before" and historical["benchmarks"] == ["one"]
    assert card["data_identity"] == source()
    assert card["data_identity_status"] == "known"


def test_unidentified_branches_get_separate_unknown_cards(repository):
    unknown = append(repository, "branch-c", data=None)
    unknown_other = append(repository, "branch-d", data=None)
    assert unknown["library_id"] != unknown_other["library_id"]
    assert repository.get_owned(1, unknown["library_id"])["data_identity_status"] == "unknown"


def test_selectors_cannot_force_a_different_dataset_into_a_card(repository):
    first = append(repository, data=source())
    with pytest.raises(BranchIdentityConflictError):
        append(repository, "branch-b", data=source("source-b"), selected=first["library_id"])
    with pytest.raises(BranchIdentityConflictError):
        append(repository, data=source("source-b"))
    with pytest.raises(SelectorNotFoundError):
        append(repository, "branch-b", data=source(), owner=2, selected=first["library_id"])


def test_merge_is_owner_only_and_transactional(repository):
    first = append(repository, data=source())
    second = append(repository, "branch-b", data=source("source-b"))
    ids = dict(source_library_id=first["library_id"], target_library_id=second["library_id"])
    with pytest.raises(SelectorNotFoundError):
        repository.merge_libraries(owner_id=2, **ids)
    with repository.database.transaction() as connection:
        connection.execute("""CREATE TRIGGER abort_merge BEFORE DELETE ON logical_libraries
            BEGIN SELECT RAISE(ABORT, 'simulated storage failure'); END""")
    with pytest.raises(sqlite3.IntegrityError, match="simulated"):
        repository.merge_libraries(owner_id=1, **ids)
    assert repository.get_owned(1, first["library_id"])["branch_count"] == 1
    assert repository.get_owned(1, second["library_id"])["branch_count"] == 1
    assert repository.resolve_library_id(first["library_id"]) == first["library_id"]


def test_sources_and_owners_get_separate_cards_with_immutable_membership(repository):
    first = append(repository, data=source())
    different = append(repository, "branch-b", data=source("source-b"))
    other = append(repository, "branch-c", data=source(), owner=2)
    assert len({first["library_id"], different["library_id"], other["library_id"]}) == 3
    with repository.database.transaction() as connection:
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute(
                "UPDATE library_versions SET model_id='rewritten' WHERE id=?", (first["id"],)
            )
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute(
                "UPDATE library_versions SET library_id=? WHERE id=?",
                (other["library_id"], first["id"]),
            )


def test_normalizer_matches_set_identity_and_rejects_credentials():
    data = source()
    data["sources"] += [{"kind": "local", "sha256": "a" * 64}, data["sources"][0]]
    data["sources"][0]["origin"] = "https://DATA.EXAMPLE:443/"
    expected = normalize_data_identity(data)
    assert normalize_data_identity(identity(*reversed(data["sources"]))) == expected
    with pytest.raises(ValueError):
        normalize_data_identity(identity(
            {"kind": "evalrdl", "origin": "https://@data.example", "run_id": "source-a"}
        ))


@pytest.mark.parametrize("existing_target", [False, True])
def test_legacy_unknown_branch_upgrades_once_without_losing_history(repository, existing_target):
    old = append(repository, data=None)
    target = (
        append(repository, "known-branch", data=source(), model="other-model")
        if existing_target
        else None
    )
    make_private(repository, old["library_id"])
    upgraded = append(repository, data=source(), selected=old["library_id"])
    assert upgraded["branch_id"] == old["branch_id"]
    assert upgraded["version_number"] == 2
    assert upgraded["library_id"] == (target["library_id"] if target else old["library_id"])
    historic = repository.get_version_visible(1, old["library_id"], old["branch_id"], old["id"])
    assert historic["data_identity"] is None
    assert historic["library_key"] == old["library_key"]
    assert repository.get_visible(None, upgraded["library_id"]) is None
    with pytest.raises(BranchIdentityConflictError):
        append(repository, data=source("different"), selected=old["library_id"])
    with pytest.raises(BranchIdentityConflictError):
        append(repository, data=None, selected=old["library_id"])


def test_manual_merge_does_not_allow_an_existing_branch_to_change_dataset(repository):
    first = append(repository, data=source())
    second = append(repository, "branch-b", data=source("source-b"))
    repository.merge_libraries(
        owner_id=1, source_library_id=first["library_id"], target_library_id=second["library_id"]
    )
    with pytest.raises(BranchIdentityConflictError, match="different source data"):
        append(repository, data=source("source-b"))
    same = append(repository, data=source(), selected=first["library_id"])
    assert same["branch_id"] == first["branch_id"] and same["version_number"] == 2


def test_identity_upgrade_keeps_merged_membership_and_historical_links(repository):
    """An identified branch brings its explicitly merged peers to the source card."""
    unknown = append(repository, "unknown", data=None)
    first = append(repository, "known-a", data=source("a"))
    repository.merge_libraries(
        owner_id=1, source_library_id=unknown["library_id"], target_library_id=first["library_id"],
    )
    target = append(repository, "known-b", data=source("b"))
    make_private(repository, first["library_id"])
    originals = saved_versions(repository)

    upgraded = append(repository, "unknown", data=source("b"), selected=unknown["library_id"])

    cards, count = repository.list_owned(1, limit=10, offset=0)
    assert count == 1
    assert cards[0]["is_merged"] and cards[0]["data_group_count"] == 2
    assert cards[0]["is_public"] == 0
    for old in (unknown, first, target):
        assert repository.resolve_library_id(old["library_id"], old["branch_id"]) == (
            upgraded["library_id"]
        )
        assert repository.get_visible(None, old["library_id"]) is None
        saved = repository.get_version_visible(1, old["library_id"], old["branch_id"], old["id"])
        assert saved["library_key"] == old["library_key"]
        assert saved["data_identity"] == old["data_identity"]
    current = saved_versions(repository)
    assert {key: current[key] for key in originals} == originals
    for run in ("a", "b"):
        future = append(repository, f"future-{run}", data=source(run))
        assert future["library_id"] == upgraded["library_id"]
    resumed = append(repository, "known-a", data=source("a"), selected=first["library_id"])
    assert resumed["branch_id"] == first["branch_id"]


def test_failed_identity_upgrade_rolls_back_whole_merged_membership(repository):
    """Publication failure must roll back card moves as well as the new version."""
    unknown = append(repository, "unknown", data=None)
    first = append(repository, "known-a", data=source("a"))
    repository.merge_libraries(
        owner_id=1, source_library_id=unknown["library_id"], target_library_id=first["library_id"],
    )
    target = append(repository, "known-b", data=source("b"))
    with repository.database.transaction() as connection:
        connection.execute("""CREATE TRIGGER abort_upgrade BEFORE INSERT ON library_versions
            BEGIN SELECT RAISE(ABORT, 'simulated publication failure'); END""")
    with pytest.raises(sqlite3.IntegrityError, match="simulated publication failure"):
        append(repository, "unknown", data=source("b"), selected=unknown["library_id"])
    assert repository.list_owned(1, limit=10, offset=0)[1] == 2
    assert repository.resolve_library_id(unknown["library_id"], unknown["branch_id"]) == (
        first["library_id"]
    )
    assert repository.get_owned(1, first["library_id"])["branch_count"] == 2
    assert repository.get_owned(1, target["library_id"])["branch_count"] == 1
    assert repository.get_owned(1, first["library_id"])["data_identity_status"] == "unknown"


def test_repeated_branch_moves_flatten_historical_aliases(repository):
    database = repository.database
    cards = [append(repository, f"branch-{index}", data=source(str(index))) for index in range(3)]
    branch_id = cards[0]["branch_id"]
    original = cards[0]
    current_id = original["library_id"]
    visited = {current_id}
    for target in (cards[1], cards[0], cards[2], cards[1]):
        with database.transaction() as connection:
            move_branch(connection, branch_id, current_id, target["library_id"])
        current_id = target["library_id"]
        visited.add(current_id)
        for prior in visited:
            assert repository.resolve_library_id(prior, branch_id) == current_id
            assert repository.get_version_visible(1, prior, branch_id, original["id"])


def test_resolver_recovers_existing_alias_cycle_at_actual_branch_residence(repository):
    first = append(repository, data=source("a"))
    second = append(repository, "other", data=source("b"))
    with repository.database.transaction() as connection:
        for start, end in ((first, second), (second, first)):
            connection.execute(
                "INSERT INTO library_branch_aliases VALUES(?,?,?)",
                (start["library_id"], first["branch_id"], end["library_id"]),
            )
    for old in (first, second):
        assert repository.resolve_library_id(old["library_id"], first["branch_id"]) == (
            first["library_id"]
        )
        assert repository.get_version_visible(1, old["library_id"], first["branch_id"], first["id"])


def test_identity_upgrade_after_merge_into_unknown_card_keeps_history_and_grouping(repository):
    """A known card merged into an unknown branch's card follows that branch's upgrade."""
    unknown = append(repository, "unknown-branch", data=None)
    known = append(repository, "known-b", data=source("source-b"))
    other = append(repository, "known-a", data=source("source-a"))
    repository.merge_libraries(
        owner_id=1, source_library_id=other["library_id"],
        target_library_id=unknown["library_id"],
    )
    upgraded = append(repository, "unknown-branch", data=source("source-b"))
    assert upgraded["library_id"] == known["library_id"]
    card = repository.get_owned(1, upgraded["library_id"])
    assert card["branch_count"] == 3
    assert repository.list_owned(1, limit=10, offset=0)[1] == 1
    for original_id in (unknown["library_id"], other["library_id"]):
        historic = repository.get_version_visible(
            1, original_id, unknown["branch_id"], unknown["id"],
        )
        assert historic["library_id"] == upgraded["library_id"]
        assert historic["data_identity"] is None
        assert historic["library_key"] == unknown["library_key"]
    assert repository.get_version_visible(1, other["library_id"], other["branch_id"], other["id"])
    assert {json.dumps(item, sort_keys=True) for item in card["data_identities"]} == {
        json.dumps(source("source-a"), sort_keys=True),
        json.dumps(source("source-b"), sort_keys=True),
    }
    with repository.database.read() as connection:
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []


def test_historically_mixed_card_still_reads_and_validates(repository):
    """Databases upgraded to schema 7 before GA can hold 'legacy_mixed' source groups."""
    saved = append(repository, data=source())
    with repository.database.transaction() as connection:
        connection.execute(
            "UPDATE library_data_identities SET data_key=?, data_identity_json='null', "
            "status='legacy_mixed' WHERE library_id=?",
            ("mixed:" + saved["branch_id"], saved["library_id"]),
        )
    card = repository.get_owned(1, saved["library_id"])
    assert card["data_identity_status"] == "legacy_mixed"
    assert card["data_identity"] is None and card["version_count"] == 1
    assert to_record(card).data_identity_status == "legacy_mixed"
    assert repository.get_version_visible(
        1, saved["library_id"], saved["branch_id"], saved["id"],
    )["data_identity"] == source()


# Schema-7 database


SCHEMA_7 = {
    "schema_info": ("singleton INTEGER, version INTEGER", set()),
    "owners": (
        "id INTEGER, username TEXT, name TEXT, avatar_url TEXT, updated_at TEXT, is_public INTEGER",
        set(),
    ),
    "logical_libraries": (
        "id TEXT, owner_id INTEGER, model_id TEXT, benchmarks_json TEXT, created_at TEXT, "
        "is_public INTEGER",
        {"pk(id)", "u(id, owner_id)", "idx_logical_libraries_owner_model(owner_id, model_id)"},
    ),
    "library_branches": (
        "id TEXT, library_id TEXT, owner_id INTEGER, branch_key TEXT, created_at TEXT",
        {"pk(id)", "u(owner_id, branch_key)", "u(id, library_id)",
         "idx_library_branches_library(library_id, created_at)"},
    ),
    "library_versions": (
        "sequence INTEGER, id TEXT, library_id TEXT, branch_id TEXT, version_number INTEGER, "
        "source_run_id TEXT, created_at TEXT, function_count INTEGER, library_sha256 TEXT, "
        "report_sha256 TEXT, library_key TEXT, report_key TEXT, harness TEXT, "
        "source_harness TEXT, reference_data_json TEXT, publish_trigger TEXT, model_id TEXT, "
        "benchmarks_json TEXT, data_identity_json TEXT",
        {"u(id)", "u(library_key)", "u(report_key)", "u(branch_id, version_number)",
         "idx_library_versions_library_latest(library_id, sequence)",
         "idx_library_versions_branch_latest(branch_id, version_number)"},
    ),
    "local_file_snapshots": (
        "source_path TEXT, fingerprint TEXT, signature_json TEXT, report_pending INTEGER, "
        "source_kind TEXT, version_id TEXT",
        {"pk(source_path)"},
    ),
    "local_source_activity": (
        "source_run_id TEXT, published_at TEXT, saved_at TEXT", {"pk(source_run_id)"},
    ),
    "report_assets": (
        "version_id TEXT, path TEXT, artifact_key TEXT, size_bytes INTEGER, sha256 TEXT",
        {"pk(version_id, path)", "u(artifact_key)", "idx_report_assets_version(version_id)"},
    ),
    "library_data_identities": (
        "owner_id INTEGER, data_key TEXT, library_id TEXT, data_identity_json TEXT, status TEXT",
        {"pk(owner_id, data_key)", "idx_library_data_identities_card(library_id)"},
    ),
    "library_aliases": ("alias_id TEXT, library_id TEXT", {"pk(alias_id)"}),
    "library_branch_aliases": (
        "alias_id TEXT, branch_id TEXT, library_id TEXT", {"pk(alias_id, branch_id)"},
    ),
}
# The stored DDL (whitespace-normalised) and each table's foreign keys, as the previous
# release created them for a new database: NOT NULL, DEFAULT and CHECK clauses, the
# foreign-key actions and the trigger body.
STORED_SCHEMA_7 = {
    "schema_info": (
        "CREATE TABLE schema_info ( singleton INTEGER PRIMARY KEY CHECK(singleton = 1), version"
        " INTEGER NOT NULL )"
    ),
    "owners": (
        "CREATE TABLE owners ( id INTEGER PRIMARY KEY, username TEXT NOT NULL, name TEXT NOT "
        "NULL, avatar_url TEXT NOT NULL DEFAULT '', updated_at TEXT NOT NULL, is_public INTEGER"
        " NOT NULL DEFAULT 1 CHECK(is_public IN (0, 1)) )"
    ),
    "logical_libraries": (
        'CREATE TABLE "logical_libraries" ( id TEXT PRIMARY KEY, owner_id INTEGER NOT NULL '
        "REFERENCES owners(id) ON UPDATE CASCADE ON DELETE CASCADE, model_id TEXT NOT NULL, "
        "benchmarks_json TEXT NOT NULL, created_at TEXT NOT NULL, is_public INTEGER NOT NULL "
        "DEFAULT 1 CHECK(is_public IN (0,1)), UNIQUE(id, owner_id) )"
    ),
    "library_branches": (
        "CREATE TABLE library_branches ( id TEXT PRIMARY KEY, library_id TEXT NOT NULL, "
        "owner_id INTEGER NOT NULL, branch_key TEXT NOT NULL, created_at TEXT NOT NULL, FOREIGN"
        " KEY(library_id, owner_id) REFERENCES logical_libraries(id, owner_id) ON UPDATE "
        "CASCADE ON DELETE CASCADE, UNIQUE(owner_id, branch_key), UNIQUE(id, library_id) )"
    ),
    "library_versions": (
        "CREATE TABLE library_versions ( sequence INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT "
        "NOT NULL UNIQUE, library_id TEXT NOT NULL, branch_id TEXT NOT NULL, version_number "
        "INTEGER NOT NULL CHECK(version_number >= 1), source_run_id TEXT NOT NULL, created_at "
        "TEXT NOT NULL, function_count INTEGER NOT NULL CHECK(function_count >= 0), "
        "library_sha256 TEXT NOT NULL, report_sha256 TEXT NOT NULL, library_key TEXT NOT NULL "
        "UNIQUE, report_key TEXT NOT NULL UNIQUE, harness TEXT NOT NULL DEFAULT 'unknown', "
        "source_harness TEXT NOT NULL DEFAULT 'unknown', reference_data_json TEXT NOT NULL "
        "DEFAULT '[]', publish_trigger TEXT NOT NULL DEFAULT 'unknown', model_id TEXT, "
        "benchmarks_json TEXT, data_identity_json TEXT, FOREIGN KEY(branch_id, library_id) "
        "REFERENCES library_branches(id, library_id) ON UPDATE CASCADE ON DELETE CASCADE, "
        "UNIQUE(branch_id, version_number) )"
    ),
    "sqlite_sequence": "CREATE TABLE sqlite_sequence(name,seq)",
    "idx_logical_libraries_owner_model": (
        "CREATE INDEX idx_logical_libraries_owner_model ON logical_libraries(owner_id, "
        "model_id)"
    ),
    "idx_library_branches_library": (
        "CREATE INDEX idx_library_branches_library ON library_branches(library_id, created_at "
        "DESC)"
    ),
    "idx_library_versions_library_latest": (
        "CREATE INDEX idx_library_versions_library_latest ON library_versions(library_id, "
        "sequence DESC)"
    ),
    "idx_library_versions_branch_latest": (
        "CREATE INDEX idx_library_versions_branch_latest ON library_versions(branch_id, "
        "version_number DESC)"
    ),
    "local_file_snapshots": (
        "CREATE TABLE local_file_snapshots ( source_path TEXT PRIMARY KEY, fingerprint TEXT NOT"
        " NULL, signature_json TEXT NOT NULL, report_pending INTEGER NOT NULL DEFAULT 0, "
        "source_kind TEXT NOT NULL, version_id TEXT NOT NULL REFERENCES library_versions(id) ON"
        " DELETE CASCADE )"
    ),
    "local_source_activity": (
        "CREATE TABLE local_source_activity ( source_run_id TEXT PRIMARY KEY, published_at "
        "TEXT, saved_at TEXT NOT NULL )"
    ),
    "report_assets": (
        "CREATE TABLE report_assets ( version_id TEXT NOT NULL REFERENCES library_versions(id) "
        "ON DELETE CASCADE, path TEXT NOT NULL, artifact_key TEXT NOT NULL UNIQUE, size_bytes "
        "INTEGER NOT NULL CHECK(size_bytes >= 0), sha256 TEXT NOT NULL, PRIMARY KEY(version_id,"
        " path) )"
    ),
    "idx_report_assets_version": (
        "CREATE INDEX idx_report_assets_version ON report_assets(version_id)"
    ),
    "library_data_identities": (
        "CREATE TABLE library_data_identities ( owner_id INTEGER NOT NULL, data_key TEXT NOT "
        "NULL, library_id TEXT NOT NULL, data_identity_json TEXT NOT NULL, status TEXT NOT NULL"
        " CHECK(status IN ('known','unknown','legacy_mixed')), PRIMARY KEY(owner_id, data_key),"
        " FOREIGN KEY(library_id, owner_id) REFERENCES logical_libraries(id, owner_id) ON "
        "UPDATE CASCADE ON DELETE CASCADE )"
    ),
    "idx_library_data_identities_card": (
        "CREATE INDEX idx_library_data_identities_card ON library_data_identities(library_id)"
    ),
    "library_aliases": (
        "CREATE TABLE library_aliases ( alias_id TEXT PRIMARY KEY, library_id TEXT NOT NULL "
        "REFERENCES logical_libraries(id) ON DELETE CASCADE )"
    ),
    "library_branch_aliases": (
        "CREATE TABLE library_branch_aliases ( alias_id TEXT NOT NULL, branch_id TEXT NOT NULL "
        "REFERENCES library_branches(id) ON DELETE CASCADE, library_id TEXT NOT NULL REFERENCES"
        " logical_libraries(id) ON DELETE CASCADE, PRIMARY KEY(alias_id, branch_id) )"
    ),
    "prevent_library_version_updates": (
        "CREATE TRIGGER prevent_library_version_updates BEFORE UPDATE ON library_versions WHEN "
        "(NEW.sequence IS NOT OLD.sequence OR NEW.id IS NOT OLD.id OR NEW.branch_id IS NOT "
        "OLD.branch_id OR NEW.version_number IS NOT OLD.version_number OR NEW.source_run_id IS "
        "NOT OLD.source_run_id OR NEW.created_at IS NOT OLD.created_at OR NEW.function_count IS"
        " NOT OLD.function_count OR NEW.library_sha256 IS NOT OLD.library_sha256 OR "
        "NEW.report_sha256 IS NOT OLD.report_sha256 OR NEW.library_key IS NOT OLD.library_key "
        "OR NEW.report_key IS NOT OLD.report_key OR NEW.harness IS NOT OLD.harness OR "
        "NEW.source_harness IS NOT OLD.source_harness OR NEW.reference_data_json IS NOT "
        "OLD.reference_data_json OR NEW.publish_trigger IS NOT OLD.publish_trigger OR "
        "NEW.model_id IS NOT OLD.model_id OR NEW.benchmarks_json IS NOT OLD.benchmarks_json OR "
        "NEW.data_identity_json IS NOT OLD.data_identity_json) OR NOT EXISTS ( SELECT 1 FROM "
        "logical_libraries old_card JOIN logical_libraries new_card ON new_card.owner_id = "
        "old_card.owner_id WHERE old_card.id = OLD.library_id AND new_card.id = NEW.library_id "
        ") BEGIN SELECT RAISE(ABORT, 'library versions are immutable'); END"
    ),
}
FOREIGN_KEYS_7 = {
    "schema_info": [],
    "owners": [],
    "logical_libraries": [
        (0, 0, "owners", "owner_id", "id", "CASCADE", "CASCADE", "NONE"),
    ],
    "library_branches": [
        (0, 0, "logical_libraries", "library_id", "id", "CASCADE", "CASCADE", "NONE"),
        (0, 1, "logical_libraries", "owner_id", "owner_id", "CASCADE", "CASCADE", "NONE"),
    ],
    "library_versions": [
        (0, 0, "library_branches", "branch_id", "id", "CASCADE", "CASCADE", "NONE"),
        (0, 1, "library_branches", "library_id", "library_id", "CASCADE", "CASCADE", "NONE"),
    ],
    "local_file_snapshots": [
        (0, 0, "library_versions", "version_id", "id", "NO ACTION", "CASCADE", "NONE"),
    ],
    "local_source_activity": [],
    "report_assets": [
        (0, 0, "library_versions", "version_id", "id", "NO ACTION", "CASCADE", "NONE"),
    ],
    "library_data_identities": [
        (0, 0, "logical_libraries", "library_id", "id", "CASCADE", "CASCADE", "NONE"),
        (0, 1, "logical_libraries", "owner_id", "owner_id", "CASCADE", "CASCADE", "NONE"),
    ],
    "library_aliases": [
        (0, 0, "logical_libraries", "library_id", "id", "NO ACTION", "CASCADE", "NONE"),
    ],
    "library_branch_aliases": [
        (0, 0, "logical_libraries", "library_id", "id", "NO ACTION", "CASCADE", "NONE"),
        (1, 0, "library_branches", "branch_id", "id", "NO ACTION", "CASCADE", "NONE"),
    ],
}
SCHEMA_7_LAYOUT = (
    SCHEMA_7, {("prevent_library_version_updates", "library_versions")}, [(7,)],
    STORED_SCHEMA_7, FOREIGN_KEYS_7,
)


def schema_layout(database):
    """Typed columns and indexed columns per table, the triggers, the schema version, the
    whitespace-normalised stored DDL and each table's foreign keys."""
    with database.read() as connection:
        tables = [row["name"] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite%'")]
        layout = {}
        for table in tables:
            columns = ", ".join(f"{row['name']} {row['type']}"
                                for row in connection.execute(f"PRAGMA table_info({table})"))
            indexes = set()
            for index in connection.execute(f"PRAGMA index_list({table})").fetchall():
                names = ", ".join(row["name"] for row in connection.execute(
                    f"PRAGMA index_info({index['name']})"))
                # Constraint indexes are named by origin: 'pk' or 'u'.
                label = index["name"] if index["origin"] == "c" else index["origin"]
                indexes.add(f"{label}({names})")
            layout[table] = (columns, indexes)
        triggers = {(row["name"], row["tbl_name"]) for row in connection.execute(
            "SELECT name, tbl_name FROM sqlite_master WHERE type='trigger'")}
        version = connection.execute("SELECT version FROM schema_info").fetchall()
        stored = {row["name"]: " ".join(row["sql"].split()) for row in connection.execute(
            "SELECT name, sql FROM sqlite_master WHERE sql IS NOT NULL")}
        keys = {table: [tuple(row) for row in connection.execute(
            f"PRAGMA foreign_key_list({table})")] for table in tables}
    return layout, triggers, [tuple(row) for row in version], stored, keys


def test_fresh_database_has_schema_7_and_reopens_unchanged(tmp_path):
    """A new database gets the schema-7 layout directly; reopening it changes nothing."""
    database = Database(tmp_path / "library.db")
    database.initialize()
    assert schema_layout(database) == SCHEMA_7_LAYOUT
    append(LibraryRepository(database), data=source())
    before = table_rows(database)
    database.initialize()
    assert schema_layout(database) == SCHEMA_7_LAYOUT
    assert table_rows(database) == before


def test_concurrent_starters_share_one_new_schema(tmp_path, monkeypatch):
    """Both starters find an empty file; the one that waits for the lock changes nothing."""
    path = tmp_path / "library.db"
    ready = Barrier(2)
    connect = Database.connect

    class Connection:
        def __init__(self, connection):
            self.connection = connection

        def __getattr__(self, name):
            return getattr(self.connection, name)

        def executescript(self, script):
            ready.wait(timeout=5)  # Both have seen an empty file before either creates.
            return self.connection.executescript(script)

    monkeypatch.setattr(Database, "connect", lambda self: Connection(connect(self)))
    with ThreadPoolExecutor(max_workers=2) as workers:
        for job in [workers.submit(Database(path).initialize) for _ in range(2)]:
            job.result(timeout=10)
    monkeypatch.undo()
    assert schema_layout(Database(path)) == SCHEMA_7_LAYOUT


def test_concurrent_starters_retry_a_lost_switch_to_wal(tmp_path, monkeypatch):
    """Starters that lose the race to switch a new file to WAL retry instead of failing.

    SQLite reports a lost read-to-write lock upgrade at once ("database is locked"), without
    waiting for busy_timeout. A third connection holds the write lock that the winning
    starter holds during its switch, so both starters lose their first attempt; it is
    released only after both have.
    """
    path = tmp_path / "library.db"
    switching = sqlite3.connect(path, isolation_level=None)
    switching.execute("BEGIN IMMEDIATE")
    both_lost = Barrier(3)
    attempts = local()
    connect = Database.connect

    def counted_connect(self):
        attempts.count = getattr(attempts, "count", 0) + 1
        if attempts.count == 2:  # This starter's first attempt failed.
            both_lost.wait(timeout=10)
        return connect(self)

    monkeypatch.setattr(Database, "connect", counted_connect)
    with ThreadPoolExecutor(max_workers=2) as workers:
        jobs = [workers.submit(Database(path).initialize) for _ in range(2)]
        for job in jobs:  # A starter that gives up must not leave the barrier waiting.
            job.add_done_callback(lambda job: job.exception() and both_lost.abort())
        try:
            both_lost.wait(timeout=10)
        except BrokenBarrierError:
            pass  # A starter failed; its result below says why.
        finally:
            switching.rollback()
            switching.close()
        for job in jobs:
            job.result(timeout=10)
    monkeypatch.undo()
    assert schema_layout(Database(path)) == SCHEMA_7_LAYOUT


def saved_database(path, version, *, pending_wal=False):
    """A WAL-mode database at ``version`` (None: unversioned) that holds one card.

    With ``pending_wal`` the writes stay in the -wal, not checkpointed, as after a crash.
    """
    with closing(sqlite3.connect(path)) as connection:
        if pending_wal:
            connection.setconfig(sqlite3.SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE, True)
        connection.execute("PRAGMA journal_mode=WAL")  # As earlier servers left it.
        if version is not None:
            connection.execute(
                "CREATE TABLE schema_info(singleton INTEGER PRIMARY KEY, version INTEGER)"
            )
            connection.execute("INSERT INTO schema_info VALUES(1, ?)", (version,))
        connection.execute("CREATE TABLE logical_libraries(id TEXT PRIMARY KEY)")
        connection.execute("INSERT INTO logical_libraries VALUES('saved-card')")
        connection.commit()


@pytest.mark.parametrize("version", [None, 2, 6])
def test_older_database_is_rejected_without_changes(tmp_path, version):
    """Earlier schemas are not upgraded; startup fails before writing anything."""
    path = tmp_path / "library.db"
    saved_database(path, version)
    before = path.read_bytes()
    message = rf"schema version {version or 'none'}, which is no longer supported"
    with pytest.raises(LegacySchemaError, match=message) as error:
        Database(path).initialize()
    assert "Set the file aside" in str(error.value) and str(path) in str(error.value)
    assert path.read_bytes() == before
    assert [item.name for item in tmp_path.iterdir()] == ["library.db"]


def test_newer_database_asks_for_an_upgrade_without_changes(tmp_path):
    """A database from a newer release is kept for that release, not set aside."""
    path = tmp_path / "library.db"
    saved_database(path, 8)
    before = path.read_bytes()
    message = (r"schema version 8, which is newer than this release supports "
               r"\(schema version 7\); upgrade AutoFA to open it\.")
    with pytest.raises(NewerSchemaError, match=message) as error:
        Database(path).initialize()
    assert "Set the file aside" not in str(error.value) and str(path) in str(error.value)
    assert path.read_bytes() == before
    assert [item.name for item in tmp_path.iterdir()] == ["library.db"]


def test_rejection_leaves_pending_wal_frames_in_place(tmp_path):
    """The probe is read-only, so it never checkpoints a leftover -wal or deletes it."""
    path = tmp_path / "library.db"
    wal = tmp_path / "library.db-wal"
    saved_database(path, 6, pending_wal=True)
    before = path.read_bytes(), wal.read_bytes()
    assert before[1]  # The card and the schema version exist only as WAL frames.
    with pytest.raises(LegacySchemaError, match="schema version 6, which is no longer supported"):
        Database(path).initialize()
    assert (path.read_bytes(), wal.read_bytes()) == before


# Local hub API


@pytest.fixture
def client(local_app):
    _, browser, settings = local_app
    browser.headers.update({"Authorization": f"Bearer {settings.local_token}"})
    return browser


@pytest.fixture
def app(local_app):
    return local_app[0]


def publish(client, branch, data_identity, *, model="model-a", benchmark="bench", selectors=None):
    library = deepcopy(LIBRARY)
    library.update(model_id=model, benchmark=benchmark)
    metadata = {"branch_key": branch, "source_run_id": branch, "model_id": model,
                "benchmarks": [benchmark], "data_identity": data_identity, **(selectors or {})}
    return client.post("/api/v2/publications", data={"metadata": json.dumps(metadata)}, files={
        "library": ("library.json", json.dumps(library), "application/json"),
        "failure_card": ("report.html", "<html>saved report</html>", "text/html"),
    })


def published(client, branch, data_identity, **kwargs):
    response = publish(client, branch, data_identity, **kwargs)
    assert response.status_code == 201, response.text
    return response.json()


def csrf_token(client, library_id):
    page = Page(client.get(f"/libraries/{library_id}").text)
    return next(attrs["value"] for tag, attrs in page.tags
                if tag == "input" and attrs.get("name") == "csrf_token")


def test_explicit_merge_preserves_versions_aliases_privacy_and_future_grouping(client, app):
    first = published(client, "source-analysis", source())
    target = published(client, "target-analysis", source("data-b"), model="different-model")
    make_private(app.state.repository, first["library_id"])
    before = client.get(first["library_url"]).content
    response = client.post(f"/api/v2/libraries/{first['library_id']}/merge",
                           json={"target_library_id": target["library_id"]})
    assert response.status_code == 200, response.text
    merged = response.json()
    assert merged["id"] == target["library_id"] and merged["is_public"] is False
    assert merged["branch_count"] == merged["version_count"] == 2
    assert merged["is_merged"] is True and len(merged["data_identities"]) == 2
    assert client.get(first["library_url"]).content == before
    assert published(client, "next-analysis", source())["library_id"] == target["library_id"]
    resumed = published(client, "source-analysis", source(), selectors={
        "library_id": first["library_id"], "branch_id": first["branch_id"],
    })
    assert resumed["library_id"] == target["library_id"]
    assert resumed["branch_id"] == first["branch_id"] and resumed["version_number"] == 2


@pytest.mark.parametrize("bad_identity", [
    identity(),
    identity({"kind": "local", "sha256": "not-a-hash"}),
    identity({"kind": "local", "sha256": "a" * 64, "path": "/private"}),
    identity({"kind": "evalrdl", "origin": "https://user:secret@example.com", "run_id": "run"}),
], ids=["no_sources", "malformed_hash", "local_path", "credentials"])
def test_invalid_or_private_identity_fields_rejected(client, bad_identity):
    assert publish(client, "invalid-data", bad_identity).status_code == 422


def snapshot(client, app, publications):
    # Keep the original publication URLs as well as current canonical URLs. A
    # rejected merge must preserve both old links and the complete saved history.
    urls = {"/api/v2/libraries"}
    for publication in publications:
        original_id = publication["library_id"]
        canonical_id = app.state.repository.resolve_library_id(original_id)
        urls.update([publication["library_url"], publication["failure_card_url"]])
        for library_id in {original_id, canonical_id}:
            branches = f"/api/v2/libraries/{library_id}/branches"
            history = f"{branches}/{publication['branch_id']}/versions"
            version = f"{history}/{publication['version_id']}"
            urls.update([branches, history, version,
                         f"{version}/library.json", f"{version}/failure-card"])
    responses = {}
    for url in sorted(urls):
        response = client.get(url)
        assert response.status_code == 200, f"{url}: {response.text}"
        responses[url] = response.content
    # API counts alone would miss changed alias mappings or artifact keys.
    rows = table_rows(app.state.database, (*CARD_TABLES, "report_assets"))
    artifact_root = app.state.artifact_store.root
    artifacts = {
        str(path.relative_to(artifact_root)): path.read_bytes()
        for path in artifact_root.rglob("*") if path.is_file()
    }
    assert len(rows["library_branches"]) == 3
    assert len(rows["library_versions"]) == 4
    assert len(artifacts) == 8
    return responses, rows, artifacts


@pytest.mark.parametrize("case", ["same-card", "canonical-to-alias"])
def test_self_merge_is_rejected_by_api_and_repository_without_changes(client, app, case):
    """Self-merges, including through historical aliases, leave saved history intact."""
    first = published(client, "source-analysis", source())
    second = published(client, "source-analysis", source(), model="updated-model")
    sibling = published(client, "source-sibling", source())
    target = published(client, "target-analysis", source("data-b"))
    source_id, target_id = first["library_id"], target["library_id"]
    if case == "same-card":
        attempt = source_id, source_id
    else:
        response = client.post(f"/api/v2/libraries/{source_id}/merge",
                               json={"target_library_id": target_id})
        assert response.status_code == 200, response.text
        attempt = target_id, source_id
    publications = [first, second, sibling, target]
    before = snapshot(client, app, publications)

    response = client.post(f"/api/v2/libraries/{attempt[0]}/merge",
                           json={"target_library_id": attempt[1]})
    assert response.status_code == 409, response.text
    assert response.json() == {"detail": "select two different data cards"}
    assert snapshot(client, app, publications) == before

    with pytest.raises(ValueError, match="^select two different data cards$"):
        app.state.repository.merge_libraries(
            owner_id=1, source_library_id=attempt[0], target_library_id=attempt[1],
        )
    assert snapshot(client, app, publications) == before


# Deletion through historical aliases


def merged_history(app, client):
    """Create two obsolete IDs and a surviving card, with multiple saved versions."""
    first = published(client, "first", source("first"))
    revision = published(client, "first", source("first"))
    second = published(client, "second", source("second"))
    target = published(client, "target", source("target"))
    for old, destination in ((first, second), (second, target)):
        app.state.repository.merge_libraries(
            owner_id=1, source_library_id=old["library_id"],
            target_library_id=destination["library_id"],
        )
    return [first, revision, second, target]


@pytest.mark.parametrize("route", ["api", "browser"])
def test_stale_delete_rejected_before_artifacts_or_metadata_change(app, client, monkeypatch, route):
    """Old card IDs stay readable, but only the current ID may delete merged history."""
    versions = merged_history(app, client)
    stale_id, target_id = versions[0]["library_id"], versions[-1]["library_id"]
    repository, store = app.state.repository, app.state.artifact_store
    before = table_rows(repository.database)
    keys = list(repository.iter_artifact_keys(1, target_id))
    artifacts = {key: store.get(key) for key in keys}
    delete = Mock(wraps=store.delete)
    monkeypatch.setattr(store, "delete", delete)
    if route == "api":
        response = client.delete(f"/api/v1/libraries/{stale_id}")
    else:
        response = client.post(f"/libraries/{stale_id}/delete",
                               data={"csrf_token": csrf_token(client, target_id)},
                               follow_redirects=False)
    assert response.status_code == 409, response.text
    delete.assert_not_called()
    assert table_rows(repository.database) == before
    assert {key: store.get(key) for key in keys} == artifacts
    for version in versions:
        assert client.get(version["library_url"]).status_code == 200
        assert repository.get_version_visible(
            1, version["library_id"], version["branch_id"], version["version_id"],
        ) is not None

    # A deliberate delete addressed to the current card still removes all its history.
    response = client.delete(f"/api/v1/libraries/{target_id}")
    assert response.status_code == 204, response.text
    assert {call.args[0] for call in delete.call_args_list} == set(keys)
    assert repository.get_owned(1, target_id) is None
    for version in versions:
        assert client.get(version["library_url"]).status_code == 404


def test_repository_destructive_helpers_never_follow_aliases(app, client):
    """Internal deletion helpers require canonical IDs even without the API guard."""
    versions = merged_history(app, client)
    repository = app.state.repository
    before = table_rows(repository.database)
    for stale_id in (versions[0]["library_id"], versions[2]["library_id"]):
        assert list(repository.iter_artifact_keys(1, stale_id)) == []
        assert repository.delete_owned(1, stale_id) is False
    assert table_rows(repository.database) == before


def test_live_card_id_deletes_only_its_current_card(repository):
    """A live canonical ID stays deletable without following its moved branch aliases."""
    moved = append(repository, "moved", data=source("a"))
    kept = append(repository, "kept", data=source("a"))
    target = append(repository, "target", data=source("b"))
    card_id = kept["library_id"]
    assert moved["library_id"] == card_id
    with repository.database.transaction() as connection:
        move_branch(connection, moved["branch_id"], card_id, target["library_id"])
    moved = repository.get_version_visible(1, card_id, moved["branch_id"], moved["id"])
    assert moved["library_id"] == target["library_id"]
    assert repository.get_owned(1, card_id)["id"] == card_id
    assert set(repository.iter_artifact_keys(1, card_id)) == {
        kept["library_key"], kept["report_key"],
    }
    assert repository.delete_owned(1, card_id) is True
    assert repository.get_version_visible(1, card_id, moved["branch_id"], moved["id"]) == moved
    assert repository.get_owned(1, target["library_id"])["version_count"] == 2


# Visibility of moved branch URLs


@pytest.fixture
def moved_branch_app(local_app):
    """Branch 0 left the public card 'original' for a private card of its own.

    Schema-7 databases hold this shape where an earlier card was split by source data.
    """
    app, client, _ = local_app
    repository, store = app.state.repository, app.state.artifact_store
    target = append(repository, "branch-0", data=local_source("a"),
                    ids=("moved-card", "branch-0", "version-0"))["library_id"]
    for index in (1, 2):
        append(repository, f"branch-{index}", data=local_source("b"),
               ids=("original", f"branch-{index}", f"version-{index}"))
    with repository.database.transaction() as connection:
        connection.execute(
            "INSERT INTO library_branch_aliases VALUES('original', 'branch-0', ?)", (target,),
        )
    store.put("version-0/library", b'{"version": 1, "functions": []}')
    store.put("version-0/report", b"<html>Private moved report</html>")
    assert repository.resolve_library_id("original", branch_id="branch-0") == target
    make_private(repository, target)
    assert repository.get_visible(None, "original") is not None
    assert repository.get_visible(None, target) is None
    return app, client, target


def view_as(app, monkeypatch, viewer_id):
    # The local browser normally uses the profile owner. Vary the authenticated
    # viewer here to exercise the shared route visibility contract as on Cloud.
    context = None if viewer_id is None else AuthContext(
        identity=Identity(id=viewer_id, username=f"viewer-{viewer_id}", name="Viewer"),
        method="session", csrf_token="test-csrf",
    )
    monkeypatch.setattr(app.state.sessions, "from_request", lambda request: context)


VERSION_LIST = "/api/v2/libraries/original/branches/branch-0/versions"


@pytest.mark.parametrize("viewer_id", [None, 2, 1], ids=["anonymous", "other-owner", "owner"])
def test_moved_private_branch_alias_checks_target_visibility(
    moved_branch_app, monkeypatch, viewer_id,
):
    app, client, target = moved_branch_app
    view_as(app, monkeypatch, viewer_id)
    public_branches = client.get("/api/v2/libraries/original/branches")
    assert public_branches.status_code == 200
    assert "branch-0" not in [item["id"] for item in public_branches.json()["items"]]
    for path in (
        VERSION_LIST,
        VERSION_LIST + "/version-0",
        VERSION_LIST + "/version-0/library.json",
        VERSION_LIST + "/version-0/failure-card",
        "/libraries/original?branch_id=branch-0&version_id=version-0",
        "/libraries/original/report?branch_id=branch-0&version_id=version-0",
    ):
        response = client.get(path)
        assert response.status_code == (200 if viewer_id == 1 else 404), f"{path}: {response.text}"
    if viewer_id == 1:
        payload = client.get(VERSION_LIST).json()
        assert payload["total"] == 1
        assert payload["items"][0]["id"] == "version-0"
        assert payload["items"][0]["library_id"] == target


@pytest.mark.parametrize("viewer_id", [2, 1], ids=["other-owner", "owner"])
@pytest.mark.parametrize("listing,path", [
    ("list_branches", "/api/v2/libraries/original/branches"),
    ("list_versions", "/api/v2/libraries/original/branches/branch-1/versions"),
    ("list_versions", "/libraries/original?branch_id=branch-1"),
    ("list_branches", "/libraries/original?branch_id=branch-1"),
])
def test_concurrent_private_merge_does_not_disclose_new_listing_metadata(
    moved_branch_app, monkeypatch, viewer_id, listing, path,
):
    app, client, target = moved_branch_app
    repository = app.state.repository
    data = repository.get_version_visible(1, "original", "branch-1", "version-1")["data_identity"]
    view_as(app, monkeypatch, viewer_id)
    original_list = getattr(repository, listing)
    merged = False

    def merge_before_listing(*args, **kwargs):
        nonlocal merged
        if not merged:
            merged = True
            repository.merge_libraries(
                owner_id=1, source_library_id="original", target_library_id=target,
            )
            append(repository, "branch-1", data=data, selected=target,
                   model="private-after-merge")
        return original_list(*args, **kwargs)

    monkeypatch.setattr(repository, listing, merge_before_listing)
    response = client.get(path)
    assert merged
    assert response.status_code == (200 if viewer_id == 1 else 404), response.text
    if viewer_id != 1:
        assert "private-after-merge" not in response.text
    elif listing == "list_versions" and path.startswith("/api/"):
        assert any(item["model_id"] == "private-after-merge"
                   for item in response.json()["items"])


# Browser views


def test_browser_merge_requires_csrf_and_preserves_old_version_links(client):
    first, second = (published(client, f"analysis-{x}", local_source(x), model=f"model-{x}")
                     for x in "ab")
    source_url = f"/libraries/{first['library_id']}"
    target_url = f"/libraries/{second['library_id']}"
    old_library_url = first["library_url"]
    original = client.get(old_library_url).content
    valid_csrf = csrf_token(client, first["library_id"])
    rejected = client.post(
        source_url + "/merge",
        data={"target_library_id": second["library_id"], "csrf_token": "invalid-csrf"},
        follow_redirects=False,
    )
    assert rejected.status_code == 403, rejected.text
    assert client.get("/").text.count('class="run-card"') == 2
    assert client.get(source_url).status_code == client.get(target_url).status_code == 200
    assert client.get(old_library_url).content == original

    merged = client.post(
        source_url + "/merge",
        data={"target_library_id": second["library_id"], "csrf_token": valid_csrf},
        follow_redirects=False,
    )
    assert merged.status_code == 303, merged.text
    assert merged.headers["location"] == target_url
    assert client.get("/").text.count('class="run-card"') == 1
    retained = client.get(source_url, params={
        "branch_id": first["branch_id"], "version_id": first["version_id"],
    })
    assert retained.status_code == 200
    assert "<h1>model-a</h1>" in retained.text
    saved = client.get(old_library_url)
    assert saved.status_code == 200 and saved.content == original


# Shared hub templates


def render(template, *, identities=(), is_owner=True, candidates=()):
    environment = Environment(
        loader=FileSystemLoader(Path(autofa_library_server.__file__).parent / "templates"),
        autoescape=select_autoescape(["html"]),
    )
    environment.filters["display_time"] = lambda value: value
    environment.globals["ui_theme"] = lambda request: "light"
    user = {"id": 1, "username": "alice", "name": "Alice", "avatar_url": ""}
    library = {
        "id": "source-card", "model_id": "example-model", "benchmarks": ["example-benchmark"],
        "owner_id": 1, "owner_username": "alice", "owner_name": "Alice", "owner_avatar": "",
        "source_run_id": "analysis-run-42", "source_harness": "test", "revision": 2,
        "updated_at": "2026-10-29", "is_public": True, "owner_is_public": True,
        "function_count": 3, "branch_count": 2, "version_count": 4,
        "data_identities": list(identities), "is_merged": False,
        "data_group_count": len(identities), "data_identity_status": "known",
    }
    branch = {"id": "branch-one", "branch_key": "analysis-one"}
    version = {
        "id": "version-one", "version_number": 1, "function_count": 3,
        "created_at": "2026-10-29", "source_harness": "test", "publish_trigger": "manual_cli",
        "reference_data": [],
    }
    return environment.get_template(template).render(
        request=SimpleNamespace(url=SimpleNamespace(path="/libraries/source-card", query="")),
        me=user, csrf_token="csrf-token", libraries=[library], library=library,
        is_owner=is_owner, merge_candidates=candidates, selected_branch=branch, branches=[branch],
        selected_version=version, versions=[version],
        canonical_url="https://hub.example/libraries/source-card", report_query="",
    )


def test_merge_control_is_owner_only_explicit_and_csrf_protected():
    candidate = {
        "id": "target-card", "model_id": "different-model", "benchmarks": ["other-benchmark"],
        "data_identities": [local_source("b")], "branch_count": 2, "version_count": 7,
    }
    page = Page(render("detail.html", candidates=[candidate]))
    merge = next(attrs for tag, attrs in page.tags if attrs.get("action", "").endswith("/merge"))
    assert (merge["action"], merge["method"]) == ("/libraries/source-card/merge", "post")
    assert any(tag == "input" and attrs.get("name") == "csrf_token"
               and attrs.get("value") == "csrf-token" for tag, attrs in page.tags)
    assert any(tag == "option" and attrs.get("value") == "target-card" for tag, attrs in page.tags)
    viewer = render("detail.html", candidates=[candidate], is_owner=False)
    assert "/merge" not in viewer and "target-card" not in viewer


def test_source_identity_html_is_escaped_in_text_and_search_attributes():
    page = Page(render("index.html", identities=[source('<svg onload="bad">')]))
    assert not any(tag == "svg" or "onload" in attrs for tag, attrs in page.tags)
    assert '<svg onload="bad">' in "".join(page.text)
    card = next(attrs for tag, attrs in page.tags if attrs.get("class") == "run-card")
    assert '<svg onload="bad">' in card["data-search"]
