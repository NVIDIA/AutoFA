# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Release gates: DCO sign-offs, license headers, dependency notices and the lazy-import closure."""

import ast
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path

from tools.check_dco import has_author_signoff, validate_pages
from tools.check_license_headers import header_errors, needs_header


def test_header_scan_includes_additions_but_excludes_worktree_deletions(tmp_path, monkeypatch):
    from tools import check_license_headers

    def files(args, **kwargs):
        return (b"retired.py\0" if "--deleted" in args
                else b"retired.py\0retained.py\0added.py\0retained.py\0")

    monkeypatch.setattr(check_license_headers.subprocess, "check_output", files)
    assert check_license_headers.working_tree_paths(tmp_path) == ["added.py", "retained.py"]


def test_dco_requires_the_author_and_a_real_trailer():
    assert has_author_signoff(
        "Fix parsing\n\nSigned-off-by: Example Author <author@example.com>\n",
        "author@example.com",
    )
    assert not has_author_signoff(
        "Fix parsing\n\nSigned-off-by: Another Author <other@example.com>\n",
        "author@example.com",
    )
    assert not has_author_signoff(
        "Explain Signed-off-by: Example Author <author@example.com>\n\n"
        "This is a description, not a certification.\n",
        "author@example.com",
    )
    assert has_author_signoff(
        "Fix\n\nSigned-off-by: Example Author <author@example.com>\n"
        "Reviewed-by: Reviewer <reviewer@example.com>\n", "AUTHOR@example.com"
    )
    assert not has_author_signoff("Fix without a sign-off\n", "author@example.com")


def test_dco_does_not_accept_a_partial_or_empty_api_response():
    commit = {
        "sha": "abcdef1234567890",
        "commit": {
            "author": {"email": "author@example.com"},
            "message": "Fix\n\nSigned-off-by: Example Author <author@example.com>\n",
        },
    }
    assert validate_pages([[commit], [commit]], 2) == []
    assert validate_pages([[commit]], 2)
    assert validate_pages([], 0)


def _commit(sha: str, email: str, message: str, account: dict | None) -> dict:
    return {"sha": sha, "author": account,
            "commit": {"author": {"email": email}, "message": message}}


def test_dco_exempts_only_commits_github_attributes_to_the_named_bot():
    # Dependabot signs off with support@github.com, not its commit author email.
    bot = _commit("b0b0b0b0b0b0b0", "49699333+dependabot[bot]@users.noreply.github.com",
                  "ci: bump\n\nSigned-off-by: dependabot[bot] <support@github.com>\n",
                  {"login": "dependabot[bot]", "type": "Bot"})
    human = _commit("c0c0c0c0c0c0c0", "maintainer@example.com", "Adjust the bump\n",
                    {"login": "maintainer", "type": "User"})
    same_login_user = _commit("d0d0d0d0d0d0d0", "someone@example.com", "Bump\n",
                              {"login": "dependabot[bot]", "type": "User"})
    unattributed = _commit("e0e0e0e0e0e0e0", "someone@example.com", "Bump\n", None)

    assert validate_pages([[bot]], 1)
    assert validate_pages([[bot]], 1, "dependabot[bot]") == []
    assert validate_pages([[bot, human]], 2, "dependabot[bot]") == [
        "c0c0c0c0c0c0: missing the commit author's DCO Signed-off-by trailer"]
    assert validate_pages([[same_login_user]], 1, "dependabot[bot]")
    assert validate_pages([[unattributed]], 1, "dependabot[bot]")


def test_header_exemptions_do_not_cover_source_or_fixtures():
    assert needs_header("tests/vendor/evalrdl/readers/tests/convert/example.py")
    assert needs_header("tests/fixtures/example.py")
    assert needs_header("src/autofa/example.py")
    assert not needs_header("docs/example.md")
    assert not needs_header(".github/CODEOWNERS")
    assert not needs_header("eval_data/inkling.yaml")
    assert not needs_header("third_party/licenses/example-1.0/AUTHORS.py")


def test_source_headers_keep_upstream_ownership_and_validate_nvidia_spelling():
    assert header_errors("src/example.py", "print('missing notices')")
    assert not header_errors(
        "src/example.py",
        "# Copyright 2026 Example Contributor\n"
        "# Licensed under the Apache License, Version 2.0 (the \"License\");\n",
    )
    assert header_errors(
        "src/example.py",
        "# SPDX-FileCopyrightText: Copyright 2026 Example Contributor\n"
        "# SPDX-License-Identifier: Apache-2.0\n",
    )
    assert header_errors(
        "src/example.py",
        "# SPDX-FileCopyrightText: Copyright 2026 NVIDIA\n"
        "# SPDX-License-Identifier: Apache-2.0\n",
    )


def test_runtime_import_dependency_closure(tmp_path, monkeypatch):
    # The old curated export omitted modules imported inside functions, so CLI
    # startup could succeed while final review/repair failed much later. Inspect
    # those imports too, without running discovery or making model requests.
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path / "profile"))
    monkeypatch.setenv("AUTOFA_SITE_CONFIG", "")
    source = Path(__file__).resolve().parents[1] / "src"
    packages = {path.name for path in source.iterdir() if (path / "__init__.py").is_file()}
    modules = {}
    missing = []

    def inspect_module(name, location):
        if name.split(".")[0] not in packages:
            return None
        if name not in modules:
            try:
                modules[name] = importlib.import_module(name)
            except ImportError as exc:
                missing.append(f"{location}: {name}: {exc}")
                modules[name] = None
        return modules[name]

    for package in ("autofa", "autofa_library_server"):
        for path in sorted((source / package).rglob("*.py")):
            relative = path.relative_to(source).with_suffix("")
            parts = relative.parts
            current_package = ".".join(parts[:-1])
            for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
                location = f"{path.relative_to(source)}:{getattr(node, 'lineno', '?')}"
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        inspect_module(alias.name, location)
                elif isinstance(node, ast.ImportFrom):
                    name = node.module or ""
                    if node.level:
                        name = importlib.util.resolve_name("." * node.level + name, current_package)
                    module = inspect_module(name, location)
                    if module is None:
                        continue
                    for alias in node.names:
                        if alias.name == "*" or hasattr(module, alias.name):
                            continue
                        # A from-import can name a package child, rather than
                        # an attribute defined by its __init__ module.
                        inspect_module(f"{name}.{alias.name}", location)

    assert not missing, "Runtime dependency missing from distribution:\n" + "\n".join(missing)


def test_dependency_notice_manifest_matches_retained_files():
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "third_party/dependency-notices.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recorded, hashed, errors = set(), set(), []

    def check(path, sha256=None):
        recorded.add(path)
        target = root / path
        if not target.is_file():
            errors.append(f"missing {path}")
        elif sha256 is not None:
            if hashlib.sha256(target.read_bytes()).hexdigest() != sha256:
                errors.append(f"hash differs: {path}")
            hashed.add(path)

    def walk(node):
        if isinstance(node, dict):
            # Native-library evidence also has wheel-relative paths; only repo copies count.
            if isinstance(node.get("path"), str) and node["path"].startswith("third_party/"):
                if not node.get("sha256"):
                    errors.append(f"no recorded sha256: {node['path']}")
                check(node["path"], node.get("sha256") or None)
            for key in ("notice_files", "upstream_license_files"):
                for path in node.get(key) or []:
                    check(path)
            if isinstance(node.get("report"), str):
                check(node["report"])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(manifest)
    retained = {path.relative_to(root).as_posix()
                for path in (root / "third_party/licenses").rglob("*") if path.is_file()}
    assert not errors, "\n".join(errors)
    # Every retained license text needs a recorded digest, however else it is listed.
    assert retained <= hashed, sorted(retained - hashed)

    # The audit section and the published notice must state the recorded wheel set.
    linux = [(package["name"], artifact) for package in manifest["packages"]
             for artifact in package.get("artifacts", [])
             if artifact["platform"].startswith("linux")]
    wheels, packages = len(linux), len({name for name, _ in linux})
    audit = manifest["linux_wheel_audit"]
    assert audit["wheel_count"] == wheels
    assert f"{wheels} wheels for {packages} platform-specific" in audit["scope"]
    notices = (root / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    assert f"**{wheels} Linux wheels for {packages} platform-specific packages**" in notices
