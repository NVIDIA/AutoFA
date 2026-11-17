# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Suite-wide isolation fixtures and shared doubles for harness CLIs and claim compilation."""

import pytest

from autofa.metrics.claim_compiler import compiler_policy_hash
from autofa.metrics.claim_contract import build_claim_contract


@pytest.fixture(autouse=True)
def isolated_deployment_defaults(monkeypatch):
    """Ordinary tests use public defaults, independent of the developer's deployment."""
    monkeypatch.setenv("AUTOFA_SITE_CONFIG", "")


@pytest.fixture(autouse=True)
def isolated_mcp_runtime(monkeypatch):
    """Each test starts without a live runtime, so none can pass on another test's leftovers."""
    from autofa.mcp import runtime

    monkeypatch.setattr(runtime, "_RUNTIME", None)


@pytest.fixture(autouse=True)
def isolated_capabilities(monkeypatch):
    """Each test starts and ends with an empty process-global LLM capability cache."""
    from autofa.llm import capabilities

    monkeypatch.setattr(capabilities, "_CACHE", {})
    monkeypatch.setattr(capabilities, "_DISCOVERY_LOCKS", {})


@pytest.fixture
def installed_codex(tmp_path_factory, monkeypatch):
    """Config/settings tests must not depend on the developer's installed harnesses."""
    binary = tmp_path_factory.mktemp("installed-harness") / "codex"
    binary.write_text('#!/bin/sh\n[ "$*" = "login status" ] || exit 99\nexit 0\n')
    binary.chmod(0o755)
    monkeypatch.setenv("AUTOFA_CODEX_CLI", str(binary))
    return binary


@pytest.fixture
def stub_claim_compiler(monkeypatch):
    """Compile each nonempty description and mechanism into one local clause, without a model."""
    from autofa.metrics import claude_judge

    calls = []

    def compile_claim(**kwargs):
        calls.append(kwargs)
        strictness = kwargs.get("behavior_strictness", "causal")
        clauses = [
            {"id": f"clause_{index}", "source_field": field, "quote": kwargs[field],
             "required_review_scope": "local", "predicate": kwargs[field]}
            for index, field in enumerate(("description", "mechanism"), 1)
            if kwargs[field].strip()
        ]
        return build_claim_contract(
            **{field: kwargs[field] for field in ("description", "mechanism", "source")},
            defining_clauses=clauses, behavior_strictness=strictness,
            compilation={"policy_hash": compiler_policy_hash(strictness),
                         "model": kwargs["model"], "exclusions": [],
                         "coverage_review": {"approved": True, "issues": []}},
        )

    monkeypatch.setattr(claude_judge, "compile_claim_contract", compile_claim)
    return calls
