# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Creative exploration helpers (String Seed of Thought — arXiv:2510.21150)."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from typing import Any

from autofa.defaults import default_value
from autofa.session.behavior_guidance import mechanism_evidence_guidance

# Optional soft priors (only when config exploration_templates=true).
# Prefer concrete behavioral why + remediation — not “task is hard / kept trying.”
#
# Catalogs are scoped by corpus domain: the tool/verification templates below
# presuppose an agentic-coding run with an executable environment, and mislead on
# math/QA corpora where there is no test suite to skip and no tool to ignore.
AGENTIC_CODING_LENSES: tuple[str, ...] = (
    "never_ran_tests",
    "self_check_as_proof",
    "ignored_tool_errors",
    "claimed_done_unread",
    "skipped_verification",
    "invented_env",
    "abandoned_mid_tool",
    "shipped_happy_path",
)

# Domain-neutral: applies to reasoning / QA / math as well as agentic runs.
GENERIC_LENSES: tuple[str, ...] = (
    "answered_different_question",
    "dropped_a_constraint",
    "unstated_assumption",
    "reasoning_answer_mismatch",
    "abandoned_correct_path",
    "confident_without_support",
)

LENS_CATALOGS: dict[str, tuple[str, ...]] = {
    "agentic_coding": AGENTIC_CODING_LENSES,
    "generic": GENERIC_LENSES,
    # Unknown domain: stay neutral rather than importing SWE-agent priors.
    "unknown": GENERIC_LENSES,
}

_LENS_HINTS: dict[str, str] = {
    "never_ran_tests": (
        "never ran the official / required tests (or the command the prompt said to run) "
        "before declaring done — remediation: always execute that check and read its output"
    ),
    "self_check_as_proof": (
        "wrote its own test / local script, saw it pass, shipped on that signal while the "
        "hidden grader still fails — remediation: treat self-checks as non-proof"
    ),
    "ignored_tool_errors": (
        "tool/command returned an error (or empty result) and the model continued as if "
        "success — remediation: stop on tool failure and diagnose before the next edit"
    ),
    "claimed_done_unread": (
        "said fixed/done without reading the failing test or grader output in the trace — "
        "remediation: require reading the failure before any success claim"
    ),
    "skipped_verification": (
        "skipped an explicit verification step from the prompt (lint, typecheck, sample "
        "run, schema check) — remediation: make that step mandatory before finish"
    ),
    "invented_env": (
        "called invented APIs, files, flags, or packages that do not exist in the env — "
        "remediation: verify names against the workspace before use"
    ),
    "abandoned_mid_tool": (
        "started a tool call / long command and never consumed its result (or aborted mid "
        "turn) then moved on — remediation: always wait for and act on tool output"
    ),
    "shipped_happy_path": (
        "patched the happy path only and skipped documented edge cases the grader checks — "
        "remediation: enumerate stated edge cases and verify each"
    ),
    "answered_different_question": (
        "solved a nearby but different question than the one asked (wrong quantity, wrong "
        "entity, wrong units) — remediation: restate the asked question before answering"
    ),
    "dropped_a_constraint": (
        "satisfied most of the prompt but silently dropped an explicit constraint "
        "(format, range, exclusion) — remediation: enumerate constraints and check each"
    ),
    "unstated_assumption": (
        "introduced an assumption the prompt never granted and never flagged it — "
        "remediation: surface the assumption or resolve the ambiguity first"
    ),
    "reasoning_answer_mismatch": (
        "reached the right intermediate result in the reasoning, then emitted a final "
        "answer inconsistent with it — remediation: reconcile answer against own work"
    ),
    "abandoned_correct_path": (
        "found a correct approach, discarded it after weak counter-evidence, and "
        "committed to a worse one — remediation: keep the best candidate and compare"
    ),
    "confident_without_support": (
        "asserted a specific fact/value with no derivation or source anywhere in the "
        "trace — remediation: require the supporting step before asserting"
    ),
}


def domain_for_corpus(
    *,
    trajectory_parsing: bool,
    tool_names: Iterable[str] = (),
) -> str:
    """Classify the corpus so mechanism templates are not applied out of domain."""
    if not trajectory_parsing:
        return "generic"
    coding_markers = {
        "bash", "shell", "execute_bash", "run", "python", "ipython",
        "edit", "str_replace", "str_replace_editor", "create", "write",
        "read", "view", "grep", "glob", "patch", "apply_patch",
    }
    names = {str(t).lower() for t in tool_names}
    if names & coding_markers:
        return "agentic_coding"
    return "generic"


def resolve_exploration_seed(
    *,
    run_id: str,
    round_idx: int = 0,
    override: int | None = None,
) -> int:
    """Resolve a round in the sequence generated by a fixed or automatic main seed."""
    if override is None:
        digest = hashlib.sha256(f"{run_id}:0".encode()).hexdigest()
        override = int(digest[:8], 16)
    return round_exploration_seed(int(override), round_idx)


def round_exploration_seed(base_seed: int, sequence_index: int) -> int:
    """Derive a repeatable sequence without mutable RNG state or run-ID dependence.

    The first entry retains the main seed for existing initial scoring panels.
    Later entries diversify discovery while the main seed remains unchanged.
    """
    base_seed = int(base_seed) & 0x7FFFFFFF
    if sequence_index < 0:
        raise ValueError("Exploration sequence index must be nonnegative")
    if sequence_index == 0:
        return base_seed
    digest = hashlib.sha256(
        f"autofa:discovery-round:{base_seed}:{int(sequence_index)}".encode(),
    ).hexdigest()
    return int(digest[:8], 16) & 0x7FFFFFFF


def initialize_run_exploration(run: Any) -> None:
    """Resolve the run's current round seed from its sequence cursor, without saving."""
    run.exploration_seed = round_exploration_seed(
        run.exploration_base_seed, run.exploration_sequence_index,
    )


def ssot_seed_string(seed: int) -> str:
    """Stable task-irrelevant exploration string, independent of model sampling."""
    return hashlib.sha256(f"autofa:ssot:{int(seed)}".encode()).hexdigest()


def exploration_lens(seed: int, *, domain: str | None = None) -> str:
    """Map seed → soft mechanism-template id from the domain's catalog."""
    catalog = LENS_CATALOGS.get(domain or "unknown", GENERIC_LENSES)
    return catalog[int(seed) % len(catalog)]


def exploration_lens_hint(lens: str) -> str:
    return _LENS_HINTS.get(
        lens,
        "actionable behavioral failure with a clear remediation (not ‘task is hard’)",
    )


def ssot_prompt_block(
    *,
    exploration_seed: int,
    exploration_templates: bool | None = None,
    lens: str | None = None,
    domain: str | None = None,
    objective_config=None,
) -> str:
    """Round-start SSoT guidance; evidence drives investigations within the round."""
    if exploration_templates is None:
        exploration_templates = default_value("exploration_templates")
    if exploration_templates:
        lens = lens or exploration_lens(exploration_seed, domain=domain)
        mode = (
            f"Mode: **mechanism templates** (``exploration_templates=true``).\n"
            f"Soft mechanism template: **{lens}** ({exploration_lens_hint(lens)}).\n"
            "Use this angle only when supported by the traces; it is not a required family."
        )
    else:
        mode = (
            "Mode: **open-ended** (``exploration_templates=false``).\n"
            "Seek non-obvious, evidence-backed angles without a fixed failure catalog."
        )
    return f"""## Exploration (SSoT — String Seed of Thought)
Creative diversity is enabled (config ``exploration_ssot=true``; arXiv:2510.21150).
{mode}
External exploration_seed={exploration_seed}. The newest runtime round seed supersedes this
initial seed after advance_round.

Use SSoT **once at the start of each round**, including a new Continue-from-library round,
to choose a useful investigation focus. It is a pivot aid, not a per-candidate ritual.
The protocol and kickoff describe the same exercise. If a kickoff repeats in the same
round, retain its focus; do not repeat the string exercise.
1. Inspect the current library, list_failure_modes and the previous round's findings.
   Identify what evidence supported, contradicted or left the earlier hypotheses unresolved.
2. Manipulate the supplied task-irrelevant string ``{ssot_seed_string(exploration_seed)}``
   (for example chunk it or compute a rolling hash), briefly show the derivation, and use it
   to consider a different evidence-supported angle. Do not invent the derivation.
3. State this round's investigation focus and how the evidence motivates it. A new round
   can extend a promising mechanism or pivot from an exhausted one. Existing open modes are
   options, not mandatory first work. Overlapping detectors and useful subsets are valuable
   when they add a supported explanation, remedy or quality improvement. Renaming an
   equivalent detector adds no discovery value; task overlap alone does not make a duplicate.
4. During the round, let findings choose the next candidate, repair or investigation.
   Do not rerun SSoT for every submission or change direction solely to satisfy the seed.
   Explain the behavior under the selected Judge Policy:
   {mechanism_evidence_guidance(objective_config)}
   “The task is difficult,” pure thrash and grader noise alone are not behavioral mechanisms.
5. {_ssot_sampling_guidance(exploration_seed)}

Keep descriptions, mechanisms and content-only detectors evidence-backed. SSoT changes
where to investigate, never the review standard or the evidence required by a claim.
"""


def _ssot_sampling_guidance(exploration_seed: int) -> str:
    return (
        f"sample_items() uses this round's seed ({exploration_seed}) automatically. "
        "Advancing a round or continuing from a library takes the next seed in the fixed main "
        "seed's sequence. Do not pin an earlier round's seed. Within a round, the same seed, "
        "sampling options and candidate pool produce the same slice. Explore further within "
        "the same round with coverage filters, different peek_chunk calls, or an explicit "
        "sample seed. Do not advance merely to refresh the sample."
    )


def kickoff_ssot_lines(
    *,
    exploration_seed: int,
    exploration_templates: bool | None = None,
    lens: str | None = None,
    domain: str | None = None,
    objective_config=None,
) -> str:
    if exploration_templates is None:
        exploration_templates = default_value("exploration_templates")
    if exploration_templates:
        lens = lens or exploration_lens(exploration_seed, domain=domain)
        mode = (f"SSoT on (templates): exploration_seed={exploration_seed}, "
                f"mechanism_template={lens}. Use the soft template only if evidence supports it. ")
    else:
        mode = f"SSoT on (open-ended): exploration_seed={exploration_seed}. "
    return (
        mode + "At the start of this round, inspect the library, ledger and previous findings. "
        f"Use seed string {ssot_seed_string(exploration_seed)} once to derive an investigation "
        "focus, briefly showing the manipulation and the evidence motivating the focus. "
        "If this kickoff repeats in the same round, retain the focus instead of repeating "
        "SSoT. Within the round, follow evidence for candidate choices and pivots. Overlap "
        "and useful subsets are welcome when they add a supported explanation, remedy or "
        "quality improvement; equivalent renamings are not new discoveries. "
        + mechanism_evidence_guidance(objective_config) + " "
        + _ssot_sampling_guidance(exploration_seed)
    )


_TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text or "") if len(t) > 2}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def novelty_warning(
    *,
    description: str,
    mechanism: str,
    existing: list[dict[str, Any]],
    threshold: float = 0.72,
) -> str | None:
    """Soft warn when description/mechanism is near-duplicate of an existing function."""
    probe = _tokens(f"{description} {mechanism}")
    if len(probe) < 4:
        return None
    best_name = ""
    best = 0.0
    for fn in existing:
        other = _tokens(
            f"{fn.get('description') or ''} {fn.get('mechanism') or ''}"
        )
        score = jaccard(probe, other)
        if score > best:
            best = score
            best_name = str(fn.get("name") or "?")
    if best >= threshold and best_name:
        status = ""
        for fn in existing:
            if str(fn.get("name") or "") == best_name and fn.get("status"):
                status = f" [ledger status={fn.get('status')}]"
                break
        return (
            f"Similar wording to existing mode '{best_name}'{status} "
            f"(text similarity≈{best:.2f}). Compare the actual claims and evidence: a useful "
            "overlap or subset can add an explanation, remedy or quality improvement. "
            "If the claims are equivalent, repair/restore the existing mode rather than "
            "re-submitting a renamed twin. Let evidence guide the next investigation "
            "within this round; no new SSoT exercise is needed."
        )
    return None
