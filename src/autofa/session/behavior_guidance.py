# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared authoring guidance selected by the saved behavioral review policy."""

from __future__ import annotations

from pathlib import Path

from autofa.behavior_policy import strictness_for_config
from autofa.metrics.claim_contract import FAILURE_CONTRIBUTION, FAILURE_RELEVANCE

OUTCOME_EVIDENCE_GUIDANCE = """Benchmark outcomes and source-grounded review are separate.
Original benchmark labels define precision, coverage and failed-task counts; preserve them.
A PASS label alone does not refute a review or prove successful recovery, and a FAIL label
alone does not establish a mistake or causal contribution. Under the causal policy, exclude
success or recovery established by the task requirements and original source evidence, not
by substituting an external grader label for that evidence. When a review and a benchmark
label disagree, inspect the relevant source and identify whether it supports a review error,
a benchmark/source discrepancy, or unresolved evidence. Do not declare a causal-policy
violation, propose a changed verdict, or exclude an entire detector solely because a reviewed
example is labeled PASS. Keep label-based metrics and source-supported adherence distinct;
this guidance does not authorize banned outcome fields as detector inputs.
"""

SAVED_RESULT_PAGING_GUIDANCE = """### Read saved tool results without loops
The operation already ran; do not repeat it to retrieve its saved output.
Use `read_tool_result(result_id, describe=True)` to inspect paths, types and sizes.
Use `pointer="/results/0/value"` for an exact field, or omit it for the whole response.
For every page, decode a JSON-string tool response if needed and require `ok == true`.
**Stop when `next_offset` is null. Never use `complete` as the loop condition.**
The last page of a multi-page result has `complete=false` and `next_offset=null`.
Never pass that null cursor back as `offset`, or restart at zero after a read error.
Keep result_id, pointer and describe unchanged between pages. A non-null next_offset
must be an integer greater than the current offset. Stop on errors or invalid cursors.
Bound automated reads; reaching the cap is an explicit incomplete read, never success.
For JavaScript orchestration, this recipe reads JSON text (not a structural listing):
```javascript
async function readSavedJson(result_id, pointer = "", maxPages = 100) {
  let text = "", offset = 0;
  for (let n = 0; n < maxPages; n++) {
    const raw = await tools.read_tool_result({
      result_id, pointer, describe: false, offset, limit: 12000
    });
    const page = typeof raw === "string" ? JSON.parse(raw) : raw;
    if (!page || page.ok !== true) throw new Error(page?.error || "Saved-result read failed");
    if (page.offset !== offset || typeof page.text !== "string") {
      throw new Error("Unexpected result page");
    }
    text += page.text;
    if (page.next_offset === null) return JSON.parse(text);
    if (!Number.isSafeInteger(page.next_offset) || page.next_offset <= offset) {
      throw new Error("Invalid or nonadvancing paging cursor");
    }
    offset = page.next_offset;
  }
  throw Object.assign(new Error("Page limit reached; evidence remains incomplete"),
                      {result_id, pointer, offset, text});
}
```
At the cap, the error retains the assembled text and continuation offset. Inspect the size
before explicitly extending the read budget or selecting smaller fields; full evidence
remains accessible. Join all text pages before decoding their JSON. Structural reads instead
collect `children`, with offsets counting children rather than characters; use the same
null-cursor stop and error/progress guards.
`complete` means this single response contains the entire selected value or child listing.
`full_result_complete` means it contains the entire original JSON response; a complete field
or structural preview is not full evidence. Resolve `#/...` references from the original
response root, dropping `#` for pointer. Compact reviews link to originals via `full_result`.
Previews and summaries are navigation: inspect required source scope, later recovery and
contradictions, including the full trace when required.
"""

REPORT_EXPLANATION_GUIDANCE = """### Write report explanations for a first reading
The report shows `mechanism` as "Why this happens". Write for a technical reader who has
not inspected the traces or detector code. Start with the concrete action or omission.
Explain what made it mistaken or worth investigating, then connect it to the observed
effect when the evidence and selected Judge Policy support that connection.
Use ordinary words, active verbs and short sentences, with one main idea per sentence.
Name the actor, object and consequence explicitly; make references to earlier facts clear.
Explain an unfamiliar technical term where it is needed. Avoid dense noun phrases,
unexplained category names and chains of abstract terms. Give the plain-language
explanation before any useful technical label.
Write one compact paragraph, usually 3–6 short sentences and never more than six.
Use fewer sentences when they explain the issue fully. Remove repetition and unnecessary
detail rather than packing conditions, exceptions and remedies into long sentences.
Keep the conditions, relevant recovery and uncertainty that distinguish this finding.
Simpler language must not broaden the claim, invent a cause or imply stronger validation.
Include contrasts only when they help the reader recognize the mistake; avoid routine
lists of what it is not. A useful remedy may follow in a separate sentence, clearly marked
as a possible improvement rather than a demonstrated fix. Do not add generic advice.
Keep detector implementation details in `function_mechanism`, written just as clearly.
Before submission, reread the explanation without the code or traces: can a new reader
identify what happened, why it matters and what remains uncertain on the first reading?
If not, rewrite it before submitting the claim for evaluation and review.
"""

CAUSAL_CANDIDATE_CHECK = """Choose the causal hypothesis before the detector strategy. Start from an
inspected task-breaking defect and work back to the concrete decision that produced it.
Express the distinction the detector must establish. Use exact operations where the evidence
is explicit, semantic interpretation where meaning must be resolved, or a combination.
Choose the simplest implementation that preserves the claim and its coverage across tasks.
Python comparisons and state tracking can establish deep causal mechanisms; LLM use does
not establish causal validity by itself. The implementation recipes in autofa_protocol.md
demonstrate composable API operations, not failure hypotheses or required architectures.
Names such as 'hallucination', 'fabricated result', 'ignored error' or 'shell hang' identify
outcomes or search regions; they do not explain the mechanism by themselves.
Before writing or submitting a detector, record a short evidence map:
- The task requirement and its source; the specific decision or omission; the resulting
  defect or incompletion; and the evidence linking them. Record item IDs and step/span
  locations. Read whether the task requires, permits or forbids the action; mentioning an
  object or operation does not establish the requirement. The cause may occur early and
  need not be the only cause.
- What specifically went wrong in that decision: an incorrect rule, unsupported premise,
  substituted success criterion, or concrete operation that violated a requirement.
  Separate observations from hypotheses. Do not invent a mental state or keep asking
  'why' beyond what the record supports; an evidenced technical mistake can be the cause.
  State the closest competing explanation and the observation that distinguishes it.
- A narrowly targeted change that could prevent this mechanism, and why it addresses the
  evidenced causal step. This is an untested intervention proposal, not proof of causality.
  'Be more careful', 'don't hallucinate' and 'verify everything' are not diagnostic fixes.
- What happened after the incident: correction, successful completion, a lasting effect,
  or an unknown ending. Verify recovery in the relevant tool result or delivered state;
  an attempted repair, a command string or a success claim is not proof it worked.
  Distinguish the user's original bug and intermediate proposals from the applied final
  implementation. A repaired original bug cannot establish a defect in the submitted change.
  An intermediate defect with no observed completion remains unknown unless its task
  consequence is already irreversible.
- The closest available hard negative: the same surface error with recovery, an unrelated
  failure, the same action permitted by a different requirement, or a correct result despite
  doubt. Compare the proposed predicate against it.
  Seek another positive where available, but do not discard a rare mode for lacking one.
Choose the category breadth using the selected optimization preset. A specific mechanism
shares a mistaken rule/operation; a causal family may contain different technical subtypes
under a clear shared causal description. In Balanced/Recall, seek the observable operation
or omission linking defects, either directly or by generalizing supported narrow detectors.
The same causal operation can arise from different beliefs or scope decisions; require a motive
only when it is evidenced and necessary to distinguish the claimed cause. Keep supported
intent-specific subtypes beneath that family rather than excluding other supported subtypes.
An undifferentiated 'incorrect patch' or 'task failed' category does not explain a shared cause.
Each matched item still needs its own concrete
requirement → action → defect → unrecovered contribution evidence. A broad title alone,
such as 'malformed tool call', is not enough; 'Malformed tool calls prevent required actions
from executing' can qualify when this connection is evidenced in every reviewed positive.
State alternatives as alternatives, not conditions that every example must satisfy.
Retain useful supported parents when adding specific mechanisms. Use explains/depends_on
only for intentional detector composition: explains changes evaluation to within-parent
metrics. A semantic relationship alone does not require gating matches through the parent.
Distinct remedies do not invalidate a causal family; explain them in its specific children.
Do not manufacture a family by OR-ing unrelated defects without a shared causal meaning.
Keep the claim minimal: include facts necessary to distinguish the mechanism, not incidental
details such as 'late in the run', a prior failed attempt, or exact terminal echo wording
unless those facts define the cause. Do this before seeing judge labels. A frozen claim's
conditions cannot later be waived for inconvenient examples; revise and validate a new claim.
For an open causal family, define its shared operation and task consequence first, then mark
subtypes as illustrative ("for example"), not an exhaustive list of allowed implementations.
Use a closed list only when that restriction defines the intended cause. This is an authoring
decision before review; a broader revised claim always needs its own evaluation and review.
Prioritize candidates with a specific causal chain visible in detector-accessible evidence.
Use errors, closing doubts and counts to locate evidence or build unscored helpers; their
occurrence alone is not a failure mode. If decisive evidence is missing, gather it or leave
the mode open and investigate another candidate. Do not fill top-x with unsupported claims.
Probe the predicate/rubric against inspected contrasts, then call smoke_test_feature with
item_ids for those exact examples, expected_outputs for positive/negative controls, and
expected_abstentions for missing decisive evidence. Inspect each prediction and resolve
mismatches or processing errors before submit_function and corpus evaluation. Execute the
actual detector, including its bound helpers/dependencies and configured classifier;
a prose audit or random histogram cannot test its predictions. Recheck controls after a
repair, including previously supported positives so an exclusion does not silently erase
coverage. If no natural contrast exists, disclose it;
never invent a corpus example or claim that author-selected controls are independent validation.
Implement the evidence conditions as gates in the code, not only in the mechanism prose:
- Establish the actual requirement and its direction, not keyword co-occurrence. If this
  distinction needs semantic interpretation, use a semantic classifier instead of assuming
  a lexical match proves it. Missing decisive requirement evidence needs abstention.
- Return True only after establishing the specific cause and its task-breaking consequence.
- Identify the operation in the applied final code or delivered artifact before calling it
  the submitted defect; the original issue, a rejected proposal, or a similar earlier edit
  is insufficient. Observed output or a code-derived witness can establish the consequence;
  do not require an unavailable test execution when visible code and its contract suffice.
- Check whether a later repair actually succeeded before excluding the original cause.
  A command mentioning the affected file does not establish that its defect was repaired.
- For 'before this command' checks, use trace.command_prefix(item, step_index=...,
  command_quote=...) and require complete=True. Include earlier commands in the SAME
  assistant event and text before the quote in the same keystrokes field; checking only
  earlier steps misses 'cp original backup; mutate original'. The prefix records command
  text, not execution or backup success: inspect results, required contents, shell control
  flow and parallel calls before concluding that a copy protected the source.
- Use library.abstain when necessary completion or causal evidence is unavailable. Reaching
  the end of the recorded list is not itself evidence the task ended.
Before submission, challenge the code with the same action under a requirement that permits
it, missing-ending and attempted-but-failed-repair cases, as well as confirmed recovery.
Constructed variants can test implementation logic;
they are never additional corpus examples or evidence for a failure-mode claim.
The card title should name the mistaken decision or shared causal family at the selected
level of detail; the mechanism must explain the per-item causal connection and alternatives.
For example, only when
evidenced, 'Replaces a required general transformation with a lookup of visible test inputs'
is more informative than 'Fabricates results'. A permitted finite lookup or a temporary stub
later replaced is a contrast, not the same cause. This is an illustration, not a mode catalog.
"""

# The operational checks are identical; only the evidence threshold differs.
OBSERVATIONAL_CANDIDATE_BASE = """Choose an evidence-backed behavioral hypothesis before
the detector strategy.
Express the distinction the detector must establish. Use exact operations where evidence
is explicit, semantic interpretation where meaning must be resolved, or a combination.
Choose the simplest implementation that preserves the claim and its coverage across tasks.
Python comparisons and state tracking can establish meaningful distinctions; using an LLM
is not stronger evidence. Implementation recipes are composable API operations, not failure
hypotheses or required architectures.
Before submission, record supporting item IDs and source spans, the observable action or
omission, the relevant task context, and the closest competing interpretation. Separate
observations, plausible explanations and demonstrated consequences. State visibility limits;
never invent hidden beliefs, final defects, grader causes or successful recovery.
Choose a coherent pattern or family that transfers across tasks. State illustrative subtypes
as alternatives. Preserve supported narrow detectors while independently reviewing a broader
parent's new matches. A union of unrelated incidents is not a shared behavioral mechanism.
Keep claims minimal and avoid incidental timing, wording, filenames or task-specific answers.
If the claim itself asserts causality, irreversibility, absence of later recovery, or a final
outcome, verify that assertion in every mode; the selected standard never waives literal
claim conditions. Reaching the end of a stored trace alone does not prove task termination.
Inspect later actions when they can contradict the claim. Do not treat a repair command or
success assertion as verified recovery; inspect the relevant result or delivered artifact.
Do not demand missing final-outcome evidence for a claim that does not need it under the
selected standard. Abstain only when evidence needed for that claim and standard is missing.
Smoke-test the actual assembled detector with inspected positives, hard negatives and missing-
evidence controls before submission. Recheck supported positives after repairs. Constructed
variants test implementation logic; they are never additional corpus evidence or independent
validation. Keep full source evidence retrievable. Do not use outcome labels as detector inputs.
Use `trace.command_prefix(..., command_quote=...)` for command-history claims and inspect
its completeness; command text alone does not prove execution or success.
A submitted claim remains frozen during review. A revised claim needs new evaluation and
review; silently dropping inconvenient conditions cannot turn its old review into a success.
"""

PROFILE_CANDIDATE_CHECKS = {
    "behavioral": """Behavioral candidate check: identify an accurately described recurring
reasoning,
action, or tool-use pattern. Outcome association is measured separately from adherence.
A behavioral pattern can qualify without proving that it caused the task to fail. Useful
patterns can also occur in passing or recovered attempts; these affect failure precision,
not adherence, unless they contradict the actual claim. Explain why the pattern is worth
investigating, but label causal interpretations as hypotheses when not established.
Pure task identifiers, token/length counts and benchmark-result proxies are not behavioral
explanations. Meaningful recurring behavior is allowed; do not silently upgrade it into a
mandatory root-cause claim or require a final artifact defect to label its occurrence.
""",
    "contextual": """Contextual candidate check: identify an observable mistake, unsupported
decision,
or harmful shortcut with a plausible, trace-supported connection to difficulty or failure
in the task. Establish what made the behavior problematic in context, rather than inferring
it from a failed benchmark label, distress language or counts alone. For a deadline shortcut,
look for abandoned work or an unsupported simplification; deadline language alone is not it.
An observed adverse local effect or evidenced abandoned requirement can establish this
connection without tracing a complete, unrecovered chain to the final benchmark failure.
Inspect later recovery and competing explanations: exclude fully recovered incidents with
no remaining relevance and mistakes unrelated to the observed difficulty or failure. Require
trace-supported continuing relevance, but do not demand reconstruction of the precise final
artifact defect or proof that this was the sole cause. An unknown ending alone does not
establish recovery or irrelevance; decide from the context supporting the stated claim.
A failed label cannot supply missing context, and clear evidence that the alleged mistake
was valid or harmless is a negative. Abstain when the contextual connection cannot be assessed.
Preserve stronger proof when a claim explicitly asserts it.
""",
}


def candidate_check(config=None) -> str:
    mode = strictness_for_config(config)
    if mode == "causal":
        return CAUSAL_CANDIDATE_CHECK
    return PROFILE_CANDIDATE_CHECKS[mode] + "\n" + OBSERVATIONAL_CANDIDATE_BASE


def mechanism_evidence_guidance(config=None) -> str:
    """Short, policy-specific explanation requirements shared by exploration and authoring."""
    requirement = {
        "behavioral": (
            "Describe the recurring action or reasoning pattern, its observed context, and why "
            "it is useful to investigate. A pattern need not cause the benchmark failure; "
            "qualify causal interpretations as hypotheses when not established."
        ),
        "contextual": (
            "Describe the mistake or harmful shortcut and its plausible, trace-supported "
            "connection to the task's difficulty or failure. Inspect recovery and continuing "
            "relevance; a complete causal chain to the final defect is not required."
        ),
        "causal": (
            "Describe the concrete mistake, its observed task-breaking consequence, and "
            "the evidence connecting them. Inspect later recovery and alternative explanations "
            "to establish that the mistake contributed to failure."
        ),
    }[strictness_for_config(config)]
    return (requirement + " Every explicit causal or final-outcome claim still requires its "
            "full evidence under every Judge Policy. State visibility limits and distinguish "
            "possible remedies from demonstrated fixes.")


def behavior_policy_prompt(config=None, *, include_discovery_guidance: bool = True) -> str:
    mode = strictness_for_config(config)
    if mode == "causal":
        requirement = ("Mandatory judge acceptance: failure_contribution\n"
                       "The judge applies this even when every literal behavior clause is true:\n"
                       + FAILURE_CONTRIBUTION["text"])
    elif mode == "contextual":
        requirement = ("Mandatory judge acceptance: failure_relevance\n"
                       + FAILURE_RELEVANCE["text"] + "\n" + PROFILE_CANDIDATE_CHECKS[mode])
    else:
        requirement = PROFILE_CANDIDATE_CHECKS[mode]
    return (f"## Behavioral review standard: {mode}\nbehavior_strictness={mode}\n"
            + requirement + "\nEvery explicitly asserted claim condition remains mandatory. "
            "The analysis focus and precision/recall preset do not change this standard. "
            "Report adherence under this selected standard, not as proof of stronger causality. "
            "Historical reviews under another standard are not current confirmations.\n"
            + (OUTCOME_EVIDENCE_GUIDANCE if include_discovery_guidance else ""))


def render_skill(text: str, config=None) -> str:
    """Fill only named authoring slots; leave code recipes and ordinary prose intact."""
    return (text.replace("<!-- AUTOFA_BEHAVIOR_POLICY -->", behavior_policy_prompt(config))
            .replace("<!-- AUTOFA_CANDIDATE_CHECK -->", candidate_check(config))
            .replace("<!-- AUTOFA_MECHANISM_GUIDANCE -->", mechanism_evidence_guidance(config))
            .replace("<!-- AUTOFA_SAVED_RESULT_PAGING -->", SAVED_RESULT_PAGING_GUIDANCE)
            .replace("<!-- AUTOFA_REPORT_EXPLANATION -->", REPORT_EXPLANATION_GUIDANCE))


def guidance_config(config=None, run_dir: Path | None = None):
    """Resolve a saved run before rendering standalone regenerated guidance."""
    if config is not None or run_dir is None:
        return config
    from autofa.objectives import config_from_run_dir

    return config_from_run_dir(run_dir)
