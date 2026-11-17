---
name: discover
description: Discover top failure modes and write detectors for them from the dataset. Use when the user says /discover or wants new failure detectors.
---

# /discover

Build a library of supported failure mechanisms and informative subtypes in the bound
parquet dataset, and submit a detector for each useful discovery.
Explain observable behavioral patterns under the selected review standard. Schema-token
and length correlations alone do not establish a meaningful behavioral mechanism.

<!-- AUTOFA_BEHAVIOR_POLICY -->
Aim for **the active detector-quality target with high Coverage**, using the live grading
table and target guidance in `autofa_protocol.md`. Follow `pareto_status.target_grade` and
the live round assessment until the runtime reports a revised target. Do not infer a
different target from round or grade counts.
Keep candidates meeting the Good or Excellent requirements
as useful progress while pursuing the active target. Increase Coverage when the evidence
supports a broader mechanism, preserving precision and adherence. A larger weighted score or
hypervolume alone is not the goal. Compare breadth on the common scoring panel across
distinct tasks, preserve useful narrow seeds, and review new matches when broadening or
synthesizing a related mechanism. Missing scores require evaluation or review. The runtime
grading table supplies the requested target thresholds below; final qualification
also requires completed evidence and review.
Retain Good and Excellent detectors as valuable findings in their own right when they
identify concrete damaging decisions, add failure coverage, or suggest distinct remedies.
The active Excellent or Outstanding aspiration is not a retention threshold: assess each
candidate against its own quality requirements rather than discard it for falling below
the current target's Coverage. Do not present an unreviewed or one-task lead as qualified.
Broad families and specific mechanisms can coexist even when their matches overlap or one
is a subset of another. A deep dive into an existing Good detector can add a useful subtype,
explanation or remedy without adding new task coverage. Establish the subtype's own claim
and support across tasks; an equivalent duplicate does not add value merely by being renamed.
Follow the quality guidance in `autofa_protocol.md`: aim for supported Outstanding mechanisms
and choose the next investigation based on its potential to improve the library. Investigating
a distinct mechanism, repairing a promising candidate, and broadening supported detectors
are all productive options. Preserve the supported version while testing a broader candidate.
No fixed mix of grades is required.
Diversify across **failure mode**, **detector strategy**, and **input surface**.
Read `min_failure_modes` (default 4) in the run configuration and `pareto_status` for the
supported target: a requested minimum of 1 means **one failure detector at the active
grade or better**.
Each target detector must meet the live **active grade** threshold for whole-dataset observed
Coverage, normalized precision (`ppv_at_pi0`) and current detector adherence
(behavioral purity), with completed claim review and matched failed examples
spanning at least **five distinct source tasks**. Repeated rollouts of
one task count once. Include unreviewed matches in breadth; current judge
reviews establish detector adherence separately. Do not add reviews solely to meet the
five-task breadth floor. Review when adherence evidence is missing, stale, or uncertain.
During sampled scoring, compare the estimated full-corpus matched task support with the
five-task floor for provisional grades. Show the actual observed matches separately
and retain the estimate's uncertainty. A small sampled bucket does not establish narrow
corpus support: preserve promising candidates for planned final evaluation. If a full scan
matches fewer than five failed tasks and the runtime requests further exploration, consider
supported broadening or synthesis. Keep narrow supported seeds
as at most OK; do not repeatedly review an unchanged small bucket to reach Good. Projections
do not replace the five actual matched distinct failed tasks required for final qualification.
During discovery, sampled estimates establish provisional grades. At each completed round,
the first `advance_round` validates retained detectors on a fixed proportional 30% panel and
keeps the current round open for one focused refinement pass. The last planned round and extra
rounds use the full dataset for selected OK-or-better candidates instead. Use the score changes
and counterexamples to repair,
restore, retain, or defer those detectors before choosing new exploration directions.
During ordinary round refinement, test supplied problematic datapoints separately with
`smoke_test_feature`; keep the fixed 200-example panel unchanged, and never mix selected
regression checks into population estimates. After 100% validation, follow the dedicated
polishing phase below; its runtime-managed cumulative checks do not use the smoke-test cap.
Complete the pass with `advance_round(validation_response="what changed and what the checks showed")`.
The runtime revalidates revisions and reuses unchanged results before advancing the round.
At `goal_met`, freeze the
versions for driver-managed validation; the runtime rechecks completion after full scoring and
representative review and may return findings for further discovery. Execution errors
or unvisited rows cannot establish final qualification; honest abstentions may remain. Use
whole-dataset failure Coverage, estimated during sampled scoring; panel recall and
within-parent recall cannot replace it. Coverage below 1% and fewer than five matched distinct
failed tasks do not block adherence review; normalized precision schedules automatic review
independently of these grade thresholds. Retain useful reviewed OK detectors as findings or synthesis
seeds: they remain visible in reports without satisfying a higher-grade target.
This does not change the grade thresholds or requested target.
Previously inspected tasks can provide reviewed support. Independent confirmation on unseen
tasks is a separate property, not a requirement for the minimum. Abstentions leave coverage
unresolved without erasing confirmed positive matches. Preserve useful single-task findings
as diagnostics or synthesis seeds, but do not count them toward the minimum.
The selected round count and plateau are soft until this minimum is met (0 disables it),
subject to the runtime's configured stopping policy. **Terminate if stalled** defaults On:
the runtime can end discovery below the minimum after repeated completed investigations
without measurable library improvement. It then completes final scoring and review, saves
the library and reports the shortfall. Off preserves continued discovery while the minimum
remains unmet. Only the runtime decides that work has stalled; a refusal, final reply or
unchanged tool call does not complete an investigation or authorize stopping.
Continue with repairs or new failure families; unjudged, rejected, stale, unfinished, proxy,
and exact duplicate candidates do not fill the target. Do not weaken evidence standards to
reach it. Explicit cancellation and research time/experiment limits still stop the run;
report any shortfall honestly.
In regular discovery, **Continue until coverage plateaus** defaults Off: once the planned
round limit and the provisional minimum are reached, hand off validation without waiting for unrelated
pending reviews or synthesis. On retains the coverage-plateau and pending-work requirements.
Check `pareto_status` before starting or resuming work and after evaluation/review: when
`goal_met` is true in regular discovery, return the summary immediately. The driver validates
selected OK-or-better candidates on the full dataset and representative review cohort, then
confirms completion or returns further work.
At `ready_for_final_validation`, follow `final_validation_guidance`: run a full scan
only if explicitly requested, otherwise return the summary for driver-managed scoring.
The runtime owns round-end validation; do not duplicate its scans. Honor saved supervisor
lifecycle intent.

### Polishing after full validation

After full-dataset validation, refinement.mode='polishing'
marks a dedicated polishing phase. Its goal is to improve existing detectors to Good or better.
This phase overrides discovery's higher-grade aspiration and family exploration guidance.
Do not add new detector names, start a new family investigation, or satisfy a comparison quota.
Repair the retained detector versions using their grade_gap and qualification_gap; preserve
supported matches and the other qualifying dimensions. Missing evidence requires its specific
retrieval or an evidence-backed deferral, not an invented predicate repair.

Keep downgrade feedback compact and layered: first the changed scores, their evaluation scopes,
and the remaining Good requirements; then a few evidence groups with counts and representative
item IDs; then the general distinction worth testing. Distinguish passing matches that reduce
failure precision, reviewed counterexamples that reduce adherence, lost failed matches that
reduce coverage, insufficient distinct-task support, and unavailable or erroneous predictions.
When cohorts differ, describe the revised estimate rather than claiming a measured regression.
Use benchmark breakdowns when available to expose population differences; a benchmark label is
not a mechanism explanation. Group by a shared observed behavior or review finding only when
the evidence supports it, and label possible explanations as hypotheses. A passing example
is not itself a rejected mechanism. Prefer representatives from distinct tasks, including a
nearby supported contrast; cite review reasons when present without inventing missing reasons.
Use these cases to formulate a rule that applies beyond their IDs, wording and benchmark.
Selected cases do not establish a population ceiling or prove that an improvement is impossible.

Use the ordinary repair, submission and capped-evaluation tools. For detectors that can call
an LLM, the runtime evaluates the original fixed scoring panel (200 examples by default,
or the configured limit) together with
all cumulative problematic datapoints and retained support controls from full validation.
It deduplicates the union, retains these controls across edits, and reports checks in batches;
there is no half-panel cap during polishing. Population estimates and provisional grades use
only the original scoring panel. The larger targeted union is regression evidence, not a
representative sample: do not compute corpus coverage, precision or grades from it. Compare
added/lost matches, execution errors and unresolved evidence by group. Start with the compact
summary; retrieve relevant cases or pages when needed to resolve a specific distinction, with
the full trace scope required by the claim. A preview is not the complete set, but there is no
need to load every retained control into the conversation: the runtime checks the entire union.
Reuse unchanged cached predictions.

Reconcile known_counterexamples in polishing feedback before describing a repair as successful.
When a previously rejected case still matches, inspect the original rejected requirement and
the current claim against the full relevant evidence, including later recovery. Repair the
general distinction or give an evidence-backed explanation of why the old finding no longer
applies; use a current-claim diagnostic review when needed. Removing a smoke expectation does
not resolve the underlying finding. Report remaining counterexamples as explicit limitations,
not repaired cases; perfect exclusion is not a new grading requirement.
Historical reviews are not automatically transferred to
new code or changed claims, and these targeted checks do not alter representative adherence.

Detectors whose complete execution uses no LLM, including helpers and parents, scan the full
corpus by default. Their repair checks come from that same scan, and their population scores
use the full corpus. An explicit debug limit still restricts the scan. Shared-panel search
comparisons remain separate from full-corpus measurements. Full prediction scoring does not
request an adherence review of every example.

Check the proposed rule beyond the cases that suggested it: use existing retrieval, smoke and
review tools on task-diverse, previously uninspected cases when available, including supported
positives and nearby contrasts. Preserve reserved independent-review evidence. If such cases
are unavailable, report that limit; rechecking edited-on examples is regression evidence, not
independent confirmation. This advice adds no mandatory stage or review calls. Final full
validation remains necessary; success on these targeted checks does not establish generalization.

Finish polishing with advance_round(validation_response="repairs, checks, and remaining limits").
An extra recovery round must include a completed tested revision, new diagnostic evidence,
or a newly established evidence-backed limitation. Repeating an unchanged retain/defer
summary leaves the same round open and does not spend patience. If a remaining detector
cannot be improved from the available evidence, inspect get_function_feedback.recovery_findings
and call defer_function_repair with its finding_ids, evidence_ids and a concrete reason.
This does not require another repair allocation or a contradicted causal judgment. When all
remaining below-Good detectors have current grounded limitations, the runtime can explicitly
terminate recovery under the stall setting after the planned rounds. Do not invent repairs
or imply that a selected sample proves an unfixable population ceiling.
An ordinary evaluation finding plus a free-text deferral is insufficient: each limitation
must also have a completed relevant test or review since polishing began, or an explicit
current unavailable-source finding. The closure reports those investigations and limitations.
At phase close the runtime revalidates selected revised detectors on 100% of the dataset and
completes representative review, reusing unchanged results. Final report grades come from that
validation, not successful targeted checks. Follow the returned closing decision; do not launch
a separate full scan after every edit or resume family discovery inside this phase.


### Discovery rounds

A round is a batch of family comparisons under the round-completion rules in
`autofa_protocol.md`. Pivots and new samples can stay in the current round; successful
detectors are not required to complete the batch. Check `family_discovery(action="status")`
and resolve current hypotheses and required synthesis before `advance_round`.

Report two kinds of improvement separately: **useful discovery/quality progress** adds a
supported explanation or remedy, or repairs a detector; **breadth progress** adds confirmed
distinct failed tasks. An informative subset or overlapping mechanism can improve the library
without increasing coverage, as can a precision or adherence improvement. Neither counts
as breadth without additional tasks. Equivalent duplicates are not useful discoveries.
Use **confirmed distinct-task coverage** in `pareto_status` and
`library_coverage` (`task_coverage`). Count the union once across detectors meeting the
Good precision and adherence thresholds and confirmations on at least two tasks.
Completed capped evaluations can establish provisional search progress without a full
census; this does not satisfy final whole-corpus qualification or held-out validation.
Precision-only score improvements, duplicate children and more rollouts of already covered
tasks do not increase breadth. The runtime's existing coverage-plateau rules remain unchanged;
useful discovery/quality progress does not reset those counters. Preserve supported coverage while expanding a mechanism.
Keep narrow seeds for synthesis; independently review a broader parent's new matches.
At the end of every turn, briefly report detector grades, useful discoveries or quality
improvements, additional distinct-task coverage, regressions and their evidence, and the next concrete action toward
the active target. Use the runtime grading table; do not invent progress or change a claim merely
to chase a label.

At a round transition, use the runtime's round bulletin for one compact library-level
reflection within the existing turn:

- **Library:** report the supplied grade counts and distinguish supported claims from open leads.
- **Round progress:** summarize useful discoveries, comparable quality improvements (including
  within a grade), and breadth changes since the saved round start. Keep confirmed-task gains
  separate from matched-task coverage. If the baseline or cohort is unavailable or incompatible,
  say the change is unknown instead of estimating it.
- **Mechanism coverage:** explain what the supported claims cover, where mechanisms overlap,
  and which gaps or useful subtypes deserve investigation. Different detector names, grades or
  match sets do not establish different mechanisms. An informative subtype can add an
  explanation or remedy without adding task coverage.
- **Lessons:** carry forward contradicted, exhausted or evidence-blocked directions, with their
  claim/evidence scope and what new evidence or changed hypothesis would justify revisiting.
  A historical rejection does not reject every related mechanism; a missing fact is not a
  counterexample.
- **Next directions:** when evidence supports them, give two or three library-level exploration
  hypotheses and each one's first evidence check. Choose from unexplained failures, a
  meaningful subtype of a supported family, or a plausible shared mechanism across seeds.
  Explain the link to current coverage and lessons; do not fill a quota with invented gaps.
  Use existing tools such as `sample_items(coverage="unconfirmed_tasks")`, `library_coverage`,
  `family_discovery(action="status")`, and full evidence retrieval for the needed checks.
  No known match does not mean an evaluated negative, and unconfirmed does not mean undetected.

Keep individual detector repairs inside the round; the next-round directions describe what
to explore across the library. Briefly state the chosen direction and why. Reuse the supplied
bulletin instead of repeating it, and retrieve details only when a direction needs them.
This reflection adds no model call, review, scoring stage or round-completion prerequisite.
Follow the runtime's active target and stopping decision.

## MCP tools and evidence sources

The tools below belong to the `autofa` MCP server (`mcp__autofa__<tool>`):

- Parsed inputs and captured requests: `get_input`, `get_model_calls`, `read_parsed_field`.
- Ordered trace exploration: `get_trajectory`, `search_trajectory`,
  `get_trajectory_context`, `get_trajectory_step`, `list_trajectory_tools`.
- Complete stored or detector-visible fields: `get_field_meta`, `read_field_chunk`.
- Artifact evidence: `download_artifact`, `read_artifact`, `view_artifact`.
- Detector validation: `smoke_test_feature`, `evaluate_functions`, `judge_function`.
- Distinct coverage and repair comparison: `library_coverage`, `get_function_repair`.
- Preserved detector versions: `list_function_versions`, `restore_function_version`.
- Failure ledger and library feedback: `list_failure_modes`, `list_library_functions`,
  `get_function_feedback`.
- Adaptive family workflow: `family_discovery` (status, record_family, assess_family,
  no_coherent_family, defer_family). Top-down and bottom-up evidence survive resume.

`judge_function` starts a separate judge session with the `judge_evidence` MCP server.
That session uses `read_evidence`, `search_evidence`, `cite_evidence`, `review_status`,
and `save_review_notes`; discovery reaches that workflow through `judge_function`.

If the current session exposes additional source, trace, or artifact MCP servers,
inspect their advertised tools and use relevant read-only readers to investigate
missing evidence. Match the run, task, and repeat, and record the evidence source.
Keep predicates grounded in the bound dataset's parsed fields; externally inspected
content is not automatically available to detectors. Tools and MCP servers mentioned
in a recorded trajectory or harness skill are the evaluated agent's capabilities,
not necessarily tools available to this discovery session.

<!-- AUTOFA_SAVED_RESULT_PAGING -->

## Steps
0. **ALWAYS** `list_failure_modes`, `list_library_functions`, then `list_helpers`. The ledger tracks every
   failure mode this run — including dropped detectors (`open` / `rejected` / `redundant`).
   Consult open modes before inventing a nearby duplicate and preserve supported versions.
   Call `family_discovery(action="status")`. Choose top-down exploration or bottom-up synthesis
   from the evidence found; inspect any `synthesis_required` checkpoint. Top-down surveys
   aim for 3/6/8 distinct failed tasks for Precision/Balanced/Recall, when available. Useful
   narrow detectors can be submitted first and then generalized; no initial family survey
   is required before finding them. Record a shared behavioral hypothesis with exact source
   quotes and negative/recovery boundaries using `record_family`. For bottom-up synthesis,
   pass `seed_functions=["narrow_a", "narrow_b"]` naming existing current evaluated detectors
   with individually confirmed reviewed cases, and cite support from each seed. Include
   `scope_limit` when only one task supports the hypothesis; do not describe it as cross-task.
   Record_family validates evidence without a separate relationship judge. Submit and
   evaluate the family, read ordinary judge feedback, then `assess_family`.
   Before retaining another detector from the same family, use `boundary_notes` or
   `assessment` to explain its contribution: a different mechanism, useful subtype,
   different remedy, or distinction that resolves an observed counterexample. A confidence
   filter, wording change or alternate implementation of the same claim is normally a
   revision. Compare it with the supported version before replacing anything; do not drop
   a useful detector merely because its coverage overlaps another's.
   No separate causal screen is required. Compare up to 1/3/5 families across both directions;
   there is no separate synthesis quota. Effort counts guide search, not edit permissions.
   When supported seeds span distinct tasks and top-down work stalls, repeats repairs or
   reaches its comparison limit, inspect a bottom-up synthesis before another top-down
   comparison, new mutating repair plan or `advance_round`. Finish/defer active work first.
   Record a parent with `seed_functions`, or `no_coherent_family(seed_functions=[...])` with
   checked support and an explanation of incompatible causes. The same unchanged seed/task
   support is not repeatedly scheduled; the inspection uses existing comparison effort.
   During the runtime's closing validation-refinement pass, repair its retained candidates
   without opening another synthesis investigation. When `refinement.mode='polishing'`,
   new detector names and family investigations are prohibited. Finish that phase before
   any runtime-authorized return to exploration.
   Before advancing, finish the inspection and proposed-parent assessment or explicit
   deferral. The existing stopping decision remains authoritative. After the round's budget
   is exhausted, a checked proposal is saved for the next
   ordinary comparison; `advance_round` activates it. Explicit limits still apply.
   For additional breadth, explore uncovered tasks between comparisons. To investigate a
   subtype, inspect the supported parent's matches and relevant contrasts; additional
   coverage is not required. Repeated rollouts of one task do not establish breadth.
   Use source task identities, never attempt-ID parsing. Compare distinct
   matched tasks, current judge-confirmed tasks and marginal additional task coverage with
   `library_coverage`; keep partial observations labeled. Cross-task reviewed support needs
   confirmations on at least two distinct tasks. Retain useful one-task diagnostics with that
   scope labeled; matches alone do not establish judge confirmation.
   Preserve supported broad parents when adding subtypes and inspect lost supported matches
   before tightening. Semantic parentage is separate from `explains` / `depends_on`, which
   should be used only for intended execution dependencies. The selected behavioral review standard still applies.
   Test a parent's behavioral distinction on similar traces sharing its surface cues or unresolved
   outcome but lacking the claimed mistaken operation, including failed tasks when relevant.
   Pair those contrasts with inspected seed support and supported new variants across tasks.
   After rejection, test the contradicted claim condition with `smoke_test_feature` and
   recheck those positives. Avoid task-specific exceptions or automatically shrinking the
   family back to its seeds; use the existing evaluation and review workflow and budgets.
   Record `no_coherent_family` with checked evidence when appropriate, or `defer_family` with
   a blocker and next step. Continue bounded comparison while useful evidence remains; do not
   invent families to reach a count. Scope limits explain missing support, not fabricated data.
   Before closing an alternative mechanism as unsupported, inspect the evidence needed to
   distinguish it from competing explanations. Short excerpts, summaries and the traced
   agent's explanation are leads, not grounds to rule it out. Read the relevant requirements,
   actions, results and later recovery; inspect source artifacts or the full trace when the
   distinction requires them. Check the decisive distinction within the existing investigation
   budget, not an exhaustive catalog. Missing decisive evidence does not refute the mechanism:
   for a recorded family awaiting assessment, use `defer_family` with `blocker` and
   `next_step`; otherwise save the open lead, evidence gap and next step with
   `report_status(message=...)`. Do not invent support just to record a family.
   Initial automatic judging is development feedback. If seeking independent confirmation
   after family comparison, freeze the detector and claim before calling
   `judge_function(name, review_phase="confirmation")` on reserved evidence.
   Development feedback is not independent confirmation. If fresh confirmation is unavailable,
   retain supported development results with that limitation; independent confirmation is not
   required for Good. Gather new evidence when needed; do not repeatedly retry unchanged reviews for a score.
1. `dataset_info` — benchmarks, fail rates, `estimated_corpus_rows` / `estimated_chunks`,
   `trajectory_coverage` / `capabilities`. Call `describe_item_fields` for per-column byte costs.
2. If **exploration_ssot** is on (default): once at the start of each round, inspect the
   library, ledger and prior findings, then manipulate the supplied seed string and state
   an evidence-supported investigation focus. The protocol and kickoff describe this same
   exercise; a repeated kickoff in the same round retains the focus. Within the round,
   follow findings rather than repeating SSoT for each candidate. An open mode is an option,
   not mandatory first work. Useful overlaps and subsets are welcome; equivalent renamings
   add no value. SSoT is open-ended by default; an enabled soft template needs trace support.
   Explain the behavior under the selected Judge Policy. “Task is hard” or thrash alone is
   not a mechanism. Call `sample_items()` without a seed for the current round's examples.
   The fixed main seed generates a reproducible sequence of different round seeds; `advance_round`
   and Continue from a library take the next position automatically. Do not pin an earlier
   round's seed. Within a round, identical requests reproduce the same slice; explore further
   within the same round with coverage filters, different `peek_chunk` calls, or an explicit
   sample seed. Do not advance merely to refresh the sample.
   Call `library_coverage` to inspect distinct matched and review-confirmed failures separately.
   For additional breadth, use `sample_items(coverage="unconfirmed_tasks")`: failed rollouts
   of tasks already covered by quality-confirmed detectors are excluded together. For a useful
   subtype, inspect existing detector matches and contrasts without that exclusion.
   Discovery samples default
   to distinct tasks, while failed/successful rollouts of one prompt remain useful controls.
   Use `coverage="unmatched"` or `coverage="unconfirmed"` for attempt-level diagnosis.
   Include contrastive exploration alongside broad sampling: investigate similar
   requirements/actions with opposite outcomes, including across different source tasks.
   Use `search_items(query=..., is_fail=False/True)` and inspect candidate evidence;
   the sampler does not automatically pair similar examples. Neither method takes priority. A task's
   pass/fail outcome is not the mechanism's label: inspect the actual claim and relevant recovery.
   Same-task rollout comparisons can also explain what changed; they do not add task breadth.
   Inspect whether these cases contain an existing mechanism that its detector missed.
   Execute that detector with `smoke_test_feature` on the inspected IDs; distinguish a false
   negative from abstention or an unscored item. A failed task alone is not a positive control.
   Keep passing controls. Absence of a saved match is not proof of a missing cause.
   Prefer different tasks when available, without imposing an example quota.
   Retain useful supported parents alongside specific children. `depends_on` / `explains`
   are for intentional executable composition; `explains` gives within-parent metrics.
   A semantic relationship alone does not require gating matches through a parent.
   Count parent/child overlaps once; child within-parent recall and corpus recall differ.
3. Inspect in pieces: `sample_items` / `peek_chunk` / `corpus_trace_stats` (fail-vs-pass lift —
   use to decide WHERE to look, not as the detector). Never load the full parquet into context.
   For saved outputs, use the guarded paging recipe above. Stop on `next_offset=null`,
   not `complete`. Preserve required evidence scope. Explicit `benchmark` filters in sampling/search
   narrow the run corpus; a nonempty `benchmarks` list takes precedence over `benchmark`.
   **Agent-run corpora:** start with `get_trajectory`, search the full trace with
   `search_trajectory`, then read ordered surrounding events with `get_trajectory_context`.
   Expand full contents with `get_trajectory_step` / `read_parsed_field`. Follow paired
   tool calls/results and later recovery before concluding that a failure persisted.
   Inspect `get_trace_record(section="coverage")` before using execution metadata.
   Its events, rollout, attempts, artifacts, tool_definitions, provenance and relations
   retain recorded source details; page nested values with JSON pointers. `events` covers
   retained non-grading lanes; identify source events by `(lane, event_idx)`.
   Python detectors can use `autofa.trace.record`, `events`, `metadata`, `attempts`,
   `artifacts`, `tool_definitions`, `provenance`, `relations` and `coverage`.
   `Step.metadata` and `Step.parts` preserve event details. Check availability before
   interpreting an empty collection, and never replay model-call snapshots as actions.
   `record` also exposes exported `prompt_messages`, `prompt_summaries`, extension
   tables in `extra_resources`, and source-file `recovered_steps` when present.
   If an investigation needs missing `provenance`, `relations` or `prompt_summaries`,
   call `get_trace_record(item_id, section=..., fetch_missing=True)`. This explicitly
   fetches and caches just that resource; repeat the flag for subsequent pages.
   Supplemental results do not change the dataset or detector inputs. Do not assume
   they are available to detectors or fetch them for every rollout during preparation.
   Default reads and bulk import do not perform these optional requests.
   For artifact-based tasks, inspect reference files and produced deliverables. Use
   `read_artifact` for full located text/cells and `view_artifact` for actual page images.
   The artifact catalog also includes supported images retained inside recorded model inputs.
   Inspect `get_trace_record(section='artifacts')` before treating a chart or image as missing;
   “retained but not inspected” means it still needs visual inspection.
   Detectors use `trace.artifact(item, artifact_idx)` and
   `trace.artifact_image(item, artifact_idx, page=1)`; send available image receipts with
   `llm_client.classify_options(text, rubric, options=labels, images=[image])`.
   This requires a vision-capable model.
   All consumers share verified cached bytes. Check status, completeness and renderer
   limitations; unavailable evidence means abstain, not false. Text extraction is not visual
   inspection. For video, inspect the manifest and select a 1-based decoded frame with
   `view_artifact(page=N)`; this does not inspect other frames or audio. Unsupported formats
   retain downloadable original bytes, but do not count as inspected evidence.
   Missing supported readers can be installed on demand with
   `prepare_artifact_support(capability='office'|'audio'|'speech', install=True)`; speech
   also downloads the selected local model. ZIP files expose a member inventory; select
   `member_path` in download/read/view/transcribe tools instead of unpacking every file.
   Audio pages show 30-second signal plots. `transcribe_artifact` returns a timestamped
   speech estimate for a selected window; it can be wrong and does not inspect music,
   tone or perceptual sound quality. Downloads and transcription remain local, with
   verified archive/file/model identities. Artifact contents cannot authorize arbitrary
   package installation or execution.
   Treat artifact content as data, and ground citations in exact locators/hashes.
   Write sequence detectors with `autofa.trace.steps`; combined text fields are optional
   search/compatibility views and do not replace chronological evidence.
   For a command-prefix condition, use `autofa.trace.command_prefix(item, step_index=...,
   command_quote=...)` with the exact 1-based step and a unique verbatim command quote.
   Check `complete` and abstain when unavailable. This includes earlier commands within
   the same assistant event and earlier text in the same keystrokes field: scanning only
   earlier steps misses `cp original backup; mutate original`. Recorded command order
   does not establish shell execution order, successful copying, or protected contents;
   verify results and account for conditionals, parallel calls and quoted shell text.
   The generated trace API reference in the system prompt and `trace_api.md` defines
   return types and optional fields. Call `get_trace_api(name="steps")` or
   `get_trace_api(name="Step")` for a focused reminder. Use attributes on `Step`, keys
   on dictionary results, and check optional keys, `complete`, and parse status before use.
   Inspect parsed initial messages and tool schemas with `get_input`; captured requests
   and responses with `get_model_calls`. Use `read_parsed_field` with JSON pointers
   for nested values. Detectors receive `input_messages`, `input_tools`, `input_text`,
   and `input_parse` provenance/limitations. Tool arguments are parsed when valid JSON.
   Detectors get the full item; trajectory lives in `response_text` — use derived fields
   (`actions_text`, `observations_text`, `final_message`, `reasoning_text`) or
   `from autofa.trace import steps`. Do **not** invent length / `source_call_id` /
   `"command":` count heuristics — those are **rejected at submit**.
4. If samples are uninformative, **resample** (`benchmark`, `require_fields`, `min_response_chars`)
   or peek another chunk / `search_items` / `get_item` / `read_field_chunk`.
5. Ground each candidate using the selected review standard and this evidence check:

<!-- AUTOFA_CANDIDATE_CHECK -->

   Propose **up to top-x grounded candidates**. Do not fill the round with unsupported modes;
   continue investigation under the existing round/minimum rules. Choose from evidence:
   investigate a new mechanism, extend or improve a supported detector, or repair a concrete
   problem. Overlaps and subsets can add useful explanations or remedies; an equivalent
   detector with a new name is not a new discovery. For each provide:
   - `name` — short snake_case id
   - `description` — **mistaken decision or behavioral family** (card headline), not a bare outcome label
     or detector rule. For example, only when supported: 'Replaces a required general
     transformation with a lookup of visible test inputs', rather than 'Fabricates results'.
   - `mechanism` — **clear, evidence-backed paragraph** under the selected Judge Policy:
     <!-- AUTOFA_MECHANISM_GUIDANCE -->
     Never invent cognitive causes, an unresolved final state, or hidden-grader causality.
   - `function_mechanism` — surfaces read + strategy + when true
   - Python `source` — choose strategy deliberately (regex / trace walk / LLM / composed /
     cross-field). Content-only by default. Optional `depends_on` / `explains`.
   - `kind`: `regex` | `llm` | `python`. Use `python` for hybrid code; strategy and
     LLM use are inferred from the implementation and its dependencies.
6. Before first submission, check that the code preserves the behavioral distinction from step 5,
   including required context, relevant recovery and missing-evidence handling.
   Use semantic classification when keyword matching cannot establish the claimed distinction.
   True needs the evidence required by the claim and selected review standard;
   required unknown facts need `library.abstain`. End of the recorded list is not proof the
   task ended. Challenge the code with a changed requirement permitting the same action,
   missing-ending, confirmed-recovery and failed-repair cases. Constructed variants test logic;
   they do not add corpus evidence or validate a mode.
   Probe semantic rubrics with `llm_classify` on inspected positive/counterexample evidence.
   When semantic conditions read the same evidence, combine them into one claim-specific
   classification where possible. Add another LLM stage only for a distinct need shown by
   inspected controls; overlapping audits are
   not a default safeguard. Preserve abstention when decisive facts remain unavailable.
   Then call `smoke_test_feature(source=..., item_ids=[...], expected_outputs={...},
   expected_abstentions=[...])` on those exact examples. For example, use
   `expected_outputs={"positive-id": True, "contrast-id": False}` and place the ID of
   an insufficient-evidence case in `expected_abstentions`. These must be actual item IDs;
   do not pass placeholder IDs or call a fabricated variant a corpus example.
   Inspect each returned prediction. Fix mismatches and execution errors before submitting.
   Execute the actual detector with its bound helpers/dependencies and configured classifier;
   reasoning through the examples or testing only the rubric does not test the full pipeline.
   After repairs, recheck previously supported positives as well as the new counterexamples.
   Random histograms do not test the behavioral distinction. If no natural contrast is available,
   disclose that limit; there is no example-count quota. Author-selected controls diagnose
   the implementation and cannot establish independent validation or corpus purity.
   Smoke tests are small diagnostic checks: at most 30 rows for LLM detectors or
   500 for pure Python, with separate limits on model calls and work between quality
   improvements. Read the returned allowance and saved-result reuse counts. Use regular
   capped evaluation for larger scoring; do not split a large scan into repeated smoke
   calls to bypass its limit. Record an evidence-backed `no_coherent_family` conclusion
   when appropriate: it counts as a completed investigation, and helps the runtime decide
   when to stop. Withholding a conclusion does not preserve an exploration opportunity.
   `submit_function` with `mechanism=`, `description=`, `function_mechanism=` — **read
   `verdict` + `feedback` + `review_schedule` + `review_deferred` + `judge_feedback` +
   `example_fail_ids` before the next edit**.
   When a claim needs an externally recorded premise, inspect the original source
   before expensive corpus scoring. Optional `evidence_prerequisites` entries declare
   the required `fact`, actual support `item_id`, retained content `field` and optional
   JSON `pointer` found through MCP. At most three selected support items are checked.
   Retention does not prove the fact; a missing selected field does not imply corpus-wide
   absence. Follow prerequisite `next_action`: restore source access, provide genuine
   supporting evidence, revise the claim to what is evidenced or defer it. Rewriting
   code or sampling other items cannot repair a missing premise in an existing review.
   Default submission evaluates first, then uses `review_mode="automatic"` internally to
   schedule development review for eligible new, changed, or stale versions:
   - `ppv_at_pi0 < 0.40`: defer standard review. Inspect passing matches and repair precision,
     or specialize to a coherent, evidence-backed subtype and accept lower coverage.
   - `0.40 <= ppv_at_pi0 < 0.70`: defer automatic standard review and prioritize precision
     repair. When per-case claim evidence is needed to choose that repair, deliberately call
     `judge_function(name, review_mode="diagnostic")` for at most three examples. Diagnostic
     findings do not produce a main adherence score or confirmed support. Deliberately call
     `judge_function(name, review_mode="standard")` if validating a useful OK detector is worthwhile.
   - `ppv_at_pi0 >= 0.70`: schedule automatic standard review when current scoring provides
     usable matches and the review prerequisites are satisfied. Coverage below 1% or fewer
     than five matched distinct failed tasks does not block adherence review.
   Explainers use their existing within-parent normalized precision for scheduling.
   Coverage and task support still determine grades and the minimum target, independently
   of review eligibility. A precise narrow detector can receive an adherence score.
   Specialization is a valid precision repair even when coverage falls. Define the supported
   behavioral boundary, revise description and mechanism consistently with the new predicate,
   and check retained positives and passing controls across tasks. Never add task-ID exceptions
   or retain a broad claim for a narrowed subtype; preserve useful supported original versions.
   Choose repairs or specialization for the current limiting dimension in `grade_gap`. State that
   purpose in existing repair rationale or `report_status` arguments; distinguish the goal from
   the actual operation, such as scoring or reviewing adherence.
   Use `grade_gap.target_grade` for the runtime's current aspiration; `next_grade` is the immediate
   milestone. For a promising OK or Weak candidate, use `qualification_gap` to identify the remaining
   Good requirements from the live grading table and address those blockers before pursuing a
   higher coverage tier. Its dimension-specific actions use the same current scores and thresholds.
   For a supported claim, use `get_function_repair`'s `quality_gap` findings with `plan_function_repair`
   and `action="refine_detector"` when allowed. Preserve the claim and relationships and evaluate
   and review the changed source.
   Association counterexamples are not automatically causal contradictions.
   Lead repair feedback with the limiting scores and evaluation scope, then a few task-diverse
   representative cases and the general distinction to test. Retrieve further evidence as needed;
   do not turn selected examples into task-ID exceptions or treat a benchmark group as a mechanism.
   Check supported positives and nearby contrasts beyond the examples that suggested the edit when
   available, using the existing evaluation and review paths; state any remaining evidence limits.
   The active aspiration remains unchanged: a supported Good detector is progress toward it,
   not permission to stop exploring. Missing measurements need evaluation or evidence recovery,
   not an assumed predicate defect. Do not infer the aspiration from the public UI's Good label.
   When coverage is the limiting dimension, broaden the current detector's supported scope or
   synthesize related seeds into a parent for their shared mechanism. Compare on the same cohort
   and deduplicate matched tasks; do not OR unrelated detectors into a claimed mechanism.
   The original panel supports fair comparison; added scope needs its own evidence. Inspect newly
   included matches across distinct tasks, including passing matches and boundary counterexamples,
   and review the broadened claim through the existing review path before claiming broader support.
   Seed confirmations do not transfer to a new claim. Selected diagnostics do not estimate
   population performance; use existing capped evaluation and review paths during discovery,
   with full-corpus validation at the existing final handoff.
   Do most detector refinement within the round, including its 30% validation feedback pass.
   Before retaining a promising downgraded candidate unchanged, test an identified, feasible
   refinement with bounded diagnostics and comparable evaluation, or state the specific evidence
   gap, failed experiment or trade-off that justifies retaining or deferring it. An untested repair
   is not a failed experiment. Selected controls can reveal a distinction worth testing, but cannot
   establish a population performance ceiling or prove that no better predicate exists. Preserve
   the supported version and test other qualifying dimensions as well as the intended improvement.
   Do not postpone a known feasible refinement to final validation. The last planned round still
   automatically validates 100% of the dataset for OK-or-better candidates; treat it as
   verification, with its returned closing feedback
   available for corrections, rather than the first opportunity to improve retained candidates.
   Follow the schedule's actionable diagnosis. Do not call standard review merely because a
   candidate is unjudged or `review_deferred`; choose a repair for the actual `grade_gap` bottleneck,
   supported specialization,
   missing-evidence investigation, a deliberate diagnostic, or useful OK validation.
   Explicit `judge_function(name)` still defaults to standard review, subject to the existing
   hard prefilter and review prerequisites. Coverage is not a review gate.
   Deferral is not rejection, zero adherence, or automatic
   deletion, and does not change grades or the selected Judge Policy. Preserve frozen review
   cohorts and processing-retry recovery; scheduling must not restart them.
   Automatic full-dataset scoring and final review select OK-or-better candidates using the
   latest current checkpoint. Promising candidates missing adherence receive bounded review
   before selection; an unavailable score is not a quality rejection. Weak, No matches, and
   nonqualifying unresolved candidates keep their existing scores and history with an explicit
   selection reason; they do not trigger full scans. Reconsider them after a meaningful revision.
   Intentionally skipped candidates are not pending final work. Selected candidates that downgrade
   still receive refinement feedback and the minimum is rechecked. Explicit full-evaluation
   requests remain available; do not duplicate the runtime's automatic validation scans.
   Read `judge_prefilter`: `skip` means
   **not judged**, never zero purity. It skips no-fire census candidates or failure
   association worse than chance. Follow the returned pre-filter criteria.
   Limited/sample evaluations require conservative confidence bounds; errors or missing
   evidence cannot establish rejection. Explainers use within-parent association.
   Improve association and re-submit/re-evaluate, or remove the candidate with a reason;
   do not retry `judge_function` while the same skip applies. Specificity still needs judging.
   Unchanged current reviews are retained, including unknown/error results;
   use a grounded repair plan for follow-up instead of resubmitting to retry.
   When blocked, follow the returned `next_action` and change the unmet prerequisite before
   retrying. Do not overlap evaluation, judging or mutation of the same detector, including
   automatic work started by `submit_function`. For `operation_in_progress`, let the current
   operation finish; inspect evidence or work on other detectors meanwhile.
7. **Evidence check:** `get_item` / `get_trajectory` / `read_field_chunk` on `example_fail_ids`.
   Confirm the full claim under the selected behavioral review standard;
   inspect relevant recovery, the ending and competing explanations. Then `report_status` with the reviewed
   sample and result; these checks do not validate every detector match.
8. NO MATCHES / WEAK / OK → inspect the evidence and compare repair coverage; if dropping use
   `remove_function(reason=…, status='open'|'rejected'|'redundant')`.
   - `open` (default): mode still real, detector failed (e.g. low purity) — revisit later
   - `rejected`: mode claim was wrong
   - `redundant`: folded into another keeper
   Inspect with `get_failure_mode` / bring back with `restore_failure_mode`.
   Also `report_status` summarizing why (one sentence).
   `behavior_specificity` describes detail and is not one of the three standard objectives.
   Never reject a supported behavioral family merely for being broad or low-specificity.
   Reject vacuous outcome labels and unsupported claim conditions using the selected
   review standard. Never drop solely because `beh=—`.
   Read `behavior_validation`: `skipped_pre_filter` needs association repair or removal,
   `review_deferred` or unjudged needs the `review_schedule`'s actionable diagnosis,
   unknown needs specific missing
   evidence, and processing errors need retry/repair. Keep unresolved candidates open.
9. During iteration use `evaluate_functions(mode="capped")`.
   At `ready_for_final_validation`, follow `final_validation_guidance`: freeze keeper versions
   and either run the requested full scan once or return the summary for driver-managed
   validation. Round boundaries automatically validate 30%, or 100% on the last planned
   round and extra rounds for selected OK-or-better candidates; read the updated findings
   before continuing.
   If seeking independent confirmation, freeze the claim and request
   `judge_function(review_phase="confirmation")` on eligible reserved evidence before
   inspecting final-census feedback. Development reviews remain useful with their limitations;
   unavailable independent confirmation does not prevent Good qualification.
   Current judgments survive re-evaluation. For missing/stale reviews, follow `review_schedule`
   before choosing a deliberate review; honor a frozen repair's validation action.
   Do not repeat unchanged current reviews merely because the round ended.
   Read per-item assessments via `get_function_feedback`: repair contradicted conditions,
   investigate unknown ones, and retry processing errors. Low confirmation alone does
   not establish a claim mismatch. Any revised claim must be submitted and judged afresh.
   Inspect every acceptance condition under the selected review standard. Missing required
   evidence needs investigation or deferral; strong association does not waive a condition.
   A revised claim needs its own review; historical confirmations do not transfer.
   Compare repairs against the preserved original using `get_function_repair`: inspect lost
   supported matches, unresolved cases and the precision/purity/recall trade-off. Check
   `library_coverage` for distinct matches and individually reviewed confirmations beyond
   other detectors. A narrow high-purity child does not automatically replace a useful
   supported family. Revised claims require their own review; preserve unresolved or
   contradicted versions as research history, not validated coverage.
   After an unproductive edit or quality regression, including within the same grade, call
   `list_function_versions(name)` before another revision. Read comparisons and recommendation
   there, and retained lessons from `get_function_repair(name)`.
   Check score scope, lost supported matches and the configured trade-off;
   different cohorts do not establish a measured regression. If an earlier supported version better
   serves the evidence, first defer any active repair, then call
   `restore_function_version(name, version_hash, reason, expected_current_version_hash)` with
   the exact returned hashes and an evidence-backed reason. Restoration is an explicit choice,
   never an automatic rollback.
   Recheck retained lessons against the restored source and claim: a broader claim's counterexample
   may not apply to an earlier narrow claim, while applicable counterevidence is not erased by
   restoration. Preserve what each attempt learned about edits, contradictions, missing facts and
   execution errors. Historical lessons guide investigation; they are not new verdicts for the
   restored claim. Before another edit, inspect the suggested bounded smoke check and the old claim's
   evidence. Use `name` and `expected_version_hash` to check the exact restored version when needed.
   Continued firing needs claim-specific counterevidence inspection; nonfire only excludes that
   observed match, and abstention or error remains unresolved. Smoke checks do not establish
   causal validity. Restoring does not reset repair budgets or review exposure, run new evaluation,
   or turn historical reviews into fresh validation.
   For refinement, call `get_function_repair(name)` and use its grounded findings to call
   `plan_function_repair(name, action, rationale, finding_ids, evidence_ids)` before editing.
   Choose `refine_detector`, tighten, revise, split, gather, processing retry, or retire
   from `allowed_actions`. `refine_detector` uses returned `quality_gap` findings to improve
   a supported claim's measured quality while preserving its claim and relationships.
   A passing causal review does not mean its coverage, precision or task support cannot
   be improved. It does not turn association counterexamples into causal contradictions.
   `repair_implementation` fixes recorded detector execution errors while preserving the
   description, mechanism and relationships, even when the behavior review passed. Inspect
   its error examples, smoke-test failed cases and supported controls, then reevaluate and
   review the changed implementation. It uses the existing repair budget. For a transient
   provider fault, restore the service and retry evaluation rather than changing the claim.
   Submit split children with `repair_plan_id`; default submission evaluates and follows
   `review_schedule` for development review or a repair for the actual `grade_gap` bottleneck. An already
   frozen repair review keeps its cohort and recovery path. Freeze the repaired claim before explicitly calling
   `judge_function(name, review_phase="confirmation")` on reserved examples.
   Gathering/retry plans use `judge_function`; follow the returned validation action.
   The workflow excludes recorded discovery/repair evidence from fresh confirmation, preserving the
   original claim/results. A revised claim does not validate the old one. Processing retries
   keep prior semantic verdicts. If fresh examples or necessary facts are unavailable, call
   `defer_function_repair(name, reason)`; do not turn missing evidence into rejection.
   The repair-plan workflow allows three repair attempts per detector during discovery. Each
   runtime-assigned final polishing round grants three additional plan attempts while
   preserving the history; `get_function_repair` reports the active allowance.
   Processing retries do not consume that budget. Small fresh samples retain scores with
   limited-evidence status. Reused matches
   can support development checks; they never establish fresh confirmation of a repair.
   Retain blocked modes on the ledger.
   `mode=full` attempts every selected row after the runtime handoff. Mid-loop submission uses
   capped evaluation with the configured LLM sample budget. Before confirming completion, the
   driver performs full scoring and representative review, reusing applicable completed
   judgments and rechecking the target; do not start a duplicate census. `/save` and publication
   preserve existing results without running detectors or judges. Completed inconclusive
   reviews retain unavailable scores; a missing numeric metric is not a reason to repeat them.
10. Follow the round-completion rules in `autofa_protocol.md`: advance only after the batch
    is complete or source-evidence exhaustion is verified, with current work resolved.
    Inspect dependent mutation results and `pareto_status` before a separate `advance_round`
    call. When progress stalls and exploration is still required, pivot angle **or strategy
    or surface** within the current round. Re-check `list_failure_modes` and call `report_status`.

## Choose operations that establish the mechanism

Express the distinction the detector must establish. Use exact operations where the
evidence is explicit, semantic interpretation where meaning must be resolved, or a
combination. Choose the simplest implementation that preserves the claim and its coverage.
Python can establish behavioral distinctions through ordered actions, linked results, arithmetic,
argument comparisons and state transitions; it is not limited to lexical signatures.
An LLM rubric does not establish stronger causal validity by itself.

Use the short Python, hybrid and LLM recipes in the discovery protocol and the generated
trace API reference as composable API demonstrations, not detector templates to copy:

- **Python:** read ordered `autofa.trace.steps` / `tool_calls`, follow linked results,
  compare parsed values or track state when those facts establish the distinction.
- **Hybrid:** extract structured evidence and classify a specific unresolved semantic
  distinction. Combine the results only as the claim requires; extraction is not an
  automatic keyword prefilter and an additional LLM stage needs a distinct purpose.
- **LLM:** use `llm_client.classify_options` for a claim-specific semantic decision,
  preserving supported, contradicted and insufficient-evidence outcomes. Prototype with
  `llm_classify` on inspected evidence.

Reuse `helpers.call` or `library.fire` with declared dependencies when useful. None of
these approaches requires a particular architecture, engine quota or alternating schedule.
All must handle missing decisive evidence, inspect relevant recovery and generalize across
tasks; required unknown facts need `library.abstain`, not an assumed negative.

For fresh runs and Continue, read `pareto_status.detector_exploration` /
`advance_round.detector_exploration` as exploration context. Library counts can reveal an
unexplored implementation option, but the available evidence determines whether it is useful.
Hybrid and composed detectors count as LLM when any helper or dependency calls one.
Retain detectors for their evidence, coverage and validation quality, never to balance counts.

Optional [operation recipes](implementation_recipes.md) demonstrate additional combinations:

| Operation | Building blocks |
|---|---|
| Compare requirements with actions | Tool definitions, recorded arguments and linked results |
| Track state and later recovery | Ordered steps and explicit state updates |
| Inspect the resulting artifact | Artifact inventory/content, exact checks or semantic interpretation |
| Reuse existing logic | Helper discovery, `helpers.call` and `library.fire` with declared dependencies |

Read the relevant section when useful. These fragments are optional building blocks, not
failure-mode templates, required stages or a quota for implementation types or helper calls.

## Surfaces
prompt · reasoning/CoT · tool calls/args · observations · final response

## Mechanism quality

<!-- AUTOFA_REPORT_EXPLANATION -->

**Good** (only if labeled fails actually show this)
- description: punchy failure-mode title
- mechanism: requirement → mistaken decision/omission → specific defect and linked task consequence,
  with recovery checked and evidence limits stated
- function_mechanism: surfaces + strategy + predicate (not "scans response_text / CoT")

**Bad (rejected or PROXY)**
- `rt.count("source_call_id") >= 110` / `"command":` counts / bare `len(rt)`
- Absence of `"name":"finish"` alone as a mode
- Mechanism that only restates the regex
- Unbacked story vs `example_fail_ids`
- Recovered parse errors followed by an unrelated wrong artifact
- Generic doubt or submission without a check, without a link to the actual defect
- A detector that matches only the symptom while its prose supplies an unsupported cause

## LLM detectors
Use the configured **OpenAI-compatible endpoint** and `feature_model` at eval time.
The default model for new configurations is `openai/openai/gpt-6-luna` through NVIDIA.
`dataset_info.llm_detectors` and startup guidance report the active model, label API,
and readiness. Use `classify_options(text, rubric, options=[...])` for fixed labels;
use `classify(prompt, text)` for free text/JSON. Never hardcode provider IDs or keys.
Check `getattr(llm_client, "available", True)`. Prefer derived surfaces over envelope slices.

An LLM rubric is one way to detect meaning, intent, contradiction, or recovery;
deterministic trace logic and precise textual rules remain valid alternatives.
Use the current library's exploration guidance to choose what to try next and validate
positive and negative examples. Report endpoint failures and retain blocked candidates
for retry; an unavailable endpoint is not evidence that the failure mode is absent.
For missed causes and false fires, distinguish extraction failures from rubric errors,
incorrect final-state/recovery interpretation, and unsupported claims. Preserve the task
requirement, applied operation and decisive later evidence in classifier input. More clauses
do not establish improvement: avoid incidental motive requirements and task-specific
exclusion lists, and compare actual pipeline outputs and retained coverage after revisions.

**API routing.** With `feature_api = "auto"` (default), the runtime checks the configured
endpoint and model and prefers `/decisions` for `classify_options`. Use this helper for
every finite-label verdict, including yes/no. Keep population scoring to decisions and
deterministic checks. Do not request generated quotes, rationales or JSON explanations
merely to justify each verdict. Several fixed-choice fields can be separate typed
decision questions over shared evidence, without generating a JSON explanation.
Optional supporting passages are retrieved on reviewer request or during final report
preparation; detector code need not generate report highlights. For evidence already
present as candidate spans, prefer deterministic extraction or selection among those
spans. Free-text extraction is still necessary when its output actually determines a
verdict (for example, Python verifies an extracted numerical claim). Do not simply skip
that check or move it after classification: revise and revalidate the detector first.
Necessary generated text and improvement explanations use `/responses` when supported. Explicit Responses or
Chat Completions settings are respected; startup readiness reports the verified routes.
Only use explicit `llm_client.decide(questions, text, focus=...)` when Decision access
is configured and ready. It returns ordered answers for these question objects:

- Predicate: `{"type": "predicate", "instructions": "Evaluate the claim-specific condition using the supplied evidence..."}` → `probability`.
- Choice: `{"type": "choice", "instructions": "Apply the rubric...", "choices": [{"value": "fires"}, {"value": "other"}]}` → `choice`, `confidence`, `probabilities`.
- Score: `{"type": "score", "instructions": "Rate the evidence...", "levels": [{"label": "absent"}, {"label": "present"}]}` → `score`, `confidence`, `probabilities`.

For each raw answer, check `answer["type"] == "refusal"` and abstain (`"unknown"`)
before reading its value. State and validate any probability/score threshold in
the detector source. Use `kind="llm"` for LLM detectors and `kind="python"` for hybrid
implementations; LLM execution and sampling still follow the code's actual calls. Classification errors,
incomplete answers and refusals must never become negative detections. The helpers
discard separate reasoning and leading marked reasoning blocks; do not parse
labels by substring or assume a model returns no reasoning.

Execution metadata (tool status, timings, token usage, finish reasons, retry records)
is legitimate evidence when tied to an observed mechanism. It is not a target label.
Never use task correctness, reward or grader verdicts as detector inputs. Reject
"the task failed because it failed", tool failure alone, or long-input/high-token/slow-run
thresholds alone as explanations. Read the action, result and later recovery; compare
counterexamples with similar lengths or difficulty where possible. A useful mode explains
what the agent did beyond those correlates. Preserve the data and enforce this distinction
in the detector claim, implementation and evidence review.

**Evidence windows — deliberate, not a fixed 4k budget.**
- Retain the relevant task requirement, decision/action, defect and recovery/ending evidence.
  Closing reasoning alone cannot establish what is wrong with an artifact or why it failed.
  If the needed context is unavailable or clipped, abstain or gather it before classification.
- Do **not** write `text[-4000:]` / `text[:4000]` (rejected at submit).
- Pass the full surface into `llm_client.classify_options(text, rubric, options=labels, focus=...)`.
  - `focus="tail"` / `"head"` / `"ends"` / `"full"` / `"auto"` (`auto` = ends for prose,
    tail for raw JSON blobs).
- Or `from autofa.trace import evidence_text` then
  `evidence_text(item, surface="reasoning", focus="closing")`.
  - `closing` / `opening` use last/first agent steps when a trajectory exists;
    on non-agentic corpora (no steps) they fall back to a char tail/head.
- For already selected evidence, use `classify_options(..., focus="full", max_chars=len(text))`
  to preserve the supplied text. `full` alone still applies a default character limit;
  matching the text length does not expand endpoint capacity or recover omitted evidence.
- `text_window_evidence` returns exact source fragments and separate analyzer omission
  metadata. The runtime also preserves legacy `text_window` provenance through ordinary
  string/JSON packing within a detector call. Analyzer omissions are never original-code
  defects. For whole-reply completeness or absence claims requiring all supplied text, use
  `classify_options(text, rubric, ..., require_complete=True)`; known window omissions
  abstain before dispatch. This checks windowing, not original-source availability.
  Excerpts may still support other claims; their classifier has an insufficient-evidence
  answer that abstains rather than silently returning a negative.
- Inspect `evidence_warnings` from smoke/submission. Returning `False` for an evidence-budget
  limit conceals unresolved items; retrieve enough evidence or call `library.abstain`.
  A real semantic length condition remains valid. Test an unresolved over-budget example
  with `expected_abstentions`, plus a complete long-reply contrast.

**Derive the semantic rubric from the claim.** Ask:

- What observable evidence establishes this claim under the selected review standard?
- What similar-looking evidence contradicts it or establishes relevant recovery?
- What missing evidence makes the decision insufficient?

Choose the conditions and output format for that distinction, rather than copying a fixed
checklist. Apply the selected behavioral review standard and every explicit claim condition:
a success claim or repair attempt alone does not establish recovery, and missing decisive
evidence remains unresolved. With `classify_options`, choose explicit labels,
compare them exactly and map insufficient evidence to `library.abstain(reason)`, never `False`
or a truthy string. Additional conditions should distinguish the cause, not exclude otherwise
supported variants merely because they differ from the seed.

Use current findings to choose the repair: parsing/linking failures need implementation
repairs; missing termination semantics, tool contracts or decisive source evidence need
retrieval or a newly validated supported claim. Switching detector engines cannot supply
absent facts. Unread evidence and reviewer processing failures need recovery within existing
scope and budgets. Actual counterexamples justify revising the predicate or mechanism.
For evidenced semantic false fires, refine the rubric against those examples, probe with
`llm_classify` and recheck supported positives before re-submitting.
When false positives arise from unrelated or mixed evidence, consider refining the existing
LLM detector into a hybrid: use regex or Python to locate candidate actions, context, linked
results and relevant counterevidence/recovery, then classify that evidence. Preserve access
to the full source and broaden retrieval when required facts are unresolved. Compare the
original and refined detector on the same inspected positives and hard negatives across
distinct tasks through the existing smoke/repair workflow; check precision gains and lost
supported variants. Narrower input is not automatically more precise, and a convenient
keyword is not automatically a necessary condition.
An uncertain judgment instead needs its missing evidence; a processing error needs
repair/retry. Do not tighten the predicate merely because confirmation is incomplete.
Do not only change `focus`.

LLM detectors may use one or more API calls per row. Mid-loop capped evaluation uses the
configured stratified sample (`llm_eval_sample`) with bounded workers and cached
unchanged outputs. New panels balance failed and successful rollouts within the same
budget, preserving benchmark strata and weighting estimates back to the corpus. If one
outcome has too few attempts, the other fills the remaining places. Panel identity stays
fixed across revisions; changes of allocation or cohort cannot establish a score trend.
At the runtime's final-validation handoff or driver-managed final scoring,
`mode="full"` attempts every selected row with versions and claims frozen. Saving and
publishing preserve sample labels and unavailable metrics.

- Report sampled estimates with their interval and sample size.
- The existing hard prefilter's sampled low-coverage skip requires its upper confidence bound
  to be below that floor. Automatic review scheduling is separate: uncertain or insufficient
  precision, coverage or task support can defer review without rejecting the mechanism.
  A small sample with no failed matches is inconclusive.
- If compatible positive smoke controls exist outside a panel with no failed matches,
  up to three distinct failed tasks can receive a separate diagnostic review. Inspect its
  findings to establish whether the mechanism is supported; selected controls do not
  change population estimates, the main adherence score or confirmed support.
- Scoring completion and review completion are separate. Use available association
  metrics while resolving the specific missing review or evidence prerequisite.
- Adherence describes the reviewed matches, not a population-weighted estimate.
  Task-diverse and adaptive reviews do not have the scoring panel's inclusion probabilities.
- Do not rank a gap smaller than its uncertainty as a measured gain.
- Deliberately increase the sample budget when needed; do not reroll for a favorable result.
- Association uncertainty is Wilson recall/FPR CIs and Fisher's exact `fisher_p`.
- Deterministic detectors score selected rows within the configured evaluation limit.

Never read `is_correct` / `accuracy` inside detectors.

## Reusable helpers

`autofa.trace.command_prefix` illustrates an ordered extraction helper: given an exact
step and command quote, it returns earlier command fragments with `step_index`, `path`
and `text`, plus completeness information. Check `complete` before interpreting an empty
prefix. Recorded command order does not prove execution or success, and this view does
not establish the absence of unrecorded or non-command actions. Use the trace API reference
for the call signature and the discovery protocol for composable implementation recipes.

Search `list_helpers` before writing detector code. Use `get_helper` for contracts and exact
source. Create ordinary extraction, normalization, parsing, or transformation functions with
`submit_helper`; they need no failure claim, dataset score, judge, or promotion. Functional
checks through `test_helper` return values/errors only.

Declare `uses_helpers={"alias": "name_or_revision"}` in `submit_function` or `submit_helper`;
call `helpers.call("alias", ...)`. Existing detectors can also be called through
`library.fire` with `depends_on`. All execution dependencies are pinned in the export.

For semantic detection, reuse extraction helpers when they preserve the evidence needed for
the claim. Skip semantic classification after an empty extraction only if that absence
reliably rules out the mechanism and the search is complete; a convenient keyword is not
automatically a necessary condition. Otherwise broaden retrieval or preserve uncertainty.
Retain access to the original evidence, including the requirement, linked consequence, later
recovery and exact source locations even when the literal claim only describes an incident.
Missing or incomplete evidence should cause `library.abstain(reason)`, never the truthy label `unknown`.
`regex_passages` / `render_passages` expose completeness explicitly. Pass the rendered
`budget_chars` to `classify_options(..., focus="full", max_chars=...)`; full focus still clips
at its character budget (16,000 by default). The runtime's final-validation handoff or final
validation before completion evaluates selected OK-or-better detectors on the full dataset; helpers are never ranked
or counted as failure modes.

For long ordered executions, use `autofa.trace.step_evidence(item, pattern, ...)` to scan all
steps and select matching steps, context, linked calls/results and closing evidence. Inspect
`selection_complete`, `full_trace`, omitted/truncated step indices and source offsets; read
the original steps when needed. Selected neighborhoods do not establish absence of later
repair: inspect relevant later edits/results or a decisive submitted state. Global absence
claims require the whole relevant history. Do not return empty evidence solely because the
full serialized trace exceeds a budget. Test extraction on long traces and successful or
repaired controls, not only the initial seed.

Use `retain_as_helper` to salvage useful deterministic logic from an active or dropped
detector, while preserving its original claim history. Extract duplicated nested functions
when they serve multiple detectors. Creating helpers directly is equally valid.


## Coverage and final validation

Standard presets rank precision, `task_recall` and behavioral purity. Task recall counts distinct
source-verified failed tasks with a match once, across the selected corpus; repeat rollouts
cannot inflate it. Attempt-level `recall` remains separate and retains its existing 1% final
eligibility floor. Explicit historical objectives retain their saved axes. For sampled,
capped or unresolved evaluations, task recall is an observed lower bound, not an estimate
of unvisited task matches. Read its scope and counts, and do not infer misses from unvisited
tasks or confuse observed matches with confirmed reviewed support.

Read each detector's `verdict` for its current quality label and reason; it summarizes
measurements, not independent confirmation. Use `discovery_feedback` to choose concrete
investigations after scoring: inspect boundaries
for low precision, claim conditions and recovery for low purity, and missed variants or related
seed synthesis for high-quality narrow breadth. Per-case findings, missing evidence and
existing repair constraints take precedence. This advice adds no gate, quota or review stage.
Choose the next investigation toward the active target from the evidence: investigate a
distinct mechanism, repair a concrete quality issue, or test a supported broader variant.
Overlapping and subset detectors remain useful when they add supported understanding;
report that separately from additional task coverage. Use `search_score` and
`pareto_status.search_ranking` as diagnostics of the configured objective and common-panel
task breadth; their weighted calculations remain unchanged. They can use current
limited/development claim reviews, label that
evidence provisional, and keep unknown cases in the purity denominator. Read evidence counts
and limitations. Validated ranking, final keeper eligibility and confirmed-task progress
remain separate; an empty validated frontier does not erase useful search evidence.
Iterative task breadth uses the same fixed scoring panel for sampled and full-scan detectors.
Check panel identity and availability before comparing candidates; missing or incompatible
panels are not comparable. Corpus task recall remains separate, including genuine matches
outside the panel. Low observed corpus coverage from a sample is not measured narrowness.
Use provisional panel search_ranking during discovery; complete corpus measurements for the
validated task-coverage frontier belong to final validation, not extra mid-loop scans.

Discovery can proceed top-down or bottom-up; choose from the evidence found. If useful
high-precision, review-supported narrow detectors arrive first and broad coverage is weak,
retain them and inspect their actual supported cases for a shared behavioral distinction
under the selected review standard. Repeated rollouts of one task, names and surface tokens
do not establish a common mechanism. Substituting a canned answer for required computation
and discarding that computation's output may support a computation-bypass family when the
shared claim is evidenced; the parent should recognize other variants of that behavior.

Use `record_family` with `seed_functions` to preserve the derivation, not to create executable
dependencies. Write an independently evaluated parent; do not set `explains` / `depends_on`
to its children, which would gate new matches through the narrow seeds. Reuse helpers where
useful. Do not merely OR unrelated bugs and call that generalization. If no coherent shared
cause emerges, keep the useful detectors and their union coverage, then explore another angle.
Keep supported seeds while testing the parent; their precision and purity do not transfer.
Smoke-test seed positives and contrasts, then compare the parent to the union of seeds on
comparable scored examples. Inspect preserved support, seed matches absent from the parent,
and additional failed tasks; partial observations cannot establish lost recall. Review newly found cases through the ordinary judge before claiming generalization.
A parent covering only seeds may organize them but has not demonstrated additional recall.
The existing parent review budget should cover new failed tasks, retained seed support and
passing controls when available. Check the recorded sample and confirmed additional tasks;
reviewing only seed examples cannot validate the added coverage.

Improve library coverage across independent tasks.
A task's repeated rollouts are not separate tasks. Use source task identities, not
attempt-ID parsing or invented IDs. Cross-task reviewed support needs current ordinary judge
confirmations on at least two distinct tasks; keep useful single-task diagnostics labeled.
`sample_items` defaults to distinct tasks within each outcome group. Failed and successful
rollouts of the same prompt are useful controls for what changed and whether recovery worked.
Use `distinct_tasks=False` deliberately for repeated-rollout comparisons; they remain one
task, never additional examples of generalization. A detector may fire on multiple repeats,
but its behavioral rule must transfer across different prompts/tasks. After a useful narrow seed,
look for the same mechanism on another failed task instead of cataloguing more bugs in one
prompt. Generalize requirements, actions and consequences; do not use task names, literal
answers, filenames or repeat identities as the failure rule.
Use library_coverage's fixed selected-corpus failure denominator, distinct matched and
confirmed tasks, marginal additional tasks and unresolved/error counts. Count overlapping
detectors once. Matching multiple tasks alone is not judge confirmation. Preserve supported
broad parents while adding children, and inspect lost supported matches before tightening.

Use sample_items(coverage="unconfirmed_tasks") for additional confirmed task coverage;
unmatched/unconfirmed remain attempt-level diagnostic filters. Check whether an
existing mechanism was missed by its detector, evidence/execution is incomplete, or another
family is supported. Seek support on another task before describing a finding as cross-task.
Smoke-test known positive and contrast controls; uncertainty or missing execution is not a
confirmed negative. These investigations guide search without becoming claim judgments.

Iterate with evaluate_functions(mode="capped"). LLM detectors use the configured fixed
stratified sample (llm_eval_sample). Detectors with no LLM calls in their execution, helpers
or parents scan the full corpus by default; an explicit debug limit still caps the scan.
Search comparisons use the same fixed panel for both kinds. Reuse unchanged cached outputs.
Read sampled labels, sample sizes and confidence intervals. Deliberately expand the sample
budget when needed rather than rerolling for a favorable result. At `ready_for_final_validation`,
follow `final_validation_guidance`: freeze the listed versions and run full evaluation only
if requested, otherwise return the summary for driver-managed scoring. Full evaluation
attempts every selected row. The runtime owns round-end validation; do not duplicate its scans.
Standard detector adherence reviews use 10% of flagged examples, rounded up, with a minimum of
three and capped by available eligible matches. During iterative evaluation this uses
observed flagged examples on the scoring panel; final review uses full-dataset matches.
Family comparison does not impose a separate standard-review cap. Explicit diagnostic reviews
remain capped at three examples and do not produce adherence or confirmed support.
Detector execution and sampled
claim review are separate. If seeking independent confirmation, request eligible reserved
review after freezing and before inspecting final-census feedback. Its unavailability does
not invalidate supported development evidence or prevent Good qualification under the normal
grade requirements. There is no new upfront dataset split. Previously exposed
tasks or missing review history do not become independent evidence on resume.
