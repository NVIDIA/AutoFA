# Detector metrics and feedback

[Back to README](../README.md) · [Detector reference](detectors.md) · [Repairing detectors](detectors.md#repairing-a-judged-detector)

## Feedback metrics (what “good” means)

When AutoFA scores a detector, pay attention to these:

| Metric | Meaning |
|--------|---------|
| **Flagged** (`pct_caught`) | Share of *all* examples the detector fires on |
| **Precision** | Among fires, share of benchmark failures; this is association, not claim confirmation |
| **Coverage** (`recall`) | Share of failed rollouts it catches; each rollout counts separately, including repeats of a task (attempt-level recall) |
| **task_recall** | Distinct failed tasks with an observed match / all distinct failed tasks in the selected corpus; each task counts once |
| **Normalized precision** (`ppv@π₀`) | Precision standardized at fail rate π₀=0.30 (fairer across corpora) |
| **Detector adherence** (`behavior_purity`) | Reviewed support for the full claim under the selected judge policy |

Adherence review has no minimum **Coverage** or distinct-task count. A precise narrow
detector can receive a review even below 1% Coverage or five matched failed tasks.
Final reports retain scored detectors at their actual grade. CLI **Flagged**
instead measures fires across all examples.
HTML reports show standard attempt-level recall as Coverage; distinct-task recall remains
an internal search metric and is not shown as another report card.
During sampled scoring, association metrics already estimate the sampling-frame population
using stratified weights. They are not raw metrics on just the 200 selected examples, and
HTML generation does not extrapolate them again. Live rows disclose the sample size and
scope separately from detector adherence. Final full-corpus scoring replaces these
estimates with observed metrics; abstentions and errors still remain unresolved.
Negative association can still exclude a candidate from discovery review. Current scores
and usable matches are required before reviewing a detector. Candidates stay in the working
library for refinement, and report cards require computed scores on every metric.
Good qualification and minimum-Good stopping remain separate:
an OK detector is useful evidence or a synthesis seed, but does not meet the Good minimum.

Behavior validation also reports confirmed, rejected, uncertain, and processing-error
counts, plus **decisive coverage** (confirmed + rejected, divided by sampled items).
The confirmation rate remains conservative; missing evidence is not a contradiction.
Reviews also distinguish unfinished evidence reads from genuinely unavailable or
ambiguous evidence. Claude and Codex can resume an explicitly unfinished read once,
within the existing review budget. An exhausted read budget remains inconclusive;
it does not become a reason to narrow the detector or a new unbounded retry loop.
Incomplete candidates remain available for research but are not assigned zero quality
or promoted into the validated behavior ranking. `judge_function` and
`get_function_feedback` expose the failed or unresolved conditions and next actions.
Reports distinguish no detected matches, unresolved detector predictions, rejected causal
claims, and unfinished reviews. Positive development reviews are labeled as development;
an attempted confirmation with no unexposed tasks is disclosed without replacing its
earlier results. Zero adherence from completed rejections is different from unavailable
adherence after processing errors or unfinished evidence review.

Default `submit_function` evaluates a detector, then schedules development review for a new,
changed, or stale version using `review_mode="automatic"` internally. Read `review_schedule`
and `review_deferred` for the decision and next action alongside `judge_prefilter` and
`judge_feedback`:

| Evaluated evidence | Automatic review and next action |
| --- | --- |
| `ppv_at_pi0 < 0.40` | Defer standard review. Repair precision from passing-match evidence, or specialize to a coherent supported subtype and accept lower coverage. |
| `0.40 <= ppv_at_pi0 < 0.70` | Defer automatic standard review and prioritize precision repair. If per-case claim evidence is needed to choose the repair, deliberately request a diagnostic review. A useful OK detector may instead receive deliberate standard validation. |
| `ppv_at_pi0 >= 0.70` | Schedule standard review when current scoring provides usable matches, subject to the existing prefilter and review prerequisites. Low Coverage or task support does not block review. |
| Missing precision or unavailable scoring/match evidence | Obtain the required evaluation or evidence; this does not reject the mechanism. |

Scheduling uses corpus normalized precision for ordinary detectors and within-parent
normalized precision for explainers. Sampled precision remains provisional, with its
uncertainty retained. Coverage and distinct-task support still determine quality grades
and the Good minimum; they do not determine whether a detector can receive an adherence review.

Call `judge_function(name, review_mode="diagnostic")` deliberately when its per-case findings
will guide the next repair. It reviews at most three examples without producing a main adherence
score or confirmed support. Call `judge_function(name, review_mode="standard")` deliberately
when validating a useful OK candidate is worthwhile. Explicit `judge_function(name)` still
defaults to standard review; normal prefilters and review prerequisites continue to apply.
An unjudged or deferred status alone is not a reason to force standard review.

Specialization can trade coverage for precision when evidence supports a coherent subtype.
Revise the description and mechanism consistently with the narrower predicate, test retained
positives and passing controls across tasks, and preserve useful supported original versions.
Task-ID exceptions do not establish a reusable behavioral boundary.

Deferral is not claim rejection, zero adherence, or automatic deletion. Scheduling changes
neither grades nor the selected judge policy. Unchanged current reviews are retained,
including unknowns and processing errors; follow up through a repair plan rather than
resubmitting to retry. Frozen review cohorts and interrupted-review recovery are preserved.
Final full-dataset scoring and missing/unfinished review completion remain unchanged.
Empty detectors skip judging.
During family discovery, these automatic reviews use **development evidence**.
Independent confirmation is optional and follows the existing task exposure policy; there
is no upfront dataset partition. Supported development reviews can qualify for Good under
the shared quality criteria, with their limitations retained. For fresh confirmation, freeze
the candidate before final-census feedback, then call
`judge_function(name="…", review_phase="confirmation")` to review reserved or otherwise
unexposed tasks. Repeated rollouts of a task already inspected during discovery or
review do not count as fresh task evidence. If none remain, confirmation is reported
as unavailable without another paid review; development scores remain available.
Unavailable confirmation does not reserve an exhausted cohort, so newly available
unexposed tasks can be considered later. Once a review starts, interruption and retry
preserve its original cohort and target size.
This tracks structured AutoFA tool exposure, not arbitrary external file reads, and
does not make the reviewed sample representative of every detector match.

Review and exposure history travels with saved libraries, including Continue,
renames, splits and removed detectors. Legacy libraries without complete history
remain usable, but cannot establish fresh confirmation from that missing history.
Quoted localisation previews count as inspected evidence. Continue retains prior
exposure; it cannot reuse another run's completed review as fresh confirmation.
Repair feedback compares the frozen baseline with the candidate: changes to
precision, recall, purity, and the selected objective, plus previously confirmed,
rejected, and unreviewed matches lost. Incomplete evaluations cannot establish that
a missing match was removed. The comparison is diagnostic; it never automatically
promotes a repair or treats all lost matches as false positives.
`auto_evaluate=False` saves without evaluation or judgment. Follow the runtime's
final-validation handoff for a requested full-corpus scan or driver-managed scoring;
the census preserves current judgments instead of requiring duplicate reviews.

Public grades: **Good**, **OK**, **Weak**, **No matches**, and **Not evaluated**.
The agent's higher breadth grades, Excellent and Outstanding, display as Good publicly.
Main `pareto_status.top_x` and CLI/report ordering use the shared
quality-first order, then highest Coverage within each grade. `validated_top_x` separately
reports the stricter corpus-frontier ranking and its review eligibility.

Default optimization is **Recall** over `(ppv_at_pi0, task_recall, behavior_purity)`.
The geometric-mean formula and Balanced/Precision/Recall weights apply to those three
metrics; custom settings adjust their weights. New discovery and the weight editor normalize
retired metric sets to the current Recall preset; no other aggregation is supported.
Existing geometric-mean runs retain their saved axes and weights on Resume; a run saved
with another `objective_rank` fails to load. Attempt recall remains report Coverage;
specificity is diagnostic.

Search preferences guide exploration and scoring. They do not change the judge's treatment
of a frozen claim under its selected judge policy. Compatible saved reviews retain their
original provenance when weights change; changed claims, Judge Policies or evidence
requirements still require current review.

`task_recall` uses verified source task identities and the current evaluation's matches.
It is **observed task coverage**, not causal confirmation. Sampled, capped or unresolved
evaluations provide a lower bound against the full task denominator; row weights never
extrapolate distinct tasks. More rollouts can provide more chances to observe a mechanism,
but each task contributes at most once. Missing identities or stale evaluations leave the
metric unavailable. Later judge scans do not silently expand the frozen scoring panel.
Final minimum eligibility uses **Good** quality thresholds, including Coverage
(**attempt recall**) strictly above 1%, plus completed evaluation and review evidence.

`search_score` and `pareto_status.search_ranking` apply the configured objective to current
development evidence, including limited or reused causal reviews. They report provisional
scores, review counts and limitations; unknown cases remain in the purity denominator.
Missing or stale evidence cannot become an invented score. This gives discovery a useful
ordering while the validated frontier is still empty. During discovery, the task-recall
component is matched failed tasks divided by failed tasks on the shared fixed panel,
not the corpus lower bound described above. Compare only matching panel identities.
Useful mechanisms, remedies and quality improvements count as discoveries even when their
matches overlap. Report those gains separately from confirmed task-coverage growth; the
runtime's coverage-plateau counter still measures that distinct-task union.

The CLI search trend shows the **best current comparable search score**, with public-grade
counts and the last assessed confirmed task coverage alongside it. Adding another candidate
cannot lower this score while the existing candidates and comparison cohort remain unchanged.
A changed or removed best detector can lower it. Changed cohorts, objectives, or telemetry
versions reset comparisons. When sample identity is unknown, the current score remains
visible without a trend. Historical `frontier_value` still averages frontier members and
remains an explicitly separate corpus-frontier diagnostic.

Submission, evaluation, judgment and inspection responses include `discovery_feedback`:
concrete investigations suggested by current scores. Low precision suggests inspecting
passing-match boundaries; low purity suggests inspecting causal claims and recovery;
high-quality narrow task coverage suggests missed variants or synthesis of related seeds.
Incomplete evidence takes precedence. This is advice, not an extra acceptance gate, review
stage or automatic detector edit; detailed per-case findings and repair constraints still apply.
