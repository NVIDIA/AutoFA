# Goals and routing

Read the conversation and available source metadata before asking for information. Ask only
when the answer changes execution. Existing instructions and authorization survive a status
message. Do useful independent inspection while waiting for consequential missing information.

For a campaign, establish:

- Outcome and sources: local/remote runs, model/cohort, immutable seed library version,
  individual analyses and any concatenated-source analysis.
- Success definition: supported mechanisms, failed-attempt union recall, distinct task
  coverage, classification completion or the discovered workflow's own objective.
- Time/resources when relevant: absolute deadline/timezone, native rounds/experiment budget,
  allowed model/API profiles, spending/resource constraints and deliverable destination.
- Finishing: prefer a report after exploration ends. For a generic stop request, ask whether
  to score the full dataset and publish the report, recommending yes to both; follow
  [graceful stopping](operations.md#graceful-stopping). Explicit stop-time choices and
  local-only/no-more-inference constraints take priority over those recommendations.
- Initial setup: review discovery, behavior-judge and LLM-detector models and effort using
  [model-setup.md](model-setup.md), including the effective discovery objective and weights;
  reuse explicit choices and explain consequential tradeoffs.
- Reuse: check relevant cloud libraries using [cloud-reuse.md](cloud-reuse.md), distinguish
  exact dataset/cohort matches from possible transfers, and record the user's import/fresh
  choice with immutable seed identities. Keep the requested input sources unchanged.
- Assistance: native execution with supervision plus useful driver contributions within
  scope; a separately protected unassisted comparison when requested.
- Monitoring: ten-minute checks/reviews and routine user updates by default, with earlier
  meaningful round/GOOD/coverage/blocker/intervention/outcome updates and direct status replies.
  Do not turn internal polling or ordinary token/cost increments into minute-by-minute messages.
  The user can change either cadence or choose event-only updates; persist and apply that
  preference to the existing campaign and its owned host tasks. Closing the host or stopping
  monitoring normally leaves AutoFA
  running. Offer pause-on-host-loss only when wanted; monitoring can resume later on the same
  campaign without changing its native limits or restarting a healthy process.

For discovery without a numeric goal, propose five distinct supported GOOD detectors and
aspirational 80% deduplicated combined error coverage, adapting to the dataset. Agree the
denominator and grade cutoff: failure-attempt union is a natural interpretation of “errors,”
while the Recall ranking preset optimizes failed-task recall. Show both without substituting
one for the other; neither the count nor 80% is a guarantee or a changed grading threshold.
Agree the discovery/judge harnesses and models, LLM detectors, effort and permitted escalation
before launch, surfacing missing access or unsupported capabilities. For a time/spend campaign,
resolve whether to improve until the delivery reserve or finish as soon as targets are met.
Default five rounds or four GOOD is not the endpoint of an agreed improvement horizon.

Agree the limited response to an unmet target at the boundary: partial delivery or a finite
additional time/spend allowance. Unanswered extension preferences mean no extension. Time and
money permissions are separate. Record whether spend covers native runs, outer-driver work,
or both, plus starting baselines for existing runs. Per-run allocations plus driver/delivery
reserves must fit the campaign allowance; copying the full cap to each run multiplies spending.
`budgets.spend_limit_usd`, `run_allocations_usd`, `driver_reserve_usd` and
`delivery_reserve_usd` are portable bookkeeping fields; native policy must enforce the shares.
For continued/existing runs, `budgets.usage_baselines_usd` maps each native run ID to the
already-accounted lifetime spend at campaign admission. The native lifetime cap is that baseline
plus its additional `run_allocations_usd` share. Do not invent a zero baseline for unknown history.
`budgets.run_overtime_allocations_usd` maps each run to its share of the explicitly approved
`overtime.usd`; the sum cannot exceed that one campaign extension. Without explicit shares,
multi-run campaigns do not independently receive the full overtime allowance.
Record `budgets.cost_scope: campaign|native_only` explicitly. A campaign-wide cap additionally
needs the approved supervisor `host.max_review_cost_usd` estimate, used to reserve outer-review
allowance before launch. Report measured cost and unresolved reservations separately; an estimate
cannot guarantee provider billing. See [scheduled-supervision.md](scheduled-supervision.md).

Do not ask every item for every request. Opening a library does not need a discovery budget.
Classifying runs must route to the discovered classification contract, not automatically start
failure discovery. New workflows use the same intent-to-interface process; these examples
are not a list of supported workflows.

Summarize the interpretation before launch. Clarify whether “eight failures” means mechanisms,
failed rollouts or assignments. Resolve “tomorrow morning” to a date/time in the user's timezone.
Keep **additional** rounds separate from the native total limit. Record per-detector recall,
deduplicated failed-attempt union and confirmed distinct-task coverage as different metrics.

Persist the goal, sources, native run IDs, success criteria, constraints and deliverables. Use
the host goal/task facility if available and authorized, plus portable campaign state.
`driver.py init` accepts a goal object; `revise` records a reasoned patch and history. Suggested
fields: `objective`, `workflow`, `sources`, `success_criteria`, `deadline`, `timezone`, `budgets`,
`assistance`, `model_configuration`, `optimization`, `library_reuse`, `deliverables`,
`next_actions`. The helper does not prescribe workflow names or native schemas.
`deadline` requires an absolute ISO timestamp with its UTC offset. `driver.py preferences`
can save explicitly agreed role/model/escalation defaults to a chosen preferences file;
it never installs those preferences into existing native runs. Keep credentials as references.
Keep recommended follow-ups separate from actions
already authorized and from the current campaign's completion state.

Track “add rounds,” “minimum eight modes,” “finish by 21:00,” “just upload,” and “enough
exploration” as goal revisions. Delivery and quality targets are independent: a delivered
report can document unmet recall. Completion of execution/publication does not prove targets
were met. Record unknown cost as unknown.

For live changes, preserve the run/process/harness session and use the discovered native
`supervise update` interface with a stable ID and expected policy revision. Read its schema:
total rounds and additional rounds differ. Queue changes once, verify the applied revision
and native phase transition, and preserve consumed review/repair allowances. Updating only
campaign JSON or externally rewriting `config.json` does not change a running native loop.
An active horizon can reopen discovery after polishing through its native transition; do not
use stop-and-Resume as a substitute for a supported live goal change. Actual code activation
can still require a deliberate restart.

Measure remaining native work, throughput and final validation before estimating a deadline.
Reserve delivery time as it approaches. Do not silently change denominators, grade thresholds,
native review/repair budgets or spending limits to satisfy a target. Preserve explicit user
configuration and AutoFA defaults instead of copying historical campaign settings.

For multiple analyses, maintain stable labels and separate source/run/library/version maps.
The combined analysis receives the intended union of source inputs with source-qualified item
IDs and selection metadata. Do not confuse a combined dataset with detector-library union
recall. Preserve source membership through transfer and publication.

“Stop monitoring” changes monitoring state only; it is not an analysis stop or goal failure.
Resolve a generic analysis stop through the graceful-finish questions before choosing native
finalization or stop-all. Persist explicit immediate-stop intent promptly. A later confirmed
final-scoring choice permits a validation-only handoff; it does not permit new discovery.
Goal revision alone never clears a native stop. “Status?” is not resume authorization.
A simple request need not create durable campaign files at all.
