# Operating AutoFA through the driver

The [driver skill](../agent-skills/autofa-driver/SKILL.md) is the main conversational
entrypoint. Point the host at this checkout's [AGENTS.md](../AGENTS.md) to install AutoFA,
register the skill and start using it in the same conversation, as shown in the [README](../README.md).
It inspects the installed AutoFA version and uses the workflows available there;
it does not replace AutoFA's discovery, taxonomy, scoring or judging implementations.

Tell it the evaluation runs or data paths, the outcome you want, and any deadline or
resource constraints. It can ask what “eight failures” means, keep a durable goal,
launch individual and combined analyses, monitor their actual progress, and continue
or stop them within that scope. Direct sample inspection, candidate injection and
repairs are available and attributed separately from native discovery and validation.
The managed native process runs independently of the conversational host. Closing that host
or stopping monitoring normally leaves discovery, finalization and configured publication
running; native deadlines, spend limits and explicit user stops still apply.

The driver uses
[`scripts/supervisor.py`](../agent-skills/autofa-driver/scripts/supervisor.py) to observe work,
with the current conversation providing reviews every ten minutes by default and after
meaningful changes. The user can change that interval or pause/resume monitoring without
restarting AutoFA. Healthy work continues unchanged; review is not an intervention cadence.
An explicit detached reviewer and verified communication adapter remain optional. Review
decisions, action outcomes and delivery receipts are durable. No conversational reviews or
updates occur while the host is absent; a local message log is not a notification. The local
machine must remain available. See [monitoring controls](../agent-skills/autofa-driver/references/scheduled-supervision.md#change-or-pause-monitoring).

Before launching, the driver reviews models and supported reasoning/effort settings separately
for discovery, behavior judging and LLM detectors. It explains quality, cost and latency
tradeoffs, reuses explicit preferences and checks saved-run overrides. See the skill's
[model setup guidance](../agent-skills/autofa-driver/references/model-setup.md).

After a report is generated, it explains supported findings and evidence gaps and proposes
prioritized improvements or follow-up analyses. These can include more rounds when marginal
coverage is still growing, targeted analysis of an error cluster, or a source-artifact deep dive.
Failures-only exploration retains separate full-cohort validation and coverage denominators.
Recommendations do not automatically restart stopped work. See
[findings and next actions](../agent-skills/autofa-driver/references/evidence-and-reporting.md).

## Native supervision interfaces

Use the current CLI help rather than assuming these examples fit another installation:

```sh
uv run autofa supervise --help
uv run autofa supervise status --run RUN_ID
uv run autofa supervise cost --run RUN_ID
uv run autofa supervise tools --name library_import_candidates
```

`status` reads saved observations without loading a mutable runtime, scanning datasets,
starting a dashboard or making model calls. It shows native rounds, turn and operation
state, recent progress, validation, coverage and publication snapshots. Snapshot entries
contain bounded summaries, file paths, sizes, modification times, digests and key counts.
Use `status --run RUN_ID --full-snapshots` only when the complete records are needed;
they can be tens of megabytes. Each snapshot has its own age. Operations record their
owner PID, hostname and available numeric process start marker. Status excludes dead owners from the active
list and reports them as stale without rewriting saved evidence. A present process does
not prove the operation is progressing; records without owner information have
unknown liveness; remote-host owners cannot be checked locally. Acquiring the native run lock
prunes confirmed dead operation records. Missing metrics remain unknown. Stored rounds count
completed native transitions.
Status preserves the difference between provisional grades and final validation.
`control` reports live requests and their revision. `campaign` reports the effective horizon,
quality milestones and stopping reason. `admission` is a timestamped owner observation of
estimated in-flight reservations, not a process-liveness guarantee.

The `usage` field and `cost` command read the native usage ledger. They show accumulated
tokens/cost, per-round/role/model groups and a qualified projection when enough completed
work is accounted. Missing monetary data stays unknown; provider-reported USD, estimates
and account credits remain distinct. The driver uses these observations to discuss current
and projected spending, and reports the final snapshot after validation/highlighting.
See [usage accounting](usage-accounting.md) for pricing setup and projection limits.

`tools` generates schemas from the current native tool registry. It is not a separate
catalog maintained by the driver. To execute a discovered tool against an inactive run:

```sh
uv run autofa supervise invoke --run RUN_ID --tool TOOL_NAME --arguments-file arguments.json
```

This acquires the native run lock and uses native tool admission, checkpoints and exposure
tracking. Depending on the selected tool it can perform inference or change saved results;
it is not a read-only inspection command. Live sessions use their own tool interface.
Invoke and finalization restore the run's saved harness and judge settings before loading
the runtime. Supervisor invocations carry driver attribution and do not consume discovery
steering notes.

## Live goals and contributions

Change discovery limits while the process remains active:

```sh
uv run autofa supervise update --run RUN_ID --id extend-1 \
  --expected-revision 0 --settings-file goals.json
```

`goals.json` can contain `{"additional_rounds": 14, "min_failure_modes": 8}`. Alternatively
set `max_rounds` to an absolute total. Additional rounds are normalized once against the
accepted saved limit/current round; retrying the same request ID cannot add them again. Use
the control revision shown by status. A reused ID with different contents or a stale revision
is rejected.

The initial receipt is `queued`. The owning runtime applies it at a safe tool/turn boundary,
preserving an already-started operation's settings and completed evidence. It can reopen
discovery from active polishing without replacing the process. Verify `applied` before
announcing success; deferred/rejected requests explain their outcome. User pause/stop wins.
An inactive run consumes queued changes when it resumes. Review/repair allowances are preserved.

Queue supported driver contributions through the same owner:

```sh
uv run autofa supervise tools --name library_import_candidates
uv run autofa supervise contribute --run RUN_ID --id candidate-1 \
  --expected-revision 1 --tool library_import_candidates --arguments-file candidate.json
```

Discover arguments from the native tool schema. Contributions use normal admission, repair
guards, exposure tracking and driver attribution. Transfers enter unscored and require native
target evaluation. Optional `--library-sha256` pins the expected saved library; imported file
hashes are checked again at execution. An interrupted contribution is `uncertain`: inspect its
native operation and library before submitting another request. `--fence-file` binds a campaign
path, goal revision and owner generation; superseded owners cannot apply delayed changes or
steering. Ordinary manual CLI usage does not require a campaign fence.

## Time and spending horizons

`campaign_policy` in scoped configuration or batch YAML separates discovery checkpoint
cadence from completion. The default is `finish_on_target` with `round_limit = 5`,
`coverage_target = 0.8` over failed rollouts and a full checkpoint every three rounds. An
empty policy (`{}`) restores round-limited behavior. A time and spend example:

```json
{
  "mode": "improve_until",
  "deadline": "2026-11-05T09:00:00+01:00",
  "spend_limit_usd": 50,
  "checkpoint_every_rounds": 3,
  "delivery_reserve_seconds": 1800,
  "delivery_reserve_usd": 5,
  "coverage_target": 0.8,
  "coverage_unit": "failed_attempts",
  "unknown_cost_policy": "stop"
}
```

Replace the example deadline and amounts with agreed limits. Three rounds then establish
checkpoint cadence, not an implicit stopping point. `finish_on_target` permits early quality
completion; `improve_until` keeps seeking useful improvement within the horizon. An explicit
`round_limit` remains binding. Coverage uses the agreed failed-attempt or failed-task denominator
and native GOOD qualification.

Discovery stops admitting new paid work at the delivery reserve; final validation can use the
reserve until the actual limit. Already dispatched work may finish. Monetary enforcement is an
estimated guard using available charges/rates and observed dispatch reservations, not a provider
billing cap. `unknown_cost_policy="allow_estimate"` accepts complete explicitly priced estimates;
unpriced gaps still prevent safe monetary admission. Install agreed `usage_pricing` before startup
inference. For multiple runs, allocate separate shares whose sum, including driver/delivery
allowances, fits the campaign budget; do not copy the whole budget into every run.

Optional `overtime` contains `when: "minimum_unmet"`, `seconds` and/or `usd`. Only an explicitly
agreed finite extension is consumed, once, when the validated GOOD count or an explicit
coverage target remains unmet; extra time does not lift a money limit. Change policy
through `supervise update` with a `campaign_policy` object. Saved state and usage survive Resume.
A resource boundary yields a normal `campaign_boundary` tool result; unavailable final
measurements remain partial and saved findings remain publishable.

## Steering, stopping and validation

For a conversational "stop here," the driver uses
[graceful stopping](../agent-skills/autofa-driver/references/operations.md#graceful-stopping)
to finish with a report. It asks two independent questions: finish final scoring on the full
dataset, and publish the report. Both default to a recommended **yes**; an unanswered question
is not permission to spend or publish. Reuse answers already explicit in the request, including
local-only or no-publication choices. Explain the available final-scoring time/cost estimate and
preserve agreed limits. Declining scoring selects a report from saved results with its incomplete
validation disclosed; declining publication keeps the report local. An explicit immediate stop
or "no more paid work" takes effect promptly and rules out fresh scoring.

These conversational choices do not change the low-level CLI: native `stop` still prevents
further work, while `finalize` hands off to validation. Keep the campaign open through the
chosen finalization and delivery instead of marking it stopped before a graceful finish.

```sh
uv run autofa supervise steer --run RUN_ID --id investigation-1 --message-file guidance.txt
uv run autofa supervise request --run RUN_ID stop
uv run autofa supervise checkpoint --run RUN_ID --output /path/to/new-checkpoint
```

Steering is discovery-only. It arrives at a discovery prompt/tool boundary, with a
stable receipt; the native agent records its response through `respond_to_steering`.
Each queued note is delivered once, using a bounded excerpt and the complete note's file
path. Unanswered notes stay visible in status but do not start additional paid turns or
override budget/finalization decisions. Identical responses are idempotent; invalid IDs or
conflicting responses return ordinary tool errors. Steering stays out of static system
prompts, judge rubrics and automatically loaded analysis skills. Interactive slash commands
such as `/compact` leave notes queued for the next real discovery turn.

Lifecycle requests are cooperative. `pause` and `stop` prevent further native work at
cancellation boundaries and remain saved across Resume. In-flight provider requests may
finish; inspect progress/process ownership before claiming a process has exited. The driver
can use an owned process's normal interrupt/escalation procedure when necessary. Checkpointing
requires an inactive run, holds its native lock, and uses SQLite backup to retain committed
WAL data. A commit already in progress finishes before cancellation is observed. A cancelled
tool returns `status: stopped_by_supervisor` so its harness receives a reply; a completed
tool keeps its result. The session observes the saved intent at its next turn boundary.
CLI stop messages name the intent and show
`autofa supervise request --run RUN_ID release` instead of claiming Ctrl+C was pressed.
A supervisor-stopped one-shot command exits with status 1; terminal Ctrl+C remains 130.
Optional highlighting returns a pending `supervisor_stop` result, leaving saved results
publishable. Checkpoints omit credentials, locks and superseded publication artifacts.
Copying does not hold the steering/lifecycle receipt lock. The destination must be new and
outside the run; a failed copy leaves it available for retry.

After stopping exploration, explicit validation-only handoff is available:

```sh
uv run autofa supervise request --run RUN_ID finalize
uv run autofa supervise finalize --run RUN_ID
```

Wait for the active discovery process to release its lock before invoking `finalize`.
For semantic detectors, a fresh finalizer can recheck a previously verified endpoint
before comparing evaluation settings. Refreshing a capped evaluation uses its original
row limit. A round checkpoint retains a completed full review when the cohort, evidence,
review policy and verdicts still match the full review.

Finalization reuses the native full-scoring/review checkpoints and preserves discovery
rounds. It records selected/skipped detectors, unresolved evidence and retained-result
completion separately. It does not publish automatically. A numerical target can remain
unmet after validation completes. `request ... release` explicitly clears a supervisor
pause/stop restriction, but does not itself start a process; then use the appropriate
native Resume or Continue workflow.

The ordinary `autofa publish --run RUN_ID` command retains report highlighting preparation.
For discovery delivery after stopping all inference, use the saved-results command. It
holds the run lock, renders saved results, and verifies the immutable uploaded version
without invoking scoring, judging or highlighting:

```sh
uv run autofa publish --run RUN_ID --saved-only --json
```

After an interrupted upload, inspect and reconcile its saved outcome first:

```sh
uv run autofa publish --run RUN_ID --saved-only --verify-only --json
```

Verify the configured destination before publication. Large review evidence is compressed
losslessly; source identity and complete history are retained. Saved-only delivery reconciles
an unresolved upload before attempting another append. A `publication_uncertain` result means
the outcome remains unknown; it does not authorize a fresh upload. A `verified` result contains
the immutable report URL. See [libraries and publication](library.md).

## Throughput, evidence and autonomy

`llm_eval_workers` requests scoring concurrency; `llm_eval_max_workers` is its explicit
ceiling (default 128, range 1–256), and `llm_eval_queue_mib` bounds retained scoring inputs
(default 64 MiB, range 1–4096). The default requested concurrency remains eight. These do
not change unrelated dataset-reader buffers, classifier identity or grade thresholds.
`scoring_capacity.json` reports observed queue pressure and stream throughput, including
cache reuse; it is not provider-request throughput or proof of a speedup. A supervisor
should measure changes, consider total host/provider demand, and back off on errors.

For scoped reallocation, stop/checkpoint the affected run and pass a JSON object containing
only the capacity fields to `autofa supervise configure --run RUN_ID --settings-file capacity.json`.
`judge_workers` (1–8) is also supported. The native run lock rejects changes to active runs;
overrides persist in `scoring_settings.json` and take effect on the next runtime load. Global
configuration, models, samples, scores and cache identities remain intact. The configuration
receipt previews effective values; runtime capacity observations show what was actually used.

Use `autofa preflight classifiers --help` to inspect or explicitly configure permitted
classifier alternatives. Native comparison/adoption retains controls, budgets and judge
routing, and requires fresh scoring. Provider and image-delivery problems are distinct
from negative detector predictions.

The native contribution tools preview unscored transfers, register external example
exposure, retain authorship/edit history, and record source-audit disputes. Existing-name
changes follow the native repair workflow. Reports disclose assistance and disputes;
coverage sensitivity is hypothetical and never edits official verdicts or denominators.

The [autonomy evaluation procedure](autonomy-evaluation.md) prepares isolated baseline and
updated arms without injecting solutions. Synthetic regression tests verify contracts and
native workflow execution with model doubles. Live discovery quality, recall improvements
and host image fidelity require their own explicitly scoped evaluation.
