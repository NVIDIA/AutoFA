# Launch, monitor and recover

## Launch and ownership

New analysis, continued library work and saved-run resume differ. Use discovered native
semantics. Pin source/seed identities; snapshot baseline settings/metrics; validate harness,
model and data access. Apply run-scoped settings; preserve judge routing when tuning classifiers.

Track each run ID, config, process identity/start time, logs, source map, budget and supervision
owner. Reconcile existing processes and locks before launch; PID alone is not identity. Use
persistent host execution when available; a foreground tool returning does not prove work
survives the agent/session ending. Record the next review and supervision boundary honestly.

For continuing analysis, establish host-owned supervision in
[scheduled-supervision.md](scheduled-supervision.md) **before** launch. Persist the campaign goal,
bind the exact current Codex/Claude session, start the inexpensive guard and consume/acknowledge
its first observation. Launch through its managed receipt; source preparation is covered before
a native run ID exists. Use the current host's goal tools when exposed and verify the goal;
otherwise keep the durable campaign goal and active foreground/Monitor review loop. Do not
replace another active goal or pretend a slash command is an available tool.

By default, check/review progress and send a routine recap every ten minutes (600 seconds),
plus meaningful round/GOOD/coverage/blocker/intervention/delivery events, in this same
conversation while monitoring is enabled. Ordinary token/cost increments, refreshed status
timestamps, poll timeouts or a healthy process in the same phase do not justify an early
message. Do not post minute-by-minute “still running” updates. Answer explicit user status
requests immediately. The interval is configurable without restarting
AutoFA. The default guard launches no additional model. A polling log is not a substantive
review, and a queued outbox is not a delivered update. A host acknowledgment records liveness;
only an acknowledgment with an assessment resets the substantive-review due time. Keep that
time visible, and verify continuation when promising ongoing reviews. Closing the host or
pausing monitoring leaves native discovery, finalization and publication running within the
agreed limits. Reattach later to review what happened; do not silently start a paid substitute
reviewer. An explicit detached review worker remains available when agreed.

## Monitor without another analysis

Prefer native read-only status. Otherwise use bounded tails/receipts, handling partial final
JSONL writes and records larger than the first buffer. A parse fault is not a stalled run.
Avoid full-library/corpus scans on every poll. Track phase, pending work, predictions/checkpoint
activity, review completions and sanitized errors across successive timestamped observations.

Use `supervisor.py observe` or bounded `watch` results, then `ack` after actually consuming
them. Prefer an owned `watch --stream --duration 600` in a supported host background/event
tool and a 600-second host review schedule when exposed. Renew its explicit expiration while
host monitoring remains enabled. Bounded foreground waits are internal housekeeping and do
not require a user message when unchanged. Include a concise
diagnosis/next action with `ack --assessment` at each quality review. A healthy review can
say why no action is needed. Internal service/control wakes need inspection, not automatic
user messages. Omit `--assessment` for transport-only acknowledgments; silent housekeeping
must not keep resetting the review deadline and delaying routine recaps. A substantive early
review/update can start a new interval. Keep native commands fenced with the returned `fence_file`.

A live terminal is not progress; quiet logs are not a stall when checkpoints advance. Perform
deeper quality checks after meaningful evidence or phase changes. Adapt intervals to workload,
failure risk and deadline. Sometimes the correct action is to let healthy native work continue.

Review does not imply intervention. Give a change time to reach its verification boundary,
unless a new urgent blocker requires action. After each completed round, explain discovered/
repaired mechanisms, new versus overlapping coverage and the next useful investigation.
Send meaningful-event updates and the ten-minute routine recap without waiting for a nudge,
unless the user chose another cadence. Apply a changed preference to the saved campaign and
owned host timers, even when the active session loaded older instructions; see
[monitoring controls](scheduled-supervision.md#change-or-pause-monitoring).

Answer status with a small table: label, completed/current/planned rounds (or workflow
equivalent), phase, validated versus provisional GOOD/OK, baseline→current requested coverage
and scope, recent progress/blocker, publication state. Estimate remaining time with uncertainty
from current measured work, not worker settings or a brief cache-write burst. Status requests
do not authorize rescoring or restarts.

Include available cost when reporting campaign progress. The new ledger covers discovery
and explicit classifier-setup probes. For other workflows, use only their existing native
accounting and report unavailable cost explicitly. Discover the native read-only cost interface
(`supervise cost` and `status.usage` in this version); use its durable aggregates rather than
scraping harness screens or adding cumulative session snapshots. State current token counts,
known USD subtotal split into reported/estimated portions, missing-accounting counts and
per-round changes. Include the projection only with its completed-round basis, remaining-work
assumptions, validation/delivery allowance and explicitly unestimated components.
A partial subtotal is not the full spend;
an unavailable forecast is not zero. Refresh projections after meaningful workload/model
changes, and report uncertainty when earlier rounds are no longer representative.
Do not invent per-round splits from a cumulative harness result: a Claude turn can cover
multiple rounds, and interruption before its final result can leave cost unknown. Use the
native attribution and notices even when the overall total is available.
Codex compactions can be absent from native token counters. If the summary reports
`compaction_usage_gaps`, describe the token total as partial; retain independently reported
cost or credits and do not estimate missing tokens from context size. Zero known USD with
missing cost observations means unavailable cost, not free work.

Keep native AutoFA execution separate from the analyzed model's efficiency metrics and from
the outer driver's own resource use. Sum campaign totals only for distinct native runs.
For discovery, Resume preserves a run's cumulative ledger, while Continue creates a separately
accounted run. Account for an already running request before claiming a spending limit stopped billing.
Reading cost/status or publishing a saved snapshot must never launch inference to fill missing
accounting. See [usage accounting](../../../docs/usage-accounting.md) for current semantics.

Budget shared workers, queue memory and provider pressure across runs. Inspect effective limits,
reallocate freed capacity, pilot increases and measure sustained benefit. Back off on connection
or rate-limit failures. Operational throughput changes should preserve evaluation identity;
model/API/reasoning changes require comparison and ordinary native evaluation.

## Recovery and lifecycle

After host resume, read the campaign, current guard generation, managed launch receipts and
bound run IDs before acting. Reattach with a fresh observation/acknowledgment; do not reissue
launch or infer the run from the newest directory. Stable launch IDs make repeated launch
requests return the same receipt. The default `on_host_loss: continue` leaves native work
running when acknowledgments, reviews or the guard disappear. `monitoring pause` disables
reviews without changing native lifecycle; `monitoring resume` reattaches to the existing run.
An explicitly configured `on_host_loss: pause` can pause discovery after loss of enabled
monitoring. Reconnection never erases that pause or any user stop: inspect lifecycle and
release only when authorized. A dead native process may need native Resume; a healthy one
does not. See [monitoring controls](scheduled-supervision.md#change-or-pause-monitoring).

Prefer queued live updates for rounds, targets and horizon changes. Read the native schema,
use a stable request ID and expected policy revision, and verify `applied` values in status.
Do not stop/restart an active process merely to edit live goal fields. The same native owner
must perform any reopening from polishing while preserving evidence and consumed allowances.

Classify interruptions: transient transport, context exhaustion, execution error, native
round/experiment/repair limit or provider/account cap. Resume cannot cure a hard account cap.
Continue completed work only within authorized budgets. Never rename claims or recreate runs
to reset exhausted repair/review limits or reroll unchanged adverse judgments.

Treat `review_capacity_exhausted` as a saved operational block, distinct from a completed
quality-stall round or detector rejection. Inspect the named review's budget, workload,
completed judgments and unread evidence. Do not repeatedly Resume an unchanged frozen
review or promise that more discovery rounds replenish its allowance. Use only a supported,
bounded recovery that preserves the cohort, completed evidence and consumed budget; otherwise
deliver the partial results with the unmet target and blocker stated explicitly.

Use one run owner and native exclusive locking. Gracefully stop owned workers, allow checkpoint
writes to drain and verify lock release before external changes. Make a native checkpoint
through the discovered interface or verified adapter in the active runtime. Preserve config,
definitions/helpers, turn/progress state, scores/reviews/exposures and consistent predictions.
SQLite backup under native locking preserves committed WAL state; copying the main `.db` alone
is not a backup. Verify synthetic restoration and retain hashes/rollback.

Inspect interrupt semantics: first interrupt may stop discovery and start final validation.
Use [graceful stopping](#graceful-stopping) to resolve a generic stop request. Persist the
chosen intent before stopping watchers; stop newly begun finalization too when the user
explicitly ends all inference. Escalate only against verified owned processes. Never
automatically resume a user-stopped run.

Distinguish discovery, validation, report and publication completion. Use validation-only
handoff when retained detectors need final measurements; preserve rounds/sources and reuse
checkpoints without starting another discovery turn. Retry execution failures only through
bounded native recovery, retaining successful work and errors/contrary evidence.

For small authorized runtime fixes: reproduce, isolate, test, checkpoint, record source identity,
deliberately activate the correct installation and verify progress; preserve rollback. A
checkout edit does not patch an installed release. Coordinate changes to shared installations.

Before repeating an interrupted import, continuation or upload, reconcile receipts/native
effects; an exception does not mean nothing happened. After delivery, close owned monitors,
publishers and keep-awake processes, preserving unrelated processes and disclosing outstanding work.

## Graceful stopping

For an ordinary “stop,” “stop here” or “enough exploration,” the preferred outcome is a
finished report. Ask these two independent questions together, with **Yes** recommended
and preselected where the host supports it:

1. **Run final scoring on the full dataset?** Yes / No, use saved results only.
2. **Publish the report to AutoFA Cloud?** Yes / No, keep a local report.

Reuse any answers already explicit in the stop request or a standing instruction for this
situation. Preserve an explicit local-only/no-upload preference. Briefly state the remaining
scoring/review workload, estimated time and cost when known; full validation can take hours.
Do not ask the user to reconfirm unchanged model choices. An unanswered question is not a
yes; do not begin a newly selected finishing action or interpret silence as extra budget.
Use the supported native pause while answers are pending, preserving completed work and
verifying the owner has quiesced. Do not turn the outer campaign into a hard-stopped campaign
while the graceful finish is being chosen. If the user says “stop immediately” or “no more
paid work,” persist that stop promptly; a report from saved results remains a separate choice.

| Final scoring | Publish | Finish |
|---|---|---|
| Yes (default) | Yes (default) | End exploration, run native full-dataset scoring and required review, generate and publish the report, then send the verified link and findings. |
| Yes | No | End exploration, run native final validation, and deliver a local report and findings. |
| No | Yes | Stop inference and publish a saved-results-only report, explicitly marking sampled/incomplete evidence. |
| No | No | Stop inference and render a local saved-results-only report with the same qualifications. |

Record the answers as a campaign revision and preserve source identity, existing scores,
completed predictions and already consumed allowances. Use the bound native executable and
frozen profile for each command; do not rewrite launch snapshots. Full scoring uses native finalization
and its cache; it does not authorize new discovery rounds or override time/spend limits.
If finalization is blocked or the allowance is exhausted, preserve the partial report and
explain the gaps rather than quietly dropping delivery or claiming final validation passed.

For the normal **yes/yes** path, keep the outer campaign active through finalization and
delivery. Request native `finalize`, not `driver.py stop`: the latter makes the watchdog
stop paid finalization as well. Monitor the owner's finalization, or, once it has released
the run lock, use the validation-only `autofa supervise finalize --run RUN_ID` if still
needed. That command does not publish; complete the existing delivery/reconciliation workflow.
Do not rerun completed validation or launch another discovery session.

A newly confirmed scoring choice can authorize changing an existing native pause/stop to
`finalize`. Record that explicit user decision and use the native user-request interface;
a stale automated fence must not clear a saved stop. Do not use `release` plus a generic
Resume merely to obtain final scores. If an outer campaign was already stopped, explicitly
authorize and record reopening for validation/delivery only before handing off; otherwise
its watchdog will keep forwarding stop. Preserve budgets and prevent another discovery launch.

For **publish=no**, a running batch may still have `execution.publish=true` captured at
launch. Changing only campaign JSON does not disable it. Quiesce that owner through the
supported interruption path and verify its exit before a validation-only handoff; then use
`autofa publish --run RUN_ID --saved-only --local-only --json` to render the local artifact.
Never replace an agreed local report with a cloud upload because of the old launch setting.

For a saved-only finish, preserve the native stop and use `--saved-only` publication/rendering
without scoring, judging or optional highlights. Record `goal.delivery.destination` and
`publish_on_stop: true` when delivery after this stop is chosen. The existing flag permits
stopped delivery to the selected destination, including a local report; it does not turn
`destination: local` into cloud authorization. Verify the artifact and send its link/path,
findings and cost qualifications before completing the campaign.

## Local helper: portable bookkeeping

`python3 <skill>/scripts/driver.py --help` describes commands. They return JSON and never launch
AutoFA or model calls. Keep state outside skill source and analyzed data; do not store secrets.

- `init --campaign DIR --goal FILE`: create goal state; goal requires `objective`.
- `revise --campaign DIR --patch FILE --reason TEXT`: merge goal changes and preserve history.
- `stop`, `reopen`, `complete --campaign DIR --reason TEXT`: record lifecycle decisions, without
  signaling processes. Completion records delivery, not achievement of numeric targets.
- `show --campaign DIR`: read recorded state; reconcile with native state before acting.
- `identify --executable PATH --source-root DIR [--config FILE ...]`: fingerprint runtime inputs.
- `cache-put --campaign DIR --identity FILE --map FILE`, `cache-get --campaign DIR`: bind and
  validate a discovered capability map against current source/config/evidence hashes.
- `snapshot --campaign DIR --label TEXT --file FILE ...`: immutable observational copies of
  selected non-database artifacts with hashes. **Not a consistent native checkpoint**; SQLite
  and WAL files are rejected. Native checkpointing owns recovery.
- `record --campaign DIR --kind intervention|checkpoint|publication --receipt FILE`: append a
  receipt with stable `id`, `state`, `reason` and action evidence. Identical repeats are
  idempotent; changed states append history; conflicting content for one ID/state is rejected.
- `verify --manifest FILE`: verify local downloaded bytes described by `artifacts` entries
  containing `path` and `sha256`. Remote identity/semantics need native verification too.

Receipts should include goal/authorization revision, run/process/source IDs, checkpoint,
baseline, proposed/applied/rejected/reverted decision, native result, exposed examples, measured
cost and next action. A recorded checkpoint receipt does not create or certify a checkpoint.

Additional record kinds are `review`, `delivery` and `allocation`; their receipts carry evidence
rather than causing those effects. `preferences --path FILE [--settings-file FILE --reason TEXT]`
reads or saves explicitly agreed outer-driver defaults without editing native configuration.
Existing saved-run settings remain authoritative.
