# Run the outer driver without status nudges

Use this reference for an authorized continuing campaign. A report-only request does not
need a scheduler. Native AutoFA remains the analyst. The current user-facing host owns the
goal and reviews its work; a healthy review can choose no action.

## Default: supervise in the user's current host session

`supervisor.py` has two modes. `supervision_mode: "host"` observes native work and protects
the agreement without buying any model reviews. The existing `"reviewer"` mode starts detached
review workers and requires a separate communication binding; see below. Use host mode by default.

The default is one substantive check/review and routine user recap every **600 seconds**,
plus earlier meaningful events: completed rounds, changed GOOD detectors or material coverage,
blockers, interventions and outcomes. Answer an explicit status request when it arrives.
Elapsed one-minute waits, cached status timestamps, ordinary token/cost increments and a
healthy process continuing in the same phase are not significant events. Keep those quiet;
do not post minute-by-minute “still running” messages. The guard's inexpensive `poll_seconds`
is an internal observation interval, not the host review or communication cadence.

Persist the campaign with `driver.py init`, including `goal.delivery.destination` (`cloud` or
explicitly chosen `local`), before preparing the launch. The completion goal covers source
preparation, discovery, validation, verified publication and a findings message. Partial quality
at an agreed bound is a valid honest outcome; an exited process alone is not completion.

Configure the guard **before a native run exists**:

```json
{
  "supervision_mode": "host",
  "native": {"argv": ["/path/to/checkout/.venv/bin/python", "-m", "autofa"],
             "cwd": "/path/to/checkout",
             "timeout_seconds": 900,
             "env": {"AUTOFA_CONFIG": "/path/to/campaign/scoped.toml"}},
  "host": {"kind": "codex", "session_id": "the-exact-current-host-session"},
  "runs": [],
  "poll_seconds": 15,
  "review_seconds": 600,
  "host_grace_seconds": 420,
  "on_host_loss": "continue",
  "allowed_actions": ["steer", "update", "contribute"]
}
```

Use `kind: "claude"` for Claude Code, independently of the native AutoFA harness. The helper
does not guess a session ID from an unrelated recent session. Obtain it from the host's supported
session surface. `host.executable` can identify a non-default installed CLI. Configure/doctor are
read-only with respect to inference. Doctor checks local CLI help, not an active goal or wakeup.

Saved-report publication and verification each allow 900 seconds by default. Set
`native.timeout_seconds` to a positive finite number for larger reports or slower connections.
When a run has its own `runs[].native` binding, configure the timeout there; delivery uses that
resolved binding rather than the global one. A timed-out upload remains uncertain and must be
reconciled before retrying. This deadline is separate from polling and review timeouts.

```sh
python3 /path/to/skill/scripts/supervisor.py configure --campaign /path/to/campaign --settings-file /path/to/supervision.json
python3 /path/to/skill/scripts/supervisor.py doctor --campaign /path/to/campaign
python3 /path/to/skill/scripts/supervisor.py start --campaign /path/to/campaign
python3 /path/to/skill/scripts/supervisor.py status --campaign /path/to/campaign
```

Read the live owner's `generation` from status. Consume the next observation in this host,
then acknowledge it using its returned `observation_id` and the real host task/monitor ID:

```sh
python3 /path/to/skill/scripts/supervisor.py observe --campaign /path/to/campaign --generation GENERATION
python3 /path/to/skill/scripts/supervisor.py ack --campaign /path/to/campaign --generation GENERATION --observation-id OBSERVATION --session-id SESSION --task-id TASK --assessment 'Configuration agreed; monitoring source preparation before native run creation.'
```

This explicit read/ack handshake is required before launch. An outbox, a Popen PID or a successful
`codex queue --resume` exit is insufficient. Only the actual host acknowledges a consumed result;
never put `ack` in an automatic shell polling loop. A heartbeat acknowledgment keeps the connection
live; only a meaningful `--assessment` resets the substantive-review due time (ten minutes by default).
For transport-only acknowledgments, omit `--assessment` so the review deadline stays intact.
Do not record repeated silent technical checks as substantive assessments and thereby keep
postponing the user's routine recap. A genuine early review and meaningful user update can
start the next ten-minute interval. When available, an independent supported host review
schedule helps preserve cadence; do not claim exact timed delivery from an unsupported host.
These receipts protect current ownership and mutations; continued native execution does not
require a live host acknowledgment under the default loss policy.

Prepare scoped settings with the native config contract interface. Its `launch_spec` pins the
exact native argv, cwd, config/contract file hashes, explicit `AUTOFA_CONFIG` and
`AUTOFA_SETUP_CONTRACT`. Launch's publication setting must match the agreed destination; cloud
delivery rejects missing or loopback Library URLs. Keep the generated stable `launch_id`:

```sh
python3 /path/to/skill/scripts/supervisor.py launch --campaign /path/to/campaign --generation GENERATION --spec-file /path/to/launch.json
python3 /path/to/skill/scripts/supervisor.py watch --campaign /path/to/campaign --generation GENERATION --after FINGERPRINT --stream --duration 600
```

The launch receipt precedes subprocess creation. The worker records process birth identity,
local log path and exit status even if download fails before a native run exists. Native code
binds its run through `AUTOFA_DRIVER_LAUNCH` and `AUTOFA_DRIVER_LAUNCH_ID`; no directory-order guess
is used. Each bound run retains its launch's explicit config/home and native command for later
status, actions and publication; changing the guard's initial profile cannot redirect these calls.
Repeating the same launch request returns its receipt. A new attempt needs a new ID and
a justified recovery decision. Failed preparation never becomes a successful empty analysis.

`observe` and `watch` return bounded native summaries, launch/preparation events and review-due
state. `watch` wakes for scheduled reviews and stable changes, including internal queued-control
or service-state observations. Consume those cues internally and decide whether they establish
meaningful progress, an intervention or a blocker before sending an early message. A wake or
`review_due` flag alone is not a notification instruction. Changing countdowns, usage totals or
status timestamps alone do not trigger a review. An explicit `observe` returns
the fresh full summary. Run `watch --stream --duration 600` in an owned, supported host
background/event tool; it emits JSON lines followed by `watch_expired`. Pair it with a
600-second host review schedule when that capability is exposed, and renew the owned stream
while monitoring remains enabled. Individual waits are bounded to 60 seconds; streams expire
within 30 minutes. An unchanged timeout or ordinary stream renewal is internal housekeeping,
not a user update. Look at referenced local logs only as needed.
Use the returned fence file for native steering/update/contribution commands. A changed campaign
revision or owner invalidates old acknowledgments and delayed actions.

### Host continuation and recovery

- **Codex:** use the exposed goal tool when available, and keep the current task waiting on
  bounded watch/process events. Official `/goal` is also available to the user. A supported
  `codex queue --thread … --message … --resume` adapter can wake that exact session, but its exit
  code does not prove resume succeeded. Require a subsequent host acknowledgment. The default
  implementation does not silently install a queue or messaging adapter.
- **Claude Code:** persist the campaign goal, use `/goal` or a goal API only when actually
  available, and run the watch stream with the exposed `Monitor` tool. Schedule a ten-minute
  review with `CronCreate`/`/loop` when supported. `/goal` background check-ins alone are too
  sparse. Renew expired monitors and delete owned scheduled tasks on completion/stop.
- **No supported asynchronous continuation:** use bounded foreground waits while monitoring
  is active, keeping unchanged waits silent until the agreed review/update time or a meaningful
  event. The wait limit does not reduce the ten-minute communication interval. If the user
  closes the host or pauses monitoring, record/disclose that review is
  suspended while native AutoFA continues. An explicit detached-reviewer handoff is optional;
  do not start a replacement model worker merely because the host goes away.

Native execution and host monitoring have separate lifetimes. With the default
`on_host_loss: continue`, missing acknowledgments, an overdue review or a stopped guard never
pause native discovery. AutoFA continues its analysis, finalization and configured publication
within the agreed native limits. While the host is absent there is no conversational review,
corrective guidance or user update; deterministic observation and saved-result publication can
continue in the detached guard. The driver must describe that distinction honestly.

Users who want discovery to require a live reviewer can explicitly choose `on_host_loss: pause`.
While monitoring is enabled, that policy pauses new discovery at a safe boundary after host/guard
loss. `host_grace_seconds` controls that policy's grace; it is not the review interval. Pausing
monitoring explicitly disables the loss gate. Neither monitoring resume nor a new host session
clears an existing native pause/stop, restarts an exited analysis or resets its spending ledger.

The host goal and tasks are subject to host permissions/lifetime. Use the managed detached worker
for native execution; host background-task lifetime alone does not prove process survival. A
sleeping/offline machine cannot execute local AutoFA or its guard. State that concrete boundary
before promising unattended execution. Sources: [Codex goals](https://learn.chatgpt.com/docs/long-running-work),
[Claude goals](https://code.claude.com/docs/en/goal),
[Claude scheduling](https://code.claude.com/docs/en/scheduled-tasks),
[Claude Monitor](https://code.claude.com/docs/en/tools-reference#monitor-tool).

### Change or pause monitoring

Monitoring controls use the current guard `generation` from `status` for mutations. They do not
change native rounds, models, budgets or lifecycle. Examples:

```sh
python3 /path/to/skill/scripts/supervisor.py monitoring status --campaign /path/to/campaign
python3 /path/to/skill/scripts/supervisor.py monitoring configure --campaign /path/to/campaign --generation GENERATION --review-seconds 1200
python3 /path/to/skill/scripts/supervisor.py monitoring pause --campaign /path/to/campaign --generation GENERATION
python3 /path/to/skill/scripts/supervisor.py monitoring resume --campaign /path/to/campaign --generation GENERATION --session-id CURRENT_SESSION
```

`configure` accepts a review interval from 1 to 86,400 seconds and optional
`--on-host-loss continue|pause`. The default is a ten-minute review and routine user recap,
plus important events;
intervene only when evidence warrants it. The saved `supervision/monitoring.json` overrides
initial monitor settings live. A cadence change does not restart the guard or native process.
Apply the user's current instruction to an existing campaign as well as future launches:
persist the conversational cadence in its goal using `driver.py revise`, apply the review
interval through `monitoring configure`, and reschedule only this campaign's owned host
watch/review tasks. A session that loaded an older skill must still honor the new preference;
editing the skill file alone does not change its saved campaign or host timers. Users can
choose a different review or message cadence, including event-only messages.

`pause` stops host/detached model reviews, not AutoFA. The inexpensive status, deadline, native
stop forwarding and publication guard remains active. Do not use `driver.py stop` to stop
monitoring: that command requests an actual analysis stop. Closing the host under the default
loss policy also leaves native work running, but the host must not claim it is still reviewing.
After pausing monitoring, cancel the host's owned watch streams and periodic review tasks,
and pause any host goal that keeps invoking supervision. Keep the native process and detached
guard running. On monitoring resume, recreate those host tasks after observing and acknowledging
the current state; do not keep waking a paused monitor just to acknowledge that it is paused.

Updating source files does not replace modules already loaded by an older running process.

`resume` can bind the current or a new host session. Resuming paused monitoring or attaching a
different session invalidates earlier acknowledgments/fences; repeating resume for the already
active session is idempotent. Read current campaign/launch/native status, then `observe` and
`ack` before issuing any
intervention. Reattach to the saved run IDs without launching another analysis. If the guard
itself exited, start it first, read its new generation and resume monitoring there. A user stop
still requires separate explicit authorization to release.

Final saved-result publication runs asynchronously from the guard, independent of paid reviews.
After preparing and sending the verified findings summary, acknowledge its returned digest with
`ack --summary-sha256 SHA256`. The completion validator checks this exact message/evidence;
never acknowledge a summary merely written to an outbox. Close owned watches when monitoring
is paused or ends; preserve campaign and delivery receipts for later reattachment. Verified delivery, an explicitly reported blocker or an actual
analysis stop determines campaign completion separately from monitoring state.

An ordinary “stop here” first follows the
[graceful-finish questions](operations.md#graceful-stopping), with full scoring and publication
both recommended. Keep the outer campaign active for authorized final validation and delivery;
`driver.py stop` is the hard stop of inference, not the default graceful-finalization path.
Record a declined publication choice before handoff and quiesce any batch owner whose frozen
configuration still enables automatic upload. A changed campaign preference alone cannot
change that owner's captured publication setting.

When a stop of inference is selected and `goal.delivery.publish_on_stop` is true, the guard continues
bounded status and saved-only reconciliation. `stop_delivery_grace_seconds` defaults to 300;
expiration records a delivery blocker and leaves local results and receipts available. Once the
guard stops, the same session/generation may still `observe` and acknowledge the exact prepared
summary digest. Such an acknowledgment has `scope: saved_delivery`; it renews no lease, creates
no mutation fence, and never releases a saved stop. Complete delivery without reopening discovery.

## Optional detached review mode: execution and communication bindings

`scripts/supervisor.py` is a standard-library Python 3.9+ service. It supports Codex and
Claude CLI review workers, plus a command adapter for a host's supported continuation API.
The outer host may differ from AutoFA's discovery/judge harness. Before paying for reviews:

1. Follow workflow discovery and verify the native executable, checkout and scoped config.
2. Resolve the user's review model and permitted effort settings. Builtin Codex reviews use
   `exec` with a read-only filesystem sandbox; Claude reviews expose only Read/Glob/Grep and
   use `--safe-mode` to disable user/project hooks and customizations while preserving auth.
   The capability probe requires this flag; older CLIs need a separately reviewed custom adapter.
   Administrator-managed policy can still apply. Reviews propose
   interventions as JSON; the supervisor performs native mutations under revision fencing.
   Authenticated CLI access and model availability are separate checks.
   Codex's filesystem sandbox does not prove that configured MCP/apps/hooks cannot mutate
   external state. Inspect the effective tool configuration, disable inappropriate tools through
   supported settings, and record `host.external_tool_review: "reviewed"` before starting.
   Doctor reports this distinction instead of claiming automatic external-tool isolation.
3. Bind the user channel. CLI review output does **not** automatically post into the original
   Codex/Claude conversation. Use a verified host continuation/delivery adapter, or explicitly
   agree foreground supervision or an outbox-only boundary. Never invent an app messaging API
   or silently add Slack/email. A local file is not a delivered user update.
4. Confirm the machine/runner stays available. A detached process survives the initiating
   terminal turn; it does not make a sleeping/offline laptop execute timers. Native run policy
   enforces its own deadlines and spend admission even when the outer driver is unavailable.

In detached review mode the native runs must exist before binding them; establish host-owned
launch monitoring first if preparation is still pending. Multiple runs share
one campaign owner and review; keep separate source and immutable library/version mappings.

## Configure and start

Persist the agreement with `driver.py init`/`revise`. The native runs must carry the agreed
policy and spending shares; setting an outer goal alone does not configure their admission.
Create a supervision settings JSON, adapting paths and the approved action set:

```json
{
  "native": {
    "argv": ["/path/to/AutoFA/.venv/bin/autofa"],
    "cwd": "/path/to/AutoFA",
    "env": {"AUTOFA_CONFIG": "/path/to/campaign/autofa.toml"}
  },
  "runs": [{"run_id": "SAVED_RUN_ID", "label": "Dataset"}],
  "host": {
    "kind": "codex", "executable": "/path/to/codex", "model": "AGREED_MODEL",
    "external_tool_review": "reviewed"
  },
  "review_seconds": 600,
  "poll_seconds": 60,
  "review_timeout_seconds": 240,
  "closing_review_limit": 2,
  "heartbeat_seconds": 600,
  "allowed_actions": ["steer", "update", "contribute"],
  "allowed_update_fields": ["max_rounds", "min_failure_modes"],
  "delivery": {"kind": "outbox", "outbox_only_acknowledged": true},
  "watchdog": {"enabled": true, "intent": "finalize", "reserve_seconds": 900}
}
```

The example records an explicitly accepted outbox-only arrangement; set the acknowledgment
only after agreeing that boundary. Start refuses an unacknowledged outbox. Replace it with an
agreed adapter below before promising updates will arrive without the user checking. Likewise,
record the Codex tool review only after inspecting the actual environment.
Choose a measured delivery reserve; 900 seconds is an example, not a workload estimate.
`goal.deadline` (or `goal.campaign_policy.deadline`) must include its UTC offset. The watchdog
uses the effective native deadline and reserve, including already-approved overtime and a pending
native extension. With unavailable status it falls back to the finite extension recorded in
the agreement. It does not consume overtime itself or preempt it with an early finalize request.
Before requesting finalization it rechecks native state. Native admission also refuses an
automatic finalize that would preempt pending approved overtime or overwrite a saved user stop.
It requests native finalize/stop at the resulting boundary. It does not itself render,
publish or verify a report, and never grants overtime. The driver coordinates finishing through
approved native operations and verifies the resulting artifacts.

If the agreement includes money, explicitly record `goal.budgets.cost_scope` as `campaign`
or `native_only`. A campaign-wide cap requires a positive `driver_reserve_usd` and a user-approved
`host.max_review_cost_usd` estimate. There is no dollar default. Native shares plus driver/delivery
reserves must fit the campaign cap. Before each review the supervisor reserves its estimate;
it stops admitting paid reviews when that allowance is exhausted, preserving deterministic
watchdog and factual round/event/heartbeat updates. A restart does not erase reservations.
Detailed older review costs move to immutable receipt files; exact totals and unresolved
reservations remain in bounded active state. Claude's returned
`total_cost_usd` and custom `cost.json` receipts can replace a reservation with reported spend.
Without a cost receipt (including the built-in Codex adapter), the reservation stays explicitly
estimated/unknown. A provider can exceed an estimate: this is an admission guard, not a hard
provider billing cap. Report native and outer costs separately and qualify their combined total.
If the user explicitly scopes the cap to native analysis, outer review spend remains outside
that cap and this exclusion must appear in the startup agreement and final accounting.

For Claude set `host.kind` to `claude`, select its executable/model and optional `effort`.
For Codex, `host.extra_argv` permits model flags and `-c`/`--config` with
`model_reasoning_effort="AGREED_EFFORT"`; for Claude it permits model/effort flags.
Permission, sandbox, profile and tool overrides are rejected, including their short aliases.
Specify each model/effort once, through its named host field or the supported extra flag.
See the [official Codex configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference)
for reasoning and MCP/app controls. Use a separately reviewed custom adapter for other host
configuration needs; it must honor the same agreed read-only review scope.

```sh
python3 /path/to/skill/scripts/supervisor.py configure --campaign /path/to/campaign --settings-file /path/to/supervision.json
python3 /path/to/skill/scripts/supervisor.py doctor --campaign /path/to/campaign
python3 /path/to/skill/scripts/supervisor.py start --campaign /path/to/campaign
python3 /path/to/skill/scripts/supervisor.py status --campaign /path/to/campaign
```

Configure does not call a model. Doctor checks local CLI help or the custom adapter's explicit
read-only capability probe. It does not verify authentication, model access or actual channel
delivery. Start launches real paid host reviews: execute it only within the campaign agreement.
Verify `owner.json` heartbeat, a **completed** substantive review and a real channel receipt.
`launched_verification_pending` is not a supervision success claim.

Use `run` instead of `start` under an existing process manager or foreground host facility.
The lifetime OS lock prevents duplicate owners. State includes owner generation, campaign
revision, review evidence, bounded native observations, pending actions and the outbox.
After a crash, restart the service and reconcile uncertain actions; never infer that an
interrupted command had no effect. Report missed reviews and the last successful review time.
Unexpected service errors have bounded diagnostic receipts and retries; stop forwarding and
message delivery remain active. A normal goal revision supersedes the old review without
being reported as a model failure and schedules an immediate review of the new goal.

Paid periodic reviews stop when all runs have fresh terminal native evidence or the latest
agreed deadline (including finite overtime) has elapsed. `closing_review_limit` permits at most
two closing reviews per goal revision by default, including failed attempts; set it to zero
when no additional paid delivery review is authorized. The first coordinates delivery; another
requires a new native event rather than an idle timer. Existing cost admission still applies.
Agree this closing allowance at setup. Afterwards factual status and delivery remain active,
without interpreting a deadline or `completed_with_gaps` as quality success. The reviewer can
record an evidence-backed `complete`/`partial` outcome after checking terminal work and delivery.
An explicit goal revision can authorize renewed reviews.

## Review and action protocol

Reviews and routine recaps default to every ten minutes while monitoring is enabled, with
earlier reviews after completed rounds, meaningful grade/coverage changes, blockers,
interventions and lifecycle/delivery outcomes. Several round transitions can be coalesced
into one message, retaining each round's substantive recap. The heartbeat/message preference
can differ from the review interval when the user requests it; internal polling never sets
the conversational cadence.

The review context includes the agreement, current observations, changed events, previous
review, authorized operations and action receipts. Inspect referenced native evidence when
summaries are insufficient. Record exposed examples before using them as development inputs.
The response includes `assessment`, `evidence`, `message` and `actions`. Each action has a
stable `id`, `kind`, bound `run_id`, `reason` and JSON-encoded `payload_json`:

| Kind | Payload | Native/evidence semantics |
|---|---|---|
| `steer` | `message` | Discovery-only guidance; delivered/addressed does not prove its effect. |
| `update` | `settings`, `expected_revision` | Native queued live update; only fields in the authorization manifest. Native status must confirm application. |
| `contribute` | `tool`, `arguments`, `expected_revision` | Native safe-boundary contribution queue; inspect its current allowed tools/schema first. Candidates remain unscored until ordinary native measurement. |
| `operation` | `name`, `values`, optional `arguments` | Explicitly configured native capability command; no shell. Use for discovered validation, publication, diagnostics or authorized recovery. |
| `verify` | `action_id`, `state`, `evidence` | Record `effect_verified`, `deferred` or `rejected` after inspecting actual results. This does not change native scores. |
| `complete` | `outcome: complete/partial`, `evidence` | Record the checked campaign outcome after terminal native work and delivery; quality achievement is not inferred. |

An update changing time/spend policy must be authorized in the goal and manifest; do not
expand `allowed_update_fields` merely because a review asked for more budget. Revise the goal
and queue the corresponding native update together. Wait for native `applied` status, including
the effective revision, before saying a live setting changed. A user stop remains authoritative.
An empty or null native `campaign_policy` disables its limits; it is rejected when the agreement
requires those limits. Contributions pin the library hash from the review observation. When a
contribution follows an acknowledged update in the same review, the scheduler passes its
`--after-update` dependency; unrelated intervening revisions remain stale.

Configure operations from the discovered capability map, with a fixed argument array,
`effect: read|mutate|launch`, and the exact allowed parameter names. Substitutions replace
whole argv elements only: `{run_id}`, `{campaign}`, `{fence_file}`, and named parameters.
Named parameters cannot override the reserved run/campaign/fence/arguments-file bindings.
An operation may set `accepts_arguments_file: true` and use `{arguments_file}`; the supervisor
then writes the proposed JSON arguments into its action directory. Mutation operations should
use native fencing wherever supported. Never grant arbitrary command strings or an unrestricted
shell operation as a shortcut. Long operations run separately so driver reviews/watchdog remain
available. A successful process exit still requires native result verification.

Action receipts distinguish prepared, issued, acknowledged, uncertain and verified effects.
Identical action IDs are not executed twice; a different payload under the same ID is rejected.
Reconcile uncertain actions against native receipts before deciding whether another request
is safe. Candidate authorship, experiments, code changes and source inspection follow
[supervision.md](supervision.md), including attribution and judge-context separation.

## Custom review and delivery adapters

A custom review adapter declares `host.kind: command`, an `argv` array and a `probe_argv`.
The probe must be read-only and return `{"review_protocol":1}`. Review argv may contain
`{context_file}`, `{prompt_file}` and `{decision_file}`. It must finish within the configured
timeout, write the structured decision object, and return zero. It may resume a verified host
conversation or create a bounded review with durable campaign context. Do not confuse the
host's session with the native AutoFA harness session. The scheduler does not assume an
undocumented Codex/Claude app API.

A custom adapter may also write `cost.json` beside `decision.json`, containing
`{"total_cost_usd":0.12,"usage":{"input_tokens":100,"output_tokens":30}}` from its real provider
receipt. An absent or malformed optional cost receipt never turns a successful review into
a failed one; its reservation remains unresolved. Do not invent usage or a dollar receipt.

A delivery adapter declares `delivery.kind: command`, `channel` and `argv`, with optional
`timeout_seconds`. Its argv receives `{message_file}` and optionally `{receipt_file}`. Read the
message's stable `id` and text; implement idempotence/reconciliation against the actual channel.
Only after delivery return this object on stdout or in the receipt file:

```json
{"id":"ORIGINAL_MESSAGE_ID","delivered":true,"channel":"AGREED_CHANNEL","remote_id":"CHANNEL_MESSAGE_ID"}
```

An exit code of zero, a log line or a locally generated ID does not prove delivery. Missing or
uncertain receipts stay unresolved and are not automatically resent; reconcile them through
the host channel to avoid duplicates. The outbox preserves the undelivered update.
Messages are delivered in creation order, one asynchronous operation at a time. Backlogs cannot
block the watchdog; idle ticks do not reread the entire message history.

Use `driver.py stop --campaign DIR --reason TEXT` for a user stop. The scheduler requests
native stop, terminates its own read-only reviewer and stops reviewing. Rejected or timed-out
stop requests remain pending with bounded retries and reconciliation. The service does not exit
or call the stop confirmed until native saved intent is acknowledged. Configured delivery drains
independently. A later explicit
`reopen` changes the campaign revision; reconcile native lifecycle and explicitly release it
before authorized work resumes. Killing only the scheduler is not a request to stop AutoFA.
The OS owner lock and revision fence prevent an older review from applying after takeover.
